using System.Diagnostics;
using System.Runtime.InteropServices;
using OpenTail.Stingray.Cpu;

/// <summary>
/// Single-thread Q4_K row-dot microbenchmark: our <c>DotQ4K_Q8KS</c> / <c>DotQ4K_Q8KS_2Row</c> against ggml's own <c>vec_dot</c> for Q4_K (reached through
/// <c>ggml_get_type_traits_cpu</c> in the vendored <c>ggml-cpu-haswell.dll</c>, the backend llama.cpp picks on this Zen 3 CPU), on the SAME raw weight rows,
/// each kernel fed its own int8 activation (ours Q8_KS, ggml Q8_K) quantised from the SAME float vector. No scheduler, no model object, no token loop,
/// no allocation inside the timed loop. Weight working sets from L1 to DRAM, then real expert/shared-expert shapes with the weights rotated past the L3.
/// Run with DOTNET_TieredCompilation=0 (tiered JIT invalidates the numbers).
/// </summary>
internal static unsafe class Q4KDotBench
{
    private const int K = 2048;                  // columns per row; 8 super-blocks, 1152 bytes per row
    private const int RowBytes = K / 256 * 144;

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate nint GetTraits(int type);

    public static int Run(string[] args)
    {
        if (Environment.GetEnvironmentVariable("DOTNET_TieredCompilation") != "0" && Environment.GetEnvironmentVariable("DOTNET_TC_QuickJitForLoops") != "0" && Environment.GetEnvironmentVariable("STINGRAY_BENCH_ALLOW_TIERED") != "1")
        {
            Console.Error.WriteLine("Refusing to run: set DOTNET_TieredCompilation=0 or DOTNET_TC_QuickJitForLoops=0 (or STINGRAY_BENCH_ALLOW_TIERED=1 to run the default tiered JIT after the built-in warm-up).");
            return 1;
        }

        string dir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "llama.cpp"));
        if (args.Length > 0) dir = args[0];
        string dll = Path.Combine(dir, "ggml-cpu-haswell.dll");
        if (!File.Exists(dll)) { Console.Error.WriteLine($"not found: {dll} (pass the tools/llama.cpp directory as the first argument)"); return 1; }
        Environment.SetEnvironmentVariable("PATH", dir + ";" + Environment.GetEnvironmentVariable("PATH"));
        nint lib = NativeLibrary.Load(dll);
        ((delegate* unmanaged[Cdecl]<void>)NativeLibrary.GetExport(lib, "ggml_cpu_init"))();
        var getTraits = (delegate* unmanaged[Cdecl]<int, nint>)NativeLibrary.GetExport(lib, "ggml_get_type_traits_cpu");

        const int Q4_K = 12, Q8_K = 15;
        nint tq4 = getTraits(Q4_K), tq8 = getTraits(Q8_K);
        var vecDot = (delegate* unmanaged[Cdecl]<int, float*, nuint, void*, nuint, void*, nuint, int, void>)*(nint*)(tq4 + 8);
        int vecDotType = *(int*)(tq4 + 16);
        long nrows = *(long*)(tq4 + 24);
        var fromFloatQ8K = (delegate* unmanaged[Cdecl]<float*, void*, long, void>)*(nint*)tq8;
        Console.WriteLine($"ggml Q4_K traits: vec_dot_type={vecDotType} (expect {Q8_K}), nrows={nrows}; AVX2={System.Runtime.Intrinsics.X86.Avx2.IsSupported} FMA={System.Runtime.Intrinsics.X86.Fma.IsSupported}");
        if (vecDotType != Q8_K) { Console.Error.WriteLine("unexpected vec_dot_type; struct layout differs"); return 1; }

        // One activation vector, quantised for each kernel.
        float* x = (float*)NativeMemory.AlignedAlloc((nuint)(K * sizeof(float)), 64);
        var rng = new Random(7);
        for (int i = 0; i < K; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
        byte* actOurs = (byte*)NativeMemory.AlignedAlloc((nuint)SimdKernels.Q8KSScratchBytes(K), 64);
        byte* actGgml = (byte*)NativeMemory.AlignedAlloc((nuint)(K / 256 * 292), 64);   // block_q8_K: float d + int8[256] + int16[16] = 292 bytes
        SimdKernels.QuantizeRowToQ8KS(x, K, actOurs);
        fromFloatQ8K(x, actGgml, K);

        // Shapes: (label, rows, rotate weights past the L3?)
        (string Label, int Rows, bool Rotate)[] shapes =
        [
            ("L1   (16 rows,   18 KB)", 16, false),
            ("L2   (256 rows, 295 KB)", 256, false),
            ("L3   (4096 rows, 4.7 MB)", 4096, false),
            ("DRAM (65536 rows, 75 MB)", 65536, false),
            ("real: shared-expert gate/up 5632 x 2048", 5632, true),
            ("real: Qwen3-Coder expert gate/up 768 x 2048", 768, true),
        ];

        // Accuracy against an exact double dot (4096 random rows, K = 2048): activations uniform [-1,1] with a few outliers, as in real hidden states.
        {
            var rng2 = new Random(99);
            float* xa = (float*)NativeMemory.AlignedAlloc((nuint)(K * sizeof(float)), 64);
            byte* a8 = (byte*)NativeMemory.AlignedAlloc((nuint)SimdKernels.Q8KSScratchBytes(K), 64);
            byte* aK = (byte*)NativeMemory.AlignedAlloc((nuint)(K / 256 * 292), 64);
            int rowsA = 4096; byte* wa = (byte*)NativeMemory.AlignedAlloc((nuint)((long)rowsA * RowBytes), 64);
            FillQ4K(wa, rowsA, new Random(5));
            double eOurs = 0, eGgml = 0, eProto = 0, norm = 0;
            for (int trial = 0; trial < 8; trial++)
            {
                for (int i = 0; i < K; i++) xa[i] = (float)(rng2.NextDouble() * 2 - 1) * (rng2.Next(64) == 0 ? 8f : 1f);
                SimdKernels.QuantizeRowToQ8KS(xa, K, a8); fromFloatQ8K(xa, aK, K);
                for (int r = 0; r < rowsA; r++)
                {
                    double exact = Q4KAccuracy.ExactDot(wa + (long)r * RowBytes, xa, K / 256);
                    float o = SimdKernels.DotQ4K_Q8KS(wa + (long)r * RowBytes, a8, K);
                    float g; vecDot(K, &g, 0, wa + (long)r * RowBytes, 0, aK, 0, 1);
                    float p = Q4KProto.Dot(wa + (long)r * RowBytes, aK, K / 256);
                    eOurs += (o - exact) * (o - exact); eGgml += (g - exact) * (g - exact); eProto += (p - exact) * (p - exact); norm += exact * exact;
                }
            }
            Console.WriteLine($"accuracy vs exact double dot (relative RMS error over {8 * rowsA} dots): ours Q8_KS {Math.Sqrt(eOurs / norm):E2}   ggml Q8_K {Math.Sqrt(eGgml / norm):E2}   prototype {Math.Sqrt(eProto / norm):E2}");
        }
        Console.WriteLine($"{"working set",-46} {"kernel",-14} {"ns/row",8} {"ns/superblk",12} {"GB/s",7}   checksum");
        foreach (var (label, rows, rotate) in shapes)
        {
            long matBytes = (long)rows * RowBytes;
            int copies = rotate ? (int)Math.Max(1, (96L << 20) / matBytes + 1) : 1;
            var mats = new byte*[copies];
            for (int c = 0; c < copies; c++)
            {
                mats[c] = (byte*)NativeMemory.AlignedAlloc((nuint)matBytes, 64);
                FillQ4K(mats[c], rows, new Random(11 + c));
            }
            float* outBuf = (float*)NativeMemory.AlignedAlloc((nuint)(rows * sizeof(float)), 64);

            double Run1(Action pass, long rowsPerPass)
            {
                { long w0 = Stopwatch.GetTimestamp(); while (Stopwatch.GetElapsedTime(w0).TotalMilliseconds < 600) pass(); }   // warm up for 600 ms so tiered JIT has promoted the kernel to tier 1
                var times = new List<double>();
                double budgetMs = 250; double spent = 0;
                while (spent < budgetMs || times.Count < 5)
                {
                    long t0 = Stopwatch.GetTimestamp(); pass();
                    double ms = Stopwatch.GetElapsedTime(t0).TotalMilliseconds; times.Add(ms); spent += ms;
                    if (times.Count > 400) break;
                }
                times.Sort();
                return times[times.Count / 2] * 1e6 / rowsPerPass;           // median ns per row
            }

            int cur = 0;
            void PassOurs1() { byte* m = mats[cur]; cur = (cur + 1) % copies; for (int r = 0; r < rows; r++) outBuf[r] = SimdKernels.DotQ4K_Q8KS(m + (long)r * RowBytes, actOurs, K); }
            void PassOurs2() { byte* m = mats[cur]; cur = (cur + 1) % copies; for (int r = 0; r + 1 < rows; r += 2) { SimdKernels.DotQ4K_Q8KS_2Row(m + (long)r * RowBytes, m + (long)(r + 1) * RowBytes, actOurs, K, out float a, out float b); outBuf[r] = a; outBuf[r + 1] = b; } }
            void PassProto() { byte* m = mats[cur]; cur = (cur + 1) % copies; int nb = K / 256; for (int r = 0; r < rows; r++) outBuf[r] = Q4KProto.Dot(m + (long)r * RowBytes, actGgml, nb); }
            void PassProto2() { byte* m = mats[cur]; cur = (cur + 1) % copies; int nb = K / 256; for (int r = 0; r + 1 < rows; r += 2) { Q4KProto2.Dot2(m + (long)r * RowBytes, m + (long)(r + 1) * RowBytes, actGgml, nb, out float a, out float b); outBuf[r] = a; outBuf[r + 1] = b; } }
            void PassA() { byte* m = mats[cur]; cur = (cur + 1) % copies; int nb = K / 256; for (int r = 0; r < rows; r++) outBuf[r] = Q4KProto3.DotA(m + (long)r * RowBytes, actGgml, nb); }
            void PassB() { byte* m = mats[cur]; cur = (cur + 1) % copies; int nb = K / 256; for (int r = 0; r < rows; r++) outBuf[r] = Q4KProto3.DotB(m + (long)r * RowBytes, actGgml, nb); }
            void PassAB() { byte* m = mats[cur]; cur = (cur + 1) % copies; int nb = K / 256; for (int r = 0; r < rows; r++) outBuf[r] = Q4KProto3.DotAB(m + (long)r * RowBytes, actGgml, nb); }
            void PassR() { byte* m = mats[cur]; cur = (cur + 1) % copies; Q4KProto3.Rows(m, rows, RowBytes, actGgml, K / 256, outBuf); }
            void PassRA() { byte* m = mats[cur]; cur = (cur + 1) % copies; Q4KProto3.RowsA(m, rows, RowBytes, actGgml, K / 256, outBuf); }
            void PassRAB() { byte* m = mats[cur]; cur = (cur + 1) % copies; Q4KProto3.RowsAB(m, rows, RowBytes, actGgml, K / 256, outBuf); }
            void PassGgml() { byte* m = mats[cur]; cur = (cur + 1) % copies; for (int r = 0; r < rows; r++) vecDot(K, outBuf + r, 0, m + (long)r * RowBytes, 0, actGgml, 0, 1); }

            foreach (var (name, pass) in new (string, Action)[] { ("ours 1-row", PassOurs1), ("ours 2-row", PassOurs2), ("proto (ggml-style)", PassProto), ("proto 2-row", PassProto2), ("A 2acc", PassA),("B batchHalf", PassB), ("AB", PassAB), ("R rows-in", PassR), ("RA", PassRA), ("RAB", PassRAB),("ggml vec_dot", PassGgml) })
            {
                cur = 0;
                double nsRow = Run1(pass, rows);
                pass();
                double cs = 0; for (int r = 0; r < Math.Min(rows, 64); r++) cs += outBuf[r];
                Console.WriteLine($"{label,-46} {name,-14} {nsRow,8:F1} {nsRow / (K / 256),12:F1} {RowBytes / nsRow,7:F2}   {cs:F3}");
            }
            for (int c = 0; c < copies; c++) NativeMemory.AlignedFree(mats[c]);
            NativeMemory.AlignedFree(outBuf);
        }
        return 0;
    }

    /// <summary>Random valid Q4_K blocks (144 B: d, dmin as small positive halves, random scales and nibbles).</summary>
    private static void FillQ4K(byte* w, int rows, Random rng)
    {
        long n = (long)rows * RowBytes;
        var span = new Span<byte>(w, (int)Math.Min(n, int.MaxValue));
        rng.NextBytes(span);
        long blocks = n / 144;
        for (long b = 0; b < blocks; b++)
        {
            byte* blk = w + b * 144;
            ushort d = BitConverter.HalfToUInt16Bits((Half)(0.004f + 0.002f * (float)rng.NextDouble()));
            ushort dmin = BitConverter.HalfToUInt16Bits((Half)(0.002f + 0.002f * (float)rng.NextDouble()));
            blk[0] = (byte)d; blk[1] = (byte)(d >> 8); blk[2] = (byte)dmin; blk[3] = (byte)(dmin >> 8);
        }
    }
}
