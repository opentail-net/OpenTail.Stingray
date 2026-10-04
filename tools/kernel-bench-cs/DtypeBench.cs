using System.Diagnostics;
using System.Runtime.InteropServices;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Core;

/// <summary>
/// Single-thread production matvec timing per quantised dtype: <c>SimdKernels.MatVec</c> with 63 rows per call (below the parallel threshold, so no scheduler),
/// cols = 8192, a pool of random-but-valid blocks rotated past the L3 (~48 MB per dtype). Reports ns/row and weight GB/s plus a checksum so before/after
/// runs of a kernel change can be compared bit-for-bit. Usage: <c>dtype-matvec [DType names...]</c>; no names = every dtype MatVec dispatches.
/// </summary>
internal static unsafe class DtypeBench
{
    private const int Cols = 8192, RowsPerCall = 63;

    // byte offsets of the fp16 fields inside one block, so random fill leaves sane (small positive) scales
    private static readonly (DType T, int[] HalfOffsets)[] Types =
    [
        (DType.Q4_0, [0]), (DType.Q8_0, [0]), (DType.IQ4_NL, [0]),
        (DType.Q2_K, [80, 82]), (DType.Q3_K, [108]), (DType.Q4_K, [0, 2]), (DType.Q5_K, [0, 2]), (DType.Q6_K, [208]),
        (DType.IQ4_XS, [0]), (DType.IQ2_XXS, [0]), (DType.IQ2_XS, [0]), (DType.IQ2_S, [0]), (DType.IQ3_XXS, [0]), (DType.IQ3_S, [0]),
    ];

    public static int Run(string[] args)
    {
        var want = new HashSet<string>(args, StringComparer.OrdinalIgnoreCase);
        float* input = (float*)NativeMemory.AlignedAlloc((nuint)(Cols * sizeof(float)), 64);
        var rng = new Random(3);
        for (int i = 0; i < Cols; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);
        float* output = (float*)NativeMemory.AlignedAlloc((nuint)(RowsPerCall * sizeof(float)), 64);
        string dllDir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "llama.cpp"));
        Environment.SetEnvironmentVariable("PATH", dllDir + ";" + Environment.GetEnvironmentVariable("PATH"));
        nint lib = NativeLibrary.Load(Path.Combine(dllDir, "ggml-cpu-haswell.dll"));
        ((delegate* unmanaged[Cdecl]<void>)NativeLibrary.GetExport(lib, "ggml_cpu_init"))();
        var getTraits = (delegate* unmanaged[Cdecl]<int, nint>)NativeLibrary.GetExport(lib, "ggml_get_type_traits_cpu");
        Console.WriteLine($"{"dtype",-9} {"ours ns/row",12} {"ggml ns/row",12} {"ours/ggml",10} {"ns/block",9}   checksums (random weights; ours and ggml quantise the activation differently)");
        foreach (var (t, offs) in Types)
        {
            if (want.Count > 0 && !want.Contains(t.ToString())) continue;
            int bs = DTypeInfo.BlockSize(t), bpb = DTypeInfo.BytesPerBlock(t);
            int rowBytes = Cols / bs * bpb;
            long callBytes = (long)RowsPerCall * rowBytes;
            int calls = (int)Math.Max(2, (48L << 20) / callBytes);
            byte* w = (byte*)NativeMemory.AlignedAlloc((nuint)(callBytes * calls), 64);
            var r2 = new Random(11);
            new Span<byte>(w, (int)(callBytes * calls)).ForEach(r2);
            long blocks = callBytes * calls / bpb;
            for (long b = 0; b < blocks; b++)
                foreach (int o in offs)
                {
                    ushort h = BitConverter.HalfToUInt16Bits((Half)(0.003f + 0.002f * (float)r2.NextDouble()));
                    w[b * bpb + o] = (byte)h; w[b * bpb + o + 1] = (byte)(h >> 8);
                }

            double TimeRows(Action pass)
            {
                { long t0 = Stopwatch.GetTimestamp(); while (Stopwatch.GetElapsedTime(t0).TotalMilliseconds < 500) pass(); }
                var times = new List<double>(); double spent = 0;
                while (spent < 600 || times.Count < 7) { long t0 = Stopwatch.GetTimestamp(); pass(); double ms = Stopwatch.GetElapsedTime(t0).TotalMilliseconds; times.Add(ms); spent += ms; if (times.Count > 300) break; }
                times.Sort();
                return times[times.Count / 2] * 1e6 / ((double)calls * RowsPerCall);
            }

            void Pass() { for (int c = 0; c < calls; c++) SimdKernels.MatVec(output, w + c * callBytes, input, RowsPerCall, Cols, t); }
            double nsRow = TimeRows(Pass);
            double cs = 0; Pass(); for (int r = 0; r < RowsPerCall; r++) cs += output[r];

            // ggml: its own activation quantiser (per call, like MatVec's) + its own vec_dot per row, same weight bytes
            nint tr = getTraits((int)t);
            var vecDot = (delegate* unmanaged[Cdecl]<int, float*, nuint, void*, nuint, void*, nuint, int, void>)*(nint*)(tr + 8);
            int vdt = *(int*)(tr + 16);
            var fromFloat = (delegate* unmanaged[Cdecl]<float*, void*, long, void>)*(nint*)(getTraits(vdt));
            byte* act = (byte*)NativeMemory.AlignedAlloc(1 << 16, 64);
            float* gOut = (float*)NativeMemory.AlignedAlloc((nuint)(RowsPerCall * sizeof(float)), 64);
            void GPass() { for (int c = 0; c < calls; c++) { fromFloat(input, act, Cols); byte* m = w + c * callBytes; for (int r = 0; r < RowsPerCall; r++) vecDot(Cols, gOut + r, 0, m + (long)r * rowBytes, 0, act, 0, 1); } }
            double nsGgml = TimeRows(GPass);
            double gcs = 0; GPass(); for (int r = 0; r < RowsPerCall; r++) gcs += gOut[r];
            Console.WriteLine($"{t,-9} {nsRow,12:F0} {nsGgml,12:F0} {nsRow / nsGgml,9:F2}x {nsRow / (Cols / bs),9:F1}   ours {cs:F3} ggml {gcs:F3}");
            NativeMemory.AlignedFree(act); NativeMemory.AlignedFree(gOut);
            NativeMemory.AlignedFree(w);
        }
        return 0;
    }

    private static void ForEach(this Span<byte> s, Random r) => r.NextBytes(s);
}
