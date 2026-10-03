using System.Diagnostics;
using System.Runtime.InteropServices;
using OpenTail.Stingray.Cpu;

/// <summary>
/// <c>OpenTail.Stingray.KernelBench sgemm [reps=7]</c>: OpenBLAS <c>cblas_sgemm</c> vs <see cref="CpuSgemm"/>
/// (and <see cref="PackedSgemmF32"/>, weights pre-packed) on the <c>CpuBackend.Sgemm</c> layout,
/// C[M,N] = A[M,K]·B[N,K]ᵀ, at shapes the CPU diffusion / TTS callers issue. Arms alternate within each
/// repetition so a machine-wide slowdown hits both equally; medians are reported.
/// </summary>
internal static unsafe class SgemmBench
{
    private static readonly (string Name, int M, int K, int N)[] Shapes =
    [
        ("flux qkv 512px",        1024, 3072, 9216),
        ("flux mlp-up 512px",     1024, 3072, 12288),
        ("flux mlp-down 512px",   1024, 12288, 3072),
        ("sd3 attn 512px",        1101, 1536, 1536),
        ("sd3 mlp-up 512px",      1101, 1536, 6144),
        ("zimage qkv 512px",      1024, 3840, 3840),
        ("f5 attn",                800, 1024, 1024),
        ("f5 ff-up",               800, 1024, 2048),
        ("t5 ff-up",               128, 4096, 10240),
        ("clip mlp",                77, 768, 3072),
        ("vae conv 512ch 64x64",  4096, 4608, 512),
        ("vae conv 128ch 256x256 (1/4)", 16384, 1152, 128),
        ("narrow conv_out",      16384, 1152, 3),
        ("narrow n=16",           4096, 512, 16),
        ("skinny m=4",               4, 3072, 3072),
        ("small m=16",              16, 1024, 4096),
    ];

    /// <summary><c>sgemm-sweep [reps=5]</c>: CpuSgemm cache-blocking sweep (KC, MC, NC) on the large
    /// compute-bound shapes, configurations interleaved within each repetition.</summary>
    public static int Sweep(string[] args)
    {
        int reps = args.Length > 0 && int.TryParse(args[0], out int r) && r > 0 ? r : 5;
        (int Kc, int Mc, int Nc)[] configs =
        [
            (0, 0, 0), (128, 0, 0), (192, 0, 0), (384, 0, 0),
            (0, 72, 0), (0, 96, 0), (0, 240, 0), (0, 0, 512), (0, 0, 2048), (0, 0, 4096),
            (192, 96, 0), (384, 72, 0), (128, 240, 2048),
        ];
        var shapes = Shapes.Where(s => (double)s.M * s.N * s.K >= 2e9).ToArray();
        Console.WriteLine($"kernel {CpuSgemm.ActiveKernelName}; median of {reps}; ms per shape (0 = default)");
        Console.WriteLine("KC/MC/NC        | " + string.Join(" | ", shapes.Select(s => $"{s.Name,22}")) + " |  geo-mean vs default");
        var results = new double[configs.Length, shapes.Length];
        for (int si = 0; si < shapes.Length; si++)
        {
            var (_, m, k, n) = shapes[si];
            float* a = Alloc((long)m * k), b = Alloc((long)n * k), c = Alloc((long)m * n);
            Fill(a, (long)m * k, 1); Fill(b, (long)n * k, 2);
            var times = new List<double>[configs.Length];
            for (int ci = 0; ci < configs.Length; ci++) times[ci] = new List<double>();
            for (int rep = -1; rep < reps; rep++)
                for (int ci = 0; ci < configs.Length; ci++)
                {
                    (CpuSgemm.OverrideKc, CpuSgemm.OverrideMc, CpuSgemm.OverrideNc) = configs[ci];
                    double t = Time(() => CpuSgemm.Gemm(m, n, k, 1f, a, k, 1, b, 1, k, 0f, c, n));
                    if (rep >= 0) times[ci].Add(t);
                }
            for (int ci = 0; ci < configs.Length; ci++) results[ci, si] = Median(times[ci]);
            NativeMemory.AlignedFree(a); NativeMemory.AlignedFree(b); NativeMemory.AlignedFree(c);
        }
        (CpuSgemm.OverrideKc, CpuSgemm.OverrideMc, CpuSgemm.OverrideNc) = (0, 0, 0);
        for (int ci = 0; ci < configs.Length; ci++)
        {
            double logSum = 0;
            for (int si = 0; si < shapes.Length; si++) logSum += Math.Log(results[ci, si] / results[0, si]);
            var (kc, mc, nc) = configs[ci];
            Console.WriteLine($"{kc,4}/{mc,4}/{nc,4}  | " + string.Join(" | ", Enumerable.Range(0, shapes.Length).Select(si => $"{results[ci, si],22:F2}")) +
                              $" |  {Math.Exp(logSum / shapes.Length),6:F3}");
        }
        return 0;
    }

    /// <summary><c>q4k-m1 [reps=200]</c>: decode-shape Q4_K, the Q8_KS matvec (today's decode) vs the
    /// repacked Path 2 GEMM at N = 1 (STINGRAY_CPU_DECODE_VIA_GEMM). Interleaved, median.</summary>
    public static int Q4KM1(string[] args)
    {
        int reps = args.Length > 0 && int.TryParse(args[0], out int r) && r > 0 ? r : 200;
        (string Name, int Rows, int Cols)[] shapes =
        [
            ("360M q/o 960x960", 960, 960), ("360M ffn-up 2560x960", 2560, 960),
            ("1.7B q 2048x2048", 2048, 2048), ("1.7B ffn-up 8192x2048", 8192, 2048),
            ("7B ffn-up 14336x4096", 14336, 4096),
        ];
        Console.WriteLine($"{"shape",-24} | {"matvec us",10} | {"gemm N=1 us",11} | ratio");
        foreach (var (name, rows, cols) in shapes)
        {
            if (cols % 256 != 0) { Console.WriteLine($"{name,-24} | skipped (cols % 256)"); continue; }
            long rowBytes = (long)cols / 256 * 144;
            byte* w = (byte*)NativeMemory.AlignedAlloc((nuint)(rowBytes * rows), 64);
            byte* packed = (byte*)NativeMemory.AlignedAlloc((nuint)SimdKernels.Q4Kx8PackedBytes(rows, cols), 64);
            float* x = Alloc(cols); float* o1 = Alloc(rows); float* o2 = Alloc(rows);
            var rng = new Random(5);
            for (long i = 0; i < rowBytes * rows; i++) w[i] = (byte)rng.Next(256);
            for (int i = 0; i < rows; i++)
                for (int b = 0; b < cols / 256; b++)
                {
                    byte* blk = w + i * rowBytes + b * 144;
                    short d = BitConverter.HalfToInt16Bits((Half)0.01f);
                    blk[0] = (byte)d; blk[1] = (byte)(d >> 8); blk[2] = (byte)d; blk[3] = (byte)(d >> 8);
                }
            Fill(x, cols, 6);
            SimdKernels.RepackQ4KMatrix(w, packed, rows, cols);
            // Decode streams every matrix from DRAM: rotate through copies totalling > 96 MB per arm so
            // no call finds its weights in the 16 MB L3 (TensorSharp's "weights rotated past the L3").
            long matBytes = rowBytes * rows;
            int copies = (int)Math.Max(1, (96L << 20) / matBytes + 1);
            var ws = new nint[copies]; var ps = new nint[copies];
            for (int c = 0; c < copies; c++)
            {
                ws[c] = (nint)NativeMemory.AlignedAlloc((nuint)matBytes, 64);
                Buffer.MemoryCopy(w, (void*)ws[c], matBytes, matBytes);
                long pb = SimdKernels.Q4Kx8PackedBytes(rows, cols);
                ps[c] = (nint)NativeMemory.AlignedAlloc((nuint)pb, 64);
                Buffer.MemoryCopy(packed, (void*)ps[c], pb, pb);
            }
            int ci = 0, cj = 0;
            void Mv() { SimdKernels.MatVec(o1, (byte*)ws[ci], x, rows, cols, OpenTail.Stingray.Core.DType.Q4_K); ci = (ci + 1) % copies; }
            void Gm() { SimdKernels.TryMatMulBatchedQ4Kx8(o2, (byte*)ps[cj], x, 1, rows, cols); cj = (cj + 1) % copies; }
            for (int i = 0; i < 2 * copies; i++) { Mv(); Gm(); }
            var t1 = new List<double>(); var t2 = new List<double>();
            for (int i = 0; i < reps; i++) { t1.Add(Time(Mv) * 1000); t2.Add(Time(Gm) * 1000); }
            double a = Median(t1), b2 = Median(t2);
            Console.WriteLine($"{name,-24} | {a,10:F1} | {b2,11:F1} | {a / b2:F2}   ({copies} copies, {matBytes * copies >> 20} MB/arm)");
            // Decode's gate+up: one fused MatVecDual over two matrices vs two GEMM calls.
            if (copies >= 2)
            {
                float* o3 = Alloc(rows);
                int di = 0, dj = 0;
                void Dual() { SimdKernels.MatVecDual(o1, (byte*)ws[di], o3, (byte*)ws[(di + 1) % copies], x, rows, cols, OpenTail.Stingray.Core.DType.Q4_K, OpenTail.Stingray.Core.DType.Q4_K); di = (di + 2) % copies; }
                void Gm2() { SimdKernels.TryMatMulBatchedQ4Kx8(o2, (byte*)ps[dj], x, 1, rows, cols); SimdKernels.TryMatMulBatchedQ4Kx8(o3, (byte*)ps[(dj + 1) % copies], x, 1, rows, cols); dj = (dj + 2) % copies; }
                for (int i = 0; i < copies; i++) { Dual(); Gm2(); }
                var t3 = new List<double>(); var t4 = new List<double>();
                for (int i = 0; i < reps; i++) { t3.Add(Time(Dual) * 1000); t4.Add(Time(Gm2) * 1000); }
                Console.WriteLine($"{"  x2 (MatVecDual vs 2 GEMM)",-24} | {Median(t3),10:F1} | {Median(t4),11:F1} | {Median(t3) / Median(t4):F2}");
                NativeMemory.AlignedFree(o3);
            }
            for (int c = 0; c < copies; c++) { NativeMemory.AlignedFree((void*)ws[c]); NativeMemory.AlignedFree((void*)ps[c]); }
            NativeMemory.AlignedFree(w); NativeMemory.AlignedFree(packed);
            NativeMemory.AlignedFree(x); NativeMemory.AlignedFree(o1); NativeMemory.AlignedFree(o2);
        }
        Console.WriteLine("ratio = matvec / gemm (>1: the GEMM at N=1 is faster). Weights L3/DRAM-resident as in decode for the larger shapes.");

        // Q6_K (ffn_down / attn_v on half the layers of Q4_K_M, and usually the LM head).
        (string Name, int Rows, int Cols)[] q6 =
        [
            ("1.7B ffn-down 2048x8192", 2048, 8192), ("1.7B lm-head 49152x2048", 49152, 2048),
            ("7B ffn-down 4096x14336", 4096, 14336), ("7B lm-head 32768x4096", 32768, 4096),
        ];
        Console.WriteLine($"{"Q6_K shape",-24} | {"matvec us",10} | {"gemm N=1 us",11} | ratio");
        foreach (var (name, rows, cols) in q6)
        {
            long rowBytes = (long)cols / 256 * 210;
            byte* w = (byte*)NativeMemory.AlignedAlloc((nuint)(rowBytes * rows), 64);
            float* x = Alloc(cols); float* o1 = Alloc(rows); float* o2 = Alloc(rows);
            var rng = new Random(7);
            for (long i = 0; i < rowBytes * rows; i++) w[i] = (byte)rng.Next(256);
            for (int i = 0; i < rows; i++)
                for (int b = 0; b < cols / 256; b++)
                {
                    byte* blk = w + i * rowBytes + b * 210;
                    short d = BitConverter.HalfToInt16Bits((Half)0.005f);
                    blk[208] = (byte)d; blk[209] = (byte)(d >> 8);
                }
            Fill(x, cols, 8);
            void Mv() => SimdKernels.MatVec(o1, w, x, rows, cols, OpenTail.Stingray.Core.DType.Q6_K);
            void Gm() => Q6KPrefillGemm.TryMatMulBatched(o2, w, x, 1, rows, cols);
            for (int i = 0; i < 10; i++) { Mv(); Gm(); }
            var t1 = new List<double>(); var t2 = new List<double>();
            for (int i = 0; i < reps / 4; i++) { t1.Add(Time(Mv) * 1000); t2.Add(Time(Gm) * 1000); }
            double a = Median(t1), b2 = Median(t2);
            Console.WriteLine($"{name,-24} | {a,10:F1} | {b2,11:F1} | {a / b2:F2}");
            NativeMemory.AlignedFree(w); NativeMemory.AlignedFree(x); NativeMemory.AlignedFree(o1); NativeMemory.AlignedFree(o2);
        }
        return 0;
    }

    public static int Run(string[] args)
    {
        int reps = args.Length > 0 && int.TryParse(args[0], out int r) && r > 0 ? r : 7;
        bool blas = BlasInterop.IsAvailable;
        Console.WriteLine($"CpuSgemm kernel: {CpuSgemm.ActiveKernelName}; OpenBLAS: {(blas ? "loaded" : "NOT FOUND")}; " +
                          $"threads: {SimdKernels.ParallelOpts.MaxDegreeOfParallelism}; reps: {reps} (median, interleaved)");
        Console.WriteLine($"{"shape",-30} {"M",6} {"K",6} {"N",6} | {"OpenBLAS ms",11} {"GF/s",6} | {"CpuSgemm ms",11} {"GF/s",6} | {"ratio",6} | {"Packed* ms",10} | {"maxRel",8}");

        foreach (var (name, m, k, n) in Shapes)
        {
            float* a = Alloc((long)m * k), b = Alloc((long)n * k);
            float* cBlas = Alloc((long)m * n), cOurs = Alloc((long)m * n), cPacked = Alloc((long)m * n);
            float* packed = null;
            try
            {
                Fill(a, (long)m * k, 1);
                Fill(b, (long)n * k, 2);
                if (PackedSgemmF32.IsSupported) packed = PackedSgemmF32.PackWeights(b, n, k);

                void RunBlas() => BlasInterop.Sgemm(BlasInterop.RowMajor, BlasInterop.NoTrans, BlasInterop.Trans,
                    m, n, k, 1f, a, k, b, k, 0f, cBlas, n);
                void RunOurs() => CpuSgemm.Gemm(m, n, k, 1f, a, k, 1, b, 1, k, 0f, cOurs, n);
                void RunPacked() => PackedSgemmF32.Gemm(cPacked, a, packed, null, m, n, k);

                // Warm up every arm (thread pools, page faults, per-thread packing buffers).
                for (int w = 0; w < 2; w++)
                {
                    if (blas) RunBlas();
                    RunOurs();
                    if (packed != null) RunPacked();
                }

                var tBlas = new List<double>(); var tOurs = new List<double>(); var tPacked = new List<double>();
                for (int i = 0; i < reps; i++)
                {
                    if (blas) tBlas.Add(Time(RunBlas));
                    tOurs.Add(Time(RunOurs));
                    if (packed != null) tPacked.Add(Time(RunPacked));
                }

                double gflop = 2.0 * m * n * k / 1e9;
                double mb = blas ? Median(tBlas) : double.NaN, mo = Median(tOurs);
                double mp = packed != null ? Median(tPacked) : double.NaN;
                double rel = blas ? MaxRel(cBlas, cOurs, (long)m * n) : MaxRelReference(a, b, cOurs, m, n, k);
                Console.WriteLine($"{name,-30} {m,6} {k,6} {n,6} | {mb,11:F2} {gflop / mb * 1e3,6:F0} | {mo,11:F2} {gflop / mo * 1e3,6:F0} | " +
                                  $"{mb / mo,6:F2} | {mp,10:F2} | {rel,8:E1}");
            }
            finally
            {
                if (packed != null) NativeMemory.AlignedFree(packed);
                NativeMemory.AlignedFree(a); NativeMemory.AlignedFree(b);
                NativeMemory.AlignedFree(cBlas); NativeMemory.AlignedFree(cOurs); NativeMemory.AlignedFree(cPacked);
            }
        }
        Console.WriteLine("ratio = OpenBLAS ms / CpuSgemm ms (>1: CpuSgemm faster). Packed* excludes its one-off weight repack.");
        return 0;
    }

    private static float* Alloc(long n) => (float*)NativeMemory.AlignedAlloc((nuint)(n * sizeof(float)), 64);

    private static void Fill(float* p, long n, int seed)
    {
        var rng = new Random(seed);
        for (long i = 0; i < n; i++) p[i] = (float)(rng.NextDouble() * 2 - 1);
    }

    private static double Time(Action f)
    {
        long t0 = Stopwatch.GetTimestamp();
        f();
        return Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
    }

    private static double Median(List<double> xs)
    {
        xs.Sort();
        return xs.Count % 2 == 1 ? xs[xs.Count / 2] : 0.5 * (xs[xs.Count / 2 - 1] + xs[xs.Count / 2]);
    }

    /// <summary>Max |x−y| / max(1, |y|) over the output.</summary>
    private static double MaxRel(float* x, float* y, long n)
    {
        double worst = 0;
        for (long i = 0; i < n; i++)
            worst = Math.Max(worst, Math.Abs(x[i] - y[i]) / Math.Max(1.0, Math.Abs(y[i])));
        return worst;
    }

    /// <summary>Without OpenBLAS: spot-check 64 outputs against a double-precision dot product.</summary>
    private static double MaxRelReference(float* a, float* b, float* c, int m, int n, int k)
    {
        double worst = 0;
        var rng = new Random(3);
        for (int s = 0; s < 64; s++)
        {
            int i = rng.Next(m), j = rng.Next(n);
            double acc = 0;
            for (int p = 0; p < k; p++) acc += (double)a[(long)i * k + p] * b[(long)j * k + p];
            worst = Math.Max(worst, Math.Abs(acc - c[(long)i * n + j]) / Math.Max(1.0, Math.Abs(acc)));
        }
        return worst;
    }
}
