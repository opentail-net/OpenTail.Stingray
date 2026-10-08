using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using OpenTail.Stingray.Cpu;

/// <summary>
/// One-thread IQ3_S microbenchmark for the MTP batched-verify question (docs/00-current-work.md item 17):
/// is the IQ3_S dot limited by weight reconstruction that could be shared across N activations?
/// Variants (same rows, same Q8_K activations, default JIT with a warm-up, no TC=0):
///   A  current  : N x DotIq3S_Q8K(row, q8_i)                     (what MatVec2In/4In fall back to)
///   S2 / S4     : shared-decode prototype, grid+sign mask built once per 32-weight group, N activations inside
///   P  predecoded: row pre-expanded to signed int8 + group scales (diagnostic upper bound, no codebook work)
/// Modes: hot (one row repeated: pure execution cost) and stream (many rows: cache/DRAM included).
/// </summary>
internal static unsafe class Iq3SMultiBench
{
    private const int Bpb = 110; // bytes per 256-weight IQ3_S block

    public static int Run(string[] args)
    {
        int cols = args.Length > 0 ? int.Parse(args[0]) : 5120;
        int rows = args.Length > 1 ? int.Parse(args[1]) : 17408;
        int nb = cols / 256;
        int rowBytes = nb * Bpb;
        int q8Bytes = SimdKernels.Q8KScratchBytes(cols);

        var rng = new Random(12345);
        byte* w = (byte*)NativeMemory.AlignedAlloc((nuint)((long)rows * rowBytes), 64);
        for (long i = 0; i < (long)rows * rowBytes; i++) w[i] = (byte)rng.Next(256);
        // Valid fp16 super-block scale (0.5 .. 2.0) so results are finite.
        for (int r = 0; r < rows; r++)
            for (int b = 0; b < nb; b++)
            {
                ushort h = (ushort)BitConverter.HalfToUInt16Bits((Half)(0.5f + (float)rng.NextDouble()));
                *(ushort*)(w + (long)r * rowBytes + b * Bpb) = h;
            }

        byte*[] q8 = new byte*[4];
        float* tmp = (float*)NativeMemory.AlignedAlloc((nuint)(cols * 4), 64);
        for (int t = 0; t < 4; t++)
        {
            for (int i = 0; i < cols; i++) tmp[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            q8[t] = (byte*)NativeMemory.AlignedAlloc((nuint)q8Bytes, 64);
            SimdKernels.QuantizeRowToQ8K(tmp, cols, q8[t]);
        }

        // ---- correctness: every variant must equal the production dot on row 0..63 ----
        byte* pre = (byte*)NativeMemory.AlignedAlloc((nuint)(cols + cols / 32), 64);
        double maxErr = 0;
        float* o = stackalloc float[4];
        float* o2 = stackalloc float[2];
        for (int r = 0; r < 64; r++)
        {
            byte* row = w + (long)r * rowBytes;

            Dot4Shared(row, q8[0], q8[1], q8[2], q8[3], cols, o);

            Dot2Shared(row, q8[0], q8[1], cols, o2);
            Predecode(row, cols, pre);
            for (int t = 0; t < 4; t++)
            {
                float refv = SimdKernels.DotIq3S_Q8K(row, q8[t], cols);
                maxErr = Math.Max(maxErr, Math.Abs(refv - o[t]) / (Math.Abs(refv) + 1e-3));
                if (t < 2) maxErr = Math.Max(maxErr, Math.Abs(refv - o2[t]) / (Math.Abs(refv) + 1e-3));
                maxErr = Math.Max(maxErr, Math.Abs(refv - DotPre(pre, q8[t], cols)) / (Math.Abs(refv) + 1e-3));
            }
        }
        Console.WriteLine($"cols={cols} rows={rows} rowBytes={rowBytes}  correctness max rel err vs production = {maxErr:E2}");
        if (maxErr > 1e-4) { Console.WriteLine("MISMATCH: kernels disagree, aborting."); return 2; }

        // ---- timing ----
        Console.WriteLine("mode    variant         ns/row(all N)   ns/row/token   vs A(N=4)");
        foreach (bool hot in new[] { true, false })
        {
            string mode = hot ? "hot" : "stream";
            int nRows = hot ? 1 : rows;
            double a4 = Time(hot, nRows, w, rowBytes, q8, cols, pre, 'A', 4);
            double a2 = Time(hot, nRows, w, rowBytes, q8, cols, pre, 'A', 2);
            double s2 = Time(hot, nRows, w, rowBytes, q8, cols, pre, 'S', 2);
            double s4 = Time(hot, nRows, w, rowBytes, q8, cols, pre, 'S', 4);
            double p4 = Time(hot, nRows, w, rowBytes, q8, cols, pre, 'P', 4);
            double a1 = Time(hot, nRows, w, rowBytes, q8, cols, pre, 'A', 1);
            void P(string v, double ns, int n) => Console.WriteLine($"{mode,-7} {v,-14} {ns,12:F0} {ns / n,14:F0} {a4 / ns,10:F2}x");
            P("A  N=1", a1, 1); P("A  N=2", a2, 2); P("S  N=2", s2, 2);
            P("A  N=4", a4, 4); P("S  N=4", s4, 4); if (hot) P("P  N=4(pre)", p4, 4);
        }
        return 0;
    }

    private static double Time(bool hot, int nRows, byte* w, int rowBytes, byte*[] q8, int cols, byte* pre, char kind, int n)
    {
        float* o = (float*)NativeMemory.AlignedAlloc(64, 64);
        // Predecoded copy of the rows used (hot: one row; stream: not supported -> reuse first row, flagged by caller)
        Predecode(w, cols, pre);
        double best = double.MaxValue;
        float sink = 0;
        // warm-up (tier-up, OSR) then timed trials
        for (int trial = 0; trial < 9; trial++)
        {
            long ticks0 = Stopwatch.GetTimestamp();
            long calls = 0;
            double budget = trial < 3 ? 0.35 : 0.5; // seconds per trial; first 3 are warm-up and discarded
            while ((Stopwatch.GetTimestamp() - ticks0) < budget * Stopwatch.Frequency)
            {
                for (int r = 0; r < nRows; r++)
                {
                    byte* row = w + (long)r * rowBytes;
                    switch (kind)
                    {
                        case 'A':
                            for (int t = 0; t < n; t++) sink += SimdKernels.DotIq3S_Q8K(row, q8[t], cols);
                            break;
                        case 'S':
                            if (n == 2) { Dot2Shared(row, q8[0], q8[1], cols, o); sink += o[0] + o[1]; }
                            else { Dot4Shared(row, q8[0], q8[1], q8[2], q8[3], cols, o); sink += o[0] + o[3]; }
                            break;
                        case 'P':
                            for (int t = 0; t < n; t++) sink += DotPre(pre, q8[t], cols);
                            break;
                    }
                }
                calls += nRows;
            }
            double ns = (Stopwatch.GetTimestamp() - ticks0) * 1e9 / Stopwatch.Frequency / calls;
            if (trial >= 3 && ns < best) best = ns;
        }
        GC.KeepAlive(sink);
        NativeMemory.AlignedFree(o);
        return best;
    }

    // ---- shared-decode kernels -------------------------------------------------------------------------------
    private static readonly Vector256<byte> SignShuffle = Vector256.Create(
        (byte)0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    private static readonly Vector256<byte> SignBits = Vector256.Create(
        (byte)1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<byte> NegMask(uint signs4)
    {
        var bits = Avx2.And(Avx2.Shuffle(Vector256.Create(signs4).AsByte(), SignShuffle), SignBits);
        return Avx2.CompareEqual(bits, SignBits);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<short> Dot32(Vector256<byte> grid, Vector256<byte> neg, sbyte* q8)
    {
        var q8s = Avx2.Subtract(Avx2.Xor(Avx.LoadVector256((byte*)q8), neg), neg).AsSByte();
        return Avx2.MultiplyAddAdjacent(grid, q8s);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float HSum(Vector256<float> v)
    {
        var s = Sse.Add(v.GetLower(), v.GetUpper());
        s = Sse.Add(s, Sse.MoveHighToLow(s, s));
        s = Sse.AddScalar(s, Sse.Shuffle(s, s, 0x55));
        return s.ToScalar();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Grid(byte* qs, byte qh0, byte qh1, int ib32, uint* grid, uint* idx,
        Vector256<uint> idxShift, Vector256<uint> idxMask, out Vector256<byte> g1, out Vector256<byte> g2)
    {
        var lo1 = Avx2.ConvertToVector256Int32(qs + ib32 * 8).AsUInt32();
        var lo2 = Avx2.ConvertToVector256Int32(qs + ib32 * 8 + 8).AsUInt32();
        var hi1 = Avx2.And(Avx2.ShiftLeftLogicalVariable(Vector256.Create((uint)qh0), idxShift), idxMask);
        var hi2 = Avx2.And(Avx2.ShiftLeftLogicalVariable(Vector256.Create((uint)qh1), idxShift), idxMask);
        Avx.Store(idx, Avx2.Or(lo1, hi1));
        g1 = Vector256.Create(grid[idx[0]], grid[idx[1]], grid[idx[2]], grid[idx[3]],
            grid[idx[4]], grid[idx[5]], grid[idx[6]], grid[idx[7]]).AsByte();
        Avx.Store(idx + 8, Avx2.Or(lo2, hi2));
        g2 = Vector256.Create(grid[idx[8]], grid[idx[9]], grid[idx[10]], grid[idx[11]],
            grid[idx[12]], grid[idx[13]], grid[idx[14]], grid[idx[15]]).AsByte();
    }

    public static void Dot2Shared(byte* row, byte* s0, byte* s1, int cols, float* o)
    {
        int nb = cols / 256;
        float* d0 = (float*)s0; float* d1 = (float*)s1;
        sbyte* qa0 = (sbyte*)(s0 + nb * 4); sbyte* qa1 = (sbyte*)(s1 + nb * 4);
        var idxShift = Vector256.Create(8u, 7u, 6u, 5u, 4u, 3u, 2u, 1u);
        var idxMask = Vector256.Create(256u);
        uint* idx = stackalloc uint[16];
        var acc0 = Vector256<float>.Zero; var acc1 = Vector256<float>.Zero;
        fixed (uint* grid = IqCodebooks.Iq3SGrid)
        {
            for (int i = 0; i < nb; i++)
            {
                byte* blk = row + i * Bpb;
                float dh = HalfConv.ToFloat(blk);
                byte* qs = blk + 2, qh = blk + 66, signs = blk + 74, scales = blk + 106;
                sbyte* q0 = qa0 + i * 256; sbyte* q1 = qa1 + i * 256;
                var sum0 = Vector256<int>.Zero; var sum1 = Vector256<int>.Zero;
                for (int ib32 = 0; ib32 < 8; ib32 += 2)
                {
                    int sc = scales[ib32 >> 1];
                    var ls1 = Vector256.Create((short)(2 * (sc & 0xF) + 1));
                    var ls2 = Vector256.Create((short)(2 * (sc >> 4) + 1));
                    Grid(qs, qh[ib32], qh[ib32 + 1], ib32, grid, idx, idxShift, idxMask, out var g1, out var g2);
                    var n1 = NegMask(*(uint*)(signs + ib32 * 4));
                    var n2 = NegMask(*(uint*)(signs + ib32 * 4 + 4));
                    int o1 = ib32 * 32, o2 = ib32 * 32 + 32;
                    sum0 = Avx2.Add(sum0, Avx2.Add(Avx2.MultiplyAddAdjacent(Dot32(g1, n1, q0 + o1), ls1), Avx2.MultiplyAddAdjacent(Dot32(g2, n2, q0 + o2), ls2)));
                    sum1 = Avx2.Add(sum1, Avx2.Add(Avx2.MultiplyAddAdjacent(Dot32(g1, n1, q1 + o1), ls1), Avx2.MultiplyAddAdjacent(Dot32(g2, n2, q1 + o2), ls2)));
                }
                acc0 = Fma.MultiplyAdd(Vector256.Create(dh * d0[i]), Avx.ConvertToVector256Single(sum0), acc0);
                acc1 = Fma.MultiplyAdd(Vector256.Create(dh * d1[i]), Avx.ConvertToVector256Single(sum1), acc1);
            }
        }
        o[0] = HSum(acc0); o[1] = HSum(acc1);
    }

    public static void Dot4Shared(byte* row, byte* s0, byte* s1, byte* s2, byte* s3, int cols, float* o)
    {
        int nb = cols / 256;
        float* d0 = (float*)s0; float* d1 = (float*)s1; float* d2 = (float*)s2; float* d3 = (float*)s3;
        sbyte* qa0 = (sbyte*)(s0 + nb * 4); sbyte* qa1 = (sbyte*)(s1 + nb * 4);
        sbyte* qa2 = (sbyte*)(s2 + nb * 4); sbyte* qa3 = (sbyte*)(s3 + nb * 4);
        var idxShift = Vector256.Create(8u, 7u, 6u, 5u, 4u, 3u, 2u, 1u);
        var idxMask = Vector256.Create(256u);
        uint* idx = stackalloc uint[16];
        var acc0 = Vector256<float>.Zero; var acc1 = Vector256<float>.Zero;
        var acc2 = Vector256<float>.Zero; var acc3 = Vector256<float>.Zero;
        fixed (uint* grid = IqCodebooks.Iq3SGrid)
        {
            for (int i = 0; i < nb; i++)
            {
                byte* blk = row + i * Bpb;
                float dh = HalfConv.ToFloat(blk);
                byte* qs = blk + 2, qh = blk + 66, signs = blk + 74, scales = blk + 106;
                sbyte* q0 = qa0 + i * 256; sbyte* q1 = qa1 + i * 256; sbyte* q2 = qa2 + i * 256; sbyte* q3 = qa3 + i * 256;
                var sum0 = Vector256<int>.Zero; var sum1 = Vector256<int>.Zero;
                var sum2 = Vector256<int>.Zero; var sum3 = Vector256<int>.Zero;
                for (int ib32 = 0; ib32 < 8; ib32 += 2)
                {
                    int sc = scales[ib32 >> 1];
                    var ls1 = Vector256.Create((short)(2 * (sc & 0xF) + 1));
                    var ls2 = Vector256.Create((short)(2 * (sc >> 4) + 1));
                    Grid(qs, qh[ib32], qh[ib32 + 1], ib32, grid, idx, idxShift, idxMask, out var g1, out var g2);
                    var n1 = NegMask(*(uint*)(signs + ib32 * 4));
                    var n2 = NegMask(*(uint*)(signs + ib32 * 4 + 4));
                    int o1 = ib32 * 32, o2 = ib32 * 32 + 32;
                    sum0 = Avx2.Add(sum0, Avx2.Add(Avx2.MultiplyAddAdjacent(Dot32(g1, n1, q0 + o1), ls1), Avx2.MultiplyAddAdjacent(Dot32(g2, n2, q0 + o2), ls2)));
                    sum1 = Avx2.Add(sum1, Avx2.Add(Avx2.MultiplyAddAdjacent(Dot32(g1, n1, q1 + o1), ls1), Avx2.MultiplyAddAdjacent(Dot32(g2, n2, q1 + o2), ls2)));
                    sum2 = Avx2.Add(sum2, Avx2.Add(Avx2.MultiplyAddAdjacent(Dot32(g1, n1, q2 + o1), ls1), Avx2.MultiplyAddAdjacent(Dot32(g2, n2, q2 + o2), ls2)));
                    sum3 = Avx2.Add(sum3, Avx2.Add(Avx2.MultiplyAddAdjacent(Dot32(g1, n1, q3 + o1), ls1), Avx2.MultiplyAddAdjacent(Dot32(g2, n2, q3 + o2), ls2)));
                }
                acc0 = Fma.MultiplyAdd(Vector256.Create(dh * d0[i]), Avx.ConvertToVector256Single(sum0), acc0);
                acc1 = Fma.MultiplyAdd(Vector256.Create(dh * d1[i]), Avx.ConvertToVector256Single(sum1), acc1);
                acc2 = Fma.MultiplyAdd(Vector256.Create(dh * d2[i]), Avx.ConvertToVector256Single(sum2), acc2);
                acc3 = Fma.MultiplyAdd(Vector256.Create(dh * d3[i]), Avx.ConvertToVector256Single(sum3), acc3);
            }
        }
        o[0] = HSum(acc0); o[1] = HSum(acc1); o[2] = HSum(acc2); o[3] = HSum(acc3);
    }

    // ---- predecoded diagnostic ---------------------------------------------------------------------------------
    // Layout: cols bytes of signed int8 weights (grid value with sign applied), then cols/32 group scales (1 byte each,
    // already the odd 2*s+1 multiplier) and the per-256 super-block half scale is folded into a float table after.
    // Only row 0 is predecoded (the diagnostic is run in hot mode); half scales are stored in a trailing float array.
    private static float[] s_preD = new float[64];

    public static void Predecode(byte* row, int cols, byte* dst)
    {
        int nb = cols / 256;
        var grid = IqCodebooks.Iq3SGrid;
        sbyte* wq = (sbyte*)dst;
        byte* sc = dst + cols;
        if (s_preD.Length < nb) s_preD = new float[nb];
        for (int i = 0; i < nb; i++)
        {
            byte* blk = row + i * Bpb;
            s_preD[i] = HalfConv.ToFloat(blk);
            byte* qs = blk + 2, qh = blk + 66, signs = blk + 74, scales = blk + 106;
            for (int ib32 = 0; ib32 < 8; ib32++)
            {
                int s = scales[ib32 >> 1];
                sc[i * 8 + ib32] = (byte)((ib32 & 1) == 0 ? 2 * (s & 0xF) + 1 : 2 * (s >> 4) + 1);
                for (int l = 0; l < 4; l++) // 4 x 8 weights = 32
                {
                    int lane = ib32 * 4 + l;
                    // 8 weights per lane group: index = qs[...] | (qh bit) << 8 -> uint grid with 4 bytes; two grids per 8 weights
                    int hiBits = qh[ib32];
                    for (int h = 0; h < 2; h++)
                    {
                        int qIdx = ib32 * 8 + l * 2 + h;
                        int gi = qs[qIdx] | (((hiBits >> (l * 2 + h)) & 1) << 8);
                        uint g = grid[gi];
                        for (int j = 0; j < 4; j++)
                        {
                            int wPos = i * 256 + ib32 * 32 + l * 8 + h * 4 + j;
                            int mag = (int)((g >> (8 * j)) & 0xFF);
                            int signBit = (signs[ib32 * 4 + l] >> (h * 4 + j)) & 1;
                            wq[wPos] = (sbyte)(signBit != 0 ? -mag : mag);
                        }
                    }
                }
            }
        }
    }

    public static float DotPre(byte* pre, byte* scratch, int cols)
    {
        int nb = cols / 256;
        float* dArr = (float*)scratch;
        sbyte* qsArr = (sbyte*)(scratch + nb * 4);
        sbyte* wq = (sbyte*)pre;
        byte* sc = pre + cols;
        var acc = Vector256<float>.Zero;
        for (int i = 0; i < nb; i++)
        {
            var sum = Vector256<int>.Zero;
            for (int g = 0; g < 8; g++)
            {
                var w = Avx.LoadVector256(wq + i * 256 + g * 32);
                var a = Avx.LoadVector256(qsArr + i * 256 + g * 32);
                var absW = Avx2.Abs(w);
                var qs = Avx2.Sign(a, w);
                var p = Avx2.MultiplyAddAdjacent(absW, qs);
                sum = Avx2.Add(sum, Avx2.MultiplyAddAdjacent(p, Vector256.Create((short)sc[i * 8 + g])));
            }
            acc = Fma.MultiplyAdd(Vector256.Create(s_preD[i] * dArr[i]), Avx.ConvertToVector256Single(sum), acc);
        }
        return HSum(acc);
    }
}
