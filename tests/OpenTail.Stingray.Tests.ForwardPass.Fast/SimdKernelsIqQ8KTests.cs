
namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>
/// AVX2-vs-scalar equivalence for the six Q8_K-paired IQ matvec kernels added 2026-08-28
/// (docs/05-cpu-architecture-kernel-opportunities.md) — IQ4_XS, IQ2_XS, IQ2_S, IQ3_XXS, IQ2_XXS,
/// IQ3_S. Mirrors the same pattern <c>SimdKernelsQ8KSTests</c> already uses for Q3_K/Q4_K/Q8_0's
/// Q8_KS-paired kernels: fill a block with random bytes (grid-index / sign / scale fields are all
/// derived from bits that stay in range regardless of value, so random data cannot crash the
/// kernel — it exercises every code path, not just "plausible" quantized weights), force the
/// FP16 scale field to a small positive value to avoid NaN propagation, then assert the AVX2
/// dispatcher's output matches the internal scalar reference within a tight tolerance.
/// </summary>
public sealed unsafe class SimdKernelsIqQ8KTests
{
    private static ushort HalfToUshort(Half h) => BitConverter.HalfToUInt16Bits(h);

    private static byte[] BuildIqMatrix(int rows, int cols, int bytesPerBlock, Random rng)
    {
        if ((cols & 0xff) != 0)
            throw new ArgumentException("cols must be a multiple of 256.");
        int blocksPerRow = cols / 256;
        int bytesPerRow = blocksPerRow * bytesPerBlock;
        var bytes = new byte[rows * bytesPerRow];

        for (int r = 0; r < rows; r++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                int off = r * bytesPerRow + b * bytesPerBlock;
                for (int i = 2; i < bytesPerBlock; i++)
                    bytes[off + i] = (byte)rng.Next(256);
                float d = (float)(rng.NextDouble() * 0.09 + 0.01);
                ushort dHalf = HalfToUshort((Half)d);
                bytes[off] = (byte)(dHalf & 0xFF);
                bytes[off + 1] = (byte)(dHalf >> 8);
            }
        }
        return bytes;
    }

    private static void AssertAvx2MatchesScalar(
        string label, int bytesPerBlock,
        Func<nint, nint, int, float> avxDot, Func<nint, nint, int, float> scalarDot)
    {
        if (!Avx2.IsSupported || !Avx.IsSupported || !Fma.IsSupported) return;

        foreach ((int rows, int cols) in new[] { (4, 256), (5, 512), (8, 2048), (3, 4096) })
        {
            var rng = new Random(unchecked((int)0x108B5EED) ^ (rows * 131 + cols) ^ label.GetHashCode());
            byte[] weightBytes = BuildIqMatrix(rows, cols, bytesPerBlock, rng);

            var input = new float[cols];
            for (int i = 0; i < cols; i++)
                input[i] = (float)(rng.NextDouble() * 2 - 1);

            int scratchBytes = SimdKernels.Q8KScratchBytes(cols);
            var scratch = new byte[scratchBytes];

            var avxOut = new float[rows];
            var scalarOut = new float[rows];

            fixed (byte* wPtr = weightBytes)
            fixed (byte* sPtr = scratch)
            fixed (float* iPtr = input)
            {
                SimdKernels.QuantizeRowToQ8K(iPtr, cols, sPtr);
                int bytesPerRow = (cols / 256) * bytesPerBlock;
                for (int r = 0; r < rows; r++)
                {
                    nint rowPtr = (nint)(wPtr + (long)r * bytesPerRow);
                    avxOut[r] = avxDot(rowPtr, (nint)sPtr, cols);
                    scalarOut[r] = scalarDot(rowPtr, (nint)sPtr, cols);
                }
            }

            int mismatches = 0;
            float maxAbs = 0, maxRel = 0;
            for (int r = 0; r < rows; r++)
            {
                float diff = MathF.Abs(avxOut[r] - scalarOut[r]);
                float rel = diff / (MathF.Abs(scalarOut[r]) + 1e-6f);
                if (diff > maxAbs) maxAbs = diff;
                if (rel > maxRel) maxRel = rel;
                if (diff > 1e-3f && rel > 1e-3f) mismatches++;
            }
            Console.WriteLine(
                $"{label} avx-vs-scalar rows={rows} cols={cols}: maxAbs={maxAbs:E2} maxRel={maxRel:E2} mismatches={mismatches}/{rows}");
            Assert.True(mismatches == 0,
                $"{label} AVX2 vs scalar mismatch ({mismatches}/{rows}) rows={rows} cols={cols}, maxAbs={maxAbs:E3}, maxRel={maxRel:E3}");
        }
    }

    [Fact]
    public void DotIq4Xs_Q8K_Avx2MatchesScalar() =>
        AssertAvx2MatchesScalar("IQ4_XS", 136,
            (row, s, c) => SimdKernels.DotIq4Xs_Q8K((byte*)row, (byte*)s, c),
            (row, s, c) => SimdKernels.DotIq4Xs_Q8K_Scalar((byte*)row, (byte*)s, c));

    [Fact]
    public void DotIq2Xs_Q8K_Avx2MatchesScalar() =>
        AssertAvx2MatchesScalar("IQ2_XS", 74,
            (row, s, c) => SimdKernels.DotIq2Xs_Q8K((byte*)row, (byte*)s, c),
            (row, s, c) => SimdKernels.DotIq2Xs_Q8K_Scalar((byte*)row, (byte*)s, c));

    [Fact]
    public void DotIq2S_Q8K_Avx2MatchesScalar() =>
        AssertAvx2MatchesScalar("IQ2_S", 82,
            (row, s, c) => SimdKernels.DotIq2S_Q8K((byte*)row, (byte*)s, c),
            (row, s, c) => SimdKernels.DotIq2S_Q8K_Scalar((byte*)row, (byte*)s, c));

    [Fact]
    public void DotIq3Xxs_Q8K_Avx2MatchesScalar() =>
        AssertAvx2MatchesScalar("IQ3_XXS", 98,
            (row, s, c) => SimdKernels.DotIq3Xxs_Q8K((byte*)row, (byte*)s, c),
            (row, s, c) => SimdKernels.DotIq3Xxs_Q8K_Scalar((byte*)row, (byte*)s, c));

    [Fact]
    public void DotIq2Xxs_Q8K_Avx2MatchesScalar() =>
        AssertAvx2MatchesScalar("IQ2_XXS", 66,
            (row, s, c) => SimdKernels.DotIq2Xxs_Q8K((byte*)row, (byte*)s, c),
            (row, s, c) => SimdKernels.DotIq2Xxs_Q8K_Scalar((byte*)row, (byte*)s, c));

    [Fact]
    public void DotIq3S_Q8K_Avx2MatchesScalar() =>
        AssertAvx2MatchesScalar("IQ3_S", 110,
            (row, s, c) => SimdKernels.DotIq3S_Q8K((byte*)row, (byte*)s, c),
            (row, s, c) => SimdKernels.DotIq3S_Q8K_Scalar((byte*)row, (byte*)s, c));

    /// <summary>
    /// Decode-shape throughput of <see cref="SimdKernels.MatVec"/> per weight dtype at Qwen3.8-27B's FFN shape
    /// (17408 × 5120), GB/s of weight bytes streamed, Q4_K as the baseline. Heavy: <c>STINGRAY_RUN_HEAVY_TESTS=1</c>.
    /// </summary>
    [Fact]
    public void MatVecThroughput_ByDType()
    {
        Assert.SkipUnless(Environment.GetEnvironmentVariable("STINGRAY_RUN_HEAVY_TESTS") == "1", "benchmark: set STINGRAY_RUN_HEAVY_TESTS=1");
        const int rows = 17408, cols = 5120;
        var input = new float[cols];
        var rng = new Random(7);
        for (int i = 0; i < cols; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);
        var output = new float[rows];
        foreach (var (dt, bpb) in new[] { (DType.Q4_K, 144), (DType.Q3_K, 110), (DType.IQ4_XS, 136), (DType.IQ3_S, 110),
                     (DType.IQ3_XXS, 98), (DType.IQ2_S, 82), (DType.IQ2_XS, 74), (DType.IQ2_XXS, 66) })
        {
            byte[] w = BuildIqMatrix(rows, cols, bpb, rng);
            fixed (byte* wp = w)
            fixed (float* ip = input, op = output)
            {
                SimdKernels.MatVec(op, wp, ip, rows, cols, dt);
                var times = new List<double>();
                for (int r = 0; r < 7; r++)
                {
                    var sw = System.Diagnostics.Stopwatch.StartNew();
                    SimdKernels.MatVec(op, wp, ip, rows, cols, dt);
                    times.Add(sw.Elapsed.TotalMilliseconds);
                }
                times.Sort();
                double ms = times[3];
                Console.WriteLine($"[MatVecBench] {dt,-8} {ms,8:F2} ms  {w.Length / 1e6 / ms,7:F1} GB/s");
            }
        }
    }

    /// <summary>
    /// Independent reference for every Q8_K-paired IQ kernel: <see cref="Dequantize.ToFloat32"/> (the path that produced
    /// the Qwen3.8-27B llama.cpp greedy receipt) followed by a float dot. AVX2-vs-scalar agreement alone cannot catch a
    /// mistake both variants share: IQ3_S shipped 2026-08-28 returning <c>0.25f * sumf</c> (a factor that belongs to
    /// IQ3_XXS; ggml's <c>ggml_vec_dot_iq3_s_q8_K</c> returns <c>sumf</c>), which made every IQ3_S matmul 4x too small
    /// and broke Qwen3.8-27B's output until found by bisect on 2026-09-25. Tolerance covers Q8_K activation rounding.
    /// </summary>
    [Theory]
    [InlineData(DType.IQ4_XS, 136)]
    [InlineData(DType.IQ2_XS, 74)]
    [InlineData(DType.IQ2_S, 82)]
    [InlineData(DType.IQ3_XXS, 98)]
    [InlineData(DType.IQ2_XXS, 66)]
    [InlineData(DType.IQ3_S, 110)]
    public void IqQ8KDot_MatchesDequantizedFloatDot(DType dtype, int bytesPerBlock)
    {
        Func<nint, nint, int, float> dot = dtype switch
        {
            DType.IQ4_XS => (r, s, c) => SimdKernels.DotIq4Xs_Q8K((byte*)r, (byte*)s, c),
            DType.IQ2_XS => (r, s, c) => SimdKernels.DotIq2Xs_Q8K((byte*)r, (byte*)s, c),
            DType.IQ2_S => (r, s, c) => SimdKernels.DotIq2S_Q8K((byte*)r, (byte*)s, c),
            DType.IQ3_XXS => (r, s, c) => SimdKernels.DotIq3Xxs_Q8K((byte*)r, (byte*)s, c),
            DType.IQ2_XXS => (r, s, c) => SimdKernels.DotIq2Xxs_Q8K((byte*)r, (byte*)s, c),
            _ => (r, s, c) => SimdKernels.DotIq3S_Q8K((byte*)r, (byte*)s, c),
        };
        const int rows = 6, cols = 1024;
        var rng = new Random(12345 + (int)dtype);
        byte[] w = BuildIqMatrix(rows, cols, bytesPerBlock, rng);
        var input = new float[cols];
        for (int i = 0; i < cols; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);
        var scratch = new byte[SimdKernels.Q8KScratchBytes(cols)];
        int bytesPerRow = cols / 256 * bytesPerBlock;
        var deq = new float[cols];
        fixed (byte* wp = w)
        fixed (byte* sp = scratch)
        fixed (float* ip = input)
        {
            SimdKernels.QuantizeRowToQ8K(ip, cols, sp);
            for (int r = 0; r < rows; r++)
            {
                Dequantize.ToFloat32(w.AsSpan(r * bytesPerRow, bytesPerRow), deq, dtype, cols);
                float expected = 0, scale = 0;
                for (int i = 0; i < cols; i++) { expected += deq[i] * input[i]; scale += MathF.Abs(deq[i] * input[i]); }
                float got = dot((nint)(wp + (long)r * bytesPerRow), (nint)sp, cols);
                Assert.True(MathF.Abs(got - expected) <= 0.02f * scale + 1e-4f,
                    $"{dtype} row {r}: kernel {got} vs dequantized dot {expected} (sum|terms| {scale})");
            }
        }
    }
}
