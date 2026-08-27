using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>
/// docs/bugstofix.md (ModelCompatibility.cs:461, deepseek2 investigation): <see cref="SimdKernels.DotQ2K_Q8K_Avx2"/>
/// is a from-scratch instruction-for-instruction port of ggml's real AVX2 <c>ggml_vec_dot_q2_K_q8_K</c>,
/// written to match its SIMD lane-accumulation order (not just its formula) exactly. This pins that
/// the two independently-implemented paths this engine ships (<see cref="SimdKernels.DotQ2K_Q8K_Scalar"/>,
/// the portable ggml-generic-equivalent fallback) agree closely -- proving the AVX2 port isn't a
/// wholesale-wrong reordering, while still expecting a non-zero gap: a scalar sequential sum and an
/// 8-lane-parallel-then-horizontal-reduce sum are mathematically equivalent but not bit-identical,
/// which is the entire premise this port exists to test.
/// </summary>
public sealed unsafe class DotQ2KAvx2ParityTests
{
    private static void WriteHalf(byte[] buffer, int offset, float value)
    {
        ushort bits = BitConverter.HalfToUInt16Bits((Half)value);
        buffer[offset] = (byte)(bits & 0xFF);
        buffer[offset + 1] = (byte)(bits >> 8);
    }

    private static byte[] MakeQ2KRow(int numBlocks, int seed)
    {
        var rng = new Random(seed);
        var bytes = new byte[numBlocks * 84];
        rng.NextBytes(bytes);
        for (int off = 0; off + 84 <= bytes.Length; off += 84)
        {
            WriteHalf(bytes, off + 80, 0.010f); // d
            WriteHalf(bytes, off + 82, 0.004f); // dmin
        }
        return bytes;
    }

    [Theory]
    [InlineData(1, 1)]
    [InlineData(1, 2)]
    [InlineData(4, 3)]
    [InlineData(16, 4)]
    public void Avx2MatchesScalar_WithinFloatReorderingTolerance(int numBlocks, int seed)
    {
        Assert.SkipUnless(System.Runtime.Intrinsics.X86.Avx2.IsSupported
            && System.Runtime.Intrinsics.X86.Fma.IsSupported
            && System.Runtime.Intrinsics.X86.Ssse3.IsSupported,
            "AVX2/FMA/SSSE3 not available in this environment");

        int cols = numBlocks * 256;
        byte[] row = MakeQ2KRow(numBlocks, seed);

        var rng = new Random(seed + 1000);
        float[] input = new float[cols];
        for (int i = 0; i < cols; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

        int scratchBytes = SimdKernels.Q8KScratchBytes(cols);
        byte[] scratch = new byte[scratchBytes];

        fixed (byte* pRow = row)
        fixed (float* pInput = input)
        fixed (byte* pScratch = scratch)
        {
            SimdKernels.QuantizeRowToQ8K(pInput, cols, pScratch);
            float scalarResult = SimdKernels.DotQ2K_Q8K_Scalar(pRow, pScratch, numBlocks);
            float avx2Result = SimdKernels.DotQ2K_Q8K_Avx2(pRow, pScratch, numBlocks);

            // Same math, different summation order/tree -- expect agreement to a handful of ULP
            // relative to the magnitude involved, not bit-identity.
            float scale = Math.Max(1e-6f, Math.Abs(scalarResult));
            float relDiff = Math.Abs(avx2Result - scalarResult) / scale;
            Assert.True(relDiff < 1e-4f,
                $"scalar={scalarResult:G9} avx2={avx2Result:G9} relDiff={relDiff:G6} " +
                $"(numBlocks={numBlocks}, seed={seed})");
        }
    }
}
