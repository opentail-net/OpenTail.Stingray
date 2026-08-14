using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>
/// Contract for <see cref="SimdKernels.BatchedMatVecTierEnabled"/> — the fp32 multi-input tiering
/// in <see cref="SimdKernels.MatMulBatched"/>'s non-BLAS fallback (session runtime plan §3.4.6).
///
/// <para><b>Why the existing coverage is not enough.</b>
/// <c>SimdKernelsQ8KSTests.MatVec4In_BitwiseMatchesSingleMatVec</c> already proves
/// <see cref="SimdKernels.MatVec4In"/> is bit-identical to four single
/// <see cref="SimdKernels.MatVec"/> calls. What it does NOT cover is the composite
/// quads → pairs → remainder walk added here, whose slot-to-output pointer arithmetic is new code.
/// A batch of 7 becomes one quad, one pair and one single, and every one of those boundaries is a
/// place to mis-map a token to the wrong output row. That mistake produces perfectly plausible
/// numbers — every value present, just attributed to the wrong sequence — which in a multi-user
/// server means one user receiving another's logits.</para>
///
/// <para>The oracle is the same function with the tier switched off, so the two arms differ in
/// nothing but the walk. The contract is bit equality: the tiering is a scheduling change and is
/// required to be arithmetically inert.</para>
/// </summary>
public sealed unsafe class BatchedMatVecTierTests
{
    private static ushort HalfToUshort(Half h) => BitConverter.HalfToUInt16Bits(h);

    private static byte[] BuildQ4KMatrix(int rows, int cols, Random rng)
    {
        int blocksPerRow = cols / 256;
        int bytesPerRow = blocksPerRow * 144;
        var bytes = new byte[rows * bytesPerRow];
        for (int r = 0; r < rows; r++)
            for (int b = 0; b < blocksPerRow; b++)
            {
                int off = r * bytesPerRow + b * 144;
                float d = (float)(rng.NextDouble() * 0.05 + 0.005);
                float dmin = (float)(rng.NextDouble() * 0.03 + 0.005);
                ushort dh = HalfToUshort((Half)d), dmh = HalfToUshort((Half)dmin);
                bytes[off + 0] = (byte)(dh & 0xFF); bytes[off + 1] = (byte)(dh >> 8);
                bytes[off + 2] = (byte)(dmh & 0xFF); bytes[off + 3] = (byte)(dmh >> 8);
                for (int i = 0; i < 12; i++) bytes[off + 4 + i] = (byte)rng.Next(256);
                for (int i = 0; i < 128; i++) bytes[off + 16 + i] = (byte)rng.Next(256);
            }
        return bytes;
    }

    private static byte[] BuildF32Matrix(int rows, int cols, Random rng)
    {
        var f = new float[rows * cols];
        for (int i = 0; i < f.Length; i++) f[i] = (float)((rng.NextDouble() - 0.5) * 0.5);
        var bytes = new byte[f.Length * sizeof(float)];
        Buffer.BlockCopy(f, 0, bytes, 0, bytes.Length);
        return bytes;
    }

    private static float[] RunBatched(byte[] weights, float[] input, int batch, int rows, int cols,
        DType dtype, bool tier)
    {
        bool prev = SimdKernels.BatchedMatVecTierEnabled;
        SimdKernels.BatchedMatVecTierEnabled = tier;
        try
        {
            var output = new float[(long)batch * rows];
            fixed (byte* w = weights)
            fixed (float* inp = input)
            fixed (float* o = output)
                SimdKernels.MatMulBatched(o, w, inp, batch, rows, cols, dtype, allowQ8: false);
            return output;
        }
        finally { SimdKernels.BatchedMatVecTierEnabled = prev; }
    }

    /// <summary>
    /// Batch sizes chosen to land on every tier boundary: 1/2/3 skip the quad entirely, 4 is one
    /// clean quad, 5-7 are quad+remainder in each shape, 8 is two quads, 9 is two quads plus a
    /// single. 15 stays below <c>MinBatchForBlas</c> so the fallback is still the path under test.
    /// </summary>
    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    [InlineData(6)]
    [InlineData(7)]
    [InlineData(8)]
    [InlineData(9)]
    [InlineData(15)]
    public void TieredFallback_IsBitIdenticalToSequentialMatVec_Q4K(int batch)
    {
        const int rows = 128, cols = 512;
        var rng = new Random(1234 + batch);
        var weights = BuildQ4KMatrix(rows, cols, rng);
        var input = new float[(long)batch * cols];
        for (int i = 0; i < input.Length; i++) input[i] = (float)((rng.NextDouble() - 0.5) * 2.0);

        var plain = RunBatched(weights, input, batch, rows, cols, DType.Q4_K, tier: false);
        var tiered = RunBatched(weights, input, batch, rows, cols, DType.Q4_K, tier: true);

        AssertBitIdentical(plain, tiered, batch, rows, $"Q4_K batch={batch}");
    }

    /// <summary>
    /// F32 weights take a different branch inside <see cref="SimdKernels.MatVec4In"/> than the
    /// register-tiled quantised kernels, so the walk is exercised against both.
    /// </summary>
    [Theory]
    [InlineData(3)]
    [InlineData(7)]
    [InlineData(8)]
    public void TieredFallback_IsBitIdenticalToSequentialMatVec_F32(int batch)
    {
        const int rows = 64, cols = 256;
        var rng = new Random(99 + batch);
        var weights = BuildF32Matrix(rows, cols, rng);
        var input = new float[(long)batch * cols];
        for (int i = 0; i < input.Length; i++) input[i] = (float)((rng.NextDouble() - 0.5) * 2.0);

        var plain = RunBatched(weights, input, batch, rows, cols, DType.Float32, tier: false);
        var tiered = RunBatched(weights, input, batch, rows, cols, DType.Float32, tier: true);

        AssertBitIdentical(plain, tiered, batch, rows, $"F32 batch={batch}");
    }

    /// <summary>
    /// The slot-mapping guard. Every batch row is given a DISTINCT input, so if the walk ever
    /// attributed one token's result to another token's output row the values would still all be
    /// present and plausible — only misfiled. This asserts each row is not merely correct but
    /// correct *and different from its neighbours*, so a swap cannot hide behind coincidence.
    /// </summary>
    [Fact]
    public void TieredFallback_DoesNotMisattributeSlots()
    {
        const int rows = 64, cols = 256, batch = 7;
        var rng = new Random(4242);
        var weights = BuildQ4KMatrix(rows, cols, rng);

        // Each token gets a strongly distinct input so its output row is unmistakable.
        var input = new float[(long)batch * cols];
        for (int t = 0; t < batch; t++)
            for (int c = 0; c < cols; c++)
                input[(long)t * cols + c] = (t + 1) * (1.0f + 0.001f * c);

        var tiered = RunBatched(weights, input, batch, rows, cols, DType.Q4_K, tier: true);

        for (int t = 0; t < batch; t++)
        {
            var single = new float[rows];
            fixed (byte* w = weights)
            fixed (float* inp = input)
            fixed (float* o = single)
                SimdKernels.MatVec(o, w, inp + (long)t * cols, rows, cols, DType.Q4_K);

            for (int r = 0; r < rows; r++)
                Assert.True(
                    BitConverter.SingleToInt32Bits(single[r])
                        == BitConverter.SingleToInt32Bits(tiered[(long)t * rows + r]),
                    $"token {t} row {r}: tiered={tiered[(long)t * rows + r]} single={single[r]} "
                    + "— a token's output must come from its own input, not a neighbour's.");
        }

        // Sanity: the rows really are distinguishable, so the assertion above has something to
        // catch. Without this a degenerate weight matrix could make every token's output equal.
        Assert.True(
            BitConverter.SingleToInt32Bits(tiered[0]) != BitConverter.SingleToInt32Bits(tiered[rows]),
            "test is vacuous: token 0 and token 1 produced identical outputs");
    }

    private static void AssertBitIdentical(float[] plain, float[] tiered, int batch, int rows, string label)
    {
        Assert.Equal(plain.Length, tiered.Length);
        for (int t = 0; t < batch; t++)
            for (int r = 0; r < rows; r++)
            {
                long i = (long)t * rows + r;
                if (BitConverter.SingleToInt32Bits(plain[i]) != BitConverter.SingleToInt32Bits(tiered[i]))
                    Assert.Fail($"{label}: token {t} row {r} tiered={tiered[i]} "
                        + $"(0x{BitConverter.SingleToInt32Bits(tiered[i]):X8}) != sequential={plain[i]} "
                        + $"(0x{BitConverter.SingleToInt32Bits(plain[i]):X8}). The tiering is a "
                        + "scheduling change and must be arithmetically inert.");
            }
    }
}
