using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>
/// A51 (performance item A11): the 2-row Q4_K Q8_KS kernel must give, for every row, the same float bits as the 1-row kernel.
/// If that holds, the decode path may switch to the 1-row kernel without changing any output. Random valid Q4_K rows and random
/// activations, compared bit for bit (not with a tolerance), at several column counts.
/// </summary>
public sealed unsafe class DotQ4K2RowBitIdentityTests
{
    private const int Rows = 64;

    [Theory]
    [InlineData(256)]
    [InlineData(2048)]
    [InlineData(5632)]
    public void DotQ4K_2Row_IsBitIdenticalToDotQ4K_Q8KS_OnEveryRow(int cols)
    {
        var rng = new Random(20241124 + cols);
        int bytesPerRow = (cols / 256) * 144;

        // Valid Q4_K row bytes: d and dmin are real half-precision values (small, positive, finite); scales and quants are random.
        byte[] weights = new byte[Rows * bytesPerRow];
        rng.NextBytes(weights);
        for (int r = 0; r < Rows; r++)
            for (int b = 0; b < cols / 256; b++)
            {
                int off = r * bytesPerRow + b * 144;
                byte[] d = BitConverter.GetBytes((Half)(rng.NextSingle() * 0.05f + 0.001f));
                byte[] dmin = BitConverter.GetBytes((Half)(rng.NextSingle() * 0.02f));
                weights[off] = d[0]; weights[off + 1] = d[1];
                weights[off + 2] = dmin[0]; weights[off + 3] = dmin[1];
            }

        float[] input = new float[cols];
        for (int i = 0; i < cols; i++) input[i] = (rng.NextSingle() * 2f - 1f) * (1f + rng.Next(8));

        int scratchBytes = SimdKernels.Q8KSScratchBytes(cols);
        byte[] scratch = new byte[scratchBytes];
        fixed (byte* w = weights)
        fixed (float* x = input)
        fixed (byte* s = scratch)
        {
            SimdKernels.QuantizeRowToQ8KS(x, cols, s);
            int nonZero = 0;
            for (int r = 0; r + 1 < Rows; r += 2)
            {
                byte* row0 = w + (long)r * bytesPerRow;
                byte* row1 = w + (long)(r + 1) * bytesPerRow;

                float single0 = SimdKernels.DotQ4K_Q8KS(row0, s, cols);
                float single1 = SimdKernels.DotQ4K_Q8KS(row1, s, cols);
                SimdKernels.DotQ4K_Q8KS_2Row(row0, row1, s, cols, out float pair0, out float pair1);

                Assert.Equal(BitConverter.SingleToInt32Bits(single0), BitConverter.SingleToInt32Bits(pair0));
                Assert.Equal(BitConverter.SingleToInt32Bits(single1), BitConverter.SingleToInt32Bits(pair1));
                if (single0 != 0f) nonZero++;
                if (single1 != 0f) nonZero++;
            }
            // Non-vacuity: identical zeros would also pass the comparison above.
            Assert.True(nonZero > Rows / 2, $"only {nonZero} of {Rows} outputs were non-zero; the comparison would be vacuous");
        }
    }
}
