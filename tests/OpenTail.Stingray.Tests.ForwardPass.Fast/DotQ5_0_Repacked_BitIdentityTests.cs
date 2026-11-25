using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>
/// The Q5_0 repacked layout (24-byte blocks: float32 scale, qh, qs) must give bit-identical results to the original 22-byte
/// layout: the repack field by field, the per-row dot (dispatched, and the scalar path on its own), and the matvec at serial
/// and parallel row counts, including an odd row count. Random valid rows, plus rows holding inf, -inf, NaN, zero and
/// subnormal scales, compared bit for bit (not with a tolerance).
/// </summary>
public sealed unsafe class DotQ5_0_Repacked_BitIdentityTests
{
    private const int BlockBytes = 22;
    private const int RepackedBlockBytes = 24;

    private static byte[] RandomQ5_0Rows(Random rng, int rows, int cols, bool specialScales)
    {
        int blocks = cols / 32;
        byte[] w = new byte[rows * blocks * BlockBytes];
        rng.NextBytes(w);
        for (int blk = 0; blk * BlockBytes < w.Length; blk++)
        {
            int off = blk * BlockBytes;
            ushort scale;
            int k = blk % 97;
            if (specialScales && k == 3) scale = 0x7C00;        // +inf
            else if (specialScales && k == 11) scale = 0xFC00;  // -inf
            else if (specialScales && k == 19) scale = 0x7E00;  // NaN
            else if (specialScales && k == 27) scale = 0x0000;  // zero
            else if (specialScales && k == 35) scale = 0x0001;  // smallest subnormal
            else scale = BitConverter.HalfToUInt16Bits((Half)(rng.NextSingle() * 0.05f + 0.001f));
            w[off] = (byte)(scale & 0xFF);
            w[off + 1] = (byte)(scale >> 8);
        }
        return w;
    }

    private static float[] RandomActivations(Random rng, int cols)
    {
        float[] input = new float[cols];
        for (int i = 0; i < cols; i++) input[i] = (rng.NextSingle() * 2f - 1f) * (1f + rng.Next(8));
        return input;
    }

    private static byte[] Repack(byte[] original)
    {
        long blocks = original.Length / BlockBytes;
        byte[] repacked = new byte[blocks * RepackedBlockBytes];
        fixed (byte* s = original)
        fixed (byte* d = repacked)
            SimdKernels.RepackQ5_0Blocks(s, d, blocks);
        return repacked;
    }

    [Theory]
    [InlineData(896, false)]
    [InlineData(896, true)]
    [InlineData(2048, true)]
    [InlineData(4864, false)]
    public void Repack_CarriesTheSameScaleAndBits(int cols, bool specialScales)
    {
        var rng = new Random(20241125 + cols + (specialScales ? 1 : 0));
        byte[] original = RandomQ5_0Rows(rng, 4, cols, specialScales);
        byte[] repacked = Repack(original);
        long blocks = original.Length / BlockBytes;

        for (long b = 0; b < blocks; b++)
        {
            ushort scale = (ushort)(original[b * BlockBytes] | (original[b * BlockBytes + 1] << 8));
            int scaleBits = BitConverter.SingleToInt32Bits(HalfConv.ToFloat(scale));
            int repackedScaleBits = BitConverter.ToInt32(repacked, (int)(b * RepackedBlockBytes));
            Assert.Equal(scaleBits, repackedScaleBits);
            for (int i = 0; i < 20; i++)
                Assert.Equal(original[b * BlockBytes + 2 + i], repacked[b * RepackedBlockBytes + 4 + i]);
        }
    }

    [Theory]
    [InlineData(896, false)]
    [InlineData(896, true)]
    [InlineData(2048, true)]
    [InlineData(4864, false)]
    public void DotQ5_0_Q8_0_Repacked_IsBitIdenticalToOriginal_OnEveryRow(int cols, bool specialScales)
    {
        const int rows = 64;
        var rng = new Random(20241126 + cols + (specialScales ? 1 : 0));
        int bytesPerRow = (cols / 32) * BlockBytes;
        int bytesPerRowR = (cols / 32) * RepackedBlockBytes;
        byte[] original = RandomQ5_0Rows(rng, rows, cols, specialScales);
        byte[] repacked = Repack(original);
        float[] input = RandomActivations(rng, cols);
        byte[] scratch = new byte[SimdKernels.Q8_0ScratchBytes(cols)];
        int nonZero = 0;

        fixed (byte* w = original)
        fixed (byte* wr = repacked)
        fixed (float* x = input)
        fixed (byte* s = scratch)
        {
            SimdKernels.QuantizeRowToQ8_0(x, cols, s);
            for (int r = 0; r < rows; r++)
            {
                float a = SimdKernels.DotQ5_0_Q8_0(w + (long)r * bytesPerRow, s, cols);
                float b = SimdKernels.DotQ5_0_Q8_0_Repacked(wr + (long)r * bytesPerRowR, s, cols);
                Assert.Equal(BitConverter.SingleToInt32Bits(a), BitConverter.SingleToInt32Bits(b));
                if (a != 0f) nonZero++;
            }
        }
        // Non-vacuity: identical zeros would also pass the comparison above.
        Assert.True(nonZero > rows / 2, $"only {nonZero} of {rows} outputs were non-zero; the comparison would be vacuous");
    }

    [Theory]
    [InlineData(896, false)]
    [InlineData(896, true)]
    [InlineData(2048, true)]
    public void DotQ5_0_Q8_0_Repacked_Scalar_IsBitIdenticalToScalarOriginal_OnEveryRow(int cols, bool specialScales)
    {
        const int rows = 64;
        var rng = new Random(20241127 + cols + (specialScales ? 1 : 0));
        int bytesPerRow = (cols / 32) * BlockBytes;
        int bytesPerRowR = (cols / 32) * RepackedBlockBytes;
        byte[] original = RandomQ5_0Rows(rng, rows, cols, specialScales);
        byte[] repacked = Repack(original);
        float[] input = RandomActivations(rng, cols);
        byte[] scratch = new byte[SimdKernels.Q8_0ScratchBytes(cols)];

        fixed (byte* w = original)
        fixed (byte* wr = repacked)
        fixed (float* x = input)
        fixed (byte* s = scratch)
        {
            SimdKernels.QuantizeRowToQ8_0(x, cols, s);
            for (int r = 0; r < rows; r++)
            {
                float a = SimdKernels.DotQ5_0_Q8_0_Scalar(w + (long)r * bytesPerRow, s, cols);
                float b = SimdKernels.DotQ5_0_Q8_0_Repacked_Scalar(wr + (long)r * bytesPerRowR, s, cols);
                Assert.Equal(BitConverter.SingleToInt32Bits(a), BitConverter.SingleToInt32Bits(b));
            }
        }
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(63)]
    [InlineData(64)]
    [InlineData(65)]
    [InlineData(129)]
    public void MatVec_Q5_0_Repacked_IsBitIdenticalToOriginal_IncludingOddRowCounts(int rows)
    {
        const int cols = 896;
        var rng = new Random(20241128 + rows);
        byte[] original = RandomQ5_0Rows(rng, rows, cols, specialScales: rows % 2 == 1);
        byte[] repacked = Repack(original);
        float[] input = RandomActivations(rng, cols);
        float[] reference = new float[rows];
        float[] output = new float[rows];

        fixed (byte* w = original)
        fixed (byte* wr = repacked)
        fixed (float* x = input)
        fixed (float* refp = reference)
        fixed (float* o = output)
        {
            SimdKernels.MatVecQ5_0(refp, w, x, rows, cols);
            SimdKernels.MatVecQ5_0Repacked(o, wr, x, rows, cols);
            for (int r = 0; r < rows; r++)
                Assert.Equal(BitConverter.SingleToInt32Bits(reference[r]), BitConverter.SingleToInt32Bits(output[r]));
        }
        Assert.True(reference.Any(v => v != 0f), "every reference output is zero; the comparison would be vacuous");
    }
}
