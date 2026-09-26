namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>
/// Parity for <see cref="Q6KPrefillGemm"/> (the group-paired Q6_K prefill GEMM) against the
/// stock row-major <see cref="SimdKernels.DotQ6K_Q8K"/> on the same Q8_K activations. The integer
/// dot per super-block is exact in both; only the float lane accumulation order differs, so results
/// agree to float rounding — a permutation or scale-pairing bug shows up as O(1) relative error.
/// Pure-kernel test: synthetic weights, no model file.
/// </summary>
public sealed unsafe class Q6KPrefillGemmTests
{
    [Theory]
    [InlineData(1, 16, 256)]
    [InlineData(3, 40, 512)]
    [InlineData(17, 33, 2048)]
    [InlineData(64, 64, 1024)]
    [InlineData(600, 24, 768)]
    public void PairedGemm_MatchesRowMajorQ8Dot(int batch, int rows, int cols)
    {
        if (!Q6KPrefillGemm.CanUse(rows, cols)) return;
        var rng = new Random(batch * 131 + rows * 7 + cols);
        int bytesPerRow = cols / 256 * 210;
        var w = new byte[rows * bytesPerRow];
        rng.NextBytes(w);
        for (int off = 0; off + 210 <= w.Length; off += 210)
        {
            var h = BitConverter.HalfToUInt16Bits((Half)(0.004f + 0.01f * (float)rng.NextDouble()));
            w[off + 208] = (byte)h; w[off + 209] = (byte)(h >> 8);
        }
        var x = new float[batch * cols];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1) * (i % 97 == 0 ? 8f : 1f);

        var got = new float[batch * rows];
        var want = new float[batch * rows];
        var scratch = new byte[SimdKernels.Q8KScratchBytes(cols)];

        fixed (byte* pw = w) fixed (float* px = x)
        fixed (float* pg = got) fixed (float* pr = want) fixed (byte* ps = scratch)
        {
            Assert.True(Q6KPrefillGemm.TryMatMulBatched(pg, pw, px, batch, rows, cols));
            for (int n = 0; n < batch; n++)
            {
                SimdKernels.QuantizeRowToQ8K(px + (long)n * cols, cols, ps);
                for (int r = 0; r < rows; r++)
                    pr[(long)n * rows + r] = SimdKernels.DotQ6K_Q8K(pw + (long)r * bytesPerRow, ps, cols);
            }
        }

        double worst = 0, scale = 0;
        for (int i = 0; i < want.Length; i++) scale = Math.Max(scale, Math.Abs(want[i]));
        for (int i = 0; i < want.Length; i++) worst = Math.Max(worst, Math.Abs(got[i] - want[i]));
        Assert.True(scale > 0);
        Assert.True(worst <= 1e-5 * scale, $"worst abs diff {worst} vs scale {scale}");
    }
}
