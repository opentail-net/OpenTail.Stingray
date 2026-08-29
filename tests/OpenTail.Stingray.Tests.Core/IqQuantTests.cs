
namespace OpenTail.Stingray.Tests.Core;

public class IqQuantTests
{
    [Fact]
    public void IqCodebooks_GridSizesAndValuesAreValid()
    {
        // These are ggml's real reference tables (examples/ggml/src/ggml-common.h's
        // iq2xxs_grid/iq3xxs_grid/iq3s_grid), not the previous fabricated bit-pattern grids --
        // each entry packs its N signed-magnitude bytes into one integer (byte j is
        // (value >> (8*j)), matching ggml's little-endian pointer-cast reading order).
        Assert.Equal(256, IqCodebooks.Iq2XxsGrid.Length);
        Assert.Equal(256, IqCodebooks.Iq3XxsGrid.Length);
        Assert.Equal(512, IqCodebooks.Iq3SGrid.Length);

        // IQ4_XS is not grid-based at all -- it shares IQ4_NL's 16-entry non-linear codebook.
        Assert.Equal(16, IqCodebooks.Iq4NlCodebook.Length);
        Assert.Equal(128, IqCodebooks.KSignsIq2Xs.Length);
        Assert.Equal(8, IqCodebooks.KMaskIq2Xs.Length);

        // Spot-check a couple of entries against ggml's own table literals.
        Assert.Equal(0x0808080808080808UL, IqCodebooks.Iq2XxsGrid[0]);
        Assert.Equal(0x04040404U, IqCodebooks.Iq3XxsGrid[0]);
        Assert.Equal(0x01010101U, IqCodebooks.Iq3SGrid[0]);
    }

    [Theory]
    [InlineData(DType.IQ2_XXS, 66)]
    [InlineData(DType.IQ3_XXS, 98)]
    [InlineData(DType.IQ3_S, 110)]
    [InlineData(DType.IQ4_XS, 136)]
    public void Dequantize_IQFormats_DequantizesBlockToFloat32WithoutNaN(DType dtype, int bytesPerBlock)
    {
        const int elementsPerBlock = 256;
        byte[] block = new byte[bytesPerBlock];

        // Set d scale float16 bytes to 1.0f (0x3C00)
        block[0] = 0x00;
        block[1] = 0x3C;

        for (int i = 2; i < bytesPerBlock; i++)
        {
            block[i] = (byte)(i % 256);
        }

        float[] output = new float[elementsPerBlock];
        Dequantize.ToFloat32(block, output, dtype, elementsPerBlock);

        for (int i = 0; i < elementsPerBlock; i++)
        {
            Assert.False(float.IsNaN(output[i]), $"Element {i} was NaN for {dtype}");
            Assert.False(float.IsInfinity(output[i]), $"Element {i} was Infinity for {dtype}");
        }
    }

    /// <summary>
    /// Edge cases mined from ggml's own correctness suite (examples/ggml/tests/test-quantize-fns.cpp):
    /// its <c>generate_data</c> seeds test blocks with a smooth cosine wave (<c>0.1 + 2*cos(i+offset)</c>)
    /// rather than a raw byte ramp, and its <c>total_quantization_error</c>/<c>dot_product_error</c>
    /// checks assert a bounded RMSE per dtype (2-bit budget 0.0075, 3-bit 0.004/0.005 for XXS).
    ///
    /// This engine has no IQ-format quantizer (from_float) to round-trip against, only the
    /// dequantizer fixed this session, so a literal round-trip-RMSE port isn't possible here.
    /// What IS portable and worth mining: ggml's cosine-seeded byte pattern (not a linear ramp,
    /// which can accidentally hit only one codebook/grid index repeatedly and mask decode bugs)
    /// and a coarse magnitude-sanity bound -- with d=1.0, no IQ2/3/4 dequantized value should ever
    /// exceed the largest possible per-group scale times the largest codebook/grid magnitude,
    /// which catches gross index/shift errors that the NaN-only check above would miss.
    /// </summary>
    [Theory]
    [InlineData(DType.IQ2_XXS, 66, 4.0f)]     // db = d*(0.5+q)*0.25, q<=15 -> db<=4.0; grid entries are ±1/±2-scale bytes (max ~2 per elem after grid decode, bounded by db*3)
    [InlineData(DType.IQ3_XXS, 98, 8.0f)]      // db = d*(0.5+q)*0.5, q<=15 -> db<=8.0
    [InlineData(DType.IQ3_S, 110, 4.0f)]       // linear per-group scale, 4-bit nibble -> up to ~4x d
    [InlineData(DType.IQ4_XS, 136, 32.0f)]     // dl = d*(ls-32), ls in [0,63] -> up to 31*d magnitude, codebook max 113
    public void Dequantize_IQFormats_CosineSeededBlock_StaysWithinMagnitudeBound(
        DType dtype, int bytesPerBlock, float scaleBoundMultiplier)
    {
        const int elementsPerBlock = 256;
        byte[] block = new byte[bytesPerBlock];

        // d = 1.0f (float16 0x3C00), matching ggml's convention of a unit reference scale so the
        // magnitude bound below is directly comparable to the codebook's own largest entries.
        block[0] = 0x00;
        block[1] = 0x3C;

        // ggml's generate_data(offset, n, dst): dst[i] = 0.1 + 2*cos(i + offset). Ported here as a
        // deterministic, smoothly-varying byte fill (rather than the i%256 ramp used by the NaN
        // test above) so every index/shift/mask path in the four decoders gets exercised with
        // non-repeating, non-degenerate byte values instead of a pattern that could alias.
        for (int i = 2; i < bytesPerBlock; i++)
        {
            double cosineValue = 0.1 + 2.0 * System.Math.Cos(i + 0.0);
            block[i] = unchecked((byte)(int)System.Math.Round((cosineValue + 2.1) / 4.2 * 255.0));
        }

        float[] output = new float[elementsPerBlock];
        Dequantize.ToFloat32(block, output, dtype, elementsPerBlock);

        // IQ4_NL/XS codebook's largest-magnitude entry is 113 (see IqCodebooks.Iq4NlCodebook);
        // the IQ2/3 grids are built from small integers (roughly {1,2,3} per byte) scaled by db.
        float bound = scaleBoundMultiplier * 130.0f;
        for (int i = 0; i < elementsPerBlock; i++)
        {
            Assert.False(float.IsNaN(output[i]), $"Element {i} was NaN for {dtype}");
            Assert.True(System.Math.Abs(output[i]) <= bound,
                $"Element {i} = {output[i]} exceeded magnitude bound {bound} for {dtype} -- likely an index/shift decode error");
        }
    }
}
