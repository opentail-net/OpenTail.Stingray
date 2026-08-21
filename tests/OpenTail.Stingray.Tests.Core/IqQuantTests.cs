using System;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using Xunit;

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
}
