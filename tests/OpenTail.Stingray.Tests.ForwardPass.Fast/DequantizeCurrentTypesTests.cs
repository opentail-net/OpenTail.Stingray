
namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>
/// Byte-layout anchors for the post-2025 llama.cpp storage types. These are deliberately
/// hand-constructed blocks rather than self-quantized data, so a shared encoder/decoder bug
/// cannot make the test pass. Expected values follow ggml-quants.c.
/// </summary>
public sealed class DequantizeCurrentTypesTests
{
    [Fact]
    public void Q1_0_DecodesSignedOneBitValues()
    {
        var source = new byte[18];
        source[1] = 0x3C; // IEEE FP16 1.0, little endian
        source[2] = 0b_0000_0001;
        var actual = new float[128];

        Dequantize.ToFloat32(source, actual, DType.Q1_0, actual.Length);

        Assert.Equal(1f, actual[0]);
        Assert.Equal(-1f, actual[1]);
        Assert.Equal(-1f, actual[127]);
    }

    [Fact]
    public void Q2_0_DecodesPackedTwoBitValues()
    {
        var source = new byte[18];
        source[1] = 0x40; // IEEE FP16 2.0, little endian
        source[2] = 0b_1110_0100; // 00, 01, 10, 11 => -1, 0, +1, +2
        var actual = new float[64];

        Dequantize.ToFloat32(source, actual, DType.Q2_0, actual.Length);

        Assert.Equal(-2f, actual[0]);
        Assert.Equal(0f, actual[1]);
        Assert.Equal(2f, actual[2]);
        Assert.Equal(4f, actual[3]);
    }

    [Fact]
    public void Nvfp4_DecodesPerSubBlockUe4M3ScalesAndE2M1Nibbles()
    {
        var source = new byte[36];
        // UE4M3 0x38 represents raw 1.0; ggml applies half-scale for its doubled E2M1 table.
        source[0] = source[1] = source[2] = source[3] = 0x38;
        source[4] = 0x91;  // low = +1, high = -1 in sub-block 0
        source[12] = 0xA2; // low = +2, high = -2 in sub-block 1
        var actual = new float[64];

        Dequantize.ToFloat32(source, actual, DType.NVFP4, actual.Length);

        Assert.Equal(0.5f, actual[0]);
        Assert.Equal(-0.5f, actual[8]);
        Assert.Equal(1f, actual[16]);
        Assert.Equal(-1f, actual[24]);
    }

    [Theory]
    [InlineData(DType.Q1_0, 127)]
    [InlineData(DType.Q2_0, 63)]
    [InlineData(DType.NVFP4, 63)]
    public void NewBlockFormats_RejectPartialBlocks(DType dtype, int count)
    {
        Assert.Throws<ArgumentException>(() =>
            Dequantize.ToFloat32(new byte[36], new float[count], dtype, count));
    }
}
