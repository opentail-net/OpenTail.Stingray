
namespace OpenTail.Stingray.Tests.Vision;

public sealed class Fp8ConverterTests
{
    [Theory]
    [InlineData(0.0f)]
    [InlineData(1.0f)]
    [InlineData(-1.0f)]
    [InlineData(2.0f)]
    [InlineData(0.5f)]
    [InlineData(-0.5f)]
    // Powers of two (mantissa = 0) round-trip exactly in E4M3; 100 is not representable
    // (nearest fp8 values around it are 96/104) and belongs in the precision test below.
    public void RoundTrip_ExactlyRepresentableValues_AreLossless(float value)
    {
        byte fp8 = OpenTail.Stingray.Diffusion.Fp8Converter.FloatToFp8E4M3(value);
        float back = OpenTail.Stingray.Diffusion.Fp8Converter.Fp8E4M3ToFloat(fp8);
        Assert.Equal(value, back, precision: 5);
    }

    [Fact]
    public void FloatToFp8_PositiveZero_RoundTripsToZero()
    {
        byte fp8 = OpenTail.Stingray.Diffusion.Fp8Converter.FloatToFp8E4M3(0.0f);
        Assert.Equal(0.0f, OpenTail.Stingray.Diffusion.Fp8Converter.Fp8E4M3ToFloat(fp8));
    }

    [Fact]
    public void FloatToFp8_NegativeZero_SetsSignBit()
    {
        byte fp8 = OpenTail.Stingray.Diffusion.Fp8Converter.FloatToFp8E4M3(-0.0f);
        Assert.Equal(0x80, fp8);
    }

    [Theory]
    [InlineData(1000.0f)]
    [InlineData(448.1f)]
    [InlineData(float.PositiveInfinity)]
    public void FloatToFp8_AboveMax_SaturatesTo448(float value)
    {
        byte fp8 = OpenTail.Stingray.Diffusion.Fp8Converter.FloatToFp8E4M3(value);
        float back = OpenTail.Stingray.Diffusion.Fp8Converter.Fp8E4M3ToFloat(fp8);
        Assert.Equal(448.0f, back, precision: 3);
    }

    [Theory]
    [InlineData(-1000.0f)]
    [InlineData(-448.1f)]
    [InlineData(float.NegativeInfinity)]
    public void FloatToFp8_BelowNegMax_SaturatesToNeg448(float value)
    {
        byte fp8 = OpenTail.Stingray.Diffusion.Fp8Converter.FloatToFp8E4M3(value);
        float back = OpenTail.Stingray.Diffusion.Fp8Converter.Fp8E4M3ToFloat(fp8);
        Assert.Equal(-448.0f, back, precision: 3);
    }

    [Fact]
    public void FloatToFp8_NaN_EncodesToCanonicalNaNByte()
    {
        byte fp8 = OpenTail.Stingray.Diffusion.Fp8Converter.FloatToFp8E4M3(float.NaN);
        Assert.Equal(0x7F, fp8);
    }

    [Theory]
    [InlineData((byte)0x7F)]
    [InlineData((byte)0xFF)]
    public void Fp8ToFloat_NaNEncodings_DecodeToNaN(byte fp8)
    {
        float back = OpenTail.Stingray.Diffusion.Fp8Converter.Fp8E4M3ToFloat(fp8);
        Assert.True(float.IsNaN(back));
    }

    [Fact]
    public void FloatToFp8_NearZeroSubnormal_DoesNotUnderflowToNaNOrOverflow()
    {
        // A tiny value should round to a subnormal fp8 (near zero), never a NaN encoding.
        byte fp8 = OpenTail.Stingray.Diffusion.Fp8Converter.FloatToFp8E4M3(0.0001f);
        Assert.NotEqual(0x7F, fp8 & 0x7F);
        float back = OpenTail.Stingray.Diffusion.Fp8Converter.Fp8E4M3ToFloat(fp8);
        Assert.True(back is >= 0f and < 0.01f);
    }

    [Fact]
    public void ConvertToFp8_Batch_MatchesElementwiseConversion()
    {
        float[] src = [0.0f, 1.5f, -3.25f, 100f, -100f];
        var dst = new byte[src.Length];
        OpenTail.Stingray.Diffusion.Fp8Converter.ConvertToFp8(src, dst);

        for (int i = 0; i < src.Length; i++)
        {
            byte expected = OpenTail.Stingray.Diffusion.Fp8Converter.FloatToFp8E4M3(src[i]);
            Assert.Equal(expected, dst[i]);
        }
    }

    [Theory]
    [InlineData(100.0f)]
    [InlineData(-100.0f)]
    public void RoundTrip_NonPowerOfTwoValue_RoundsToNearestRepresentable(float value)
    {
        // 100 is not exactly representable in E4M3 (3 mantissa bits at this exponent give
        // steps of 8: ..., 96, 104, ...); it rounds to the nearest of those.
        byte fp8 = OpenTail.Stingray.Diffusion.Fp8Converter.FloatToFp8E4M3(value);
        float back = OpenTail.Stingray.Diffusion.Fp8Converter.Fp8E4M3ToFloat(fp8);
        Assert.InRange(System.MathF.Abs(back - value), 0f, 8f);
    }

    [Fact]
    public void RoundTrip_ValueNearMax_StaysWithinFp8Precision()
    {
        // 300 is representable-ish in E4M3 (exp range covers it); the round-trip should be
        // close but not necessarily exact given only 3 mantissa bits.
        byte fp8 = OpenTail.Stingray.Diffusion.Fp8Converter.FloatToFp8E4M3(300.0f);
        float back = OpenTail.Stingray.Diffusion.Fp8Converter.Fp8E4M3ToFloat(fp8);
        Assert.InRange(back, 280f, 320f);
    }
}
