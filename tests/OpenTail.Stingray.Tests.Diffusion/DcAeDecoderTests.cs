using System;
using OpenTail.Stingray.Diffusion;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

public sealed class DcAeDecoderTests
{
    [Fact]
    public void NearestUpsample2x_ScalesSpatialDimensionsCorrectly()
    {
        int c = 3, h = 4, w = 4;
        var input = new float[c * h * w];
        for (int i = 0; i < input.Length; i++) input[i] = i;

        var (output, outH, outW) = DcAeDecoder.NearestUpsample2x(input, c, h, w);

        Assert.Equal(8, outH);
        Assert.Equal(8, outW);
        Assert.Equal(c * 8 * 8, output.Length);

        // Top-left 2x2 of channel 0 should all equal input[0]
        Assert.Equal(input[0], output[0]);
        Assert.Equal(input[0], output[1]);
        Assert.Equal(input[0], output[8]);
        Assert.Equal(input[0], output[9]);
    }

    [Fact]
    public void Decode_ReconstructsFullImageWithoutNaNs()
    {
        using var decoder = new DcAeDecoder(compressionRatio: 32, latentChannels: 32);
        int latH = 2, latW = 2;
        var latent = new float[32 * latH * latW];
        for (int i = 0; i < latent.Length; i++) latent[i] = (i * 0.19f) % 2.0f - 1.0f;

        var rgb = decoder.Decode(latent, latH, latW);

        int expectedH = latH * 32;
        int expectedW = latW * 32;
        Assert.Equal(3 * expectedH * expectedW, rgb.Length);

        foreach (float pixel in rgb)
        {
            Assert.False(float.IsNaN(pixel));
            Assert.False(float.IsInfinity(pixel));
            Assert.InRange(pixel, 0.0f, 1.0f);
        }
    }
}
