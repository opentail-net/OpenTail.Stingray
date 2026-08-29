
namespace OpenTail.Stingray.Tests.Diffusion;

public sealed class TinyVaeTests
{
    [Fact]
    public void NearestUpsample2x_ScalesSpatialDimensionsCorrectly()
    {
        int c = 2, h = 4, w = 4;
        var input = new float[c * h * w];
        for (int i = 0; i < input.Length; i++) input[i] = i + 1.0f;

        var (output, outH, outW) = TinyVaeDecoder.NearestUpsample2x(input, c, h, w);

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
    public void Decode_4ChannelLatent_ProducesCorrectSpatialDimensionsAndValidRanges()
    {
        using var decoder = new TinyVaeDecoder(latentChannels: 4);
        int latH = 4, latW = 4;
        var latent = new float[4 * latH * latW];
        for (int i = 0; i < latent.Length; i++) latent[i] = (i % 7 - 3) * 0.5f;

        var rgb = decoder.Decode(latent, latH, latW);

        int expectedH = latH * 8;
        int expectedW = latW * 8;
        Assert.Equal(3 * expectedH * expectedW, rgb.Length);

        foreach (float pixel in rgb)
        {
            Assert.False(float.IsNaN(pixel));
            Assert.False(float.IsInfinity(pixel));
            Assert.InRange(pixel, 0.0f, 1.0f);
        }
    }

    [Fact]
    public void Decode_16ChannelLatent_ProducesCorrectSpatialDimensions()
    {
        using var decoder = new TinyVaeDecoder(latentChannels: 16);
        int latH = 2, latW = 2;
        var latent = new float[16 * latH * latW];
        for (int i = 0; i < latent.Length; i++) latent[i] = ((i % 11) - 5) * 0.2f;

        var rgb = decoder.Decode(latent, latH, latW);

        int expectedH = latH * 8;
        int expectedW = latW * 8;
        Assert.Equal(3 * expectedH * expectedW, rgb.Length);

        foreach (float pixel in rgb)
        {
            Assert.False(float.IsNaN(pixel));
            Assert.False(float.IsInfinity(pixel));
            Assert.InRange(pixel, 0.0f, 1.0f);
        }
    }

    [Fact]
    public void DecodeToRgb24_PopulatesInterleavedBuffer()
    {
        using var decoder = new TinyVaeDecoder(latentChannels: 4);
        int latH = 2, latW = 2;
        var latent = new float[4 * latH * latW];
        Array.Fill(latent, 0.5f);

        int outH = latH * 8;
        int outW = latW * 8;
        var buffer = new byte[outH * outW * 3];

        decoder.DecodeToRgb24(latent, latH, latW, buffer);

        Assert.Equal(outH * outW * 3, buffer.Length);
        bool hasNonZero = false;
        for (int i = 0; i < buffer.Length; i++)
        {
            if (buffer[i] > 0) hasNonZero = true;
        }
        Assert.True(hasNonZero);
    }

    [Fact]
    public void DecodeToRgba32_SetsAlphaChannelTo255()
    {
        using var decoder = new TinyVaeDecoder(latentChannels: 4);
        int latH = 2, latW = 2;
        var latent = new float[4 * latH * latW];
        Array.Fill(latent, 0.2f);

        int outH = latH * 8;
        int outW = latW * 8;
        var buffer = new byte[outH * outW * 4];

        decoder.DecodeToRgba32(latent, latH, latW, buffer);

        for (int i = 0; i < outH * outW; i++)
        {
            Assert.Equal(255, buffer[i * 4 + 3]); // Alpha channel
        }
    }

    [Fact]
    public void Encode_ProducesExpectedLatentDimensions()
    {
        using var encoder = new TinyVaeEncoder(latentChannels: 4);
        int height = 16, width = 16;
        var rgb = new float[3 * height * width];
        for (int i = 0; i < rgb.Length; i++) rgb[i] = 0.5f;

        var latent = encoder.Encode(rgb, height, width);

        int expectedLatH = height / 8;
        int expectedLatW = width / 8;
        Assert.Equal(4 * expectedLatH * expectedLatW, latent.Length);

        foreach (float val in latent)
        {
            Assert.False(float.IsNaN(val));
            Assert.False(float.IsInfinity(val));
        }
    }

    [Fact]
    public void IVaeDecoder_InterfaceConformance()
    {
        IVaeDecoder decoder = new TinyVaeDecoder(latentChannels: 4);
        int latH = 2, latW = 2;
        var latent = new float[4 * latH * latW];
        var rgb = decoder.Decode(latent, latH, latW);

        Assert.Equal(3 * 16 * 16, rgb.Length);
    }
}
