using System;
using System.Collections.Generic;
using OpenTail.Stingray.Diffusion;
using OpenTail.Stingray.Diffusion.Wan;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

public sealed class WanVae3DTests
{
    [Fact]
    public void DupUp3D_CorrectlyUpscalesSpatialAndTemporalDimensions()
    {
        int c = 2, t = 2, h = 2, w = 2;
        var input = new float[c * t * h * w];
        for (int i = 0; i < input.Length; i++) input[i] = i + 1;

        int factorT = 2;
        int factorS = 2;

        var (output, outT, outH, outW) = WanVaeDecoder3D.DupUp3D(input, c, t, h, w, factorT, factorS);

        Assert.Equal(4, outT);
        Assert.Equal(4, outH);
        Assert.Equal(4, outW);
        Assert.Equal(c * 4 * 4 * 4, output.Length);

        // Value at channel 0, outT=0, outH=0, outW=0 should match inT=0, inH=0, inW=0
        Assert.Equal(input[0], output[0]);
    }

    [Fact]
    public void CausalConv3D_PreservesTemporalCausality()
    {
        using var decoder = new WanVaeDecoder3D();
        int inCh = 1, outCh = 1, t = 4, h = 2, w = 2;

        // An impulse at time t=2
        var x = new float[inCh * t * h * w];
        int frameSize = h * w;
        for (int i = 0; i < frameSize; i++) x[2 * frameSize + i] = 10.0f;

        var outVal = decoder.CausalConv3D(x, "test.conv", inCh, outCh, t, h, w, kt: 3, kh: 1, kw: 1);

        // Due to causal temporal padding (pads left, not right):
        // Output at t=0 and t=1 must NOT be affected by impulse at t=2 (must be 0)
        for (int s = 0; s < frameSize; s++)
        {
            Assert.Equal(0.0f, outVal[0 * frameSize + s]);
            Assert.Equal(0.0f, outVal[1 * frameSize + s]);
        }
    }

    [Fact]
    public void Decode_ProducesValidRGBFramesWithoutNaNs()
    {
        using var decoder = new WanVaeDecoder3D();
        int t = 2, latH = 2, latW = 2;
        int totalLatent = WanVaeDecoder3D.LatentChannels * t * latH * latW;
        var latent = new float[totalLatent];
        for (int i = 0; i < latent.Length; i++) latent[i] = (i * 0.17f) % 2.0f - 1.0f;

        var frames = decoder.Decode(latent, t, latH, latW);

        Assert.NotEmpty(frames);
        foreach (var frame in frames)
        {
            Assert.NotEmpty(frame);
            foreach (float pixel in frame)
            {
                Assert.False(float.IsNaN(pixel));
                Assert.False(float.IsInfinity(pixel));
                Assert.InRange(pixel, 0.0f, 1.0f);
            }
        }
    }
}
