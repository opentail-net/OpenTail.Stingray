using System;
using OpenTail.Stingray.Vision;
using Xunit;

namespace OpenTail.Stingray.Tests.Vision;

public sealed class CogVlmVisionTests
{
    [Fact]
    public void ImagePreprocessor_GeneratesSquareTargetGrid()
    {
        int w = 640;
        int h = 480;
        var rgb = new byte[w * h * 3];
        for (int i = 0; i < rgb.Length; i += 3)
        {
            rgb[i] = 120;
            rgb[i + 1] = 140;
            rgb[i + 2] = 160;
        }

        var pre = CogVlmImagePreprocessor.Preprocess(rgb, w, h, imageSize: 490, patchSize: 14);
        Assert.NotNull(pre);
        Assert.Equal(490, pre.TargetWidth);
        Assert.Equal(490, pre.TargetHeight);
        Assert.Equal(35, pre.PatchesX);
        Assert.Equal(35, pre.PatchesY);
        Assert.Equal(3 * 490 * 490, pre.Chw.Length);

        for (int i = 0; i < 100; i++)
        {
            Assert.False(float.IsNaN(pre.Chw[i]));
            Assert.False(float.IsInfinity(pre.Chw[i]));
        }
    }
}
