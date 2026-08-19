using System;
using OpenTail.Stingray.Vision;
using Xunit;

namespace OpenTail.Stingray.Tests.Vision;

public sealed class MobileNetV5VisionTests
{
    [Fact]
    public void ImagePreprocessor_GeneratesSquareTargetGrid()
    {
        int w = 320;
        int h = 240;
        var rgb = new byte[w * h * 3];
        for (int i = 0; i < rgb.Length; i += 3)
        {
            rgb[i] = 50;
            rgb[i + 1] = 60;
            rgb[i + 2] = 70;
        }

        var pre = MobileNetV5ImagePreprocessor.Preprocess(rgb, w, h, imageSize: 224, patchSize: 16);
        Assert.NotNull(pre);
        Assert.Equal(224, pre.TargetWidth);
        Assert.Equal(224, pre.TargetHeight);
        Assert.Equal(14, pre.PatchesX);
        Assert.Equal(14, pre.PatchesY);
        Assert.Equal(3 * 224 * 224, pre.Chw.Length);

        for (int i = 0; i < 100; i++)
        {
            Assert.False(float.IsNaN(pre.Chw[i]));
            Assert.False(float.IsInfinity(pre.Chw[i]));
        }
    }
}
