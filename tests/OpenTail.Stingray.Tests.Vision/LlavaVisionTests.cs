using System;
using System.IO;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Vision;
using Xunit;

namespace OpenTail.Stingray.Tests.Vision;

public sealed class LlavaVisionTests
{
    [Fact]
    public void ImagePreprocessor_GeneratesSquare336Grid()
    {
        int w = 640;
        int h = 480;
        var rgb = new byte[w * h * 3];
        for (int i = 0; i < rgb.Length; i += 3)
        {
            rgb[i] = 160;
            rgb[i + 1] = 120;
            rgb[i + 2] = 220;
        }

        var pre = LlavaImagePreprocessor.Preprocess(rgb, w, h, imageSize: 336, patchSize: 14);
        Assert.NotNull(pre);
        Assert.Equal(336, pre.TargetWidth);
        Assert.Equal(336, pre.TargetHeight);
        Assert.Equal(24, pre.PatchesX);
        Assert.Equal(24, pre.PatchesY);
        Assert.Equal(3 * 336 * 336, pre.Chw.Length);

        for (int i = 0; i < 100; i++)
        {
            Assert.False(float.IsNaN(pre.Chw[i]));
            Assert.False(float.IsInfinity(pre.Chw[i]));
        }
    }
}
