using System;
using System.IO;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Vision;
using Xunit;

namespace OpenTail.Stingray.Tests.Vision;

public sealed class DeepSeekOcrVisionTests
{
    [Fact]
    public void ImagePreprocessor_GeneratesSquare1024Grid()
    {
        int w = 800;
        int h = 600;
        var rgb = new byte[w * h * 3];
        for (int i = 0; i < rgb.Length; i += 3)
        {
            rgb[i] = 120;
            rgb[i + 1] = 180;
            rgb[i + 2] = 230;
        }

        var pre = DeepSeekOcrImagePreprocessor.Preprocess(rgb, w, h, imageSize: 1024, patchSize: 16);
        Assert.NotNull(pre);
        Assert.Equal(1024, pre.TargetWidth);
        Assert.Equal(1024, pre.TargetHeight);
        Assert.Equal(64, pre.PatchesX);
        Assert.Equal(64, pre.PatchesY);
        Assert.Equal(3 * 1024 * 1024, pre.Chw.Length);

        for (int i = 0; i < 100; i++)
        {
            Assert.False(float.IsNaN(pre.Chw[i]));
            Assert.False(float.IsInfinity(pre.Chw[i]));
        }
    }
}
