using System;
using OpenTail.Stingray.Vision;
using Xunit;

namespace OpenTail.Stingray.Tests.Vision;

public sealed class Exaone4VisionTests
{
    [Fact]
    public void ImagePreprocessor_GeneratesTargetGrid()
    {
        int w = 640;
        int h = 480;
        var rgb = new byte[w * h * 3];
        for (int i = 0; i < rgb.Length; i += 3)
        {
            rgb[i] = 100;
            rgb[i + 1] = 150;
            rgb[i + 2] = 200;
        }

        var pre = Exaone4ImagePreprocessor.Preprocess(rgb, w, h, patchSize: 14, mergeFactor: 2, maxDim: 980);
        Assert.NotNull(pre);
        Assert.True(pre.TargetWidth > 0);
        Assert.True(pre.TargetHeight > 0);
        Assert.Equal(pre.TargetWidth / 14, pre.PatchesX);
        Assert.Equal(pre.TargetHeight / 14, pre.PatchesY);
        Assert.Equal(3 * pre.TargetWidth * pre.TargetHeight, pre.Chw.Length);

        for (int i = 0; i < 100; i++)
        {
            Assert.False(float.IsNaN(pre.Chw[i]));
            Assert.False(float.IsInfinity(pre.Chw[i]));
        }
    }
}
