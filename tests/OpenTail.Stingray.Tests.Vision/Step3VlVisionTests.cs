using System;
using OpenTail.Stingray.Vision;
using Xunit;

namespace OpenTail.Stingray.Tests.Vision;

public sealed class Step3VlVisionTests
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

        var pre = Step3VlImagePreprocessor.Preprocess(rgb, w, h, imageSize: 378, patchSize: 14);
        Assert.NotNull(pre);
        Assert.Equal(378, pre.TargetWidth);
        Assert.Equal(280, pre.TargetHeight);
        Assert.Equal(27, pre.PatchesX);
        Assert.Equal(20, pre.PatchesY);
        Assert.Equal(3 * 378 * 280, pre.Chw.Length);

        for (int i = 0; i < 100; i++)
        {
            Assert.False(float.IsNaN(pre.Chw[i]));
            Assert.False(float.IsInfinity(pre.Chw[i]));
        }
    }
}
