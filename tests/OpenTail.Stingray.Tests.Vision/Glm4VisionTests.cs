using System;
using System.IO;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Vision;
using Xunit;

namespace OpenTail.Stingray.Tests.Vision;

public sealed class Glm4VisionTests
{
    [Fact]
    public void ImagePreprocessor_SnapsToGridMultiplesOf28()
    {
        int w = 700;
        int h = 500;
        var rgb = new byte[w * h * 3];
        for (int i = 0; i < rgb.Length; i += 3)
        {
            rgb[i] = 110;
            rgb[i + 1] = 170;
            rgb[i + 2] = 230;
        }

        var pre = Glm4ImagePreprocessor.Preprocess(rgb, w, h, patchSize: 14, mergeFactor: 2);
        Assert.NotNull(pre);
        Assert.Equal(0, pre.TargetWidth % 28);
        Assert.Equal(0, pre.TargetHeight % 28);
        Assert.Equal(pre.PatchesX * pre.PatchesY, (pre.TargetWidth / 14) * (pre.TargetHeight / 14));
        Assert.Equal(3 * pre.TargetWidth * pre.TargetHeight, pre.Chw.Length);

        for (int i = 0; i < 100; i++)
        {
            Assert.False(float.IsNaN(pre.Chw[i]));
            Assert.False(float.IsInfinity(pre.Chw[i]));
        }
    }
}
