using System;
using System.IO;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Vision;
using Xunit;

namespace OpenTail.Stingray.Tests.Vision;

public sealed class KimiVisionTests
{
    [Fact]
    public void ImagePreprocessor_SnapsToMultiplesOf28()
    {
        int w = 640;
        int h = 480;
        var rgb = new byte[w * h * 3];
        for (int i = 0; i < rgb.Length; i += 3)
        {
            rgb[i] = 150;
            rgb[i + 1] = 190;
            rgb[i + 2] = 240;
        }

        var pre = KimiImagePreprocessor.Preprocess(rgb, w, h, patchSize: 14, mergeFactor: 2);
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
