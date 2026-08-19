using System;
using System.IO;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Vision;
using Xunit;

namespace OpenTail.Stingray.Tests.Vision;

public sealed class InternVlVisionTests
{
    [Fact]
    public void ImagePreprocessor_NormalizesToStandardGrid()
    {
        int w = 800;
        int h = 600;
        var rgb = new byte[w * h * 3];
        for (int i = 0; i < rgb.Length; i += 3)
        {
            rgb[i] = 130;
            rgb[i + 1] = 180;
            rgb[i + 2] = 220;
        }

        var pre = InternVlImagePreprocessor.Preprocess(rgb, w, h, imageSize: 448, patchSize: 14);
        Assert.NotNull(pre);
        Assert.Equal(448, pre.TargetWidth);
        Assert.Equal(448, pre.TargetHeight);
        Assert.Equal(32, pre.PatchesX);
        Assert.Equal(32, pre.PatchesY);
        Assert.Equal(3 * 448 * 448, pre.Chw.Length);

        for (int i = 0; i < 100; i++)
        {
            Assert.False(float.IsNaN(pre.Chw[i]));
            Assert.False(float.IsInfinity(pre.Chw[i]));
        }
    }
}
