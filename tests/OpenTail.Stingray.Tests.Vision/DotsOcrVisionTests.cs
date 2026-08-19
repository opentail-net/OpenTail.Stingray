using System;
using OpenTail.Stingray.Vision;
using Xunit;

namespace OpenTail.Stingray.Tests.Vision;

public sealed class DotsOcrVisionTests
{
    [Fact]
    public void ImagePreprocessor_GeneratesSnappedGrid()
    {
        int w = 600;
        int h = 400;
        var rgb = new byte[w * h * 3];
        for (int i = 0; i < rgb.Length; i += 3)
        {
            rgb[i] = 110;
            rgb[i + 1] = 170;
            rgb[i + 2] = 210;
        }

        var pre = DotsOcrImagePreprocessor.Preprocess(rgb, w, h, patchSize: 14);
        Assert.NotNull(pre);
        Assert.Equal(602, pre.TargetWidth); // Snap to multiple of 14: 602 / 14 = 43
        Assert.Equal(406, pre.TargetHeight); // Snap to multiple of 14: 406 / 14 = 29
        Assert.Equal(43, pre.PatchesX);
        Assert.Equal(29, pre.PatchesY);
        Assert.Equal(3 * 602 * 406, pre.Chw.Length);

        for (int i = 0; i < 100; i++)
        {
            Assert.False(float.IsNaN(pre.Chw[i]));
            Assert.False(float.IsInfinity(pre.Chw[i]));
        }
    }
}
