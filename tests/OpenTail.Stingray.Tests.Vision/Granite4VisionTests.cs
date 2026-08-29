
namespace OpenTail.Stingray.Tests.Vision;

public sealed class Granite4VisionTests
{
    [Fact]
    public void ImagePreprocessor_GeneratesSquareTargetGrid()
    {
        int w = 500;
        int h = 300;
        var rgb = new byte[w * h * 3];
        for (int i = 0; i < rgb.Length; i += 3)
        {
            rgb[i] = 80;
            rgb[i + 1] = 90;
            rgb[i + 2] = 100;
        }

        var pre = Granite4ImagePreprocessor.Preprocess(rgb, w, h, imageSize: 384, patchSize: 14);
        Assert.NotNull(pre);
        Assert.Equal(384, pre.TargetWidth);
        Assert.Equal(384, pre.TargetHeight);
        Assert.Equal(384 / 14, pre.PatchesX);
        Assert.Equal(384 / 14, pre.PatchesY);
        Assert.Equal(3 * 384 * 384, pre.Chw.Length);

        for (int i = 0; i < 100; i++)
        {
            Assert.False(float.IsNaN(pre.Chw[i]));
            Assert.False(float.IsInfinity(pre.Chw[i]));
        }
    }
}
