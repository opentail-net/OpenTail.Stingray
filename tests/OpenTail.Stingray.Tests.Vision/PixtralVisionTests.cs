
namespace OpenTail.Stingray.Tests.Vision;

public sealed class PixtralVisionTests
{
    [Fact]
    public void ImagePreprocessor_AlignsToPatchMultiples()
    {
        int w = 800;
        int h = 600;
        var rgb = new byte[w * h * 3];
        for (int i = 0; i < rgb.Length; i += 3)
        {
            rgb[i] = 100;
            rgb[i + 1] = 150;
            rgb[i + 2] = 200;
        }

        var pre = PixtralImagePreprocessor.Preprocess(rgb, w, h, patchSize: 16, maxDim: 1024);
        Assert.NotNull(pre);
        Assert.Equal(0, pre.TargetWidth % 16);
        Assert.Equal(0, pre.TargetHeight % 16);
        Assert.Equal(pre.PatchesX * pre.PatchesY, (pre.TargetWidth / 16) * (pre.TargetHeight / 16));
        Assert.Equal(3 * pre.TargetWidth * pre.TargetHeight, pre.Chw.Length);

        for (int i = 0; i < 100; i++)
        {
            Assert.False(float.IsNaN(pre.Chw[i]));
            Assert.False(float.IsInfinity(pre.Chw[i]));
        }
    }

    [Fact]
    public void ImagePreprocessor_HandlesSquareImages()
    {
        int w = 512;
        int h = 512;
        var rgb = new byte[w * h * 3];
        var pre = PixtralImagePreprocessor.Preprocess(rgb, w, h, patchSize: 16, maxDim: 1024);
        Assert.Equal(512, pre.TargetWidth);
        Assert.Equal(512, pre.TargetHeight);
        Assert.Equal(32, pre.PatchesX);
        Assert.Equal(32, pre.PatchesY);
    }
}
