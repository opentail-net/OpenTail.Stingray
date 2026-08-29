
namespace OpenTail.Stingray.Tests.Vision;

public sealed class NemotronVisionTests
{
    [Fact]
    public void ImagePreprocessor_Generates512Grid()
    {
        int w = 640;
        int h = 480;
        var rgb = new byte[w * h * 3];
        for (int i = 0; i < rgb.Length; i += 3)
        {
            rgb[i] = 120;
            rgb[i + 1] = 160;
            rgb[i + 2] = 200;
        }

        var pre = NemotronImagePreprocessor.Preprocess(rgb, w, h, imageSize: 512, patchSize: 14);
        Assert.NotNull(pre);
        Assert.Equal(512, pre.TargetWidth);
        Assert.Equal(512, pre.TargetHeight);
        Assert.Equal(36, pre.PatchesX); // 512 / 14 = 36
        Assert.Equal(36, pre.PatchesY);
        Assert.Equal(3 * 512 * 512, pre.Chw.Length);

        for (int i = 0; i < 100; i++)
        {
            Assert.False(float.IsNaN(pre.Chw[i]));
            Assert.False(float.IsInfinity(pre.Chw[i]));
        }
    }
}
