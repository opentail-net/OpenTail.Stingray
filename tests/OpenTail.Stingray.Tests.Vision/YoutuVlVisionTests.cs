
namespace OpenTail.Stingray.Tests.Vision;

public sealed class YoutuVlVisionTests
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

        var pre = YoutuVlImagePreprocessor.Preprocess(rgb, w, h, patchSize: 16, mergeFactor: 2);
        Assert.NotNull(pre);
        Assert.Equal(pre.TargetWidth / 16, pre.PatchesX);
        Assert.Equal(pre.TargetHeight / 16, pre.PatchesY);
        Assert.Equal(3 * pre.TargetWidth * pre.TargetHeight, pre.Chw.Length);

        for (int i = 0; i < 100; i++)
        {
            Assert.False(float.IsNaN(pre.Chw[i]));
            Assert.False(float.IsInfinity(pre.Chw[i]));
        }
    }

    [Fact]
    public void Preprocess_RoundsUpToMergeAlignment_LikeLlamaCpp()
    {
        // mtmd_image_preprocessor_youtuvl: 640x488 -> 640x512 (ceil to 32 px), 40x32 patches = 320
        // tokens after the 2x2 merge; llama-server reports the same image token count for test-1.jpeg.
        var pre = YoutuVlImagePreprocessor.Preprocess(new byte[640 * 488 * 3], 640, 488, patchSize: 16, mergeFactor: 2);
        Assert.Equal((640, 512), (pre.TargetWidth, pre.TargetHeight));
        Assert.Equal((40, 32), (pre.PatchesX, pre.PatchesY));
    }
}
