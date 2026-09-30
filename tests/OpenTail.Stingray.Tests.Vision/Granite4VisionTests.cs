
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

        var pre = Granite4ImagePreprocessor.Preprocess(rgb, w, h, imageSize: 384, patchSize: 14,
            imageMean: [0.5f, 0.5f, 0.5f], imageStd: [0.5f, 0.5f, 0.5f]);
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

    [Fact]
    public void Granite4Vision_PromptFraming_MatchesCanonicalFormat()
    {
        // Granite 4 Vision uses <image> without enclosing open/close markers,
        // and requires canonical Granite role tags without leading indentation or
        // unrequested system prompt.
        string userMsg = "<image>Describe this picture.";
        string formatted = $"<|start_of_role|>user<|end_of_role|>{userMsg}<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>";
        Assert.Equal("<|start_of_role|>user<|end_of_role|><image>Describe this picture.<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>", formatted);
    }
}
