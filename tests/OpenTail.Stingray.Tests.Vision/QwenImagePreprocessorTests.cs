using OpenTail.Stingray.Vision;

namespace OpenTail.Stingray.Tests.Vision;

/// <summary>
/// Qwen-VL preprocessing for the qwen4exp projector and the shared resize it uses. Synthetic images, no weights.
/// </summary>
public sealed class QwenImagePreprocessorTests
{
    private static byte[] Solid(int w, int h, byte v)
    {
        var rgb = new byte[w * h * 3];
        Array.Fill(rgb, v);
        return rgb;
    }

    [Theory]
    [InlineData(100, 60)]
    [InlineData(8, 8)]
    [InlineData(640, 480)]
    public void Preprocess_AlignsToTheMergeGrid_AndFitsTheTokenBudget(int srcW, int srcH)
    {
        var pre = QwenImagePreprocessor.Preprocess(Solid(srcW, srcH, 128), srcW, srcH);

        Assert.Equal(0, pre.Width % QwenImagePreprocessor.Align);
        Assert.Equal(0, pre.Height % QwenImagePreprocessor.Align);
        int tokens = (pre.Width / QwenImagePreprocessor.Align) * (pre.Height / QwenImagePreprocessor.Align);
        Assert.InRange(tokens, 1, QwenImagePreprocessor.MaxTokens);
        Assert.Equal(3 * pre.Width * pre.Height, pre.Chw.Length);
    }

    [Fact]
    public void Preprocess_NormalisesToMinusOneOne_MeanHalfStdHalf()
    {
        var black = QwenImagePreprocessor.Preprocess(Solid(64, 64, 0), 64, 64);
        var white = QwenImagePreprocessor.Preprocess(Solid(64, 64, 255), 64, 64);

        Assert.All(black.Chw, v => Assert.Equal(-1f, v));
        Assert.All(white.Chw, v => Assert.Equal(1f, v));
    }

    [Fact]
    public void ResizeToU8Chw_SameSize_CopiesThePlanesExactly()
    {
        // 2x1 image, planar output is R plane, G plane, B plane.
        byte[] rgb = { 10, 20, 30, 40, 50, 60 };
        byte[] chw = ImagePreprocessor.ResizeToU8Chw(rgb, 2, 1, 2, 1);

        Assert.Equal(new byte[] { 10, 40, 20, 50, 30, 60 }, chw);
    }
}
