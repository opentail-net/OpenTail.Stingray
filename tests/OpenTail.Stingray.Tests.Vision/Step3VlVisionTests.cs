namespace OpenTail.Stingray.Tests.Vision;

/// <summary>
/// Step3-VL slicing rules, as llama.cpp <c>mtmd_image_preprocessor_step3vl</c> (image_size 728, crop 504).
/// The 640x200 case reproduces llama-server's 503 image-side tokens for that image (4 crops x 81 + 169
/// overview + 10 markers); it also pins the float rounding in calc_crop_extent (640/200 - 3 = 0.2000000476
/// is above the 0.2 threshold, so 4 windows, not 3).
/// </summary>
public sealed class Step3VlVisionTests
{
    private static readonly float[] Mean = [0.481455f, 0.457828f, 0.408211f];
    private static readonly float[] Std = [0.26863f, 0.261303f, 0.275777f];

    private static Step3VlPreprocessedImage Run(int w, int h) =>
        Step3VlImagePreprocessor.Preprocess(new byte[w * h * 3], w, h, imageSize: 728, longestEdge: 3024, Mean, Std);

    [Fact]
    public void WideSmallImage_SlicesOnShortSideWindows()
    {
        var pre = Run(640, 200);
        Assert.Equal((4, 1), (pre.GridW, pre.GridH));
        Assert.All(pre.Crops, c => Assert.Equal(504, c.Size));
        Assert.Equal(728, pre.Overview.Size);
        Assert.Equal(3 * 728 * 728, pre.Overview.Chw.Length);
    }

    [Fact]
    public void NearSquareSmallImage_HasOverviewOnly()
    {
        var pre = Run(640, 480);
        Assert.Empty(pre.Crops);
        Assert.Equal(728, pre.Overview.Size);
    }

    [Fact]
    public void LargeImage_Uses504Windows()
    {
        // 2000/504 = 3.97 -> 4 windows (2016); 1000/504 = 1.98 -> 2 windows (1008).
        var pre = Run(2000, 1000);
        Assert.Equal((4, 2), (pre.GridW, pre.GridH));
        Assert.Equal(8, pre.Crops.Count);
    }
}
