using OpenTail.Stingray.Engine.Verification;
using OpenTail.Stingray.Vision;

namespace OpenTail.Stingray.Tests.Vision;

/// <summary>
/// Phase 5.3 (Qwen 3.8 plan), step (a)-(b): a video temporal patch takes two consecutive frames. Identical frames must
/// reproduce the still-image path exactly, and the two frames must enter their own temporal kernels (additively). Real
/// projector; skipped when the mmproj is absent.
/// </summary>
public sealed class QwenVlVideoFrameTests
{
    private const int Side = 224;

    private static string? FindMmproj() => ModelLocator.FindDeep("mmproj-Qwen3.8-Flash-Next-F16.gguf");

    private static float[] Pattern(int seed)
    {
        var chw = new float[3 * Side * Side];
        for (int i = 0; i < chw.Length; i++)
            chw[i] = (float)Math.Sin((i + seed) * 0.017) * 0.8f;
        return chw;
    }

    [Fact]
    public void VideoWithIdenticalFrames_MatchesTheStillImagePath()
    {
        string? mmproj = FindMmproj();
        Assert.SkipUnless(mmproj is not null, "mmproj-Qwen3.8-Flash-Next-F16.gguf is required (904 MB).");

        using var vision = QwenVlVisionModel.Open(mmproj!);
        var encoder = new QwenVlVisionEncoder(vision);
        var frame = Pattern(1);

        float[] still = encoder.Forward(frame, Side, Side, out int stillTokens);
        float[] video = encoder.ForwardVideo(frame, frame, Side, Side, out int videoTokens);

        Assert.Equal(stillTokens, videoTokens);
        Assert.Equal(still.Length, video.Length);
        for (int i = 0; i < still.Length; i++)
            Assert.Equal(still[i], video[i]);
    }

    [Fact]
    public void SecondFrame_ChangesTheVideoTokens()
    {
        // Additivity holds only at the patch-embedding stage, which is private; the layers after it are nonlinear. So this
        // checks the observable property: the second frame reaches the output. Frame-level agreement is checked later
        // against the reference.
        string? mmproj = FindMmproj();
        Assert.SkipUnless(mmproj is not null, "mmproj-Qwen3.8-Flash-Next-F16.gguf is required (904 MB).");

        using var vision = QwenVlVisionModel.Open(mmproj!);
        var encoder = new QwenVlVisionEncoder(vision);
        var f0 = Pattern(3);
        var f1 = Pattern(11);

        float[] same = encoder.ForwardVideo(f0, f0, Side, Side, out _);
        float[] different = encoder.ForwardVideo(f0, f1, Side, Side, out _);

        double maxDiff = 0;
        for (int i = 0; i < same.Length; i++)
            maxDiff = Math.Max(maxDiff, Math.Abs(same[i] - different[i]));
        Assert.True(maxDiff > 1e-3, $"the second frame had no effect on the output (max difference {maxDiff})");
    }

    [Fact]
    public void VideoFramesOfDifferentSizes_AreRejected()
    {
        string? mmproj = FindMmproj();
        Assert.SkipUnless(mmproj is not null, "mmproj-Qwen3.8-Flash-Next-F16.gguf is required (904 MB).");

        using var vision = QwenVlVisionModel.Open(mmproj!);
        var encoder = new QwenVlVisionEncoder(vision);
        var big = Pattern(5);
        var small = new float[3 * 112 * 112];

        Assert.Throws<ArgumentException>(() => encoder.ForwardVideo(big, small, Side, Side, out _));
    }
}
