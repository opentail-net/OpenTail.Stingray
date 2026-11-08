using OpenTail.Stingray.Vision;

namespace OpenTail.Stingray.Tests.Vision;

/// <summary>
/// Phase 5.3 step (d) (Qwen 3.8 plan): the video prompt layout follows the reference's text timestamps and its
/// adjacency-based chunking. Pure logic, no weights.
/// </summary>
public sealed class VideoPromptLayoutTests
{
    private static List<string> Texts(IEnumerable<VideoPromptPart> parts) =>
        parts.Where(p => !p.IsFrame).Select(p => p.Text!).ToList();

    [Fact]
    public void EightFramesAtFourFps_TimestampSplitsTheFirstPair()
    {
        var parts = VideoPromptLayout.Parts(8, 4f);

        Assert.Equal(["Video:", "[0m0.00s]"], Texts(parts));
        Assert.Equal(new List<int[]> { new[] { 0 }, new[] { 1, 2 }, new[] { 3, 4 }, new[] { 5, 6 }, new[] { 7 } },
            VideoPromptLayout.Chunks(parts));
    }

    [Fact]
    public void TimestampsEveryTwentyFrames_AtFourFpsAndFiveSeconds()
    {
        var parts = VideoPromptLayout.Parts(45, 4f);

        Assert.Equal(["Video:", "[0m0.00s]", "[0m5.00s]", "[0m10.00s]"], Texts(parts));
        var chunks = VideoPromptLayout.Chunks(parts);
        Assert.Equal(23, chunks.Count);            // [0], then 22 pairs (1,2) ... (43,44)
        Assert.Equal(new[] { 19, 20 }, chunks[10]); // the pair closed by the 5 s timestamp
        Assert.Equal(new[] { 21, 22 }, chunks[11]);
        Assert.Equal(new[] { 43, 44 }, chunks[^1]);
    }

    [Fact]
    public void MinuteBoundary_IsComputedFromFrameIndexOverFps()
    {
        var parts = VideoPromptLayout.Parts(241, 4f); // frame 240 is at 60.00 s

        var texts = Texts(parts);
        Assert.Equal("[1m0.00s]", texts[^1]);
        Assert.Equal(14, texts.Count); // "Video:" + thresholds 0, 5, ..., 60 s (13 timestamps)
    }

    [Fact]
    public void Timestamp_UsesMinutesAndTwoDecimalSeconds()
    {
        Assert.Equal("[0m0.00s]", VideoPromptLayout.Timestamp(0f));
        Assert.Equal("[0m0.25s]", VideoPromptLayout.Timestamp(250f));
        Assert.Equal("[1m5.50s]", VideoPromptLayout.Timestamp(65500f));
    }

    [Fact]
    public void NoTimestamps_PairsConsecutiveFrames()
    {
        var parts = VideoPromptLayout.Parts(8, 4f, timestampIntervalMs: 0);

        Assert.Equal(["Video:"], Texts(parts));
        Assert.Equal(new List<int[]> { new[] { 0, 1 }, new[] { 2, 3 }, new[] { 4, 5 }, new[] { 6, 7 } },
            VideoPromptLayout.Chunks(parts));
    }
}
