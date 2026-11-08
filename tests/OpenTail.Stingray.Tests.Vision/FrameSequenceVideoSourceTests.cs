using OpenTail.Stingray.Diffusion;
using OpenTail.Stingray.Engine.Verification;
using OpenTail.Stingray.Vision;

namespace OpenTail.Stingray.Tests.Vision;

/// <summary>
/// Phase 5.3 frame-input path (Qwen 3.8 plan): a directory of PNG frames is a video source with no decoder. The rate is
/// stated by the caller; frames come back in name order, as RGB24, and the two-frame projector path accepts them.
/// </summary>
public sealed class FrameSequenceVideoSourceTests : IDisposable
{
    private const int Side = 224;
    private const double Fps = 4.0;

    private readonly string _dir = Path.Combine(Path.GetTempPath(), "stingray-frames-" + Guid.NewGuid().ToString("N"));

    public FrameSequenceVideoSourceTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        if (Directory.Exists(_dir)) Directory.Delete(_dir, recursive: true);
    }

    /// <summary>RGB24 frame whose content moves with k, so consecutive frames differ.</summary>
    private static byte[] RgbFrame(int k, int side)
    {
        var rgb = new byte[side * side * 3];
        int bar = (16 + k * 12) * side / Side;
        for (int y = 0; y < side; y++)
            for (int x = 0; x < side; x++)
            {
                int i = (y * side + x) * 3;
                bool inBar = x >= bar && x < bar + side / 5;
                rgb[i] = (byte)(inBar ? 230 : 30);
                rgb[i + 1] = (byte)(y < side / 2 ? 90 : 160);
                rgb[i + 2] = (byte)(x < side / 2 ? 60 : 200);
            }
        return rgb;
    }

    private static float[] ToChw(byte[] rgb, int side)
    {
        int plane = side * side;
        var chw = new float[3 * plane];
        for (int p = 0; p < plane; p++)
            for (int c = 0; c < 3; c++)
                chw[c * plane + p] = rgb[p * 3 + c] / 255f;
        return chw;
    }

    private void WriteFrames(int count, int side)
    {
        for (int k = 0; k < count; k++)
            PngWriter.Write(Path.Combine(_dir, $"frame-{k:D4}.png"), ToChw(RgbFrame(k, side), side), side, side);
    }

    [Fact]
    public void Probe_ReportsTheFirstFrameSizeAndTheStatedRate()
    {
        WriteFrames(3, Side);

        var info = new FrameSequenceVideoSource(Fps).Probe(_dir);

        Assert.Equal(Side, info.Width);
        Assert.Equal(Side, info.Height);
        Assert.Equal(Fps, info.Fps);
    }

    [Fact]
    public void ReadFrames_ReturnsEveryFrameInNameOrder()
    {
        const int count = 6;
        WriteFrames(count, Side);

        var frames = new FrameSequenceVideoSource(Fps).ReadFrames(_dir, Fps, Side, Side).ToList();

        Assert.Equal(count, frames.Count);
        for (int k = 0; k < count; k++)
        {
            byte[] expected = RgbFrame(k, Side);
            int worst = 0;
            for (int i = 0; i < expected.Length; i++) worst = Math.Max(worst, Math.Abs(expected[i] - frames[k][i]));
            Assert.True(worst <= 1, $"frame {k} differs from the source by {worst} levels");
        }
    }

    [Fact]
    public void ReadFrames_ResizesFramesOfAnotherSize()
    {
        WriteFrames(2, 112);

        var frames = new FrameSequenceVideoSource(Fps).ReadFrames(_dir, Fps, Side, Side).ToList();

        Assert.All(frames, f => Assert.Equal(Side * Side * 3, f.Length));
    }

    [Fact]
    public void ReadFrames_RefusesAnotherRate()
    {
        WriteFrames(2, Side);

        Assert.Throws<NotSupportedException>(() => new FrameSequenceVideoSource(Fps).ReadFrames(_dir, 2.0, Side, Side).ToList());
    }

    [Fact]
    public void ReadFrames_OrdersNumericallyNotLexically()
    {
        // frame-10 must come after frame-2; lexical order would put it second.
        PngWriter.Write(Path.Combine(_dir, "frame-1.png"), ToChw(RgbFrame(0, Side), Side), Side, Side);
        PngWriter.Write(Path.Combine(_dir, "frame-2.png"), ToChw(RgbFrame(1, Side), Side), Side, Side);
        PngWriter.Write(Path.Combine(_dir, "frame-10.png"), ToChw(RgbFrame(2, Side), Side), Side, Side);

        var frames = new FrameSequenceVideoSource(Fps).ReadFrames(_dir, Fps, Side, Side).ToList();

        Assert.Equal(3, frames.Count);
        for (int k = 0; k < 3; k++)
        {
            byte[] expected = RgbFrame(k, Side);
            int worst = 0;
            for (int i = 0; i < expected.Length; i++) worst = Math.Max(worst, Math.Abs(expected[i] - frames[k][i]));
            Assert.True(worst <= 1, $"position {k} is not frame {k} (differs by {worst} levels)");
        }
    }

    [Fact]
    public void EmptyDirectory_IsRejected()
    {
        Assert.Throws<InvalidDataException>(() => new FrameSequenceVideoSource(Fps).Probe(_dir));
    }

    [Fact]
    public void FrameSequence_RunsThroughTheTwoFramePath()
    {
        string? mmproj = ModelLocator.FindDeep("mmproj-Qwen3.8-Flash-Next-F16.gguf");
        Assert.SkipUnless(mmproj is not null, "mmproj-Qwen3.8-Flash-Next-F16.gguf is required (904 MB).");
        const int count = 8;
        WriteFrames(count, Side);

        var frames = new FrameSequenceVideoSource(Fps).ReadFrames(_dir, Fps, Side, Side)
            .Select(f => VideoTemporalPairs.RgbToChw(f, Side, Side)).ToList();
        using var vision = QwenVlVisionModel.Open(mmproj!);
        var encoder = new QwenVlVisionEncoder(vision);
        int merged = vision.PatchSize * 2; // patch size from the mmproj, times the 2x2 merge

        // The reference's chunking (a timestamp between frames splits a pair), not consecutive pairs.
        var chunks = VideoPromptLayout.Chunks(VideoPromptLayout.Parts(frames.Count, (float)Fps));
        foreach (int[] chunk in chunks)
        {
            float[] tokens = encoder.ForwardVideo(frames[chunk[0]], frames[chunk[^1]], Side, Side, out int n);
            Assert.Equal((Side / merged) * (Side / merged), n);
            Assert.True(tokens.Length > 0);
            Assert.All(tokens.Take(256), v => Assert.True(float.IsFinite(v)));
        }
        Assert.Equal(5, chunks.Count); // 8 frames at 4 fps: [0] [1,2] [3,4] [5,6] [7]
    }
}
