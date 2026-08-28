using System.IO;
using OpenTail.Stingray.Audio.QwenTTS;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real end-to-end smoke test for <see cref="QwenTtsPipeline"/> -- text -&gt; real Talker semantic
/// codes -&gt; real Code Predictor acoustic depth-expansion (using the real
/// <see cref="OpenTail.Stingray.Engine.ForwardPass.LastHidden"/> bridge this session added) -&gt;
/// real, independently golden-verified codec decode chain -&gt; waveform. First time all of
/// QwenTTS's real components (Talker, Code Predictor, and every codec stage) run chained
/// together in one call. Matches this project's established first-pass bar for a from-scratch
/// end-to-end pipeline (finite + non-silent RMS, not a fabricated numeric oracle).
/// </summary>
public sealed class QwenTtsPipelineTests : HeavyTestBase
{
    private static string? FindRepoFile(string relativePath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relativePath);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void Generate_RealWeights_ProducesFiniteNonSilentWaveform()
    {
        string? talkerPath = FindRepoFile("models/qwen-talker-0.6b-base-Q8_0.gguf");
        string? codecPath = FindRepoFile("models/qwen-tokenizer-12hz-Q8_0.gguf");
        Assert.SkipUnless(talkerPath != null, "models/qwen-talker-0.6b-base-Q8_0.gguf not found");
        Assert.SkipUnless(codecPath != null, "models/qwen-tokenizer-12hz-Q8_0.gguf not found");

        using var pipeline = QwenTtsPipeline.Load(talkerPath!, codecPath!);
        Assert.Equal(24000, pipeline.SampleRate);

        var wav = pipeline.Generate("Hello there.", maxFrames: 6);

        Assert.True(wav.Length > 0, "pipeline produced zero samples");

        double sumSq = 0;
        foreach (var s in wav)
        {
            Assert.False(float.IsNaN(s) || float.IsInfinity(s), "waveform contains NaN/Inf");
            sumSq += (double)s * s;
        }
        double rms = System.Math.Sqrt(sumSq / wav.Length);
        Assert.True(rms > 1e-6, $"waveform looks silent/degenerate: rms={rms}");
    }

    // TEMP bisection harness for the golden-verification investigation (docs/audio-review-
    // progress.md's QwenTTS entries) -- set STINGRAY_QWENTTS_GOLDEN_DUMP and STINGRAY_QWENTTS_BISECT_LAYERS
    // to dump the N-layer talker trunk's hidden state for comparison against the real PyTorch
    // reference. Not a real correctness assertion; TODO revert/remove once the bug is found.
    [Fact]
    public void Bisect_TalkerLayers()
    {
        string? talkerPath = FindRepoFile("models/qwen-talker-0.6b-base-Q8_0.gguf");
        string? codecPath = FindRepoFile("models/qwen-tokenizer-12hz-Q8_0.gguf");
        Assert.SkipUnless(talkerPath != null, "models/qwen-talker-0.6b-base-Q8_0.gguf not found");
        Assert.SkipUnless(codecPath != null, "models/qwen-tokenizer-12hz-Q8_0.gguf not found");
        string? nLayersEnv = System.Environment.GetEnvironmentVariable("STINGRAY_QWENTTS_BISECT_LAYERS");
        Assert.SkipUnless(nLayersEnv != null, "STINGRAY_QWENTTS_BISECT_LAYERS not set");
        int nLayers = int.Parse(nLayersEnv!);

        using var pipeline = QwenTtsPipeline.Load(talkerPath!, codecPath!);
        _ = pipeline.Generate("Hello there", talkerNumLayers: nLayers, maxFrames: 1);
    }
}
