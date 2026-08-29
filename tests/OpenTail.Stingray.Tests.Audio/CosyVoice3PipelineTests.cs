
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real end-to-end smoke test for <see cref="CosyVoice3Pipeline"/> -- text -&gt; real LLM speech
/// tokens -&gt; real flow-encoder conditioning -&gt; real DiT CFM ODE solve -&gt; real HiFT vocoder,
/// chaining every stage built this session on real weights. Matches this project's established
/// first-pass bar for a from-scratch end-to-end pipeline (finite + non-silent RMS + correct
/// sample rate, not a fabricated numeric oracle -- same bar Fish Speech's end-to-end test used).
/// </summary>
public sealed class CosyVoice3PipelineTests : HeavyTestBase
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
        string? path = FindRepoFile("models/cosyvoice3/CosyVoice3-2512_F16.gguf");
        Assert.SkipUnless(path != null, "models/cosyvoice3/CosyVoice3-2512_F16.gguf not found");

        using var pipeline = CosyVoice3Pipeline.Load(path!);
        Assert.Equal(24000, pipeline.SampleRate);

        var wav = pipeline.Generate("Hello there, this is a test.", maxNewSpeechTokens: 20, odeSteps: 4, seed: 42);

        Assert.True(wav.Length > 0, "pipeline produced zero samples");

        double sumSq = 0;
        foreach (var s in wav)
        {
            Assert.False(float.IsNaN(s) || float.IsInfinity(s), "waveform contains NaN/Inf");
            Assert.InRange(s, -1.5f, 1.5f); // HiFT has no hard clamp; allow real small overshoot
            sumSq += (double)s * s;
        }
        double rms = System.Math.Sqrt(sumSq / wav.Length);
        Assert.True(rms > 1e-4, $"waveform looks silent/degenerate: rms={rms}");
    }
}
