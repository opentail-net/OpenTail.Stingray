using OpenTail.Stingray.Audio.AudioGen;
using OpenTail.Stingray.Audio.Primitives;
using OpenTail.Stingray.Core;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio.AudioGen;

/// <summary>
/// First real end-to-end smoke test for the AudioGen port: real `facebook/audiogen-medium` LM +
/// codec weights (converted from AudioCraft's native `.bin` checkpoint format to safetensors,
/// see docs/063-audiogen-implementation-plan.md) + real stock `t5-large` conditioning, full
/// pipeline run against real weights. NON-DEGENERACY receipt (finite, non-silent), not yet a
/// numeric golden-parity test against an independent Python/AudioCraft reference run.
/// </summary>
public sealed class AudioGenGenerationSmokeTests : HeavyTestBase
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

    private static (string lm, string codec, string t5, string tokenizer)? FindWeights()
    {
        string? lm = FindRepoFile("models/audiogen-medium/audiogen-medium-lm.safetensors");
        string? codec = FindRepoFile("models/audiogen-medium/audiogen-medium-encodec16k.safetensors");
        string? t5 = FindRepoFile("models/audiogen-medium/t5-large.safetensors");
        string? tokenizer = FindRepoFile("models/audiogen-medium/t5-large-tokenizer.json");
        if (lm is null || codec is null || t5 is null || tokenizer is null) return null;
        return (lm, codec, t5, tokenizer);
    }

    [Fact]
    public void Generate_RealWeights_ProducesNonDegenerateAudio()
    {
        var weights = FindWeights();
        Assert.SkipUnless(weights is not null,
            "models/audiogen-medium/{audiogen-medium-lm.safetensors,audiogen-medium-encodec16k.safetensors,t5-large.safetensors,t5-large-tokenizer.json} not found");
        var (lmPath, codecPath, t5Path, tokenizerPath) = weights!.Value;

        using var lmLoader = SafetensorsLoader.Open(lmPath);
        using var codecLoader = SafetensorsLoader.Open(codecPath);
        using var t5Loader = SafetensorsLoader.Open(t5Path);

        var textEncoderWeights = AudioGenTextEncoderWeights.Load(t5Loader);
        var tokenizer = T5Tokenizer.FromFile(tokenizerPath);
        var transformerWeights = new AudioGenTransformerWeights(lmLoader);
        var codecWeights = AudioGenEncodecDecoderWeights.Load(codecLoader);

        var generator = new AudioGenGenerator(textEncoderWeights, tokenizer, transformerWeights, codecWeights);

        // Short + greedy (topK<=1) for a fast, deterministic smoke run.
        var pcm = generator.Generate("dog barking", durationSeconds: 1.0f, seed: 0, guidanceScale: 3.0f, topK: 1);

        Assert.True(pcm.Length > 0, "decoder produced zero samples");
        foreach (var sample in pcm)
            Assert.True(float.IsFinite(sample), "PCM contains NaN/Inf -- pipeline produced degenerate output");

        double sumSq = 0;
        foreach (var sample in pcm) sumSq += (double)sample * sample;
        double rms = Math.Sqrt(sumSq / pcm.Length);
        Assert.True(rms > 1e-6, $"PCM RMS energy ({rms}) is near-silent -- likely a wiring bug, not real audio");
    }

    /// <summary>Generates a real, listenable sample (real top-k/temperature sampling, not greedy) into `docs/audio-samples/` for manual by-ear review -- per CLAUDE.md rule 9, this directory is gitignored and local-only.</summary>
    [Fact]
    public void Generate_RealWeights_WritesListenableSample()
    {
        var weights = FindWeights();
        Assert.SkipUnless(weights is not null,
            "models/audiogen-medium/{audiogen-medium-lm.safetensors,audiogen-medium-encodec16k.safetensors,t5-large.safetensors,t5-large-tokenizer.json} not found");
        var (lmPath, codecPath, t5Path, tokenizerPath) = weights!.Value;

        string? repoRoot = FindRepoFile("README.md");
        Assert.SkipUnless(repoRoot != null, "repo root not found");
        string samplesDir = Path.Combine(Path.GetDirectoryName(repoRoot!)!, "docs", "audio-samples");
        Directory.CreateDirectory(samplesDir);

        using var lmLoader = SafetensorsLoader.Open(lmPath);
        using var codecLoader = SafetensorsLoader.Open(codecPath);
        using var t5Loader = SafetensorsLoader.Open(t5Path);

        var textEncoderWeights = AudioGenTextEncoderWeights.Load(t5Loader);
        var tokenizer = T5Tokenizer.FromFile(tokenizerPath);
        var transformerWeights = new AudioGenTransformerWeights(lmLoader);
        var codecWeights = AudioGenEncodecDecoderWeights.Load(codecLoader);
        var generator = new AudioGenGenerator(textEncoderWeights, tokenizer, transformerWeights, codecWeights);

        var pcm = generator.Generate("heavy rain falling on a metal roof", durationSeconds: 5.0f, seed: 42,
            guidanceScale: AudioGenConfig.DefaultGuidanceScale, topK: AudioGenConfig.DefaultTopK, temperature: AudioGenConfig.DefaultTemperature);

        string outPath = Path.Combine(samplesDir, "audiogen-medium-first-real-sample.wav");
        WavWriter.WriteWav(outPath, pcm, sampleRate: AudioGenConfig.SampleRate, channels: 1);

        Assert.True(File.Exists(outPath));
    }
}
