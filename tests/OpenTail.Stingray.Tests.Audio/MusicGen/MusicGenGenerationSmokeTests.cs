using OpenTail.Stingray.Audio.MusicGen;
using OpenTail.Stingray.Core;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio.MusicGen;

/// <summary>
/// First real end-to-end smoke test for the MusicGen port: real `facebook/musicgen-small` +
/// real stock `t5-base` weights, full pipeline (T5 conditioning -> delayed-pattern decode with
/// CFG -> EnCodec decode) run against real weights. This is a NON-DEGENERACY receipt (finite,
/// non-silent, has real spectral content), not yet a numeric golden-parity test against an
/// independent Python/HF reference -- no local Python/torch reference run exists for this model
/// yet (see docs/062-musicgen-implementation-plan.md's testing-strategy section for the
/// still-open golden-verification work, in particular the CFG null-condition convention this
/// implementation guesses at). Treat a pass here as "the pipeline runs and produces real audio
/// energy," not "the audio is musically/numerically correct."
/// </summary>
public sealed class MusicGenGenerationSmokeTests : HeavyTestBase
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
    public void Generate_RealWeights_ProducesNonDegenerateAudio()
    {
        string? musicGenPath = FindRepoFile("models/musicgen-small/musicgen-small.safetensors");
        string? tokenizerPath = FindRepoFile("models/musicgen-small/t5-base-tokenizer.json");
        Assert.SkipUnless(musicGenPath != null && tokenizerPath != null,
            "models/musicgen-small/{musicgen-small.safetensors,t5-base-tokenizer.json} not found");

        // Single checkpoint: musicgen-small's own model.safetensors bundles a full `text_encoder.*`
        // tensor tree (see MusicGenTextEncoderWeights' doc comment) -- no separate t5-base download needed.
        using var musicGenLoader = SafetensorsLoader.Open(musicGenPath!);

        var textEncoderWeights = MusicGenTextEncoderWeights.Load(musicGenLoader);
        var tokenizer = T5Tokenizer.FromFile(tokenizerPath!);
        var transformerWeights = new MusicGenTransformerWeights(musicGenLoader);
        var codecWeights = MusicGenEncodecDecoderWeights.Load(musicGenLoader);

        var generator = new MusicGenGenerator(textEncoderWeights, tokenizer, transformerWeights, codecWeights);

        // Short + greedy (topK<=1) for a fast, deterministic smoke run -- not real generation quality settings.
        var pcm = generator.Generate("acoustic guitar melody", durationSeconds: 1.0f, seed: 0, guidanceScale: 3.0f, topK: 1);

        Assert.True(pcm.Length > 0, "decoder produced zero samples");
        foreach (var sample in pcm)
            Assert.True(float.IsFinite(sample), "PCM contains NaN/Inf -- pipeline produced degenerate output");

        double sumSq = 0;
        foreach (var sample in pcm) sumSq += (double)sample * sample;
        double rms = Math.Sqrt(sumSq / pcm.Length);
        Assert.True(rms > 1e-6, $"PCM RMS energy ({rms}) is near-silent -- likely a wiring bug, not real audio");
    }

    /// <summary>Generates a real, listenable sample (real top-k/temperature sampling, not greedy) into `docs/audio-samples/` for manual by-ear review -- per CLAUDE.md rule 9, this directory is gitignored and local-only, so this is not committing generated media.</summary>
    [Fact]
    public void Generate_RealWeights_WritesListenableSample()
    {
        string? musicGenPath = FindRepoFile("models/musicgen-small/musicgen-small.safetensors");
        string? tokenizerPath = FindRepoFile("models/musicgen-small/t5-base-tokenizer.json");
        Assert.SkipUnless(musicGenPath != null && tokenizerPath != null,
            "models/musicgen-small/{musicgen-small.safetensors,t5-base-tokenizer.json} not found");

        string? repoRoot = FindRepoFile("README.md");
        Assert.SkipUnless(repoRoot != null, "repo root not found");
        string samplesDir = Path.Combine(Path.GetDirectoryName(repoRoot!)!, "docs", "audio-samples");
        Directory.CreateDirectory(samplesDir);

        using var loader = SafetensorsLoader.Open(musicGenPath!);
        var textEncoderWeights = MusicGenTextEncoderWeights.Load(loader);
        var tokenizer = T5Tokenizer.FromFile(tokenizerPath!);
        var transformerWeights = new MusicGenTransformerWeights(loader);
        var codecWeights = MusicGenEncodecDecoderWeights.Load(loader);
        var generator = new MusicGenGenerator(textEncoderWeights, tokenizer, transformerWeights, codecWeights);

        var pcm = generator.Generate("upbeat acoustic guitar melody, happy", durationSeconds: 5.0f, seed: 42,
            guidanceScale: MusicGenConfig.DefaultGuidanceScale, topK: MusicGenConfig.DefaultTopK, temperature: MusicGenConfig.DefaultTemperature);

        string outPath = Path.Combine(samplesDir, "musicgen-small-first-real-sample.wav");
        WavWriter.WriteWav(outPath, pcm, sampleRate: MusicGenConfig.SampleRate, channels: 1);

        Assert.True(File.Exists(outPath));
    }
}
