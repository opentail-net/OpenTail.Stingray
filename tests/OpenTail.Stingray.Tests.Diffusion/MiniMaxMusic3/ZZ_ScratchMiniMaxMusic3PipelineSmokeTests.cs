using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.MiniMaxMusic3;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion.MiniMaxMusic3;

/// <summary>
/// Real-weight end-to-end smoke test chaining AR generation -> condition encoder -> Flow scheduler
/// -> vocoder, entirely with real weights. NOT golden-parity (blocked on the real tokenizer / a
/// real reference generation run, same caveat as <c>MiniMaxMusic3AutoregressiveGeneratorSmokeTests</c>)
/// -- confirms the whole chain runs without exceptions and produces finite, in-range PCM. Scratch:
/// exercises the full pipeline shape end-to-end; delete once a real golden/regression test
/// supersedes it.
/// </summary>
public sealed class ZZ_ScratchMiniMaxMusic3PipelineSmokeTests
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

    private static string? FindRepoDir(string relativePath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relativePath);
            if (Directory.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void Synthesize_RealWeights_ProducesFinitePcm()
    {
        string? langDir = FindRepoDir("models/minimax-music3/language_model");
        string? depthPath = FindRepoFile("models/minimax-music3/rvq_depth_decoder.safetensors");
        string? condPath = FindRepoFile("models/minimax-music3/condition_encoder.safetensors");
        string? transformerDir = FindRepoDir("models/minimax-music3/transformer");
        string? vocoderPath = FindRepoFile("models/minimax-music3/vocoder.safetensors");
        string? tokenizerDir = FindRepoDir("models/minimax-music3/tokenizer");
        Assert.SkipUnless(langDir != null && depthPath != null && condPath != null && transformerDir != null && vocoderPath != null && tokenizerDir != null,
            "one or more models/minimax-music3/* real weight/tokenizer files not found");

        using var langLoader = SafetensorsLoader.OpenDirectory(langDir!);
        using var globalModel = new MiniMaxMusic3GlobalModel(langLoader);

        using var depthLoader = SafetensorsLoader.Open(depthPath!);
        var depthWeights = MiniMaxMusic3RvqDepthDecoderWeights.Load(depthLoader);

        var promptEncoder = MiniMaxMusic3PromptEncoder.Load(tokenizerDir!);
        int[] promptTokens = promptEncoder.BuildConditionalPrompt(
            "Intimate acoustic folk, male vocal, fingerpicked guitar",
            "[Verse]\nWalking through the morning rain");
        var random = new Random(42);
        var representation = MiniMaxMusic3AutoregressiveGenerator.Generate(globalModel, depthWeights, promptTokens, maxFrames: 20, random);
        Assert.True(representation.FrameCount >= 1, "AR generator produced zero frames");

        using var condLoader = SafetensorsLoader.Open(condPath!);
        var conditionWeights = MiniMaxMusic3ConditionEncoderWeights.Load(condLoader);

        using var transformerLoader = SafetensorsLoader.OpenDirectory(transformerDir!);
        var transformerWeights = MiniMaxMusic3TransformerWeights.Load(transformerLoader);

        using var vocoderLoader = SafetensorsLoader.Open(vocoderPath!);
        var vocoderWeights = MiniMaxMusic3VocoderWeights.Load(vocoderLoader);

        var pcm = MiniMaxMusic3Pipeline.Synthesize(conditionWeights, transformerWeights, vocoderWeights, representation, numFlowSteps: 4, seed: 7);

        Assert.True(pcm.Length > 0, "vocoder produced zero samples");
        bool anyNonFinite = false;
        bool anyOutOfRange = false;
        foreach (var s in pcm)
        {
            if (!float.IsFinite(s)) anyNonFinite = true;
            if (s < -1.5f || s > 1.5f) anyOutOfRange = true; // real decoder isn't hard-clamped in this pipeline yet
        }
        Assert.False(anyNonFinite, "PCM contains NaN/Inf");
        Assert.False(anyOutOfRange, "PCM wildly out of expected [-1,1]-ish range");
    }
}
