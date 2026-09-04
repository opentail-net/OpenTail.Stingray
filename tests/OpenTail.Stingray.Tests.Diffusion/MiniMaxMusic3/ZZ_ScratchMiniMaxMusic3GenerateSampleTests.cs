using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.MiniMaxMusic3;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion.MiniMaxMusic3;

/// <summary>
/// Scratch: generates a real V1-scope sample (single chunk, up to the real `_CHUNK_FRAMES=200`
/// boundary, ~8s) with real weights and the real tokenizer, and writes it to
/// docs/diffusion-samples/ for listening. NOT a golden-parity check (see the other MiniMax-Music3
/// smoke tests' own caveats -- the AR loop/scheduler glue isn't numerically verified against a real
/// reference yet). Delete once superseded by a real regression/sample-generation tool.
/// </summary>
public sealed class ZZ_ScratchMiniMaxMusic3GenerateSampleTests
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

    private static string? FindRepoRoot()
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            if (Directory.Exists(Path.Combine(dir, "docs"))) return dir;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void Generate_V1Scope_WritesRealWavSample()
    {
        string? langDir = FindRepoDir("models/minimax-music3/language_model");
        string? depthPath = FindRepoFile("models/minimax-music3/rvq_depth_decoder.safetensors");
        string? condPath = FindRepoFile("models/minimax-music3/condition_encoder.safetensors");
        string? transformerDir = FindRepoDir("models/minimax-music3/transformer");
        string? vocoderPath = FindRepoFile("models/minimax-music3/vocoder.safetensors");
        string? tokenizerDir = FindRepoDir("models/minimax-music3/tokenizer");
        string? repoRoot = FindRepoRoot();
        Assert.SkipUnless(langDir != null && depthPath != null && condPath != null && transformerDir != null && vocoderPath != null && tokenizerDir != null && repoRoot != null,
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
        var representation = MiniMaxMusic3AutoregressiveGenerator.Generate(globalModel, depthWeights, promptTokens, maxFrames: 200, random);
        Assert.True(representation.FrameCount >= 1, "AR generator produced zero frames");

        using var condLoader = SafetensorsLoader.Open(condPath!);
        var conditionWeights = MiniMaxMusic3ConditionEncoderWeights.Load(condLoader);

        using var transformerLoader = SafetensorsLoader.OpenDirectory(transformerDir!);
        var transformerWeights = MiniMaxMusic3TransformerWeights.Load(transformerLoader);

        using var vocoderLoader = SafetensorsLoader.Open(vocoderPath!);
        var vocoderWeights = MiniMaxMusic3VocoderWeights.Load(vocoderLoader);

        var pcm = MiniMaxMusic3Pipeline.Synthesize(conditionWeights, transformerWeights, vocoderWeights, representation, numFlowSteps: 8, seed: 7);
        Assert.True(pcm.Length > 0, "vocoder produced zero samples");

        // Real vocoder output is channel-planar ([L...][R...]), WavWriter needs interleaved.
        int samplesPerChannel = pcm.Length / 2;
        var interleaved = new float[pcm.Length];
        for (int i = 0; i < samplesPerChannel; i++)
        {
            interleaved[2 * i] = pcm[i];
            interleaved[2 * i + 1] = pcm[samplesPerChannel + i];
        }

        string outDir = Path.Combine(repoRoot!, "docs", "diffusion-samples");
        Directory.CreateDirectory(outDir);
        string outPath = Path.Combine(outDir, $"minimax_music3_v1_folk_verse_{representation.FrameCount}frames.wav");
        WavWriter.WriteWav(outPath, interleaved, sampleRate: MiniMaxMusic3Config.VocoderSampleRate, channels: 2);

        Assert.True(File.Exists(outPath));
    }
}
