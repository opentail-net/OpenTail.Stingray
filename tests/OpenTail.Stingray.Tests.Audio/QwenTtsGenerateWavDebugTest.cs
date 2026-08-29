using System;
using System.IO;
using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Audio.QwenTTS;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>TEMPORARY debug test: QwenTTS is disabled at the CLI level (`-e qwentts` throws,
/// TtsCommand.cs), but this test calls the pipeline directly to check whether the just-applied
/// fix (ModelHyperparams.FromGgufMetadata was missing the tensorSource argument in all three
/// QwenTTS call sites -- QwenTtsPipeline.cs, QwenTtsTalkerGeneration.cs,
/// QwenTtsCodePredictorGeneration.cs -- silently skipping real QK-RMSNorm weights confirmed
/// present via list-tensors, `talker.blk.N.attn_q_norm.weight`/`attn_k_norm.weight`, same bug
/// CLASS as the one just fixed for CosyVoice2/3's missing attention bias) makes any real,
/// listenable difference.</summary>
public sealed class QwenTtsGenerateWavDebugTest : HeavyTestBase
{
    private static string? FindModelPath(string relPath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relPath);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void Generate_RealQwenTtsWav()
    {
        string? talkerPath = FindModelPath("models/qwen-talker-0.6b-base-Q8_0.gguf");
        Assert.SkipUnless(talkerPath != null, "QwenTTS talker model not found");

        using var pipeline = QwenTtsPipeline.Load(talkerPath!);
        var wav = pipeline.Generate("This is a test of voice synthesis.", seed: 42);

        Assert.True(wav.Length > 0, "QwenTTS produced empty audio");

        string repoRoot = Directory.GetParent(Path.GetDirectoryName(talkerPath!)!)!.FullName;
        var result = new AudioGenerationResult(wav, 24000);
        string outPath = Path.Combine(repoRoot, "docs", "audio-samples", "qwentts-qknorm-fix-check.wav");
        result.SaveWav(outPath);
        Console.WriteLine($"Wrote {outPath}, {wav.Length} samples, {wav.Length / 24000.0:F2}s");
    }
}
