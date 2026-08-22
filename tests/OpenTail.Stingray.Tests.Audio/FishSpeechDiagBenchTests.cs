using System;
using System.Diagnostics;
using System.IO;
using OpenTail.Stingray.Audio.FishSpeech;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>TEMPORARY diagnostic-only bench, isolates trunk-vs-fast-AR cost -- delete after use.</summary>
public sealed class FishSpeechDiagBenchTests : HeavyTestBase
{
    [Fact]
    public void Diag_IsolateTrunkVsFastAr()
    {
        string? modelPath = FindModelPath("s2-pro-q4_k_m.gguf");
        string? tokDir = FindRepoDir("examples/s2.cpp");
        Assert.SkipUnless(modelPath != null && tokDir != null, "S2 Pro GGUF or examples/s2.cpp not found");

        using var pipeline = new FishSpeechPipeline(modelPath!, tokDir!);

        var prompt = pipeline.BuildPrompt("Hello there.");

        // Warmup
        pipeline.GenerateSemanticTokens("Hello there.", maxTokens: 3);

        var swTotal = Stopwatch.StartNew();
        var frames = pipeline.GenerateFrames("Hello there.", maxTokens: 10);
        swTotal.Stop();

        File.WriteAllText(Path.Combine(Path.GetTempPath(), "fishspeech_diag_result.txt"),
            $"total_ms={swTotal.Elapsed.TotalMilliseconds:F1} tokens={frames.SemanticTokens.Count} prompt_len={prompt.Count}\n");
    }

    private static string? FindModelPath(string fileName)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, "models", fileName);
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
}
