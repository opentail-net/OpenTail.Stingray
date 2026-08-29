using System;
using System.IO;
using OpenTail.Stingray.Audio.FishSpeech;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// TEMP clip-generation helper (docs/audio-review-progress.md's Fish Speech investigation,
/// 2026-08-28): generates a real WAV at a given seed for the user to listen to directly (the CLI
/// has no --seed flag). TODO remove once the investigation concludes.
/// </summary>
public sealed class FishSpeechClipGenDebugTests : HeavyTestBase
{
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

    private static string? FindOutDir()
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, "docs", "audio-samples");
            if (Directory.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void GenerateClip_Seed7_PostCodecFix()
    {
        string? modelPath = FindModelPath("s2-pro-q4_k_m.gguf");
        string? tokDir = FindRepoDir("examples/s2.cpp");
        string? outDir = FindOutDir();
        Assert.SkipUnless(modelPath != null && tokDir != null && outDir != null, "prerequisites not found");

        using var pipeline = FishSpeechFullPipeline.Load(modelPath!, tokDir!, modelPath!);
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var pcm = pipeline.Synthesize("Hello! I will make some lunch, darling!", maxTokens: 200, seed: 7);
        sw.Stop();
        Assert.NotEmpty(pcm);

        var result = new OpenTail.Stingray.Audio.AudioGenerationResult(pcm, 44100);
        string outPath = Path.Combine(outDir!, "fishspeech-lunch-v13-fastar-zeroalloc-seed7.wav");
        result.SaveWav(outPath);
        Console.WriteLine($"[FastArZeroAlloc] saved {outPath} samples={pcm.Length} durationSec={pcm.Length / 44100.0:F2} elapsedSec={sw.Elapsed.TotalSeconds:F2}s");
    }
}
