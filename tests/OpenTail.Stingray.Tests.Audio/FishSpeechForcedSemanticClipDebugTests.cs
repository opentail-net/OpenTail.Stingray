using System;
using System.Globalization;
using System.IO;
using OpenTail.Stingray.Audio.FishSpeech;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// TEMP debug harness (docs/audio-review-progress.md's Fish Speech "goat" investigation,
/// 2026-08-28): the decisive isolation test -- forces the REAL reference's own semantic-token
/// sequence (from scratch_ref_codes.txt) through OUR trunk + OUR fast-AR (greedy) + OUR fixed
/// codec, producing a real WAV. If this sounds close to the reference, the bug is isolated to
/// how OUR slow-AR CHOOSES semantic tokens (not the trunk's per-step hidden-state computation,
/// not fast-AR, not the codec). If it still sounds wrong, the trunk itself has a real bug that
/// doesn't show up in single-step logit comparisons. TODO remove once resolved.
/// </summary>
public sealed class FishSpeechForcedSemanticClipDebugTests : HeavyTestBase
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
    public void ForceReferenceSemanticSequence_ThroughOurFastArAndCodec()
    {
        string? modelPath = FindModelPath("s2-pro-q4_k_m.gguf");
        string? tokDir = FindRepoDir("examples/s2.cpp");
        string? codesPath = FindRepoFile("scratch_ref_codes.txt");
        string? outDir = FindOutDir();
        Assert.SkipUnless(modelPath != null && tokDir != null && codesPath != null && outDir != null,
            "prerequisites not found");

        var lines = File.ReadAllLines(codesPath!);
        var header = lines[0].Split(' ', StringSplitOptions.RemoveEmptyEntries);
        int nFrames = int.Parse(header[0], CultureInfo.InvariantCulture);
        var refSemantic = new int[nFrames];
        for (int f = 0; f < nFrames; f++)
        {
            var row = lines[f + 1].Split(' ', StringSplitOptions.RemoveEmptyEntries);
            refSemantic[f] = int.Parse(row[0], CultureInfo.InvariantCulture);
        }
        Console.WriteLine($"nFrames={nFrames}");

        using var pipeline = new FishSpeechPipeline(modelPath!, tokDir!);
        var (semanticTokens, frames) = pipeline.ForceGenerateFrames("Hello! I will make some lunch, darling!", refSemantic);

        var semanticCodes = semanticTokens.ToArray();
        int numResidual = frames[0].Length - 1;
        var residualCodes = new int[numResidual][];
        for (int cb = 0; cb < numResidual; cb++)
        {
            residualCodes[cb] = new int[nFrames];
            for (int ti = 0; ti < nFrames; ti++)
                residualCodes[cb][ti] = frames[ti][cb + 1];
        }

        using var codecWeights = new FishSpeechCodecWeights(modelPath!);
        var pcm = FishSpeechCodec.Decode(codecWeights, semanticCodes, residualCodes);

        float peak = 0f;
        for (int i = 0; i < pcm.Length; i++) { float a = MathF.Abs(pcm[i]); if (a > peak) peak = a; }
        if (peak > 1e-4f && peak < 0.8f)
        {
            float gain = 0.85f / peak;
            for (int i = 0; i < pcm.Length; i++) pcm[i] *= gain;
        }

        var result = new OpenTail.Stingray.Audio.AudioGenerationResult(pcm, 44100);
        string outPath = Path.Combine(outDir!, "fishspeech-lunch-forced-ref-semantic-our-fastar.wav");
        result.SaveWav(outPath);
        Console.WriteLine($"saved {outPath} samples={pcm.Length} durationSec={pcm.Length / 44100.0:F2}");
    }
}
