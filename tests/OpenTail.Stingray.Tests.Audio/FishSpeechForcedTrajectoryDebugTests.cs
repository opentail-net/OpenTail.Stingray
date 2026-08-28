using System;
using System.Globalization;
using System.IO;
using OpenTail.Stingray.Audio.FishSpeech;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// TEMP debug harness (docs/audio-review-progress.md's Fish Speech "goat" investigation,
/// 2026-08-28): forces the REAL reference's own generated semantic-token trajectory (captured
/// earlier in scratch_ref_codes.txt) through our own slow-AR trunk + fast-AR step by step
/// (bypassing our own semantic-token CHOICE entirely, but exercising our real decode-step
/// ForwardEmbedding/KV-cache path across many positions), then compares our resulting greedy
/// residual codes against the reference's own real residual codes for the SAME frames. If they
/// match closely, our trunk+fast-AR are correct even across many decode steps and the "goat"
/// symptom is really about OUR OWN semantic-token choices diverging from a good trajectory
/// (points at the slow-AR's post-prefill decode path or its own greedy/RAS choices). If they
/// diverge sharply, that points at a real decode-step bug (stale KV cache, RoPE position, etc.)
/// analogous to the QwenTTS stale-pointer bug found earlier this project. TODO remove once
/// resolved.
/// </summary>
public sealed class FishSpeechForcedTrajectoryDebugTests : HeavyTestBase
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

    [Fact]
    public void ForceReferenceSemanticTrajectory_CompareResidualCodes()
    {
        string? modelPath = FindModelPath("s2-pro-q4_k_m.gguf");
        string? tokDir = FindRepoDir("examples/s2.cpp");
        string? codesPath = FindRepoFile("scratch_ref_codes.txt");
        Assert.SkipUnless(modelPath != null && tokDir != null && codesPath != null, "prerequisites not found");

        var lines = File.ReadAllLines(codesPath!);
        var header = lines[0].Split(' ', StringSplitOptions.RemoveEmptyEntries);
        int nFrames = int.Parse(header[0], CultureInfo.InvariantCulture);
        int numCodebooks = int.Parse(header[1], CultureInfo.InvariantCulture);

        var refSemantic = new int[nFrames];
        var refResidual = new int[numCodebooks - 1][];
        for (int cb = 0; cb < numCodebooks - 1; cb++) refResidual[cb] = new int[nFrames];
        for (int f = 0; f < nFrames; f++)
        {
            var row = lines[f + 1].Split(' ', StringSplitOptions.RemoveEmptyEntries);
            refSemantic[f] = int.Parse(row[0], CultureInfo.InvariantCulture);
            for (int cb = 0; cb < numCodebooks - 1; cb++)
                refResidual[cb][f] = int.Parse(row[cb + 1], CultureInfo.InvariantCulture);
        }
        Console.WriteLine($"nFrames={nFrames} numCodebooks={numCodebooks}");

        using var pipeline = new FishSpeechPipeline(modelPath!, tokDir!);
        var (ourSemantic, ourFrames) = pipeline.ForceGenerateFrames("Hello! I will make some lunch, darling!", refSemantic);

        Assert.Equal(nFrames, ourFrames.Count);

        for (int cb = 0; cb < numCodebooks - 1; cb++)
        {
            int matches = 0;
            for (int f = 0; f < nFrames; f++)
                if (ourFrames[f][cb + 1] == refResidual[cb][f]) matches++;
            Console.WriteLine($"cb[{cb + 1}] exact-match rate vs reference: {matches}/{nFrames} ({100.0 * matches / nFrames:F1}%)");
        }
    }
}
