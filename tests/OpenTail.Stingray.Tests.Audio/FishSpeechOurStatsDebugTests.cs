using System;
using System.IO;
using System.Linq;
using OpenTail.Stingray.Audio.FishSpeech;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// TEMP debug harness (docs/audio-review-progress.md's Fish Speech "goat" investigation,
/// 2026-08-28): dumps the SAME per-codebook stats (distinct count, longest repeat run, value
/// range) for OUR own current pipeline's generated codes, to compare directly against the two
/// real reference trajectories already captured (scratch_ref_codes.txt = "slightly off, maxing
/// out"; scratch_ref_codes_clean.txt = clean). TODO remove once resolved.
/// </summary>
public sealed class FishSpeechOurStatsDebugTests : HeavyTestBase
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

    [Fact]
    public void DumpOurStats()
    {
        string? modelPath = FindModelPath("s2-pro-q4_k_m.gguf");
        string? tokDir = FindRepoDir("examples/s2.cpp");
        Assert.SkipUnless(modelPath != null && tokDir != null, "prerequisites not found");

        using var pipeline = new FishSpeechPipeline(modelPath!, tokDir!);
        var (tokens, frames) = pipeline.GenerateFrames("Hello! I will make some lunch, darling!", maxTokens: 200);

        int nFrames = frames.Count;
        Console.WriteLine($"=== OURS: nframes={nFrames} ===");
        int numCb = frames[0].Length;
        for (int cb = 0; cb < numCb; cb++)
        {
            var vals = cb == 0 ? tokens : frames.Select(f => f[cb]).ToList();
            int distinct = vals.Distinct().Count();
            int maxRun = 1, curRun = 1;
            for (int i = 1; i < vals.Count; i++)
            {
                if (vals[i] == vals[i - 1]) { curRun++; maxRun = Math.Max(maxRun, curRun); }
                else curRun = 1;
            }
            var mostCommon = vals.GroupBy(v => v).OrderByDescending(g => g.Count()).First();
            string label = cb == 0 ? "semantic" : $"residual{cb}";
            Console.WriteLine($"  cb[{cb}]({label}): distinct={distinct}/{nFrames} maxRun={maxRun} mostCommon=({mostCommon.Key},{mostCommon.Count()}) range=[{vals.Min()},{vals.Max()}]");
        }
    }
}
