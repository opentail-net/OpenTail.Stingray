using System;
using System.Globalization;
using System.IO;
using System.Linq;
using OpenTail.Stingray.Audio.FishSpeech;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// TEMP debug harness (docs/audio-review-progress.md's Fish Speech "closer but not perfect"
/// investigation, 2026-08-28): compares our C# slow-AR's real prefill logits (fully deterministic,
/// no RNG involved) against the REAL reference's own prefill logits for the byte-identical prompt
/// (captured via a temporary env-var-gated dump in examples/s2.cpp's s2_generate.cpp, reverted
/// after use). This isolates the slow-AR trunk itself, independent of any sampling/RNG variance,
/// as the codec is already proven correct (see FishSpeechCodecReferenceCodesDebugTests). TODO
/// remove once resolved.
/// </summary>
public sealed class FishSpeechPrefillLogitsCompareDebugTests : HeavyTestBase
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
    public void ComparePrefillLogits_AgainstRealReference()
    {
        string? modelPath = FindModelPath("s2-pro-q4_k_m.gguf");
        string? tokDir = FindRepoDir("examples/s2.cpp");
        string? logitsPath = FindRepoFile("scratch_ref_prefill_logits.txt");
        Assert.SkipUnless(modelPath != null && tokDir != null && logitsPath != null,
            "S2 Pro GGUF, examples/s2.cpp, or scratch_ref_prefill_logits.txt not found");

        var lines = File.ReadAllLines(logitsPath!);
        int refLen = int.Parse(lines[0], CultureInfo.InvariantCulture);
        var refLogits = new double[refLen];
        for (int i = 0; i < refLen; i++) refLogits[i] = double.Parse(lines[i + 1], CultureInfo.InvariantCulture);

        using var pipeline = new FishSpeechPipeline(modelPath!, tokDir!);
        var prompt = pipeline.BuildPrompt("Hello! I will make some lunch, darling!");
        var ours = pipeline.PrefillForBisection(prompt);

        Console.WriteLine($"refLen={refLen} oursLen={ours.Length}");
        Assert.Equal(refLen, ours.Length);

        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < refLen; i++)
        {
            dot += ours[i] * refLogits[i];
            na += (double)ours[i] * ours[i];
            nb += refLogits[i] * refLogits[i];
        }
        double cosine = dot / (Math.Sqrt(na) * Math.Sqrt(nb) + 1e-12);
        Console.WriteLine($"cosine similarity (full vocab): {cosine}");

        // Also report top-10 argmax agreement within the semantic range, since that's what
        // actually drives the first generated token.
        var refTop = Enumerable.Range(0, refLen).OrderByDescending(i => refLogits[i]).Take(10).ToArray();
        var oursTop = Enumerable.Range(0, ours.Length).OrderByDescending(i => ours[i]).Take(10).ToArray();
        Console.WriteLine("ref top10 indices: " + string.Join(",", refTop));
        Console.WriteLine("ours top10 indices: " + string.Join(",", oursTop));
        Console.WriteLine("ref top10 values: " + string.Join(",", refTop.Select(i => refLogits[i].ToString("F4"))));
        Console.WriteLine("ours top10 values: " + string.Join(",", oursTop.Select(i => ours[i].ToString("F4"))));
    }
}
