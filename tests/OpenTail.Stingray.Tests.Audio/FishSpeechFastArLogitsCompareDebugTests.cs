using System;
using System.Globalization;
using System.IO;
using OpenTail.Stingray.Audio.FishSpeech;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// TEMP debug harness (docs/audio-review-progress.md's Fish Speech "closer but not perfect"
/// investigation, 2026-08-28): feeds the REAL reference's own dumped hidden state and sampled
/// semantic code (captured via a temporary env-var-gated dump in examples/s2.cpp's
/// s2_generate.cpp, reverted after use) into our C# fast-AR directly, and compares the resulting
/// codebook-1 logits against the reference's own. This isolates the fast-AR forward math itself
/// (independent of RNG/sampling and independent of the slow-AR, which is already confirmed to
/// match closely). TODO remove once resolved.
/// </summary>
public sealed class FishSpeechFastArLogitsCompareDebugTests : HeavyTestBase
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

    [Fact]
    public void CompareFastArLogits_AgainstRealReference()
    {
        string? modelPath = FindModelPath("s2-pro-q4_k_m.gguf");
        string? dumpPath = FindRepoFile("scratch_ref_fastar_logits.txt");
        Assert.SkipUnless(modelPath != null && dumpPath != null,
            "S2 Pro GGUF or scratch_ref_fastar_logits.txt not found");

        var lines = File.ReadAllLines(dumpPath!);
        int idx = 0;
        int semCode = int.Parse(lines[idx++], CultureInfo.InvariantCulture);
        int hiddenSize = int.Parse(lines[idx++], CultureInfo.InvariantCulture);
        var hidden = new float[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) hidden[i] = float.Parse(lines[idx++], CultureInfo.InvariantCulture);
        int logitsSize = int.Parse(lines[idx++], CultureInfo.InvariantCulture);
        var refLogits = new double[logitsSize];
        for (int i = 0; i < logitsSize; i++) refLogits[i] = double.Parse(lines[idx++], CultureInfo.InvariantCulture);

        Console.WriteLine($"semCode={semCode} hiddenSize={hiddenSize} logitsSize={logitsSize}");

        using var weights = new FishSpeechWeights(modelPath!);
        var ourLogits = FishSpeechFastAr.Forward(weights, hidden, [semCode]);

        Assert.Equal(logitsSize, ourLogits.Length);

        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < logitsSize; i++)
        {
            dot += ourLogits[i] * refLogits[i];
            na += (double)ourLogits[i] * ourLogits[i];
            nb += refLogits[i] * refLogits[i];
        }
        double cosine = dot / (Math.Sqrt(na) * Math.Sqrt(nb) + 1e-12);
        Console.WriteLine($"fast-AR cb1 logits cosine similarity: {cosine}");

        int refArgmax = 0; double refMax = double.NegativeInfinity;
        int ourArgmax = 0; float ourMax = float.NegativeInfinity;
        for (int i = 0; i < logitsSize; i++)
        {
            if (refLogits[i] > refMax) { refMax = refLogits[i]; refArgmax = i; }
            if (ourLogits[i] > ourMax) { ourMax = ourLogits[i]; ourArgmax = i; }
        }
        Console.WriteLine($"ref argmax={refArgmax} val={refMax:F4}  ours argmax={ourArgmax} val={ourMax:F4}");
    }
}
