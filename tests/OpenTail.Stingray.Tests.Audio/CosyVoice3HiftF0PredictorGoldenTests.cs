using System;
using System.IO;
using OpenTail.Stingray.Audio.CosyVoice;
using OpenTail.Stingray.Audio.Primitives;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real numeric golden verification for CosyVoice3's HiFT F0 predictor (`HiFTVocoderKernels.
/// PredictF0`, real weights via <see cref="CosyVoice3HiftWeights.F0Predictor"/>). Carved out as
/// its own deterministic golden test because the rest of `CosyVoiceHiftVocoder.Generate`'s chain
/// (`SineGen`'s NSF excitation) consumes a real `System.Random` stream that has no numpy
/// equivalent to reproduce bit-for-bit -- F0 prediction itself has no such randomness and is
/// fully golden-verifiable the same way `InputEmbed`/`RunBackbone` were for the DiT.
/// </summary>
public sealed class CosyVoice3HiftF0PredictorGoldenTests : HeavyTestBase
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

    [Fact]
    public void PredictF0_RealWeights_MatchesGoldenOracle()
    {
        string? modelPath = FindRepoFile("models/cosyvoice3/CosyVoice3-2512_F16.gguf");
        Assert.SkipUnless(modelPath != null, "models/cosyvoice3/CosyVoice3-2512_F16.gguf not found");

        string? inputPath = FindRepoFile("scratch-llamacpp-ref/cosyvoice3_hift_f0predictor_golden_input.txt");
        string? outputPath = FindRepoFile("scratch-llamacpp-ref/cosyvoice3_hift_f0predictor_golden_output.txt");
        Assert.SkipUnless(inputPath != null && outputPath != null, "golden HiFT F0 predictor fixture not found");

        var inLines = File.ReadAllText(inputPath!).Split('\n');
        int t = int.Parse(inLines[0].Trim());
        var mel = Array.ConvertAll(inLines[1].Trim().Split(','), float.Parse); // channel-first [80, T] flat

        var golden = Array.ConvertAll(File.ReadAllText(outputPath!).Trim().Split(','), float.Parse);
        Assert.Equal(t, golden.Length);

        using var w = new CosyVoice3HiftWeights(modelPath!);

        var f0 = HiFTVocoderKernels.PredictF0ForTest(w.F0Predictor, mel, t, melDim: 80);

        Assert.Equal(golden.Length, f0.Length);

        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < f0.Length; i++)
        {
            dot += f0[i] * golden[i];
            normA += f0[i] * f0[i];
            normB += golden[i] * golden[i];
        }
        double cosine = dot / (Math.Sqrt(normA) * Math.Sqrt(normB));
        Assert.True(cosine > 0.999, $"cosine similarity {cosine} too low vs golden F0 predictor output");
    }
}
