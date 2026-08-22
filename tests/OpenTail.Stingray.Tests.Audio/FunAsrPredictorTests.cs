using System;
using System.IO;
using OpenTail.Stingray.Audio.FunASR;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real numeric golden verification for <see cref="FunAsrPredictor"/> (see docs/audio-review-
/// progress.md's FunASR section) -- compares against `scratch-llamacpp-ref/
/// funasr_golden_predictor.py`, which chains off the already golden-verified encoder output
/// (same golden encoder input as <see cref="FunAsrEncoderTests"/>) so this test isolates the
/// predictor's own correctness.
/// </summary>
public sealed class FunAsrPredictorTests : HeavyTestBase
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

    private static float[] ParseCsv(string path, int expectedLength)
    {
        var parts = File.ReadAllText(path).Trim().Split(',');
        Assert.Equal(expectedLength, parts.Length);
        var arr = new float[expectedLength];
        for (int i = 0; i < expectedLength; i++) arr[i] = float.Parse(parts[i]);
        return arr;
    }

    [Fact]
    public void Predict_RealWeights_MatchesGoldenAlphasAndEmbeds()
    {
        string? modelPath = FindRepoFile("models/paraformer-q8.gguf");
        Assert.SkipUnless(modelPath != null, "models/paraformer-q8.gguf not found");
        string? encInPath = FindRepoFile("scratch-llamacpp-ref/funasr_golden_encoder_input.txt");
        string? alphasPath = FindRepoFile("scratch-llamacpp-ref/funasr_golden_predictor_alphas.txt");
        string? embedsPath = FindRepoFile("scratch-llamacpp-ref/funasr_golden_predictor_embeds.txt");
        Assert.SkipUnless(encInPath != null && alphasPath != null && embedsPath != null,
            "golden predictor files not found (re-run scratch-llamacpp-ref/funasr_golden_predictor.py)");

        const int t = 10, inDim = 560, outDim = 512;
        var flatIn = ParseCsv(encInPath!, t * inDim);
        var input = new float[t][];
        for (int i = 0; i < t; i++) input[i] = flatIn.AsSpan(i * inDim, inDim).ToArray();

        using var w = new FunAsrWeights(modelPath!);
        var encoderOut = FunAsrEncoder.Forward(w, input);
        var (acousticEmbeds, tokenCount) = FunAsrPredictor.Predict(w, encoderOut);

        var lines = File.ReadAllText(embedsPath!).Split('\n');
        int goldenTokenCount = int.Parse(lines[0].Trim());
        Assert.Equal(goldenTokenCount, tokenCount);

        // Parse the embeds line (second line) directly since it may be empty when tokenCount==0.
        string embedsLine = lines.Length > 1 ? lines[1] : "";
        if (goldenTokenCount > 0)
        {
            var parts = embedsLine.Trim().Split(',');
            Assert.Equal(goldenTokenCount * outDim, parts.Length);
            double dot = 0, normA = 0, normB = 0;
            int idx = 0;
            for (int i = 0; i < goldenTokenCount; i++)
            {
                for (int d = 0; d < outDim; d++)
                {
                    float a = acousticEmbeds[i][d];
                    float b = float.Parse(parts[idx++]);
                    dot += a * b;
                    normA += a * a;
                    normB += b * b;
                }
            }
            double cosine = dot / (Math.Sqrt(normA) * Math.Sqrt(normB));
            Assert.True(cosine > 0.99, $"cosine similarity {cosine} too low vs golden predictor acoustic_embeds");
        }
    }
}
