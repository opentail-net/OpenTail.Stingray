using System;
using System.IO;
using OpenTail.Stingray.Audio.FunASR;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real numeric golden verification for <see cref="FunAsrRealDecoder"/> (see docs/audio-review-
/// progress.md's FunASR section) -- compares against `scratch-llamacpp-ref/
/// funasr_golden_decoder.py`, which chains off the already golden-verified encoder+predictor
/// output so this test isolates the decoder's own correctness.
/// </summary>
public sealed class FunAsrRealDecoderTests : HeavyTestBase
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
    public void Forward_RealWeights_MatchesGoldenDecoderLogitsAndArgmax()
    {
        string? modelPath = FindRepoFile("models/paraformer-q8.gguf");
        Assert.SkipUnless(modelPath != null, "models/paraformer-q8.gguf not found");
        string? encInPath = FindRepoFile("scratch-llamacpp-ref/funasr_golden_encoder_input.txt");
        string? logitsPath = FindRepoFile("scratch-llamacpp-ref/funasr_golden_decoder_logits.txt");
        string? tokenIdsPath = FindRepoFile("scratch-llamacpp-ref/funasr_golden_decoder_tokenids.txt");
        Assert.SkipUnless(encInPath != null && logitsPath != null && tokenIdsPath != null,
            "golden decoder files not found (re-run scratch-llamacpp-ref/funasr_golden_decoder.py)");

        const int t = 10, inDim = 560, vocab = 8404;
        var flatIn = ParseCsv(encInPath!, t * inDim);
        var input = new float[t][];
        for (int i = 0; i < t; i++) input[i] = flatIn.AsSpan(i * inDim, inDim).ToArray();

        using var w = new FunAsrWeights(modelPath!);
        var encoderOut = FunAsrEncoder.Forward(w, input);
        var (acousticEmbeds, tokenCount) = FunAsrPredictor.Predict(w, encoderOut);
        var logits = FunAsrRealDecoder.Forward(w, acousticEmbeds, encoderOut);

        Assert.Equal(tokenCount, logits.Length);

        var goldenTokenIds = File.ReadAllText(tokenIdsPath!).Trim().Split(',');
        Assert.Equal(tokenCount, goldenTokenIds.Length);

        for (int i = 0; i < tokenCount; i++)
        {
            int argmax = 0;
            float best = float.NegativeInfinity;
            for (int v = 0; v < vocab; v++)
                if (logits[i][v] > best) { best = logits[i][v]; argmax = v; }

            int goldenId = int.Parse(goldenTokenIds[i]);
            Assert.Equal(goldenId, argmax);
        }

        var goldenLogits = ParseCsv(logitsPath!, tokenCount * vocab);
        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < tokenCount; i++)
        {
            for (int v = 0; v < vocab; v++)
            {
                float a = logits[i][v];
                float b = goldenLogits[i * vocab + v];
                dot += a * b;
                normA += a * a;
                normB += b * b;
            }
        }
        double cosine = dot / (Math.Sqrt(normA) * Math.Sqrt(normB));
        Assert.True(cosine > 0.99, $"cosine similarity {cosine} too low vs golden decoder logits");
    }
}
