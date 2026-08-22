using System;
using System.IO;
using OpenTail.Stingray.Audio.FishSpeech;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real numeric golden verification for <see cref="FishSpeechFastAr"/> -- compares against
/// `scratch-llamacpp-ref/fish_speech_fastar_golden.py`, which fetches the real 4-layer
/// `audio_decoder.*` weights (~800MB) directly from the real `fishaudio/s2-pro` safetensors via
/// byte-range HTTP requests (no full 9.1GB download) and computes the real math in numpy,
/// transcribed from `fish_speech/models/text2semantic/llama.py`. Deterministic input: hidden
/// state = all 0.1, prefix codebook values = [7, 42, 99].
/// </summary>
public sealed class FishSpeechFastArTests : HeavyTestBase
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
    public void Forward_RealWeights_MatchesGoldenOracle()
    {
        string? modelPath = FindModelPath("s2-pro-q4_k_m.gguf");
        Assert.SkipUnless(modelPath != null, "models/s2-pro-q4_k_m.gguf not found");

        string? inputPath = FindRepoFile("scratch-llamacpp-ref/fishspeech_fastar_golden_input.txt");
        string? logitsPath = FindRepoFile("scratch-llamacpp-ref/fishspeech_fastar_golden_logits.txt");
        Assert.SkipUnless(inputPath != null && logitsPath != null,
            "golden fast-AR files not found (re-run scratch-llamacpp-ref/fish_speech_fastar_golden.py)");

        var inputLines = File.ReadAllText(inputPath!).Trim().Split('\n');
        var hiddenCsv = inputLines[0].Split(',');
        var hidden = new float[hiddenCsv.Length];
        for (int i = 0; i < hidden.Length; i++) hidden[i] = float.Parse(hiddenCsv[i]);

        var prefixCsv = inputLines[1].Split(',');
        var prefix = new int[prefixCsv.Length];
        for (int i = 0; i < prefix.Length; i++) prefix[i] = int.Parse(prefixCsv[i]);

        var logitsLines = File.ReadAllText(logitsPath!).Trim().Split('\n');
        int codebookSize = int.Parse(logitsLines[0]);
        var goldenParts = logitsLines[1].Split(',');
        Assert.Equal(codebookSize, goldenParts.Length);
        var golden = new float[codebookSize];
        for (int i = 0; i < codebookSize; i++) golden[i] = float.Parse(goldenParts[i]);

        using var weights = new FishSpeechWeights(modelPath!);
        Assert.Equal(hidden.Length, weights.EmbeddingDim);

        var logits = FishSpeechFastAr.Forward(weights, hidden, prefix);
        Assert.Equal(codebookSize, logits.Length);

        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < codebookSize; i++)
        {
            dot += logits[i] * golden[i];
            normA += logits[i] * logits[i];
            normB += golden[i] * golden[i];
        }
        double cosine = dot / (Math.Sqrt(normA) * Math.Sqrt(normB));
        Assert.True(cosine > 0.99, $"cosine similarity {cosine} too low vs golden fast-AR logits");
    }

    /// <summary>Same golden comparison but against the Q8_0 (near-lossless) quant instead of Q4_K_M -- tests whether the Q4_K_M mismatch found earlier this fire is quantization-compounding (cosine should improve substantially) or a real code bug (cosine would stay low).</summary>
    [Fact]
    public void Forward_Q8_0Weights_MatchesGoldenOracle()
    {
        string? modelPath = FindModelPath("s2-pro-q8_0.gguf");
        Assert.SkipUnless(modelPath != null, "models/s2-pro-q8_0.gguf not found");

        string? inputPath = FindRepoFile("scratch-llamacpp-ref/fishspeech_fastar_golden_input.txt");
        string? logitsPath = FindRepoFile("scratch-llamacpp-ref/fishspeech_fastar_golden_logits.txt");
        Assert.SkipUnless(inputPath != null && logitsPath != null,
            "golden fast-AR files not found (re-run scratch-llamacpp-ref/fish_speech_fastar_golden.py)");

        var inputLines = File.ReadAllText(inputPath!).Trim().Split('\n');
        var hiddenCsv = inputLines[0].Split(',');
        var hidden = new float[hiddenCsv.Length];
        for (int i = 0; i < hidden.Length; i++) hidden[i] = float.Parse(hiddenCsv[i]);

        var prefixCsv = inputLines[1].Split(',');
        var prefix = new int[prefixCsv.Length];
        for (int i = 0; i < prefix.Length; i++) prefix[i] = int.Parse(prefixCsv[i]);

        var logitsLines = File.ReadAllText(logitsPath!).Trim().Split('\n');
        int codebookSize = int.Parse(logitsLines[0]);
        var goldenParts = logitsLines[1].Split(',');
        var golden = new float[codebookSize];
        for (int i = 0; i < codebookSize; i++) golden[i] = float.Parse(goldenParts[i]);

        using var weights = new FishSpeechWeights(modelPath!);
        var logits = FishSpeechFastAr.Forward(weights, hidden, prefix);

        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < codebookSize; i++)
        {
            dot += logits[i] * golden[i];
            normA += logits[i] * logits[i];
            normB += golden[i] * golden[i];
        }
        double cosine = dot / (Math.Sqrt(normA) * Math.Sqrt(normB));

        File.WriteAllText(Path.Combine(Path.GetTempPath(), "fishspeech_fastar_q8_cosine.txt"), $"cosine={cosine}\n");
        Assert.True(cosine > 0.99, $"cosine similarity {cosine} too low vs golden fast-AR logits (Q8_0)");
    }
}
