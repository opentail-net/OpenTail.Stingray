using System;
using System.IO;
using OpenTail.Stingray.Audio.QwenTTS;
using OpenTail.Stingray.Core;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real numeric golden verification for <see cref="QwenTtsCodecRvq"/> -- compares against a real
/// oracle (`scratch-llamacpp-ref/qwentts_rvq_golden_*.txt`) built directly from the real,
/// already-local `models/qwen-tokenizer-12hz-Q8_0.gguf` weights via the `gguf` Python package's
/// dequantization, computing the real split-RVQ math (per-group codebook sum in 256-dim internal
/// space, then that group's own 256-&gt;512 projection, then sum semantic+acoustic) directly in
/// numpy, transcribed from the real `quantizer-decode.h`/official `SplitResidualVectorQuantizer`.
/// </summary>
public sealed class QwenTtsCodecRvqTests : HeavyTestBase
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
    public void Decode_RealWeights_MatchesGoldenOracle()
    {
        string? modelPath = FindModelPath("qwen-tokenizer-12hz-Q8_0.gguf");
        Assert.SkipUnless(modelPath != null, "models/qwen-tokenizer-12hz-Q8_0.gguf not found");

        string? codesPath = FindRepoFile("scratch-llamacpp-ref/qwentts_rvq_golden_codes.txt");
        string? outputPath = FindRepoFile("scratch-llamacpp-ref/qwentts_rvq_golden_output.txt");
        Assert.SkipUnless(codesPath != null && outputPath != null, "golden RVQ fixture not found");

        var codeLines = File.ReadAllText(codesPath!).Trim().Split('\n');
        var codes = new int[16][];
        for (int i = 0; i < 16; i++) codes[i] = Array.ConvertAll(codeLines[i].Split(','), int.Parse);

        var outLines = File.ReadAllText(outputPath!).Split('\n');
        var dims = outLines[0].Trim().Split(',');
        int goldenT = int.Parse(dims[0]);
        int goldenHidden = int.Parse(dims[1]);
        var goldenParts = outLines[1].Trim().Split(',');
        Assert.Equal(goldenT * goldenHidden, goldenParts.Length);
        var golden = new float[goldenT * goldenHidden];
        for (int i = 0; i < golden.Length; i++) golden[i] = float.Parse(goldenParts[i]);

        using var model = GgufModel.Open(modelPath!);
        var weights = new QwenTtsCodecRvqWeights(model);
        var output = QwenTtsCodecRvq.Decode(weights, codes);

        Assert.Equal(goldenT, output.Length);

        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < goldenT; i++)
        {
            for (int d = 0; d < goldenHidden; d++)
            {
                float a = output[i][d];
                float b = golden[i * goldenHidden + d];
                dot += a * b;
                normA += a * a;
                normB += b * b;
            }
        }
        double cosine = dot / (Math.Sqrt(normA) * Math.Sqrt(normB));
        Assert.True(cosine > 0.999, $"cosine similarity {cosine} too low vs golden RVQ decode output");
    }
}
