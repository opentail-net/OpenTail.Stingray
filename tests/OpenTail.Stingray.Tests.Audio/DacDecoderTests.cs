using System;
using System.IO;
using OpenTail.Stingray.Audio.Parler;
using OpenTail.Stingray.Core;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real numeric golden verification for <see cref="DacDecoder"/> -- compares against
/// `scratch-llamacpp-ref/parler_dac_golden.py`, which uses the real, already-local
/// `models/parler-tts-mini-v1.safetensors` and computes the real DAC decode math directly in
/// numpy, transcribed from the real `descript-audio-codec` package.
/// </summary>
public sealed class DacDecoderTests : HeavyTestBase
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
    public void Decode_RealWeights_MatchesGoldenPcmOutput()
    {
        string? modelPath = FindRepoFile("models/parler-tts-mini-v1.safetensors");
        Assert.SkipUnless(modelPath != null, "models/parler-tts-mini-v1.safetensors not found");

        string? codesPath = FindRepoFile("scratch-llamacpp-ref/parler_dac_golden_codes.txt");
        string? pcmPath = FindRepoFile("scratch-llamacpp-ref/parler_dac_golden_pcm.txt");
        Assert.SkipUnless(codesPath != null && pcmPath != null,
            "golden DAC files not found (re-run scratch-llamacpp-ref/parler_dac_golden.py)");

        var codeLines = File.ReadAllText(codesPath!).Trim().Split('\n');
        var codes = new int[codeLines.Length][];
        for (int i = 0; i < codeLines.Length; i++) codes[i] = Array.ConvertAll(codeLines[i].Split(','), int.Parse);

        var pcmLines = File.ReadAllText(pcmPath!).Split('\n');
        int goldenLen = int.Parse(pcmLines[0].Trim());
        var goldenParts = pcmLines[1].Trim().Split(',');
        Assert.Equal(goldenLen, goldenParts.Length);
        var golden = new float[goldenLen];
        for (int i = 0; i < goldenLen; i++) golden[i] = float.Parse(goldenParts[i]);

        using var loader = SafetensorsLoader.Open(modelPath!);
        var weights = new DacWeights(loader);
        var pcm = DacDecoder.Decode(weights, codes);

        Assert.Equal(goldenLen, pcm.Length);

        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < goldenLen; i++)
        {
            dot += pcm[i] * golden[i];
            normA += pcm[i] * pcm[i];
            normB += golden[i] * golden[i];
        }
        double cosine = dot / (Math.Sqrt(normA) * Math.Sqrt(normB));
        Assert.True(cosine > 0.99, $"cosine similarity {cosine} too low vs golden DAC PCM output");
    }
}
