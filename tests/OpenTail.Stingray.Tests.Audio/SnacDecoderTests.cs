using System;
using System.IO;
using OpenTail.Stingray.Audio.Orpheus;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real numeric golden verification for <see cref="SnacDecoder"/> -- compares against
/// `scratch-llamacpp-ref/snac_golden.py`'s output, which runs the REAL `hubertsiuzdak/snac_24khz`
/// PyTorch model directly (via the real `snac` pip package), with `NoiseBlock` patched to a
/// documented no-op (matching this port's own convention, see SnacDecoder's class doc comment).
/// Deterministic input: 4 super-frames, every slot filled with code 17 (fill_code, arbitrary
/// non-zero value in the real [0, 4096) codebook range).
/// </summary>
public sealed class SnacDecoderTests : HeavyTestBase
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

    private static float[] ParseCsvLine(string csv, int expectedLength)
    {
        var parts = csv.Trim().Split(',');
        Assert.Equal(expectedLength, parts.Length);
        var arr = new float[expectedLength];
        for (int i = 0; i < expectedLength; i++) arr[i] = float.Parse(parts[i]);
        return arr;
    }

    [Fact]
    public void Decode_RealWeights_MatchesGoldenPcmOutput()
    {
        string? modelPath = FindRepoFile("models/snac-24khz.gguf");
        Assert.SkipUnless(modelPath != null, "models/snac-24khz.gguf not found");

        string? pcmPath = FindRepoFile("scratch-llamacpp-ref/snac_golden_pcm.txt");
        Assert.SkipUnless(pcmPath != null, "golden SNAC PCM file not found (re-run scratch-llamacpp-ref/snac_golden.py)");

        // Deterministic 4-super-frame input, every slot = code 17, matching snac_golden.py exactly.
        const int numSuperFrames = 4;
        const int fillCode = 17;
        var codes = new int[3][]
        {
            new int[numSuperFrames * 1],
            new int[numSuperFrames * 2],
            new int[numSuperFrames * 4],
        };
        for (int i = 0; i < codes[0].Length; i++) codes[0][i] = fillCode;
        for (int i = 0; i < codes[1].Length; i++) codes[1][i] = fillCode;
        for (int i = 0; i < codes[2].Length; i++) codes[2][i] = fillCode;

        using var w = new SnacWeights(modelPath!);
        var pcm = SnacDecoder.Decode(w, codes);

        var lines = File.ReadAllText(pcmPath!).Split('\n');
        int goldenLen = int.Parse(lines[0].Trim());
        var golden = ParseCsvLine(lines[1], goldenLen);

        Assert.Equal(goldenLen, pcm.Length);

        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < goldenLen; i++)
        {
            dot += pcm[i] * golden[i];
            normA += pcm[i] * pcm[i];
            normB += golden[i] * golden[i];
        }
        double cosine = dot / (Math.Sqrt(normA) * Math.Sqrt(normB));
        Assert.True(cosine > 0.99, $"cosine similarity {cosine} too low vs golden SNAC PCM output");
    }
}
