using OpenTail.Stingray.Diffusion.TextEncoders;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Numeric parity check of <see cref="T5GemmaEncoder"/> against a golden encode dumped from
/// HuggingFace `transformers`' real `T5GemmaEncoderModel`, loaded from the real
/// `stabilityai/stable-audio-3-small-music-base`'s bundled `t5gemma-b-b-ul2/` subfolder (ungated,
/// downloaded locally to `models/stable-audio-3-t5gemma/` -- see
/// docs/057-stable-audio-3-implementation-plan.md). This is Stable Audio 3's real text
/// conditioner, not a placeholder.
/// </summary>
public sealed class StableAudioT5GemmaEncoderGoldenParityTests
{
    private const string EncoderDirRelative = "models/stable-audio-3-t5gemma";

    private static string? FindRepoFile(string relativePath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relativePath.Replace('/', Path.DirectorySeparatorChar));
            if (File.Exists(p) || Directory.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    private static string? FindGoldenDir()
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, "tests", "OpenTail.Stingray.Tests.Diffusion", "TestData", "StableAudioT5GemmaGolden");
            if (Directory.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    private static float CosineSimilarity(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += (double)a[i] * b[i];
            na += (double)a[i] * a[i];
            nb += (double)b[i] * b[i];
        }
        return (float)(dot / (Math.Sqrt(na) * Math.Sqrt(nb) + 1e-12));
    }

    [Fact]
    public void T5GemmaEncoder_MatchesRealTransformersReference_OnRealTokenIds()
    {
        string? encoderDir = FindRepoFile(EncoderDirRelative);
        string? goldenDir = FindGoldenDir();
        if (encoderDir is null || goldenDir is null) return; // skip: needs local T5Gemma weights + fixtures

        using var st = SafetensorsLoader.OpenDirectory(encoderDir);
        using var encoder = T5GemmaEncoder.FromLoader(st);

        var idsBytes = File.ReadAllBytes(Path.Combine(goldenDir, "ids.bin"));
        var ids = new int[idsBytes.Length / 4];
        Buffer.BlockCopy(idsBytes, 0, ids, 0, idsBytes.Length);

        var maskBytes = File.ReadAllBytes(Path.Combine(goldenDir, "mask.bin"));
        var maskInts = new int[maskBytes.Length / 4];
        Buffer.BlockCopy(maskBytes, 0, maskInts, 0, maskBytes.Length);
        var mask = new bool[maskInts.Length];
        for (int i = 0; i < mask.Length; i++) mask[i] = maskInts[i] != 0;

        var outBytes = File.ReadAllBytes(Path.Combine(goldenDir, "output.bin"));
        var goldenOutput = new float[outBytes.Length / 4];
        Buffer.BlockCopy(outBytes, 0, goldenOutput, 0, outBytes.Length);

        var output = encoder.Encode(ids, mask);

        Assert.Equal(goldenOutput.Length, output.Length);
        float cos = CosineSimilarity(output, goldenOutput);
        Assert.True(cos > 0.999f, $"T5Gemma encoder cosine-sim too low: {cos}");
    }
}
