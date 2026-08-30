using OpenTail.Stingray.Audio.Xtts;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real numeric golden verification for <see cref="XttsConditioningEncoder"/>, against
/// `scratch-llamacpp-ref/xtts_conditioning_golden.py`'s real `model.gpt.get_style_emb(...)`
/// output on a fixed deterministic mel input.</summary>
public sealed class XttsConditioningEncoderTests : HeavyTestBase
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

    private static float[] ReadCsv(string path) =>
        Array.ConvertAll(File.ReadAllText(path).Trim().Split(','), float.Parse);

    private static double Cosine(float[] a, float[] b)
    {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
        return dot / (Math.Sqrt(na) * Math.Sqrt(nb));
    }

    [Fact]
    public void Encode_RealWeights_MatchesGoldenOracle()
    {
        string? weightsPath = FindRepoFile("models/xtts-v2/model.safetensors");
        Assert.SkipUnless(weightsPath != null, "models/xtts-v2/model.safetensors not found");
        string? melPath = FindRepoFile("scratch-llamacpp-ref/xtts_conditioning_golden_mel.txt");
        string? outPath = FindRepoFile("scratch-llamacpp-ref/xtts_conditioning_golden_output.txt");
        Assert.SkipUnless(melPath != null && outPath != null, "golden conditioning files not found (re-run scratch-llamacpp-ref/xtts_conditioning_golden.py)");

        using var loader = SafetensorsLoader.Open(weightsPath!);
        var weights = new XttsConditioningWeights(loader);

        // Golden mel is real PyTorch [1, 80, T] -- already channel-first, no transpose needed.
        float[] mel = ReadCsv(melPath!);
        int t = mel.Length / XttsConditioningWeights.MelDim;

        float[] output = XttsConditioningEncoder.Encode(weights, mel, t);

        // Golden output is real PyTorch [1, 1024, 32] -- already channel-first too.
        float[] golden = ReadCsv(outPath!);

        Assert.Equal(golden.Length, output.Length);
        foreach (float v in output)
        {
            Assert.False(float.IsNaN(v), "conditioning output must not contain NaN");
            Assert.False(float.IsInfinity(v), "conditioning output must not contain Infinity");
        }
        double cosine = Cosine(output, golden);
        Assert.True(cosine > 0.99, $"conditioning output cosine {cosine} too low vs golden");
    }
}
