using OpenTail.Stingray.Audio.Xtts;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real numeric golden verification for <see cref="XttsGptTrunk"/> (the bare, standard
/// HF GPT2 trunk in isolation -- no text/audio tokenization or conditioning), against
/// `scratch-llamacpp-ref/xtts_gpt_trunk_golden.py`'s real `model.gpt.gpt(inputs_embeds=...)`
/// output. Isolates the Conv1D-transpose/attention/MLP math from everything else in the port.</summary>
public sealed class XttsGptTrunkTests : HeavyTestBase
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
    public void Forward_RealWeights_MatchesGoldenOracle()
    {
        string? weightsPath = FindRepoFile("models/xtts-v2/model.safetensors");
        Assert.SkipUnless(weightsPath != null, "models/xtts-v2/model.safetensors not found (run scratch-llamacpp-ref/xtts_convert_to_safetensors.py)");
        string? inputPath = FindRepoFile("scratch-llamacpp-ref/xtts_gpt_trunk_golden_input.txt");
        string? outputPath = FindRepoFile("scratch-llamacpp-ref/xtts_gpt_trunk_golden_output.txt");
        Assert.SkipUnless(inputPath != null && outputPath != null, "golden GPT trunk files not found (re-run scratch-llamacpp-ref/xtts_gpt_trunk_golden.py)");

        using var loader = SafetensorsLoader.Open(weightsPath!);
        var weights = new XttsGptWeights(loader);

        // Golden input is real PyTorch [1, T, dim] token-major (index = t*dim+d) -- transpose to
        // this codebase's channel-first [dim, T] (index = d*T+t) convention.
        float[] tokenMajor = ReadCsv(inputPath!);
        const int dim = XttsGptWeights.ModelDim;
        int t = tokenMajor.Length / dim;
        var channelFirst = new float[dim * t];
        for (int ti = 0; ti < t; ti++)
            for (int d = 0; d < dim; d++)
                channelFirst[d * t + ti] = tokenMajor[ti * dim + d];

        float[] output = XttsGptTrunk.Forward(weights, channelFirst, t);

        // Golden output is also real PyTorch [1, T, dim] token-major -- transpose for comparison.
        float[] goldenTokenMajor = ReadCsv(outputPath!);
        var goldenChannelFirst = new float[goldenTokenMajor.Length];
        for (int ti = 0; ti < t; ti++)
            for (int d = 0; d < dim; d++)
                goldenChannelFirst[d * t + ti] = goldenTokenMajor[ti * dim + d];

        Assert.Equal(goldenChannelFirst.Length, output.Length);
        foreach (float v in output)
        {
            Assert.False(float.IsNaN(v), "GPT trunk output must not contain NaN");
            Assert.False(float.IsInfinity(v), "GPT trunk output must not contain Infinity");
        }
        double cosine = Cosine(output, goldenChannelFirst);
        Assert.True(cosine > 0.99, $"GPT trunk output cosine {cosine} too low vs golden");
    }
}
