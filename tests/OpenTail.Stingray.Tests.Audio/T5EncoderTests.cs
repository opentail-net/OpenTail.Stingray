
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real numeric golden verification for <see cref="T5Encoder"/> -- compares against
/// `scratch-llamacpp-ref/t5_golden.py`, which runs the REAL `transformers.T5EncoderModel`
/// loaded with the REAL fine-tuned weights sliced out of `parler-tts-mini-v1`'s own
/// `model.safetensors` (not a stock flan-t5-large checkpoint -- see T5EncoderWeights' doc
/// comment for why that would be wrong for this model).
/// </summary>
public sealed class T5EncoderTests : HeavyTestBase
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

    private static float[] ParseCsv(string csv, int expectedLength)
    {
        var parts = csv.Trim().Split(',');
        Assert.Equal(expectedLength, parts.Length);
        var arr = new float[expectedLength];
        for (int i = 0; i < expectedLength; i++) arr[i] = float.Parse(parts[i]);
        return arr;
    }

    [Fact]
    public void Forward_RealWeights_MatchesGoldenOutput()
    {
        string? modelPath = FindRepoFile("models/parler-tts-mini-v1.safetensors");
        Assert.SkipUnless(modelPath != null, "models/parler-tts-mini-v1.safetensors not found");

        string? idsPath = FindRepoFile("scratch-llamacpp-ref/t5_golden_input_ids.txt");
        string? outPath = FindRepoFile("scratch-llamacpp-ref/t5_golden_output.txt");
        Assert.SkipUnless(idsPath != null && outPath != null, "golden T5 files not found (re-run scratch-llamacpp-ref/t5_golden.py)");

        var idsCsv = File.ReadAllText(idsPath!).Trim().Split(',');
        var tokenIds = new int[idsCsv.Length];
        for (int i = 0; i < idsCsv.Length; i++) tokenIds[i] = int.Parse(idsCsv[i]);

        var lines = File.ReadAllText(outPath!).Split('\n');
        var dims = lines[0].Trim().Split(',');
        int goldenT = int.Parse(dims[0]);
        int goldenDim = int.Parse(dims[1]);
        var golden = ParseCsv(lines[1], goldenT * goldenDim);

        using var loader = SafetensorsLoader.Open(modelPath!);
        var weights = new T5EncoderWeights(loader);
        var output = T5Encoder.Forward(weights, tokenIds);

        Assert.Equal(goldenT, output.Length);
        Assert.Equal(goldenDim, output[0].Length);

        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < goldenT; i++)
        {
            for (int d = 0; d < goldenDim; d++)
            {
                float a = output[i][d];
                float b = golden[i * goldenDim + d];
                dot += a * b;
                normA += a * a;
                normB += b * b;
            }
        }
        double cosine = dot / (Math.Sqrt(normA) * Math.Sqrt(normB));
        Assert.True(cosine > 0.99, $"cosine similarity {cosine} too low vs golden T5 encoder output");
    }
}
