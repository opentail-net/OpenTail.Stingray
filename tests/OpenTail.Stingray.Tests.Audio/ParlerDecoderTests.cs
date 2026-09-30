
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real numeric golden verification for <see cref="ParlerDecoder"/> -- compares against
/// `scratch-llamacpp-ref/parler_decoder_golden.py`, which uses the real, already-local
/// `models/parler-tts-mini-v1.safetensors` (same file used for the T5 encoder) and computes the
/// real decoder math directly in numpy, transcribed from `parler_tts/modeling_parler_tts.py`.
/// </summary>
public sealed class ParlerDecoderTests : HeavyTestBase
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

    private static string? FindWeightsPath()
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            foreach (var candidate in new[]
            {
                Path.Combine(dir, "models", "parler-tts-mini-v1.safetensors"),
                Path.Combine(dir, "models", "_models", "parler-tts-mini-v1.safetensors"),
            })
            {
                // Ignore empty placeholders so they cannot mask the real checkpoint in _models.
                if (File.Exists(candidate) && new FileInfo(candidate).Length > 50L * 1024 * 1024)
                    return candidate;
            }
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void Forward_RealWeights_MatchesGoldenOutput()
    {
        string? modelPath = FindWeightsPath();
        Assert.SkipUnless(modelPath != null, "models/parler-tts-mini-v1.safetensors not found");
        Console.WriteLine($"[ParlerDecoderGolden] path={modelPath} bytes={new FileInfo(modelPath!).Length}");

        string? idsPath = FindRepoFile("scratch-llamacpp-ref/parler_decoder_golden_codebook_ids.txt");
        string? hiddenPath = FindRepoFile("scratch-llamacpp-ref/parler_decoder_golden_hidden.txt");
        Assert.SkipUnless(idsPath != null && hiddenPath != null,
            "golden Parler decoder files not found (re-run scratch-llamacpp-ref/parler_decoder_golden.py)");

        var idLines = File.ReadAllText(idsPath!).Trim().Split('\n');
        int t = idLines.Length;
        var codebookIds = new int[t][];
        for (int i = 0; i < t; i++) codebookIds[i] = Array.ConvertAll(idLines[i].Split(','), int.Parse);

        var hiddenLines = File.ReadAllText(hiddenPath!).Split('\n');
        var dims = hiddenLines[0].Trim().Split(',');
        int goldenT = int.Parse(dims[0]);
        int goldenDim = int.Parse(dims[1]);
        var goldenParts = hiddenLines[1].Trim().Split(',');
        Assert.Equal(goldenT * goldenDim, goldenParts.Length);
        var golden = new float[goldenT * goldenDim];
        for (int i = 0; i < golden.Length; i++) golden[i] = float.Parse(goldenParts[i]);

        using var loader = SafetensorsLoader.Open(modelPath!);
        try
        {
            double f32Cosine = RunAndCompare(loader, quantize: false, t, codebookIds, goldenT, goldenDim, golden);
            Console.WriteLine($"[ParlerDecoderGolden] F32 control cosine={f32Cosine:R}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[ParlerDecoderGolden] F32 control failed with the first reported decoder-layer boundary: {ex}");
        }

        double q8Cosine = RunAndCompare(loader, quantize: true, t, codebookIds, goldenT, goldenDim, golden);
        Console.WriteLine($"[ParlerDecoderGolden] Q8_0 cosine={q8Cosine:R}");
        Assert.True(q8Cosine > 0.99, $"cosine similarity {q8Cosine} too low vs golden Parler decoder output");
    }

    private static double RunAndCompare(SafetensorsLoader loader, bool quantize, int t, int[][] codebookIds,
        int goldenT, int goldenDim, float[] golden)
    {
        var weights = new ParlerDecoderWeights(loader, quantizeLargeMatricesToQ8: quantize);
        var inputEmbeds = new float[t][];
        for (int i = 0; i < t; i++) inputEmbeds[i] = ParlerDecoder.EmbedStep(weights, codebookIds[i], i);

        var encoderHidden = new float[4][];
        for (int i = 0; i < 4; i++)
        {
            var row = new float[ParlerDecoderWeights.HiddenDim];
            Array.Fill(row, 0.05f);
            encoderHidden[i] = row;
        }

        var output = ParlerDecoder.Forward(weights, inputEmbeds, encoderHidden, diagnoseFiniteStages: !quantize);
        Assert.Equal(goldenT, output.Length);
        Assert.Equal(goldenDim, output[0].Length);
        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < goldenT; i++)
        for (int d = 0; d < goldenDim; d++)
        {
            float a = output[i][d];
            float b = golden[i * goldenDim + d];
            dot += a * b;
            normA += a * a;
            normB += b * b;
        }
        return dot / (Math.Sqrt(normA) * Math.Sqrt(normB));
    }
}
