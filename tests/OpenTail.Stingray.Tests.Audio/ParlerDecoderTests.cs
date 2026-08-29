
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

    [Fact]
    public void Forward_RealWeights_MatchesGoldenOutput()
    {
        string? modelPath = FindRepoFile("models/parler-tts-mini-v1.safetensors");
        Assert.SkipUnless(modelPath != null, "models/parler-tts-mini-v1.safetensors not found");

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
        var weights = new ParlerDecoderWeights(loader);

        var inputEmbeds = new float[t][];
        for (int i = 0; i < t; i++) inputEmbeds[i] = ParlerDecoder.EmbedStep(weights, codebookIds[i], i);

        // Fake "encoder" hidden matching the oracle's deterministic 4-position, all-0.05 stand-in.
        var encoderHidden = new float[4][];
        for (int i = 0; i < 4; i++)
        {
            var row = new float[ParlerDecoderWeights.HiddenDim];
            for (int d = 0; d < row.Length; d++) row[d] = 0.05f;
            encoderHidden[i] = row;
        }

        var output = ParlerDecoder.Forward(weights, inputEmbeds, encoderHidden);

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
        Assert.True(cosine > 0.99, $"cosine similarity {cosine} too low vs golden Parler decoder output");
    }
}
