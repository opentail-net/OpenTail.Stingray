
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real numeric golden verification for <see cref="QwenTtsSpeakerEncoder"/> -- real GGUF
/// weights (`spk_enc.*`, stored in the Talker GGUF), real oracle transcribed from the official
/// `TimeDelayNetBlock`/`Res2NetBlock`/`SqueezeExcitation`/`AttentiveStatisticsPooling`/
/// `SpeakerEncoder` (`qwen_tts/core/models/modeling_qwen3_tts.py`). Operates directly on a
/// deterministic real log-mel input, isolating the encoder network from mel-frontend extraction
/// (same isolation strategy used for the codec's deterministic-codes RVQ test).
/// </summary>
public sealed class QwenTtsSpeakerEncoderTests : HeavyTestBase
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

    private static (int T, int Dim, float[][] Rows) ReadFixture(string path)
    {
        var lines = File.ReadAllText(path).Split('\n');
        var dims = lines[0].Trim().Split(',');
        int t = int.Parse(dims[0]);
        int dim = int.Parse(dims[1]);
        var parts = lines[1].Trim().Split(',');
        var rows = new float[t][];
        for (int i = 0; i < t; i++)
        {
            rows[i] = new float[dim];
            for (int d = 0; d < dim; d++) rows[i][d] = float.Parse(parts[i * dim + d]);
        }
        return (t, dim, rows);
    }

    [Fact]
    public void Forward_RealWeights_MatchesGoldenOracle()
    {
        string? modelPath = FindModelPath("qwen-talker-0.6b-base-Q8_0.gguf");
        Assert.SkipUnless(modelPath != null, "models/qwen-talker-0.6b-base-Q8_0.gguf not found");

        string? inputPath = FindRepoFile("scratch-llamacpp-ref/qwentts_spkenc_golden_input.txt");
        string? outputPath = FindRepoFile("scratch-llamacpp-ref/qwentts_spkenc_golden_output.txt");
        Assert.SkipUnless(inputPath != null && outputPath != null, "golden speaker encoder fixture not found");

        var (_, melDim, mel) = ReadFixture(inputPath!);
        var (_, embDim, goldenRows) = ReadFixture(outputPath!);
        Assert.Equal(128, melDim);
        var golden = goldenRows[0];

        using var model = GgufModel.Open(modelPath!);
        var weights = new QwenTtsSpeakerEncoderWeights(model);
        Assert.Equal(embDim, weights.EncDim);

        var embedding = QwenTtsSpeakerEncoder.Forward(weights, mel);

        Assert.Equal(embDim, embedding.Length);

        double dot = 0, normA = 0, normB = 0;
        for (int d = 0; d < embDim; d++)
        {
            float a = embedding[d];
            float b = golden[d];
            dot += a * b;
            normA += a * a;
            normB += b * b;
        }
        double cosine = dot / (Math.Sqrt(normA) * Math.Sqrt(normB));
        Assert.True(cosine > 0.999, $"cosine similarity {cosine} too low vs golden speaker embedding");
    }
}
