
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real numeric golden verification for <see cref="QwenTtsCodecTransformer"/> -- compares against
/// a real oracle built directly from the real, already-local `models/qwen-tokenizer-12hz-Q8_0.gguf`
/// weights via the `gguf` Python package (with a real, hand-rolled Q8_0 block dequantizer -- this
/// package does NOT auto-dequantize Q8_0 tensors, confirmed empirically this fire), computing the
/// real 8-layer transformer math (full MHA, NEOX RoPE theta=10000, causal sliding-window
/// attention window=72, per-layer LayerScale) directly in numpy, transcribed from the real, local
/// `examples/qwentts.cpp/src/tokenizer-transformer.h`.
/// </summary>
public sealed class QwenTtsCodecTransformerTests : HeavyTestBase
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
        Assert.Equal(t * dim, parts.Length);
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
        string? modelPath = FindModelPath("qwen-tokenizer-12hz-Q8_0.gguf");
        Assert.SkipUnless(modelPath != null, "models/qwen-tokenizer-12hz-Q8_0.gguf not found");

        string? inputPath = FindRepoFile("scratch-llamacpp-ref/qwentts_codec_transformer_golden_input.txt");
        string? outputPath = FindRepoFile("scratch-llamacpp-ref/qwentts_codec_transformer_golden_output.txt");
        Assert.SkipUnless(inputPath != null && outputPath != null, "golden codec transformer fixture not found");

        var (t, latentDim, input) = ReadFixture(inputPath!);
        var (goldenT, goldenDim, golden) = ReadFixture(outputPath!);
        Assert.Equal(t, goldenT);
        Assert.Equal(latentDim, goldenDim);

        using var model = GgufModel.Open(modelPath!);
        var weights = new QwenTtsCodecTransformerWeights(model);
        var output = QwenTtsCodecTransformer.Forward(weights, input);

        Assert.Equal(t, output.Length);

        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < t; i++)
        {
            for (int d = 0; d < latentDim; d++)
            {
                float a = output[i][d];
                float b = golden[i][d];
                dot += a * b;
                normA += a * a;
                normB += b * b;
            }
        }
        double cosine = dot / (Math.Sqrt(normA) * Math.Sqrt(normB));
        Assert.True(cosine > 0.999, $"cosine similarity {cosine} too low vs golden codec transformer output");
    }
}
