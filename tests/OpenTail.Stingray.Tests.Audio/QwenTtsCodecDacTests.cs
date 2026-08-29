
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real numeric golden verification for the QwenTTS DAC decoder chain's pre-conv + block 0
/// (SnakeBeta -&gt; causal ConvTranspose1d(kernel=16,stride=8) -&gt; 3x ResidualUnit) -- real GGUF
/// weights, real oracle transcribed from the official DAC decoder / local `dac-decoder-v2.h`.
/// </summary>
public sealed class QwenTtsCodecDacTests : HeavyTestBase
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
    public void PreConvThenBlock0_RealWeights_MatchesGoldenOracle()
    {
        string? modelPath = FindModelPath("qwen-tokenizer-12hz-Q8_0.gguf");
        Assert.SkipUnless(modelPath != null, "models/qwen-tokenizer-12hz-Q8_0.gguf not found");

        string? inputPath = FindRepoFile("scratch-llamacpp-ref/qwentts_dac_block0_golden_input.txt");
        string? outputPath = FindRepoFile("scratch-llamacpp-ref/qwentts_dac_block0_golden_output.txt");
        Assert.SkipUnless(inputPath != null && outputPath != null, "golden DAC block0 fixture not found");

        var (_, inDim, input) = ReadFixture(inputPath!);
        var (goldenT, goldenCh, golden) = ReadFixture(outputPath!);
        Assert.Equal(1024, inDim);

        using var model = GgufModel.Open(modelPath!);
        var weights = new QwenTtsCodecDacWeights(model);

        var preConv = QwenTtsCodecDac.CausalConv1dForTest(input, weights.PreConvWeight, weights.PreConvBias, inCh: 1024, outCh: 1536, kernel: 7, dilation: 1);
        var output = QwenTtsCodecDac.DecoderBlockForTest(preConv, weights.Blocks[0], inCh: 1536, outCh: 768, rate: 8);

        Assert.Equal(goldenT, output.Length);
        Assert.Equal(goldenCh, output[0].Length);

        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < goldenT; i++)
        {
            for (int d = 0; d < goldenCh; d++)
            {
                float a = output[i][d];
                float b = golden[i][d];
                dot += a * b;
                normA += a * a;
                normB += b * b;
            }
        }
        double cosine = dot / (Math.Sqrt(normA) * Math.Sqrt(normB));
        Assert.True(cosine > 0.999, $"cosine similarity {cosine} too low vs golden DAC block0 output");
    }
}
