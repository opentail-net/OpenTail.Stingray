
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Integration golden test chaining the three real codec components implemented so far
/// (RVQ decode -&gt; pre-conv -&gt; 8-layer transformer) on real codes, catching data-flow/shape
/// bugs between components that each piece's own isolated golden test can't -- e.g. a wrong
/// causal-padding offset or a transposed weight between stages. Same real oracle technique as the
/// per-component tests (real GGUF weights, real Q8_0 dequantization, numpy reference transcribed
/// from the real local `examples/qwentts.cpp` sources).
/// </summary>
public sealed class QwenTtsCodecChainTests : HeavyTestBase
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

    [Fact]
    public void RvqThenPreConvThenTransformer_RealWeights_MatchesGoldenOracle()
    {
        string? modelPath = FindModelPath("qwen-tokenizer-12hz-Q8_0.gguf");
        Assert.SkipUnless(modelPath != null, "models/qwen-tokenizer-12hz-Q8_0.gguf not found");

        string? codesPath = FindRepoFile("scratch-llamacpp-ref/qwentts_codec_chain_golden_codes.txt");
        string? outputPath = FindRepoFile("scratch-llamacpp-ref/qwentts_codec_chain_golden_output.txt");
        Assert.SkipUnless(codesPath != null && outputPath != null, "golden codec chain fixture not found");

        var codeLines = File.ReadAllText(codesPath!).Trim().Split('\n');
        var codes = new int[16][];
        for (int i = 0; i < 16; i++) codes[i] = Array.ConvertAll(codeLines[i].Split(','), int.Parse);

        var outLines = File.ReadAllText(outputPath!).Split('\n');
        var dims = outLines[0].Trim().Split(',');
        int goldenT = int.Parse(dims[0]);
        int goldenDim = int.Parse(dims[1]);
        var goldenParts = outLines[1].Trim().Split(',');
        var golden = new float[goldenT * goldenDim];
        for (int i = 0; i < golden.Length; i++) golden[i] = float.Parse(goldenParts[i]);

        using var model = GgufModel.Open(modelPath!);
        var rvqWeights = new QwenTtsCodecRvqWeights(model);
        var preConvWeights = new QwenTtsCodecPreConvWeights(model);
        var transformerWeights = new QwenTtsCodecTransformerWeights(model);

        var rvqOut = QwenTtsCodecRvq.Decode(rvqWeights, codes);
        var preConvOut = QwenTtsCodecPreConv.Forward(preConvWeights, rvqOut);
        var output = QwenTtsCodecTransformer.Forward(transformerWeights, preConvOut);

        Assert.Equal(goldenT, output.Length);

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
        Assert.True(cosine > 0.999, $"cosine similarity {cosine} too low vs golden codec chain output");
    }
}
