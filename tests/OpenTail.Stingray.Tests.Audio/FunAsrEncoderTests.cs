
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real numeric golden verification for <see cref="FunAsrEncoder"/> (see docs/audio-review-
/// progress.md's FunASR section) -- compares against `scratch-llamacpp-ref/
/// funasr_golden_encoder.py`, an independent from-scratch numpy re-implementation of the real
/// SAN-M encoder formula (transcribed from the real `funasr` Python package) applied to the
/// SAME real GGUF weights via the `gguf` Python package's dequantizer. We don't have the
/// original PyTorch checkpoint locally (only the GGUF conversion), so this -- not literally
/// running `funasr` itself -- is the independent oracle here, same spirit as every other
/// pipeline's golden-dump script in this repo.
/// </summary>
public sealed class FunAsrEncoderTests : HeavyTestBase
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

    private static float[] ParseCsv(string path, int expectedLength)
    {
        var parts = File.ReadAllText(path).Trim().Split(',');
        Assert.Equal(expectedLength, parts.Length);
        var arr = new float[expectedLength];
        for (int i = 0; i < expectedLength; i++) arr[i] = float.Parse(parts[i]);
        return arr;
    }

    [Fact]
    public void Forward_RealWeights_MatchesGoldenEncoderOutput()
    {
        string? modelPath = FindRepoFile("models/paraformer-q8.gguf");
        Assert.SkipUnless(modelPath != null, "models/paraformer-q8.gguf not found");
        string? inPath = FindRepoFile("scratch-llamacpp-ref/funasr_golden_encoder_input.txt");
        string? outPath = FindRepoFile("scratch-llamacpp-ref/funasr_golden_encoder_output.txt");
        Assert.SkipUnless(inPath != null && outPath != null, "golden encoder input/output not found (re-run scratch-llamacpp-ref/funasr_golden_encoder.py)");

        const int t = 10, inDim = 560, outDim = 512;
        var flatIn = ParseCsv(inPath!, t * inDim);
        var flatOutGolden = ParseCsv(outPath!, t * outDim);

        var input = new float[t][];
        for (int i = 0; i < t; i++) input[i] = flatIn.AsSpan(i * inDim, inDim).ToArray();

        using var w = new FunAsrWeights(modelPath!);
        var output = FunAsrEncoder.Forward(w, input);

        Assert.Equal(t, output.Length);
        Assert.Equal(outDim, output[0].Length);

        // Cosine similarity over the flattened output, matching this repo's established bar
        // (>0.99) for a correct-but-not-necessarily-bitwise-identical port.
        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < t; i++)
        {
            for (int d = 0; d < outDim; d++)
            {
                float a = output[i][d];
                float b = flatOutGolden[i * outDim + d];
                dot += a * b;
                normA += a * a;
                normB += b * b;
            }
        }
        double cosine = dot / (Math.Sqrt(normA) * Math.Sqrt(normB));
        Assert.True(cosine > 0.99, $"cosine similarity {cosine} too low vs golden encoder output");
    }
}
