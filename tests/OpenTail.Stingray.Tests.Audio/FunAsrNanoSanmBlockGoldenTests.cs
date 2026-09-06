
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real golden-verification test for FunAsrNanoSanmBlock against
/// `examples/audio.cpp/tests/fun_asr_nano/sanm_reference.{json,bin}` -- a real, small reference
/// fixture generated directly from the real `transformers` `FunAsrNanoEncoderStem`/
/// `FunAsrNanoEncoderLayer` PyTorch implementation (real weights, real input, real expected
/// output/checkpoints), NOT synthetic. Covers both block variants: the "projection" (stem, no
/// input residual, input_size != model_size) and "residual" (main/timestamp layer, input_size ==
/// model_size) forms.</summary>
public sealed class FunAsrNanoSanmBlockGoldenTests : HeavyTestBase
{
    private static string? FindRepoFile(string relPath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relPath);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void ProjectionAndResidualBlocks_MatchRealPyTorchReference()
    {
        string? jsonPath = FindRepoFile("examples/audio.cpp/tests/fun_asr_nano/sanm_reference.json");
        string? binPath = FindRepoFile("examples/audio.cpp/tests/fun_asr_nano/sanm_reference.bin");
        Assert.SkipUnless(jsonPath != null && binPath != null, "sanm_reference.{json,bin} not found");

        var data = ReadAllF32(binPath!);
        using var doc = System.Text.Json.JsonDocument.Parse(File.ReadAllText(jsonPath!));
        var root = doc.RootElement;
        var config = root.GetProperty("config");
        int inputSize = config.GetProperty("input_size").GetInt32();
        int modelSize = config.GetProperty("model_size").GetInt32();
        int numHeads = config.GetProperty("num_heads").GetInt32();
        int ffnSize = config.GetProperty("ffn_size").GetInt32();
        int kernelSize = config.GetProperty("fsmn_kernel_size").GetInt32();
        float eps = config.GetProperty("layer_norm_eps").GetSingle();
        int frames = config.GetProperty("frames").GetInt32();

        var blocks = root.GetProperty("blocks");

        // --- Projection block (stem): input_size -> model_size, no input residual ---
        var projection = blocks.GetProperty("projection");
        var projInput = ReadRows(data, projection.GetProperty("input"), frames, inputSize);
        var projExpected = ReadFlat(data, projection.GetProperty("output"));
        var projWeights = ReadSanmWeights(data, projection.GetProperty("weights"));

        var projOutput = OpenTail.Stingray.Audio.FunASR.FunAsrNanoSanmBlock.ProjectionBlock(
            projInput, projWeights, inputSize, modelSize, numHeads, ffnSize, kernelSize, eps);
        AssertClose("projection.output", Flatten(projOutput), projExpected);

        // --- Residual block (main/timestamp layer): model_size -> model_size, with input residual ---
        var residual = blocks.GetProperty("residual");
        var resInput = ReadRows(data, residual.GetProperty("input"), frames, modelSize);
        var resExpected = ReadFlat(data, residual.GetProperty("output"));
        var resWeights = ReadSanmWeights(data, residual.GetProperty("weights"));

        var resOutput = OpenTail.Stingray.Audio.FunASR.FunAsrNanoSanmBlock.ResidualBlock(
            resInput, resWeights, modelSize, numHeads, ffnSize, kernelSize, eps);
        AssertClose("residual.output", Flatten(resOutput), resExpected);
    }

    private static void AssertClose(string label, float[] actual, float[] expected)
    {
        Assert.Equal(expected.Length, actual.Length);
        double maxAbsDiff = 0;
        for (int i = 0; i < actual.Length; i++)
            maxAbsDiff = Math.Max(maxAbsDiff, Math.Abs(actual[i] - expected[i]));
        Console.Error.WriteLine($"[FunAsrNanoSanmGolden] {label} maxAbsDiff={maxAbsDiff:F6}");
        Assert.True(maxAbsDiff < 1e-3, $"{label} maxAbsDiff {maxAbsDiff} too high vs real PyTorch reference");
    }

    private static float[] Flatten(float[][] rows)
    {
        int dim = rows[0].Length;
        var flat = new float[rows.Length * dim];
        for (int i = 0; i < rows.Length; i++) Array.Copy(rows[i], 0, flat, i * dim, dim);
        return flat;
    }

    private static float[] ReadAllF32(string path)
    {
        var bytes = File.ReadAllBytes(path);
        var values = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
        return values;
    }

    private static float[] ReadFlat(float[] data, System.Text.Json.JsonElement entry)
    {
        int offset = entry.GetProperty("offset_f32").GetInt32();
        int count = entry.GetProperty("count").GetInt32();
        var result = new float[count];
        Array.Copy(data, offset, result, 0, count);
        return result;
    }

    private static float[][] ReadRows(float[] data, System.Text.Json.JsonElement entry, int frames, int dim)
    {
        var flat = ReadFlat(data, entry);
        var rows = new float[frames][];
        for (int i = 0; i < frames; i++)
        {
            var row = new float[dim];
            Array.Copy(flat, i * dim, row, 0, dim);
            rows[i] = row;
        }
        return rows;
    }

    private static float[] ReadWeight(float[] data, System.Text.Json.JsonElement weights, string name) => ReadFlat(data, weights.GetProperty(name));

    private static OpenTail.Stingray.Audio.FunASR.FunAsrNanoSanmWeights ReadSanmWeights(float[] data, System.Text.Json.JsonElement weights) => new()
    {
        SelfAttnLayerNormWeight = ReadWeight(data, weights, "self_attn_layer_norm.weight"),
        SelfAttnLayerNormBias = ReadWeight(data, weights, "self_attn_layer_norm.bias"),
        QWeight = ReadWeight(data, weights, "self_attn.q_proj.weight"),
        QBias = ReadWeight(data, weights, "self_attn.q_proj.bias"),
        KWeight = ReadWeight(data, weights, "self_attn.k_proj.weight"),
        KBias = ReadWeight(data, weights, "self_attn.k_proj.bias"),
        VWeight = ReadWeight(data, weights, "self_attn.v_proj.weight"),
        VBias = ReadWeight(data, weights, "self_attn.v_proj.bias"),
        OutWeight = ReadWeight(data, weights, "self_attn.out_proj.weight"),
        OutBias = ReadWeight(data, weights, "self_attn.out_proj.bias"),
        FsmnConvWeight = ReadWeight(data, weights, "feedforward_sequential_memory.conv.weight"),
        FinalLayerNormWeight = ReadWeight(data, weights, "final_layer_norm.weight"),
        FinalLayerNormBias = ReadWeight(data, weights, "final_layer_norm.bias"),
        Fc1Weight = ReadWeight(data, weights, "fc1.weight"),
        Fc1Bias = ReadWeight(data, weights, "fc1.bias"),
        Fc2Weight = ReadWeight(data, weights, "fc2.weight"),
        Fc2Bias = ReadWeight(data, weights, "fc2.bias"),
    };
}
