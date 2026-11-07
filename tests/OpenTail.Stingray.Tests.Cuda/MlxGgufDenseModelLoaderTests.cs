using System.Text;
using OpenTail.Stingray.Mlx;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Tests.Cuda;

/// <summary>
/// <see cref="MlxGgufDenseModelLoader"/> reads real GGUF metadata and dequantizes real tensor bytes
/// via Stingray's own <see cref="OpenTail.Stingray.Cpu.Dequantize"/> -- both run fine on this
/// machine (no Mac needed; nothing here calls into mlx-c), so this is fully, really tested, not just
/// "fails predictably." A small standalone GGUF builder (not Tests.Core's internal
/// <c>SyntheticGguf</c>, which this project doesn't reference) produces a tiny one-layer F32 dense
/// checkpoint with the exact tensor names <c>MlxGgufDenseModelLoader</c> looks for.
/// </summary>
public sealed class MlxGgufDenseModelLoaderTests : IDisposable
{
    private const string FlagVar = "STINGRAY_MLX_DENSE";
    private readonly string _path = Path.Combine(Path.GetTempPath(), "mlx-gguf-loader-" + Guid.NewGuid().ToString("N") + ".gguf");
    private readonly string? _originalFlag = Environment.GetEnvironmentVariable(FlagVar);

    public MlxGgufDenseModelLoaderTests() => Environment.SetEnvironmentVariable(FlagVar, "1");

    public void Dispose()
    {
        Environment.SetEnvironmentVariable(FlagVar, _originalFlag);
        try { File.Delete(_path); } catch (IOException) { }
    }

    [Fact]
    public void Load_FlagNotSet_ThrowsNotSupported()
    {
        Environment.SetEnvironmentVariable(FlagVar, null);
        WriteSyntheticCheckpoint();
        using var gguf = GgufModel.Open(_path);
        Assert.Throws<NotSupportedException>(() => MlxGgufDenseModelLoader.Load(gguf));
    }

    // hiddenDim=6, numHeads=2, numKvHeads=1, headDim=3, intermDim=8, 1 layer, vocab=3.
    // Every tensor contains distinct values so matrix orientation errors are observable.
    private void WriteSyntheticCheckpoint(bool includeOutput = false, string architecture = "llama",
        int? ropeDimensionCount = null, string? ropeScalingType = null)
    {
        const int hiddenDim = 8, numHeads = 2, numKvHeads = 1, headDim = 4, intermDim = 10, vocab = 3;
        using var ms = new MemoryStream();
        using var w = new BinaryWriter(ms);

        void Str(string s) { var b = Encoding.UTF8.GetBytes(s); w.Write((ulong)b.Length); w.Write(b); }
        void KvStr(string k, string v) { Str(k); w.Write(8u); Str(v); }
        void KvU32(string k, uint v) { Str(k); w.Write(4u); w.Write(v); }
        void KvF32(string k, float v) { Str(k); w.Write(6u); w.Write(v); }

        var tensors = new List<(string Name, long[] Dims, ulong Offset, int FloatCount)>
        {
            ("blk.0.attn_norm.weight", [hiddenDim], 0, hiddenDim),
            ("blk.0.attn_q.weight", [hiddenDim, numHeads * headDim], 0, hiddenDim * numHeads * headDim),
            ("blk.0.attn_k.weight", [hiddenDim, numKvHeads * headDim], 0, hiddenDim * numKvHeads * headDim),
            ("blk.0.attn_v.weight", [hiddenDim, numKvHeads * headDim], 0, hiddenDim * numKvHeads * headDim),
            ("blk.0.attn_output.weight", [numHeads * headDim, hiddenDim], 0, numHeads * headDim * hiddenDim),
            ("blk.0.ffn_norm.weight", [hiddenDim], 0, hiddenDim),
            ("blk.0.ffn_gate.weight", [hiddenDim, intermDim], 0, hiddenDim * intermDim),
            ("blk.0.ffn_up.weight", [hiddenDim, intermDim], 0, hiddenDim * intermDim),
            ("blk.0.ffn_down.weight", [intermDim, hiddenDim], 0, intermDim * hiddenDim),
            ("token_embd.weight", [hiddenDim, vocab], 0, hiddenDim * vocab),
            ("output_norm.weight", [hiddenDim], 0, hiddenDim),
        };
        if (includeOutput)
            tensors.Add(("output.weight", [hiddenDim, vocab], 0, hiddenDim * vocab));
        ulong offset = 0;
        for (int i = 0; i < tensors.Count; i++)
        {
            var t = tensors[i];
            tensors[i] = (t.Name, t.Dims, offset, t.FloatCount);
            offset += (ulong)t.FloatCount * sizeof(float);
        }

        w.Write(0x46554747u);           // GGUF magic
        w.Write(3u);                    // version
        w.Write((ulong)tensors.Count);  // tensor count
        w.Write((ulong)(7 + (ropeDimensionCount.HasValue ? 1 : 0) + (ropeScalingType is not null ? 1 : 0)));

        KvStr("general.architecture", architecture);
        KvU32($"{architecture}.embedding_length", hiddenDim);
        KvU32($"{architecture}.attention.head_count", numHeads);
        KvU32($"{architecture}.attention.head_count_kv", numKvHeads);
        KvU32($"{architecture}.block_count", 1);
        KvU32($"{architecture}.feed_forward_length", intermDim);
        KvF32($"{architecture}.attention.layer_norm_rms_epsilon", 1e-5f);
        if (ropeDimensionCount is { } ropeDims)
            KvU32($"{architecture}.rope.dimension_count", (uint)ropeDims);
        if (ropeScalingType is not null)
            KvStr($"{architecture}.rope.scaling.type", ropeScalingType);

        foreach (var t in tensors)
        {
            Str(t.Name);
            w.Write((uint)t.Dims.Length);
            foreach (var d in t.Dims) w.Write((ulong)d);
            w.Write(0u); // DType.Float32
            w.Write(t.Offset);
        }
        while (ms.Length % 32 != 0) w.Write((byte)0);

        // Distinct row-major source values expose transposition and dimension errors.
        for (int i = 0; i < tensors.Count; i++)
            for (int j = 0; j < tensors[i].FloatCount; j++)
                w.Write((i + 1) * 100f + j + 1);

        File.WriteAllBytes(_path, ms.ToArray());
    }

    [Fact]
    public void Load_RealSyntheticCheckpoint_ProducesCorrectlyShapedModel()
    {
        WriteSyntheticCheckpoint();
        using var gguf = GgufModel.Open(_path);

        var model = MlxGgufDenseModelLoader.Load(gguf);

        Assert.Equal(8, model.HiddenDim);
        Assert.Equal(3, model.VocabSize);
        Assert.Single(model.Layers);
        var layer = model.Layers[0];
        Assert.Equal(2, layer.NumHeads);
        Assert.Equal(1, layer.NumKvHeads);
        Assert.Equal(4, layer.HeadDim);
        Assert.Equal(10, layer.IntermDim);

        Assert.Equal(ExpectedLinear(1, 8, 8), layer.Wq);
        Assert.Equal(ExpectedLinear(2, 8, 4), layer.Wk);
        Assert.Equal(ExpectedLinear(3, 8, 4), layer.Wv);
        Assert.Equal(ExpectedLinear(4, 8, 8), layer.Wo);
        Assert.Equal(ExpectedLinear(6, 8, 10), layer.WGate);
        Assert.Equal(ExpectedLinear(7, 8, 10), layer.WUp);
        Assert.Equal(ExpectedLinear(8, 10, 8), layer.WDown);
        Assert.Equal(Enumerable.Range(1001, 24).Select(v => (float)v), model.TokenEmbedding);
        Assert.Equal(Enumerable.Range(1101, 8).Select(v => (float)v), model.OutputNormWeight);

        float[] gateProduct = Project([1, 2, 3, 4, 5, 6, 7, 8], layer.WGate, inputDim: 8, outputDim: 10);
        Assert.Equal(new float[] { 25404, 25692, 25980, 26268, 26556, 26844, 27132, 27420, 27708, 27996 }, gateProduct);

        // With no output.weight tensor, the embedding's [vocab, hidden] data is separately
        // transposed for MatMul's [hidden, vocab] output projection.
        Assert.Equal(ExpectedLinear(9, 8, 3), model.OutputWeight);
        Assert.NotSame(model.TokenEmbedding, model.OutputWeight);
        Assert.Equal(new float[] { 36204, 36492, 36780 }, Project([1, 2, 3, 4, 5, 6, 7, 8], model.OutputWeight, 8, 3));
    }

    [Fact]
    public void Load_SeparateOutputWeight_UsesLinearLayout()
    {
        WriteSyntheticCheckpoint(includeOutput: true);
        using var gguf = GgufModel.Open(_path);

        var model = MlxGgufDenseModelLoader.Load(gguf);

        Assert.Equal(ExpectedLinear(11, 8, 3), model.OutputWeight);
        Assert.Equal(new float[] { 43404, 43692, 43980 }, Project([1, 2, 3, 4, 5, 6, 7, 8], model.OutputWeight, 8, 3));
    }

    private static float[] ExpectedLinear(int tensorIndex, int inputDim, int outputDim)
    {
        var result = new float[inputDim * outputDim];
        for (int output = 0; output < outputDim; output++)
            for (int input = 0; input < inputDim; input++)
                result[input * outputDim + output] = (tensorIndex + 1) * 100f + output * inputDim + input + 1;
        return result;
    }

    private static float[] Project(ReadOnlySpan<float> input, ReadOnlySpan<float> weight, int inputDim, int outputDim)
    {
        var result = new float[outputDim];
        for (int output = 0; output < outputDim; output++)
            for (int i = 0; i < inputDim; i++)
                result[output] += input[i] * weight[i * outputDim + output];
        return result;
    }

    [Fact]
    public void Load_MissingArchitecture_Throws()
    {
        using var ms = new MemoryStream();
        using var w = new BinaryWriter(ms);
        w.Write(0x46554747u); w.Write(3u); w.Write((ulong)0); w.Write((ulong)0);
        File.WriteAllBytes(_path, ms.ToArray());
        using var gguf = GgufModel.Open(_path);

        Assert.Throws<InvalidOperationException>(() => MlxGgufDenseModelLoader.Load(gguf));
    }

    [Fact]
    public void Load_UnverifiedArchitecture_RefusesBeforeTensorLoading()
    {
        WriteSyntheticCheckpoint(architecture: "mistral");
        using var gguf = GgufModel.Open(_path);

        var ex = Assert.Throws<NotSupportedException>(() => MlxGgufDenseModelLoader.Load(gguf));

        Assert.Contains("only the verified 'llama'", ex.Message);
    }

    [Theory]
    [InlineData(2, null, "full-head RoPE")]
    [InlineData(null, "linear", "RoPE setting")]
    public void Load_UnsupportedRopeMetadata_Refuses(int? ropeDimensionCount, string? scalingType, string expectedMessage)
    {
        WriteSyntheticCheckpoint(ropeDimensionCount: ropeDimensionCount, ropeScalingType: scalingType);
        using var gguf = GgufModel.Open(_path);

        var ex = Assert.Throws<NotSupportedException>(() => MlxGgufDenseModelLoader.Load(gguf));

        Assert.Contains(expectedMessage, ex.Message);
    }
}
