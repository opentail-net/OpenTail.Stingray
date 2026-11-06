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

    public MlxGgufDenseModelLoaderTests() => Environment.SetEnvironmentVariable(FlagVar, "1");

    public void Dispose()
    {
        Environment.SetEnvironmentVariable(FlagVar, null);
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

    // hiddenDim=4, numHeads=2, headDim=2, intermDim=8, 1 layer, vocab=3. Every weight tensor is F32,
    // filled with its own recognizable constant so a wrong tensor landing in the wrong field would
    // produce an obviously wrong value rather than a coincidentally-plausible one.
    private void WriteSyntheticCheckpoint()
    {
        const int hiddenDim = 4, numHeads = 2, headDim = 2, intermDim = 8, vocab = 3;
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
            ("blk.0.attn_k.weight", [hiddenDim, numHeads * headDim], 0, hiddenDim * numHeads * headDim),
            ("blk.0.attn_v.weight", [hiddenDim, numHeads * headDim], 0, hiddenDim * numHeads * headDim),
            ("blk.0.attn_output.weight", [numHeads * headDim, hiddenDim], 0, numHeads * headDim * hiddenDim),
            ("blk.0.ffn_norm.weight", [hiddenDim], 0, hiddenDim),
            ("blk.0.ffn_gate.weight", [hiddenDim, intermDim], 0, hiddenDim * intermDim),
            ("blk.0.ffn_up.weight", [hiddenDim, intermDim], 0, hiddenDim * intermDim),
            ("blk.0.ffn_down.weight", [intermDim, hiddenDim], 0, intermDim * hiddenDim),
            ("token_embd.weight", [hiddenDim, vocab], 0, hiddenDim * vocab),
            ("output_norm.weight", [hiddenDim], 0, hiddenDim),
        };
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
        w.Write((ulong)7);              // kv count

        KvStr("general.architecture", "llama");
        KvU32("llama.embedding_length", hiddenDim);
        KvU32("llama.attention.head_count", numHeads);
        KvU32("llama.attention.head_count_kv", numHeads);
        KvU32("llama.block_count", 1);
        KvU32("llama.feed_forward_length", intermDim);
        KvF32("llama.attention.layer_norm_rms_epsilon", 1e-5f);

        foreach (var t in tensors)
        {
            Str(t.Name);
            w.Write((uint)t.Dims.Length);
            foreach (var d in t.Dims) w.Write((ulong)d);
            w.Write(0u); // DType.Float32
            w.Write(t.Offset);
        }
        while (ms.Length % 32 != 0) w.Write((byte)0);

        // Tensor data: each tensor filled with a distinct constant (its index + 1).
        for (int i = 0; i < tensors.Count; i++)
            for (int j = 0; j < tensors[i].FloatCount; j++)
                w.Write((float)(i + 1));

        File.WriteAllBytes(_path, ms.ToArray());
    }

    [Fact]
    public void Load_RealSyntheticCheckpoint_ProducesCorrectlyShapedModel()
    {
        WriteSyntheticCheckpoint();
        using var gguf = GgufModel.Open(_path);

        var model = MlxGgufDenseModelLoader.Load(gguf);

        Assert.Equal(4, model.HiddenDim);
        Assert.Equal(3, model.VocabSize);
        Assert.Single(model.Layers);
        var layer = model.Layers[0];
        Assert.Equal(2, layer.NumHeads);
        Assert.Equal(2, layer.NumKvHeads);
        Assert.Equal(2, layer.HeadDim);
        Assert.Equal(8, layer.IntermDim);

        // Each weight was filled with its own distinct constant (tensor index + 1 in the list
        // above) -- confirms the loader wired each GGUF tensor into the right MlxDenseLayer field,
        // not a transposed or swapped one.
        Assert.All(layer.AttnNormWeight, v => Assert.Equal(1f, v));
        Assert.All(layer.Wq, v => Assert.Equal(2f, v));
        Assert.All(layer.Wk, v => Assert.Equal(3f, v));
        Assert.All(layer.Wv, v => Assert.Equal(4f, v));
        Assert.All(layer.Wo, v => Assert.Equal(5f, v));
        Assert.All(layer.FfnNormWeight, v => Assert.Equal(6f, v));
        Assert.All(layer.WGate, v => Assert.Equal(7f, v));
        Assert.All(layer.WUp, v => Assert.Equal(8f, v));
        Assert.All(layer.WDown, v => Assert.Equal(9f, v));
        Assert.All(model.TokenEmbedding, v => Assert.Equal(10f, v));
        Assert.All(model.OutputNormWeight, v => Assert.Equal(11f, v));
        // Tied embeddings: no separate output.weight tensor in this checkpoint.
        Assert.Same(model.TokenEmbedding, model.OutputWeight);
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
}
