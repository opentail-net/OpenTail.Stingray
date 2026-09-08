using OpenTail.Stingray.Audio.Rvc;

namespace OpenTail.Stingray.Audio.NeuTts;

/// <summary>
/// Presents NeuTTS's real Qwen3-style AR speech-token backbone (`backbone/model.*` tensors inside
/// the packed `neutts-2e-orig.gguf`) to `OpenTail.Stingray.Engine`'s existing, unmodified
/// `ForwardPass` as a standard `qwen3` model -- same bridging technique as
/// `HiggsAudio.HiggsLlmTensorSource`/`VibeVoice.VibeVoiceLlmTensorSource`. Confirmed real from
/// `neutts/backbone.cpp`'s `load_layer_weights`/`make_neutts_qwen_config` and the checkpoint's own
/// embedded `config.json` (not guessed): a genuine Qwen3-family GQA decoder -- bias-free
/// `q_proj`/`k_proj`/`v_proj` with real per-head `q_norm`/`k_norm` RMSNorm, NEOX RoPE, SwiGLU MLP,
/// real TIED embeddings (`tie_word_embeddings: true`, `lm_head` = `token_embedding` in the
/// reference -- no separate `lm_head` tensor exists, matching this class's generic tied-embedding
/// fallback). Real config: `hidden_size=512, intermediate_size=1536, num_hidden_layers=28,
/// num_attention_heads=12, num_key_value_heads=4, head_dim=128, vocab_size=217232,
/// rope_theta=10000, rms_norm_eps=1e-6`.
/// </summary>
public sealed unsafe class NeuTtsBackboneTensorSource : IModelTensorSource, IDisposable
{
    public const int HiddenSize = 512;
    public const int IntermediateSize = 1536;
    public const int NumLayers = 28;
    public const int NumHeads = 12;
    public const int NumKvHeads = 4;
    public const int HeadDim = 128;
    public const int VocabSize = 217232;
    public const float RopeTheta = 10000.0f;
    public const float RmsNormEps = 1e-6f;

    private readonly RvcPackedTensorSource _source;
    private readonly Dictionary<string, GgufTensorInfo> _byName = new(StringComparer.Ordinal);
    private readonly Dictionary<string, string> _sourceNameByCanonical = new(StringComparer.Ordinal);
    private readonly List<nint> _ownedPointers = [];
    private readonly Dictionary<string, nint> _resolvedPointers = new(StringComparer.Ordinal);
    private readonly List<GgufTensorInfo> _tensors;
    private readonly Dictionary<string, object> _metadata;
    private bool _disposed;

    public NeuTtsBackboneTensorSource(RvcPackedTensorSource source)
    {
        _source = source;

        MapIfPresent2D("backbone/model.embed_tokens.weight", "token_embd.weight", VocabSize, HiddenSize);
        MapIfPresent1D("backbone/model.norm.weight", "output_norm.weight", HiddenSize);
        // No separate lm_head tensor -- tied embeddings, "output.weight" deliberately left unmapped.

        int qOut = NumHeads * HeadDim;
        int kvOut = NumKvHeads * HeadDim;
        for (int i = 0; i < NumLayers; i++)
        {
            string p = $"backbone/model.layers.{i}.";
            string b = $"blk.{i}.";
            MapIfPresent1D(p + "input_layernorm.weight", b + "attn_norm.weight", HiddenSize);
            MapIfPresent2D(p + "self_attn.q_proj.weight", b + "attn_q.weight", qOut, HiddenSize);
            MapIfPresent2D(p + "self_attn.k_proj.weight", b + "attn_k.weight", kvOut, HiddenSize);
            MapIfPresent2D(p + "self_attn.v_proj.weight", b + "attn_v.weight", kvOut, HiddenSize);
            MapIfPresent2D(p + "self_attn.o_proj.weight", b + "attn_output.weight", HiddenSize, qOut);
            MapIfPresent1D(p + "self_attn.q_norm.weight", b + "attn_q_norm.weight", HeadDim);
            MapIfPresent1D(p + "self_attn.k_norm.weight", b + "attn_k_norm.weight", HeadDim);
            MapIfPresent1D(p + "post_attention_layernorm.weight", b + "ffn_norm.weight", HiddenSize);
            MapIfPresent2D(p + "mlp.gate_proj.weight", b + "ffn_gate.weight", IntermediateSize, HiddenSize);
            MapIfPresent2D(p + "mlp.up_proj.weight", b + "ffn_up.weight", IntermediateSize, HiddenSize);
            MapIfPresent2D(p + "mlp.down_proj.weight", b + "ffn_down.weight", HiddenSize, IntermediateSize);
        }

        _tensors = [.. _byName.Values];

        _metadata = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            ["general.architecture"] = "qwen3",
            ["qwen3.embedding_length"] = HiddenSize,
            ["qwen3.block_count"] = NumLayers,
            ["qwen3.attention.head_count"] = NumHeads,
            ["qwen3.attention.head_count_kv"] = NumKvHeads,
            ["qwen3.attention.key_length"] = HeadDim,
            ["qwen3.attention.value_length"] = HeadDim,
            ["qwen3.feed_forward_length"] = IntermediateSize,
            ["qwen3.attention.layer_norm_rms_epsilon"] = RmsNormEps,
            ["qwen3.rope.freq_base"] = RopeTheta,
            ["qwen3.vocab_size"] = VocabSize,
            ["qwen3.context_length"] = 32768,
        };
    }

    private void MapIfPresent2D(string sourceName, string canonicalName, int outDim, int inDim)
    {
        if (!_source.HasTensor(sourceName)) return;
        var dtype = _source.GetRawInfo(sourceName).DType;
        _byName[canonicalName] = new GgufTensorInfo(canonicalName, 2, [inDim, outDim], dtype, DataOffset: 0);
        _sourceNameByCanonical[canonicalName] = sourceName;
    }

    private void MapIfPresent1D(string sourceName, string canonicalName, int dim)
    {
        if (!_source.HasTensor(sourceName)) return;
        _byName[canonicalName] = new GgufTensorInfo(canonicalName, 1, [dim], DType.Float32, DataOffset: 0);
        _sourceNameByCanonical[canonicalName] = sourceName;
    }

    public IReadOnlyList<GgufTensorInfo> Tensors => _tensors;
    public IReadOnlyDictionary<string, object> Metadata => _metadata;

    public GgufTensorInfo? FindTensor(string name) => _byName.TryGetValue(name, out var info) ? info : null;

    public ReadOnlySpan<byte> GetTensorData(GgufTensorInfo tensor)
    {
        byte* pointer = GetTensorDataPtr(tensor);
        long byteSize = tensor.DType == DType.Float32
            ? tensor.ElementCount * sizeof(float)
            : DTypeInfo.ByteSize(tensor.ElementCount, tensor.DType);
        return new ReadOnlySpan<byte>(pointer, checked((int)byteSize));
    }

    // Real DType passthrough (same as VibeVoiceLlmTensorSource): per-layer weight matrices keep
    // their real on-disk DType so ForwardPass computes on them natively-quantized instead of this
    // class eagerly materializing a larger FP32 copy of every weight.
    public byte* GetTensorDataPtr(GgufTensorInfo tensor)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_resolvedPointers.TryGetValue(tensor.Name, out nint cached))
            return (byte*)cached;

        string sourceName = _sourceNameByCanonical[tensor.Name];
        if (tensor.DType != DType.Float32)
            return _source.GetRawDataPtr(sourceName);

        var data = _source.GetTensor(sourceName);
        long byteCount = (long)data.Length * sizeof(float);
        float* buffer = (float*)NativeMemory.Alloc((nuint)byteCount);
        fixed (float* src = data)
            Buffer.MemoryCopy(src, buffer, byteCount, byteCount);
        _ownedPointers.Add((nint)buffer);
        _resolvedPointers[tensor.Name] = (nint)buffer;
        return (byte*)buffer;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (var p in _ownedPointers) NativeMemory.Free((void*)p);
        _ownedPointers.Clear();
    }
}
