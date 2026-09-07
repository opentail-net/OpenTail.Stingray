namespace OpenTail.Stingray.Audio.VoxCpm2;

/// <summary>
/// Presents VoxCPM2's MiniCPM LLM backbone (`base_lm.*` tensors, real prefix confirmed via
/// `assets.cpp`'s `validate_weight_anchors`) to `OpenTail.Stingray.Engine`'s existing,
/// unmodified `ForwardPass` as a standard `minicpm` model -- same bridging technique as
/// `VibeVoice.VibeVoiceLlmTensorSource`/`OmniVoice.OmniVoiceLlmTensorSource`. Real, resolved
/// architectural finding this session (see `docs/audio-review-progress.md`'s VoxCPM2 section):
/// `"minicpm"` is an ADMITTED architecture in `ModelCompatibility.cs` (reuses Granite's graph --
/// the real OpenBMB "mup" `scale_emb`/`scale_depth`/`dim_model_base` convention maps to GGUF's
/// `minicpm.embedding_scale`/`minicpm.residual_scale`/`minicpm.logit_scale` metadata keys), AND
/// `IForwardPass.ForwardEmbedding`/`LastHidden` together give exactly the embeddings-in/
/// hidden-states-out capability `base_lm`'s per-step DiT/FSQ/fusion-projection generation loop
/// needs -- no bespoke Transformer port required for this piece, unlike this session's other
/// from-scratch GPT2/ConvNeXt-family ports.
///
/// <para><b>Real, non-obvious detail</b>: standard llama-family Q/K/V/O + gate/up/down MLP naming
/// (same as <see cref="VibeVoice.VibeVoiceLlmTensorSource"/>'s Qwen2 mapping), NO QKV bias (unlike
/// VibeVoice's Qwen2 decoder) -- confirmed via `text_decoder.cpp`'s sibling `minicpm.cpp`
/// (`load_layer_weights`, no `.bias` tensors requested for `q_proj`/`k_proj`/`v_proj`).</para>
///
/// <para><b>Not yet wired</b>: the real `embedding_scale`/`residual_scale`/`logit_scale` values
/// (OpenBMB `scale_emb`/`scale_depth`/`dim_model_base` -&gt; GGUF metadata formula) are passed
/// through as caller-supplied floats rather than derived here -- deriving the exact real
/// conversion formula from `lm_config`'s raw fields needs the checkpoint's actual `config.json`
/// numbers, a real next step once resumed.</para>
/// </summary>
public sealed unsafe class VoxCpm2LlmTensorSource : IModelTensorSource, IDisposable
{
    private readonly Rvc.RvcPackedTensorSource _source;
    private readonly Dictionary<string, GgufTensorInfo> _byName = new(StringComparer.Ordinal);
    private readonly Dictionary<string, string> _sourceNameByCanonical = new(StringComparer.Ordinal);
    private readonly List<nint> _ownedPointers = [];
    private readonly Dictionary<string, nint> _resolvedPointers = new(StringComparer.Ordinal);
    private readonly List<GgufTensorInfo> _tensors;
    private readonly Dictionary<string, object> _metadata;
    private bool _disposed;

    public VoxCpm2LlmTensorSource(
        Rvc.RvcPackedTensorSource source,
        int numLayers, int hiddenDim, int numHeads, int numKvHeads, int headDim,
        int ffDim, int vocabSize, float ropeTheta, float rmsNormEps,
        float embeddingScale, float residualScale, float logitScale)
    {
        _source = source;

        MapIfPresent2D("base_lm.embed_tokens.weight", "token_embd.weight", vocabSize, hiddenDim);
        MapIfPresent1D("base_lm.norm.weight", "output_norm.weight", hiddenDim);

        bool hasSeparateLmHead = _source.HasTensor("base_lm.lm_head.weight");
        if (hasSeparateLmHead)
            MapIfPresent2D("base_lm.lm_head.weight", "output.weight", vocabSize, hiddenDim);
        // else: tied embeddings, "output.weight" deliberately left unmapped.

        int qOut = numHeads * headDim;
        int kvOut = numKvHeads * headDim;
        for (int i = 0; i < numLayers; i++)
        {
            string p = $"base_lm.layers.{i}.";
            string b = $"blk.{i}.";
            MapIfPresent1D(p + "input_layernorm.weight", b + "attn_norm.weight", hiddenDim);
            MapIfPresent2D(p + "self_attn.q_proj.weight", b + "attn_q.weight", qOut, hiddenDim);
            MapIfPresent2D(p + "self_attn.k_proj.weight", b + "attn_k.weight", kvOut, hiddenDim);
            MapIfPresent2D(p + "self_attn.v_proj.weight", b + "attn_v.weight", kvOut, hiddenDim);
            MapIfPresent2D(p + "self_attn.o_proj.weight", b + "attn_output.weight", hiddenDim, qOut);
            MapIfPresent1D(p + "post_attention_layernorm.weight", b + "ffn_norm.weight", hiddenDim);
            MapIfPresent2D(p + "mlp.gate_proj.weight", b + "ffn_gate.weight", ffDim, hiddenDim);
            MapIfPresent2D(p + "mlp.up_proj.weight", b + "ffn_up.weight", ffDim, hiddenDim);
            MapIfPresent2D(p + "mlp.down_proj.weight", b + "ffn_down.weight", hiddenDim, ffDim);
        }

        _tensors = [.. _byName.Values];

        _metadata = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            ["general.architecture"] = "minicpm",
            ["minicpm.embedding_length"] = hiddenDim,
            ["minicpm.block_count"] = numLayers,
            ["minicpm.attention.head_count"] = numHeads,
            ["minicpm.attention.head_count_kv"] = numKvHeads,
            ["minicpm.feed_forward_length"] = ffDim,
            ["minicpm.attention.layer_norm_rms_epsilon"] = rmsNormEps,
            ["minicpm.rope.freq_base"] = ropeTheta,
            ["minicpm.vocab_size"] = vocabSize,
            ["minicpm.context_length"] = 32768,
            ["minicpm.embedding_scale"] = embeddingScale,
            ["minicpm.residual_scale"] = residualScale,
            ["minicpm.logit_scale"] = logitScale,
        };
    }

    private void MapIfPresent2D(string sourceName, string canonicalName, int outDim, int inDim)
    {
        if (!_source.HasTensor(sourceName)) return;
        _byName[canonicalName] = new GgufTensorInfo(canonicalName, 2, [inDim, outDim], DType.Float32, DataOffset: 0);
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
        return new ReadOnlySpan<byte>(pointer, checked((int)(tensor.ElementCount * sizeof(float))));
    }

    public byte* GetTensorDataPtr(GgufTensorInfo tensor)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_resolvedPointers.TryGetValue(tensor.Name, out nint cached))
            return (byte*)cached;

        string sourceName = _sourceNameByCanonical[tensor.Name];
        var data = _source.GetTensor(sourceName);
        float* buffer = (float*)NativeMemory.Alloc((nuint)(data.Length * sizeof(float)));
        fixed (float* src = data)
            Buffer.MemoryCopy(src, buffer, data.Length * sizeof(float), data.Length * sizeof(float));
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
