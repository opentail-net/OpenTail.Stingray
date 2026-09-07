using OpenTail.Stingray.Audio.Rvc;

namespace OpenTail.Stingray.Audio.HiggsAudio;

/// <summary>
/// Presents Higgs Audio TTS's real text/AR backbone (`body.*`/`tied.embedding.*` tensors) to
/// `OpenTail.Stingray.Engine`'s existing, unmodified `ForwardPass` as a standard `qwen3` model --
/// same bridging technique as `OmniVoice.OmniVoiceLlmTensorSource`/
/// `VibeVoice.VibeVoiceLlmTensorSource`. Confirmed real from `ar.cpp`'s `load_layer_weights`/
/// `load_higgs_ar_weights` (not guessed): a genuine Qwen3-family GQA decoder -- fused-bias-free
/// `q_proj`/`k_proj`/`v_proj` with real per-head `q_norm`/`k_norm` RMSNorm (Qwen3's real
/// convention, unlike VibeVoice's Qwen2 decoder which has QKV bias and no q/k norm), standard
/// RoPE, SwiGLU MLP. Real prefix `body.layers.{i}.*`, final norm `body.norm.weight`, text
/// embedding `tied.embedding.text_embedding.weight` (tied to the text LM head -- no separate
/// text `lm_head` tensor in the reference, matching this class's generic tied-embedding
/// fallback).
///
/// <para><b>Real, genuinely different multimodal mechanism from every other bridge class this
/// session</b>: Higgs's discrete audio-codebook tokens are embedded via a SEPARATE, SHARED
/// `tied.embedding.modality_embeddings.0.embedding.weight` table (`[numCodebooks*audioVocabSize,
/// hiddenSize]`, each codebook occupying its own disjoint contiguous id range) that is ALSO
/// reused directly as the audio-codebook LM head (real weight tying, confirmed via
/// `build_modality_logits`'s reuse of the same tensor). This class exposes that table via
/// <see cref="ModalityEmbeddingWeight"/>/<see cref="ModalityEmbeddingVocabSize"/> for a caller to
/// build the real per-frame multi-codebook sum-embedding and gated text/audio mix
/// (`build_higgs_prefill_input_embedding`'s real `text*textGate + code*codeGate` formula, driven
/// by a separate real gate predictor NOT yet ported this session) -- unlike OmniVoice/MOSS/
/// VibeVoice's "splice as extra vocab rows on `token_embd.weight`" trick, this checkpoint's own
/// real architecture already keeps text and audio embeddings in separate tables summed at
/// runtime, so no vocab-extension workaround is needed here.</para>
/// </summary>
public sealed unsafe class HiggsLlmTensorSource : IModelTensorSource, IDisposable
{
    private readonly RvcPackedTensorSource _source;
    private readonly Dictionary<string, GgufTensorInfo> _byName = new(StringComparer.Ordinal);
    private readonly Dictionary<string, string> _sourceNameByCanonical = new(StringComparer.Ordinal);
    private readonly List<nint> _ownedPointers = [];
    private readonly Dictionary<string, nint> _resolvedPointers = new(StringComparer.Ordinal);
    private readonly List<GgufTensorInfo> _tensors;
    private readonly Dictionary<string, object> _metadata;
    private bool _disposed;

    public int ModalityEmbeddingVocabSize { get; }
    public int HiddenDim { get; }

    public HiggsLlmTensorSource(
        RvcPackedTensorSource source,
        int numLayers, int hiddenDim, int numHeads, int numKvHeads, int headDim,
        int ffDim, int vocabSize, float ropeTheta, float rmsNormEps,
        int numCodebooks, int audioVocabSize)
    {
        _source = source;
        HiddenDim = hiddenDim;
        ModalityEmbeddingVocabSize = numCodebooks * audioVocabSize;

        MapIfPresent2D("tied.embedding.text_embedding.weight", "token_embd.weight", vocabSize, hiddenDim);
        MapIfPresent1D("body.norm.weight", "output_norm.weight", hiddenDim);
        // No separate lm_head tensor -- tied embeddings, "output.weight" deliberately left unmapped.

        int qOut = numHeads * headDim;
        int kvOut = numKvHeads * headDim;
        for (int i = 0; i < numLayers; i++)
        {
            string p = $"body.layers.{i}.";
            string b = $"blk.{i}.";
            MapIfPresent1D(p + "input_layernorm.weight", b + "attn_norm.weight", hiddenDim);
            MapIfPresent2D(p + "self_attn.q_proj.weight", b + "attn_q.weight", qOut, hiddenDim);
            MapIfPresent2D(p + "self_attn.k_proj.weight", b + "attn_k.weight", kvOut, hiddenDim);
            MapIfPresent2D(p + "self_attn.v_proj.weight", b + "attn_v.weight", kvOut, hiddenDim);
            MapIfPresent2D(p + "self_attn.o_proj.weight", b + "attn_output.weight", hiddenDim, qOut);
            MapIfPresent1D(p + "self_attn.q_norm.weight", b + "attn_q_norm.weight", headDim);
            MapIfPresent1D(p + "self_attn.k_norm.weight", b + "attn_k_norm.weight", headDim);
            MapIfPresent1D(p + "post_attention_layernorm.weight", b + "ffn_norm.weight", hiddenDim);
            MapIfPresent2D(p + "mlp.gate_proj.weight", b + "ffn_gate.weight", ffDim, hiddenDim);
            MapIfPresent2D(p + "mlp.up_proj.weight", b + "ffn_up.weight", ffDim, hiddenDim);
            MapIfPresent2D(p + "mlp.down_proj.weight", b + "ffn_down.weight", hiddenDim, ffDim);
        }

        _tensors = [.. _byName.Values];

        _metadata = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            ["general.architecture"] = "qwen3",
            ["qwen3.embedding_length"] = hiddenDim,
            ["qwen3.block_count"] = numLayers,
            ["qwen3.attention.head_count"] = numHeads,
            ["qwen3.attention.head_count_kv"] = numKvHeads,
            ["qwen3.attention.key_length"] = headDim,
            ["qwen3.attention.value_length"] = headDim,
            ["qwen3.feed_forward_length"] = ffDim,
            ["qwen3.attention.layer_norm_rms_epsilon"] = rmsNormEps,
            ["qwen3.rope.freq_base"] = ropeTheta,
            ["qwen3.vocab_size"] = vocabSize,
            ["qwen3.context_length"] = 32768,
        };
    }

    /// <summary>Real modality (audio-codebook) embedding/LM-head table, `[numCodebooks*
    /// audioVocabSize, hiddenSize]`, row-major (row = combined codebook-offset id). Materialized
    /// on first access.</summary>
    public float[] ModalityEmbeddingWeight => _source.GetTensor("tied.embedding.modality_embeddings.0.embedding.weight");

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
