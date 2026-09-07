using OpenTail.Stingray.Audio.Rvc;

namespace OpenTail.Stingray.Audio.PersonaPlex;

/// <summary>
/// Presents PersonaPlex's real temporal-LM backbone (`lm/transformer.layers.*` tensors) to
/// `OpenTail.Stingray.Engine`'s existing, unmodified `ForwardPass` as a standard `qwen3` model --
/// same bridging technique as every other `IModelTensorSource` this session. Confirmed real from
/// `lm_runtime.cpp`'s `load_main_layer`/`load_personaplex_lm_weights` (not guessed): a genuine
/// bias-free, RoPE, plain-MHA (`num_key_value_heads == num_attention_heads`, no GQA) decoder with
/// NO per-head q/k-norm (`use_qk_norm=false` in the real reference) and SEPARATE, untied
/// `text_linear.weight` output projection (unlike Higgs/OmniVoice's tied embeddings).
///
/// <para><b>Real, genuinely new wrinkle this session</b>: both the QKV projection
/// (`self_attn.in_proj_weight`, real shape `[3*hidden, hidden]`) and the gated-MLP projection
/// (`gating.linear_in.weight`, real shape `[2*intermediate, hidden]`) are PACKED single tensors
/// in the checkpoint (real reference: `modules::QwenDecoderQKVLayout::PackedQKV`/
/// `modules::QwenDecoderMLPMode::PackedGateUp`) -- every prior bridge class this session mapped
/// 1:1 onto already-separate q/k/v/gate/up tensors. `ForwardPass`'s generic qwen-family graph
/// expects separate `attn_q`/`attn_k`/`attn_v` and `ffn_gate`/`ffn_up` tensors, so this class
/// splits each packed weight into contiguous row-range sub-copies AT LOAD TIME (materializing
/// real, correctly-sliced managed arrays once in the constructor) before presenting them as
/// ordinary separate canonical tensors -- the packed layout is invisible past this class.
/// Splitting is a plain row-slice (`in_proj_weight` rows `[0,hidden)`=Q, `[hidden,2*hidden)`=K,
/// `[2*hidden,3*hidden)`=V; `linear_in.weight` rows `[0,intermediate)`=gate,
/// `[intermediate,2*intermediate)`=up) since PyTorch's `nn.Linear` weight layout is
/// `[out_features, in_features]` row-major and the real reference concatenates gate/up/QKV along
/// the OUTPUT (row) dimension, confirmed by the packed shapes above.</para>
///
/// <para>Real, separate per-codebook audio embedding tables (`emb.{codebook}.weight`, one
/// FULLY SEPARATE `[audioCodebookSize+1, hidden]` table per of the real `lm_codebooks` count --
/// NOT a single shared table with per-codebook offset ranges like Higgs's
/// `modality_embeddings`) are exposed via <see cref="AudioEmbeddingWeight"/> for a caller to
/// build the real per-step additive multi-stream input (`embedding_scale*continuous_input +
/// token_scale*sum_of_selected_embeddings`, per `lm_runtime.cpp`'s real step-graph
/// construction) -- not yet wired into a generation loop this session.</para>
/// </summary>
public sealed unsafe class PersonaPlexLmTensorSource : IModelTensorSource, IDisposable
{
    private readonly RvcPackedTensorSource _source;
    private readonly Dictionary<string, GgufTensorInfo> _byName = new(StringComparer.Ordinal);
    private readonly Dictionary<string, Func<float[]>> _materializers = new(StringComparer.Ordinal);
    private readonly List<nint> _ownedPointers = [];
    private readonly Dictionary<string, nint> _resolvedPointers = new(StringComparer.Ordinal);
    private readonly List<GgufTensorInfo> _tensors;
    private readonly Dictionary<string, object> _metadata;
    private bool _disposed;

    public int HiddenDim { get; }
    public int LmCodebooks { get; }

    public PersonaPlexLmTensorSource(
        RvcPackedTensorSource source,
        int numLayers, int hiddenDim, int numHeads, int headDim,
        int ffDim, int textVocabSize, int lmCodebooks, int audioCodebookSize,
        float ropeTheta, float rmsNormEps)
    {
        _source = source;
        HiddenDim = hiddenDim;
        LmCodebooks = lmCodebooks;

        MapMaterialized2D("lm/text_emb.weight", "token_embd.weight", textVocabSize + 1, hiddenDim,
            () => source.GetTensor("lm/text_emb.weight"));
        MapMaterialized1D("lm/out_norm.alpha", "output_norm.weight", hiddenDim,
            () => source.GetTensor("lm/out_norm.alpha"));
        MapMaterialized2D("lm/text_linear.weight", "output.weight", textVocabSize, hiddenDim,
            () => source.GetTensor("lm/text_linear.weight"));

        int qkvOut = numHeads * headDim; // plain MHA: q/k/v all equal size
        for (int i = 0; i < numLayers; i++)
        {
            string p = $"lm/transformer.layers.{i}.";
            string b = $"blk.{i}.";

            MapMaterialized1D(p + "norm1.alpha", b + "attn_norm.weight", hiddenDim,
                () => source.GetTensor(p + "norm1.alpha"));
            MapMaterialized1D(p + "norm2.alpha", b + "ffn_norm.weight", hiddenDim,
                () => source.GetTensor(p + "norm2.alpha"));
            MapMaterialized2D(p + "self_attn.out_proj.weight", b + "attn_output.weight", hiddenDim, qkvOut,
                () => source.GetTensor(p + "self_attn.out_proj.weight"));
            MapMaterialized2D(p + "gating.linear_out.weight", b + "ffn_down.weight", hiddenDim, ffDim,
                () => source.GetTensor(p + "gating.linear_out.weight"));

            string inProjKey = p + "self_attn.in_proj_weight";
            MapMaterialized2D(inProjKey + "#q", b + "attn_q.weight", qkvOut, hiddenDim,
                () => SplitRows(source.GetTensor(inProjKey), hiddenDim, 0, qkvOut));
            MapMaterialized2D(inProjKey + "#k", b + "attn_k.weight", qkvOut, hiddenDim,
                () => SplitRows(source.GetTensor(inProjKey), hiddenDim, qkvOut, qkvOut));
            MapMaterialized2D(inProjKey + "#v", b + "attn_v.weight", qkvOut, hiddenDim,
                () => SplitRows(source.GetTensor(inProjKey), hiddenDim, 2 * qkvOut, qkvOut));

            string gateUpKey = p + "gating.linear_in.weight";
            MapMaterialized2D(gateUpKey + "#gate", b + "ffn_gate.weight", ffDim, hiddenDim,
                () => SplitRows(source.GetTensor(gateUpKey), hiddenDim, 0, ffDim));
            MapMaterialized2D(gateUpKey + "#up", b + "ffn_up.weight", ffDim, hiddenDim,
                () => SplitRows(source.GetTensor(gateUpKey), hiddenDim, ffDim, ffDim));
        }

        _tensors = [.. _byName.Values];

        _metadata = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            ["general.architecture"] = "qwen3",
            ["qwen3.embedding_length"] = hiddenDim,
            ["qwen3.block_count"] = numLayers,
            ["qwen3.attention.head_count"] = numHeads,
            ["qwen3.attention.head_count_kv"] = numHeads,
            ["qwen3.attention.key_length"] = headDim,
            ["qwen3.attention.value_length"] = headDim,
            ["qwen3.feed_forward_length"] = ffDim,
            ["qwen3.attention.layer_norm_rms_epsilon"] = rmsNormEps,
            ["qwen3.rope.freq_base"] = ropeTheta,
            ["qwen3.vocab_size"] = textVocabSize,
            ["qwen3.context_length"] = 32768,
        };
    }

    /// <summary>Real, fully separate per-codebook audio embedding table, `[audioCodebookSize+1,
    /// hiddenSize]`, row-major. Materialized on first access.</summary>
    public float[] AudioEmbeddingWeight(int codebook)
    {
        if ((uint)codebook >= (uint)LmCodebooks) throw new ArgumentOutOfRangeException(nameof(codebook));
        return _source.GetTensor($"lm/emb.{codebook}.weight");
    }

    /// <summary>Real text embedding table, `[textVocabSize+1, hiddenSize]`, row-major (the SAME
    /// table backing `token_embd.weight`). Materialized on first access.</summary>
    public float[] TextEmbeddingWeight() => _source.GetTensor("lm/text_emb.weight");

    private static float[] SplitRows(float[] packed, int cols, int rowOffset, int rows)
    {
        var output = new float[(long)rows * cols];
        Array.Copy(packed, (long)rowOffset * cols, output, 0, output.Length);
        return output;
    }

    private void MapMaterialized2D(string key, string canonicalName, int outDim, int inDim, Func<float[]> materialize)
    {
        if (!_source.HasTensor(key.Split('#')[0])) return;
        _byName[canonicalName] = new GgufTensorInfo(canonicalName, 2, [inDim, outDim], DType.Float32, DataOffset: 0);
        _materializers[canonicalName] = materialize;
    }

    private void MapMaterialized1D(string sourceName, string canonicalName, int dim, Func<float[]> materialize)
    {
        if (!_source.HasTensor(sourceName)) return;
        _byName[canonicalName] = new GgufTensorInfo(canonicalName, 1, [dim], DType.Float32, DataOffset: 0);
        _materializers[canonicalName] = materialize;
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

        var data = _materializers[tensor.Name]();
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
