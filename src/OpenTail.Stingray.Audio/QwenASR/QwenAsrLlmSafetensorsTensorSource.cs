
namespace OpenTail.Stingray.Audio.QwenASR;

/// <summary>
/// Presents the LLM (`thinker.model.*`/`thinker.lm_head.*`) half of the real, canonical
/// Hugging Face `Qwen/Qwen3-ASR-0.6B` Safetensors checkpoint to `OpenTail.Stingray.Engine`'s
/// existing, unmodified `ForwardPass` as a standard `qwen3` model -- the Safetensors
/// counterpart of <see cref="QwenAsrLlmTensorSource"/> (GGUF-based), same architectural bet,
/// different source format.
///
/// <para>Real, confirmed (not guessed) facts about this checkpoint, verified directly against
/// the real downloaded `model.safetensors` tensor list before writing this loader: real wrapper
/// prefix `thinker.` (NOT a bare `model.*`/`lm_head.*` layout -- easy to miss), real BF16
/// storage throughout (`SafetensorsLoader.ReadF32` already converts this generically), decoder
/// attention projections (`q/k/v/o_proj`) are genuinely bias-free (unlike the audio tower's own
/// attention, which DOES have biases -- two different real conventions in the same checkpoint),
/// real per-head `q_norm`/`k_norm` (RMSNorm, `[128]` each, confirmed shape = head_dim), real GQA
/// (`q_proj` out=2048=16×128, `k/v_proj` out=1024=8×128), and both `thinker.model.embed_tokens.
/// weight` AND `thinker.lm_head.weight` are separately materialized in the file despite
/// `tie_word_embeddings=true` in `config.json` -- do not assume tying eliminates one physical
/// tensor.</para>
/// </summary>
public sealed unsafe class QwenAsrLlmSafetensorsTensorSource : IModelTensorSource, IDisposable, IQwenAsrAudioConditionableSource
{
    private readonly SafetensorsLoader _loader;
    private readonly Dictionary<string, GgufTensorInfo> _byName = new(StringComparer.Ordinal);
    private readonly Dictionary<string, string> _sourceNameByCanonical = new(StringComparer.Ordinal);
    private readonly List<nint> _ownedPointers = [];
    private readonly Dictionary<string, nint> _resolvedPointers = new(StringComparer.Ordinal);
    private readonly Dictionary<string, nint> _syntheticBuffers = new(StringComparer.Ordinal);
    private readonly List<GgufTensorInfo> _tensors;
    private readonly Dictionary<string, object> _metadata;
    private readonly int _hiddenDim;
    private bool _disposed;

    public QwenAsrLlmSafetensorsTensorSource(string safetensorsPath, int numLayers, int hiddenDim, int numHeads, int numKvHeads, int headDim, int ffDim, int vocabSize, float ropeTheta, float rmsNormEps)
    {
        _loader = SafetensorsLoader.Open(safetensorsPath);
        _hiddenDim = hiddenDim;

        MapIfPresent("thinker.model.embed_tokens.weight", "token_embd.weight");
        MapIfPresent("thinker.model.norm.weight", "output_norm.weight");
        MapIfPresent("thinker.lm_head.weight", "output.weight");

        for (int i = 0; i < numLayers; i++)
        {
            string p = $"thinker.model.layers.{i}.";
            string b = $"blk.{i}.";
            MapIfPresent(p + "input_layernorm.weight", b + "attn_norm.weight");
            MapIfPresent(p + "self_attn.q_proj.weight", b + "attn_q.weight");
            MapIfPresent(p + "self_attn.k_proj.weight", b + "attn_k.weight");
            MapIfPresent(p + "self_attn.v_proj.weight", b + "attn_v.weight");
            MapIfPresent(p + "self_attn.o_proj.weight", b + "attn_output.weight");
            MapIfPresent(p + "self_attn.q_norm.weight", b + "attn_q_norm.weight");
            MapIfPresent(p + "self_attn.k_norm.weight", b + "attn_k_norm.weight");
            MapIfPresent(p + "post_attention_layernorm.weight", b + "ffn_norm.weight");
            MapIfPresent(p + "mlp.gate_proj.weight", b + "ffn_gate.weight");
            MapIfPresent(p + "mlp.up_proj.weight", b + "ffn_up.weight");
            MapIfPresent(p + "mlp.down_proj.weight", b + "ffn_down.weight");
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

    private void MapIfPresent(string sourceName, string canonicalName)
    {
        if (!_loader.TensorNames.Contains(sourceName)) return;
        int[] shape = _loader.GetShape(sourceName);
        long[] dims = ToGgufDimensionOrder(shape);
        _byName[canonicalName] = new GgufTensorInfo(canonicalName, dims.Length, dims, DType.Float32, DataOffset: 0);
        _sourceNameByCanonical[canonicalName] = sourceName;
    }

    /// <summary>Reverses a row-major HF shape into GGUF's fastest-varying-first order (same convention as `SafetensorsTensorSource`/`CosyVoiceLlmTensorSource`).</summary>
    private static long[] ToGgufDimensionOrder(int[] shape)
    {
        var dims = new long[shape.Length];
        for (int i = 0; i < shape.Length; i++) dims[i] = shape[shape.Length - 1 - i];
        return dims;
    }

    public IReadOnlyList<GgufTensorInfo> Tensors => _tensors;
    public IReadOnlyDictionary<string, object> Metadata => _metadata;

    public GgufTensorInfo? FindTensor(string name) =>
        _byName.TryGetValue(name, out var info) ? info : null;

    public ReadOnlySpan<byte> GetTensorData(GgufTensorInfo tensor)
    {
        byte* pointer = GetTensorDataPtr(tensor);
        return new ReadOnlySpan<byte>(pointer, checked((int)(tensor.ElementCount * sizeof(float))));
    }

    public byte* GetTensorDataPtr(GgufTensorInfo tensor)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_syntheticBuffers.TryGetValue(tensor.Name, out nint syntheticPtr))
            return (byte*)syntheticPtr;
        if (_resolvedPointers.TryGetValue(tensor.Name, out nint cached))
            return (byte*)cached;

        string sourceName = _sourceNameByCanonical[tensor.Name];
        // Real BF16 storage throughout -- convert once, cache the pointer for the lifetime of
        // this source (same technique CosyVoiceLlmTensorSource uses for its own BF16 fallback).
        var data = _loader.ReadF32(sourceName);
        float* buffer = (float*)NativeMemory.Alloc((nuint)(data.Length * sizeof(float)));
        fixed (float* src = data)
            Buffer.MemoryCopy(src, buffer, data.Length * sizeof(float), data.Length * sizeof(float));
        _ownedPointers.Add((nint)buffer);
        _resolvedPointers[tensor.Name] = (nint)buffer;
        return (byte*)buffer;
    }

    /// <summary>
    /// The Safetensors counterpart of <see cref="QwenAsrLlmTensorSource.AudioTokenIdOffset"/> --
    /// -1 until <see cref="EnableAudioConditioning"/> has been called.
    /// </summary>
    public int AudioTokenIdOffset { get; private set; } = -1;

    /// <summary>
    /// The Safetensors counterpart of <see cref="QwenAsrLlmTensorSource.EnableAudioConditioning"/>
    /// -- identical real technique (synthetic combined `token_embd.weight`: real text rows,
    /// already resolved to F32 via <see cref="GetTensorDataPtr"/>'s BF16 conversion, followed by
    /// the AuT encoder's own per-frame projected output rows), just reading the text embedding
    /// through this class's own resolution path instead of `Dequantize.ToFloat32` directly.
    /// </summary>
    public void EnableAudioConditioning(ReadOnlySpan<float> audioEmbeddings, int numAudioTokens)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (numAudioTokens <= 0) throw new ArgumentOutOfRangeException(nameof(numAudioTokens));

        var textEmbedInfo = _byName["token_embd.weight"];
        int hiddenDim = checked((int)textEmbedInfo.Dimensions[0]);
        int textVocab = checked((int)textEmbedInfo.Dimensions[1]);
        if (audioEmbeddings.Length != (long)numAudioTokens * hiddenDim)
            throw new ArgumentException($"audioEmbeddings length {audioEmbeddings.Length} != numAudioTokens*hiddenDim ({numAudioTokens}*{hiddenDim}).", nameof(audioEmbeddings));

        byte* textEmbedPtr = GetTensorDataPtr(textEmbedInfo);

        int combinedVocab = textVocab + numAudioTokens;
        long combinedElementCount = (long)combinedVocab * hiddenDim;
        float* combined = (float*)NativeMemory.Alloc((nuint)(combinedElementCount * sizeof(float)));
        Buffer.MemoryCopy(textEmbedPtr, combined, combinedElementCount * sizeof(float), (long)textVocab * hiddenDim * sizeof(float));
        fixed (float* audioPtr = audioEmbeddings)
        {
            long audioElementCount = (long)numAudioTokens * hiddenDim;
            Buffer.MemoryCopy(audioPtr, combined + (long)textVocab * hiddenDim,
                audioElementCount * sizeof(float), audioElementCount * sizeof(float));
        }

        _ownedPointers.Add((nint)combined);
        _syntheticBuffers["token_embd.weight"] = (nint)combined;
        _byName["token_embd.weight"] = new GgufTensorInfo("token_embd.weight", 2, [hiddenDim, combinedVocab], DType.Float32, DataOffset: 0);
        _tensors.Clear();
        _tensors.AddRange(_byName.Values);

        AudioTokenIdOffset = textVocab;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (var p in _ownedPointers) NativeMemory.Free((void*)p);
        _ownedPointers.Clear();
        _loader.Dispose();
    }
}
