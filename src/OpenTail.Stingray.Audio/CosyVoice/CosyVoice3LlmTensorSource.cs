using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Audio.CosyVoice;

/// <summary>
/// Presents CosyVoice3's LLM backbone (`models/cosyvoice3/CosyVoice3-2512_F16.gguf`, the
/// official pre-converted single-file GGUF from `Lourdle/Fun-CosyVoice3-0.5B-2512-GGUF`,
/// bundling LLM+flow+HiFT+tokenizer together under a custom `general.architecture =
/// cosyvoice3-2512` tag) to `OpenTail.Stingray.Engine`'s real `ForwardPass` as a `qwen2`
/// model -- same architectural bet as `QwenAsrLlmTensorSource` (GGUF, metadata-remap-only)
/// and `CosyVoiceLlmTensorSource` (CosyVoice2, safetensors, name-remap-with-bias), now
/// confirmed a THIRD time. Verified directly from real tensor shapes: `layers.0.self_attn.
/// q_proj.weight [896,896]`, `k_proj.weight [896,128]` (head_dim=64, n_kv_heads=2),
/// `mlp.gate_proj.weight [896,4864]` -- identical dims to CosyVoice2's Qwen2 backbone (896
/// hidden, 14 heads, 2 kv heads, 4864 ff, 24 layers), just packaged under this checkpoint's
/// own bare tensor-naming convention (`layers.N.*`/`embed_tokens.weight`/`norm.weight`, no
/// prefix at all -- the simplest of the three namings seen across QwenASR/CosyVoice2/
/// CosyVoice3 so far) and its own metadata keys (`num_hidden_layers`, `rms_norm_eps`,
/// `rope_theta`, not GGUF's usual `{arch}.*` convention).
///
/// This checkpoint's speech vocabulary is 6761 (`speech_embedding.weight`/`llm_decoder.
/// weight` both `[896,6761]`), NOT CosyVoice2's 6564 -- read from the real tensor shape here,
/// never hardcoded, since the two versions genuinely differ.
/// </summary>
public sealed unsafe class CosyVoice3LlmTensorSource : IModelTensorSource, IDisposable
{
    private readonly GgufModel _inner;
    private readonly Dictionary<string, object> _metadata;
    private readonly List<GgufTensorInfo> _tensors;
    private readonly int _hiddenDim;
    private readonly int _textVocabSize;
    private readonly List<nint> _ownedPointers = [];
    private bool _disposed;
    private bool _speechGenerationMode;
    private readonly Dictionary<string, GgufTensorInfo> _overrides = new(StringComparer.Ordinal);
    private readonly Dictionary<string, nint> _syntheticBuffers = new(StringComparer.Ordinal);
    private readonly Dictionary<string, string> _canonicalToReal = new(StringComparer.Ordinal);

    public CosyVoice3LlmTensorSource(GgufModel inner)
    {
        _inner = inner;

        int numLayers = GetInt("num_hidden_layers", 24);
        int numHeads = GetInt("num_attention_heads", 14);
        int numKvHeads = GetInt("num_key_value_heads", 2);
        float rmsNormEps = GetFloat("rms_norm_eps", 1e-6f);
        float ropeTheta = GetFloat("rope_theta", 1_000_000f);

        var embedInfo = inner.FindTensor("embed_tokens.weight") ?? throw new InvalidDataException("CosyVoice3 GGUF missing 'embed_tokens.weight'.");
        _hiddenDim = (int)embedInfo.Dimensions[0]; // ne=[hidden, vocab]
        _textVocabSize = (int)embedInfo.Dimensions[1];
        int headDim = _hiddenDim / numHeads;
        int ffDim = GetTensorDim("layers.0.mlp.gate_proj.weight", axis: 1, fallback: 4864);

        MapCanonical("embed_tokens.weight", "token_embd.weight");
        MapCanonical("norm.weight", "output_norm.weight");
        for (int i = 0; i < numLayers; i++)
        {
            string p = $"layers.{i}.";
            string b = $"blk.{i}.";
            MapCanonical(p + "input_layernorm.weight", b + "attn_norm.weight");
            MapCanonical(p + "self_attn.q_proj.weight", b + "attn_q.weight");
            MapCanonical(p + "self_attn.q_proj.bias", b + "attn_q.bias");
            MapCanonical(p + "self_attn.k_proj.weight", b + "attn_k.weight");
            MapCanonical(p + "self_attn.k_proj.bias", b + "attn_k.bias");
            MapCanonical(p + "self_attn.v_proj.weight", b + "attn_v.weight");
            MapCanonical(p + "self_attn.v_proj.bias", b + "attn_v.bias");
            MapCanonical(p + "self_attn.o_proj.weight", b + "attn_output.weight");
            MapCanonical(p + "post_attention_layernorm.weight", b + "ffn_norm.weight");
            MapCanonical(p + "mlp.gate_proj.weight", b + "ffn_gate.weight");
            MapCanonical(p + "mlp.up_proj.weight", b + "ffn_up.weight");
            MapCanonical(p + "mlp.down_proj.weight", b + "ffn_down.weight");
        }
        // No separate lm_head in this checkpoint -- tied to the text embedding.
        MapCanonical("embed_tokens.weight", "output.weight");

        var tensors = new List<GgufTensorInfo>();
        foreach (var (canonicalName, realName) in _canonicalToReal)
        {
            var real = inner.FindTensor(realName)!.Value;
            tensors.Add(real with { Name = canonicalName });
        }
        _tensors = tensors;

        SpeechEmbeddingInfo = inner.FindTensor("speech_embedding.weight");
        LlmDecoderInfo = inner.FindTensor("llm_decoder.weight");
        int speechVocab = SpeechEmbeddingInfo is { } sei ? (int)sei.Dimensions[1] : 0;
        SpeechVocabSize = speechVocab;

        _metadata = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            ["general.architecture"] = "qwen2",
            ["qwen2.embedding_length"] = _hiddenDim,
            ["qwen2.block_count"] = numLayers,
            ["qwen2.attention.head_count"] = numHeads,
            ["qwen2.attention.head_count_kv"] = numKvHeads,
            ["qwen2.attention.key_length"] = headDim,
            ["qwen2.attention.value_length"] = headDim,
            ["qwen2.feed_forward_length"] = ffDim,
            ["qwen2.attention.layer_norm_rms_epsilon"] = rmsNormEps,
            ["qwen2.rope.freq_base"] = ropeTheta,
            ["qwen2.vocab_size"] = _textVocabSize,
            ["qwen2.context_length"] = 32768,
        };
    }

    private void MapCanonical(string realName, string canonicalName)
    {
        if (_inner.FindTensor(realName) is null) return;
        _canonicalToReal[canonicalName] = realName;
    }

    private int GetInt(string key, int fallback) => _inner.Metadata.TryGetValue(key, out var v) ? Convert.ToInt32(v) : fallback;
    private float GetFloat(string key, float fallback) => _inner.Metadata.TryGetValue(key, out var v) ? Convert.ToSingle(v) : fallback;

    private int GetTensorDim(string name, int axis, int fallback)
    {
        var t = _inner.FindTensor(name);
        return t is { } info && axis < info.NDimensions ? (int)info.Dimensions[axis] : fallback;
    }

    public int SpeechVocabSize { get; }
    public int SpeechTokenIdOffset => _textVocabSize;
    private GgufTensorInfo? SpeechEmbeddingInfo { get; }
    private GgufTensorInfo? LlmDecoderInfo { get; }

    /// <summary>Same composition trick as `CosyVoiceLlmTensorSource.EnableSpeechGenerationMode` -- see that method's doc comment for the full rationale (verified against the real `ForwardPass.EmbedTokenInto` lookup-by-name behavior, no Engine changes needed).</summary>
    public void EnableSpeechGenerationMode()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_speechGenerationMode) return;
        if (SpeechEmbeddingInfo is not { } speechInfo || LlmDecoderInfo is not { } decoderInfo)
            throw new InvalidOperationException("CosyVoice3 GGUF has no speech_embedding/llm_decoder tensors to switch into speech-generation mode with.");

        var textEmbedInfo = _inner.FindTensor("embed_tokens.weight")!.Value;
        byte* textEmbedPtr = _inner.GetTensorDataPtr(textEmbedInfo);
        var textEmbedBytes = _inner.GetTensorData(textEmbedInfo);
        var speechEmbedBytes = _inner.GetTensorData(speechInfo);

        // These GGUF tensors may be F16 (confirmed: embed_tokens/llm_decoder/speech_embedding
        // are all Float16 in this checkpoint) -- dequantize both halves to F32 for the
        // synthetic combined buffer, since ForwardPass's dense CPU path expects a consistent
        // dtype per tensor and F32 is the simplest to hand-assemble correctly.
        int speechVocab = SpeechVocabSize;
        int combinedVocab = _textVocabSize + speechVocab;
        long combinedElementCount = (long)combinedVocab * _hiddenDim;
        float* combined = (float*)System.Runtime.InteropServices.NativeMemory.Alloc((nuint)(combinedElementCount * sizeof(float)));
        var combinedSpanText = new Span<float>(combined, _textVocabSize * _hiddenDim);
        Cpu.Dequantize.ToFloat32(textEmbedBytes, combinedSpanText, textEmbedInfo.DType, (long)_textVocabSize * _hiddenDim);
        var combinedSpanSpeech = new Span<float>(combined + (long)_textVocabSize * _hiddenDim, speechVocab * _hiddenDim);
        Cpu.Dequantize.ToFloat32(speechEmbedBytes, combinedSpanSpeech, speechInfo.DType, (long)speechVocab * _hiddenDim);
        _ownedPointers.Add((nint)combined);
        _syntheticBuffers["token_embd.weight"] = (nint)combined;
        _overrides["token_embd.weight"] = new GgufTensorInfo("token_embd.weight", 2, [_hiddenDim, combinedVocab], DType.Float32, DataOffset: 0);

        var decoderBytes = _inner.GetTensorData(decoderInfo);
        long decoderElementCount = (long)speechVocab * _hiddenDim;
        float* decoderCopy = (float*)System.Runtime.InteropServices.NativeMemory.Alloc((nuint)(decoderElementCount * sizeof(float)));
        Cpu.Dequantize.ToFloat32(decoderBytes, new Span<float>(decoderCopy, (int)decoderElementCount), decoderInfo.DType, decoderElementCount);
        _ownedPointers.Add((nint)decoderCopy);
        _syntheticBuffers["output.weight"] = (nint)decoderCopy;
        _overrides["output.weight"] = new GgufTensorInfo("output.weight", 2, [_hiddenDim, speechVocab], DType.Float32, DataOffset: 0);

        _metadata["qwen2.vocab_size"] = speechVocab;
        _tensors.RemoveAll(t => t.Name is "token_embd.weight" or "output.weight");
        _tensors.Add(_overrides["token_embd.weight"]);
        _tensors.Add(_overrides["output.weight"]);
        _speechGenerationMode = true;
    }

    public IReadOnlyList<GgufTensorInfo> Tensors => _tensors;
    public IReadOnlyDictionary<string, object> Metadata => _metadata;

    public GgufTensorInfo? FindTensor(string name)
    {
        if (_overrides.TryGetValue(name, out var overridden)) return overridden;
        if (!_canonicalToReal.TryGetValue(name, out var realName)) return null;
        var found = _inner.FindTensor(realName);
        return found is null ? null : found.Value with { Name = name };
    }

    public ReadOnlySpan<byte> GetTensorData(GgufTensorInfo tensor)
    {
        if (_syntheticBuffers.TryGetValue(tensor.Name, out nint ptr))
            return new ReadOnlySpan<byte>((void*)ptr, checked((int)(tensor.ElementCount * sizeof(float))));
        return _inner.GetTensorData(FindRealTensor(tensor.Name));
    }

    public byte* GetTensorDataPtr(GgufTensorInfo tensor)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_syntheticBuffers.TryGetValue(tensor.Name, out nint ptr)) return (byte*)ptr;
        return _inner.GetTensorDataPtr(FindRealTensor(tensor.Name));
    }

    private GgufTensorInfo FindRealTensor(string canonicalName) =>
        _inner.FindTensor(_canonicalToReal[canonicalName])
        ?? throw new KeyNotFoundException($"CosyVoice3 GGUF has no tensor for '{canonicalName}'.");

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (var ptr in _ownedPointers) System.Runtime.InteropServices.NativeMemory.Free((void*)ptr);
        _ownedPointers.Clear();
    }
}
