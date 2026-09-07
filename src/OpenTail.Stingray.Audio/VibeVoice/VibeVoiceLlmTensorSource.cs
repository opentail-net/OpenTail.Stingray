using OpenTail.Stingray.Audio.Rvc;

namespace OpenTail.Stingray.Audio.VibeVoice;

/// <summary>
/// Presents VibeVoice ASR's real text decoder (`model.language_model.*` tensors inside the packed
/// `vibevoice-asr-q8_0.gguf`) to `OpenTail.Stingray.Engine`'s existing, unmodified `ForwardPass`
/// as a standard `qwen2` model -- same bridging technique as `OmniVoice.OmniVoiceLlmTensorSource`/
/// `QwenASR.QwenAsrLlmTensorSource` (present real weights under `ForwardPass`'s native GGUF tensor
/// naming instead of writing a bespoke Transformer forward pass), confirmed real from
/// `examples/audio.cpp/src/models/vibevoice_asr/text_decoder.cpp`'s `make_qwen_decoder_config`/
/// `load_layer_weights`: a genuine Qwen2/2.5-family GQA decoder -- fused-bias `q_proj`/`k_proj`/
/// `v_proj` (QKV bias present, matching Qwen2's real convention, unlike Qwen3's bias-free +
/// `q_norm`/`k_norm` variant -- this class deliberately does NOT map any `q_norm`/`k_norm` tensors),
/// standard un-normalized RoPE, SwiGLU MLP, RMSNorm. Real, non-obvious detail confirmed from the
/// reference: `lm_head` is EITHER a separate `lm_head.weight`/`model.lm_head.weight` tensor OR
/// (when absent -- tied embeddings) the same tensor as `embed_tokens.weight`; this class checks
/// for a real `lm_head.weight`/`model.lm_head.weight` tensor and falls through to
/// `OpenTail.Stingray.Core`'s existing generic tied-embedding fallback when neither exists, same
/// convention as the sibling classes above.
///
/// <para>Unlike `OmniVoiceLlmTensorSource` (backed by a separate safetensors file) or
/// `QwenAsrLlmTensorSource` (backed by a separate native GGUF), this class is backed by
/// <see cref="RvcPackedTensorSource"/> over the SAME single packed checkpoint the tokenizer
/// encoders/connector already read (`audiocpp.tensor_names`-indexed, real names, not mmap
/// pointers) -- so every tensor is materialized into an owned unmanaged buffer at first access,
/// same as the sibling classes' own synthetic-buffer mechanism, just with a different upstream
/// data source.</para>
/// </summary>
public sealed unsafe class VibeVoiceLlmTensorSource : IModelTensorSource, IDisposable
{
    private readonly RvcPackedTensorSource _source;
    private readonly Dictionary<string, GgufTensorInfo> _byName = new(StringComparer.Ordinal);
    private readonly Dictionary<string, string> _sourceNameByCanonical = new(StringComparer.Ordinal);
    private readonly List<nint> _ownedPointers = [];
    private readonly Dictionary<string, nint> _resolvedPointers = new(StringComparer.Ordinal);
    private readonly Dictionary<string, nint> _syntheticBuffers = new(StringComparer.Ordinal);
    private readonly List<GgufTensorInfo> _tensors;
    private readonly Dictionary<string, object> _metadata;
    private bool _disposed;

    public VibeVoiceLlmTensorSource(
        RvcPackedTensorSource source,
        int numLayers, int hiddenDim, int numHeads, int numKvHeads, int headDim,
        int ffDim, int vocabSize, float ropeTheta, float rmsNormEps)
    {
        _source = source;

        MapIfPresent2D("model.language_model.embed_tokens.weight", "token_embd.weight", vocabSize, hiddenDim);
        MapIfPresent1D("model.language_model.norm.weight", "output_norm.weight", hiddenDim);

        bool hasSeparateLmHead = _source.HasTensor("model.language_model.lm_head.weight") || _source.HasTensor("lm_head.weight");
        if (hasSeparateLmHead)
        {
            string real = _source.HasTensor("model.language_model.lm_head.weight") ? "model.language_model.lm_head.weight" : "lm_head.weight";
            MapIfPresent2D(real, "output.weight", vocabSize, hiddenDim);
        }
        // else: tied embeddings, "output.weight" deliberately left unmapped (generic tied fallback).

        int qOut = numHeads * headDim;
        int kvOut = numKvHeads * headDim;
        for (int i = 0; i < numLayers; i++)
        {
            string p = $"model.language_model.layers.{i}.";
            string b = $"blk.{i}.";
            MapIfPresent1D(p + "input_layernorm.weight", b + "attn_norm.weight", hiddenDim);
            MapIfPresent2D(p + "self_attn.q_proj.weight", b + "attn_q.weight", qOut, hiddenDim);
            MapIfPresent1D(p + "self_attn.q_proj.bias", b + "attn_q.bias", qOut);
            MapIfPresent2D(p + "self_attn.k_proj.weight", b + "attn_k.weight", kvOut, hiddenDim);
            MapIfPresent1D(p + "self_attn.k_proj.bias", b + "attn_k.bias", kvOut);
            MapIfPresent2D(p + "self_attn.v_proj.weight", b + "attn_v.weight", kvOut, hiddenDim);
            MapIfPresent1D(p + "self_attn.v_proj.bias", b + "attn_v.bias", kvOut);
            MapIfPresent2D(p + "self_attn.o_proj.weight", b + "attn_output.weight", hiddenDim, qOut);
            MapIfPresent1D(p + "post_attention_layernorm.weight", b + "ffn_norm.weight", hiddenDim);
            MapIfPresent2D(p + "mlp.gate_proj.weight", b + "ffn_gate.weight", ffDim, hiddenDim);
            MapIfPresent2D(p + "mlp.up_proj.weight", b + "ffn_up.weight", ffDim, hiddenDim);
            MapIfPresent2D(p + "mlp.down_proj.weight", b + "ffn_down.weight", hiddenDim, ffDim);
        }

        _tensors = [.. _byName.Values];

        _metadata = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            ["general.architecture"] = "qwen2",
            ["qwen2.embedding_length"] = hiddenDim,
            ["qwen2.block_count"] = numLayers,
            ["qwen2.attention.head_count"] = numHeads,
            ["qwen2.attention.head_count_kv"] = numKvHeads,
            ["qwen2.feed_forward_length"] = ffDim,
            ["qwen2.attention.layer_norm_rms_epsilon"] = rmsNormEps,
            ["qwen2.rope.freq_base"] = ropeTheta,
            ["qwen2.vocab_size"] = vocabSize,
            ["qwen2.context_length"] = 32768,
        };
    }

    // Real GGUF dimension convention: reference declares shapes numpy-style [out,in] (row-major,
    // out slowest); ne-order (fastest-first) is the reverse, [in,out] -- matches every other
    // bridging class in this codebase (see OmniVoiceLlmTensorSource.ToGgufDimensionOrder).
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
        if (_syntheticBuffers.TryGetValue(tensor.Name, out nint syntheticPtr))
            return (byte*)syntheticPtr;
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

    /// <summary>Real -1 until <see cref="EnableSpeechConditioning"/> has been called: the first
    /// synthetic vocab id assigned to the injected speech-embedding rows.</summary>
    public int SpeechTokenIdOffset { get; private set; } = -1;

    /// <summary>
    /// Splices real per-frame speech embeddings (already combined from the acoustic + semantic
    /// connectors' outputs -- the combination rule itself lives in `session.cpp`, not yet ported,
    /// so this method takes the already-combined embeddings as an input rather than owning that
    /// decision) into the LLM's vocabulary as extra rows on `token_embd.weight`, same real
    /// technique as `OmniVoiceLlmTensorSource.EnableAudioConditioning`/
    /// `QwenAsrLlmTensorSource`'s equivalent: `ForwardPass` has no dedicated raw-embedding
    /// injection API, so a fixed, known-before-prefill set of continuous embeddings is instead
    /// appended as synthetic vocab entries, and the real prompt's `&lt;|box_start|&gt;` speech
    /// placeholder token ids (see `VibeVoiceASRTextTokenizer::build_prompt`'s
    /// `speech_positions`) get REMAPPED by the caller to
    /// `SpeechTokenIdOffset + slotIndex` before this source is handed to `ForwardPass`.
    /// </summary>
    public void EnableSpeechConditioning(ReadOnlySpan<float> speechEmbeddings, int numSpeechTokens)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (numSpeechTokens <= 0) throw new ArgumentOutOfRangeException(nameof(numSpeechTokens));

        var textEmbedInfo = _byName["token_embd.weight"];
        int hiddenDim = checked((int)textEmbedInfo.Dimensions[0]);
        int textVocab = checked((int)textEmbedInfo.Dimensions[1]);
        if (speechEmbeddings.Length != (long)numSpeechTokens * hiddenDim)
            throw new ArgumentException($"speechEmbeddings length {speechEmbeddings.Length} != numSpeechTokens*hiddenDim ({numSpeechTokens}*{hiddenDim}).", nameof(speechEmbeddings));

        byte* textEmbedPtr = GetTensorDataPtr(textEmbedInfo);

        int combinedVocab = textVocab + numSpeechTokens;
        long combinedElementCount = (long)combinedVocab * hiddenDim;
        float* combined = (float*)NativeMemory.Alloc((nuint)(combinedElementCount * sizeof(float)));
        Buffer.MemoryCopy(textEmbedPtr, combined, combinedElementCount * sizeof(float), (long)textVocab * hiddenDim * sizeof(float));
        fixed (float* speechPtr = speechEmbeddings)
        {
            long speechElementCount = (long)numSpeechTokens * hiddenDim;
            Buffer.MemoryCopy(speechPtr, combined + (long)textVocab * hiddenDim,
                speechElementCount * sizeof(float), speechElementCount * sizeof(float));
        }

        _ownedPointers.Add((nint)combined);
        _syntheticBuffers["token_embd.weight"] = (nint)combined;
        _byName["token_embd.weight"] = new GgufTensorInfo("token_embd.weight", 2, [hiddenDim, combinedVocab], DType.Float32, DataOffset: 0);
        _tensors.Clear();
        _tensors.AddRange(_byName.Values);

        SpeechTokenIdOffset = textVocab;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (var p in _ownedPointers) NativeMemory.Free((void*)p);
        _ownedPointers.Clear();
    }
}
