using System.Runtime.InteropServices;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Audio.CosyVoice;

/// <summary>
/// Presents CosyVoice2's LLM backbone (`models/cosyvoice2_llm.safetensors`, converted this
/// session from the official `llm.pt` checkpoint -- see docs/audio-review-progress.md's
/// CosyVoice section) to `OpenTail.Stingray.Engine`'s existing, unmodified text-generation
/// `ForwardPass` as if it were a standard `qwen2` model -- same architectural bet as
/// `QwenASR/QwenAsrLlmTensorSource.cs`, since `models/cosyvoice2_config.json` confirms this
/// backbone is a plain `Qwen2ForCausalLM` (verified directly, see the CosyVoice Phase-0 audit
/// entry above).
///
/// Unlike QwenASR's adapter (which wraps a `GgufModel` and only needs to remap metadata keys,
/// since the GGUF's `blk.*` tensor names already match llama.cpp's standard convention), this
/// checkpoint's real tensor names use a doubled `llm.model.model.layers.{i}.*` prefix (an
/// artifact of the original PyTorch module nesting: `CosyVoice2Model.llm.model` wraps a
/// standard HF `Qwen2Model`) and carry Q/K/V *biases* that
/// `SafetensorsTextModelPackage.TryMapToOpenTailTensorName` (the existing generic HF-name
/// mapper in Core, used by `SafetensorsTensorSource`) does not handle -- that mapper only
/// covers the bias-less standard HF layer-weight names. Rather than extend that shared mapper
/// (higher blast radius, touches every other safetensors-backed pipeline) this class maps the
/// specific tensor names this checkpoint uses directly, including biases. Keeps
/// `SafetensorsTensorSource.cs`'s BF16-to-F32 conversion path (dead code in practice here --
/// despite `cosyvoice2_config.json` declaring `torch_dtype: bfloat16`, the actual `llm.pt`
/// tensors this session converted are already plain F32, confirmed directly from the
/// converted safetensors header) since a future re-conversion from a genuinely BF16 source
/// should still work without this class needing a second pass.
/// </summary>
public sealed unsafe class CosyVoiceLlmTensorSource : IModelTensorSource, IDisposable
{
    private readonly SafetensorsLoader _loader;
    private readonly Dictionary<string, GgufTensorInfo> _byCanonicalName = new(StringComparer.Ordinal);
    private readonly Dictionary<string, string> _sourceNameByCanonicalName = new(StringComparer.Ordinal);
    private readonly Dictionary<string, nint> _convertedBf16Buffers = new(StringComparer.Ordinal);
    private readonly Dictionary<string, nint> _syntheticBuffers = new(StringComparer.Ordinal);
    private readonly List<nint> _ownedPointers = [];
    private readonly Lock _convertLock = new();
    private readonly List<GgufTensorInfo> _tensors;
    private readonly Dictionary<string, object> _metadata;
    private readonly int _hiddenDim;
    private readonly int _textVocabSize;
    private bool _disposed;
    private bool _speechGenerationMode;

    public CosyVoiceLlmTensorSource(string safetensorsPath, int numLayers, int hiddenDim, int numHeads, int numKvHeads, int headDim, int ffDim, int vocabSize, float ropeTheta, float rmsNormEps)
    {
        _loader = SafetensorsLoader.Open(safetensorsPath);

        MapIfPresent("llm.model.model.embed_tokens.weight", "token_embd.weight");
        MapIfPresent("llm.model.model.norm.weight", "output_norm.weight");
        MapIfPresent("llm.model.lm_head.weight", "output.weight");

        for (int i = 0; i < numLayers; i++)
        {
            string p = $"llm.model.model.layers.{i}.";
            string b = $"blk.{i}.";
            MapIfPresent(p + "input_layernorm.weight", b + "attn_norm.weight");
            MapIfPresent(p + "self_attn.q_proj.weight", b + "attn_q.weight");
            MapIfPresent(p + "self_attn.q_proj.bias", b + "attn_q.bias");
            MapIfPresent(p + "self_attn.k_proj.weight", b + "attn_k.weight");
            MapIfPresent(p + "self_attn.k_proj.bias", b + "attn_k.bias");
            MapIfPresent(p + "self_attn.v_proj.weight", b + "attn_v.weight");
            MapIfPresent(p + "self_attn.v_proj.bias", b + "attn_v.bias");
            MapIfPresent(p + "self_attn.o_proj.weight", b + "attn_output.weight");
            MapIfPresent(p + "post_attention_layernorm.weight", b + "ffn_norm.weight");
            MapIfPresent(p + "mlp.gate_proj.weight", b + "ffn_gate.weight");
            MapIfPresent(p + "mlp.up_proj.weight", b + "ffn_up.weight");
            MapIfPresent(p + "mlp.down_proj.weight", b + "ffn_down.weight");
        }

        // tie_word_embeddings=true in this checkpoint's real config (confirmed during the
        // Phase-0 audit) -- but this specific llm.pt DOES ship a separate lm_head.weight
        // (untied), so only fall back to the tied embedding if output.weight is genuinely
        // absent.
        if (!_byCanonicalName.ContainsKey("output.weight") && _byCanonicalName.TryGetValue("token_embd.weight", out var tokenEmbd))
        {
            _byCanonicalName["output.weight"] = tokenEmbd;
            _sourceNameByCanonicalName["output.weight"] = _sourceNameByCanonicalName["token_embd.weight"];
        }

        // CosyVoice-specific tensors NOT exposed through the IModelTensorSource seam (they
        // aren't part of the standard qwen2 tensor set ForwardPass expects) -- these are what
        // real CosyVoice generation actually needs on top of the backbone: a speech-token
        // embedding table extending the input embedding space, and a separate output head
        // over the speech vocabulary. See this class's doc comment and docs/audio-review-
        // progress.md's CosyVoice section for why they aren't wired into a working generation
        // loop yet (ForwardPass.Prefill only accepts token ids, no raw-embedding injection or
        // output-head-override entry point exists to mix text/speech embeddings or swap the
        // head mid-sequence).
        SpeechEmbeddingWeight = _loader.Contains("speech_embedding.weight") ? _loader.ReadF32("speech_embedding.weight") : null;
        LlmDecoderWeight = _loader.Contains("llm_decoder.weight") ? _loader.ReadF32("llm_decoder.weight") : null;
        LlmDecoderBias = _loader.Contains("llm_decoder.bias") ? _loader.ReadF32("llm_decoder.bias") : null;
        LlmEmbeddingWeight = _loader.Contains("llm_embedding.weight") ? _loader.ReadF32("llm_embedding.weight") : null;

        _hiddenDim = hiddenDim;
        _textVocabSize = vocabSize;
        _tensors = [.. _byCanonicalName.Values];

        var metadata = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            ["general.architecture"] = "qwen2",
            ["qwen2.embedding_length"] = hiddenDim,
            ["qwen2.block_count"] = numLayers,
            ["qwen2.attention.head_count"] = numHeads,
            ["qwen2.attention.head_count_kv"] = numKvHeads,
            ["qwen2.attention.key_length"] = headDim,
            ["qwen2.attention.value_length"] = headDim,
            ["qwen2.feed_forward_length"] = ffDim,
            ["qwen2.attention.layer_norm_rms_epsilon"] = rmsNormEps,
            ["qwen2.rope.freq_base"] = ropeTheta,
            ["qwen2.vocab_size"] = vocabSize,
            ["qwen2.context_length"] = 32768,
        };
        _metadata = metadata;
    }

    private void MapIfPresent(string sourceName, string canonicalName)
    {
        if (!_loader.TensorNames.Contains(sourceName)) return;
        string dtype = _loader.GetDtype(sourceName);
        DType mapped = dtype switch
        {
            "F32" => DType.Float32,
            "F16" => DType.Float16,
            "BF16" => DType.Float32, // converted at read time, see GetRawFloat32Ptr
            _ => throw new NotSupportedException($"CosyVoice LLM tensor '{sourceName}' has unsupported dtype '{dtype}'."),
        };
        int[] shape = _loader.GetShape(sourceName);
        long[] dims = ToGgufDimensionOrder(shape);
        _byCanonicalName[canonicalName] = new GgufTensorInfo(canonicalName, dims.Length, dims, mapped, DataOffset: 0);
        _sourceNameByCanonicalName[canonicalName] = sourceName;
    }

    /// <summary>Reverses a row-major HF shape into GGUF's fastest-varying-first order (same convention as <see cref="SafetensorsTensorSource"/>).</summary>
    private static long[] ToGgufDimensionOrder(int[] shape)
    {
        var dims = new long[shape.Length];
        for (int i = 0; i < shape.Length; i++) dims[i] = shape[shape.Length - 1 - i];
        return dims;
    }

    public IReadOnlyList<GgufTensorInfo> Tensors => _tensors;
    public IReadOnlyDictionary<string, object> Metadata => _metadata;

    /// <summary>FSQ speech-token embedding table, [6564, 896] -- extends the input embedding space beyond the 151936-token text vocab. Null if the checkpoint doesn't ship it.</summary>
    public float[]? SpeechEmbeddingWeight { get; }
    /// <summary>Speech-vocab output head weight, [6564, 896] -- the real head used at inference for speech-token prediction (distinct from the text `output.weight`/`lm_head`). Null if absent.</summary>
    public float[]? LlmDecoderWeight { get; }
    public float[]? LlmDecoderBias { get; }
    /// <summary>Small 2-row embedding, purpose not yet independently confirmed (likely sos/task special-token conditioning). Null if absent.</summary>
    public float[]? LlmEmbeddingWeight { get; }

    /// <summary>
    /// The token id a speech-vocab index maps to once <see cref="EnableSpeechGenerationMode"/>
    /// has been called: real speech token `s` (0..6563) becomes `SpeechTokenIdOffset + s` in
    /// the synthetic combined vocabulary. Prefill/Decode callers must apply this offset to
    /// every speech token id before passing it to <c>ForwardPass</c>, and subtract it back off
    /// (and add <see cref="LlmDecoderBias"/>, since <c>ForwardPass</c> has no final-layer-bias
    /// support -- see this method's doc comment) when interpreting returned logits.
    /// </summary>
    public int SpeechTokenIdOffset => _textVocabSize;

    /// <summary>
    /// Switches this source's `token_embd.weight`/`output.weight` from the text backbone to a
    /// real speech-token-generation configuration, entirely via composition -- no
    /// `OpenTail.Stingray.Engine` changes needed. `ForwardPass.EmbedTokenInto` (verified by
    /// reading `ForwardPass.Helpers.cs` directly) looks up embedding rows purely by tensor
    /// name + token id, with no hardcoded vocab assumption, so presenting a *synthetic*
    /// concatenated embedding table under the same `token_embd.weight` name works: real text
    /// rows `[0, textVocabSize)` followed by real `speech_embedding.weight` rows
    /// `[textVocabSize, textVocabSize+6564)`. Similarly `output.weight` is swapped to
    /// `llm_decoder.weight` (the real speech-vocab head). `ForwardPass` has no final-layer-
    /// bias support (checked directly -- only per-block `attn_output.bias` exists, no
    /// equivalent for the LM head), so `LlmDecoderBias` must be added to `Prefill`'s returned
    /// logits by the caller afterward, not applied here.
    ///
    /// Irreversible on this instance (construct a second source for text-mode use if both are
    /// needed concurrently) -- kept simple since real usage only ever needs one mode per
    /// `ForwardPass` instance at a time.
    /// </summary>
    public void EnableSpeechGenerationMode()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_speechGenerationMode) return;
        if (SpeechEmbeddingWeight is null || LlmDecoderWeight is null)
            throw new InvalidOperationException("CosyVoiceLlmTensorSource has no real speech_embedding/llm_decoder tensors to switch into speech-generation mode with.");

        var textEmbedInfo = _byCanonicalName["token_embd.weight"];
        byte* textEmbedPtr = GetTensorDataPtr(textEmbedInfo);

        int speechVocab = SpeechEmbeddingWeight.Length / _hiddenDim;
        int combinedVocab = _textVocabSize + speechVocab;
        long combinedElementCount = (long)combinedVocab * _hiddenDim;
        float* combined = (float*)NativeMemory.Alloc((nuint)(combinedElementCount * sizeof(float)));
        Buffer.MemoryCopy(textEmbedPtr, combined, combinedElementCount * sizeof(float), (long)_textVocabSize * _hiddenDim * sizeof(float));
        fixed (float* speechPtr = SpeechEmbeddingWeight)
        {
            Buffer.MemoryCopy(speechPtr, combined + (long)_textVocabSize * _hiddenDim,
                (long)speechVocab * _hiddenDim * sizeof(float), (long)speechVocab * _hiddenDim * sizeof(float));
        }
        _ownedPointers.Add((nint)combined);
        _syntheticBuffers["token_embd.weight"] = (nint)combined;
        _byCanonicalName["token_embd.weight"] = new GgufTensorInfo("token_embd.weight", 2, [_hiddenDim, combinedVocab], DType.Float32, DataOffset: 0);

        fixed (float* decoderPtr = LlmDecoderWeight)
        {
            long decoderBytes = (long)LlmDecoderWeight.Length * sizeof(float);
            float* decoderCopy = (float*)NativeMemory.Alloc((nuint)decoderBytes);
            Buffer.MemoryCopy(decoderPtr, decoderCopy, decoderBytes, decoderBytes);
            _ownedPointers.Add((nint)decoderCopy);
            _syntheticBuffers["output.weight"] = (nint)decoderCopy;
        }
        _byCanonicalName["output.weight"] = new GgufTensorInfo("output.weight", 2, [_hiddenDim, speechVocab], DType.Float32, DataOffset: 0);

        // ModelHyperparams.FromGgufMetadata reads qwen2.vocab_size to size ForwardPass's
        // internal logits/output buffers -- must match the real (now much smaller) output.
        // weight tensor exactly, or ForwardPass reads/writes past the real allocation and
        // segfaults (caught by direct testing, not by inspection -- see docs/audio-review-
        // progress.md's CosyVoice section).
        _metadata["qwen2.vocab_size"] = speechVocab;

        _tensors.Clear();
        _tensors.AddRange(_byCanonicalName.Values);
        _speechGenerationMode = true;
    }

    public GgufTensorInfo? FindTensor(string name) =>
        _byCanonicalName.TryGetValue(name, out var info) ? info : null;

    public ReadOnlySpan<byte> GetTensorData(GgufTensorInfo tensor)
    {
        byte* pointer = GetTensorDataPtr(tensor);
        long length = tensor.DType == DType.Float32 ? tensor.ElementCount * sizeof(float) : ByteLengthRaw(tensor.Name);
        return new ReadOnlySpan<byte>(pointer, checked((int)length));
    }

    private long ByteLengthRaw(string canonicalName)
    {
        string sourceName = _sourceNameByCanonicalName[canonicalName];
        _loader.TryGetMappedPointer(sourceName, out _, out long length, out _);
        return length;
    }

    public byte* GetTensorDataPtr(GgufTensorInfo tensor)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_syntheticBuffers.TryGetValue(tensor.Name, out nint syntheticPtr))
            return (byte*)syntheticPtr;
        string sourceName = _sourceNameByCanonicalName[tensor.Name];
        string dtype = _loader.GetDtype(sourceName);
        if (dtype != "BF16")
        {
            if (!_loader.TryGetMappedPointer(sourceName, out byte* pointer, out _, out _))
                throw new KeyNotFoundException($"CosyVoice LLM shard has no tensor '{sourceName}'.");
            return pointer;
        }

        lock (_convertLock)
        {
            if (_convertedBf16Buffers.TryGetValue(tensor.Name, out nint existing))
                return (byte*)existing;

            if (!_loader.TryGetMappedPointer(sourceName, out byte* srcPointer, out long byteLength, out _))
                throw new KeyNotFoundException($"CosyVoice LLM shard has no tensor '{sourceName}'.");

            long elementCount = byteLength / 2;
            float* dstPtr = (float*)NativeMemory.Alloc((nuint)(elementCount * sizeof(float)));
            ushort* srcU16 = (ushort*)srcPointer;
            for (long i = 0; i < elementCount; i++)
                dstPtr[i] = BitConverter.UInt32BitsToSingle((uint)srcU16[i] << 16);

            _convertedBf16Buffers[tensor.Name] = (nint)dstPtr;
            _ownedPointers.Add((nint)dstPtr);
            return (byte*)dstPtr;
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        lock (_convertLock)
        {
            foreach (nint ptr in _ownedPointers) NativeMemory.Free((void*)ptr);
            _ownedPointers.Clear();
            _convertedBf16Buffers.Clear();
        }
        _loader.Dispose();
    }
}
