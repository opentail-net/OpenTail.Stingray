using System.Runtime.InteropServices;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Audio.QwenASR;

/// <summary>
/// Presents the LLM half of a Qwen3-ASR GGUF (`general.architecture = qwen3asr`) to
/// `OpenTail.Stingray.Engine`'s existing, unmodified text-generation `ForwardPass` as if it
/// were a standard `qwen3` model -- letting the real Qwen3 GQA/QK-norm/RoPE/SwiGLU forward
/// pass run this checkpoint's `blk.*`/`token_embd`/`output`/`output_norm` tensors directly,
/// rather than hand-writing a second scalar Qwen3 transformer implementation the way
/// Parakeet's Conformer or Chatterbox's T3 needed for genuinely novel architectures.
///
/// This is the metadata-remapping-adapter path chosen over splitting a second physical GGUF
/// file to disk (see docs/audio-review-progress.md for the two options this was weighed
/// against): `OpenTail.Stingray.Core.IModelTensorSource` exists exactly for this purpose
/// ("lets another format feed the existing, unmodified transformer loop" -- its own doc
/// comment) and `ForwardPass`'s constructor already accepts the interface, not a concrete
/// `GgufModel`, so this adapter requires zero changes to Engine code.
///
/// Two things this adapter does, both required because Qwen3-ASR bundles the audio encoder
/// and LLM decoder in one multimodal GGUF with a custom `qwen3asr.llm.*`/`qwen3asr.audio.*`
/// metadata namespace instead of the standard llama.cpp `{arch}.*` convention `ModelGraph.cs`
/// expects:
/// 1. Remaps the handful of `qwen3asr.llm.*` hyperparameter keys `ModelGraph.cs` actually
///    reads (traced directly from its source: `embedding_length`, `attention.head_count`,
///    `attention.head_count_kv`, `attention.key_length`/`value_length`, `block_count`,
///    `feed_forward_length`, `attention.layer_norm_rms_epsilon`, `rope.freq_base`,
///    `vocab_size`, `context_length`) into `qwen3.*` names, and overrides
///    `general.architecture` to the literal string `"qwen3"` so `ModelCompatibility`'s
///    architecture gate and `ModelGraph`'s hyperparameter reader both recognize it. Every
///    other metadata key (tokenizer.*, the `audio.*`-prefixed ones, etc.) passes through
///    unchanged.
/// 2. Filters `Tensors`/`FindTensor` down to only the standard `blk.*`/`token_embd.weight`/
///    `output.weight`/`output_norm.weight` tensors -- the `audio.*`-prefixed AuT encoder
///    tensors are NOT exposed here (they're irrelevant to the text-generation forward pass
///    and would only confuse a strict weight-loading pass expecting a closed, known tensor
///    set for the architecture).
///
/// <see cref="EnableAudioConditioning"/> adds the real multimodal audio-embedding injection:
/// same composition-only technique CosyVoice's `CosyVoiceLlmTensorSource.
/// EnableSpeechGenerationMode` uses (see docs/audio-review-progress.md) -- `ForwardPass.
/// EmbedTokenInto` resolves an embedding row purely by looking up whatever tensor is bound to
/// the canonical name `"token_embd.weight"` and indexing by token id, with no hardcoded
/// vocabulary assumption, so a synthetic combined table (real text rows, dequantized, followed
/// by the audio encoder's own per-frame continuous output rows) presented under that same name
/// works transparently. Unlike CosyVoice's case, `output.weight`/vocab_size do NOT need
/// swapping here -- QwenASR still predicts real text tokens, only the *input* embedding space
/// grows for the duration of one utterance's audio positions.
/// </summary>
public sealed unsafe class QwenAsrLlmTensorSource : IModelTensorSource, IDisposable, IQwenAsrAudioConditionableSource
{
    private readonly GgufModel _inner;
    private readonly Dictionary<string, object> _metadata;
    private readonly Dictionary<string, GgufTensorInfo> _byName = new(StringComparer.Ordinal);
    private readonly List<nint> _ownedPointers = [];
    private List<GgufTensorInfo> _tensors;
    private bool _disposed;
    private nint _combinedEmbeddingPtr;

    public QwenAsrLlmTensorSource(GgufModel inner)
    {
        _inner = inner;

        var metadata = new Dictionary<string, object>(inner.Metadata, StringComparer.Ordinal);
        metadata["general.architecture"] = "qwen3";
        RemapIfPresent(metadata, "qwen3asr.llm.d_model", "qwen3.embedding_length");
        RemapIfPresent(metadata, "qwen3asr.llm.n_heads", "qwen3.attention.head_count");
        RemapIfPresent(metadata, "qwen3asr.llm.n_kv_heads", "qwen3.attention.head_count_kv");
        RemapIfPresent(metadata, "qwen3asr.llm.head_dim", "qwen3.attention.key_length");
        RemapIfPresent(metadata, "qwen3asr.llm.head_dim", "qwen3.attention.value_length");
        RemapIfPresent(metadata, "qwen3asr.llm.n_layers", "qwen3.block_count");
        RemapIfPresent(metadata, "qwen3asr.llm.ff_dim", "qwen3.feed_forward_length");
        RemapIfPresent(metadata, "qwen3asr.llm.rms_norm_eps", "qwen3.attention.layer_norm_rms_epsilon");
        RemapIfPresent(metadata, "qwen3asr.llm.rope_theta", "qwen3.rope.freq_base");
        RemapIfPresent(metadata, "qwen3asr.llm.vocab_size", "qwen3.vocab_size");
        RemapIfPresent(metadata, "qwen3asr.llm.max_pos", "qwen3.context_length");
        _metadata = metadata;

        foreach (var t in inner.Tensors)
        {
            if (t.Name.StartsWith("blk.", StringComparison.Ordinal) ||
                t.Name is "token_embd.weight" or "output.weight" or "output_norm.weight")
            {
                _byName[t.Name] = t;
            }
        }
        _tensors = [.. _byName.Values];
    }

    private static void RemapIfPresent(Dictionary<string, object> metadata, string sourceKey, string targetKey)
    {
        if (metadata.TryGetValue(sourceKey, out var value))
            metadata[targetKey] = value;
    }

    /// <summary>
    /// The token id an audio frame position maps to once <see cref="EnableAudioConditioning"/>
    /// has been called: audio frame `f` (0..numAudioTokens-1) becomes
    /// <c>AudioTokenIdOffset + f</c> in the synthetic combined embedding space. Callers build
    /// the prompt token sequence with these ids at the real checkpoint's `&lt;|audio_pad|&gt;`
    /// positions before calling <c>ForwardPass.Prefill</c>. -1 until enabled.
    /// </summary>
    public int AudioTokenIdOffset { get; private set; } = -1;

    /// <summary>
    /// Builds a synthetic combined `token_embd.weight` (real text-vocab rows, dequantized to
    /// F32, followed by <paramref name="numAudioTokens"/> rows taken directly from
    /// <paramref name="audioEmbeddings"/> -- the AuT audio encoder's own projected output, see
    /// <see cref="QwenAsrAudioEncoder.Forward"/>) and presents it under the same name, exactly
    /// as <c>CosyVoiceLlmTensorSource.EnableSpeechGenerationMode</c> does for its speech-token
    /// case. `audioEmbeddings` must be row-major [numAudioTokens, hiddenDim] where hiddenDim
    /// matches `qwen3.embedding_length` (the AuT encoder's adapter already projects into this
    /// exact space -- see `QwenAsrEncoderConfig.QwenHiddenDim`).
    ///
    /// Irreversible on this instance and only valid for the audio clip it was built from --
    /// construct a fresh source (or call again to rebuild) per utterance, since the audio rows
    /// are utterance-specific, unlike CosyVoice's small fixed speech vocabulary.
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

        int combinedVocab = textVocab + numAudioTokens;
        long combinedElementCount = (long)combinedVocab * hiddenDim;
        float* combined = (float*)NativeMemory.Alloc((nuint)(combinedElementCount * sizeof(float)));

        // Dequantize the real text rows in one pass (this checkpoint's token_embd.weight is
        // Q4_K/Q8_0-quantized, not F32 -- Dequantize.ToFloat32 handles any DType uniformly, same
        // helper KokoroWeights.cs already uses for its own tensor loading).
        var rawText = _inner.GetTensorData(textEmbedInfo);
        Dequantize.ToFloat32(rawText, new Span<float>(combined, checked((int)((long)textVocab * hiddenDim))), textEmbedInfo.DType, (long)textVocab * hiddenDim);

        fixed (float* audioPtr = audioEmbeddings)
        {
            long audioElementCount = (long)numAudioTokens * hiddenDim;
            Buffer.MemoryCopy(audioPtr, combined + (long)textVocab * hiddenDim,
                audioElementCount * sizeof(float), audioElementCount * sizeof(float));
        }

        _ownedPointers.Add((nint)combined);
        _combinedEmbeddingPtr = (nint)combined;
        _byName["token_embd.weight"] = new GgufTensorInfo("token_embd.weight", 2, [hiddenDim, combinedVocab], DType.Float32, DataOffset: 0);
        _tensors = [.. _byName.Values];

        AudioTokenIdOffset = textVocab;
    }

    public IReadOnlyList<GgufTensorInfo> Tensors => _tensors;

    public IReadOnlyDictionary<string, object> Metadata => _metadata;

    public GgufTensorInfo? FindTensor(string name) =>
        _byName.TryGetValue(name, out var info) ? info : null;

    public ReadOnlySpan<byte> GetTensorData(GgufTensorInfo tensor)
    {
        if (_combinedEmbeddingPtr != 0 && tensor.Name == "token_embd.weight")
            return new ReadOnlySpan<byte>((void*)_combinedEmbeddingPtr, checked((int)(tensor.ElementCount * sizeof(float))));
        return _inner.GetTensorData(tensor);
    }

    public byte* GetTensorDataPtr(GgufTensorInfo tensor)
    {
        if (_combinedEmbeddingPtr != 0 && tensor.Name == "token_embd.weight")
            return (byte*)_combinedEmbeddingPtr;
        return _inner.GetTensorDataPtr(tensor);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (var p in _ownedPointers) NativeMemory.Free((void*)p);
        _ownedPointers.Clear();
    }
}
