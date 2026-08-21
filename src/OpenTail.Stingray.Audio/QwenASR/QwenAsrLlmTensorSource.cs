using OpenTail.Stingray.Core;

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
/// file to disk (see docs/audio-review-progress.md's QwenASR section for the two options this
/// was weighed against): `OpenTail.Stingray.Core.IModelTensorSource` exists exactly for this
/// purpose ("lets another format feed the existing, unmodified transformer loop" -- its own
/// doc comment) and `ForwardPass`'s constructor already accepts the interface, not a concrete
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
/// NOT YET WIRED into an actual `ForwardPass`/generation loop, KV cache, or the multimodal
/// audio-embedding injection point (replacing `audio_pad_token_id` embeddings with the AuT
/// encoder's projected features before decode) -- this class only establishes that the
/// Engine's real Qwen3 forward pass CAN consume this checkpoint's tensors under its own
/// architecture-detection logic. Structural-only, not yet exercised end-to-end.
/// </summary>
public sealed unsafe class QwenAsrLlmTensorSource : IModelTensorSource
{
    private readonly GgufModel _inner;
    private readonly IReadOnlyDictionary<string, object> _metadata;
    private readonly IReadOnlyList<GgufTensorInfo> _tensors;

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

        var tensors = new List<GgufTensorInfo>();
        foreach (var t in inner.Tensors)
        {
            if (t.Name.StartsWith("blk.", StringComparison.Ordinal) ||
                t.Name is "token_embd.weight" or "output.weight" or "output_norm.weight")
            {
                tensors.Add(t);
            }
        }
        _tensors = tensors;
    }

    private static void RemapIfPresent(Dictionary<string, object> metadata, string sourceKey, string targetKey)
    {
        if (metadata.TryGetValue(sourceKey, out var value))
            metadata[targetKey] = value;
    }

    public IReadOnlyList<GgufTensorInfo> Tensors => _tensors;

    public IReadOnlyDictionary<string, object> Metadata => _metadata;

    public GgufTensorInfo? FindTensor(string name)
    {
        // Only the filtered LLM-half tensor names are valid lookups through this source;
        // delegate directly since the underlying GgufModel already indexes by exact name.
        var found = _inner.FindTensor(name);
        if (found is null) return null;
        return found.Value.Name.StartsWith("blk.", StringComparison.Ordinal) ||
               found.Value.Name is "token_embd.weight" or "output.weight" or "output_norm.weight"
            ? found
            : null;
    }

    public ReadOnlySpan<byte> GetTensorData(GgufTensorInfo tensor) => _inner.GetTensorData(tensor);

    public byte* GetTensorDataPtr(GgufTensorInfo tensor) => _inner.GetTensorDataPtr(tensor);
}
