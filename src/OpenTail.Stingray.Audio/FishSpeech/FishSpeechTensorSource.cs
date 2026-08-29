
namespace OpenTail.Stingray.Audio.FishSpeech;

/// <summary>
/// Remaps Fish Speech S2 Pro's real GGUF tensor names (`embeddings.weight`, `norm.weight`,
/// `layers.{i}.attention.{wqkv,wo,q_norm,k_norm}.weight`, `layers.{i}.attention_norm.weight`,
/// `layers.{i}.feed_forward.{w1,w2,w3}.weight`, `layers.{i}.ffn_norm.weight` -- confirmed via
/// `list-tensors` on `models/s2-pro-q4_k_m.gguf`, cross-checked against
/// `examples/s2.cpp/src/s2_model.cpp`'s real tensor-loading code, e.g. `w1`=gate/`w3`=up/`w2`=
/// down confirmed from the real FFN forward pass at that file's line ~1044) into the canonical
/// llama.cpp-style names <see cref="OpenTail.Stingray.Engine.ForwardPass"/> expects
/// (`token_embd.weight`, `output_norm.weight`, `blk.{i}.attn_qkv.weight`, etc).
///
/// <para>This is the sanctioned reuse seam documented on <see cref="IModelTensorSource"/> itself
/// ("a type swap that cannot alter GGUF behaviour, and it lets another format feed the existing,
/// unmodified transformer loop") -- confirmed empirically, not assumed, that every per-layer
/// tensor Fish Speech's real GGUF has maps 1:1 onto a name `ForwardPass` already knows how to
/// consume: fused `attn_qkv.weight` (ForwardPass already supports this exact fused-QKV shape,
/// same code path used by other fused-QKV architectures), separate `attn_q_norm`/`attn_k_norm`
/// (matches this checkpoint's real `attention_qk_norm=true`), and NO separate `output.weight`
/// tensor (Fish Speech ties its output head to the input embedding -- `tie_word_embeddings=true`
/// metadata, confirmed absent from the real tensor list -- `ForwardPass` already falls back to
/// the embedding tensor automatically when `output.weight` is missing, exactly the tied-embedding
/// case). This means the ENTIRE slow-AR transformer trunk (RMSNorm, fused-QKV split, QK-norm,
/// RoPE, GQA attention, SwiGLU FFN) runs through `ForwardPass` completely unmodified -- only the
/// name translation is new code, not a from-scratch reimplementation of any of that math.</para>
///
/// <para>Does NOT expose the codec (`c.*`-prefixed tensors) or the fast-AR (`fast_*`-prefixed
/// tensors) -- those are separate components with their own forward passes, not part of this
/// remapping (the fast-AR in particular has a materially different shape: separate head_dim,
/// tiny context length, conditioned on a hidden-state tap rather than driven through the same
/// trunk -- see docs/audio-review-progress.md's Fish Speech section).</para>
/// </summary>
public sealed unsafe class FishSpeechTensorSource : IModelTensorSource
{
    private readonly GgufModel _inner;
    private readonly Dictionary<string, string> _rename;
    private readonly List<GgufTensorInfo> _tensors;
    private readonly Dictionary<string, GgufTensorInfo> _byCanonicalName;
    private readonly Dictionary<string, object> _metadata;

    public FishSpeechTensorSource(GgufModel inner, int numLayers)
    {
        _inner = inner;
        _rename = new Dictionary<string, string>
        {
            ["token_embd.weight"] = "embeddings.weight",
            ["output_norm.weight"] = "norm.weight",
        };

        // REAL BUG FOUND AND FIXED (this fire): ModelHyperparams.FromGgufMetadata falls back to
        // `embeddingDim / numHeads` (= 2560/32 = 80) when `{arch}.attention.key_length` is
        // absent -- and this checkpoint's real GGUF genuinely has no such key. The REAL head_dim
        // is 128, confirmed independently three ways: (1) the real HF `fishaudio/s2-pro`
        // `config.json`'s `text_config.head_dim=128`, (2) the real `wqkv.weight` tensor's output
        // width (6144) only factors as `(32 + 2*8) * 128`, never as `* 80`, (3) the real
        // `attn_q_norm.weight` tensor's own element count (128). Without this override,
        // `ForwardPass` silently sliced the real 6144-wide fused QKV tensor using WRONG offsets
        // (expecting only 3840 columns) -- it did not crash (no shape validation against the
        // tensor's actual width), so the earlier "ForwardPass_Constructs_And_ForwardEmbedding_
        // Runs" test passing was misleading: it only checked shape/finiteness, not correctness,
        // exactly the gap this project's golden-verification discipline exists to catch. Fixed
        // by synthesizing the missing metadata key here, the same sanctioned mechanism
        // `ModelHyperparams.FromGgufMetadata` already reads generically -- not a new code path.
        _metadata = new Dictionary<string, object>(inner.Metadata)
        {
            ["fish-speech.attention.key_length"] = 128,
            // Keep block_count consistent with numLayers, same fix as QwenTtsTalkerTensorSource:
            // this source only ever aliases blk.0..numLayers-1 below, so ForwardPass must not
            // believe there are more (throws "Missing tensor: blk.N.*" otherwise). Also a real
            // bisection knob for the NaN investigation (docs/audio-review-progress.md).
            ["fish-speech.block_count"] = numLayers,
        };
        for (int i = 0; i < numLayers; i++)
        {
            _rename[$"blk.{i}.attn_norm.weight"] = $"layers.{i}.attention_norm.weight";
            _rename[$"blk.{i}.attn_qkv.weight"] = $"layers.{i}.attention.wqkv.weight";
            _rename[$"blk.{i}.attn_output.weight"] = $"layers.{i}.attention.wo.weight";
            _rename[$"blk.{i}.attn_q_norm.weight"] = $"layers.{i}.attention.q_norm.weight";
            _rename[$"blk.{i}.attn_k_norm.weight"] = $"layers.{i}.attention.k_norm.weight";
            _rename[$"blk.{i}.ffn_norm.weight"] = $"layers.{i}.ffn_norm.weight";
            _rename[$"blk.{i}.ffn_gate.weight"] = $"layers.{i}.feed_forward.w1.weight";
            _rename[$"blk.{i}.ffn_up.weight"] = $"layers.{i}.feed_forward.w3.weight";
            _rename[$"blk.{i}.ffn_down.weight"] = $"layers.{i}.feed_forward.w2.weight";
        }

        _byCanonicalName = new Dictionary<string, GgufTensorInfo>();
        _tensors = new List<GgufTensorInfo>();
        foreach (var (canonical, real) in _rename)
        {
            var info = _inner.FindTensor(real);
            if (info is null) continue; // e.g. attn_q_norm absent on a non-qk-norm layer -- tolerate, ForwardPass probes with FindTensor too
            _byCanonicalName[canonical] = info.Value;
            _tensors.Add(info.Value);
        }
    }

    public IReadOnlyList<GgufTensorInfo> Tensors => _tensors;
    public IReadOnlyDictionary<string, object> Metadata => _metadata;

    public GgufTensorInfo? FindTensor(string name) =>
        _byCanonicalName.TryGetValue(name, out var info) ? info : null;

    public ReadOnlySpan<byte> GetTensorData(GgufTensorInfo tensor) => _inner.GetTensorData(tensor);

    public byte* GetTensorDataPtr(GgufTensorInfo tensor) => _inner.GetTensorDataPtr(tensor);
}
