using System;
using System.Collections.Generic;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Audio.QwenTTS;

/// <summary>
/// Remaps Qwen3-TTS Code Predictor's real GGUF tensor names (`code_pred.blk.{i}.*`, same file as
/// the Talker -- confirmed via `list-tensors` on the real `Serveurperso/Qwen3-TTS-GGUF`
/// conversion) into the canonical llama.cpp-style names <see cref="OpenTail.Stingray.Engine.ForwardPass"/>
/// expects, the same pattern as <see cref="QwenTtsTalkerTensorSource"/>.
///
/// <para><b>Confirmed real, not guessed</b>: SAME per-layer shape family as the Talker (separate
/// attn_q/k/v/output, per-head QK-RMSNorm, SwiGLU FFN) but genuinely smaller (5 layers vs. 28,
/// confirmed via `code_pred.block_count=5` metadata matching the official `code_predictor_config`
/// exactly) and structurally different at the top level: NO single shared embedding/output pair
/// -- instead 15 SEPARATE per-codebook tables, `code_pred.codec_embd.{0..14}.weight` and
/// `code_pred.lm_head.{0..14}.weight` (real autoregressive depth-expansion architecture already
/// documented in docs/audio-review-progress.md's QwenTTS entries: codebook g's input embeds via
/// table g, output projects via lm_head g). This tensor source aliases `token_embd.weight`/
/// `output.weight` to table 0 ONLY for `ForwardPass`'s construction-time metadata/shape probing
/// -- real per-codebook composition is separate, not-yet-built logic, not exposed generically
/// here.</para>
/// </summary>
public sealed class QwenTtsCodePredictorTensorSource : IModelTensorSource
{
    private readonly GgufModel _inner;
    private readonly Dictionary<string, string> _rename;
    private readonly List<GgufTensorInfo> _tensors;
    private readonly Dictionary<string, GgufTensorInfo> _byCanonicalName;
    private readonly Dictionary<string, object> _metadata;

    public QwenTtsCodePredictorTensorSource(GgufModel inner, int numLayers)
    {
        _inner = inner;

        _metadata = new Dictionary<string, object>(inner.Metadata);
        foreach (var (key, value) in inner.Metadata)
        {
            const string prefix = "qwen3-tts.code_pred.";
            if (key.StartsWith(prefix, StringComparison.Ordinal))
                _metadata[$"qwen3-tts.{key[prefix.Length..]}"] = value;
        }

        _rename = new Dictionary<string, string>
        {
            ["token_embd.weight"] = "code_pred.codec_embd.0.weight",
            ["output_norm.weight"] = "code_pred.output_norm.weight",
            ["output.weight"] = "code_pred.lm_head.0.weight",
        };
        for (int i = 0; i < numLayers; i++)
        {
            _rename[$"blk.{i}.attn_norm.weight"] = $"code_pred.blk.{i}.attn_norm.weight";
            _rename[$"blk.{i}.attn_q.weight"] = $"code_pred.blk.{i}.attn_q.weight";
            _rename[$"blk.{i}.attn_k.weight"] = $"code_pred.blk.{i}.attn_k.weight";
            _rename[$"blk.{i}.attn_v.weight"] = $"code_pred.blk.{i}.attn_v.weight";
            _rename[$"blk.{i}.attn_output.weight"] = $"code_pred.blk.{i}.attn_output.weight";
            _rename[$"blk.{i}.attn_q_norm.weight"] = $"code_pred.blk.{i}.attn_q_norm.weight";
            _rename[$"blk.{i}.attn_k_norm.weight"] = $"code_pred.blk.{i}.attn_k_norm.weight";
            _rename[$"blk.{i}.ffn_norm.weight"] = $"code_pred.blk.{i}.ffn_norm.weight";
            _rename[$"blk.{i}.ffn_gate.weight"] = $"code_pred.blk.{i}.ffn_gate.weight";
            _rename[$"blk.{i}.ffn_up.weight"] = $"code_pred.blk.{i}.ffn_up.weight";
            _rename[$"blk.{i}.ffn_down.weight"] = $"code_pred.blk.{i}.ffn_down.weight";
        }

        _byCanonicalName = new Dictionary<string, GgufTensorInfo>();
        _tensors = new List<GgufTensorInfo>();
        foreach (var (canonical, real) in _rename)
        {
            var info = _inner.FindTensor(real);
            if (info is null) continue;
            _byCanonicalName[canonical] = info.Value;
            _tensors.Add(info.Value);
        }
    }

    public IReadOnlyList<GgufTensorInfo> Tensors => _tensors;
    public IReadOnlyDictionary<string, object> Metadata => _metadata;

    public GgufTensorInfo? FindTensor(string name) =>
        _byCanonicalName.TryGetValue(name, out var info) ? info : null;

    public ReadOnlySpan<byte> GetTensorData(GgufTensorInfo tensor) => _inner.GetTensorData(tensor);

    public unsafe byte* GetTensorDataPtr(GgufTensorInfo tensor) => _inner.GetTensorDataPtr(tensor);
}
