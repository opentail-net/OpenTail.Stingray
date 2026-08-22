using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Audio.QwenTTS;

/// <summary>
/// Remaps Qwen3-TTS Talker's real GGUF tensor names (`talker.blk.{i}.*`, confirmed via
/// `list-tensors` on the real `Serveurperso/Qwen3-TTS-GGUF` conversion of the official
/// `Qwen/Qwen3-TTS-12Hz-0.6B-Base` checkpoint -- see docs/audio-review-progress.md's QwenTTS
/// entries) into the canonical llama.cpp-style names <see cref="OpenTail.Stingray.Engine.ForwardPass"/>
/// expects (`blk.{i}.*`), the same sanctioned reuse pattern as `FishSpeech.FishSpeechTensorSource`.
///
/// <para><b>Confirmed real, not guessed</b>: `talker.blk.{i}.attn_q/attn_k/attn_v/attn_output.
/// weight` (SEPARATE Q/K/V, unlike Fish Speech's fused `attn_qkv` -- this is actually the more
/// common/standard llama.cpp convention `ForwardPass` already natively expects, confirmed via
/// `ForwardPass.cs`'s own real `blk.{i}.attn_q.weight`/`attn_k.weight`/`attn_v.weight` resolution
/// code), `attn_q_norm`/`attn_k_norm` (real per-head QK-RMSNorm, confirmed via the real
/// `talker-forward.h`'s `q = ggml_rms_norm(...); q = ggml_mul(..., q_norm_w)` sequence -- BEFORE
/// RoPE), `attn_norm`/`ffn_norm`, `ffn_gate`/`ffn_up`/`ffn_down` (SwiGLU). Real per-tensor dtype:
/// all big matmul weights Q8_0, all norm vectors plain Float32 (confirmed via `list-tensors`).
/// </para>
///
/// <para>Real GGUF metadata confirmed to match the official `Qwen/Qwen3-TTS-12Hz-0.6B-Base`
/// `config.json`'s `talker_config` exactly (not assumed): `qwen3-tts.talker.block_count=28`,
/// `embedding_length=1024`, `feed_forward_length=3072`, `attention.head_count=16`,
/// `attention.head_count_kv=8`, `attention.key_length=128`, `rope.freq_base=1000000`. Real RoPE
/// convention confirmed from `talker-forward.h`'s own `ggml_rope_ext(..., GGML_ROPE_TYPE_NEOX,
/// ...)` call -- NEOX (half-split), not the interleaved-pairs convention Fish Speech's fast-AR
/// uses. The "mrope_interleaved"/"mrope_section" metadata describes MULTIMODAL AXIS
/// interleaving (irrelevant for TTS-only inference, where all three mrope axes share one
/// position id and the whole scheme collapses to plain 1D NEOX RoPE, confirmed by
/// `talker-forward.h`'s own comment) -- NOT the rotation style itself.</para>
///
/// <para>Does NOT expose `talker.codec_embd.weight`/`text_embd.weight`/`text_proj.*`/
/// `codec_head.weight` under generic names -- those need real per-timestep embedding
/// COMPOSITION (text projection, codec embedding lookup) analogous to
/// `FishSpeechPipeline.EmbedTextToken`/`EmbedSemanticToken`, not a plain token-id lookup
/// `ForwardPass.Forward` already does -- so this tensor source exposes `token_embd.weight`
/// (aliased to `codec_embd.weight`, needed only so `ForwardPass`'s construction-time metadata
/// probing finds an embedding tensor to infer vocab size from) and `output.weight`/
/// `output_norm.weight` for the trunk's final projection, while the real composition logic lives
/// in a separate weights loader (not yet built this fire -- see docs/audio-review-progress.md).
/// </para>
/// </summary>
public sealed unsafe class QwenTtsTalkerTensorSource : IModelTensorSource, IDisposable
{
    private readonly GgufModel _inner;
    private readonly Dictionary<string, string> _rename;
    private readonly List<GgufTensorInfo> _tensors;
    private readonly Dictionary<string, GgufTensorInfo> _byCanonicalName;
    private readonly Dictionary<string, object> _metadata;
    private readonly List<nint> _ownedPointers = [];
    private readonly Dictionary<string, nint> _syntheticBuffers = new(StringComparer.Ordinal);
    private readonly int _hiddenDim;

    public QwenTtsTalkerTensorSource(GgufModel inner, int numLayers)
    {
        _inner = inner;

        // Real metadata keys carry a "talker." infix (`qwen3-tts.talker.attention.head_count`,
        // confirmed via list-metadata on the real Serveurperso conversion) that
        // ModelHyperparams.FromGgufMetadata's generic `{arch}.attention.head_count`-style lookup
        // (arch = general.architecture = "qwen3-tts") does not know to strip -- this GGUF packs
        // the Talker, Code Predictor (qwen3-tts.code_pred.*), and speaker encoder
        // (qwen3-tts.spk_enc.*) configs side by side under one architecture name, unlike a
        // single-model GGUF. Synthesize the flat `{arch}.*` keys ForwardPass actually needs by
        // stripping the "talker." infix, the same sanctioned mechanism FishSpeechTensorSource
        // already uses for a differently-shaped real gap (a genuinely missing key there; a
        // differently-nested key here).
        _metadata = new Dictionary<string, object>(inner.Metadata);
        foreach (var (key, value) in inner.Metadata)
        {
            const string prefix = "qwen3-tts.talker.";
            if (key.StartsWith(prefix, StringComparison.Ordinal))
                _metadata[$"qwen3-tts.{key[prefix.Length..]}"] = value;
        }

        _rename = new Dictionary<string, string>
        {
            ["token_embd.weight"] = "talker.codec_embd.weight",
            ["output_norm.weight"] = "talker.output_norm.weight",
            ["output.weight"] = "talker.codec_head.weight",
        };
        for (int i = 0; i < numLayers; i++)
        {
            _rename[$"blk.{i}.attn_norm.weight"] = $"talker.blk.{i}.attn_norm.weight";
            _rename[$"blk.{i}.attn_q.weight"] = $"talker.blk.{i}.attn_q.weight";
            _rename[$"blk.{i}.attn_k.weight"] = $"talker.blk.{i}.attn_k.weight";
            _rename[$"blk.{i}.attn_v.weight"] = $"talker.blk.{i}.attn_v.weight";
            _rename[$"blk.{i}.attn_output.weight"] = $"talker.blk.{i}.attn_output.weight";
            _rename[$"blk.{i}.attn_q_norm.weight"] = $"talker.blk.{i}.attn_q_norm.weight";
            _rename[$"blk.{i}.attn_k_norm.weight"] = $"talker.blk.{i}.attn_k_norm.weight";
            _rename[$"blk.{i}.ffn_norm.weight"] = $"talker.blk.{i}.ffn_norm.weight";
            _rename[$"blk.{i}.ffn_gate.weight"] = $"talker.blk.{i}.ffn_gate.weight";
            _rename[$"blk.{i}.ffn_up.weight"] = $"talker.blk.{i}.ffn_up.weight";
            _rename[$"blk.{i}.ffn_down.weight"] = $"talker.blk.{i}.ffn_down.weight";
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

        _hiddenDim = _metadata.TryGetValue("qwen3-tts.embedding_length", out var hd) ? Convert.ToInt32(hd) : 1024;
    }

    public IReadOnlyList<GgufTensorInfo> Tensors => _tensors;
    public IReadOnlyDictionary<string, object> Metadata => _metadata;

    public GgufTensorInfo? FindTensor(string name) =>
        _byCanonicalName.TryGetValue(name, out var info) ? info : null;

    public ReadOnlySpan<byte> GetTensorData(GgufTensorInfo tensor)
    {
        if (_syntheticBuffers.TryGetValue(tensor.Name, out nint syntheticPtr))
            return new ReadOnlySpan<byte>((void*)syntheticPtr, checked((int)(tensor.ElementCount * sizeof(float))));
        return _inner.GetTensorData(tensor);
    }

    public byte* GetTensorDataPtr(GgufTensorInfo tensor)
    {
        if (_syntheticBuffers.TryGetValue(tensor.Name, out nint syntheticPtr))
            return (byte*)syntheticPtr;
        return _inner.GetTensorDataPtr(tensor);
    }

    /// <summary>
    /// Swaps `token_embd.weight` (real name `talker.codec_embd.weight`, otherwise a plain
    /// codec-token lookup table) to a synthetic buffer of caller-composed rows -- the same
    /// synthetic-embedding-table technique <c>CosyVoiceLlmTensorSource.EnableSpeechGenerationMode</c>
    /// already established for a different real gap (there: extending the vocab; here: feeding
    /// PRECOMPUTED per-position embeddings, since the real Talker prompt sums a text-projection
    /// stream and a codec-embedding stream at each position -- an operation no single token id
    /// can express). <c>ForwardPass</c> has no raw-embedding-input API (`ForwardEmbedding`/
    /// `LastHidden` are both unimplemented on it, confirmed by inspection -- see
    /// docs/audio-review-progress.md's QwenTTS Talker generation entry), so real per-position
    /// embeddings are instead exposed as `token_embd.weight` rows and the caller feeds
    /// sequential dummy ids `0..numRows-1` into `Prefill`/`Forward`, exploiting the fact
    /// `ForwardPass`'s embedding lookup only cares about `token_embd.weight[id]`, not what the
    /// id conventionally means.
    /// </summary>
    public void SetPromptEmbedding(float[] rows, int numRows)
    {
        long elementCount = (long)numRows * _hiddenDim;
        if (rows.Length != elementCount)
            throw new ArgumentException($"SetPromptEmbedding: expected {elementCount} elements ({numRows}x{_hiddenDim}), got {rows.Length}.");

        float* buffer = (float*)NativeMemory.Alloc((nuint)(elementCount * sizeof(float)));
        fixed (float* src = rows)
            Buffer.MemoryCopy(src, buffer, elementCount * sizeof(float), elementCount * sizeof(float));
        _ownedPointers.Add((nint)buffer);
        _syntheticBuffers["token_embd.weight"] = (nint)buffer;

        _byCanonicalName["token_embd.weight"] = new GgufTensorInfo("token_embd.weight", 2, [_hiddenDim, numRows], DType.Float32, DataOffset: 0);
        _tensors.Clear();
        _tensors.AddRange(_byCanonicalName.Values);
    }

    public void Dispose()
    {
        foreach (var ptr in _ownedPointers)
            NativeMemory.Free((void*)ptr);
        _ownedPointers.Clear();
    }
}
