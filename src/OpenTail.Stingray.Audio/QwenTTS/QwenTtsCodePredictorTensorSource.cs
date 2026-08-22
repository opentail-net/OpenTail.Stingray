using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
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
public sealed unsafe class QwenTtsCodePredictorTensorSource : IModelTensorSource, IDisposable
{
    private readonly GgufModel _inner;
    private readonly Dictionary<string, string> _rename;
    private readonly List<GgufTensorInfo> _tensors;
    private readonly Dictionary<string, GgufTensorInfo> _byCanonicalName;
    private readonly Dictionary<string, object> _metadata;
    private readonly Dictionary<string, nint> _syntheticBuffers = new(StringComparer.Ordinal);
    private readonly List<nint> _ownedPointers = [];
    private readonly int _hiddenDim;

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
    /// Swaps `token_embd.weight` to a synthetic buffer of caller-composed per-position
    /// embeddings -- the exact same technique <see cref="QwenTtsTalkerTensorSource.SetPromptEmbedding"/>
    /// uses, needed here because the real Code Predictor's first pass input is `[talker_hidden,
    /// embed(c0)]` (a raw hidden-state bridge from the Talker plus a codec-table lookup), not a
    /// plain token id. Caller feeds sequential dummy ids `0..numRows-1` into `Prefill`/`Forward`.
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

    /// <summary>
    /// Swaps `output.weight` to a caller-supplied real per-codebook `code_pred.lm_head.{g}.weight`
    /// buffer -- real autoregressive depth-expansion needs a DIFFERENT output head at each of the
    /// 15 acoustic-codebook steps, all sharing the same real shape (vocab=2048, dim=1024), so no
    /// metadata/shape change is needed, only the underlying data pointer.
    /// </summary>
    public void SetOutputHead(float[] lmHeadWeight, int vocabSize)
    {
        long elementCount = (long)vocabSize * _hiddenDim;
        if (lmHeadWeight.Length != elementCount)
            throw new ArgumentException($"SetOutputHead: expected {elementCount} elements ({vocabSize}x{_hiddenDim}), got {lmHeadWeight.Length}.");

        float* buffer = (float*)NativeMemory.Alloc((nuint)(elementCount * sizeof(float)));
        fixed (float* src = lmHeadWeight)
            Buffer.MemoryCopy(src, buffer, elementCount * sizeof(float), elementCount * sizeof(float));
        _ownedPointers.Add((nint)buffer);
        _syntheticBuffers["output.weight"] = (nint)buffer;

        _byCanonicalName["output.weight"] = new GgufTensorInfo("output.weight", 2, [_hiddenDim, vocabSize], DType.Float32, DataOffset: 0);
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
