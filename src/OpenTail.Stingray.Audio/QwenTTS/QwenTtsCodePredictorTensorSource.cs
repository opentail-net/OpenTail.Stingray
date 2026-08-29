
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

        // Pre-allocate synthetic buffers at constructor time so ForwardPass's constructor immediately
        // captures pointers to these mutable buffers rather than immutable raw GGUF weights.
        _embedBufferCapacityElements = 2 * _hiddenDim;
        _embedBuffer = (nint)NativeMemory.Alloc((nuint)(_embedBufferCapacityElements * sizeof(float)));
        _ownedPointers.Add(_embedBuffer);
        _syntheticBuffers["token_embd.weight"] = _embedBuffer;
        _byCanonicalName["token_embd.weight"] = new GgufTensorInfo("token_embd.weight", 2, [_hiddenDim, 2], DType.Float32, DataOffset: 0);

        _outputBufferCapacityElements = 2048 * _hiddenDim;
        _outputBuffer = (nint)NativeMemory.Alloc((nuint)(_outputBufferCapacityElements * sizeof(float)));
        _ownedPointers.Add(_outputBuffer);
        _syntheticBuffers["output.weight"] = _outputBuffer;
        _byCanonicalName["output.weight"] = new GgufTensorInfo("output.weight", 2, [_hiddenDim, 2048], DType.Float32, DataOffset: 0);

        _tensors.Clear();
        _tensors.AddRange(_byCanonicalName.Values);
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

    private nint _embedBuffer;
    private long _embedBufferCapacityElements;
    private nint _outputBuffer;
    private long _outputBufferCapacityElements;

    /// <summary>
    /// Writes `token_embd.weight` with a synthetic buffer of caller-composed per-position
    /// embeddings -- the exact same technique <see cref="QwenTtsTalkerTensorSource.SetPromptEmbedding"/>
    /// uses, needed here because the real Code Predictor's first pass input is `[talker_hidden,
    /// embed(c0)]` (a raw hidden-state bridge from the Talker plus a codec-table lookup), not a
    /// plain token id. Caller feeds sequential dummy ids `0..numRows-1` into `Prefill`/`Forward`.
    ///
    /// <para><b>REAL BUG FOUND AND FIXED (2026-08-28), same class as
    /// <see cref="QwenTtsTalkerTensorSource.SetPromptEmbedding"/>'s fix</b>: this allocated a NEW
    /// buffer at a NEW address every call, invisible to the `ForwardPass` already constructed
    /// against this source (its `_embTensor`/`_outputWeight` are captured ONCE, in the
    /// constructor). Every codebook step g=1..14 in `QwenTtsCodePredictorGeneration.
    /// GenerateAcousticCodes` calls this once per step on the SAME `ForwardPass` instance -- so
    /// every step after the first was reading the STALE first-call buffer's row 0 (always the
    /// prefill's `talkerLastHidden`, never the actual previous codebook's embedding). Fixed the
    /// same way: one persistent buffer, written in place.</para>
    /// </summary>
    public void SetPromptEmbedding(float[] rows, int numRows)
    {
        long elementCount = (long)numRows * _hiddenDim;
        if (rows.Length != elementCount)
            throw new ArgumentException($"SetPromptEmbedding: expected {elementCount} elements ({numRows}x{_hiddenDim}), got {rows.Length}.");

        if (elementCount > _embedBufferCapacityElements)
        {
            throw new InvalidOperationException(
                $"SetPromptEmbedding: requested {numRows} rows ({elementCount} elements) exceeds the " +
                $"{_embedBufferCapacityElements}-element buffer already allocated for a live ForwardPass. " +
                "Call this with the largest row count first, then only equal-or-smaller row counts.");
        }

        fixed (float* src = rows)
            Buffer.MemoryCopy(src, (void*)_embedBuffer, _embedBufferCapacityElements * sizeof(float), elementCount * sizeof(float));
    }

    public void SetOutputHead(float[] lmHeadWeight, int vocabSize)
    {
        long elementCount = (long)vocabSize * _hiddenDim;
        if (lmHeadWeight.Length != elementCount)
            throw new ArgumentException($"SetOutputHead: expected {elementCount} elements ({vocabSize}x{_hiddenDim}), got {lmHeadWeight.Length}.");

        if (elementCount > _outputBufferCapacityElements)
        {
            throw new InvalidOperationException(
                $"SetOutputHead: requested {elementCount} elements exceeds the {_outputBufferCapacityElements}-" +
                "element buffer already allocated for a live ForwardPass.");
        }

        fixed (float* src = lmHeadWeight)
            Buffer.MemoryCopy(src, (void*)_outputBuffer, _outputBufferCapacityElements * sizeof(float), elementCount * sizeof(float));
    }

    public void Dispose()
    {
        foreach (var ptr in _ownedPointers)
            NativeMemory.Free((void*)ptr);
        _ownedPointers.Clear();
    }
}
