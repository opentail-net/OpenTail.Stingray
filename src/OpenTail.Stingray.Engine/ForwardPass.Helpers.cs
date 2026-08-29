
namespace OpenTail.Stingray.Engine;

// Part of ForwardPass (see ForwardPass.cs for the type summary). Embedding lookup (including
// Gemma 4 per-layer embedding), fused quantized matvec, norm-weight caching, tensor/bias
// resolution, and small buffer utilities shared across the decode and prefill paths.
public sealed unsafe partial class ForwardPass
{
    // ================================================================
    //  Embedding lookup (single-row dequant)
    // ================================================================

    private void EmbedToken(int token, int position) => EmbedTokenInto(token, _hidden, position);

    // ================================================================
    //  Gemma 4 Per-Layer-Embedding (PLE)
    // ================================================================

    // Token-major layout: per_layer_token_embd shape (PleAll=10752, vocab=262144) stores
    // one row of length PleAll per token (GGUF dim[0] is row width). Gather + dequant
    // the row, then per-layer normalise + add projection + scale.
    private void BuildPerLayerProjections(int token)
    {
        int stackedDim = _hp.NumLayers * _pleWidth;

        int bytesPerRow = (stackedDim / DTypeInfo.BlockSize(_pleTokenEmbed.DType))
                        * DTypeInfo.BytesPerBlock(_pleTokenEmbed.DType);
        byte* rowPtr = _pleTokenEmbed.DataPtr + (long)token * bytesPerRow;
        if (_pleTokenEmbed.DType == DType.Float32)
        {
            new ReadOnlySpan<float>((float*)rowPtr, stackedDim)
                .CopyTo(new Span<float>(_pleRowBuf, stackedDim));
        }
        else
        {
            SimdKernels.DequantRow(rowPtr, _pleRowBuf, stackedDim, _pleTokenEmbed.DType);
        }

        // Gemma scales every embedding table by sqrt(its hidden dim). The PLE table's
        // hidden dim is PleWidth (256 → 16×), matching the trunk's sqrt(EmbeddingDim)
        // scale on token_embd.
        float pleScale = MathF.Sqrt(_pleWidth);
        SimdKernels.ScaleInPlace(_pleRowBuf, pleScale, stackedDim);

        SimdKernels.MatVec(_projPerLayer, (byte*)_perLayerModelProj,
            _hidden, stackedDim, _embDim, DType.Float32);

        float embScale = 1.0f / MathF.Sqrt(_embDim);
        SimdKernels.ScaleInPlace(_projPerLayer, embScale, stackedDim);

        float invSqrt2 = 1.0f / MathF.Sqrt(2.0f);
        var projNormW = GetNormWeight(_perLayerProjNormTensor);
        for (int L = 0; L < _hp.NumLayers; L++)
        {
            float* slice = _projPerLayer + (long)L * _pleWidth;
            FastRmsNorm(slice, slice, projNormW, _pleWidth, _hp.RmsNormEps);
            SimdKernels.AddInPlace(slice, _pleRowBuf + (long)L * _pleWidth, _pleWidth);
            SimdKernels.ScaleInPlace(slice, invSqrt2, _pleWidth);
        }

    }

    private void ApplyPerLayerEmbedding(int layer)
    {
        float* slice = _projPerLayer + (long)layer * _pleWidth;
        FusedMatVec(_pleX, _pleInpGate![layer], _hidden, _pleWidth, _embDim);
        SimdKernels.GeluTanhMul(_pleX, slice, _pleX, _pleWidth);
        FusedMatVec(_pleY, _plePostProj![layer], _pleX, _embDim, _pleWidth);
        var postW = GetNormWeight(_plePostNorm![layer]);
        FastRmsNorm(_pleY, _pleY, postW, _embDim, _hp.RmsNormEps);
        SimdKernels.AddInPlace(_hidden, _pleY, _embDim);
    }

    private void EmbedTokenInto(int token, float* dest, int position = -1)
    {
        int bytesPerRow = (_embDim / DTypeInfo.BlockSize(_embTensor.DType))
                        * DTypeInfo.BytesPerBlock(_embTensor.DType);
        byte* rowPtr = _embTensor.DataPtr + (long)token * bytesPerRow;
        if (_embTensor.DType == DType.Float32)
        {
            new ReadOnlySpan<float>((float*)rowPtr, _embDim)
                .CopyTo(new Span<float>(dest, _embDim));
        }
        else
        {
            SimdKernels.DequantRow(rowPtr, dest, _embDim, _embTensor.DType);
        }

        // GPT-2: learned absolute position embedding, added once to the token embedding
        // (src/models/gpt2.cpp: inpL = ggml_add(tok_embd_lookup, pos_embd_lookup)).
        if (_posEmbdTensor is { } posEmbd && position >= 0)
        {
            int posBytesPerRow = (_embDim / DTypeInfo.BlockSize(posEmbd.DType))
                                * DTypeInfo.BytesPerBlock(posEmbd.DType);
            byte* posRowPtr = posEmbd.DataPtr + (long)position * posBytesPerRow;
            if (posEmbd.DType == DType.Float32)
            {
                SimdKernels.AddInPlace(dest, (float*)posRowPtr, _embDim);
            }
            else
            {
                SimdKernels.DequantRow(posRowPtr, _posEmbdScratch, _embDim, posEmbd.DType);
                SimdKernels.AddInPlace(dest, _posEmbdScratch, _embDim);
            }
        }
    }

    // ================================================================
    //  Fused quantized MatVec (no intermediate F32 weight buffer)
    // ================================================================

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void FusedMatVec(float* output, in TensorRef tensor, float* input, int rows, int cols)
    {
        SimdKernels.MatVec(output, tensor.DataPtr, input, rows, cols, tensor.DType);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void FastRmsNorm(float* output, float* input, float* weight, int size, float eps)
    {
        if (_useWideNorms) SimdKernels.RmsNormWide(output, input, weight, size, eps);
        else               SimdKernels.RmsNorm    (output, input, weight, size, eps);
    }

    /// <summary>
    /// Dispatches to LayerNorm (GPT-NeoX/Falcon/StarCoder2 with a bias tensor, or Command-R/
    /// cohere2 without one — see <see cref="_usesLayerNorm"/>) or the ordinary RMSNorm path
    /// every other architecture uses. <paramref name="bias"/> may be null (no bias tensor for
    /// this architecture, e.g. cohere2) — <see cref="SimdKernels.LayerNorm"/> handles that
    /// directly; it is ignored entirely on the RMSNorm path.
    /// </summary>
    private void FastNorm(float* output, float* input, float* weight, float* bias, int size, float eps)
    {
        if (_usesLayerNorm) SimdKernels.LayerNorm(output, input, weight, bias, size, eps);
        else                FastRmsNorm(output, input, weight, size, eps);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void FastPureRmsNorm(float* output, float* input, int size, float eps)
    {
        if (_useWideNorms) SimdKernels.PureRmsNormWide(output, input, size, eps);
        else               SimdKernels.PureRmsNorm    (output, input, size, eps);
    }

    // ================================================================
    //  Norm weight cache (tiny F32 weights, cached permanently)
    // ================================================================

    private float* GetNormWeight(in TensorRef tensor)
    {
        if (_normCache.TryGetValue(tensor.Name, out var cached))
            return (float*)cached;

        var data = _model.GetTensorData(tensor.Info);
        int count = (int)tensor.Info.ElementCount;
        var buf = Alloc(count);

        if (tensor.DType == DType.Float32)
            MemoryMarshal.Cast<byte, float>(data).Slice(0, count).CopyTo(new Span<float>(buf, count));
        else
            Dequantize.ToFloat32(data, new Span<float>(buf, count), tensor.DType, count);

        // GGUF converter for gemma family already bakes the HF "(1 + w)" RMSNorm
        // convention; we multiply by stored `w` directly (mirrors llama.cpp build_norm
        // in src/llama-graph.cpp). Verified vs actual GGUF: attn_norm ~8, attn_q_norm
        // ~0.98 — already final multipliers.

        _normCache[tensor.Name] = (nint)buf;
        return buf;
    }

    // ================================================================
    //  Tensor resolution
    // ================================================================

    private TensorRef ResolveTensor(string name)
    {
        var info = _model.FindTensor(name)
            ?? throw new InvalidOperationException($"Missing tensor: {name}");
        return new TensorRef(name, info, info.DType, _model.GetTensorDataPtr(info));
    }

    private float LoadScalarF32(string name)
    {
        var info = _model.FindTensor(name)
            ?? throw new InvalidOperationException($"Missing tensor: {name}");
        var data = _model.GetTensorData(info);
        float[] buf = new float[1];
        if (info.DType == DType.Float32)
            MemoryMarshal.Cast<byte, float>(data).Slice(0, 1).CopyTo(buf);
        else
            Dequantize.ToFloat32(data, buf.AsSpan(), info.DType, 1);
        return buf[0];
    }

    private float* LoadBias(string name, int count)
    {
        var info = _model.FindTensor(name)
            ?? throw new InvalidOperationException($"Missing bias tensor: {name}");
        var data = _model.GetTensorData(info);
        var buf = Alloc(count);
        if (info.DType == DType.Float32)
            MemoryMarshal.Cast<byte, float>(data).Slice(0, count).CopyTo(new Span<float>(buf, count));
        else
            Dequantize.ToFloat32(data, new Span<float>(buf, count), info.DType, count);
        return buf;
    }

    private readonly unsafe struct TensorRef
    {
        public readonly string Name;
        public readonly GgufTensorInfo Info;
        public readonly DType DType;
        public readonly byte* DataPtr;

        public TensorRef(string name, GgufTensorInfo info, DType dtype, byte* dataPtr)
        {
            Name = name; Info = info; DType = dtype; DataPtr = dataPtr;
        }
    }

    // ================================================================
    //  Utilities
    // ================================================================

    /// <summary>
    /// Apply RMSNorm independently to each head-sized chunk.
    /// weight has [headDim] elements and is shared across all heads.
    /// </summary>
    private static void PerHeadRmsNorm(float* data, float* weight, int numHeads, int headDim, float eps)
    {
        for (int h = 0; h < numHeads; h++)
            SimdKernels.RmsNorm(data + h * headDim, data + h * headDim, weight, headDim, eps);
    }

    /// <summary>
    /// OLMoE-shaped QK-norm: one RMS over the WHOLE projection vector, then a per-channel weight.
    ///
    /// <para>The reduction width is the whole point, and it is easy to get wrong. This used to loop
    /// per head and normalise over <paramref name="headDim"/> elements at a time, using that head's
    /// slice of the weight — which looks reasonable, and is what "per-head QK-norm with a
    /// per-channel weight" would mean. It is not what the model does. llama.cpp's
    /// <c>models/olmoe.cpp</c> applies <c>build_norm</c> to <c>Qcur</c>/<c>Kcur</c> while they are
    /// still <c>[n_embd, n_tokens]</c> and only reshapes into heads afterwards, so the RMS
    /// denominator spans all heads (2048 elements for OLMoE-1B-7B, not 128).</para>
    ///
    /// <para>Per-head and whole-vector RMS agree only if every head has the same RMS, so the two
    /// diverge immediately — at layer 0, on the first token. That was the OLMoE parity defect;
    /// see <c>OlmoeGreedyParityTests</c>.</para>
    /// </summary>
    private static void PerChannelRmsNorm(float* data, float* weight, int numHeads, int headDim, float eps) =>
        SimdKernels.RmsNorm(data, data, weight, numHeads * headDim, eps);

    private void ApplyQkNorm(float* q, float* k, int layer)
    {
        if (_perChannelQkNorm)
        {
            PerChannelRmsNorm(q, _qNorm[layer], _numHeads,   _headDim, _hp.RmsNormEps);
            PerChannelRmsNorm(k, _kNorm[layer], _numKvHeads, _headDim, _hp.RmsNormEps);
        }
        else
        {
            PerHeadRmsNorm(q, _qNorm[layer], _numHeads,   _headDim, _hp.RmsNormEps);
            PerHeadRmsNorm(k, _kNorm[layer], _numKvHeads, _headDim, _hp.RmsNormEps);
        }
    }

    /// <summary>
    /// Per-layer-head-dim QK-norm. <paramref name="k"/> may be null on KV-share layers
    /// where the K projection didn't run (the source layer already normed its own K).
    /// </summary>
    private void ApplyQkNormLayer(float* q, float* k, int layer, int layerHd, int kvHeads)
    {
        if (_perChannelQkNorm)
        {
            PerChannelRmsNorm(q, _qNorm[layer], _numHeads, layerHd, _hp.RmsNormEps);
            if (k != null)
                PerChannelRmsNorm(k, _kNorm[layer], kvHeads, layerHd, _hp.RmsNormEps);
        }
        else
        {
            PerHeadRmsNorm(q, _qNorm[layer], _numHeads, layerHd, _hp.RmsNormEps);
            if (k != null)
                PerHeadRmsNorm(k, _kNorm[layer], kvHeads, layerHd, _hp.RmsNormEps);
        }
    }

    private static void PerHeadPureRmsNorm(float* data, int numHeads, int headDim, float eps)
    {
        for (int h = 0; h < numHeads; h++)
            SimdKernels.PureRmsNorm(data + h * headDim, data + h * headDim, headDim, eps);
    }

    private static float* Alloc(int count) =>
        (float*)NativeMemory.AllocZeroed((nuint)(count * sizeof(float)));

    /// <summary>
    /// Widen one token's K or V row from a compact per-layer head packing (head <c>h</c> at
    /// <c>h * headDim</c>) to the KV cache's own head stride (head <c>h</c> at
    /// <c>h * cacheHeadStride</c>), which is fixed model-wide at <c>_maxHeadDim</c>.
    /// <para>The destination must already be zeroed; the gaps between heads are never written, so
    /// re-zeroing per token would be pure waste — every call writes exactly the same head slots.</para>
    /// </summary>
    private static void ScatterToCacheStride(float* dst, float* src, int numHeads,
        int headDim, int cacheHeadStride)
    {
        if (headDim == cacheHeadStride)
        {
            Copy(dst, src, numHeads * headDim);
            return;
        }
        for (int h = 0; h < numHeads; h++)
            Copy(dst + (long)h * cacheHeadStride, src + (long)h * headDim, headDim);
    }

    private static void Copy(float* dst, float* src, int size) =>
        new ReadOnlySpan<float>(src, size).CopyTo(new Span<float>(dst, size));

}
