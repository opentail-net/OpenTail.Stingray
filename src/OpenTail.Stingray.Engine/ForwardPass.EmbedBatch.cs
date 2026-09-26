using System.Runtime.InteropServices;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Engine;

public sealed unsafe partial class ForwardPass
{
    /// <inheritdoc />
    public bool SupportsBatchedHiddenStateExtraction =>
        !_usesUnweightedNorm &&
        _tqKvCache is null &&
        _layerKvSrc is null &&
        (!_hp.IsMoE || MoeBatchedPrefillSupported);

    /// <inheritdoc />
    public void ExtractHiddenStatesBatch(
        IReadOnlyList<IReadOnlyList<int>> sequences,
        Span<float> destination,
        ReadOnlySpan<int> destinationOffsets = default)
    {
        int M = sequences.Count;
        if (M == 0) return;

        if (M == 1)
        {
            int off = destinationOffsets.IsEmpty ? 0 : destinationOffsets[0];
            ExtractHiddenStates(sequences[0], destination.Slice(off * _embDim, sequences[0].Count * _embDim));
            return;
        }

        int Ntotal = 0;
        for (int s = 0; s < M; s++)
        {
            Ntotal += sequences[s].Count;
        }

        if (Ntotal == 0) return;

        if (destination.Length < Ntotal * _embDim)
            throw new ArgumentException(
                $"Destination length ({destination.Length}) must be at least {Ntotal * _embDim} (total tokens: {Ntotal}, embDim: {_embDim})",
                nameof(destination));

        if (!destinationOffsets.IsEmpty && destinationOffsets.Length < M)
            throw new ArgumentException(
                $"DestinationOffsets length ({destinationOffsets.Length}) must match sequence count ({M})",
                nameof(destinationOffsets));

        // Fallback for configurations not supported by the batched trunk
        if (!SupportsBatchedHiddenStateExtraction)
        {
            for (int s = 0; s < M; s++)
            {
                int dstTokenOffset = destinationOffsets.IsEmpty
                    ? (s == 0 ? 0 : CountPrecedingTokens(sequences, s))
                    : destinationOffsets[s];
                ExtractHiddenStates(sequences[s], destination.Slice(dstTokenOffset * _embDim, sequences[s].Count * _embDim));
            }
            return;
        }

        // Build flat layout
        var flatTokens = new int[Ntotal];
        var posOf = new int[Ntotal];
        var seqStartTokenIdx = new int[M];
        var seqLens = new int[M];
        int cur = 0;
        for (int s = 0; s < M; s++)
        {
            seqStartTokenIdx[s] = cur;
            int len = sequences[s].Count;
            seqLens[s] = len;
            for (int i = 0; i < len; i++)
            {
                flatTokens[cur] = sequences[s][i];
                posOf[cur] = i; // Reset position per sequence
                cur++;
            }
        }

        // Allocate batch buffers
        int maxKvHeads = _hp.LayerKvHeads is { } lkvs ? lkvs.Max() : _numKvHeads;
        int qDimMax = _numHeads * _maxHeadDim;
        int kvDimMax = maxKvHeads * _maxHeadDim;

        var batchHidden = (float*)NativeMemory.AllocZeroed((nuint)((long)Ntotal * _embDim * sizeof(float)));
        var batchResidual = (float*)NativeMemory.AllocZeroed((nuint)((long)Ntotal * _embDim * sizeof(float)));
        var batchNorm = (float*)NativeMemory.AllocZeroed((nuint)((long)Ntotal * _embDim * sizeof(float)));
        var batchQ = (float*)NativeMemory.AllocZeroed((nuint)((long)Ntotal * qDimMax * sizeof(float)));
        var batchK = (float*)NativeMemory.AllocZeroed((nuint)((long)Ntotal * kvDimMax * sizeof(float)));
        var batchV = (float*)NativeMemory.AllocZeroed((nuint)((long)Ntotal * kvDimMax * sizeof(float)));
        var batchAttnOut = (float*)NativeMemory.AllocZeroed((nuint)((long)Ntotal * qDimMax * sizeof(float)));
        var batchFfnGate = (float*)NativeMemory.AllocZeroed((nuint)((long)Ntotal * _intermDim * sizeof(float)));
        var batchFfnUp = (float*)NativeMemory.AllocZeroed((nuint)((long)Ntotal * _intermDim * sizeof(float)));

        var kStage = (_layerHeadDim is not null || _hp.LayerKvHeads is not null)
            ? (float*)NativeMemory.AllocZeroed((nuint)(kvDimMax * sizeof(float))) : null;
        var vStage = (_layerHeadDim is not null || _hp.LayerKvHeads is not null)
            ? (float*)NativeMemory.AllocZeroed((nuint)(kvDimMax * sizeof(float))) : null;

        bool batchedMoe = _hp.IsMoE;
        var batchMoeOut = batchedMoe
            ? (float*)NativeMemory.AllocZeroed((nuint)((long)Ntotal * _embDim * sizeof(float)))
            : null;

        var batchMlaAttnOutCompact = _isMla
            ? (float*)NativeMemory.AllocZeroed((nuint)((long)Ntotal * _numHeads * _mlaVDim * sizeof(float)))
            : null;

        int stackedPleDim = _hp.HasPerLayerTokenEmbd ? _hp.NumLayers * _pleWidth : 0;
        var batchPleProj = _hp.HasPerLayerTokenEmbd
            ? (float*)NativeMemory.AllocZeroed((nuint)((long)Ntotal * stackedPleDim * sizeof(float)))
            : null;

        try
        {
            // 1. Embed all tokens with sequence-relative positions
            for (int n = 0; n < Ntotal; n++)
                EmbedTokenInto(flatTokens[n], batchHidden + (long)n * _embDim, posOf[n]);

            if (_hp.EmbeddingScale != 1f)
            {
                for (int n = 0; n < Ntotal; n++)
                    SimdKernels.ScaleInPlace(batchHidden + (long)n * _embDim, _hp.EmbeddingScale, _embDim);
            }

            if (_hp.HasPerLayerTokenEmbd)
            {
                for (int n = 0; n < Ntotal; n++)
                    BuildPerLayerProjectionsBatched(flatTokens[n], batchHidden + (long)n * _embDim, batchPleProj + (long)n * stackedPleDim);
            }

            // 2. Transformer layers
            for (int layer = 0; layer < _hp.NumLayers; layer++)
            {
                int layerHd = _layerHeadDim?[layer] ?? _headDim;
                int layerKv = _hp.LayerKvHeads is { } lkv ? lkv[layer] : _numKvHeads;
                int qDim = _numHeads * layerHd;
                int kvDim = layerKv * layerHd;
                bool isSwa = _isSwaLayer is not null && _isSwaLayer[layer];
                int windowSize = isSwa ? _hp.SlidingWindowSize : -1;
                bool kEqV = _hp.AttentionKEqV && !isSwa && _wv[layer].DataPtr is null;

                var normW = GetNormWeight(_attnNorm[layer]);
                var attnNormB = _hasNormBias && _bAttnNorm is not null ? _bAttnNorm[layer] : null;

                // Batched attention-norm across all tokens
                for (int n = 0; n < Ntotal; n++)
                {
                    Copy(batchResidual + (long)n * _embDim, batchHidden + (long)n * _embDim, _embDim);
                    FastNorm(batchNorm + (long)n * _embDim,
                        batchHidden + (long)n * _embDim, normW, attnNormB, _embDim, _hp.RmsNormEps);
                }

                // Batched QKV projections
                if (_isMla)
                {
                    MlaComputeQkvBatched(layer, batchNorm, Ntotal, batchQ, batchK, batchV);
                }
                else
                {
                    MatMulBatchedCached(batchQ, in _wq[layer], batchNorm, Ntotal, qDim, _embDim);
                    if (kEqV)
                    {
                        MatMulBatchedCached(batchK, in _wk[layer], batchNorm, Ntotal, kvDim, _embDim);
                        for (int n = 0; n < Ntotal; n++)
                            Copy(batchV + (long)n * kvDim, batchK + (long)n * kvDim, kvDim);
                    }
                    else
                    {
                        MatMulBatchedCached(batchK, in _wk[layer], batchNorm, Ntotal, kvDim, _embDim);
                        MatMulBatchedCached(batchV, in _wv[layer], batchNorm, Ntotal, kvDim, _embDim);
                    }
                }

                if (_hasAttnBias)
                {
                    for (int n = 0; n < Ntotal; n++)
                    {
                        SimdKernels.AddInPlace(batchQ + (long)n * qDim, _bq[layer], qDim);
                        SimdKernels.AddInPlace(batchK + (long)n * kvDim, _bk[layer], kvDim);
                        SimdKernels.AddInPlace(batchV + (long)n * kvDim, _bv[layer], kvDim);
                    }
                }

                // Per-token QK-norm and RoPE using sequence-relative posOf[n]
                bool useRoPE = _hp.NoRopeLayerStep == 0 || (layer + 1) % _hp.NoRopeLayerStep != 0;
                if (_hp.RopeOnlySwaLayers) useRoPE = useRoPE && isSwa;

                for (int n = 0; n < Ntotal; n++)
                {
                    float* qn = batchQ + (long)n * qDim;
                    float* kn = batchK + (long)n * kvDim;
                    float* vn = batchV + (long)n * kvDim;
                    int pos = posOf[n];

                    if (_hasQkNorm && !_hp.UseL2QkNorm && !_hp.QkNormAfterRope)
                    {
                        ApplyQkNormLayer(qn, kn, layer, layerHd, layerKv);
                    }

                    if (_layerHeadDim is not null)
                    {
                        PerHeadPureRmsNorm(vn, layerKv, layerHd, _hp.RmsNormEps);
                    }

                    if (useRoPE)
                    {
                        ApplyRopeLayer(qn, pos, _numHeads, layer, layerHd);
                        ApplyRopeLayer(kn, pos, layerKv, layer, layerHd);
                    }

                    if (_hasQkNorm && !_hp.UseL2QkNorm && _hp.QkNormAfterRope)
                    {
                        ApplyQkNormLayer(qn, kn, layer, layerHd, layerKv);
                    }

                    if (_hasQkNorm && _hp.UseL2QkNorm && useRoPE)
                    {
                        PerHeadPureRmsNorm(qn, _numHeads, layerHd, _hp.RmsNormEps);
                        PerHeadPureRmsNorm(kn, layerKv, layerHd, _hp.RmsNormEps);
                    }
                }

                // Per-sequence attention execution: TruncateTo(0), append K/V for sequence s, run PrefillCoreAttention
                for (int s = 0; s < M; s++)
                {
                    int sLen = seqLens[s];
                    if (sLen == 0) continue;
                    int sStart = seqStartTokenIdx[s];

                    _kvCache.TruncateTo(0);
                    for (int i = 0; i < sLen; i++)
                    {
                        int n = sStart + i;
                        float* kn = batchK + (long)n * kvDim;
                        float* vn = batchV + (long)n * kvDim;

                        if (kStage is null)
                        {
                            _kvCache.Append(layer,
                                new ReadOnlySpan<float>(kn, kvDim),
                                new ReadOnlySpan<float>(vn, kvDim));
                        }
                        else
                        {
                            StageCompactKv(kStage, kn, kvDim, kvDimMax);
                            StageCompactKv(vStage!, vn, kvDim, kvDimMax);
                            _kvCache.Append(layer,
                                new ReadOnlySpan<float>(kStage, kvDimMax),
                                new ReadOnlySpan<float>(vStage, kvDimMax));
                        }
                        _kvCache.IncrementPosition();
                    }

                    PrefillCoreAttention(
                        batchQ + (long)sStart * qDim,
                        _kvCache,
                        layer,
                        sLen,
                        startPos: 0,
                        batchAttnOut + (long)sStart * qDim,
                        windowSize);
                }

                // Batched output projection across all tokens
                if (_isMla)
                {
                    MlaCompactAttnOutBatched(batchAttnOut, batchMlaAttnOutCompact!, Ntotal);
                    MatMulBatchedCached(batchNorm, in _wo[layer], batchMlaAttnOutCompact!, Ntotal, _embDim, _numHeads * _mlaVDim);
                }
                else
                {
                    MatMulBatchedCached(batchNorm, in _wo[layer], batchAttnOut, Ntotal, _embDim, qDim);
                }

                if (_hasAttnOutputBias)
                {
                    for (int n = 0; n < Ntotal; n++)
                        SimdKernels.AddInPlace(batchNorm + (long)n * _embDim, _bo[layer], _embDim);
                }

                if (_postAttnNorm is not null)
                {
                    var paNormW = GetNormWeight(_postAttnNorm[layer]);
                    for (int n = 0; n < Ntotal; n++)
                        FastRmsNorm(batchNorm + (long)n * _embDim, batchNorm + (long)n * _embDim, paNormW, _embDim, _hp.RmsNormEps);
                }

                if (!_hp.UseParallelResidual)
                {
                    for (int n = 0; n < Ntotal; n++)
                    {
                        float* h = batchHidden + (long)n * _embDim;
                        float* proj = batchNorm + (long)n * _embDim;
                        float* r = batchResidual + (long)n * _embDim;
                        Copy(h, proj, _embDim);
                        if (_hp.ResidualScale != 1f)
                            SimdKernels.ScaleInPlace(h, _hp.ResidualScale, _embDim);
                        SimdKernels.AddInPlace(h, r, _embDim);
                    }
                    for (int n = 0; n < Ntotal; n++)
                        Copy(batchResidual + (long)n * _embDim, batchHidden + (long)n * _embDim, _embDim);
                }

                // Batched FFN norm
                if (_ffnNorm[layer].DataPtr is null)
                {
                    for (int n = 0; n < Ntotal; n++)
                        Copy(batchNorm + (long)n * _embDim, batchHidden + (long)n * _embDim, _embDim);
                }
                else
                {
                    var ffnNormW = GetNormWeight(_ffnNorm[layer]);
                    var ffnNormB = _hasNormBias && _bFfnNorm is not null ? _bFfnNorm[layer] : null;
                    if (_hp.UseParallelResidual)
                    {
                        for (int n = 0; n < Ntotal; n++)
                        {
                            Copy(batchHidden + (long)n * _embDim, batchNorm + (long)n * _embDim, _embDim);
                            FastNorm(batchNorm + (long)n * _embDim,
                                batchResidual + (long)n * _embDim, ffnNormW, ffnNormB, _embDim, _hp.RmsNormEps);
                        }
                    }
                    else
                    {
                        for (int n = 0; n < Ntotal; n++)
                            FastNorm(batchNorm + (long)n * _embDim,
                                batchHidden + (long)n * _embDim, ffnNormW, ffnNormB, _embDim, _hp.RmsNormEps);
                    }
                }

                // Batched FFN computation
                bool layerIsMoe = IsMoeLayer(layer);
                float* ffnOut = layerIsMoe ? batchMoeOut : batchNorm;
                if (layerIsMoe)
                {
                    MoeFfnBatched(layer, batchNorm, batchMoeOut, Ntotal);
                }
                else if (_wGate[layer].DataPtr is null)
                {
                    MatMulBatchedCached(batchFfnUp, in _wUp[layer], batchNorm, Ntotal, _intermDim, _embDim);
                    if (_xieluAlphaN is not null)
                    {
                        for (int n = 0; n < Ntotal; n++)
                            SimdKernels.XieluInPlace(batchFfnUp + (long)n * _intermDim, _intermDim,
                                _xieluAlphaN![layer], _xieluAlphaP![layer], _xieluBeta![layer], _xieluEps![layer]);
                        MatMulBatchedCached(batchNorm, in _wDown[layer], batchFfnUp, Ntotal, _embDim, _intermDim);
                    }
                    else if (_usesReluSquared)
                    {
                        for (int n = 0; n < Ntotal; n++)
                        {
                            float* up = batchFfnUp + (long)n * _intermDim;
                            if (_hasFfnBias && _bFfnUp is not null)
                                SimdKernels.AddInPlace(up, _bFfnUp[layer], _intermDim);
                            SimdKernels.ReluSqrInPlace(up, _intermDim);
                        }
                        MatMulBatchedCached(batchNorm, in _wDown[layer], batchFfnUp, Ntotal, _embDim, _intermDim);
                        if (_hasFfnBias && _bFfnDown is not null)
                        {
                            for (int n = 0; n < Ntotal; n++)
                                SimdKernels.AddInPlace(batchNorm + (long)n * _embDim, _bFfnDown[layer], _embDim);
                        }
                    }
                    else
                    {
                        for (int n = 0; n < Ntotal; n++)
                        {
                            float* up = batchFfnUp + (long)n * _intermDim;
                            if (_hasFfnBias && _bFfnUp is not null)
                                SimdKernels.AddInPlace(up, _bFfnUp[layer], _intermDim);
                            SimdKernels.GeluInPlace(up, _intermDim);
                        }
                        MatMulBatchedCached(batchNorm, in _wDown[layer], batchFfnUp, Ntotal, _embDim, _intermDim);
                        if (_hasFfnBias && _bFfnDown is not null)
                        {
                            for (int n = 0; n < Ntotal; n++)
                                SimdKernels.AddInPlace(batchNorm + (long)n * _embDim, _bFfnDown[layer], _embDim);
                        }
                    }
                }
                else
                {
                    MatMulBatchedDualCached(batchFfnGate, in _wGate[layer], batchFfnUp, in _wUp[layer], batchNorm, Ntotal, _intermDim, _embDim);

                    if (_hp.FfnActivation == FfnActivation.GeluApprox)
                    {
                        for (int n = 0; n < Ntotal; n++)
                        {
                            float* g = batchFfnGate + (long)n * _intermDim;
                            float* u = batchFfnUp + (long)n * _intermDim;
                            SimdKernels.GeluTanhMul(g, u, g, _intermDim);
                        }
                    }
                    else
                    {
                        for (int n = 0; n < Ntotal; n++)
                            SimdKernels.SiLuMul(batchFfnGate + (long)n * _intermDim,
                                batchFfnUp + (long)n * _intermDim, _intermDim);
                    }

                    MatMulBatchedCached(batchNorm, in _wDown[layer], batchFfnGate, Ntotal, _embDim, _intermDim);
                }

                if (_postFfwNorm is not null)
                {
                    var pfNormW = GetNormWeight(_postFfwNorm[layer]);
                    for (int n = 0; n < Ntotal; n++)
                        FastRmsNorm(ffnOut + (long)n * _embDim, ffnOut + (long)n * _embDim, pfNormW, _embDim, _hp.RmsNormEps);
                }

                // Residual add
                if (_hp.UseParallelResidual)
                {
                    for (int n = 0; n < Ntotal; n++)
                    {
                        float* h = batchHidden + (long)n * _embDim;
                        float* ffn = ffnOut + (long)n * _embDim;
                        float* r = batchResidual + (long)n * _embDim;
                        SimdKernels.AddInPlace(h, ffn, _embDim);
                        SimdKernels.AddInPlace(h, r, _embDim);
                        Copy(r, h, _embDim);
                    }
                }
                else
                {
                    for (int n = 0; n < Ntotal; n++)
                    {
                        float* h = batchHidden + (long)n * _embDim;
                        Copy(h, ffnOut + (long)n * _embDim, _embDim);
                        if (_hp.ResidualScale != 1f)
                            SimdKernels.ScaleInPlace(h, _hp.ResidualScale, _embDim);
                        SimdKernels.AddInPlace(h, batchResidual + (long)n * _embDim, _embDim);
                    }
                }

                if (_hp.HasPerLayerTokenEmbd)
                {
                    for (int n = 0; n < Ntotal; n++)
                        ApplyPerLayerEmbeddingBatched(layer, batchHidden + (long)n * _embDim, batchPleProj + (long)n * stackedPleDim + (long)layer * _pleWidth);
                }

                if (_layerOutputScale is not null)
                {
                    float scale = _layerOutputScale[layer];
                    for (int n = 0; n < Ntotal; n++)
                        SimdKernels.ScaleInPlace(batchHidden + (long)n * _embDim, scale, _embDim);
                }
            }

            // 3. Final norm for all tokens
            var outNormW = GetNormWeight(_outputNorm);
            var outNormB = _hasNormBias ? _bOutputNorm : null;

            void ApplyFinalNorm(float* dest, float* src)
            {
                if (_usesUnweightedNorm)
                    SimdKernels.PureLayerNorm(dest, src, _embDim, _hp.RmsNormEps);
                else
                    FastNorm(dest, src, outNormW, outNormB, _embDim, _hp.RmsNormEps);
            }

            for (int n = 0; n < Ntotal; n++)
            {
                float* hn = batchHidden + (long)n * _embDim;
                ApplyFinalNorm(hn, hn);
            }

            // 4. Scatter output to destination
            fixed (float* dstBase = destination)
            {
                for (int s = 0; s < M; s++)
                {
                    int sLen = seqLens[s];
                    if (sLen == 0) continue;
                    int sStart = seqStartTokenIdx[s];
                    int dstTokenOffset = destinationOffsets.IsEmpty ? sStart : destinationOffsets[s];
                    Copy(dstBase + (long)dstTokenOffset * _embDim, batchHidden + (long)sStart * _embDim, sLen * _embDim);
                }
            }
        }
        finally
        {
            NativeMemory.Free(batchHidden);
            NativeMemory.Free(batchResidual);
            NativeMemory.Free(batchNorm);
            NativeMemory.Free(batchQ);
            NativeMemory.Free(batchK);
            NativeMemory.Free(batchV);
            NativeMemory.Free(batchAttnOut);
            NativeMemory.Free(batchFfnGate);
            NativeMemory.Free(batchFfnUp);
            if (kStage != null) NativeMemory.Free(kStage);
            if (vStage != null) NativeMemory.Free(vStage);
            if (batchMoeOut != null) NativeMemory.Free(batchMoeOut);
            if (batchMlaAttnOutCompact != null) NativeMemory.Free(batchMlaAttnOutCompact);
            if (batchPleProj != null) NativeMemory.Free(batchPleProj);

            _kvCache.Reset();
        }
    }

    private static int CountPrecedingTokens(IReadOnlyList<IReadOnlyList<int>> sequences, int s)
    {
        int sum = 0;
        for (int i = 0; i < s; i++)
            sum += sequences[i].Count;
        return sum;
    }
}
