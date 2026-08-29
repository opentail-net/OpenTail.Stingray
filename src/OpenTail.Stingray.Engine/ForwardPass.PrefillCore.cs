
namespace OpenTail.Stingray.Engine;

// Part of ForwardPass (see ForwardPass.cs for the type summary). Batched multi-token prefill:
// PrefillCore (the dense/hybrid batched trunk), PrefillCoreTq (TurboQuant's sibling), hidden-tap
// capture for speculative decoding, and BatchVerify (speculative-decode verification).
public sealed unsafe partial class ForwardPass
{
    private ReadOnlySpan<float> PrefillCore(IReadOnlyList<int> tokens, PagedKvCache cache, int startPos,
        PositionLogitsCallback? onAllPositionLogits = null)
    {
        int N = tokens.Count;

        // SnapKV gating (issue #51): only run eviction when this is a fresh
        // prefill (startPos==0), the budget is configured, AND the prompt is
        // long enough that eviction would actually drop something. On short
        // prompts the scoring cost outweighs the savings.
        bool snapKvActive = _snapKvCfg.Enabled
                         && startPos == 0
                         && N > _snapKvCfg.Budget
                         && N > _snapKvCfg.Window;
        if (snapKvActive)
        {
            _snapKv ??= new SnapKvSelector(_numHeads, _numKvHeads, _headDim);
            _snapKv.Reset(N);
        }

        // Batch hidden states: [N, embDim]
        var batchHidden = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _embDim * sizeof(float)));
        var batchResidual = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _embDim * sizeof(float)));
        try
        {
            // 1. Embed all tokens
            for (int n = 0; n < N; n++)
                EmbedTokenInto(tokens[n], batchHidden + (long)n * _embDim, startPos + n);

            // Gemma 4 always takes the sequential Forward() path (perLayerHdUnsupported), so
            // this batched trunk previously never needed to apply EmbeddingScale — Granite and
            // MiniCPM are the first dense architectures to reach PrefillCore with it set.
            if (_hp.EmbeddingScale != 1f)
                for (int n = 0; n < N; n++)
                    SimdKernels.ScaleInPlace(batchHidden + (long)n * _embDim, _hp.EmbeddingScale, _embDim);

            // Temp buffers for batched operations
            int qDimMax = _numHeads * _maxHeadDim;
            int kvDimMax = _numKvHeads * _maxHeadDim;
            var batchNorm = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _embDim * sizeof(float)));
            var batchQ = (float*)NativeMemory.AllocZeroed((nuint)((long)N * qDimMax * sizeof(float)));
            var batchK = (float*)NativeMemory.AllocZeroed((nuint)((long)N * kvDimMax * sizeof(float)));
            var batchV = (float*)NativeMemory.AllocZeroed((nuint)((long)N * kvDimMax * sizeof(float)));
            var batchAttnOut = (float*)NativeMemory.AllocZeroed((nuint)((long)N * qDimMax * sizeof(float)));
            var batchFfnGate = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _intermDim * sizeof(float)));
            var batchFfnUp = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _intermDim * sizeof(float)));
            // MoE needs a separate FFN output buffer: the dense path writes the down projection
            // straight back over batchNorm, but every expert re-reads batchNorm, so they cannot
            // share. Only allocated for MoE — dense keeps the in-place buffer it always had.
            // Per-layer head dims only: one zeroed staging row each for K and V, widened from
            // the layer's compact head packing to the cache's _maxHeadDim head stride. Zeroed
            // once — the padding between heads is never re-dirtied, since every scatter writes
            // exactly the same head slots.
            var kStage = _layerHeadDim is not null
                ? (float*)NativeMemory.AllocZeroed((nuint)(kvDimMax * sizeof(float))) : null;
            var vStage = _layerHeadDim is not null
                ? (float*)NativeMemory.AllocZeroed((nuint)(kvDimMax * sizeof(float))) : null;
            bool batchedMoe = _hp.IsMoE;
            var batchMoeOut = batchedMoe
                ? (float*)NativeMemory.AllocZeroed((nuint)((long)N * _embDim * sizeof(float)))
                : null;
            // MLA: attention output before _wo needs numHeads*_mlaVDim per token, not the
            // zero-padded numHeads*_maxHeadDim batchAttnOut carries (see MlaCompactAttnOutBatched).
            var batchMlaAttnOutCompact = _isMla
                ? (float*)NativeMemory.AllocZeroed((nuint)((long)N * _numHeads * _mlaVDim * sizeof(float)))
                : null;

            try
            {
                bool profPrefill = PrefillProfileTimers.Enabled;
                if (profPrefill) PrefillProfileTimers.CountTokens(N);
                long pStage;

                // 2. Process layer-by-layer
                for (int layer = 0; layer < _hp.NumLayers; layer++)
                {
                    // Per-layer head dim (gemma4 issue #351): every shape below is derived from
                    // THIS layer's head dim, not a model-wide one. The batch buffers above are
                    // sized from _maxHeadDim, so a smaller layer simply packs its rows tighter;
                    // qDim/kvDim are the per-token STRIDE into those buffers for this layer only.
                    // Mirrors GpuForwardPass.RunGemma4Layers, which cuts per-layer views the same way.
                    int layerHd = _layerHeadDim?[layer] ?? _headDim;
                    // Quantity 2: the BUFFER / MATMUL shape, which is the weight's actual row
                    // count — _wq[layer] has _numHeads * layerHd rows and no more, so asking a
                    // narrow layer's projection for qDimMax rows reads past the tensor and faults
                    // inside SimdKernels.DotF32. PrefillCoreAttention independently derives this
                    // same compact value as its own qDim, so making the buffers compact is what
                    // makes producer and consumer agree.
                    //
                    // Quantity 1 — the CACHE's stride — is deliberately NOT this. It stays
                    // _maxHeadDim-wide, and K/V are widened to it at the single point that needs
                    // it, the Append below. Using the compact width there is what handed WriteKv a
                    // short span and threw at PagedKvCache.cs:455.
                    int qDim = _numHeads * layerHd;
                    int kvDim = _numKvHeads * layerHd;
                    long pLayerStart = profPrefill ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
                    long pNamedTicks = 0;

                    cache.TruncateTo(startPos);
                    var normW = GetNormWeight(_attnNorm[layer]);
                    var attnNormB = _hasNormBias && _bAttnNorm is not null ? _bAttnNorm[layer] : null;

                    // Batch norm (LayerNorm w/ bias for gptneox, RMSNorm otherwise) for all tokens
                    pStage = profPrefill ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
                    for (int n = 0; n < N; n++)
                    {
                        Copy(batchResidual + (long)n * _embDim, batchHidden + (long)n * _embDim, _embDim);
                        FastNorm(batchNorm + (long)n * _embDim,
                            batchHidden + (long)n * _embDim, normW, attnNormB, _embDim, _hp.RmsNormEps);
                    }
                    if (profPrefill)
                    {
                        long d = System.Diagnostics.Stopwatch.GetTimestamp() - pStage;
                        PrefillProfileTimers.Add(PrefillProfileTimers.Category.RmsNorm, d);
                        pNamedTicks += d;
                    }

                    pStage = profPrefill ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
                    if (_isMla)
                    {
                        // MLA (DeepSeek-V2/V3): compressed-latent K/V, not a direct wk/wv
                        // projection — see MlaComputeQkvBatched's doc comment.
                        MlaComputeQkvBatched(layer, batchNorm, N, batchQ, batchK, batchV);

                        if (s_mlaTrace)
                        {
                            double sAttnNorm = 0, sAttnNormAbs = 0, sQ = 0;
                            for (int n = 0; n < N; n++)
                            {
                                float* nb = batchNorm + (long)n * _embDim;
                                for (int d = 0; d < _embDim; d++) { sAttnNorm += nb[d]; sAttnNormAbs += Math.Abs((double)nb[d]); }
                                float* qb = batchQ + (long)n * qDim;
                                for (int d = 0; d < qDim; d++) sQ += qb[d];
                            }
                            Console.Error.WriteLine($"[MLA-TRACE] L{layer} attn_norm sum={sAttnNorm:F6} sumAbs={sAttnNormAbs:F6} q(pre-rope) sum={sQ:F6}");
                        }
                    }
                    else
                    {
                        // Batched Q/K/V projections (single GEMM per weight matrix)
                        MatMulBatchedCached(batchQ, in _wq[layer], batchNorm, N, qDim, _embDim);
                        MatMulBatchedCached(batchK, in _wk[layer], batchNorm, N, kvDim, _embDim);
                        MatMulBatchedCached(batchV, in _wv[layer], batchNorm, N, kvDim, _embDim);
                    }

                    // Apply QKV biases per token (Qwen/GPT-NeoX models)
                    if (_hasAttnBias)
                    {
                        for (int n = 0; n < N; n++)
                        {
                            SimdKernels.AddInPlace(batchQ + (long)n * qDim, _bq[layer], qDim);
                            SimdKernels.AddInPlace(batchK + (long)n * kvDim, _bk[layer], kvDim);
                            SimdKernels.AddInPlace(batchV + (long)n * kvDim, _bv[layer], kvDim);
                        }
                    }
                    if (profPrefill)
                    {
                        long d = System.Diagnostics.Stopwatch.GetTimestamp() - pStage;
                        PrefillProfileTimers.Add(PrefillProfileTimers.Category.QkvProj, d);
                        pNamedTicks += d;
                    }

                    // Per-head Q/K RMSNorm and RoPE — ordering and NoPE layers
                    bool useRoPE = _hp.NoRopeLayerStep == 0
                        || (layer + 1) % _hp.NoRopeLayerStep != 0;

                    long pRopeTicks = 0, pAttnTicks = 0;
                    pStage = profPrefill ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
                    for (int n = 0; n < N; n++)
                    {
                        float* qn = batchQ + (long)n * qDim;
                        float* kn = batchK + (long)n * kvDim;
                        float* vn = batchV + (long)n * kvDim;

                        // Qwen3 (weighted QK-norm): norm BEFORE RoPE
                        if (_hasQkNorm && !_hp.UseL2QkNorm && !_hp.QkNormAfterRope)
                        {
                            ApplyQkNorm(qn, kn, layer);
                        }

                        if (useRoPE)
                        {
                            ApplyRopeLayer(qn, startPos + n, _numHeads, layer, layerHd);
                            ApplyRopeLayer(kn, startPos + n, _numKvHeads, layer, layerHd);
                        }

                        // Hunyuan-Dense (weighted QK-norm): norm AFTER RoPE
                        if (_hasQkNorm && !_hp.UseL2QkNorm && _hp.QkNormAfterRope)
                        {
                            ApplyQkNorm(qn, kn, layer);
                        }

                        // L2 QK-norm (Llama-4): norm AFTER RoPE, only on RoPE layers
                        if (_hasQkNorm && _hp.UseL2QkNorm && useRoPE)
                        {
                            PerHeadPureRmsNorm(qn, _numHeads, layerHd, _hp.RmsNormEps);
                            PerHeadPureRmsNorm(kn, _numKvHeads, layerHd, _hp.RmsNormEps);
                        }

                        if (kStage is null)
                        {
                            cache.Append(layer,
                                new ReadOnlySpan<float>(kn, kvDim),
                                new ReadOnlySpan<float>(vn, kvDim));
                        }
                        else
                        {
                            // Quantity 1. A flat copy into a kvDimMax-long span would have the
                            // right LENGTH and therefore throw nothing, while placing every head
                            // but the first at the wrong offset — the cache strides heads by
                            // _maxHeadDim, so this is a per-head SCATTER, not a pad.
                            ScatterToCacheStride(kStage, kn, _numKvHeads, layerHd, _maxHeadDim);
                            ScatterToCacheStride(vStage!, vn, _numKvHeads, layerHd, _maxHeadDim);
                            cache.Append(layer,
                                new ReadOnlySpan<float>(kStage, kvDimMax),
                                new ReadOnlySpan<float>(vStage, kvDimMax));
                        }
                        cache.IncrementPosition();
                    }
                    if (profPrefill) pRopeTicks = System.Diagnostics.Stopwatch.GetTimestamp() - pStage;

                    if (s_mlaTrace && _isMla)
                    {
                        // This class's MLA layout is rope-FIRST per head ([rope(_ropeDim),
                        // nope]) -- see MlaComputeQkv's doc comment -- so the post-RoPE q_pe/
                        // k_pe equivalent is just the leading _ropeDim channels of each head.
                        double sQpe = 0, sKpe = 0;
                        for (int n = 0; n < N; n++)
                        {
                            float* qn = batchQ + (long)n * qDim;
                            float* kn = batchK + (long)n * kvDim;
                            for (int h = 0; h < _numHeads; h++)
                                for (int d = 0; d < _ropeDim; d++)
                                    sQpe += qn[h * _maxHeadDim + d];
                            for (int d = 0; d < _ropeDim; d++)
                                sKpe += kn[d]; // MQA: k_pe shared across heads, head 0 suffices
                        }
                        Console.Error.WriteLine($"[MLA-TRACE] L{layer} q_pe(post-rope) sum={sQpe:F6} k_pe(post-rope) sum={sKpe:F6}");
                    }

                    pStage = profPrefill ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
                    PrefillCoreAttention(batchQ, cache, layer, N, startPos, batchAttnOut);
                    if (profPrefill) pAttnTicks = System.Diagnostics.Stopwatch.GetTimestamp() - pStage;

                    if (profPrefill)
                    {
                        PrefillProfileTimers.Add(PrefillProfileTimers.Category.RoPE, pRopeTicks);
                        PrefillProfileTimers.Add(PrefillProfileTimers.Category.Attention, pAttnTicks);
                        pNamedTicks += pRopeTicks + pAttnTicks;
                    }

                    // SnapKV (issue #51): accumulate per-layer last-W query
                    // attention into the global score buffer. batchQ here is
                    // post-RoPE / post-Q-norm — the same vectors that just
                    // wrote scores against the K cache in the per-token loop
                    // above, so the scoring math is internally consistent.
                    if (snapKvActive)
                    {
                        _snapKv!.AccumulateLayer(batchQ, N, cache, layer, startPos,
                            _snapKvCfg.Window);
                    }

                    pStage = profPrefill ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
                    // Batched output projection
                    if (_isMla)
                    {
                        // batchAttnOut is zero-padded to _maxHeadDim per head (see
                        // MlaComputeQkvBatched); _wo's real GGUF shape expects
                        // numHeads*_mlaVDim, not numHeads*_maxHeadDim.
                        MlaCompactAttnOutBatched(batchAttnOut, batchMlaAttnOutCompact!, N);

                        if (s_mlaTrace)
                        {
                            double s = 0, sAbs = 0;
                            int compactDim = _numHeads * _mlaVDim;
                            for (int n = 0; n < N; n++)
                                for (int d = 0; d < compactDim; d++)
                                {
                                    double v = batchMlaAttnOutCompact![(long)n * compactDim + d];
                                    s += v;
                                    sAbs += Math.Abs(v);
                                }
                            float* t0 = batchMlaAttnOutCompact!;
                            Console.Error.WriteLine(
                                $"[MLA-TRACE] L{layer} kqv_out(compact) sum={s:F6} sumAbs={sAbs:F6} " +
                                $"tok0first3=[{t0[0]:F4},{t0[1]:F4},{t0[2]:F4}] " +
                                $"tok0last3=[{t0[compactDim - 3]:F4},{t0[compactDim - 2]:F4},{t0[compactDim - 1]:F4}]");
                        }

                        MatMulBatchedCached(batchNorm, in _wo[layer], batchMlaAttnOutCompact!, N, _embDim, _numHeads * _mlaVDim);
                    }
                    else
                    {
                        MatMulBatchedCached(batchNorm, in _wo[layer], batchAttnOut, N, _embDim, qDim);
                    }

                    // Apply output projection bias (Qwen models)
                    if (_hasAttnOutputBias)
                    {
                        for (int n = 0; n < N; n++)
                            SimdKernels.AddInPlace(batchNorm + (long)n * _embDim, _bo[layer], _embDim);
                    }
                    if (profPrefill)
                    {
                        long d = System.Diagnostics.Stopwatch.GetTimestamp() - pStage;
                        PrefillProfileTimers.Add(PrefillProfileTimers.Category.OutProj, d);
                        pNamedTicks += d;
                    }

                    if (!_hp.UseParallelResidual)
                    {
                    // Add output projection + residual → batchHidden
                    for (int n = 0; n < N; n++)
                    {
                        float* h = batchHidden + (long)n * _embDim;
                        float* proj = batchNorm + (long)n * _embDim;
                        float* r = batchResidual + (long)n * _embDim;
                        Copy(h, proj, _embDim);
                        // Granite/MiniCPM: scale the sublayer output before it joins the residual.
                        if (_hp.ResidualScale != 1f)
                            SimdKernels.ScaleInPlace(h, _hp.ResidualScale, _embDim);
                        SimdKernels.AddInPlace(h, r, _embDim);
                    }

                    // Save residual for FFN (post-attn residual, sequential graph only).
                    for (int n = 0; n < N; n++)
                        Copy(batchResidual + (long)n * _embDim, batchHidden + (long)n * _embDim, _embDim);
                    }
                    // else (parallel residual): batchNorm already holds attn_out (from the
                    // output projection above) and batchResidual still holds inpL, untouched
                    // since the attn-norm step at the top of this layer. Stashed into
                    // batchHidden and combined with the FFN output below.

                    // FFN norm: sequential reads batchHidden (attn_out + inpL); parallel (GPT-
                    // NeoX) reads batchResidual (inpL) directly — a SEPARATE LayerNorm from the
                    // attn one, both fed the same incoming residual. OLMo2 has no ffn_norm tensor
                    // at all (see _ffnNorm's fallback in the constructor) — FFN reads the raw
                    // residual unmodified, normed only by a POST-FFN norm applied to the
                    // sublayer's output below, before the residual add.
                    pStage = profPrefill ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
                    if (_ffnNorm[layer].DataPtr is null)
                    {
                        for (int n = 0; n < N; n++)
                            Copy(batchNorm + (long)n * _embDim, batchHidden + (long)n * _embDim, _embDim);
                    }
                    else
                    {
                        var ffnNormW = GetNormWeight(_ffnNorm[layer]);
                        var ffnNormB = _hasNormBias && _bFfnNorm is not null ? _bFfnNorm[layer] : null;
                        if (_hp.UseParallelResidual)
                        {
                            for (int n = 0; n < N; n++)
                            {
                                Copy(batchHidden + (long)n * _embDim, batchNorm + (long)n * _embDim, _embDim);
                                FastNorm(batchNorm + (long)n * _embDim,
                                    batchResidual + (long)n * _embDim, ffnNormW, ffnNormB, _embDim, _hp.RmsNormEps);
                            }
                        }
                        else
                        {
                            for (int n = 0; n < N; n++)
                                FastNorm(batchNorm + (long)n * _embDim,
                                    batchHidden + (long)n * _embDim, ffnNormW, ffnNormB, _embDim, _hp.RmsNormEps);
                        }
                    }
                    if (profPrefill)
                    {
                        long d = System.Diagnostics.Stopwatch.GetTimestamp() - pStage;
                        PrefillProfileTimers.Add(PrefillProfileTimers.Category.RmsNorm, d);
                        pNamedTicks += d;
                        pStage = System.Diagnostics.Stopwatch.GetTimestamp();
                    }

                    // Where this layer's FFN output lands: dense reuses batchNorm in place,
                    // MoE needs its own buffer (see batchMoeOut's declaration). Per-LAYER check
                    // (not the model-level batchedMoe) so DeepSeek-V2's leading dense block(s)
                    // correctly take the dense path even though the model overall IsMoE.
                    bool layerIsMoe = IsMoeLayer(layer);
                    float* ffnOut = layerIsMoe ? batchMoeOut : batchNorm;
                    if (layerIsMoe)
                    {
                        MoeFfnBatched(layer, batchNorm, batchMoeOut, N);
                    }
                    else if (_wGate[layer].DataPtr is null)
                    {
                        // Apertus/GPT-NeoX: no ffn_gate tensor — plain up -> activation -> down.
                        MatMulBatchedCached(batchFfnUp, in _wUp[layer], batchNorm, N, _intermDim, _embDim);
                        if (_xieluAlphaN is not null)
                        {
                            for (int n = 0; n < N; n++)
                                SimdKernels.XieluInPlace(batchFfnUp + (long)n * _intermDim, _intermDim,
                                    _xieluAlphaN![layer], _xieluAlphaP![layer], _xieluBeta![layer], _xieluEps![layer]);
                            MatMulBatchedCached(batchNorm, in _wDown[layer], batchFfnUp, N, _embDim, _intermDim);
                        }
                        else if (_usesReluSquared)
                        {
                            // JAIS-2: biased ReLU-squared — same bias placement as GELU below.
                            for (int n = 0; n < N; n++)
                            {
                                float* up = batchFfnUp + (long)n * _intermDim;
                                if (_hasFfnBias && _bFfnUp is not null)
                                    SimdKernels.AddInPlace(up, _bFfnUp[layer], _intermDim);
                                SimdKernels.ReluSqrInPlace(up, _intermDim);
                            }
                            MatMulBatchedCached(batchNorm, in _wDown[layer], batchFfnUp, N, _embDim, _intermDim);
                            if (_hasFfnBias && _bFfnDown is not null)
                            {
                                for (int n = 0; n < N; n++)
                                    SimdKernels.AddInPlace(batchNorm + (long)n * _embDim, _bFfnDown[layer], _embDim);
                            }
                        }
                        else
                        {
                            // GPT-NeoX: biased GELU — up bias goes INSIDE the activation
                            // (gelu(Wx + b), not gelu(Wx) + b), down bias after.
                            for (int n = 0; n < N; n++)
                            {
                                float* up = batchFfnUp + (long)n * _intermDim;
                                if (_hasFfnBias && _bFfnUp is not null)
                                    SimdKernels.AddInPlace(up, _bFfnUp[layer], _intermDim);
                                SimdKernels.GeluInPlace(up, _intermDim);
                            }
                            MatMulBatchedCached(batchNorm, in _wDown[layer], batchFfnUp, N, _embDim, _intermDim);
                            if (_hasFfnBias && _bFfnDown is not null)
                            {
                                for (int n = 0; n < N; n++)
                                    SimdKernels.AddInPlace(batchNorm + (long)n * _embDim, _bFfnDown[layer], _embDim);
                            }
                        }
                    }
                    else
                    {
                        MatMulBatchedDualCached(batchFfnGate, in _wGate[layer], batchFfnUp, in _wUp[layer], batchNorm, N, _intermDim, _embDim);

                        // Per-token SiLU(gate) * up
                        for (int n = 0; n < N; n++)
                            SimdKernels.SiLuMul(batchFfnGate + (long)n * _intermDim,
                                batchFfnUp + (long)n * _intermDim, _intermDim);

                        MatMulBatchedCached(batchNorm, in _wDown[layer], batchFfnGate, N, _embDim, _intermDim);
                    }
                    if (profPrefill)
                    {
                        long d = System.Diagnostics.Stopwatch.GetTimestamp() - pStage;
                        PrefillProfileTimers.Add(PrefillProfileTimers.Category.Ffn, d);
                        pNamedTicks += d;
                    }

                    // Residual add
                    if (_hp.UseParallelResidual)
                    {
                        // 3-way sum: batchHidden(attn_out) + ffnOut(ffn_out) + batchResidual(inpL).
                        for (int n = 0; n < N; n++)
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
                        for (int n = 0; n < N; n++)
                        {
                            float* h = batchHidden + (long)n * _embDim;
                            Copy(h, ffnOut + (long)n * _embDim, _embDim);
                            // Granite/MiniCPM: scale the sublayer output before it joins the residual.
                            if (_hp.ResidualScale != 1f)
                                SimdKernels.ScaleInPlace(h, _hp.ResidualScale, _embDim);
                            SimdKernels.AddInPlace(h, batchResidual + (long)n * _embDim, _embDim);
                        }
                    }

                    if (s_mlaTrace)
                    {
                        double s = 0, sAbs = 0;
                        for (int n = 0; n < N; n++)
                            for (int d = 0; d < _embDim; d++)
                            {
                                double v = batchHidden[(long)n * _embDim + d];
                                s += v; sAbs += Math.Abs(v);
                            }
                        Console.Error.WriteLine($"[MLA-TRACE] L{layer} l_out sum={s:F6} sumAbs={sAbs:F6} " +
                            $"tok0first3=[{batchHidden[0]:F4},{batchHidden[1]:F4},{batchHidden[2]:F4}] " +
                            $"tok0last3=[{batchHidden[_embDim - 3]:F4},{batchHidden[_embDim - 2]:F4},{batchHidden[_embDim - 1]:F4}]");
                    }

                    // Hidden-state taps: batchHidden rows are this layer's outputs.
                    if (_taps is { } taps && taps.SlotOf(layer) is int tapSlot && tapSlot >= 0)
                        for (int n = 0; n < N; n++)
                            CaptureTap(startPos + n, tapSlot, batchHidden + (long)n * _embDim);

                    if (profPrefill)
                    {
                        long layerTotal = System.Diagnostics.Stopwatch.GetTimestamp() - pLayerStart;
                        PrefillProfileTimers.Add(PrefillProfileTimers.Category.Other, Math.Max(0, layerTotal - pNamedTicks));
                    }
                }

                // Set KV cache length to startPos + N for subsequent decode calls.
                cache.TruncateTo(startPos + N);

                // SnapKV (issue #51): compact the cache to the selected keep
                // set. Runs once per prefill — the per-token decode path is
                // untouched and pays no extra cost. After compaction
                // cache.Length is the kept-slot count and cache.LogicalLength
                // is the original prompt length, so decode RoPE continues from
                // the right reference frame.
                if (snapKvActive)
                {
                    var keep = _snapKv!.SelectKeepSet(N, _snapKvCfg.Budget, _snapKvCfg.Recency);
                    if (keep.Length < N)
                    {
                        cache.Compact(keep);
                    }
                }
            }
            finally
            {
                NativeMemory.Free(batchNorm);
                NativeMemory.Free(batchQ);
                NativeMemory.Free(batchK);
                NativeMemory.Free(batchV);
                NativeMemory.Free(batchAttnOut);
                NativeMemory.Free(batchFfnGate);
                NativeMemory.Free(batchFfnUp);
                if (batchMoeOut != null) NativeMemory.Free(batchMoeOut);
                if (batchMlaAttnOutCompact != null) NativeMemory.Free(batchMlaAttnOutCompact);
                if (kStage != null) NativeMemory.Free(kStage);
                if (vStage != null) NativeMemory.Free(vStage);
            }

            // 3. Final norm + output projection. Normally last token only; when
            // onAllPositionLogits is set (diagnostic use, see Prefill's doc comment) every
            // position is projected instead, reusing the same _logits buffer per position
            // (the callback must consume it before the next iteration overwrites it) so this
            // stays a streaming O(vocab) buffer rather than an O(N*vocab) allocation.
            var outNormW = GetNormWeight(_outputNorm);

            var outNormB = _hasNormBias ? _bOutputNorm : null;

            if (onAllPositionLogits != null)
            {
                for (int n = 0; n < N; n++)
                {
                    float* hn = batchHidden + (long)n * _embDim;
                    FastNorm(hn, hn, outNormW, outNormB, _embDim, _hp.RmsNormEps);
                    FusedMatVec(_logits, _outputWeight, hn, _hp.VocabSize, _embDim);
                    onAllPositionLogits(n, new ReadOnlySpan<float>(_logits, _hp.VocabSize));
                }
                return new ReadOnlySpan<float>(_logits, _hp.VocabSize);
            }

            float* lastHidden = batchHidden + (long)(N - 1) * _embDim;
            FastNorm(lastHidden, lastHidden, outNormW, outNormB, _embDim, _hp.RmsNormEps);
            FusedMatVec(_logits, _outputWeight, lastHidden, _hp.VocabSize, _embDim);

            return new ReadOnlySpan<float>(_logits, _hp.VocabSize);
        }
        finally
        {
            NativeMemory.Free(batchHidden);
            NativeMemory.Free(batchResidual);
        }
    }

    /// <summary>
    /// TurboQuant variant of <see cref="PrefillCore"/>: identical batched matmul
    /// structure (QKV, attn output, FFN gate/up/down all run as one GEMM per
    /// weight matrix across N tokens), but routes K/V into the TQ cache and
    /// uses TqAttention per token. Between layers the global TQ position
    /// counter snaps back to <paramref name="startPos"/> while per-layer
    /// FastScan tile/staging/FP32 state stays intact — each layer's TQ window
    /// evolves independently as the N tokens stream through it.
    ///
    /// <para>Unlike <see cref="PrefillCore"/>, this deliberately does not opt into
    /// <see cref="SimdKernels.MatMulBatched"/>'s int8 path (it leaves <c>allowQ8</c> at its
    /// default of <c>false</c>). TurboQuant already trades accuracy for KV footprint, and
    /// stacking int8 activation quantization on top of it is a separate quality question that no
    /// perplexity/greedy-parity measurement covers yet. A deliberate scope boundary, not an
    /// oversight — TQ prefill forgoes the ~+47% int8 win until that is measured.</para>
    /// </summary>
    private ReadOnlySpan<float> PrefillCoreTq(IReadOnlyList<int> tokens, int startPos)
    {
        var cache = _tqKvCache!;
        int N = tokens.Count;

        // SnapKV (issue #60) gating: fresh prefill, explicit budget, prompt
        // long enough to drop something. TQ is fine to compose with — the
        // selector reads from the same cache the per-token TqAttention writes
        // and the compaction promotes the oldest FP32-window survivors into
        // the TQ region as needed.
        bool snapKvActive = _snapKvCfg.Enabled
                         && startPos == 0
                         && N > _snapKvCfg.Budget
                         && N > _snapKvCfg.Window;
        if (snapKvActive)
        {
            _snapKv ??= new SnapKvSelector(_numHeads, _numKvHeads, _headDim);
            _snapKv.Reset(N);
        }

        var batchHidden = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _embDim * sizeof(float)));
        var batchResidual = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _embDim * sizeof(float)));
        try
        {
            for (int n = 0; n < N; n++)
                EmbedTokenInto(tokens[n], batchHidden + (long)n * _embDim, startPos + n);

            int qDimMax = _numHeads * _maxHeadDim;
            int kvDimMax = _numKvHeads * _maxHeadDim;
            var batchNorm = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _embDim * sizeof(float)));
            var batchQ = (float*)NativeMemory.AllocZeroed((nuint)((long)N * qDimMax * sizeof(float)));
            var batchK = (float*)NativeMemory.AllocZeroed((nuint)((long)N * kvDimMax * sizeof(float)));
            var batchV = (float*)NativeMemory.AllocZeroed((nuint)((long)N * kvDimMax * sizeof(float)));
            var batchAttnOut = (float*)NativeMemory.AllocZeroed((nuint)((long)N * qDimMax * sizeof(float)));
            var batchFfnGate = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _intermDim * sizeof(float)));
            var batchFfnUp = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _intermDim * sizeof(float)));

            try
            {
                for (int layer = 0; layer < _hp.NumLayers; layer++)
                {
                    // Per-layer head dim (gemma4 issue #351): every shape below is derived from
                    // THIS layer's head dim, not a model-wide one. The batch buffers above are
                    // sized from _maxHeadDim, so a smaller layer simply packs its rows tighter;
                    // qDim/kvDim are the per-token STRIDE into those buffers for this layer only.
                    // Mirrors GpuForwardPass.RunGemma4Layers, which cuts per-layer views the same way.
                    int layerHd = _layerHeadDim?[layer] ?? _headDim;
                    // STRIDES STAY UNIFORM. PagedKvCache holds one model-wide _kvDim and copies
                    // exactly that many floats in WriteKv, so a per-layer stride hands it a short
                    // span and throws (PagedKvCache.cs:455). layerHd belongs in the ARITHMETIC —
                    // norms, RoPE, attention head dim — never in the addressing. Narrow layers
                    // simply leave the row tail unused; attention only ever reads h*layerHd for
                    // h < the layer's head count, so the tail is written to cache but never read.
                    int qDim = qDimMax;
                    int kvDim = kvDimMax;
                    // Snap the shared global position counter back to startPos
                    // for this layer's per-token loop. Per-layer TQ tile + FP32
                    // window state from prior layers is untouched.
                    cache.ResetTotalLengthForBatchedPrefill(startPos);
                    var normW = GetNormWeight(_attnNorm[layer]);

                    for (int n = 0; n < N; n++)
                    {
                        Copy(batchResidual + (long)n * _embDim, batchHidden + (long)n * _embDim, _embDim);
                        SimdKernels.RmsNorm(batchNorm + (long)n * _embDim,
                            batchHidden + (long)n * _embDim, normW, _embDim, _hp.RmsNormEps);
                    }

                    SimdKernels.MatMulBatched(batchQ, _wq[layer].DataPtr, batchNorm,
                        N, qDim, _embDim, _wq[layer].DType);
                    SimdKernels.MatMulBatched(batchK, _wk[layer].DataPtr, batchNorm,
                        N, kvDim, _embDim, _wk[layer].DType);
                    SimdKernels.MatMulBatched(batchV, _wv[layer].DataPtr, batchNorm,
                        N, kvDim, _embDim, _wv[layer].DType);

                    if (_hasAttnBias)
                    {
                        for (int n = 0; n < N; n++)
                        {
                            SimdKernels.AddInPlace(batchQ + (long)n * qDim, _bq[layer], qDim);
                            SimdKernels.AddInPlace(batchK + (long)n * kvDim, _bk[layer], kvDim);
                            SimdKernels.AddInPlace(batchV + (long)n * kvDim, _bv[layer], kvDim);
                        }
                    }

                    bool useRoPE = _hp.NoRopeLayerStep == 0
                        || (layer + 1) % _hp.NoRopeLayerStep != 0;

                    for (int n = 0; n < N; n++)
                    {
                        float* qn = batchQ + (long)n * qDim;
                        float* kn = batchK + (long)n * kvDim;
                        float* vn = batchV + (long)n * kvDim;

                        if (_hasQkNorm && !_hp.UseL2QkNorm)
                            ApplyQkNorm(qn, kn, layer);

                        if (useRoPE)
                        {
                            ApplyRopeLayer(qn, startPos + n, _numHeads, layer, layerHd);
                            ApplyRopeLayer(kn, startPos + n, _numKvHeads, layer, layerHd);
                        }

                        if (_hasQkNorm && _hp.UseL2QkNorm && useRoPE)
                        {
                            PerHeadPureRmsNorm(qn, _numHeads, layerHd, _hp.RmsNormEps);
                            PerHeadPureRmsNorm(kn, _numKvHeads, layerHd, _hp.RmsNormEps);
                        }

                        cache.Append(layer,
                            new ReadOnlySpan<float>(kn, kvDim),
                            new ReadOnlySpan<float>(vn, kvDim));
                        cache.IncrementPosition();

                        Copy(_q, qn, qDim);
                        TqAttention(layer, startPos + n);

                        Copy(batchAttnOut + (long)n * qDim, _attnOut, qDim);
                    }

                    // SnapKV (issue #60): same shape as PrefillCore's call but
                    // against the TQ cache. batchQ is post-RoPE / post-Q-norm —
                    // the same vectors TqAttention just used to write scores
                    // against the TQ-compressed + FP32-ring K state.
                    if (snapKvActive)
                    {
                        _snapKv!.AccumulateLayer(batchQ, N, cache, layer, startPos,
                            _snapKvCfg.Window);
                    }

                    SimdKernels.MatMulBatched(batchNorm, _wo[layer].DataPtr, batchAttnOut,
                        N, _embDim, qDim, _wo[layer].DType);

                    if (_hasAttnOutputBias)
                    {
                        for (int n = 0; n < N; n++)
                            SimdKernels.AddInPlace(batchNorm + (long)n * _embDim, _bo[layer], _embDim);
                    }

                    for (int n = 0; n < N; n++)
                    {
                        float* h = batchHidden + (long)n * _embDim;
                        float* proj = batchNorm + (long)n * _embDim;
                        float* r = batchResidual + (long)n * _embDim;
                        Copy(h, proj, _embDim);
                        SimdKernels.AddInPlace(h, r, _embDim);
                    }

                    var ffnNormW = GetNormWeight(_ffnNorm[layer]);
                    for (int n = 0; n < N; n++)
                    {
                        Copy(batchResidual + (long)n * _embDim, batchHidden + (long)n * _embDim, _embDim);
                        SimdKernels.RmsNorm(batchNorm + (long)n * _embDim,
                            batchHidden + (long)n * _embDim, ffnNormW, _embDim, _hp.RmsNormEps);
                    }

                    SimdKernels.MatMulBatched(batchFfnGate, _wGate[layer].DataPtr, batchNorm,
                        N, _intermDim, _embDim, _wGate[layer].DType);
                    SimdKernels.MatMulBatched(batchFfnUp, _wUp[layer].DataPtr, batchNorm,
                        N, _intermDim, _embDim, _wUp[layer].DType);

                    for (int n = 0; n < N; n++)
                        SimdKernels.SiLuMul(batchFfnGate + (long)n * _intermDim,
                            batchFfnUp + (long)n * _intermDim, _intermDim);

                    SimdKernels.MatMulBatched(batchNorm, _wDown[layer].DataPtr, batchFfnGate,
                        N, _embDim, _intermDim, _wDown[layer].DType);

                    for (int n = 0; n < N; n++)
                    {
                        float* h = batchHidden + (long)n * _embDim;
                        Copy(h, batchNorm + (long)n * _embDim, _embDim);
                        SimdKernels.AddInPlace(h, batchResidual + (long)n * _embDim, _embDim);
                    }
                }

                // _totalLength was advanced to startPos + N by the last layer's
                // per-token loop, which is the state subsequent decode calls expect.

                // SnapKV (issue #60): compact the TQ cache to the selected keep
                // set. Runs once per prefill — per-token decode is untouched.
                // After compaction Length is the kept-slot count; decode RoPE
                // for the next token continues from `startPos + N` (the caller's
                // position counter is unchanged), which is the right post-eviction
                // reference frame because RoPE depends on absolute position not
                // on cache slot index.
                if (snapKvActive)
                {
                    var keep = _snapKv!.SelectKeepSet(N, _snapKvCfg.Budget, _snapKvCfg.Recency);
                    if (keep.Length < N)
                    {
                        cache.Compact(keep, N);
                    }
                }
            }
            finally
            {
                NativeMemory.Free(batchNorm);
                NativeMemory.Free(batchQ);
                NativeMemory.Free(batchK);
                NativeMemory.Free(batchV);
                NativeMemory.Free(batchAttnOut);
                NativeMemory.Free(batchFfnGate);
                NativeMemory.Free(batchFfnUp);
            }

            float* lastHidden = batchHidden + (long)(N - 1) * _embDim;
            var outNormW = GetNormWeight(_outputNorm);
            SimdKernels.RmsNorm(lastHidden, lastHidden, outNormW, _embDim, _hp.RmsNormEps);
            FusedMatVec(_logits, _outputWeight, lastHidden, _hp.VocabSize, _embDim);

            return new ReadOnlySpan<float>(_logits, _hp.VocabSize);
        }
        finally
        {
            NativeMemory.Free(batchHidden);
            NativeMemory.Free(batchResidual);
        }
    }

    /// <summary>
    /// Whether <see cref="BatchVerify"/> can run (issue #207): everything except the two
    /// configurations it throws for — the TurboQuant KV cache (compressed ring can't take
    /// the batched appends) and gemma4-style per-layer head_dim (not wired into the batched
    /// trunk) — and a SnapKV-compacted cache. After <c>Compact</c> the physical slot count
    /// (<see cref="PagedKvCache.Length"/>) sits below the logical RoPE position
    /// (<see cref="PagedKvCache.LogicalLength"/>), but <see cref="BatchVerify"/> appends at
    /// the LOGICAL position via <c>TruncateTo(startPos)</c>, which would declare slots
    /// past the compacted length valid and read garbage K/V — same #130 gate the CUDA and
    /// GDN passes already have; the sequential <see cref="Forward"/> fallback handles the
    /// compacted frame correctly. MoE stays <c>true</c>: <see cref="BatchVerify"/> itself
    /// falls back to sequential <see cref="Forward"/> calls for MoE, which is still correct.
    /// </summary>
    // ── Hidden-state taps (DSpark draft conditioning, PR #413 spec) ──

    /// <summary>
    /// Taps require stable absolute positions, which SnapKV compaction breaks,
    /// and are captured on the standard dense paths only (no TurboQuant KV).
    /// </summary>
    public bool SupportsHiddenTaps => !_snapKvCfg.Enabled && _tqKvCache is null;

    public int HiddenTapDim => _taps?.TapDim ?? 0;

    public void EnableHiddenTaps(ReadOnlySpan<int> layerIds)
    {
        if (!SupportsHiddenTaps)
            throw new NotSupportedException(
                "Hidden-state taps are not supported with SnapKV eviction or a TurboQuant KV cache " +
                "(both break the absolute-position indexing taps rely on).");
        // Gemma-family post-layer transforms (post-FFW norm, PLE injection, per-layer
        // output scale) run only on the sequential RunTrunk path; the batched
        // Prefill/BatchVerify cores capture at the plain FFN-residual point. Until the
        // batched cores mirror those transforms, taps on such models would record
        // different values per path — reject rather than desync silently. (Gemma 4
        // per-layer head_dim models already route every batched call to sequential
        // Forward, but the guard keeps the contract explicit.)
        // The exception is a model whose batched calls ALL route to sequential Forward anyway —
        // per-layer head_dim, i.e. Gemma 4. There is then no second capture point to disagree
        // with, which is the only thing this guard exists to prevent, so rejecting it buys
        // nothing and costs the ability to diff Gemma 4 layer-by-layer against another backend.
        if ((_postFfwNorm is not null || _layerOutputScale is not null || _hp.HasPerLayerTokenEmbd)
            && _layerHeadDim is null)
            throw new NotSupportedException(
                "Hidden-state taps are not supported on models with post-FFW norm / per-layer " +
                "output scale / PLE (capture points differ between sequential and batched paths).");

        _taps?.Dispose();
        _taps = new HiddenTapBuffer(layerIds, _hp.NumLayers, _embDim, _hp.ContextLength);
    }

    public ReadOnlySpan<float> HiddenTapsAt(int position) =>
        _taps is { } tb ? tb.At(position) : default;

    /// <summary>Copy one tapped layer output (embDim floats) into position/slot.</summary>
    private void CaptureTap(int position, int slot, float* layerOutput)
    {
        new ReadOnlySpan<float>(layerOutput, _embDim).CopyTo(_taps!.RowSlot(position, slot));
    }

    public bool SupportsBatchVerify =>
        _tqKvCache is null
        && _layerHeadDim is null
        && _kvCache.Length == _kvCache.LogicalLength;

    /// <summary>
    /// Batched verification for speculative decoding: processes <paramref name="tokens"/> starting
    /// at <paramref name="startPos"/> using the existing KV cache as context.
    /// All K/V entries are appended to the cache; caller must call TruncateTo to rewind on rejection.
    /// Returns <c>result[i]</c> = logits after processing <c>tokens[i]</c>.
    /// </summary>
    /// <exception cref="NotSupportedException">If TurboQuant KV cache is enabled.</exception>
    public float[][] BatchVerify(int[] tokens, int startPos)
    {
        if (_tqKvCache != null)
            throw new NotSupportedException("BatchVerify is not supported when TurboQuant KV cache is enabled.");
        if (_layerHeadDim is not null)
            throw new NotSupportedException(
                "gemma4 per-layer head_dim not yet supported on the batched BatchVerify path.");

        int N = tokens.Length;
        if (N == 0) return Array.Empty<float[]>();

        if (N == 1 || _hp.IsMoE)
        {
            // Single token or MoE: fall back to sequential Forward calls
            var seq = new float[N][];
            for (int i = 0; i < N; i++)
            {
                var logits = Forward(tokens[i], startPos + i);
                seq[i] = new float[_hp.VocabSize];
                logits.CopyTo(seq[i]);
            }
            return seq;
        }

        var batchHidden = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _embDim * sizeof(float)));
        var batchResidual = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _embDim * sizeof(float)));
        try
        {
            // 1. Embed all tokens
            for (int n = 0; n < N; n++)
                EmbedTokenInto(tokens[n], batchHidden + (long)n * _embDim, startPos + n);

            int qDim = _numHeads * _headDim;
            int kvDim = _numKvHeads * _headDim;
            var batchNorm = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _embDim * sizeof(float)));
            var batchQ = (float*)NativeMemory.AllocZeroed((nuint)((long)N * qDim * sizeof(float)));
            var batchK = (float*)NativeMemory.AllocZeroed((nuint)((long)N * kvDim * sizeof(float)));
            var batchV = (float*)NativeMemory.AllocZeroed((nuint)((long)N * kvDim * sizeof(float)));
            var batchAttnOut = (float*)NativeMemory.AllocZeroed((nuint)((long)N * qDim * sizeof(float)));
            var batchFfnGate = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _intermDim * sizeof(float)));
            var batchFfnUp = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _intermDim * sizeof(float)));

            try
            {
                // 2. Process layer-by-layer (same batch structure as Prefill, starting at startPos)
                for (int layer = 0; layer < _hp.NumLayers; layer++)
                {
                    // Restore cache length to startPos so K/V appends land at the right positions
                    _kvCache.TruncateTo(startPos);

                    var normW = GetNormWeight(_attnNorm[layer]);
                    for (int n = 0; n < N; n++)
                    {
                        Copy(batchResidual + (long)n * _embDim, batchHidden + (long)n * _embDim, _embDim);
                        SimdKernels.RmsNorm(batchNorm + (long)n * _embDim,
                            batchHidden + (long)n * _embDim, normW, _embDim, _hp.RmsNormEps);
                    }

                    SimdKernels.MatMulBatched(batchQ, _wq[layer].DataPtr, batchNorm,
                        N, qDim, _embDim, _wq[layer].DType);
                    SimdKernels.MatMulBatched(batchK, _wk[layer].DataPtr, batchNorm,
                        N, kvDim, _embDim, _wk[layer].DType);
                    SimdKernels.MatMulBatched(batchV, _wv[layer].DataPtr, batchNorm,
                        N, kvDim, _embDim, _wv[layer].DType);

                    if (_hasAttnBias)
                    {
                        for (int n = 0; n < N; n++)
                        {
                            SimdKernels.AddInPlace(batchQ + (long)n * qDim, _bq[layer], qDim);
                            SimdKernels.AddInPlace(batchK + (long)n * kvDim, _bk[layer], kvDim);
                            SimdKernels.AddInPlace(batchV + (long)n * kvDim, _bv[layer], kvDim);
                        }
                    }

                    bool useRoPE = _hp.NoRopeLayerStep == 0
                        || (layer + 1) % _hp.NoRopeLayerStep != 0;

                    // Sequential: RoPE (at startPos+n), K/V append, causal attention
                    for (int n = 0; n < N; n++)
                    {
                        float* qn = batchQ + (long)n * qDim;
                        float* kn = batchK + (long)n * kvDim;
                        float* vn = batchV + (long)n * kvDim;

                        int pos = startPos + n;

                        // Qwen3 (weighted QK-norm): norm BEFORE RoPE
                        if (_hasQkNorm && !_hp.UseL2QkNorm)
                        {
                            ApplyQkNorm(qn, kn, layer);
                        }

                        if (useRoPE)
                        {
                            ApplyRope(qn, pos, _numHeads);
                            ApplyRope(kn, pos, _numKvHeads);
                        }

                        // L2 QK-norm (Llama-4): norm AFTER RoPE, only on RoPE layers
                        if (_hasQkNorm && _hp.UseL2QkNorm && useRoPE)
                        {
                            PerHeadPureRmsNorm(qn, _numHeads, _headDim, _hp.RmsNormEps);
                            PerHeadPureRmsNorm(kn, _numKvHeads, _headDim, _hp.RmsNormEps);
                        }

                        _kvCache.Append(layer,
                            new ReadOnlySpan<float>(kn, kvDim),
                            new ReadOnlySpan<float>(vn, kvDim));
                        _kvCache.IncrementPosition();  // _length = startPos + n + 1

                        Copy(_q, qn, qDim);
                        Attention(_kvCache, layer, pos);  // seqLen = startPos + n + 1, uses K/V for 0..pos

                        Copy(batchAttnOut + (long)n * qDim, _attnOut, qDim);
                    }

                    SimdKernels.MatMulBatched(batchNorm, _wo[layer].DataPtr, batchAttnOut,
                        N, _embDim, qDim, _wo[layer].DType);

                    if (_hasAttnOutputBias)
                    {
                        for (int n = 0; n < N; n++)
                            SimdKernels.AddInPlace(batchNorm + (long)n * _embDim, _bo[layer], _embDim);
                    }

                    for (int n = 0; n < N; n++)
                    {
                        float* h = batchHidden + (long)n * _embDim;
                        float* proj = batchNorm + (long)n * _embDim;
                        float* r = batchResidual + (long)n * _embDim;
                        Copy(h, proj, _embDim);
                        SimdKernels.AddInPlace(h, r, _embDim);
                    }

                    var ffnNormW = GetNormWeight(_ffnNorm[layer]);
                    for (int n = 0; n < N; n++)
                    {
                        Copy(batchResidual + (long)n * _embDim, batchHidden + (long)n * _embDim, _embDim);
                        SimdKernels.RmsNorm(batchNorm + (long)n * _embDim,
                            batchHidden + (long)n * _embDim, ffnNormW, _embDim, _hp.RmsNormEps);
                    }

                    SimdKernels.MatMulBatched(batchFfnGate, _wGate[layer].DataPtr, batchNorm,
                        N, _intermDim, _embDim, _wGate[layer].DType);
                    SimdKernels.MatMulBatched(batchFfnUp, _wUp[layer].DataPtr, batchNorm,
                        N, _intermDim, _embDim, _wUp[layer].DType);

                    for (int n = 0; n < N; n++)
                        SimdKernels.SiLuMul(batchFfnGate + (long)n * _intermDim,
                            batchFfnUp + (long)n * _intermDim, _intermDim);

                    SimdKernels.MatMulBatched(batchNorm, _wDown[layer].DataPtr, batchFfnGate,
                        N, _embDim, _intermDim, _wDown[layer].DType);

                    for (int n = 0; n < N; n++)
                    {
                        float* h = batchHidden + (long)n * _embDim;
                        Copy(h, batchNorm + (long)n * _embDim, _embDim);
                        SimdKernels.AddInPlace(h, batchResidual + (long)n * _embDim, _embDim);
                    }

                    // Hidden-state taps: batchHidden rows are this layer's outputs.
                    if (_taps is { } taps && taps.SlotOf(layer) is int tapSlot && tapSlot >= 0)
                        for (int n = 0; n < N; n++)
                            CaptureTap(startPos + n, tapSlot, batchHidden + (long)n * _embDim);
                }

                // Ensure cache length is startPos + N
                _kvCache.TruncateTo(startPos);
                for (int i = 0; i < N; i++) _kvCache.IncrementPosition();
            }
            finally
            {
                NativeMemory.Free(batchNorm);
                NativeMemory.Free(batchQ);
                NativeMemory.Free(batchK);
                NativeMemory.Free(batchV);
                NativeMemory.Free(batchAttnOut);
                NativeMemory.Free(batchFfnGate);
                NativeMemory.Free(batchFfnUp);
            }

            // 3. Final norm + output projection per position
            var outNormW = GetNormWeight(_outputNorm);
            var result = new float[N][];
            for (int n = 0; n < N; n++)
            {
                float* h = batchHidden + (long)n * _embDim;
                SimdKernels.RmsNorm(h, h, outNormW, _embDim, _hp.RmsNormEps);
                FusedMatVec(_logits, _outputWeight, h, _hp.VocabSize, _embDim);
                result[n] = new float[_hp.VocabSize];
                new ReadOnlySpan<float>(_logits, _hp.VocabSize).CopyTo(result[n]);
            }
            return result;
        }
        finally
        {
            NativeMemory.Free(batchHidden);
            NativeMemory.Free(batchResidual);
        }
    }

}
