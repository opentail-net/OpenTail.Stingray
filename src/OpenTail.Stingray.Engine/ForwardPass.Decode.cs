using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.TurboQuant;

namespace OpenTail.Stingray.Engine;

// Part of ForwardPass (see ForwardPass.cs for the type summary). Single-token decode: Forward /
// ForwardEmbedding / RunTrunk (the per-token sequential path), RoPE application, MLA (DeepSeek-V2/
// V3 compressed-latent attention) QKV compute, and norm/router tracing.
public sealed unsafe partial class ForwardPass
{
    /// <summary>
    /// Run one token through the full transformer. Returns logits span.
    /// </summary>
    public ReadOnlySpan<float> Forward(int token, int position)
    {
        _currentPos = position;

        // 1. Embedding lookup (single-row dequant, no full table materialization)
        EmbedToken(token, position);

        // Gemma family scales embeddings by sqrt(EmbeddingDim) before the trunk.
        if (_hp.EmbeddingScale != 1f)
            SimdKernels.ScaleInPlace(_hidden, _hp.EmbeddingScale, _embDim);

        if (_hp.HasPerLayerTokenEmbd)
            BuildPerLayerProjections(token);

        return RunTrunk(position, token);
    }

    /// <summary>
    /// Forward a single position from a PRECOMPUTED embedding (e.g. a vision soft token)
    /// instead of a token-table lookup, returning the next-token logits.
    ///
    /// Differs from <see cref="Forward"/> in two model-faithful ways for multimodal input
    /// (llama.cpp src/models/gemma4.cpp):
    ///   • does NOT apply the Gemma sqrt(EmbeddingDim) embedding scale — raw image/audio
    ///     embeddings arrive already final (gemma4.cpp:182, "do not normalize weights for
    ///     raw embeddings input"); and
    ///   • uses the padding token (id 0) for the per-layer-embedding (PLE) table lookup,
    ///     while still projecting the supplied embedding (gemma4.cpp build_inp_per_layer
    ///     multimodal branch).
    ///
    /// Attention is causal, consistent with the existing sequential Gemma path. (Gemma's
    /// reference toggles bidirectional attention within an image span; replicating that
    /// needs the batched layer-by-layer path, which is gated off for per-layer-head-dim
    /// models — tracked as a follow-up on issue #250.)
    /// </summary>
    /// <inheritdoc/>
    public bool SupportsEmbeddingInput => true;

    public ReadOnlySpan<float> ForwardEmbedding(ReadOnlySpan<float> embedding, int position)
    {
        if (embedding.Length != _embDim)
            throw new ArgumentException(
                $"embedding length {embedding.Length} != model embedding dim {_embDim}.");

        _currentPos = position;
        embedding.CopyTo(new Span<float>(_hidden, _embDim));

        // Note: no EmbeddingScale here (see remarks). PLE uses the padding token row.
        if (_hp.HasPerLayerTokenEmbd)
            BuildPerLayerProjections(0);

        return RunTrunk(position, traceToken: -1);
    }

    /// <summary>
    /// Shared transformer trunk for <see cref="Forward"/> and <see cref="ForwardEmbedding"/>:
    /// assumes <c>_hidden</c> (and, for PLE models, the per-layer projections) are already
    /// populated for <paramref name="position"/>. <paramref name="traceToken"/> is used only
    /// for the optional norm trace (−1 for embedding input).
    /// </summary>
    private ReadOnlySpan<float> RunTrunk(int position, int traceToken)
    {
        float embNorm = _traceNorms ? L2Norm(_hidden, _embDim) : 0f;

        StageCapture.Record("cpu", -1, StageCapture.Stages.Embed,
            new ReadOnlySpan<float>(_hidden, _embDim));

        bool profDecode = DecodeProfileTimers.Enabled;
        if (profDecode) DecodeProfileTimers.CountToken();

        // 2. Transformer layers
        for (int layer = 0; layer < _hp.NumLayers; layer++)
        {
            long layerStart = profDecode ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
            long namedTicks = 0;
            long stageStart;

            int layerHd = _layerHeadDim?[layer] ?? _headDim;
            // Per-layer KV head count (Gemma 4 12B: 8 GQA on SWA, 1 MQA on the global
            // k_eq_v layers). Falls back to the model-level count for every other arch.
            int layerKv = _hp.LayerKvHeads is { } lkv ? lkv[layer] : _numKvHeads;
            int qDimL = _numHeads * layerHd;
            int kvDimL = layerKv * layerHd;
            int kvSrc = _layerKvSrc is not null ? _layerKvSrc[layer] : -1;
            bool kvShared = kvSrc >= 0;
            int effLayer = kvShared ? kvSrc : layer;
            bool isSwa = _isSwaLayer is not null && _isSwaLayer[layer];
            int windowSize = isSwa ? _hp.SlidingWindowSize : -1;
            // Gemma 4 12B global layers carry no attn_v (attention_k_eq_v): V reuses the
            // raw K projection (pre QK-norm, pre-RoPE). These layers always own their KV.
            bool kEqV = _hp.AttentionKEqV && !isSwa && _wv[layer].DataPtr is null;

            // Save residual
            Copy(_residual, _hidden, _embDim);

            // Pre-attention norm (LayerNorm w/ bias for gptneox, RMSNorm otherwise). OLMo2 has no
            // attn_norm tensor at all — attention reads the raw residual, normed only by a POST-
            // attention norm applied to the sublayer's output (below, before the residual add).
            stageStart = profDecode ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
            if (_usesUnweightedNorm)
            {
                SimdKernels.PureLayerNorm(_normBuf, _hidden, _embDim, _hp.RmsNormEps);
            }
            else if (_attnNorm[layer].DataPtr is null)
            {
                Copy(_normBuf, _hidden, _embDim);
            }
            else
            {
                var normW = GetNormWeight(_attnNorm[layer]);
                var attnNormB = _hasNormBias && _bAttnNorm is not null ? _bAttnNorm[layer] : null;
                FastNorm(_normBuf, _hidden, normW, attnNormB, _embDim, _hp.RmsNormEps);
            }
            StageCapture.Record("cpu", layer, StageCapture.Stages.AttnNorm,
                new ReadOnlySpan<float>(_normBuf, _embDim));
            if (profDecode)
            {
                long d = System.Diagnostics.Stopwatch.GetTimestamp() - stageStart;
                DecodeProfileTimers.Add(DecodeProfileTimers.Category.RmsNorm, d);
                namedTicks += d;
                stageStart = System.Diagnostics.Stopwatch.GetTimestamp();
            }

            // Q projection always runs on the active layer's weights.
            // For per-layer head_dim models the trailing bytes of _q/_k/_v are stale
            // from a wider prior layer; zero them so subsequent Attention reads (which
            // stride by the active layerHd) don't pick up garbage on heads beyond
            // numHeads*layerHd, and KV cache pages don't carry inter-layer pollution.
            if (_layerHeadDim is not null)
            {
                int qBytes = _numHeads * _maxHeadDim;
                int kvBytes = _numKvHeads * _maxHeadDim;
                new Span<float>(_q, qBytes).Clear();
                new Span<float>(_k, kvBytes).Clear();
                new Span<float>(_v, kvBytes).Clear();
            }
            if (_isMla)
            {
                // MLA (DeepSeek-V2/V3): compressed-latent K/V, not a direct wk/wv projection --
                // see MlaComputeQkv's doc comment for the internal layout this writes.
                MlaComputeQkv(layer, _normBuf, _q, _k, _v);
            }
            else
            {
                FusedMatVec(_q, _wq[layer], _normBuf, qDimL, _embDim);
                if (!kvShared)
                {
                    if (kEqV)
                    {
                        // Gemma 4 12B global layers: no attn_v weight — V is the raw K
                        // projection (copied BEFORE QK-norm and RoPE, then plain-RMS-normed
                        // below). Mirrors CudaForwardPass CopyDevice(vView, kView).
                        FusedMatVec(_k, _wk[layer], _normBuf, kvDimL, _embDim);
                        Copy(_v, _k, kvDimL);
                    }
                    else if (_layerHeadDim is not null)
                    {
                        // Gemma 4: K and V share row count and dtype — fuse via
                        // MatVecDual so the row loops interleave and the input
                        // vector reads amortize. The row-interleave changes the FP
                        // ordering vs sequential matvecs by ~ULP; gated to the
                        // per-layer head_dim path because cumulative trunk drift
                        // breaks Qwen3.6-27B-MTP byte parity (see
                        // feedback_qkv_matvecdual_breaks_mtp_parity).
                        SimdKernels.MatVecDual(_k, _wk[layer].DataPtr, _v, _wv[layer].DataPtr,
                            _normBuf, kvDimL, _embDim, _wk[layer].DType, _wv[layer].DType);
                    }
                    else
                    {
                        FusedMatVec(_k, _wk[layer], _normBuf, kvDimL, _embDim);
                        FusedMatVec(_v, _wv[layer], _normBuf, kvDimL, _embDim);
                    }
                }
            }
            if (profDecode)
            {
                long d = System.Diagnostics.Stopwatch.GetTimestamp() - stageStart;
                DecodeProfileTimers.Add(DecodeProfileTimers.Category.QkvProj, d);
                namedTicks += d;
                stageStart = System.Diagnostics.Stopwatch.GetTimestamp();
            }

            StageCapture.Record("cpu", layer, StageCapture.Stages.VProj,
                new ReadOnlySpan<float>(_v, kvDimL));

            if (_hasAttnBias)
            {
                SimdKernels.AddInPlace(_q, _bq[layer], qDimL);
                if (!kvShared)
                {
                    SimdKernels.AddInPlace(_k, _bk[layer], kvDimL);
                    SimdKernels.AddInPlace(_v, _bv[layer], kvDimL);
                }
            }

            // NoPE: skip RoPE for NoPE layers. Command-R (cohere2) has the opposite rule from
            // Llama-4/SmolLM3's period-based skip: RoPE runs ONLY on SWA layers, never on
            // global ones (see ModelHyperparams.RopeOnlySwaLayers) — ANDed in on top of the
            // period-based check, which cohere2 never sets anyway.
            bool useRoPE = _hp.NoRopeLayerStep == 0
                || (layer + 1) % _hp.NoRopeLayerStep != 0;
            if (_hp.RopeOnlySwaLayers) useRoPE = useRoPE && isSwa;

            // Qwen3 (weighted QK-norm): apply norm BEFORE RoPE (per reference implementation)
            // Llama-4 (L2 QK-norm): apply norm AFTER RoPE (per llama.cpp)
            if (_hasQkNorm && !_hp.UseL2QkNorm && !_hp.QkNormAfterRope)
            {
                ApplyQkNormLayer(_q, kvShared ? null : _k, layer, layerHd, layerKv);
            }

            // Gemma 4: V is plain per-head RmsNorm (no learned weight) before cache.
            // Matches llama.cpp src/models/gemma4.cpp line 227:
            //   Vcur = ggml_rms_norm(ctx0, Vcur, hparams.f_norm_rms_eps)
            if (_layerHeadDim is not null && !kvShared)
            {
                PerHeadPureRmsNorm(_v, layerKv, layerHd, _hp.RmsNormEps);
            }

            StageCapture.Record("cpu", layer, StageCapture.Stages.VNorm,
                new ReadOnlySpan<float>(_v, kvDimL));

            if (useRoPE)
            {
                ApplyRopeLayer(_q, position, _numHeads, layer, layerHd);
                if (!kvShared)
                    ApplyRopeLayer(_k, position, layerKv, layer, layerHd);
            }

            // Hunyuan-Dense (weighted QK-norm): apply norm AFTER RoPE, on the already-rotated Q/K
            if (_hasQkNorm && !_hp.UseL2QkNorm && _hp.QkNormAfterRope)
            {
                ApplyQkNormLayer(_q, kvShared ? null : _k, layer, layerHd, layerKv);
            }

            // L2 QK-norm (Llama-4): only on RoPE layers, applied after RoPE
            if (_hasQkNorm && _hp.UseL2QkNorm && useRoPE)
            {
                PerHeadPureRmsNorm(_q, _numHeads, layerHd, _hp.RmsNormEps);
                if (!kvShared)
                    PerHeadPureRmsNorm(_k, _numKvHeads, layerHd, _hp.RmsNormEps);
            }

            if (profDecode)
            {
                long d = System.Diagnostics.Stopwatch.GetTimestamp() - stageStart;
                DecodeProfileTimers.Add(DecodeProfileTimers.Category.RoPE, d);
                namedTicks += d;
                stageStart = System.Diagnostics.Stopwatch.GetTimestamp();
            }

            // Store K, V in cache. KV-share layers don't append — the source layer's
            // cache slot is shared via effLayer in the Attention call below.
            // PagedKvCache.Append requires exactly cache.KvDim floats; for per-layer
            // head_dim models the trailing (KvDim - kvDimL) floats were just zeroed.
            if (!kvShared)
            {
                int appendLen = _layerHeadDim is not null ? _kvCache.KvDim : kvDimL;
                if (_tqKvCache != null)
                {
                    _tqKvCache.Append(layer,
                        new ReadOnlySpan<float>(_k, appendLen),
                        new ReadOnlySpan<float>(_v, appendLen));
                }
                else
                {
                    _kvCache.Append(layer,
                        new ReadOnlySpan<float>(_k, appendLen),
                        new ReadOnlySpan<float>(_v, appendLen));
                }
            }

            // Attention
            if (profDecode) stageStart = System.Diagnostics.Stopwatch.GetTimestamp();
            if (_tqKvCache != null)
                TqAttention(layer, position);
            else
                Attention(_kvCache, effLayer, layer, position, layerHd, windowSize, layerKv);
            if (profDecode)
            {
                long d = System.Diagnostics.Stopwatch.GetTimestamp() - stageStart;
                DecodeProfileTimers.Add(DecodeProfileTimers.Category.Attention, d);
                namedTicks += d;
                stageStart = System.Diagnostics.Stopwatch.GetTimestamp();
            }

            // Output projection (input width is per-layer qDim).
            StageCapture.Record("cpu", layer, StageCapture.Stages.AttnOut,
                new ReadOnlySpan<float>(_attnOut, qDimL));

            if (_isMla)
            {
                // _attnOut is zero-padded to _maxHeadDim per head (see MlaComputeQkv); _wo's
                // real GGUF shape expects numHeads*_mlaVDim, not numHeads*_maxHeadDim.
                MlaCompactAttnOut(_attnOut, _mlaAttnOutCompact);
                FusedMatVec(_hidden, _wo[layer], _mlaAttnOutCompact, _embDim, _numHeads * _mlaVDim);
            }
            else
            {
                FusedMatVec(_hidden, _wo[layer], _attnOut, _embDim, qDimL);
            }
            if (_hasAttnOutputBias)
                SimdKernels.AddInPlace(_hidden, _bo[layer], _embDim);
            StageCapture.Record("cpu", layer, StageCapture.Stages.OProj,
                new ReadOnlySpan<float>(_hidden, _embDim));
            if (profDecode)
            {
                long d = System.Diagnostics.Stopwatch.GetTimestamp() - stageStart;
                DecodeProfileTimers.Add(DecodeProfileTimers.Category.OutProj, d);
                namedTicks += d;
            }

            // Gemma 4: post-attention RmsNorm BEFORE the residual add.
            if (_postAttnNorm is not null)
            {
                var paNormW = GetNormWeight(_postAttnNorm[layer]);
                FastRmsNorm(_hidden, _hidden, paNormW, _embDim, _hp.RmsNormEps);
            }

            // Granite/MiniCPM: scale the sublayer output before it joins the residual stream.
            if (_hp.ResidualScale != 1f)
                SimdKernels.ScaleInPlace(_hidden, _hp.ResidualScale, _embDim);

            if (_hp.UseParallelResidual)
            {
                // GPT-NeoX: x = inpL + attn(ln1(inpL)) + ffn(ln2(inpL)). _hidden currently
                // holds attn_out (pre-residual, bias/scale already applied above); _residual
                // still holds inpL untouched (saved before the attn-norm, never overwritten
                // since — unlike the sequential branch below, nothing here copies into it
                // until the very end). Stash attn_out before DenseFfn/MoeFfn overwrite
                // _hidden, and compute the FFN norm from the SAME inpL, not from attn_out+inpL.
                Copy(_parAttnOut, _hidden, _embDim);
                if (_traceNorms) _normTraceAttn![layer] = L2Norm(_parAttnOut, _embDim);

                var ffnNormWp = GetNormWeight(_ffnNorm[layer]);
                stageStart = profDecode ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
                var ffnNormBp = _hasNormBias && _bFfnNorm is not null ? _bFfnNorm[layer] : null;
                FastNorm(_normBuf, _residual, ffnNormWp, ffnNormBp, _embDim, _hp.RmsNormEps);
                if (profDecode)
                {
                    long d = System.Diagnostics.Stopwatch.GetTimestamp() - stageStart;
                    DecodeProfileTimers.Add(DecodeProfileTimers.Category.RmsNorm, d);
                    namedTicks += d;
                    stageStart = System.Diagnostics.Stopwatch.GetTimestamp();
                }

                if (IsMoeLayer(layer))
                    MoeFfn(layer);
                else
                    DenseFfn(layer);
                if (profDecode)
                {
                    long d = System.Diagnostics.Stopwatch.GetTimestamp() - stageStart;
                    DecodeProfileTimers.Add(DecodeProfileTimers.Category.Ffn, d);
                    namedTicks += d;
                }

                // 3-way residual sum: _hidden(ffn_out) + _parAttnOut(attn_out) + _residual(inpL).
                SimdKernels.AddInPlace(_hidden, _parAttnOut, _embDim);
                SimdKernels.AddInPlace(_hidden, _residual, _embDim);
                Copy(_residual, _hidden, _embDim);
            }
            else
            {
            // Residual
            SimdKernels.AddInPlace(_hidden, _residual, _embDim);

            if (_traceNorms) _normTraceAttn![layer] = L2Norm(_hidden, _embDim);

            StageCapture.Record("cpu", layer, StageCapture.Stages.PostAttnResidual,
                new ReadOnlySpan<float>(_hidden, _embDim));

            // Save residual for FFN
            Copy(_residual, _hidden, _embDim);

            // Pre-FFN norm (LayerNorm w/ bias for starcoder2/gptneox-style archs, RMSNorm
            // otherwise). OLMo2 has no ffn_norm tensor at all — FFN reads the raw post-attention
            // residual directly, normed only by a POST-FFN norm applied to the sublayer's output
            // below, before the residual add.
            stageStart = profDecode ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
            if (_usesUnweightedNorm)
            {
                SimdKernels.PureLayerNorm(_normBuf, _hidden, _embDim, _hp.RmsNormEps);
            }
            else if (_ffnNorm[layer].DataPtr is null)
                Copy(_normBuf, _hidden, _embDim);
            else
            {
                var ffnNormW = GetNormWeight(_ffnNorm[layer]);
                var ffnNormBSeq = _hasNormBias && _bFfnNorm is not null ? _bFfnNorm[layer] : null;
                FastNorm(_normBuf, _hidden, ffnNormW, ffnNormBSeq, _embDim, _hp.RmsNormEps);
            }
            if (profDecode)
            {
                long d = System.Diagnostics.Stopwatch.GetTimestamp() - stageStart;
                DecodeProfileTimers.Add(DecodeProfileTimers.Category.RmsNorm, d);
                namedTicks += d;
                stageStart = System.Diagnostics.Stopwatch.GetTimestamp();
            }

            if (IsMoeLayer(layer))
                MoeFfn(layer);
            else
                DenseFfn(layer);
            if (profDecode)
            {
                long d = System.Diagnostics.Stopwatch.GetTimestamp() - stageStart;
                DecodeProfileTimers.Add(DecodeProfileTimers.Category.Ffn, d);
                namedTicks += d;
            }

            // Gemma 4: post-FFN RmsNorm before the residual add.
            if (_postFfwNorm is not null)
            {
                var pfNormW = GetNormWeight(_postFfwNorm[layer]);
                FastRmsNorm(_hidden, _hidden, pfNormW, _embDim, _hp.RmsNormEps);
            }

            // Granite/MiniCPM: scale the sublayer output before it joins the residual stream.
            if (_hp.ResidualScale != 1f)
                SimdKernels.ScaleInPlace(_hidden, _hp.ResidualScale, _embDim);

            // Residual (post-attn output that includes its own residual).
            SimdKernels.AddInPlace(_hidden, _residual, _embDim);
            }

            StageCapture.Record("cpu", layer, StageCapture.Stages.PostFfnResidual,
                new ReadOnlySpan<float>(_hidden, _embDim));

            if (_hp.HasPerLayerTokenEmbd)
                ApplyPerLayerEmbedding(layer);

            StageCapture.Record("cpu", layer, StageCapture.Stages.PostPle,
                new ReadOnlySpan<float>(_hidden, _embDim));

            // Gemma 4: per-layer learned output scale applies AFTER the PLE injection
            // (matches llama.cpp gemma4 build order — applying it before PLE breaks the
            // residual balance and produces unbounded hidden L2 growth).
            if (_layerOutputScale is not null)
                SimdKernels.ScaleInPlace(_hidden, _layerOutputScale[layer], _embDim);

            // Hidden-state tap: _hidden now holds this layer's output (= next layer's input).
            if (_taps is { } taps && taps.SlotOf(layer) is int tapSlot && tapSlot >= 0)
                CaptureTap(position, tapSlot, _hidden);

            StageCapture.Record("cpu", layer, StageCapture.Stages.LayerOutput,
                new ReadOnlySpan<float>(_hidden, _embDim));

            if (_traceNorms) _normTraceFfn![layer] = L2Norm(_hidden, _embDim);

            if (profDecode)
            {
                long layerTotal = System.Diagnostics.Stopwatch.GetTimestamp() - layerStart;
                DecodeProfileTimers.Add(DecodeProfileTimers.Category.Other, Math.Max(0, layerTotal - namedTicks));
            }
        }

        // Increment KV cache position
        if (_tqKvCache != null)
            _tqKvCache.IncrementPosition();
        else
            _kvCache.IncrementPosition();

        float preFinalNorm = _traceNorms ? L2Norm(_hidden, _embDim) : 0f;

        // 3. Final norm
        long finalNormStart = profDecode ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
        if (_usesUnweightedNorm)
        {
            SimdKernels.PureLayerNorm(_hidden, _hidden, _embDim, _hp.RmsNormEps);
        }
        else
        {
            var outNormW = GetNormWeight(_outputNorm);
            FastNorm(_hidden, _hidden, outNormW, _hasNormBias ? _bOutputNorm : null, _embDim, _hp.RmsNormEps);
        }
        if (profDecode) DecodeProfileTimers.Add(DecodeProfileTimers.Category.RmsNorm, System.Diagnostics.Stopwatch.GetTimestamp() - finalNormStart);

        float postFinalNorm = _traceNorms ? L2Norm(_hidden, _embDim) : 0f;

        // 4. Output projection → logits (fused, no 400MB intermediate buffer)
        long logitsStart = profDecode ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
        FusedMatVec(_logits, _outputWeight, _hidden, _hp.VocabSize, _embDim);
        if (profDecode) DecodeProfileTimers.Add(DecodeProfileTimers.Category.OutProj, System.Diagnostics.Stopwatch.GetTimestamp() - logitsStart);

        // Granite/MiniCPM final-logit scale (already carries llama.cpp's 1/f_logit_scale
        // reciprocal — see ModelHyperparams.LogitScale).
        if (_hp.LogitScale != 1f)
            SimdKernels.ScaleInPlace(_logits, _hp.LogitScale, _hp.VocabSize);

        // Gemma 4 final-logit softcap: x = tanh(x/cap) * cap.
        if (_hp.FinalLogitSoftcap > 0f)
            SimdKernels.SoftcapInPlace(_logits, _hp.VocabSize, _hp.FinalLogitSoftcap);

        if (_traceNorms)
            EmitNormTrace(traceToken, position, embNorm, preFinalNorm, postFinalNorm);

        return new ReadOnlySpan<float>(_logits, _hp.VocabSize);
    }

    private void ApplyRope(float* x, int pos, int heads)
    {
        var cos = _ropeCosTable + (long)pos * _ropeHalfDim;
        var sin = _ropeSinTable + (long)pos * _ropeHalfDim;
        if (_hp.IsNeoxRope)
            SimdKernels.ApplyRoPECachedNeox(x, cos, sin, heads, _headDim);
        else
            SimdKernels.ApplyRoPECached(x, cos, sin, heads, _headDim);
    }

    /// <summary>
    /// Per-layer-aware RoPE: selects the global or SWA cos/sin table for Gemma 4 and
    /// rotates each head's leading <paramref name="layerHd"/> dims.
    /// </summary>
    private void ApplyRopeLayer(float* x, int pos, int heads, int layer, int layerHd)
    {
        bool useSwa = _ropeCosTableSwa != null && _isSwaLayer is not null && _isSwaLayer[layer];
        int halfDim = useSwa ? _ropeHalfDimSwa : _ropeHalfDim;
        float* cosTab = useSwa ? _ropeCosTableSwa : _ropeCosTable;
        float* sinTab = useSwa ? _ropeSinTable    : _ropeSinTable;
        if (useSwa) sinTab = _ropeSinTableSwa;
        var cos = cosTab + (long)pos * halfDim;
        var sin = sinTab + (long)pos * halfDim;
        if (_hp.IsNeoxRope)
        {
            // Partial RoPE (GPT-NeoX/Pythia: rope.dimension_count < headDim, e.g. 16 of 64):
            // rotate only the leading _ropeDim channels, dims [_ropeDim, layerHd) pass
            // through. _ropeDim == layerHd for every other architecture, so this is a no-op
            // fast path everywhere except gptneox.
            if (_ropeDim < layerHd)
                SimdKernels.ApplyRoPECachedNeoxPartial(x, cos, sin, heads, layerHd, _ropeDim);
            else
                SimdKernels.ApplyRoPECachedNeox(x, cos, sin, heads, layerHd);
        }
        else
        {
            // Partial RoPE, "normal"/interleaved convention (GLM4 non-multimodal:
            // rope.dimension_count=64, headDim=128) — same _ropeDim mechanism as the NEOX
            // branch above, just the interleaved-pair kernel instead of the halfDim-offset one.
            if (_ropeDim < layerHd)
                SimdKernels.ApplyRoPECachedPartial(x, cos, sin, heads, layerHd, _ropeDim);
            else
                SimdKernels.ApplyRoPECached(x, cos, sin, heads, layerHd);
        }
    }

    /// <summary>
    /// MLA (DeepSeek-V2/V3/R1) Q/K/V for one token, replacing the standard wq/wk/wv projections
    /// for this architecture. Writes <paramref name="q"/>/<paramref name="k"/>/<paramref name="v"/>
    /// (each <c>_numHeads * _maxHeadDim</c>-wide, same buffers/stride the standard path uses) with
    /// this class's internal per-head layout: K is <c>[rope(_ropeDim), nope(_mlaNopeDim)]</c> --
    /// ROPE FIRST, unlike ggml's <c>[nope, rope]</c> -- so the existing partial-RoPE path
    /// (<see cref="ApplyRopeLayer"/>, which rotates only the LEADING <c>_ropeDim</c> channels) can
    /// rotate it with no new RoPE code; attention's dot product doesn't care which permutation Q
    /// and K agree on, only that they DO agree, which this and Q's own reordering below guarantee.
    /// V is <c>[v(_mlaVDim), zero-pad(_maxHeadDim - _mlaVDim)]</c> -- padded because
    /// <see cref="PagedKvCache"/>/<see cref="Attention(PagedKvCache,int,int)"/> assume one shared
    /// width for K and V per head, which MLA's independent 192/128 GGUF dims don't naturally have;
    /// the padding contributes exactly zero to the attention-weighted sum by construction, and
    /// callers must read only the first <c>_mlaVDim</c> floats of each head's attention OUTPUT
    /// before the wo projection (see PrefillCoreAttention/RunTrunk's MLA branches).
    ///
    /// <para>Q is a plain per-head projection (<c>_wq[layer]</c>, ggml <c>[nope, rope]</c> order)
    /// since only q_lora_rank==0 "lite" checkpoints are handled -- reordered into this class's
    /// <c>[rope, nope]</c> convention below. Only the legacy unsplit <c>wkv_b</c> tensor layout is
    /// handled for K/V (DeepSeek-V2-Lite ships this); the split <c>wk_b</c>/<c>wv_b</c>
    /// "absorption" layout some newer checkpoints use is not implemented -- such a GGUF has no
    /// <c>attn_kv_b</c> tensor at all, so <see cref="ResolveTensor"/> already fails closed
    /// ("Missing tensor") at load time rather than silently mis-attending.</para>
    /// </summary>
    private void MlaComputeQkv(int layer, float* normBuf, float* q, float* k, float* v)
    {
        int qDimMla = _numHeads * _headDim;

        // Q: standard per-head projection, ggml [nope, rope] order -> reorder to [rope, nope].
        FusedMatVec(_q, _wq[layer], normBuf, qDimMla, _embDim);
        for (int h = 0; h < _numHeads; h++)
        {
            float* src = _q + (long)h * _headDim;      // [nope(_mlaNopeDim), rope(_ropeDim)]
            float* dst = q + (long)h * _maxHeadDim;     // [rope(_ropeDim), nope(_mlaNopeDim)]
            Copy(dst, src + _mlaNopeDim, _ropeDim);
            Copy(dst + _ropeDim, src, _mlaNopeDim);
        }

        // K/V: compress -> norm -> decompress (see the doc comment above for the full shape story).
        FusedMatVec(_mlaKvCmprPe, _wKvAMqa![layer], normBuf, _mlaKvLoraRank + _ropeDim, _embDim);

        var kvANormW = GetNormWeight(_kvANorm![layer]);
        FastNorm(_mlaKvCmprPe, _mlaKvCmprPe, kvANormW, null, _mlaKvLoraRank, _hp.RmsNormEps);

        FusedMatVec(_mlaDecompressed, _wKvB![layer], _mlaKvCmprPe,
            _numHeads * (_mlaNopeDim + _mlaVDim), _mlaKvLoraRank);

        float* kPe = _mlaKvCmprPe + _mlaKvLoraRank; // MQA: identical across every head
        int decompStride = _mlaNopeDim + _mlaVDim;
        for (int h = 0; h < _numHeads; h++)
        {
            float* kHead = k + (long)h * _maxHeadDim;
            float* vHead = v + (long)h * _maxHeadDim;
            float* decompHead = _mlaDecompressed + (long)h * decompStride;

            Copy(kHead, kPe, _ropeDim);
            Copy(kHead + _ropeDim, decompHead, _mlaNopeDim);

            Copy(vHead, decompHead + _mlaNopeDim, _mlaVDim);
            if (_mlaVDim < _maxHeadDim)
                new Span<float>(vHead + _mlaVDim, _maxHeadDim - _mlaVDim).Clear();
        }
    }

    /// <summary>
    /// Compacts an MLA attention output from this class's zero-padded per-head width
    /// (<c>_maxHeadDim</c>, see <see cref="MlaComputeQkv"/>) down to the REAL per-head V width
    /// (<c>_mlaVDim</c>) that <c>_wo</c>'s actual GGUF shape expects
    /// (<c>n_head * n_embd_head_v_mla</c>, not <c>n_head * _maxHeadDim</c>). The dropped tail is
    /// always exactly zero by construction, so this is a lossless truncation, not an approximation.
    /// </summary>
    private void MlaCompactAttnOut(float* attnOutPadded, float* attnOutCompact)
    {
        for (int h = 0; h < _numHeads; h++)
            Copy(attnOutCompact + (long)h * _mlaVDim, attnOutPadded + (long)h * _maxHeadDim, _mlaVDim);
    }

    /// <summary>
    /// Batched (N-token) sibling of <see cref="MlaCompactAttnOut"/> for <see cref="PrefillCore"/>:
    /// <paramref name="batchAttnOutPadded"/> is <c>N * (numHeads * _maxHeadDim)</c>-strided,
    /// <paramref name="batchAttnOutCompact"/> is <c>N * (numHeads * _mlaVDim)</c>-strided.
    /// </summary>
    private void MlaCompactAttnOutBatched(float* batchAttnOutPadded, float* batchAttnOutCompact, int N)
    {
        int paddedStride = _numHeads * _maxHeadDim;
        int compactStride = _numHeads * _mlaVDim;
        for (int n = 0; n < N; n++)
            MlaCompactAttnOut(batchAttnOutPadded + (long)n * paddedStride, batchAttnOutCompact + (long)n * compactStride);
    }

    /// <summary>
    /// Batched (N-token) sibling of <see cref="MlaComputeQkv"/> for <see cref="PrefillCore"/>:
    /// same per-token math and layout (see that method's doc comment), but Q/K-nope-and-V's three
    /// projections each run as ONE batched GEMM across all N tokens (via
    /// <see cref="MatMulBatchedCached"/>) instead of N separate MatVec calls, matching how the
    /// standard (non-MLA) path amortizes weight reads across a whole prefill chunk.
    /// <paramref name="batchQ"/>/<paramref name="batchK"/>/<paramref name="batchV"/> are each
    /// <c>N * (numHeads * _maxHeadDim)</c>-strided, matching PrefillCore's existing batch buffers.
    /// </summary>
    private void MlaComputeQkvBatched(int layer, float* batchNorm, int N, float* batchQ, float* batchK, float* batchV)
    {
        int qDimMla = _numHeads * _headDim;
        int kvCmprPeDim = _mlaKvLoraRank + _ropeDim;
        int decompDim = _numHeads * (_mlaNopeDim + _mlaVDim);

        var batchQRaw = (float*)NativeMemory.AllocZeroed((nuint)((long)N * qDimMla * sizeof(float)));
        var batchKvCmprPe = (float*)NativeMemory.AllocZeroed((nuint)((long)N * kvCmprPeDim * sizeof(float)));
        var batchKvCmprNormed = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _mlaKvLoraRank * sizeof(float)));
        var batchDecompressed = (float*)NativeMemory.AllocZeroed((nuint)((long)N * decompDim * sizeof(float)));
        try
        {
            MatMulBatchedCached(batchQRaw, in _wq[layer], batchNorm, N, qDimMla, _embDim);
            MatMulBatchedCached(batchKvCmprPe, in _wKvAMqa![layer], batchNorm, N, kvCmprPeDim, _embDim);

            var kvANormW = GetNormWeight(_kvANorm![layer]);
            for (int n = 0; n < N; n++)
            {
                float* src = batchKvCmprPe + (long)n * kvCmprPeDim;
                float* dst = batchKvCmprNormed + (long)n * _mlaKvLoraRank;
                FastNorm(dst, src, kvANormW, null, _mlaKvLoraRank, _hp.RmsNormEps);
            }

            MatMulBatchedCached(batchDecompressed, in _wKvB![layer], batchKvCmprNormed, N, decompDim, _mlaKvLoraRank);

            int decompStride = _mlaNopeDim + _mlaVDim;
            int qStride = _numHeads * _maxHeadDim;
            for (int n = 0; n < N; n++)
            {
                float* qRawTok = batchQRaw + (long)n * qDimMla;
                float* qTok = batchQ + (long)n * qStride;
                float* kTok = batchK + (long)n * qStride;
                float* vTok = batchV + (long)n * qStride;
                float* kPe = batchKvCmprPe + (long)n * kvCmprPeDim + _mlaKvLoraRank; // MQA: shared across heads
                float* decompTok = batchDecompressed + (long)n * decompDim;

                for (int h = 0; h < _numHeads; h++)
                {
                    float* qSrc = qRawTok + (long)h * _headDim;   // ggml [nope, rope]
                    float* qDst = qTok + (long)h * _maxHeadDim;   // this class's [rope, nope]
                    Copy(qDst, qSrc + _mlaNopeDim, _ropeDim);
                    Copy(qDst + _ropeDim, qSrc, _mlaNopeDim);

                    float* kHead = kTok + (long)h * _maxHeadDim;
                    float* vHead = vTok + (long)h * _maxHeadDim;
                    float* decompHead = decompTok + (long)h * decompStride;

                    Copy(kHead, kPe, _ropeDim);
                    Copy(kHead + _ropeDim, decompHead, _mlaNopeDim);

                    Copy(vHead, decompHead + _mlaNopeDim, _mlaVDim);
                    if (_mlaVDim < _maxHeadDim)
                        new Span<float>(vHead + _mlaVDim, _maxHeadDim - _mlaVDim).Clear();
                }
            }
        }
        finally
        {
            NativeMemory.Free(batchQRaw);
            NativeMemory.Free(batchKvCmprPe);
            NativeMemory.Free(batchKvCmprNormed);
            NativeMemory.Free(batchDecompressed);
        }
    }

    private static float L2Norm(float* x, int n)
    {
        double s = 0;
        for (int i = 0; i < n; i++) { double v = x[i]; s += v * v; }
        return (float)Math.Sqrt(s);
    }


    private void EmitNormTrace(int token, int position,
        float embNorm, float preFinalNorm, float postFinalNorm)
    {
        // Top-1 logit + index
        int topIdx = 0; float topVal = float.MinValue;
        for (int i = 0; i < _hp.VocabSize; i++)
            if (_logits[i] > topVal) { topVal = _logits[i]; topIdx = i; }

        var sb = new System.Text.StringBuilder(2048);
        var inv = System.Globalization.CultureInfo.InvariantCulture;
        sb.Append("[norms pos=").Append(position)
          .Append(" tok=").Append(token)
          .Append(" emb=").Append(embNorm.ToString("F2", inv));
        for (int i = 0; i < _hp.NumLayers; i++)
        {
            sb.Append(" L").Append(i).Append(":a=")
              .Append(_normTraceAttn![i].ToString("F1", inv))
              .Append("/f=").Append(_normTraceFfn![i].ToString("F1", inv));
        }
        sb.Append(" preFN=").Append(preFinalNorm.ToString("F2", inv))
          .Append(" postFN=").Append(postFinalNorm.ToString("F2", inv))
          .Append(" top=").Append(topIdx)
          .Append('@').Append(topVal.ToString("F2", inv));
        Console.Error.WriteLine(sb.ToString());
    }

}
