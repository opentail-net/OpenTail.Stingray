
namespace OpenTail.Stingray.Engine;

// Part of ForwardPass (see ForwardPass.cs for the type summary). The IBatchedForwardPass /
// IPrefixCacheableBatchedForwardPass surface: CreateCache, ForwardCore, PrefillWithCache (the
// per-sequence-cache API ContinuousBatchingEngine drives), BatchForwardMulti, PrefillPackedMulti,
// and Dispose.
public sealed unsafe partial class ForwardPass
{
    // ================================================================
    //  Continuous Batching API
    // ================================================================

    /// <summary>
    /// Creates a new empty <see cref="PagedKvCache"/> compatible with this model's layer/head dimensions.
    /// Used by <see cref="ContinuousBatchingEngine"/> to allocate per-sequence caches.
    /// </summary>
    public PagedKvCache CreateCache() =>
        new PagedKvCache(_hp.NumLayers, _hp.NumKvHeads, _maxHeadDim,
            bf16Store: PagedKvCache.Bf16StoreRequested,
            autoBf16: PagedKvCache.Bf16AutoRequested,
            layerHeadDim: _layerHeadDim);

    // ── IBatchedForwardPass (issue #190) ────────────────────────────────────────
    // The engine drives this forward pass through the backend-agnostic interface, holding
    // caches as opaque ISequenceKvCache handles. For the CPU path the handle IS the concrete
    // PagedKvCache the methods above already take, so these explicit implementations just
    // unwrap it. SnapKvEnabled / KvBytesPerToken / PrefillDequantCacheActive are public and
    // satisfy the interface implicitly.
    ISequenceKvCache IBatchedForwardPass.CreateCache() => CreateCache();

    int IPrefixCacheableBatchedForwardPass.PrefixCacheBlockSize => PagedKvCache.PageSize;

    ISequenceKvCache IPrefixCacheableBatchedForwardPass.CapturePrefix(ISequenceKvCache cache, int prefixLength) =>
        ((PagedKvCache)cache).ForkSharedPrefix(prefixLength);

    ISequenceKvCache IPrefixCacheableBatchedForwardPass.ForkPrefix(ISequenceKvCache prefix) =>
        ((PagedKvCache)prefix).ForkSharedPrefix(((PagedKvCache)prefix).Length);

    ReadOnlySpan<float> IBatchedForwardPass.PrefillWithCache(
        IReadOnlyList<int> tokens, ISequenceKvCache cache, int startPos)
        => PrefillWithCache(tokens, (PagedKvCache)cache, startPos);

    float[]?[] IBatchedForwardPass.PrefillPackedMulti(
        ReadOnlyMemory<int>[] chunks, int[] startPos, ISequenceKvCache[] caches, bool[] wantLogits)
        => PrefillPackedMulti(chunks, startPos, AsPaged(caches), wantLogits);

    float[][] IBatchedForwardPass.BatchForwardMulti(int[] tokens, int[] positions, ISequenceKvCache[] caches)
        => BatchForwardMulti(tokens, positions, AsPaged(caches));

    private static PagedKvCache[] AsPaged(ISequenceKvCache[] caches)
    {
        var r = new PagedKvCache[caches.Length];
        for (int i = 0; i < caches.Length; i++)
            r[i] = (PagedKvCache)caches[i];
        return r;
    }

    /// <summary>
    /// Forward pass for a single token using the provided explicit cache (no TurboQuant).
    /// Used by <see cref="PrefillWithCache"/> for single-token prompts and MoE sequential prefill.
    /// </summary>
    private ReadOnlySpan<float> ForwardCore(int token, int pos, PagedKvCache cache)
    {
        // Scratch sized from _ctxLen, but the KV cache is not: PagedKvCache defaults to 8192
        // blocks (131,072 positions), so it keeps accepting appends long after `pos` has run off
        // the end of the ctxLen-sized buffers. Attention writes scores[h * _ctxLen + t] for
        // t < pos + 1, and RoPE reads _ropeCosTable + pos * _ropeHalfDim; both are unchecked
        // native accesses, so overrunning corrupts memory rather than failing. Callers are
        // expected to stop at MaxSeqLen — this makes the invariant unbypassable instead of
        // trusting each one, and turns silent corruption into a diagnosable throw.
        if ((uint)pos >= (uint)_ctxLen)
        {
            throw new ArgumentOutOfRangeException(nameof(pos), pos,
                $"Position exceeds the active context length ({_ctxLen}). Generation must stop at " +
                $"MaxSeqLen; continuing would write past the attention-score and RoPE scratch buffers.");
        }

        EmbedToken(token, pos);
        for (int layer = 0; layer < _hp.NumLayers; layer++)
        {
            Copy(_residual, _hidden, _embDim);
            var normW = GetNormWeight(_attnNorm[layer]);
            SimdKernels.RmsNorm(_normBuf, _hidden, normW, _embDim, _hp.RmsNormEps);
            FusedMatVec(_q, _wq[layer], _normBuf, _numHeads * _headDim, _embDim);
            FusedMatVec(_k, _wk[layer], _normBuf, _numKvHeads * _headDim, _embDim);
            FusedMatVec(_v, _wv[layer], _normBuf, _numKvHeads * _headDim, _embDim);
            if (_hasAttnBias)
            {
                SimdKernels.AddInPlace(_q, _bq[layer], _numHeads * _headDim);
                SimdKernels.AddInPlace(_k, _bk[layer], _numKvHeads * _headDim);
                SimdKernels.AddInPlace(_v, _bv[layer], _numKvHeads * _headDim);
            }
            {
                bool useRoPE = _hp.NoRopeLayerStep == 0
                    || (layer + 1) % _hp.NoRopeLayerStep != 0;
                if (_hasQkNorm && !_hp.UseL2QkNorm)
                {
                    ApplyQkNorm(_q, _k, layer);
                }
                if (useRoPE)
                {
                    ApplyRope(_q, pos, _numHeads);
                    ApplyRope(_k, pos, _numKvHeads);
                }
                if (_hasQkNorm && _hp.UseL2QkNorm && useRoPE)
                {
                    PerHeadPureRmsNorm(_q, _numHeads, _headDim, _hp.RmsNormEps);
                    PerHeadPureRmsNorm(_k, _numKvHeads, _headDim, _hp.RmsNormEps);
                }
            }
            cache.Append(layer,
                new ReadOnlySpan<float>(_k, _numKvHeads * _headDim),
                new ReadOnlySpan<float>(_v, _numKvHeads * _headDim));
            Attention(cache, layer, pos);
            FusedMatVec(_hidden, _wo[layer], _attnOut, _embDim, _numHeads * _headDim);
            if (_hasAttnOutputBias)
                SimdKernels.AddInPlace(_hidden, _bo[layer], _embDim);
            SimdKernels.AddInPlace(_hidden, _residual, _embDim);
            Copy(_residual, _hidden, _embDim);
            var ffnNormW = GetNormWeight(_ffnNorm[layer]);
            SimdKernels.RmsNorm(_normBuf, _hidden, ffnNormW, _embDim, _hp.RmsNormEps);
            if (_hp.IsMoE)
                MoeFfn(layer);
            else
                DenseFfn(layer);
            SimdKernels.AddInPlace(_hidden, _residual, _embDim);
        }
        cache.IncrementPosition();
        var outNormW = GetNormWeight(_outputNorm);
        SimdKernels.RmsNorm(_hidden, _hidden, outNormW, _embDim, _hp.RmsNormEps);
        FusedMatVec(_logits, _outputWeight, _hidden, _hp.VocabSize, _embDim);
        return new ReadOnlySpan<float>(_logits, _hp.VocabSize);
    }

    /// <summary>
    /// Prefill prompt tokens into an explicitly provided KV cache instead of the engine's primary cache.
    /// Used by <see cref="ContinuousBatchingEngine"/> to prefill per-sequence caches concurrently.
    /// Not supported when TurboQuant KV cache is enabled.
    /// </summary>
    /// <param name="tokens">Prompt token IDs to process.</param>
    /// <param name="cache">The KV cache to write into.</param>
    /// <param name="startPos">Starting position in the cache (default 0).</param>
    /// <returns>Logits for the last token.</returns>
    public ReadOnlySpan<float> PrefillWithCache(IReadOnlyList<int> tokens, PagedKvCache cache, int startPos = 0)
    {
        if (_tqKvCache != null)
            throw new NotSupportedException("PrefillWithCache is not supported when TurboQuant KV cache is enabled.");
        if (_layerHeadDim is not null)
            throw new NotSupportedException(
                "gemma4 per-layer head_dim not yet supported on PrefillWithCache.");
        int N = tokens.Count;
        if (N == 0) throw new ArgumentException("Token list is empty", nameof(tokens));
        // Keep the externally supplied-cache route coherent with PrefillDispatch. Continuous
        // batching calls this method, so leaving the all-control safeguard only on Prefill would
        // make server-admitted structural probes take the numerically unsafe Q8 activation path.
        //
        // Deliberately NOT short-circuiting on N == 1 here, unlike PrefillDispatch (issue: hot-
        // session replay divergence investigation, 2026-08-13). This method is the one path a
        // retained session's per-turn admission calls (ContinuousBatchingEngine), and a session's
        // turns are chunked arbitrarily by caller-supplied prompt length -- a one-word continuation
        // (e.g. " capital") is exactly as likely to land here with N == 1 as N == 8. PrefillCore's
        // MatMulBatchedCached already takes the Q8/repacked-Q4Kx8 path for any batch size,
        // including 1 (see its "ragged tail" handling), so there is no throughput reason to route
        // N == 1 through the non-quantized PrefillWithCacheSequential/ForwardCore path here the way
        // there is for a genuinely fresh, large single-shot Prefill() call. Taking the sequential
        // shortcut at N == 1 meant a retained session's short continuations silently ran full-F32
        // while its own longer turns and its full-replay oracle both ran Q8-quantized -- the same
        // logical sequence computed with two different, non-bit-exact precisions depending on how
        // the caller happened to chunk it across turns. Measured impact: maxAbsDiff ~0.85 across
        // every vocab logit for the same position (see PrefillPathParityTests.cs) -- large enough
        // to flip a close greedy choice a few tokens later, which is what
        // HotSessionGreedyReplayTests catches. The other conditions (control/degenerate prompts,
        // unsupported MoE) are unrelated numerical-safety cases and still take the sequential path.
        if (IsAllControlTokenPrompt(tokens) || IsSingleDistinctTokenPrompt(tokens)
            || (_hp.IsMoE && !MoeBatchedPrefillSupported))
            return PrefillWithCacheSequential(tokens, cache, startPos);
        return PrefillCore(tokens, cache, startPos);
    }

    /// <summary>
    /// Exact, token-at-a-time external-cache admission. Used for individually ineligible
    /// requests and for the whole packed group when one member is an all-control structural
    /// probe: packing must not make an ordinary neighbour's numerical route timing-dependent.
    /// </summary>
    private ReadOnlySpan<float> PrefillWithCacheSequential(
        IReadOnlyList<int> tokens, PagedKvCache cache, int startPos)
    {
        ReadOnlySpan<float> logits = default;
        for (int i = 0; i < tokens.Count; i++)
            logits = ForwardCore(tokens[i], startPos + i, cache);
        return logits;
    }

    /// <summary>
    /// Batched decode step for N sequences simultaneously: one token per sequence, each with its own
    /// KV cache at the given position.
    ///
    /// <para><b>On what batching actually buys here.</b> This used to claim it "amortizes weight
    /// reads N× across concurrent users". Measured on CPU (docs/session-native-inference-runtime-plan.md
    /// §3.4.11): going from 1 to 4 sequences cuts per-token trunk time by 12.7%, and four
    /// concurrent sessions aggregate 1.18x a single session — not 4x. The weight reuse is real
    /// (<see cref="SimdKernels.BatchedMatVecTierEnabled"/> routes the batch through
    /// <c>MatVec4In</c>, which reads and dequantizes each weight row once for four inputs) but the
    /// per-input FMA work is still N×, and that is where the time goes: FFN alone is ~70% of decode
    /// trunk time and improves only 11% at N=4. CPU decode at these sizes is not weight-bandwidth
    /// bound. Attention is ~3% of decode, so per-sequence attention is not the limiter either.</para>
    /// Not supported when TurboQuant KV cache is enabled or for MoE models.
    /// </summary>
    /// <param name="tokens">Next token for each sequence (length N).</param>
    /// <param name="positions">Current decode position for each sequence (= cache.Length before this call).</param>
    /// <param name="caches">Per-sequence KV cache (length N).</param>
    /// <returns>Logits array for each sequence (length N × VocabSize).</returns>
    public float[][] BatchForwardMulti(int[] tokens, int[] positions, PagedKvCache[] caches)
    {
        if (_tqKvCache != null)
            throw new NotSupportedException("BatchForwardMulti is not supported when TurboQuant KV cache is enabled.");
        if (_hp.IsMoE)
            throw new NotSupportedException("BatchForwardMulti is not supported for MoE models; use individual ForwardCore calls.");
        if (_layerHeadDim is not null)
            throw new NotSupportedException(
                "gemma4 per-layer head_dim not yet supported on BatchForwardMulti.");
        int N = tokens.Length;
        if (N == 0) return Array.Empty<float[]>();
        int qDim = _numHeads * _headDim;
        int kvDim = _numKvHeads * _headDim;
        bool profDecodeB = DecodeProfileTimers.Enabled;
        // One decoded token per SEQUENCE, so the profiler's per-token averages mean the same thing
        // here as on the sequential path rather than counting a whole batch as a single token.
        if (profDecodeB) for (int pt = 0; pt < N; pt++) DecodeProfileTimers.CountToken();
        var batchHidden = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _embDim * sizeof(float)));
        var batchResidual = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _embDim * sizeof(float)));
        try
        {
            for (int n = 0; n < N; n++)
                EmbedTokenInto(tokens[n], batchHidden + (long)n * _embDim, positions[n]);
            var batchNorm = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _embDim * sizeof(float)));
            var batchQ = (float*)NativeMemory.AllocZeroed((nuint)((long)N * qDim * sizeof(float)));
            var batchK = (float*)NativeMemory.AllocZeroed((nuint)((long)N * kvDim * sizeof(float)));
            var batchV = (float*)NativeMemory.AllocZeroed((nuint)((long)N * kvDim * sizeof(float)));
            var batchAttnOut = (float*)NativeMemory.AllocZeroed((nuint)((long)N * qDim * sizeof(float)));
            var batchFfnGate = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _intermDim * sizeof(float)));
            var batchFfnUp = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _intermDim * sizeof(float)));
            try
            {
                for (int layer = 0; layer < _hp.NumLayers; layer++)
                {
                    // Stage timing (STINGRAY_PROFILE_DECODE=1). BatchForwardMulti had none,
                    // so the multi-sequence decode path -- the one whose flat scaling the session
                    // runtime plan is trying to explain -- could not be profiled at all, only
                    // speculated about. Same categories as ForwardCore so the two decode paths
                    // compare bucket for bucket.
                    long bStage = profDecodeB ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
                    var normW = GetNormWeight(_attnNorm[layer]);
                    for (int n = 0; n < N; n++)
                    {
                        Copy(batchResidual + (long)n * _embDim, batchHidden + (long)n * _embDim, _embDim);
                        SimdKernels.RmsNorm(batchNorm + (long)n * _embDim,
                            batchHidden + (long)n * _embDim, normW, _embDim, _hp.RmsNormEps);
                    }
                    if (profDecodeB)
                    {
                        DecodeProfileTimers.Add(DecodeProfileTimers.Category.RmsNorm,
                            System.Diagnostics.Stopwatch.GetTimestamp() - bStage);
                        bStage = System.Diagnostics.Stopwatch.GetTimestamp();
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
                    if (profDecodeB)
                    {
                        DecodeProfileTimers.Add(DecodeProfileTimers.Category.QkvProj,
                            System.Diagnostics.Stopwatch.GetTimestamp() - bStage);
                    }
                    bool useRoPE = _hp.NoRopeLayerStep == 0
                        || (layer + 1) % _hp.NoRopeLayerStep != 0;
                    // Per-sequence: RoPE, KV append to individual cache, causal attention.
                    // RoPE/norm/append and the attention itself are timed separately: the open
                    // question is specifically whether per-sequence ATTENTION is the irreducible
                    // residual, and lumping the cache bookkeeping in with it would not answer that.
                    long bRope = 0, bAttn = 0;
                    for (int n = 0; n < N; n++)
                    {
                        long bSeq = profDecodeB ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
                        float* qn = batchQ + (long)n * qDim;
                        float* kn = batchK + (long)n * kvDim;
                        float* vn = batchV + (long)n * kvDim;
                        int pos = positions[n];
                        // Soft-reset this layer's position so the Append lands at pos
                        caches[n].TruncateTo(pos);
                        if (useRoPE)
                        {
                            ApplyRope(qn, pos, _numHeads);
                            ApplyRope(kn, pos, _numKvHeads);
                        }
                        if (_hasQkNorm)
                        {
                            if (_hp.UseL2QkNorm)
                            {
                                PerHeadPureRmsNorm(qn, _numHeads, _headDim, _hp.RmsNormEps);
                                PerHeadPureRmsNorm(kn, _numKvHeads, _headDim, _hp.RmsNormEps);
                            }
                            else
                            {
                                ApplyQkNorm(qn, kn, layer);
                            }
                        }
                        caches[n].Append(layer,
                            new ReadOnlySpan<float>(kn, kvDim),
                            new ReadOnlySpan<float>(vn, kvDim));
                        caches[n].IncrementPosition(); // _length = pos+1
                        if (profDecodeB)
                        {
                            long now = System.Diagnostics.Stopwatch.GetTimestamp();
                            bRope += now - bSeq;
                            bSeq = now;
                        }
                        Copy(_q, qn, qDim);
                        Attention(caches[n], layer, pos);
                        Copy(batchAttnOut + (long)n * qDim, _attnOut, qDim);
                        if (profDecodeB) bAttn += System.Diagnostics.Stopwatch.GetTimestamp() - bSeq;
                    }
                    if (profDecodeB)
                    {
                        DecodeProfileTimers.Add(DecodeProfileTimers.Category.RoPE, bRope);
                        DecodeProfileTimers.Add(DecodeProfileTimers.Category.Attention, bAttn);
                        bStage = System.Diagnostics.Stopwatch.GetTimestamp();
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
                        Copy(h, batchNorm + (long)n * _embDim, _embDim);
                        SimdKernels.AddInPlace(h, batchResidual + (long)n * _embDim, _embDim);
                    }
                    if (profDecodeB)
                    {
                        DecodeProfileTimers.Add(DecodeProfileTimers.Category.OutProj,
                            System.Diagnostics.Stopwatch.GetTimestamp() - bStage);
                        bStage = System.Diagnostics.Stopwatch.GetTimestamp();
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
                    if (profDecodeB)
                        DecodeProfileTimers.Add(DecodeProfileTimers.Category.Ffn,
                            System.Diagnostics.Stopwatch.GetTimestamp() - bStage);
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

    /// <summary>
    /// Packed multi-sequence prefill (issue #183 Gap 2): processes one chunk of prompt
    /// tokens from each of S sequences in a single forward pass. All chunks are
    /// concatenated into one packed batch so every GEMM amortizes weight reads across
    /// the combined token count — the multi-prompt analogue of what
    /// <see cref="BatchForwardMulti"/> does for decode. Attention stays per-token
    /// against each sequence's own cache (varlen / cu_seqlens-style: no cross-sequence
    /// attention, no padding).
    ///
    /// SnapKV eviction is never applied here — chunked admission feeds this with
    /// startPos &gt; 0 segments where SnapKV scoring doesn't run; callers that want
    /// SnapKV must use whole-prompt <see cref="PrefillWithCache"/> instead (the engine
    /// gates on <see cref="SnapKvEnabled"/>).
    /// </summary>
    /// <param name="chunks">Per-sequence token chunk (each non-empty).</param>
    /// <param name="startPos">Per-sequence cache position at which its chunk begins.</param>
    /// <param name="caches">Per-sequence KV cache.</param>
    /// <param name="wantLogits">
    /// Per-sequence: compute logits for the chunk's last token (true for a sequence's
    /// final chunk; intermediate chunks skip the vocab projection).
    /// </param>
    /// <returns>Per-sequence logits array, null where <paramref name="wantLogits"/> was false.</returns>
    public float[]?[] PrefillPackedMulti(
        ReadOnlyMemory<int>[] chunks, int[] startPos, PagedKvCache[] caches, bool[] wantLogits)
    {
        if (_tqKvCache != null)
            throw new NotSupportedException("PrefillPackedMulti is not supported when TurboQuant KV cache is enabled.");
        if (_hp.IsMoE)
            throw new NotSupportedException("PrefillPackedMulti is not supported for MoE models.");
        if (_layerHeadDim is not null)
            throw new NotSupportedException(
                "gemma4 per-layer head_dim not yet supported on PrefillPackedMulti.");

        int S = chunks.Length;
        if (S == 0) return Array.Empty<float[]?>();
        if (startPos.Length != S || caches.Length != S || wantLogits.Length != S)
            throw new ArgumentException("chunks/startPos/caches/wantLogits lengths must match.");

        // Packed admission is normally the highest-throughput prefill route. It must not,
        // however, reintroduce Q8 activation quantisation for an all-control structural probe
        // which Prefill and PrefillWithCache deliberately keep on the F32 route. Such prompts
        // are rare and short; falling back for the whole affected packed batch avoids silently
        // changing numerical behaviour based solely on whether another request arrived nearby.
        for (int s = 0; s < S; s++)
        {
            if (!IsAllControlTokenPrompt(chunks[s].Span) && !IsSingleDistinctTokenPrompt(chunks[s].Span)) continue;
            var fallback = new float[]?[S];
            for (int i = 0; i < S; i++)
            {
                ReadOnlySpan<float> logits = PrefillWithCacheSequential(
                    chunks[i].ToArray(), caches[i], startPos[i]);
                if (wantLogits[i]) fallback[i] = logits.ToArray();
            }
            return fallback;
        }

        // Packed offsets: sequence s owns packed rows [off[s], off[s+1]).
        var off = new int[S + 1];
        for (int s = 0; s < S; s++)
        {
            if (chunks[s].IsEmpty)
                throw new ArgumentException($"Chunk for sequence {s} is empty.", nameof(chunks));
            off[s + 1] = off[s] + chunks[s].Length;
        }
        int N = off[S];

        int qDim = _numHeads * _headDim;
        int kvDim = _numKvHeads * _headDim;
        var batchHidden = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _embDim * sizeof(float)));
        var batchResidual = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _embDim * sizeof(float)));
        try
        {
            for (int s = 0; s < S; s++)
            {
                var span = chunks[s].Span;
                for (int i = 0; i < span.Length; i++)
                    EmbedTokenInto(span[i], batchHidden + (long)(off[s] + i) * _embDim, startPos[s] + i);
            }

            var batchNorm = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _embDim * sizeof(float)));
            var batchQ = (float*)NativeMemory.AllocZeroed((nuint)((long)N * qDim * sizeof(float)));
            var batchK = (float*)NativeMemory.AllocZeroed((nuint)((long)N * kvDim * sizeof(float)));
            var batchV = (float*)NativeMemory.AllocZeroed((nuint)((long)N * kvDim * sizeof(float)));
            var batchAttnOut = (float*)NativeMemory.AllocZeroed((nuint)((long)N * qDim * sizeof(float)));
            var batchFfnGate = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _intermDim * sizeof(float)));
            var batchFfnUp = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _intermDim * sizeof(float)));
            try
            {
                for (int layer = 0; layer < _hp.NumLayers; layer++)
                {
                    // Rewind each cache so this layer's appends land at startPos[s]
                    // (Append advances the shared position counter every layer; same
                    // per-layer soft reset PrefillCore does for its single cache).
                    for (int s = 0; s < S; s++)
                        caches[s].TruncateTo(startPos[s]);

                    var normW = GetNormWeight(_attnNorm[layer]);
                    for (int n = 0; n < N; n++)
                    {
                        Copy(batchResidual + (long)n * _embDim, batchHidden + (long)n * _embDim, _embDim);
                        SimdKernels.RmsNorm(batchNorm + (long)n * _embDim,
                            batchHidden + (long)n * _embDim, normW, _embDim, _hp.RmsNormEps);
                    }

                    // allowBlas: false throughout PrefillPackedMulti -- N here is the SUM of
                    // multiple independent sessions' chunk lengths, not positions within one
                    // prompt. Letting that combined size decide BLAS-vs-tiered kernel choice would
                    // make a session's own numerics depend on how many OTHER, unrelated sessions
                    // happened to be packed alongside it in the same call -- see allowBlas's doc on
                    // SimdKernels.MatMulBatched and docs/031-concurrent-decode-batch-tier-divergence-bug.md.
                    MatMulBatchedCached(batchQ, in _wq[layer], batchNorm, N, qDim, _embDim, allowBlas: false);
                    MatMulBatchedCached(batchK, in _wk[layer], batchNorm, N, kvDim, _embDim, allowBlas: false);
                    MatMulBatchedCached(batchV, in _wv[layer], batchNorm, N, kvDim, _embDim, allowBlas: false);

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

                    // Per-token: RoPE at the token's own absolute position, KV append
                    // into the token's own cache, causal attention over that cache only.
                    for (int s = 0; s < S; s++)
                    {
                        for (int i = 0; i < chunks[s].Length; i++)
                        {
                            int n = off[s] + i;
                            int pos = startPos[s] + i;
                            float* qn = batchQ + (long)n * qDim;
                            float* kn = batchK + (long)n * kvDim;
                            float* vn = batchV + (long)n * kvDim;

                            if (_hasQkNorm && !_hp.UseL2QkNorm)
                            {
                                ApplyQkNorm(qn, kn, layer);
                            }
                            if (useRoPE)
                            {
                                ApplyRope(qn, pos, _numHeads);
                                ApplyRope(kn, pos, _numKvHeads);
                            }
                            if (_hasQkNorm && _hp.UseL2QkNorm && useRoPE)
                            {
                                PerHeadPureRmsNorm(qn, _numHeads, _headDim, _hp.RmsNormEps);
                                PerHeadPureRmsNorm(kn, _numKvHeads, _headDim, _hp.RmsNormEps);
                            }

                            caches[s].Append(layer,
                                new ReadOnlySpan<float>(kn, kvDim),
                                new ReadOnlySpan<float>(vn, kvDim));
                            caches[s].IncrementPosition();

                            Copy(_q, qn, qDim);
                            Attention(caches[s], layer, pos);
                            Copy(batchAttnOut + (long)n * qDim, _attnOut, qDim);
                        }
                    }

                    MatMulBatchedCached(batchNorm, in _wo[layer], batchAttnOut, N, _embDim, qDim, allowBlas: false);
                    if (_hasAttnOutputBias)
                    {
                        for (int n = 0; n < N; n++)
                            SimdKernels.AddInPlace(batchNorm + (long)n * _embDim, _bo[layer], _embDim);
                    }
                    for (int n = 0; n < N; n++)
                    {
                        float* h = batchHidden + (long)n * _embDim;
                        Copy(h, batchNorm + (long)n * _embDim, _embDim);
                        SimdKernels.AddInPlace(h, batchResidual + (long)n * _embDim, _embDim);
                    }

                    var ffnNormW = GetNormWeight(_ffnNorm[layer]);
                    for (int n = 0; n < N; n++)
                    {
                        Copy(batchResidual + (long)n * _embDim, batchHidden + (long)n * _embDim, _embDim);
                        SimdKernels.RmsNorm(batchNorm + (long)n * _embDim,
                            batchHidden + (long)n * _embDim, ffnNormW, _embDim, _hp.RmsNormEps);
                    }
                    MatMulBatchedCached(batchFfnGate, in _wGate[layer], batchNorm, N, _intermDim, _embDim, allowBlas: false);
                    MatMulBatchedCached(batchFfnUp, in _wUp[layer], batchNorm, N, _intermDim, _embDim, allowBlas: false);
                    for (int n = 0; n < N; n++)
                        SimdKernels.SiLuMul(batchFfnGate + (long)n * _intermDim,
                            batchFfnUp + (long)n * _intermDim, _intermDim);
                    MatMulBatchedCached(batchNorm, in _wDown[layer], batchFfnGate, N, _embDim, _intermDim, allowBlas: false);
                    for (int n = 0; n < N; n++)
                    {
                        float* h = batchHidden + (long)n * _embDim;
                        Copy(h, batchNorm + (long)n * _embDim, _embDim);
                        SimdKernels.AddInPlace(h, batchResidual + (long)n * _embDim, _embDim);
                    }
                }

                // Leave every cache at its post-chunk length for subsequent decode/chunks.
                for (int s = 0; s < S; s++)
                    caches[s].TruncateTo(startPos[s] + chunks[s].Length);
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

            // Final norm + vocab projection only for sequences whose chunk completes
            // their prompt — intermediate chunks never need logits.
            var outNormW = GetNormWeight(_outputNorm);
            var result = new float[]?[S];
            for (int s = 0; s < S; s++)
            {
                if (!wantLogits[s]) continue;
                float* lastHidden = batchHidden + (long)(off[s + 1] - 1) * _embDim;
                SimdKernels.RmsNorm(lastHidden, lastHidden, outNormW, _embDim, _hp.RmsNormEps);
                FusedMatVec(_logits, _outputWeight, lastHidden, _hp.VocabSize, _embDim);
                var arr = new float[_hp.VocabSize];
                new ReadOnlySpan<float>(_logits, _hp.VocabSize).CopyTo(arr);
                result[s] = arr;
            }
            return result;
        }
        finally
        {
            NativeMemory.Free(batchHidden);
            NativeMemory.Free(batchResidual);
        }
    }


    public void Dispose()
    {
        // Idempotency guard (matches the codebase-wide convention elsewhere, e.g.
        // OpenTail.Stingray.Core.ModelHandle / SharedModelCache): InferenceEngine.DisposeCore
        // calls _fwd.Dispose() explicitly AND separately disposes every item in its _owned
        // array, which also contains this same ForwardPass instance (InferenceEngineLoader adds
        // it to owned[] when constructing the CPU dense path) -- without this guard, every
        // NativeMemory.Free() call below runs twice, a double-free that corrupts the native heap
        // (bugstofix.md, discovered via docs/032's Phase 5 real-model concurrency work: even a
        // single plain, non-batching InferenceEngine crashed on Dispose() after ANY real model
        // load, with or without ever generating -- ContinuousBatchingEngine's OwnedDisposableEngine
        // never hit this because its DrainedOnDispose guard is a different, unrelated check).
        if (_disposed) return;
        _disposed = true;

        NativeMemory.Free(_hidden);
        NativeMemory.Free(_residual);
        NativeMemory.Free(_normBuf);
        NativeMemory.Free(_q);
        NativeMemory.Free(_k);
        NativeMemory.Free(_v);
        NativeMemory.Free(_attnOut);
        NativeMemory.Free(_ffnGate);
        NativeMemory.Free(_ffnUp);
        NativeMemory.Free(_logits);
        NativeMemory.Free(_attnScores);
        NativeMemory.Free(_ropeCosTable);
        NativeMemory.Free(_ropeSinTable);
        if (_posEmbdScratch != null) NativeMemory.Free(_posEmbdScratch);
        if (_ropeCosTableSwa != null) NativeMemory.Free(_ropeCosTableSwa);
        if (_ropeSinTableSwa != null) NativeMemory.Free(_ropeSinTableSwa);
        _taps?.Dispose();

        foreach (var ptr in _normCache.Values)
            NativeMemory.Free((void*)ptr);
        _normCache.Clear();

        foreach (var ptr in _q4kx8Cache.Values)
            NativeMemory.Free((void*)ptr);
        _q4kx8Cache.Clear();
        _q4kx8CacheUsedBytes = 0;

        foreach (var ptr in _dequantWeightCache.Values)
            NativeMemory.Free((void*)ptr);
        _dequantWeightCache.Clear();

        if (_hasAttnBias)
        {
            for (int i = 0; i < _hp.NumLayers; i++)
            {
                if (_bq[i] != null) NativeMemory.Free(_bq[i]);
                if (_bk[i] != null) NativeMemory.Free(_bk[i]);
                if (_bv[i] != null) NativeMemory.Free(_bv[i]);
                if (_bo[i] != null) NativeMemory.Free(_bo[i]);
            }
        }

        if (_hasQkNorm && !_hp.UseL2QkNorm)
        {
            for (int i = 0; i < _hp.NumLayers; i++)
            {
                NativeMemory.Free(_qNorm[i]);
                NativeMemory.Free(_kNorm[i]);
            }
        }

        if (_hasNormBias)
        {
            for (int i = 0; i < _hp.NumLayers; i++)
            {
                if (_bAttnNorm![i] != null) NativeMemory.Free(_bAttnNorm[i]);
                // Falcon has no ffn_norm.bias tensor: _bFfnNorm[i] aliases _bAttnNorm[i] (see
                // the constructor's fallback) rather than owning an independent allocation —
                // freeing it too would double-free the same pointer. Every other architecture
                // with HasNormBias allocates the two independently, so the pointer-equality
                // check is the discriminator, not the architecture string.
                if (_bFfnNorm![i] != null && _bFfnNorm[i] != _bAttnNorm[i])
                    NativeMemory.Free(_bFfnNorm[i]);
            }
            if (_bOutputNorm != null) NativeMemory.Free(_bOutputNorm);
        }
        if (_hasFfnBias)
        {
            for (int i = 0; i < _hp.NumLayers; i++)
            {
                if (_bFfnUp![i] != null) NativeMemory.Free(_bFfnUp[i]);
                if (_bFfnDown![i] != null) NativeMemory.Free(_bFfnDown[i]);
            }
        }
        if (_parAttnOut != null) NativeMemory.Free(_parAttnOut);

        if (_hp.IsMoE)
        {
            NativeMemory.Free(_routerLogits);
            NativeMemory.Free(_sharedOut);
            NativeMemory.Free(_expertGate);
            NativeMemory.Free(_expertUp);
            NativeMemory.Free(_moeDownTemp);
            FreeMoeBatchScratch();
        }

        if (_hp.HasPerLayerTokenEmbd)
        {
            NativeMemory.Free(_perLayerModelProj);
            NativeMemory.Free(_pleRowBuf);
            NativeMemory.Free(_projPerLayer);
            NativeMemory.Free(_pleX);
            NativeMemory.Free(_pleY);
        }

        _kvCache.Dispose();
    }
}
