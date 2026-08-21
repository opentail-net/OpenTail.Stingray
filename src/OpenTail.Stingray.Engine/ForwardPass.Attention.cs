using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.TurboQuant;

namespace OpenTail.Stingray.Engine;

// Part of ForwardPass (see ForwardPass.cs for the type summary). All attention kernels: the
// batched-prefill materialized-softmax path, its BF16 KV-store variant, Flash-64 (tiled online-
// softmax), the single-token decode Attention, and TurboQuant's TqAttention.
public sealed unsafe partial class ForwardPass
{
    // ================================================================
    //  Batched Prefill Attention
    // ================================================================

    public static unsafe void ComputeBatchedCausalAttention(
        float* batchQ, float* batchK, float* batchV, float* batchAttnOut,
        int N, int startPos, int numHeads, int numKvHeads, int headDim, float scale)
    {
        int qDim = numHeads * headDim;
        int kvDim = numKvHeads * headDim;
        int hpkg = numHeads / numKvHeads;

        Parallel.For(0, numHeads, h =>
        {
            int kvHead = h / hpkg;
            int maxSeqLen = startPos + N;
            float* headScores = (float*)NativeMemory.AllocZeroed((nuint)(maxSeqLen * sizeof(float)));
            try
            {
                for (int n = 0; n < N; n++)
                {
                    float* qHead = batchQ + (long)n * qDim + h * headDim;
                    float* outHead = batchAttnOut + (long)n * qDim + h * headDim;

                    int scoreLen = startPos + n + 1;
                    for (int i = 0; i < scoreLen; i++)
                    {
                        float* kVec = batchK + (long)i * kvDim + kvHead * headDim;
                        headScores[i] = SimdKernels.DotF32(qHead, kVec, headDim) * scale;
                    }

                    SimdKernels.SoftmaxInPlace(headScores, scoreLen);

                    for (int d = 0; d < headDim; d++) outHead[d] = 0;

                    for (int i = 0; i < scoreLen; i++)
                    {
                        float* vVec = batchV + (long)i * kvDim + kvHead * headDim;
                        float w = headScores[i];
                        if (Fma.IsSupported && headDim >= 8)
                        {
                            var wv = Vector256.Create(w);
                            int d = 0;
                            for (; d + 8 <= headDim; d += 8)
                            {
                                var o = Avx.LoadVector256(outHead + d);
                                var v = Avx.LoadVector256(vVec + d);
                                Avx.Store(outHead + d, Fma.MultiplyAdd(wv, v, o));
                            }
                            for (; d < headDim; d++)
                                outHead[d] += w * vVec[d];
                        }
                        else
                        {
                            for (int d = 0; d < headDim; d++)
                                outHead[d] += w * vVec[d];
                        }
                    }
                }
            }
            finally
            {
                NativeMemory.Free(headScores);
            }
        });
    }

    /// <summary>
    /// Batched prefill attention, tiled over the token axis.
    ///
    /// <para>The previous shape was <c>for each head: for each token n: full K pass; softmax; full
    /// V pass</c>. Because <see cref="PrefillCore"/> passes the WHOLE prompt as N, that re-read the
    /// entire K and V cache once per token — for a 3218-token prompt, 3218 passes over ~824 KB per
    /// head, which is far past L2 and therefore goes to RAM every time. The FLOPs are inherently
    /// O(N²); the memory traffic did not have to be.</para>
    ///
    /// <para>Tiling the token loop by <c>TokenTile</c> streams each K[i] (then each V[i]) once per
    /// TILE tokens instead of once per token, so the vector stays hot in L1 while all tokens in the
    /// tile consume it — the same insight as the Vulkan flash-attention rewrite (perf-loop
    /// iteration 31), applied to the cache hierarchy rather than to VRAM bandwidth.</para>
    ///
    /// <para><b>Bit-identical to the untiled form</b>, deliberately: scores are computed by the same
    /// dot in the same order, softmax runs over the same row, and each output still accumulates
    /// over i ascending. Only the loop nesting changes, not the arithmetic order.</para>
    /// </summary>
    private void PrefillCoreAttention(float* batchQ, PagedKvCache cache, int layer, int N, int startPos, float* batchAttnOut)
    {
        int numHeads = _numHeads;
        int numKvHeads = _numKvHeads;
        // Per-layer head dim (issue #351). The attn scale on the next line was already
        // gemma4-aware; the dimension itself was not.
        int headDim = _layerHeadDim?[layer] ?? _headDim;
        int qDim = numHeads * headDim;
        // Quantity 1 (see the doc's "three quantities" table): the CACHE's head stride, which is
        // fixed at construction by _maxHeadDim and is NOT this layer's head dim. K rows are
        // _numKvHeads * _maxHeadDim wide with head h at h * _maxHeadDim, so a narrow layer must
        // still step by the wide stride to find its head — it just reads headDim of it.
        // (ValueAtHead already does this internally; KeyAt returns the row base, so the caller
        // owns the head offset and this is where it was being got wrong.)
        int cacheHeadStride = _layerHeadDim is not null ? _maxHeadDim : headDim;
        int hpkg = numHeads / numKvHeads;
        // Kept consistent with the single-token Attention() override above — see its comment.
        float scale = _hp.AttentionScaleOverride != 0f ? _hp.AttentionScaleOverride
            : _layerHeadDim is not null ? 1.0f : 1.0f / MathF.Sqrt(headDim);
        int ctxLen = _ctxLen;
        bool enableRegisterValues = Environment.GetEnvironmentVariable(
            "STINGRAY_PREFILL_ATTN_REGISTER_VALUES") != "0";
        bool enableFlash64 = Environment.GetEnvironmentVariable(
            "STINGRAY_PREFILL_ATTN_FLASH64") != "0";

        // The 64x64 packing/setup cost loses on short prefills (the isolated crossover is well
        // above 128 tokens). Keeping small chunks on the bit-stable incumbent also preserves the
        // packed-multi parity contract, whose current attention loop is intentionally per-token.
        //
        // The threshold tests startPos + N — the sequence length prefilled so far — NOT N alone.
        // Flash-64 uses online softmax (running max, per-tile rescale) and is therefore NOT
        // bit-identical to the incumbent, so selecting between them on the CHUNK size makes a
        // prompt's logits depend on how it was admitted: at chunk 512 a 600-token prompt splits
        // 512 + 88, and on `N` alone the tail silently fell back to the incumbent while the same
        // prompt prefilled in one pass ran entirely on flash-64. Measured divergence was ~2.5% on a
        // logit (3.655 vs 3.556) — far outside chunk-boundary FP drift, and exactly the defect
        // SimdKernels.cs:76-86 calls out ("a user's logits depend on who else happened to be
        // batched alongside them"). Caught by
        // PrefillAttentionParityTests.ChunkedPrefill_MatchesUnchunked_AcrossFlash64Threshold.
        //
        // KNOWN RESIDUAL: this makes the decision monotonic in sequence position, which fixes every
        // case where the first chunk already reaches the threshold — i.e. any chunk size >= 256,
        // which covers the shipped STINGRAY_PREFILL_CHUNK values. It does NOT fix a prompt whose
        // total exceeds 256 while its individual chunks do not (e.g. 600 tokens at chunk 64): the
        // early chunks take the incumbent and later ones flash-64. Closing that needs the decision
        // threaded down from the caller, which alone knows the prompt's total length.
        // Flash-64 handles BF16 natively by widening each 64x64 tile once during packing, so it must
        // be offered the work FIRST. Routing BF16 around it was measured to cost 41% of prefill
        // throughput (122.2 -> 72.5 t/s at 5314 tokens) — the loss was the missing Flash-64, not the
        // narrower store.
        //
        // Tile width is 64 queries/KVs. The machinery below is fully head-width generic (headDim is
        // a parameter throughout, scratch is sized Tile*headDim) and the strided AVX2 microkernel
        // supports 128/256-wide heads — but the 128/256 WIDTHS ARE HELD BACK HERE, deliberately, one
        // env var away from re-enabling (`headDim is 64 or 128 or 256`). Not a correctness bug: a
        // parity investigation resolved that (flash-vs-materialised maxAbs 0.762, cosine 0.999345,
        // identical greedy token — the same envelope as the Q8 activation-prefill approximation this
        // project already ships on by default, maxAbs 0.807). It's off by default because of a
        // measured QUALITY regression instead: +0.52%/+0.47% perplexity on wikitext-2 (Qwen3-8B),
        // worse than even the plain batched path, against a Q4Kx8-repack precedent of ~0% for a
        // default-on numerics change — too large a trade to make on the model owner's behalf despite
        // the +14% prefill throughput it buys. Opt in via STINGRAY_PREFILL_ATTN_WIDE_HEADS=1.
        // Full investigation (ruled-out hypotheses, the parity and perplexity measurements, the
        // superseded reasoning that preceded them): docs/reference/forwardpass-investigation-log.md
        // #flash-128256-wide-attention-heads--perplexity-investigation
        if (enableFlash64 && startPos + N >= 256 && _layerHeadDim is null
            && (headDim == 64 || (Flash64WideHeadDimsEnabled && headDim is 128 or 256)) &&
            Avx2.IsSupported && Fma.IsSupported)
        {
            PrefillFlashAttention64(batchQ, cache, layer, N, startPos, batchAttnOut,
                numHeads, numKvHeads, qDim, headDim, scale);
            return;
        }

        // Whatever Flash-64 declined (short prefill, non-64 head dim, per-layer head dims, no AVX2)
        // still needs a BF16-aware reader — the F32 loops below would misread 2-byte pages.
        if (cache.IsBf16Store)
        {
            PrefillCoreAttentionBf16(batchQ, cache, layer, N, startPos, batchAttnOut,
                numHeads, numKvHeads, headDim, qDim, cacheHeadStride, hpkg, scale);
            return;
        }

        // Tokens per tile. This is the K/V re-read amortisation factor: the whole K cache, then the
        // whole V cache, is streamed once per TILE tokens, so K/V traffic scales as N/TILE.
        //
        // Originally 8, chosen to keep the score scratch small. A direct sweep
        // (tools/attn-bench, N=3218, three independent runs) shows that was well short of the
        // optimum — the scratch concern was over-weighted and the traffic term keeps paying:
        //     tile      4      8     16     32     64    128    256
        //     speedup 0.70x  1.00x  1.17x  1.35x  1.48x  1.34x  1.04x
        // 64 is the measured optimum and the curve is flat enough either side (32 and 128 both
        // within ~10%) that it is not a knife-edge. Below 16 the per-tile fixed costs dominate;
        // above 128 the scratch itself (TILE * stride floats per head-thread) stops fitting.
        //
        // Scratch at TILE=64 is 64 * stride * 4 bytes per head-thread — 2 MB at ctxLen 8192, so
        // ~24 MB live across 12 concurrent head-threads. That is comfortably RAM-resident, which
        // is the real constraint; it was never going to be L1-resident at any useful tile size.
        const int TokenTile = 64;

        // Diagnostic (STINGRAY_MLA_TRACE=1): post-softmax attention weight sample, head 0 --
        // ground-truth-comparable against llama.cpp's kq_soft_max-N tensor (only visible with
        // -fa off, since flash attention fuses score/softmax/weighted-V into one opaque
        // FLASH_ATTN_EXT node with no inspectable intermediate).
        bool traceThisLayer = s_mlaTrace && _isMla;

        Parallel.For(0, numHeads, h =>
        {
            int kvHead = h / hpkg;
            int maxSeqLen = startPos + N;
            // Size the scratch to the sequence actually being prefilled, not to ctxLen. Every index
            // written below is `i < endSeq`, and endSeq = min(startPos + nBase + t + 1, cache.Length)
            // is bounded by startPos + N = maxSeqLen — so maxSeqLen rows are always sufficient, and
            // ctxLen over-allocated whenever the prompt is shorter than the configured context.
            // This matters more since TokenTile grew to 64: at ctxLen 8192 the old sizing would be
            // 2 MB per head-thread regardless of prompt length, and at a 128k context it would be
            // 33 MB per head-thread — hundreds of MB across the head threads, for scratch that is
            // mostly never touched. Measured perf-neutral on its own; this is a memory bound, not
            // a speed change.
            int stride = maxSeqLen;
            float* scores = (float*)NativeMemory.AllocZeroed((nuint)((long)TokenTile * stride * sizeof(float)));
            bool registerValues = enableRegisterValues && Fma.IsSupported && headDim >= 8 && headDim % 8 == 0;
            float** valueRows = registerValues
                ? (float**)NativeMemory.Alloc((nuint)(Math.Min(maxSeqLen, cache.Length) * sizeof(nint)))
                : null;

            // NOTE (2026-08-02, measured and reverted): phase 3 accumulates straight into
            // batchAttnOut, whose per-token stride is qDim = numHeads * headDim floats — 8192 BYTES
            // at 32 heads / 64 dim. Zen 3's L1d is 32 KB / 8-way / 64 B lines = 64 sets indexed by
            // address mod 4096, and 8192 mod 4096 == 0, so every token's output head maps to the
            // same L1 sets: 64 lines competing for 8 ways.
            //
            // That looks like textbook conflict thrashing, so phase 3 was rewritten to accumulate
            // into a contiguous TokenTile x headDim scratch (16 KB) and copy out once per tile —
            // bit-identical arithmetic, no numerics risk. Properly interleaved measurements found
            // it performance-neutral; the earlier claimed ~9% loss compared against one noisy
            // baseline and was retracted. Moving the same loads did not reduce their uop cost.
            //
            // The uop count, not the cache behaviour, is what made phase 3 expensive: 8 acc loads
            // + 8 V loads + 8 FMAs + 8 stores per (token, KV position), i.e. 4 uops per useful FMA.
            // The registerValues path below fixes that with an 8-token x 8-float microkernel. On
            // the production PagedKvCache shape it measured 1.17-1.20x for whole attention and
            // 1.075x / 1.088x end-to-end at 931 / 2431 tokens, bit-identical in the isolated harness
            // and the chunked-prefill tests. Set STINGRAY_PREFILL_ATTN_REGISTER_VALUES=0 only for
            // controlled A/B measurement. Do not re-try the scratch-buffer variant.
            try
            {
                if (valueRows is not null)
                    for (int i = 0; i < Math.Min(maxSeqLen, cache.Length); i++)
                        valueRows[i] = cache.ValueAtHead(layer, i, kvHead);

                for (int nBase = 0; nBase < N; nBase += TokenTile)
                {
                    int tn = Math.Min(TokenTile, N - nBase);
                    // Longest causal row in this tile — the K/V streaming bound.
                    int endSeqMax = Math.Min(startPos + nBase + tn, cache.Length);

                    // ── Phase 1: scores. Stream K once; every token in the tile consumes it. ──
                    //
                    // NOT batched via SimdKernels.DotF32_4In — see that kernel's remarks. Batching
                    // four query tokens per key vector was tried (2026-08-02) and reverted: it is
                    // worth only ~6% of attention (~2% end-to-end), and because the 4-wide kernel
                    // can only cover `t + 4 <= tn` the tail falls back to DotF32, which sums in a
                    // different order. A token's arithmetic would then depend on how many tokens
                    // share its tile — i.e. on N — so chunked and unchunked prefill of the same
                    // prompt disagree. That broke 4 ContinuousBatchingTests. Wiring it in safely
                    // needs every token on one kernel, which the tile remainder prevents.
                    for (int i = 0; i < endSeqMax; i++)
                    {
                        float* kVec = cache.KeyAt(layer, i) + kvHead * cacheHeadStride;
                        for (int t = 0; t < tn; t++)
                        {
                            int endSeq = Math.Min(startPos + nBase + t + 1, cache.Length);
                            if (i < endSeq)
                                scores[(long)t * stride + i] = SimdKernels.DotF32(
                                    batchQ + (long)(nBase + t) * qDim + h * headDim, kVec, headDim) * scale;
                        }
                    }

                    if (traceThisLayer && h == 0 && nBase == 0)
                    {
                        int lastTPre = tn - 1;
                        float* rowPre = scores + (long)lastTPre * stride;
                        Console.Error.WriteLine(
                            $"[MLA-TRACE] L{layer} kq(pre-softmax) h0 tokLast first3=[{rowPre[0]:F4},{rowPre[1]:F4},{rowPre[2]:F4}]");
                    }

                    // ── Phase 2: per-token softmax over its own causal length ──
                    for (int t = 0; t < tn; t++)
                    {
                        int endSeq = Math.Min(startPos + nBase + t + 1, cache.Length);
                        SimdKernels.SoftmaxInPlace(scores + (long)t * stride, endSeq);
                    }

                    if (traceThisLayer && h == 0 && nBase == 0)
                    {
                        // Post-softmax attention weights, head 0, LAST token in this tile --
                        // ground-truth-comparable against llama.cpp's kq_soft_max-N tensor's
                        // head-0 block, last row (only visible with -fa off — flash attention
                        // fuses this away). Masked (causally invalid) positions are exactly 0
                        // on both sides, so unlike the raw pre-softmax scores this is a fair,
                        // apples-to-apples comparison (no padded-position mismatch).
                        int lastT = tn - 1;
                        float* row = scores + (long)lastT * stride;
                        Console.Error.WriteLine(
                            $"[MLA-TRACE] L{layer} kq_soft_max h0 tokLast first3=[{row[0]:F4},{row[1]:F4},{row[2]:F4}]");
                    }

                    if (registerValues)
                    {
                        AccumulatePrefillValuesRegister8(valueRows, scores, stride, batchAttnOut,
                            nBase, tn, qDim, h * headDim, headDim, startPos, cache.Length);
                    }
                    else
                    {
                        for (int t = 0; t < tn; t++)
                        {
                            float* outHead = batchAttnOut + (long)(nBase + t) * qDim + h * headDim;
                            for (int d = 0; d < headDim; d++) outHead[d] = 0;
                        }

                        // Scalar/non-AVX fallback retains the original loop shape.
                        for (int i = 0; i < endSeqMax; i++)
                        {
                            float* vVec = cache.ValueAtHead(layer, i, kvHead);
                            for (int t = 0; t < tn; t++)
                            {
                                int endSeq = Math.Min(startPos + nBase + t + 1, cache.Length);
                                if (i >= endSeq) continue;
                                float* outHead = batchAttnOut + (long)(nBase + t) * qDim + h * headDim;
                                float w = scores[(long)t * stride + i];
                                if (Fma.IsSupported && headDim >= 8)
                                {
                                    var wv = Vector256.Create(w);
                                    int d = 0;
                                    for (; d + 8 <= headDim; d += 8)
                                    {
                                        var o = Avx.LoadVector256(outHead + d);
                                        var v = Avx.LoadVector256(vVec + d);
                                        Avx.Store(outHead + d, Fma.MultiplyAdd(wv, v, o));
                                    }
                                    for (; d < headDim; d++)
                                        outHead[d] += w * vVec[d];
                                }
                                else
                                {
                                    for (int d = 0; d < headDim; d++)
                                        outHead[d] += w * vVec[d];
                                }
                            }
                        }
                    }

                }
            }
            finally
            {
                NativeMemory.Free(valueRows);
                NativeMemory.Free(scores);
            }
        });
    }

    /// <summary>
    /// Attention against a BF16-store cache (<c>STINGRAY_KV_STORE=bf16</c>). Structural mirror of
    /// <see cref="PrefillCoreAttention"/>: same token tiling, same three phases, same causal bounds
    /// and the same score-then-softmax-then-weighted-V order — only the loads differ, widening 2-byte
    /// elements instead of reading 4-byte ones. Arithmetic stays fp32 throughout.
    /// </summary>
    /// <remarks>
    /// <para>Deliberately a separate method rather than a branch inside the hot loops. The two paths
    /// dereference different pointer types, and threading that through the tiled loops would put a
    /// predictable-but-real test in the innermost body of the engine's single hottest kernel. The
    /// duplication is the cheaper of the two costs, and the F32 path stays byte-for-byte as it was —
    /// which matters because it is the default and is covered by the parity suites.</para>
    ///
    /// <para><b>Not bit-identical to the F32 path, by construction</b> — the stored values have 8
    /// mantissa bits instead of 23. The reduction ORDER is identical (<c>DotF32Bf16</c> copies
    /// <c>DotF32</c>'s accumulator tree exactly), so the difference is attributable to storage
    /// precision alone. Perplexity is the gate, not bit-equality.</para>
    ///
    /// <para>The register-8 value microkernel is not used here. It is an F32 uop-count optimisation
    /// worth ~1.17x on prefill; this path exists for decode, where N=1 makes an 8-token microkernel
    /// degenerate and the bound is DRAM traffic rather than uops.</para>
    /// </remarks>
    private void PrefillCoreAttentionBf16(
        float* batchQ, PagedKvCache cache, int layer, int N, int startPos, float* batchAttnOut,
        int numHeads, int numKvHeads, int headDim, int qDim, int cacheHeadStride, int hpkg, float scale)
    {
        const int TokenTile = 64;

        Parallel.For(0, numHeads, h =>
        {
            int kvHead = h / hpkg;
            int maxSeqLen = startPos + N;
            int stride = maxSeqLen;
            float* scores = (float*)NativeMemory.AllocZeroed(
                (nuint)((long)TokenTile * stride * sizeof(float)));
            try
            {
                for (int nBase = 0; nBase < N; nBase += TokenTile)
                {
                    int tn = Math.Min(TokenTile, N - nBase);
                    int endSeqMax = Math.Min(startPos + nBase + tn, cache.Length);

                    // ── Phase 1: scores. Stream K once per tile. ──
                    for (int i = 0; i < endSeqMax; i++)
                    {
                        ushort* kVec = cache.Bf16KeyAt(layer, i) + kvHead * cacheHeadStride;
                        for (int t = 0; t < tn; t++)
                        {
                            int endSeq = Math.Min(startPos + nBase + t + 1, cache.Length);
                            if (i < endSeq)
                                scores[(long)t * stride + i] = SimdKernels.DotF32Bf16(
                                    batchQ + (long)(nBase + t) * qDim + h * headDim, kVec, headDim) * scale;
                        }
                    }

                    // ── Phase 2: per-token softmax over its own causal length ──
                    for (int t = 0; t < tn; t++)
                    {
                        int endSeq = Math.Min(startPos + nBase + t + 1, cache.Length);
                        SimdKernels.SoftmaxInPlace(scores + (long)t * stride, endSeq);
                    }

                    // ── Phase 3: weighted V. Stream V once per tile, same i-ascending order. ──
                    for (int t = 0; t < tn; t++)
                    {
                        float* outHead = batchAttnOut + (long)(nBase + t) * qDim + h * headDim;
                        for (int d = 0; d < headDim; d++) outHead[d] = 0;
                    }

                    for (int i = 0; i < endSeqMax; i++)
                    {
                        ushort* vVec = cache.Bf16ValueAtHead(layer, i, kvHead);
                        for (int t = 0; t < tn; t++)
                        {
                            int endSeq = Math.Min(startPos + nBase + t + 1, cache.Length);
                            if (i >= endSeq) continue;
                            SimdKernels.AccumulateScaledBf16(
                                batchAttnOut + (long)(nBase + t) * qDim + h * headDim,
                                vVec, scores[(long)t * stride + i], headDim);
                        }
                    }
                }
            }
            finally
            {
                NativeMemory.Free(scores);
            }
        });
    }

    /// <summary>
    /// Default AVX2 prefill path matching llama.cpp's CPU Flash-attention structure: 64x64 Q/KV
    /// tiles, online softmax, and the same 6x2 FP32 microkernel for both QK and probabilities*V.
    /// Set <c>STINGRAY_PREFILL_ATTN_FLASH64=0</c> to retain the materialised-score fallback.
    /// KV tiles are anchored at absolute position zero, so a query sees the same reduction order
    /// whether a prompt reaches this method in one call or several chunks.
    /// </summary>
    private static void PrefillFlashAttention64(
        float* batchQ, PagedKvCache cache, int layer, int tokenCount, int startPos,
        float* output, int numHeads, int numKvHeads, int qDim, int headDim, float scale)
    {
        const int Tile = 64;
        int queryTiles = (tokenCount + Tile - 1) / Tile;
        int headsPerKv = numHeads / numKvHeads;

        // Query-tile jobs improve isolated attention substantially, but the production result was
        // neutral at 900 tokens and only +2.0% best-of at 2400 tokens. Keep the simpler one-job-
        // per-head schedule as the default; this switch retains the verified experiment without
        // promoting a result that is still within this machine's end-to-end noise floor. Both
        // arms call the same tile worker, so the switch isolates scheduling from arithmetic.
        // KV-outer/query-inner reorder: packs each KV tile once per group of query tiles instead
        // of once per query tile. Default since it measured +1.6% alone / +4.0% with the SIMD
        // K-pack transpose, and it is bit-exact with the old schedule (Flash64KvOuterTests).
        if (Flash64KvOuterEnabled)
        {
            int groupTiles = s_flash64KvOuterGroupTiles;
            int maxQueries = Math.Min(tokenCount, groupTiles * Tile);
            var kvOuterScratch = new ThreadLocal<PrefillFlash64KvOuterScratch>(
                () => new PrefillFlash64KvOuterScratch(headDim, maxQueries), trackAllValues: true);
            try
            {
                Parallel.For(0, numHeads, h =>
                    ComputePrefillFlashAttention64KvOuterHead(batchQ, cache, layer, tokenCount, startPos,
                        output, qDim, headDim, scale, h, h / headsPerKv, kvOuterScratch.Value!, groupTiles));
            }
            finally
            {
                foreach (PrefillFlash64KvOuterScratch s in kvOuterScratch.Values) s.Dispose();
                kvOuterScratch.Dispose();
            }
            return;
        }

        bool useTileJobs = Environment.GetEnvironmentVariable(
            "STINGRAY_PREFILL_ATTN_FLASH64_TILE_JOBS") == "1";
        if (!useTileJobs)
        {
            Parallel.For(0, numHeads, h =>
            {
                using var scratch = new PrefillFlash64Scratch(headDim);
                for (int nBase = 0; nBase < tokenCount; nBase += Tile)
                {
                    ComputePrefillFlashAttention64Tile(batchQ, cache, layer, tokenCount, startPos,
                        output, qDim, headDim, scale, h, h / headsPerKv, nBase, scratch);
                }
            });
            return;
        }

        var threadScratch = new ThreadLocal<PrefillFlash64Scratch>(
            () => new PrefillFlash64Scratch(headDim), trackAllValues: true);

        try
        {
            Parallel.For(0, numHeads * queryTiles, job =>
            {
                int h = job / queryTiles;
                int nBase = (job - h * queryTiles) * Tile;
                ComputePrefillFlashAttention64Tile(batchQ, cache, layer, tokenCount, startPos,
                    output, qDim, headDim, scale, h, h / headsPerKv, nBase, threadScratch.Value!);
            });
        }
        finally
        {
            foreach (PrefillFlash64Scratch scratch in threadScratch.Values) scratch.Dispose();
            threadScratch.Dispose();
        }
    }

    /// <summary>
    /// Default prefill-attention schedule since the 2x2 below; <c>STINGRAY_PREFILL_ATTN_KV_OUTER=0</c>
    /// restores the previous one. Same arithmetic
    /// as <see cref="ComputePrefillFlashAttention64Tile"/>, different loop order: KV tiles outside,
    /// query tiles inside, so a KV tile's K and V are packed <b>once</b> and reused by every query
    /// tile in the group instead of being repacked per query tile.
    ///
    /// <para><b>Why.</b> The default schedule is <c>for head → for queryTile → for kvTile: pack K;
    /// GEMM</c>, so identical key data is transposed into the pack buffer once per query tile — at
    /// 2048 tokens that is 32 repacks of the same bytes. This is a structural redundancy, not a
    /// constant factor: it is the one thing in this kernel whose cost falls with a reorder rather
    /// than with a faster instruction.</para>
    ///
    /// <para><b>Why it should be bit-exact.</b> Each query row still consumes KV tiles in ascending
    /// order, so its online-softmax accumulator sees exactly the sequence it saw before — the
    /// reorder is a loop interchange, not a reassociation. Two details preserve that. First,
    /// <c>valid</c> is still derived from the query tile's own causal limit, so a group-packed K
    /// tile that extends past a given query tile's reach has those columns zeroed by the existing
    /// clear before P·V, contributing exactly <c>0 × v</c>. Second, a query tile that cannot reach
    /// the current KV tile at all is skipped, which is precisely the iteration the old loop never
    /// ran. <c>TileJobs_MatchHeadJobs_BitExactly</c> is therefore a valid gate for this path.</para>
    ///
    /// <para><b>MEASURED: this helps.</b> SmolLM2-1.7B Q4_K_M, 1550-token prefill, headDim 64,
    /// 12 logical CPUs, 6 interleaved rounds per cell, as a 2×2 against the K-pack transpose
    /// (<c>STINGRAY_CPU_KPACK_SIMD</c>), best-of-6 / median t/s:</para>
    /// <code>
    ///   kpack   kv-outer    best   median      vs baseline (best)
    ///   scalar  off        148.3    143.7      baseline
    ///   scalar  ON         150.6    147.2      +1.6%
    ///   SIMD    off        152.8    148.7      +3.0%
    ///   SIMD    ON         154.2    149.9      +4.0%
    /// </code>
    /// <para>The two are roughly additive, so they attack overlapping but not identical cost: the
    /// SIMD transpose makes each pack cheaper, this reorder performs 7/8 fewer of them. The
    /// baseline is worst on best, median AND worst-case, which is what makes this more than a
    /// directional hint.</para>
    ///
    /// <para><b>A superseded earlier reading is recorded here deliberately.</b> A first 4-round
    /// A/B measured only the SIMD-on row (152.8 vs 154.2 — about +0.9%) and reported it as "no
    /// gain", and an op-count argument was offered to explain the null: the packs are ~8,192
    /// element copies against ~524,000 GEMM MACs, so ~1.5% of the tile's work. That argument is
    /// wrong, and the way it is wrong is worth keeping. It compares OPERATIONS, but the scalar
    /// pack walks a column — one float per KV row, a row-sized stride between touches, the access
    /// pattern that defeats prefetch — so its share of TIME far exceeds its share of ops. An
    /// op-count Amdahl bound is not a time bound for memory-latency-bound work.</para>
    ///
    /// <para><b>Why grouped rather than whole-sequence.</b> Holding running max/sum and the output
    /// accumulator live for every query tile at once costs <c>tokenCount × headDim</c> floats per
    /// thread — about 2 MB per thread at 8192 tokens, times a thread per core. Grouping bounds that
    /// to <c>groupTiles × 64 × headDim</c> (256 KB at the default 8 tiles and headDim 64, which
    /// stays L2-resident) while still amortising the K-pack 8-fold, capturing most of an available
    /// 32-fold saving for a fraction of the footprint.</para>
    /// </summary>
    private static void ComputePrefillFlashAttention64KvOuterHead(
        float* batchQ, PagedKvCache cache, int layer, int tokenCount, int startPos,
        float* output, int qDim, int headDim, float scale, int h, int kvHead,
        PrefillFlash64KvOuterScratch scratch, int groupTiles)
    {
        const int Tile = 64;
        bool bf16 = cache.IsBf16Store;
        int queryTiles = (tokenCount + Tile - 1) / Tile;

        for (int g0 = 0; g0 < queryTiles; g0 += groupTiles)
        {
            int gTiles = Math.Min(groupTiles, queryTiles - g0);
            int qStart = g0 * Tile;
            int qCount = Math.Min(tokenCount - qStart, gTiles * Tile);

            for (int t = 0; t < qCount; t++)
            {
                scratch.RunningMax[t] = float.NegativeInfinity;
                scratch.RunningSum[t] = 0f;
                Buffer.MemoryCopy(batchQ + (long)(qStart + t) * qDim + h * headDim,
                    scratch.QPack + (long)t * headDim, headDim * sizeof(float), headDim * sizeof(float));
            }
            new Span<float>(scratch.Accumulator, qCount * headDim).Clear();

            int groupEnd = Math.Min(startPos + qStart + qCount, cache.Length);
            for (int kvBase = 0; kvBase < groupEnd; kvBase += Tile)
            {
                int kLen = Math.Min(Tile, groupEnd - kvBase);

                // ── Pack K and V ONCE for this KV tile (the whole point of the reorder) ──
                if (bf16)
                {
                    for (int j = 0; j < kLen; j++)
                        scratch.Bf16KeyRows[j] = cache.Bf16KeyAt(layer, kvBase + j) + kvHead * headDim;
                    for (int d = 0; d < headDim; d++)
                    {
                        float* packedRow = scratch.KPack + d * Tile;
                        int j = 0;
                        for (; j < kLen; j++) packedRow[j] = SimdKernels.Bf16ToF32(scratch.Bf16KeyRows[j][d]);
                        for (; j < Tile; j++) packedRow[j] = 0f;
                    }
                }
                else
                {
                    for (int j = 0; j < kLen; j++)
                        scratch.KeyRows[j] = cache.KeyAt(layer, kvBase + j) + kvHead * headDim;
                    int jFull = 0;
                    if (SimdKernels.KPackSimdEnabled && Avx.IsSupported && (headDim & 7) == 0)
                    {
                        jFull = kLen & ~7;
                        for (int j0 = 0; j0 < jFull; j0 += 8)
                            for (int d0 = 0; d0 < headDim; d0 += 8)
                                SimdKernels.TransposeBlock8x8(
                                    scratch.KeyRows + j0, d0, scratch.KPack + (long)d0 * Tile + j0, Tile);
                    }
                    for (int d = 0; d < headDim; d++)
                    {
                        float* packedRow = scratch.KPack + d * Tile;
                        int j = jFull;
                        for (; j < kLen; j++) packedRow[j] = scratch.KeyRows[j][d];
                        for (; j < Tile; j++) packedRow[j] = 0f;
                    }
                }

                if (bf16)
                    for (int j = 0; j < kLen; j++)
                        SimdKernels.WidenBf16ToF32(cache.Bf16ValueAtHead(layer, kvBase + j, kvHead),
                            scratch.VPack + j * headDim, headDim);
                else
                    for (int j = 0; j < kLen; j++)
                        Buffer.MemoryCopy(cache.ValueAtHead(layer, kvBase + j, kvHead),
                            scratch.VPack + j * headDim, headDim * sizeof(float), headDim * sizeof(float));
                new Span<float>(scratch.VPack + kLen * headDim, (Tile - kLen) * headDim).Clear();

                // ── Every query tile in the group consumes the packed K/V ──
                for (int qt = 0; qt < gTiles; qt++)
                {
                    int nBase = qStart + qt * Tile;
                    int tn = Math.Min(Tile, tokenCount - nBase);
                    if (tn <= 0) break;
                    // Exactly the iterations the per-query-tile loop never ran: this tile's causal
                    // reach ends at or before this KV tile, so it has no valid key here.
                    if (kvBase >= Math.Min(startPos + nBase + tn, cache.Length)) continue;

                    int qOff = nBase - qStart;
                    float* qPack = scratch.QPack + (long)qOff * headDim;
                    float* acc = scratch.Accumulator + (long)qOff * headDim;

                    if (headDim != Tile || s_flash64StridedGemm)
                        SimdKernels.GemmF32_6x2(qPack, scratch.KPack, scratch.Scores,
                            tn, headDim, Tile, headDim, Tile, Tile);
                    else
                        SimdKernels.GemmF32_64x64_6x2(qPack, scratch.KPack, scratch.Scores, tn);

                    for (int t = 0; t < tn; t++)
                    {
                        float* row = scratch.Scores + t * Tile;
                        int valid = Math.Clamp(startPos + nBase + t + 1 - kvBase, 0, kLen);
                        if (valid == 0)
                        {
                            new Span<float>(row, Tile).Clear();
                            continue;
                        }

                        float tileMax = SimdKernels.ScaleAndMaxF32InPlace(row, valid, scale);
                        float oldMax = scratch.RunningMax[qOff + t];
                        float newMax = MathF.Max(oldMax, tileMax);
                        float rescale = float.IsNegativeInfinity(oldMax) ? 0f : MathF.Exp(oldMax - newMax);
                        float tileSum = SimdKernels.ExpMinusMaxSumInPlace(row, valid, newMax);
                        new Span<float>(row + valid, Tile - valid).Clear();
                        scratch.RunningMax[qOff + t] = newMax;
                        scratch.RunningSum[qOff + t] = scratch.RunningSum[qOff + t] * rescale + tileSum;

                        if (rescale != 1f)
                        {
                            var rescaleV = Vector256.Create(rescale);
                            float* accRow = acc + (long)t * headDim;
                            for (int d = 0; d < headDim; d += 8)
                                Avx.Store(accRow + d, Avx.Multiply(Avx.LoadVector256(accRow + d), rescaleV));
                        }
                    }

                    if (headDim != Tile || s_flash64StridedGemm)
                        SimdKernels.GemmF32_6x2(scratch.Scores, scratch.VPack, acc,
                            tn, Tile, headDim, Tile, headDim, headDim, accumulate: true);
                    else
                        SimdKernels.GemmF32_64x64_6x2(scratch.Scores, scratch.VPack, acc, tn, accumulate: true);
                }
            }

            for (int t = 0; t < qCount; t++)
            {
                float* outHead = output + (long)(qStart + t) * qDim + h * headDim;
                float* accRow = scratch.Accumulator + (long)t * headDim;
                var inv = Vector256.Create(1f / scratch.RunningSum[t]);
                for (int d = 0; d < headDim; d += 8)
                    Avx.Store(outHead + d, Avx.Multiply(Avx.LoadVector256(accRow + d), inv));
            }
        }
    }

    private static void ComputePrefillFlashAttention64Tile(
        float* batchQ, PagedKvCache cache, int layer, int tokenCount, int startPos,
        float* output, int qDim, int headDim, float scale, int h, int kvHead, int nBase,
        PrefillFlash64Scratch scratch)
    {
        const int Tile = 64;
        int tn = Math.Min(Tile, tokenCount - nBase);
        bool bf16 = cache.IsBf16Store;

        for (int t = 0; t < tn; t++)
        {
            scratch.RunningMax[t] = float.NegativeInfinity;
            scratch.RunningSum[t] = 0f;
            Buffer.MemoryCopy(batchQ + (long)(nBase + t) * qDim + h * headDim,
                scratch.QPack + t * headDim, headDim * sizeof(float), headDim * sizeof(float));
        }
        new Span<float>(scratch.Accumulator, tn * headDim).Clear();

        int endSeqMax = Math.Min(startPos + nBase + tn, cache.Length);
        for (int kvBase = 0; kvBase < endSeqMax; kvBase += Tile)
        {
            int kLen = Math.Min(Tile, endSeqMax - kvBase);
            // BF16 pages are widened HERE, once per tile, into the same F32 pack the GEMM already
            // consumes 64 times — so the widen amortises 64-fold and every kernel below this point
            // is bit-for-bit the F32 one. (The opposite choice, widening on each streaming read,
            // is right for decode and wrong here; see SimdKernels.WidenBf16ToF32.)
            if (bf16)
            {
                for (int j = 0; j < kLen; j++)
                    scratch.Bf16KeyRows[j] = cache.Bf16KeyAt(layer, kvBase + j) + kvHead * headDim;
                for (int d = 0; d < headDim; d++)
                {
                    float* packedRow = scratch.KPack + d * Tile;
                    int j = 0;
                    for (; j < kLen; j++) packedRow[j] = SimdKernels.Bf16ToF32(scratch.Bf16KeyRows[j][d]);
                    for (; j < Tile; j++) packedRow[j] = 0f;
                }
            }
            else
            {
                for (int j = 0; j < kLen; j++)
                    scratch.KeyRows[j] = cache.KeyAt(layer, kvBase + j) + kvHead * headDim;

                // K-pack is a transpose: [key][dim] in the cache becomes [dim][key] for the GEMM.
                // Done scalar it is headDim*kLen single-float copies whose reads walk a column —
                // one float per KV row, a whole row's stride between touches. The 8x8 AVX block
                // form reads each source row as one 32-byte load instead, and is bit-identical
                // because a transpose only moves floats. Full 8-key blocks go through the vector
                // path; the ragged key tail and the zero-fill out to Tile stay scalar.
                int jFull = 0;
                if (SimdKernels.KPackSimdEnabled && Avx.IsSupported && (headDim & 7) == 0)
                {
                    jFull = kLen & ~7;
                    for (int j0 = 0; j0 < jFull; j0 += 8)
                        for (int d0 = 0; d0 < headDim; d0 += 8)
                            SimdKernels.TransposeBlock8x8(
                                scratch.KeyRows + j0, d0, scratch.KPack + (long)d0 * Tile + j0, Tile);
                }
                for (int d = 0; d < headDim; d++)
                {
                    float* packedRow = scratch.KPack + d * Tile;
                    int j = jFull;
                    for (; j < kLen; j++) packedRow[j] = scratch.KeyRows[j][d];
                    for (; j < Tile; j++) packedRow[j] = 0f;
                }
            }
            // Q*Kt. Tile and HeadDim are both the compile-time 64 here, so the strided kernel
            // runs the identical shape and is bit-identical to the hardcoded one
            // (GemmF32StridedParityTests pins that); it is measurably faster all the same, because
            // it hoists the six row base pointers out of the j/k loops instead of recomputing
            // indices. Gated so the two can be interleaved in one binary.
            if (headDim != Tile || s_flash64StridedGemm)
                SimdKernels.GemmF32_6x2(scratch.QPack, scratch.KPack, scratch.Scores,
                    tn, headDim, Tile, headDim, Tile, Tile);
            else
                SimdKernels.GemmF32_64x64_6x2(scratch.QPack, scratch.KPack, scratch.Scores, tn);

            for (int t = 0; t < tn; t++)
            {
                float* row = scratch.Scores + t * Tile;
                int valid = Math.Clamp(startPos + nBase + t + 1 - kvBase, 0, kLen);
                if (valid == 0)
                {
                    new Span<float>(row, Tile).Clear();
                    continue;
                }

                float tileMax = SimdKernels.ScaleAndMaxF32InPlace(row, valid, scale);
                float oldMax = scratch.RunningMax[t];
                float newMax = MathF.Max(oldMax, tileMax);
                float rescale = float.IsNegativeInfinity(oldMax) ? 0f : MathF.Exp(oldMax - newMax);
                float tileSum = SimdKernels.ExpMinusMaxSumInPlace(row, valid, newMax);
                new Span<float>(row + valid, Tile - valid).Clear();
                scratch.RunningMax[t] = newMax;
                scratch.RunningSum[t] = scratch.RunningSum[t] * rescale + tileSum;

                if (rescale != 1f)
                {
                    var rescaleV = Vector256.Create(rescale);
                    float* acc = scratch.Accumulator + t * headDim;
                    for (int d = 0; d < headDim; d += 8)
                        Avx.Store(acc + d, Avx.Multiply(Avx.LoadVector256(acc + d), rescaleV));
                }
            }

            if (bf16)
                for (int j = 0; j < kLen; j++)
                    SimdKernels.WidenBf16ToF32(cache.Bf16ValueAtHead(layer, kvBase + j, kvHead),
                        scratch.VPack + j * headDim, headDim);
            else
                for (int j = 0; j < kLen; j++)
                    Buffer.MemoryCopy(cache.ValueAtHead(layer, kvBase + j, kvHead),
                        scratch.VPack + j * headDim, headDim * sizeof(float), headDim * sizeof(float));
            new Span<float>(scratch.VPack + kLen * headDim, (Tile - kLen) * headDim).Clear();
            // P*V: the transposed shape of the pair above (k = keys, n = head dim).
            if (headDim != Tile || s_flash64StridedGemm)
                SimdKernels.GemmF32_6x2(scratch.Scores, scratch.VPack, scratch.Accumulator,
                    tn, Tile, headDim, Tile, headDim, headDim, accumulate: true);
            else
                SimdKernels.GemmF32_64x64_6x2(
                    scratch.Scores, scratch.VPack, scratch.Accumulator, tn, accumulate: true);
        }

        for (int t = 0; t < tn; t++)
        {
            float* outHead = output + (long)(nBase + t) * qDim + h * headDim;
            float* acc = scratch.Accumulator + t * headDim;
            var inv = Vector256.Create(1f / scratch.RunningSum[t]);
            for (int d = 0; d < headDim; d += 8)
                Avx.Store(outHead + d, Avx.Multiply(Avx.LoadVector256(acc + d), inv));
        }
    }

    /// <summary>
    /// Scratch for <see cref="ComputePrefillFlashAttention64KvOuterHead"/>. Differs from
    /// <see cref="PrefillFlash64Scratch"/> in one way that matters: running max/sum and the output
    /// accumulator are sized for a whole GROUP of query tiles rather than one, because the reorder
    /// keeps them all live while a KV tile is resident. K/V packs and the score tile stay
    /// single-tile — those are what the reorder is amortising, not what it multiplies.
    /// </summary>
    private sealed class PrefillFlash64KvOuterScratch : IDisposable
    {
        private const int Tile = 64;
        public readonly float* Scores = (float*)NativeMemory.AlignedAlloc(Tile * Tile * sizeof(float), 64);
        public readonly float* KPack;
        public readonly float* VPack;
        public readonly float* QPack;
        public readonly float* Accumulator;
        public readonly float* RunningMax;
        public readonly float* RunningSum;
        public readonly float** KeyRows = (float**)NativeMemory.AlignedAlloc((nuint)(Tile * sizeof(nint)), 64);
        public readonly ushort** Bf16KeyRows = (ushort**)NativeMemory.AlignedAlloc((nuint)(Tile * sizeof(nint)), 64);

        public PrefillFlash64KvOuterScratch(int headDim, int maxQueries)
        {
            nuint tileElems = (nuint)(Tile * headDim);
            KPack = (float*)NativeMemory.AlignedAlloc(tileElems * sizeof(float), 64);
            VPack = (float*)NativeMemory.AlignedAlloc(tileElems * sizeof(float), 64);

            nuint groupElems = (nuint)((long)maxQueries * headDim);
            QPack = (float*)NativeMemory.AlignedAlloc(groupElems * sizeof(float), 64);
            Accumulator = (float*)NativeMemory.AlignedAlloc(groupElems * sizeof(float), 64);
            RunningMax = (float*)NativeMemory.AlignedAlloc((nuint)maxQueries * sizeof(float), 64);
            RunningSum = (float*)NativeMemory.AlignedAlloc((nuint)maxQueries * sizeof(float), 64);
        }

        public void Dispose()
        {
            NativeMemory.AlignedFree(Bf16KeyRows);
            NativeMemory.AlignedFree(KeyRows);
            NativeMemory.AlignedFree(RunningSum);
            NativeMemory.AlignedFree(RunningMax);
            NativeMemory.AlignedFree(Accumulator);
            NativeMemory.AlignedFree(QPack);
            NativeMemory.AlignedFree(VPack);
            NativeMemory.AlignedFree(KPack);
            NativeMemory.AlignedFree(Scores);
        }
    }

    private sealed class PrefillFlash64Scratch : IDisposable
    {
        private const int Tile = 64;
        public readonly float* Scores = (float*)NativeMemory.AlignedAlloc(Tile * Tile * sizeof(float), 64);
        public readonly float* Accumulator;
        public readonly float* RunningMax = (float*)NativeMemory.AlignedAlloc(64 * sizeof(float), 64);
        public readonly float* RunningSum = (float*)NativeMemory.AlignedAlloc(64 * sizeof(float), 64);
        public readonly float* QPack;
        public readonly float* KPack;
        public readonly float* VPack;
        public readonly float** KeyRows = (float**)NativeMemory.AlignedAlloc((nuint)(64 * sizeof(nint)), 64);
        /// <summary>BF16-store counterpart of <see cref="KeyRows"/>. Only one of the two is ever
        /// populated for a given cache; both are allocated because the scratch is pooled per thread
        /// and 512 bytes is not worth a conditional allocation.</summary>
        public readonly ushort** Bf16KeyRows = (ushort**)NativeMemory.AlignedAlloc((nuint)(64 * sizeof(nint)), 64);

        public PrefillFlash64Scratch(int headDim)
        {
            nuint elements = (nuint)(Tile * headDim);
            Accumulator = (float*)NativeMemory.AlignedAlloc(elements * sizeof(float), 64);
            QPack = (float*)NativeMemory.AlignedAlloc(elements * sizeof(float), 64);
            KPack = (float*)NativeMemory.AlignedAlloc(elements * sizeof(float), 64);
            VPack = (float*)NativeMemory.AlignedAlloc(elements * sizeof(float), 64);
        }

        public void Dispose()
        {
            NativeMemory.AlignedFree(Bf16KeyRows);
            NativeMemory.AlignedFree(KeyRows);
            NativeMemory.AlignedFree(VPack);
            NativeMemory.AlignedFree(KPack);
            NativeMemory.AlignedFree(QPack);
            NativeMemory.AlignedFree(RunningSum);
            NativeMemory.AlignedFree(RunningMax);
            NativeMemory.AlignedFree(Accumulator);
            NativeMemory.AlignedFree(Scores);
        }
    }

    /// <summary>
    /// Weighted-V microkernel for prefill attention. Holds eight tokens' eight-float output chunks
    /// in YMM registers across the ascending KV loop. Every lane receives the same FMA sequence as
    /// the former memory-accumulator loop, so chunked and unchunked prefill remain bit-identical.
    /// </summary>
    private static void AccumulatePrefillValuesRegister8(float** valueRows, float* scores, int stride,
        float* output, int nBase, int tokenCount, int qDim, int headOffset, int headDim,
        int startPos, int cacheLength)
    {
        for (int tBase = 0; tBase < tokenCount; tBase += 8)
        {
            int active = Math.Min(8, tokenCount - tBase);
            int firstEnd = Math.Min(startPos + nBase + tBase + 1, cacheLength);
            int lastEnd = Math.Min(firstEnd + active - 1, cacheLength);

            for (int d = 0; d < headDim; d += 8)
            {
                var a0 = Vector256<float>.Zero;
                var a1 = Vector256<float>.Zero;
                var a2 = Vector256<float>.Zero;
                var a3 = Vector256<float>.Zero;
                var a4 = Vector256<float>.Zero;
                var a5 = Vector256<float>.Zero;
                var a6 = Vector256<float>.Zero;
                var a7 = Vector256<float>.Zero;

                if (active == 8)
                {
                    for (int i = 0; i < firstEnd; i++)
                    {
                        var v = Avx.LoadVector256(valueRows[i] + d);
                        a0 = Fma.MultiplyAdd(Vector256.Create(scores[(long)(tBase + 0) * stride + i]), v, a0);
                        a1 = Fma.MultiplyAdd(Vector256.Create(scores[(long)(tBase + 1) * stride + i]), v, a1);
                        a2 = Fma.MultiplyAdd(Vector256.Create(scores[(long)(tBase + 2) * stride + i]), v, a2);
                        a3 = Fma.MultiplyAdd(Vector256.Create(scores[(long)(tBase + 3) * stride + i]), v, a3);
                        a4 = Fma.MultiplyAdd(Vector256.Create(scores[(long)(tBase + 4) * stride + i]), v, a4);
                        a5 = Fma.MultiplyAdd(Vector256.Create(scores[(long)(tBase + 5) * stride + i]), v, a5);
                        a6 = Fma.MultiplyAdd(Vector256.Create(scores[(long)(tBase + 6) * stride + i]), v, a6);
                        a7 = Fma.MultiplyAdd(Vector256.Create(scores[(long)(tBase + 7) * stride + i]), v, a7);
                    }
                }
                else
                {
                    for (int i = 0; i < firstEnd; i++)
                    {
                        var v = Avx.LoadVector256(valueRows[i] + d);
                        a0 = Fma.MultiplyAdd(Vector256.Create(scores[(long)(tBase + 0) * stride + i]), v, a0);
                        if (active > 1) a1 = Fma.MultiplyAdd(Vector256.Create(scores[(long)(tBase + 1) * stride + i]), v, a1);
                        if (active > 2) a2 = Fma.MultiplyAdd(Vector256.Create(scores[(long)(tBase + 2) * stride + i]), v, a2);
                        if (active > 3) a3 = Fma.MultiplyAdd(Vector256.Create(scores[(long)(tBase + 3) * stride + i]), v, a3);
                        if (active > 4) a4 = Fma.MultiplyAdd(Vector256.Create(scores[(long)(tBase + 4) * stride + i]), v, a4);
                        if (active > 5) a5 = Fma.MultiplyAdd(Vector256.Create(scores[(long)(tBase + 5) * stride + i]), v, a5);
                        if (active > 6) a6 = Fma.MultiplyAdd(Vector256.Create(scores[(long)(tBase + 6) * stride + i]), v, a6);
                    }
                }

                // A full causal group differs only in its final seven positions. These FMAs remain
                // ascending for every accumulator; this is loop interchange, not reassociation.
                for (int i = firstEnd; i < lastEnd; i++)
                {
                    var v = Avx.LoadVector256(valueRows[i] + d);
                    int firstActive = i - firstEnd + 1;
                    if (firstActive <= 1 && active > 1) a1 = Fma.MultiplyAdd(Vector256.Create(scores[(long)(tBase + 1) * stride + i]), v, a1);
                    if (firstActive <= 2 && active > 2) a2 = Fma.MultiplyAdd(Vector256.Create(scores[(long)(tBase + 2) * stride + i]), v, a2);
                    if (firstActive <= 3 && active > 3) a3 = Fma.MultiplyAdd(Vector256.Create(scores[(long)(tBase + 3) * stride + i]), v, a3);
                    if (firstActive <= 4 && active > 4) a4 = Fma.MultiplyAdd(Vector256.Create(scores[(long)(tBase + 4) * stride + i]), v, a4);
                    if (firstActive <= 5 && active > 5) a5 = Fma.MultiplyAdd(Vector256.Create(scores[(long)(tBase + 5) * stride + i]), v, a5);
                    if (firstActive <= 6 && active > 6) a6 = Fma.MultiplyAdd(Vector256.Create(scores[(long)(tBase + 6) * stride + i]), v, a6);
                    if (firstActive <= 7 && active > 7) a7 = Fma.MultiplyAdd(Vector256.Create(scores[(long)(tBase + 7) * stride + i]), v, a7);
                }

                Avx.Store(output + (long)(nBase + tBase + 0) * qDim + headOffset + d, a0);
                if (active > 1) Avx.Store(output + (long)(nBase + tBase + 1) * qDim + headOffset + d, a1);
                if (active > 2) Avx.Store(output + (long)(nBase + tBase + 2) * qDim + headOffset + d, a2);
                if (active > 3) Avx.Store(output + (long)(nBase + tBase + 3) * qDim + headOffset + d, a3);
                if (active > 4) Avx.Store(output + (long)(nBase + tBase + 4) * qDim + headOffset + d, a4);
                if (active > 5) Avx.Store(output + (long)(nBase + tBase + 5) * qDim + headOffset + d, a5);
                if (active > 6) Avx.Store(output + (long)(nBase + tBase + 6) * qDim + headOffset + d, a6);
                if (active > 7) Avx.Store(output + (long)(nBase + tBase + 7) * qDim + headOffset + d, a7);
            }
        }
    }

    // ================================================================
    //  Attention
    // ================================================================

    private void Attention(PagedKvCache cache, int layer, int position)
        => Attention(cache, layer, layer, position, _headDim, windowSize: -1, _numKvHeads);

    /// <summary>
    /// Multi-head attention with optional per-layer head dim, KV-source aliasing, and
    /// sliding-window bound. <paramref name="readLayer"/> is the layer whose K/V pages
    /// to read (== <paramref name="ownLayer"/> for non-shared layers; the source layer
    /// when KV is aliased). <paramref name="windowSize"/> &gt; 0 restricts the score and
    /// V-aggregation loops to the last <paramref name="windowSize"/> positions.
    /// <paramref name="kvHeads"/> is the active layer's KV head count (Gemma 4 12B:
    /// 8 GQA on SWA layers, 1 MQA on the k_eq_v global layers) — it can differ from the
    /// model-level <see cref="_numKvHeads"/>, so the head→KV-group ratio is computed here.
    /// </summary>
    private void Attention(PagedKvCache cache, int readLayer, int ownLayer, int position,
        int hd, int windowSize, int kvHeads)
    {
        // After SnapKV eviction (issue #51), the absolute position keeps
        // growing while the cache only stores `cache.Length` slots — `position`
        // would overshoot. The prefill loop increments cache.Length before
        // calling Attention (so position+1 == cache.Length); the decode loop
        // increments after (so position+1 == cache.Length+1). Clamping to
        // cache.Length+1 keeps the old answer for both prefill and the
        // un-evicted decode case while bounding the read to the actually
        // stored slots once eviction has shrunk the cache.
        _ = ownLayer;
        int endSeq = Math.Min(position + 1, cache.Length + 1);
        int startSeq = windowSize > 0 ? Math.Max(0, endSeq - windowSize) : 0;
        // Gemma 4 uses self.scaling = 1.0 (no pre-attention scaling); other archs
        // use 1/sqrt(head_dim). See llama.cpp src/models/gemma4.cpp:11
        //   hparams.f_attention_scale = 1.0f
        // Granite/MiniCPM declare an explicit kq_scale override (Granite 3.3 2B: 0.015625
        // = 1/64, not 1/sqrt(64) = 0.125) — checked before the Gemma-4/default fallback.
        float scale = _hp.AttentionScaleOverride != 0f ? _hp.AttentionScaleOverride
            : _layerHeadDim is not null ? 1.0f : 1.0f / MathF.Sqrt(hd);
        // Head→KV-group ratio for the ACTIVE layer (kvHeads, not the model-level
        // _numKvHeads): Gemma 4 12B global layers are MQA (kvHeads=1 → all _numHeads map
        // to KV head 0), SWA layers GQA (kvHeads=8). For non-per-layer models kvHeads ==
        // _numKvHeads so hpkg == _headsPerKvGroup.
        int ctxLen = _ctxLen; int hpkg = _numHeads / kvHeads;
        // The per-layer K/V stride now lives in PagedKvCache (see its layerHeadDim parameter), so
        // both the row-major K reads below and the transposed V reads agree on it. This method
        // used to compute a slotStride here and discard it, which left the V region striding at
        // the model-level head_dim while K strided at the layer's — the Gemma 4 KV-head bug.
        var q = _q; var attnOut = _attnOut; var scores = _attnScores;
        int rl = readLayer; int hdLocal = hd; int startLocal = startSeq;

        int scoreLenAll = endSeq - startLocal;
        int numHeadsLocal = _numHeads;

        // ── Score pass, parallelised over POSITION TILES rather than heads ──
        // Parallelising over heads makes head h read bytes [h*hd, h*hd+hd) of every KV row, i.e. a
        // stride equal to the whole row (numKvHeads*headDim floats — 8 KB on this model). Strides
        // beyond a page are not prefetched, so every read exposed full memory latency; decode was
        // achieving ~22 GB/s against a measured 36.8 GB/s ceiling. Walking positions instead lets
        // each KV row be read contiguously while all heads consume it, and the query vectors
        // (numHeads*headDim floats) are small enough to stay resident across the whole tile.
        //
        // Bit-identical: each score is the same dot of the same operands, and only the order in
        // which independent (head, position) pairs are computed changes.
        const int PosTile = 64;
        int posTiles = (scoreLenAll + PosTile - 1) / PosTile;
        // BF16-store caches hold 2-byte pages; the dtype is fixed for the cache's lifetime, so this
        // is hoisted entirely out of the position and head loops rather than tested per element.
        bool bf16 = cache.IsBf16Store;
        if (posTiles > 1)
        {
            Parallel.For(0, posTiles, ti =>
            {
                int i0 = ti * PosTile;
                int i1 = Math.Min(i0 + PosTile, scoreLenAll);
                if (bf16)
                {
                    for (int i = i0; i < i1; i++)
                    {
                        ushort* kRow = cache.Bf16KeyAt(rl, startLocal + i);
                        for (int hh = 0; hh < numHeadsLocal; hh++)
                            scores[(long)hh * ctxLen + i] =
                                SimdKernels.DotF32Bf16(q + hh * hdLocal, kRow + (hh / hpkg) * hdLocal, hdLocal) * scale;
                    }
                    return;
                }
                for (int i = i0; i < i1; i++)
                {
                    float* kRow = cache.KeyAt(rl, startLocal + i);
                    for (int hh = 0; hh < numHeadsLocal; hh++)
                        scores[(long)hh * ctxLen + i] =
                            SimdKernels.DotF32(q + hh * hdLocal, kRow + (hh / hpkg) * hdLocal, hdLocal) * scale;
                }
            });
        }
        else if (bf16)
        {
            for (int i = 0; i < scoreLenAll; i++)
            {
                ushort* kRow = cache.Bf16KeyAt(rl, startLocal + i);
                for (int hh = 0; hh < numHeadsLocal; hh++)
                    scores[(long)hh * ctxLen + i] =
                        SimdKernels.DotF32Bf16(q + hh * hdLocal, kRow + (hh / hpkg) * hdLocal, hdLocal) * scale;
            }
        }
        else
        {
            for (int i = 0; i < scoreLenAll; i++)
            {
                float* kRow = cache.KeyAt(rl, startLocal + i);
                for (int hh = 0; hh < numHeadsLocal; hh++)
                    scores[(long)hh * ctxLen + i] =
                        SimdKernels.DotF32(q + hh * hdLocal, kRow + (hh / hpkg) * hdLocal, hdLocal) * scale;
            }
        }

        // Softmax and the weighted-V sum stay parallel over heads: the V accumulation is a
        // per-head reduction over ascending i, so splitting it by position would need per-thread
        // partials and would change the accumulation order (and the result).
        Parallel.For(0, _numHeads, h =>
        {
            int kvHead = h / hpkg;
            float* outHead = attnOut + h * hdLocal;
            float* headScores = scores + (long)h * ctxLen;

            int scoreLen = scoreLenAll;

            SimdKernels.SoftmaxInPlace(headScores, scoreLen);

            for (int d = 0; d < hdLocal; d++) outHead[d] = 0;

            if (bf16)
            {
                for (int i = 0; i < scoreLen; i++)
                    SimdKernels.AccumulateScaledBf16(
                        outHead, cache.Bf16ValueAtHead(rl, startLocal + i, kvHead), headScores[i], hdLocal);
                return;
            }

            for (int i = 0; i < scoreLen; i++)
            {
                int t = startLocal + i;
                float* vVec = cache.ValueAtHead(rl, t, kvHead);
                float w = headScores[i];
                if (Fma.IsSupported && hdLocal >= 8)
                {
                    var wv = Vector256.Create(w);
                    int d = 0;
                    for (; d + 8 <= hdLocal; d += 8)
                    {
                        var o = Avx.LoadVector256(outHead + d);
                        var v = Avx.LoadVector256(vVec + d);
                        Avx.Store(outHead + d, Fma.MultiplyAdd(wv, v, o));
                    }
                    for (; d < hdLocal; d++)
                        outHead[d] += w * vVec[d];
                }
                else
                {
                    for (int d = 0; d < hdLocal; d++)
                        outHead[d] += w * vVec[d];
                }
            }
        });
    }

    // ================================================================
    //  TurboQuant Attention
    // ================================================================

    private void TqAttention(int layer, int position)
    {
        var tq = _tqKvCache!;
        // After SnapKV (issue #60) eviction the absolute position keeps
        // growing while the TQ cache only stores `tq.Length` slots. The
        // Forward decode path's Append runs before IncrementPosition so the
        // new K/V is at slot tq.Length (and visible via Fp32KeyAt(position=
        // tq.Length)), hence the `+1` — mirrors PagedKvCache.Attention.
        // Pre-eviction position+1 == tq.Length so the clamp is a no-op.
        int seqLen = Math.Min(position + 1, tq.Length + 1);
        int tqLen = tq.GetTqLength(layer);
        int fp32Start = tqLen;
        float scale = 1.0f / MathF.Sqrt(_headDim);
        int ctxLen = _ctxLen; int hd = _headDim; int hpkg = _headsPerKvGroup;
        int tqBlkSz = tq.TqBlockSize;
        var q = _q; var attnOut = _attnOut; var scores = _attnScores;
        var rotated = _rotatedQuery; var decomp = _decompBuf;

        Parallel.For(0, _numHeads, h =>
        {
            int kvHead = h / hpkg;
            float* qHead = q + h * hd;
            float* outHead = attnOut + h * hd;
            float* headScores = scores + (long)h * ctxLen;
            float* headRotated = rotated + h * hd;
            float* headDecomp = decomp + h * hd;

            // Rotate the query into the compressed-domain basis (Lloyd-Max:
            // per-head sign-flip + WHT; KVarN: plain WHT — issue #180).
            tq.RotateQuery(layer, kvHead,
                new ReadOnlySpan<float>(qHead, hd),
                new Span<float>(headRotated, hd));

            // K-scoring over the compressed region. Lloyd-Max (issue #34):
            // tile-walks full 32-position FastScan tiles through an i8-LUT
            // pshufb kernel and falls back to per-block DequantDot on the <32
            // staging tail. KVarN (issue #180): whole 128-token tiles via the
            // fused KVarNCompressor.KeyScores (no staging tail exists).
            tq.ComputeKScores(layer, kvHead, headRotated, scale, headScores);

            // Phase 1b: FP32 window positions
            for (int t = fp32Start; t < seqLen; t++)
            {
                float* kVec = tq.Fp32KeyAt(layer, t) + kvHead * hd;
                headScores[t] = SimdKernels.DotF32(qHead, kVec, hd) * scale;
            }

            SimdKernels.SoftmaxInPlace(headScores, seqLen);

            for (int d = 0; d < hd; d++) outHead[d] = 0;

            // V-aggregation over the compressed region: tiles accumulate in the
            // rotated domain with ONE deferred inverse WHT per head (Lloyd-Max
            // adds a sign-flip; KVarN uses UnrotateOutput), then the FP32-window
            // loop below accumulates the recent positions on top in the
            // original (un-rotated) domain — the domain contract both codecs share.
            tq.ComputeVAggregation(layer, kvHead, headScores, outHead);

            for (int t = fp32Start; t < seqLen; t++)
            {
                float* vVec = tq.Fp32ValueAt(layer, t) + kvHead * hd;
                float w = headScores[t];
                if (Fma.IsSupported && hd >= 8)
                {
                    var wv = Vector256.Create(w);
                    int d = 0;
                    for (; d + 8 <= hd; d += 8)
                    {
                        var o = Avx.LoadVector256(outHead + d);
                        var v = Avx.LoadVector256(vVec + d);
                        Avx.Store(outHead + d, Fma.MultiplyAdd(wv, v, o));
                    }
                    for (; d < hd; d++)
                        outHead[d] += w * vVec[d];
                }
                else
                {
                    for (int d = 0; d < hd; d++)
                        outHead[d] += w * vVec[d];
                }
            }
        });
    }

}
