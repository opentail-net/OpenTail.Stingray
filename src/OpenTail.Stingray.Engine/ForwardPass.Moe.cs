using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.TurboQuant;

namespace OpenTail.Stingray.Engine;

// Part of ForwardPass (see ForwardPass.cs for the type summary). Feed-forward: DenseFfn and the
// MoE FFN family (MoeFfn, batched MoeFfnBatched, expert-slot matvec helpers, top-k routing).
public sealed unsafe partial class ForwardPass
{
    // ================================================================
    //  Dense FFN (non-MoE)
    // ================================================================

    private void DenseFfn(int layer)
    {
        // Apertus/GPT-NeoX: no ffn_gate tensor — plain up -> activation -> down, no gate
        // multiply. Apertus uses xIELU (unbiased); GPT-NeoX uses biased GELU (up bias goes
        // INSIDE the activation — gelu(Wx + b), not gelu(Wx) + b — down bias after).
        if (_wGate[layer].DataPtr is null)
        {
            FusedMatVec(_ffnUp, _wUp[layer], _normBuf, _intermDim, _embDim);
            if (_xieluAlphaN is not null)
            {
                SimdKernels.XieluInPlace(_ffnUp, _intermDim,
                    _xieluAlphaN![layer], _xieluAlphaP![layer], _xieluBeta![layer], _xieluEps![layer]);
            }
            else if (_usesReluSquared)
            {
                if (_hasFfnBias && _bFfnUp is not null)
                    SimdKernels.AddInPlace(_ffnUp, _bFfnUp[layer], _intermDim);
                SimdKernels.ReluSqrInPlace(_ffnUp, _intermDim);
            }
            else
            {
                if (_hasFfnBias && _bFfnUp is not null)
                    SimdKernels.AddInPlace(_ffnUp, _bFfnUp[layer], _intermDim);
                SimdKernels.GeluInPlace(_ffnUp, _intermDim);
            }
            FusedMatVec(_hidden, _wDown[layer], _ffnUp, _embDim, _intermDim);
            if (_hasFfnBias && _bFfnDown is not null)
                SimdKernels.AddInPlace(_hidden, _bFfnDown[layer], _embDim);
            return;
        }

        SimdKernels.MatVecDual(_ffnGate, _wGate[layer].DataPtr, _ffnUp, _wUp[layer].DataPtr,
            _normBuf, _intermDim, _embDim, _wGate[layer].DType, _wUp[layer].DType);
        if (_hp.FfnActivation == FfnActivation.GeluApprox)
            SimdKernels.GeluTanhMul(_ffnGate, _ffnUp, _ffnGate, _intermDim);
        else
            SimdKernels.SiLuMul(_ffnGate, _ffnUp, _intermDim);
        FusedMatVec(_hidden, _wDown[layer], _ffnGate, _embDim, _intermDim);
    }

    // ================================================================
    //  MoE FFN (Mixture of Experts)
    // ================================================================

    private void MoeFfn(int layer)
    {
        int numExperts = _hp.NumExperts;
        int numActive = _hp.NumActiveExperts;
        int expertDim = _hp.ExpertIntermediateDim;

        // Step 1: Router — compute expert logits and select top-k
        FusedMatVec(_routerLogits, _wGateInp![layer], _normBuf, numExperts, _embDim);

        // Gating: sigmoid for Llama-4, softmax for others
        if (_hp.UseSigmoidGating)
            SimdKernels.SigmoidInPlace(_routerLogits, numExperts);
        else
            SimdKernels.SoftmaxInPlace(_routerLogits, numExperts);

        // Find top-k experts (for k=1, just argmax)
        Span<int> selectedExperts = stackalloc int[numActive];
        Span<float> expertWeights = stackalloc float[numActive];
        SelectTopK(_routerLogits, numExperts, numActive, selectedExperts, expertWeights,
            normalize: _hp.NormalizeMoeTopKWeights);

        if (_hp.ExpertWeightsScale != 1f)
            for (int k = 0; k < numActive; k++)
                expertWeights[k] *= _hp.ExpertWeightsScale;

        if (_traceRouters && (_traceRouterPos < 0 || _traceRouterPos == _currentPos))
        {
            var inv = System.Globalization.CultureInfo.InvariantCulture;
            var sb = new System.Text.StringBuilder(512);
            sb.Append("[router pos=").Append(_currentPos).Append(" L").Append(layer).Append(']');
            float wsum = 0;
            for (int i = 0; i < numActive; i++)
            {
                sb.Append(' ').Append(selectedExperts[i]).Append('=')
                  .Append(expertWeights[i].ToString("F4", inv));
                wsum += expertWeights[i];
            }
            sb.Append(" sum=").Append(wsum.ToString("F4", inv));

            // docs/bugstofix.md (ModelCompatibility.cs:461 entry, external-consultation follow-up):
            // top-k boundary margin -- the gap between the LEAST-confident SELECTED expert and the
            // MOST-confident NOT-selected one. _routerLogits holds post-softmax probabilities at
            // this point (softmax applied above, before SelectTopK). A tiny margin here is exactly
            // the condition under which a small upstream numerical difference can flip which expert
            // set gets selected -- the mechanism the external consultation's route-replay experiment
            // was designed to test.
            float minSelected = float.MaxValue;
            for (int i = 0; i < numActive; i++)
                if (_routerLogits[selectedExperts[i]] < minSelected) minSelected = _routerLogits[selectedExperts[i]];
            float maxUnselected = float.MinValue;
            for (int e = 0; e < numExperts; e++)
            {
                bool selected = false;
                for (int i = 0; i < numActive; i++)
                    if (selectedExperts[i] == e) { selected = true; break; }
                if (!selected && _routerLogits[e] > maxUnselected) maxUnselected = _routerLogits[e];
            }
            sb.Append(" margin=").Append((minSelected - maxUnselected).ToString("F6", inv));

            Console.Error.WriteLine(sb.ToString());
        }

        // Step 2: Shared expert (runs on every token if present)
        // Shared expert uses the same dim as routed experts (ExpertIntermediateDim)
        if (_hp.HasSharedExpert)
        {
            FusedMatVec(_expertGate, _wGateShexp![layer], _normBuf, expertDim, _embDim);
            FusedMatVec(_expertUp, _wUpShexp![layer], _normBuf, expertDim, _embDim);
            SimdKernels.SiLuMul(_expertGate, _expertUp, expertDim);
            FusedMatVec(_sharedOut, _wDownShexp![layer], _expertGate, _embDim, expertDim);
        }

        // Step 3: Selected expert(s) — sparse execution
        // Zero the output accumulator
        new Span<float>(_hidden, _embDim).Clear();

        for (int k = 0; k < numActive; k++)
        {
            int expertIdx = selectedExperts[k];
            float weight = expertWeights[k];

            // Expert weights are packed: all experts concatenated in one tensor.
            // Each expert's gate/up is [expertDim, embDim], down is [embDim, expertDim].
            // Expert slice offset in packed tensor: expertIdx * expertDim * (bytes per row)
            ExpertMatVec(_expertGate, _wGateExps![layer], expertIdx, expertDim, _embDim, _normBuf);
            ExpertMatVec(_expertUp, _wUpExps![layer], expertIdx, expertDim, _embDim, _normBuf);

            if (_hp.UseSigmoidGating)
            {
                // Llama-4: apply sigmoid weight before FFN (scale gate/up ≡ scaling input)
                SimdKernels.ScaleInPlace(_expertGate, weight, expertDim);
                SimdKernels.ScaleInPlace(_expertUp, weight, expertDim);
                weight = 1.0f;
            }

            SimdKernels.SiLuMul(_expertGate, _expertUp, expertDim);
            ExpertMatVecDown(_hidden, _wDownExps![layer], expertIdx, _embDim, expertDim, _expertGate, weight);
        }

        // Step 4: Add shared expert output
        if (_hp.HasSharedExpert)
            SimdKernels.AddInPlace(_hidden, _sharedOut, _embDim);
    }

    /// <summary>
    /// Master switch for the batched MoE prefill FFN. Set <c>STINGRAY_MOE_BATCHED_PREFILL=0</c>
    /// to force MoE prompts back onto the per-token sequential trunk. Settable so parity tests can
    /// run both arms in one process (the sequential arm is the oracle, and re-launching to get it
    /// would make the comparison depend on process state rather than on this one flag).
    /// </summary>
    /// <summary>
    /// Force batched prefill for per-layer-head-dim models (Gemma 4). Default OFF, and it must
    /// stay off until the batched core grows sliding-window attention, per-layer KV head counts
    /// and KV-layer sharing — see the gate in <c>PrefillWithPerPositionLogits</c>. Exists so the
    /// remaining work can be exercised and measured, not as a supported configuration.
    /// </summary>
    private static readonly bool s_perLayerHeadDimPrefillForced =
        Environment.GetEnvironmentVariable("STINGRAY_PER_LAYER_HD_PREFILL") == "1";

    /// <summary>
    /// Whether prefill flash-64 uses <see cref="SimdKernels.GemmF32_6x2"/> (strided) instead of
    /// <see cref="SimdKernels.GemmF32_64x64_6x2"/> (shape-hardcoded). The two are bit-identical at
    /// this shape, so this is a pure speed switch with no numerics question attached — it exists
    /// only so both arms can be interleaved in one binary rather than compared across rebuilds.
    /// <c>STINGRAY_FLASH64_STRIDED_GEMM=0</c> restores the hardcoded kernel.
    /// </summary>
    /// <summary>
    /// KV-outer prefill-attention reorder. <b>On by default</b>; <c>STINGRAY_PREFILL_ATTN_KV_OUTER=0</c>
    /// restores the per-query-tile schedule. Measured at +1.6% alone and +4.0% combined with the
    /// SIMD K-pack transpose — see the 2×2 table on
    /// <see cref="ComputePrefillFlashAttention64KvOuterHead"/>. It is bit-exact with the old
    /// schedule (<c>Flash64KvOuterTests</c>), so the default carries no numerical risk; the cost is
    /// scratch, ~256 KB per thread instead of ~16 KB, because a group of query tiles stays live
    /// while a KV tile is resident.
    ///
    /// <para>Settable rather than a readonly env snapshot so a test can flip it inside one process
    /// and diff the two schedules against each other. Reading it only from the environment would
    /// have made the natural gate useless: the reorder short-circuits before the tile-jobs branch,
    /// so an env-configured run of the existing schedule-comparison test would put BOTH arms on
    /// this path and compare it with itself — a confident pass proving nothing.</para>
    /// </summary>
    /// <summary>
    /// Admits head dimensions 128/256 to the Flash-64 prefill path. <b>Off by default</b> — the
    /// widths are implemented but held back pending the parity decision documented at the gate.
    /// Settable rather than env-only so the decision can be measured at all: the comparison needs
    /// flash-on and flash-off logits from within one process, which an env snapshot read once at
    /// type-init cannot provide.
    /// </summary>
    internal static bool Flash64WideHeadDimsEnabled { get; set; } =
        Environment.GetEnvironmentVariable("STINGRAY_PREFILL_ATTN_WIDE_HEADS") == "1";

    internal static bool Flash64KvOuterEnabled { get; set; } =
        Environment.GetEnvironmentVariable("STINGRAY_PREFILL_ATTN_KV_OUTER") != "0";

    /// <summary>
    /// Query tiles held live per KV pack in the reordered path. Trades scratch footprint for
    /// K-pack amortisation: 8 tiles is 512 queries, ~256 KB of accumulator+Q at headDim 64.
    /// </summary>
    private static readonly int s_flash64KvOuterGroupTiles =
        int.TryParse(Environment.GetEnvironmentVariable("STINGRAY_PREFILL_ATTN_KV_OUTER_TILES"),
            System.Globalization.NumberStyles.Integer,
            System.Globalization.CultureInfo.InvariantCulture, out int g) && g > 0 ? g : 8;

    private static readonly bool s_flash64StridedGemm =
        Environment.GetEnvironmentVariable("STINGRAY_FLASH64_STRIDED_GEMM") != "0";

    public static bool MoeBatchedPrefillEnabled { get; set; } =
        Environment.GetEnvironmentVariable("STINGRAY_MOE_BATCHED_PREFILL") != "0";

    /// <summary>
    /// Whether this model may take the batched MoE prefill path instead of falling back to the
    /// per-token sequential trunk.
    ///
    /// <para>The excluded cases are all things the batched cores do not model at all, for dense
    /// models either — post-attention/post-FFW norms, per-layer output scale and PLE are applied
    /// only on <c>RunTrunk</c>. Admitting a MoE model that has them would produce a silent
    /// numerics divergence between chunked and unchunked prefill of the same prompt, so they stay
    /// on the sequential path. The router trace is excluded because it prints
    /// <c>_currentPos</c>, which the batched path does not advance per token; a trace that lied
    /// about position would be worse than no batching.</para>
    /// </summary>
    private bool MoeBatchedPrefillSupported =>
        MoeBatchedPrefillEnabled
        && _hp.IsMoE
        && _wGateInp is not null && _wGateExps is not null
        && _wUpExps is not null && _wDownExps is not null
        && _tqKvCache is null
        && !_traceRouters && !_traceNorms
        && _postAttnNorm is null && _postFfwNorm is null
        && _layerOutputScale is null && !_hp.HasPerLayerTokenEmbd;

    /// <summary>
    /// Grow the batched-MoE scratch to hold <paramref name="n"/> token rows. Buffers are kept
    /// across layers and chunks (the reuse distance is one layer) and released in
    /// <see cref="Dispose"/>.
    /// </summary>
    private void EnsureMoeBatchScratch(int n)
    {
        if (n <= _moeBatchCap) return;
        FreeMoeBatchScratch();

        int numExperts = _hp.NumExperts;
        int na = _hp.NumActiveExperts;
        int expertDim = _hp.ExpertIntermediateDim;
        long pairs = (long)n * na;

        _moeBatchRouter   = (float*)NativeMemory.Alloc((nuint)((long)n * numExperts * sizeof(float)));
        _moeBatchSel      = (int*)  NativeMemory.Alloc((nuint)(pairs * sizeof(int)));
        _moeBatchWts      = (float*)NativeMemory.Alloc((nuint)(pairs * sizeof(float)));
        _moeExpStart      = (int*)  NativeMemory.Alloc((nuint)((numExperts + 1) * sizeof(int)));
        _moeExpCursor     = (int*)  NativeMemory.Alloc((nuint)(numExperts * sizeof(int)));
        _moeExpTokI       = (int*)  NativeMemory.Alloc((nuint)(pairs * sizeof(int)));
        _moeExpTokK       = (int*)  NativeMemory.Alloc((nuint)(pairs * sizeof(int)));
        _moeBatchGathered = (float*)NativeMemory.Alloc((nuint)((long)n * _embDim * sizeof(float)));
        _moeBatchGate     = (float*)NativeMemory.Alloc((nuint)((long)n * expertDim * sizeof(float)));
        _moeBatchUp       = (float*)NativeMemory.Alloc((nuint)((long)n * expertDim * sizeof(float)));
        _moeBatchDown     = (float*)NativeMemory.Alloc((nuint)(pairs * _embDim * sizeof(float)));
        _moeBatchCap = n;
    }

    private void FreeMoeBatchScratch()
    {
        if (_moeBatchCap == 0) return;
        NativeMemory.Free(_moeBatchRouter);   _moeBatchRouter = null;
        NativeMemory.Free(_moeBatchSel);      _moeBatchSel = null;
        NativeMemory.Free(_moeBatchWts);      _moeBatchWts = null;
        NativeMemory.Free(_moeExpStart);      _moeExpStart = null;
        NativeMemory.Free(_moeExpCursor);     _moeExpCursor = null;
        NativeMemory.Free(_moeExpTokI);       _moeExpTokI = null;
        NativeMemory.Free(_moeExpTokK);       _moeExpTokK = null;
        NativeMemory.Free(_moeBatchGathered); _moeBatchGathered = null;
        NativeMemory.Free(_moeBatchGate);     _moeBatchGate = null;
        NativeMemory.Free(_moeBatchUp);       _moeBatchUp = null;
        NativeMemory.Free(_moeBatchDown);     _moeBatchDown = null;
        _moeBatchCap = 0;
    }

    /// <summary>
    /// Batched MoE FFN for one prefill layer: the MoE twin of the dense
    /// gate/up GEMM → SiLU → down GEMM sequence in <see cref="PrefillCore"/>.
    ///
    /// <para><paramref name="batchNorm"/> holds the <paramref name="n"/> pre-FFN-normed rows and
    /// <paramref name="batchOut"/> receives the FFN output (fully overwritten; the caller adds the
    /// residual). The two must not alias — unlike the dense path, which writes the down projection
    /// straight back over its normed input, every expert re-reads <paramref name="batchNorm"/>.</para>
    ///
    /// <para>Structure, and why it is not just "widen the GEMMs": routing is per token, so the
    /// tokens sharing an expert are an arbitrary subset. This (1) routes every token — the router
    /// itself is dense and stays per token in F32, deliberately, see below; (2) buckets the
    /// (token, slot) pairs by selected expert into CSR order; (3) gathers each expert's tokens
    /// into one contiguous batch and runs three ordinary batched GEMMs over it, so the expert's
    /// weight rows are streamed once for the whole bucket rather than once per token; (4) reduces
    /// the unweighted down partials per token in top-k slot order. Step 2 is the part with no
    /// analogue in the dense path; step 4's ordering is load-bearing, see its comment.</para>
    ///
    /// <para>The router stays on the exact per-token F32 <see cref="FusedMatVec"/> the sequential
    /// path uses. Batching it would be nearly free in cost terms, but top-k selection is discrete:
    /// int8 activation quantisation could flip a marginal expert choice, and then the batched and
    /// sequential paths would not merely round differently, they would run different experts.
    /// The router is ~0.3% of this FFN's MACs, so there is nothing to win by risking that.</para>
    ///
    /// <para>The expert GEMMs do take the int8 batched path (<c>allowQ8: true</c>) — the same one
    /// the dense batched prefill has used since the Q8-prefill ship, and admissible for the same
    /// reason: the rows are positions within one prompt. That is the only source of divergence
    /// from the sequential F32 trunk — the same class the dense batched path already carries.
    /// With <c>Q8PrefillEnabled</c> off this path is bit-identical to sequential, which
    /// MoeBatchedPrefillParityTests pins.</para>
    /// </summary>
    private void MoeFfnBatched(int layer, float* batchNorm, float* batchOut, int n)
    {
        int numExperts = _hp.NumExperts;
        int na = _hp.NumActiveExperts;
        int expertDim = _hp.ExpertIntermediateDim;

        EnsureMoeBatchScratch(n);

        // ── 1. Route every token (per-token F32, identical to the sequential path) ──────────
        for (int t = 0; t < n; t++)
        {
            float* logits = _moeBatchRouter + (long)t * numExperts;
            FusedMatVec(logits, _wGateInp![layer], batchNorm + (long)t * _embDim, numExperts, _embDim);

            if (_hp.UseSigmoidGating)
                SimdKernels.SigmoidInPlace(logits, numExperts);
            else
                SimdKernels.SoftmaxInPlace(logits, numExperts);

            var wts = new Span<float>(_moeBatchWts + (long)t * na, na);
            SelectTopK(logits, numExperts, na,
                new Span<int>(_moeBatchSel + (long)t * na, na),
                wts,
                normalize: _hp.NormalizeMoeTopKWeights);

            if (_hp.ExpertWeightsScale != 1f)
                for (int k = 0; k < na; k++)
                    wts[k] *= _hp.ExpertWeightsScale;

            if (s_mlaTrace && t == 0)
            {
                var sel = new Span<int>(_moeBatchSel, na);
                Console.Error.WriteLine($"[MLA-TRACE] L{layer} tok0 experts=[{string.Join(",", sel.ToArray())}] weights=[{string.Join(",", wts.ToArray().Select(w => w.ToString("F4")))}]");
            }
        }

        // ── 2. Bucket the (token, slot) pairs by expert, CSR-style ─────────────────────────
        int* expStart = _moeExpStart;
        int* cursor = _moeExpCursor;
        long pairs = (long)n * na;
        for (int e = 0; e <= numExperts; e++) expStart[e] = 0;
        for (long s = 0; s < pairs; s++) expStart[_moeBatchSel[s] + 1]++;
        for (int e = 0; e < numExperts; e++) expStart[e + 1] += expStart[e];
        for (int e = 0; e < numExperts; e++) cursor[e] = expStart[e];
        for (int t = 0; t < n; t++)
            for (int k = 0; k < na; k++)
            {
                long s = (long)t * na + k;
                int p = cursor[_moeBatchSel[s]]++;
                _moeExpTokI[p] = t;
                _moeExpTokK[p] = k;
            }

        // ── 3. One batch of GEMMs per used expert ──────────────────────────────────────────
        ref readonly TensorRef gateExps = ref _wGateExps![layer];
        ref readonly TensorRef upExps = ref _wUpExps![layer];
        ref readonly TensorRef downExps = ref _wDownExps![layer];
        int bprGate = RowBytes(gateExps.DType, _embDim);
        int bprUp = RowBytes(upExps.DType, _embDim);
        int bprDown = RowBytes(downExps.DType, expertDim);

        for (int e = 0; e < numExperts; e++)
        {
            int p0 = expStart[e], p1 = expStart[e + 1];
            int cnt = p1 - p0;
            if (cnt == 0) continue;

            for (int i = 0; i < cnt; i++)
                Copy(_moeBatchGathered + (long)i * _embDim,
                     batchNorm + (long)_moeExpTokI[p0 + i] * _embDim, _embDim);

            SimdKernels.MatMulBatched(_moeBatchGate,
                gateExps.DataPtr + (long)e * expertDim * bprGate, _moeBatchGathered,
                cnt, expertDim, _embDim, gateExps.DType, allowQ8: true);
            SimdKernels.MatMulBatched(_moeBatchUp,
                upExps.DataPtr + (long)e * expertDim * bprUp, _moeBatchGathered,
                cnt, expertDim, _embDim, upExps.DType, allowQ8: true);

            // Llama-4 sigmoid gating scales the FFN input rather than its output, exactly as
            // MoeFfn does; the reduce below then uses a weight of 1 for those models.
            if (_hp.UseSigmoidGating)
                for (int i = 0; i < cnt; i++)
                {
                    float w = _moeBatchWts[(long)_moeExpTokI[p0 + i] * na + _moeExpTokK[p0 + i]];
                    SimdKernels.ScaleInPlace(_moeBatchGate + (long)i * expertDim, w, expertDim);
                    SimdKernels.ScaleInPlace(_moeBatchUp + (long)i * expertDim, w, expertDim);
                }

            // The bucket's rows are contiguous, so one SiLuMul covers the whole batch.
            SimdKernels.SiLuMul(_moeBatchGate, _moeBatchUp, cnt * expertDim);

            // Down projection into a scratch batch, then scattered UNWEIGHTED into
            // (token, slot) order — the weighting happens in phase 4.
            SimdKernels.MatMulBatched(_moeBatchGathered,
                downExps.DataPtr + (long)e * _embDim * bprDown, _moeBatchGate,
                cnt, _embDim, expertDim, downExps.DType, allowQ8: true);

            for (int i = 0; i < cnt; i++)
                Copy(_moeBatchDown + ((long)_moeExpTokI[p0 + i] * na + _moeExpTokK[p0 + i]) * _embDim,
                     _moeBatchGathered + (long)i * _embDim, _embDim);
        }

        // ── 4. Reduce per token, in TOP-K SLOT ORDER ──────────────────────────────────────
        // Not expert order, which is what the CSR loop above naturally produces. FP32 addition
        // is not associative, and reducing 8 expert contributions in a different order than
        // MoeFfn's `for k in 0..numActive` loop is not a last-bit difference: measured on OLMoE
        // it moved the final logits by up to 0.20 with every kernel otherwise identical, enough
        // to change the sampled token. Storing unweighted partials per (token, slot) and
        // reducing them here costs one extra pass over N*k*embDim floats and buys back exact
        // agreement with the sequential trunk. (This is also what the CUDA hybrid's
        // BatchedRoutedExpertsCpu does, for the same reason.)
        for (int t = 0; t < n; t++)
        {
            float* dst = batchOut + (long)t * _embDim;
            new Span<float>(dst, _embDim).Clear();
            for (int k = 0; k < na; k++)
                SimdKernels.WeightedAddInPlace(dst,
                    _moeBatchDown + ((long)t * na + k) * _embDim,
                    _hp.UseSigmoidGating ? 1f : _moeBatchWts[(long)t * na + k], _embDim);
        }

        // ── 5. Shared expert: dense over every token, so an ordinary batched FFN ───────────
        if (_hp.HasSharedExpert)
        {
            MatMulBatchedCached(_moeBatchGate, in _wGateShexp![layer], batchNorm, n, expertDim, _embDim);
            MatMulBatchedCached(_moeBatchUp, in _wUpShexp![layer], batchNorm, n, expertDim, _embDim);
            SimdKernels.SiLuMul(_moeBatchGate, _moeBatchUp, n * expertDim);
            MatMulBatchedCached(_moeBatchDown, in _wDownShexp![layer], _moeBatchGate, n, _embDim, expertDim);
            for (int t = 0; t < n; t++)
                SimdKernels.AddInPlace(batchOut + (long)t * _embDim,
                    _moeBatchDown + (long)t * _embDim, _embDim);
        }
    }

    /// <summary>Bytes one weight row of <paramref name="cols"/> elements occupies in this dtype.</summary>
    private static int RowBytes(DType dtype, int cols) =>
        (cols / DTypeInfo.BlockSize(dtype)) * DTypeInfo.BytesPerBlock(dtype);

    /// <summary>
    /// MatVec for a single expert slice from a packed expert tensor.
    /// The packed tensor has shape [numExperts * rows, cols]. Expert i's slice
    /// starts at row offset (i * rows).
    /// </summary>
    private void ExpertMatVec(float* output, in TensorRef packedTensor,
        int expertIdx, int rows, int cols, float* input)
    {
        int bytesPerRow = (cols / DTypeInfo.BlockSize(packedTensor.DType))
                        * DTypeInfo.BytesPerBlock(packedTensor.DType);
        long expertOffset = (long)expertIdx * rows * bytesPerRow;
        byte* expertData = packedTensor.DataPtr + expertOffset;
        SimdKernels.MatVec(output, expertData, input, rows, cols, packedTensor.DType);
    }

    /// <summary>
    /// MatVec for expert down projection, with weighted accumulation into output.
    /// output += weight * (expertDown[expertIdx] × input)
    /// </summary>
    private void ExpertMatVecDown(float* output, in TensorRef packedTensor,
        int expertIdx, int rows, int cols, float* input, float weight)
    {
        int bytesPerRow = (cols / DTypeInfo.BlockSize(packedTensor.DType))
                        * DTypeInfo.BytesPerBlock(packedTensor.DType);
        long expertOffset = (long)expertIdx * rows * bytesPerRow;
        byte* expertData = packedTensor.DataPtr + expertOffset;

        SimdKernels.MatVec(_moeDownTemp, expertData, input, rows, cols, packedTensor.DType);

        SimdKernels.WeightedAddInPlace(output, _moeDownTemp, weight, rows);
    }

    private static void SelectTopK(float* logits, int n, int k,
        Span<int> indices, Span<float> weights, bool normalize)
    {
        // Simple selection for small k (typically 1 or 2)
        for (int ki = 0; ki < k; ki++)
        {
            int bestIdx = 0;
            float bestVal = float.MinValue;
            for (int i = 0; i < n; i++)
            {
                bool alreadySelected = false;
                for (int j = 0; j < ki; j++)
                    if (indices[j] == i) { alreadySelected = true; break; }
                if (!alreadySelected && logits[i] > bestVal)
                { bestVal = logits[i]; bestIdx = i; }
            }
            indices[ki] = bestIdx;
            weights[ki] = bestVal;
        }

        // Renormalize selected weights to sum to 1 (Qwen3-MoE / Mixtral convention).
        // OLMoE skips this — its router uses raw post-softmax probabilities, so
        // unused mass on non-selected experts intentionally shrinks the MoE block's
        // contribution to the residual.
        if (normalize && k > 1)
        {
            float sum = 0;
            for (int i = 0; i < k; i++) sum += weights[i];
            if (sum > 0)
                for (int i = 0; i < k; i++) weights[i] /= sum;
        }
    }

}

