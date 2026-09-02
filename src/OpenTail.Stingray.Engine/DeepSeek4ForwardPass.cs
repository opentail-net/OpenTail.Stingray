using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Engine;

// ============================================================================================
// ALPHA / UNTESTED -- see DeepSeek4Alpha.cs's file header for the overall status/scope note;
// everything there applies here too, doubly so for this file. This is the forward-pass dispatch
// that drives DeepSeek4Graph + DeepSeek4TensorSet + DeepSeek4CompressedState through a real
// prompt. It has NEVER been run -- there is no DeepSeek-V4 GGUF available this session, so
// nothing below has executed even once, let alone been checked against a reference.
//
// HONEST SCOPE LIMIT, read before using this for anything: compress_ratio==0 ("raw attention")
// and compress_ratio==128 (HCA) layers are implemented. compress_ratio==4 (CSA) still throws
// NotSupportedException at construction. Reason CSA is deferred and HCA isn't: the compression
// projection's output width is coff*n_embd_head where coff = (ratio==4 ? 2 : 1)
// (deepseek4.cpp:129-136) -- for HCA (ratio 128, coff=1) this is one compressed row per raw
// token, structurally identical to the simple "one token in, one compressed row out" model this
// port already assumes. For CSA (ratio 4, coff=2), ONE token's compression call produces TWO
// sub-rows via an overlapping-window scheme (build_overlap_compressed_kv_from_state,
// deepseek4.cpp:524-606) this session could not reverse-engineer with confidence in the time
// available -- implementing it wrong would be worse than leaving it explicitly unimplemented.
// CSA also needs the lightning indexer's top-k mask folded into its attention
// (build_csa_lid_attention), which HCA does not. A real V4-Flash checkpoint likely uses BOTH
// ratios across its layers, so this class still cannot run such a checkpoint end-to-end -- it now
// covers two of the three attention variants, not all three.
//
// TWO SPECIFIC AREAS OF THE RAW-ATTENTION PORT BELOW ARE HIGH-RISK AND NOT CONFIDENTLY VERIFIED
// (flagged in-line at the point they're used, repeated here for visibility):
//  1. The raw path's kv projection is genuinely MQA-with-K==V: deepseek4.cpp's build_raw_attention
//     caches ONE n_embd_head-wide vector per token (not separate K and V) and calls
//     build_attn_mha(q, k, k, ...) -- the same tensor for both the key and value argument. This
//     port replicates that (one cached vector, used as both K and V), but it was inferred by
//     reading the reference's call sites, not confirmed against any executed trace.
//  2. The reference applies ggml_rope_ext_back (an INVERSE rotation) to the attention OUTPUT's
//     rope-dim slice before concatenating it with the nope slice (deepseek4.cpp:1252-1258), which
//     is unusual -- most architectures never touch RoPE after attention. This port replicates it
//     as literally "rotate by -position instead of +position," which is what "_back" suggests,
//     but the reference's actual ggml_rope_ext_back implementation was NOT read this session to
//     confirm that's the correct interpretation. Get real ground-truth intermediates (the same
//     methodology docs/done/032-...md used for deepseek2) before trusting this specific step.
// ============================================================================================

/// <summary>
/// ALPHA/UNTESTED, RAW-ATTENTION-ONLY. See this file's header. Implements <see cref="IForwardPass"/>
/// well enough to structurally exercise DeepSeek-V4's hyper-connection + MoE + raw-attention
/// wiring end-to-end for a synthetic or ratio-0-only checkpoint; throws on any layer with
/// compress_ratio != 0 (CSA/HCA), which real DeepSeek-V4 checkpoints are expected to use
/// extensively.
/// </summary>
public sealed unsafe class DeepSeek4ForwardPass : IForwardPass
{
    private readonly GgufModel _model;
    private readonly DeepSeek4Hyperparams _hp;
    private readonly DeepSeek4TensorSet _tensors;
    private readonly int _hc, _embedDim, _numHeads, _headDim, _ropeDim, _nopeDim, _numLayer;

    // Simplified single-sequence, no-rewind raw KV cache: one n_embd_head-wide vector per token
    // per layer (see this file's header point 1 -- K and V are the SAME vector in the raw path).
    // Populated for BOTH ratio==0 and ratio==128 layers (HCA still keeps the raw recent-token
    // cache alongside its compressed blocks, per build_hca_attention's raw_k concat).
    private readonly List<float[]>[] _kvCache;
    private readonly int[] _compressRatio;
    private readonly DeepSeek4CompressedState _compressedState;

    // Per-layer HCA compression scratch: raw comp-kv/comp-score rows accumulated since the last
    // finalized block, cleared every `ratio` tokens.
    private readonly List<float[]>[] _hcaKvBuffer;
    private readonly List<float[]>[] _hcaScoreBuffer;
    private readonly int[] _hcaTokensSinceBlockStart;

    public DeepSeek4ForwardPass(GgufModel model, DeepSeek4Hyperparams hp)
    {
        _model = model;
        _hp = hp;
        _tensors = DeepSeek4TensorSet.Load(model, hp);
        _hc = hp.HyperConnectionMultiplier;
        _embedDim = hp.EmbedDim;
        _numHeads = hp.NumHeads;
        _headDim = hp.HeadDim;
        _ropeDim = hp.RopeDim;
        _nopeDim = _headDim - _ropeDim;
        _numLayer = hp.NumLayer;

        _compressRatio = new int[_numLayer];
        for (int il = 0; il < _numLayer; il++)
        {
            int ratio = il < hp.CompressRatios.Count ? hp.CompressRatios[il] : 0;
            if (ratio != 0 && ratio != 128)
            {
                throw new NotSupportedException(
                    $"DeepSeek4ForwardPass (alpha): layer {il} has compress_ratio={ratio} -- " +
                    "only 0 (raw attention) and 128 (HCA) are implemented; 4 (CSA) is explicitly " +
                    "deferred, see this file's header. See " +
                    "docs/058-deepseek-full-lineage-implementation-plan.md Phase 0.");
            }
            _compressRatio[il] = ratio;
        }

        _kvCache = new List<float[]>[_numLayer];
        _hcaKvBuffer = new List<float[]>[_numLayer];
        _hcaScoreBuffer = new List<float[]>[_numLayer];
        _hcaTokensSinceBlockStart = new int[_numLayer];
        for (int il = 0; il < _numLayer; il++)
        {
            _kvCache[il] = [];
            _hcaKvBuffer[il] = [];
            _hcaScoreBuffer[il] = [];
        }

        _compressedState = new DeepSeek4CompressedState(_numLayer, _headDim, hp.IndexerHeadSize, hp.CompressRatios);
    }

    public int VocabSize { get; private set; }
    public int MaxSeqLen => 1 << 20; // no structural limit in this simplified cache; not tuned.

    public ReadOnlySpan<float> Forward(int token, int position)
    {
        // hc streams, each [embedDim]; all initialized to the same embedding row
        // (deepseek4.cpp:1286-1287: ggml_repeat_4d broadcasts the single embedding to every
        // stream).
        var inpL = new float[_hc * _embedDim];
        EmbedTokenInto(token, inpL.AsSpan(0, _embedDim));
        for (int h = 1; h < _hc; h++)
        {
            inpL.AsSpan(0, _embedDim).CopyTo(inpL.AsSpan(h * _embedDim, _embedDim));
        }

        for (int il = 0; il < _numLayer; il++)
        {
            RunAttentionBlock(il, inpL, position);
            RunFfnBlock(il, inpL, token);
        }

        // Final: hc_head mix-down -> output_norm -> LM head (deepseek4.cpp:1392-1401).
        var flatNormed = new float[_hc * _embedDim];
        fixed (float* inpPtr = inpL, outPtr = flatNormed)
        {
            SimdKernels.RmsNorm(outPtr, inpPtr, null, _hc * _embedDim, _hp.RmsNormEps);
        }
        // NOTE: PureRmsNorm (no learned weight) -- build_hc_head's normed input has no weight arg
        // either (deepseek4.cpp:453-454: ggml_rms_norm with no follow-up mul), matching Q's own
        // unweighted norm pattern noted in this file's header discussion.

        var pre = new float[_hc];
        DeepSeek4Graph.HyperConnectionHeadGate(
            flatNormed, _hc, _embedDim,
            AsFloatSpan(_tensors.HcHeadFn), AsFloatSpan(_tensors.HcHeadScale), AsFloatSpan(_tensors.HcHeadBase),
            _hp.HyperConnectionEpsilon, pre);

        var headMixed = new float[_embedDim];
        DeepSeek4Graph.HyperConnectionMixDown(inpL, pre, _hc, _embedDim, headMixed);

        var normed = new float[_embedDim];
        fixed (float* headMixedPtr = headMixed, normedPtr = normed)
        {
            float* weightPtr = (float*)_tensors.OutputNorm.DataPtr;
            SimdKernels.RmsNorm(normedPtr, headMixedPtr, weightPtr, _embedDim, _hp.RmsNormEps);
        }

        VocabSize = (int)_tensors.Output.Info.Dimensions[1];
        var logits = new float[VocabSize];
        fixed (float* normedPtr = normed, logitsPtr = logits)
        {
            SimdKernels.MatVec(logitsPtr, _tensors.Output.DataPtr, normedPtr, VocabSize, _embedDim, _tensors.Output.DType);
        }
        return logits;
    }

    private void RunAttentionBlock(int il, float[] inpL, int position)
    {
        var layer = _tensors.Layers[il];
        var residual = (float[])inpL.Clone();

        var flatNormed = new float[_hc * _embedDim];
        fixed (float* inpPtr = inpL, outPtr = flatNormed)
        {
            SimdKernels.RmsNorm(outPtr, inpPtr, null, _hc * _embedDim, _hp.RmsNormEps);
        }

        var pre = new float[_hc];
        var post = new float[_hc];
        var comb = new float[_hc * _hc];
        DeepSeek4Graph.HyperConnectionGate(
            flatNormed, _hc, _embedDim,
            AsFloatSpan(layer.HcAttnFn), AsFloatSpan(layer.HcAttnScale), AsFloatSpan(layer.HcAttnBase),
            _hp.HyperConnectionEpsilon, pre, post, comb);
        // HyperConnectionGate's internal Sinkhorn call uses iterations=1 hardcoded (see its own
        // doc comment) -- re-derive with hp.HyperConnectionSinkhornIterations once this is
        // exercised for real; not fixed in this pass to avoid changing DeepSeek4Graph's already-
        // tested public signature under time pressure.

        var cur = new float[_embedDim];
        DeepSeek4Graph.HyperConnectionMixDown(inpL, pre, _hc, _embedDim, cur);

        var attnNormed = new float[_embedDim];
        fixed (float* curPtr = cur, outPtr = attnNormed)
        {
            float* weightPtr = (float*)layer.AttnNorm!.Value.DataPtr;
            SimdKernels.RmsNorm(outPtr, curPtr, weightPtr, _embedDim, _hp.RmsNormEps);
        }

        var attnOut = RawAttention(il, layer, attnNormed, position);

        var mixedUp = new float[_hc * _embedDim];
        DeepSeek4Graph.HyperConnectionMixUp(attnOut, residual, post, comb, _hc, _embedDim, mixedUp);
        mixedUp.CopyTo(inpL, 0);
    }

    private void RunFfnBlock(int il, float[] inpL, int token)
    {
        var layer = _tensors.Layers[il];
        var residual = (float[])inpL.Clone();

        var flatNormed = new float[_hc * _embedDim];
        fixed (float* inpPtr = inpL, outPtr = flatNormed)
        {
            SimdKernels.RmsNorm(outPtr, inpPtr, null, _hc * _embedDim, _hp.RmsNormEps);
        }

        var pre = new float[_hc];
        var post = new float[_hc];
        var comb = new float[_hc * _hc];
        DeepSeek4Graph.HyperConnectionGate(
            flatNormed, _hc, _embedDim,
            AsFloatSpan(layer.HcFfnFn), AsFloatSpan(layer.HcFfnScale), AsFloatSpan(layer.HcFfnBase),
            _hp.HyperConnectionEpsilon, pre, post, comb);

        var cur = new float[_embedDim];
        DeepSeek4Graph.HyperConnectionMixDown(inpL, pre, _hc, _embedDim, cur);

        var ffnNormed = new float[_embedDim];
        fixed (float* curPtr = cur, outPtr = ffnNormed)
        {
            float* weightPtr = (float*)layer.FfnNorm!.Value.DataPtr;
            SimdKernels.RmsNorm(outPtr, curPtr, weightPtr, _embedDim, _hp.RmsNormEps);
        }

        var moeOut = MoeFfn(il, layer, ffnNormed, token);
        var sharedOut = SharedExpertFfn(layer, ffnNormed);
        var ffnOut = new float[_embedDim];
        for (int i = 0; i < _embedDim; i++) ffnOut[i] = moeOut[i] + sharedOut[i];

        var mixedUp = new float[_hc * _embedDim];
        DeepSeek4Graph.HyperConnectionMixUp(ffnOut, residual, post, comb, _hc, _embedDim, mixedUp);
        mixedUp.CopyTo(inpL, 0);
    }

    /// <summary>
    /// Raw (compress_ratio==0) multi-head attention. See this file's header for the two
    /// high-risk, not-independently-verified aspects (MQA-with-K==V, and the output-side
    /// rope_ext_back). Single-token decode only (position-by-position, not batched).
    /// </summary>
    private float[] RawAttention(int il, DeepSeek4LayerTensors layer, float[] normedInput, int position)
    {
        // q = wq_b(RMSNorm_unweighted(wq_a(normedInput))) reshaped [numHeads, headDim], then
        // per-head RMSNorm (no learned weight, deepseek4.cpp:940-946) then RoPE on the rope slice.
        int qLoraRank = _hp.QLoraRank;
        var qr = new float[qLoraRank];
        fixed (float* inPtr = normedInput, outPtr = qr)
        {
            SimdKernels.MatVec(outPtr, layer.WqA!.Value.DataPtr, inPtr, qLoraRank, _embedDim, layer.WqA.Value.DType);
        }
        var qrNormed = new float[qLoraRank];
        fixed (float* inPtr = qr, outPtr = qrNormed)
        {
            float* weightPtr = (float*)layer.AttnQANorm!.Value.DataPtr;
            SimdKernels.RmsNorm(outPtr, inPtr, weightPtr, qLoraRank, _hp.RmsNormEps);
        }
        var q = new float[_numHeads * _headDim];
        fixed (float* inPtr = qrNormed, outPtr = q)
        {
            SimdKernels.MatVec(outPtr, layer.WqB!.Value.DataPtr, inPtr, _numHeads * _headDim, qLoraRank, layer.WqB.Value.DType);
        }
        for (int h = 0; h < _numHeads; h++)
        {
            fixed (float* headPtr = q.AsSpan(h * _headDim, _headDim))
            {
                SimdKernels.RmsNorm(headPtr, headPtr, null, _headDim, _hp.RmsNormEps);
            }
            ApplyRopeInterleaved(q.AsSpan(h * _headDim + _nopeDim, _ropeDim), position, _hp.RopeFreqBase);
        }

        // kv = wkv(normedInput), RMSNorm'd with attn_kv_norm (learned weight), RoPE on rope slice.
        // headDim-wide, ONE vector (MQA-style) -- see this file's header point 1.
        var kv = new float[_headDim];
        fixed (float* inPtr = normedInput, outPtr = kv)
        {
            SimdKernels.MatVec(outPtr, layer.Wkv!.Value.DataPtr, inPtr, _headDim, _embedDim, layer.Wkv.Value.DType);
        }
        fixed (float* kvPtr = kv)
        {
            float* weightPtr = (float*)layer.AttnKvNorm!.Value.DataPtr;
            SimdKernels.RmsNorm(kvPtr, kvPtr, weightPtr, _headDim, _hp.RmsNormEps);
        }
        ApplyRopeInterleaved(kv.AsSpan(_nopeDim, _ropeDim), position, _hp.RopeFreqBase);

        _kvCache[il].Add(kv);

        // HCA layers (ratio==128): also accumulate/finalize compressed blocks and attend over
        // raw+compressed together (build_hca_attention's raw_k + hca_k concat, deepseek4.cpp:
        // 832-833). CSA (ratio==4) is not reached here -- gated out at construction.
        int ratio = _compressRatio[il];
        if (ratio == 128)
        {
            AccumulateHcaCompression(il, layer, normedInput, position);
        }

        var keys = _kvCache[il];
        var compressedKeys = ratio == 128 ? _compressedState.Hca(il)!.BlockCount : 0;

        int numKeys = keys.Count + compressedKeys;

        // Multi-head attention: each Q head attends over the SAME cached kv sequence
        // (deepseek4.cpp's raw path uses one shared kv "head" -- true MQA).
        var attnOut = new float[_numHeads * _headDim];
        var scores = new float[numKeys];
        float scale = 1f / MathF.Sqrt(_headDim);
        for (int h = 0; h < _numHeads; h++)
        {
            var qHead = q.AsSpan(h * _headDim, _headDim);
            for (int t = 0; t < numKeys; t++)
            {
                var kt = GetKeyOrCompressed(il, keys, t, compressedKeys);
                float dot = 0f;
                for (int d = 0; d < _headDim; d++) dot += qHead[d] * kt[d];
                scores[t] = dot * scale;
            }
            fixed (float* scoresPtr = scores)
            {
                SimdKernels.SoftmaxInPlace(scoresPtr, numKeys);
            }
            var outHead = attnOut.AsSpan(h * _headDim, _headDim);
            outHead.Clear();
            for (int t = 0; t < numKeys; t++)
            {
                var vt = GetKeyOrCompressed(il, keys, t, compressedKeys);
                float w = scores[t];
                for (int d = 0; d < _headDim; d++) outHead[d] += vt[d] * w;
            }

            // Output-side rope_ext_back on the rope slice -- see this file's header point 2.
            // Applied as "rotate by -position" (inverse of the forward rotation), NOT confirmed
            // against the reference's actual ggml_rope_ext_back implementation.
            ApplyRopeInterleaved(outHead.Slice(_nopeDim, _ropeDim), -position, _hp.RopeFreqBase);
        }

        // wo_a/wo_b grouped output LoRA (deepseek4.cpp:1247-1263-ish region; grouping is NOT
        // implemented here -- this treats wo_a/wo_b as a single ungrouped down-projection, which
        // is wrong whenever OutputGroupCount > 1. Flagged as a known gap, not silently assumed
        // correct: re-derive the per-group reshape (wo_a's {n_head*headDim/groups, loraRank,
        // groups} shape, deepseek4.cpp:119) before trusting this for a checkpoint with groups>1.
        int outLoraRank = _hp.OutputLoraRank;
        var oa = new float[outLoraRank];
        fixed (float* inPtr = attnOut, outPtr = oa)
        {
            SimdKernels.MatVec(outPtr, layer.WoA!.Value.DataPtr, inPtr, outLoraRank, _numHeads * _headDim, layer.WoA.Value.DType);
        }
        var result = new float[_embedDim];
        fixed (float* inPtr = oa, outPtr = result)
        {
            SimdKernels.MatVec(outPtr, layer.WoB!.Value.DataPtr, inPtr, _embedDim, outLoraRank, layer.WoB.Value.DType);
        }
        return result;
    }

    /// <summary>
    /// Indexes into the combined "raw cache followed by compressed HCA blocks" key sequence
    /// (deepseek4.cpp's build_hca_attention concatenates raw_k then hca_k, deepseek4.cpp:832-833
    /// -- this port matches that ordering: indices [0, rawCount) are raw, [rawCount, rawCount+
    /// compressedCount) are compressed blocks).
    /// </summary>
    private float[] GetKeyOrCompressed(int il, List<float[]> rawKeys, int index, int compressedCount)
    {
        if (index < rawKeys.Count) return rawKeys[index];
        var (kv, _) = _compressedState.Hca(il)!.GetBlock(index - rawKeys.Count);
        return kv.ToArray();
    }

    /// <summary>
    /// HCA (ratio==128, coff==1) block accumulation: projects this token's comp-kv/comp-score
    /// rows (deepseek4.cpp:989-993, via attn_comp_wkv/attn_comp_wgate + the attn_comp_ape
    /// positional table), buffers them, and finalizes+persists a compressed block via
    /// <see cref="DeepSeek4Graph.HcaCompressBlock"/> once exactly <c>ratio</c> raw tokens have
    /// accumulated since the last block boundary. Simplification vs. the reference: this port
    /// persists a zero vector as the block's companion "score" in
    /// <see cref="DeepSeek4CompressedState"/> (see that type's Persist signature) since, for
    /// HCA's non-overlapping scheme, a finalized block's score is never read again by a later
    /// compression step (only CSA's overlapping scheme reads a prior block's raw score, and CSA
    /// is not implemented here) -- only the block's KV is ever consumed downstream, by attention.
    /// </summary>
    private void AccumulateHcaCompression(int il, DeepSeek4LayerTensors layer, float[] normedInput, int position)
    {
        int coffHeadDim = _headDim; // coff==1 for ratio==128 (deepseek4.cpp:131: coff = ratio==4?2:1)
        int ratio = 128;

        var compKv = new float[coffHeadDim];
        var compScore = new float[coffHeadDim];
        fixed (float* inPtr = normedInput, kvPtr = compKv, scorePtr = compScore)
        {
            SimdKernels.MatVec(kvPtr, layer.AttnCompWkv!.Value.DataPtr, inPtr, coffHeadDim, _embedDim, layer.AttnCompWkv.Value.DType);
            SimdKernels.MatVec(scorePtr, layer.AttnCompWgate!.Value.DataPtr, inPtr, coffHeadDim, _embedDim, layer.AttnCompWgate.Value.DType);
        }

        int posInBlock = _hcaTokensSinceBlockStart[il];
        var apeInfo = layer.AttnCompApe!.Value.Info;
        int apeBytesPerRow = (coffHeadDim / DTypeInfo.BlockSize(layer.AttnCompApe.Value.DType)) * DTypeInfo.BytesPerBlock(layer.AttnCompApe.Value.DType);
        byte* apeRowPtr = layer.AttnCompApe.Value.DataPtr + (long)posInBlock * apeBytesPerRow;
        var apeRow = new float[coffHeadDim];
        fixed (float* apeOutPtr = apeRow)
        {
            SimdKernels.DequantRow(apeRowPtr, apeOutPtr, coffHeadDim, layer.AttnCompApe.Value.DType);
        }
        for (int i = 0; i < coffHeadDim; i++) compScore[i] += apeRow[i];

        _hcaKvBuffer[il].Add(compKv);
        _hcaScoreBuffer[il].Add(compScore);
        _hcaTokensSinceBlockStart[il]++;

        if (_hcaTokensSinceBlockStart[il] < ratio) return;

        var kvFlat = new float[ratio * coffHeadDim];
        var scoreFlat = new float[ratio * coffHeadDim];
        for (int r = 0; r < ratio; r++)
        {
            _hcaKvBuffer[il][r].CopyTo(kvFlat.AsSpan(r * coffHeadDim, coffHeadDim));
            _hcaScoreBuffer[il][r].CopyTo(scoreFlat.AsSpan(r * coffHeadDim, coffHeadDim));
        }

        int blockIndex = _compressedState.Hca(il)!.BlockCount;
        var result = new float[coffHeadDim];
        DeepSeek4Graph.HcaCompressBlock(
            kvFlat, scoreFlat, ratio, coffHeadDim, _ropeDim,
            AsFloatSpan(layer.AttnCompNorm), _hp.RmsNormEps,
            (span, ropeDim, blockPos, freqBase) => ApplyRopeInterleaved(span, (int)blockPos, freqBase),
            _hp.CompressRopeFreqBase, blockIndex,
            result);

        _compressedState.Hca(il)!.Persist(result, new float[coffHeadDim]);

        _hcaKvBuffer[il].Clear();
        _hcaScoreBuffer[il].Clear();
        _hcaTokensSinceBlockStart[il] = 0;
    }

    private float[] MoeFfn(int il, DeepSeek4LayerTensors layer, float[] normedInput, int token)
    {
        int numExperts = _hp.NumExperts;
        int topK = _hp.NumExpertsUsed;
        var expertIndices = new int[topK];
        var expertWeights = new float[topK];

        if (il < _hp.HashLayerCount)
        {
            // Hash-routed layer: direct token->expert lookup, unit weight (see
            // DeepSeek4Graph.HashLayerSelectExperts's doc comment -- the "unit weight" assumption
            // there is NOT independently re-verified against build_moe_ffn's actual hash-layer
            // call site).
            var tid2eidData = new int[topK];
            var tid2eidInfo = layer.FfnGateTid2Eid!.Value;
            // tid2eid is stored as a quantized/float GGUF tensor [numExpertsUsed, vocabSize] --
            // dequantize the one row for this token.
            var rowFloat = new float[topK];
            fixed (float* rowPtr = rowFloat)
            {
                int bytesPerRow = (topK / DTypeInfo.BlockSize(tid2eidInfo.DType)) * DTypeInfo.BytesPerBlock(tid2eidInfo.DType);
                byte* rowSrc = tid2eidInfo.DataPtr + (long)token * bytesPerRow;
                SimdKernels.DequantRow(rowSrc, rowPtr, topK, tid2eidInfo.DType);
            }
            for (int k = 0; k < topK; k++)
            {
                expertIndices[k] = (int)MathF.Round(rowFloat[k]);
                expertWeights[k] = 1f;
            }
        }
        else
        {
            var logits = new float[numExperts];
            fixed (float* inPtr = normedInput, outPtr = logits)
            {
                SimdKernels.MatVec(outPtr, layer.FfnGateInp!.Value.DataPtr, inPtr, numExperts, _embedDim, layer.FfnGateInp.Value.DType);
            }
            if (layer.FfnExpProbsB is { } bias)
            {
                float* biasPtr = (float*)bias.DataPtr;
                for (int e = 0; e < numExperts; e++) logits[e] += biasPtr[e];
            }
            var scores = new float[numExperts];
            DeepSeek4Graph.SqrtSoftplusGate(logits, scores);
            DeepSeek4Graph.SelectAndWeightExperts(
                scores, topK, _hp.ExpertWeightsNorm, _hp.ExpertWeightsScale, expertIndices, expertWeights);
        }

        var result = new float[_embedDim];
        int ffnDim = _hp.ExpertFeedForwardLength;
        var gate = new float[ffnDim];
        var up = new float[ffnDim];
        var down = new float[_embedDim];
        for (int k = 0; k < topK; k++)
        {
            int e = expertIndices[k];
            float w = expertWeights[k];
            ExpertMatVec(layer.FfnGateExps!.Value, e, normedInput, gate, ffnDim, numExperts);
            ExpertMatVec(layer.FfnUpExps!.Value, e, normedInput, up, ffnDim, numExperts);
            for (int i = 0; i < ffnDim; i++)
            {
                // SiLU(gate) * up -- ggml's exact single-divide form (see docs/done/032-...md's
                // SiLU fidelity fix); reused here as the same formula shape, not re-verified
                // against a real V4 checkpoint.
                float g = gate[i];
                float silu = g / (1f + MathF.Exp(-g));
                up[i] = silu * up[i];
            }
            ExpertMatVecDown(layer.FfnDownExps!.Value, e, up, down, ffnDim, numExperts);
            for (int i = 0; i < _embedDim; i++) result[i] += down[i] * w;
        }
        return result;
    }

    private float[] SharedExpertFfn(DeepSeek4LayerTensors layer, float[] normedInput)
    {
        int sharedFfnDim = _hp.ExpertFeedForwardLength * _hp.ExpertSharedCount;
        var gate = new float[sharedFfnDim];
        var up = new float[sharedFfnDim];
        fixed (float* inPtr = normedInput, gatePtr = gate, upPtr = up)
        {
            SimdKernels.MatVec(gatePtr, layer.FfnGateShexp!.Value.DataPtr, inPtr, sharedFfnDim, _embedDim, layer.FfnGateShexp.Value.DType);
            SimdKernels.MatVec(upPtr, layer.FfnUpShexp!.Value.DataPtr, inPtr, sharedFfnDim, _embedDim, layer.FfnUpShexp.Value.DType);
        }
        for (int i = 0; i < sharedFfnDim; i++)
        {
            float g = gate[i];
            float silu = g / (1f + MathF.Exp(-g));
            up[i] = silu * up[i];
        }
        var result = new float[_embedDim];
        fixed (float* inPtr = up, outPtr = result)
        {
            SimdKernels.MatVec(outPtr, layer.FfnDownShexp!.Value.DataPtr, inPtr, _embedDim, sharedFfnDim, layer.FfnDownShexp.Value.DType);
        }
        return result;
    }

    /// <summary>
    /// Matrix-vector multiply against one expert's slice of a 3D [in, out, numExperts]-shaped
    /// GGUF tensor. GGUF stores the expert axis as the SLOWEST-varying dimension (row-major, so
    /// expert e's [out, in] weight matrix starts at byte offset e * (out*in-scaled row size)) --
    /// matches this codebase's existing MoE expert-slicing convention
    /// (ExpertMatVec/ExpertMatVecDown, ForwardPass.Moe.cs), reproduced here rather than reused
    /// since those are private to ForwardPass.
    /// </summary>
    private void ExpertMatVec(DeepSeek4TensorRef tensor, int expert, float[] input, float[] output, int outDim, int numExperts)
    {
        int inDim = (int)tensor.Info.Dimensions[0];
        long bytesPerRow = ((long)inDim / DTypeInfo.BlockSize(tensor.DType)) * DTypeInfo.BytesPerBlock(tensor.DType);
        long expertByteOffset = (long)expert * outDim * bytesPerRow;
        byte* expertPtr = tensor.DataPtr + expertByteOffset;
        fixed (float* inPtr = input, outPtr = output)
        {
            SimdKernels.MatVec(outPtr, expertPtr, inPtr, outDim, inDim, tensor.DType);
        }
    }

    private void ExpertMatVecDown(DeepSeek4TensorRef tensor, int expert, float[] input, float[] output, int inDim, int numExperts)
    {
        int outDim = (int)tensor.Info.Dimensions[1];
        long bytesPerRow = ((long)inDim / DTypeInfo.BlockSize(tensor.DType)) * DTypeInfo.BytesPerBlock(tensor.DType);
        long expertByteOffset = (long)expert * outDim * bytesPerRow;
        byte* expertPtr = tensor.DataPtr + expertByteOffset;
        fixed (float* inPtr = input, outPtr = output)
        {
            SimdKernels.MatVec(outPtr, expertPtr, inPtr, outDim, inDim, tensor.DType);
        }
    }

    /// <summary>
    /// Interleaved (LLAMA_ROPE_TYPE_NORM) RoPE: rotates consecutive pairs (2i, 2i+1), confirmed
    /// as DeepSeek-V4's convention via llama-model.cpp:2589-2593 (LLM_ARCH_DEEPSEEK/DEEPSEEK2/
    /// DEEPSEEK32/DEEPSEEK4 all map to LLAMA_ROPE_TYPE_NORM) -- NOT the NEOX convention the
    /// lightning indexer uses elsewhere in the same file (a separate, confirmed-different case,
    /// deepseek4.cpp's indexer_q_pe/indexer_k_pe calls hardcode LLAMA_ROPE_TYPE_NEOX explicitly).
    /// </summary>
    private static void ApplyRopeInterleaved(Span<float> x, int position, float freqBase)
    {
        int dim = x.Length;
        int half = dim / 2;
        for (int i = 0; i < half; i++)
        {
            float freq = MathF.Pow(freqBase, -2f * i / dim);
            float theta = position * freq;
            float cos = MathF.Cos(theta), sin = MathF.Sin(theta);
            float x0 = x[2 * i], x1 = x[2 * i + 1];
            x[2 * i] = x0 * cos - x1 * sin;
            x[2 * i + 1] = x0 * sin + x1 * cos;
        }
    }

    private void EmbedTokenInto(int token, Span<float> dest)
    {
        var info = _tensors.TokEmbd.Info;
        int bytesPerRow = (_embedDim / DTypeInfo.BlockSize(info.DType)) * DTypeInfo.BytesPerBlock(info.DType);
        byte* rowPtr = _tensors.TokEmbd.DataPtr + (long)token * bytesPerRow;
        fixed (float* destPtr = dest)
        {
            SimdKernels.DequantRow(rowPtr, destPtr, _embedDim, info.DType);
        }
    }

    private static ReadOnlySpan<float> AsFloatSpan(DeepSeek4TensorRef? tensorRef)
    {
        if (tensorRef is not { } t) return default;
        int count = (int)t.Info.ElementCount;
        return new ReadOnlySpan<float>((float*)t.DataPtr, count);
        // NOTE: assumes the tensor is stored Float32 (true for every HC gate/scale/base tensor
        // per the reference's create_tensor calls, which never specify a quantized type for
        // these small per-layer gate tensors -- not independently re-verified against a real
        // GGUF's actual on-disk dtype).
    }

    // ── IForwardPass minimal surface (defaults handle the rest) ────────────────────────────

    public ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, int startPos = 0)
    {
        ReadOnlySpan<float> last = default;
        for (int i = 0; i < tokens.Count; i++)
        {
            last = Forward(tokens[i], startPos + i).ToArray();
        }
        return last;
    }

    public void TruncateTo(int length)
    {
        if (length == 0) { ResetCache(); return; }
        int current = _kvCache.Length > 0 ? _kvCache[0].Count : 0;
        if (length != current)
        {
            throw new NotSupportedException(
                "DeepSeek4ForwardPass (alpha): only full reset (TruncateTo(0)) or a no-op " +
                "(current length) is supported -- see DeepSeek4CompressedState's file header on " +
                "the simplified, non-rewindable cache this pass uses.");
        }
    }

    public void ResetCache()
    {
        for (int il = 0; il < _numLayer; il++) _kvCache[il].Clear();
    }

    public void Dispose() { }
}
