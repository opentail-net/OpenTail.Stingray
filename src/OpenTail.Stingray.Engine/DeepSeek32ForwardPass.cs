using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Engine;

// ============================================================================================
// ALPHA / UNTESTED -- see DeepSeek32Alpha.cs's file header for overall status/scope; everything
// there applies here. Forward-pass dispatch for deepseek32 (V3.2): classic MLA attention with
// absorption, DSA lightning-indexer sparse masking over a SINGLE raw K cache (structurally
// simpler than deepseek4's CSA/HCA compressed-block scheme -- deepseek32 has a real
// indexer_attn_k projection, no compression needed), and dense/MoE FFN dispatch by
// leading_dense_block_count. NEVER RUN -- no DeepSeek-V3.2 GGUF available this session.
//
// KNOWN, DELIBERATE SIMPLIFICATIONS/GAPS (mirroring the honesty discipline used for deepseek4):
//  1. YaRN RoPE scaling IS implemented (ApplyYarnRope + the constructor's kqScale derivation),
//     porting the deepseek2 investigation's already-verified formula chain
//     (docs/done/032-...md's attn_factor_org/mscale/kq_scale derivation, and
//     SimdKernels.BuildYarnRopeTable's corr-dim/ramp-mix math, reused as formula not as code --
//     see ApplyYarnRope's own doc comment for why it's a fresh single-position inline rather than
//     a BuildYarnRopeTable call). NOT independently re-verified against a real checkpoint or
//     ground-truth intermediates -- "ported from verified code" is not the same claim as
//     "verified in this new context." Also: freqScale/extFactor are derived from
//     RopeYarnFactor > 1 as a proxy for "YaRN active" since this port's GGUF loader doesn't read
//     rope.scaling.type itself -- a real checkpoint whose YaRN is signaled differently would slip
//     through as "inactive."
//  2. The Hadamard rotation applied to the indexer's Q/K (`self_k_rot_lid`,
//     ggml_mul_mat(inp_attn_dsa->self_k_rot_lid, indexer_q/k) in the reference) is NOT
//     implemented -- same gap as deepseek4, same reason (no Hadamard primitive in this codebase,
//     unclear whether it's a loaded weight or generated).
//  3. MTP is not implemented (HasMtpHead defaults to false).
//  4. The absorption-optimization MLA math (q_nope "absorbed" through wk_b before attention,
//     Kcur/Vcur built from the SAME kv_lora_rank-wide compressed vector, wv_b decompressing the
//     attention-weighted result AFTER the softmax sum rather than expanding V upfront) was
//     re-derived from reading deepseek32.cpp's graph constructor (lines 214-441, read in full
//     earlier this session), not reused from this codebase's existing dead
//     Engine.MlaAttention/DeepSeekMoeGraph classes (confirmed dead code, see DeepSeek4Alpha.cs's
//     equivalent note) -- believed correct from that reading, not independently verified.
// ============================================================================================

/// <summary>
/// ALPHA/UNTESTED. Implements <see cref="IForwardPass"/> for deepseek32 (V3.2): MLA attention
/// with DSA/lightning-indexer sparse masking, dense/MoE FFN dispatch. See this file's header for
/// the four specific, deliberate simplifications (no YaRN, no Hadamard, no MTP, MLA math
/// unverified against ground truth).
/// </summary>
public sealed unsafe class DeepSeek32ForwardPass : IForwardPass
{
    private readonly GgufModel _model;
    private readonly DeepSeek32Hyperparams _hp;
    private readonly DeepSeek32TensorSet _tensors;
    private readonly int _embedDim, _numHeads, _headDimK, _headDimV, _ropeDim, _nopeDim, _kvLoraRank, _numLayer;

    // Single raw MLA K cache per layer: each entry is [kvLoraRank + ropeDim] wide (Kcur, MQA-
    // shared across all Q heads). Vcur is a VIEW of the same cache (the leading kvLoraRank
    // channels) -- "K-only cache" per the reference's own comment, matching deepseek4's raw path.
    private readonly List<float[]>[] _kvCache;

    // Raw indexer K cache: one indexerHeadDim-wide vector per token per layer (deepseek32 has a
    // real indexer_attn_k projection -- no compression needed, unlike deepseek4's indexer).
    private readonly List<float[]>[] _indexerKCache;

    // YaRN RoPE scaling parameters and the derived deepseek-specific kq_scale, per
    // docs/058-...md's Phase 1 update: ported from the deepseek2 investigation's already-verified
    // formula chain (docs/done/032-...md) rather than left unimplemented. freqScale/extFactor
    // follow llama.cpp's standard convention (freqScale = 1/factor and extFactor = 1 when YaRN
    // scaling is declared, 1/0 otherwise) -- NOT independently re-verified against a real
    // checkpoint's rope.scaling.type metadata parse, since this port has no GGUF loader wired to
    // read that key at all yet (RopeYarnFactor > 1 is used as the "YaRN active" proxy).
    private readonly float _freqScale, _extFactor, _attnFactorBaseline, _kqScale;
    private readonly int _origCtxLen;

    public DeepSeek32ForwardPass(GgufModel model, DeepSeek32Hyperparams hp)
    {
        _model = model;
        _hp = hp;
        _tensors = DeepSeek32TensorSet.Load(model, hp);
        _embedDim = hp.EmbedDim;
        _numHeads = hp.NumHeads;
        _headDimK = hp.EffectiveHeadDimK;
        _headDimV = hp.EffectiveHeadDimV;
        _ropeDim = hp.RopeDim;
        _nopeDim = _headDimK - _ropeDim;
        _kvLoraRank = hp.KvLoraRank;
        _numLayer = hp.NumLayer;

        bool yarnActive = hp.RopeYarnFactor > 1f;
        _freqScale = yarnActive ? 1f / hp.RopeYarnFactor : 1f;
        _extFactor = yarnActive ? 1f : 0f;
        _attnFactorBaseline = 1f; // llama.cpp default absent an explicit {arch}.rope.scaling.attn_factor key.
        _origCtxLen = hp.RopeYarnOrigCtxLen;

        // deepseek32.cpp:193-199 -- attn_factor_org cancels BuildYarnRopeTable's own mscale
        // correction (see ApplyYarnRope) to recover the "pre-adjustment" attn_factor, then
        // reapplies it with deepseek's OWN rope_yarn_log_mul in place of the generic 0.1
        // constant, squared into kq_scale. When YaRN is inactive (freqScale==1), log(1/1)==0 and
        // this collapses to the plain kqScale = 1/sqrt(headDimK) form.
        float attnFactorOrg = _attnFactorBaseline * (1f + 0.1f * MathF.Log(1f / _freqScale));
        float mscale = attnFactorOrg * (1f + 0.1f * hp.RopeYarnLogMul * MathF.Log(1f / _freqScale));
        _kqScale = mscale * mscale / MathF.Sqrt(_headDimK);

        _kvCache = new List<float[]>[_numLayer];
        _indexerKCache = new List<float[]>[_numLayer];
        for (int il = 0; il < _numLayer; il++)
        {
            _kvCache[il] = [];
            _indexerKCache[il] = [];
        }
    }

    public int VocabSize { get; private set; }
    public int MaxSeqLen => 1 << 20;

    public ReadOnlySpan<float> Forward(int token, int position)
    {
        var cur = new float[_embedDim];
        EmbedTokenInto(token, cur);

        for (int il = 0; il < _numLayer; il++)
        {
            var layer = _tensors.Layers[il];
            var residual = (float[])cur.Clone();

            var attnNormed = new float[_embedDim];
            fixed (float* inPtr = cur, outPtr = attnNormed)
            {
                float* weightPtr = (float*)layer.AttnNorm!.Value.DataPtr;
                SimdKernels.RmsNorm(outPtr, inPtr, weightPtr, _embedDim, _hp.RmsNormEps);
            }

            var attnOut = MlaAttention(il, layer, attnNormed, position);
            for (int i = 0; i < _embedDim; i++) cur[i] = residual[i] + attnOut[i];

            residual = (float[])cur.Clone();
            var ffnNormed = new float[_embedDim];
            fixed (float* inPtr = cur, outPtr = ffnNormed)
            {
                float* weightPtr = (float*)layer.FfnNorm!.Value.DataPtr;
                SimdKernels.RmsNorm(outPtr, inPtr, weightPtr, _embedDim, _hp.RmsNormEps);
            }

            var ffnOut = il < _hp.LeadingDenseBlockCount ? DenseFfn(layer, ffnNormed) : MoeFfn(layer, ffnNormed);
            for (int i = 0; i < _embedDim; i++) cur[i] = residual[i] + ffnOut[i];
        }

        var normed = new float[_embedDim];
        fixed (float* inPtr = cur, outPtr = normed)
        {
            float* weightPtr = (float*)_tensors.OutputNorm.DataPtr;
            SimdKernels.RmsNorm(outPtr, inPtr, weightPtr, _embedDim, _hp.RmsNormEps);
        }

        VocabSize = (int)_tensors.Output.Info.Dimensions[1];
        var logits = new float[VocabSize];
        fixed (float* inPtr = normed, outPtr = logits)
        {
            SimdKernels.MatVec(outPtr, _tensors.Output.DataPtr, inPtr, VocabSize, _embedDim, _tensors.Output.DType);
        }
        return logits;
    }

    /// <summary>
    /// Classic MLA attention with absorption, per deepseek32.cpp:222-440 (the non-MTP trunk
    /// graph). See this file's header for the YaRN/Hadamard gaps. kq_scale here is the
    /// NO-YARN-CORRECTION plain form (1/sqrt(headDimK)) -- see header point 1.
    /// </summary>
    private float[] MlaAttention(int il, DeepSeek32LayerTensors layer, float[] normedInput, int position)
    {
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
        var q = new float[_numHeads * _headDimK];
        fixed (float* inPtr = qrNormed, outPtr = q)
        {
            SimdKernels.MatVec(outPtr, layer.WqB!.Value.DataPtr, inPtr, _numHeads * _headDimK, qLoraRank, layer.WqB.Value.DType);
        }
        for (int h = 0; h < _numHeads; h++)
        {
            ApplyYarnRope(q.AsSpan(h * _headDimK + _nopeDim, _ropeDim), position);
        }

        var kvCmprPe = new float[_kvLoraRank + _ropeDim];
        fixed (float* inPtr = normedInput, outPtr = kvCmprPe)
        {
            SimdKernels.MatVec(outPtr, layer.WkvAMqa!.Value.DataPtr, inPtr, _kvLoraRank + _ropeDim, _embedDim, layer.WkvAMqa.Value.DType);
        }
        var kPe = kvCmprPe.AsSpan(_kvLoraRank, _ropeDim).ToArray();
        ApplyYarnRope(kPe, position);
        var kvCmpr = new float[_kvLoraRank];
        fixed (float* inPtr = kvCmprPe, outPtr = kvCmpr)
        {
            float* weightPtr = (float*)layer.AttnKvANorm!.Value.DataPtr;
            SimdKernels.RmsNorm(outPtr, inPtr, weightPtr, _kvLoraRank, _hp.RmsNormEps);
        }

        // Kcur = concat(kv_cmpr, k_pe), single MQA-shared K/V source cached raw.
        var kCur = new float[_kvLoraRank + _ropeDim];
        kvCmpr.CopyTo(kCur, 0);
        kPe.CopyTo(kCur, _kvLoraRank);
        _kvCache[il].Add(kCur);

        // Absorb q_nope through wk_b per head: q_nope_absorbed[h] = wk_b[h]^T . q_nope[h],
        // [kvLoraRank]-wide, then concat with q_pe[h] to form the effective per-head query.
        var qEff = new float[_numHeads * (_kvLoraRank + _ropeDim)];
        for (int h = 0; h < _numHeads; h++)
        {
            var qNopeHead = q.AsSpan(h * _headDimK, _nopeDim);
            var absorbed = new float[_kvLoraRank];
            PerHeadMatVec(layer.WkB!.Value, h, _kvLoraRank, _nopeDim, qNopeHead, absorbed);
            absorbed.CopyTo(qEff.AsSpan(h * (_kvLoraRank + _ropeDim), _kvLoraRank));
            q.AsSpan(h * _headDimK + _nopeDim, _ropeDim).CopyTo(qEff.AsSpan(h * (_kvLoraRank + _ropeDim) + _kvLoraRank, _ropeDim));
        }

        int numKeys = _kvCache[il].Count;
        int[]? attendable = null;
        if (layer.IndexerProj is not null && numKeys > _hp.IndexerTopK)
        {
            attendable = SelectIndexerTopK(il, layer, qrNormed, normedInput, position, numKeys);
        }

        float kqScale = _kqScale; // precomputed in the constructor -- YaRN-aware, see its derivation there.
        var attnOut = new float[_numHeads * _headDimV];
        int effectiveKeys = attendable?.Length ?? numKeys;
        var scores = new float[effectiveKeys];
        int keyDim = _kvLoraRank + _ropeDim;
        for (int h = 0; h < _numHeads; h++)
        {
            var qHead = qEff.AsSpan(h * keyDim, keyDim);
            for (int t = 0; t < effectiveKeys; t++)
            {
                var kt = _kvCache[il][attendable?[t] ?? t];
                float dot = 0f;
                for (int d = 0; d < keyDim; d++) dot += qHead[d] * kt[d];
                scores[t] = dot * kqScale;
            }
            fixed (float* scoresPtr = scores)
            {
                SimdKernels.SoftmaxInPlace(scoresPtr, effectiveKeys);
            }
            var weighted = new float[_kvLoraRank];
            for (int t = 0; t < effectiveKeys; t++)
            {
                var vt = _kvCache[il][attendable?[t] ?? t]; // V is the leading kvLoraRank channels of the SAME cached vector.
                float w = scores[t];
                for (int d = 0; d < _kvLoraRank; d++) weighted[d] += vt[d] * w;
            }
            var outHead = attnOut.AsSpan(h * _headDimV, _headDimV);
            PerHeadMatVec(layer.WvB!.Value, h, _headDimV, _kvLoraRank, weighted, outHead);
        }

        var result = new float[_embedDim];
        fixed (float* inPtr = attnOut, outPtr = result)
        {
            SimdKernels.MatVec(outPtr, layer.Wo!.Value.DataPtr, inPtr, _embedDim, _numHeads * _headDimV, layer.Wo.Value.DType);
        }
        return result;
    }

    /// <summary>
    /// Lightning indexer, deepseek32's simpler (real raw K, no compression) variant. Indexer Q
    /// from <c>indexer_attn_q_b(qrNormed)</c> + NEOX rope; indexer K from
    /// <c>indexer_attn_k(normedInput)</c>, LayerNorm'd (NOT RMSNorm -- <c>indexer_k_norm</c> has
    /// both a weight and a bias, deepseek32.cpp:116-117/263, matching the reference's plain
    /// <c>LLM_NORM</c> not <c>LLM_NORM_RMS</c>) + NEOX rope, cached raw every token. No Hadamard
    /// (see header point 2).
    /// </summary>
    private int[] SelectIndexerTopK(int il, DeepSeek32LayerTensors layer, float[] qrNormed, float[] normedInput, int position, int numKeys)
    {
        int numIndexerHeads = _hp.IndexerNumHeads;
        int indexerHeadDim = _hp.IndexerHeadSize;
        int indexerNopeDim = indexerHeadDim - _ropeDim;

        var indexerQ = new float[numIndexerHeads * indexerHeadDim];
        fixed (float* inPtr = qrNormed, outPtr = indexerQ)
        {
            SimdKernels.MatVec(outPtr, layer.IndexerAttnQB!.Value.DataPtr, inPtr, numIndexerHeads * indexerHeadDim, qrNormed.Length, layer.IndexerAttnQB.Value.DType);
        }
        for (int h = 0; h < numIndexerHeads; h++)
        {
            ApplyRopeNeox(indexerQ.AsSpan(h * indexerHeadDim + indexerNopeDim, _ropeDim), position, _hp.RopeFreqBase);
        }

        var indexerK = new float[indexerHeadDim];
        fixed (float* inPtr = normedInput, outPtr = indexerK)
        {
            SimdKernels.MatVec(outPtr, layer.IndexerAttnK!.Value.DataPtr, inPtr, indexerHeadDim, _embedDim, layer.IndexerAttnK.Value.DType);
        }
        LayerNormInPlace(indexerK, layer.IndexerKNorm!.Value, layer.IndexerKNormBias);
        ApplyRopeNeox(indexerK.AsSpan(indexerNopeDim, _ropeDim), position, _hp.RopeFreqBase);
        _indexerKCache[il].Add(indexerK);

        var indexerWeightsPerHead = new float[numIndexerHeads];
        fixed (float* inPtr = normedInput, outPtr = indexerWeightsPerHead)
        {
            SimdKernels.MatVec(outPtr, layer.IndexerProj!.Value.DataPtr, inPtr, numIndexerHeads, _embedDim, layer.IndexerProj.Value.DType);
        }
        float indexerScale = 1f / MathF.Sqrt(indexerHeadDim * numIndexerHeads);
        for (int h = 0; h < numIndexerHeads; h++) indexerWeightsPerHead[h] *= indexerScale;

        var k = new float[numKeys * numIndexerHeads * indexerHeadDim];
        var weights = new float[numKeys * numIndexerHeads];
        for (int t = 0; t < numKeys; t++)
        {
            var kt = _indexerKCache[il][t];
            for (int h = 0; h < numIndexerHeads; h++)
            {
                kt.CopyTo(k.AsSpan((t * numIndexerHeads + h) * indexerHeadDim, indexerHeadDim));
                weights[t * numIndexerHeads + h] = indexerWeightsPerHead[h];
            }
        }

        var mask = new float[numKeys]; // all raw cache entries so far are causal by construction.
        var scores = new float[numKeys];
        DeepSeek4Graph.LightningIndexerScore(indexerQ, k, weights, mask, numIndexerHeads, indexerHeadDim, numKeys, scores);

        int topK = Math.Min(_hp.IndexerTopK, numKeys);
        var selected = DeepSeek4Graph.SelectTopKIndices(scores, topK);
        Array.Sort(selected); // keep attention order-independent of score rank; not load-bearing.
        return selected;
    }

    private float[] DenseFfn(DeepSeek32LayerTensors layer, float[] normedInput)
    {
        int ffnDim = (int)layer.FfnGate!.Value.Info.Dimensions[1];
        var gate = new float[ffnDim];
        var up = new float[ffnDim];
        fixed (float* inPtr = normedInput, gatePtr = gate, upPtr = up)
        {
            SimdKernels.MatVec(gatePtr, layer.FfnGate!.Value.DataPtr, inPtr, ffnDim, _embedDim, layer.FfnGate.Value.DType);
            SimdKernels.MatVec(upPtr, layer.FfnUp!.Value.DataPtr, inPtr, ffnDim, _embedDim, layer.FfnUp.Value.DType);
        }
        for (int i = 0; i < ffnDim; i++)
        {
            float g = gate[i];
            up[i] = (g / (1f + MathF.Exp(-g))) * up[i];
        }
        var result = new float[_embedDim];
        fixed (float* inPtr = up, outPtr = result)
        {
            SimdKernels.MatVec(outPtr, layer.FfnDown!.Value.DataPtr, inPtr, _embedDim, ffnDim, layer.FfnDown.Value.DType);
        }
        return result;
    }

    private float[] MoeFfn(DeepSeek32LayerTensors layer, float[] normedInput)
    {
        int numExperts = _hp.NumExperts;
        int topK = _hp.NumExpertsUsed;

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

        // Standard softmax gating (deepseek32 reads expert_gating_func generically rather than
        // hard-requiring sqrt-softplus like deepseek4 -- softmax is the DeepSeek-V2/V3 norm;
        // this port assumes softmax unconditionally, NOT dispatching on the read
        // ExpertGatingFunc value -- a known simplification if a real checkpoint declares
        // something else).
        var scores = new float[numExperts];
        float max = float.NegativeInfinity;
        for (int e = 0; e < numExperts; e++) max = MathF.Max(max, logits[e]);
        float sum = 0f;
        for (int e = 0; e < numExperts; e++) { scores[e] = MathF.Exp(logits[e] - max); sum += scores[e]; }
        for (int e = 0; e < numExperts; e++) scores[e] /= sum;

        var expertIndices = new int[topK];
        var expertWeights = new float[topK];
        DeepSeek4Graph.SelectAndWeightExperts(scores, topK, _hp.ExpertWeightsNorm, _hp.ExpertWeightsScale, expertIndices, expertWeights);

        var result = new float[_embedDim];
        int ffnDim = _hp.ExpertFeedForwardLength;
        var gate = new float[ffnDim];
        var up = new float[ffnDim];
        var down = new float[_embedDim];
        for (int k = 0; k < topK; k++)
        {
            int e = expertIndices[k];
            float w = expertWeights[k];
            PerExpertMatVec(layer.FfnGateExps!.Value, e, normedInput, gate, ffnDim);
            PerExpertMatVec(layer.FfnUpExps!.Value, e, normedInput, up, ffnDim);
            for (int i = 0; i < ffnDim; i++)
            {
                float g = gate[i];
                up[i] = (g / (1f + MathF.Exp(-g))) * up[i];
            }
            PerExpertMatVecDown(layer.FfnDownExps!.Value, e, up, down, ffnDim);
            for (int i = 0; i < _embedDim; i++) result[i] += down[i] * w;
        }

        int sharedFfnDim = ffnDim * _hp.ExpertSharedCount;
        var sGate = new float[sharedFfnDim];
        var sUp = new float[sharedFfnDim];
        fixed (float* inPtr = normedInput, gatePtr = sGate, upPtr = sUp)
        {
            SimdKernels.MatVec(gatePtr, layer.FfnGateShexp!.Value.DataPtr, inPtr, sharedFfnDim, _embedDim, layer.FfnGateShexp.Value.DType);
            SimdKernels.MatVec(upPtr, layer.FfnUpShexp!.Value.DataPtr, inPtr, sharedFfnDim, _embedDim, layer.FfnUpShexp.Value.DType);
        }
        for (int i = 0; i < sharedFfnDim; i++)
        {
            float g = sGate[i];
            sUp[i] = (g / (1f + MathF.Exp(-g))) * sUp[i];
        }
        var sharedResult = new float[_embedDim];
        fixed (float* inPtr = sUp, outPtr = sharedResult)
        {
            SimdKernels.MatVec(outPtr, layer.FfnDownShexp!.Value.DataPtr, inPtr, _embedDim, sharedFfnDim, layer.FfnDownShexp.Value.DType);
        }
        for (int i = 0; i < _embedDim; i++) result[i] += sharedResult[i];

        return result;
    }

    private void PerExpertMatVec(DeepSeek4TensorRef tensor, int expert, float[] input, float[] output, int outDim)
    {
        int inDim = (int)tensor.Info.Dimensions[0];
        long bytesPerRow = ((long)inDim / DTypeInfo.BlockSize(tensor.DType)) * DTypeInfo.BytesPerBlock(tensor.DType);
        byte* expertPtr = tensor.DataPtr + (long)expert * outDim * bytesPerRow;
        fixed (float* inPtr = input, outPtr = output)
        {
            SimdKernels.MatVec(outPtr, expertPtr, inPtr, outDim, inDim, tensor.DType);
        }
    }

    private void PerExpertMatVecDown(DeepSeek4TensorRef tensor, int expert, float[] input, float[] output, int inDim)
    {
        int outDim = (int)tensor.Info.Dimensions[1];
        long bytesPerRow = ((long)inDim / DTypeInfo.BlockSize(tensor.DType)) * DTypeInfo.BytesPerBlock(tensor.DType);
        byte* expertPtr = tensor.DataPtr + (long)expert * outDim * bytesPerRow;
        fixed (float* inPtr = input, outPtr = output)
        {
            SimdKernels.MatVec(outPtr, expertPtr, inPtr, outDim, inDim, tensor.DType);
        }
    }

    /// <summary>Matrix-vector multiply against head <paramref name="head"/>'s slice of a 3D [in, out, numHeads]-shaped GGUF tensor (wk_b/wv_b's layout).</summary>
    private void PerHeadMatVec(DeepSeek4TensorRef tensor, int head, int outDim, int inDim, ReadOnlySpan<float> input, Span<float> output)
    {
        long bytesPerRow = ((long)inDim / DTypeInfo.BlockSize(tensor.DType)) * DTypeInfo.BytesPerBlock(tensor.DType);
        byte* headPtr = tensor.DataPtr + (long)head * outDim * bytesPerRow;
        fixed (float* inPtr = input, outPtr = output)
        {
            SimdKernels.MatVec(outPtr, headPtr, inPtr, outDim, inDim, tensor.DType);
        }
    }

    private static void LayerNormInPlace(float[] x, DeepSeek4TensorRef weight, DeepSeek4TensorRef? bias)
    {
        int n = x.Length;
        float mean = 0f;
        for (int i = 0; i < n; i++) mean += x[i];
        mean /= n;
        float var = 0f;
        for (int i = 0; i < n; i++) { float d = x[i] - mean; var += d * d; }
        var /= n;
        float inv = 1f / MathF.Sqrt(var + 1e-5f);
        float* w = (float*)weight.DataPtr;
        float* b = bias is { } bv ? (float*)bv.DataPtr : null;
        for (int i = 0; i < n; i++)
        {
            x[i] = (x[i] - mean) * inv * w[i] + (b is not null ? b[i] : 0f);
        }
    }

    /// <summary>
    /// Interleaved (LLAMA_ROPE_TYPE_NORM) YaRN-aware RoPE for ONE position -- the single-position
    /// analog of <see cref="SimdKernels.BuildYarnRopeTable"/> (reused formula, not reused code:
    /// that method fills every position from 0 up in one pass, which would be wasteful to call
    /// per-token at large positions in this alpha's unbatched decode loop; this inlines the same
    /// corr-dim/ramp-mix/mscale math for just the position being rotated right now). When YaRN is
    /// inactive (<see cref="_extFactor"/>==0), this collapses to plain RoPE with magnitude
    /// unchanged by <paramref name="attnFactor"/> (mscale stays exactly `attnFactor`, matching
    /// BuildYarnRopeTable's own no-YaRN branch) -- deepseek32.cpp's OWN kq_scale correction
    /// (computed once in the constructor, not here) is layered on top of this, not a
    /// substitute for it.
    /// </summary>
    private void ApplyYarnRope(Span<float> x, int position)
    {
        int dim = x.Length;
        int half = dim / 2;
        float theta = _hp.RopeFreqBase;

        static float CorrDim(int nDims, int nCtxOrig, float nRot, float b) =>
            nDims * MathF.Log(nCtxOrig / (nRot * 2f * MathF.PI)) / (2f * MathF.Log(b));

        float corrLow = 0f, corrHigh = half * 2f - 1f;
        if (_origCtxLen > 0)
        {
            corrLow = MathF.Max(0f, MathF.Floor(CorrDim(dim, _origCtxLen, 32f, theta)));
            corrHigh = MathF.Min(dim - 1, MathF.Ceiling(CorrDim(dim, _origCtxLen, 1f, theta)));
        }

        float thetaScale = MathF.Pow(theta, -2f / dim);
        float thetaExtrap = position;
        for (int i = 0; i < half; i++)
        {
            float thetaInterp = _freqScale * thetaExtrap;
            float thetaFinal = thetaInterp;
            float mscale = _attnFactorBaseline;
            if (_extFactor != 0f)
            {
                float y = (i - corrLow) / MathF.Max(0.001f, corrHigh - corrLow);
                float rampMix = (1f - MathF.Min(1f, MathF.Max(0f, y))) * _extFactor;
                thetaFinal = thetaInterp * (1f - rampMix) + thetaExtrap * rampMix;
                mscale *= 1f + 0.1f * MathF.Log(1f / _freqScale);
            }
            float cos = MathF.Cos(thetaFinal) * mscale, sin = MathF.Sin(thetaFinal) * mscale;
            float x0 = x[2 * i], x1 = x[2 * i + 1];
            x[2 * i] = x0 * cos - x1 * sin;
            x[2 * i + 1] = x0 * sin + x1 * cos;
            thetaExtrap *= thetaScale;
        }
    }

    private static void ApplyRopeNeox(Span<float> x, int position, float freqBase)
    {
        int dim = x.Length;
        int half = dim / 2;
        for (int i = 0; i < half; i++)
        {
            float freq = MathF.Pow(freqBase, -2f * i / dim);
            float theta = position * freq;
            float cos = MathF.Cos(theta), sin = MathF.Sin(theta);
            float x0 = x[i], x1 = x[i + half];
            x[i] = x0 * cos - x1 * sin;
            x[i + half] = x0 * sin + x1 * cos;
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

    // ── IForwardPass minimal surface ────────────────────────────────────────────────────────

    public ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, int startPos = 0)
    {
        ReadOnlySpan<float> last = default;
        for (int i = 0; i < tokens.Count; i++) last = Forward(tokens[i], startPos + i).ToArray();
        return last;
    }

    public void TruncateTo(int length)
    {
        if (length == 0) { ResetCache(); return; }
        int current = _kvCache.Length > 0 ? _kvCache[0].Count : 0;
        if (length != current)
        {
            throw new NotSupportedException(
                "DeepSeek32ForwardPass (alpha): only full reset (TruncateTo(0)) or a no-op is supported.");
        }
    }

    public void ResetCache()
    {
        for (int il = 0; il < _numLayer; il++)
        {
            _kvCache[il].Clear();
            _indexerKCache[il].Clear();
        }
    }

    public void Dispose() { }
}
