using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Engine;

// ============================================================================================
// ALPHA / UNTESTED -- Forward-pass implementation for GLM-DSA (GLM-5.2). Ported 2026-10-03 from
// examples/llama.cpp/llama.cpp/src/models/glm-dsa.cpp.
//
// Key components:
//  1. Classic MLA attention with weight absorption (576-wide compressed latent KV cache per token).
//  2. DSA Lightning Indexer with orthonormal Sylvester-Walsh-Hadamard transform (PrismHadamard)
//     and top-k sparse key selection.
//  3. Shared indexer reuse schedule: full layers refresh top_k; intermediate layers reuse prev_top_k.
//  4. Leading dense SwiGLU blocks followed by 256-expert MoE (top-8 routed, sigmoid gating,
//     renormalized weights, scale 2.5) + shared expert.
// ============================================================================================

/// <summary>
/// Forward pass for GLM-DSA ("glm-dsa" GGUF architecture).
/// </summary>
public sealed unsafe class GlmDsaForwardPass : IForwardPass
{
    private readonly GgufModel _model;
    private readonly GlmDsaHyperparams _hp;
    private readonly GlmDsaTensorSet _tensors;
    private readonly int _embedDim, _numHeads, _headDimK, _headDimV, _ropeDim, _nopeDim, _kvLoraRank, _numLayer;

    // Single raw MLA K cache per layer: each entry is [kvLoraRank + ropeDim] wide (512 + 64 = 576 floats).
    private readonly List<float[]>[] _kvCache;

    // Raw indexer K cache: one indexerHeadDim-wide vector per token per layer.
    private readonly List<float[]>[] _indexerKCache;

    private readonly float _freqScale, _extFactor, _attnFactorBaseline, _kqScale;
    private readonly int _origCtxLen;

    public GlmDsaForwardPass(GgufModel model, GlmDsaHyperparams hp)
    {
        _model = model;
        _hp = hp;
        _tensors = GlmDsaTensorSet.Load(model, hp);
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
        _attnFactorBaseline = 1f;
        _origCtxLen = hp.RopeYarnOrigCtxLen;

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

        VocabSize = (int)_tensors.TokEmbd.Info.Dimensions[1];
    }

    public int VocabSize { get; private set; }
    public int MaxSeqLen => 1 << 20;

    public ReadOnlySpan<float> Forward(int token, int position)
    {
        var cur = new float[_embedDim];
        EmbedTokenInto(token, cur);

        int[]? prevTopK = null;

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

            var attnOut = MlaAttention(il, layer, attnNormed, position, ref prevTopK);
            for (int i = 0; i < _embedDim; i++) cur[i] = residual[i] + attnOut[i];

            residual = (float[])cur.Clone();
            var ffnNormed = new float[_embedDim];
            fixed (float* inPtr = cur, outPtr = ffnNormed)
            {
                float* weightPtr = (float*)layer.FfnNorm!.Value.DataPtr;
                SimdKernels.RmsNorm(outPtr, inPtr, weightPtr, _embedDim, _hp.RmsNormEps);
            }

            var ffnOut = il < _hp.LeadingDenseBlockCount
                ? DenseFfn(layer, ffnNormed)
                : MoeFfn(layer, ffnNormed);

            for (int i = 0; i < _embedDim; i++) cur[i] = residual[i] + ffnOut[i];
        }

        var outputNormed = new float[_embedDim];
        fixed (float* inPtr = cur, outPtr = outputNormed)
        {
            float* weightPtr = (float*)_tensors.OutputNorm.DataPtr;
            SimdKernels.RmsNorm(outPtr, inPtr, weightPtr, _embedDim, _hp.RmsNormEps);
        }

        var logits = new float[VocabSize];
        fixed (float* inPtr = outputNormed, outPtr = logits)
        {
            SimdKernels.MatVec(outPtr, _tensors.Output.DataPtr, inPtr, VocabSize, _embedDim, _tensors.Output.DType);
        }
        return logits;
    }

    private void EmbedTokenInto(int token, float[] destination)
    {
        var tensor = _tensors.TokEmbd;
        int row = token;
        int cols = _embedDim;
        long bytesPerRow = ((long)cols / DTypeInfo.BlockSize(tensor.DType)) * DTypeInfo.BytesPerBlock(tensor.DType);
        byte* src = tensor.DataPtr + (long)row * bytesPerRow;
        fixed (float* dst = destination)
        {
            SimdKernels.DequantRow(src, dst, cols, tensor.DType);
        }
    }

    private float[] MlaAttention(int il, GlmDsaLayerTensors layer, float[] normedInput, int position, ref int[]? prevTopK)
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

        int numKeys = _kvCache[il].Count + 1; // including current token
        int[]? attendable;

        // DSA Lightning Indexer
        if (_hp.IsIndexerFull(il) && layer.IndexerProj is not null)
        {
            attendable = RunFullIndexer(il, layer, qrNormed, normedInput, position, numKeys);
            prevTopK = attendable;
        }
        else
        {
            // Reuse previous top-k for shared layers, ensuring current token index (numKeys - 1) is present
            if (prevTopK is not null)
            {
                int currentTokenIdx = numKeys - 1;
                if (!Array.Exists(prevTopK, idx => idx == currentTokenIdx))
                {
                    attendable = new int[prevTopK.Length + 1];
                    prevTopK.CopyTo(attendable, 0);
                    attendable[^1] = currentTokenIdx;
                }
                else
                {
                    attendable = prevTopK;
                }
            }
            else
            {
                attendable = null;
            }
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

        // Kcur = concat(kv_cmpr, k_pe), single MQA-shared K/V source cached raw
        var kCur = new float[_kvLoraRank + _ropeDim];
        kvCmpr.CopyTo(kCur, 0);
        kPe.CopyTo(kCur, _kvLoraRank);
        _kvCache[il].Add(kCur);

        // Absorb q_nope through wk_b per head: q_nope_absorbed[h] = wk_b[h]^T . q_nope[h]
        var qEff = new float[_numHeads * (_kvLoraRank + _ropeDim)];
        for (int h = 0; h < _numHeads; h++)
        {
            var qNopeHead = q.AsSpan(h * _headDimK, _nopeDim);
            var absorbed = new float[_kvLoraRank];
            PerHeadMatVec(layer.WkB!.Value, h, _kvLoraRank, _nopeDim, qNopeHead, absorbed);
            absorbed.CopyTo(qEff.AsSpan(h * (_kvLoraRank + _ropeDim), _kvLoraRank));
            q.AsSpan(h * _headDimK + _nopeDim, _ropeDim).CopyTo(qEff.AsSpan(h * (_kvLoraRank + _ropeDim) + _kvLoraRank, _ropeDim));
        }

        float kqScale = _kqScale;
        var attnOut = new float[_numHeads * _headDimV];
        int effectiveKeys = attendable?.Length ?? numKeys;
        var scores = new float[effectiveKeys];
        int keyDim = _kvLoraRank + _ropeDim;

        for (int h = 0; h < _numHeads; h++)
        {
            var qHead = qEff.AsSpan(h * keyDim, keyDim);
            for (int t = 0; t < effectiveKeys; t++)
            {
                int kIdx = attendable?[t] ?? t;
                var kt = _kvCache[il][kIdx];
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
                int kIdx = attendable?[t] ?? t;
                var vt = _kvCache[il][kIdx];
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

    private int[] RunFullIndexer(int il, GlmDsaLayerTensors layer, float[] qrNormed, float[] normedInput, int position, int numKeys)
    {
        int numIndexerHeads = _hp.IndexerNumHeads;
        int indexerHeadDim = _hp.IndexerHeadSize; // 128
        int nRot = _hp.RopeDim; // 64

        var indexerQ = new float[numIndexerHeads * indexerHeadDim];
        fixed (float* inPtr = qrNormed, outPtr = indexerQ)
        {
            SimdKernels.MatVec(outPtr, layer.IndexerAttnQB!.Value.DataPtr, inPtr, numIndexerHeads * indexerHeadDim, qrNormed.Length, layer.IndexerAttnQB.Value.DType);
        }

        // Layout per head is [rope | nope] -> first nRot channels are rotated
        for (int h = 0; h < numIndexerHeads; h++)
        {
            ApplyRopeNorm(indexerQ.AsSpan(h * indexerHeadDim, nRot), position, _hp.RopeFreqBase);
        }

        // Orthonormal Sylvester-Walsh-Hadamard transform on indexer_q in-place
        fixed (float* qPtr = indexerQ)
        {
            PrismHadamard.ApplySylvesterHadamard(qPtr, numIndexerHeads * indexerHeadDim, 128);
        }

        var indexerK = new float[indexerHeadDim];
        fixed (float* inPtr = normedInput, outPtr = indexerK)
        {
            SimdKernels.MatVec(outPtr, layer.IndexerAttnK!.Value.DataPtr, inPtr, indexerHeadDim, _embedDim, layer.IndexerAttnK.Value.DType);
        }
        LayerNormInPlace(indexerK, layer.IndexerKNorm!.Value, layer.IndexerKNormBias);
        ApplyRopeNorm(indexerK.AsSpan(0, nRot), position, _hp.RopeFreqBase);

        // Orthonormal Sylvester-Walsh-Hadamard transform on indexer_k in-place
        fixed (float* kPtr = indexerK)
        {
            PrismHadamard.ApplySylvesterHadamard(kPtr, indexerHeadDim, 128);
        }
        _indexerKCache[il].Add(indexerK);

        var indexerWeightsPerHead = new float[numIndexerHeads];
        fixed (float* inPtr = normedInput, outPtr = indexerWeightsPerHead)
        {
            SimdKernels.MatVec(outPtr, layer.IndexerProj!.Value.DataPtr, inPtr, numIndexerHeads, _embedDim, layer.IndexerProj.Value.DType);
        }
        float indexerScale = 1f / MathF.Sqrt(indexerHeadDim * numIndexerHeads);
        for (int h = 0; h < numIndexerHeads; h++) indexerWeightsPerHead[h] *= indexerScale;

        var kAll = new float[numKeys * numIndexerHeads * indexerHeadDim];
        var weightsAll = new float[numKeys * numIndexerHeads];
        for (int t = 0; t < numKeys; t++)
        {
            var kt = _indexerKCache[il][t];
            for (int h = 0; h < numIndexerHeads; h++)
            {
                kt.CopyTo(kAll.AsSpan((t * numIndexerHeads + h) * indexerHeadDim, indexerHeadDim));
                weightsAll[t * numIndexerHeads + h] = indexerWeightsPerHead[h];
            }
        }

        var mask = new float[numKeys];
        var scores = new float[numKeys];
        DeepSeek4Graph.LightningIndexerScore(indexerQ, kAll, weightsAll, mask, numIndexerHeads, indexerHeadDim, numKeys, scores);

        int topK = Math.Min(_hp.IndexerTopK, numKeys);
        var selected = DeepSeek4Graph.SelectTopKIndices(scores, topK);
        Array.Sort(selected);
        return selected;
    }

    private float[] DenseFfn(GlmDsaLayerTensors layer, float[] normedInput)
    {
        int interDim = (int)layer.FfnGate!.Value.Info.Dimensions[1];
        var gate = new float[interDim];
        var up = new float[interDim];
        fixed (float* inPtr = normedInput, gPtr = gate, uPtr = up)
        {
            SimdKernels.MatVec(gPtr, layer.FfnGate.Value.DataPtr, inPtr, interDim, _embedDim, layer.FfnGate.Value.DType);
            SimdKernels.MatVec(uPtr, layer.FfnUp!.Value.DataPtr, inPtr, interDim, _embedDim, layer.FfnUp.Value.DType);
        }
        for (int i = 0; i < interDim; i++)
        {
            float g = gate[i];
            float silu = g / (1f + MathF.Exp(-g));
            gate[i] = silu * up[i];
        }
        var down = new float[_embedDim];
        fixed (float* inPtr = gate, outPtr = down)
        {
            SimdKernels.MatVec(outPtr, layer.FfnDown!.Value.DataPtr, inPtr, _embedDim, interDim, layer.FfnDown.Value.DType);
        }
        return down;
    }

    private float[] MoeFfn(GlmDsaLayerTensors layer, float[] normedInput)
    {
        int numExperts = _hp.NumExperts;
        int topK = _hp.NumExpertsUsed;
        int interDim = _hp.ExpertFeedForwardLength > 0
            ? _hp.ExpertFeedForwardLength
            : (int)layer.FfnGateExps!.Value.Info.Dimensions[1];

        // 1. Router logits
        var logits = new float[numExperts];
        fixed (float* inPtr = normedInput, outPtr = logits)
        {
            SimdKernels.MatVec(outPtr, layer.FfnGateInp!.Value.DataPtr, inPtr, numExperts, _embedDim, layer.FfnGateInp.Value.DType);
        }
        if (layer.FfnExpProbsB is { } biasRef)
        {
            float* bias = (float*)biasRef.DataPtr;
            for (int e = 0; e < numExperts; e++) logits[e] += bias[e];
        }

        // 2. Sigmoid gating
        var scores = new float[numExperts];
        for (int e = 0; e < numExperts; e++)
        {
            scores[e] = 1f / (1f + MathF.Exp(-logits[e]));
        }

        // 3. Top-k selection
        int[] topIndices = DeepSeek4Graph.SelectTopKIndices(scores, topK);
        var weights = new float[topK];
        float sumWeight = 0f;
        for (int k = 0; k < topK; k++)
        {
            weights[k] = scores[topIndices[k]];
            sumWeight += weights[k];
        }
        if (_hp.ExpertWeightsNorm && sumWeight > 0f)
        {
            float inv = 1f / sumWeight;
            for (int k = 0; k < topK; k++) weights[k] *= inv;
        }
        for (int k = 0; k < topK; k++) weights[k] *= _hp.ExpertWeightsScale;

        // 4. Routed experts accumulation
        var accumulated = new float[_embedDim];
        var expertGate = new float[interDim];
        var expertUp = new float[interDim];
        var expertDown = new float[_embedDim];

        for (int k = 0; k < topK; k++)
        {
            int expert = topIndices[k];
            float w = weights[k];

            PerExpertMatVec(layer.FfnGateExps!.Value, expert, normedInput, expertGate, interDim);
            PerExpertMatVec(layer.FfnUpExps!.Value, expert, normedInput, expertUp, interDim);

            for (int i = 0; i < interDim; i++)
            {
                float g = expertGate[i];
                float silu = g / (1f + MathF.Exp(-g));
                expertGate[i] = silu * expertUp[i];
            }

            PerExpertMatVecDown(layer.FfnDownExps!.Value, expert, expertGate, expertDown, interDim);
            for (int i = 0; i < _embedDim; i++) accumulated[i] += expertDown[i] * w;
        }

        // 5. Shared expert
        int sharedInterDim = (int)layer.FfnGateShexp!.Value.Info.Dimensions[1];
        var shGate = new float[sharedInterDim];
        var shUp = new float[sharedInterDim];
        fixed (float* inPtr = normedInput, gPtr = shGate, uPtr = shUp)
        {
            SimdKernels.MatVec(gPtr, layer.FfnGateShexp.Value.DataPtr, inPtr, sharedInterDim, _embedDim, layer.FfnGateShexp.Value.DType);
            SimdKernels.MatVec(uPtr, layer.FfnUpShexp!.Value.DataPtr, inPtr, sharedInterDim, _embedDim, layer.FfnUpShexp.Value.DType);
        }
        for (int i = 0; i < sharedInterDim; i++)
        {
            float g = shGate[i];
            float silu = g / (1f + MathF.Exp(-g));
            shGate[i] = silu * shUp[i];
        }
        var shDown = new float[_embedDim];
        fixed (float* inPtr = shGate, outPtr = shDown)
        {
            SimdKernels.MatVec(outPtr, layer.FfnDownShexp!.Value.DataPtr, inPtr, _embedDim, sharedInterDim, layer.FfnDownShexp.Value.DType);
        }

        for (int i = 0; i < _embedDim; i++) accumulated[i] += shDown[i];
        return accumulated;
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

    private static void ApplyRopeNorm(Span<float> x, int position, float freqBase)
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
                "GlmDsaForwardPass (alpha): only full reset (TruncateTo(0)) or a no-op is supported.");
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
