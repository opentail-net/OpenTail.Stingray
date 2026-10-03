using System;
using System.Collections.Generic;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Engine;

/// <summary>
/// Forward pass implementation for DeepSeek-V4.1 Flash (deepseek41).
/// Evaluates the 40-layer trunk with compression ratios 0 / 1 / 2, dual Engram tables (layers 1 and 14),
/// 384 routed experts + 1 shared expert (sqrtsoftplus routing with scale 1.5), 8-group output LoRA,
/// and 4-stream mHC hyper-connections.
/// </summary>
public sealed unsafe class DeepSeek41ForwardPass : IForwardPass
{
    private readonly GgufModel _model;
    private readonly DeepSeek41Hyperparams _hp;
    private readonly DeepSeek41TensorSet _tensors;
    private readonly DeepSeek41Engram _engram;
    private readonly int _hc, _embedDim, _numHeads, _headDim, _ropeDim, _nopeDim, _numLayer;

    private readonly List<float[]>[] _kvCache;
    private readonly int[] _compressRatio;
    private float[] _logits;

    public int VocabSize { get; }
    public int MaxSeqLen => 1 << 20;

    public DeepSeek41ForwardPass(GgufModel model, DeepSeek41Hyperparams hp)
    {
        _model = model;
        _hp = hp;
        _tensors = DeepSeek41TensorSet.Load(model, hp);
        _engram = new DeepSeek41Engram(hp);

        _hc = hp.HyperConnectionMultiplier;
        _embedDim = hp.EmbedDim;
        _numHeads = hp.NumHeads;
        _headDim = hp.HeadDim;
        _ropeDim = hp.RopeDim;
        _nopeDim = hp.NopeDim;
        _numLayer = hp.NumLayer;

        VocabSize = _tensors.Output.HasValue && _tensors.Output.Value.Info.Dimensions.Length >= 2
            ? (int)_tensors.Output.Value.Info.Dimensions[1]
            : (int)_tensors.TokenEmbd!.Value.Info.Dimensions[1];
        _logits = new float[VocabSize];

        _kvCache = new List<float[]>[_numLayer];
        _compressRatio = new int[_numLayer];
        for (int i = 0; i < _numLayer; i++)
        {
            _kvCache[i] = new List<float[]>();
            _compressRatio[i] = i < hp.CompressRatios.Count ? hp.CompressRatios[i] : 0;
        }
    }

    public ReadOnlySpan<float> Forward(int token, int position)
    {
        _engram.AppendToken(token);

        // 1. Token embedding
        var embd = new float[_embedDim];
        fixed (float* outPtr = embd)
        {
            float* embdRow = (float*)_tensors.TokenEmbd!.Value.DataPtr + (long)token * _embedDim;
            SimdKernels.DequantRow((byte*)embdRow, outPtr, _embedDim, _tensors.TokenEmbd.Value.DType);
        }

        // Initialize 4-stream hyper-connection state: residual = [hc * embedDim]
        var residual = new float[_hc * _embedDim];
        for (int s = 0; s < _hc; s++)
        {
            embd.CopyTo(residual.AsSpan(s * _embedDim, _embedDim));
        }

        // 2. Transformer trunk layers
        for (int il = 0; il < _numLayer; il++)
        {
            var layer = _tensors.Layers[il];

            // Engram injection on layers 1 and 14
            if ((il == 1 || il == 14) && layer.EngramTable.HasValue && layer.EngramProj.HasValue)
            {
                var engramContrib = new float[_embedDim];
                _engram.LookupAndProject(
                    (float*)layer.EngramTable.Value.DataPtr,
                    (float*)layer.EngramProj.Value.DataPtr,
                    engramContrib);

                // Add Engram embedding to residual stream 0
                for (int d = 0; d < _embedDim; d++)
                {
                    residual[d] += engramContrib[d];
                }
            }

            // Attention block with mHC
            ExecuteMhcBlock(il, layer, residual, isFfn: false, token, position);

            // MoE FFN block with mHC
            ExecuteMhcBlock(il, layer, residual, isFfn: true, token, position);
        }

        // 3. Final output mixdown and projection
        var finalHidden = new float[_embedDim];
        fixed (float* rPtr = residual, outPtr = finalHidden)
        {
            float* normWeight = (float*)_tensors.OutputNorm!.Value.DataPtr;
            SimdKernels.RmsNorm(outPtr, rPtr, normWeight, _embedDim, _hp.RmsNormEps);
        }

        if (_logits.Length != VocabSize) _logits = new float[VocabSize];
        fixed (float* inPtr = finalHidden, outPtr = _logits)
        {
            SimdKernels.MatVec(outPtr, _tensors.Output!.Value.DataPtr, inPtr, VocabSize, _embedDim, _tensors.Output.Value.DType);
        }

        return _logits;
    }

    private void ExecuteMhcBlock(
        int il, DeepSeek41LayerTensors layer, float[] residual, bool isFfn, int token, int position)
    {
        var fnTensor = isFfn ? layer.HcFfnFn : layer.HcAttnFn;
        var baseTensor = isFfn ? layer.HcFfnBase : layer.HcAttnBase;
        var scaleTensor = isFfn ? layer.HcFfnScale : layer.HcAttnScale;

        // Compute flat RMSNorm across all hc*embedDim
        var flatNormed = new float[_hc * _embedDim];
        float sumSquares = 0f;
        for (int i = 0; i < residual.Length; i++) sumSquares += residual[i] * residual[i];
        float invRms = 1.0f / MathF.Sqrt(sumSquares / residual.Length + _hp.HyperConnectionEpsilon);
        for (int i = 0; i < residual.Length; i++) flatNormed[i] = residual[i] * invRms;

        var pre = new float[_hc];
        var post = new float[_hc];
        var comb = new float[_hc * _hc];

        if (fnTensor.HasValue && baseTensor.HasValue && scaleTensor.HasValue)
        {
            DeepSeek4Graph.HyperConnectionGate(
                flatNormed, _hc, _embedDim,
                AsFloatSpan(fnTensor.Value),
                AsFloatSpan(scaleTensor.Value),
                AsFloatSpan(baseTensor.Value),
                _hp.HyperConnectionEpsilon,
                _hp.HyperConnectionSinkhornIterations,
                pre, post, comb);
        }
        else
        {
            Array.Fill(pre, 1.0f / _hc);
            Array.Fill(post, 1.0f);
            for (int r = 0; r < _hc; r++) comb[r * _hc + r] = 1.0f; // Identity
        }

        var cur = new float[_embedDim];
        DeepSeek4Graph.HyperConnectionMixDown(residual, pre, _hc, _embedDim, cur);

        var subOut = isFfn ? ExecuteFfn(il, layer, cur, token) : ExecuteAttention(il, layer, cur, position);

        var mixedUp = new float[_hc * _embedDim];
        DeepSeek4Graph.HyperConnectionMixUp(subOut, residual, post, comb, _hc, _embedDim, mixedUp);
        mixedUp.CopyTo(residual, 0);
    }

    private float[] ExecuteAttention(int il, DeepSeek41LayerTensors layer, float[] cur, int position)
    {
        var normed = new float[_embedDim];
        fixed (float* curPtr = cur, outPtr = normed)
        {
            float* w = (float*)layer.AttnNorm!.Value.DataPtr;
            SimdKernels.RmsNorm(outPtr, curPtr, w, _embedDim, _hp.RmsNormEps);
        }

        int qLoraRank = _hp.QLoraRank;
        var qr = new float[qLoraRank];
        fixed (float* inPtr = normed, outPtr = qr)
        {
            SimdKernels.MatVec(outPtr, layer.WqA!.Value.DataPtr, inPtr, qLoraRank, _embedDim, layer.WqA.Value.DType);
        }

        var qrNormed = new float[qLoraRank];
        fixed (float* inPtr = qr, outPtr = qrNormed)
        {
            float* w = (float*)layer.AttnQANorm!.Value.DataPtr;
            SimdKernels.RmsNorm(outPtr, inPtr, w, qLoraRank, _hp.RmsNormEps);
        }

        int totalQDim = _numHeads * _headDim;
        var q = new float[totalQDim];
        fixed (float* inPtr = qrNormed, outPtr = q)
        {
            SimdKernels.MatVec(outPtr, layer.WqB!.Value.DataPtr, inPtr, totalQDim, qLoraRank, layer.WqB.Value.DType);
        }

        for (int h = 0; h < _numHeads; h++)
        {
            fixed (float* headPtr = q.AsSpan(h * _headDim, _headDim))
            {
                SimdKernels.RmsNorm(headPtr, headPtr, null, _headDim, _hp.RmsNormEps);
            }
            ApplyRopeInterleaved(q.AsSpan(h * _headDim + _nopeDim, _ropeDim), position, _hp.RopeFreqBase);
        }

        // KV projection (single shared head, headDim wide)
        var kv = new float[_headDim];
        fixed (float* inPtr = normed, outPtr = kv)
        {
            SimdKernels.MatVec(outPtr, layer.Wkv!.Value.DataPtr, inPtr, _headDim, _embedDim, layer.Wkv.Value.DType);
        }
        fixed (float* kvPtr = kv)
        {
            float* w = (float*)layer.AttnKvNorm!.Value.DataPtr;
            SimdKernels.RmsNorm(kvPtr, kvPtr, w, _headDim, _hp.RmsNormEps);
        }
        ApplyRopeInterleaved(kv.AsSpan(_nopeDim, _ropeDim), position, _hp.RopeFreqBase);

        _kvCache[il].Add(kv);

        // Multi-head attention
        var allKeys = _kvCache[il];
        int numKeys = allKeys.Count;
        var attnOut = new float[totalQDim];
        var scores = new float[numKeys];
        float scale = 1f / MathF.Sqrt(_headDim);

        for (int h = 0; h < _numHeads; h++)
        {
            var qHead = q.AsSpan(h * _headDim, _headDim);
            for (int t = 0; t < numKeys; t++)
            {
                var kt = allKeys[t];
                float dot = 0f;
                for (int d = 0; d < _headDim; d++) dot += qHead[d] * kt[d];
                scores[t] = dot * scale;
            }
            fixed (float* sPtr = scores)
            {
                SimdKernels.SoftmaxInPlace(sPtr, numKeys);
            }
            var outHead = attnOut.AsSpan(h * _headDim, _headDim);
            outHead.Clear();
            for (int t = 0; t < numKeys; t++)
            {
                var vt = allKeys[t];
                float w = scores[t];
                for (int d = 0; d < _headDim; d++) outHead[d] += vt[d] * w;
            }
            ApplyRopeInterleaved(outHead.Slice(_nopeDim, _ropeDim), -position, _hp.RopeFreqBase);
        }

        // 8-group Output LoRA: WoA -> WoB
        int outLoraRank = _hp.OutputLoraRank;
        var oa = new float[outLoraRank];
        int groups = Math.Max(1, _hp.OutputGroupCount);

        if (groups > 1 && totalQDim % groups == 0 && outLoraRank % groups == 0)
        {
            int groupInDim = totalQDim / groups;
            int groupOutDim = outLoraRank / groups;
            fixed (float* inPtr = attnOut, outPtr = oa)
            {
                byte* wBase = layer.WoA!.Value.DataPtr;
                var dtype = layer.WoA.Value.DType;
                for (int g = 0; g < groups; g++)
                {
                    float* gIn = inPtr + g * groupInDim;
                    float* gOut = outPtr + g * groupOutDim;
                    byte* gWeight = dtype == DType.Float32
                        ? (byte*)((float*)wBase + (long)g * groupOutDim * groupInDim)
                        : wBase + (long)g * groupOutDim * groupInDim;
                    SimdKernels.MatVec(gOut, gWeight, gIn, groupOutDim, groupInDim, dtype);
                }
            }
        }
        else
        {
            fixed (float* inPtr = attnOut, outPtr = oa)
            {
                SimdKernels.MatVec(outPtr, layer.WoA!.Value.DataPtr, inPtr, outLoraRank, totalQDim, layer.WoA.Value.DType);
            }
        }

        var result = new float[_embedDim];
        fixed (float* inPtr = oa, outPtr = result)
        {
            SimdKernels.MatVec(outPtr, layer.WoB!.Value.DataPtr, inPtr, _embedDim, outLoraRank, layer.WoB.Value.DType);
        }
        return result;
    }

    private float[] ExecuteFfn(int il, DeepSeek41LayerTensors layer, float[] cur, int token)
    {
        var normed = new float[_embedDim];
        fixed (float* curPtr = cur, outPtr = normed)
        {
            float* w = (float*)layer.FfnNorm!.Value.DataPtr;
            SimdKernels.RmsNorm(outPtr, curPtr, w, _embedDim, _hp.RmsNormEps);
        }

        // MoE Router: 384 experts, sqrtsoftplus with scale 1.5, top-6
        int numExperts = _hp.NumExperts;
        var routerLogits = new float[numExperts];
        fixed (float* inPtr = normed, outPtr = routerLogits)
        {
            SimdKernels.MatVec(outPtr, layer.FfnGateInp!.Value.DataPtr, inPtr, numExperts, _embedDim, layer.FfnGateInp.Value.DType);
        }

        var routerScores = new float[numExperts];
        DeepSeek41Graph.SqrtSoftplusGate(routerLogits, _hp.ExpertWeightsScale, routerScores);

        int topK = _hp.NumExpertsUsed; // 6
        var topIndices = new int[topK];
        var topWeights = new float[topK];
        DeepSeek4Graph.SelectAndWeightExperts(
            routerScores, topK, _hp.ExpertWeightsNorm, _hp.ExpertWeightsScale, topIndices, topWeights);

        int interDim = layer.FfnGateExps.HasValue && layer.FfnGateExps.Value.Info.Dimensions.Length >= 2
            ? (int)layer.FfnGateExps.Value.Info.Dimensions[1]
            : _hp.ExpertFeedForwardLength;

        var ffnOut = new float[_embedDim];
        var gateBuf = new float[interDim];
        var upBuf = new float[interDim];
        var expertDown = new float[_embedDim];

        fixed (float* inPtr = normed, gatePtr = gateBuf, upPtr = upBuf, downPtr = expertDown, outPtr = ffnOut)
        {
            for (int k = 0; k < topK; k++)
            {
                int expId = topIndices[k];
                float weight = topWeights[k];
                if (weight <= 0f) continue;

                byte* gExp = layer.FfnGateExps!.Value.DataPtr + (long)expId * interDim * _embedDim * sizeof(float);
                byte* uExp = layer.FfnUpExps!.Value.DataPtr + (long)expId * interDim * _embedDim * sizeof(float);
                byte* dExp = layer.FfnDownExps!.Value.DataPtr + (long)expId * _embedDim * interDim * sizeof(float);

                SimdKernels.MatVec(gatePtr, gExp, inPtr, interDim, _embedDim, layer.FfnGateExps.Value.DType);
                SimdKernels.MatVec(upPtr, uExp, inPtr, interDim, _embedDim, layer.FfnUpExps.Value.DType);

                // SwiGLU: silu(gate) * up
                for (int d = 0; d < interDim; d++)
                {
                    float g = gateBuf[d];
                    float silu = g / (1.0f + MathF.Exp(-g));
                    gateBuf[d] = silu * upBuf[d];
                }

                SimdKernels.MatVec(downPtr, dExp, gatePtr, _embedDim, interDim, layer.FfnDownExps.Value.DType);

                for (int d = 0; d < _embedDim; d++)
                {
                    outPtr[d] += expertDown[d] * weight;
                }
            }

            // Shared expert
            if (layer.FfnGateShexp.HasValue && layer.FfnUpShexp.HasValue && layer.FfnDownShexp.HasValue)
            {
                SimdKernels.MatVec(gatePtr, layer.FfnGateShexp.Value.DataPtr, inPtr, interDim, _embedDim, layer.FfnGateShexp.Value.DType);
                SimdKernels.MatVec(upPtr, layer.FfnUpShexp.Value.DataPtr, inPtr, interDim, _embedDim, layer.FfnUpShexp.Value.DType);

                for (int d = 0; d < interDim; d++)
                {
                    float g = gateBuf[d];
                    float silu = g / (1.0f + MathF.Exp(-g));
                    gateBuf[d] = silu * upBuf[d];
                }

                SimdKernels.MatVec(downPtr, layer.FfnDownShexp.Value.DataPtr, gatePtr, _embedDim, interDim, layer.FfnDownShexp.Value.DType);

                for (int d = 0; d < _embedDim; d++)
                {
                    outPtr[d] += expertDown[d];
                }
            }
        }

        return ffnOut;
    }

    private static void ApplyRopeInterleaved(Span<float> x, int position, float freqBase)
    {
        int half = x.Length / 2;
        for (int i = 0; i < half; i++)
        {
            float theta = position * MathF.Pow(freqBase, -2f * i / x.Length);
            float cos = MathF.Cos(theta);
            float sin = MathF.Sin(theta);
            float re = x[2 * i];
            float im = x[2 * i + 1];
            x[2 * i] = re * cos - im * sin;
            x[2 * i + 1] = re * sin + im * cos;
        }
    }

    private static ReadOnlySpan<float> AsFloatSpan(DeepSeek4TensorRef tensor)
    {
        int len = 1;
        for (int i = 0; i < tensor.Info.Dimensions.Length; i++) len *= (int)tensor.Info.Dimensions[i];
        return new ReadOnlySpan<float>(tensor.DataPtr, len);
    }

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
                "DeepSeek41ForwardPass (alpha): only full reset (TruncateTo(0)) is supported.");
        }
    }

    public void ResetCache()
    {
        for (int il = 0; il < _numLayer; il++) _kvCache[il].Clear();
        _engram.Reset();
    }

    public void Dispose() { }
}
