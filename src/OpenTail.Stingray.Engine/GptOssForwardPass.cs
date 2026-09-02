using System.Numerics.Tensors;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Engine;

// ============================================================================================
// Forward-pass dispatch for gpt-oss: standard GQA attention with biased
// QKVO, attention sinks, alternating sliding/full-window masking, and biased MoE with the OAI
// SwiGLU + select-then-softmax gating.
// ============================================================================================

/// <summary>
/// Implements <see cref="IForwardPass"/> for gpt-oss: GQA attention with biased
/// QKVO projections, attention sinks, alternating sliding-window/full masking, and MoE with
/// per-expert biases, the OAI SwiGLU activation, and select-then-softmax gating.
/// </summary>
public sealed unsafe class GptOssForwardPass : IForwardPass
{
    private readonly GgufModel _model;
    private readonly GptOssHyperparams _hp;
    private readonly GptOssTensorSet _tensors;
    private readonly int _embedDim, _numHeads, _numHeadsKv, _headDim, _numLayer;
    private readonly int _kvDim;

    // Precomputed RoPE frequency tables
    private readonly float[] _invFreqGlobal;
    private readonly float[] _invFreqSwa;
    private readonly float[] _cosBuf;
    private readonly float[] _sinBuf;

    // Zero-allocation reusable workspace buffers
    private readonly float[] _curBuf;
    private readonly float[] _residualBuf;
    private readonly float[] _attnNormedBuf;
    private readonly float[] _ffnNormedBuf;
    private readonly float[] _qBuf;
    private readonly float[] _attnOutBuf;
    private readonly float[] _logitsBuf;
    private readonly float[] _moeLogitsBuf;
    private readonly float[] _gateBuf;
    private readonly float[] _upBuf;
    private readonly float[] _actBuf;
    private readonly float[] _downBuf;
    private readonly int[] _expertIndices;
    private readonly float[] _expertWeights;
    private readonly byte[] _q8ScratchEmbed;
    private readonly byte[] _q8ScratchFfn;
    private float[] _scoresBuf;

    // Flat contiguous per-layer KV cache
    private int _maxCapacity = 4096;
    private int _cacheLength;
    private float[][] _kCacheFlat;
    private float[][] _vCacheFlat;

    public GptOssForwardPass(GgufModel model, GptOssHyperparams hp)
    {
        _model = model;
        _hp = hp;
        _tensors = GptOssTensorSet.Load(model, hp);
        _embedDim = hp.EmbedDim;
        _numHeads = hp.NumHeads;
        _numHeadsKv = hp.NumHeadsKv;
        _headDim = hp.HeadDim;
        _numLayer = hp.NumLayer;
        _kvDim = _numHeadsKv * _headDim;

        int half = _headDim / 2;
        _invFreqGlobal = new float[half];
        _invFreqSwa = new float[half];
        _cosBuf = new float[half];
        _sinBuf = new float[half];

        for (int i = 0; i < half; i++)
        {
            _invFreqGlobal[i] = MathF.Pow(_hp.RopeFreqBase, -2f * i / _headDim);
            _invFreqSwa[i] = MathF.Pow(_hp.RopeFreqBaseSwa, -2f * i / _headDim);
        }

        _curBuf = new float[_embedDim];
        _residualBuf = new float[_embedDim];
        _attnNormedBuf = new float[_embedDim];
        _ffnNormedBuf = new float[_embedDim];
        _qBuf = new float[_numHeads * _headDim];
        _attnOutBuf = new float[_numHeads * _headDim];
        _logitsBuf = new float[hp.VocabSize];
        _moeLogitsBuf = new float[hp.NumExperts];

        int ffnDim = hp.ExpertFeedForwardLength;
        _gateBuf = new float[ffnDim];
        _upBuf = new float[ffnDim];
        _actBuf = new float[ffnDim];
        _downBuf = new float[_embedDim];

        _expertIndices = new int[hp.NumExpertsUsed];
        _expertWeights = new float[hp.NumExpertsUsed];
        _scoresBuf = new float[128];

        _q8ScratchEmbed = new byte[SimdKernels.Q8_0ScratchBytes(_embedDim)];
        _q8ScratchFfn = new byte[SimdKernels.Q8_0ScratchBytes(ffnDim)];

        _kCacheFlat = new float[_numLayer][];
        _vCacheFlat = new float[_numLayer][];
        for (int il = 0; il < _numLayer; il++)
        {
            _kCacheFlat[il] = new float[_maxCapacity * _kvDim];
            _vCacheFlat[il] = new float[_maxCapacity * _kvDim];
        }
    }

    public int VocabSize => _hp.VocabSize;
    public int MaxSeqLen => 1 << 20;

    public ReadOnlySpan<float> Forward(int token, int position)
    {
        EnsureCacheCapacity(position + 1);
        _cacheLength = Math.Max(_cacheLength, position + 1);

        EmbedTokenInto(token, _curBuf);

        fixed (float* pCur = _curBuf, pRes = _residualBuf, pAttnNorm = _attnNormedBuf, pFfnNorm = _ffnNormedBuf, pLogits = _logitsBuf)
        {
            float* curPtr = pCur;
            float* resPtr = pRes;
            float* attnNormPtr = pAttnNorm;
            float* ffnNormPtr = pFfnNorm;

            for (int il = 0; il < _numLayer; il++)
            {
                var layer = _tensors.Layers[il];
                Buffer.BlockCopy(_curBuf, 0, _residualBuf, 0, _embedDim * sizeof(float));

                float* attnNormWeight = (float*)layer.AttnNorm!.Value.DataPtr;
                SimdKernels.RmsNorm(attnNormPtr, curPtr, attnNormWeight, _embedDim, _hp.RmsNormEps);

                Attention(il, layer, _attnNormedBuf, position, _curBuf);
                for (int i = 0; i < _embedDim; i++) curPtr[i] += resPtr[i];

                Buffer.BlockCopy(_curBuf, 0, _residualBuf, 0, _embedDim * sizeof(float));

                float* ffnNormWeight = (float*)layer.AttnPostNorm!.Value.DataPtr;
                SimdKernels.RmsNorm(ffnNormPtr, curPtr, ffnNormWeight, _embedDim, _hp.RmsNormEps);

                MoeFfn(layer, _ffnNormedBuf, _curBuf);
                for (int i = 0; i < _embedDim; i++) curPtr[i] += resPtr[i];
            }

            float* outputNormWeight = (float*)_tensors.OutputNorm.DataPtr;
            SimdKernels.RmsNorm(attnNormPtr, curPtr, outputNormWeight, _embedDim, _hp.RmsNormEps);

            SimdKernels.MatVec(pLogits, _tensors.Output.DataPtr, attnNormPtr, VocabSize, _embedDim, _tensors.Output.DType);
        }

        return _logitsBuf;
    }

    private void EnsureCacheCapacity(int needed)
    {
        if (needed <= _maxCapacity) return;
        int newCap = Math.Max(needed, _maxCapacity * 2);
        for (int il = 0; il < _numLayer; il++)
        {
            Array.Resize(ref _kCacheFlat[il], newCap * _kvDim);
            Array.Resize(ref _vCacheFlat[il], newCap * _kvDim);
        }
        _maxCapacity = newCap;
    }

    private void Attention(int il, GptOssLayerTensors layer, float[] normedInput, int position, float[] output)
    {
        bool isSwa = _hp.IsSwaLayer(il);
        float[] invFreq = isSwa ? _invFreqSwa : _invFreqGlobal;

        int half = _headDim / 2;
        for (int i = 0; i < half; i++)
        {
            float theta = position * invFreq[i];
            _cosBuf[i] = MathF.Cos(theta);
            _sinBuf[i] = MathF.Sin(theta);
        }

        var kFlat = _kCacheFlat[il];
        var vFlat = _vCacheFlat[il];
        int kvOffset = position * _kvDim;

        fixed (float* inPtr = normedInput, qPtr = _qBuf, kPtr = &kFlat[kvOffset], vPtr = &vFlat[kvOffset])
        {
            SimdKernels.MatVec(qPtr, layer.Wq!.Value.DataPtr, inPtr, _numHeads * _headDim, _embedDim, layer.Wq.Value.DType);
            SimdKernels.MatVec(kPtr, layer.Wk!.Value.DataPtr, inPtr, _kvDim, _embedDim, layer.Wk.Value.DType);
            SimdKernels.MatVec(vPtr, layer.Wv!.Value.DataPtr, inPtr, _kvDim, _embedDim, layer.Wv.Value.DType);
        }
        AddBiasIfPresent(_qBuf, layer.WqB);
        AddBiasIfPresent(kFlat.AsSpan(kvOffset, _kvDim), layer.WkB);
        AddBiasIfPresent(vFlat.AsSpan(kvOffset, _kvDim), layer.WvB);

        for (int h = 0; h < _numHeads; h++)
        {
            ApplyRopeNeoxFast(_qBuf.AsSpan(h * _headDim, _headDim), _cosBuf, _sinBuf);
        }
        for (int h = 0; h < _numHeadsKv; h++)
        {
            ApplyRopeNeoxFast(kFlat.AsSpan(kvOffset + h * _headDim, _headDim), _cosBuf, _sinBuf);
        }

        int cacheLen = position + 1;
        int windowStart = isSwa && _hp.SlidingWindow > 0 ? Math.Max(0, cacheLen - _hp.SlidingWindow) : 0;
        int numKeys = cacheLen - windowStart;

        if (_scoresBuf.Length < numKeys)
        {
            _scoresBuf = new float[Math.Max(numKeys * 2, 128)];
        }
        var scoresSpan = _scoresBuf.AsSpan(0, numKeys);

        int groupSize = _numHeads / _numHeadsKv; // GQA: groupSize Q heads share one KV head.
        float scale = 1f / MathF.Sqrt(_headDim);
        float? sink = null;

        for (int h = 0; h < _numHeads; h++)
        {
            int kvHead = h / groupSize;
            int kvHeadOffset = kvHead * _headDim;
            if (layer.AttnSinks is { } sinksTensor)
            {
                sink = ((float*)sinksTensor.DataPtr)[h];
            }

            var qHead = new ReadOnlySpan<float>(_qBuf, h * _headDim, _headDim);
            for (int t = 0; t < numKeys; t++)
            {
                int keyTokenIdx = windowStart + t;
                var kt = new ReadOnlySpan<float>(kFlat, keyTokenIdx * _kvDim + kvHeadOffset, _headDim);
                scoresSpan[t] = TensorPrimitives.Dot(qHead, kt) * scale;
            }
            GptOssGraph.SoftmaxWithSink(scoresSpan, sink);

            var outHead = _attnOutBuf.AsSpan(h * _headDim, _headDim);
            outHead.Clear();
            for (int t = 0; t < numKeys; t++)
            {
                int keyTokenIdx = windowStart + t;
                var vt = new ReadOnlySpan<float>(vFlat, keyTokenIdx * _kvDim + kvHeadOffset, _headDim);
                float w = scoresSpan[t];
                if (w != 0f)
                {
                    TensorPrimitives.MultiplyAdd(vt, w, outHead, outHead);
                }
            }
        }

        fixed (float* inPtr = _attnOutBuf, outPtr = output)
        {
            SimdKernels.MatVec(outPtr, layer.Wo!.Value.DataPtr, inPtr, _embedDim, _numHeads * _headDim, layer.Wo.Value.DType);
        }
        AddBiasIfPresent(output, layer.WoB);
    }

    private void MoeFfn(GptOssLayerTensors layer, float[] normedInput, float[] output)
    {
        int numExperts = _hp.NumExperts;
        int topK = _hp.NumExpertsUsed;
        int ffnDim = _hp.ExpertFeedForwardLength;

        fixed (float* inPtr = normedInput, outPtr = _moeLogitsBuf, gatePtr = _gateBuf, upPtr = _upBuf, actPtr = _actBuf, downPtr = _downBuf)
        {
            SimdKernels.MatVec(outPtr, layer.FfnGateInp!.Value.DataPtr, inPtr, numExperts, _embedDim, layer.FfnGateInp.Value.DType);
            AddBiasIfPresent(_moeLogitsBuf, layer.FfnGateInpB);

            GptOssGraph.SelectThenSoftmaxGate(_moeLogitsBuf.AsSpan(0, numExperts), topK, _expertIndices.AsSpan(0, topK), _expertWeights.AsSpan(0, topK));

            Array.Clear(output, 0, _embedDim);

            var gateExps = layer.FfnGateExps!.Value;
            var upExps = layer.FfnUpExps!.Value;
            var downExps = layer.FfnDownExps!.Value;

            long gateBytesPerRow = ((long)_embedDim / DTypeInfo.BlockSize(gateExps.DType)) * DTypeInfo.BytesPerBlock(gateExps.DType);
            long downBytesPerRow = ((long)ffnDim / DTypeInfo.BlockSize(downExps.DType)) * DTypeInfo.BytesPerBlock(downExps.DType);

            for (int k = 0; k < topK; k++)
            {
                int e = _expertIndices[k];
                float w = _expertWeights[k];

                byte* gateExpPtr = gateExps.DataPtr + (long)e * ffnDim * gateBytesPerRow;
                byte* upExpPtr = upExps.DataPtr + (long)e * ffnDim * gateBytesPerRow;
                byte* downExpPtr = downExps.DataPtr + (long)e * _embedDim * downBytesPerRow;

                SimdKernels.MatVecDual(gatePtr, gateExpPtr, upPtr, upExpPtr, inPtr, ffnDim, _embedDim, gateExps.DType, upExps.DType);
                AddPerExpertBiasIfPresent(_gateBuf, layer.FfnGateExpsB, e, ffnDim);
                AddPerExpertBiasIfPresent(_upBuf, layer.FfnUpExpsB, e, ffnDim);

                GptOssGraph.SwigluOai(_gateBuf.AsSpan(0, ffnDim), _upBuf.AsSpan(0, ffnDim), _actBuf.AsSpan(0, ffnDim));

                SimdKernels.MatVec(downPtr, downExpPtr, actPtr, _embedDim, ffnDim, downExps.DType);
                AddPerExpertBiasIfPresent(_downBuf, layer.FfnDownExpsB, e, _embedDim);

                var outSpan = output.AsSpan(0, _embedDim);
                var downSpan = new ReadOnlySpan<float>(_downBuf, 0, _embedDim);
                TensorPrimitives.MultiplyAdd(downSpan, w, outSpan, outSpan);
            }
        }
    }

    private static void AddPerExpertBiasIfPresent(float[] data, DeepSeek4TensorRef? bias, int expert, int dim)
    {
        if (bias is not { } b) return;
        float* biasPtr = (float*)b.DataPtr + (long)expert * dim;
        for (int i = 0; i < dim; i++) data[i] += biasPtr[i];
    }

    private static void AddBiasIfPresent(Span<float> data, DeepSeek4TensorRef? bias)
    {
        if (bias is not { } b) return;
        float* biasPtr = (float*)b.DataPtr;
        for (int i = 0; i < data.Length; i++) data[i] += biasPtr[i];
    }

    private static void ApplyRopeNeoxFast(Span<float> x, ReadOnlySpan<float> cos, ReadOnlySpan<float> sin)
    {
        int half = cos.Length;
        for (int i = 0; i < half; i++)
        {
            float c = cos[i], s = sin[i];
            float x0 = x[i], x1 = x[i + half];
            x[i] = x0 * c - x1 * s;
            x[i + half] = x0 * s + x1 * c;
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
        for (int i = 0; i < tokens.Count; i++) last = Forward(tokens[i], startPos + i);
        return last;
    }

    public void TruncateTo(int length)
    {
        if (length == 0) { ResetCache(); return; }
        if (length > _cacheLength)
        {
            throw new NotSupportedException("GptOssForwardPass (alpha): cannot extend cache length via TruncateTo.");
        }
        _cacheLength = length;
    }

    public void ResetCache()
    {
        _cacheLength = 0;
    }

    public void Dispose() { }
}
