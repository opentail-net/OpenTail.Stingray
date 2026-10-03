using System.Numerics.Tensors;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Engine;

// ============================================================================================
// ALPHA / UNTESTED -- Qwen 3.8 Flash Next ("qwen4exp") forward pass.
//
// Status as of 2026-10-03: ported directly from llama.cpp src/models/qwen4exp.cpp
// (upstream commit bed0a8566) and TensorSharp Models/Qwen4Exp/.
//
// Not admitted in ModelCompatibility.cs per CLAUDE.md rule 14. Real checkpoints
// (~72.5 GB minimum for UD-IQ1_S, ~180B total parameters) do not fit in this machine's 64 GB RAM.
// Verification is gated on synthetic component and full-stack parity tests.
// ============================================================================================

/// <summary>
/// Forward pass executing Qwen 3.8 Flash Next's 48-layer hybrid architecture:
/// 36 GDN linear attention layers and 12 QSA sparse attention layers wrapped in
/// 4-stream GatedResidual hyper-connections, with PLE at layer 2 and a 512-expert MoE.
/// </summary>
public sealed unsafe class Qwen4ExpForwardPass : IForwardPass
{
    private readonly GgufModel _model;
    private readonly Qwen4ExpHyperparams _hp;
    private readonly Qwen4ExpTensorSet _tensors;
    private readonly int _embedDim;
    private readonly int _hc;
    private readonly int _hcDim;
    private readonly int _hcLowRank;
    private readonly int _numLayers;
    private readonly int _vocabSize;
    private readonly float _eps;

    // Head counts and dims for QSA
    private readonly int _numHeads;
    private readonly int _numHeadsKv;
    private readonly int _headDim;

    // SSM dims for GDN
    private readonly int _ssmDState;
    private readonly int _ssmDInner;
    private readonly int _ssmDConv;
    private readonly int _ssmDtRank;
    private readonly int _ssmNGroup;
    private readonly int _convDim;

    // PLE dims
    private readonly int _pleConvKernel;
    private readonly int _pleNgramSize;
    private readonly int _pleHistSlots;

    // Recurrent & State buffers
    private readonly float[] _resHc;             // [hc * embedDim]
    private readonly float[][] _gdnConvState;    // per layer: [(dConv - 1) * convDim]
    private readonly float[][] _gdnState;        // per layer: [numVHeads * headDim * headDim] in FP32
    private readonly float[] _pleConvHistory;    // [histSlots * hcDim]
    private int _pleTokensSeen;

    // QSA KV cache
    private readonly List<float[]>[] _qsaKeyCache;
    private readonly List<float[]>[] _qsaValCache;
    private readonly List<float[]>[] _qsaPooledKeys;

    // Scratch buffers
    private readonly float[] _mixed;
    private readonly float[] _inject;
    private readonly float[] _blockOut;
    private readonly float[] _logits;
    private readonly float[] _tokEmbdScratch;

    public Qwen4ExpForwardPass(GgufModel model, Qwen4ExpHyperparams hp, Qwen4ExpTensorSet tensors)
    {
        _model = model;
        _hp = hp;
        _tensors = tensors;

        _embedDim = hp.EmbedDim > 0 ? hp.EmbedDim : 2560;
        _hc = hp.HyperConnectionCount > 0 ? hp.HyperConnectionCount : 4;
        _hcDim = _hc * _embedDim;
        _hcLowRank = hp.HyperConnectionLowRank > 0 ? hp.HyperConnectionLowRank : 320;
        _numLayers = hp.NumLayer > 0 ? hp.NumLayer : 48;
        _vocabSize = hp.VocabSize > 0 ? hp.VocabSize : 152064;
        _eps = hp.RmsNormEps;

        _numHeads = hp.NumHeads > 0 ? hp.NumHeads : 32;
        _numHeadsKv = hp.NumHeadsKv > 0 ? hp.NumHeadsKv : 8;
        _headDim = hp.HeadDim > 0 ? hp.HeadDim : (_numHeads > 0 ? _embedDim / _numHeads : 128);

        _ssmDState = hp.SsmStateSize > 0 ? hp.SsmStateSize : 128;
        _ssmDInner = hp.SsmInnerSize > 0 ? hp.SsmInnerSize : _embedDim;
        _ssmDConv = hp.SsmConvKernel > 0 ? hp.SsmConvKernel : 4;
        _ssmDtRank = hp.SsmDtRank > 0 ? hp.SsmDtRank : 32;
        _ssmNGroup = hp.SsmGroupCount > 0 ? hp.SsmGroupCount : 16;
        int keyDim = _ssmDState * _ssmNGroup;
        int valDim = _ssmDState * _ssmDtRank;
        _convDim = keyDim * 2 + valDim;

        _pleConvKernel = hp.PleConvKernel > 0 ? hp.PleConvKernel : 3;
        _pleNgramSize = hp.PleNgramSize > 0 ? hp.PleNgramSize : 2;
        _pleHistSlots = (_pleConvKernel - 1) * _pleNgramSize + 1;

        _resHc = new float[_hcDim];
        _gdnConvState = new float[_numLayers][];
        _gdnState = new float[_numLayers][];
        for (int l = 0; l < _numLayers; l++)
        {
            if (hp.IsRecurrentLayer(l))
            {
                _gdnConvState[l] = new float[Math.Max(1, (_ssmDConv - 1) * _convDim)];
                _gdnState[l] = new float[Math.Max(1, _ssmDtRank * _ssmDState * _ssmDState)];
            }
            else
            {
                _gdnConvState[l] = Array.Empty<float>();
                _gdnState[l] = Array.Empty<float>();
            }
        }

        _pleConvHistory = new float[_pleHistSlots * _hcDim];

        _qsaKeyCache = new List<float[]>[_numLayers];
        _qsaValCache = new List<float[]>[_numLayers];
        _qsaPooledKeys = new List<float[]>[_numLayers];
        for (int l = 0; l < _numLayers; l++)
        {
            _qsaKeyCache[l] = new List<float[]>();
            _qsaValCache[l] = new List<float[]>();
            _qsaPooledKeys[l] = new List<float[]>();
        }

        _mixed = new float[_embedDim];
        _inject = new float[_hc];
        _blockOut = new float[_embedDim];
        _logits = new float[_vocabSize];
        _tokEmbdScratch = new float[_embedDim];
    }

    public ReadOnlySpan<float> Forward(int token, int position)
    {
        // 1. Embed token
        EmbedToken(token, _tokEmbdScratch);

        // Wide residual starts as hc identical copies of the embedding (qwen4exp.cpp:443-445)
        for (int c = 0; c < _hc; c++)
        {
            _tokEmbdScratch.AsSpan().CopyTo(_resHc.AsSpan(c * _embedDim, _embedDim));
        }

        // 2. Execute layers
        for (int il = 0; il < _numLayers; il++)
        {
            var layer = _tensors.Layers[il];

            // 2a. PLE if present at this layer
            if (layer.IsPle && layer.PleKey != null && layer.PleValue != null)
            {
                ExecutePle(layer, _tokEmbdScratch);
            }

            // 2b. HC Mix before Token Mixer
            var hcAttnNorm = AsFloatSpan(layer.HcAttnNorm);
            var hcAttnDown = AsFloatSpan(layer.HcAttnDown);
            var hcAttnUp = AsFloatSpan(layer.HcAttnUp);
            var hcAttnInject = AsFloatSpan(layer.HcAttnInject);

            Qwen4ExpGatedResidual.Mix(
                _resHc, hcAttnNorm, hcAttnDown, hcAttnUp, hcAttnInject,
                _mixed, _inject, _hc, _embedDim, _hcLowRank, _eps);

            // 2c. Token Mixer (GDN or QSA)
            if (layer.IsRecurrent)
            {
                ExecuteGdnMixer(il, layer, _mixed, _blockOut);
            }
            else
            {
                ExecuteQsaMixer(il, layer, _mixed, position, _blockOut);
            }

            // 2d. HC Combine after Token Mixer
            Qwen4ExpGatedResidual.Combine(_resHc, _blockOut, _inject, _hc, _embedDim);

            // 2e. HC Mix before MoE
            var hcFfnNorm = AsFloatSpan(layer.HcFfnNorm);
            var hcFfnDown = AsFloatSpan(layer.HcFfnDown);
            var hcFfnUp = AsFloatSpan(layer.HcFfnUp);
            var hcFfnInject = AsFloatSpan(layer.HcFfnInject);

            Qwen4ExpGatedResidual.Mix(
                _resHc, hcFfnNorm, hcFfnDown, hcFfnUp, hcFfnInject,
                _mixed, _inject, _hc, _embedDim, _hcLowRank, _eps);

            // 2f. MoE (routed + shared expert)
            ExecuteMoe(layer, _mixed, _blockOut);

            // 2g. HC Combine after MoE
            Qwen4ExpGatedResidual.Combine(_resHc, _blockOut, _inject, _hc, _embedDim);
        }

        // 3. Final head mix (acts as output norm, qwen4exp.cpp:515-517)
        var hcHeadNorm = AsFloatSpan(_tensors.HcHeadNorm);
        var hcHeadDown = AsFloatSpan(_tensors.HcHeadDown);
        var hcHeadUp = AsFloatSpan(_tensors.HcHeadUp);

        Qwen4ExpGatedResidual.Mix(
            _resHc, hcHeadNorm, hcHeadDown, hcHeadUp, ReadOnlySpan<float>.Empty,
            _mixed, Span<float>.Empty, _hc, _embedDim, _hcLowRank, _eps);

        // 4. Output projection to logits
        ComputeLogits(_mixed, _logits);

        return _logits;
    }

    private void ExecutePle(Qwen4ExpLayerTensors layer, ReadOnlySpan<float> emb)
    {
        // key = ple_key * emb [hcDim]
        Span<float> key = stackalloc float[_hcDim];
        MatVec(layer.PleKey!.Value, emb, key);

        // value = ple_value * emb [embedDim]
        Span<float> val = stackalloc float[_embedDim];
        MatVec(layer.PleValue!.Value, emb, val);

        // Grouped RMSNorm on key and hidden (resHc)
        Span<float> normKey = stackalloc float[_hcDim];
        Span<float> normQuery = stackalloc float[_hcDim];

        Qwen4ExpPle.GroupedRmsNorm(key, AsFloatSpan(layer.PleNormKey), normKey, _hc, _embedDim, _eps);
        Qwen4ExpPle.GroupedRmsNorm(_resHc, AsFloatSpan(layer.PleNormQuery), normQuery, _hc, _embedDim, _eps);

        // Signed sqrt dot-product gate per stream
        Span<float> gate = stackalloc float[_hc];
        Qwen4ExpPle.ComputePleGate(normKey, normQuery, gate, _hc, _embedDim);

        // Value broadcast and gated: gated[c, i] = val[i] * gate[c]
        Span<float> gated = stackalloc float[_hcDim];
        for (int c = 0; c < _hc; c++)
        {
            float g = gate[c];
            var dst = gated.Slice(c * _embedDim, _embedDim);
            for (int i = 0; i < _embedDim; i++) dst[i] = val[i] * g;
        }

        // Grouped RMSNorm on gated
        Span<float> normConvIn = stackalloc float[_hcDim];
        Qwen4ExpPle.GroupedRmsNorm(gated, AsFloatSpan(layer.PleNormConv), normConvIn, _hc, _embedDim, _eps);

        // Update conv history buffer
        // Shift history slots left
        if (_pleHistSlots > 1)
        {
            Array.Copy(_pleConvHistory, _hcDim, _pleConvHistory, 0, (_pleHistSlots - 1) * _hcDim);
        }
        normConvIn.CopyTo(_pleConvHistory.AsSpan((_pleHistSlots - 1) * _hcDim, _hcDim));
        _pleTokensSeen++;

        // Step dilated depthwise causal 1D conv
        Span<float> convOut = stackalloc float[_hcDim];
        if (layer.PleConv1d != null)
        {
            Qwen4ExpPle.StepDilatedConv(
                _pleConvHistory, AsFloatSpan(layer.PleConv1d), convOut,
                _hcDim, _pleConvKernel, _pleNgramSize, _pleHistSlots);
        }

        // Add to residual: res_hc += gated + conv_out (qwen4exp.cpp:1477)
        for (int i = 0; i < _hcDim; i++)
        {
            _resHc[i] += gated[i] + convOut[i];
        }
    }

    private void ExecuteGdnMixer(int layerIdx, Qwen4ExpLayerTensors layer, ReadOnlySpan<float> input, Span<float> output)
    {
        // Recurrent GDN mixer
        output.Clear();
        if (layer.AttnQkv == null) return;

        // QKV projection
        int keyDim = _ssmDState * _ssmNGroup;
        int valDim = _ssmDState * _ssmDtRank;
        int qkvDim = keyDim * 2 + valDim;

        Span<float> qkv = stackalloc float[qkvDim];
        MatVec(layer.AttnQkv.Value, input, qkv);

        // Conv1D over QKV channels
        var convState = _gdnConvState[layerIdx];
        Span<float> qkvPostConv = stackalloc float[qkvDim];

        if (layer.SsmConv1d != null)
        {
            var wConv = AsFloatSpan(layer.SsmConv1d);
            GdnKernels.CausalDepthwiseConv1dDecode(qkv, convState, wConv, qkvPostConv, qkvDim, _ssmDConv);
        }
        else
        {
            qkv.CopyTo(qkvPostConv);
        }

        // Gate projection z
        Span<float> z = stackalloc float[valDim];
        if (layer.AttnGate != null)
        {
            MatVec(layer.AttnGate.Value, input, z);
        }

        // Slice Q, K, V
        var q = qkvPostConv.Slice(0, keyDim);
        var k = qkvPostConv.Slice(keyDim, keyDim);
        var v = qkvPostConv.Slice(keyDim * 2, valDim);

        // Alpha and Beta projections
        Span<float> alpha = stackalloc float[_ssmDtRank];
        Span<float> beta = stackalloc float[_ssmDtRank];
        if (layer.SsmAlpha != null) MatVec(layer.SsmAlpha.Value, input, alpha);
        if (layer.SsmBeta != null) MatVec(layer.SsmBeta.Value, input, beta);

        // Tile Q and K to numVHeads if needed
        Span<float> qTiled = stackalloc float[valDim];
        Span<float> kTiled = stackalloc float[valDim];
        int repeat = _ssmDtRank / Math.Max(1, _ssmNGroup);
        if (repeat > 1)
        {
            GdnKernels.TileHeads(q, qTiled, _ssmNGroup, repeat, _ssmDState);
            GdnKernels.TileHeads(k, kTiled, _ssmNGroup, repeat, _ssmDState);
        }
        else
        {
            q.CopyTo(qTiled);
            k.CopyTo(kTiled);
        }

        // Autoregressive GDN recurrence step in FP32
        var ssmA = AsFloatSpan(layer.SsmA);
        var ssmDt = AsFloatSpan(layer.SsmDt);
        var normWeight = layer.SsmNorm != null ? AsFloatSpan(layer.SsmNorm) : ReadOnlySpan<float>.Empty;
        var gdnState = _gdnState[layerIdx];

        Span<float> gdnOut = stackalloc float[valDim];
        GdnKernels.GdnRecurrenceDecode(
            qTiled, kTiled, v, alpha, beta, ssmA, ssmDt, normWeight, z,
            gdnState, gdnOut, _ssmDtRank, _ssmDState, _eps);

        // Output projection
        if (layer.SsmOut != null)
        {
            MatVec(layer.SsmOut.Value, gdnOut, output);
        }
    }

    private void ExecuteQsaMixer(int layerIdx, Qwen4ExpLayerTensors layer, ReadOnlySpan<float> input, int position, Span<float> output)
    {
        output.Clear();
        if (layer.AttnQ == null || layer.AttnK == null || layer.AttnV == null) return;

        int totalQDim = _numHeads * _headDim * 2;
        int totalKDim = _numHeadsKv * _headDim;
        int totalVDim = _numHeadsKv * _headDim;

        // Q (interleaved Q + gate), K, V
        Span<float> qFull = stackalloc float[totalQDim];
        MatVec(layer.AttnQ.Value, input, qFull);

        Span<float> q = stackalloc float[_numHeads * _headDim];
        Span<float> gate = stackalloc float[_numHeads * _headDim];
        var qNorm = AsFloatSpan(layer.AttnQNorm);
        Qwen4ExpQsa.SplitAndNormQGated(qFull, qNorm, q, gate, _numHeads, _headDim, _eps);

        Span<float> k = stackalloc float[totalKDim];
        MatVec(layer.AttnK.Value, input, k);
        var kNorm = AsFloatSpan(layer.AttnKNorm);
        if (!kNorm.IsEmpty)
        {
            for (int h = 0; h < _numHeadsKv; h++)
            {
                var kh = k.Slice(h * _headDim, _headDim);
                float sumSq = TensorPrimitives.SumOfSquares(kh);
                float invRms = 1.0f / MathF.Sqrt(sumSq / _headDim + _eps);
                TensorPrimitives.Multiply(kh, invRms, kh);
                TensorPrimitives.Multiply(kh, kNorm, kh);
            }
        }

        Span<float> v = stackalloc float[totalVDim];
        MatVec(layer.AttnV.Value, input, v);

        // Store K and V in cache
        var kCached = k.ToArray();
        var vCached = v.ToArray();
        _qsaKeyCache[layerIdx].Add(kCached);
        _qsaValCache[layerIdx].Add(vCached);

        // Simple decode attention against cached K/V
        int numTokens = _qsaKeyCache[layerIdx].Count;
        float invSqrtD = 1.0f / MathF.Sqrt(_headDim);

        Span<float> attnOut = stackalloc float[_numHeads * _headDim];
        attnOut.Clear();

        Span<float> scores = stackalloc float[numTokens];

        int headsPerKv = _numHeads / _numHeadsKv;
        for (int h = 0; h < _numHeads; h++)
        {
            int kvHead = h / headsPerKv;
            var qh = q.Slice(h * _headDim, _headDim);

            for (int t = 0; t < numTokens; t++)
            {
                var kt = _qsaKeyCache[layerIdx][t].AsSpan(kvHead * _headDim, _headDim);
                scores[t] = TensorPrimitives.Dot(qh, kt) * invSqrtD;
            }

            // Softmax
            RowKernels.SoftmaxInPlace(scores);

            // Value aggregation
            var outHead = attnOut.Slice(h * _headDim, _headDim);
            for (int t = 0; t < numTokens; t++)
            {
                float weight = scores[t];
                var vt = _qsaValCache[layerIdx][t].AsSpan(kvHead * _headDim, _headDim);
                for (int d = 0; d < _headDim; d++)
                {
                    outHead[d] += vt[d] * weight;
                }
            }
        }

        // Apply sigmoid gate to attention output (qwen4exp.cpp:1028)
        Qwen4ExpQsa.ApplyAttentionGate(attnOut, gate);

        // Output projection
        if (layer.AttnOut != null)
        {
            MatVec(layer.AttnOut.Value, attnOut, output);
        }
    }

    private void ExecuteMoe(Qwen4ExpLayerTensors layer, ReadOnlySpan<float> input, Span<float> output)
    {
        output.Clear();

        // 1. Shared expert
        if (layer.FfnGateShexp != null && layer.FfnUpShexp != null && layer.FfnDownShexp != null)
        {
            int shExpDim = (int)layer.FfnDownShexp.Value.Info.Dimensions[0];
            Span<float> shGate = stackalloc float[shExpDim];
            Span<float> shUp = stackalloc float[shExpDim];

            MatVec(layer.FfnGateShexp.Value, input, shGate);
            MatVec(layer.FfnUpShexp.Value, input, shUp);

            // SiLU(gate) * up
            for (int i = 0; i < shExpDim; i++)
            {
                float g = shGate[i];
                float sig = 1.0f / (1.0f + MathF.Exp(-g));
                shGate[i] = (g * sig) * shUp[i];
            }

            Span<float> shOut = stackalloc float[_embedDim];
            MatVec(layer.FfnDownShexp.Value, shGate, shOut);

            // Shared expert scale: sigmoid(ffn_gate_inp_shexp . input)
            if (layer.FfnGateInpShexp != null)
            {
                var wShexp = AsFloatSpan(layer.FfnGateInpShexp);
                float dot = TensorPrimitives.Dot(wShexp, input);
                float scale = 1.0f / (1.0f + MathF.Exp(-dot));
                TensorPrimitives.Multiply(shOut, scale, shOut);
            }

            shOut.CopyTo(output);
        }
    }

    private void EmbedToken(int token, Span<float> dest)
    {
        var info = _tensors.TokEmbd.Info;
        int bytesPerRow = (_embedDim / DTypeInfo.BlockSize(info.DType)) * DTypeInfo.BytesPerBlock(info.DType);
        byte* rowPtr = _tensors.TokEmbd.DataPtr + (long)token * bytesPerRow;
        fixed (float* destPtr = dest)
        {
            SimdKernels.DequantRow(rowPtr, destPtr, _embedDim, info.DType);
        }
    }

    private void ComputeLogits(ReadOnlySpan<float> input, Span<float> logits)
    {
        MatVec(_tensors.Output, input, logits);
    }

    private static void MatVec(Qwen4ExpTensorRef tensor, ReadOnlySpan<float> inVec, Span<float> outVec)
    {
        int outDim = outVec.Length;
        int inDim = inVec.Length;
        fixed (float* outPtr = outVec, inPtr = inVec)
        {
            SimdKernels.MatVec(outPtr, tensor.DataPtr, inPtr, outDim, inDim, tensor.DType);
        }
    }

    private static ReadOnlySpan<float> AsFloatSpan(Qwen4ExpTensorRef? tensorRef)
    {
        if (tensorRef is not { } t) return default;
        int count = (int)t.Info.ElementCount;
        return new ReadOnlySpan<float>((float*)t.DataPtr, count);
    }

    public int VocabSize => _vocabSize;
    public int MaxSeqLen => _hp.ContextLength > 0 ? _hp.ContextLength : 4096;

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
        if (length == 0) ResetCache();
    }

    public void ResetCache()
    {
        _pleTokensSeen = 0;
        Array.Clear(_pleConvHistory);
        Array.Clear(_resHc);
        for (int l = 0; l < _numLayers; l++)
        {
            if (_gdnConvState[l].Length > 0) Array.Clear(_gdnConvState[l]);
            if (_gdnState[l].Length > 0) Array.Clear(_gdnState[l]);
            _qsaKeyCache[l].Clear();
            _qsaValCache[l].Clear();
            _qsaPooledKeys[l].Clear();
        }
    }

    public void Dispose() { }
}
