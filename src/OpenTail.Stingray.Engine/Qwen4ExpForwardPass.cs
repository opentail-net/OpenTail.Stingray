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
    private readonly int _indexerHeadCount;
    private readonly int _indexerKeyLength;
    private readonly int _indexerTopK;
    private readonly int _indexerKPool;

    // SSM dims for GDN
    private readonly int _ssmDState;
    private readonly int _ssmDInner;
    private readonly int _ssmDConv;
    private readonly int _ssmDtRank;
    private readonly int _ssmNGroup;
    private readonly int _convDim;

    // RoPE and PLE dims & state
    private readonly int _ropeDim;
    private readonly float _ropeTheta;
    private readonly int _pleHeadDim;
    private readonly int _pleConvKernel;
    private readonly int _pleNgramSize;
    private readonly int _pleHistSlots;
    private readonly Qwen4ExpPleHasher? _pleHasher;
    private readonly long[]? _pleHeadRows;

    // Recurrent & State buffers
    private readonly float[] _resHc;             // [hc * embedDim]
    private readonly float[][] _gdnConvState;    // per layer: [(dConv - 1) * convDim]
    private readonly float[][] _gdnState;        // per layer: [numVHeads * headDim * headDim] in FP32
    private readonly float[] _pleConvHistory;    // [histSlots * hcDim]
    private int _pleTokensSeen;

    // QSA KV cache and Indexer cache
    private readonly List<float[]>[] _qsaKeyCache;
    private readonly List<float[]>[] _qsaValCache;
    private readonly List<float[]>[] _qsaRawIndexKCache;
    private readonly List<float[]>[] _qsaPooledKeys;

    // Scratch buffers
    private readonly float[] _mixed;
    private readonly float[] _inject;
    private readonly float[] _blockOut;
    private readonly float[] _logits;
    private readonly float[] _tokEmbdScratch;
    private readonly float[] _pleEmbScratch;
    private readonly float[] _qFullScratch;
    private readonly float[] _qScratch;
    private readonly float[] _qGateScratch;
    private readonly float[] _kScratch;
    private readonly float[] _vScratch;
    private readonly float[] _attnOutScratch;
    private readonly float[] _routerLogitsScratch;
    private readonly int[] _moeTopIdxScratch;
    private readonly float[] _moeTopWScratch;
    private readonly float[] _moeGateScratch;
    private readonly float[] _moeUpScratch;
    private readonly float[] _moeDownScratch;
    private readonly float[] _shGateUpScratch;
    private readonly float[] _shOutScratch;

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

        _indexerHeadCount = hp.IndexerHeadCount > 0 ? hp.IndexerHeadCount : 4;
        _indexerKeyLength = hp.IndexerKeyLength > 0 ? hp.IndexerKeyLength : _headDim;
        _indexerTopK = hp.IndexerTopK;
        _indexerKPool = hp.IndexerKPool > 0 ? hp.IndexerKPool : 4;

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

        _ropeTheta = hp.RopeTheta > 0 ? hp.RopeTheta : 10000000.0f;
        _ropeDim = hp.RopeDim > 0 ? hp.RopeDim : _headDim;

        if (hp.PleLayers != null && hp.PleLayers.Count > 0)
        {
            _pleHasher = new Qwen4ExpPleHasher(hp);
            _pleHeadRows = new long[_pleHasher.NumHeads];
        }

        int numPleHeads = _pleHasher != null ? _pleHasher.NumHeads : 16;
        _pleHeadDim = hp.PleEmbeddingLengthPerLayer > 0
            ? hp.PleEmbeddingLengthPerLayer
            : (numPleHeads > 0 ? _embedDim / numPleHeads : 160);

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
        _qsaRawIndexKCache = new List<float[]>[_numLayers];
        _qsaPooledKeys = new List<float[]>[_numLayers];
        for (int l = 0; l < _numLayers; l++)
        {
            _qsaKeyCache[l] = new List<float[]>();
            _qsaValCache[l] = new List<float[]>();
            _qsaRawIndexKCache[l] = new List<float[]>();
            _qsaPooledKeys[l] = new List<float[]>();
        }

        _mixed = new float[_embedDim];
        _inject = new float[_hc];
        _blockOut = new float[_embedDim];
        _logits = new float[_vocabSize];
        _tokEmbdScratch = new float[_embedDim];
        _pleEmbScratch = new float[_embedDim];

        int totalQDim = _numHeads * _headDim * 2;
        int totalKDim = _numHeadsKv * _headDim;
        int totalVDim = _numHeadsKv * _headDim;
        _qFullScratch = new float[totalQDim];
        _qScratch = new float[_numHeads * _headDim];
        _qGateScratch = new float[_numHeads * _headDim];
        _kScratch = new float[totalKDim];
        _vScratch = new float[totalVDim];
        _attnOutScratch = new float[_numHeads * _headDim];

        int numExperts = hp.ExpertCount > 0 ? hp.ExpertCount : 512;
        int topK = Math.Min(hp.ExpertUsedCount > 0 ? hp.ExpertUsedCount : 10, numExperts);
        _routerLogitsScratch = new float[numExperts];
        _moeTopIdxScratch = new int[topK];
        _moeTopWScratch = new float[topK];

        int maxInterDim = Math.Max(2048, _embedDim * 2);
        _moeGateScratch = new float[maxInterDim];
        _moeUpScratch = new float[maxInterDim];
        _moeDownScratch = new float[_embedDim];
        _shGateUpScratch = new float[maxInterDim];
        _shOutScratch = new float[_embedDim];
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
                ExecutePle(layer, _tokEmbdScratch, token);
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

    private void ExecutePle(Qwen4ExpLayerTensors layer, ReadOnlySpan<float> tokEmb, int token)
    {
        // 1. Determine PLE embedding input:
        // If the PLE n-gram table (PerLayerTokEmbd) is present and hasher is initialized,
        // gather the embedding row for each head and concatenate them into pleEmb.
        // Otherwise fall back to tokEmb.
        Span<float> pleEmb = _pleEmbScratch;
        bool gatheredFromTable = false;

        if (_pleHasher != null && _tensors.PerLayerTokEmbd != null && _pleHeadRows != null)
        {
            _pleHasher.PushToken(token);
            _pleHasher.ComputeRowIndices(_pleHeadRows);

            var info = _tensors.PerLayerTokEmbd.Value.Info;
            long totalRows = info.Dimensions.Length > 1 ? info.Dimensions[1] : 1;
            int pleHeadDim = _pleHeadDim;
            int bytesPerRow = (pleHeadDim / DTypeInfo.BlockSize(info.DType)) * DTypeInfo.BytesPerBlock(info.DType);

            for (int h = 0; h < _pleHeadRows.Length; h++)
            {
                long row = _pleHeadRows[h];
                if (row < 0 || row >= totalRows)
                {
                    throw new InvalidOperationException(
                        $"PLE head {h} computed row index {row} out of bounds [0, {totalRows}) in PerLayerTokEmbd.");
                }

                byte* rowPtr = _tensors.PerLayerTokEmbd.Value.DataPtr + row * bytesPerRow;
                var headSlice = pleEmb.Slice(h * pleHeadDim, pleHeadDim);
                fixed (float* pHead = headSlice)
                {
                    SimdKernels.DequantRow(rowPtr, pHead, pleHeadDim, info.DType);
                }
            }
            gatheredFromTable = true;
        }

        if (!gatheredFromTable)
        {
            tokEmb.CopyTo(pleEmb);
        }

        // key = ple_key * emb [hcDim]
        Span<float> key = stackalloc float[_hcDim];
        MatVec(layer.PleKey!.Value, pleEmb, key);

        // value = ple_value * emb [embedDim]
        Span<float> val = stackalloc float[_embedDim];
        MatVec(layer.PleValue!.Value, pleEmb, val);

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
        Span<float> qFull = _qFullScratch.AsSpan(0, totalQDim);
        MatVec(layer.AttnQ.Value, input, qFull);

        Span<float> q = _qScratch;
        Span<float> gate = _qGateScratch;
        var qNorm = AsFloatSpan(layer.AttnQNorm);
        Qwen4ExpQsa.SplitAndNormQGated(qFull, qNorm, q, gate, _numHeads, _headDim, _eps);

        Span<float> k = _kScratch.AsSpan(0, totalKDim);
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

        // Apply 4-section IMRoPE to Q and K at current token position
        Qwen4ExpRope.ApplyImRope(q, position, _numHeads, _headDim, _hp.RopeDimensionSections, _ropeDim, _ropeTheta);
        Qwen4ExpRope.ApplyImRope(k, position, _numHeadsKv, _headDim, _hp.RopeDimensionSections, _ropeDim, _ropeTheta);

        Span<float> v = _vScratch.AsSpan(0, totalVDim);
        MatVec(layer.AttnV.Value, input, v);

        // Store K and V in cache
        var kCached = k.ToArray();
        var vCached = v.ToArray();
        _qsaKeyCache[layerIdx].Add(kCached);
        _qsaValCache[layerIdx].Add(vCached);

        // Maintain QSA raw indexer-K cache and form K-pools
        if (layer.IndexKProj != null)
        {
            Span<float> rawIdxK = stackalloc float[_indexerKeyLength];
            MatVec(layer.IndexKProj.Value, input, rawIdxK);
            _qsaRawIndexKCache[layerIdx].Add(rawIdxK.ToArray());

            // Check if a full pool of kpool tokens has just been completed
            int rawCount = _qsaRawIndexKCache[layerIdx].Count;
            if (rawCount % _indexerKPool == 0)
            {
                int poolIdx = (rawCount / _indexerKPool) - 1;
                Span<float> blockKeys = stackalloc float[_indexerKPool * _indexerKeyLength];
                for (int p = 0; p < _indexerKPool; p++)
                {
                    int t = poolIdx * _indexerKPool + p;
                    _qsaRawIndexKCache[layerIdx][t].CopyTo(blockKeys.Slice(p * _indexerKeyLength, _indexerKeyLength));
                }
                Span<float> pooledKey = stackalloc float[_indexerKeyLength];
                var indexKNorm = AsFloatSpan(layer.IndexKNorm);
                Qwen4ExpQsa.PoolIndexerKeys(blockKeys, indexKNorm, pooledKey, _indexerKPool, _indexerKeyLength, _eps);

                // Apply IMRoPE to pooled key at first token position of this block
                int blockStartPos = poolIdx * _indexerKPool;
                int indexerRopeDim = Math.Min(_ropeDim, _indexerKeyLength);
                Qwen4ExpRope.ApplyImRope(pooledKey, blockStartPos, 1, _indexerKeyLength, _hp.RopeDimensionSections, indexerRopeDim, _ropeTheta);
                _qsaPooledKeys[layerIdx].Add(pooledKey.ToArray());
            }
        }

        int numTokens = _qsaKeyCache[layerIdx].Count;
        float invSqrtD = 1.0f / MathF.Sqrt(_headDim);

        Span<float> attnOut = _attnOutScratch;
        attnOut.Clear();

        // Perform sparse block selection if indexer is active and topK is specified
        HashSet<int>? selectedTokenIndices = null;
        if (_indexerTopK > 0 && layer.IndexQProj != null && _qsaPooledKeys[layerIdx].Count > 0)
        {
            int totalPools = _qsaPooledKeys[layerIdx].Count;
            int topKPoolCount = Math.Max(1, _indexerTopK / _indexerKPool);

            // Project Indexer Q
            Span<float> idxQFull = stackalloc float[_indexerHeadCount * _indexerKeyLength];
            MatVec(layer.IndexQProj.Value, input, idxQFull);

            // Norm and RoPE Indexer Q
            var idxQNorm = AsFloatSpan(layer.IndexQNorm);
            if (!idxQNorm.IsEmpty)
            {
                for (int h = 0; h < _indexerHeadCount; h++)
                {
                    var qh = idxQFull.Slice(h * _indexerKeyLength, _indexerKeyLength);
                    float sumSq = TensorPrimitives.SumOfSquares(qh);
                    float invRms = 1.0f / MathF.Sqrt(sumSq / _indexerKeyLength + _eps);
                    TensorPrimitives.Multiply(qh, invRms, qh);
                    TensorPrimitives.Multiply(qh, idxQNorm, qh);
                }
            }
            int indexerRopeDim = Math.Min(_ropeDim, _indexerKeyLength);
            Qwen4ExpRope.ApplyImRope(idxQFull, position, _indexerHeadCount, _indexerKeyLength, _hp.RopeDimensionSections, indexerRopeDim, _ropeTheta);

            // Score candidate pools
            Span<float> poolScores = stackalloc float[totalPools];
            for (int p = 0; p < totalPools; p++)
            {
                poolScores[p] = Qwen4ExpQsa.ComputeBlockScore(idxQFull, _qsaPooledKeys[layerIdx][p], _indexerHeadCount, _indexerKeyLength);
            }

            Span<int> selectedPools = stackalloc int[totalPools];
            int nSelected = Qwen4ExpQsa.SelectTopKPools(poolScores, topKPoolCount, selectedPools);

            selectedTokenIndices = new HashSet<int>();
            for (int i = 0; i < nSelected; i++)
            {
                int p = selectedPools[i];
                int startToken = p * _indexerKPool;
                for (int t = 0; t < _indexerKPool; t++)
                {
                    selectedTokenIndices.Add(startToken + t);
                }
            }

            // Incomplete tail tokens are always retained
            int completedPoolTokens = totalPools * _indexerKPool;
            for (int t = completedPoolTokens; t < numTokens; t++)
            {
                selectedTokenIndices.Add(t);
            }
        }

        Span<float> scores = stackalloc float[numTokens];
        int headsPerKv = _numHeads / _numHeadsKv;

        for (int h = 0; h < _numHeads; h++)
        {
            int kvHead = h / headsPerKv;
            var qh = q.Slice(h * _headDim, _headDim);

            for (int t = 0; t < numTokens; t++)
            {
                if (selectedTokenIndices != null && !selectedTokenIndices.Contains(t))
                {
                    scores[t] = float.NegativeInfinity;
                }
                else
                {
                    var kt = _qsaKeyCache[layerIdx][t].AsSpan(kvHead * _headDim, _headDim);
                    scores[t] = TensorPrimitives.Dot(qh, kt) * invSqrtD;
                }
            }

            // Softmax
            RowKernels.SoftmaxInPlace(scores);

            // Value aggregation
            var outHead = attnOut.Slice(h * _headDim, _headDim);
            for (int t = 0; t < numTokens; t++)
            {
                float weight = scores[t];
                if (weight <= 0f || float.IsNaN(weight)) continue;

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

    internal void ExecuteMoe(Qwen4ExpLayerTensors layer, ReadOnlySpan<float> input, Span<float> output)
    {
        output.Clear();

        // 0. Routed experts: softmax router, top-k, renormalised (+ optional scale), SiLU(gate)*up -> down
        //    (llama.cpp qwen4exp.cpp build_moe_ffn call).
        if (_hp.ExpertCount > 0)
        {
            ExecuteRoutedExperts(layer, input, output);
        }

        // 1. Shared expert
        if (layer.FfnGateShexp != null && layer.FfnUpShexp != null && layer.FfnDownShexp != null)
        {
            int shExpDim = (int)layer.FfnDownShexp.Value.Info.Dimensions[0];
            Span<float> shGate = _moeGateScratch.AsSpan(0, shExpDim);
            Span<float> shUp = _moeUpScratch.AsSpan(0, shExpDim);

            MatVec(layer.FfnGateShexp.Value, input, shGate);
            MatVec(layer.FfnUpShexp.Value, input, shUp);

            // SiLU(gate) * up
            for (int i = 0; i < shExpDim; i++)
            {
                float g = shGate[i];
                float sig = 1.0f / (1.0f + MathF.Exp(-g));
                shGate[i] = (g * sig) * shUp[i];
            }

            Span<float> shOut = _shOutScratch;
            MatVec(layer.FfnDownShexp.Value, shGate, shOut);

            // Shared expert scale: sigmoid(ffn_gate_inp_shexp . input)
            if (layer.FfnGateInpShexp != null)
            {
                var wShexp = AsFloatSpan(layer.FfnGateInpShexp);
                float dot = TensorPrimitives.Dot(wShexp, input);
                float scale = 1.0f / (1.0f + MathF.Exp(-dot));
                TensorPrimitives.Multiply(shOut, scale, shOut);
            }

            TensorPrimitives.Add(output, shOut, output);
        }
    }

    private void ExecuteRoutedExperts(Qwen4ExpLayerTensors layer, ReadOnlySpan<float> input, Span<float> output)
    {
        if (layer.FfnGateInp is not { } router || layer.FfnDownExps is not { } downExps)
        {
            throw new InvalidOperationException(
                $"qwen4exp layer {layer.LayerIndex}: expert_count={_hp.ExpertCount} but routed expert tensors (ffn_gate_inp / ffn_down_exps) are missing.");
        }

        bool hasSeparate = layer.FfnGateExps is not null && layer.FfnUpExps is not null;
        bool hasFused = layer.FfnGateUpExps is not null;

        if (!hasSeparate && !hasFused)
        {
            throw new InvalidOperationException(
                $"qwen4exp layer {layer.LayerIndex}: expert_count={_hp.ExpertCount} but neither separate (ffn_gate_exps, ffn_up_exps) nor fused (ffn_gate_up_exps) tensors were found.");
        }

        int numExperts = _hp.ExpertCount;
        int topK = Math.Min(_hp.ExpertUsedCount, numExperts);
        int interDim = hasSeparate
            ? (int)layer.FfnGateExps!.Value.Info.Dimensions[1]
            : (int)layer.FfnGateUpExps!.Value.Info.Dimensions[1] / 2;

        Span<float> logits = _routerLogitsScratch.AsSpan(0, numExperts);
        MatVec(router, input, logits);

        Span<int> topIdx = _moeTopIdxScratch.AsSpan(0, topK);
        Span<float> topW = _moeTopWScratch.AsSpan(0, topK);
        Qwen4ExpMoeRouting.Route(logits, topK, _hp.ExpertWeightsScale, topIdx, topW);

        Span<float> gate = _moeGateScratch.AsSpan(0, interDim);
        Span<float> up = _moeUpScratch.AsSpan(0, interDim);
        Span<float> down = _moeDownScratch;
        for (int k = 0; k < topK; k++)
        {
            int expert = topIdx[k];
            if (hasSeparate)
            {
                ExpertMatVec(layer.FfnGateExps!.Value, expert, input, gate);
                ExpertMatVec(layer.FfnUpExps!.Value, expert, input, up);
            }
            else
            {
                ExpertMatVec(layer.FfnGateUpExps!.Value, expert, input, gate, rowOffset: 0);
                ExpertMatVec(layer.FfnGateUpExps!.Value, expert, input, up, rowOffset: interDim);
            }

            for (int i = 0; i < interDim; i++)
            {
                float g = gate[i];
                gate[i] = (g / (1.0f + MathF.Exp(-g))) * up[i];
            }
            ExpertMatVec(downExps, expert, gate, down);
            TensorPrimitives.MultiplyAdd(down, topW[k], output, output);
        }
    }

    /// <summary>Row-sliced matvec into expert <paramref name="expert"/> of a stacked [in, out, experts] tensor.</summary>
    private static void ExpertMatVec(Qwen4ExpTensorRef tensor, int expert, ReadOnlySpan<float> inVec, Span<float> outVec, int rowOffset = 0)
    {
        int inDim = (int)tensor.Info.Dimensions[0];
        int outDim = outVec.Length;
        int totalRows = (int)tensor.Info.Dimensions[1];
        long bytesPerRow = (long)(inDim / DTypeInfo.BlockSize(tensor.DType)) * DTypeInfo.BytesPerBlock(tensor.DType);
        byte* expertPtr = tensor.DataPtr + ((long)expert * totalRows + rowOffset) * bytesPerRow;
        fixed (float* outPtr = outVec, inPtr = inVec)
        {
            SimdKernels.MatVec(outPtr, expertPtr, inPtr, outDim, inDim, tensor.DType);
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
        _pleHasher?.Reset();
        Array.Clear(_pleConvHistory);
        Array.Clear(_resHc);
        for (int l = 0; l < _numLayers; l++)
        {
            if (_gdnConvState[l].Length > 0) Array.Clear(_gdnConvState[l]);
            if (_gdnState[l].Length > 0) Array.Clear(_gdnState[l]);
            _qsaKeyCache[l].Clear();
            _qsaValCache[l].Clear();
            _qsaRawIndexKCache[l].Clear();
            _qsaPooledKeys[l].Clear();
        }
    }

    public void Dispose() { }
}
