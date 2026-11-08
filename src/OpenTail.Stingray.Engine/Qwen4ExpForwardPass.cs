using System.Numerics.Tensors;
using System.Runtime.InteropServices;
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
    // Dequantized copies of the small "control" tensors AsFloatSpan serves (norm weights, HC
    // gate/down/up projections, SSM/GDN scalars, conv kernels) -- these are all small per-layer
    // tensors, not the large MoE expert matrices (those go through DType-aware MatVec dispatch
    // elsewhere, not this cache), so caching every one of them as F32 is bounded (a few GB at
    // most across 48 layers) and safe on this machine's RAM even for the full paged 72.5 GB model.
    // Keyed by the tensor's raw data pointer, which is stable for the model's lifetime (backed by
    // GgufModel's mmap).
    private readonly Dictionary<nint, float[]> _dequantCache = new();
    private readonly Dictionary<nint, float[]> _convWeightTransposeCache = new();
    private readonly bool _q4eDebug = Environment.GetEnvironmentVariable("STINGRAY_Q4E_DEBUG") == "1";
    // Diagnostic dump of l_last-N and result_output at one position (see Phase 0.2 of the Qwen plan).
    private readonly string? _dumpDir = Environment.GetEnvironmentVariable("STINGRAY_Q4E_DUMP_DIR");
    private readonly int _dumpPos = int.TryParse(Environment.GetEnvironmentVariable("STINGRAY_Q4E_DUMP_POS"), out int p) ? p : -1;
    private bool _dumpActive;
    // Noise probe (Phase 0.2): relative embedding perturbation, 0 by default.
    private static readonly float PerturbEps = float.TryParse(Environment.GetEnvironmentVariable("STINGRAY_Q4E_PERTURB_EPS"), System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out float pe) ? pe : 0f;
    // Phase 0.2 experiment: ssm_out as an f32 dot instead of the quantised MatVec. Off by default.
    private static readonly bool SsmOutF32Experiment = Environment.GetEnvironmentVariable("STINGRAY_Q4E_SSMOUT_F32") == "1";
    // Phase 0.2 experiment: routed experts as dequantised f32 dots. Off by default.
    private static readonly bool ExpertF32Experiment = Environment.GetEnvironmentVariable("STINGRAY_Q4E_EXPERT_F32") == "1";
    private double[]? _layerSumAccum;
    private double[]? _hcCombine0Accum;
    private double[]? _linearAttnOut0Accum;
    private double[]? _alphaAccum;
    private double[]? _betaAccum;
    private double[]? _qAccum;
    private double[]? _kAccum;
    private double[]? _vAccum;
    private double[]? _qkvRawAccum;
    private double[]? _convRawAccum;
    private double[]? _convSiluAccum;
    private double[]? _ffnOut0Accum;
    private double[]? _ffnInject0Accum;
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
    private Qwen4ExpSequenceState _seq = null!; // per-sequence state, see Qwen4ExpSequenceState

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
        _seq = CreateSequenceState();

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
        _dumpActive = _dumpDir != null && position == _dumpPos;
        _seq.TokensProcessed = position + 1;

        // 1. Embed token
        EmbedToken(token, _tokEmbdScratch);
        if (_dumpActive) DumpFloats(Path.Combine(_dumpDir!, "model_input_embed.bin"), _tokEmbdScratch);
        if (PerturbEps != 0f)
        {
            // Noise probe (Phase 0.2): deterministic relative perturbation of the embedding.
            for (int i = 0; i < _tokEmbdScratch.Length; i++)
            {
                double u = Math.Sin(i * 12.9898 + position * 78.233) * 43758.5453;
                u -= Math.Floor(u);
                _tokEmbdScratch[i] *= 1f + PerturbEps * (float)(2 * u - 1);
            }
        }

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
                ExecutePle(il, position, layer, _tokEmbdScratch, token);
            }

            // 2b. HC Mix before Token Mixer
            var hcAttnNorm = AsFloatSpan(layer.HcAttnNorm);
            var hcAttnDown = AsFloatSpan(layer.HcAttnDown);
            var hcAttnUp = AsFloatSpan(layer.HcAttnUp);
            var hcAttnInject = AsFloatSpan(layer.HcAttnInject);

            Qwen4ExpGatedResidual.Mix(
                _resHc, hcAttnNorm, hcAttnDown, hcAttnUp, hcAttnInject,
                _mixed, _inject, _hc, _embedDim, _hcLowRank, _eps);

            if (il == 0 && Environment.GetEnvironmentVariable("STINGRAY_Q4E_DEBUG") == "1")
            {
                double s = 0; foreach (var v in _mixed) s += v;
                Console.Error.WriteLine($"[q4e-dbg] pos={position} hc_mixed(l0,attn) sum={s:F6}");
            }

            // 2c. Token Mixer (GDN or QSA)
            if (layer.IsRecurrent)
            {
                ExecuteGdnMixer(il, layer, _mixed, _blockOut, position);
            }
            else
            {
                ExecuteQsaMixer(il, layer, _mixed, position, _blockOut);
            }

            if (_q4eDebug && il == 0 && position <= 4)
            {
                double s = 0; foreach (var v in _blockOut) s += v;
                (_linearAttnOut0Accum ??= new double[1])[0] += s;
                if (position == 4)
                    Console.Error.WriteLine($"[q4e-dbg] linear_attn_out-0 sum={_linearAttnOut0Accum[0]:F6}");
            }

            if (_dumpDir != null && position == _dumpPos)
                DumpFloats(Path.Combine(_dumpDir, $"linear_attn_out_{il}.bin"), _blockOut);

            // 2d. HC Combine after Token Mixer
            Qwen4ExpGatedResidual.Combine(_resHc, _blockOut, _inject, _hc, _embedDim);

            if (_dumpDir != null && position == _dumpPos)
                DumpFloats(Path.Combine(_dumpDir, $"hc_combine_{il}.bin"), _resHc);

            if (_q4eDebug && il == 0 && position <= 4)
            {
                double s = 0; foreach (var v in _resHc) s += v;
                (_hcCombine0Accum ??= new double[1])[0] += s;
                if (position == 4)
                    Console.Error.WriteLine($"[q4e-dbg] hc_combine-0 sum={_hcCombine0Accum[0]:F6}");
            }

            // 2e. HC Mix before MoE
            var hcFfnNorm = AsFloatSpan(layer.HcFfnNorm);
            var hcFfnDown = AsFloatSpan(layer.HcFfnDown);
            var hcFfnUp = AsFloatSpan(layer.HcFfnUp);
            var hcFfnInject = AsFloatSpan(layer.HcFfnInject);

            Qwen4ExpGatedResidual.Mix(
                _resHc, hcFfnNorm, hcFfnDown, hcFfnUp, hcFfnInject,
                _mixed, _inject, _hc, _embedDim, _hcLowRank, _eps);

            if (_q4eDebug && il == 0 && position <= 4)
            {
                double s = 0; foreach (var v in _inject) s += v;
                (_ffnInject0Accum ??= new double[1])[0] += s;
                if (position == 4)
                    Console.Error.WriteLine($"[q4e-dbg] ffn hc_inject-0 sum={_ffnInject0Accum[0]:F6}");
            }

            if (_dumpActive) DumpFloats(Path.Combine(_dumpDir!, $"hc_mixed_occ2_{il}.bin"), _mixed);

            // 2f. MoE (routed + shared expert)
            ExecuteMoe(layer, _mixed, _blockOut);

            if (_dumpDir != null && position == _dumpPos)
                DumpFloats(Path.Combine(_dumpDir, $"ffn_out_{il}.bin"), _blockOut);

            if (_q4eDebug && il == 0 && position <= 4)
            {
                double s = 0; foreach (var v in _blockOut) s += v;
                (_ffnOut0Accum ??= new double[1])[0] += s;
                if (position == 4)
                    Console.Error.WriteLine($"[q4e-dbg] ffn_out-0 sum={_ffnOut0Accum[0]:F6}");
            }

            // 2g. HC Combine after MoE
            Qwen4ExpGatedResidual.Combine(_resHc, _blockOut, _inject, _hc, _embedDim);

            if (_q4eDebug && position <= 4)
            {
                double s = 0; foreach (var v in _resHc) s += v;
                (_layerSumAccum ??= new double[_numLayers])[il] += s;
                if (il == _numLayers - 1 && position == 4)
                    for (int li = 0; li < _numLayers; li++)
                        Console.Error.WriteLine($"[q4e-dbg] l_last-{li} sum={_layerSumAccum[li]:F6}");
            }

            if (_dumpDir != null && position == _dumpPos)
                DumpFloats(Path.Combine(_dumpDir, $"l_last_{il}.bin"), _resHc);
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

        if (Environment.GetEnvironmentVariable("STINGRAY_Q4E_DEBUG") == "1")
        {
            double s = 0; foreach (var v in _logits) s += v;
            Console.Error.WriteLine($"[q4e-dbg] pos={position} result_output sum={s:F6}");
        }

        if (_dumpDir != null && position == _dumpPos)
            DumpFloats(Path.Combine(_dumpDir, "result_output.bin"), _logits);

        return _logits;
    }

    private static void DumpFloats(string path, ReadOnlySpan<float> data)
    {
        using var fs = File.Create(path);
        fs.Write(MemoryMarshal.AsBytes(data));
    }

    private static void DumpInts(string path, ReadOnlySpan<int> data)
    {
        using var fs = File.Create(path);
        fs.Write(MemoryMarshal.AsBytes(data));
    }

    private void ExecutePle(int il, int position, Qwen4ExpLayerTensors layer, ReadOnlySpan<float> tokEmb, int token)
    {
        bool dump = _dumpDir != null && position == _dumpPos;
        // 1. Determine PLE embedding input:
        // If the PLE n-gram table (PerLayerTokEmbd) is present and hasher is initialized,
        // gather the embedding row for each head and concatenate them into pleEmb.
        // Otherwise fall back to tokEmb.
        Span<float> pleEmb = _pleEmbScratch;
        bool gatheredFromTable = false;

        if (_pleHasher != null && _tensors.PerLayerTokEmbd != null && _pleHeadRows != null)
        {
            _seq.PleHasher!.PushToken(token);
            _seq.PleHasher!.ComputeRowIndices(_pleHeadRows);

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

        if (dump)
        {
            DumpFloats(Path.Combine(_dumpDir!, "ple_embd.bin"), pleEmb);
            DumpFloats(Path.Combine(_dumpDir!, $"ple_gate_{il}.bin"), gate);
            DumpFloats(Path.Combine(_dumpDir!, $"ple_gated_value_{il}.bin"), gated);
        }

        // Grouped RMSNorm on gated
        Span<float> normConvIn = stackalloc float[_hcDim];
        Qwen4ExpPle.GroupedRmsNorm(gated, AsFloatSpan(layer.PleNormConv), normConvIn, _hc, _embedDim, _eps);

        // Update conv history buffer
        // Shift history slots left
        if (_pleHistSlots > 1)
        {
            Array.Copy(_seq.PleConvHistory, _hcDim, _seq.PleConvHistory, 0, (_pleHistSlots - 1) * _hcDim);
        }
        normConvIn.CopyTo(_seq.PleConvHistory.AsSpan((_pleHistSlots - 1) * _hcDim, _hcDim));
        _seq.PleTokensSeen++;

        // Step dilated depthwise causal 1D conv
        Span<float> convOut = stackalloc float[_hcDim];
        if (layer.PleConv1d != null)
        {
            Qwen4ExpPle.StepDilatedConv(
                _seq.PleConvHistory, GetTransposedPleConv(layer), convOut,
                _hcDim, _pleConvKernel, _pleNgramSize, _pleHistSlots);
        }

        if (dump)
        {
            DumpFloats(Path.Combine(_dumpDir!, $"ple_conv_out_{il}.bin"), convOut);
        }

        // Add to residual: res_hc += gated + conv_out (qwen4exp.cpp:1477)
        for (int i = 0; i < _hcDim; i++)
        {
            _resHc[i] += gated[i] + convOut[i];
        }
    }

    private void ExecuteGdnMixer(int layerIdx, Qwen4ExpLayerTensors layer, ReadOnlySpan<float> input, Span<float> output, int position)
    {
        // Recurrent GDN mixer
        output.Clear();
        if (_dumpActive) DumpFloats(Path.Combine(_dumpDir!, $"hc_mixed_{layerIdx}.bin"), input);
        if (layer.AttnQkv == null) return;

        // QKV projection
        int keyDim = _ssmDState * _ssmNGroup;
        int valDim = _ssmDState * _ssmDtRank;
        int qkvDim = keyDim * 2 + valDim;

        Span<float> qkv = stackalloc float[qkvDim];
        MatVec(layer.AttnQkv.Value, input, qkv);
        if (_dumpActive) DumpFloats(Path.Combine(_dumpDir!, $"linear_attn_qkv_mixed_{layerIdx}.bin"), qkv);

        if (_q4eDebug && layerIdx == 0 && position <= 4)
        {
            double s = 0; foreach (var x in qkv) s += x;
            (_qkvRawAccum ??= new double[1])[0] += s;
            if (position == 4)
                Console.Error.WriteLine($"[q4e-dbg] linear_attn_qkv_mixed-0 sum={_qkvRawAccum[0]:F6}");
        }

        // Conv1D over QKV channels
        var convState = _seq.GdnConvState[layerIdx];
        Span<float> qkvPostConv = stackalloc float[qkvDim];

        if (layer.SsmConv1d != null)
        {
            // GGUF stores blk.N.ssm_conv1d.weight as ne=[kernel, channels] (kernel innermost/
            // fastest in memory per ggml's ggml_compute_forward_ssm_conv_f32, c[i0 + i1*nc]) --
            // i.e. on-disk layout is [channel][kernel], the transpose of what Conv1dStep's
            // weight.Slice(k*channels, channels) expects ([kernel][channel]). HybridGdnForwardPass
            // (Qwen 3.5 hybrid, already verified) already transposes this at load time
            // (LoadConv1dTransposed); this port never did, so it silently ran the wrong conv
            // weights on every token. Found 2026-11-07 via real-weight layer-by-layer sum diffing
            // against llama.cpp's eval-callback dump (conv_output_raw was off by ~3.5x; after this
            // fix the raw QKV mixer output -- linear_attn_out-0 -- moved from +30.5 (wrong sign) to
            // -2.76, matching llama.cpp's -2.59 to within quantisation noise).
            var wConvT = GetTransposedConvWeight(layer, qkvDim);
            GdnKernels.CausalDepthwiseConv1dDecode(qkv, convState, wConvT, qkvPostConv, qkvDim, _ssmDConv);
        }
        else
        {
            qkv.CopyTo(qkvPostConv);
        }

        if (_q4eDebug && layerIdx == 0 && position <= 4)
        {
            double s = 0; foreach (var x in qkvPostConv) s += x;
            (_convRawAccum ??= new double[1])[0] += s;
            if (position == 4)
                Console.Error.WriteLine($"[q4e-dbg] conv_output_raw-0 sum={_convRawAccum[0]:F6}");
        }

        if (_dumpActive) DumpFloats(Path.Combine(_dumpDir!, $"conv_output_raw_{layerIdx}.bin"), qkvPostConv);

        // SiLU on the conv output, confirmed 2026-11-07 against both llama.cpp (qwen4exp.cpp:1160,
        // "conv_output_silu = ggml_silu(conv_output_proper)") and TensorSharp's own doc comment
        // ("PrepareGdnInputs has already done ... conv1d + SiLU"). This port had NO activation
        // here at all -- found via real-weight layer-by-layer sum diffing against llama.cpp's
        // eval-callback dump: hc_mixed (pre-mixer) matched within 0.4%, but hc_combine (post-GDN-
        // mixer, same layer) was already off by ~90%, isolating the bug to inside this mixer.
        for (int i = 0; i < qkvDim; i++)
        {
            float x = qkvPostConv[i];
            qkvPostConv[i] = x / (1.0f + MathF.Exp(-x));
        }

        if (_q4eDebug && layerIdx == 0 && position <= 4)
        {
            double s = 0; foreach (var x in qkvPostConv) s += x;
            (_convSiluAccum ??= new double[1])[0] += s;
            if (position == 4)
                Console.Error.WriteLine($"[q4e-dbg] conv_output_silu-0 sum={_convSiluAccum[0]:F6}");
        }

        if (_dumpActive) DumpFloats(Path.Combine(_dumpDir!, $"conv_output_silu_{layerIdx}.bin"), qkvPostConv);

        // Gate projection z
        Span<float> z = stackalloc float[valDim];
        if (layer.AttnGate != null)
        {
            MatVec(layer.AttnGate.Value, input, z);
        }
        if (_dumpActive) DumpFloats(Path.Combine(_dumpDir!, $"z_{layerIdx}.bin"), z);

        // Slice Q, K, V
        var q = qkvPostConv.Slice(0, keyDim);
        var k = qkvPostConv.Slice(keyDim, keyDim);
        var v = qkvPostConv.Slice(keyDim * 2, valDim);

        // Per-head L2 norm on Q and K (no learned weight), confirmed 2026-11-07 against
        // llama.cpp's build_gdn_l2_norm (models.h:14-18: rms_norm(x, eps/n) scaled by
        // 1/sqrt(n), which works out to x / sqrt(sum(x^2) + eps) -- plain L2 norm, not
        // RMSNorm-with-gamma). Applied per head, before the K-head tiling below. This port
        // had no normalization here at all -- found the same way as the GDN conv SiLU fix,
        // continuing the layer-by-layer diff after that fix narrowed but did not close the gap.
        for (int h = 0; h < _ssmNGroup; h++)
        {
            var qh = q.Slice(h * _ssmDState, _ssmDState);
            float qSumSq = TensorPrimitives.SumOfSquares(qh);
            TensorPrimitives.Multiply(qh, 1.0f / MathF.Sqrt(qSumSq + _eps), qh);

            var kh = k.Slice(h * _ssmDState, _ssmDState);
            float kSumSq = TensorPrimitives.SumOfSquares(kh);
            TensorPrimitives.Multiply(kh, 1.0f / MathF.Sqrt(kSumSq + _eps), kh);
        }

        // Alpha and Beta projections
        Span<float> alpha = stackalloc float[_ssmDtRank];
        Span<float> beta = stackalloc float[_ssmDtRank];
        if (layer.SsmAlpha != null) MatVec(layer.SsmAlpha.Value, input, alpha);
        if (layer.SsmBeta != null) MatVec(layer.SsmBeta.Value, input, beta);

        if (_dumpActive)
        {
            DumpFloats(Path.Combine(_dumpDir!, $"alpha_{layerIdx}.bin"), alpha);
            DumpFloats(Path.Combine(_dumpDir!, $"beta_raw_{layerIdx}.bin"), beta);
        }

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
        var gdnState = _seq.GdnState[layerIdx];

        if (_q4eDebug && layerIdx == 0 && position <= 4)
        {
            double sA = 0; foreach (var x in alpha) sA += x;
            double sB = 0; foreach (var x in beta) sB += x;
            double sQ = 0; foreach (var x in qTiled) sQ += x;
            double sK = 0; foreach (var x in kTiled) sK += x;
            double sV = 0; foreach (var x in v) sV += x;
            (_alphaAccum ??= new double[1])[0] += sA;
            (_betaAccum ??= new double[1])[0] += sB;
            (_qAccum ??= new double[1])[0] += sQ;
            (_kAccum ??= new double[1])[0] += sK;
            (_vAccum ??= new double[1])[0] += sV;
            if (position == 4)
                Console.Error.WriteLine(
                    $"[q4e-dbg] alpha-0(raw) sum={_alphaAccum[0]:F6} beta-0(raw) sum={_betaAccum[0]:F6} " +
                    $"q_tiled-0 sum={_qAccum[0]:F6} k_tiled-0 sum={_kAccum[0]:F6} v-0 sum={_vAccum[0]:F6}");
        }

        Span<float> gdnOut = stackalloc float[valDim];
        // qwen4exp gates the GDN output with sigmoid(z), not SiLU(z) -- confirmed 2026-11-07
        // against TensorSharp's GatedDeltaNetRecurrenceManaged, which documents this exact
        // one-flag difference from the already-verified Qwen 3.5 hybrid model that this shared
        // kernel defaults to.
        GdnKernels.GdnRecurrenceDecode(
            qTiled, kTiled, v, alpha, beta, ssmA, ssmDt, normWeight, z,
            gdnState, gdnOut, _ssmDtRank, _ssmDState, _eps, useSigmoidGate: true);

        if (_dumpActive) DumpFloats(Path.Combine(_dumpDir!, $"final_output_{layerIdx}.bin"), gdnOut);

        // Output projection
        if (layer.SsmOut != null)
        {
            if (SsmOutF32Experiment)
            {
                // Experiment (Phase 0.2): dot the dequantised f32 weights, no activation rounding.
                var w = AsFloatSpan(layer.SsmOut);
                int inDim = gdnOut.Length;
                for (int r = 0; r < output.Length; r++)
                    output[r] = TensorPrimitives.Dot(w.Slice(r * inDim, inDim), gdnOut);
            }
            else
            {
                MatVec(layer.SsmOut.Value, gdnOut, output);
            }
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
        _seq.QsaKeyCache[layerIdx].Add(kCached);
        _seq.QsaValCache[layerIdx].Add(vCached);

        // Maintain QSA raw indexer-K cache and form K-pools
        if (layer.IndexKProj != null)
        {
            Span<float> rawIdxK = stackalloc float[_indexerKeyLength];
            MatVec(layer.IndexKProj.Value, input, rawIdxK);
            _seq.QsaRawIndexKCache[layerIdx].Add(rawIdxK.ToArray());

            // Check if a full pool of kpool tokens has just been completed
            int rawCount = _seq.QsaRawIndexKCache[layerIdx].Count;
            if (rawCount % _indexerKPool == 0)
            {
                int poolIdx = (rawCount / _indexerKPool) - 1;
                Span<float> blockKeys = stackalloc float[_indexerKPool * _indexerKeyLength];
                for (int p = 0; p < _indexerKPool; p++)
                {
                    int t = poolIdx * _indexerKPool + p;
                    _seq.QsaRawIndexKCache[layerIdx][t].CopyTo(blockKeys.Slice(p * _indexerKeyLength, _indexerKeyLength));
                }
                Span<float> pooledKey = stackalloc float[_indexerKeyLength];
                var indexKNorm = AsFloatSpan(layer.IndexKNorm);
                Qwen4ExpQsa.PoolIndexerKeys(blockKeys, indexKNorm, pooledKey, _indexerKPool, _indexerKeyLength, _eps);

                // Apply IMRoPE to pooled key at first token position of this block
                int blockStartPos = poolIdx * _indexerKPool;
                int indexerRopeDim = Math.Min(_ropeDim, _indexerKeyLength);
                Qwen4ExpRope.ApplyImRope(pooledKey, blockStartPos, 1, _indexerKeyLength, _hp.RopeDimensionSections, indexerRopeDim, _ropeTheta);
                _seq.QsaPooledKeys[layerIdx].Add(pooledKey.ToArray());
            }
        }

        int numTokens = _seq.QsaKeyCache[layerIdx].Count;
        float invSqrtD = 1.0f / MathF.Sqrt(_headDim);

        Span<float> attnOut = _attnOutScratch;
        attnOut.Clear();

        // Perform sparse block selection if indexer is active and topK is specified
        HashSet<int>? selectedTokenIndices = null;
        if (_indexerTopK > 0 && layer.IndexQProj != null && _seq.QsaPooledKeys[layerIdx].Count > 0)
        {
            int totalPools = _seq.QsaPooledKeys[layerIdx].Count;
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
                poolScores[p] = Qwen4ExpQsa.ComputeBlockScore(idxQFull, _seq.QsaPooledKeys[layerIdx][p], _indexerHeadCount, _indexerKeyLength);
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
                    var kt = _seq.QsaKeyCache[layerIdx][t].AsSpan(kvHead * _headDim, _headDim);
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

                var vt = _seq.QsaValCache[layerIdx][t].AsSpan(kvHead * _headDim, _headDim);
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
        if (_dumpActive) DumpFloats(Path.Combine(_dumpDir!, $"ffn_moe_out_{layer.LayerIndex}.bin"), output);

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
                if (_dumpActive)
                {
                    DumpFloats(Path.Combine(_dumpDir!, $"ffn_shexp_{layer.LayerIndex}.bin"), shOut);
                    DumpFloats(Path.Combine(_dumpDir!, $"shared_expert_gate_sigmoid_{layer.LayerIndex}.bin"), new[] { scale });
                }
                TensorPrimitives.Multiply(shOut, scale, shOut);
            }

            if (_dumpActive) DumpFloats(Path.Combine(_dumpDir!, $"ffn_shexp_gated_{layer.LayerIndex}.bin"), shOut);
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

        if (_dumpActive)
        {
            int il = layer.LayerIndex;
            DumpFloats(Path.Combine(_dumpDir!, $"ffn_moe_logits_{il}.bin"), logits);
            DumpFloats(Path.Combine(_dumpDir!, $"ffn_moe_weights_norm_{il}.bin"), topW);
            DumpInts(Path.Combine(_dumpDir!, $"ffn_moe_topk_{il}.bin"), topIdx);
        }

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
        if (ExpertF32Experiment)
        {
            // Experiment (Phase 0.2): dequantised f32 weights, no activation rounding.
            int count = outDim * inDim;
            var wf = new float[count];
            Dequantize.ToFloat32(new ReadOnlySpan<byte>(expertPtr, checked((int)(outDim * bytesPerRow))), wf, tensor.DType, count);
            for (int r = 0; r < outDim; r++)
                outVec[r] = TensorPrimitives.Dot(wf.AsSpan(r * inDim, inDim), inVec);
            return;
        }
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
            MatVecLlamaNumerics(outPtr, tensor.DataPtr, inPtr, outDim, inDim, tensor.DType);
        }
    }

    /// <summary>
    /// Matvec with llama.cpp's activation numerics for the Qwen 3.8 path. llama.cpp's Q5_K matmul quantises the
    /// activation to Q8_K (ggml_vec_dot_q5_K_q8_K). The generic <see cref="SimdKernels.MatVec"/> keeps f32 activations
    /// for Q5_K, which left a 2.2% error at ssm_out (Phase 0.2). Other dtypes already quantise like llama.cpp.
    /// </summary>
    public static void MatVecLlamaNumerics(float* output, byte* weights, float* input, int rows, int cols, DType dtype)
    {
        if (dtype == DType.Q5_K)
            SimdKernels.MatVecQ5K_Q8K(output, weights, input, rows, cols);
        else
            SimdKernels.MatVec(output, weights, input, rows, cols, dtype);
    }

    private ReadOnlySpan<float> AsFloatSpan(Qwen4ExpTensorRef? tensorRef)
    {
        if (tensorRef is not { } t) return default;
        int count = (int)t.Info.ElementCount;
        // Confirmed by crashing on a real checkpoint (2026-11-07): this used to cast every tensor
        // as Float32 with no dtype check at all. Qwen4Exp's real GGUF stores these small control
        // tensors (hc_attn/ffn_down/up/norm etc.) as Q8_0, so (float*)t.DataPtr with
        // Info.ElementCount read as float-count massively overran the actual (much smaller,
        // quantized) buffer -> a native AccessViolationException reading garbage/protected memory
        // inside Qwen4ExpGatedResidual.Mix. Fixed to dequantize via the same Dequantize.ToFloat32
        // path the rest of the engine uses, cached per tensor pointer since these weights never
        // change across calls (see _dequantCache's own doc comment for why this is safe here).
        if (t.DType == DType.Float32)
            return new ReadOnlySpan<float>((float*)t.DataPtr, count);

        nint key = (nint)t.DataPtr;
        if (!_dequantCache.TryGetValue(key, out var cached))
        {
            cached = new float[count];
            long byteSize = DTypeInfo.ByteSize(count, t.DType);
            var src = new ReadOnlySpan<byte>(t.DataPtr, checked((int)byteSize));
            Dequantize.ToFloat32(src, cached, t.DType, count);
            _dequantCache[key] = cached;
        }
        return cached;
    }

    /// <summary>
    /// Dequantized and transposed copy of a layer's <c>ssm_conv1d.weight</c>, from GGUF's
    /// on-disk <c>[channel][kernel]</c> layout into the <c>[kernel][channel]</c> layout
    /// <see cref="GdnKernels.CausalDepthwiseConv1dDecode"/> expects (see the call site's comment
    /// for the source). Cached per tensor pointer, same rationale as <see cref="AsFloatSpan"/>.
    /// </summary>
    private ReadOnlySpan<float> GetTransposedConvWeight(Qwen4ExpLayerTensors layer, int channels)
    {
        var t = layer.SsmConv1d!.Value;
        int count = (int)t.Info.ElementCount;
        nint key = (nint)t.DataPtr;
        if (_convWeightTransposeCache.TryGetValue(key, out var cached)) return cached;

        float[] src;
        if (t.DType == DType.Float32)
        {
            src = new float[count];
            new ReadOnlySpan<float>((float*)t.DataPtr, count).CopyTo(src);
        }
        else
        {
            src = new float[count];
            long byteSize = DTypeInfo.ByteSize(count, t.DType);
            Dequantize.ToFloat32(new ReadOnlySpan<byte>(t.DataPtr, checked((int)byteSize)), src, t.DType, count);
        }

        var dst = new float[count];
        for (int c = 0; c < channels; c++)
            for (int k = 0; k < _ssmDConv; k++)
                dst[k * channels + c] = src[c * _ssmDConv + k];

        _convWeightTransposeCache[key] = dst;
        return dst;
    }

    // GGUF stores blk.N.ple_conv1d.weight as [channel][tap] (tap innermost, bf16). StepDilatedConv reads [tap][channel].
    private ReadOnlySpan<float> GetTransposedPleConv(Qwen4ExpLayerTensors layer)
    {
        var t = layer.PleConv1d!.Value;
        int count = (int)t.Info.ElementCount;
        nint key = (nint)t.DataPtr;
        if (_convWeightTransposeCache.TryGetValue(key, out var cached)) return cached;

        var src = new float[count];
        if (t.DType == DType.Float32)
        {
            new ReadOnlySpan<float>((float*)t.DataPtr, count).CopyTo(src);
        }
        else
        {
            long byteSize = DTypeInfo.ByteSize(count, t.DType);
            Dequantize.ToFloat32(new ReadOnlySpan<byte>(t.DataPtr, checked((int)byteSize)), src, t.DType, count);
        }

        var dst = new float[count];
        Qwen4ExpPle.TransposeConvWeights(src, _hcDim, _pleConvKernel, dst);

        _convWeightTransposeCache[key] = dst;
        return dst;
    }

    /// <summary>
    /// Sum of layer 0's raw GDN-mixer output (post output-projection, pre HC-combine) across
    /// positions 0-4 of a 5-token prefill -- only populated when STINGRAY_Q4E_DEBUG=1. Exposed for
    /// <see cref="OpenTail.Stingray.Tests.Core.Qwen4ExpRealWeightsSmokeTests"/>'s regression test
    /// for the conv1d weight-transpose fix (2026-11-07); not meant for production use.
    /// </summary>
    internal double? DebugLinearAttnOut0Sum => _q4eDebug ? _linearAttnOut0Accum?[0] : null;

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
        if (length == 0) { ResetCache(); return; }
        if (length != _seq.PleTokensSeen)
        {
            // GDN conv/recurrent state and the QSA raw-key/pool caches are destructively updated
            // per token (SupportsPartialRewind defaults to false and is not overridden here), so
            // only a full reset or a no-op at the current length is representable -- any other
            // value previously fell through silently instead of rejecting the call, which left
            // whatever state existed unchanged while callers believed the rewind had happened.
            throw new NotSupportedException(
                $"Qwen4ExpForwardPass (alpha): TruncateTo({length}) is not supported -- only " +
                $"full reset (TruncateTo(0)) or a no-op (current length {_seq.PleTokensSeen}) is " +
                "representable, because GDN/QSA state is destructively updated per token.");
        }
    }

    public void ResetCache()
    {
        _seq.Reset();
        Array.Clear(_resHc);
    }

    /// <summary>A new, empty sequence state for this model (its own PLE token history and caches).</summary>
    public Qwen4ExpSequenceState CreateSequenceState() => new(
        _numLayers + 1, // + the MTP head's block, whose attention cache lives at index _numLayers
        l => l < _numLayers && _hp.IsRecurrentLayer(l),
        (_ssmDConv - 1) * _convDim,
        _ssmDtRank * _ssmDState * _ssmDState,
        _pleHistSlots * _hcDim,
        _hp.PleLayers != null && _hp.PleLayers.Count > 0 ? new Qwen4ExpPleHasher(_hp) : null);

    /// <summary>Hyperparameters this pass was built with.</summary>
    public Qwen4ExpHyperparams Hyperparameters => _hp;

    /// <summary>Number of transformer layers.</summary>
    public int NumLayers => _numLayers;

    /// <summary>
    /// HC residual after the last <see cref="Forward(int, int, Qwen4ExpSequenceState)"/> call: the trunk's h_nextn for that
    /// position. Valid until the next forward call.
    /// </summary>
    public ReadOnlySpan<float> LastHcResidual => _resHc;

    private Qwen4ExpMtpHead? _mtpHead;
    private float[]? _mtpEmbed, _mtpEnorm, _mtpHnorm, _mtpRes, _mtpConcat;

    /// <summary>Attaches the qwen4exp MTP draft head (Phase 4). Required before <see cref="ForwardMtp"/>.</summary>
    public void AttachMtpHead(Qwen4ExpMtpHead head)
    {
        _mtpHead = head ?? throw new ArgumentNullException(nameof(head));
        _mtpEmbed ??= new float[_embedDim];
        _mtpEnorm ??= new float[_embedDim];
        _mtpHnorm ??= new float[_hcDim];
        _mtpRes ??= new float[_hcDim];
        _mtpConcat ??= new float[2 * _embedDim];
    }

    /// <summary>
    /// One MTP draft step (reference graph_mtp, qwen4exp.cpp ~529-612). Given the draft token and the trunk's HC residual
    /// <paramref name="hIn"/> at the previous position, returns the draft logits and writes the next residual into
    /// <paramref name="hOut"/>. Uses the head's own attention cache in <paramref name="sequence"/> (index NumLayers).
    /// </summary>
    public ReadOnlySpan<float> ForwardMtp(
        int draftToken, ReadOnlySpan<float> hIn, int position, Qwen4ExpSequenceState sequence, Span<float> hOut)
    {
        var head = _mtpHead ?? throw new InvalidOperationException("No MTP head attached (AttachMtpHead).");
        if (hIn.Length != _hcDim || hOut.Length != _hcDim)
            throw new ArgumentException($"MTP residual must have {_hcDim} values.");
        sequence.ThrowIfDisposed();

        var previous = _seq;
        _seq = sequence;
        try
        {
            int head0 = _numLayers;
            var layer = head.Layer;
            var res = _mtpRes!;

            // 1. token embedding (shared with the trunk), then RMS norm with enorm.
            EmbedToken(draftToken, _mtpEmbed!);
            Qwen4ExpPle.GroupedRmsNorm(_mtpEmbed!, AsFloatSpan(head.EnNorm), _mtpEnorm!, 1, _embedDim, _eps);

            // 2. hidden state: grouped RMS norm per stream with hnorm.
            Qwen4ExpPle.GroupedRmsNorm(hIn, AsFloatSpan(head.HNorm), _mtpHnorm!, _hc, _embedDim, _eps);

            // 3. eh_proj over concat(enorm, hnorm_c) for every stream c.
            var concat = _mtpConcat!;
            for (int c = 0; c < _hc; c++)
            {
                _mtpEnorm!.AsSpan().CopyTo(concat.AsSpan(0, _embedDim));
                _mtpHnorm!.AsSpan(c * _embedDim, _embedDim).CopyTo(concat.AsSpan(_embedDim, _embedDim));
                MatVec(head.EhProj, concat, res.AsSpan(c * _embedDim, _embedDim));
            }

            // 4. attention block with the head's own cache.
            var hcAttnNorm = AsFloatSpan(layer.HcAttnNorm);
            Qwen4ExpGatedResidual.Mix(res, hcAttnNorm, AsFloatSpan(layer.HcAttnDown), AsFloatSpan(layer.HcAttnUp),
                AsFloatSpan(layer.HcAttnInject), _mixed, _inject, _hc, _embedDim, _hcLowRank, _eps);
            ExecuteQsaMixer(head0, layer, _mixed, position, _blockOut);
            Qwen4ExpGatedResidual.Combine(res, _blockOut, _inject, _hc, _embedDim);

            // 5. FFN block (routed MoE + shared expert).
            Qwen4ExpGatedResidual.Mix(res, AsFloatSpan(layer.HcFfnNorm), AsFloatSpan(layer.HcFfnDown), AsFloatSpan(layer.HcFfnUp),
                AsFloatSpan(layer.HcFfnInject), _mixed, _inject, _hc, _embedDim, _hcLowRank, _eps);
            ExecuteMoe(layer, _mixed, _blockOut);
            Qwen4ExpGatedResidual.Combine(res, _blockOut, _inject, _hc, _embedDim);

            res.AsSpan().CopyTo(hOut);

            // 6. head output mixer (no injection), then the shared output projection.
            Qwen4ExpGatedResidual.Mix(res, AsFloatSpan(head.HcHeadNorm), AsFloatSpan(head.HcHeadDown), AsFloatSpan(head.HcHeadUp),
                ReadOnlySpan<float>.Empty, _mixed, Span<float>.Empty, _hc, _embedDim, _hcLowRank, _eps);
            ComputeLogits(_mixed, _logits);
            return _logits;
        }
        finally
        {
            _seq = previous;
        }
    }

    /// <summary>Forward pass for one token of a given sequence. Calls on different sequences may be interleaved.</summary>
    public ReadOnlySpan<float> Forward(int token, int position, Qwen4ExpSequenceState sequence)
    {
        sequence.ThrowIfDisposed();
        var previous = _seq;
        _seq = sequence;
        try { return Forward(token, position); }
        finally { _seq = previous; }
    }

    public void Dispose() { }
}
