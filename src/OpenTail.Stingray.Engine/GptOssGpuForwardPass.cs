using System.Runtime.InteropServices;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Vulkan;

namespace OpenTail.Stingray.Engine;

/// <summary>
/// gpt-oss on Vulkan, full offload, token by token — the GPU twin of <see cref="GptOssForwardPass"/>
/// (same math, op for op): biased Q8_0 QKVO, YaRN NEOX RoPE (expressed exactly as per-pair
/// frequency factors + a cos/sin scale through <see cref="VulkanBackend.RoPEFactorsBatched"/>),
/// attention with per-head sinks and the alternating sliding window
/// (<see cref="VulkanBackend.AttentionSinks"/>), and the MoE with raw-MXFP4 experts
/// (<see cref="VulkanBackend.MatVecMxfp4"/>), per-expert biases and the OAI SwiGLU. Router top-k
/// selection runs on the CPU from a 32-float readback per layer, like GpuForwardPass's MoE.
/// </summary>
public sealed unsafe class GptOssGpuForwardPass : IForwardPass
{
    private readonly GgufModel _model;
    private readonly VulkanBackend _gpu;
    private readonly GptOssHyperparams _hp;
    private readonly GptOssTensorSet _tensors;
    private readonly int _embDim, _numHeads, _numKv, _headDim, _kvDim, _qDim, _ffnDim, _maxSeq;

    private sealed class Layer
    {
        public required Tensor AttnNorm, PostNorm, Wq, Wk, Wv, Wo, Bq, Bk, Bv, Bo, Sinks;
        public required Tensor Router, RouterB, GateExps, UpExps, DownExps, GateB, UpB, DownB;
        public required Tensor KCache, VCache;
    }

    private readonly Layer[] _layers;
    private readonly Tensor _outputNorm, _output;
    private readonly Tensor _ropeFactorsFull, _ropeFactorsSwa;
    private readonly float _ropeMscale;

    // Scratch
    private readonly Tensor _hidden, _residual, _norm, _q, _k, _v, _attnOut, _scores;
    private readonly Tensor _routerLogits, _gate, _up, _down, _moeOut, _biasRow, _logits;
    private readonly float[] _routerHost, _logitsHost, _embedHost;
    private readonly int[] _expertIdx;
    private readonly float[] _expertW;

    public GptOssGpuForwardPass(GgufModel model, VulkanBackend gpu, GptOssHyperparams hp, int maxContextLength = 4096)
    {
        _model = model;
        _gpu = gpu;
        _hp = hp;
        _tensors = GptOssTensorSet.Load(model, hp);
        _embDim = hp.EmbedDim;
        _numHeads = hp.NumHeads;
        _numKv = hp.NumHeadsKv;
        _headDim = hp.HeadDim;
        _kvDim = _numKv * _headDim;
        _qDim = _numHeads * _headDim;
        _ffnDim = hp.ExpertFeedForwardLength;
        // 0 / negative = caller's "pick a default" (the CLI passes 0 without -c).
        _maxSeq = maxContextLength > 0 ? maxContextLength : 4096;

        Console.Error.Write($"[GptOssGpuForwardPass] Uploading {hp.NumLayer} layers to VRAM...");
        _layers = new Layer[hp.NumLayer];
        for (int il = 0; il < hp.NumLayer; il++)
        {
            var t = _tensors.Layers[il];
            _layers[il] = new Layer
            {
                AttnNorm = Upload(t.AttnNorm!.Value), PostNorm = Upload(t.AttnPostNorm!.Value),
                Wq = Upload(t.Wq!.Value), Wk = Upload(t.Wk!.Value), Wv = Upload(t.Wv!.Value), Wo = Upload(t.Wo!.Value),
                Bq = Upload(t.WqB!.Value), Bk = Upload(t.WkB!.Value), Bv = Upload(t.WvB!.Value), Bo = Upload(t.WoB!.Value),
                Sinks = Upload(t.AttnSinks!.Value),
                Router = Upload(t.FfnGateInp!.Value), RouterB = Upload(t.FfnGateInpB!.Value),
                GateExps = Upload(t.FfnGateExps!.Value), UpExps = Upload(t.FfnUpExps!.Value), DownExps = Upload(t.FfnDownExps!.Value),
                GateB = Upload(t.FfnGateExpsB!.Value), UpB = Upload(t.FfnUpExpsB!.Value), DownB = Upload(t.FfnDownExpsB!.Value),
                KCache = _gpu.Allocate(TensorShape.D1((long)_maxSeq * _kvDim)),
                VCache = _gpu.Allocate(TensorShape.D1((long)_maxSeq * _kvDim)),
            };
            Console.Error.Write('.');
        }
        _outputNorm = Upload(_tensors.OutputNorm);
        _output = Upload(_tensors.Output);
        Console.Error.WriteLine(" done.");

        // YaRN, from SimdKernels.BuildYarnRopeRow: the final angle is pos * freq_i * (s(1-r_i) + r_i)
        // with freq scale s and ramp r_i — linear in position, so it is exactly a per-pair frequency
        // factor 1/(s(1-r_i) + r_i) and a constant cos/sin scale.
        _ropeFactorsFull = Upload(YarnFactors(hp.RopeFreqBase, out _ropeMscale));
        _ropeFactorsSwa = Upload(YarnFactors(hp.RopeFreqBaseSwa, out _));

        _hidden = Alloc(_embDim); _residual = Alloc(_embDim); _norm = Alloc(_embDim);
        _q = Alloc(_qDim); _k = Alloc(_kvDim); _v = Alloc(_kvDim); _attnOut = Alloc(_qDim);
        _scores = Alloc((long)_numHeads * _maxSeq);
        _routerLogits = Alloc(hp.NumExperts);
        _gate = Alloc(_ffnDim); _up = Alloc(_ffnDim); _down = Alloc(_embDim); _moeOut = Alloc(_embDim);
        _biasRow = Alloc(Math.Max(_ffnDim, _embDim));
        _logits = Alloc(hp.VocabSize);
        _routerHost = new float[hp.NumExperts];
        _logitsHost = new float[hp.VocabSize];
        _embedHost = new float[_embDim];
        _expertIdx = new int[hp.NumExpertsUsed];
        _expertW = new float[hp.NumExpertsUsed];
    }

    public int VocabSize => _hp.VocabSize;
    public int MaxSeqLen => _maxSeq;

    private Tensor Alloc(long n) => _gpu.Allocate(TensorShape.D1(n));

    private Tensor Upload(ReadOnlySpan<float> data) => _gpu.Upload(data, TensorShape.D1(data.Length));

    private Tensor Upload(DeepSeek4TensorRef t)
    {
        long bytes = t.Info.ByteSize;
        var span = new ReadOnlySpan<byte>(t.DataPtr, checked((int)bytes));
        if (t.DType == DType.Float32)
            return Upload(MemoryMarshal.Cast<byte, float>(span));
        // Raw quantized bytes (Q8_0 / MXFP4), stored in a float buffer the shaders read as uints.
        var raw = new float[(bytes + 3) / 4];
        span.CopyTo(MemoryMarshal.AsBytes(raw.AsSpan()));
        return Upload(raw);
    }

    private float[] YarnFactors(float theta, out float mscale)
    {
        int half = _headDim / 2;
        var f = new float[half];
        float s = 1f / _hp.RopeScalingFactor;
        bool yarn = _hp.RopeScalingFactor > 1f;
        mscale = yarn ? 1f + 0.1f * MathF.Log(1f / s) : 1f;
        static float CorrDim(int nDims, int nCtxOrig, float nRot, float b) =>
            nDims * MathF.Log(nCtxOrig / (nRot * 2f * MathF.PI)) / (2f * MathF.Log(b));
        float lo = 0f, hi = half * 2f - 1f;
        if (_hp.RopeOrigContext > 0)
        {
            lo = MathF.Max(0f, MathF.Floor(CorrDim(_headDim, _hp.RopeOrigContext, _hp.YarnBetaFast, theta)));
            hi = MathF.Min(_headDim - 1, MathF.Ceiling(CorrDim(_headDim, _hp.RopeOrigContext, _hp.YarnBetaSlow, theta)));
        }
        for (int i = 0; i < half; i++)
        {
            float r = 0f;
            if (yarn)
            {
                float y = (i - lo) / MathF.Max(0.001f, hi - lo);
                r = 1f - MathF.Min(1f, MathF.Max(0f, y));
            }
            f[i] = 1f / (s * (1f - r) + r);
        }
        return f;
    }

    public ReadOnlySpan<float> Forward(int token, int position)
    {
        if ((uint)position >= (uint)_maxSeq)
            throw new ArgumentOutOfRangeException(nameof(position), position, $"Context is {_maxSeq} tokens.");

        // Embedding row (Q8_0) dequantized on the CPU; EmbedLookup has no Q8_0 path.
        var info = _tensors.TokEmbd.Info;
        int bytesPerRow = (_embDim / DTypeInfo.BlockSize(info.DType)) * DTypeInfo.BytesPerBlock(info.DType);
        fixed (float* dst = _embedHost)
            SimdKernels.DequantRow(_tensors.TokEmbd.DataPtr + (long)token * bytesPerRow, dst, _embDim, info.DType);
        _gpu.UploadInto(_hidden, _embedHost);

        int seqLen = position + 1;
        for (int il = 0; il < _layers.Length; il++)
        {
            var L = _layers[il];
            bool swa = _hp.IsSwaLayer(il);
            _gpu.BeginRecord();

            // ── attention ──
            _gpu.RecordComputeCopy(_residual, _hidden);
            _gpu.RmsNorm(_norm, _hidden, L.AttnNorm, _hp.RmsNormEps);
            _gpu.RecordBarrier();
            _gpu.MatMul(_q, L.Wq, _norm, DType.Q8_0);
            _gpu.MatMul(_k, L.Wk, _norm, DType.Q8_0);
            _gpu.MatMul(_v, L.Wv, _norm, DType.Q8_0);
            _gpu.RecordBarrier();
            _gpu.AddInPlace(_q, L.Bq);
            _gpu.AddInPlace(_k, L.Bk);
            _gpu.AddInPlace(_v, L.Bv);
            _gpu.RecordBarrier();
            float theta = swa ? _hp.RopeFreqBaseSwa : _hp.RopeFreqBase;
            var factors = swa ? _ropeFactorsSwa : _ropeFactorsFull;
            _gpu.RoPEFactorsBatched(_q, position, _headDim, _numHeads, 1, theta, neox: true, factors, _ropeMscale);
            _gpu.RoPEFactorsBatched(_k, position, _headDim, _numKv, 1, theta, neox: true, factors, _ropeMscale);
            _gpu.RecordBarrier();
            _gpu.KvAppend(_k, _v, L.KCache, L.VCache, (uint)_kvDim, (uint)position, (uint)_maxSeq);
            _gpu.RecordBarrier();
            uint window = swa && _hp.SlidingWindow > 0 ? (uint)_hp.SlidingWindow : 0u;
            _gpu.AttentionSinks(_q, L.KCache, L.VCache, _attnOut, _scores, L.Sinks,
                (uint)_numHeads, (uint)_numKv, (uint)_headDim, (uint)seqLen, (uint)_maxSeq, window);
            _gpu.RecordBarrier();
            _gpu.MatMul(_hidden, L.Wo, _attnOut, DType.Q8_0);
            _gpu.RecordBarrier();
            _gpu.AddInPlace(_hidden, L.Bo);
            _gpu.RecordBarrier();
            _gpu.AddInPlace(_hidden, _residual);
            _gpu.RecordBarrier();

            // ── MoE: router on GPU, top-k on CPU ──
            _gpu.RecordComputeCopy(_residual, _hidden);
            _gpu.RmsNorm(_norm, _hidden, L.PostNorm, _hp.RmsNormEps);
            _gpu.RecordBarrier();
            _gpu.MatMul(_routerLogits, L.Router, _norm, DType.Float32);
            _gpu.RecordBarrier();
            _gpu.AddInPlace(_routerLogits, L.RouterB);
            _gpu.EndRecordAndSubmit();
            _gpu.Download(_routerLogits, _routerHost);
            GptOssGraph.SelectThenSoftmaxGate(_routerHost, _hp.NumExpertsUsed, _expertIdx, _expertW);

            _gpu.BeginRecord();
            _gpu.Clear(_moeOut);
            for (int k = 0; k < _expertIdx.Length; k++)
            {
                int e = _expertIdx[k];
                _gpu.MatVecMxfp4(_gate, L.GateExps, _norm, _embDim, e * _ffnDim);
                _gpu.MatVecMxfp4(_up, L.UpExps, _norm, _embDim, e * _ffnDim);
                _gpu.RecordBarrier();
                AddBiasSlice(_gate, L.GateB, e, _ffnDim);
                AddBiasSlice(_up, L.UpB, e, _ffnDim);
                _gpu.SwigluOai(_gate, _up, _ffnDim);
                _gpu.RecordBarrier();
                _gpu.MatVecMxfp4(_down, L.DownExps, _gate, _ffnDim, e * _embDim);
                _gpu.RecordBarrier();
                AddBiasSlice(_down, L.DownB, e, _embDim);
                _gpu.AddScaledInPlace(_moeOut, _down, _expertW[k]);
                _gpu.RecordBarrier();
            }
            _gpu.RecordComputeCopy(_hidden, _moeOut);
            _gpu.RecordBarrier();
            _gpu.AddInPlace(_hidden, _residual);
            if (il < _layers.Length - 1) _gpu.EndRecordAndSubmit();
        }

        _gpu.RecordBarrier();
        _gpu.RmsNorm(_norm, _hidden, _outputNorm, _hp.RmsNormEps);
        _gpu.RecordBarrier();
        _gpu.MatMul(_logits, _output, _norm, _tensors.Output.DType);
        _gpu.EndRecordAndSubmit();
        _gpu.Download(_logits, _logitsHost);
        return _logitsHost;
    }

    /// <summary>x += bias[e*dim .. +dim) — one expert's row of a [experts][dim] bias tensor.</summary>
    private void AddBiasSlice(Tensor x, Tensor bias, int expert, int dim)
    {
        _gpu.RecordComputeCopyRegion(_biasRow, 0, bias, (long)expert * dim * sizeof(float), (long)dim * sizeof(float));
        _gpu.RecordBarrier();
        _gpu.AddRowBroadcastInPlace(x, _biasRow, 1, dim);
        _gpu.RecordBarrier();
    }

    public ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, int startPos = 0)
    {
        ReadOnlySpan<float> last = default;
        for (int i = 0; i < tokens.Count; i++) last = Forward(tokens[i], startPos + i);
        return last;
    }

    // The KV cache is position-addressed; attention reads only [0, position], so rewinding needs
    // no device work.
    public void TruncateTo(int length) { }

    public void ResetCache() { }

    public void Dispose()
    {
        foreach (var L in _layers)
            foreach (var t in new[] { L.AttnNorm, L.PostNorm, L.Wq, L.Wk, L.Wv, L.Wo, L.Bq, L.Bk, L.Bv, L.Bo, L.Sinks,
                L.Router, L.RouterB, L.GateExps, L.UpExps, L.DownExps, L.GateB, L.UpB, L.DownB, L.KCache, L.VCache })
                _gpu.Free(t);
        foreach (var t in new[] { _outputNorm, _output, _ropeFactorsFull, _ropeFactorsSwa, _hidden, _residual, _norm,
            _q, _k, _v, _attnOut, _scores, _routerLogits, _gate, _up, _down, _moeOut, _biasRow, _logits })
            _gpu.Free(t);
    }
}
