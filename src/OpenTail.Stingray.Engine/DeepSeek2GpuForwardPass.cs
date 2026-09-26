using System.Runtime.InteropServices;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Vulkan;

namespace OpenTail.Stingray.Engine;

/// <summary>
/// DeepSeek-V2 (MLA, q_lora_rank = 0 "lite" checkpoints) on Vulkan, full offload, token by token.
/// Mirrors <see cref="ForwardPass"/>'s MLA decode (MlaComputeQkv / MlaCompactAttnOut) op for op, in
/// the same per-head layout: Q and K are <c>[rope(ropeDim), nope]</c> at the full key width, V is
/// <c>[v(vDim), zero pad]</c> at that same width so the standard attention shader applies, and the
/// attention output is compacted back to vDim per head before wo. RoPE is YaRN on the leading
/// ropeDim channels, as per-pair factors through <see cref="VulkanBackend.RoPEFactorsBatched"/>. FFN:
/// the leading dense block(s), then softmax-gated top-k MoE with the always-on shared expert; router
/// top-k runs on the CPU from a readback per layer.
/// </summary>
public sealed unsafe class DeepSeek2GpuForwardPass : IForwardPass
{
    private readonly GgufModel _model;
    private readonly VulkanBackend _gpu;
    private readonly ModelHyperparams _hp;
    private readonly int _embDim, _numHeads, _headDim, _ropeDim, _nopeDim, _vDim, _kvLora, _kvDim, _maxSeq;
    private readonly int _expertDim, _sharedDim, _denseDim;

    private sealed class Layer
    {
        public required Tensor AttnNorm, Wq, WKvA, KvANorm, WKvB, Wo, FfnNorm, KCache, VCache;
        public required DType WqT, WKvAT, WKvBT, WoT;
        // Dense FFN (leading layers) or MoE; unused slots are null.
        public Tensor? Gate, Up, Down, Router, GateExps, UpExps, DownExps, GateS, UpS, DownS;
        public DType GateT, UpT, DownT, GateExpsT, UpExpsT, DownExpsT, GateST, UpST, DownST;
    }

    private readonly Layer[] _layers;
    private readonly Tensor _outputNorm, _output, _ropeFactors;
    private readonly DType _outputT;
    private readonly float _ropeMscale;
    private readonly GgufTensorInfo _embInfo;

    private readonly Tensor _hidden, _residual, _norm, _qRaw, _q, _kvCmprPe, _kvCmpr, _kvNorm, _decomp, _k, _v;
    private readonly Tensor _attnOut, _attnCompact, _scores, _router, _gate, _up, _gateD, _upD, _gateS, _upS;
    private readonly Tensor _down, _moeOut, _sharedOut, _logits;
    private readonly float[] _embedHost, _routerHost, _logitsHost;
    private readonly int[] _expertIdx;
    private readonly float[] _expertW;

    public DeepSeek2GpuForwardPass(GgufModel model, VulkanBackend gpu, ModelHyperparams hp, int maxContextLength = 0)
    {
        if (hp.KvLoraRank <= 0) throw new ArgumentException("Not an MLA model.", nameof(hp));
        if (model.FindTensor("blk.0.attn_q_a.weight") is not null)
            throw new NotSupportedException("DeepSeek2 with q_lora_rank > 0 is not supported on Vulkan yet (only the lite layout).");
        _model = model;
        _gpu = gpu;
        _hp = hp;
        _embDim = hp.EmbeddingDim;
        _numHeads = hp.NumHeads;
        _headDim = hp.HeadDim;                            // key width: nope + rope
        _ropeDim = hp.RopeDim > 0 ? hp.RopeDim : _headDim;
        _nopeDim = _headDim - _ropeDim;
        _vDim = hp.MlaVHeadDim > 0 ? hp.MlaVHeadDim : _nopeDim;
        _kvLora = hp.KvLoraRank;
        _kvDim = _numHeads * _headDim;                    // K/V expanded per head, V padded to _headDim
        _maxSeq = maxContextLength > 0 ? Math.Min(maxContextLength, hp.ContextLength) : Math.Min(4096, hp.ContextLength);
        _expertDim = hp.ExpertIntermediateDim;
        _sharedDim = hp.SharedExpertIntermediateDim > 0 ? hp.SharedExpertIntermediateDim : _expertDim;
        _denseDim = hp.IntermediateDim;

        Console.Error.Write($"[DeepSeek2GpuForwardPass] Uploading {hp.NumLayers} layers to VRAM...");
        _layers = new Layer[hp.NumLayers];
        for (int il = 0; il < hp.NumLayers; il++)
        {
            string p = $"blk.{il}.";
            var L = new Layer
            {
                AttnNorm = Up(p + "attn_norm.weight", out _),
                Wq = Up(p + "attn_q.weight", out var wqT),
                WKvA = Up(p + "attn_kv_a_mqa.weight", out var kvaT),
                KvANorm = Up(p + "attn_kv_a_norm.weight", out _),
                WKvB = Up(p + "attn_kv_b.weight", out var kvbT),
                Wo = Up(p + "attn_output.weight", out var woT),
                FfnNorm = Up(p + "ffn_norm.weight", out _),
                KCache = _gpu.Allocate(TensorShape.D1((long)_maxSeq * _kvDim)),
                VCache = _gpu.Allocate(TensorShape.D1((long)_maxSeq * _kvDim)),
                WqT = wqT, WKvAT = kvaT, WKvBT = kvbT, WoT = woT,
            };
            if (il < hp.LeadingDenseBlockCount)
            {
                L.Gate = Up(p + "ffn_gate.weight", out L.GateT);
                L.Up = Up(p + "ffn_up.weight", out L.UpT);
                L.Down = Up(p + "ffn_down.weight", out L.DownT);
            }
            else
            {
                L.Router = Up(p + "ffn_gate_inp.weight", out _);
                L.GateExps = Up(p + "ffn_gate_exps.weight", out L.GateExpsT);
                L.UpExps = Up(p + "ffn_up_exps.weight", out L.UpExpsT);
                L.DownExps = Up(p + "ffn_down_exps.weight", out L.DownExpsT);
                if (hp.HasSharedExpert)
                {
                    L.GateS = Up(p + "ffn_gate_shexp.weight", out L.GateST);
                    L.UpS = Up(p + "ffn_up_shexp.weight", out L.UpST);
                    L.DownS = Up(p + "ffn_down_shexp.weight", out L.DownST);
                }
            }
            _layers[il] = L;
            Console.Error.Write('.');
        }
        _outputNorm = Up("output_norm.weight", out _);
        _output = model.FindTensor("output.weight") is not null ? Up("output.weight", out _outputT) : Up("token_embd.weight", out _outputT);
        _embInfo = model.FindTensor("token_embd.weight")!.Value;
        Console.Error.WriteLine(" done.");

        _ropeFactors = Upload(YarnFactors(out _ropeMscale));

        _hidden = Alloc(_embDim); _residual = Alloc(_embDim); _norm = Alloc(_embDim);
        _qRaw = Alloc(_numHeads * _headDim); _q = Alloc(_numHeads * _headDim);
        _kvCmprPe = Alloc(_kvLora + _ropeDim); _kvCmpr = Alloc(_kvLora); _kvNorm = Alloc(_kvLora);
        _decomp = Alloc(_numHeads * (_nopeDim + _vDim));
        _k = Alloc(_kvDim); _v = Alloc(_kvDim);
        _attnOut = Alloc(_numHeads * _headDim); _attnCompact = Alloc(_numHeads * _vDim);
        _scores = Alloc((long)_numHeads * _maxSeq);
        _router = Alloc(Math.Max(1, hp.NumExperts));
        _gate = Alloc(_expertDim); _up = Alloc(_expertDim);
        _gateD = Alloc(Math.Max(1, _denseDim)); _upD = Alloc(Math.Max(1, _denseDim));
        _gateS = Alloc(_sharedDim); _upS = Alloc(_sharedDim);
        _down = Alloc(_embDim); _moeOut = Alloc(_embDim); _sharedOut = Alloc(_embDim);
        _logits = Alloc(hp.VocabSize);
        _embedHost = new float[_embDim];
        _routerHost = new float[Math.Max(1, hp.NumExperts)];
        _logitsHost = new float[hp.VocabSize];
        _expertIdx = new int[Math.Max(1, hp.NumActiveExperts)];
        _expertW = new float[Math.Max(1, hp.NumActiveExperts)];
    }

    public int VocabSize => _hp.VocabSize;
    public int MaxSeqLen => _maxSeq;
    // Position-addressed KV (attention reads only [0, position]), so any rewind is exact.
    public bool SupportsPartialRewind => true;

    private Tensor Alloc(long n) => _gpu.Allocate(TensorShape.D1(n));
    private Tensor Upload(ReadOnlySpan<float> d) => _gpu.Upload(d, TensorShape.D1(d.Length));

    /// <summary>Uploads a GGUF tensor: F32 as floats, anything else as its raw quantized bytes.</summary>
    private Tensor Up(string name, out DType dtype)
    {
        var info = _model.FindTensor(name) ?? throw new InvalidOperationException($"Missing tensor: {name}");
        dtype = info.DType;
        var bytes = _model.GetTensorData(info);
        if (dtype == DType.Float32) return Upload(MemoryMarshal.Cast<byte, float>(bytes));
        var raw = new float[(bytes.Length + 3) / 4];
        bytes.CopyTo(MemoryMarshal.AsBytes(raw.AsSpan()));
        return Upload(raw);
    }

    /// <summary>
    /// ForwardPass's MLA RoPE table (BuildYarnRopeTable over ropeDim, freq scale 1/factor, ext 1,
    /// attn factor pre-divided when yarn_log_multiplier is set) as per-pair factors and a scale:
    /// the YaRN angle is pos * freq_i * (s(1 - r_i) + r_i), linear in position.
    /// </summary>
    private float[] YarnFactors(out float mscale)
    {
        int half = _ropeDim / 2;
        var f = new float[half];
        bool yarn = _hp.RopeYarnFactor > 1f;
        if (!yarn) { Array.Fill(f, 1f); mscale = 1f; return f; }
        float s = 1f / _hp.RopeYarnFactor;
        float attnFactor = _hp.RopeYarnLogMul != 0f ? 1f / (1f + 0.1f * MathF.Log(_hp.RopeYarnFactor)) : 1f;
        mscale = attnFactor * (1f + 0.1f * MathF.Log(1f / s));
        static float CorrDim(int nDims, int nCtxOrig, float nRot, float b) =>
            nDims * MathF.Log(nCtxOrig / (nRot * 2f * MathF.PI)) / (2f * MathF.Log(b));
        float lo = 0f, hi = half * 2f - 1f;
        if (_hp.RopeYarnOrigCtxLen > 0)
        {
            lo = MathF.Max(0f, MathF.Floor(CorrDim(_ropeDim, _hp.RopeYarnOrigCtxLen, 32f, _hp.RopeTheta)));
            hi = MathF.Min(_ropeDim - 1, MathF.Ceiling(CorrDim(_ropeDim, _hp.RopeYarnOrigCtxLen, 1f, _hp.RopeTheta)));
        }
        for (int i = 0; i < half; i++)
        {
            float y = (i - lo) / MathF.Max(0.001f, hi - lo);
            float r = 1f - MathF.Min(1f, MathF.Max(0f, y));
            f[i] = 1f / (s * (1f - r) + r);
        }
        return f;
    }

    private void Copy(Tensor dst, int dstOff, Tensor src, int srcOff, int count)
        => _gpu.RecordComputeCopyRegion(dst, (long)dstOff * sizeof(float), src, (long)srcOff * sizeof(float), (long)count * sizeof(float));

    public ReadOnlySpan<float> Forward(int token, int position)
    {
        if ((uint)position >= (uint)_maxSeq)
            throw new ArgumentOutOfRangeException(nameof(position), position, $"Context is {_maxSeq} tokens.");

        int bytesPerRow = (_embDim / DTypeInfo.BlockSize(_embInfo.DType)) * DTypeInfo.BytesPerBlock(_embInfo.DType);
        var embData = _model.GetTensorData(_embInfo);
        fixed (byte* src = embData)
        fixed (float* dst = _embedHost)
            SimdKernels.DequantRow(src + (long)token * bytesPerRow, dst, _embDim, _embInfo.DType);
        _gpu.UploadInto(_hidden, _embedHost);

        float qPrescale = _hp.AttentionScaleOverride != 0f ? _hp.AttentionScaleOverride * MathF.Sqrt(_headDim) : 1f;
        int decompStride = _nopeDim + _vDim;

        for (int il = 0; il < _layers.Length; il++)
        {
            var L = _layers[il];
            _gpu.BeginRecord();

            // ── MLA attention ──
            _gpu.RecordComputeCopy(_residual, _hidden);
            _gpu.RmsNorm(_norm, _hidden, L.AttnNorm, _hp.RmsNormEps);
            _gpu.RecordBarrier();
            _gpu.MatMul(_qRaw, L.Wq, _norm, L.WqT);
            _gpu.MatMul(_kvCmprPe, L.WKvA, _norm, L.WKvAT);
            _gpu.Clear(_v);
            _gpu.RecordBarrier();
            for (int h = 0; h < _numHeads; h++)
            {
                // ggml [nope, rope] -> [rope, nope]
                Copy(_q, h * _headDim, _qRaw, h * _headDim + _nopeDim, _ropeDim);
                Copy(_q, h * _headDim + _ropeDim, _qRaw, h * _headDim, _nopeDim);
            }
            Copy(_kvCmpr, 0, _kvCmprPe, 0, _kvLora);
            _gpu.RecordBarrier();
            _gpu.RmsNorm(_kvNorm, _kvCmpr, L.KvANorm, _hp.RmsNormEps);
            _gpu.RecordBarrier();
            _gpu.MatMul(_decomp, L.WKvB, _kvNorm, L.WKvBT);
            _gpu.RecordBarrier();
            for (int h = 0; h < _numHeads; h++)
            {
                Copy(_k, h * _headDim, _kvCmprPe, _kvLora, _ropeDim);             // MQA rope part, shared
                Copy(_k, h * _headDim + _ropeDim, _decomp, h * decompStride, _nopeDim);
                Copy(_v, h * _headDim, _decomp, h * decompStride + _nopeDim, _vDim);
            }
            _gpu.RecordBarrier();
            _gpu.RoPEFactorsBatched(_q, position, _headDim, _numHeads, 1, _hp.RopeTheta, _hp.IsNeoxRope, _ropeFactors, _ropeMscale, _ropeDim);
            _gpu.RoPEFactorsBatched(_k, position, _headDim, _numHeads, 1, _hp.RopeTheta, _hp.IsNeoxRope, _ropeFactors, _ropeMscale, _ropeDim);
            _gpu.RecordBarrier();
            if (qPrescale != 1f)
            {
                // The shader divides by sqrt(head_dim); DeepSeek2's scale is mscale^2 / sqrt(head_dim).
                _gpu.ScaleInPlace(_q, qPrescale);
                _gpu.RecordBarrier();
            }
            _gpu.KvAppend(_k, _v, L.KCache, L.VCache, (uint)_kvDim, (uint)position, (uint)_maxSeq);
            _gpu.RecordBarrier();
            _gpu.Attention(_q, L.KCache, L.VCache, _attnOut, _scores,
                (uint)_numHeads, (uint)_numHeads, (uint)_headDim, (uint)(position + 1), (uint)_maxSeq);
            _gpu.RecordBarrier();
            for (int h = 0; h < _numHeads; h++)
                Copy(_attnCompact, h * _vDim, _attnOut, h * _headDim, _vDim);
            _gpu.RecordBarrier();
            _gpu.MatMul(_hidden, L.Wo, _attnCompact, L.WoT);
            _gpu.RecordBarrier();
            _gpu.AddInPlace(_hidden, _residual);
            _gpu.RecordBarrier();

            // ── FFN ──
            _gpu.RecordComputeCopy(_residual, _hidden);
            _gpu.RmsNorm(_norm, _hidden, L.FfnNorm, _hp.RmsNormEps);
            _gpu.RecordBarrier();
            if (L.Gate is not null)
            {
                _gpu.MatMul(_gateD, L.Gate, _norm, L.GateT);
                _gpu.MatMul(_upD, L.Up!, _norm, L.UpT);
                _gpu.RecordBarrier();
                _gpu.SiLuMul(_gateD, _upD);
                _gpu.RecordBarrier();
                _gpu.MatMul(_hidden, L.Down!, _gateD, L.DownT);
                _gpu.RecordBarrier();
            }
            else
            {
                _gpu.MatMul(_router, L.Router!, _norm, DType.Float32);
                _gpu.EndRecordAndSubmit();
                _gpu.Download(_router, _routerHost);
                var idx = _expertIdx;
                var w = _expertW;
                SelectExperts(idx, w);

                _gpu.BeginRecord();
                _gpu.Clear(_moeOut);
                if (L.GateS is not null)
                {
                    _gpu.MatMul(_gateS, L.GateS, _norm, L.GateST);
                    _gpu.MatMul(_upS, L.UpS!, _norm, L.UpST);
                    _gpu.RecordBarrier();
                    _gpu.SiLuMul(_gateS, _upS);
                    _gpu.RecordBarrier();
                    _gpu.MatMul(_sharedOut, L.DownS!, _gateS, L.DownST);
                    _gpu.RecordBarrier();
                    _gpu.AddInPlace(_moeOut, _sharedOut);
                    _gpu.RecordBarrier();
                }
                for (int k = 0; k < idx.Length; k++)
                {
                    int e = idx[k];
                    _gpu.MatVecRowOffset(_gate, L.GateExps!, _norm, _embDim, e * _expertDim, L.GateExpsT);
                    _gpu.MatVecRowOffset(_up, L.UpExps!, _norm, _embDim, e * _expertDim, L.UpExpsT);
                    _gpu.RecordBarrier();
                    _gpu.SiLuMul(_gate, _up);
                    _gpu.RecordBarrier();
                    _gpu.MatVecRowOffset(_down, L.DownExps!, _gate, _expertDim, e * _embDim, L.DownExpsT);
                    _gpu.RecordBarrier();
                    _gpu.AddScaledInPlace(_moeOut, _down, w[k]);
                    _gpu.RecordBarrier();
                }
                _gpu.RecordComputeCopy(_hidden, _moeOut);
                _gpu.RecordBarrier();
            }
            _gpu.AddInPlace(_hidden, _residual);
            if (il < _layers.Length - 1) _gpu.EndRecordAndSubmit();
        }

        _gpu.RecordBarrier();
        _gpu.RmsNorm(_norm, _hidden, _outputNorm, _hp.RmsNormEps);
        _gpu.RecordBarrier();
        _gpu.MatMul(_logits, _output, _norm, _outputT);
        _gpu.EndRecordAndSubmit();
        _gpu.Download(_logits, _logitsHost);
        return _logitsHost;
    }

    /// <summary>ForwardPass.MoeFfn's gating: softmax (or sigmoid), top-k, optional renorm, scale.</summary>
    private void SelectExperts(Span<int> idx, Span<float> w)
    {
        int n = _hp.NumExperts, k = idx.Length;
        var p = _routerHost.AsSpan(0, n);
        if (_hp.UseSigmoidGating)
            for (int i = 0; i < n; i++) p[i] = 1f / (1f + MathF.Exp(-p[i]));
        else
        {
            float max = float.NegativeInfinity;
            for (int i = 0; i < n; i++) max = MathF.Max(max, p[i]);
            float sum = 0;
            for (int i = 0; i < n; i++) { p[i] = MathF.Exp(p[i] - max); sum += p[i]; }
            for (int i = 0; i < n; i++) p[i] /= sum;
        }
        for (int ki = 0; ki < k; ki++)
        {
            int best = 0; float bestVal = float.MinValue;
            for (int i = 0; i < n; i++)
            {
                bool taken = false;
                for (int j = 0; j < ki; j++) if (idx[j] == i) { taken = true; break; }
                if (!taken && p[i] > bestVal) { bestVal = p[i]; best = i; }
            }
            idx[ki] = best; w[ki] = bestVal;
        }
        if (_hp.NormalizeMoeTopKWeights && k > 1)
        {
            float sum = 0;
            for (int i = 0; i < k; i++) sum += w[i];
            if (sum > 0) for (int i = 0; i < k; i++) w[i] /= sum;
        }
        if (_hp.ExpertWeightsScale != 1f)
            for (int i = 0; i < k; i++) w[i] *= _hp.ExpertWeightsScale;
    }

    public ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, int startPos = 0)
    {
        ReadOnlySpan<float> last = default;
        for (int i = 0; i < tokens.Count; i++) last = Forward(tokens[i], startPos + i);
        return last;
    }

    // Position-addressed KV: attention reads [0, position] only, so rewinding needs no device work.
    public void TruncateTo(int length) { }
    public void ResetCache() { }

    public void Dispose()
    {
        foreach (var L in _layers)
            foreach (var t in new Tensor?[] { L.AttnNorm, L.Wq, L.WKvA, L.KvANorm, L.WKvB, L.Wo, L.FfnNorm, L.KCache, L.VCache,
                L.Gate, L.Up, L.Down, L.Router, L.GateExps, L.UpExps, L.DownExps, L.GateS, L.UpS, L.DownS })
                if (t is not null) _gpu.Free(t);
        foreach (var t in new[] { _outputNorm, _output, _ropeFactors, _hidden, _residual, _norm, _qRaw, _q, _kvCmprPe,
            _kvCmpr, _kvNorm, _decomp, _k, _v, _attnOut, _attnCompact, _scores, _router, _gate, _up, _gateD, _upD,
            _gateS, _upS, _down, _moeOut, _sharedOut, _logits })
            _gpu.Free(t);
    }
}
