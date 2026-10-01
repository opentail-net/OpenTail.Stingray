using System.Numerics.Tensors;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Engine;

/// <summary>
/// CPU forward pass for RWKV-6 ("Finch", <c>general.architecture = rwkv6</c>). Each layer is a
/// time-mix block (token shift, a data-dependent 5-way lerp through a two-stage low-rank adapter,
/// a low-rank decay, the WKV6 recurrence with its time_first bonus, per-head group norm, SiLU gate)
/// followed by a receptance-gated squared-ReLU channel mix; every <c>rescale_every_n_layers</c>
/// layers the residual is halved. Port of llama.cpp's <c>llm_build_rwkv6</c> /
/// <c>llm_build_rwkv6_base</c> (the QRWKV variant without time_first is not handled). The shared
/// RWKV plumbing lives in <see cref="RwkvForwardPassBase"/>.
/// </summary>
public sealed unsafe class Rwkv6ForwardPass : RwkvForwardPassBase
{
    private sealed class Layer
    {
        public required DeepSeek4TensorRef AttnNormW, AttnNormB, FfnNormW, FfnNormB;
        public required DeepSeek4TensorRef LerpX, W1, W2, Decay, DecayW1, DecayW2, First, LnW, LnB;
        public DeepSeek4TensorRef? LerpFused;
        public DeepSeek4TensorRef?[] Lerps = [];                    // w, k, v, r, g when not fused
        public required DeepSeek4TensorRef Receptance, Key, Value, Gate, Output;
        public DeepSeek4TensorRef? ReceptanceB, KeyB, ValueB;
        public required DeepSeek4TensorRef CmLerpK, CmLerpR, CmReceptance, CmKey, CmValue;
    }

    private readonly Layer[] _layers;
    private readonly int _mixDim, _decayDim, _rescaleEvery;

    // Chunk scratch, [rows × width] row-major.
    private float[] _xa = [], _xx = [], _lw = [], _slice = [], _ddd = [], _mix = [];
    private float[] _r = [], _k = [], _v = [], _g = [], _w = [], _dw = [], _y = [], _cur = [], _ffnK = [];

    public Rwkv6ForwardPass(GgufModel model) : base(model, "rwkv6")
    {
        _rescaleEvery = model.Metadata.TryGetValue("rwkv6.rescale_every_n_layers", out var re) ? Convert.ToInt32(re) : 0;
        _layers = new Layer[NumLayer];
        for (int il = 0; il < NumLayer; il++)
        {
            string p = $"blk.{il}.";
            var fused = Opt(model, p + "time_mix_lerp_fused.weight");
            _layers[il] = new Layer
            {
                AttnNormW = Req(model, p + "attn_norm.weight"), AttnNormB = Req(model, p + "attn_norm.bias"),
                FfnNormW = Req(model, p + "attn_norm_2.weight"), FfnNormB = Req(model, p + "attn_norm_2.bias"),
                LerpX = Req(model, p + "time_mix_lerp_x.weight"),
                W1 = Req(model, p + "time_mix_w1.weight"), W2 = Req(model, p + "time_mix_w2.weight"),
                LerpFused = fused,
                Lerps = fused is null
                    ? new[] { "w", "k", "v", "r", "g" }.Select(s => (DeepSeek4TensorRef?)Req(model, $"{p}time_mix_lerp_{s}.weight")).ToArray()
                    : [],
                Decay = Req(model, p + "time_mix_decay.weight"),
                DecayW1 = Req(model, p + "time_mix_decay_w1.weight"), DecayW2 = Req(model, p + "time_mix_decay_w2.weight"),
                First = Req(model, p + "time_mix_first.weight"),
                LnW = Req(model, p + "time_mix_ln.weight"), LnB = Req(model, p + "time_mix_ln.bias"),
                Receptance = Req(model, p + "time_mix_receptance.weight"), Key = Req(model, p + "time_mix_key.weight"),
                Value = Req(model, p + "time_mix_value.weight"), Gate = Req(model, p + "time_mix_gate.weight"),
                Output = Req(model, p + "time_mix_output.weight"),
                ReceptanceB = Opt(model, p + "time_mix_receptance.bias"), KeyB = Opt(model, p + "time_mix_key.bias"),
                ValueB = Opt(model, p + "time_mix_value.bias"),
                CmLerpK = Req(model, p + "channel_mix_lerp_k.weight"), CmLerpR = Req(model, p + "channel_mix_lerp_r.weight"),
                CmReceptance = Req(model, p + "channel_mix_receptance.weight"),
                CmKey = Req(model, p + "channel_mix_key.weight"), CmValue = Req(model, p + "channel_mix_value.weight"),
            };
        }
        _mixDim = (int)_layers[0].W1.Info.Dimensions[1] / 5;          // time_mix_extra_dim
        _decayDim = (int)_layers[0].DecayW1.Info.Dimensions[1];       // time_decay_extra_dim
    }

    protected override void EnsureScratch(int n)
    {
        int d = D;
        _xa = new float[n * d]; _xx = new float[n * d]; _lw = new float[n * 5 * _mixDim];
        _slice = new float[n * _mixDim]; _ddd = new float[5 * n * d]; _mix = new float[5 * n * d];
        _r = new float[n * d]; _k = new float[n * d]; _v = new float[n * d]; _g = new float[n * d];
        _w = new float[n * d]; _dw = new float[n * _decayDim]; _y = new float[n * d]; _cur = new float[n * d];
        _ffnK = new float[n * FfnDim];
    }

    private static void AddBias(float* rows, DeepSeek4TensorRef? bias, int n, int d)
    {
        if (bias is not { } b) return;
        float* bp = (float*)b.DataPtr;
        for (int t = 0; t < n; t++)
            for (int i = 0; i < d; i++) rows[t * d + i] += bp[i];
    }

    protected override void RunLayer(int il, float* x, int n)
    {
        var L = _layers[il];
        int d = D, md = _mixDim;
        fixed (float* xa = _xa, xx = _xx, lw = _lw, slice = _slice, ddd = _ddd, mix = _mix,
               r = _r, k = _k, v = _v, g = _g, w = _w, dw = _dw, y = _y, cur = _cur, ffnK = _ffnK,
               attShift = AttShift[il], ffnShift = FfnShift[il], state = WkvState[il])
        {
            // ── time mix ──
            LayerNormRows(xa, x, L.AttnNormW, L.AttnNormB, n);

            // Data-dependent lerp: xxx = tanh(W1 · lerp_x(xa)) split in 5 blocks of md, then
            // ddd_m = W2[m] · xxx_m, and x_m = (prev - xa) * (lerp_m + ddd_m) + xa for m = w, k, v, r, g.
            TokenShiftLerp(xx, xa, attShift, (float*)L.LerpX.DataPtr, null, n);
            MatMul(lw, L.W1, xx, n);
            TensorPrimitives.Tanh(Sp(lw, n * 5 * md), Sp(lw, n * 5 * md));
            var w2 = L.W2;                                          // [md, d, 5]
            long w2BlockBytes = (long)d * md / DTypeInfo.BlockSize(w2.DType) * DTypeInfo.BytesPerBlock(w2.DType);
            for (int m = 0; m < 5; m++)
            {
                for (int t = 0; t < n; t++)
                    Buffer.MemoryCopy(lw + (t * 5 + m) * md, slice + t * md, md * sizeof(float), md * sizeof(float));
                float* dst = ddd + (long)m * n * d;
                if (n == 1) SimdKernels.MatVec(dst, w2.DataPtr + m * w2BlockBytes, slice, d, md, w2.DType);
                else SimdKernels.MatMulBatched(dst, w2.DataPtr + m * w2BlockBytes, slice, n, d, md, w2.DType, allowQ8: true);
                float* lerpM = L.LerpFused is { } lf ? (float*)lf.DataPtr + m * d : (float*)L.Lerps[m]!.Value.DataPtr;
                TokenShiftLerp(mix + (long)m * n * d, xa, attShift, lerpM, dst, n);
            }
            SaveShift(xa, n, AttShift[il]);
            float* xw = mix, xk = mix + n * d, xv = mix + 2 * n * d, xr = mix + 3 * n * d, xg = mix + 4 * n * d;

            MatMul(r, L.Receptance, xr, n); AddBias(r, L.ReceptanceB, n, d);
            MatMul(k, L.Key, xk, n); AddBias(k, L.KeyB, n, d);
            MatMul(v, L.Value, xv, n); AddBias(v, L.ValueB, n, d);
            MatMul(g, L.Gate, xg, n);
            TensorPrimitives.Sigmoid(Sp(g, n * d), Sp(cur, n * d));                     // SiLU: g * sigmoid(g)
            TensorPrimitives.Multiply(Sp(g, n * d), Sp(cur, n * d), Sp(g, n * d));

            // w = exp(-exp(decay_w2 · tanh(decay_w1 · xw) + decay))
            MatMul(dw, L.DecayW1, xw, n);
            TensorPrimitives.Tanh(Sp(dw, n * _decayDim), Sp(dw, n * _decayDim));
            MatMul(w, L.DecayW2, dw, n);
            float* decay = (float*)L.Decay.DataPtr;
            AddRowVector(w, decay, n);
            TensorPrimitives.Exp(Sp(w, n * d), Sp(w, n * d));
            TensorPrimitives.Negate(Sp(w, n * d), Sp(w, n * d));
            TensorPrimitives.Exp(Sp(w, n * d), Sp(w, n * d));

            // Head-local from here to the output projection: heads in parallel, tokens in order.
            float* u = (float*)L.First.DataPtr, lnW = (float*)L.LnW.DataPtr, lnB = (float*)L.LnB.DataPtr;
            float* pr = r, pw = w, pk = k, pv = v, py = y, pg = g, pState = state;
            int hs = HeadSize;
            Parallel.For(0, Heads, h =>
            {
                int o = h * hs;
                float* sh = pState + (long)h * hs * hs;
                for (int t = 0; t < n; t++)
                {
                    int to = t * d + o;
                    float* yt = py + to;
                    WkvKernels.Wkv6StepHead(sh, pr + to, pw + to, pk + to, pv + to, u + o, yt, hs);
                    SimdKernels.LayerNorm(yt, yt, lnW + o, lnB + o, hs, GroupNormEps);
                    for (int j = 0; j < hs; j++) yt[j] *= pg[to + j];
                }
            });
            MatMul(cur, L.Output, y, n);
            TensorPrimitives.Add(Sp(x, n * d), Sp(cur, n * d), Sp(x, n * d));      // x is now ffn_inp

            // ── channel mix ── (xa reused as ffn_norm; xx and r reused as the k / r lerps)
            LayerNormRows(xa, x, L.FfnNormW, L.FfnNormB, n);
            TokenShiftLerp(xx, xa, ffnShift, (float*)L.CmLerpK.DataPtr, null, n);
            TokenShiftLerp(y, xa, ffnShift, (float*)L.CmLerpR.DataPtr, null, n);
            SaveShift(xa, n, FfnShift[il]);
            MatMul(r, L.CmReceptance, y, n);
            MatMul(ffnK, L.CmKey, xx, n);
            TensorPrimitives.Max(Sp(ffnK, n * FfnDim), 0f, Sp(ffnK, n * FfnDim));
            TensorPrimitives.Multiply(Sp(ffnK, n * FfnDim), Sp(ffnK, n * FfnDim), Sp(ffnK, n * FfnDim));
            MatMul(cur, L.CmValue, ffnK, n);
            TensorPrimitives.Sigmoid(Sp(r, n * d), Sp(r, n * d));
            TensorPrimitives.MultiplyAdd(Sp(r, n * d), Sp(cur, n * d), Sp(x, n * d), Sp(x, n * d));

            if (_rescaleEvery != 0 && (il + 1) % _rescaleEvery == 0)
                TensorPrimitives.Multiply(Sp(x, n * d), 0.5f, Sp(x, n * d));
        }
    }
}
