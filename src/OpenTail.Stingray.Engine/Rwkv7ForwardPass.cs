using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Engine;

/// <summary>
/// CPU forward pass for RWKV-7 ("Goose", <c>general.architecture = rwkv7</c>). Each layer is a
/// time-mix block (token shift, a fused 6-way lerp, low-rank decay/ICLR/value-residual/gate
/// adapters, the WKV7 recurrence, per-head group norm) followed by a squared-ReLU channel mix.
/// Port of llama.cpp's <c>llm_build_rwkv7</c> / <c>llm_build_rwkv7_base</c>; the shared RWKV
/// plumbing lives in <see cref="RwkvForwardPassBase"/>.
/// </summary>
public sealed unsafe class Rwkv7ForwardPass : RwkvForwardPassBase
{
    private const float DecayScale = -0.606531f;    // w = exp(-0.606531 * sigmoid(w)) = exp(-e^-0.5 * σ)

    private sealed class Layer
    {
        public required DeepSeek4TensorRef AttnNormW, AttnNormB, FfnNormW, FfnNormB;
        public required DeepSeek4TensorRef LerpFused, W0, W1, W2, A0, A1, A2, KK, KA, RK, LnW, LnB;
        public required DeepSeek4TensorRef Receptance, Key, Value, Output;
        public DeepSeek4TensorRef? V0, V1, V2, G1, G2;
        public required DeepSeek4TensorRef CmLerpK, CmKey, CmValue;
    }

    private readonly Layer[] _layers;
    private readonly int _maxLora;

    // Chunk scratch, [rows × width] row-major.
    private float[] _xa = [], _sx = [], _mix = [], _r = [], _w = [], _k = [], _v = [], _a = [], _g = [];
    private float[] _kk = [], _b = [], _vFirst = [], _y = [], _cur = [], _lora = [], _ffnK = [];

    public Rwkv7ForwardPass(GgufModel model) : base(model, "rwkv7")
    {
        _layers = new Layer[NumLayer];
        for (int il = 0; il < NumLayer; il++)
        {
            string p = $"blk.{il}.";
            _layers[il] = new Layer
            {
                AttnNormW = Req(model, p + "attn_norm.weight"), AttnNormB = Req(model, p + "attn_norm.bias"),
                FfnNormW = Req(model, p + "attn_norm_2.weight"), FfnNormB = Req(model, p + "attn_norm_2.bias"),
                LerpFused = Req(model, p + "time_mix_lerp_fused.weight"),
                W0 = Req(model, p + "time_mix_w0.weight"), W1 = Req(model, p + "time_mix_w1.weight"), W2 = Req(model, p + "time_mix_w2.weight"),
                A0 = Req(model, p + "time_mix_a0.weight"), A1 = Req(model, p + "time_mix_a1.weight"), A2 = Req(model, p + "time_mix_a2.weight"),
                V0 = Opt(model, p + "time_mix_v0.weight"), V1 = Opt(model, p + "time_mix_v1.weight"), V2 = Opt(model, p + "time_mix_v2.weight"),
                G1 = Opt(model, p + "time_mix_g1.weight"), G2 = Opt(model, p + "time_mix_g2.weight"),
                KK = Req(model, p + "time_mix_k_k.weight"), KA = Req(model, p + "time_mix_k_a.weight"), RK = Req(model, p + "time_mix_r_k.weight"),
                LnW = Req(model, p + "time_mix_ln.weight"), LnB = Req(model, p + "time_mix_ln.bias"),
                Receptance = Req(model, p + "time_mix_receptance.weight"), Key = Req(model, p + "time_mix_key.weight"),
                Value = Req(model, p + "time_mix_value.weight"), Output = Req(model, p + "time_mix_output.weight"),
                CmLerpK = Req(model, p + "channel_mix_lerp_k.weight"),
                CmKey = Req(model, p + "channel_mix_key.weight"), CmValue = Req(model, p + "channel_mix_value.weight"),
            };
        }
        foreach (var l in _layers)
            foreach (var t in new[] { l.W1, l.A1, l.V1, l.G1 })
                if (t is { } tt) _maxLora = Math.Max(_maxLora, (int)tt.Info.Dimensions[1]);
    }

    protected override void EnsureScratch(int n)
    {
        int d = D;
        _xa = new float[n * d]; _sx = new float[n * d]; _mix = new float[6 * n * d];
        _r = new float[n * d]; _w = new float[n * d]; _k = new float[n * d]; _v = new float[n * d];
        _a = new float[n * d]; _g = new float[n * d]; _kk = new float[n * d]; _b = new float[n * d];
        _vFirst = new float[n * d]; _y = new float[n * d]; _cur = new float[n * d];
        _lora = new float[n * _maxLora]; _ffnK = new float[n * FfnDim];
    }

    protected override void RunLayer(int il, float* x, int n)
    {
        var L = _layers[il];
        int d = D, hs = HeadSize;
        fixed (float* xa = _xa, sx = _sx, mix = _mix, r = _r, w = _w, k = _k, v = _v, a = _a, g = _g,
               kk = _kk, b = _b, vFirst = _vFirst, y = _y, cur = _cur, lora = _lora, ffnK = _ffnK,
               attShift = AttShift[il], ffnShift = FfnShift[il], state = WkvState[il])
        {
            // ── time mix ── six lerps (r, w, k, v, a, g); block m of `mix` holds n rows for lerp m.
            LayerNormRows(xa, x, L.AttnNormW, L.AttnNormB, n);
            float* lerp = (float*)L.LerpFused.DataPtr;              // [d, 6]
            for (int m = 0; m < 6; m++)
                TokenShiftLerp(mix + (long)m * n * d, xa, attShift, lerp + m * d, null, n);
            SaveShift(xa, n, AttShift[il]);
            float* xr = mix, xw = mix + n * d, xk = mix + 2 * n * d, xv = mix + 3 * n * d,
                   xaa = mix + 4 * n * d, xg = mix + 5 * n * d;

            MatMul(r, L.Receptance, xr, n);

            int nw = (int)L.W1.Info.Dimensions[1];
            MatMul(lora, L.W1, xw, n);
            for (int i = 0; i < n * nw; i++) lora[i] = MathF.Tanh(lora[i]);
            MatMul(w, L.W2, lora, n);
            float* w0 = (float*)L.W0.DataPtr;
            for (int t = 0; t < n; t++)
                for (int i = 0; i < d; i++) w[t * d + i] = MathF.Exp(DecayScale * Sigmoid(w[t * d + i] + w0[i]));

            MatMul(k, L.Key, xk, n);
            MatMul(v, L.Value, xv, n);
            if (il == 0)
            {
                Buffer.MemoryCopy(v, vFirst, (long)n * d * sizeof(float), (long)n * d * sizeof(float));
            }
            else
            {
                MatMul(lora, L.V1!.Value, xv, n);
                MatMul(cur, L.V2!.Value, lora, n);
                float* v0 = (float*)L.V0!.Value.DataPtr;
                for (int t = 0; t < n; t++)
                    for (int i = 0; i < d; i++)
                    {
                        int o = t * d + i;
                        v[o] += (vFirst[o] - v[o]) * Sigmoid(cur[o] + v0[i]);
                    }
            }

            bool hasGate = L.G1 is not null && L.G2 is not null;
            if (hasGate)
            {
                int ng = (int)L.G1!.Value.Info.Dimensions[1];
                MatMul(lora, L.G1.Value, xg, n);
                for (int i = 0; i < n * ng; i++) lora[i] = Sigmoid(lora[i]);
                MatMul(g, L.G2!.Value, lora, n);
            }

            MatMul(lora, L.A1, xaa, n);
            MatMul(a, L.A2, lora, n);
            float* a0 = (float*)L.A0.DataPtr;
            for (int t = 0; t < n; t++)
                for (int i = 0; i < d; i++) a[t * d + i] = Sigmoid(a[t * d + i] + a0[i]);

            float* kkW = (float*)L.KK.DataPtr, kaW = (float*)L.KA.DataPtr, rkW = (float*)L.RK.DataPtr;
            for (int t = 0; t < n; t++)
            {
                float* kt = k + t * d, at = a + t * d, kkt = kk + t * d, bt = b + t * d;
                // kk = l2_norm(k * k_k) per head (ggml_l2_norm eps 1e-12: x / max(‖x‖, eps)).
                for (int i = 0; i < d; i++) kkt[i] = kt[i] * kkW[i];
                for (int h = 0; h < Heads; h++)
                {
                    float* kh = kkt + h * hs;
                    float ss = 0f;
                    for (int j = 0; j < hs; j++) ss += kh[j] * kh[j];
                    float scale = 1f / MathF.Max(MathF.Sqrt(ss), 1e-12f);
                    for (int j = 0; j < hs; j++) kh[j] *= scale;
                }
                // k = k + (a*ka - ka), ka = k * k_a; WKV7 a' = -kk, b' = kk * a.
                for (int i = 0; i < d; i++)
                {
                    float ka = kt[i] * kaW[i];
                    kt[i] = kt[i] + (at[i] * ka - ka);
                    bt[i] = kkt[i] * at[i];
                    kkt[i] = -kkt[i];
                }

                float* yt = y + t * d, rt = r + t * d, vt = v + t * d;
                WkvKernels.Wkv7Step(state, rt, w + t * d, kt, vt, kkt, bt, yt, Heads, hs);

                // Group norm, then + v · Σ_head(k · r · r_k).
                GroupNorm(yt, L.LnW, L.LnB);
                for (int h = 0; h < Heads; h++)
                {
                    int o = h * hs;
                    float rk = 0f;
                    for (int j = 0; j < hs; j++) rk += kt[o + j] * rt[o + j] * rkW[o + j];
                    for (int j = 0; j < hs; j++) yt[o + j] += vt[o + j] * rk;
                }
                if (hasGate)
                    for (int i = 0; i < d; i++) yt[i] *= g[t * d + i];
            }
            MatMul(cur, L.Output, y, n);
            for (int i = 0; i < n * d; i++) x[i] += cur[i];      // x is now ffn_inp

            // ── channel mix ── (xa reused as ffn_norm)
            LayerNormRows(xa, x, L.FfnNormW, L.FfnNormB, n);
            TokenShiftLerp(sx, xa, ffnShift, (float*)L.CmLerpK.DataPtr, null, n);
            SaveShift(xa, n, FfnShift[il]);
            MatMul(ffnK, L.CmKey, sx, n);
            for (int i = 0; i < n * FfnDim; i++) { float t = MathF.Max(ffnK[i], 0f); ffnK[i] = t * t; }
            MatMul(cur, L.CmValue, ffnK, n);
            for (int i = 0; i < n * d; i++) x[i] += cur[i];
        }
    }
}
