using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Engine;

/// <summary>Hyperparameters of an <c>rwkv7</c> GGUF (llama.cpp LLM_ARCH_RWKV7).</summary>
public sealed record Rwkv7Hyperparams(int EmbedDim, int NumLayer, int HeadSize, int FfnDim, int VocabSize, float NormEps)
{
    public int NumHeads => EmbedDim / HeadSize;

    public static Rwkv7Hyperparams FromModel(GgufModel model)
    {
        int Int(string key) => Convert.ToInt32(model.Metadata[key]);
        int embed = Int("rwkv7.embedding_length");
        int vocab = model.Metadata.TryGetValue("rwkv7.vocab_size", out var v)
            ? Convert.ToInt32(v)
            : ((object[])model.Metadata["tokenizer.ggml.tokens"]).Length;
        float eps = model.Metadata.TryGetValue("rwkv7.attention.layer_norm_epsilon", out var e) ? Convert.ToSingle(e) : 1e-5f;
        return new Rwkv7Hyperparams(embed, Int("rwkv7.block_count"), Int("rwkv7.wkv.head_size"),
            Int("rwkv7.feed_forward_length"), vocab, eps);
    }
}

/// <summary>
/// CPU forward pass for RWKV-7 ("Goose", <c>general.architecture = rwkv7</c>): a recurrent model
/// with no KV cache. Each layer is a time-mix block (token shift, low-rank decay/ICLR/value-residual/
/// gate adapters, the WKV7 recurrence, per-head group norm) followed by a squared-ReLU channel-mix
/// block, each with its own one-token shift state. Port of llama.cpp's <c>llm_build_rwkv7</c> and
/// <c>llm_build_rwkv7_base</c> (src/models/rwkv7.cpp, rwkv7-base.cpp), evaluated one token at a time
/// (llama.cpp batches a ubatch through the same recurrence; the math per token is identical).
/// </summary>
public sealed unsafe class Rwkv7ForwardPass : IForwardPass
{
    private const float GroupNormEps = 64e-5f;      // rwkv7-base.cpp: ggml_norm(ctx0, cur, 64e-5f)
    private const float DecayScale = -0.606531f;    // w = exp(-0.606531 * sigmoid(w)) = exp(-e^-0.5 * σ)

    private sealed class Layer
    {
        public required DeepSeek4TensorRef AttnNormW, AttnNormB, FfnNormW, FfnNormB;
        public required DeepSeek4TensorRef LerpFused, W0, W1, W2, A0, A1, A2, KK, KA, RK, LnW, LnB;
        public required DeepSeek4TensorRef Receptance, Key, Value, Output;
        public DeepSeek4TensorRef? V0, V1, V2, G1, G2;
        public required DeepSeek4TensorRef CmLerpK, CmKey, CmValue;
    }

    private readonly Rwkv7Hyperparams _hp;
    private readonly int _d, _heads, _headSize;
    private readonly DeepSeek4TensorRef _tokEmbd, _tokNormW, _tokNormB, _outNormW, _outNormB, _output;
    private readonly Layer[] _layers;

    // Recurrent state: per layer the previous token's attn_norm and ffn_norm (token shift) and the
    // WKV7 matrix state.
    private readonly float[][] _attShift, _ffnShift, _wkvState;
    private int _length;

    // Scratch.
    private readonly float[] _x, _xa, _xn, _sx, _mix, _r, _w, _k, _v, _a, _g, _kk, _b, _vFirst, _y, _cur;
    private readonly float[] _lora, _ffnK, _logits;

    public Rwkv7ForwardPass(GgufModel model, Rwkv7Hyperparams hp)
    {
        _hp = hp;
        _d = hp.EmbedDim;
        _headSize = hp.HeadSize;
        _heads = hp.NumHeads;

        DeepSeek4TensorRef Req(string name)
        {
            var info = model.FindTensor(name) ?? throw new InvalidOperationException($"Missing required rwkv7 tensor: {name}");
            return new DeepSeek4TensorRef(name, info, model.GetTensorDataPtr(info));
        }
        DeepSeek4TensorRef? Opt(string name) =>
            model.FindTensor(name) is { } info ? new DeepSeek4TensorRef(name, info, model.GetTensorDataPtr(info)) : null;

        _tokEmbd = Req("token_embd.weight");
        _tokNormW = Req("token_embd_norm.weight");
        _tokNormB = Req("token_embd_norm.bias");
        _outNormW = Req("output_norm.weight");
        _outNormB = Req("output_norm.bias");
        _output = Req("output.weight");

        _layers = new Layer[hp.NumLayer];
        for (int il = 0; il < hp.NumLayer; il++)
        {
            string p = $"blk.{il}.";
            _layers[il] = new Layer
            {
                AttnNormW = Req(p + "attn_norm.weight"), AttnNormB = Req(p + "attn_norm.bias"),
                FfnNormW = Req(p + "attn_norm_2.weight"), FfnNormB = Req(p + "attn_norm_2.bias"),
                LerpFused = Req(p + "time_mix_lerp_fused.weight"),
                W0 = Req(p + "time_mix_w0.weight"), W1 = Req(p + "time_mix_w1.weight"), W2 = Req(p + "time_mix_w2.weight"),
                A0 = Req(p + "time_mix_a0.weight"), A1 = Req(p + "time_mix_a1.weight"), A2 = Req(p + "time_mix_a2.weight"),
                V0 = Opt(p + "time_mix_v0.weight"), V1 = Opt(p + "time_mix_v1.weight"), V2 = Opt(p + "time_mix_v2.weight"),
                G1 = Opt(p + "time_mix_g1.weight"), G2 = Opt(p + "time_mix_g2.weight"),
                KK = Req(p + "time_mix_k_k.weight"), KA = Req(p + "time_mix_k_a.weight"), RK = Req(p + "time_mix_r_k.weight"),
                LnW = Req(p + "time_mix_ln.weight"), LnB = Req(p + "time_mix_ln.bias"),
                Receptance = Req(p + "time_mix_receptance.weight"), Key = Req(p + "time_mix_key.weight"),
                Value = Req(p + "time_mix_value.weight"), Output = Req(p + "time_mix_output.weight"),
                CmLerpK = Req(p + "channel_mix_lerp_k.weight"),
                CmKey = Req(p + "channel_mix_key.weight"), CmValue = Req(p + "channel_mix_value.weight"),
            };
        }

        _attShift = new float[hp.NumLayer][];
        _ffnShift = new float[hp.NumLayer][];
        _wkvState = new float[hp.NumLayer][];
        for (int il = 0; il < hp.NumLayer; il++)
        {
            _attShift[il] = new float[_d];
            _ffnShift[il] = new float[_d];
            _wkvState[il] = new float[_heads * _headSize * _headSize];
        }

        _x = new float[_d]; _xa = new float[_d]; _xn = new float[_d]; _sx = new float[_d];
        _mix = new float[6 * _d];
        _r = new float[_d]; _w = new float[_d]; _k = new float[_d]; _v = new float[_d]; _a = new float[_d];
        _g = new float[_d]; _kk = new float[_d]; _b = new float[_d]; _vFirst = new float[_d];
        _y = new float[_d]; _cur = new float[_d];
        int maxLora = 0;
        foreach (var l in _layers)
            foreach (var t in new[] { l.W1, l.A1, l.V1, l.G1 })
                if (t is { } tt) maxLora = Math.Max(maxLora, (int)tt.Info.Dimensions[1]);
        _lora = new float[maxLora];
        _ffnK = new float[hp.FfnDim];
        _logits = new float[hp.VocabSize];
    }

    public int VocabSize => _hp.VocabSize;
    public int MaxSeqLen => int.MaxValue;

    private static void MatVec(float* output, in DeepSeek4TensorRef w, float* input)
    {
        int cols = (int)w.Info.Dimensions[0], rows = (int)w.Info.Dimensions[1];
        SimdKernels.MatVec(output, w.DataPtr, input, rows, cols, w.DType);
    }

    private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

    public ReadOnlySpan<float> Forward(int token, int position)
    {
        if (position != _length)
            throw new NotSupportedException($"Rwkv7ForwardPass is recurrent: expected position {_length}, got {position}.");

        int d = _d;
        fixed (float* x = _x, xa = _xa, xn = _xn, sx = _sx, mix = _mix, r = _r, w = _w, k = _k, v = _v, a = _a,
               g = _g, kk = _kk, b = _b, vFirst = _vFirst, y = _y, cur = _cur, lora = _lora, ffnK = _ffnK, logits = _logits)
        {
            int bytesPerRow = d / DTypeInfo.BlockSize(_tokEmbd.DType) * DTypeInfo.BytesPerBlock(_tokEmbd.DType);
            SimdKernels.DequantRow(_tokEmbd.DataPtr + (long)token * bytesPerRow, x, d, _tokEmbd.DType);
            SimdKernels.LayerNorm(x, x, (float*)_tokNormW.DataPtr, (float*)_tokNormB.DataPtr, d, _hp.NormEps);

            for (int il = 0; il < _layers.Length; il++)
            {
                var L = _layers[il];
                fixed (float* attShift = _attShift[il], ffnShift = _ffnShift[il], state = _wkvState[il])
                {
                    // ── time mix ──
                    SimdKernels.LayerNorm(xa, x, (float*)L.AttnNormW.DataPtr, (float*)L.AttnNormB.DataPtr, d, _hp.NormEps);
                    for (int i = 0; i < d; i++) sx[i] = attShift[i] - xa[i];
                    float* lerp = (float*)L.LerpFused.DataPtr;              // [d, 6]: r, w, k, v, a, g
                    for (int m = 0; m < 6; m++)
                        for (int i = 0; i < d; i++) mix[m * d + i] = sx[i] * lerp[m * d + i] + xa[i];
                    float* xr = mix, xw = mix + d, xk = mix + 2 * d, xv = mix + 3 * d, xaa = mix + 4 * d, xg = mix + 5 * d;
                    Buffer.MemoryCopy(xa, attShift, d * sizeof(float), d * sizeof(float));

                    MatVec(r, L.Receptance, xr);

                    MatVec(lora, L.W1, xw);
                    int nw = (int)L.W1.Info.Dimensions[1];
                    for (int i = 0; i < nw; i++) lora[i] = MathF.Tanh(lora[i]);
                    MatVec(w, L.W2, lora);
                    float* w0 = (float*)L.W0.DataPtr;
                    for (int i = 0; i < d; i++) w[i] = MathF.Exp(DecayScale * Sigmoid(w[i] + w0[i]));

                    MatVec(k, L.Key, xk);
                    MatVec(v, L.Value, xv);
                    if (il == 0)
                    {
                        Buffer.MemoryCopy(v, vFirst, d * sizeof(float), d * sizeof(float));
                    }
                    else
                    {
                        MatVec(lora, L.V1!.Value, xv);
                        MatVec(cur, L.V2!.Value, lora);
                        float* v0 = (float*)L.V0!.Value.DataPtr;
                        for (int i = 0; i < d; i++) v[i] += (vFirst[i] - v[i]) * Sigmoid(cur[i] + v0[i]);
                    }

                    bool hasGate = L.G1 is not null && L.G2 is not null;
                    if (hasGate)
                    {
                        MatVec(lora, L.G1!.Value, xg);
                        int ng = (int)L.G1.Value.Info.Dimensions[1];
                        for (int i = 0; i < ng; i++) lora[i] = Sigmoid(lora[i]);
                        MatVec(g, L.G2!.Value, lora);
                    }

                    MatVec(lora, L.A1, xaa);
                    MatVec(a, L.A2, lora);
                    float* a0 = (float*)L.A0.DataPtr;
                    for (int i = 0; i < d; i++) a[i] = Sigmoid(a[i] + a0[i]);

                    // kk = l2_norm(k * k_k) per head (ggml_l2_norm eps 1e-12: x / max(‖x‖, eps)).
                    float* kkW = (float*)L.KK.DataPtr, kaW = (float*)L.KA.DataPtr;
                    for (int i = 0; i < d; i++) kk[i] = k[i] * kkW[i];
                    for (int h = 0; h < _heads; h++)
                    {
                        float* kh = kk + h * _headSize;
                        float ss = 0f;
                        for (int j = 0; j < _headSize; j++) ss += kh[j] * kh[j];
                        float scale = 1f / MathF.Max(MathF.Sqrt(ss), 1e-12f);
                        for (int j = 0; j < _headSize; j++) kh[j] *= scale;
                    }
                    // k = k + (a*ka - ka), ka = k * k_a
                    for (int i = 0; i < d; i++)
                    {
                        float ka = k[i] * kaW[i];
                        k[i] = k[i] + (a[i] * ka - ka);
                    }
                    // WKV7 inputs: a' = -kk, b' = kk * a (reuse kk in place as a', b holds b').
                    for (int i = 0; i < d; i++) { b[i] = kk[i] * a[i]; kk[i] = -kk[i]; }

                    Wkv7Kernels.Step(state, r, w, k, v, kk, b, y, _heads, _headSize);

                    // Group norm over heads, then ln weight/bias.
                    float* lnW = (float*)L.LnW.DataPtr, lnB = (float*)L.LnB.DataPtr;
                    for (int h = 0; h < _heads; h++)
                    {
                        int o = h * _headSize;
                        SimdKernels.LayerNorm(y + o, y + o, lnW + o, lnB + o, _headSize, GroupNormEps);
                    }
                    // + v * Σ_head(k * r * r_k)
                    float* rkW = (float*)L.RK.DataPtr;
                    for (int h = 0; h < _heads; h++)
                    {
                        int o = h * _headSize;
                        float rk = 0f;
                        for (int j = 0; j < _headSize; j++) rk += k[o + j] * r[o + j] * rkW[o + j];
                        for (int j = 0; j < _headSize; j++) y[o + j] += v[o + j] * rk;
                    }
                    if (hasGate)
                        for (int i = 0; i < d; i++) y[i] *= g[i];
                    MatVec(cur, L.Output, y);

                    // ffn_inp = cur + x (x keeps the layer input; becomes ffn_inp)
                    for (int i = 0; i < d; i++) x[i] += cur[i];

                    // ── channel mix ──
                    SimdKernels.LayerNorm(xn, x, (float*)L.FfnNormW.DataPtr, (float*)L.FfnNormB.DataPtr, d, _hp.NormEps);
                    float* lerpK = (float*)L.CmLerpK.DataPtr;
                    for (int i = 0; i < d; i++) sx[i] = (ffnShift[i] - xn[i]) * lerpK[i] + xn[i];
                    Buffer.MemoryCopy(xn, ffnShift, d * sizeof(float), d * sizeof(float));
                    MatVec(ffnK, L.CmKey, sx);
                    for (int i = 0; i < _hp.FfnDim; i++) { float t = MathF.Max(ffnK[i], 0f); ffnK[i] = t * t; }
                    MatVec(cur, L.CmValue, ffnK);
                    for (int i = 0; i < d; i++) x[i] += cur[i];
                }
            }

            SimdKernels.LayerNorm(xn, x, (float*)_outNormW.DataPtr, (float*)_outNormB.DataPtr, d, _hp.NormEps);
            MatVec(logits, _output, xn);
        }

        _length++;
        return _logits;
    }

    public ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, int startPos = 0)
    {
        ReadOnlySpan<float> last = default;
        for (int i = 0; i < tokens.Count; i++) last = Forward(tokens[i], startPos + i);
        return last;
    }

    /// <summary>A recurrent state cannot be rewound: only a full reset or a no-op is possible.</summary>
    public void TruncateTo(int length)
    {
        if (length == 0) { ResetCache(); return; }
        if (length != _length)
            throw new NotSupportedException($"Rwkv7ForwardPass is recurrent and cannot rewind from {_length} to {length}.");
    }

    public void ResetCache()
    {
        foreach (var s in _attShift) Array.Clear(s);
        foreach (var s in _ffnShift) Array.Clear(s);
        foreach (var s in _wkvState) Array.Clear(s);
        _length = 0;
    }

    public void Dispose() { }
}
