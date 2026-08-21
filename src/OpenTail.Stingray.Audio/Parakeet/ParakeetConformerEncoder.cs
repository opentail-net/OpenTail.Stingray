using System;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Audio.Parakeet;

/// <summary>
/// NVIDIA NeMo Parakeet/Canary-CTC FastConformer acoustic encoder: dw_striding 8x subsampling
/// front-end, 24x Conformer blocks (macaron FFN, Transformer-XL rel-pos self-attention with
/// untied u/v biases, GLU depthwise-conv module), CTC head. Ported directly from
/// `examples/crispasr/src/core/fastconformer.h` (`build_pre_encode`/`build_block`) and
/// `canary_ctc.cpp` -- see docs/audio-review-progress.md's Parakeet section for the full
/// per-tensor derivation this was written against. NOT YET golden-verified against a real
/// oracle run (no crispasr build/dump done yet this iteration) -- treat as structurally
/// complete but numerically unconfirmed until that happens.
/// </summary>
public static class ParakeetConformerEncoder
{
    /// <summary>
    /// mel is channel-first-by-frame [T, NMels] (row-major, frame-major -- mel[t*NMels+f]),
    /// as produced by <see cref="ParakeetMelExtractor"/>. Returns per-frame encoder hidden
    /// states [TEnc][HiddenDim] and CTC logits [TEnc][VocabSize+1].
    /// </summary>
    public static (float[][] Hidden, float[][] CtcLogits, int TEnc) Forward(ParakeetWeights w, float[] mel, int tMel)
    {
        var (sub, tEnc) = Subsample(w, mel, tMel);

        var posEnc = SinusoidalRelPosTable(tEnc, w.HiddenDim);

        var x = sub;
        foreach (var layer in w.Layers)
            x = ConformerBlock(w, layer, x, posEnc, tEnc);

        var logits = new float[tEnc][];
        for (int t = 0; t < tEnc; t++)
            logits[t] = Linear(x[t], w.CtcWeight, w.CtcBias, w.HiddenDim, w.VocabSize + 1);

        return (x, logits, tEnc);
    }

    // -----------------------------------------------------------------
    // Subsampling: dw_striding, 3 stages, 8x time downsampling (and freq: 80 -> 10).
    // Image layout throughout: IDX(c,h,w) = c*H*W + h*W + w, with W=freq axis (fast, ne0 in
    // ggml), H=time axis (ne1). Weight layout (raw GGUF storage order, ne=[KW,KH,Cin,Cout]):
    // idx = kw + kh*K + cin*K*K + cout*K*K*Cin.
    // -----------------------------------------------------------------
    private static (float[][] X, int TEnc) Subsample(ParakeetWeights w, float[] mel, int tMel)
    {
        int nMels = w.NMels;
        int c = w.SubsampleChannels;

        // mel input as a 1-channel image: IDX(0,h=t,w=f) = t*nMels + f -- already matches mel's
        // frame-major [T,NMels] layout directly (cin=1 collapses the channel term to zero).
        var stage0 = Conv2dFull(mel, cin: 1, hin: tMel, win: nMels, w.PreConv0Weight, w.PreConv0Bias, c, k: 3, stride: 2, pad: 1, out int h1, out int w1);
        ReluInPlace(stage0);

        var stage1dw = Conv2dDepthwise(stage0, c, h1, w1, w.PreConv2Weight, w.PreConv2Bias, k: 3, stride: 2, pad: 1, out int h2, out int w2);
        var stage1 = Conv2dPointwise(stage1dw, c, h2, w2, w.PreConv3Weight, w.PreConv3Bias, c);
        ReluInPlace(stage1);

        var stage2dw = Conv2dDepthwise(stage1, c, h2, w2, w.PreConv5Weight, w.PreConv5Bias, k: 3, stride: 2, pad: 1, out int h3, out int w3);
        var stage2 = Conv2dPointwise(stage2dw, c, h3, w3, w.PreConv6Weight, w.PreConv6Bias, c);
        ReluInPlace(stage2);

        // permute(0,2,1,3) + flatten: feature[k] = channel*W3 + freq_w (channel-major).
        int flatDim = c * w3;
        var flat = new float[h3][];
        for (int t = 0; t < h3; t++)
        {
            var row = new float[flatDim];
            for (int ch = 0; ch < c; ch++)
                for (int fw = 0; fw < w3; fw++)
                    row[ch * w3 + fw] = stage2[ch * h3 * w3 + t * w3 + fw];
            flat[t] = row;
        }

        var output = new float[h3][];
        for (int t = 0; t < h3; t++)
            output[t] = Linear(flat[t], w.PreOutWeight, w.PreOutBias, flatDim, w.HiddenDim);

        return (output, h3);
    }

    private static float[] Conv2dFull(float[] input, int cin, int hin, int win, float[] weight, float[] bias, int cout, int k, int stride, int pad, out int hout, out int wout)
    {
        hout = (hin + 2 * pad - k) / stride + 1;
        wout = (win + 2 * pad - k) / stride + 1;
        var output = new float[cout * hout * wout];
        for (int co = 0; co < cout; co++)
        {
            for (int ho = 0; ho < hout; ho++)
            {
                for (int wo = 0; wo < wout; wo++)
                {
                    float sum = bias[co];
                    for (int ci = 0; ci < cin; ci++)
                    {
                        for (int kh = 0; kh < k; kh++)
                        {
                            int hi = ho * stride - pad + kh;
                            if (hi < 0 || hi >= hin) continue;
                            for (int kw = 0; kw < k; kw++)
                            {
                                int wi = wo * stride - pad + kw;
                                if (wi < 0 || wi >= win) continue;
                                float wt = weight[kw + kh * k + ci * k * k + co * k * k * cin];
                                sum += wt * input[ci * hin * win + hi * win + wi];
                            }
                        }
                    }
                    output[co * hout * wout + ho * wout + wo] = sum;
                }
            }
        }
        return output;
    }

    /// <summary>Depthwise conv2d: groups == channels, each output channel uses only its matching input channel.</summary>
    private static float[] Conv2dDepthwise(float[] input, int channels, int hin, int win, float[] weight, float[] bias, int k, int stride, int pad, out int hout, out int wout)
    {
        hout = (hin + 2 * pad - k) / stride + 1;
        wout = (win + 2 * pad - k) / stride + 1;
        var output = new float[channels * hout * wout];
        for (int c = 0; c < channels; c++)
        {
            for (int ho = 0; ho < hout; ho++)
            {
                for (int wo = 0; wo < wout; wo++)
                {
                    float sum = bias[c];
                    for (int kh = 0; kh < k; kh++)
                    {
                        int hi = ho * stride - pad + kh;
                        if (hi < 0 || hi >= hin) continue;
                        for (int kw = 0; kw < k; kw++)
                        {
                            int wi = wo * stride - pad + kw;
                            if (wi < 0 || wi >= win) continue;
                            float wt = weight[kw + kh * k + c * k * k];
                            sum += wt * input[c * hin * win + hi * win + wi];
                        }
                    }
                    output[c * hout * wout + ho * wout + wo] = sum;
                }
            }
        }
        return output;
    }

    /// <summary>Pointwise (1x1, stride 1, no padding) full conv2d, i.e. a per-pixel Linear across channels.</summary>
    private static float[] Conv2dPointwise(float[] input, int cin, int h, int w, float[] weight, float[] bias, int cout)
    {
        var output = new float[cout * h * w];
        for (int co = 0; co < cout; co++)
        {
            float b = bias[co];
            int wBase = co * cin;
            for (int hp = 0; hp < h; hp++)
            {
                for (int wp = 0; wp < w; wp++)
                {
                    float sum = b;
                    for (int ci = 0; ci < cin; ci++)
                        sum += weight[wBase + ci] * input[ci * h * w + hp * w + wp];
                    output[co * h * w + hp * w + wp] = sum;
                }
            }
        }
        return output;
    }

    private static void ReluInPlace(float[] x)
    {
        for (int i = 0; i < x.Length; i++) if (x[i] < 0f) x[i] = 0f;
    }

    // -----------------------------------------------------------------
    // Positional encoding: sinusoidal, length 2T-1, positions descending from +(T-1) to -(T-1).
    // Returned as posEnc[p][dim] for p = 0..2T-2 (p=T-1 is the zero-offset row).
    // -----------------------------------------------------------------
    private static float[][] SinusoidalRelPosTable(int t, int dim)
    {
        int len = 2 * t - 1;
        var table = new float[len][];
        for (int p = 0; p < len; p++)
        {
            float pos = t - 1 - p;
            var row = new float[dim];
            for (int i = 0; i < dim; i += 2)
            {
                double div = Math.Exp(-Math.Log(10000.0) * i / dim);
                row[i] = (float)Math.Sin(pos * div);
                if (i + 1 < dim) row[i + 1] = (float)Math.Cos(pos * div);
            }
            table[p] = row;
        }
        return table;
    }

    // -----------------------------------------------------------------
    // Conformer block: FFN1(0.5) -> rel-pos self-attn -> conv module -> FFN2(0.5) -> LayerNorm.
    // -----------------------------------------------------------------
    private static float[][] ConformerBlock(ParakeetWeights w, ParakeetConformerLayer l, float[][] x, float[][] posEnc, int t)
    {
        int dim = w.HiddenDim;

        // FFN1 (macaron, half-step)
        var afterFf1 = new float[t][];
        for (int i = 0; i < t; i++)
        {
            var normed = LayerNorm(x[i], l.NormFf1Weight, l.NormFf1Bias);
            var h1 = Linear(normed, l.Ff1Linear1Weight, l.Ff1Linear1Bias, dim, w.FfDim);
            SiluInPlace(h1);
            var h2 = Linear(h1, l.Ff1Linear2Weight, l.Ff1Linear2Bias, w.FfDim, dim);
            var row = new float[dim];
            for (int d = 0; d < dim; d++) row[d] = x[i][d] + 0.5f * h2[d];
            afterFf1[i] = row;
        }

        // Self-attention (rel-pos, untied u/v, full Transformer-XL rel_shift)
        var attnOut = RelPosSelfAttention(w, l, afterFf1, posEnc, t);
        var afterAttn = new float[t][];
        for (int i = 0; i < t; i++)
        {
            var row = new float[dim];
            for (int d = 0; d < dim; d++) row[d] = afterFf1[i][d] + attnOut[i][d];
            afterAttn[i] = row;
        }

        // Conv module
        var convOut = ConvModule(w, l, afterAttn, t);
        var afterConv = new float[t][];
        for (int i = 0; i < t; i++)
        {
            var row = new float[dim];
            for (int d = 0; d < dim; d++) row[d] = afterAttn[i][d] + convOut[i][d];
            afterConv[i] = row;
        }

        // FFN2 (macaron, half-step)
        var output = new float[t][];
        for (int i = 0; i < t; i++)
        {
            var normed = LayerNorm(afterConv[i], l.NormFf2Weight, l.NormFf2Bias);
            var h1 = Linear(normed, l.Ff2Linear1Weight, l.Ff2Linear1Bias, dim, w.FfDim);
            SiluInPlace(h1);
            var h2 = Linear(h1, l.Ff2Linear2Weight, l.Ff2Linear2Bias, w.FfDim, dim);
            var row = new float[dim];
            for (int d = 0; d < dim; d++) row[d] = afterConv[i][d] + 0.5f * h2[d];
            output[i] = LayerNorm(row, l.NormOutWeight, l.NormOutBias);
        }

        return output;
    }

    private static float[][] RelPosSelfAttention(ParakeetWeights w, ParakeetConformerLayer l, float[][] x, float[][] posEnc, int t)
    {
        int dim = w.HiddenDim;
        int heads = w.NumHeads;
        int headDim = w.HeadDim;
        float scale = 1f / MathF.Sqrt(headDim);

        var q = new float[t][];
        var k = new float[t][];
        var v = new float[t][];
        var normed = new float[t][];
        for (int i = 0; i < t; i++)
        {
            normed[i] = LayerNorm(x[i], l.NormAttnWeight, l.NormAttnBias);
            q[i] = Linear(normed[i], l.AttnQWeight, l.AttnQBias, dim, dim);
            k[i] = Linear(normed[i], l.AttnKWeight, l.AttnKBias, dim, dim);
            v[i] = Linear(normed[i], l.AttnVWeight, l.AttnVBias, dim, dim);
        }

        int posLen = posEnc.Length; // 2T-1
        var r = new float[posLen][];
        for (int p = 0; p < posLen; p++)
            r[p] = LinearNoBias(posEnc[p], l.AttnPosWeight, dim, dim);

        var output = new float[t][];
        for (int i = 0; i < t; i++) output[i] = new float[dim];

        for (int h = 0; h < heads; h++)
        {
            int off = h * headDim;
            var qu = new float[t][];
            var qv = new float[t][];
            for (int i = 0; i < t; i++)
            {
                var u = new float[headDim];
                var vv = new float[headDim];
                for (int d = 0; d < headDim; d++)
                {
                    u[d] = q[i][off + d] + l.AttnPosBiasU[h * headDim + d];
                    vv[d] = q[i][off + d] + l.AttnPosBiasV[h * headDim + d];
                }
                qu[i] = u;
                qv[i] = vv;
            }

            System.Threading.Tasks.Parallel.For(0, t, qi =>
            {
                var scores = new float[t];
                unsafe
                {
                    fixed (float* up = qu[qi], vp = qv[qi])
                    {
                        for (int ki = 0; ki < t; ki++)
                        {
                            fixed (float* kp = k[ki])
                            {
                                float ac = SimdKernels.DotF32(up, kp + off, headDim);
                                int relIdx = (t - 1) + ki - qi;
                                fixed (float* rp = r[relIdx])
                                {
                                    float bd = SimdKernels.DotF32(vp, rp + off, headDim);
                                    scores[ki] = (ac + bd) * scale;
                                }
                            }
                        }
                    }
                }
                SoftmaxInPlace(scores);
                var outSpan = output[qi].AsSpan(off, headDim);
                for (int ki = 0; ki < t; ki++)
                {
                    var vRow = v[ki].AsSpan(off, headDim);
                    System.Numerics.Tensors.TensorPrimitives.MultiplyAdd(vRow, scores[ki], outSpan, outSpan);
                }
            });
        }

        var projected = new float[t][];
        for (int i = 0; i < t; i++)
            projected[i] = Linear(output[i], l.AttnOutWeight, l.AttnOutBias, dim, dim);
        return projected;
    }

    /// <summary>LN -> pw1(d->2d) -> GLU (first_half * sigmoid(second_half), matches ggml_siglu_swapped/PyTorch F.glu) -> depthwise conv1d (BN-folded) -> SiLU -> pw2(d->d).</summary>
    private static float[][] ConvModule(ParakeetWeights w, ParakeetConformerLayer l, float[][] x, int t)
    {
        int dim = w.HiddenDim;
        int kernel = w.ConvKernel;
        int pad = (kernel - 1) / 2;

        var glu = new float[t][];
        for (int i = 0; i < t; i++)
        {
            var normed = LayerNorm(x[i], l.NormConvWeight, l.NormConvBias);
            var pw1 = Linear(normed, l.ConvPw1Weight, l.ConvPw1Bias, dim, 2 * dim);
            var row = new float[dim];
            for (int d = 0; d < dim; d++)
            {
                float a = pw1[d];
                float b = pw1[dim + d];
                row[d] = a * (1f / (1f + MathF.Exp(-b)));
            }
            glu[i] = row;
        }

        var dwOut = new float[t][];
        for (int ti = 0; ti < t; ti++)
        {
            var row = new float[dim];
            for (int d = 0; d < dim; d++)
            {
                float sum = l.ConvDwBias[d];
                for (int kk = 0; kk < kernel; kk++)
                {
                    int srcT = ti - pad + kk;
                    if (srcT < 0 || srcT >= t) continue;
                    sum += l.ConvDwWeight[d * kernel + kk] * glu[srcT][d];
                }
                row[d] = sum;
            }
            dwOut[ti] = row;
        }

        var output = new float[t][];
        for (int i = 0; i < t; i++)
        {
            SiluInPlace(dwOut[i]);
            output[i] = Linear(dwOut[i], l.ConvPw2Weight, l.ConvPw2Bias, dim, dim);
        }
        return output;
    }

    private static void SiluInPlace(float[] x)
    {
        for (int i = 0; i < x.Length; i++)
        {
            float v = x[i];
            x[i] = v / (1f + MathF.Exp(-v));
        }
    }

    private static void SoftmaxInPlace(float[] scores)
    {
        float max = float.NegativeInfinity;
        for (int i = 0; i < scores.Length; i++) if (scores[i] > max) max = scores[i];
        float sum = 0f;
        for (int i = 0; i < scores.Length; i++)
        {
            float e = MathF.Exp(scores[i] - max);
            scores[i] = e;
            sum += e;
        }
        float invSum = 1f / sum;
        for (int i = 0; i < scores.Length; i++) scores[i] *= invSum;
    }

    private static unsafe float[] Linear(float[] input, float[] weight, float[]? bias, int inDim, int outDim)
    {
        var output = new float[outDim];
        fixed (float* wp = weight, xp = input, yp = output)
        {
            SimdKernels.MatVecF32(yp, wp, xp, outDim, inDim);
        }
        if (bias != null)
            for (int o = 0; o < outDim; o++) output[o] += bias[o];
        return output;
    }

    private static unsafe float[] LinearNoBias(float[] input, float[] weight, int inDim, int outDim)
    {
        var output = new float[outDim];
        fixed (float* wp = weight, xp = input, yp = output)
        {
            SimdKernels.MatVecF32(yp, wp, xp, outDim, inDim);
        }
        return output;
    }

    private static float[] LayerNorm(float[] x, float[] weight, float[] bias)
    {
        int n = x.Length;
        double mean = 0;
        for (int i = 0; i < n; i++) mean += x[i];
        mean /= n;
        double var = 0;
        for (int i = 0; i < n; i++) { double d = x[i] - mean; var += d * d; }
        var /= n;
        float invStd = (float)(1.0 / Math.Sqrt(var + ParakeetWeights.LayerNormEps));

        var output = new float[n];
        for (int i = 0; i < n; i++)
            output[i] = (float)((x[i] - mean) * invStd) * weight[i] + bias[i];
        return output;
    }
}
