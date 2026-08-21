using System;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Audio.Chatterbox;

/// <summary>
/// S3Gen stage 2: the CFM (Conditional Flow Matching) flow-matching UNet estimator
/// (examples/chatterbox-tts-py/chatterbox/models/s3gen/decoder.py's ConditionalDecoder) plus the
/// meanflow-distilled 2-step Euler ODE solver (flow_matching.py's CausalConditionalCFM.
/// basic_euler). Turns [mu, cond, spk-embedding] conditioning (from ChatterboxFlowEncoder) plus
/// gaussian noise into a mel-spectrogram.
///
/// Structure (channels=[256] in the reference, a single-element list, so both the "down" and
/// "up" stages degrade to no-op-resolution CausalConv1d resamples -- see decoder.py and
/// ChatterboxS3GenWeights's stage-loading comments):
///   down:  CausalResnetBlock1D(320-&gt;256) + 4x BasicTransformerBlock(256) + CausalConv1d(256-&gt;256, k=3)
///   mid:   12x [CausalResnetBlock1D(256-&gt;256) + 4x BasicTransformerBlock(256)]
///   up:    CausalResnetBlock1D(512-&gt;256, input = concat(x, down-stage skip)) + 4x BasicTransformerBlock(256) + CausalConv1d(256-&gt;256, k=3)
///   final: CausalBlock1D(256-&gt;256) + Conv1d(256-&gt;80, k=1)
///
/// Timestep conditioning (meanflow): t and r (start/end of this Euler step) are each embedded via
/// SinusoidalPosEmb(320) -> Linear(320,1024)+SiLU+Linear(1024,1024), concatenated to 2048-dim and
/// mixed down to 1024-dim via a single Linear (no bias) -- decoder.py's time_embed_mixer.
/// </summary>
public static class ChatterboxCfmDecoder
{
    /// <summary>
    /// mu, cond are channel-first [80, T]. spkEmbed is [80]. Returns the generated mel-spectrogram,
    /// channel-first [80, T] (still the FULL prompt+generated length -- caller slices off the
    /// prompt-length prefix, matching flow.py's `feat[:, :, mel_len1:]`).
    /// </summary>
    public static float[] Generate(ChatterboxS3GenWeights w, float[] mu, float[] cond, float[] spkEmbed, int t, Random rng, int nSteps = 2)
    {
        int mel = w.DecOutChannels; // 80

        // x0 ~ N(0, 1), channel-first [80, T]
        var x = new float[mel * t];
        for (int i = 0; i < x.Length; i++) x[i] = SampleGaussian(rng);

        // t_span = linspace(0, 1, nSteps+1); meanflow skips the cosine warp.
        var tSpan = new float[nSteps + 1];
        for (int i = 0; i <= nSteps; i++) tSpan[i] = (float)i / nSteps;

        for (int step = 0; step < nSteps; step++)
        {
            float tCur = tSpan[step];
            float tNext = tSpan[step + 1];
            float[] dxdt = Estimator(w, x, mu, cond, spkEmbed, tCur, tNext, t);
            float dt = tNext - tCur;
            for (int i = 0; i < x.Length; i++) x[i] += dt * dxdt[i];
        }

        return x;
    }

    private static float SampleGaussian(Random rng)
    {
        // Box-Muller.
        double u1 = Math.Max(1e-12, rng.NextDouble());
        double u2 = rng.NextDouble();
        return (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
    }

    /// <summary>ConditionalDecoder.forward: the UNet's velocity-field estimate dx/dt at (x, t, r).</summary>
    private static float[] Estimator(ChatterboxS3GenWeights w, float[] x, float[] mu, float[] cond, float[] spkEmbed, float tCur, float tNext, int t)
    {
        int mel = w.DecOutChannels; // 80

        var timeEmb = MeanflowTimeEmbed(w, tCur, tNext);

        // input = concat([x, mu, spks_broadcast, cond], channels) -> [320, T]
        int inCh = w.DecInChannels; // 320
        var input = new float[inCh * t];
        CopyChannels(x, input, 0, mel, t);
        CopyChannels(mu, input, mel, mel, t);
        for (int c = 0; c < mel; c++)
            for (int ti = 0; ti < t; ti++)
                input[(2 * mel + c) * t + ti] = spkEmbed[c];
        CopyChannels(cond, input, 3 * mel, mel, t);

        int ch = w.DecChannels; // 256

        // --- down stage ---
        var down = ResnetBlock(input, inCh, t, timeEmb, w.DownStage.Resnet, ch);
        foreach (var tb in w.DownStage.TransformerBlocks)
            down = TransformerBlock(down, ch, t, tb, w.DecNumHeads, w.DecHeadDim);
        var skip = down; // saved for the up-stage skip connection
        down = CausalConv1d(down, ch, t, w.DownStage.ResampleConvWeight!, w.DownStage.ResampleConvBias!, ch, kernel: 3);

        // --- mid stages ---
        var mid = down;
        foreach (var stage in w.MidStages)
        {
            mid = ResnetBlock(mid, ch, t, timeEmb, stage.Resnet, ch);
            foreach (var tb in stage.TransformerBlocks)
                mid = TransformerBlock(mid, ch, t, tb, w.DecNumHeads, w.DecHeadDim);
        }

        // --- up stage: concat(mid, skip) along channels -> 2*ch ---
        var upIn = new float[2 * ch * t];
        CopyChannels(mid, upIn, 0, ch, t);
        CopyChannels(skip, upIn, ch, ch, t);
        var up = ResnetBlock(upIn, 2 * ch, t, timeEmb, w.UpStage.Resnet, ch);
        foreach (var tb in w.UpStage.TransformerBlocks)
            up = TransformerBlock(up, ch, t, tb, w.DecNumHeads, w.DecHeadDim);
        up = CausalConv1d(up, ch, t, w.UpStage.ResampleConvWeight!, w.UpStage.ResampleConvBias!, ch, kernel: 3);

        // --- final block + projection ---
        var finalConv = CausalConv1d(up, ch, t, w.FinalBlockConvWeight, w.FinalBlockConvBias, ch, kernel: 3);
        var finalNormed = LayerNormChannelFirst(finalConv, ch, t, w.FinalBlockLnWeight, w.FinalBlockLnBias);
        MishInPlace(finalNormed);

        return Conv1dK1(finalNormed, ch, t, w.FinalProjWeight, w.FinalProjBias, w.DecOutChannels);
    }

    /// <summary>
    /// SinusoidalPosEmb(320) -> Linear(320,1024)+SiLU+Linear(1024,1024) applied to both t and r,
    /// concatenated (2048) and mixed down to 1024 via a bias-free Linear (decoder.py's
    /// time_embed_mixer, only present because meanflow=True).
    /// </summary>
    private static float[] MeanflowTimeEmbed(ChatterboxS3GenWeights w, float tCur, float tNext)
    {
        var tEmbRaw = SinusoidalPosEmb(tCur, w.DecInChannels);
        var rEmbRaw = SinusoidalPosEmb(tNext, w.DecInChannels);
        int timeEmbedDim = w.DecChannels * 4; // 1024

        var tEmb = TimeMlp(w, tEmbRaw, timeEmbedDim);
        var rEmb = TimeMlp(w, rEmbRaw, timeEmbedDim);

        var concat = new float[timeEmbedDim * 2];
        Array.Copy(tEmb, 0, concat, 0, timeEmbedDim);
        Array.Copy(rEmb, 0, concat, timeEmbedDim, timeEmbedDim);

        return LinearNoBias(concat, w.TimeMixerWeight, timeEmbedDim * 2, timeEmbedDim);
    }

    private static float[] TimeMlp(ChatterboxS3GenWeights w, float[] emb, int timeEmbedDim)
    {
        var h = Linear(emb, w.TimeMlpLinear1Weight, w.TimeMlpLinear1Bias, w.DecInChannels, timeEmbedDim);
        SiluInPlace(h);
        return Linear(h, w.TimeMlpLinear2Weight, w.TimeMlpLinear2Bias, timeEmbedDim, timeEmbedDim);
    }

    /// <summary>SinusoidalPosEmb(dim), matcha/decoder.py: scale=1000 default.</summary>
    private static float[] SinusoidalPosEmb(float x, int dim)
    {
        int half = dim / 2;
        var emb = new float[dim];
        double logStep = Math.Log(10000.0) / (half - 1);
        for (int i = 0; i < half; i++)
        {
            double freq = Math.Exp(-i * logStep);
            double angle = 1000.0 * x * freq;
            emb[i] = (float)Math.Sin(angle);
            emb[half + i] = (float)Math.Cos(angle);
        }
        return emb;
    }

    /// <summary>
    /// CausalResnetBlock1D: block1 (causal conv k=3 + LayerNorm + Mish) -> += mlp(time_emb)
    /// broadcast over time -> block2 (same as block1) -> + res_conv(x) (1x1 conv, always applied).
    /// </summary>
    private static float[] ResnetBlock(float[] x, int dimIn, int t, float[] timeEmb, ChatterboxCfmResnetWeights rw, int dimOut)
    {
        var h = CausalConv1d(x, dimIn, t, rw.Block1ConvWeight, rw.Block1ConvBias, dimOut, kernel: 3);
        h = LayerNormChannelFirst(h, dimOut, t, rw.Block1LnWeight, rw.Block1LnBias);
        MishInPlace(h);

        var timeProj = Linear(MishScalarArray(timeEmb), rw.MlpWeight, rw.MlpBias, timeEmb.Length, dimOut);
        for (int c = 0; c < dimOut; c++)
        {
            float bias = timeProj[c];
            int rowBase = c * t;
            for (int ti = 0; ti < t; ti++) h[rowBase + ti] += bias;
        }

        h = CausalConv1d(h, dimOut, t, rw.Block2ConvWeight, rw.Block2ConvBias, dimOut, kernel: 3);
        h = LayerNormChannelFirst(h, dimOut, t, rw.Block2LnWeight, rw.Block2LnBias);
        MishInPlace(h);

        var resConv = Conv1dK1(x, dimIn, t, rw.ResConvWeight, rw.ResConvBias, dimOut);
        for (int i = 0; i < h.Length; i++) h[i] += resConv[i];
        return h;
    }

    private static float[] MishScalarArray(float[] x)
    {
        var y = new float[x.Length];
        for (int i = 0; i < x.Length; i++) y[i] = Mish(x[i]);
        return y;
    }

    /// <summary>BasicTransformerBlock: x = x + attn1(LayerNorm(x)); x = x + FF(LayerNorm(x)) -- self-attention only.</summary>
    private static float[] TransformerBlock(float[] x, int dim, int t, ChatterboxCfmTransformerBlockWeights tw, int heads, int headDim)
    {
        var normed = LayerNormChannelFirstToRowMajor(x, dim, t, tw.Norm1Weight, tw.Norm1Bias);
        var attnOut = SelfAttention(normed, t, dim, tw, heads, headDim);

        var afterAttn = new float[dim * t];
        for (int c = 0; c < dim; c++)
            for (int ti = 0; ti < t; ti++)
                afterAttn[c * t + ti] = x[c * t + ti] + attnOut[ti * dim + c];

        var normed3 = LayerNormChannelFirstToRowMajor(afterAttn, dim, t, tw.Norm3Weight, tw.Norm3Bias);
        var output = new float[dim * t];
        for (int ti = 0; ti < t; ti++)
        {
            var row = new float[dim];
            Array.Copy(normed3, ti * dim, row, 0, dim);
            var up = Linear(row, tw.FfUpWeight, tw.FfUpBias, dim, dim * 4);
            GeluInPlace(up);
            var down = Linear(up, tw.FfDownWeight, tw.FfDownBias, dim * 4, dim);
            for (int c = 0; c < dim; c++) output[c * t + ti] = afterAttn[c * t + ti] + down[c];
        }
        return output;
    }

    /// <summary>Standard (non-relative) multi-head self-attention. input is row-major [T, dim]; q/k/v have no bias.</summary>
    private static float[] SelfAttention(float[] inputRowMajor, int t, int dim, ChatterboxCfmTransformerBlockWeights tw, int heads, int headDim)
    {
        int qkvDim = heads * headDim;
        var q = new float[t][];
        var k = new float[t][];
        var v = new float[t][];
        for (int i = 0; i < t; i++)
        {
            var row = new float[dim];
            Array.Copy(inputRowMajor, i * dim, row, 0, dim);
            q[i] = LinearNoBias(row, tw.QWeight, dim, qkvDim);
            k[i] = LinearNoBias(row, tw.KWeight, dim, qkvDim);
            v[i] = LinearNoBias(row, tw.VWeight, dim, qkvDim);
        }

        var context = new float[t][];
        for (int i = 0; i < t; i++) context[i] = new float[qkvDim];

        float scale = 1f / MathF.Sqrt(headDim);
        for (int h = 0; h < heads; h++)
        {
            int hOff = h * headDim;
            var scores = new float[t];
            for (int i = 0; i < t; i++)
            {
                for (int j = 0; j < t; j++)
                {
                    float dot = 0f;
                    for (int d = 0; d < headDim; d++) dot += q[i][hOff + d] * k[j][hOff + d];
                    scores[j] = dot * scale;
                }
                SoftmaxInPlace(scores);
                for (int j = 0; j < t; j++)
                {
                    float wgt = scores[j];
                    for (int d = 0; d < headDim; d++) context[i][hOff + d] += wgt * v[j][hOff + d];
                }
            }
        }

        var output = new float[t * dim];
        for (int i = 0; i < t; i++)
        {
            var projected = Linear(context[i], tw.OutWeight, tw.OutBias, qkvDim, dim);
            Array.Copy(projected, 0, output, i * dim, dim);
        }
        return output;
    }

    private static void CopyChannels(float[] src, float[] dst, int dstChannelOffset, int channels, int t)
    {
        Array.Copy(src, 0, dst, dstChannelOffset * t, channels * t);
    }

    /// <summary>Causal (left-pad kernel-1, no right pad) stride-1 Conv1d. Channel-first [inCh, t] -> [outCh, t].</summary>
    private static float[] CausalConv1d(float[] input, int inCh, int t, float[] weight, float[] bias, int outCh, int kernel)
    {
        int pad = kernel - 1;
        var output = new float[outCh * t];
        for (int oc = 0; oc < outCh; oc++)
        {
            float b = bias[oc];
            int wOcBase = oc * inCh * kernel;
            for (int ti = 0; ti < t; ti++)
            {
                float sum = b;
                for (int ic = 0; ic < inCh; ic++)
                {
                    int wBase = wOcBase + ic * kernel;
                    int srcBase = ic * t;
                    for (int k = 0; k < kernel; k++)
                    {
                        int src = ti - pad + k;
                        if (src >= 0) sum += weight[wBase + k] * input[srcBase + src];
                    }
                }
                output[oc * t + ti] = sum;
            }
        }
        return output;
    }

    private static float[] Conv1dK1(float[] input, int inCh, int t, float[] weight, float[] bias, int outCh)
    {
        var output = new float[outCh * t];
        for (int oc = 0; oc < outCh; oc++)
        {
            float b = bias[oc];
            int wBase = oc * inCh;
            for (int ti = 0; ti < t; ti++)
            {
                float sum = b;
                for (int ic = 0; ic < inCh; ic++) sum += weight[wBase + ic] * input[ic * t + ti];
                output[oc * t + ti] = sum;
            }
        }
        return output;
    }

    private static float[] LayerNormChannelFirst(float[] x, int ch, int t, float[] weight, float[] bias, float eps = 1e-5f)
    {
        var output = new float[ch * t];
        for (int ti = 0; ti < t; ti++)
        {
            double mean = 0;
            for (int c = 0; c < ch; c++) mean += x[c * t + ti];
            mean /= ch;
            double var = 0;
            for (int c = 0; c < ch; c++) { double d = x[c * t + ti] - mean; var += d * d; }
            var /= ch;
            float invStd = (float)(1.0 / Math.Sqrt(var + eps));
            for (int c = 0; c < ch; c++)
                output[c * t + ti] = (float)((x[c * t + ti] - mean) * invStd) * weight[c] + bias[c];
        }
        return output;
    }

    /// <summary>Same per-timestep LayerNorm as <see cref="LayerNormChannelFirst"/>, but returns row-major [T, ch] (what BasicTransformerBlock's attention/FF want).</summary>
    private static float[] LayerNormChannelFirstToRowMajor(float[] x, int ch, int t, float[] weight, float[] bias, float eps = 1e-5f)
    {
        var output = new float[t * ch];
        for (int ti = 0; ti < t; ti++)
        {
            double mean = 0;
            for (int c = 0; c < ch; c++) mean += x[c * t + ti];
            mean /= ch;
            double var = 0;
            for (int c = 0; c < ch; c++) { double d = x[c * t + ti] - mean; var += d * d; }
            var /= ch;
            float invStd = (float)(1.0 / Math.Sqrt(var + eps));
            for (int c = 0; c < ch; c++)
                output[ti * ch + c] = (float)((x[c * t + ti] - mean) * invStd) * weight[c] + bias[c];
        }
        return output;
    }

    private static float Mish(float x) => x * MathF.Tanh(MathF.Log(1f + MathF.Exp(x)));

    private static void MishInPlace(float[] x)
    {
        for (int i = 0; i < x.Length; i++) x[i] = Mish(x[i]);
    }

    private static void SiluInPlace(float[] x)
    {
        for (int i = 0; i < x.Length; i++) x[i] = x[i] / (1f + MathF.Exp(-x[i]));
    }

    private static void GeluInPlace(float[] x)
    {
        const float c = 0.7978845608028654f; // sqrt(2/pi)
        for (int i = 0; i < x.Length; i++)
        {
            float v = x[i];
            float inner = c * (v + 0.044715f * v * v * v);
            x[i] = 0.5f * v * (1f + MathF.Tanh(inner));
        }
    }

    private static void SoftmaxInPlace(float[] scores)
    {
        float max = float.NegativeInfinity;
        for (int i = 0; i < scores.Length; i++) if (scores[i] > max) max = scores[i];
        double sum = 0;
        for (int i = 0; i < scores.Length; i++)
        {
            double e = Math.Exp(scores[i] - max);
            scores[i] = (float)e;
            sum += e;
        }
        float invSum = (float)(1.0 / sum);
        for (int i = 0; i < scores.Length; i++) scores[i] *= invSum;
    }

    private static unsafe float[] Linear(float[] input, float[] weight, float[] bias, int inDim, int outDim)
    {
        var output = new float[outDim];
        fixed (float* w = weight, x = input, y = output)
        {
            SimdKernels.MatVecF32(y, w, x, outDim, inDim);
        }
        for (int o = 0; o < outDim; o++) output[o] += bias[o];
        return output;
    }

    private static unsafe float[] LinearNoBias(float[] input, float[] weight, int inDim, int outDim)
    {
        var output = new float[outDim];
        fixed (float* w = weight, x = input, y = output)
        {
            SimdKernels.MatVecF32(y, w, x, outDim, inDim);
        }
        return output;
    }
}
