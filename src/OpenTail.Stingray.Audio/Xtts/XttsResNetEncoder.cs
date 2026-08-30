
namespace OpenTail.Stingray.Audio.Xtts;

/// <summary>
/// Real XTTS-v2 `ResNetSpeakerEncoder.forward` (`TTS/encoder/models/resnet.py`), given an
/// ALREADY-EXTRACTED power-mel-spectrogram (via <see cref="XttsSpeakerMelExtractor"/> -- this
/// class does NOT itself run `torch_spec`, matching the split already established by this port's
/// other stages). Real order: `log_input` (log(x+1e-6)) -> InstanceNorm1d(64, affine=False,
/// per-mel-bin normalize across time) -> unsqueeze to [1,64,T] -> conv1+relu+bn1 -> 4 ResNet
/// layers ([3,4,6,3] `SEBasicBlock`s, 3 stride-2 downsamples) -> reshape [C*H',W'] -> attentive
/// statistics pooling ("ASP": softmax-weighted mean+std over time) -> `fc` projection to a
/// 512-dim d-vector.
/// </summary>
public static class XttsResNetEncoder
{
    /// <summary>mel is channel-first [64, T] (real, un-logged, un-normalized power mel-spectrogram straight from <see cref="XttsSpeakerMelExtractor"/>). Returns the real 512-dim speaker embedding (no L2-norm applied -- matches the real reference's own default `l2_norm=False`).</summary>
    public static float[] Forward(XttsResNetWeights w, float[] mel, int t)
    {
        const int melDim = XttsResNetWeights.InputDim;

        // log_input=True: log(x + 1e-6).
        var x = new float[mel.Length];
        for (int i = 0; i < x.Length; i++) x[i] = MathF.Log(mel[i] + 1e-6f);

        // InstanceNorm1d(64, affine=False): normalize each of the 64 mel-bin "channels" across time, per-clip, no learnable scale/shift.
        InstanceNormInPlace(x, melDim, t);

        // conv1 (1->32, k3, s1, p1) + relu + bn1. Input is [1, 64(H), T(W)] (mel bins as height, time as width).
        int h = melDim, wid = t;
        var cur = XttsResNetKernels.Conv2d(x, 1, h, wid, w.Conv1Weight, XttsResNetWeights.NumFilters[0], kernel: 3, stride: 1, pad: 1, out h, out wid, bias: w.Conv1Bias);
        XttsResNetKernels.ReluInPlace(cur);
        XttsResNetKernels.BatchNorm2dInPlace(cur, XttsResNetWeights.NumFilters[0], h, wid, w.Bn1.Weight, w.Bn1.Bias, w.Bn1.RunningMean, w.Bn1.RunningVar);

        int ch = XttsResNetWeights.NumFilters[0];
        for (int layerIdx = 0; layerIdx < 4; layerIdx++)
        {
            var blocks = w.ResLayers[layerIdx];
            for (int b = 0; b < blocks.Length; b++)
            {
                int stride = (layerIdx > 0 && b == 0) ? 2 : 1;
                int outCh = XttsResNetWeights.NumFilters[layerIdx];
                (cur, h, wid) = ResBlockForward(cur, ch, h, wid, blocks[b], outCh, stride);
                ch = outCh;
            }
        }

        // reshape [C, H', W'] -> [C*H', W'] -- PyTorch row-major flatten groups channel-then-height (index = c*H' + hh), confirmed from the real `x.reshape(x.size(0), -1, x.size(-1))` on a [B,C,H,W] tensor.
        int attnCh = ch * h;
        var reshaped = cur; // already exactly this layout: channel-first [ch,h,wid] flat IS [ch*h, wid] flat (same memory order, c*h*wid+hh*wid+ww = (c*h+hh)*wid+ww).

        var attnWeights = AttentionForward(w, reshaped, attnCh, wid);

        // ASP: mu = sum(x*w, dim=time); sg = sqrt(clamp(sum(x^2*w,dim=time) - mu^2, min=1e-5)).
        var mu = new float[attnCh];
        var sg = new float[attnCh];
        for (int c = 0; c < attnCh; c++)
        {
            float sumXw = 0f, sumX2w = 0f;
            int baseIdx = c * wid;
            for (int ti = 0; ti < wid; ti++)
            {
                float xv = reshaped[baseIdx + ti];
                float wv = attnWeights[baseIdx + ti];
                sumXw += xv * wv;
                sumX2w += xv * xv * wv;
            }
            mu[c] = sumXw;
            sg[c] = MathF.Sqrt(MathF.Max(1e-5f, sumX2w - sumXw * sumXw));
        }

        var pooled = new float[2 * attnCh];
        Array.Copy(mu, 0, pooled, 0, attnCh);
        Array.Copy(sg, 0, pooled, attnCh, attnCh);

        return Linear(pooled, w.FcWeight, w.FcBias, XttsResNetWeights.ProjDim);
    }

    private static void InstanceNormInPlace(float[] x, int ch, int t, float eps = 1e-5f)
    {
        for (int c = 0; c < ch; c++)
        {
            int baseIdx = c * t;
            double mean = 0;
            for (int i = 0; i < t; i++) mean += x[baseIdx + i];
            mean /= t;
            double var = 0;
            for (int i = 0; i < t; i++) { double d = x[baseIdx + i] - mean; var += d * d; }
            var /= t; // biased variance (PyTorch InstanceNorm default)
            float invStd = (float)(1.0 / Math.Sqrt(var + eps));
            for (int i = 0; i < t; i++) x[baseIdx + i] = (float)((x[baseIdx + i] - mean) * invStd);
        }
    }

    private static (float[] Output, int H, int W) ResBlockForward(float[] x, int inCh, int h, int w, XttsResBlockWeights bw, int outCh, int stride)
    {
        var out1 = XttsResNetKernels.Conv2d(x, inCh, h, w, bw.Conv1Weight, outCh, kernel: 3, stride: stride, pad: 1, out int h1, out int w1);
        XttsResNetKernels.ReluInPlace(out1);
        XttsResNetKernels.BatchNorm2dInPlace(out1, outCh, h1, w1, bw.Bn1.Weight, bw.Bn1.Bias, bw.Bn1.RunningMean, bw.Bn1.RunningVar);

        var out2 = XttsResNetKernels.Conv2d(out1, outCh, h1, w1, bw.Conv2Weight, outCh, kernel: 3, stride: 1, pad: 1, out int h2, out int w2);
        XttsResNetKernels.BatchNorm2dInPlace(out2, outCh, h2, w2, bw.Bn2.Weight, bw.Bn2.Bias, bw.Bn2.RunningMean, bw.Bn2.RunningVar);

        var se = XttsResNetKernels.SqueezeExcite(out2, outCh, h2, w2, bw.SeFc0Weight, bw.SeFc0Bias, bw.SeFc2Weight, bw.SeFc2Bias, bw.SeReducedCh);

        float[] residual;
        if (bw.DownsampleConvWeight is not null)
        {
            residual = XttsResNetKernels.Conv2d(x, inCh, h, w, bw.DownsampleConvWeight, outCh, kernel: 1, stride: stride, pad: 0, out _, out _);
            XttsResNetKernels.BatchNorm2dInPlace(residual, outCh, h2, w2, bw.DownsampleBn!.Weight, bw.DownsampleBn.Bias, bw.DownsampleBn.RunningMean, bw.DownsampleBn.RunningVar);
        }
        else
        {
            residual = x;
        }

        var output = new float[se.Length];
        for (int i = 0; i < output.Length; i++) output[i] = se[i] + residual[i];
        XttsResNetKernels.ReluInPlace(output);
        return (output, h2, w2);
    }

    /// <summary>Real attention head: Conv1d(2048->128,k1)+bias -> ReLU -> BatchNorm1d(128) -> Conv1d(128->2048,k1)+bias -> Softmax(dim=time).</summary>
    private static float[] AttentionForward(XttsResNetWeights w, float[] x, int ch, int t)
    {
        var h1 = Conv1x1WithBias(x, ch, t, w.Attn0Weight, w.Attn0Bias, 128);
        XttsResNetKernels.ReluInPlace(h1);
        XttsResNetKernels.BatchNorm1dInPlace(h1, 128, t, w.Attn2Bn.Weight, w.Attn2Bn.Bias, w.Attn2Bn.RunningMean, w.Attn2Bn.RunningVar);
        var h2 = Conv1x1WithBias(h1, 128, t, w.Attn3Weight, w.Attn3Bias, ch);

        // Softmax over TIME (dim=2 in the real [B,C,T] tensor) -- per-channel, across all t.
        for (int c = 0; c < ch; c++)
        {
            int baseIdx = c * t;
            float max = float.NegativeInfinity;
            for (int i = 0; i < t; i++) if (h2[baseIdx + i] > max) max = h2[baseIdx + i];
            float sum = 0f;
            for (int i = 0; i < t; i++) { float e = MathF.Exp(h2[baseIdx + i] - max); h2[baseIdx + i] = e; sum += e; }
            float inv = 1f / sum;
            for (int i = 0; i < t; i++) h2[baseIdx + i] *= inv;
        }
        return h2;
    }

    private static float[] Conv1x1WithBias(float[] input, int inCh, int t, float[] weight, float[] bias, int outCh)
    {
        var output = new float[outCh * t];
        for (int o = 0; o < outCh; o++)
        {
            float b = bias[o];
            int wBase = o * inCh;
            for (int ti = 0; ti < t; ti++)
            {
                float sum = b;
                for (int i = 0; i < inCh; i++) sum += weight[wBase + i] * input[i * t + ti];
                output[o * t + ti] = sum;
            }
        }
        return output;
    }

    private static float[] Linear(float[] x, float[] weight, float[] bias, int outDim)
    {
        int inDim = x.Length;
        var output = new float[outDim];
        for (int o = 0; o < outDim; o++)
        {
            float sum = bias[o];
            int wBase = o * inDim;
            for (int i = 0; i < inDim; i++) sum += weight[wBase + i] * x[i];
            output[o] = sum;
        }
        return output;
    }
}
