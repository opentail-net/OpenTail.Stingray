using System;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Audio.Piper;

/// <summary>
/// Piper's VITS TextEncoder (enc_p): token embedding (scaled by sqrt(hidden)) -> N-layer
/// post-LN Transformer with windowed relative-position self-attention (Shaw et al. 2018 style,
/// NOT the Transformer-XL/ESPnet scheme used elsewhere in this codebase for Chatterbox/Kokoro) ->
/// proj to [mu, logs]. No absolute positional embedding -- VITS relies entirely on the relative-
/// position attention bias.
///
/// Relative-position attention: the reference (VITS attentions.py MultiHeadAttention) computes
/// this via a "skew/unskew" matrix trick purely for vectorized-PyTorch efficiency; mathematically
/// it is exactly a per-(query,key) lookup into a small, FIXED-size window table
/// (2*windowSize+1 entries) keyed by clamp(j-i, -windowSize, windowSize), with ZERO contribution
/// (not clamped-to-edge) for |j-i| &gt; windowSize (confirmed by the reference's `_get_relative_
/// embeddings` zero-padding the table before slicing when T &gt; windowSize+1). This class
/// implements that direct form -- correct for any T, no reshape games -- verified against real
/// ONNX golden output via cosine similarity.
/// </summary>
public static class PiperTextEncoder
{
    /// <summary>
    /// tokens are phoneme ids. Returns (encoderHidden, mu, logs), all channel-first [hidden, T].
    /// encoderHidden is the encoder stack's output BEFORE the final proj split -- VITS's
    /// SynthesizerTrn.infer feeds this (not mu) into the duration predictor and flow, per
    /// `x, m_p, logs_p, x_mask = self.enc_p(...)` followed by `self.dp(x, x_mask, ...)`.
    /// </summary>
    public static (float[] EncoderHidden, float[] Mu, float[] Logs) Forward(PiperOnnxWeights w, ReadOnlySpan<int> tokens)
    {
        int t = tokens.Length;
        int dim = w.HiddenDim;
        float embScale = MathF.Sqrt(dim);

        // Embed + scale, channel-first [dim, t].
        var x = new float[dim * t];
        for (int ti = 0; ti < t; ti++)
        {
            int tok = tokens[ti];
            int rowBase = tok * dim;
            for (int c = 0; c < dim; c++)
                x[c * t + ti] = w.EmbeddingWeight[rowBase + c] * embScale;
        }

        foreach (var layer in w.Layers)
        {
            var attnOut = RelPositionAttention(x, t, w, layer);
            for (int i = 0; i < x.Length; i++) x[i] += attnOut[i];
            x = LayerNormChannelFirst(x, dim, t, layer.Norm1Gamma, layer.Norm1Beta);

            var ffnOut = Ffn(x, t, w, layer);
            for (int i = 0; i < x.Length; i++) x[i] += ffnOut[i];
            x = LayerNormChannelFirst(x, dim, t, layer.Norm2Gamma, layer.Norm2Beta);
        }

        // proj: Conv1d(hidden, 2*hidden, kernel=1) -> split into mu/logs.
        var stats = Conv1x1(x, dim, t, w.ProjWeight, w.ProjBias, 2 * dim);
        var mu = new float[dim * t];
        var logs = new float[dim * t];
        Array.Copy(stats, 0, mu, 0, dim * t);
        Array.Copy(stats, dim * t, logs, 0, dim * t);
        return (x, mu, logs);
    }

    private static float[] RelPositionAttention(float[] x, int t, PiperOnnxWeights w, PiperEncoderLayerWeights lw)
    {
        int dim = w.HiddenDim;
        int heads = w.NumHeads;
        int headDim = w.HeadDim;
        int window = w.WindowSize;
        float scale = 1f / MathF.Sqrt(headDim);

        // conv_q/k/v are kernel=1 convs == per-timestep Linear over the channel dim.
        var q = ConvOverTime1x1(x, dim, t, lw.ConvQWeight, lw.ConvQBias, dim);
        var k = ConvOverTime1x1(x, dim, t, lw.ConvKWeight, lw.ConvKBias, dim);
        var v = ConvOverTime1x1(x, dim, t, lw.ConvVWeight, lw.ConvVBias, dim);

        var context = new float[dim * t]; // channel-first [dim, t]

        for (int h = 0; h < heads; h++)
        {
            int hOff = h * headDim;
            var scores = new float[t];
            for (int i = 0; i < t; i++)
            {
                for (int j = 0; j < t; j++)
                {
                    float ac = 0f;
                    for (int d = 0; d < headDim; d++) ac += q[(hOff + d) * t + i] * k[(hOff + d) * t + j];
                    ac *= scale;

                    int offset = j - i;
                    float bd = 0f;
                    if (offset >= -window && offset <= window)
                    {
                        int relIdx = (offset + window) * headDim;
                        for (int d = 0; d < headDim; d++)
                            bd += (q[(hOff + d) * t + i] * scale) * lw.EmbRelK[relIdx + d];
                    }
                    scores[j] = ac + bd;
                }
                SoftmaxInPlace(scores);

                for (int j = 0; j < t; j++)
                {
                    float p = scores[j];
                    if (p == 0f) continue;
                    for (int d = 0; d < headDim; d++) context[(hOff + d) * t + i] += p * v[(hOff + d) * t + j];

                    int offset = j - i;
                    if (offset >= -window && offset <= window)
                    {
                        int relIdx = (offset + window) * headDim;
                        for (int d = 0; d < headDim; d++)
                            context[(hOff + d) * t + i] += p * lw.EmbRelV[relIdx + d];
                    }
                }
            }
        }

        return Conv1x1(context, dim, t, lw.ConvOWeight, lw.ConvOBias, dim);
    }

    /// <summary>FFN: conv_1 (kernel=3, pad=1) -> ReLU -> conv_2 (kernel=3, pad=1).</summary>
    private static float[] Ffn(float[] x, int t, PiperOnnxWeights w, PiperEncoderLayerWeights lw)
    {
        int dim = w.HiddenDim;
        int ffnHidden = lw.Ffn1Bias.Length;
        var h = Conv1dSamePad(x, dim, t, lw.Ffn1Weight, lw.Ffn1Bias, ffnHidden, w.FfnKernel);
        for (int i = 0; i < h.Length; i++) if (h[i] < 0f) h[i] = 0f; // ReLU
        return Conv1dSamePad(h, ffnHidden, t, lw.Ffn2Weight, lw.Ffn2Bias, dim, w.FfnKernel);
    }

    /// <summary>Per-timestep 1x1 conv (a Linear applied independently at every timestep), input/output channel-first.</summary>
    private static float[] Conv1x1(float[] input, int inCh, int t, float[] weight, float[] bias, int outCh) =>
        ConvOverTime1x1(input, inCh, t, weight, bias, outCh);

    private static unsafe float[] ConvOverTime1x1(float[] input, int inCh, int t, float[] weight, float[] bias, int outCh)
    {
        var output = new float[outCh * t];
        var col = new float[inCh];
        var outCol = new float[outCh];
        for (int ti = 0; ti < t; ti++)
        {
            for (int c = 0; c < inCh; c++) col[c] = input[c * t + ti];
            fixed (float* w = weight, x = col, y = outCol)
            {
                SimdKernels.MatVecF32(y, w, x, outCh, inCh);
            }
            for (int c = 0; c < outCh; c++) output[c * t + ti] = outCol[c] + bias[c];
        }
        return output;
    }

    private static float[] Conv1dSamePad(float[] input, int inCh, int t, float[] weight, float[] bias, int outCh, int kernel)
    {
        int pad = kernel / 2;
        var output = new float[outCh * t];
        System.Threading.Tasks.Parallel.For(0, outCh, oc =>
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
                        if ((uint)src < (uint)t) sum += weight[wBase + k] * input[srcBase + src];
                    }
                }
                output[oc * t + ti] = sum;
            }
        });
        return output;
    }

    private static float[] LayerNormChannelFirst(float[] x, int ch, int t, float[] gamma, float[] beta, float eps = 1e-5f)
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
                output[c * t + ti] = (float)((x[c * t + ti] - mean) * invStd) * gamma[c] + beta[c];
        }
        return output;
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
}
