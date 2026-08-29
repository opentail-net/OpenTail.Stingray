
namespace OpenTail.Stingray.Audio.Primitives;

/// <summary>
/// Shared math kernels for the VITS family of TTS models (Piper, MeloTTS, and any future
/// VITS/VITS2-derived pipeline): windowed relative-position self-attention (Shaw et al. 2018
/// style, as used by VITS's `attentions.py` MultiHeadAttention), 1x1/"same"-padded Conv1d,
/// and channel-first LayerNorm. Extracted from the original Piper-only implementation so later
/// VITS-family ports (MeloTTS) don't hand-roll a second copy of the same math -- keep this the
/// single source of truth for these primitives across all VITS-derived pipelines.
/// </summary>
public static class VitsAttentionKernels
{
    /// <summary>
    /// Windowed relative-position multi-head self-attention over a channel-first [dim, t] input.
    /// Mathematically equivalent to VITS's `_get_relative_embeddings` + skew/unskew trick: for
    /// each (query i, key j) pair, the relative bias term uses embRelK/embRelV[clamp(j-i,
    /// -window, window)], contributing ZERO when |j-i| &gt; window (not clamped-to-edge).
    /// convQ/K/V/O are kernel=1 conv (i.e. per-timestep Linear) weights, [dim, dim] row-major.
    /// embRelK/embRelV are [2*window+1, headDim].
    /// </summary>
    public static float[] RelPositionSelfAttention(
        float[] x, int t, int dim, int heads, int window,
        float[] convQWeight, float[] convQBias,
        float[] convKWeight, float[] convKBias,
        float[] convVWeight, float[] convVBias,
        float[] convOWeight, float[] convOBias,
        float[] embRelK, float[] embRelV)
    {
        int headDim = dim / heads;
        float scale = 1f / MathF.Sqrt(headDim);

        var q = ConvOverTime1x1(x, dim, t, convQWeight, convQBias, dim);
        var k = ConvOverTime1x1(x, dim, t, convKWeight, convKBias, dim);
        var v = ConvOverTime1x1(x, dim, t, convVWeight, convVBias, dim);

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
                            bd += (q[(hOff + d) * t + i] * scale) * embRelK[relIdx + d];
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
                            context[(hOff + d) * t + i] += p * embRelV[relIdx + d];
                    }
                }
            }
        }

        return ConvOverTime1x1(context, dim, t, convOWeight, convOBias, dim);
    }

    /// <summary>Per-timestep 1x1 conv (a Linear applied independently at every timestep), input/output channel-first.</summary>
    public static float[] Conv1x1(float[] input, int inCh, int t, float[] weight, float[] bias, int outCh) =>
        ConvOverTime1x1(input, inCh, t, weight, bias, outCh);

    public static unsafe float[] ConvOverTime1x1(float[] input, int inCh, int t, float[] weight, float[] bias, int outCh)
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

    /// <summary>"Same"-padded Conv1d (odd kernel, pad = kernel/2), channel-first in/out.</summary>
    public static float[] Conv1dSamePad(float[] input, int inCh, int t, float[] weight, float[] bias, int outCh, int kernel)
    {
        int pad = kernel / 2;
        var output = new float[outCh * t];
        Parallel.For(0, outCh, oc =>
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

    public static float[] LayerNormChannelFirst(float[] x, int ch, int t, float[] gamma, float[] beta, float eps = 1e-5f)
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

    public static void SoftmaxInPlace(float[] scores)
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
