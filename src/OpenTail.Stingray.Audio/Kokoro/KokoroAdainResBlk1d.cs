
namespace OpenTail.Stingray.Audio.Kokoro;

/// <summary>
/// StyleTTS2 AdainResBlk1d (istftnet.py -- NOT to be confused with the similarly-named but
/// distinct `AdaINResBlock1`, which is Snake1D-activated and only used inside the Generator's
/// resblocks/noise_res, see <see cref="ResBlockWeights"/>). This class backs
/// ProsodyPredictor.F0/N and Decoder.encode/decode: LeakyReLU(0.2)-activated, with an
/// AdaIN1d (per-CHANNEL instance-norm, i.e. normalized over T not over channels -- distinct
/// from AdaLayerNorm's per-timestep channel-norm) before each conv, an optional depthwise
/// "pool" ConvTranspose1d upsample inside the residual branch, and a plain nearest-neighbor
/// x2 upsample (+ optional learned 1x1 conv) in the shortcut branch.
/// </summary>
public static class KokoroAdainResBlk1d
{
    private const float LeakyReluSlope = 0.2f;
    private const float InstanceNormEps = 1e-5f;

    /// <summary>x is channel-first [dimIn, T]. Returns channel-first [dimOut, TOut] (TOut = 2*T if w.PoolWeight is set, else T).</summary>
    public static float[] Forward(AdainResBlk1dWeights w, float[] x, int dimIn, int dimOut, int t, float[] style, int styleDim)
    {
        // _residual(x, s)
        var xt = AdaIn1d(x, style, w.Adain1Weight, w.Adain1Bias, dimIn, styleDim, t);
        LeakyReluInPlace(xt);
        int poolT = t;
        if (w.PoolWeight is not null)
        {
            xt = DepthwiseConvTranspose1d(xt, w.PoolWeight, w.PoolBias!, dimIn, t, kernel: 3, stride: 2, padding: 1, outputPadding: 1);
            poolT = t * 2;
        }
        xt = Conv1d(xt, w.Conv1Weight, w.Conv1Bias, dimIn, dimOut, poolT, kernel: 3, padding: 1);
        xt = AdaIn1d(xt, style, w.Adain2Weight, w.Adain2Bias, dimOut, styleDim, poolT);
        LeakyReluInPlace(xt);
        xt = Conv1d(xt, w.Conv2Weight, w.Conv2Bias, dimOut, dimOut, poolT, kernel: 3, padding: 1);

        // _shortcut(x)
        float[] xs = w.PoolWeight is not null ? NearestUpsample2x(x, dimIn, t) : x;
        int shortcutT = w.PoolWeight is not null ? t * 2 : t;
        if (w.Conv1x1Weight is not null)
            xs = Conv1dNoBias(xs, w.Conv1x1Weight, dimIn, dimOut, shortcutT, kernel: 1, padding: 0);

        float invSqrt2 = 1f / MathF.Sqrt(2f);
        var output = new float[dimOut * poolT];
        for (int i = 0; i < output.Length; i++)
            output[i] = (xt[i] + xs[i]) * invSqrt2;
        return output;
    }

    /// <summary>AdaIN1d: InstanceNorm1d (per-channel, over T, affine=False) then (1+gamma)*norm+beta from fc(style).</summary>
    private static float[] AdaIn1d(float[] x, float[] style, float[] fcWeight, float[] fcBias, int channels, int styleDim, int t)
    {
        var h = new float[2 * channels];
        for (int o = 0; o < 2 * channels; o++)
        {
            float sum = fcBias[o];
            int wBase = o * styleDim;
            for (int d = 0; d < styleDim; d++)
                sum += fcWeight[wBase + d] * style[d];
            h[o] = sum;
        }

        var output = new float[channels * t];
        for (int c = 0; c < channels; c++)
        {
            double mean = 0;
            for (int ti = 0; ti < t; ti++) mean += x[c * t + ti];
            mean /= t;

            double variance = 0;
            for (int ti = 0; ti < t; ti++)
            {
                double d = x[c * t + ti] - mean;
                variance += d * d;
            }
            variance /= t;
            float invStd = (float)(1.0 / Math.Sqrt(variance + InstanceNormEps));

            float gamma = h[c];
            float beta = h[channels + c];
            for (int ti = 0; ti < t; ti++)
            {
                float normed = (float)((x[c * t + ti] - mean) * invStd);
                output[c * t + ti] = (1f + gamma) * normed + beta;
            }
        }
        return output;
    }

    private static void LeakyReluInPlace(float[] x)
    {
        for (int i = 0; i < x.Length; i++)
            if (x[i] < 0f) x[i] *= LeakyReluSlope;
    }

    // Scale-and-shift-add vectorization -- see ChatterboxCfmDecoder.cs/ChatterboxVocoder.cs for
    // the full rationale (14.7x win there): for a fixed (oc,ic,k), `output[ti] += w*input[ti+shift]`
    // over the valid ti range is one TensorPrimitives.MultiplyAdd over a contiguous span, instead
    // of a per-timestep strided scalar sum with a bounds check on every element.
    private static float[] Conv1d(float[] input, float[] weight, float[] bias, int inCh, int outCh, int t, int kernel, int padding)
    {
        var output = new float[outCh * t];
        System.Threading.Tasks.Parallel.For(0, outCh, oc =>
        {
            var outRow = new float[t];
            Array.Fill(outRow, bias[oc]);
            int wOcBase = oc * inCh * kernel;
            for (int ic = 0; ic < inCh; ic++)
            {
                var inRow = input.AsSpan(ic * t, t);
                int wBase = wOcBase + ic * kernel;
                for (int k = 0; k < kernel; k++)
                    AxpyShifted(inRow, weight[wBase + k], outRow, k - padding, t);
            }
            Array.Copy(outRow, 0, output, oc * t, t);
        });
        return output;
    }

    private static float[] Conv1dNoBias(float[] input, float[] weight, int inCh, int outCh, int t, int kernel, int padding)
    {
        var output = new float[outCh * t];
        System.Threading.Tasks.Parallel.For(0, outCh, oc =>
        {
            var outRow = new float[t];
            int wOcBase = oc * inCh * kernel;
            for (int ic = 0; ic < inCh; ic++)
            {
                var inRow = input.AsSpan(ic * t, t);
                int wBase = wOcBase + ic * kernel;
                for (int k = 0; k < kernel; k++)
                    AxpyShifted(inRow, weight[wBase + k], outRow, k - padding, t);
            }
            Array.Copy(outRow, 0, output, oc * t, t);
        });
        return output;
    }

    /// <summary>UpSample1d('linear'/default StyleTTS2 usage): nearest-neighbor x2 along T.</summary>
    private static float[] NearestUpsample2x(float[] input, int channels, int t)
    {
        var output = new float[channels * t * 2];
        for (int c = 0; c < channels; c++)
            for (int ti = 0; ti < t; ti++)
            {
                float v = input[c * t + ti];
                output[c * t * 2 + ti * 2] = v;
                output[c * t * 2 + ti * 2 + 1] = v;
            }
        return output;
    }

    /// <summary>
    /// Depthwise (groups=channels) ConvTranspose1d, weight [channels, 1, kernel] row-major.
    /// Output length = (T-1)*stride - 2*padding + kernel + outputPadding.
    /// </summary>
    private static float[] DepthwiseConvTranspose1d(float[] input, float[] weight, float[] bias, int channels, int t, int kernel, int stride, int padding, int outputPadding)
    {
        int tOut = (t - 1) * stride - 2 * padding + kernel + outputPadding;
        var output = new float[channels * tOut];
        for (int c = 0; c < channels; c++)
        {
            int outBase = c * tOut;
            for (int i = 0; i < tOut; i++) output[outBase + i] = bias[c];
            int wBase = c * kernel;
            int srcBase = c * t;
            for (int ti = 0; ti < t; ti++)
            {
                float v = input[srcBase + ti];
                int outStart = ti * stride - padding;
                for (int k = 0; k < kernel; k++)
                {
                    int to = outStart + k;
                    if ((uint)to >= (uint)tOut) continue;
                    output[outBase + to] += v * weight[wBase + k];
                }
            }
        }
        return output;
    }

    /// <summary>output[ti] += scale * input[ti + shift] for every ti where ti+shift is in [0,t).</summary>
    private static void AxpyShifted(ReadOnlySpan<float> input, float scale, Span<float> output, int shift, int t)
    {
        int start = Math.Max(0, -shift);
        int end = Math.Min(t, t - shift);
        int len = end - start;
        if (len <= 0) return;
        var inSlice = input.Slice(start + shift, len);
        var outSlice = output.Slice(start, len);
        TensorPrimitives.MultiplyAdd(inSlice, scale, outSlice, outSlice);
    }
}
