using System;
using System.Numerics.Tensors;
using System.Threading.Tasks;

namespace OpenTail.Stingray.Audio.Primitives;

/// <summary>
/// Shared low-level conv primitives for the HiFi-GAN family of vocoders (Piper, MeloTTS): dilated
/// "same"-padded Conv1d, ConvTranspose1d (weight_norm-exported [inCh,outCh,kernel] layout), and
/// LeakyReLU. Extracted from the original Piper-only implementation so MeloTTS's `Generator`
/// (a different, more elaborate `ResBlock1`-based topology -- 3 conv pairs per resblock at fixed
/// dilations [1,3,5], vs. whatever simpler resblock shape Piper's checkpoint used) doesn't
/// hand-roll a second copy of the underlying conv math, even though the resblock LOOP STRUCTURE
/// itself differs enough between the two checkpoints that it isn't shared (see
/// <see cref="OpenTail.Stingray.Audio.Piper.PiperHifiGanDecoder"/> vs `MeloGenerator`).
/// </summary>
public static class HifiGanKernels
{
    public static float LeakyRelu(float v, float alpha = 0.1f) => v >= 0f ? v : v * alpha;

    public static float[] Conv1dSamePad(float[] input, int inCh, int t, float[] weight, float[]? bias, int outCh, int kernel) =>
        Conv1dDilated(input, inCh, t, weight, bias, outCh, kernel, dilation: 1);

    public static float[] Conv1dDilated(float[] input, int inCh, int t, float[] weight, float[]? bias, int outCh, int kernel, int dilation)
    {
        int pad = (kernel * dilation - dilation) / 2;
        var output = new float[outCh * t];
        Parallel.For(0, outCh, oc =>
        {
            float b = bias is null ? 0f : bias[oc];
            int wBase = oc * inCh * kernel;
            for (int ti = 0; ti < t; ti++) output[oc * t + ti] = b;

            for (int ic = 0; ic < inCh; ic++)
            {
                int wcBase = wBase + ic * kernel;
                int srcBase = ic * t;
                int dstBase = oc * t;
                for (int k = 0; k < kernel; k++)
                {
                    float weightVal = weight[wcBase + k];
                    int shift = k * dilation - pad;
                    AxpyShifted(output, dstBase, input, srcBase, t, weightVal, shift);
                }
            }
        });
        return output;
    }

    /// <summary>output[dstBase+ti] += weight * input[srcBase+ti+shift] for all ti where the shifted
    /// index is in-range, vectorized over the valid contiguous span.</summary>
    private static void AxpyShifted(float[] output, int dstBase, float[] input, int srcBase, int t, float weight, int shift)
    {
        int tiStart = Math.Max(0, -shift);
        int tiEnd = Math.Min(t, t - shift);
        if (tiStart >= tiEnd) return;
        int len = tiEnd - tiStart;
        var src = new ReadOnlySpan<float>(input, srcBase + tiStart + shift, len);
        var dst = new Span<float>(output, dstBase + tiStart, len);
        TensorPrimitives.MultiplyAdd(src, weight, dst, dst);
    }

    /// <summary>weight shape [inCh, outCh, kernel] (ConvTranspose convention). pad = (kernel-stride)/2
    /// (matches HiFi-GAN's ONNX-exported explicit pads, symmetric for every checkpoint seen so far).</summary>
    public static float[] ConvTranspose1d(float[] input, int inCh, int t, float[] weight, float[] bias, int outCh, int kernel, int stride)
    {
        int pad = (kernel - stride) / 2;
        int outT = t * stride;
        var output = new float[outCh * outT];
        Parallel.For(0, outCh, oc =>
        {
            for (int ti = 0; ti < outT; ti++) output[oc * outT + ti] = bias[oc];

            for (int ic = 0; ic < inCh; ic++)
            {
                int srcBase = ic * t;
                int wBase = ic * outCh * kernel + oc * kernel;
                for (int si = 0; si < t; si++)
                {
                    float inputVal = input[srcBase + si];
                    if (inputVal == 0f) continue;
                    int outStart = si * stride - pad;
                    for (int k = 0; k < kernel; k++)
                    {
                        int oti = outStart + k;
                        if ((uint)oti < (uint)outT) output[oc * outT + oti] += weight[wBase + k] * inputVal;
                    }
                }
            }
        });
        return output;
    }
}
