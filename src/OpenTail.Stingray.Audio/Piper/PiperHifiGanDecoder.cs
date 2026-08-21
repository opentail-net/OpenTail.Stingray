using System;

namespace OpenTail.Stingray.Audio.Piper;

/// <summary>
/// Piper's HiFi-GAN vocoder (dec), forward pass. Standard HiFi-GAN generator, confirmed via ONNX
/// node/attribute inspection: conv_pre (k=7) -&gt; 3x [LeakyRelu -&gt; ConvTranspose1d upsample -&gt;
/// 3-way-averaged "resblock2"-style residual stack (2 convs per block, dilations [1,2], kernels
/// [3,5,7])] -&gt; LeakyRelu -&gt; conv_post (k=7, no bias) -&gt; Tanh.
/// Upsample stages: stride/kernel (8,16), (8,16), (4,8) -- total upsample factor 256, matching
/// Piper's 22050 Hz / ~86 Hz mel frame rate.
/// </summary>
public static class PiperHifiGanDecoder
{
    private const float LeakyReluAlpha = 0.1f;
    private static readonly int[] ResblockKernels = [3, 5, 7];
    // Confirmed via ONNX node attribute inspection: NOT uniform [1,2] across kernel sizes.
    private static readonly int[][] ResblockDilations = [[1, 2], [2, 6], [3, 12]];
    private static readonly int[] UpKernels = [16, 16, 8];
    private static readonly int[] UpStrides = [8, 8, 4];

    /// <summary>zp is the flow's output, channel-first [192, T]. Returns mono waveform samples.</summary>
    public static float[] Forward(PiperOnnxWeights w, float[] zp, int t)
    {
        int ch = 256;
        var x = Conv1dSamePad(zp, w.HiddenDim, t, w.DecConvPreWeight, w.DecConvPreBias, ch, kernel: 7);

        for (int stage = 0; stage < 3; stage++)
        {
            for (int i = 0; i < x.Length; i++) x[i] = LeakyRelu(x[i]);

            int outCh = ch / 2;
            int newT = t * UpStrides[stage];
            x = ConvTranspose1d(x, ch, t, w.DecUpsWeight[stage], w.DecUpsBias[stage], outCh, UpKernels[stage], UpStrides[stage]);
            ch = outCh;
            t = newT;

            float[]? sum = null;
            for (int k = 0; k < 3; k++)
            {
                int rbIndex = stage * 3 + k;
                var rbOut = ResBlockForward(x, ch, t, w.DecResblocks[rbIndex], ResblockKernels[k], ResblockDilations[k]);
                if (sum is null) sum = rbOut;
                else for (int i = 0; i < sum.Length; i++) sum[i] += rbOut[i];
            }
            for (int i = 0; i < sum!.Length; i++) sum[i] /= 3f;
            x = sum;
        }

        for (int i = 0; i < x.Length; i++) x[i] = LeakyRelu(x[i]);
        var post = Conv1dSamePad(x, ch, t, w.DecConvPostWeight, null, 1, kernel: 7);
        for (int i = 0; i < post.Length; i++) post[i] = MathF.Tanh(post[i]);
        return post;
    }

    private static float LeakyRelu(float v) => v >= 0f ? v : v * LeakyReluAlpha;

    private static float[] ResBlockForward(float[] input, int ch, int t, PiperResBlockWeights rb, int kernel, int[] dilations)
    {
        var x = input;
        for (int layer = 0; layer < 2; layer++)
        {
            var y = new float[ch * t];
            for (int i = 0; i < y.Length; i++) y[i] = LeakyRelu(x[i]);
            y = Conv1dDilated(y, ch, t, rb.ConvWeight[layer], rb.ConvBias[layer], ch, kernel, dilations[layer]);
            var next = new float[ch * t];
            for (int i = 0; i < next.Length; i++) next[i] = x[i] + y[i];
            x = next;
        }
        return x;
    }

    private static float[] Conv1dSamePad(float[] input, int inCh, int t, float[] weight, float[]? bias, int outCh, int kernel)
    {
        int pad = kernel / 2;
        return Conv1dDilatedCore(input, inCh, t, weight, bias, outCh, kernel, dilation: 1, pad);
    }

    private static float[] Conv1dDilated(float[] input, int inCh, int t, float[] weight, float[] bias, int outCh, int kernel, int dilation)
    {
        int pad = (kernel * dilation - dilation) / 2;
        return Conv1dDilatedCore(input, inCh, t, weight, bias, outCh, kernel, dilation, pad);
    }

    private static float[] Conv1dDilatedCore(float[] input, int inCh, int t, float[] weight, float[]? bias, int outCh, int kernel, int dilation, int pad)
    {
        var output = new float[outCh * t];
        System.Threading.Tasks.Parallel.For(0, outCh, oc =>
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
    /// index is in-range, vectorized over the valid contiguous span (see docs/audio-review-progress.md's
    /// "loop-reorder-for-vectorization" pattern -- the same helper used across Kokoro/Chatterbox/Whisper).</summary>
    private static void AxpyShifted(float[] output, int dstBase, float[] input, int srcBase, int t, float weight, int shift)
    {
        int tiStart = Math.Max(0, -shift);
        int tiEnd = Math.Min(t, t - shift);
        if (tiStart >= tiEnd) return;
        int len = tiEnd - tiStart;
        var src = new ReadOnlySpan<float>(input, srcBase + tiStart + shift, len);
        var dst = new Span<float>(output, dstBase + tiStart, len);
        System.Numerics.Tensors.TensorPrimitives.MultiplyAdd(src, weight, dst, dst);
    }

    private static float[] ConvTranspose1d(float[] input, int inCh, int t, float[] weight, float[] bias, int outCh, int kernel, int stride)
    {
        // weight shape [inCh, outCh, kernel] (ConvTranspose convention). pad = (kernel-stride)/2
        // (matches this checkpoint's ONNX-exported explicit pads, all symmetric).
        int pad = (kernel - stride) / 2;
        int outT = t * stride;
        var output = new float[outCh * outT];
        System.Threading.Tasks.Parallel.For(0, outCh, oc =>
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
