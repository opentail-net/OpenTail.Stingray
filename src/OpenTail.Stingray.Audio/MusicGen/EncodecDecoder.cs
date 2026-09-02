
namespace OpenTail.Stingray.Audio.MusicGen;

/// <summary>
/// Real EnCodec 32kHz decoder forward pass, transcribed from the real `transformers`
/// `modeling_encodec.py` (`EncodecDecoder`/`EncodecResnetBlock`/`EncodecLSTM`) -- see
/// <see cref="EncodecDecoderWeights"/>'s doc comment for the real layer-index derivation from
/// the checkpoint's own tensor shapes.
///
/// <para><b>Real, easy-to-get-wrong quirks confirmed from the real source</b>: (1) the 2-layer
/// LSTM has a real residual/skip connection around the WHOLE stack (`y, _ = self.lstm(x); return
/// y + x`), not per-layer. (2) `use_conv_shortcut: false` for this 32kHz checkpoint's residual
/// blocks means the shortcut path is a plain identity, not a learned 1x1 conv (unlike DAC, which
/// always uses identity too, so this matches -- but do not assume it generalizes to every
/// EnCodec config). (3) The upsampling `SConvTranspose1d` trims ONLY the right side of its raw
/// (unpadded) transpose-conv output by `kernel - stride` samples (`trim_right_ratio: 1.0`),
/// which simplifies to exactly `outputLength == inputLength * stride` -- see
/// <see cref="TrimmedConvTranspose1d"/>. (4) Non-transpose convs use real EnCodec's
/// non-causal-path symmetric padding (`use_causal_conv: false`, and every conv here has stride=1
/// with an input length that already divides evenly, so the "extra padding for exact division"
/// term the real source also computes is always zero) -- same im2col symmetric-pad convolution
/// shape as <see cref="Parler.DacDecoder"/>'s `FullConv1d`, reused verbatim.</para>
/// </summary>
public static class EncodecDecoder
{
    /// <summary>Real `ResidualVectorQuantizer.decode`: per-codebook embedding lookup, summed across all 4 codebooks at the same time resolution (no repeat/upsample step -- EnCodec's codebooks all run at the same 50Hz frame rate).</summary>
    public static float[] QuantizerDecode(EncodecDecoderWeights w, int[][] codes)
    {
        int t = codes[0].Length;
        int dim = EncodecDecoderWeights.LatentDim;
        var z = new float[dim * t]; // [channel, time] layout throughout this file

        for (int q = 0; q < codes.Length; q++)
        {
            var codebook = w.Codebooks[q];
            for (int ti = 0; ti < t; ti++)
            {
                int code = codes[q][ti];
                int cbBase = code * dim;
                for (int d = 0; d < dim; d++) z[d * t + ti] += codebook[cbBase + d];
            }
        }
        return z;
    }

    /// <summary>Full real decode: 4 codebook streams -&gt; quantizer sum -&gt; decoder stack -&gt; mono float32 PCM at 32kHz. NOT tanh-clamped (unlike DAC) -- real EnCodec returns the raw final conv output.</summary>
    public static float[] Decode(EncodecDecoderWeights w, int[][] codes)
    {
        var z = QuantizerDecode(w, codes);
        int t = codes[0].Length;

        var x = FullConv1d(z, EncodecDecoderWeights.LatentDim, EncodecDecoderWeights.ChannelsPerStage[0], t, w.InitConvWeight, w.InitConvBias, kernel: MusicGenConfig.EncodecKernelSize, dilation: 1, padding: 3);

        x = LstmWithResidual(x, w.Lstm, t, EncodecDecoderWeights.ChannelsPerStage[0]);

        int ch = EncodecDecoderWeights.ChannelsPerStage[0];
        int curT = t;
        for (int i = 0; i < 4; i++)
        {
            x = Elu(x);
            int outCh = EncodecDecoderWeights.ChannelsPerStage[i + 1];
            int ratio = EncodecDecoderWeights.Ratios[i];
            int kernel = 2 * ratio;
            (x, curT) = TrimmedConvTranspose1d(x, ch, outCh, curT, w.Stages[i].UpsampleWeight, w.Stages[i].UpsampleBias, kernel, ratio);
            x = ResidualBlock(x, outCh, curT, w.Stages[i]);
            ch = outCh;
        }

        x = Elu(x);
        var pcm2d = FullConv1d(x, ch, 1, curT, w.OutConvWeight, w.OutConvBias, kernel: MusicGenConfig.EncodecLastKernelSize, dilation: 1, padding: 3);
        return pcm2d; // already [1, curT] == mono PCM
    }

    private static float[] ResidualBlock(float[] x, int channels, int t, EncodecUpsampleStageWeights w)
    {
        int hidden = channels / MusicGenConfig.EncodecCompress;
        var y = Elu(x);
        y = FullConv1d(y, channels, hidden, t, w.ResBlockConv0Weight, w.ResBlockConv0Bias, kernel: MusicGenConfig.EncodecResidualKernelSize, dilation: 1, padding: 1);
        y = Elu(y);
        y = FullConv1d(y, hidden, channels, t, w.ResBlockConv1Weight, w.ResBlockConv1Bias, kernel: 1, dilation: 1, padding: 0);

        var output = new float[y.Length];
        for (int i = 0; i < y.Length; i++) output[i] = x[i] + y[i]; // use_conv_shortcut=false: identity shortcut
        return output;
    }

    /// <summary>Real EnCodec `SConvTranspose1d`: raw (unpadded) ConvTranspose1d, then trim `kernel - stride` samples off the RIGHT only (`trim_right_ratio=1.0`) -- which always yields exactly `outT = t * stride`.</summary>
    private static (float[] Data, int T) TrimmedConvTranspose1d(float[] x, int inCh, int outCh, int t, float[] weight, float[] bias, int kernel, int stride)
    {
        var raw = ConvTranspose1dNoPad(x, inCh, outCh, t, weight, bias, kernel, stride);
        int outT = t * stride;
        var trimmed = new float[outCh * outT];
        for (int oc = 0; oc < outCh; oc++)
            Array.Copy(raw, oc * ((t - 1) * stride + kernel), trimmed, oc * outT, outT);
        return (trimmed, outT);
    }

    /// <summary>Real ConvTranspose1d with padding=0/output_padding=0 (PyTorch default), weight layout `[inCh, outCh, kernel]` flat row-major -- same convention as <see cref="Parler.DacDecoder"/>'s `ConvTranspose1d`, just without the padding/cropping it does internally (trimming is handled by <see cref="TrimmedConvTranspose1d"/> here instead).</summary>
    private static float[] ConvTranspose1dNoPad(float[] x, int inCh, int outCh, int t, float[] weight, float[] bias, int kernel, int stride)
    {
        int outT = (t - 1) * stride + kernel;
        var output = new float[outCh * outT];
        Parallel.For(0, outCh, oc =>
        {
            float b = bias[oc];
            int dstBase = oc * outT;
            for (int ti = 0; ti < outT; ti++) output[dstBase + ti] = b;

            for (int ic = 0; ic < inCh; ic++)
            {
                int srcBase = ic * t;
                int wBase = (ic * outCh + oc) * kernel;
                for (int ti = 0; ti < t; ti++)
                {
                    float v = x[srcBase + ti];
                    int outStart = ti * stride;
                    var wSpan = weight.AsSpan(wBase, kernel);
                    var dstSpan = output.AsSpan(dstBase + outStart, kernel);
                    TensorPrimitives.MultiplyAdd(wSpan, v, dstSpan, dstSpan);
                }
            }
        });
        return output;
    }

    /// <summary>Real FULL (non-depthwise) Conv1d, symmetric ("same"-style) padding -- identical technique to <see cref="Parler.DacDecoder"/>'s `FullConv1d` (im2col + per-output-channel dot product).</summary>
    private static unsafe float[] FullConv1d(float[] x, int inCh, int outCh, int t, float[] weight, float[] bias, int kernel, int dilation, int padding)
    {
        int rowLen = inCh * kernel;
        var col = new float[t * rowLen];
        Parallel.For(0, t, ti =>
        {
            int rowBase = ti * rowLen;
            for (int ic = 0; ic < inCh; ic++)
            {
                int xBase = ic * t;
                int rBase = rowBase + ic * kernel;
                for (int k = 0; k < kernel; k++)
                {
                    int src = ti - padding + k * dilation;
                    col[rBase + k] = (uint)src < (uint)t ? x[xBase + src] : 0f;
                }
            }
        });

        var output = new float[outCh * t];
        fixed (float* colPtr = col, weightPtr = weight, outputPtr = output)
        {
            var colPtrLocal = colPtr;
            var weightPtrLocal = weightPtr;
            var outputPtrLocal = outputPtr;
            Parallel.For(0, outCh, oc =>
            {
                float b = bias[oc];
                float* wOc = weightPtrLocal + oc * rowLen;
                float* outBase = outputPtrLocal + oc * t;
                for (int ti = 0; ti < t; ti++)
                    outBase[ti] = b + SimdKernels.DotF32(wOc, colPtrLocal + ti * rowLen, rowLen);
            });
        }
        return output;
    }

    /// <summary>Real 2-layer unidirectional LSTM with a residual/skip connection around the whole stack (`y, _ = lstm(x); return y + x`). `x` is `[channels, t]`; internally transposed to `[t, channels]` for the recurrence, then back.</summary>
    private static float[] LstmWithResidual(float[] x, EncodecLstmWeights w, int t, int channels)
    {
        var xt = new float[t * channels]; // [t, channels]
        for (int c = 0; c < channels; c++)
            for (int ti = 0; ti < t; ti++)
                xt[ti * channels + c] = x[c * t + ti];

        var l0 = LstmLayer(xt, w.WeightIhL0, w.WeightHhL0, w.BiasIhL0, w.BiasHhL0, t, channels, channels);
        var l1 = LstmLayer(l0, w.WeightIhL1, w.WeightHhL1, w.BiasIhL1, w.BiasHhL1, t, channels, channels);

        var output = new float[x.Length]; // back to [channels, t], with residual add
        for (int c = 0; c < channels; c++)
            for (int ti = 0; ti < t; ti++)
                output[c * t + ti] = x[c * t + ti] + l1[ti * channels + c];
        return output;
    }

    private static float[] LstmLayer(float[] input, float[] weightIh, float[] weightHh, float[] biasIh, float[] biasHh, int t, int inputSize, int hiddenSize)
    {
        var h = new float[hiddenSize];
        var c = new float[hiddenSize];
        var gates = new float[4 * hiddenSize];
        var output = new float[t * hiddenSize];

        for (int ti = 0; ti < t; ti++)
        {
            for (int g = 0; g < 4 * hiddenSize; g++)
            {
                float sum = biasIh[g] + biasHh[g];
                int ihRow = g * inputSize;
                for (int d = 0; d < inputSize; d++) sum += weightIh[ihRow + d] * input[ti * inputSize + d];
                int hhRow = g * hiddenSize;
                for (int d = 0; d < hiddenSize; d++) sum += weightHh[hhRow + d] * h[d];
                gates[g] = sum;
            }

            // PyTorch LSTM gate order: input, forget, cell(g), output.
            for (int j = 0; j < hiddenSize; j++)
            {
                float i_g = Sigmoid(gates[j]);
                float f_g = Sigmoid(gates[hiddenSize + j]);
                float g_g = MathF.Tanh(gates[2 * hiddenSize + j]);
                float o_g = Sigmoid(gates[3 * hiddenSize + j]);
                float cNew = f_g * c[j] + i_g * g_g;
                c[j] = cNew;
                h[j] = o_g * MathF.Tanh(cNew);
            }
            Array.Copy(h, 0, output, ti * hiddenSize, hiddenSize);
        }
        return output;
    }

    private static float[] Elu(float[] x)
    {
        var output = new float[x.Length];
        for (int i = 0; i < x.Length; i++)
            output[i] = x[i] > 0f ? x[i] : MathF.Exp(x[i]) - 1f;
        return output;
    }

    private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));
}
