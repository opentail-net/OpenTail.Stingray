
namespace OpenTail.Stingray.Audio.Primitives;

/// <summary>
/// Shared, ratio-parameterized real EnCodec decoder forward pass, transcribed from the real
/// `transformers` `modeling_encodec.py` (`EncodecDecoder`/`EncodecResnetBlock`/`EncodecLSTM`).
/// Extracted 2026-09-02 as a genuine DRY pass (CLAUDE.md rule 7) once TWO real, independently
/// inspected EnCodec checkpoints turned out to share the byte-for-byte identical layer skeleton
/// (initial conv -&gt; 2-layer LSTM w/ residual -&gt; 4x [upsample-transpose-conv + one residual
/// block] -&gt; final conv, `n_filters=64`/`compress=2`/`kernel_size=7`/`residual_kernel_size=3`/
/// `trim_right_ratio=1.0`/no final activation in both) and differed ONLY in their per-stage
/// upsampling ratios: MusicGen's 32kHz codec uses `[8,5,4,4]`; AudioGen's 16kHz codec (a
/// separately-trained EnCodec for environmental sound, confirmed via its own real
/// `compression_state_dict.bin`'s embedded training config) uses `[8,5,4,2]`.
///
/// <para><b>Real, easy-to-get-wrong quirks confirmed from the real source</b> (see the git history
/// of this file, formerly `MusicGen/EncodecDecoder.cs`, for the original derivation): (1) the
/// 2-layer LSTM has a real residual/skip connection around the WHOLE stack (`y, _ =
/// self.lstm(x); return y + x`), not per-layer. (2) `use_conv_shortcut: false` in both real
/// checkpoints' residual blocks means the shortcut path is a plain identity, not a learned 1x1
/// conv. (3) The upsampling `SConvTranspose1d` trims ONLY the right side of its raw (unpadded)
/// transpose-conv output by `kernel - stride` samples (`trim_right_ratio: 1.0` in both), which
/// simplifies to exactly `outputLength == inputLength * stride`. (4) Non-transpose convs use
/// real EnCodec's non-causal-path symmetric padding.</para>
/// </summary>
public sealed class EncodecDecoderWeights
{
    public required int LatentDim { get; init; } // == codebook_dim, 128 in both real checkpoints seen so far
    public required int[] Ratios { get; init; } // decoder upsample order
    public required int[] ChannelsPerStage { get; init; } // channel count entering each of the 5 top-level stages

    public required float[][] Codebooks { get; init; } // [codebook][codebookSize * LatentDim]

    public required float[] InitConvWeight { get; init; }
    public required float[] InitConvBias { get; init; }

    public required EncodecLstmWeights Lstm { get; init; }

    public required EncodecUpsampleStageWeights[] Stages { get; init; }

    public required float[] OutConvWeight { get; init; }
    public required float[] OutConvBias { get; init; }

    /// <summary>Real SEANet channel progression: starts at `nFilters * 2^numStages` (the innermost/most-downsampled width) and halves at each upsample stage down to `nFilters`. Both real checkpoints seen so far use `nFilters=64`, `numStages=4` -&gt; `[1024,512,256,128,64]`.</summary>
    public static int[] DefaultChannelsPerStage(int nFilters, int numStages)
    {
        var channels = new int[numStages + 1];
        channels[0] = nFilters << numStages;
        for (int i = 1; i <= numStages; i++) channels[i] = channels[i - 1] / 2;
        return channels;
    }

    /// <summary>Folds `weight_g`(`[outCh,1,1]`)*`weight_v`(`[outCh,inCh,K]`)/||v[outCh,:,:]||_2 into a plain conv weight -- PyTorch `weight_norm` (dim=0) convention, same as <see cref="OpenTail.Stingray.Audio.Parler.DacWeights"/>'s identically-named helper.</summary>
    public static float[] FoldConvWeight(SafetensorsLoader loader, string weightGName, string weightVName)
    {
        var g = loader.ReadF32(weightGName);
        var v = loader.ReadF32(weightVName);
        int[] vShape = loader.GetShape(weightVName);
        int outCh = vShape[0];
        int perChannel = v.Length / outCh;

        var folded = new float[v.Length];
        for (int o = 0; o < outCh; o++)
        {
            double sumSq = 0;
            int baseIdx = o * perChannel;
            for (int j = 0; j < perChannel; j++) { double vv = v[baseIdx + j]; sumSq += vv * vv; }
            float norm = (float)Math.Sqrt(sumSq);
            float scale = norm > 1e-12f ? g[o] / norm : 0f;
            for (int j = 0; j < perChannel; j++) folded[baseIdx + j] = v[baseIdx + j] * scale;
        }
        return folded;
    }
}

public sealed class EncodecLstmWeights
{
    public required float[] WeightIhL0 { get; init; }
    public required float[] WeightHhL0 { get; init; }
    public required float[] BiasIhL0 { get; init; }
    public required float[] BiasHhL0 { get; init; }
    public required float[] WeightIhL1 { get; init; }
    public required float[] WeightHhL1 { get; init; }
    public required float[] BiasIhL1 { get; init; }
    public required float[] BiasHhL1 { get; init; }
}

public sealed class EncodecUpsampleStageWeights
{
    public required float[] UpsampleWeight { get; init; } // ConvTranspose1d, [inCh, outCh, K] flat
    public required float[] UpsampleBias { get; init; }
    public required float[] ResBlockConv0Weight { get; init; } // dim -> dim/compress, k=3
    public required float[] ResBlockConv0Bias { get; init; }
    public required float[] ResBlockConv1Weight { get; init; } // dim/compress -> dim, k=1
    public required float[] ResBlockConv1Bias { get; init; }
}

public static class EncodecDecoderKernels
{
    private const int KernelSize = 7;
    private const int ResidualKernelSize = 3;
    private const int Compress = 2; // residual-block bottleneck divisor, same in both real checkpoints seen so far

    /// <summary>Real `ResidualVectorQuantizer.decode`: per-codebook embedding lookup, summed across all codebooks at the same time resolution (no repeat/upsample step -- every codebook stream runs at the same frame rate).</summary>
    public static float[] QuantizerDecode(EncodecDecoderWeights w, int[][] codes)
    {
        int t = codes[0].Length;
        int dim = w.LatentDim;
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

    /// <summary>Full real decode: N codebook streams -&gt; quantizer sum -&gt; decoder stack -&gt; mono float32 PCM. NOT tanh-clamped (unlike DAC) -- real EnCodec returns the raw final conv output.</summary>
    public static float[] Decode(EncodecDecoderWeights w, int[][] codes)
    {
        var z = QuantizerDecode(w, codes);
        int t = codes[0].Length;
        int numStages = w.Ratios.Length;

        var x = FullConv1d(z, w.LatentDim, w.ChannelsPerStage[0], t, w.InitConvWeight, w.InitConvBias, kernel: KernelSize, dilation: 1, padding: 3);

        x = LstmWithResidual(x, w.Lstm, t, w.ChannelsPerStage[0]);

        int ch = w.ChannelsPerStage[0];
        int curT = t;
        for (int i = 0; i < numStages; i++)
        {
            x = Elu(x);
            int outCh = w.ChannelsPerStage[i + 1];
            int ratio = w.Ratios[i];
            int kernel = 2 * ratio;
            (x, curT) = TrimmedConvTranspose1d(x, ch, outCh, curT, w.Stages[i].UpsampleWeight, w.Stages[i].UpsampleBias, kernel, ratio);
            x = ResidualBlock(x, outCh, curT, w.Stages[i]);
            ch = outCh;
        }

        x = Elu(x);
        var pcm2d = FullConv1d(x, ch, 1, curT, w.OutConvWeight, w.OutConvBias, kernel: KernelSize, dilation: 1, padding: 3);
        return pcm2d; // already [1, curT] == mono PCM
    }

    private static float[] ResidualBlock(float[] x, int channels, int t, EncodecUpsampleStageWeights w)
    {
        int hidden = channels / Compress;
        var y = Elu(x);
        y = FullConv1d(y, channels, hidden, t, w.ResBlockConv0Weight, w.ResBlockConv0Bias, kernel: ResidualKernelSize, dilation: 1, padding: 1);
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

    /// <summary>Real ConvTranspose1d with padding=0/output_padding=0 (PyTorch default), weight layout `[inCh, outCh, kernel]` flat row-major.</summary>
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

    /// <summary>Real FULL (non-depthwise) Conv1d, symmetric ("same"-style) padding.</summary>
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
