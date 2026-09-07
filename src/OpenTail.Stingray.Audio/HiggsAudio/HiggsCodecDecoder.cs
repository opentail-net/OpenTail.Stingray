namespace OpenTail.Stingray.Audio.HiggsAudio;

/// <summary>
/// Real forward pass for Higgs Audio TTS's acoustic codec DECODE path, ported from `codec.cpp`'s
/// `quantizer_decode`/`acoustic_decoder` (not guessed): sum of 8 real RVQ-level embedding
/// lookups projected into a shared 1024-dim space, then a standard (NON-causal, symmetric-
/// padding) DAC-lineage decoder -- `Linear(1024-&gt;256) -&gt; Conv1d(256-&gt;1024,k=7) -&gt; 5x
/// [Snake -&gt; ConvTranspose1d(ratio) -&gt; 3x ResidualUnit(dilations 1/3/9)] -&gt; Snake -&gt;
/// Conv1d(32-&gt;1,k=7)`. Genuinely different convolution convention from every DAC-lineage
/// decoder ported elsewhere this session (OmniVoice/MOSS-TTS-Nano/VoxCPM2 are all CAUSAL,
/// left-only padding) -- this one uses real symmetric padding and PyTorch's standard
/// `ConvTranspose1d(padding, output_padding)` semantics directly (confirmed via
/// `conv_transpose_with_adjusted_output_padding`'s real crop-length formula, reduced here to the
/// mathematically equivalent direct index mapping rather than the reference's
/// build-full-then-crop approach).
/// </summary>
public static class HiggsCodecDecoder
{
    /// <summary>Decodes `codes[frames][8]` (RVQ code ids per frame, each codebook a value in
    /// `[0, 1024)`) into a mono 24kHz waveform.</summary>
    public static float[] Decode(HiggsCodecDecoderWeights w, int[][] codes)
    {
        int frames = codes.Length;
        var quantized = QuantizerDecode(w, codes); // [frames][1024]

        // fc2: Linear(1024 -> 256) per frame, then transpose to channel-major [256][frames].
        var acoustic = new float[HiggsCodecDecoderWeights.AcousticHiddenSize][];
        for (int c = 0; c < HiggsCodecDecoderWeights.AcousticHiddenSize; c++) acoustic[c] = new float[frames];
        for (int t = 0; t < frames; t++)
        {
            var projected = Linear(quantized[t], w.AcousticProjectWeight, w.AcousticProjectBias,
                HiggsCodecDecoderWeights.CodecHiddenSize, HiggsCodecDecoderWeights.AcousticHiddenSize);
            for (int c = 0; c < HiggsCodecDecoderWeights.AcousticHiddenSize; c++) acoustic[c][t] = projected[c];
        }

        var hidden = Conv1d(acoustic, HiggsCodecDecoderWeights.AcousticHiddenSize, HiggsCodecDecoderWeights.DecoderChannels[0],
            w.DecoderInputConvWeight, w.DecoderInputConvBias, kernel: 7, stride: 1, padding: 3, dilation: 1);

        for (int block = 0; block < HiggsCodecDecoderWeights.UpsampleRatios.Length; block++)
        {
            int inChannels = HiggsCodecDecoderWeights.DecoderChannels[block];
            int outChannels = HiggsCodecDecoderWeights.DecoderChannels[block + 1];
            int ratio = HiggsCodecDecoderWeights.UpsampleRatios[block];
            var bw = w.DecoderBlocks[block];

            hidden = Snake(hidden, inChannels, bw.SnakeAlpha);
            int padding = (ratio + 1) / 2;
            int outputPadding = ratio % 2;
            hidden = ConvTranspose1d(hidden, inChannels, outChannels, bw.ConvTransposeWeight, bw.ConvTransposeBias,
                kernel: 2 * ratio, stride: ratio, padding: padding, outputPadding: outputPadding);

            for (int unit = 0; unit < 3; unit++)
            {
                int dilation = HiggsCodecDecoderWeights.ResidualDilations[unit];
                hidden = ResidualUnit(hidden, outChannels, bw.ResidualUnits[unit], dilation);
            }
        }

        hidden = Snake(hidden, HiggsCodecDecoderWeights.DecoderChannels[^1], w.DecoderOutputSnakeAlpha);
        var output = Conv1d(hidden, HiggsCodecDecoderWeights.DecoderChannels[^1], 1,
            w.DecoderOutputConvWeight, w.DecoderOutputConvBias, kernel: 7, stride: 1, padding: 3, dilation: 1);
        return output[0];
    }

    private static float[][] QuantizerDecode(HiggsCodecDecoderWeights w, int[][] codes)
    {
        int frames = codes.Length;
        var sum = new float[frames][];
        for (int t = 0; t < frames; t++) sum[t] = new float[HiggsCodecDecoderWeights.CodecHiddenSize];

        for (int cb = 0; cb < HiggsCodecDecoderWeights.NumCodebooks; cb++)
        {
            var q = w.Quantizers[cb];
            for (int t = 0; t < frames; t++)
            {
                int id = codes[t][cb];
                var embed = new float[HiggsCodecDecoderWeights.CodebookDim];
                Array.Copy(q.CodebookEmbed, (long)id * HiggsCodecDecoderWeights.CodebookDim, embed, 0, HiggsCodecDecoderWeights.CodebookDim);
                var projected = Linear(embed, q.ProjectOutWeight, q.ProjectOutBias,
                    HiggsCodecDecoderWeights.CodebookDim, HiggsCodecDecoderWeights.CodecHiddenSize);
                for (int c = 0; c < HiggsCodecDecoderWeights.CodecHiddenSize; c++) sum[t][c] += projected[c];
            }
        }
        return sum;
    }

    private static float[][] ResidualUnit(float[][] x, int channels, HiggsCodecResidualUnitWeights w, int dilation)
    {
        var h = Snake(x, channels, w.Snake1Alpha);
        h = Conv1d(h, channels, channels, w.Conv1Weight, w.Conv1Bias, kernel: 7, stride: 1, padding: 3 * dilation, dilation: dilation);
        h = Snake(h, channels, w.Snake2Alpha);
        h = Conv1d(h, channels, channels, w.Conv2Weight, w.Conv2Bias, kernel: 1, stride: 1, padding: 0, dilation: 1);

        int t = x[0].Length;
        var output = new float[channels][];
        for (int c = 0; c < channels; c++)
        {
            output[c] = new float[t];
            for (int i = 0; i < t; i++) output[c][i] = x[c][i] + h[c][i];
        }
        return output;
    }

    /// <summary>Real Snake activation: `x + sin(alpha*x)^2/(alpha+1e-9)`, per-channel alpha.</summary>
    private static float[][] Snake(float[][] x, int channels, float[] alpha)
    {
        int t = x[0].Length;
        var output = new float[channels][];
        for (int c = 0; c < channels; c++)
        {
            output[c] = new float[t];
            float a = alpha[c];
            float denom = a + 1e-9f;
            for (int i = 0; i < t; i++)
            {
                float s = MathF.Sin(a * x[c][i]);
                output[c][i] = x[c][i] + (s * s) / denom;
            }
        }
        return output;
    }

    private static float[][] Conv1d(float[][] input, int inChannels, int outChannels, float[] weight, float[] bias,
        int kernel, int stride, int padding, int dilation)
    {
        int inLen = input[0].Length;
        int outLen = (inLen + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1;
        var output = new float[outChannels][];
        for (int oc = 0; oc < outChannels; oc++)
        {
            output[oc] = new float[outLen];
            int wBaseOc = oc * inChannels * kernel;
            for (int o = 0; o < outLen; o++)
            {
                float sum = bias[oc];
                int start = o * stride - padding;
                for (int ic = 0; ic < inChannels; ic++)
                {
                    int wBase = wBaseOc + ic * kernel;
                    var inRow = input[ic];
                    for (int k = 0; k < kernel; k++)
                    {
                        int idx = start + k * dilation;
                        if ((uint)idx >= (uint)inLen) continue;
                        sum += weight[wBase + k] * inRow[idx];
                    }
                }
                output[oc][o] = sum;
            }
        }
        return output;
    }

    /// <summary>Standard PyTorch `ConvTranspose1d(padding, output_padding)` semantics, weight
    /// layout `[inChannels, outChannels, kernel]`.</summary>
    private static float[][] ConvTranspose1d(float[][] input, int inChannels, int outChannels, float[] weight, float[] bias,
        int kernel, int stride, int padding, int outputPadding)
    {
        int inLen = input[0].Length;
        int outLen = (inLen - 1) * stride - 2 * padding + kernel + outputPadding;
        var output = new float[outChannels][];
        for (int oc = 0; oc < outChannels; oc++) output[oc] = new float[outLen];
        for (int oc = 0; oc < outChannels; oc++)
            for (int o = 0; o < outLen; o++) output[oc][o] = bias[oc];

        for (int ic = 0; ic < inChannels; ic++)
        {
            var inRow = input[ic];
            int wBaseIc = ic * outChannels * kernel;
            for (int i = 0; i < inLen; i++)
            {
                float v = inRow[i];
                if (v == 0f) continue;
                int baseOut = i * stride - padding;
                for (int k = 0; k < kernel; k++)
                {
                    int o = baseOut + k;
                    if ((uint)o >= (uint)outLen) continue;
                    for (int oc = 0; oc < outChannels; oc++)
                        output[oc][o] += weight[wBaseIc + oc * kernel + k] * v;
                }
            }
        }
        return output;
    }

    private static float[] Linear(float[] input, float[] weight, float[] bias, int inDim, int outDim)
    {
        var output = new float[outDim];
        for (int o = 0; o < outDim; o++)
        {
            float sum = bias[o];
            int wBase = o * inDim;
            for (int i = 0; i < inDim; i++) sum += weight[wBase + i] * input[i];
            output[o] = sum;
        }
        return output;
    }
}
