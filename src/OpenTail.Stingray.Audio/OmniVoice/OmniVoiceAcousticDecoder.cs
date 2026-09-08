
namespace OpenTail.Stingray.Audio.OmniVoice;

/// <summary>Real forward pass for OmniVoice's quantizer-decode + acoustic DAC decoder -- see
/// <see cref="OmniVoiceAcousticDecoderWeights"/>'s doc comment for the architecture derivation.
/// Real Snake activation (`x + sin(alpha*x)^2/alpha`, confirmed from the reference's own
/// `Snake1dModule::build`) and `ConvTranspose1d`-then-crop upsampling (crop `[padding,
/// padding+croppedLen)` out of the raw, unpadded transpose-conv output, matching real PyTorch
/// `ConvTranspose1d(padding, output_padding)` semantics exactly).</summary>
public static class OmniVoiceAcousticDecoder
{
    /// <summary>Decodes a real RVQ code sequence (`[frames][8]`, one code per codebook per frame)
    /// into a raw waveform.</summary>
    public static float[] Decode(OmniVoiceAcousticDecoderWeights w, int[][] codes)
    {
        int frames = codes.Length;
        int quantDim = OmniVoiceAcousticDecoderWeights.QuantizerLatentDim;

        // Quantizer decode: sum project_out(codebook.embed[code]) across all 8 codebooks.
        var latentChannelMajor = new float[quantDim * frames];
        for (int q = 0; q < OmniVoiceAcousticDecoderWeights.NumCodebooks; q++)
        {
            var qw = w.Quantizers[q];
            int dim = OmniVoiceAcousticDecoderWeights.CodebookDim;
            for (int t = 0; t < frames; t++)
            {
                int code = codes[t][q];
                var embed = qw.CodebookEmbed.AsSpan(code * dim, dim);
                for (int o = 0; o < quantDim; o++)
                {
                    float sum = qw.ProjectOutBias[o];
                    int wBase = o * dim;
                    for (int d = 0; d < dim; d++) sum += qw.ProjectOutWeight[wBase + d] * embed[d];
                    latentChannelMajor[o * frames + t] += sum;
                }
            }
        }

        // Real fc2 projection: quantDim(1024) -> acoustic latent dim(256), BEFORE conv1 -- found
        // via a real shape mismatch (conv1's real weight requires a 256-wide input, not 1024),
        // see OmniVoiceAcousticDecoderWeights.Fc2Weight's doc comment.
        int acousticDim = OmniVoiceAcousticDecoderWeights.AcousticLatentDim;
        var fc2Out = new float[acousticDim * frames];
        Parallel.For(0, acousticDim, o =>
        {
            float b = w.Fc2Bias[o];
            int wBase = o * quantDim;
            for (int t = 0; t < frames; t++)
            {
                float sum = b;
                for (int i = 0; i < quantDim; i++) sum += w.Fc2Weight[wBase + i] * latentChannelMajor[i * frames + t];
                fc2Out[o * frames + t] = sum;
            }
        });

        // acoustic_decoder.conv1: Conv1d(256->decoderHidden, k=7, pad=3, bias).
        var x = Conv1dSamePad(fc2Out, acousticDim, frames, w.Conv1Weight, w.Conv1Bias, outCh: OmniVoiceAcousticDecoderWeights.DecoderHiddenSize, kernel: 7, dilation: 1);
        int channels = OmniVoiceAcousticDecoderWeights.DecoderHiddenSize;
        int t2 = frames;

        for (int b = 0; b < w.Blocks.Length; b++)
        {
            var block = w.Blocks[b];
            int stride = OmniVoiceAcousticDecoderWeights.UpsamplingRatios[b];
            int padding = (stride + 1) / 2;
            int outputPadding = stride % 2;

            SnakeInPlace(x, block.SnakeAlpha, channels, t2);
            (x, t2) = ConvTranspose1dPyTorch(x, channels, t2, block.ConvTWeight, block.ConvTBias, block.OutChannels, kernel: 2 * stride, stride: stride, padding: padding, outputPadding: outputPadding);
            channels = block.OutChannels;

            x = ResidualUnit(x, channels, t2, block.Res1, dilation: 1);
            x = ResidualUnit(x, channels, t2, block.Res2, dilation: 3);
            x = ResidualUnit(x, channels, t2, block.Res3, dilation: 9);
        }

        SnakeInPlace(x, w.FinalSnakeAlpha, channels, t2);
        var audio = Conv1dSamePad(x, channels, t2, w.Conv2Weight, w.Conv2Bias, outCh: 1, kernel: 7, dilation: 1);
        return audio; // channels=1, so this flat array is already the raw waveform
    }

    private static float[] ResidualUnit(float[] x, int channels, int frames, OmniVoiceResidualUnitWeights r, int dilation)
    {
        var h = (float[])x.Clone();
        SnakeInPlace(h, r.Snake1Alpha, channels, frames);
        int padding = 3 * dilation;
        h = Conv1dSamePad(h, channels, frames, r.Conv1Weight, r.Conv1Bias, outCh: channels, kernel: 7, dilation: dilation, explicitPadding: padding);
        SnakeInPlace(h, r.Snake2Alpha, channels, frames);
        h = Conv1dSamePad(h, channels, frames, r.Conv2Weight, r.Conv2Bias, outCh: channels, kernel: 1, dilation: 1);
        var output = new float[x.Length];
        for (int i = 0; i < x.Length; i++) output[i] = x[i] + h[i];
        return output;
    }

    /// <summary>Real Snake activation: `x + sin(alpha*x)^2/alpha`, alpha broadcast per-channel.</summary>
    private static void SnakeInPlace(float[] x, float[] alpha, int channels, int frames)
    {
        Parallel.For(0, channels, c =>
        {
            float a = alpha[c];
            int baseIdx = c * frames;
            for (int t = 0; t < frames; t++)
            {
                float v = x[baseIdx + t];
                float s = MathF.Sin(a * v);
                x[baseIdx + t] = v + (s * s) / a;
            }
        });
    }

    private static float[] Conv1dSamePad(float[] x, int inCh, int frames, float[] weight, float[] bias, int outCh, int kernel, int dilation, int? explicitPadding = null)
    {
        int pad = explicitPadding ?? ((kernel - 1) / 2) * dilation;
        var output = new float[outCh * frames];
        Parallel.For(0, outCh, oc =>
        {
            float b = bias[oc];
            int wBase = oc * inCh * kernel;
            for (int t = 0; t < frames; t++)
            {
                float sum = b;
                for (int ic = 0; ic < inCh; ic++)
                {
                    int wIcBase = wBase + ic * kernel;
                    for (int k = 0; k < kernel; k++)
                    {
                        int it = t - pad + k * dilation;
                        if ((uint)it >= (uint)frames) continue;
                        sum += weight[wIcBase + k] * x[ic * frames + it];
                    }
                }
                output[oc * frames + t] = sum;
            }
        });
        return output;
    }

    /// <summary>Real PyTorch ConvTranspose1d(stride,kernel,padding,output_padding) -- computes the
    /// unpadded transpose-conv, then crops `[padding, padding+croppedLen)`.</summary>
    private static (float[] Output, int OutLen) ConvTranspose1dPyTorch(float[] x, int inCh, int inLen, float[] weight, float[] bias, int outCh, int kernel, int stride, int padding, int outputPadding)
    {
        int rawLen = (inLen - 1) * stride + kernel;
        var raw = new float[outCh * rawLen];
        for (int oc = 0; oc < outCh; oc++)
            for (int ot = 0; ot < rawLen; ot++) raw[oc * rawLen + ot] = bias[oc];
        for (int ic = 0; ic < inCh; ic++)
        {
            for (int it = 0; it < inLen; it++)
            {
                float v = x[ic * inLen + it];
                if (v == 0f) continue;
                int outBase = stride * it;
                for (int oc = 0; oc < outCh; oc++)
                {
                    int wBase = (ic * outCh + oc) * kernel;
                    for (int k = 0; k < kernel; k++)
                        raw[oc * rawLen + outBase + k] += v * weight[wBase + k];
                }
            }
        }
        int croppedLen = (inLen - 1) * stride - 2 * padding + kernel + outputPadding;
        var output = new float[outCh * croppedLen];
        for (int oc = 0; oc < outCh; oc++)
            Array.Copy(raw, oc * rawLen + padding, output, oc * croppedLen, croppedLen);
        return (output, croppedLen);
    }
}
