
namespace OpenTail.Stingray.Audio.OmniVoice;

/// <summary>Real forward pass for OmniVoice's acoustic DAC-style encoder -- see
/// <see cref="OmniVoiceAcousticEncoderWeights"/>'s doc comment for the architecture derivation.
/// Real Snake activation and strided Conv1d downsampling matching real PyTorch
/// `Conv1d(stride, padding)` semantics.</summary>
public static class OmniVoiceAcousticEncoder
{
    /// <summary>Input: raw waveform samples (real sample rate `sampling_rate`=24000 per the real
    /// checkpoint's `audio_tokenizer/config.json`). Returns channel-major `[256, outFrames]`
    /// acoustic latent (256 = `acoustic_model_config.hidden_size`).</summary>
    public static (float[] Latent, int Frames) Encode(OmniVoiceAcousticEncoderWeights w, ReadOnlySpan<float> waveform)
    {
        int frames = waveform.Length;
        var x = Conv1dSamePad(waveform.ToArray(), 1, frames, w.Conv1Weight, w.Conv1Bias, outCh: OmniVoiceAcousticEncoderWeights.EncoderHiddenSize, kernel: 7, dilation: 1);
        int channels = OmniVoiceAcousticEncoderWeights.EncoderHiddenSize;

        for (int b = 0; b < w.Blocks.Length; b++)
        {
            var block = w.Blocks[b];
            x = ResidualUnit(x, channels, frames, block.Res1, dilation: 1);
            x = ResidualUnit(x, channels, frames, block.Res2, dilation: 3);
            x = ResidualUnit(x, channels, frames, block.Res3, dilation: 9);
            SnakeInPlace(x, block.SnakeAlpha, channels, frames);

            int stride = OmniVoiceAcousticEncoderWeights.DownsamplingRatios[b];
            int padding = (stride + 1) / 2;
            (x, frames) = Conv1dStrided(x, channels, frames, block.ConvWeight, block.ConvBias, block.OutChannels, kernel: 2 * stride, stride: stride, padding: padding);
            channels = block.OutChannels;
        }

        SnakeInPlace(x, w.FinalSnakeAlpha, channels, frames);
        var latent = Conv1dSamePad(x, channels, frames, w.Conv2Weight, w.Conv2Bias, outCh: 256, kernel: 3, dilation: 1);
        return (latent, frames);
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

    private static void SnakeInPlace(float[] x, float[] alpha, int channels, int frames)
    {
        for (int c = 0; c < channels; c++)
        {
            float a = alpha[c];
            int baseIdx = c * frames;
            for (int t = 0; t < frames; t++)
            {
                float v = x[baseIdx + t];
                float s = MathF.Sin(a * v);
                x[baseIdx + t] = v + (s * s) / a;
            }
        }
    }

    private static float[] Conv1dSamePad(float[] x, int inCh, int frames, float[] weight, float[] bias, int outCh, int kernel, int dilation, int? explicitPadding = null)
    {
        int pad = explicitPadding ?? ((kernel - 1) / 2) * dilation;
        var output = new float[outCh * frames];
        for (int oc = 0; oc < outCh; oc++)
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
        }
        return output;
    }

    private static (float[] Output, int OutLen) Conv1dStrided(float[] x, int inCh, int inLen, float[] weight, float[] bias, int outCh, int kernel, int stride, int padding)
    {
        int outLen = (inLen + 2 * padding - kernel) / stride + 1;
        var output = new float[outCh * outLen];
        for (int oc = 0; oc < outCh; oc++)
        {
            float b = bias[oc];
            int wBase = oc * inCh * kernel;
            for (int ot = 0; ot < outLen; ot++)
            {
                float sum = b;
                int start = ot * stride - padding;
                for (int ic = 0; ic < inCh; ic++)
                {
                    int wIcBase = wBase + ic * kernel;
                    for (int k = 0; k < kernel; k++)
                    {
                        int it = start + k;
                        if ((uint)it >= (uint)inLen) continue;
                        sum += weight[wIcBase + k] * x[ic * inLen + it];
                    }
                }
                output[oc * outLen + ot] = sum;
            }
        }
        return (output, outLen);
    }
}
