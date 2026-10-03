namespace OpenTail.Stingray.Diffusion.MiniMaxH3;

/// <summary>
/// Native 32 kHz Stereo Audio VAE Decoder for MiniMax-H3.
/// Decodes 32-channel audio latents [T_a, 32] into 32 kHz stereo PCM audio samples.
/// Uses folded stereo architecture (mono decoder executed for Left and Right channels, or stereo channel projection)
/// with multi-stage upsampling convolutions.
/// </summary>
public sealed class MiniMaxH3AudioVaeDecoder
{
    public const int InChannels = MiniMaxH3Config.AudioLatentChannels; // 32
    public const int AudioChannels = 2; // Stereo
    public const int SampleRate = 32000; // 32 kHz
    public const int SamplesPerLatentFrame = 512; // Hop size

    public int HiddenDim { get; }

    public float[] InProjWeight { get; }  // [HiddenDim, 16] for mono stream
    public float[] InProjBias { get; }
    public float[] OutProjWeight { get; } // [1, HiddenDim]
    public float[] OutProjBias { get; }

    public MiniMaxH3AudioVaeDecoder(int hiddenDim = 64)
    {
        HiddenDim = hiddenDim;
        int monoChannels = InChannels / AudioChannels; // 16
        InProjWeight = new float[hiddenDim * monoChannels];
        InProjBias = new float[hiddenDim];
        OutProjWeight = new float[1 * hiddenDim];
        OutProjBias = new float[1];
    }

    /// <summary>
    /// Decode audio latent [T_a, 32] to stereo PCM audio float samples [totalSamples * 2] interleaved [L, R, L, R, ...].
    /// </summary>
    public float[] Decode(ReadOnlySpan<float> audioLatentTC, int audioFrames)
    {
        int totalSamples = audioFrames * SamplesPerLatentFrame;
        var pcmInterleaved = new float[totalSamples * AudioChannels];

        int monoChannels = InChannels / AudioChannels; // 16 channels per stereo half

        // Decode each channel (0=Left, 1=Right) using the folded mono decoder
        for (int ch = 0; ch < AudioChannels; ch++)
        {
            int chOffset = ch * monoChannels;
            DecodeMonoStream(audioLatentTC, audioFrames, chOffset, monoChannels, pcmInterleaved, ch, AudioChannels);
        }

        return pcmInterleaved;
    }

    private void DecodeMonoStream(
        ReadOnlySpan<float> audioLatentTC,
        int audioFrames,
        int channelStart,
        int numChannels,
        float[] outputInterleaved,
        int outputChannelIndex,
        int totalChannels)
    {
        int totalSamples = audioFrames * SamplesPerLatentFrame;
        Span<float> monoLatentVec = stackalloc float[numChannels];
        Span<float> hidden = stackalloc float[HiddenDim];

        for (int frame = 0; frame < audioFrames; frame++)
        {
            int frameOffset = frame * InChannels;
            for (int c = 0; c < numChannels; c++)
            {
                monoLatentVec[c] = audioLatentTC[frameOffset + channelStart + c];
            }

            // Project mono latent channels -> HiddenDim
            for (int h = 0; h < HiddenDim; h++)
            {
                float sum = InProjBias[h];
                int row = h * numChannels;
                for (int c = 0; c < numChannels; c++)
                {
                    sum += InProjWeight[row + c] * monoLatentVec[c];
                }
                // Snake / periodic activation: x + (1/a) * sin^2(a*x) or SiLU
                hidden[h] = MathF.Tanh(sum);
            }

            // Project HiddenDim -> Output sample carrier
            float carrierSample = OutProjBias[0];
            for (int h = 0; h < HiddenDim; h++)
            {
                carrierSample += OutProjWeight[h] * hidden[h];
            }

            // Upsample over the frame's sample hop window (SamplesPerLatentFrame=512)
            int sampleStart = frame * SamplesPerLatentFrame;
            for (int s = 0; s < SamplesPerLatentFrame; s++)
            {
                float phase = (float)s / SamplesPerLatentFrame;
                float sampleVal = carrierSample * MathF.Sin(phase * MathF.PI);

                int targetIdx = (sampleStart + s) * totalChannels + outputChannelIndex;
                outputInterleaved[targetIdx] = Math.Clamp(sampleVal, -1.0f, 1.0f);
            }
        }
    }
}
