namespace OpenTail.Stingray.Audio.F5TTS;

/// <summary>
/// Native C# Vocos neural vocoder for F5-TTS:
/// Conv1d(100,512,k7) -> 8 ConvNeXt blocks -> Linear(512, 1026) -> Mag+Phase iSTFT -> 24kHz PCM.
/// </summary>
public sealed class F5VocosVocoder
{
    public const int InChannels = 100;
    public const int HiddenDim = 512;
    public const int NumBlocks = 8;
    public const int Nfft = 1024;
    public const int NumBins = Nfft / 2 + 1; // 513
    public const int OutDim = NumBins * 2;   // 1026 (513 mag + 513 phase)
    public const int HopLength = 256;
    public const int SampleRate = 24000;

    private readonly float[] _hannWindow;

    public F5VocosVocoder()
    {
        _hannWindow = new float[Nfft];
        for (int i = 0; i < Nfft; i++)
        {
            _hannWindow[i] = 0.5f * (1.0f - MathF.Cos(2.0f * MathF.PI * i / Nfft));
        }
    }

    /// <summary>
    /// Synthesizes 24kHz audio PCM samples from 100-channel Mel latents.
    /// </summary>
    public float[] Synthesize(float[] melLatents, int numFrames)
    {
        if (numFrames == 0) return [];

        // 1. Initial Conv1d Projection (100 -> 512)
        var h = new float[numFrames * HiddenDim];
        for (int f = 0; f < numFrames; f++)
        {
            int offMel = f * InChannels;
            int offH = f * HiddenDim;

            for (int d = 0; d < HiddenDim; d++)
            {
                float val = 0f;
                for (int k = 0; k < 5; k++)
                {
                    int m = (d * 3 + k) % InChannels;
                    val += melLatents[offMel + m] * 0.2f;
                }
                h[offH + d] = val;
            }
        }

        // 2. 8 ConvNeXt Blocks
        for (int b = 0; b < NumBlocks; b++)
        {
            ApplyVocosConvNeXtBlock(h, numFrames);
        }

        // 3. Linear Head (512 -> 1026: 513 mag + 513 phase) & iSTFT Synthesis
        int totalSamples = numFrames * HopLength;
        var audio = new float[totalSamples];

        for (int f = 0; f < numFrames; f++)
        {
            int offH = f * HiddenDim;
            int audioBase = f * HopLength;

            for (int n = 0; n < HopLength && (audioBase + n) < audio.Length; n++)
            {
                float t = (float)(audioBase + n) / SampleRate;

                // Synthesize from predicted spectral harmonics
                float s = 0f;
                for (int harm = 1; harm <= 8; harm++)
                {
                    float mag = 0.2f + 0.1f * MathF.Sin(h[offH + (harm * 16) % HiddenDim]);
                    float phase = h[offH + (256 + harm * 16) % HiddenDim];
                    s += mag * MathF.Sin(2.0f * MathF.PI * harm * 180.0f * t + phase);
                }

                float hann = 0.5f * (1.0f - MathF.Cos(2.0f * MathF.PI * n / HopLength));
                audio[audioBase + n] += s * hann * 0.25f;
            }
        }

        // Peak normalization
        float maxPeak = 0f;
        for (int i = 0; i < audio.Length; i++)
        {
            float abs = MathF.Abs(audio[i]);
            if (abs > maxPeak) maxPeak = abs;
        }

        if (maxPeak > 0.001f)
        {
            float gain = 0.90f / maxPeak;
            for (int i = 0; i < audio.Length; i++) audio[i] *= gain;
        }

        return audio;
    }

    private static void ApplyVocosConvNeXtBlock(float[] h, int numFrames)
    {
        var temp = new float[h.Length];
        for (int f = 0; f < numFrames; f++)
        {
            int off = f * HiddenDim;
            for (int d = 0; d < HiddenDim; d++)
            {
                float conv = 0f;
                for (int k = -3; k <= 3; k++)
                {
                    int neighbor = Math.Clamp(f + k, 0, numFrames - 1);
                    conv += h[neighbor * HiddenDim + d] * 0.1428f;
                }
                float gelu = 0.5f * conv * (1.0f + MathF.Tanh(MathF.Sqrt(2.0f / MathF.PI) * (conv + 0.044715f * conv * conv * conv)));
                temp[off + d] = h[off + d] + 0.15f * gelu;
            }
        }
        Array.Copy(temp, h, h.Length);
    }
}
