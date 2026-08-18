namespace OpenTail.Stingray.Audio.F5TTS;

/// <summary>
/// 100-channel Mel-Spectrogram feature extractor for 24000Hz audio in F5-TTS.
/// (n_fft=1024, hop_length=256, win_length=1024, fmin=0, fmax=12000).
/// </summary>
public sealed class F5MelExtractor
{
    public const int NumMels = 100;
    public const int SampleRate = 24000;
    public const int Nfft = 1024;
    public const int HopLength = 256;
    public const int WinLength = 1024;

    private readonly float[][] _melFilters;
    private readonly float[] _hannWindow;

    public F5MelExtractor()
    {
        _hannWindow = new float[WinLength];
        for (int i = 0; i < WinLength; i++)
        {
            _hannWindow[i] = 0.5f * (1.0f - MathF.Cos(2.0f * MathF.PI * i / WinLength));
        }

        _melFilters = CreateMelFilterbank(NumMels, Nfft, SampleRate, 0f, 12000f);
    }

    /// <summary>
    /// Computes 100-channel log Mel-spectrogram from 24kHz audio PCM samples.
    /// Returns 1D flattened array [numFrames * 100].
    /// </summary>
    public float[] ExtractMel(ReadOnlySpan<float> pcm)
    {
        if (pcm.Length < HopLength) return new float[NumMels];

        int numFrames = (pcm.Length - WinLength) / HopLength + 1;
        if (numFrames <= 0) numFrames = 1;

        var mels = new float[numFrames * NumMels];
        int numBins = Nfft / 2 + 1; // 513
        var magSpec = new float[numBins];

        for (int f = 0; f < numFrames; f++)
        {
            int pcmOff = f * HopLength;

            // Compute STFT magnitude for this frame
            for (int k = 0; k < numBins; k++)
            {
                float real = 0f;
                float imag = 0f;
                float freq = 2.0f * MathF.PI * k / Nfft;

                for (int n = 0; n < WinLength; n++)
                {
                    int sampleIdx = pcmOff + n;
                    float sample = (sampleIdx < pcm.Length) ? pcm[sampleIdx] : 0f;
                    float winSample = sample * _hannWindow[n];

                    real += winSample * MathF.Cos(freq * n);
                    imag -= winSample * MathF.Sin(freq * n);
                }

                magSpec[k] = MathF.Sqrt(real * real + imag * imag + 1e-9f);
            }

            // Apply Mel filterbank & log compression
            int melOff = f * NumMels;
            for (int m = 0; m < NumMels; m++)
            {
                float energy = 0f;
                var filter = _melFilters[m];
                for (int k = 0; k < numBins; k++)
                {
                    energy += magSpec[k] * filter[k];
                }

                mels[melOff + m] = MathF.Log(MathF.Max(energy, 1e-5f));
            }
        }

        return mels;
    }

    private static float[][] CreateMelFilterbank(int numMels, int nfft, int sr, float fmin, float fmax)
    {
        int numBins = nfft / 2 + 1;
        var filters = new float[numMels][];
        for (int i = 0; i < numMels; i++) filters[i] = new float[numBins];

        float Mel(float f) => 2595.0f * MathF.Log10(1.0f + f / 700.0f);
        float InvMel(float m) => 700.0f * (MathF.Pow(10.0f, m / 2595.0f) - 1.0f);

        float minMel = Mel(fmin);
        float maxMel = Mel(fmax);
        float melStep = (maxMel - minMel) / (numMels + 1);

        var melPoints = new float[numMels + 2];
        var binPoints = new int[numMels + 2];

        for (int i = 0; i < numMels + 2; i++)
        {
            melPoints[i] = minMel + i * melStep;
            float freq = InvMel(melPoints[i]);
            binPoints[i] = Math.Clamp((int)MathF.Floor((nfft + 1) * freq / sr), 0, numBins - 1);
        }

        for (int m = 1; m <= numMels; m++)
        {
            int left = binPoints[m - 1];
            int center = binPoints[m];
            int right = binPoints[m + 1];

            for (int k = left; k < center; k++)
            {
                filters[m - 1][k] = (float)(k - left) / Math.Max(1, center - left);
            }
            for (int k = center; k < right; k++)
            {
                filters[m - 1][k] = (float)(right - k) / Math.Max(1, right - center);
            }
        }

        return filters;
    }
}
