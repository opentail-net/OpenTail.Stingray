namespace OpenTail.Stingray.Audio.Parakeet;

/// <summary>
/// 80-channel Log-Mel Spectrogram Filterbank feature extractor configured for NVIDIA NeMo Parakeet ASR (16kHz).
/// </summary>
public sealed class ParakeetMelExtractor
{
    public const int SampleRate = 16000;
    public const int NumMels = 80;
    public const int WindowSize = 400; // 25ms @ 16kHz
    public const int HopLength = 160;   // 10ms @ 16kHz
    public const int NFft = 512;

    private readonly float[] _window;
    private readonly float[][] _melFilters;

    public ParakeetMelExtractor()
    {
        _window = CreateHanningWindow(WindowSize);
        _melFilters = CreateMelFilterBank(NumMels, NFft, SampleRate, 0.0f, 8000.0f);
    }

    /// <summary>
    /// Extracts an 80-channel log-mel spectrogram from 16kHz mono PCM audio samples.
    /// Returns float array of shape [numFrames * 80] flattened as mel[f * 80 + m].
    /// </summary>
    public float[] ExtractMel(ReadOnlySpan<float> pcm)
    {
        if (pcm.Length < WindowSize) return [];

        int numFrames = Math.Max(1, (pcm.Length - WindowSize) / HopLength + 1);
        var mel = new float[numFrames * NumMels];

        var real = new float[NFft];
        var imag = new float[NFft];
        var powerSpectrum = new float[NFft / 2 + 1];

        for (int f = 0; f < numFrames; f++)
        {
            int startSample = f * HopLength;

            // 1. Windowing & Pre-emphasis (s[i] - 0.97 * s[i-1])
            Array.Clear(real, 0, NFft);
            Array.Clear(imag, 0, NFft);

            for (int i = 0; i < WindowSize; i++)
            {
                int sampleIdx = startSample + i;
                float current = (sampleIdx < pcm.Length) ? pcm[sampleIdx] : 0.0f;
                float prev = (sampleIdx > 0 && sampleIdx - 1 < pcm.Length) ? pcm[sampleIdx - 1] : 0.0f;
                float preEmphasized = current - 0.97f * prev;

                real[i] = preEmphasized * _window[i];
            }

            // 2. Compute Power Spectrum via Discrete Fourier Transform
            for (int k = 0; k <= NFft / 2; k++)
            {
                float r = 0.0f;
                float im = 0.0f;
                float angle = -2.0f * MathF.PI * k / NFft;

                for (int n = 0; n < WindowSize; n++)
                {
                    float a = angle * n;
                    r += real[n] * MathF.Cos(a);
                    im += real[n] * MathF.Sin(a);
                }

                powerSpectrum[k] = (r * r + im * im) / NFft;
            }

            // 3. Apply Mel Filterbank & Log Compression (log(mel + 1e-5))
            for (int m = 0; m < NumMels; m++)
            {
                float energy = 0.0f;
                float[] filter = _melFilters[m];
                for (int k = 0; k < powerSpectrum.Length; k++)
                {
                    energy += powerSpectrum[k] * filter[k];
                }

                // Log-Mel scaling matching NeMo: log(max(energy, 1e-5))
                float logMel = MathF.Log(MathF.Max(energy, 1e-5f));
                mel[f * NumMels + m] = logMel;
            }
        }

        return mel;
    }

    private static float[] CreateHanningWindow(int size)
    {
        var win = new float[size];
        for (int i = 0; i < size; i++)
        {
            win[i] = 0.5f * (1.0f - MathF.Cos(2.0f * MathF.PI * i / (size - 1)));
        }
        return win;
    }

    private static float[][] CreateMelFilterBank(int numMels, int nFft, int sampleRate, float fMin, float fMax)
    {
        int numBins = nFft / 2 + 1;
        var filters = new float[numMels][];
        for (int m = 0; m < numMels; m++)
        {
            filters[m] = new float[numBins];
        }

        float melMin = 2595.0f * MathF.Log10(1.0f + fMin / 700.0f);
        float melMax = 2595.0f * MathF.Log10(1.0f + fMax / 700.0f);
        float melStep = (melMax - melMin) / (numMels + 1);

        var binFreqs = new float[numMels + 2];
        for (int i = 0; i < binFreqs.Length; i++)
        {
            float m = melMin + i * melStep;
            float hz = 700.0f * (MathF.Pow(10.0f, m / 2595.0f) - 1.0f);
            binFreqs[i] = MathF.Floor((nFft + 1) * hz / sampleRate);
        }

        for (int m = 0; m < numMels; m++)
        {
            float left = binFreqs[m];
            float center = binFreqs[m + 1];
            float right = binFreqs[m + 2];

            for (int k = 0; k < numBins; k++)
            {
                if (k >= left && k <= center && center > left)
                {
                    filters[m][k] = (k - left) / (center - left);
                }
                else if (k >= center && k <= right && right > center)
                {
                    filters[m][k] = (right - k) / (right - center);
                }
            }
        }

        return filters;
    }
}
