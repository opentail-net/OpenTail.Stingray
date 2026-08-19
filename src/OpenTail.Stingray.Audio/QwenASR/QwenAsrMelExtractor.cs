using OpenTail.Stingray.Audio.Primitives;

namespace OpenTail.Stingray.Audio.QwenASR;

/// <summary>
/// 128-channel Log-Mel Spectrogram feature extractor for Qwen3-ASR (16kHz).
/// Follows Slaney-style 128-channel filterbank with dynamic range maximum clamping and (log_spec + 4.0)/4.0 scaling.
/// </summary>
public sealed class QwenAsrMelExtractor
{
    public const int SampleRate = 16000;
    public const int NumMels = 128;
    public const int WindowSize = 400; // 25ms @ 16kHz
    public const int HopLength = 160;   // 10ms @ 16kHz
    public const int NFft = 512;

    private readonly float[] _hannWindow;
    private readonly float[][] _melFilters;

    public QwenAsrMelExtractor()
    {
        _hannWindow = SpectralKernels.CreateSymmetricHannWindow(WindowSize);
        _melFilters = CreateSlaneyMelFilterBank(NumMels, NFft, SampleRate, 0.0f, 8000.0f);
    }

    /// <summary>
    /// Computes 128-channel normalized log-mel spectrogram from 16kHz audio samples.
    /// Output shape: [NumMels, numFrames] flattened as mel[m * numFrames + f].
    /// </summary>
    public float[] ExtractMel(ReadOnlySpan<float> pcm)
    {
        if (pcm.Length < WindowSize) return [];

        int numFrames = Math.Max(1, (pcm.Length - WindowSize) / HopLength + 1);
        var mel = new float[NumMels * numFrames];

        var real = new float[NFft];
        var powerSpectrum = new float[NFft / 2 + 1];
        var rawMel = new float[NumMels * numFrames];

        float maxLogMel = float.NegativeInfinity;

        // 1. STFT & Mel Filterbank Matrix Multiplication
        for (int f = 0; f < numFrames; f++)
        {
            int startSample = f * HopLength;
            Array.Clear(real, 0, NFft);

            for (int i = 0; i < WindowSize; i++)
            {
                int sIdx = startSample + i;
                float sample = (sIdx < pcm.Length) ? pcm[sIdx] : 0.0f;
                real[i] = sample * _hannWindow[i];
            }

            // Power Spectrum (modulus squared of DFT)
            SpectralKernels.ComputePowerSpectrum(real, powerSpectrum);
            for (int k = 0; k <= NFft / 2; k++)
            {
                powerSpectrum[k] /= NFft;
            }

            for (int m = 0; m < NumMels; m++)
            {
                float energy = 0.0f;
                float[] filter = _melFilters[m];
                for (int k = 0; k < powerSpectrum.Length; k++)
                {
                    energy += powerSpectrum[k] * filter[k];
                }

                // log10(clamp(mel, min=1e-10))
                float logVal = MathF.Log10(MathF.Max(energy, 1e-10f));
                rawMel[m * numFrames + f] = logVal;

                if (logVal > maxLogMel)
                {
                    maxLogMel = logVal;
                }
            }
        }

        // 2. Dynamic Range Clamping (max - 8.0) and Normalization ((log_spec + 4.0) / 4.0)
        float floorVal = maxLogMel - 8.0f;
        for (int i = 0; i < rawMel.Length; i++)
        {
            float clamped = MathF.Max(rawMel[i], floorVal);
            mel[i] = (clamped + 4.0f) / 4.0f;
        }

        return mel;
    }

    private static float[][] CreateSlaneyMelFilterBank(int numMels, int nFft, int sampleRate, float fMin, float fMax)
    {
        int numBins = nFft / 2 + 1;
        var filters = new float[numMels][];
        for (int m = 0; m < numMels; m++) filters[m] = new float[numBins];

        // Slaney Mel scale conversion: linear below 1000 Hz, logarithmic above 1000 Hz
        float HzToMel(float hz) => (hz >= 1000.0f) ? 15.0f + MathF.Log(hz / 1000.0f) / 0.068751777f : 3.0f * hz / 200.0f;
        float MelToHz(float mel) => (mel >= 15.0f) ? 1000.0f * MathF.Exp((mel - 15.0f) * 0.068751777f) : 200.0f * mel / 3.0f;

        float melMin = HzToMel(fMin);
        float melMax = HzToMel(fMax);
        float melStep = (melMax - melMin) / (numMels + 1);

        var binFreqs = new float[numMels + 2];
        for (int i = 0; i < binFreqs.Length; i++)
        {
            float m = melMin + i * melStep;
            float hz = MelToHz(m);
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
