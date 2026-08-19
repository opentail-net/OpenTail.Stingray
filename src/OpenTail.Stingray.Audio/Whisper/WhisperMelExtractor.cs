using System.Numerics.Tensors;
using OpenTail.Stingray.Audio.Primitives;

namespace OpenTail.Stingray.Audio.Whisper;

/// <summary>
/// 80/128-channel Log-Mel Spectrogram feature extractor for 16kHz audio in OpenAI Whisper.
/// (SampleRate=16000, Nfft=400, HopLength=160, WinLength=400, Fmin=0, Fmax=8000).
/// </summary>
public sealed class WhisperMelExtractor
{
    public const int SampleRate = 16000;
    public const int Nfft = 400;
    public const int HopLength = 160;
    public const int WinLength = 400;
    public const int HalfFft = Nfft / 2 + 1; // 201 bins

    public int NumMels { get; }

    private readonly float[] _hannWindow;
    private readonly float[] _melFilters; // [NumMels * HalfFft]

    public WhisperMelExtractor(int numMels = 80, float[]? customFilters = null)
    {
        NumMels = numMels;
        _hannWindow = SpectralKernels.CreateHannWindow(WinLength);

        if (customFilters != null && customFilters.Length == NumMels * HalfFft)
        {
            _melFilters = (float[])customFilters.Clone();
        }
        else
        {
            _melFilters = CreateSlaneyMelFilterbank(NumMels, Nfft, SampleRate, 0f, 8000f);
        }
    }

    /// <summary>
    /// Computes Log-Mel spectrogram from 16kHz audio PCM samples.
    /// Returns 2D flattened array formatted as [NumMels, numFrames].
    /// </summary>
    public float[] ExtractMel(ReadOnlySpan<float> pcm, bool padTo30Seconds = true)
    {
        int stage1Pad = padTo30Seconds ? SampleRate * 30 : 0;
        int stage2Pad = WinLength / 2; // 200 samples reflective padding

        int nSamples = pcm.Length;
        int paddedLength = nSamples + stage1Pad + stage2Pad * 2;
        float[] padded = new float[paddedLength];

        // 1. Copy original samples at stage2Pad offset
        pcm.CopyTo(padded.AsSpan(stage2Pad, nSamples));

        // 2. Reflective pad start
        int reflectCount = Math.Min(stage2Pad, Math.Max(0, nSamples - 1));
        for (int i = 0; i < reflectCount; i++)
        {
            padded[stage2Pad - 1 - i] = pcm[1 + i];
        }

        // 3. Number of frames
        int numFrames = (paddedLength - WinLength) / HopLength;
        if (numFrames <= 0) numFrames = 1;

        // mel output: [NumMels, numFrames]
        float[] melData = new float[NumMels * numFrames];
        float[] fftIn = new float[WinLength];
        float[] powerSpec = new float[HalfFft];

        for (int f = 0; f < numFrames; f++)
        {
            int offset = f * HopLength;

            // Apply Hann window
            int avail = Math.Min(WinLength, paddedLength - offset);
            for (int j = 0; j < avail; j++)
            {
                fftIn[j] = padded[offset + j] * _hannWindow[j];
            }
            if (avail < WinLength)
            {
                Array.Clear(fftIn, avail, WinLength - avail);
            }

            // Compute Power Spectrum (modulus^2 of DFT)
            SpectralKernels.ComputePowerSpectrum(fftIn, powerSpec);

            // Apply Mel filterbank & log10 compression
            for (int m = 0; m < NumMels; m++)
            {
                int filterOff = m * HalfFft;
                double sum = 0.0;

                for (int k = 0; k < HalfFft; k++)
                {
                    sum += (double)powerSpec[k] * _melFilters[filterOff + k];
                }

                float logEnergy = MathF.Log10(MathF.Max((float)sum, 1e-10f));
                melData[m * numFrames + f] = logEnergy;
            }
        }

        // 4. Dynamic range clamping to (max - 8.0) and normalization to [-1, 1] range: (x + 4.0) / 4.0
        float maxVal = -1e20f;
        for (int i = 0; i < melData.Length; i++)
        {
            if (melData[i] > maxVal) maxVal = melData[i];
        }

        float minThreshold = maxVal - 8.0f;
        for (int i = 0; i < melData.Length; i++)
        {
            float val = melData[i];
            if (val < minThreshold) val = minThreshold;
            melData[i] = (val + 4.0f) / 4.0f;
        }

        return melData;
    }

    /// <summary>
    /// Constructs Slaney-style Mel filterbank with area normalization matching librosa / PyTorch / OpenAI Whisper.
    /// </summary>
    public static float[] CreateSlaneyMelFilterbank(int numMels, int nfft, int sampleRate, float fmin, float fmax)
    {
        int halfFft = nfft / 2 + 1;
        float[] filters = new float[numMels * halfFft];

        // Slaney mel scale: linear below 1000 Hz, logarithmic above
        const float minLogHz = 1000.0f;
        const float minLogMel = 15.0f; // 1000 / (200.0 / 3.0)
        const float logStep = 27.0f / 64.0f; // log(6.4) / 27.0

        float HzToMel(float hz)
        {
            if (hz < minLogHz) return hz / (200.0f / 3.0f);
            return minLogMel + MathF.Log(hz / minLogHz) / logStep;
        }

        float MelToHz(float mel)
        {
            if (mel < minLogMel) return mel * (200.0f / 3.0f);
            return minLogHz * MathF.Exp(logStep * (mel - minLogMel));
        }

        float minMel = HzToMel(fmin);
        float maxMel = HzToMel(fmax);
        float melDelta = (maxMel - minMel) / (numMels + 1);

        float[] melPoints = new float[numMels + 2];
        float[] hzPoints = new float[numMels + 2];

        for (int i = 0; i < numMels + 2; i++)
        {
            melPoints[i] = minMel + i * melDelta;
            hzPoints[i] = MelToHz(melPoints[i]);
        }

        float[] fftFreqs = new float[halfFft];
        for (int i = 0; i < halfFft; i++)
        {
            fftFreqs[i] = (float)i * sampleRate / nfft;
        }

        for (int m = 0; m < numMels; m++)
        {
            float fPrev = hzPoints[m];
            float fCurr = hzPoints[m + 1];
            float fNext = hzPoints[m + 2];

            int offset = m * halfFft;

            for (int k = 0; k < halfFft; k++)
            {
                float f = fftFreqs[k];
                float weight = 0f;

                if (f >= fPrev && f <= fCurr && fCurr > fPrev)
                {
                    weight = (f - fPrev) / (fCurr - fPrev);
                }
                else if (f >= fCurr && f <= fNext && fNext > fCurr)
                {
                    weight = (fNext - f) / (fNext - fCurr);
                }

                // Slaney normalization: 2.0 / (fNext - fPrev)
                float norm = 2.0f / MathF.Max(fNext - fPrev, 1e-6f);
                filters[offset + k] = weight * norm;
            }
        }

        return filters;
    }
}
