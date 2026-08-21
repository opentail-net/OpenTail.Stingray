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
        int stage2Pad = WinLength / 2; // 200 samples reflective padding, matches torch.stft(center=True)

        // Whisper pads/trims audio to EXACTLY 30s (N_SAMPLES = SampleRate*30) before computing
        // mel frames, not "real audio length + 30s of extra padding" -- getting this wrong
        // inflates the frame count (e.g. 4100 instead of the correct 3000 for a 30s window) and,
        // worse, corrupts the global-max normalization below since it's computed over the whole
        // (wrongly oversized) buffer instead of the true 30s window. Confirmed against a numpy
        // reference port of openai-whisper's log_mel_spectrogram; see docs/audio-review-progress.md.
        int nSamples = pcm.Length;
        int innerLength = padTo30Seconds ? SampleRate * 30 : nSamples;
        int paddedLength = innerLength + stage2Pad * 2;
        float[] padded = new float[paddedLength];

        // 1. Copy original samples at stage2Pad offset (trimmed to innerLength if longer than 30s)
        int copyLength = Math.Min(nSamples, innerLength);
        pcm[..copyLength].CopyTo(padded.AsSpan(stage2Pad, copyLength));

        // 2. Reflective pad start (reflect off the real audio's own first samples)
        int reflectCount = Math.Min(stage2Pad, Math.Max(0, copyLength - 1));
        for (int i = 0; i < reflectCount; i++)
        {
            padded[stage2Pad - 1 - i] = pcm[1 + i];
        }

        // 3. Reflective pad end (reflect off the tail of the inner [0, innerLength) buffer --
        // real audio's tail if trimmed/exact-length, otherwise the zero-padding, which reflects
        // to more zeros and is a no-op).
        int tailStart = stage2Pad + innerLength;
        int reflectTailCount = Math.Min(stage2Pad, Math.Max(0, copyLength - 1));
        for (int i = 0; i < reflectTailCount; i++)
        {
            padded[tailStart + i] = padded[tailStart - 2 - i];
        }

        // 4. Number of frames. torch.stft's raw frame count is 1 + (paddedLength-WinLength)/HopLength,
        // and whisper drops the trailing frame (stft[..., :-1]) -- the two cancel out, so this
        // integer-division form already equals the correct final frame count directly.
        int numFrames = (paddedLength - WinLength) / HopLength;
        if (numFrames <= 0) numFrames = 1;

        // mel output: [NumMels, numFrames]
        float[] melData = new float[NumMels * numFrames];

        Parallel.For(0, numFrames, f =>
        {
            int offset = f * HopLength;
            Span<float> fftIn = stackalloc float[WinLength];
            Span<float> powerSpec = stackalloc float[HalfFft];

            // Apply Hann window
            int avail = Math.Min(WinLength, paddedLength - offset);
            for (int j = 0; j < avail; j++)
            {
                fftIn[j] = padded[offset + j] * _hannWindow[j];
            }
            if (avail < WinLength)
            {
                fftIn.Slice(avail, WinLength - avail).Clear();
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
        });

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
        // BUG FIX (2026-08-21): this was hardcoded to 27f/64f (0.4219), not log(6.4)/27 (0.0688)
        // as the comment claimed -- a ~6x error that corrupted the mel scale above 1000Hz and
        // zeroed out low-index filters entirely. Confirmed against a numpy reference port of
        // librosa/whisper's mel filterbank; see docs/audio-review-progress.md.
        float logStep = MathF.Log(6.4f) / 27.0f;

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
