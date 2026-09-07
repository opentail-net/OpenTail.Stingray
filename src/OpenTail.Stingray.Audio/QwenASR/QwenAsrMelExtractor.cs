
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
    // Real n_fft=400, corrected 2026-09-07 (was 512): confirmed directly from the real
    // checkpoint's `preprocessor_config.json` (`"n_fft": 400`, `"hop_length": 160`,
    // `"feature_size": 128` -- a real, separate config file this port had not previously read,
    // only NFft's value was wrong, not WindowSize/HopLength). A wrong FFT size shifts every mel
    // filterbank bin's frequency mapping (`CreateSlaneyMelFilterBank` uses NFft directly) and the
    // real reference's mel output frame count (595) vs this port's previous 593 for the same
    // audio -- found via a real frame-by-frame comparison against the reference's dumped mel
    // output, part of this session's Qwen3 Forced Aligner audio-encoder bisection.
    public const int NFft = 400;

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
    /// <summary>Real torch/numpy "reflect" boundary index, ported from `dsp.cpp`'s real
    /// `reflect_index` (not guessed) -- NOT edge-clamping, a genuine mirrored-without-repeating-
    /// the-edge-sample reflection, applied iteratively for indices more than one period out of
    /// range (never actually reached for this extractor's real pad size vs. any real audio
    /// length, but ported faithfully anyway).</summary>
    private static int ReflectIndex(int index, int length)
    {
        while (index < 0 || index >= length)
        {
            index = index < 0 ? -index : 2 * length - index - 2;
        }
        return index;
    }

    public float[] ExtractMel(ReadOnlySpan<float> pcm)
    {
        if (pcm.Length < WindowSize) return [];

        // Real Whisper-style CENTERED framing with REFLECT padding, added 2026-09-07 (see
        // docs/audio-review-progress.md's Qwen3 Forced Aligner audio-encoder bisection): the
        // reference's real `WhisperLogMelExtractor` (`dsp.cpp`) pads the signal by `n_fft/2`
        // samples on each side (reflected, not zero/edge-clamped) before framing -- this port
        // previously started each frame's window flush against the raw (unpadded) signal start,
        // a real, silent mismatch that shifts every single frame's content and produces a
        // slightly different total frame count (595 vs this port's previous 593 for the same
        // real test audio, confirmed via a real frame-by-frame dump comparison).
        int pad = NFft / 2;
        int numFrames = Math.Max(1, 1 + pcm.Length / HopLength);
        var mel = new float[NumMels * numFrames];
        var rawMel = new float[NumMels * numFrames];
        var pcmArray = pcm.ToArray(); // captured by the per-frame parallel closure below
        var frameMax = new float[numFrames];

        // 1. STFT & Mel Filterbank Matrix Multiplication -- frames are fully independent (each
        // writes disjoint columns of rawMel), so this parallelizes cleanly across cores; each
        // task gets its own real/powerSpectrum scratch buffers (thread-local, via the
        // Parallel.For localInit overload) since the original code reused one shared pair
        // across all frames. The mel-filter energy dot product is now SIMD (TensorPrimitives.
        // Dot over the already-contiguous powerSpectrum/filter arrays) instead of a scalar loop.
        // maxLogMel's reduction is made race-safe by recording each frame's own max into
        // frameMax[] and reducing that array (single-threaded, cheap) after the parallel loop,
        // rather than one shared mutable float racing across threads.
        Parallel.For(0, numFrames,
            () => (new float[NFft], new float[NFft / 2 + 1]),
            (f, _, buffers) =>
            {
                var (real, powerSpectrum) = buffers;
                int startSample = f * HopLength - pad;
                Array.Clear(real, 0, NFft);

                for (int i = 0; i < WindowSize; i++)
                {
                    int sIdx = ReflectIndex(startSample + i, pcmArray.Length);
                    real[i] = pcmArray[sIdx] * _hannWindow[i];
                }

                // Power Spectrum (modulus squared of DFT)
                SpectralKernels.ComputePowerSpectrum(real, powerSpectrum);
                for (int k = 0; k <= NFft / 2; k++)
                {
                    powerSpectrum[k] /= NFft;
                }

                float localMax = float.NegativeInfinity;
                for (int m = 0; m < NumMels; m++)
                {
                    float energy = TensorPrimitives.Dot((ReadOnlySpan<float>)powerSpectrum, (ReadOnlySpan<float>)_melFilters[m]);

                    // log10(clamp(mel, min=1e-10))
                    float logVal = MathF.Log10(MathF.Max(energy, 1e-10f));
                    rawMel[m * numFrames + f] = logVal;

                    if (logVal > localMax) localMax = logVal;
                }
                frameMax[f] = localMax;

                return buffers;
            },
            _ => { });

        float maxLogMel = float.NegativeInfinity;
        for (int f = 0; f < numFrames; f++)
            if (frameMax[f] > maxLogMel) maxLogMel = frameMax[f];

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
