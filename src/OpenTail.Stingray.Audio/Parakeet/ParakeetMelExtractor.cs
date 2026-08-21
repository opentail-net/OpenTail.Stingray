using OpenTail.Stingray.Audio.Primitives;

namespace OpenTail.Stingray.Audio.Parakeet;

/// <summary>
/// 80-channel Log-Mel Spectrogram Filterbank feature extractor for NVIDIA NeMo Parakeet/
/// Canary-CTC ASR (16kHz). Uses the checkpoint's own shipped filterbank/window
/// (`preprocessor.fb`/`preprocessor.window`) rather than recomputing them, and matches
/// `examples/crispasr/src/core/mel.cpp`'s exact NeMo `AudioToMelSpectrogramPreprocessor`
/// pipeline: global pre-emphasis (0.97, applied before center-padding) -> zero center-pad by
/// n_fft/2 -> window centered within the n_fft buffer -> power spectrum -> mel projection ->
/// `log(x + log_eps)` (log_eps = 2^-24, natural log, NOT `log(max(x,eps))`) -> per-feature
/// Z-normalization (Bessel-corrected variance, eps=1e-5 added to std OUTSIDE the sqrt). An
/// earlier version of this file self-computed a generic librosa-style mel filterbank and used
/// `log(max(x,1e-5))` with no normalization -- both diverge from what this checkpoint actually
/// expects; see docs/audio-review-progress.md's Parakeet section.
/// </summary>
public sealed class ParakeetMelExtractor
{
    public const int SampleRate = 16000;
    public const int NumMels = 80;
    public const int NFft = 512;
    public const int WinLength = 400; // 25ms @ 16kHz
    public const int HopLength = 160;  // 10ms @ 16kHz
    private const float Preemph = 0.97f;
    private const float LogEps = 1f / (1 << 24);

    private readonly float[] _window; // n_fft-length, win_length samples centered with (n_fft-win_length)/2 zero-pad each side
    private readonly float[] _melFb;  // [NumMels, NFft/2+1], row-major (fb[m * n_freqs + k])

    public ParakeetMelExtractor() : this(BuiltinFallbackWindow(), BuiltinFallbackFilterbank())
    {
    }

    /// <summary>Constructs from the checkpoint's own real shipped tensors (preferred -- see <see cref="FromWeights"/>).</summary>
    public ParakeetMelExtractor(float[] rawWindow, float[] melFilterbank)
    {
        _window = new float[NFft];
        int lpad = (NFft - WinLength) / 2;
        int wn = Math.Min(rawWindow.Length, WinLength);
        for (int i = 0; i < wn; i++) _window[lpad + i] = rawWindow[i];
        _melFb = melFilterbank;
    }

    public static ParakeetMelExtractor FromWeights(ParakeetWeights w) => new(w.MelWindow, w.MelFilterbank);

    /// <summary>
    /// Extracts an 80-channel log-mel spectrogram from 16kHz mono PCM audio samples.
    /// Returns float array of shape [numFrames * 80] flattened as mel[f * 80 + m].
    /// </summary>
    public float[] ExtractMel(ReadOnlySpan<float> pcm)
    {
        int nSamples = pcm.Length;
        if (nSamples == 0) return [];

        // 1. Global pre-emphasis (before center-padding, matches NeMo: first sample preserved as-is).
        var preemph = new float[nSamples];
        preemph[0] = pcm[0];
        for (int i = 1; i < nSamples; i++) preemph[i] = pcm[i] - Preemph * pcm[i - 1];

        // 2. Zero center-pad by n_fft/2 on both sides.
        int pad = NFft / 2;
        var padded = new float[pad + nSamples + pad];
        preemph.CopyTo(padded.AsSpan(pad));

        int t = (padded.Length - NFft) / HopLength + 1;
        if (t <= 0) return [];

        var mel = new float[t * NumMels];
        var frame = new float[NFft];
        var powerSpectrum = new float[NFft / 2 + 1];

        for (int f = 0; f < t; f++)
        {
            int start = f * HopLength;
            for (int i = 0; i < NFft; i++)
                frame[i] = padded[start + i] * _window[i];

            SpectralKernels.ComputePowerSpectrum(frame, powerSpectrum);

            for (int m = 0; m < NumMels; m++)
            {
                float energy = 0f;
                int fbBase = m * powerSpectrum.Length;
                for (int k = 0; k < powerSpectrum.Length; k++)
                    energy += powerSpectrum[k] * _melFb[fbBase + k];

                mel[f * NumMels + m] = MathF.Log(energy + LogEps);
            }
        }

        // 3. Per-feature (per-mel-band) Z-normalization across time, Bessel-corrected variance.
        int denom = t > 1 ? t - 1 : 1;
        for (int m = 0; m < NumMels; m++)
        {
            double sum = 0;
            for (int f = 0; f < t; f++) sum += mel[f * NumMels + m];
            double mean = sum / t;

            double sq = 0;
            for (int f = 0; f < t; f++)
            {
                double d = mel[f * NumMels + m] - mean;
                sq += d * d;
            }
            float std = MathF.Sqrt((float)(sq / denom));
            if (float.IsNaN(std)) std = 0f;
            std += 1e-5f;

            for (int f = 0; f < t; f++)
                mel[f * NumMels + m] = (float)((mel[f * NumMels + m] - mean) / std);
        }

        return mel;
    }

    // Fallback filterbank/window (librosa-style, self-computed) for callers that construct this
    // class without a loaded checkpoint (e.g. structural Fast tests). NOT what the real
    // checkpoint expects -- ParakeetPipeline.Load always uses FromWeights with the checkpoint's
    // real preprocessor.fb/preprocessor.window instead.
    private static float[] BuiltinFallbackWindow() => SpectralKernels.CreateSymmetricHannWindow(WinLength);

    private static float[] BuiltinFallbackFilterbank()
    {
        int numBins = NFft / 2 + 1;
        var fb = new float[NumMels * numBins];

        float fMin = 0f, fMax = 8000f;
        float melMin = 2595.0f * MathF.Log10(1.0f + fMin / 700.0f);
        float melMax = 2595.0f * MathF.Log10(1.0f + fMax / 700.0f);
        float melStep = (melMax - melMin) / (NumMels + 1);

        var binFreqs = new float[NumMels + 2];
        for (int i = 0; i < binFreqs.Length; i++)
        {
            float m = melMin + i * melStep;
            float hz = 700.0f * (MathF.Pow(10.0f, m / 2595.0f) - 1.0f);
            binFreqs[i] = MathF.Floor((NFft + 1) * hz / SampleRate);
        }

        for (int m = 0; m < NumMels; m++)
        {
            float left = binFreqs[m], center = binFreqs[m + 1], right = binFreqs[m + 2];
            int fbBase = m * numBins;
            for (int k = 0; k < numBins; k++)
            {
                if (k >= left && k <= center && center > left)
                    fb[fbBase + k] = (k - left) / (center - left);
                else if (k >= center && k <= right && right > center)
                    fb[fbBase + k] = (right - k) / (right - center);
            }
        }
        return fb;
    }
}
