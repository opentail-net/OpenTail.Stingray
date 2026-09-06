
namespace OpenTail.Stingray.Audio.NemotronAsr;

/// <summary>
/// 128-channel Log-Mel Spectrogram Filterbank feature extractor for Nemotron 3.5 ASR (16kHz).
/// Transcribed directly from `examples/audio.cpp/src/models/nemotron_asr/frontend.cpp`'s
/// `NemotronFrontend::extract_waveform` (not guessed) -- a real, deliberately SIMPLER pipeline
/// than <see cref="Parakeet.ParakeetMelExtractor"/>'s: global pre-emphasis (0.97) -> center-pad by
/// n_fft/2 both sides (STFT `center=true`, constant/zero pad) -> Hann window (symmetric,
/// non-periodic: `0.5 - 0.5*cos(2*pi*i/(N-1))`, self-generated since this checkpoint does NOT ship
/// a window tensor) -> magnitude spectrum -> the checkpoint's own shipped `preprocessor.fb`
/// filterbank -> `log(mel + log_zero_guard)` (`log_zero_guard = 2^-24`, matching Parakeet's
/// `LogEps` exactly) -> frames beyond `valid_frames` zeroed (`asr.preprocessor.mask_invalid_frames
/// = true`). Critically, UNLIKE Parakeet, there is NO per-feature Z-normalization step here --
/// confirmed both by the reference's frontend.cpp having no such step and by this checkpoint's own
/// `asr.preprocessor.normalize = NA` metadata.
/// </summary>
public sealed class NemotronAsrMelExtractor
{
    public const int SampleRate = 16000;
    public const int NumMels = 128;
    public const int NFft = 512;
    public const int WinLength = 400; // 25ms @ 16kHz
    public const int HopLength = 160;  // 10ms @ 16kHz
    private const float Preemph = 0.97f;
    private const float LogZeroGuard = 1f / (1 << 24);

    private readonly float[] _window; // NFft-length, WinLength samples centered with (NFft-WinLength)/2 zero-pad each side
    private readonly float[] _melFb;  // [NumMels, NFft/2+1], row-major (fb[m * n_freqs + k])

    public NemotronAsrMelExtractor(float[] melFilterbank)
    {
        var rawWindow = MakeHannWindow(WinLength);
        _window = new float[NFft];
        int lpad = (NFft - WinLength) / 2;
        rawWindow.CopyTo(_window.AsSpan(lpad));
        _melFb = melFilterbank;
    }

    public static NemotronAsrMelExtractor FromWeights(NemotronAsrWeights w) => new(w.MelFilterbank);

    /// <summary>Real symmetric (non-periodic) Hann window, matching frontend.cpp's
    /// `make_hann_window` exactly: `0.5 - 0.5*cos(2*pi*i/(N-1))`.</summary>
    private static float[] MakeHannWindow(int winLength)
    {
        var window = new float[winLength];
        if (winLength == 1) { window[0] = 1f; return window; }
        for (int i = 0; i < winLength; i++)
            window[i] = 0.5f - 0.5f * MathF.Cos(2f * MathF.PI * i / (winLength - 1));
        return window;
    }

    /// <summary>
    /// Extracts a 128-channel log-mel spectrogram from 16kHz mono PCM audio samples.
    /// Returns float array of shape [numFrames * 128] flattened as mel[f * 128 + m], with the
    /// time-major frame count matching the reference's own `frames` (center-padded STFT frame
    /// count), NOT reduced to `valid_frames` (the reference keeps the full frame count and zeros
    /// out invalid tail frames rather than truncating).
    /// </summary>
    public (float[] Mel, int Frames, int ValidFrames) ExtractMel(ReadOnlySpan<float> pcm)
    {
        int nSamples = pcm.Length;
        if (nSamples == 0) return ([], 0, 0);

        var processed = new float[nSamples];
        pcm.CopyTo(processed);
        for (int i = nSamples - 1; i > 0; i--)
            processed[i] -= Preemph * processed[i - 1];

        int pad = NFft / 2;
        var padded = new float[pad + nSamples + pad];
        processed.CopyTo(padded.AsSpan(pad));

        int frames = (padded.Length - NFft) / HopLength + 1;
        if (frames <= 0) return ([], 0, 0);

        int validFrames = Math.Clamp(nSamples / HopLength, 0, frames);

        var mel = new float[frames * NumMels];
        var frame = new float[NFft];
        var powerSpectrum = new float[NFft / 2 + 1];

        for (int f = 0; f < frames; f++)
        {
            int start = f * HopLength;
            for (int i = 0; i < NFft; i++)
                frame[i] = padded[start + i] * _window[i];

            SpectralKernels.ComputePowerSpectrum(frame, powerSpectrum);
            // Reference uses the magnitude spectrum (sqrt of power) fed into the mel filterbank.
            for (int k = 0; k < powerSpectrum.Length; k++)
                powerSpectrum[k] = MathF.Sqrt(powerSpectrum[k]);

            for (int m = 0; m < NumMels; m++)
            {
                float energy = 0f;
                int fbBase = m * powerSpectrum.Length;
                for (int k = 0; k < powerSpectrum.Length; k++)
                    energy += powerSpectrum[k] * _melFb[fbBase + k];

                mel[f * NumMels + m] = f < validFrames ? MathF.Log(energy + LogZeroGuard) : 0f;
            }
        }

        return (mel, frames, validFrames);
    }
}
