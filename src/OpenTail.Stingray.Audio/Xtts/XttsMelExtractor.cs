
namespace OpenTail.Stingray.Audio.Xtts;

/// <summary>
/// Real XTTS-v2 mel-spectrogram feature extractor, matching the shared real
/// `torchaudio.transforms.MelSpectrogram(power=2, normalized=False, sample_rate=22050, f_min=0,
/// f_max=8000, n_mels=80, norm="slaney")` -&gt; `log(clamp(mel, min=1e-5))` -&gt; per-mel-bin
/// normalization against `mel_stats.pth` formula shared by BOTH real reference functions
/// (`TTS/tts/layers/xtts/dvae.py`'s `dvae_wav_to_mel` and `TTS/tts/models/xtts.py`'s
/// `wav_to_mel_cloning`) -- they differ ONLY in `n_fft`/`hop_length`/`win_length`, which this class
/// now takes as constructor parameters instead of hardcoding one function's values.
///
/// <para><b>Real call-site n_fft/hop/win values, confirmed from the actual reference call
/// sites</b> (NOT just each function's own default-argument values, which can differ from what's
/// actually passed): `dvae_wav_to_mel` (DVAE codebook path -- confirmed NOT part of the real
/// `Xtts.inference` synthesis path, see this port's DVAE-not-used correction) always uses
/// `n_fft=1024, hop_length=256, win_length=1024` (its only, hardcoded call). `wav_to_mel_cloning`
/// (the REAL conditioning-encoder path, `Xtts.get_gpt_cond_latents`) is called with
/// `n_fft=2048, hop_length=256, win_length=1024` when `gpt_use_perceiver_resampler=True` (real for
/// XTTS-v2's own `config.json`, confirmed) -- use <see cref="ForConditioningCloning"/> for that
/// real pipeline path; the DVAE-path `n_fft=1024` default on the primary constructor exists only
/// for the already-golden-verified (but pipeline-dead) `XttsDvaeDecoder` stage.</para>
///
/// <para><b>Critical, easy-to-get-wrong detail confirmed directly from the real `torchaudio`
/// source</b> (`torchaudio/functional/functional.py`'s `melscale_fbanks`/`_hz_to_mel`): the
/// `norm="slaney"` parameter ONLY affects the filterbank's AREA normalization -- the underlying
/// Hz-to-mel SCALE conversion still defaults to `mel_scale="htk"` (pure `2595*log10(1+hz/700)`),
/// NOT the librosa/"Slaney-scale" piecewise-linear-below-1000Hz formula
/// (<see cref="OpenTail.Stingray.Audio.CosyVoice.CosyVoiceMelExtractor"/> uses THAT different
/// formula for its own, unrelated checkpoint -- do not copy its `HzToMel`/`MelToHz` for XTTS, a
/// plausible-looking but wrong reuse this port deliberately avoided). The triangular-filter
/// construction and Slaney area-normalization formula themselves ARE structurally the same as
/// `CosyVoiceMelExtractor`'s, just fed the correct (HTK) mel breakpoints.</para>
/// </summary>
public sealed class XttsMelExtractor
{
    public const int NumMels = 80;
    public const int SampleRate = 22050;
    public const float FMin = 0f;
    public const float FMax = 8000f;

    public int Nfft { get; }
    public int HopLength { get; }
    public int WinLength { get; }
    public int Pad => Nfft / 2; // real torch.stft center=True reflect padding

    private readonly float[][] _melBasis; // [NumMels][Nfft/2+1]
    private readonly float[] _hannWindow; // [WinLength]

    /// <summary>Real `dvae_wav_to_mel`'s hardcoded n_fft=1024/hop=256/win=1024 (DVAE codebook path -- pipeline-dead, see class doc).</summary>
    public XttsMelExtractor() : this(nfft: 1024, hopLength: 256, winLength: 1024) { }

    public XttsMelExtractor(int nfft, int hopLength, int winLength)
    {
        Nfft = nfft;
        HopLength = hopLength;
        WinLength = winLength;

        // Real torch.stft reads the FULL n_fft-length span per frame and center-pads the
        // win_length-length window with zeros to n_fft length when win_length < n_fft (already
        // established for XttsSpeakerMelExtractor's identical real torch.stft behavior).
        _hannWindow = new float[Nfft];
        int winOffset = (Nfft - WinLength) / 2;
        for (int i = 0; i < WinLength; i++)
            _hannWindow[winOffset + i] = 0.5f * (1.0f - MathF.Cos(2.0f * MathF.PI * i / WinLength)); // periodic Hann, matches torch.hann_window's default (periodic=True)

        _melBasis = BuildMelBasis();
    }

    /// <summary>Real `wav_to_mel_cloning` as actually called from `Xtts.get_gpt_cond_latents` when `gpt_use_perceiver_resampler=True` (real for XTTS-v2): n_fft=2048, hop_length=256, win_length=1024 -- the REAL conditioning-encoder mel frontend.</summary>
    public static XttsMelExtractor ForConditioningCloning() => new(nfft: 2048, hopLength: 256, winLength: 1024);

    /// <summary>
    /// Extracts the real per-checkpoint-normalized log-mel-spectrogram from 22050Hz reference
    /// audio PCM samples, channel-first [80, numFrames] (matching this codebase's usual layout).
    /// `melStats` is the real `mel_stats.pth`'s [80] per-bin normalization divisor.
    /// </summary>
    public float[] ExtractMel(ReadOnlySpan<float> pcm, float[] melStats)
    {
        if (pcm.Length == 0) return [];

        int len = pcm.Length;
        int paddedLen = len + 2 * Pad;
        var signal = new float[paddedLen];

        // Reflect padding (torch.stft's center=True default, pad_mode="reflect").
        for (int i = 0; i < Pad; i++)
        {
            int leftIdx = (Pad - i) % len;
            if (leftIdx < 0) leftIdx += len;
            signal[i] = pcm[leftIdx];

            int rightIdx = (len - 2 - i) % len;
            if (rightIdx < 0) rightIdx += len;
            signal[i + len + Pad] = pcm[rightIdx];
        }
        pcm.CopyTo(signal.AsSpan(Pad, len));

        int numFrames = (paddedLen - Nfft) / HopLength + 1;
        if (numFrames <= 0) return [];

        int numBins = Nfft / 2 + 1;
        var mel = new float[NumMels * numFrames];
        var powerSpec = new float[numBins];
        var windowed = new float[Nfft];

        for (int f = 0; f < numFrames; f++)
        {
            int pcmOff = f * HopLength;
            for (int n = 0; n < Nfft; n++)
                windowed[n] = signal[pcmOff + n] * _hannWindow[n];

            // power=2: keep the power spectrum as-is (no sqrt to magnitude, unlike CosyVoiceMelExtractor's power=1).
            SpectralKernels.ComputePowerSpectrum(windowed, powerSpec);

            for (int m = 0; m < NumMels; m++)
            {
                float energy = 0f;
                var filter = _melBasis[m];
                for (int k = 0; k < numBins; k++)
                    energy += powerSpec[k] * filter[k];

                // log(clamp(mel, min=1e-5)) then per-bin normalization -- channel-first [mel, frame].
                float logMel = MathF.Log(MathF.Max(1e-5f, energy));
                mel[m * numFrames + f] = logMel / melStats[m];
            }
        }

        return mel;
    }

    private static float HzToMelHtk(float hz) => 2595.0f * MathF.Log10(1.0f + hz / 700.0f);
    private static float MelToHzHtk(float mel) => 700.0f * (MathF.Pow(10.0f, mel / 2595.0f) - 1.0f);

    private float[][] BuildMelBasis()
    {
        int numBins = Nfft / 2 + 1;
        float mMin = HzToMelHtk(FMin);
        float mMax = HzToMelHtk(FMax);
        float step = (mMax - mMin) / (NumMels + 1);

        var fPts = new float[NumMels + 2]; // Hz
        for (int i = 0; i < NumMels + 2; i++)
            fPts[i] = MelToHzHtk(mMin + i * step);

        var fDiff = new float[NumMels + 1];
        for (int i = 0; i < NumMels + 1; i++)
            fDiff[i] = fPts[i + 1] - fPts[i];

        // all_freqs = linspace(0, sampleRate/2, numBins) -- real torchaudio convention.
        var allFreqs = new float[numBins];
        for (int i = 0; i < numBins; i++)
            allFreqs[i] = i * (SampleRate / 2f) / (numBins - 1);

        var basis = new float[NumMels][];
        for (int i = 0; i < NumMels; i++)
        {
            basis[i] = new float[numBins];
            float enorm = 2.0f / (fPts[i + 2] - fPts[i]); // real Slaney AREA normalization (norm="slaney")

            for (int j = 0; j < numBins; j++)
            {
                float down = (allFreqs[j] - fPts[i]) / fDiff[i];
                float up = (fPts[i + 2] - allFreqs[j]) / fDiff[i + 1];
                float val = MathF.Max(0.0f, MathF.Min(down, up));
                basis[i][j] = val * enorm;
            }
        }

        return basis;
    }
}
