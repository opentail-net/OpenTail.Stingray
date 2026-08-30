
namespace OpenTail.Stingray.Audio.F5TTS;

/// <summary>
/// 100-channel Mel-Spectrogram feature extractor for 24000Hz audio in F5-TTS
/// (n_fft=1024, hop_length=256, win_length=1024, fmin=0, fmax=12000), matching
/// `examples/f5-tts-py/f5_tts/model/modules.py`'s real `get_vocos_mel_spectrogram` (this
/// checkpoint's real `mel_spec_type="vocos"`): `torchaudio.transforms.MelSpectrogram(power=1,
/// center=True, normalized=False, norm=None)`.
///
/// <para><b>Real `center=True` framing, confirmed against the reference before this was verified
/// (see docs/audio-review-progress.md's F5-TTS entry)</b>: PyTorch's `torch.stft(center=True,
/// pad_mode="reflect")` reflect-pads the waveform by `n_fft/2` samples on BOTH sides before
/// framing, so frame 0 is centered at sample 0 -- NOT framed directly from `pcm[0]` with no
/// padding, which time-shifts every frame by `n_fft/2` samples (21ms at 24kHz) relative to the
/// real reference and produces a different frame count. Fixed here to match.</para>
/// </summary>
public sealed class F5MelExtractor
{
    public const int NumMels = 100;
    public const int SampleRate = 24000;
    public const int Nfft = 1024;
    public const int HopLength = 256;
    public const int WinLength = 1024;
    public const int Pad = Nfft / 2; // 512, real torch.stft center=True reflect padding

    private readonly float[][] _melFilters;
    private readonly float[] _hannWindow;

    public F5MelExtractor()
    {
        _hannWindow = SpectralKernels.CreateHannWindow(WinLength);

        _melFilters = CreateMelFilterbank(NumMels, Nfft, SampleRate, 0f, 12000f);
    }

    /// <summary>
    /// Computes 100-channel log Mel-spectrogram from 24kHz audio PCM samples.
    /// Returns 1D flattened array [numFrames * 100].
    /// </summary>
    public float[] ExtractMel(ReadOnlySpan<float> pcm)
    {
        if (pcm.Length == 0) return new float[NumMels];

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

        int numFrames = (paddedLen - WinLength) / HopLength + 1;
        if (numFrames <= 0) numFrames = 1;

        var mels = new float[numFrames * NumMels];
        int numBins = Nfft / 2 + 1; // 513
        var magSpec = new float[numBins];

        var windowed = new float[WinLength];

        for (int f = 0; f < numFrames; f++)
        {
            int off = f * HopLength;

            for (int n = 0; n < WinLength; n++)
            {
                int sampleIdx = off + n;
                float sample = (sampleIdx < paddedLen) ? signal[sampleIdx] : 0f;
                windowed[n] = sample * _hannWindow[n];
            }

            // Compute STFT magnitude for this frame
            SpectralKernels.ComputePowerSpectrum(windowed, magSpec);
            for (int k = 0; k < numBins; k++)
            {
                magSpec[k] = MathF.Sqrt(magSpec[k] + 1e-9f);
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
