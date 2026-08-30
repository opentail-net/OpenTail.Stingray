
namespace OpenTail.Stingray.Audio.Xtts;

/// <summary>
/// Real XTTS-v2 speaker-encoder mel frontend (`ResNetSpeakerEncoder`'s `torch_spec`, confirmed
/// from `TTS/encoder/models/base_encoder.py`'s `get_torch_mel_spectrogram_class`): real
/// pre-emphasis (0.97) -&gt; `torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512,
/// win_length=400, hop_length=160, window_fn=torch.hamming_window, n_mels=64)` -- HAMMING window
/// (not Hann, unlike <see cref="XttsMelExtractor"/>), real un-overridden defaults `f_min=0`,
/// `f_max=sample_rate/2=8000`, `power=2.0`, `mel_scale="htk"`, and crucially `norm=None` (NO
/// Slaney area normalization this time -- a real, distinct mel config from the DVAE's, don't
/// reuse <see cref="XttsMelExtractor"/>'s filterbank).
///
/// <para>This is a THIRD, independent mel config in this port (after the DVAE's 22050Hz/80-mel/
/// Hann/Slaney-normalized config) -- each of XTTS-v2's sub-models was trained with its own
/// separate frontend, confirmed from source rather than assumed identical.</para>
/// </summary>
public sealed class XttsSpeakerMelExtractor
{
    public const int NumMels = 64;
    public const int SampleRate = 16000;
    public const int Nfft = 512;
    public const int HopLength = 160;
    public const int WinLength = 400;
    public const int Pad = Nfft / 2; // 256, real torch.stft center=True reflect padding
    public const float FMin = 0f;
    public const float FMax = SampleRate / 2f; // 8000
    public const float PreemphasisCoeff = 0.97f;

    private readonly float[][] _melBasis;
    private readonly float[] _hammingWindow;

    public XttsSpeakerMelExtractor()
    {
        // Real torch.stft behavior when win_length(400) < n_fft(512), confirmed directly against
        // torch.stft (NOT assumed): the FULL n_fft span of samples is read per frame, and the
        // window function is CENTER-padded to n_fft length with zeros on both sides (NOT left-
        // aligned) -- so _hammingWindow here is Nfft-length, with the real Hamming values placed
        // in the middle [((Nfft-WinLength)/2) .. +WinLength), zero elsewhere.
        _hammingWindow = new float[Nfft];
        int winOffset = (Nfft - WinLength) / 2;
        for (int i = 0; i < WinLength; i++)
            _hammingWindow[winOffset + i] = 0.54f - 0.46f * MathF.Cos(2.0f * MathF.PI * i / WinLength); // periodic=True default

        _melBasis = BuildMelBasis();
    }

    /// <summary>
    /// Real pre-emphasis: `y[0] = x[0] - coeff*x[1]` (reflect-padding boundary quirk, confirmed
    /// from the real `PreEmphasis` module's `F.pad(x,(1,0),"reflect")` + conv1d([-coeff,1])
    /// math), `y[i] = x[i] - coeff*x[i-1]` for i&gt;=1.
    /// </summary>
    public static float[] Preemphasis(ReadOnlySpan<float> pcm)
    {
        int n = pcm.Length;
        var output = new float[n];
        if (n == 0) return output;
        output[0] = pcm[0] - PreemphasisCoeff * (n > 1 ? pcm[1] : pcm[0]);
        for (int i = 1; i < n; i++)
            output[i] = pcm[i] - PreemphasisCoeff * pcm[i - 1];
        return output;
    }

    /// <summary>Extracts the real (un-normalized, no InstanceNorm/log applied here -- those are the caller's, `ResNetSpeakerEncoder.forward`'s own -- responsibility) power-mel-spectrogram, channel-first [64, numFrames].</summary>
    public float[] ExtractMel(ReadOnlySpan<float> pcmPreemphasized)
    {
        if (pcmPreemphasized.Length == 0) return [];

        int len = pcmPreemphasized.Length;
        int paddedLen = len + 2 * Pad;
        var signal = new float[paddedLen];

        for (int i = 0; i < Pad; i++)
        {
            int leftIdx = (Pad - i) % len;
            if (leftIdx < 0) leftIdx += len;
            signal[i] = pcmPreemphasized[leftIdx];

            int rightIdx = (len - 2 - i) % len;
            if (rightIdx < 0) rightIdx += len;
            signal[i + len + Pad] = pcmPreemphasized[rightIdx];
        }
        pcmPreemphasized.CopyTo(signal.AsSpan(Pad, len));

        // Each frame reads a full Nfft-length span of the padded signal (see below), so frame
        // count divides by Nfft, not WinLength (unlike XttsMelExtractor, where Nfft==WinLength
        // makes the two divisors identical).
        int numFrames = (paddedLen - Nfft) / HopLength + 1;
        if (numFrames <= 0) return [];

        int numBins = Nfft / 2 + 1;
        var mel = new float[NumMels * numFrames];
        var powerSpec = new float[numBins];
        // Real torch.stft behavior (Nfft=512 != WinLength=400 here, unlike XttsMelExtractor where
        // they're equal): read the FULL Nfft-length span of samples per frame, multiplied by the
        // center-padded window (see constructor) -- NOT a WinLength-length read + left-pad.
        var windowed = new float[Nfft];

        for (int f = 0; f < numFrames; f++)
        {
            int pcmOff = f * HopLength;
            for (int n = 0; n < Nfft; n++)
                windowed[n] = signal[pcmOff + n] * _hammingWindow[n];

            SpectralKernels.ComputePowerSpectrum(windowed, powerSpec);

            for (int m = 0; m < NumMels; m++)
            {
                float energy = 0f;
                var filter = _melBasis[m];
                for (int k = 0; k < numBins; k++)
                    energy += powerSpec[k] * filter[k];
                mel[m * numFrames + f] = energy;
            }
        }

        return mel;
    }

    private static float HzToMelHtk(float hz) => 2595.0f * MathF.Log10(1.0f + hz / 700.0f);
    private static float MelToHzHtk(float mel) => 700.0f * (MathF.Pow(10.0f, mel / 2595.0f) - 1.0f);

    /// <summary>Real triangular filterbank, HTK mel scale, NO Slaney area normalization (`norm=None`).</summary>
    private static float[][] BuildMelBasis()
    {
        int numBins = Nfft / 2 + 1;
        float mMin = HzToMelHtk(FMin);
        float mMax = HzToMelHtk(FMax);
        float step = (mMax - mMin) / (NumMels + 1);

        var fPts = new float[NumMels + 2];
        for (int i = 0; i < NumMels + 2; i++)
            fPts[i] = MelToHzHtk(mMin + i * step);

        var fDiff = new float[NumMels + 1];
        for (int i = 0; i < NumMels + 1; i++)
            fDiff[i] = fPts[i + 1] - fPts[i];

        var allFreqs = new float[numBins];
        for (int i = 0; i < numBins; i++)
            allFreqs[i] = i * (SampleRate / 2f) / (numBins - 1);

        var basis = new float[NumMels][];
        for (int i = 0; i < NumMels; i++)
        {
            basis[i] = new float[numBins];
            for (int j = 0; j < numBins; j++)
            {
                float down = (allFreqs[j] - fPts[i]) / fDiff[i];
                float up = (fPts[i + 2] - allFreqs[j]) / fDiff[i + 1];
                basis[i][j] = MathF.Max(0.0f, MathF.Min(down, up)); // no enorm -- norm=None
            }
        }

        return basis;
    }
}
