
namespace OpenTail.Stingray.Audio.CosyVoice;

/// <summary>
/// 80-channel Mel-Spectrogram feature extractor for 24kHz audio in CosyVoice3.
/// Matches <c>examples/cosyvoice.cpp</c>'s <c>cosyvoice_frontend_context::extract_speech_feat</c>
/// and <c>build_mel_basis(24000.f, 1920, 80, 0.0f, 12000.0f)</c> tensor-for-tensor:
/// <list type="bullet">
/// <item><description>Sample rate: 24000 Hz</description></item>
/// <item><description>N_FFT: 1920, WinLength: 1920, HopLength: 480</description></item>
/// <item><description>F_min: 0 Hz, F_max: 12000 Hz, NumMels: 80</description></item>
/// <item><description>Reflection padding: 720 samples on left and right</description></item>
/// <item><description>Periodic Hann window</description></item>
/// <item><description>Slaney mel filterbank normalization (2.0 / (mel_f[i+2] - mel_f[i]))</description></item>
/// <item><description>Log compression: ln(max(1e-5, energy))</description></item>
/// </list>
/// </summary>
public sealed class CosyVoiceMelExtractor
{
    public const int NumMels = 80;
    public const int SampleRate = 24000;
    public const int Nfft = 1920;
    public const int HopLength = 480;
    public const int WinLength = 1920;
    public const int Pad = 720;

    private readonly float[][] _melBasis; // [80][961]
    private readonly float[] _hannWindow; // [1920]

    private static readonly Lazy<CosyVoiceMelExtractor> s_instance = new(() => new CosyVoiceMelExtractor());
    public static CosyVoiceMelExtractor Shared => s_instance.Value;

    public CosyVoiceMelExtractor()
    {
        _hannWindow = new float[WinLength];
        for (int i = 0; i < WinLength; i++)
        {
            _hannWindow[i] = 0.5f * (1.0f - MathF.Cos(2.0f * MathF.PI * i / WinLength));
        }

        _melBasis = BuildMelBasis(SampleRate, Nfft, NumMels, 0.0f, 12000.0f);
    }

    /// <summary>
    /// Extracts 80-channel log mel-spectrogram [numFrames, 80] from 24kHz audio PCM samples.
    /// Channel-last flattened array of length <c>numFrames * 80</c>.
    /// </summary>
    public float[] ExtractMel(ReadOnlySpan<float> pcm)
    {
        if (pcm.Length == 0) return [];

        int len = pcm.Length;
        int paddedLen = len + 2 * Pad;
        var signal = new float[paddedLen];

        // Reflect padding (mode='reflect' matching PyTorch / cosyvoice.cpp)
        for (int i = 0; i < Pad; i++)
        {
            int leftIdx = (Pad - 1 - i) % len;
            signal[i] = pcm[leftIdx];

            int rightIdx = (len - 1 - i) % len;
            if (rightIdx < 0) rightIdx += len;
            signal[i + len + Pad] = pcm[rightIdx];
        }
        pcm.CopyTo(signal.AsSpan(Pad, len));

        int numFrames = (paddedLen - WinLength) / HopLength + 1;
        if (numFrames <= 0) return [];

        int numBins = Nfft / 2 + 1; // 961
        var mels = new float[numFrames * NumMels];
        var magSpec = new float[numBins];
        var windowed = new float[WinLength];

        for (int f = 0; f < numFrames; f++)
        {
            int pcmOff = f * HopLength;
            for (int n = 0; n < WinLength; n++)
            {
                windowed[n] = signal[pcmOff + n] * _hannWindow[n];
            }

            // Power spectrum -> magnitude spectrum
            SpectralKernels.ComputePowerSpectrum(windowed, magSpec);
            for (int k = 0; k < numBins; k++)
            {
                magSpec[k] = MathF.Sqrt(magSpec[k]);
            }

            int melOff = f * NumMels;
            for (int m = 0; m < NumMels; m++)
            {
                float energy = 0f;
                var filter = _melBasis[m];
                for (int k = 0; k < numBins; k++)
                {
                    energy += magSpec[k] * filter[k];
                }

                mels[melOff + m] = MathF.Log(MathF.Max(1e-5f, energy));
            }
        }

        return mels;
    }

    private static float HzToMel(float freq)
    {
        const float fMin = 0.0f;
        const float fSp = 200.0f / 3.0f;
        const float minLogHz = 1000.0f;
        const float minLogMel = (minLogHz - fMin) / fSp;
        float logstep = MathF.Log(6.4f) / 27.0f;

        if (freq >= minLogHz)
            return minLogMel + MathF.Log(freq / minLogHz) / logstep;
        else
            return (freq - fMin) / fSp;
    }

    private static float MelToHz(float mel)
    {
        const float fMin = 0.0f;
        const float fSp = 200.0f / 3.0f;
        const float minLogHz = 1000.0f;
        const float minLogMel = (minLogHz - fMin) / fSp;
        float logstep = MathF.Log(6.4f) / 27.0f;

        if (mel >= minLogMel)
            return minLogHz * MathF.Exp(logstep * (mel - minLogMel));
        else
            return mel * fSp + fMin;
    }

    private static float[][] BuildMelBasis(float sr, int nfft, int numMels, float fmin, float fmax)
    {
        int numBins = nfft / 2 + 1; // 961
        float melFmin = HzToMel(fmin);
        float melFmax = HzToMel(fmax);
        float step = (melFmax - melFmin) / (numMels + 1);

        var melF = new float[numMels + 2];
        for (int i = 0; i < numMels + 2; i++)
        {
            melF[i] = MelToHz(melFmin + i * step);
        }

        var fdiff = new float[numMels + 1];
        for (int i = 0; i < numMels + 1; i++)
        {
            fdiff[i] = melF[i + 1] - melF[i];
        }

        var fftFreqs = new float[numBins];
        for (int i = 0; i < numBins; i++)
        {
            fftFreqs[i] = i * sr / nfft;
        }

        var basis = new float[numMels][];
        for (int i = 0; i < numMels; i++)
        {
            basis[i] = new float[numBins];
            float enorm = 2.0f / (melF[i + 2] - melF[i]);

            for (int j = 0; j < numBins; j++)
            {
                float lower = -(melF[i] - fftFreqs[j]) / fdiff[i];
                float upper = (melF[i + 2] - fftFreqs[j]) / fdiff[i + 1];
                float val = MathF.Max(0.0f, MathF.Min(lower, upper));
                basis[i][j] = val * enorm;
            }
        }

        return basis;
    }
}
