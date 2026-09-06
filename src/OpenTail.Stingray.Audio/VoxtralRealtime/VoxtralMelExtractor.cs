
namespace OpenTail.Stingray.Audio.VoxtralRealtime;

/// <summary>
/// Real log-mel frontend for Voxtral Realtime's audio tower, transcribed directly from
/// `examples/audio.cpp/src/models/voxtral_realtime/frontend.cpp` (not guessed). Structurally the
/// same Slaney 128-bin filterbank family already used for Qwen3-ASR
/// (<see cref="QwenASR.QwenAsrMelExtractor"/>) -- fmin=0/fmax=8000Hz, `(log10(mel)+4)/4` scaling
/// -- but with two real, checkpoint-confirmed differences: `n_fft=win_length=400` (no
/// zero-padding beyond the analysis window, unlike Qwen3-ASR's `n_fft=512 &gt; window=400`), and a
/// **fixed** dynamic-range floor (`global_log_mel_max - 8`, from the real `config.json`'s
/// `global_log_mel_max=1.5`) rather than Whisper/Qwen3-ASR's per-utterance `max(log_spec) - 8`.
/// </summary>
public sealed class VoxtralMelExtractor
{
    public const int SampleRate = 16000;
    public const int NumMels = 128;
    public const int NFft = 400;
    public const int WindowSize = 400;
    public const int HopLength = 160;
    public const float GlobalLogMelMax = 1.5f;

    private readonly float[] _hannWindow;
    private readonly float[][] _melFilters;

    public VoxtralMelExtractor()
    {
        _hannWindow = SpectralKernels.CreateSymmetricHannWindow(WindowSize);
        _melFilters = CreateSlaneyMelFilterBank(NumMels, NFft, SampleRate, 0.0f, SampleRate / 2.0f);
    }

    /// <summary>Computes 128-channel normalized log-mel spectrogram. Output shape
    /// [NumMels, numFrames] flattened as mel[m * numFrames + f].</summary>
    public float[] ExtractMel(ReadOnlySpan<float> pcm)
    {
        if (pcm.Length < WindowSize) return [];

        int numFrames = Math.Max(1, (pcm.Length - WindowSize) / HopLength + 1);
        var mel = new float[NumMels * numFrames];
        var pcmArray = pcm.ToArray();
        float floorValue = GlobalLogMelMax - 8.0f;

        System.Threading.Tasks.Parallel.For(0, numFrames,
            () => (new float[NFft], new float[NFft / 2 + 1]),
            (f, _, buffers) =>
            {
                var (real, powerSpectrum) = buffers;
                int startSample = f * HopLength;
                Array.Clear(real, 0, NFft);
                for (int i = 0; i < WindowSize; i++)
                {
                    int sIdx = startSample + i;
                    float sample = (sIdx < pcmArray.Length) ? pcmArray[sIdx] : 0.0f;
                    real[i] = sample * _hannWindow[i];
                }

                SpectralKernels.ComputePowerSpectrum(real, powerSpectrum);
                for (int k = 0; k <= NFft / 2; k++) powerSpectrum[k] /= NFft;

                for (int m = 0; m < NumMels; m++)
                {
                    float energy = System.Numerics.Tensors.TensorPrimitives.Dot((ReadOnlySpan<float>)powerSpectrum, (ReadOnlySpan<float>)_melFilters[m]);
                    float logVal = MathF.Log10(MathF.Max(energy, 1e-10f));
                    float clamped = MathF.Max(logVal, floorValue);
                    mel[m * numFrames + f] = (clamped + 4.0f) / 4.0f;
                }
                return buffers;
            },
            _ => { });

        return mel;
    }

    private static float[][] CreateSlaneyMelFilterBank(int numMels, int nFft, int sampleRate, float fMin, float fMax)
    {
        int numBins = nFft / 2 + 1;
        var filters = new float[numMels][];
        for (int m = 0; m < numMels; m++) filters[m] = new float[numBins];

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
                    filters[m][k] = (k - left) / (center - left);
                else if (k >= center && k <= right && right > center)
                    filters[m][k] = (right - k) / (right - center);
            }
        }
        return filters;
    }
}
