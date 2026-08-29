
namespace OpenTail.Stingray.Audio.CosyVoice;

/// <summary>
/// Real speech-token extractor for CosyVoice3's zero-shot reference-audio conditioning. Matches
/// <c>examples/cosyvoice.cpp</c>'s <c>cosyvoice_frontend_context::extract_speech_token</c>
/// tensor-for-tensor:
/// <list type="bullet">
/// <item><description>Sample rate: 16000 Hz</description></item>
/// <item><description>N_FFT: 400, WinLength: 400, HopLength: 160</description></item>
/// <item><description>Replicate (edge) padding of 200 samples on each side, PyTorch stft center=True convention</description></item>
/// <item><description>Frame count: floor(len / hop_length) -- matches the reference's own <c>[..., :-1]</c> trim</description></item>
/// <item><description>Periodic Hann window</description></item>
/// <item><description>128-bin mel filterbank via the same librosa-style basis as <see cref="CosyVoiceMelExtractor"/>, params (16000, 400, 128, 0, 8000)</description></item>
/// <item><description>Mel energy is the POWER-spectrum-weighted sum (no sqrt)</description></item>
/// <item><description>log10(max(1e-10, energy)), then dynamic-range normalize: max -= 8; v = (max(v,max)+4)/4</description></item>
/// <item><description>Fed to `cosyvoice_speech_tokenizer_v2.onnx` as [1,128,T] float32 `feats` + [1] int32 `feats_length`</description></item>
/// </list>
/// </summary>
public static class CosyVoiceSpeechTokenizer
{
    public const int SampleRate = 16000;
    public const int Nfft = 400;
    public const int WinLength = 400;
    public const int HopLength = 160;
    public const int NumMels = 128;

    private static readonly Lazy<float[][]> s_melBasis = new(() => BuildMelBasis(SampleRate, Nfft, NumMels, 0.0f, 8000.0f));
    private static readonly Lazy<float[]> s_hannWindow = new(() => BuildHannWindow(WinLength));

    /// <summary>
    /// Runs the real `cosyvoice_speech_tokenizer_v2.onnx` graph on 16kHz mono reference audio and
    /// returns its speech-token IDs, or null if the ONNX file isn't available at the given path.
    /// </summary>
    public static int[]? Extract(string speechTokenizerOnnxPath, ReadOnlySpan<float> pcm16k)
    {
        using var session = OnnxModelSession.TryLoad(speechTokenizerOnnxPath);
        if (session is null) return null;

        var (mel, numFrames) = ExtractMel(pcm16k);
        if (numFrames == 0) return null;

        string featsName = session.InputNames.FirstOrDefault(n => n.Contains("feats") && !n.Contains("length"))
            ?? session.InputNames.First();
        string lengthName = session.InputNames.FirstOrDefault(n => n.Contains("length"))
            ?? session.InputNames.Skip(1).First();

        var feats = mel;
        var length = new[] { numFrames };
        return session.RunToIntArray((featsName, feats, [1, NumMels, numFrames]), (lengthName, length, [1]));
    }

    /// <summary>
    /// Extracts the mel-basis-and-window-consistent [128, T] channel-first log-mel feature that
    /// `cosyvoice_speech_tokenizer_v2.onnx` expects. Returns (mel, numFrames).
    /// </summary>
    public static (float[] Mel, int NumFrames) ExtractMel(ReadOnlySpan<float> pcm16k)
    {
        int len = pcm16k.Length;
        int numFrames = len / HopLength;
        if (numFrames <= 0) return ([], 0);

        var padded = new float[len + Nfft];
        float left = pcm16k[0];
        float right = pcm16k[len - 1];
        for (int i = 0; i < Nfft / 2; i++)
        {
            padded[i] = left;
            padded[i + len + Nfft / 2] = right;
        }
        pcm16k.CopyTo(padded.AsSpan(Nfft / 2, len));

        var hann = s_hannWindow.Value;
        var melBasis = s_melBasis.Value;
        int numBins = Nfft / 2 + 1; // 201

        var mel = new float[NumMels * numFrames]; // channel-first [128, T]
        var windowed = new float[WinLength];
        var power = new float[numBins];

        for (int f = 0; f < numFrames; f++)
        {
            int off = f * HopLength;
            for (int n = 0; n < WinLength; n++)
                windowed[n] = padded[off + n] * hann[n];

            SpectralKernels.ComputePowerSpectrum(windowed, power);

            for (int m = 0; m < NumMels; m++)
            {
                float energy = 0f;
                var basisRow = melBasis[m];
                for (int k = 0; k < numBins; k++)
                    energy += power[k] * basisRow[k];

                mel[m * numFrames + f] = energy;
            }
        }

        float maxValue = 1e-10f;
        for (int i = 0; i < mel.Length; i++)
        {
            float v = MathF.Log10(MathF.Max(1e-10f, mel[i]));
            mel[i] = v;
            if (v > maxValue) maxValue = v;
        }

        maxValue -= 8f;
        for (int i = 0; i < mel.Length; i++)
        {
            float v = MathF.Max(mel[i], maxValue);
            mel[i] = (v + 4f) / 4f;
        }

        return (mel, numFrames);
    }

    private static float[] BuildHannWindow(int winLength)
    {
        var w = new float[winLength];
        for (int i = 0; i < winLength; i++)
            w[i] = 0.5f * (1.0f - MathF.Cos(2.0f * MathF.PI * i / winLength));
        return w;
    }

    private static float HzToMel(float freq)
    {
        const float fMin = 0.0f;
        const float fSp = 200.0f / 3.0f;
        const float minLogHz = 1000.0f;
        const float minLogMel = (minLogHz - fMin) / fSp;
        float logstep = MathF.Log(6.4f) / 27.0f;

        return freq >= minLogHz
            ? minLogMel + MathF.Log(freq / minLogHz) / logstep
            : (freq - fMin) / fSp;
    }

    private static float MelToHz(float mel)
    {
        const float fMin = 0.0f;
        const float fSp = 200.0f / 3.0f;
        const float minLogHz = 1000.0f;
        const float minLogMel = (minLogHz - fMin) / fSp;
        float logstep = MathF.Log(6.4f) / 27.0f;

        return mel >= minLogMel
            ? minLogHz * MathF.Exp(logstep * (mel - minLogMel))
            : mel * fSp + fMin;
    }

    private static float[][] BuildMelBasis(float sr, int nfft, int numMels, float fmin, float fmax)
    {
        int numBins = nfft / 2 + 1;
        float melFmin = HzToMel(fmin);
        float melFmax = HzToMel(fmax);
        float step = (melFmax - melFmin) / (numMels + 1);

        var melF = new float[numMels + 2];
        for (int i = 0; i < numMels + 2; i++)
            melF[i] = MelToHz(melFmin + i * step);

        var fdiff = new float[numMels + 1];
        for (int i = 0; i < numMels + 1; i++)
            fdiff[i] = melF[i + 1] - melF[i];

        var fftFreqs = new float[numBins];
        for (int i = 0; i < numBins; i++)
            fftFreqs[i] = i * sr / nfft;

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
