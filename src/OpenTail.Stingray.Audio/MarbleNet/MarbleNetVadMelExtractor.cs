namespace OpenTail.Stingray.Audio.MarbleNet;

/// <summary>
/// Real 80-channel log-mel feature extraction for MarbleNet VAD, ported from
/// `marblenet_vad/runtime.cpp`'s `compute_marblenet_features` (not guessed). Differs from
/// <see cref="OpenTail.Stingray.Audio.Parakeet.ParakeetMelExtractor"/>'s NeMo pipeline in three
/// real ways: NO pre-emphasis, NO per-feature Z-normalization (config `normalize: "None"`), and a
/// real `pad_to=2` post-step that zero-pads the frame axis up to the next even frame count. Uses
/// the checkpoint's own shipped `preprocessor.featurizer.window`/`.fb` tensors (not recomputed).
/// STFT: `center=true`, zero (constant) padding by `n_fft/2` each side, `log(power + 2^-24)`.
/// </summary>
public sealed class MarbleNetVadMelExtractor
{
    private readonly float[] _window; // NFft-length, WinLength samples centered with zero-pad
    private readonly float[] _melFb;  // [NMels, NFft/2+1]
    private const float LogEps = 1f / (1 << 24);

    public MarbleNetVadMelExtractor(MarbleNetVadWeights w)
    {
        _window = new float[MarbleNetVadWeights.NFft];
        int lpad = (MarbleNetVadWeights.NFft - MarbleNetVadWeights.WinLength) / 2;
        w.Window.AsSpan(0, MarbleNetVadWeights.WinLength).CopyTo(_window.AsSpan(lpad));
        _melFb = w.MelFilterbank;
    }

    /// <summary>Returns `[frames * NMels]` flattened as `mel[f * NMels + m]`, frame-padded up to
    /// a multiple of `pad_to=2`.</summary>
    public float[] ExtractMel(ReadOnlySpan<float> pcm)
    {
        int nFft = MarbleNetVadWeights.NFft, hop = MarbleNetVadWeights.HopLength, nMels = MarbleNetVadWeights.NMels;
        int nSamples = pcm.Length;
        if (nSamples == 0) return [];

        int pad = nFft / 2;
        var padded = new float[pad + nSamples + pad];
        pcm.CopyTo(padded.AsSpan(pad));

        int t = (padded.Length - nFft) / hop + 1;
        if (t <= 0) return [];

        int paddedT = ((t + MarbleNetVadWeights.PadTo - 1) / MarbleNetVadWeights.PadTo) * MarbleNetVadWeights.PadTo;
        var mel = new float[paddedT * nMels];
        var frame = new float[nFft];
        var powerSpectrum = new float[nFft / 2 + 1];

        for (int f = 0; f < t; f++)
        {
            int start = f * hop;
            for (int i = 0; i < nFft; i++)
                frame[i] = padded[start + i] * _window[i];

            SpectralKernels.ComputePowerSpectrum(frame, powerSpectrum);

            for (int m = 0; m < nMels; m++)
            {
                float energy = 0f;
                int fbBase = m * powerSpectrum.Length;
                for (int k = 0; k < powerSpectrum.Length; k++)
                    energy += powerSpectrum[k] * _melFb[fbBase + k];

                mel[f * nMels + m] = MathF.Log(energy + LogEps);
            }
        }
        return mel;
    }
}
