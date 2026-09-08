namespace OpenTail.Stingray.Audio.Citrinet;

/// <summary>
/// Real 80-channel log-mel feature extraction for Citrinet ASR, ported from
/// `citrinet_asr/runtime.cpp`'s `compute_citrinet_features` (not guessed). Real, deliberately
/// different from `MarbleNet`'s extractor: NO pre-emphasis, but a real per-feature (per-mel-band)
/// Z-normalization (`normalize: "per_feature"`, `FeatureNormalizer`'s real formula -- Bessel-
/// corrected variance, `eps=1e-5` added OUTSIDE the sqrt, same formula already used by
/// `OpenTail.Stingray.Audio.Parakeet.ParakeetMelExtractor`), THEN a real `pad_to=16` frame-count
/// zero-pad (applied AFTER normalization, unlike MarbleNet where there's no normalization step at
/// all). Uses the checkpoint's own shipped window/filterbank tensors.
/// </summary>
public sealed class CitrinetAsrMelExtractor
{
    private readonly float[] _window;
    private readonly float[] _melFb;
    private const float LogEps = 1f / (1 << 24);

    public CitrinetAsrMelExtractor(CitrinetAsrWeights w)
    {
        _window = new float[CitrinetAsrWeights.NFft];
        int lpad = (CitrinetAsrWeights.NFft - CitrinetAsrWeights.WinLength) / 2;
        w.Window.AsSpan(0, CitrinetAsrWeights.WinLength).CopyTo(_window.AsSpan(lpad));
        _melFb = w.MelFilterbank;
    }

    /// <summary>Returns `(frames, [frames * NMels] flattened as mel[f * NMels + m], rawFrames)`
    /// -- `rawFrames` is the true (pre-`pad_to`) frame count, needed downstream to know how many
    /// output frames are real vs. zero-padding.</summary>
    public (float[] Values, int RawFrames, int PaddedFrames) ExtractMel(ReadOnlySpan<float> pcm, int padTo)
    {
        int nFft = CitrinetAsrWeights.NFft, hop = CitrinetAsrWeights.HopLength, nMels = CitrinetAsrWeights.NMels;
        int nSamples = pcm.Length;
        if (nSamples == 0) return ([], 0, 0);

        int pad = nFft / 2;
        var padded = new float[pad + nSamples + pad];
        pcm.CopyTo(padded.AsSpan(pad));

        int t = (padded.Length - nFft) / hop + 1;
        if (t <= 0) return ([], 0, 0);

        var mel = new float[t * nMels];
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

        // Real per-feature (per-mel-band) Z-normalization, Bessel-corrected variance, eps OUTSIDE sqrt.
        int denom = t > 1 ? t - 1 : 1;
        for (int m = 0; m < nMels; m++)
        {
            double sum = 0;
            for (int f = 0; f < t; f++) sum += mel[f * nMels + m];
            double mean = sum / t;

            double sq = 0;
            for (int f = 0; f < t; f++)
            {
                double d = mel[f * nMels + m] - mean;
                sq += d * d;
            }
            float std = MathF.Sqrt((float)(sq / denom));
            if (float.IsNaN(std)) std = 0f;
            std += 1e-5f;

            for (int f = 0; f < t; f++)
                mel[f * nMels + m] = (float)((mel[f * nMels + m] - mean) / std);
        }

        int paddedT = padTo > 1 ? ((t + padTo - 1) / padTo) * padTo : t;
        if (paddedT != t)
        {
            var withPad = new float[paddedT * nMels];
            mel.AsSpan().CopyTo(withPad);
            mel = withPad;
        }
        return (mel, t, paddedT);
    }
}
