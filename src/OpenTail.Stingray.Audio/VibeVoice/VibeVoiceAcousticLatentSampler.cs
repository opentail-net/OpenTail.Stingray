namespace OpenTail.Stingray.Audio.VibeVoice;

/// <summary>
/// Real Gaussian VAE reparameterization for VibeVoice's acoustic tokenizer, ported from
/// `speech_tokenizer.cpp`'s `sample_vibevoice_acoustic_latents_gaussian` (not guessed): a real,
/// deliberately fixed-std (not learned logvar) reparameterization trick --
/// `std_scale = fix_std / 0.8`, one scalar `std` sampled per utterance (`std = randn() *
/// std_scale`), then every latent element gets `sampled = mean + std * randn()`. Confirmed real
/// via `speech_encoder.cpp`: this step IS needed for VibeVoice ASR too (a correction to this
/// project's own earlier assumption -- see the dated correction in
/// docs/audio-review-progress.md), not just the TTS voice-cloning path.
///
/// <para><b>Known gap, not worked around</b>: the reference generates its Gaussian noise via a
/// real PyTorch-CUDA-compatible Philox counter-based RNG at BF16 precision
/// (`generate_torch_cuda_randn`/`TorchRandnPrecision::BFloat16`) so that a fixed seed reproduces
/// bit-identical noise to the real PyTorch reference implementation. This class instead uses a
/// standard Box-Muller transform over .NET's own `Random` -- structurally the same
/// reparameterization formula, but NOT bit-identical to the reference for a given seed. Flagged
/// explicitly per this project's precision-gap discipline (same class of gap as
/// `UnigramTokenizer`'s `precompiled_charsmap` stand-in) rather than silently approximated.</para>
/// </summary>
public static class VibeVoiceAcousticLatentSampler
{
    /// <summary>Samples `[dim][frames]` channel-major latents from the encoder's real mean
    /// output, using the real `std = randn() * (fixStd / 0.8)` per-utterance scalar std.</summary>
    public static float[][] Sample(float[][] meanChannelMajor, float fixStd, Random rng)
    {
        if (fixStd <= 0f) throw new ArgumentOutOfRangeException(nameof(fixStd));

        float stdScale = fixStd / 0.8f;
        float std = (float)NextGaussian(rng) * stdScale;

        int dim = meanChannelMajor.Length;
        int frames = meanChannelMajor[0].Length;
        var output = new float[dim][];
        for (int c = 0; c < dim; c++)
        {
            var row = new float[frames];
            var meanRow = meanChannelMajor[c];
            for (int t = 0; t < frames; t++)
                row[t] = meanRow[t] + std * (float)NextGaussian(rng);
            output[c] = row;
        }
        return output;
    }

    /// <summary>Standard Box-Muller transform (not the reference's real Torch-CUDA RNG -- see
    /// class doc's "Known gap").</summary>
    private static double NextGaussian(Random rng)
    {
        double u1 = 1.0 - rng.NextDouble();
        double u2 = rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
    }
}
