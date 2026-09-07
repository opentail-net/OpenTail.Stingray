namespace OpenTail.Stingray.Audio.VoxCpm2;

/// <summary>
/// Real VoxCPM2 AudioVAE config fields (`audio_vae_config` in the checkpoint's `config.json`),
/// parsed the same way `assets.cpp`'s `parse_audio_vae_config` does (not guessed). Decode-only
/// fields; the encoder-side (`sample_rate`, `encoder_rates`, `encoder_dim`) are captured too since
/// they appear in the same JSON block, but this pipeline's decoder never needs them (TTS only
/// ever DECODES generated latent features into audio, never re-encodes a reference clip inline --
/// see <see cref="VoxCpm2AudioVaeDecoder"/>'s doc comment for the analogous MOSS-TTS-Nano scoping
/// decision).
/// </summary>
public sealed class VoxCpm2AudioVaeConfig
{
    public required int LatentDim { get; init; }
    public required int DecoderDim { get; init; }
    public required int[] DecoderRates { get; init; }
    public required int OutputSampleRate { get; init; }
    public required int[] SampleRateBinBoundaries { get; init; }

    /// <summary>Real encoder-side fields, added 2026-09-07 for the AudioVAE ENCODER port (voice
    /// cloning / reference-audio conditioning). Confirmed real, NOT symmetric with the decoder
    /// (`EncoderRates=[2,5,8,8]`, 4 stages, vs `DecoderRates=[8,6,5,2,2,2]`, 6 stages) -- read
    /// directly from the real checkpoint's embedded `config.json`, not assumed/guessed.</summary>
    public required int EncoderDim { get; init; }
    public required int[] EncoderRates { get; init; }

    /// <summary>Real `sample_rate_bucket`: counts how many boundaries `output_sample_rate` exceeds.</summary>
    public int SampleRateBucket()
    {
        int bucket = 0;
        while (bucket < SampleRateBinBoundaries.Length && OutputSampleRate > SampleRateBinBoundaries[bucket])
            bucket++;
        return bucket;
    }

    public int SampleRateBucketCount => SampleRateBinBoundaries.Length + 1;
}
