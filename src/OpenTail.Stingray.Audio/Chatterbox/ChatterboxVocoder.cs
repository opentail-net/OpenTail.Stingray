using OpenTail.Stingray.Audio.Primitives;

namespace OpenTail.Stingray.Audio.Chatterbox;

/// <summary>
/// S3Gen stage 3: HiFTGenerator (examples/chatterbox-tts-py/chatterbox/models/s3gen/hifigan.py) --
/// Neural Source Filter + ISTFTNet vocoder turning a mel-spectrogram into a 24kHz waveform.
///
/// Thin wrapper over <see cref="HiFTVocoderKernels"/> -- the actual DSP math (NSF harmonic
/// sine source, learned inverse-STFT head, Snake-activated HiFiGAN resblocks, F0 prediction)
/// is shared with `CosyVoice/CosyVoiceHiftVocoder.cs`, extracted once both pipelines had real
/// weights to check against each other -- see docs/audio-review-progress.md's CosyVoice
/// section. Golden-verified against real PyTorch output (see `Tests.Audio/
/// ChatterboxVocoderTests.cs`); re-run after this extraction to confirm it's behavior-
/// preserving.
/// </summary>
public static class ChatterboxVocoder
{
    /// <summary>mel is channel-first [80, T]. Returns the waveform samples.</summary>
    public static float[] Generate(ChatterboxS3GenWeights w, float[] mel, int t, Random rng) =>
        HiFTVocoderKernels.Generate(w, mel, t, rng, melDim: 80);
}
