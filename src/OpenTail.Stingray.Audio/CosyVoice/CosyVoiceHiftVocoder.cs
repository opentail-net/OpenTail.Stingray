namespace OpenTail.Stingray.Audio.CosyVoice;

/// <summary>
/// CosyVoice's HiFT vocoder. Thin wrapper over <see cref="HiFTVocoderKernels"/> -- the actual
/// NSF-source + ISTFTNet HiFiGAN math is shared with `Chatterbox/ChatterboxVocoder.cs`.
/// </summary>
public static class CosyVoiceHiftVocoder
{
    /// <summary>mel is channel-first [MelDim=80, T]. Returns the waveform samples.</summary>
    public static float[] Generate(IHiFTVocoderWeights w, float[] mel, int t, Random rng, float pitchScale = 1.0f) =>
        HiFTVocoderKernels.Generate(w, mel, t, rng, melDim: 80, pitchScale: pitchScale);
}
