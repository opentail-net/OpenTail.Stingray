
namespace OpenTail.Stingray.Audio.CosyVoice;

/// <summary>
/// CosyVoice2's HiFT vocoder. Thin wrapper over <see cref="HiFTVocoderKernels"/> -- the actual
/// NSF-source + ISTFTNet HiFiGAN math is shared with `Chatterbox/ChatterboxVocoder.cs` (real,
/// golden-verified against PyTorch), extracted once both pipelines had real weights to check
/// against each other -- see docs/audio-review-progress.md's CosyVoice section.
///
/// NOT YET golden-verified against a real oracle -- structurally complete, same caveat as
/// every pipeline in this doc before its golden-verification pass.
/// </summary>
public static class CosyVoiceHiftVocoder
{
    /// <summary>mel is channel-first [MelDim=80, T]. Returns the waveform samples. Accepts any real <see cref="IHiFTVocoderWeights"/> implementation (CosyVoice2's weight-norm-folded safetensors or CosyVoice3's plain GGUF weights) -- the forward math itself is identical across both checkpoints.</summary>
    public static float[] Generate(IHiFTVocoderWeights w, float[] mel, int t, Random rng, float pitchScale = 1.0f) =>
        HiFTVocoderKernels.Generate(w, mel, t, rng, melDim: 80, pitchScale: pitchScale);
}
