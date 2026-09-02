using OpenTail.Stingray.Diffusion.AceStep.Conditioning;
using OpenTail.Stingray.Diffusion.AceStep.Text;
using OpenTail.Stingray.Diffusion.AceStep.Transformer;
using OpenTail.Stingray.Diffusion.AceStep.Vae;

namespace OpenTail.Stingray.Diffusion.AceStep;

/// <summary>
/// Weight bundle for ACE-Step Turbo's checkpoint components. See
/// docs/064-acestep-implementation-plan.md for status: all real components (DiT, VAE decoder+
/// encoder, Qwen3 text encoder, condition encoder including its lyric and timbre sub-encoders) are
/// now real and individually golden-parity-verified against the real `diffusers` reference. `Vae`
/// (the decoder) and `VaeEncoder` are loaded from the same `vae.safetensors` -- the encoder is
/// needed only to self-derive a real `silence_latent` (see <see cref="Vae.AceStepOobleckEncoder"/>'s
/// doc comment), not for any real encode-then-decode round trip in V1.
/// </summary>
public sealed class AceStepModel
{
    public required AceStepDiTWeights Transformer { get; init; }
    public required AceStepOobleckDecoderWeights Vae { get; init; }
    public required AceStepOobleckEncoderWeights VaeEncoder { get; init; }
    public required AceStepQwen3TextEncoder TextEncoder { get; init; }
    public required AceStepConditionEncoderWeights ConditionEncoder { get; init; }
    public required AceStepTimbreEncoderWeights TimbreEncoder { get; init; }
}
