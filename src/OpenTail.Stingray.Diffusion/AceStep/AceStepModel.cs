using OpenTail.Stingray.Diffusion.AceStep.Conditioning;
using OpenTail.Stingray.Diffusion.AceStep.Text;
using OpenTail.Stingray.Diffusion.AceStep.Transformer;
using OpenTail.Stingray.Diffusion.AceStep.Vae;

namespace OpenTail.Stingray.Diffusion.AceStep;

/// <summary>
/// Weight bundle for ACE-Step Turbo's checkpoint components. See
/// docs/064-acestep-implementation-plan.md for status: all four components (DiT, VAE decoder,
/// Qwen3 text encoder, condition encoder) are now real and individually tested against real
/// weights; only end-to-end pipeline wiring (<see cref="AceStepPipeline"/>) and the flow-matching
/// scheduler loop remain.
/// </summary>
public sealed class AceStepModel
{
    public required AceStepDiTWeights Transformer { get; init; }
    public required AceStepOobleckDecoderWeights Vae { get; init; }
    public required AceStepQwen3TextEncoder TextEncoder { get; init; }
    public required AceStepConditionEncoderWeights ConditionEncoder { get; init; }
}
