using OpenTail.Stingray.Diffusion.AceStep.Text;
using OpenTail.Stingray.Diffusion.AceStep.Transformer;
using OpenTail.Stingray.Diffusion.AceStep.Vae;

namespace OpenTail.Stingray.Diffusion.AceStep;

/// <summary>
/// Weight bundle for ACE-Step Turbo's checkpoint components. See
/// docs/064-acestep-implementation-plan.md for real-vs-placeholder status: the DiT
/// (<see cref="AceStepDiTWeights"/>) and Oobleck VAE decoder
/// (<see cref="AceStepOobleckDecoderWeights"/>) are real, tested against real weights; the text
/// encoder (<see cref="AceStepQwen3TextEncoder"/>, a stateful class owning its own GGUF handle
/// rather than a bare weights record) is real and tested; the condition encoder
/// (<see cref="AceStepConditionEncoderWeights"/>) remains an unimplemented placeholder. This
/// class exists so <see cref="AceStepPipeline"/> has a real, typed dependency to construct
/// against once the remaining piece (condition encoder) is implemented.
/// </summary>
public sealed class AceStepModel
{
    public required AceStepDiTWeights Transformer { get; init; }
    public required AceStepOobleckDecoderWeights Vae { get; init; }
    public required AceStepQwen3TextEncoder TextEncoder { get; init; }
    public required AceStepConditionEncoderWeights ConditionEncoder { get; init; }
}

/// <summary>Placeholder for the condition encoder's weights (text projector + lyric encoder + timbre encoder). Not implemented -- see docs/064-acestep-implementation-plan.md's "Immediate next steps".</summary>
public sealed class AceStepConditionEncoderWeights;
