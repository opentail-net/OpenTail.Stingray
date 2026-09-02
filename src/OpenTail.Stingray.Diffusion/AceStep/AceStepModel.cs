
namespace OpenTail.Stingray.Diffusion.AceStep;

/// <summary>
/// Weight bundle for ACE-Step Turbo's four real checkpoint components (DiT, VAE, Qwen3 text
/// encoder, condition encoder). All four weight types are placeholder shells -- see
/// docs/064-acestep-implementation-plan.md; no forward pass exists yet for any of them. This
/// class exists so <see cref="AceStepPipeline"/> has a real, typed dependency to construct
/// against once each component is actually implemented, rather than being written against loose
/// tensors.
/// </summary>
public sealed class AceStepModel
{
    public required AceStepDiTWeights Transformer { get; init; }
    public required AceStepOobleckVaeWeights Vae { get; init; }
    public required AceStepQwen3TextEncoderWeights TextEncoder { get; init; }
    public required AceStepConditionEncoderWeights ConditionEncoder { get; init; }
}

/// <summary>Placeholder for the 24-layer ACE-Step DiT's weights. Not implemented -- see docs/064-acestep-implementation-plan.md's "Immediate next steps".</summary>
public sealed class AceStepDiTWeights;

/// <summary>Placeholder for the real `AutoencoderOobleck` decoder's weights. Not implemented -- the exact Snake1d (alpha+beta) formula needs verifying against real `diffusers` source first, see the plan doc.</summary>
public sealed class AceStepOobleckVaeWeights;

/// <summary>Placeholder for the Qwen3-Embedding-0.6B text encoder's weights. Not implemented -- whether ACE-Step's real pipeline uses causal or bidirectional attention for this encoding needs verifying first, see the plan doc.</summary>
public sealed class AceStepQwen3TextEncoderWeights;

/// <summary>Placeholder for the condition encoder's weights (text projector + lyric encoder + timbre encoder). Not implemented.</summary>
public sealed class AceStepConditionEncoderWeights;
