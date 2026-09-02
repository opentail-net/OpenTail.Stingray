
namespace OpenTail.Stingray.Diffusion.AceStep;

/// <summary>
/// Public generation request for ACE-Step Turbo text-to-music. V1 scope only (see
/// docs/064-acestep-implementation-plan.md): text + optional lyrics -&gt; 48kHz stereo WAV, no
/// planner LM, no cover/repaint/extract/lego/complete editing modes, no reference-audio timbre
/// conditioning. <see cref="Shift"/> must be one of the three real supported values (1, 2, or 3) --
/// the real reference silently snaps any other value to the nearest of those three, this type
/// does not validate that itself (left to the pipeline, once it exists, to match real behavior
/// exactly rather than fail fast in a way the reference wouldn't).
/// </summary>
public sealed class AceStepGenerationParams
{
    public required string Prompt { get; init; }

    public string Lyrics { get; init; } = "";

    public float DurationSeconds { get; init; } = 30f;

    public int InferenceSteps { get; init; } = AceStepConfig.DefaultInferenceSteps;

    public float Shift { get; init; } = AceStepConfig.DefaultShift;

    public int? Seed { get; init; }

    public bool Instrumental { get; init; }
}
