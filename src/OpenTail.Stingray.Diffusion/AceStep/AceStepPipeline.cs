
namespace OpenTail.Stingray.Diffusion.AceStep;

/// <summary>
/// Top-level ACE-Step Turbo text-to-music pipeline. Deliberately unimplemented -- see
/// docs/064-acestep-implementation-plan.md's real, checkpoint-verified architecture and its
/// "Immediate next steps" for what has to exist before this can do anything (Qwen3 text encoder,
/// condition encoder, DiT, flow-matching scheduler, and the Oobleck VAE decoder, each with their
/// own golden/non-degeneracy check before being trusted, matching how MusicGen/AudioGen were
/// verified in this project).
///
/// <para>Intended real flow once implemented (real, from `AceStepConditionGenerationModel
/// .generate_audio`): encode prompt (+lyrics) via Qwen3 -&gt; condition encoder packs
/// text/lyric/timbre into one cross-attention sequence -&gt; sample Gaussian noise at the target
/// duration's latent length -&gt; run the real hardcoded shift-1/2/3 8-step Euler-ODE schedule
/// through the DiT (with cross-attention K/V computed once and reused across all 8 steps) -&gt;
/// decode the resulting latent through the real `AutoencoderOobleck` VAE -&gt; 48kHz stereo
/// PCM.</para>
/// </summary>
public sealed class AceStepPipeline
{
    private readonly AceStepModel _model;

    public AceStepPipeline(AceStepModel model)
    {
        _model = model;
    }

    public StereoAudioBuffer Generate(AceStepGenerationParams parameters) =>
        throw new NotImplementedException(
            "ACE-Step Turbo generation is not implemented yet -- see docs/064-acestep-implementation-plan.md's \"Immediate next steps\" for the real work still ahead (weight download, Qwen3/condition-encoder/DiT/scheduler/VAE ports, each golden-verified before use).");
}

/// <summary>Shared stereo audio buffer shape -- intended to eventually also be used by MusicGen/AudioGen's mono output and any future stereo audio model, per this plan's recommendation, but not yet wired into those (both remain mono `float[]` today; retrofitting them is a separate, deliberate follow-up, not bundled into this scaffold).</summary>
public readonly struct StereoAudioBuffer
{
    public required int SampleRate { get; init; }
    public required float[] Left { get; init; }
    public required float[] Right { get; init; }

    public int SampleCount => Left.Length;
    public double DurationSeconds => SampleCount / (double)SampleRate;
}
