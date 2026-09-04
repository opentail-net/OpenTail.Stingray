namespace OpenTail.Stingray.Diffusion.StableAudio;

/// <summary>
/// Real `stable_audio_tools.inference.generation.generate_diffusion_cond` duration-padding and
/// `stable_audio_tools.inference.sampling.DistributionShift` timestep-warp formulas, confirmed from
/// GitHub `main` source (fetched fresh this session -- these functions do not exist in the PyPI
/// 0.0.19 release's `sampling.py`, which has no `DistributionShift`/`dist_shift` concept at all) and
/// from the real `model_config.json` for all three shipped checkpoints (Small Music, Small SFX,
/// Medium), which all set `use_effective_length_for_schedule: true`, `mask_padding_attention: true`,
/// `distribution_shift_options: {"type": "full", "min_length": 256, "max_length": 4096}` (Small)/
/// `{"min_length": 256, "max_length": 4096}` (Medium, `type` omitted -- real code defaults to "full").
///
/// <para>Shared between <see cref="StableAudioPipeline"/> (Small) and
/// <see cref="StableAudioMediumPipeline"/> (Medium) since the real formula and config values are
/// byte-identical between all three checkpoints (CLAUDE.md rule 7).</para>
///
/// <para><b>Known, documented gap</b>: the real reference also masks self-/cross-attention over the
/// padding region (`mask_padding_attention: true`) so the DiT never attends INTO the padded tail.
/// Neither <c>StableAudioDiT</c> nor <c>StableAudioMediumDiT</c> currently accepts an attention mask
/// at all (both run full, unmasked attention over the whole latent) -- wiring that through is a
/// larger structural change than this pass covers, so it is NOT implemented here. The padding
/// itself, the schedule shift, and the final trim ARE implemented; the model still "sees" the
/// padded tail unmasked during generation, which is a real, acknowledged divergence from the
/// reference until attention masking is added.</para>
/// </summary>
public static class StableAudioScheduleKernels
{
    private const float DurationPaddingSeconds = 6.0f; // real `duration_padding_sec` default
    private const int MinLatentLength = 256;  // real `distribution_shift_options.min_length`
    private const int MaxLatentLength = 4096; // real `distribution_shift_options.max_length`
    private const float BaseShift = 0.5f;     // real `DistributionShift.base_shift` default
    private const float MaxShift = 1.15f;     // real `DistributionShift.max_shift` default

    /// <summary>Real effective (unpadded) latent frame count for <paramref name="durationSeconds"/>.</summary>
    public static int EffectiveSeqLen(float durationSeconds, float latentFrameRate) =>
        (int)Math.Ceiling(durationSeconds * latentFrameRate);

    /// <summary>
    /// Real `adapt_duration_to_conditioning` padding: generate at
    /// `requestedSeconds + duration_padding_sec` frames (capped at <see cref="MaxLatentLength"/>,
    /// which doubles as this port's practical max-length cap since the real per-checkpoint
    /// `sample_size` isn't independently available here). Never shorter than the effective length.
    /// </summary>
    public static int PaddedSeqLen(int effectiveSeqLen, float latentFrameRate)
    {
        int padded = effectiveSeqLen + (int)MathF.Ceiling(DurationPaddingSeconds * latentFrameRate);
        return Math.Clamp(padded, effectiveSeqLen, MaxLatentLength);
    }

    /// <summary>
    /// Real `DistributionShift.shift` (scalar path, `sigma=1.0`, `use_sine=false` -- neither checkpoint
    /// sets `use_sine`): re-times a uniform `t` using the EFFECTIVE (unpadded) sequence length, so
    /// short generations get a schedule shifted toward higher noise levels for longer relative to a
    /// naive linear schedule -- this is the real timestep warp EVERY real generation applies and this
    /// port previously skipped entirely (plain linear `t = 1 - step/steps`).
    /// </summary>
    public static float ShiftTimestep(float t, int effectiveSeqLen)
    {
        int clamped = Math.Clamp(effectiveSeqLen, MinLatentLength, MaxLatentLength);
        float mu = -(BaseShift + (MaxShift - BaseShift) * (clamped - MinLatentLength) / (MaxLatentLength - MinLatentLength));
        float expMu = MathF.Exp(mu);
        // Real: t_out = 1 - exp(mu) / (exp(mu) + (1/(1-t) - 1)^sigma), sigma=1.0.
        float oneMinusT = Math.Max(1e-7f, 1f - t);
        float ratioTerm = 1f / oneMinusT - 1f;
        return 1f - expMu / (expMu + ratioTerm);
    }
}
