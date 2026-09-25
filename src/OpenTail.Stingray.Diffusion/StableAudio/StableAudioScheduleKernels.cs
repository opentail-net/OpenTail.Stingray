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

    // Real SAMPLING schedule default (`DiffusionModel.sampling_dist_shift` when the config has no
    // `sampling_distribution_shift_options`, true for all three shipped base checkpoints):
    // `LogSNRShift(rate=0, anchor_logsnr=-6.2, logsnr_end=2.0)`, anchor_length default 2000.
    private const float LogSnrAnchor = -6.2f;
    private const float LogSnrEnd = 2.0f;
    private const float LogSnrRate = 0f;
    private const float LogSnrAnchorLength = 2000f;

    /// <summary>
    /// The real inference-time timestep warp: `LogSNRShift.shift` with the model's sampling defaults
    /// (`models/diffusion.py`: `sampling_dist_shift = LogSNRShift(rate=0, anchor_logsnr=-6.2,
    /// logsnr_end=2.0)`; the vendored C++ reference's `shifted_logsnr_timestep` is the same):
    /// <c>logsnr = end - t·(end - start)</c>, <c>t_out = sigmoid(-logsnr)</c>, endpoints 0 and 1 kept
    /// exact. With rate 0 the schedule doesn't depend on length; the parameter stays for configs with
    /// rate &gt; 0.
    ///
    /// <para><b>Fixed 2026-09-25</b>: this used to apply `DistributionShift` from
    /// `distribution_shift_options`, which is the TRAINING timestep distribution, not the sampling
    /// schedule. That warped every step toward the wrong noise levels and was why our generations
    /// came out muffled (7-20× less high-frequency energy than the reference) and insensitive to
    /// CFG.</para>
    /// </summary>
    public static float ShiftTimestep(float t, int effectiveSeqLen)
    {
        if (t <= 0f) return 0f;
        if (t >= 1f) return 1f;
        float logSnrStart = LogSnrAnchor - LogSnrRate * MathF.Log2(Math.Max(1, effectiveSeqLen) / LogSnrAnchorLength);
        float logSnr = LogSnrEnd - t * (LogSnrEnd - logSnrStart);
        return 1f / (1f + MathF.Exp(logSnr));
    }

    /// <summary>The TRAINING timestep distribution warp (`DistributionShift`, from the checkpoint's
    /// `distribution_shift_options`). Not used for sampling; kept for reference and tests.</summary>
    public static float TrainingDistributionShift(float t, int effectiveSeqLen)
    {
        int clamped = Math.Clamp(effectiveSeqLen, MinLatentLength, MaxLatentLength);
        float mu = -(BaseShift + (MaxShift - BaseShift) * (clamped - MinLatentLength) / (MaxLatentLength - MinLatentLength));
        float expMu = MathF.Exp(mu);
        float oneMinusT = Math.Max(1e-7f, 1f - t);
        float ratioTerm = 1f / oneMinusT - 1f;
        return 1f - expMu / (expMu + ratioTerm);
    }
}
