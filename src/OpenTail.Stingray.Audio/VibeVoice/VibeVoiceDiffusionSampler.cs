namespace OpenTail.Stingray.Audio.VibeVoice;

/// <summary>
/// Real port of VibeVoice TTS's `diffusion_sampler.cpp`'s `sample_vibevoice_speech_latents` (not
/// guessed): classic batch-doubled classifier-free guidance around
/// <see cref="VibeVoiceDiffusionHead"/> and <see cref="VibeVoiceDpmSolverScheduler"/>. Each
/// solver step runs the head ONCE on `[positive_frames; negative_frames]` concatenated in the
/// batch dimension (both halves start from the SAME noisy sample -- `duplicate_positive_half`,
/// a real, deliberate detail: this is not "predict once per condition from independently-noised
/// samples", it is the same noisy latent evaluated under two different conditioning vectors),
/// then combines `eps = uncond + cfgScale*(cond - uncond)` and feeds that single combined eps to
/// BOTH halves of the next scheduler step (the reference always writes the same guided value
/// into both the cond and uncond slots of the eps buffer it hands to `scheduler.step`).
/// </summary>
public static class VibeVoiceDiffusionSampler
{
    /// <summary>Runs the full multi-step DPM-Solver++ CFG sampling loop. `positiveCondition`/
    /// `negativeCondition` are `[hidden]` (one condition vector, real reference batch_size=1 per
    /// call), `initialSpeech` is the starting noisy `[latent]` sample. Returns the final
    /// denoised `[latent]` speech latent.</summary>
    public static float[] Sample(
        VibeVoiceDiffusionHeadWeights headWeights,
        VibeVoiceDpmSolverScheduler scheduler,
        float[] positiveCondition,
        float[] negativeCondition,
        float[] initialSpeech,
        float guidanceScale)
    {
        int hidden = headWeights.HiddenSize;
        int latent = headWeights.LatentSize;
        if (positiveCondition.Length != hidden) throw new ArgumentException("positiveCondition length mismatch.");
        if (negativeCondition.Length != hidden) throw new ArgumentException("negativeCondition length mismatch.");
        if (initialSpeech.Length != latent) throw new ArgumentException("initialSpeech length mismatch.");

        var condition = new[] { positiveCondition, negativeCondition };
        var speech = initialSpeech;
        scheduler.ResetStepState();

        foreach (int timestep in scheduler.Timesteps)
        {
            var combined = new[] { speech, speech };
            var prediction = VibeVoiceDiffusionHead.Predict(headWeights, combined, condition, timestep);

            var cond = prediction[0];
            var uncond = prediction[1];
            var eps = new float[latent];
            for (int i = 0; i < latent; i++) eps[i] = uncond[i] + guidanceScale * (cond[i] - uncond[i]);

            speech = scheduler.Step(eps, timestep, speech);
        }

        return speech;
    }
}
