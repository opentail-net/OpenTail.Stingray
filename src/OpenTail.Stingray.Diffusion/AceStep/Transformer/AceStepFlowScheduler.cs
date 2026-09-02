namespace OpenTail.Stingray.Diffusion.AceStep.Transformer;

/// <summary>
/// Real Turbo flow-matching Euler-ODE sampling loop, transcribed from
/// `AceStepConditionGenerationModel.generate_audio` (`infer_method="ode"` branch -- the only path
/// used by Turbo's real hardcoded 8-step schedules; the `"sde"` branch is a real alternative in the
/// reference but not used by any real Turbo default, so not ported here) -- see
/// docs/064-acestep-implementation-plan.md.
///
/// <para><b>Real per-step update</b>: `vt = decoder(xt, t, t)` (Turbo always calls with
/// `timestep_r=timestep`, see <see cref="AceStepDiT"/>'s doc comment); on the FINAL step,
/// `x0 = xt - vt * t` directly (`get_x0_from_noise`) instead of an Euler step, since there is no
/// next timestep to step toward. On every other step, `dt = t_curr - t_next`, `xt = xt - vt * dt`
/// (real Euler ODE: `dx/dt = -v`).</para>
///
/// <para><b>Real gap, documented rather than silently assumed</b>: the real `diffusers`
/// `AceStepConditionEncoder` ships a learned `silence_latent` buffer (VAE-encoded real audio
/// silence) used as `src_latents` for plain text-to-music generation (no reference/cover audio) --
/// confirmed from `diffusers/pipelines/ace_step/pipeline_ace_step.py`'s `prepare_src_latents`. That
/// buffer is NOT present in the real `acestep-v15-turbo/model.safetensors` checkpoint this project
/// downloaded (confirmed by inspecting its real safetensors header -- no `silence_latent` key
/// anywhere), so it is evidently supplied by a separate converter/asset this project does not have.
/// This scheduler uses all-zero `src_latents` as an explicit, flagged placeholder for the V1
/// text-to-music (non-cover) path -- the real `diffusers` pipeline's comment warns that feeding
/// zeros into the TIMBRE encoder produces OOD/drone-like audio, but V1 never calls the timbre
/// encoder at all (text+lyrics-only condition encoder scope), so that specific warning does not
/// apply here; the open question is only how much zero `src_latents` (as opposed to real encoded
/// silence) degrades the DiT's own context-conditioning input. Revisit if/when the real
/// `silence_latent` buffer is located. `chunk_masks` = all-ones IS confirmed real for plain
/// generation (the same pipeline's `_build_chunk_mask` doc comment: "dumping the chunk_masks tensor
/// that generate_audio actually receives (unique values = [True])").</para>
/// </summary>
public static class AceStepFlowScheduler
{
    /// <summary>
    /// Runs the full real Turbo Euler-ODE denoising loop and returns the final clean latent
    /// `[latentFrames][AudioAcousticHiddenDim]` (25Hz acoustic latent, ready for
    /// <see cref="Vae.AceStepOobleckDecoder"/>).
    /// </summary>
    public static float[][] Generate(
        AceStepDiTWeights w,
        float[][] conditionSequence,
        int latentFrames,
        float shift,
        int? seed)
    {
        if (!AceStepConfig.ShiftTimestepSchedules.TryGetValue(shift, out var schedule))
        {
            // Real reference: any other requested shift snaps to the nearest of {1,2,3}.
            float nearest = AceStepConfig.ShiftTimestepSchedules.Keys
                .OrderBy(s => MathF.Abs(s - shift))
                .First();
            schedule = AceStepConfig.ShiftTimestepSchedules[nearest];
        }

        int acousticDim = AceStepConfig.AudioAcousticHiddenDim;

        // Real V1 gap: no real `silence_latent` buffer available (see class doc comment) --
        // src_latents=zeros, chunk_masks=ones (the latter IS confirmed real for plain generation).
        var contextLatents = new float[latentFrames][];
        for (int t = 0; t < latentFrames; t++)
        {
            var row = new float[2 * acousticDim];
            for (int i = acousticDim; i < 2 * acousticDim; i++) row[i] = 1f; // chunk_masks half = 1
            contextLatents[t] = row;
        }

        var rng = seed.HasValue ? new Random(seed.Value) : new Random();
        var xt = new float[latentFrames][];
        for (int t = 0; t < latentFrames; t++)
        {
            var row = new float[acousticDim];
            for (int i = 0; i < acousticDim; i++) row[i] = SampleStandardNormal(rng);
            xt[t] = row;
        }

        var (initialPatches, originalSeqLen) = AceStepDiT.ProjIn(w, contextLatents, xt);
        var ctx = AceStepDiT.PrepareCrossAttention(w, conditionSequence, initialPatches.Length);

        int numSteps = schedule.Length;
        for (int step = 0; step < numSteps; step++)
        {
            float currentT = schedule[step];

            var (patches, seqLen) = AceStepDiT.ProjIn(w, contextLatents, xt);
            var patchesOut = AceStepDiT.Forward(w, patches, currentT, currentT, ctx);
            var vt = AceStepDiT.ProjOut(w, patchesOut, seqLen);

            if (step == numSteps - 1)
            {
                // Real `get_x0_from_noise`: x0 = xt - vt * t.
                for (int t = 0; t < latentFrames; t++)
                    for (int i = 0; i < acousticDim; i++)
                        xt[t][i] -= vt[t][i] * currentT;
                break;
            }

            float nextT = schedule[step + 1];
            float dt = currentT - nextT;
            for (int t = 0; t < latentFrames; t++)
                for (int i = 0; i < acousticDim; i++)
                    xt[t][i] -= vt[t][i] * dt;
        }

        return xt;
    }

    /// <summary>Box-Muller standard normal sample -- matches this project's need for i.i.d. Gaussian noise; not bit-exact against real `torch.randn` (different RNG algorithm entirely), only statistically equivalent.</summary>
    private static float SampleStandardNormal(Random rng)
    {
        double u1 = 1.0 - rng.NextDouble();
        double u2 = rng.NextDouble();
        return (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
    }
}
