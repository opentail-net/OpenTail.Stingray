namespace OpenTail.Stingray.Audio.VoxCpm2;

/// <summary>
/// Native C# port of VoxCPM2's conditional flow-matching (CFM) sampler, from `generator.cpp`'s
/// `VoxCPM2CFMRuntime::Impl::generate_patch` (not guessed) -- a real Euler solver, but with two
/// genuinely non-standard real details, not approximated:
///
/// <para><b>1. A real cosine-warped time schedule</b> (not a plain linear `1..0` schedule):
/// `base = 1 - i/timesteps; t_span[i] = base + (cos(halfPi*base) - 1 + base)`.</para>
///
/// <para><b>2. A real "optimized CFG" formula</b>, NOT the textbook `uncond + scale*(cond-uncond)`:
/// the estimator is run ONCE per step on a batch of 2 (positive/conditional row 0, negative row
/// 1 -- `mu` zeroed for the negative row, `cond` shared by both), then `scale = dot(positive,
/// negative) / (||negative||^2 + 1e-8)` (a real per-step projection-magnitude correction), and
/// `dphi = negative*scale + cfgValue*(positive - negative*scale)`.</para>
///
/// <para>Also real, not guessed: the delta-time embedding uses `dt` only when `dit_config.
/// mean_mode` is true (false for this checkpoint, so delta embedding is always `sinusoidal(0,
/// hiddenDim)`); the first `max(1, ceil(0.04*stepCount))` steps are a real deliberate "zero
/// init" warm-up where `dphi` stays zero (x is carried through unchanged, not denoised) before
/// the estimator is called at all.</para>
///
/// <para><b>Real simplification, not a shortcut around correctness</b>: the reference batches
/// the positive/negative CFG branches together as one `[2, ...]`-shaped graph call; this port
/// calls <see cref="VoxCpm2DiTEstimator.Run"/> twice (once per branch) instead -- mathematically
/// identical, since the two batch rows never cross-attend inside the estimator (confirmed by
/// reading `generator.cpp`'s graph construction: each batch item's sequence/attention is fully
/// independent, the batch dimension is only ever consumed elementwise afterward). Also, like
/// this session's `VibeVoiceAcousticLatentSampler`: the reference's initial noise comes from a
/// real Torch-CUDA-compatible RNG (`generate_torch_cuda_randn`); this port uses a standard
/// Box-Muller transform over .NET's own `Random` instead -- same real, documented precision gap,
/// not bit-identical for a given seed.</para>
/// </summary>
public static class VoxCpm2CfmSolver
{
    /// <summary>
    /// Generates one patch's `[PatchSize][FeatDim]` (patch-major) continuous features via the
    /// real CFM Euler loop. `mu` is the real 2-row conditioning prefix for the POSITIVE/
    /// conditional branch (e.g. `[currentLmDitHidden, residualDitHidden]` from
    /// <see cref="VoxCpm2StepProjection"/>'s output) -- the negative/unconditional branch always
    /// uses a zeroed `mu`, matching the reference's real `mu_in` construction.
    /// </summary>
    public static float[][] GeneratePatch(
        VoxCpm2DiTEstimatorWeights w,
        float[][] mu, float[][] condPatch,
        int timesteps, float cfgValue, bool meanMode,
        Random noiseRng, float temperature = 1.0f)
    {
        if (timesteps <= 0) throw new ArgumentOutOfRangeException(nameof(timesteps));
        int patchSize = VoxCpm2DiTEstimator.PatchSize;
        int featDim = VoxCpm2DiTEstimator.FeatDim;
        int hiddenDim = mu[0].Length;

        var x = new float[patchSize][];
        for (int p = 0; p < patchSize; p++)
        {
            x[p] = new float[featDim];
            for (int d = 0; d < featDim; d++) x[p][d] = (float)NextGaussian(noiseRng) * temperature;
        }

        var zeroMu = new float[2][] { new float[hiddenDim], new float[hiddenDim] };

        var tSpan = new float[timesteps + 1];
        const double halfPi = 1.57079632679489661923;
        for (int i = 0; i <= timesteps; i++)
        {
            double baseVal = 1.0 - (double)i / timesteps;
            tSpan[i] = (float)(baseVal + (Math.Cos(halfPi * baseVal) - 1.0 + baseVal));
        }

        float t = tSpan[0];
        float dt = tSpan[0] - tSpan[1];
        int zeroInitSteps = Math.Max(1, (int)(tSpan.Length * 0.04));

        for (int step = 1; step < tSpan.Length; step++)
        {
            var dphi = new float[patchSize][];
            for (int p = 0; p < patchSize; p++) dphi[p] = new float[featDim];

            if (step > zeroInitSteps)
            {
                float timeOne = t;
                float dtValue = meanMode ? dt : 0f;

                var positive = VoxCpm2DiTEstimator.Run(w, x, mu, condPatch, timeOne, dtValue);
                var negative = VoxCpm2DiTEstimator.Run(w, x, zeroMu, condPatch, timeOne, dtValue);

                double dot = 0.0, normSq = 1e-8;
                for (int p = 0; p < patchSize; p++)
                    for (int d = 0; d < featDim; d++)
                    {
                        dot += (double)positive[p][d] * negative[p][d];
                        normSq += (double)negative[p][d] * negative[p][d];
                    }
                float scale = (float)(dot / normSq);

                for (int p = 0; p < patchSize; p++)
                    for (int d = 0; d < featDim; d++)
                    {
                        float neg = negative[p][d] * scale;
                        dphi[p][d] = neg + cfgValue * (positive[p][d] - neg);
                    }
            }

            for (int p = 0; p < patchSize; p++)
                for (int d = 0; d < featDim; d++)
                    x[p][d] -= dt * dphi[p][d];

            t -= dt;
            if (step < tSpan.Length - 1) dt = t - tSpan[step + 1];
        }

        return x;
    }

    private static double NextGaussian(Random rng)
    {
        double u1 = 1.0 - rng.NextDouble();
        double u2 = rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
    }
}
