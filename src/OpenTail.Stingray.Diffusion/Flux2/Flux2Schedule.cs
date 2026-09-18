namespace OpenTail.Stingray.Diffusion.Flux2;

/// <summary>
/// Real FLUX.2 flow-matching timestep schedule (docs/087), confirmed against
/// `examples/flux2/src/flux2/sampling.py`'s real `get_schedule`/`compute_empirical_mu`/
/// `generalized_time_snr_shift`. NOT a plain linear ramp -- FLUX.2 (like FLUX.1/SD3.5) is trained
/// with a resolution- and step-count-dependent SHIFTED schedule: a `mu` value is computed from
/// the real image sequence length (number of DiT patch tokens) and step count via a piecewise-
/// linear empirical fit, then every linearly-spaced timestep is reshaped through a generalized
/// SNR-shift formula that concentrates more steps near the high-noise end for larger images.
/// </summary>
public static class Flux2Schedule
{
    /// <summary>Real linear-fit constants from `compute_empirical_mu` (verbatim).</summary>
    private const float A1 = 8.73809524e-05f, B1 = 1.89833333f;
    private const float A2 = 0.00016927f, B2 = 0.45666666f;

    /// <summary>
    /// Real `compute_empirical_mu(image_seq_len, num_steps)`: a piecewise-linear empirical fit,
    /// NOT resolution-independent -- larger `imageSeqLen` (more DiT patch tokens) shifts the
    /// schedule differently than smaller.
    /// </summary>
    public static float ComputeEmpiricalMu(int imageSeqLen, int numSteps)
    {
        if (imageSeqLen > 4300)
            return A2 * imageSeqLen + B2;

        float m200 = A2 * imageSeqLen + B2;
        float m10 = A1 * imageSeqLen + B1;
        float a = (m200 - m10) / 190.0f;
        float b = m200 - 200.0f * a;
        return a * numSteps + b;
    }

    /// <summary>
    /// Real `generalized_time_snr_shift(t, mu, sigma)`: `exp(mu) / (exp(mu) + (1/t - 1)^sigma)`.
    /// Boundary-preserving (t=0 -&gt; 0, t=1 -&gt; 1), reshapes the interior. `t` must be in (0, 1]
    /// for the formula as written (t=0 handled as a limit, see <see cref="GetSchedule"/>).
    /// </summary>
    public static float GeneralizedTimeSnrShift(float t, float mu, float sigma = 1.0f)
    {
        if (t <= 0f) return 0f; // limit as t->0+: exp(mu) / (exp(mu) + inf) = 0
        float expMu = MathF.Exp(mu);
        return expMu / (expMu + MathF.Pow(1f / t - 1f, sigma));
    }

    /// <summary>
    /// Real `get_schedule(num_steps, image_seq_len)`: `numSteps + 1` values, linearly spaced
    /// from 1 down to 0, then reshaped via <see cref="GeneralizedTimeSnrShift"/> with
    /// <c>sigma=1.0</c> and <c>mu</c> from <see cref="ComputeEmpiricalMu"/>.
    /// </summary>
    public static float[] GetSchedule(int numSteps, int imageSeqLen)
    {
        float mu = ComputeEmpiricalMu(imageSeqLen, numSteps);
        var timesteps = new float[numSteps + 1];
        for (int i = 0; i <= numSteps; i++)
        {
            float linear = 1.0f - (float)i / numSteps; // torch.linspace(1, 0, numSteps+1)
            timesteps[i] = GeneralizedTimeSnrShift(linear, mu, 1.0f);
        }
        return timesteps;
    }
}
