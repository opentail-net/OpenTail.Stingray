
namespace OpenTail.Stingray.Diffusion.Flux3;

/// <summary>
/// Self-Flow Rectified Flow ODE scheduler for FLUX 3.
/// Solves the continuous-time probability flow ODE from Gaussian noise (t=1.0) to coherent multimodal latents (t=0.0).
/// </summary>
public sealed class Flux3SelfFlowScheduler
{
    public float Shift { get; }

    public Flux3SelfFlowScheduler(float shift = 3.0f)
    {
        Shift = shift;
    }

    /// <summary>
    /// Computes the time-shifted schedule for flow-matching models.
    /// Higher shift values allocate more computation steps to high-frequency semantic details.
    /// </summary>
    public float[] BuildTimesteps(int steps)
    {
        if (steps <= 0) throw new ArgumentOutOfRangeException(nameof(steps), "Steps must be positive.");

        var timesteps = new float[steps + 1];
        for (int i = 0; i <= steps; i++)
        {
            float t = 1.0f - (float)i / steps;
            // Time-shifting formula: t_shifted = (shift * t) / (1 + (shift - 1) * t)
            timesteps[i] = (Shift * t) / (1.0f + (Shift - 1.0f) * t);
        }
        return timesteps;
    }

    /// <summary>
    /// Computes a single 1st-order Euler flow step: x_{t+dt} = x_t + dt * v_t.
    /// </summary>
    public void StepEuler(Span<float> latent, ReadOnlySpan<float> velocity, float dt)
    {
        if (latent.Length != velocity.Length)
            throw new ArgumentException("Latent and velocity spans must have identical length.");

        TensorPrimitives.MultiplyAdd(velocity, dt, latent, latent);
    }

    /// <summary>
    /// Computes a 2nd-order Heun predictor-corrector flow step.
    /// </summary>
    public void StepHeun(
        Span<float> latent,
        ReadOnlySpan<float> vPredicted,
        ReadOnlySpan<float> vCorrected,
        float dt)
    {
        if (latent.Length != vPredicted.Length || latent.Length != vCorrected.Length)
            throw new ArgumentException("Latent and velocity spans must have identical length.");

        float halfDt = dt * 0.5f;
        for (int i = 0; i < latent.Length; i++)
        {
            latent[i] += halfDt * (vPredicted[i] + vCorrected[i]);
        }
    }
}
