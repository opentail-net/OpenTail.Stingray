
namespace OpenTail.Stingray.Diffusion;

/// <summary>
/// Latent Consistency Model (LCM) Scheduler for ultra-fast 1 to 4 step diffusion generation.
/// Solves the Probability Flow ODE with boundary condition scaling (c_skip and c_out).
/// Reference: diffusers:src/diffusers/schedulers/scheduling_lcm.py
/// </summary>
public sealed class LcmScheduler
{
    private readonly float _sigmaData;
    private readonly float _timestepScaling;
    private readonly int _numTrainTimesteps;
    private readonly float[] _alphasCumprod;

    public int NumTrainTimesteps => _numTrainTimesteps;

    public LcmScheduler(
        int numTrainTimesteps = 1000,
        float betaStart = 0.00085f,
        float betaEnd = 0.012f,
        float sigmaData = 0.5f,
        float timestepScaling = 10.0f)
    {
        _numTrainTimesteps = numTrainTimesteps;
        _sigmaData = sigmaData;
        _timestepScaling = timestepScaling;

        // Linear beta schedule (standard SD 1.5 / SDXL)
        var betas = new float[numTrainTimesteps];
        for (int i = 0; i < numTrainTimesteps; i++)
        {
            float t = (float)i / (numTrainTimesteps - 1);
            betas[i] = betaStart + t * (betaEnd - betaStart);
        }

        _alphasCumprod = new float[numTrainTimesteps];
        float prod = 1.0f;
        for (int i = 0; i < numTrainTimesteps; i++)
        {
            prod *= (1.0f - betas[i]);
            _alphasCumprod[i] = prod;
        }
    }

    /// <summary>
    /// Computes the evenly-spaced discrete timestep schedule for N inference steps (e.g. 1, 2, 4).
    /// </summary>
    public int[] BuildTimesteps(int numInferenceSteps)
    {
        if (numInferenceSteps <= 0 || numInferenceSteps > _numTrainTimesteps)
            throw new ArgumentOutOfRangeException(nameof(numInferenceSteps));

        var timesteps = new int[numInferenceSteps];
        int stepSize = _numTrainTimesteps / numInferenceSteps;

        for (int i = 0; i < numInferenceSteps; i++)
        {
            // Descending order: e.g. for 4 steps: 999, 749, 499, 249
            timesteps[i] = _numTrainTimesteps - 1 - (i * stepSize);
        }

        return timesteps;
    }

    /// <summary>
    /// Computes boundary condition scalings (c_skip, c_out).
    /// </summary>
    public (float cSkip, float cOut) GetBoundaryScalings(int timestep)
    {
        float scaledTimestep = (timestep * _timestepScaling) / _numTrainTimesteps;
        float scaledSq = scaledTimestep * scaledTimestep;
        float sigmaSq = _sigmaData * _sigmaData;

        float denom = scaledSq + sigmaSq;
        float cSkip = sigmaSq / denom;
        float cOut = scaledTimestep / MathF.Sqrt(denom);

        return (cSkip, cOut);
    }

    /// <summary>
    /// Performs a single LCM denoising step:
    /// 1. Computes pred_x0 = c_skip * sample + c_out * model_output
    /// 2. If not final step, adds scaled noise towards prev_timestep:
    ///    prev_sample = sqrt(alpha_prev) * pred_x0 + sqrt(1 - alpha_prev) * noise
    /// </summary>
    public void Step(
        Span<float> sample,
        ReadOnlySpan<float> modelOutput,
        int timestep,
        int prevTimestep,
        ReadOnlySpan<float> noise = default)
    {
        if (sample.Length != modelOutput.Length)
            throw new ArgumentException("Sample and modelOutput spans must have matching lengths.");

        var (cSkip, cOut) = GetBoundaryScalings(timestep);

        // 1. Predict denoised x_0
        for (int i = 0; i < sample.Length; i++)
        {
            float x0 = cSkip * sample[i] + cOut * modelOutput[i];

            if (prevTimestep <= 0)
            {
                // Final step: output directly predicted x_0
                sample[i] = x0;
            }
            else
            {
                // Multistep transition
                float alphaPrev = _alphasCumprod[Math.Clamp(prevTimestep, 0, _numTrainTimesteps - 1)];
                float sqrtAlphaPrev = MathF.Sqrt(alphaPrev);
                float sqrtOneMinusAlphaPrev = MathF.Sqrt(Math.Max(0f, 1.0f - alphaPrev));

                float n = noise.Length == sample.Length ? noise[i] : 0f;
                sample[i] = sqrtAlphaPrev * x0 + sqrtOneMinusAlphaPrev * n;
            }
        }
    }
}
