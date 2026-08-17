namespace OpenTail.Stingray.Diffusion.StableDiffusion;

/// <summary>
/// Discrete Euler scheduler and CFG sampler for Stable Diffusion 1.5.
/// Uses the 1000-step scaled linear beta schedule (beta_start=0.00085, beta_end=0.012).
/// </summary>
public sealed class EulerDiscreteScheduler : IDiffusionScheduler, IDiffusionSampler
{
    private const int TotalTrainTimesteps = 1000;
    private const float BetaStart = 0.00085f;
    private const float BetaEnd = 0.012f;

    private readonly float[] _trainSigmas;
    private readonly float[] _trainLogSigmas;

    public float[] Sigmas { get; }
    public float[] Timesteps { get; }
    public int NumSteps => Timesteps.Length;

    public EulerDiscreteScheduler(int numInferenceSteps)
    {
        // 1. Build the 1000-step training betas & sigmas
        _trainSigmas = new float[TotalTrainTimesteps];
        _trainLogSigmas = new float[TotalTrainTimesteps];

        float sqrtBetaStart = MathF.Sqrt(BetaStart);
        float sqrtBetaEnd = MathF.Sqrt(BetaEnd);
        double cumulativeAlpha = 1.0;

        for (int i = 0; i < TotalTrainTimesteps; i++)
        {
            float tNorm = (float)i / (TotalTrainTimesteps - 1);
            float sqrtBeta = sqrtBetaStart + tNorm * (sqrtBetaEnd - sqrtBetaStart);
            float beta = sqrtBeta * sqrtBeta;
            float alpha = 1f - beta;
            cumulativeAlpha *= alpha;

            // sigma = sqrt((1 - alpha_cumprod) / alpha_cumprod)
            float sigma = MathF.Sqrt((float)((1.0 - cumulativeAlpha) / cumulativeAlpha));
            _trainSigmas[i] = sigma;
            _trainLogSigmas[i] = MathF.Log(sigma);
        }

        // 2. Compute inference sigmas (discrete spacing from 999 down to 0)
        Sigmas = new float[numInferenceSteps + 1];
        Timesteps = new float[numInferenceSteps];

        if (numInferenceSteps == 1)
        {
            Sigmas[0] = _trainSigmas[TotalTrainTimesteps - 1];
            Sigmas[1] = 0f;
            Timesteps[0] = TotalTrainTimesteps - 1;
        }
        else
        {
            float step = (float)(TotalTrainTimesteps - 1) / (numInferenceSteps - 1);
            for (int i = 0; i < numInferenceSteps; i++)
            {
                float t = (TotalTrainTimesteps - 1) - step * i;
                Timesteps[i] = t;
                Sigmas[i] = TimestepToSigma(t);
            }
            Sigmas[numInferenceSteps] = 0f;
        }
    }

    /// <summary>Convert continuous/fractional timestep t to sigma via log-linear interpolation.</summary>
    public float TimestepToSigma(float t)
    {
        int lowIdx = Math.Clamp((int)MathF.Floor(t), 0, TotalTrainTimesteps - 2);
        int highIdx = lowIdx + 1;
        float w = t - lowIdx;
        float logSigma = (1f - w) * _trainLogSigmas[lowIdx] + w * _trainLogSigmas[highIdx];
        return MathF.Exp(logSigma);
    }

    /// <summary>Convert sigma to continuous timestep t.</summary>
    public float SigmaToTimestep(float sigma)
    {
        float logSigma = MathF.Log(sigma);
        int lowIdx = 0;
        for (int i = 0; i < TotalTrainTimesteps; i++)
        {
            if (logSigma - _trainLogSigmas[i] >= 0)
                lowIdx++;
        }
        lowIdx = Math.Clamp(lowIdx - 1, 0, TotalTrainTimesteps - 2);
        int highIdx = lowIdx + 1;

        float low = _trainLogSigmas[lowIdx];
        float high = _trainLogSigmas[highIdx];
        float w = (low - logSigma) / (low - high);
        w = Math.Clamp(w, 0f, 1f);
        return (1f - w) * lowIdx + w * highIdx;
    }

    /// <summary>Generate Gaussian latent noise.</summary>
    public float[] SampleNoise(int elementCount, int seed)
    {
        var noise = new float[elementCount];
        var rng = seed >= 0 ? new Random(seed) : new Random();

        // Box-Muller transform
        for (int i = 0; i < elementCount - 1; i += 2)
        {
            double u1 = 1.0 - rng.NextDouble();
            double u2 = 1.0 - rng.NextDouble();
            double radius = Math.Sqrt(-2.0 * Math.Log(u1));
            double theta = 2.0 * Math.PI * u2;

            noise[i] = (float)(radius * Math.Cos(theta));
            noise[i + 1] = (float)(radius * Math.Sin(theta));
        }

        if ((elementCount & 1) == 1)
        {
            double u1 = 1.0 - rng.NextDouble();
            double u2 = 1.0 - rng.NextDouble();
            noise[^1] = (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
        }

        // Scale initial noise by max sigma
        float sigmaMax = Sigmas[0];
        for (int i = 0; i < noise.Length; i++)
            noise[i] *= sigmaMax;

        return noise;
    }

    /// <summary>
    /// Executes the Euler discrete denoising loop.
    /// predictNoise receives (scaledLatent, timestep) and returns predicted noise (or model output).
    /// </summary>
    public float[] Denoise(float[] initialLatent, Func<float[], float, float[]> predictNoise, Action<int, int>? progress = null)
    {
        var x = (float[])initialLatent.Clone();
        int steps = NumSteps;

        for (int i = 0; i < steps; i++)
        {
            float sigma = Sigmas[i];
            float sigmaNext = Sigmas[i + 1];
            float timestep = Timesteps[i];

            // 1. Scale model input: x_in = x / sqrt(sigma^2 + 1)
            float cIn = 1f / MathF.Sqrt(sigma * sigma + 1f);
            var xIn = new float[x.Length];
            for (int j = 0; j < x.Length; j++)
                xIn[j] = x[j] * cIn;

            // 2. Predict noise
            var modelOut = predictNoise(xIn, timestep);

            // 3. Euler step: x_{t+1} = x_t + modelOut * (sigma_{t+1} - sigma_t)
            float dt = sigmaNext - sigma;
            for (int j = 0; j < x.Length; j++)
                x[j] += modelOut[j] * dt;

            progress?.Invoke(i + 1, steps);
        }

        return x;
    }

    /// <summary>Combine conditional and unconditional noise predictions via Classifier-Free Guidance (CFG).</summary>
    public float[] CombineGuidance(float[] noisePredConditional, float[] noisePredUnconditional, float guidanceScale)
    {
        var result = new float[noisePredConditional.Length];
        for (int i = 0; i < result.Length; i++)
        {
            float uncond = noisePredUnconditional[i];
            float cond = noisePredConditional[i];
            result[i] = uncond + guidanceScale * (cond - uncond);
        }
        return result;
    }
}
