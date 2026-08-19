using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Diffusion.StableDiffusion;

public enum DiffusionSchedulerType
{
    Euler,
    EulerAncestral,
    Ddim,
    DpmPlusPlus2M,
    DpmPlusPlus2MKarras,
    Lcm
}

/// <summary>
/// Universal Discrete Scheduler supporting:
///   - Euler Discrete
///   - Euler Ancestral (stochastic)
///   - DDIM (deterministic inversion / step)
///   - DPM++ 2M / DPM++ 2M Karras (2nd order ODE solver)
/// Matching stable-diffusion.cpp:src/runtime/denoiser.hpp (sample_euler, sample_euler_ancestral, sample_dpmpp_2m)
/// </summary>
public sealed class EulerDiscreteScheduler
{
    public float[] Betas { get; }
    public float[] Alphas { get; }
    public float[] AlphasCumprod { get; }
    public float[] Sigmas { get; }
    public float[] Timesteps { get; }
    public int NumSteps { get; }

    private readonly DiffusionSchedulerType _schedulerType;

    public EulerDiscreteScheduler(int numInferenceSteps = 20, DiffusionSchedulerType schedulerType = DiffusionSchedulerType.Euler, float betaStart = 0.00085f, float betaEnd = 0.012f, int trainSteps = 1000)
    {
        NumSteps = numInferenceSteps;
        _schedulerType = schedulerType;

        // 1. Generate 1000-step linear beta schedule
        Betas = new float[trainSteps];
        Alphas = new float[trainSteps];
        AlphasCumprod = new float[trainSteps];

        float start = MathF.Sqrt(betaStart);
        float end = MathF.Sqrt(betaEnd);
        float step = (end - start) / (trainSteps - 1);

        float cumprod = 1.0f;
        for (int i = 0; i < trainSteps; i++)
        {
            float linear = start + i * step;
            float beta = linear * linear;
            Betas[i] = beta;
            Alphas[i] = 1.0f - beta;
            cumprod *= Alphas[i];
            AlphasCumprod[i] = cumprod;
        }

        // Full trained sigmas
        var allSigmas = new float[trainSteps];
        for (int i = 0; i < trainSteps; i++)
        {
            float alphaBar = AlphasCumprod[i];
            allSigmas[i] = MathF.Sqrt((1f - alphaBar) / alphaBar);
        }

        // 2. Select discrete timesteps & sigmas
        Timesteps = new float[numInferenceSteps];
        Sigmas = new float[numInferenceSteps + 1];

        float stepRatio = (float)(trainSteps - 1) / (numInferenceSteps - 1);
        for (int i = 0; i < numInferenceSteps; i++)
        {
            float t = (numInferenceSteps - 1 - i) * stepRatio;
            Timesteps[i] = t;

            int low = (int)MathF.Floor(t);
            int high = (int)MathF.Ceiling(t);
            float weight = t - low;

            float sigmaLow = allSigmas[Math.Clamp(low, 0, trainSteps - 1)];
            float sigmaHigh = allSigmas[Math.Clamp(high, 0, trainSteps - 1)];
            Sigmas[i] = sigmaLow + weight * (sigmaHigh - sigmaLow);
        }
        Sigmas[numInferenceSteps] = 0f;

        // If Karras noise distribution requested, remap sigmas
        if (schedulerType == DiffusionSchedulerType.DpmPlusPlus2MKarras)
        {
            Sigmas = BuildKarrasSigmas(numInferenceSteps, Sigmas[0], Sigmas[^2]);
        }
    }

    private static float[] BuildKarrasSigmas(int numSteps, float sigmaMax, float sigmaMin, float rho = 7.0f)
    {
        var sigmas = new float[numSteps + 1];
        float invRho = 1.0f / rho;
        float minInv = MathF.Pow(sigmaMin, invRho);
        float maxInv = MathF.Pow(sigmaMax, invRho);

        for (int i = 0; i < numSteps; i++)
        {
            float ramp = (float)i / (numSteps - 1);
            sigmas[i] = MathF.Pow(maxInv + ramp * (minInv - maxInv), rho);
        }
        sigmas[numSteps] = 0f;
        return sigmas;
    }

    /// <summary>
    /// Computes CFG guided noise prediction: e_cfg = e_uncond + guidance * (e_cond - e_uncond).
    /// </summary>
    public float[] CombineGuidance(float[] noiseCond, float[] noiseUncond, float guidanceScale)
    {
        var result = new float[noiseCond.Length];
        for (int i = 0; i < result.Length; i++)
        {
            float uncond = noiseUncond[i];
            float cond = noiseCond[i];
            result[i] = uncond + guidanceScale * (cond - uncond);
        }
        return result;
    }

    /// <summary>
    /// Sample random Gaussian noise vector scaled by initial sigmaMax.
    /// </summary>
    public float[] SampleNoise(int length, int seed = -1)
    {
        var noise = new float[length];
        var rng = seed >= 0 ? new Random(seed) : new Random();

        for (int i = 0; i < length - 1; i += 2)
        {
            double u1 = 1.0 - rng.NextDouble();
            double u2 = 1.0 - rng.NextDouble();
            double radius = Math.Sqrt(-2.0 * Math.Log(u1));
            double theta = 2.0 * Math.PI * u2;
            noise[i] = (float)(radius * Math.Cos(theta));
            noise[i + 1] = (float)(radius * Math.Sin(theta));
        }

        if ((length & 1) == 1)
        {
            double u1 = 1.0 - rng.NextDouble();
            double u2 = 1.0 - rng.NextDouble();
            noise[^1] = (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
        }

        float sigmaMax = Sigmas[0];
        for (int i = 0; i < noise.Length; i++)
            noise[i] *= sigmaMax;

        return noise;
    }

    /// <summary>
    /// Generates initial scaled Gaussian noise vector for the starting latent space.
    /// </summary>
    public float[] CreateInitialLatents(int batch, int channels, int height, int width, int seed = -1)
        => SampleNoise(batch * channels * height * width, seed);

    /// <summary>
    /// Adds noise to initial latent at specified start step for img2img workflows.
    /// </summary>
    public float[] CreateNoisyLatent(float[] latent, float[] noise, int startStep)
    {
        float sigma = Sigmas[Math.Clamp(startStep, 0, Sigmas.Length - 1)];
        var result = new float[latent.Length];
        for (int i = 0; i < latent.Length; i++)
            result[i] = latent[i] + noise[i] * sigma;
        return result;
    }

    /// <summary>
    /// Executes the denoising loop.
    /// predictNoise receives (scaledLatent, timestep) and returns predicted noise.
    /// </summary>
    public float[] Denoise(float[] initialLatent, Func<float[], float, float[]> predictNoise, Action<int, int>? progress = null, int startStep = 0)
    {
        var x = (float[])initialLatent.Clone();
        int steps = NumSteps;
        var rng = new Random(42);
        float[]? oldDenoised = null;

        for (int i = startStep; i < steps; i++)
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

            // 3. Step update based on scheduler algorithm
            if (_schedulerType is DiffusionSchedulerType.DpmPlusPlus2M or DiffusionSchedulerType.DpmPlusPlus2MKarras)
            {
                // DPM-Solver++ (2M) formulation (reference: stable-diffusion.cpp:sample_dpmpp_2m)
                // denoised estimate: d = x - sigma * modelOut
                var denoised = new float[x.Length];
                for (int j = 0; j < x.Length; j++)
                    denoised[j] = x[j] - sigma * modelOut[j];

                if (sigmaNext <= 0f)
                {
                    // Final step directly maps to denoised estimate
                    x = denoised;
                }
                else
                {
                    float t = -MathF.Log(sigma);
                    float tNext = -MathF.Log(sigmaNext);
                    float h = tNext - t;
                    float a = sigmaNext / sigma;
                    float b = MathF.Exp(-h) - 1.0f;

                    if (i == startStep || oldDenoised is null)
                    {
                        // 1st order Euler step on ODE
                        for (int j = 0; j < x.Length; j++)
                            x[j] = a * x[j] - b * denoised[j];
                    }
                    else
                    {
                        // 2nd order multi-step update
                        float tPrev = -MathF.Log(Sigmas[i - 1]);
                        float hLast = t - tPrev;
                        float r = hLast / h;
                        float w1 = 1.0f + 1.0f / (2.0f * r);
                        float w2 = 1.0f / (2.0f * r);

                        for (int j = 0; j < x.Length; j++)
                        {
                            float denoisedD = w1 * denoised[j] - w2 * oldDenoised[j];
                            x[j] = a * x[j] - b * denoisedD;
                        }
                    }
                }
                oldDenoised = denoised;
            }
            else if (_schedulerType == DiffusionSchedulerType.EulerAncestral)
            {
                // Euler Ancestral (stochastic)
                float sigmaUp = MathF.Sqrt(sigmaNext * sigmaNext * (sigma * sigma - sigmaNext * sigmaNext) / (sigma * sigma));
                float sigmaDown = MathF.Sqrt(sigmaNext * sigmaNext - sigmaUp * sigmaUp);
                float dt = sigmaDown - sigma;

                for (int j = 0; j < x.Length; j++)
                {
                    x[j] += modelOut[j] * dt;
                    if (sigmaUp > 0f)
                    {
                        double u1 = 1.0 - rng.NextDouble();
                        double u2 = 1.0 - rng.NextDouble();
                        float z = (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
                        x[j] += z * sigmaUp;
                    }
                }
            }
            else if (_schedulerType == DiffusionSchedulerType.Ddim)
            {
                // DDIM Step update
                float dt = sigmaNext - sigma;
                for (int j = 0; j < x.Length; j++)
                    x[j] += modelOut[j] * dt;
            }
            else if (_schedulerType == DiffusionSchedulerType.Lcm)
            {
                // LCM Step: x0 = c_skip * x + c_out * modelOut
                float sigmaData = 0.5f;
                float scaledT = timestep * 0.01f;
                float denom = scaledT * scaledT + sigmaData * sigmaData;
                float cSkip = (sigmaData * sigmaData) / denom;
                float cOut = scaledT / MathF.Sqrt(denom);

                for (int j = 0; j < x.Length; j++)
                    x[j] = cSkip * x[j] + cOut * modelOut[j];
            }
            else
            {
                // Standard Discrete Euler: x_{t-1} = x_t + d_t * dt
                float dt = sigmaNext - sigma;
                for (int j = 0; j < x.Length; j++)
                    x[j] += modelOut[j] * dt;
            }

            progress?.Invoke(i + 1, steps);
        }

        return x;
    }
}
