namespace OpenTail.Stingray.Audio.VibeVoice;

/// <summary>
/// Real port of VibeVoice TTS's `scheduler.cpp` -- a 2nd-order multistep DPM-Solver++ over a
/// cosine-beta DDPM noise schedule, `v_prediction` only (real reference throws for any other
/// config; this port hardcodes the same restriction rather than branching on config it will
/// never see). Not guessed -- every formula (cosine `alpha_bar`, `sigma=sqrt((1-a)/a)`,
/// `lambda=log(alpha)-log(sigma)`, first/second-order update coefficients, the real
/// `lower_order_nums`/`lower_order_final`/`lower_order_second` bookkeeping that decides which
/// order to use each step) is copied directly from `VibeVoiceDPMSolverScheduler`.
/// </summary>
public sealed class VibeVoiceDpmSolverScheduler
{
    private const int SolverOrder = 2;
    private const float MaxBeta = 0.999f;

    private readonly float[] _alphasCumprod;
    private readonly float[] _sigmasFull;

    private int[] _timesteps = [];
    private float[] _sigmas = [];
    private float[][] _modelOutputs = [[], []];
    private int _lowerOrderNums;
    private int _stepIndex = -1;

    public IReadOnlyList<int> Timesteps => _timesteps;

    public VibeVoiceDpmSolverScheduler(int ddpmNumSteps)
    {
        var betas = CosineBetas(ddpmNumSteps);
        _alphasCumprod = new float[ddpmNumSteps];
        float cumulative = 1f;
        for (int i = 0; i < ddpmNumSteps; i++)
        {
            cumulative *= 1f - betas[i];
            _alphasCumprod[i] = cumulative;
        }
        _sigmasFull = new float[ddpmNumSteps];
        for (int i = 0; i < ddpmNumSteps; i++)
            _sigmasFull[i] = MathF.Sqrt((1f - _alphasCumprod[i]) / _alphasCumprod[i]);
    }

    private static float AlphaBarCosine(float t)
    {
        float value = MathF.Cos((t + 0.008f) / 1.008f * MathF.PI * 0.5f);
        return value * value;
    }

    private static float[] CosineBetas(int timesteps)
    {
        var betas = new float[timesteps];
        for (int i = 0; i < timesteps; i++)
        {
            float t1 = (float)i / timesteps;
            float t2 = (float)(i + 1) / timesteps;
            betas[i] = Math.Min(1f - AlphaBarCosine(t2) / AlphaBarCosine(t1), MaxBeta);
        }
        return betas;
    }

    public void SetTimesteps(int inferenceSteps)
    {
        int lastTimestep = _alphasCumprod.Length;
        var forward = new int[inferenceSteps + 1];
        double start = 0.0, stop = lastTimestep - 1, denom = inferenceSteps;
        for (int i = 0; i <= inferenceSteps; i++)
            forward[i] = (int)Math.Round(start + (stop - start) * i / denom, MidpointRounding.ToEven);

        _timesteps = new int[inferenceSteps];
        for (int i = inferenceSteps; i >= 1; i--) _timesteps[inferenceSteps - i] = forward[i];

        _sigmas = new float[inferenceSteps + 1];
        for (int i = 0; i < inferenceSteps; i++) _sigmas[i] = _sigmasFull[_timesteps[i]];
        _sigmas[inferenceSteps] = 0f;

        ResetStepState();
    }

    public void ResetStepState()
    {
        _modelOutputs = [[], []];
        _lowerOrderNums = 0;
        _stepIndex = -1;
    }

    private static void SigmaToAlphaSigma(float sigma, out float alphaT, out float sigmaT)
    {
        alphaT = 1f / MathF.Sqrt(sigma * sigma + 1f);
        sigmaT = sigma * alphaT;
    }

    private int IndexForTimestep(int timestep)
    {
        for (int i = 0; i < _timesteps.Length; i++) if (_timesteps[i] == timestep) return i;
        return _timesteps.Length - 1;
    }

    private float[] ConvertModelOutput(float[] modelOutput, float[] sample)
    {
        SigmaToAlphaSigma(_sigmas[_stepIndex], out float alphaT, out float sigmaT);
        var output = new float[modelOutput.Length];
        for (int i = 0; i < output.Length; i++) output[i] = alphaT * sample[i] - sigmaT * modelOutput[i];
        return output;
    }

    private float[] FirstOrderUpdate(float[] modelOutput, float[] sample)
    {
        int idx = _stepIndex;
        SigmaToAlphaSigma(_sigmas[idx + 1], out float alphaT, out float sigmaT);
        SigmaToAlphaSigma(_sigmas[idx], out float alphaS, out float sigmaS);
        float lambdaT = MathF.Log(alphaT) - MathF.Log(sigmaT);
        float lambdaS = MathF.Log(alphaS) - MathF.Log(sigmaS);
        float h = lambdaT - lambdaS;
        float expNegH = MathF.Exp(-h);
        float sampleCoeff = sigmaT / sigmaS;
        float modelCoeff = -alphaT * (expNegH - 1f);

        var output = new float[sample.Length];
        for (int i = 0; i < output.Length; i++) output[i] = sampleCoeff * sample[i] + modelCoeff * modelOutput[i];
        return output;
    }

    private float[] SecondOrderUpdate(float[] sample)
    {
        int idx = _stepIndex;
        SigmaToAlphaSigma(_sigmas[idx + 1], out float alphaT, out float sigmaT);
        SigmaToAlphaSigma(_sigmas[idx], out float alphaS0, out float sigmaS0);
        SigmaToAlphaSigma(_sigmas[idx - 1], out float alphaS1, out float sigmaS1);
        float lambdaT = MathF.Log(alphaT) - MathF.Log(sigmaT);
        float lambdaS0 = MathF.Log(alphaS0) - MathF.Log(sigmaS0);
        float lambdaS1 = MathF.Log(alphaS1) - MathF.Log(sigmaS1);
        float h = lambdaT - lambdaS0;
        float h0 = lambdaS0 - lambdaS1;
        float r0 = h0 / h;
        float expNegH = MathF.Exp(-h);
        float sampleCoeff = sigmaT / sigmaS0;
        float d0Coeff = -alphaT * (expNegH - 1f);

        var m0 = _modelOutputs[0];
        var m1 = _modelOutputs[1];
        var output = new float[sample.Length];
        for (int i = 0; i < output.Length; i++)
        {
            float d0 = m1[i];
            float d1 = (1f / r0) * (m1[i] - m0[i]);
            output[i] = sampleCoeff * sample[i] + d0Coeff * d0 + 0.5f * d0Coeff * d1;
        }
        return output;
    }

    /// <summary>Advances one DPM-Solver++ step: `eps` (converted v-prediction model output),
    /// current `timestep`, current `sample`. Returns the previous (less-noisy) sample.</summary>
    public float[] Step(float[] modelOutput, int timestep, float[] sample)
    {
        if (_timesteps.Length == 0) throw new InvalidOperationException("SetTimesteps must be called before Step.");
        if (_stepIndex < 0) _stepIndex = IndexForTimestep(timestep);

        bool lowerOrderFinal = _stepIndex == _timesteps.Length - 1;
        bool lowerOrderSecond = _stepIndex == _timesteps.Length - 2 && _timesteps.Length < 15;

        var converted = ConvertModelOutput(modelOutput, sample);
        _modelOutputs[0] = _modelOutputs[1];
        _modelOutputs[1] = converted;

        float[] prevSample = _lowerOrderNums < 1 || lowerOrderFinal
            ? FirstOrderUpdate(_modelOutputs[1], sample)
            : (SolverOrder == 2 || _lowerOrderNums < 2 || lowerOrderSecond)
                ? SecondOrderUpdate(sample)
                : throw new InvalidOperationException("Unexpected third-order path.");

        if (_lowerOrderNums < SolverOrder) _lowerOrderNums++;
        _stepIndex++;
        return prevSample;
    }
}
