namespace OpenTail.Stingray.Diffusion.MiniMaxH3;

/// <summary>
/// Dual flow-matching Euler scheduler for MiniMax-H3.
/// H3 uses distinct flow-shift parameters and independent sigma schedules for video and audio:
/// Video flow shift = 12.0, Audio flow shift = 3.0.
/// CFG guidance is 1.0 (CFG-distilled; no unconditional negative-prompt pass in base path).
/// </summary>
public sealed class MiniMaxH3Scheduler
{
    private readonly float[] _videoSigmas;
    private readonly float[] _audioSigmas;

    public int NumSteps => _videoSigmas.Length - 1;
    public ReadOnlySpan<float> VideoSigmas => _videoSigmas;
    public ReadOnlySpan<float> AudioSigmas => _audioSigmas;

    public MiniMaxH3Scheduler(int numSteps, float videoShift = MiniMaxH3Config.VideoFlowShift, float audioShift = MiniMaxH3Config.AudioFlowShift)
    {
        if (numSteps < 1) throw new ArgumentOutOfRangeException(nameof(numSteps), "numSteps must be >= 1.");
        _videoSigmas = ComputeSigmas(numSteps, videoShift);
        _audioSigmas = ComputeSigmas(numSteps, audioShift);
    }

    /// <summary>
    /// Compute shifted sigma schedule for a given number of steps and flow shift.
    /// Returns array of length numSteps + 1, ending with 0.0.
    /// </summary>
    public static float[] ComputeSigmas(int numSteps, float flowShift)
    {
        var sigmas = new float[numSteps + 1];
        if (numSteps == 1)
        {
            sigmas[0] = Shift(flowShift, 1.0f);
        }
        else
        {
            for (int i = 0; i < numSteps; i++)
            {
                float linearT = 1.0f - (float)i / numSteps;
                sigmas[i] = Shift(flowShift, linearT);
            }
        }
        sigmas[numSteps] = 0.0f;
        return sigmas;
    }

    /// <summary>
    /// Shift formula: s' = (shift * s) / (1 + (shift - 1) * s).
    /// </summary>
    public static float Shift(float shift, float t)
    {
        if (shift == 1.0f) return t;
        return (shift * t) / (1.0f + (shift - 1.0f) * t);
    }

    /// <summary>
    /// Single Euler step for rectified flow matching:
    /// x_{t+1} = x_t + (sigma_{t+1} - sigma_t) * velocity.
    /// </summary>
    public static void StepEuler(Span<float> latent, ReadOnlySpan<float> velocity, float currentSigma, float nextSigma)
    {
        if (latent.Length != velocity.Length)
            throw new ArgumentException("Latent and velocity spans must have identical length.");

        float dt = nextSigma - currentSigma; // dt < 0 as sigma decreases towards 0
        for (int i = 0; i < latent.Length; i++)
        {
            latent[i] += dt * velocity[i];
        }
    }
}
