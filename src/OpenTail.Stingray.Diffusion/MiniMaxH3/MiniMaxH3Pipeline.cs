namespace OpenTail.Stingray.Diffusion.MiniMaxH3;

/// <summary>
/// MiniMax-H3 Omni-Modal Diffusion Pipeline.
/// Drives rectified flow matching over packed multimodal latents (video + 32 kHz audio),
/// with dual flow-shift schedules (video shift=12, audio shift=3) and CFG-distilled guidance (cfg=1.0).
/// </summary>
public sealed class MiniMaxH3Pipeline
{
    private readonly MiniMaxH3DiT _dit;
    private readonly float[] _adalnTable;

    public MiniMaxH3DiT DiT => _dit;

    public MiniMaxH3Pipeline(MiniMaxH3DiT dit, float[] adalnTable)
    {
        _dit = dit ?? throw new ArgumentNullException(nameof(dit));
        if (adalnTable.Length < MiniMaxH3Config.AdaLnTableRows * MiniMaxH3Config.AdaLnTableCols)
            throw new ArgumentException("AdaLN table size is invalid.", nameof(adalnTable));
        _adalnTable = adalnTable;
    }

    /// <summary>
    /// Run the dual-schedule Euler denoising loop.
    /// Returns the denoised video latent [VideoFrames, 24, VideoHeight, VideoWidth]
    /// and audio latent [AudioFrames, 32].
    /// </summary>
    public (float[] DenoisedVideoLatent, float[] DenoisedAudioLatent) Generate(
        MiniMaxH3Layout layout,
        ReadOnlySpan<float> contextTokens,
        int numSteps = 20,
        int? seed = null,
        Action<int, int>? progress = null,
        int numHeads = MiniMaxH3Config.NumHeads,
        int headDim = MiniMaxH3Config.HeadDim)
    {
        var scheduler = new MiniMaxH3Scheduler(numSteps);
        var videoSigmas = scheduler.VideoSigmas;
        var audioSigmas = scheduler.AudioSigmas;

        var rng = seed is int s ? new Random(s) : new Random();

        // 1. Allocate and initialize initial Gaussian noise for video latent [T, 24, H, W]
        int videoLatentLen = layout.VideoFrames * MiniMaxH3Config.VideoLatentChannels * layout.VideoHeight * layout.VideoWidth;
        var videoLatent = new float[videoLatentLen];
        for (int i = 0; i < videoLatentLen; i++)
        {
            videoLatent[i] = SampleStandardNormal(rng);
        }

        // 2. Allocate and initialize initial Gaussian noise for audio latent [T_a, 32]
        int audioLatentLen = layout.AudioFrames * MiniMaxH3Config.AudioLatentChannels;
        var audioLatent = new float[audioLatentLen];
        for (int i = 0; i < audioLatentLen; i++)
        {
            audioLatent[i] = SampleStandardNormal(rng);
        }

        // Buffers for patch tokens
        var videoPatchTokens = new float[layout.NumVideoTokens * MiniMaxH3Config.VideoPatchDim];
        var audioTokens = new float[layout.NumAudioTokens * MiniMaxH3Config.AudioPatchDim];
        var curveVec8 = new float[MiniMaxH3Config.AdaLnTableRows];
        var videoVelocityTCHW = new float[videoLatentLen];

        // 3. Rectified Flow Euler Denoising Loop
        for (int step = 0; step < numSteps; step++)
        {
            float vSigma = videoSigmas[step];
            float vSigmaNext = videoSigmas[step + 1];
            float aSigma = audioSigmas[step];
            float aSigmaNext = audioSigmas[step + 1];

            // Interpolate AdaLN curve vector at video timestep (0..1000)
            float timestep = vSigma * 1000.0f;
            MiniMaxH3AdaLN.InterpolateTimestepCurve(_adalnTable, timestep, curveVec8);

            // Pack current noisy video and audio latents into sequence tokens
            layout.PackVideoLatent(videoLatent, videoPatchTokens);
            layout.PackAudioLatent(audioLatent, audioTokens);

            // Omni-modal DiT forward pass
            var (vVelPatches, aVelTokens) = _dit.Forward(
                layout,
                videoPatchTokens,
                audioTokens,
                contextTokens,
                curveVec8,
                numHeads,
                headDim);

            // Unpack predicted video velocity from patches to [T, 24, H, W]
            layout.UnpackVideoLatent(vVelPatches, videoVelocityTCHW);

            // Single Euler step with independent video and audio sigmas
            MiniMaxH3Scheduler.StepEuler(videoLatent, videoVelocityTCHW, vSigma, vSigmaNext);
            MiniMaxH3Scheduler.StepEuler(audioLatent, aVelTokens, aSigma, aSigmaNext);

            progress?.Invoke(step + 1, numSteps);
        }

        return (videoLatent, audioLatent);
    }

    private static float SampleStandardNormal(Random rng)
    {
        double u1 = 1.0 - rng.NextDouble();
        double u2 = 1.0 - rng.NextDouble();
        return (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
    }
}
