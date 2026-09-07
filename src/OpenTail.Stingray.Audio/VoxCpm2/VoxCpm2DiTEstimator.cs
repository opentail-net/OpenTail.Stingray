namespace OpenTail.Stingray.Audio.VoxCpm2;

/// <summary>
/// Native C# port of VoxCPM2's DiT estimator (the flow-matching velocity-field network) forward
/// pass, from `generator.cpp`'s `VoxCPM2DiTEstimatorRuntime::Impl::build`/`run` and
/// `sinusoidal_time_embedding` (not guessed). Real, genuinely different design from F5-TTS's
/// AdaLN-Zero DiT (confirmed via this session's F5-TTS reuse check): NO modulation/gating at all
/// -- conditioning (`mu`, time, `cond`) is injected purely as PREFIX rows concatenated before the
/// noisy `x` rows in one sequence, consumed by ordinary self-attention through the SAME
/// bidirectional MiniCPM stack (<see cref="VoxCpm2MiniCpmBidirectionalStack"/>) the local encoder
/// uses. Real per-call sequence: `[mu(2 rows), time(1 row), cond(patchSize rows), x(patchSize
/// rows)]` -&gt; only the LAST `patchSize` rows (the `x` portion) survive after the transformer,
/// projected back to `featDim`.
/// </summary>
public static class VoxCpm2DiTEstimator
{
    public const int FeatDim = 64;
    public const int PatchSize = 4;
    private const int HiddenDim = VoxCpm2MiniCpmBidirectionalStack.HiddenDim;

    /// <summary>Real `sinusoidal_time_embedding`: sin/cos frequency embedding, `emb_scale =
    /// log(10000)/(half-1)`, `freq = exp(-index*emb_scale)`, `arg = 1000*timestep*freq`.</summary>
    public static float[] SinusoidalTimeEmbedding(float timestep, int hiddenSize)
    {
        if (hiddenSize <= 0 || hiddenSize % 2 != 0) throw new ArgumentException("hiddenSize must be even.", nameof(hiddenSize));
        int half = hiddenSize / 2;
        var output = new float[hiddenSize];
        double embScale = Math.Log(10000.0) / (half - 1);
        for (int i = 0; i < half; i++)
        {
            double freq = Math.Exp(-i * embScale);
            double arg = 1000.0 * timestep * freq;
            output[i] = (float)Math.Sin(arg);
            output[half + i] = (float)Math.Cos(arg);
        }
        return output;
    }

    /// <summary>
    /// One DiT estimator forward pass for a single CFG batch item. `x`/`cond` are
    /// `[PatchSize][FeatDim]` (patch-major); `mu` is the real 2-row conditioning prefix (each row
    /// `HiddenDim` wide -- `[currentLmDitHidden, residualDitHidden]` for the positive/conditional
    /// branch, or a zeroed pair for the negative/unconditional CFG branch, per the caller).
    /// Returns `[PatchSize][FeatDim]` (patch-major), matching `x`'s own layout.
    /// </summary>
    public static float[][] Run(VoxCpm2DiTEstimatorWeights w, float[][] x, float[][] mu, float[][] cond, float timestep, float deltaTimestep)
    {
        if (mu.Length != 2) throw new ArgumentException("mu must have exactly 2 rows.", nameof(mu));

        var xProj = new float[PatchSize][];
        var condProj = new float[PatchSize][];
        for (int i = 0; i < PatchSize; i++)
        {
            xProj[i] = VoxCpm2MiniCpmBidirectionalStack.Linear(x[i], w.InProjWeight, w.InProjBias, FeatDim, HiddenDim);
            condProj[i] = VoxCpm2MiniCpmBidirectionalStack.Linear(cond[i], w.CondProjWeight, w.CondProjBias, FeatDim, HiddenDim);
        }

        var time = SinusoidalTimeEmbedding(timestep, HiddenDim);
        time = VoxCpm2MiniCpmBidirectionalStack.Linear(time, w.TimeMlp1Weight, w.TimeMlp1Bias, HiddenDim, HiddenDim);
        VoxCpm2MiniCpmBidirectionalStack.SiluInPlace(time);
        time = VoxCpm2MiniCpmBidirectionalStack.Linear(time, w.TimeMlp2Weight, w.TimeMlp2Bias, HiddenDim, HiddenDim);

        var dt = SinusoidalTimeEmbedding(deltaTimestep, HiddenDim);
        dt = VoxCpm2MiniCpmBidirectionalStack.Linear(dt, w.DeltaTimeMlp1Weight, w.DeltaTimeMlp1Bias, HiddenDim, HiddenDim);
        VoxCpm2MiniCpmBidirectionalStack.SiluInPlace(dt);
        dt = VoxCpm2MiniCpmBidirectionalStack.Linear(dt, w.DeltaTimeMlp2Weight, w.DeltaTimeMlp2Bias, HiddenDim, HiddenDim);

        var timeRow = new float[HiddenDim];
        for (int i = 0; i < HiddenDim; i++) timeRow[i] = time[i] + dt[i];

        int seqLen = 2 + 1 + PatchSize + PatchSize;
        var hidden = new float[seqLen][];
        hidden[0] = mu[0];
        hidden[1] = mu[1];
        hidden[2] = timeRow;
        for (int i = 0; i < PatchSize; i++) hidden[3 + i] = condProj[i];
        for (int i = 0; i < PatchSize; i++) hidden[3 + PatchSize + i] = xProj[i];

        hidden = VoxCpm2MiniCpmBidirectionalStack.Run(hidden, w.DecoderLayers, w.DecoderNorm, seqLen);

        var output = new float[PatchSize][];
        for (int i = 0; i < PatchSize; i++)
            output[i] = VoxCpm2MiniCpmBidirectionalStack.Linear(hidden[3 + PatchSize + i], w.OutProjWeight, w.OutProjBias, HiddenDim, FeatDim);
        return output;
    }
}
