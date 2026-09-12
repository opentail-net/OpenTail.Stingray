namespace OpenTail.Stingray.Diffusion.MiniMaxMusic3;

/// <summary>
/// Persistent, reusable workspace for <see cref="MiniMaxMusic3RvqDepthDecoder"/> to make
/// incremental depth decoding (both single-branch and CFG-paired) 100% allocation-free.
/// Allocated once per generator instance and reused across all audio frames and depth steps.
/// </summary>
public sealed class MiniMaxMusic3RvqDepthWorkspace
{
    public const int MaxDepthSteps = 16;

    public readonly float[] XCond;
    public readonly float[] XUncond;

    public readonly float[] Normed1Cond;
    public readonly float[] Normed1Uncond;

    public readonly float[] QCond;
    public readonly float[] QUncond;
    public readonly float[] KCond;
    public readonly float[] KUncond;
    public readonly float[] VCond;
    public readonly float[] VUncond;

    public readonly float[] ContextCond;
    public readonly float[] ContextUncond;

    public readonly float[] AfterAttnCond;
    public readonly float[] AfterAttnUncond;

    public readonly float[] Normed2Cond;
    public readonly float[] Normed2Uncond;

    public readonly float[] GateCond;
    public readonly float[] GateUncond;
    public readonly float[] UpCond;
    public readonly float[] UpUncond;

    public readonly float[] MlpCond;
    public readonly float[] MlpUncond;

    public readonly float[] OutCond;
    public readonly float[] OutUncond;

    public readonly float[] Scores; // Scratch for attention scores [MaxDepthSteps]

    public MiniMaxMusic3RvqDepthWorkspace(
        int hidden = MiniMaxMusic3Config.RvqDepthDecoderHiddenSize,
        int intermediate = MiniMaxMusic3Config.RvqDepthDecoderIntermediateSize)
    {
        XCond = new float[hidden];
        XUncond = new float[hidden];

        Normed1Cond = new float[hidden];
        Normed1Uncond = new float[hidden];

        QCond = new float[hidden];
        QUncond = new float[hidden];
        KCond = new float[hidden];
        KUncond = new float[hidden];
        VCond = new float[hidden];
        VUncond = new float[hidden];

        ContextCond = new float[hidden];
        ContextUncond = new float[hidden];

        AfterAttnCond = new float[hidden];
        AfterAttnUncond = new float[hidden];

        Normed2Cond = new float[hidden];
        Normed2Uncond = new float[hidden];

        GateCond = new float[intermediate];
        GateUncond = new float[intermediate];
        UpCond = new float[intermediate];
        UpUncond = new float[intermediate];

        MlpCond = new float[hidden];
        MlpUncond = new float[hidden];

        OutCond = new float[hidden];
        OutUncond = new float[hidden];

        Scores = new float[MaxDepthSteps];
    }
}
