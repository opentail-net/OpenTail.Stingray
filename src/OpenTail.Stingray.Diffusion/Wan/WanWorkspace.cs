namespace OpenTail.Stingray.Diffusion.Wan;

/// <summary>
/// Reusable workspace buffers for Wan 2.1 DiT forward passes, eliminating per-block managed allocations.
/// </summary>
public sealed class WanWorkspace
{
    public float[] Norm1 { get; }
    public float[] Normed1 { get; }
    public float[] Q { get; }
    public float[] K { get; }
    public float[] V { get; }
    public float[] QkvFused { get; }
    public float[] AttnOut { get; }
    public float[] Norm3 { get; }
    public float[] NormedCross { get; }
    public float[] CrossQ { get; }
    public float[] CrossAttnOut { get; }
    public float[] Norm2 { get; }
    public float[] Normed2 { get; }
    public float[] Ffn1 { get; }
    public float[] FfnOut { get; }
    public float[] HeadShift { get; }
    public float[] HeadScale { get; }
    public float[] Mod { get; }

    /// <summary>
    /// Cached (K, V) projections of text context for each transformer block:
    /// Item1: K [numTxt * dim], Item2: V [numTxt * dim].
    /// Invariant across timesteps and CFG branches.
    /// </summary>
    public (float[] K, float[] V)[] CrossKvCache { get; set; }

    public int NumTokens { get; }
    public int Dim { get; }
    public int FfnDim { get; }

    public WanWorkspace(int numTokens, int dim, int ffnDim, int numLayers)
    {
        NumTokens = numTokens;
        Dim = dim;
        FfnDim = ffnDim;

        Norm1 = new float[numTokens * dim];
        Normed1 = new float[numTokens * dim];
        Q = new float[numTokens * dim];
        K = new float[numTokens * dim];
        V = new float[numTokens * dim];
        QkvFused = new float[numTokens * dim * 3];
        AttnOut = new float[numTokens * dim];
        Norm3 = new float[numTokens * dim];
        NormedCross = new float[numTokens * dim];
        CrossQ = new float[numTokens * dim];
        CrossAttnOut = new float[numTokens * dim];
        Norm2 = new float[numTokens * dim];
        Normed2 = new float[numTokens * dim];
        Ffn1 = new float[numTokens * ffnDim];
        FfnOut = new float[numTokens * dim];
        HeadShift = new float[dim];
        HeadScale = new float[dim];
        Mod = new float[dim * 6];
        CrossKvCache = new (float[] K, float[] V)[numLayers];
    }
}
