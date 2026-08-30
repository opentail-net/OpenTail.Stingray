namespace OpenTail.Stingray.Audio.Xtts;

/// <summary>
/// Preallocated KV-cache and scratch workspace for single-step XTTS-v2 GPT2 trunk evaluation.
/// Eliminates all per-step heap allocations and enables O(1) layer evaluation with O(T) attention.
/// </summary>
public sealed class XttsGptCache
{
    public const int MaxContextLength = 1024;
    public int ModelDim { get; }
    public int NumLayers { get; }
    public int NumHeads { get; }
    public int HeadDim { get; }

    // K and V caches: [numLayers][maxPositions][modelDim]
    public float[][][] K { get; }
    public float[][][] V { get; }
    public int[] Counts { get; }

    // Preallocated scratch workspace for zero GC allocations per step
    public float[] Normed { get; }
    public float[] Qkv { get; }
    public float[] Q { get; }
    public float[] Context { get; }
    public float[] AttnOut { get; }
    public float[] H1 { get; }
    public float[] FfnNormed { get; }
    public float[] FfnMid { get; }
    public float[] FfnOut { get; }
    public float[] Scores { get; }
    public float[] Output { get; }
    public float[] LastHidden { get; }

    public XttsGptCache(XttsGptWeights w, int maxContext = MaxContextLength)
    {
        ModelDim = XttsGptWeights.ModelDim;
        NumLayers = XttsGptWeights.NumLayers;
        NumHeads = XttsGptWeights.NumHeads;
        HeadDim = XttsGptWeights.HeadDim;

        K = new float[NumLayers][][];
        V = new float[NumLayers][][];
        Counts = new int[NumLayers];

        for (int l = 0; l < NumLayers; l++)
        {
            K[l] = new float[maxContext][];
            V[l] = new float[maxContext][];
            for (int p = 0; p < maxContext; p++)
            {
                K[l][p] = new float[ModelDim];
                V[l][p] = new float[ModelDim];
            }
        }

        Normed = new float[ModelDim];
        Qkv = new float[3 * ModelDim];
        Q = new float[ModelDim];
        Context = new float[ModelDim];
        AttnOut = new float[ModelDim];
        H1 = new float[ModelDim];
        FfnNormed = new float[ModelDim];
        FfnMid = new float[XttsGptWeights.FfnDim];
        FfnOut = new float[ModelDim];
        Scores = new float[maxContext];
        Output = new float[ModelDim];
        LastHidden = new float[ModelDim];
    }

    public void Reset()
    {
        Array.Clear(Counts, 0, Counts.Length);
    }
}
