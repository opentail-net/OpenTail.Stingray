namespace OpenTail.Stingray.Diffusion.MiniMaxMusic3;

/// <summary>
/// Per-layer KV cache for <see cref="MiniMaxMusic3RvqDepthDecoder"/>'s incremental decoding
/// across the 7 residual codebook steps within an audio frame.
/// Each layer stores one row (`[hidden]`) per cached step.
/// Reset per audio frame.
/// </summary>
public sealed class MiniMaxMusic3RvqDepthKvCache
{
    public int Length => Keys[0].Count;
    internal readonly List<float[]>[] Keys;
    internal readonly List<float[]>[] Values;

    public MiniMaxMusic3RvqDepthKvCache(int numLayers = MiniMaxMusic3Config.RvqDepthDecoderNumLayers)
    {
        Keys = new List<float[]>[numLayers];
        Values = new List<float[]>[numLayers];
        for (int i = 0; i < numLayers; i++)
        {
            Keys[i] = [];
            Values[i] = [];
        }
    }

    public void Reset()
    {
        for (int i = 0; i < Keys.Length; i++)
        {
            Keys[i].Clear();
            Values[i].Clear();
        }
    }
}
