namespace OpenTail.Stingray.Audio.Whisper;

/// <summary>
/// Lightweight per-sequence Key/Value cache for OpenAI Whisper decoder self-attention layers.
/// Enables $O(1)$ per-token decode step complexity instead of $O(N^2)$ prefix recalculation.
/// </summary>
public sealed class WhisperKvCache
{
    public float[][] Keys { get; }
    public float[][] Values { get; }
    public int MaxCtx { get; }
    public int DModel { get; }
    public int Position { get; set; }

    public WhisperKvCache(int nLayers, int maxCtx, int dModel)
    {
        MaxCtx = maxCtx;
        DModel = dModel;
        Keys = new float[nLayers][];
        Values = new float[nLayers][];

        for (int l = 0; l < nLayers; l++)
        {
            Keys[l] = new float[maxCtx * dModel];
            Values[l] = new float[maxCtx * dModel];
        }

        Position = 0;
    }

    /// <summary>
    /// Resets the cached token position to 0.
    /// </summary>
    public void Reset()
    {
        Position = 0;
    }
}
