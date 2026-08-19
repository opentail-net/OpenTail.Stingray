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

    /// <summary>
    /// Cross-attention Keys/Values projected once from the audio encoder output per layer.
    /// Encoder output is fixed for the whole utterance, so these are computed a single time
    /// (see <see cref="WhisperDecoder.PrimeCrossAttention"/>) and reused across every decode step,
    /// instead of re-projecting all audio frames on every generated token.
    /// </summary>
    public float[][]? CrossKeys { get; private set; }
    public float[][]? CrossValues { get; private set; }
    public int CrossFrames { get; private set; }

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
    /// Resets the cached token position to 0. Cross-attention cache is left intact since it
    /// only depends on the (unchanged) audio encoder output.
    /// </summary>
    public void Reset()
    {
        Position = 0;
    }

    public void SetCrossAttentionCache(float[][] crossKeys, float[][] crossValues, int audioFrames)
    {
        CrossKeys = crossKeys;
        CrossValues = crossValues;
        CrossFrames = audioFrames;
    }
}
