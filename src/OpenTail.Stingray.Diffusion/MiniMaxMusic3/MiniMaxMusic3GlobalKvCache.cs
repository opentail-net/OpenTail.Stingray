namespace OpenTail.Stingray.Diffusion.MiniMaxMusic3;

/// <summary>
/// Real per-layer KV cache for <see cref="MiniMaxMusic3GlobalModel"/>'s incremental (one-token-at-
/// a-time) decoding, matching the real `MiniMaxMusic3AutoregressiveStep` generation loop's
/// `use_cache=True` Qwen3 forward (docs/066-minimax-music3-future-plan.md, "Real per-frame
/// generation loop"). Each layer stores one row (`[nKvHeads*headDim]`, POST rope/qk-norm) per
/// cached position; grows by however many new tokens a given <see cref="MiniMaxMusic3GlobalModel.ForwardIncremental"/>
/// call appends (a multi-token prompt prefill, then one token per subsequent step).
/// </summary>
public sealed class MiniMaxMusic3GlobalKvCache
{
    public int Length { get; internal set; }
    internal readonly List<float[]>[] Keys;
    internal readonly List<float[]>[] Values;

    public MiniMaxMusic3GlobalKvCache(int numLayers)
    {
        Keys = new List<float[]>[numLayers];
        Values = new List<float[]>[numLayers];
        for (int i = 0; i < numLayers; i++)
        {
            Keys[i] = [];
            Values[i] = [];
        }
    }
}
