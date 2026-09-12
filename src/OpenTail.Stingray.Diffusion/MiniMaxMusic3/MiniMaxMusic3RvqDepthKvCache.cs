using System.Runtime.CompilerServices;

namespace OpenTail.Stingray.Diffusion.MiniMaxMusic3;

/// <summary>
/// Contiguous per-layer KV cache for <see cref="MiniMaxMusic3RvqDepthDecoder"/>'s incremental decoding
/// across the 7 residual codebook steps within an audio frame.
/// Uses preallocated flat memory buffers [numLayers][MaxSteps * HiddenSize].
/// Reset per audio frame at zero allocation cost.
/// </summary>
public sealed class MiniMaxMusic3RvqDepthKvCache
{
    public const int MaxSteps = 16;
    public int Length { get; internal set; }

    internal readonly float[][] Keys;   // [numLayers][MaxSteps * hidden]
    internal readonly float[][] Values; // [numLayers][MaxSteps * hidden]
    internal readonly int HiddenSize;

    public MiniMaxMusic3RvqDepthKvCache(
        int numLayers = MiniMaxMusic3Config.RvqDepthDecoderNumLayers,
        int hiddenSize = MiniMaxMusic3Config.RvqDepthDecoderHiddenSize)
    {
        HiddenSize = hiddenSize;
        Keys = new float[numLayers][];
        Values = new float[numLayers][];
        for (int i = 0; i < numLayers; i++)
        {
            Keys[i] = new float[MaxSteps * hiddenSize];
            Values[i] = new float[MaxSteps * hiddenSize];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Append(int layer, ReadOnlySpan<float> key, ReadOnlySpan<float> val)
    {
        int offset = Length * HiddenSize;
        key.CopyTo(Keys[layer].AsSpan(offset, HiddenSize));
        val.CopyTo(Values[layer].AsSpan(offset, HiddenSize));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ReadOnlySpan<float> GetKeySpan(int layer, int step) =>
        new ReadOnlySpan<float>(Keys[layer], step * HiddenSize, HiddenSize);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ReadOnlySpan<float> GetValSpan(int layer, int step) =>
        new ReadOnlySpan<float>(Values[layer], step * HiddenSize, HiddenSize);

    public void Reset()
    {
        Length = 0;
    }
}
