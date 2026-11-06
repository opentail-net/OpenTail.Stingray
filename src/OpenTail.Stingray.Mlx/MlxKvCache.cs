namespace OpenTail.Stingray.Mlx;

/// <summary>
/// One layer's K/V cache, host-resident. TensorSharp keeps its cache device-resident and writes into
/// it with a native <c>mlx_slice_update</c> (<c>MlxFusedOps.TryWriteKvCacheBlock</c>,
/// <c>MlxNative.cs</c>/<c>MlxFusedOps.cs</c> ~line 702) to avoid a host round trip; every op ported so
/// far in this backend already round-trips through host memory per call (see
/// docs/4-performance/gpu/2026-11-06-mlx-metal-backend-plan.md), so there is nothing a native
/// slice-update buys here -- appending is a plain host array copy instead, which needs no
/// mlx-c binding and no graceful-failure test (there is no native call to fail).
/// </summary>
public sealed class MlxKvCache
{
    private readonly float[] _k;
    private readonly float[] _v;

    public int NumHeads { get; }
    public int MaxSeqLen { get; }
    public int HeadDim { get; }
    /// <summary>Number of positions actually written so far.</summary>
    public int Length { get; private set; }

    public MlxKvCache(int numHeads, int maxSeqLen, int headDim)
    {
        if (numHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numHeads));
        if (maxSeqLen <= 0) throw new ArgumentOutOfRangeException(nameof(maxSeqLen));
        if (headDim <= 0) throw new ArgumentOutOfRangeException(nameof(headDim));
        NumHeads = numHeads;
        MaxSeqLen = maxSeqLen;
        HeadDim = headDim;
        _k = new float[numHeads * maxSeqLen * headDim];
        _v = new float[numHeads * maxSeqLen * headDim];
    }

    /// <summary>
    /// Appends <paramref name="seqLen"/> new positions' K/V, head-first [numHeads, seqLen, headDim]
    /// (the layout <see cref="MlxOps.SplitHeads"/> produces), at the current <see cref="Length"/>.
    /// </summary>
    public void Append(ReadOnlySpan<float> newKHeadFirst, ReadOnlySpan<float> newVHeadFirst, int seqLen)
    {
        int expected = NumHeads * seqLen * HeadDim;
        if (newKHeadFirst.Length != expected) throw new ArgumentException($"newKHeadFirst has {newKHeadFirst.Length} elements, expected {expected}.", nameof(newKHeadFirst));
        if (newVHeadFirst.Length != expected) throw new ArgumentException($"newVHeadFirst has {newVHeadFirst.Length} elements, expected {expected}.", nameof(newVHeadFirst));
        if (Length + seqLen > MaxSeqLen) throw new InvalidOperationException($"Appending {seqLen} positions would exceed MaxSeqLen ({MaxSeqLen}); currently at {Length}.");

        for (int h = 0; h < NumHeads; h++)
        {
            newKHeadFirst.Slice(h * seqLen * HeadDim, seqLen * HeadDim)
                .CopyTo(_k.AsSpan(h * MaxSeqLen * HeadDim + Length * HeadDim, seqLen * HeadDim));
            newVHeadFirst.Slice(h * seqLen * HeadDim, seqLen * HeadDim)
                .CopyTo(_v.AsSpan(h * MaxSeqLen * HeadDim + Length * HeadDim, seqLen * HeadDim));
        }
        Length += seqLen;
    }

    /// <summary>The valid [numHeads, Length, headDim] prefix of the K cache (not the full [numHeads, MaxSeqLen, headDim] backing array).</summary>
    public float[] Keys() => ValidPrefix(_k);
    public float[] Values() => ValidPrefix(_v);

    private float[] ValidPrefix(float[] backing)
    {
        var result = new float[NumHeads * Length * HeadDim];
        for (int h = 0; h < NumHeads; h++)
            backing.AsSpan(h * MaxSeqLen * HeadDim, Length * HeadDim).CopyTo(result.AsSpan(h * Length * HeadDim, Length * HeadDim));
        return result;
    }
}
