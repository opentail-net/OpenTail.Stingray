using OpenTail.Stingray.Mlx;

namespace OpenTail.Stingray.Tests.Cuda;

/// <summary>
/// <see cref="MlxKvCache"/> is plain host-side array bookkeeping -- no native call anywhere in it
/// (see its type doc comment for why) -- so unlike every other MLX type in this backend, it can be
/// fully tested for real here, not just "fails predictably."
/// </summary>
public sealed class MlxKvCacheTests
{
    [Fact]
    public void Append_ThenKeys_RoundTripsHeadFirstData()
    {
        // numHeads=2, headDim=2, two appends of 1 position each. Fill each append with a distinct,
        // recognizable value per head so a head/position mixup would be visible immediately.
        var cache = new MlxKvCache(numHeads: 2, maxSeqLen: 4, headDim: 2);

        // Append 1: head0=[1,1], head1=[2,2]
        cache.Append(newKHeadFirst: [1f, 1f, 2f, 2f], newVHeadFirst: [10f, 10f, 20f, 20f], seqLen: 1);
        // Append 2: head0=[3,3], head1=[4,4]
        cache.Append(newKHeadFirst: [3f, 3f, 4f, 4f], newVHeadFirst: [30f, 30f, 40f, 40f], seqLen: 1);

        Assert.Equal(2, cache.Length);
        // Head-first [numHeads=2, Length=2, headDim=2]: head0's two positions, then head1's two positions.
        Assert.Equal([1f, 1f, 3f, 3f, 2f, 2f, 4f, 4f], cache.Keys());
        Assert.Equal([10f, 10f, 30f, 30f, 20f, 20f, 40f, 40f], cache.Values());
    }

    [Fact]
    public void Append_MultiPosition_PreservesOrderWithinAHead()
    {
        var cache = new MlxKvCache(numHeads: 1, maxSeqLen: 3, headDim: 2);
        cache.Append(newKHeadFirst: [1f, 2f, 3f, 4f], newVHeadFirst: [10f, 20f, 30f, 40f], seqLen: 2);
        Assert.Equal(2, cache.Length);
        Assert.Equal([1f, 2f, 3f, 4f], cache.Keys());
    }

    [Fact]
    public void Append_BeyondMaxSeqLen_Throws()
    {
        var cache = new MlxKvCache(numHeads: 1, maxSeqLen: 1, headDim: 2);
        cache.Append(newKHeadFirst: [1f, 2f], newVHeadFirst: [1f, 2f], seqLen: 1);
        Assert.Throws<InvalidOperationException>(() => cache.Append([1f, 2f], [1f, 2f], seqLen: 1));
    }

    [Fact]
    public void Append_LengthMismatch_Throws()
    {
        var cache = new MlxKvCache(numHeads: 2, maxSeqLen: 4, headDim: 2);
        Assert.Throws<ArgumentException>(() => cache.Append(newKHeadFirst: new float[3], newVHeadFirst: new float[4], seqLen: 1));
    }

    [Fact]
    public void Keys_BeforeAnyAppend_IsEmpty()
    {
        var cache = new MlxKvCache(numHeads: 2, maxSeqLen: 4, headDim: 2);
        Assert.Empty(cache.Keys());
        Assert.Equal(0, cache.Length);
    }
}
