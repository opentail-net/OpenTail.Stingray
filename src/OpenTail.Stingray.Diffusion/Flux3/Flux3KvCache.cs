namespace OpenTail.Stingray.Diffusion.Flux3;

/// <summary>
/// Per-layer Key-Value cache for reference frames, keyframe conditioning, and multi-turn video context in FLUX 3.
/// Ported from FLUX.2/Kontext reference architecture to eliminate redundant KV projections.
/// </summary>
public sealed class Flux3LayerKvCache
{
    public float[]? KeyRef { get; private set; }
    public float[]? ValueRef { get; private set; }
    public int NumRefTokens { get; private set; }

    public bool HasCachedTokens => KeyRef != null && ValueRef != null;

    public void Store(float[] key, float[] value, int numTokens)
    {
        KeyRef = key ?? throw new ArgumentNullException(nameof(key));
        ValueRef = value ?? throw new ArgumentNullException(nameof(value));
        NumRefTokens = numTokens;
    }

    public (float[] key, float[] value) Retrieve()
    {
        if (KeyRef == null || ValueRef == null)
            throw new InvalidOperationException("Layer KV cache has not been populated.");
        return (KeyRef, ValueRef);
    }

    public void Clear()
    {
        KeyRef = null;
        ValueRef = null;
        NumRefTokens = 0;
    }
}

/// <summary>
/// Full-model KV Cache manager for FLUX 3 double-stream and single-stream transformer blocks.
/// </summary>
public sealed class Flux3KvCache
{
    private readonly Flux3LayerKvCache[] _doubleBlockCaches;
    private readonly Flux3LayerKvCache[] _singleBlockCaches;

    public int NumDoubleBlocks => _doubleBlockCaches.Length;
    public int NumSingleBlocks => _singleBlockCaches.Length;

    public Flux3KvCache(int numDoubleBlocks, int numSingleBlocks)
    {
        _doubleBlockCaches = new Flux3LayerKvCache[numDoubleBlocks];
        for (int i = 0; i < numDoubleBlocks; i++)
            _doubleBlockCaches[i] = new Flux3LayerKvCache();

        _singleBlockCaches = new Flux3LayerKvCache[numSingleBlocks];
        for (int i = 0; i < numSingleBlocks; i++)
            _singleBlockCaches[i] = new Flux3LayerKvCache();
    }

    public Flux3LayerKvCache GetDoubleLayer(int layerIdx)
    {
        if ((uint)layerIdx >= (uint)_doubleBlockCaches.Length)
            throw new ArgumentOutOfRangeException(nameof(layerIdx));
        return _doubleBlockCaches[layerIdx];
    }

    public Flux3LayerKvCache GetSingleLayer(int layerIdx)
    {
        if ((uint)layerIdx >= (uint)_singleBlockCaches.Length)
            throw new ArgumentOutOfRangeException(nameof(layerIdx));
        return _singleBlockCaches[layerIdx];
    }

    public void Clear()
    {
        foreach (var cache in _doubleBlockCaches) cache.Clear();
        foreach (var cache in _singleBlockCaches) cache.Clear();
    }
}
