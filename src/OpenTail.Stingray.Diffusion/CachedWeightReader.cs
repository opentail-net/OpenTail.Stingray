namespace OpenTail.Stingray.Diffusion;

/// <summary>
/// Shared prefix + cache wrapper around <see cref="IWeightLoader"/> for models with a single,
/// fixed weight-name prefix and no fallback-prefix resolution (SD1.5, SDXL, SD3 -- other models
/// like Wan/QwenImage/HunyuanVideo/ControlNet need real multi-prefix fallback for checkpoints
/// distributed under different container conventions, and keep their own `Resolve` logic).
/// Extracted from three byte-identical copies of `GetWeight`/`TryGetWeight`/`_weightCache`.
/// </summary>
internal sealed class CachedWeightReader
{
    private readonly IWeightLoader _weights;
    private readonly string _prefix;
    private readonly Dictionary<string, float[]> _cache = new(StringComparer.Ordinal);

    /// <summary>The prefix this reader prepends to every lookup -- exposed for callers that need
    /// to build a matching cache key of their own (e.g. a separate GPU-upload cache).</summary>
    public string Prefix => _prefix;

    public CachedWeightReader(IWeightLoader weights, string prefix)
    {
        _weights = weights;
        _prefix = prefix;
    }

    public float[] Get(string name)
    {
        string fullName = _prefix + name;
        lock (_cache)
        {
            if (!_cache.TryGetValue(fullName, out var w))
            {
                w = _weights.ReadF32(fullName);
                _cache[fullName] = w;
            }
            return w;
        }
    }

    public float[]? TryGet(string name)
    {
        string fullName = _prefix + name;
        lock (_cache)
        {
            if (_cache.TryGetValue(fullName, out var w)) return w;
            if (_weights.Contains(fullName))
            {
                w = _weights.ReadF32(fullName);
                _cache[fullName] = w;
                return w;
            }
            return null;
        }
    }

    public void Clear()
    {
        lock (_cache) _cache.Clear();
    }
}
