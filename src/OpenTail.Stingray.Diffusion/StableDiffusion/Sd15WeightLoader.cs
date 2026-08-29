
namespace OpenTail.Stingray.Diffusion.StableDiffusion;

/// <summary>
/// Sub-loader view of an underlying IWeightLoader with a key prefix filter and remapping.
/// Enables ClipLEncoder, UNet2DConditionModel, and VaeDecoder to read from a unified SD 1.5 checkpoint.
/// </summary>
public sealed class PrefixWeightLoader : IWeightLoader
{
    private readonly IWeightLoader _inner;
    private readonly string _prefix;
    private readonly Dictionary<string, string> _aliasMap = new(StringComparer.Ordinal);

    public PrefixWeightLoader(IWeightLoader inner, string prefix)
    {
        _inner = inner;
        _prefix = prefix;
    }

    public void AddAlias(string requestedKey, string actualKey)
    {
        _aliasMap[requestedKey] = actualKey;
    }

    public bool Contains(string name)
    {
        if (_aliasMap.TryGetValue(name, out var aliased))
            return _inner.Contains(aliased);

        return _inner.Contains(_prefix + name) || _inner.Contains(name);
    }

    public float[] ReadF32(string name)
    {
        if (_aliasMap.TryGetValue(name, out var aliased))
            return _inner.ReadF32(aliased);

        if (_inner.Contains(_prefix + name))
            return _inner.ReadF32(_prefix + name);

        return _inner.ReadF32(name);
    }

    public unsafe bool TryGetRaw(string name, out nint dataPtr, out long byteLen, out DType dtype, out int rows, out int cols)
    {
        if (_aliasMap.TryGetValue(name, out var aliased))
            return _inner.TryGetRaw(aliased, out dataPtr, out byteLen, out dtype, out rows, out cols);

        if (_inner.Contains(_prefix + name))
            return _inner.TryGetRaw(_prefix + name, out dataPtr, out byteLen, out dtype, out rows, out cols);

        return _inner.TryGetRaw(name, out dataPtr, out byteLen, out dtype, out rows, out cols);
    }

    public void Dispose()
    {
        // Inner loader lifecycle is managed externally
    }
}
