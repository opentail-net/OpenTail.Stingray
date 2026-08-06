namespace OpenTail.Stingray.Core;

/// <summary>
/// Presents a validated Hugging Face SafeTensors Llama/Mistral package through OpenTail's
/// canonical tensor names. Values are read as F32 on demand, preserving the source's F32/F16/BF16
/// weights rather than applying a deployment quantization format.
/// </summary>
/// <remarks>
/// This is the original-weights source for the future dense CPU execution lane. It deliberately
/// does not expose raw block-quantized pointers, so GGUF-specific CPU/GPU kernels remain GGUF-only.
/// </remarks>
public sealed class SafetensorsLlamaWeightLoader : IWeightLoader
{
    private readonly SafetensorsLoader _source;
    private readonly Dictionary<string, string> _sourceNameByCanonicalName;

    private SafetensorsLlamaWeightLoader(SafetensorsTextModelPackage package, SafetensorsLoader source)
    {
        Package = package;
        _source = source;
        _sourceNameByCanonicalName = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (string sourceName in source.TensorNames)
        {
            string? canonicalName = SafetensorsTextModelPackage.TryMapToOpenTailTensorName(sourceName);
            if (canonicalName is not null) _sourceNameByCanonicalName[canonicalName] = sourceName;
        }
    }

    /// <summary>Validated external package metadata and asset locations.</summary>
    public SafetensorsTextModelPackage Package { get; }

    /// <summary>Opens a validated package and maps its supported tensors to canonical names.</summary>
    public static SafetensorsLlamaWeightLoader Open(string path)
    {
        var package = SafetensorsTextModelPackage.Open(path);
        return new SafetensorsLlamaWeightLoader(package, SafetensorsTextModelPackage.OpenWeights(package));
    }

    /// <inheritdoc/>
    public bool Contains(string name) => _sourceNameByCanonicalName.ContainsKey(name);

    /// <inheritdoc/>
    public float[] ReadF32(string name) => _source.ReadF32(ResolveSourceName(name));

    /// <summary>Returns the validated source storage dtype for a canonical tensor name.</summary>
    public string GetDtype(string name) => _source.GetDtype(ResolveSourceName(name));

    /// <summary>Returns the source row-major tensor shape for a canonical tensor name.</summary>
    public int[] GetShape(string name) => _source.GetShape(ResolveSourceName(name));

    /// <inheritdoc/>
    public unsafe bool TryGetRaw(string name,
        out nint dataPtr, out long byteLen,
        out DType dtype, out int rows, out int cols)
    {
        dataPtr = 0; byteLen = 0; dtype = default; rows = 0; cols = 0;
        return false;
    }

    /// <inheritdoc/>
    public void Dispose() => _source.Dispose();

    private string ResolveSourceName(string canonicalName) =>
        _sourceNameByCanonicalName.TryGetValue(canonicalName, out string? sourceName)
            ? sourceName
            : throw new KeyNotFoundException($"SafeTensors Llama tensor not found: '{canonicalName}'.");
}
