
namespace OpenTail.Stingray.Core;

/// <summary>
/// Serves a Hugging Face SafeTensors package's weights to the engine through
/// <see cref="IModelTensorSource"/>, under OpenTail's canonical tensor names.
/// </summary>
/// <remarks>
/// Hugging Face records dense tensor shapes in row-major <c>[output, input]</c> order, whereas
/// GGUF records dimensions fastest-varying first as <c>[input, output]</c>. The descriptor path
/// reverses that order exactly once in <see cref="ToGgufDimensionOrder"/>; raw tensor bytes remain
/// row-major and are not transposed. SafeTensors source weights remain high precision; quantized
/// local deployment is intentionally the GGUF route.
/// </remarks>
public sealed unsafe class SafetensorsTensorSource : IModelTensorSource, IDisposable
{
    private readonly SafetensorsLoader _loader;
    private readonly Dictionary<string, GgufTensorInfo> _byCanonicalName;
    private readonly Dictionary<string, string> _sourceNameByCanonicalName;
    private readonly Dictionary<string, bool> _isBf16ByCanonicalName;
    private readonly Dictionary<string, nint> _convertedBf16Buffers = new(StringComparer.Ordinal);
    private readonly HashSet<nint> _ownedPointers = [];
    private readonly Lock _convertLock = new();
    private readonly List<GgufTensorInfo> _tensors;
    private bool _disposed;

    private SafetensorsTensorSource(
        SafetensorsLoader loader,
        Dictionary<string, GgufTensorInfo> byCanonicalName,
        Dictionary<string, string> sourceNames,
        Dictionary<string, bool> isBf16Map,
        IReadOnlyDictionary<string, object> metadata)
    {
        _loader = loader;
        _byCanonicalName = byCanonicalName;
        _sourceNameByCanonicalName = sourceNames;
        _isBf16ByCanonicalName = isBf16Map;
        _tensors = [.. byCanonicalName.Values];
        Metadata = metadata;
    }

    /// <inheritdoc/>
    public IReadOnlyList<GgufTensorInfo> Tensors => _tensors;

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> Metadata { get; }

    /// <summary>
    /// Opens a package directory and exposes its dense Llama/Mistral tensors under canonical names.
    /// </summary>
    /// <exception cref="NotSupportedException">A tensor uses a dtype outside the profile.</exception>
    public static SafetensorsTensorSource Open(string packagePath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(packagePath);
        var package = SafetensorsTextModelPackage.Open(packagePath);
        var loader = SafetensorsTextModelPackage.OpenWeights(package);
        try
        {
            var descriptors = new Dictionary<string, GgufTensorInfo>(StringComparer.Ordinal);
            var sourceNames = new Dictionary<string, string>(StringComparer.Ordinal);
            var isBf16Map = new Dictionary<string, bool>(StringComparer.Ordinal);

            foreach (string sourceName in loader.TensorNames)
            {
                string? canonical = SafetensorsTextModelPackage.TryMapToOpenTailTensorName(sourceName);
                if (canonical is null) continue;

                string dtype = loader.GetDtype(sourceName);
                DType mapped = dtype switch
                {
                    "F32" => DType.Float32,
                    "F16" => DType.Float16,
                    "BF16" => DType.Float32,
                    _ => throw new NotSupportedException(
                        $"SafeTensors tensor '{sourceName}' has dtype '{dtype}'; the dense execution " +
                        "path accepts F32, F16, or BF16."),
                };

                int[] shape = loader.GetShape(sourceName);
                long[] dims = ToGgufDimensionOrder(shape);

                descriptors[canonical] = new GgufTensorInfo(canonical, dims.Length, dims, mapped, DataOffset: 0);
                sourceNames[canonical] = sourceName;
                if (dtype == "BF16")
                {
                    isBf16Map[canonical] = true;
                }
            }

            if (!descriptors.ContainsKey("output.weight") && descriptors.TryGetValue("token_embd.weight", out var tokenEmbd))
            {
                descriptors["output.weight"] = new GgufTensorInfo("output.weight", tokenEmbd.NDimensions, tokenEmbd.Dimensions.ToArray(), tokenEmbd.DType, DataOffset: tokenEmbd.DataOffset);
                sourceNames["output.weight"] = sourceNames["token_embd.weight"];
                if (isBf16Map.TryGetValue("token_embd.weight", out bool isTiedBf16) && isTiedBf16)
                {
                    isBf16Map["output.weight"] = true;
                }
            }

            return new SafetensorsTensorSource(loader, descriptors, sourceNames, isBf16Map, package.ToOpenTailMetadata());
        }
        catch
        {
            loader.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Reverses a row-major Hugging Face shape into GGUF's fastest-varying-first dimension order.
    /// </summary>
    internal static long[] ToGgufDimensionOrder(int[] shape)
    {
        var dims = new long[shape.Length];
        for (int i = 0; i < shape.Length; i++) dims[i] = shape[shape.Length - 1 - i];
        return dims;
    }

    /// <inheritdoc/>
    public GgufTensorInfo? FindTensor(string name)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return _byCanonicalName.TryGetValue(name, out var info) ? info : null;
    }

    /// <inheritdoc/>
    public ReadOnlySpan<byte> GetTensorData(GgufTensorInfo tensor)
    {
        byte* pointer = GetTensorDataPtr(tensor);
        long length = ByteLength(tensor);
        return new ReadOnlySpan<byte>(pointer, checked((int)length));
    }

    /// <inheritdoc/>
    public byte* GetTensorDataPtr(GgufTensorInfo tensor)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (_isBf16ByCanonicalName.TryGetValue(tensor.Name, out bool isBf16) && isBf16)
        {
            lock (_convertLock)
            {
                if (_convertedBf16Buffers.TryGetValue(tensor.Name, out nint existingPtr))
                    return (byte*)existingPtr;

                // If output.weight is tied to token_embd.weight, ensure token_embd.weight is converted
                if (tensor.Name == "output.weight" && _sourceNameByCanonicalName.TryGetValue("output.weight", out string? outSrc)
                    && _sourceNameByCanonicalName.TryGetValue("token_embd.weight", out string? tokSrc) && outSrc == tokSrc)
                {
                    var tokenEmbdInfo = _byCanonicalName["token_embd.weight"];
                    byte* tokenPtr = GetTensorDataPtr(tokenEmbdInfo);
                    _convertedBf16Buffers["output.weight"] = (nint)tokenPtr;
                    return tokenPtr;
                }

                return GetRawFloat32Ptr(tensor.Name);
            }
        }

        if (!_sourceNameByCanonicalName.TryGetValue(tensor.Name, out string? sourceNameNonBf16))
            throw new KeyNotFoundException($"SafeTensors source has no tensor '{tensor.Name}'.");
        if (!_loader.TryGetMappedPointer(sourceNameNonBf16, out byte* pointer, out _, out _))
            throw new KeyNotFoundException($"SafeTensors shard has no tensor '{sourceNameNonBf16}'.");
        return pointer;
    }

    private byte* GetRawFloat32Ptr(string canonicalName)
    {
        if (!_sourceNameByCanonicalName.TryGetValue(canonicalName, out string? sourceName))
            throw new KeyNotFoundException($"SafeTensors source has no tensor '{canonicalName}'.");
        if (!_loader.TryGetMappedPointer(sourceName, out byte* pointer, out long bytes, out _))
            throw new KeyNotFoundException($"SafeTensors shard has no tensor '{sourceName}'.");

        if (_isBf16ByCanonicalName.TryGetValue(canonicalName, out bool isBf16) && isBf16)
        {
            lock (_convertLock)
            {
                if (_convertedBf16Buffers.TryGetValue(canonicalName, out nint existing))
                    return (byte*)existing;

                long elementCount = bytes / 2;
                float* dstPtr = (float*)NativeMemory.Alloc((nuint)(elementCount * sizeof(float)));
                ushort* srcPtr = (ushort*)pointer;
                for (long i = 0; i < elementCount; i++)
                {
                    uint bits = (uint)srcPtr[i] << 16;
                    dstPtr[i] = BitConverter.UInt32BitsToSingle(bits);
                }
                _convertedBf16Buffers[canonicalName] = (nint)dstPtr;
                _ownedPointers.Add((nint)dstPtr);
                return (byte*)dstPtr;
            }
        }

        return pointer;
    }

    private long ByteLength(GgufTensorInfo tensor)
    {
        if (_isBf16ByCanonicalName.TryGetValue(tensor.Name, out bool isBf16) && isBf16)
            return tensor.ElementCount * sizeof(float);

        if (!_sourceNameByCanonicalName.TryGetValue(tensor.Name, out string? sourceName))
            throw new KeyNotFoundException($"SafeTensors source has no tensor '{tensor.Name}'.");
        _loader.TryGetMappedPointer(sourceName, out _, out long length, out _);
        return length;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        lock (_convertLock)
        {
            foreach (nint ptr in _ownedPointers)
            {
                NativeMemory.Free((void*)ptr);
            }
            _ownedPointers.Clear();
            _convertedBf16Buffers.Clear();
        }
        _loader.Dispose();
    }
}


