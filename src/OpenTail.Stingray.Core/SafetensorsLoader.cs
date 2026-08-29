using System.IO.MemoryMappedFiles;
using System.Buffers.Binary;

namespace OpenTail.Stingray.Core;

/// <summary>
/// Safetensors file reader supporting both single-file and multi-shard directory layouts.
///
/// Single file:  SafetensorsLoader.Open("model.safetensors")
/// Sharded dir:  SafetensorsLoader.OpenDirectory("path/to/model/")
///   — reads model.safetensors.index.json to map tensor names to shard files,
///     OR falls back to merging all model*.safetensors files in the directory.
///
/// Format per file: [u64-LE header_size] [header_size JSON bytes] [raw tensor data]
/// JSON maps tensor_name → {dtype, shape, data_offsets:[start, end]}.
/// </summary>
public sealed class SafetensorsLoader : IWeightLoader
{
    // One entry per tensor, points to its shard
    private sealed record TensorInfo(string Dtype, int[] Shape, long Start, long End, int ShardIndex)
    {
        public long ElementCount
        {
            get { long n = 1; foreach (var d in Shape) n = checked(n * d); return n; }
        }
    }

    // The SafeTensors reference implementation caps JSON headers at 100 MiB. The cap prevents a
    // corrupt length field from turning inspection into a multi-gigabyte allocation attempt.
    private const int MaxHeaderBytes = 100 * 1024 * 1024;

    private readonly List<(FileStream file, long dataOffset)> _shards;
    private readonly Dictionary<string, TensorInfo> _tensors;

    // Lazily created, one per shard, released only at Dispose. See TryGetMappedPointer.
    private readonly MemoryMappedFile?[] _maps;
    private readonly MemoryMappedViewAccessor?[] _views;
    private readonly nint[] _mapBases;
    private readonly object _mapGate = new();
    private bool _disposed;

    private SafetensorsLoader(List<(FileStream, long)> shards, Dictionary<string, TensorInfo> tensors)
    {
        _shards  = shards;
        _tensors = tensors;
        _maps    = new MemoryMappedFile?[shards.Count];
        _views   = new MemoryMappedViewAccessor?[shards.Count];
        _mapBases = new nint[shards.Count];
    }

    /// <summary>
    /// Exposes a tensor's bytes as a pointer into a memory-mapped view of its shard, valid until this
    /// loader is disposed.
    /// </summary>
    /// <remarks>
    /// <para><b>Why mapping and not the FileStream reads above.</b> The engine's tensor-source contract
    /// (<see cref="IModelTensorSource.GetTensorDataPtr"/>) requires a pointer that stays valid for the
    /// source's lifetime, because the forward pass stores it and reads it from other threads. A
    /// stream read into a managed array cannot promise that. Mapping also matches how GGUF is served,
    /// so residency stays OS-managed and nothing materialises a whole model — which is what the plan
    /// requires of a 7B/70B package.</para>
    ///
    /// <para><b>Deliberately additive.</b> The <see cref="ReadF32"/>/<see cref="ReadRaw"/> paths keep
    /// their FileStream behaviour because the diffusion pipeline depends on them; a mapping is created
    /// alongside, only for shards someone actually asks to map.</para>
    ///
    /// <para>Returns false when the tensor is unknown. Mapping failures throw, because a shard that
    /// cannot be mapped is not a condition a caller can sensibly continue from.</para>
    /// </remarks>
    public unsafe bool TryGetMappedPointer(string name, out byte* pointer, out long byteLength, out string dtype)
    {
        pointer = null; byteLength = 0; dtype = string.Empty;
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (!_tensors.TryGetValue(name, out var info)) return false;

        nint basePtr = EnsureMapped(info.ShardIndex);
        var (_, dataOffset) = _shards[info.ShardIndex];
        pointer = (byte*)basePtr + dataOffset + info.Start;
        byteLength = info.End - info.Start;
        dtype = info.Dtype;
        return true;
    }

    private nint EnsureMapped(int shardIndex)
    {
        lock (_mapGate)
        {
            if (_mapBases[shardIndex] != 0) return _mapBases[shardIndex];

            var (file, _) = _shards[shardIndex];
            var map = MemoryMappedFile.CreateFromFile(file, mapName: null, capacity: 0,
                MemoryMappedFileAccess.Read, HandleInheritability.None, leaveOpen: true);
            var view = map.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);

            unsafe
            {
                byte* basePtr = null;
                view.SafeMemoryMappedViewHandle.AcquirePointer(ref basePtr);
                // The view's own offset is 0 here, but AcquirePointer can hand back a pointer to the
                // start of the containing allocation granule, so the documented correction still
                // applies rather than being assumed away.
                basePtr += view.PointerOffset;
                _mapBases[shardIndex] = (nint)basePtr;
            }
            _maps[shardIndex] = map;
            _views[shardIndex] = view;
            return _mapBases[shardIndex];
        }
    }

    // ── Factory methods ───────────────────────────────────────────────────

    public static SafetensorsLoader Open(string path)
    {
        var (shard, tensors) = ParseFile(path, 0);
        return new SafetensorsLoader([shard], tensors);
    }

    /// <summary>
    /// Load a multi-shard model directory.
    /// Reads model.safetensors.index.json if present; otherwise merges all model*.safetensors files.
    /// </summary>
    public static SafetensorsLoader OpenDirectory(string dir)
    {
        string indexPath = Path.Combine(dir, "model.safetensors.index.json");
        if (File.Exists(indexPath))
            return OpenFromIndex(dir, indexPath);

        // Fallback: merge all model*.safetensors (or diffusion_pytorch_model*.safetensors) in directory
        string[] candidates = [
            ..Directory.GetFiles(dir, "model*.safetensors"),
            ..Directory.GetFiles(dir, "diffusion_pytorch_model*.safetensors"),
        ];
        Array.Sort(candidates, StringComparer.Ordinal);

        if (candidates.Length == 0)
            throw new FileNotFoundException($"No safetensors files found in directory: {dir}");

        var shards  = new List<(FileStream, long)>(candidates.Length);
        var tensors = new Dictionary<string, TensorInfo>(StringComparer.Ordinal);

        try
        {
            for (int i = 0; i < candidates.Length; i++)
            {
                var (shard, shardTensors) = ParseFile(candidates[i], i);
                try
                {
                    AddTensors(tensors, shardTensors, candidates[i]);
                    shards.Add(shard);
                }
                catch
                {
                    shard.file.Dispose();
                    throw;
                }
            }
            return new SafetensorsLoader(shards, tensors);
        }
        catch
        {
            foreach (var (file, _) in shards) file.Dispose();
            throw;
        }
    }

    private static SafetensorsLoader OpenFromIndex(string dir, string indexPath)
    {
        var indexJson = File.ReadAllBytes(indexPath);
        using var doc = JsonDocument.Parse(indexJson);
        if (doc.RootElement.ValueKind != JsonValueKind.Object
            || !doc.RootElement.TryGetProperty("weight_map", out var weightMap)
            || weightMap.ValueKind != JsonValueKind.Object)
            throw new InvalidDataException("Safetensors shard index must contain a weight_map object.");

        // Collect unique shard filenames in sorted order
        var shardFileNames = new SortedSet<string>(StringComparer.Ordinal);
        var expectedShardByTensor = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var prop in weightMap.EnumerateObject())
        {
            if (prop.Value.ValueKind != JsonValueKind.String || string.IsNullOrWhiteSpace(prop.Value.GetString()))
                throw new InvalidDataException($"Safetensors shard index has no shard filename for tensor '{prop.Name}'.");
            if (!expectedShardByTensor.TryAdd(prop.Name, prop.Value.GetString()!))
                throw new InvalidDataException($"Safetensors shard index contains duplicate tensor '{prop.Name}'.");
            shardFileNames.Add(prop.Value.GetString()!);
        }

        string root = Path.GetFullPath(dir);
        string rootPrefix = Path.EndsInDirectorySeparator(root) ? root : root + Path.DirectorySeparatorChar;

        var shards  = new List<(FileStream, long)>(shardFileNames.Count);
        var tensors = new Dictionary<string, TensorInfo>(StringComparer.Ordinal);
        var shardIndexByName = new Dictionary<string, int>(StringComparer.Ordinal);

        try
        {
            foreach (var shardFile in shardFileNames)
            {
                int idx = shards.Count;
                string shardPath = Path.GetFullPath(Path.Combine(root, shardFile));
                if (!shardPath.StartsWith(rootPrefix, StringComparison.Ordinal))
                    throw new InvalidDataException($"Safetensors shard index references '{shardFile}' outside package root.");
                var (shard, shardTensors) = ParseFile(shardPath, idx);
                try
                {
                    AddTensors(tensors, shardTensors, shardPath);
                    shards.Add(shard);
                    shardIndexByName.Add(shardFile, idx);
                }
                catch
                {
                    shard.file.Dispose();
                    throw;
                }
            }

            if (tensors.Count != expectedShardByTensor.Count)
                throw new InvalidDataException("Safetensors shard index and shard tensor inventories differ.");
            foreach (var (tensorName, shardName) in expectedShardByTensor)
            {
                if (!tensors.TryGetValue(tensorName, out var tensor)
                    || tensor.ShardIndex != shardIndexByName[shardName])
                    throw new InvalidDataException($"Safetensors shard index maps tensor '{tensorName}' to the wrong shard.");
            }
            return new SafetensorsLoader(shards, tensors);
        }
        catch
        {
            foreach (var (file, _) in shards) file.Dispose();
            throw;
        }
    }

    // ── Tensor access ─────────────────────────────────────────────────────

    public bool Contains(string name) => _tensors.ContainsKey(name);

    public IEnumerable<string> TensorNames => _tensors.Keys;

    /// <summary>Number of tensors across all opened shards.</summary>
    public int TensorCount => _tensors.Count;

    /// <summary>Returns the storage dtype declared for a tensor without reading its data.</summary>


    public string GetDtype(string name)
    {
        if (!_tensors.TryGetValue(name, out var info))
            throw new KeyNotFoundException($"Safetensors tensor not found: '{name}'");
        return info.Dtype;
    }

    /// <summary>Read a tensor as float32. Handles F32, F16, BF16, F8_E4M3, F8_E5M2.</summary>
    public float[] ReadF32(string name)
    {
        if (!_tensors.TryGetValue(name, out var info))
            throw new KeyNotFoundException($"Safetensors tensor not found: '{name}'");

        long byteLen = info.End - info.Start;
        var raw = new byte[checked((int)byteLen)];
        var (file, dataOffset) = _shards[info.ShardIndex];

        lock (file)
        {
            file.Seek(dataOffset + info.Start, SeekOrigin.Begin);
            file.ReadExactly(raw);
        }

        int count  = checked((int)info.ElementCount);
        var result = new float[count];

        switch (info.Dtype)
        {
            case "F32":
                MemoryMarshal.Cast<byte, float>(raw).CopyTo(result);
                break;
            case "F16":
                var f16 = MemoryMarshal.Cast<byte, Half>(raw);
                for (int i = 0; i < count; i++) result[i] = (float)f16[i];
                break;
            case "BF16":
                var bf16 = MemoryMarshal.Cast<byte, ushort>(raw);
                for (int i = 0; i < count; i++)
                    result[i] = BitConverter.Int32BitsToSingle((int)((uint)bf16[i] << 16));
                break;
            // The per-byte helpers these replaced were not merely slower, they were wrong: neither
            // decoded the formats' non-finite encodings. E4M3FN reserves S.1111.111 for NaN and tops
            // out at 448, so the old code returned +480 — a value the format cannot represent — for
            // 0x7F; E5M2 exponent 31 is Inf/NaN, and the old code returned finite values near 2^16.
            case "F8_E4M3":
                FastVectorTypeConverter.ConvertFp8E4M3ToF32(raw.AsSpan(0, count), result);
                break;
            case "F8_E5M2":
                FastVectorTypeConverter.ConvertFp8E5M2ToF32(raw.AsSpan(0, count), result);
                break;
            default:
                throw new NotSupportedException($"Safetensors dtype '{info.Dtype}' not supported.");
        }

        return result;
    }

    /// <summary>
    /// Read a tensor's raw (unconverted) bytes plus its safetensors dtype string
    /// (e.g. "BF16", "F32"). For consumers that keep large tensors in their storage
    /// dtype and convert rows on demand (e.g. the DSpark draft head's BF16
    /// embedding/markov tables) instead of materializing a full F32 copy.
    /// </summary>
    public byte[] ReadRaw(string name, out string dtype)
    {
        if (!_tensors.TryGetValue(name, out var info))
            throw new KeyNotFoundException($"Safetensors tensor not found: '{name}'");

        long byteLen = info.End - info.Start;
        var raw = new byte[checked((int)byteLen)];
        var (file, dataOffset) = _shards[info.ShardIndex];

        lock (file)
        {
            file.Seek(dataOffset + info.Start, SeekOrigin.Begin);
            file.ReadExactly(raw);
        }

        dtype = info.Dtype;
        return raw;
    }

    /// <summary>Read tensor shape without loading data.</summary>
    public int[] GetShape(string name)
    {
        if (!_tensors.TryGetValue(name, out var info))
            throw new KeyNotFoundException($"Safetensors tensor not found: '{name}'");
        return (int[])info.Shape.Clone();
    }

    // ── Internal parsing ──────────────────────────────────────────────────

    private static ((FileStream file, long dataOffset), Dictionary<string, TensorInfo>) ParseFile(string path, int shardIdx)
    {
        var file = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read,
                                  bufferSize: 65536, useAsync: false);
        try
        {
            if (file.Length < sizeof(ulong))
                throw new InvalidDataException($"Safetensors file '{path}' is shorter than its header length field.");

            Span<byte> hdrLenBuf = stackalloc byte[sizeof(ulong)];
            file.ReadExactly(hdrLenBuf);
            ulong hdrLen = BinaryPrimitives.ReadUInt64LittleEndian(hdrLenBuf);
            long maxAvailableHeader = file.Length - sizeof(ulong);
            if (hdrLen > MaxHeaderBytes || hdrLen > (ulong)maxAvailableHeader)
                throw new InvalidDataException($"Safetensors header length {hdrLen} is invalid for '{path}'.");

            var hdrBytes = new byte[(int)hdrLen];
            file.ReadExactly(hdrBytes);
            long dataOffset = 8L + (long)hdrLen;
            long dataLength = file.Length - dataOffset;

            var tensors = new Dictionary<string, TensorInfo>(StringComparer.Ordinal);
            using var doc = JsonDocument.Parse(hdrBytes);
            if (doc.RootElement.ValueKind != JsonValueKind.Object)
                throw new InvalidDataException("Safetensors header root must be a JSON object.");
            var ranges = new List<(long Start, long End, string Name)>();
            foreach (var prop in doc.RootElement.EnumerateObject())
            {
                if (prop.Name == "__metadata__") continue;
                var obj      = prop.Value;
                if (obj.ValueKind != JsonValueKind.Object)
                    throw new InvalidDataException($"Safetensors tensor '{prop.Name}' descriptor must be an object.");
                var dtype    = obj.GetProperty("dtype").GetString();
                if (string.IsNullOrEmpty(dtype))
                    throw new InvalidDataException($"Safetensors tensor '{prop.Name}' has no dtype.");
                var shapeArr = obj.GetProperty("shape");
                var offsets  = obj.GetProperty("data_offsets");
                if (shapeArr.ValueKind != JsonValueKind.Array || offsets.ValueKind != JsonValueKind.Array
                    || offsets.GetArrayLength() != 2)
                    throw new InvalidDataException($"Safetensors tensor '{prop.Name}' has an invalid shape or data_offsets descriptor.");

                int[] shape = new int[shapeArr.GetArrayLength()];
                int si = 0;
                foreach (var el in shapeArr.EnumerateArray())
                {
                    int dim = el.GetInt32();
                    if (dim < 0) throw new InvalidDataException($"Safetensors tensor '{prop.Name}' has a negative shape dimension.");
                    shape[si++] = dim;
                }

                long start = offsets[0].GetInt64();
                long end   = offsets[1].GetInt64();
                if (start < 0 || end < start || end > dataLength)
                    throw new InvalidDataException($"Safetensors tensor '{prop.Name}' has data offsets outside its shard.");

                var info = new TensorInfo(dtype, shape, start, end, shardIdx);
                int bytesPerElement = BytesPerElement(dtype);
                if (bytesPerElement > 0 && checked(info.ElementCount * bytesPerElement) != end - start)
                    throw new InvalidDataException($"Safetensors tensor '{prop.Name}' byte range does not match its shape and dtype.");
                if (!tensors.TryAdd(prop.Name, info))
                    throw new InvalidDataException($"Safetensors header contains duplicate tensor '{prop.Name}'.");
                ranges.Add((start, end, prop.Name));
            }

            long priorEnd = 0;
            foreach (var range in ranges.OrderBy(x => x.Start))
            {
                if (range.Start < priorEnd)
                    throw new InvalidDataException($"Safetensors tensor '{range.Name}' overlaps another tensor's byte range.");
                priorEnd = range.End;
            }

            return ((file, dataOffset), tensors);
        }
        catch (OverflowException ex)
        {
            file.Dispose();
            throw new InvalidDataException($"Safetensors tensor dimensions overflow in '{path}'.", ex);
        }
        catch
        {
            file.Dispose();
            throw;
        }
    }

    private static void AddTensors(Dictionary<string, TensorInfo> target,
        Dictionary<string, TensorInfo> source, string shardPath)
    {
        foreach (var (name, info) in source)
            if (!target.TryAdd(name, info))
                throw new InvalidDataException($"Safetensors tensor '{name}' appears in more than one shard (including '{shardPath}').");
    }

    private static int BytesPerElement(string dtype) => dtype switch
    {
        "F32" => 4,
        "F16" or "BF16" => 2,
        "F8_E4M3" or "F8_E5M2" => 1,
        _ => 0,
    };

    // F8 decoding now lives in FastVectorTypeConverter, which owns the lookup tables and the
    // non-finite encodings both of the helpers that used to live here got wrong.

    public void Dispose()
    {
        lock (_mapGate)
        {
            if (_disposed) return;
            _disposed = true;
            for (int i = 0; i < _views.Length; i++)
            {
                if (_mapBases[i] != 0) _views[i]!.SafeMemoryMappedViewHandle.ReleasePointer();
                _views[i]?.Dispose();
                _maps[i]?.Dispose();
                _mapBases[i] = 0;
            }
        }
        foreach (var (file, _) in _shards) file.Dispose();
    }

    /// <inheritdoc/>
    /// Safetensors data is plain float32 (no block quantization) and is accessed via
    /// <see cref="ReadF32"/>. Raw pointer access is not supported for this backend.
    public unsafe bool TryGetRaw(string name,
        out nint dataPtr, out long byteLen,
        out DType dtype, out int rows, out int cols)
    {
        dataPtr = 0; byteLen = 0; dtype = default; rows = 0; cols = 0;
        return false;
    }
}


