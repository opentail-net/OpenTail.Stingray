using System.Buffers.Binary;
using System.IO.MemoryMappedFiles;

namespace OpenTail.Stingray.Core;

/// <summary>
/// Parses and provides zero-copy access to a GGUF model file (or split multi-shard model).
/// Uses memory-mapped I/O so tensor data is paged from disk on demand.
/// Split models are detected automatically by the -00001-of-NNNNN.gguf filename pattern
/// or by the general.split.count metadata key.
/// </summary>
public sealed unsafe class GgufModel : IDisposable, IModelTensorSource
{
    private const uint GgufMagic = 0x46554747; // "GGUF" as little-endian uint32
    private const int DefaultAlignment = 32;

    // Per-shard memory mapping state (index 0 = first/only shard)
    private readonly MemoryMappedFile[] _shardMmfs;
    private readonly MemoryMappedViewAccessor[] _shardAccessors;
    private readonly byte*[] _shardBasePtrs;
    private readonly long[] _shardFileSizes;
    private readonly long[] _shardDataStartOffsets;
    private readonly long[] _shardTensorInfoEndOffsets;
    private bool _disposed;

    public GgufHeader Header { get; }
    public IReadOnlyDictionary<string, object> Metadata { get; }
    public IReadOnlyList<GgufTensorInfo> Tensors { get; }

    private GgufModel(
        MemoryMappedFile[] shardMmfs,
        MemoryMappedViewAccessor[] shardAccessors,
        byte*[] shardBasePtrs,
        long[] shardFileSizes,
        long[] shardDataStartOffsets,
        long[] shardTensorInfoEndOffsets,
        GgufHeader header,
        IReadOnlyDictionary<string, object> metadata,
        IReadOnlyList<GgufTensorInfo> tensors)
    {
        _shardMmfs = shardMmfs;
        _shardAccessors = shardAccessors;
        _shardBasePtrs = shardBasePtrs;
        _shardFileSizes = shardFileSizes;
        _shardDataStartOffsets = shardDataStartOffsets;
        _shardTensorInfoEndOffsets = shardTensorInfoEndOffsets;
        Header = header;
        Metadata = metadata;
        Tensors = tensors;
    }

    /// <summary>
    /// Opens and parses a GGUF model file (or the first shard of a split model).
    /// If the filename matches the pattern *-00001-of-NNNNN.gguf, all sibling shards
    /// are opened automatically. Metadata is parsed eagerly; tensor data is lazy (mmap).
    /// </summary>
    public static GgufModel Open(string path)
    {
        var shardPaths = ResolveShardPaths(path);
        int shardCount = shardPaths.Length;

        var mmfs       = new MemoryMappedFile[shardCount];
        var accessors  = new MemoryMappedViewAccessor[shardCount];
        var basePtrs   = new byte*[shardCount];
        var fileSizes  = new long[shardCount];
        var dataStarts = new long[shardCount];
        var tensorInfoEnds = new long[shardCount];

        try
        {
            // Open all shard files and acquire memory-mapped pointers
            for (int s = 0; s < shardCount; s++)
            {
                if (!File.Exists(shardPaths[s]))
                    throw new FileNotFoundException("GGUF shard file not found.", shardPaths[s]);
                // FileInfo.Length reads the reparse point's own attributes without following it,
                // so it reports 0 for a symlink even though the OS transparently follows the link
                // for actual reads (e.g. models/ pointing GGUFs at an external drive). Opening a
                // handle and reading its Length follows the link and reports the real target size.
                using (var probe = File.OpenHandle(shardPaths[s], FileMode.Open, FileAccess.Read, FileShare.Read))
                    fileSizes[s] = RandomAccess.GetLength(probe);
                mmfs[s]      = MemoryMappedFile.CreateFromFile(shardPaths[s], FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
                accessors[s] = mmfs[s].CreateViewAccessor(0, fileSizes[s], MemoryMappedFileAccess.Read);
                byte* ptr = null;
                accessors[s].SafeMemoryMappedViewHandle.AcquirePointer(ref ptr);
                basePtrs[s] = ptr;
            }

            // Shard 0 carries the metadata; later shards repeat it (skipped) and add their own tensors. The same
            // ParseShard serves a remote prefix (ParseIndex), so a local and a remote read cannot disagree.
            var allTensors = new List<GgufTensorInfo>();
            var shard0 = ParseShard(basePtrs[0], fileSizes[0], fileSizes[0], shardIndex: 0, DefaultAlignment, allTensors);
            dataStarts[0] = shard0.DataStartOffset;
            tensorInfoEnds[0] = shard0.TensorInfoEndOffset;
            var metadata = shard0.Metadata!;

            for (int s = 1; s < shardCount; s++)
            {
                var shard = ParseShard(basePtrs[s], fileSizes[s], fileSizes[s], shardIndex: s, shard0.Alignment, allTensors);
                dataStarts[s] = shard.DataStartOffset;
                tensorInfoEnds[s] = shard.TensorInfoEndOffset;
            }

            InjectDerivedMetadata(metadata, allTensors);

            // Report actual version from header
            var finalHeader = new GgufHeader(GgufMagic, shard0.Version, (ulong)allTensors.Count, shard0.KvCount);

            return new GgufModel(mmfs, accessors, basePtrs, fileSizes, dataStarts, tensorInfoEnds, finalHeader, metadata, allTensors);
        }
        catch
        {
            // Cleanup on failure
            for (int s = 0; s < shardCount; s++)
            {
                if (basePtrs[s] != null) accessors[s]?.SafeMemoryMappedViewHandle.ReleasePointer();
                accessors[s]?.Dispose();
                mmfs[s]?.Dispose();
            }
            throw;
        }
    }

    /// <summary>
    /// Returns a raw pointer to the tensor data in the memory-mapped file. Zero-copy.
    /// The pointer is valid for the lifetime of this GgufModel instance.
    /// </summary>
    public unsafe byte* GetTensorDataPtr(GgufTensorInfo tensor)
    {
        ResolveTensorRange(tensor, out byte* basePtr, out long absoluteOffset, out _);
        return basePtr + absoluteOffset;
    }

    /// <summary>Absolute byte offset (from the start of the file) where shard 0's tensor-data
    /// section begins -- i.e. everywhere the header, KV metadata, and tensor-info array end and
    /// raw tensor bytes start. Every <see cref="GgufTensorInfo.DataOffset"/> is relative to this.
    /// Exposed for GGUF-splicing tools that need to append a new tensor's bytes after the existing
    /// data blob without touching (or needing to understand) the KV metadata encoding at all.</summary>
    public long Shard0DataStartOffset => _shardDataStartOffsets[0];

    /// <summary>Absolute byte offset (from the start of the file) where shard 0's tensor-info array
    /// ends, BEFORE any alignment padding -- i.e. exactly where a new <see cref="GgufTensorInfo"/>
    /// entry can be inserted so it is read contiguously with the existing ones. A GGUF-splicing
    /// tool that copies [0, Shard0TensorInfoEndOffset) verbatim, appends one new tensor-info entry,
    /// re-pads to alignment, then appends the original data blob ([Shard0DataStartOffset, EOF))
    /// verbatim, produces a file that parses identically to the original plus the new tensor --
    /// without needing to understand the KV metadata value encoding at all.</summary>
    public long Shard0TensorInfoEndOffset => _shardTensorInfoEndOffsets[0];

    /// <summary>
    /// Returns a read-only span directly into the memory-mapped file for the given tensor. Zero-copy.
    /// </summary>
    public ReadOnlySpan<byte> GetTensorData(GgufTensorInfo tensor)
    {
        ResolveTensorRange(tensor, out byte* basePtr, out long absoluteOffset, out long byteSize);

        if (byteSize > int.MaxValue)
            throw new OverflowException(
                $"Tensor '{tensor.Name}' byte size ({byteSize} bytes) exceeds ReadOnlySpan max capacity ({int.MaxValue} bytes). Use GetTensorDataPtr for 2+ GiB tensors.");

        return new ReadOnlySpan<byte>(basePtr + absoluteOffset, (int)byteSize);
    }

    /// <summary>
    /// The single validated path from a <see cref="GgufTensorInfo"/> to raw memory. Both accessors go
    /// through it so neither can be the one that forgot a check — the pointer form previously had
    /// none at all, which made it the weaker of two doors into the same mapping.
    ///
    /// <para>Parse-time validation (see <c>ParseTensorInfos</c>) has already proved that every
    /// tensor belonging to this model lies inside its shard. What cannot be assumed here is that the
    /// caller passed a tensor from THIS model: <see cref="GgufTensorInfo"/> is a public record struct
    /// that anyone can construct or mutate with <c>with</c>, so the shard index and the resolved
    /// range are re-checked rather than trusted.</para>
    /// </summary>
    private void ResolveTensorRange(
        GgufTensorInfo tensor, out byte* basePtr, out long absoluteOffset, out long byteSize)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        int s = tensor.ShardIndex;
        if ((uint)s >= (uint)_shardBasePtrs.Length)
            throw new ArgumentOutOfRangeException(nameof(tensor),
                $"Tensor '{tensor.Name}' names shard {s}, but this model has {_shardBasePtrs.Length}.");

        if (tensor.DataOffset > long.MaxValue)
            throw new InvalidDataException(
                $"Tensor '{tensor.Name}' has data offset {tensor.DataOffset}, which exceeds the addressable range.");

        try
        {
            byteSize = tensor.ByteSize;
            absoluteOffset = checked(_shardDataStartOffsets[s] + (long)tensor.DataOffset);
            _ = checked(absoluteOffset + byteSize);
        }
        catch (OverflowException ex)
        {
            throw new InvalidDataException(
                $"Tensor '{tensor.Name}' has an offset/size that overflows when resolved against shard {s}.", ex);
        }

        if (byteSize < 0 || absoluteOffset < 0 || absoluteOffset + byteSize > _shardFileSizes[s])
            throw new InvalidDataException(
                $"Tensor '{tensor.Name}' data (offset={absoluteOffset}, size={byteSize}) is outside shard {s} " +
                $"(file size {_shardFileSizes[s]}).");

        basePtr = _shardBasePtrs[s];
    }

    /// <summary>
    /// Copies tensor data from the memory-mapped file into the destination buffer.
    /// </summary>
    public void LoadTensor(GgufTensorInfo tensor, Span<byte> destination)
    {
        var data = GetTensorData(tensor);
        if (destination.Length < data.Length)
            throw new ArgumentException(
                $"Destination buffer ({destination.Length} bytes) is smaller than tensor data ({data.Length} bytes).");
        data.CopyTo(destination);
    }

    /// <summary>Finds a tensor by name, or returns null if not found.</summary>
    public GgufTensorInfo? FindTensor(string name)
    {
        for (int i = 0; i < Tensors.Count; i++)
            if (Tensors[i].Name == name) return Tensors[i];
        return null;
    }

    /// <summary>Gets a metadata value by key, or returns the default if not found.</summary>
    public T GetMetadata<T>(string key, T defaultValue = default!) =>
        Metadata.TryGetValue(key, out var value) ? (T)Convert.ChangeType(value, typeof(T)) : defaultValue;

    /// <summary>
    /// Releases every shard mapping. Idempotent: a second call is a no-op rather than a second
    /// <c>ReleasePointer</c> on an already-released handle, which unbalances the refcount and can
    /// unmap memory other callers still hold pointers into. Double disposal is easy to reach because
    /// engine teardown paths transfer ownership of this object between owner lists.
    /// </summary>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        for (int s = 0; s < _shardAccessors.Length; s++)
        {
            _shardAccessors[s].SafeMemoryMappedViewHandle.ReleasePointer();
            _shardAccessors[s].Dispose();
            _shardMmfs[s].Dispose();
        }
    }

    // ── helpers ──────────────────────────────────────────────────────────────────

    /// <summary>
    /// Given a model path, returns the ordered list of shard file paths.
    /// Detects the *-00001-of-NNNNN.gguf naming convention used by llama.cpp split models.
    /// Falls back to [path] for single-file models.
    /// </summary>
    public static string[] ResolveShardPaths(string path)
    {
        var name = Path.GetFileName(path);
        // Match pattern like "Model-00001-of-00002.gguf"
        var m = System.Text.RegularExpressions.Regex.Match(
            name, @"^(.+)-(\d+)-of-(\d+)(\.gguf)$",
            System.Text.RegularExpressions.RegexOptions.IgnoreCase);

        if (!m.Success) return [path];

        string prefix   = m.Groups[1].Value;
        int    total    = int.Parse(m.Groups[3].Value);
        string ext      = m.Groups[4].Value;
        string dir      = Path.GetDirectoryName(path) ?? ".";

        var paths = new string[total];
        for (int i = 0; i < total; i++)
            paths[i] = Path.Combine(dir, $"{prefix}-{i + 1:D5}-of-{total:D5}{ext}");

        return paths;
    }

    private static uint ValidateMagicAndVersion(ref GgufBinaryReader reader)
    {
        var magic = reader.ReadUInt32();
        if (magic != GgufMagic)
            throw new InvalidDataException($"Invalid GGUF magic: 0x{magic:X8}");
        var version = reader.ReadUInt32();
        if (version is < 2 or > 3)
            throw new InvalidDataException($"Unsupported GGUF version: {version}");
        return version;
    }

    private readonly record struct ShardParse(
        uint Version, ulong TensorCount, ulong KvCount, Dictionary<string, object>? Metadata,
        int Alignment, long DataStartOffset, long TensorInfoEndOffset);

    /// <summary>
    /// Parses one shard's header, metadata (kept for shard 0, skipped otherwise: later shards repeat it) and tensor-info array from a buffer.
    /// <paramref name="bufferLength"/> is how many bytes are readable; <paramref name="fileSize"/> is the real size of the shard, which is
    /// what tensor offsets are validated against. They are equal for a mapped file and differ for a downloaded prefix.
    /// </summary>
    private static ShardParse ParseShard(byte* ptr, long bufferLength, long fileSize, int shardIndex, int alignmentFromShard0, List<GgufTensorInfo> tensors)
    {
        var reader = new GgufBinaryReader(ptr, bufferLength);
        uint version = ValidateMagicAndVersion(ref reader);
        var tensorCount = reader.ReadUInt64();
        var kvCount = reader.ReadUInt64();
        // Counts come from the file. Bound them by the bytes that could possibly hold that many entries before any
        // allocation is sized from them, so a hostile header cannot request billions of entries.
        reader.RequirePlausibleCount(kvCount, minBytesPerEntry: 13, "metadata key/value pairs");
        reader.RequirePlausibleCount(tensorCount, minBytesPerEntry: 24, "tensors");

        Dictionary<string, object>? metadata = null;
        int alignment = alignmentFromShard0;
        if (shardIndex == 0)
        {
            metadata = new Dictionary<string, object>((int)kvCount);
            for (ulong i = 0; i < kvCount; i++)
            {
                var key       = reader.ReadGgufString();
                var valueType = (GgufValueType)reader.ReadUInt32();
                var value     = reader.ReadGgufValue(valueType);
                metadata[key] = value;
            }
            alignment = DefaultAlignment;
            if (metadata.TryGetValue("general.alignment", out var alignObj))
                alignment = Convert.ToInt32(alignObj);
        }
        else
        {
            // Skip metadata entries (present but redundant in shards > 0)
            for (ulong i = 0; i < kvCount; i++)
            {
                reader.ReadGgufString();                         // key
                var vt = (GgufValueType)reader.ReadUInt32();
                reader.SkipGgufValue(vt);                        // value
            }
        }

        ParseTensorInfos(ref reader, tensorCount, alignment, shardIndex, fileSize, out long dataStart, out long infoEnd, tensors);
        return new ShardParse(version, tensorCount, kvCount, metadata, alignment, dataStart, infoEnd);
    }

    /// <summary>Synthetic metadata derived from tensor inspection. Shared by the file loader and <see cref="ParseIndex"/> so both see the same facts.</summary>
    private static void InjectDerivedMetadata(Dictionary<string, object> metadata, List<GgufTensorInfo> allTensors)
    {
            var arch = metadata.TryGetValue("general.architecture", out var archVal) ? (string)archVal : "llama";
            int attnKCount = 0, attnVCount = 0;
            foreach (var t in allTensors)
            {
                if (t.Name == "blk.0.attn_q.bias")     metadata["_opentailllm.has_attn_bias"] = true;
                // Qwen2 carries bias on Q/K/V but NOT on the output projection; Qwen-1 and
                // some others carry it on the output too. Probe attn_output.bias separately so
                // the output bias is loaded only when it actually exists (llama.cpp treats `bo`
                // as optional). Without this, a Qwen2 GGUF fails to load: HasAttnBias is set
                // from attn_q.bias but the loaders then demand a non-existent attn_output.bias.
                else if (t.Name == "blk.0.attn_output.bias") metadata["_opentailllm.has_attn_output_bias"] = true;
                else if (t.Name == "blk.0.attn_q_norm.weight") metadata["_opentailllm.has_qk_norm"] = true;
                else if (t.Name == "per_layer_token_embd.weight") metadata["_opentailllm.has_ple"] = true;
                else if (t.Name == "blk.0.post_attention_norm.weight") metadata["_opentailllm.has_post_attn_norm"] = true;
                else if (t.Name == "blk.0.post_ffw_norm.weight") metadata["_opentailllm.has_post_ffw_norm"] = true;
                else if (t.Name == "blk.0.layer_output_scale.weight") metadata["_opentailllm.has_layer_output_scale"] = true;
                // Gated-DeltaNet recurrent blocks (qwen35moe and similar hybrids). The
                // first few layers' indices depend on full_attention_interval, so probe
                // a window of low layer indices for ssm_conv1d.weight.
                else if (t.Name is "blk.0.ssm_conv1d.weight"
                                 or "blk.1.ssm_conv1d.weight"
                                 or "blk.2.ssm_conv1d.weight"
                                 or "blk.3.ssm_conv1d.weight")
                    metadata["_opentailllm.is_hybrid_ssm"] = true;

                // Gemma 4 12B global layers omit attn_v and reuse K as V
                // (attention_k_eq_v). Detect by a V-projection deficit vs K.
                if (t.Name.EndsWith(".attn_k.weight", StringComparison.Ordinal))      attnKCount++;
                else if (t.Name.EndsWith(".attn_v.weight", StringComparison.Ordinal)) attnVCount++;
            }
            // k_eq_v is a Gemma-4-specific mechanism (its global layers drop attn_v and reuse
            // the K projection as V). Gate the V-deficit heuristic on the gemma4 arch family so
            // an unrelated architecture that ships fewer V than K projections can't be mis-driven
            // down the copy-K-into-V path. Heuristic, not a metadata read — no GGUF declares
            // attention_k_eq_v directly.
            if (attnKCount > 0 && attnVCount < attnKCount
                && arch.StartsWith("gemma4", StringComparison.Ordinal))
                metadata["_opentailllm.attention_k_eq_v"] = true;

            if (!metadata.ContainsKey($"{arch}.vocab_size") &&
                metadata.TryGetValue("tokenizer.ggml.tokens", out var tokArr) && tokArr is object[] toks)
                metadata[$"{arch}.vocab_size"] = (ulong)toks.Length;
    }

    /// <summary>
    /// Parses the header, metadata and tensor-info arrays from the START of a GGUF (or of each shard of a split one) without the tensor data.
    /// Used to inspect a model remotely from a few MB read by HTTP Range request. <paramref name="shardPrefixes"/> holds one entry per shard
    /// in order: the bytes read from the start of that shard and the shard's full size (needed to validate tensor offsets).
    /// Throws an <see cref="InvalidDataException"/> that <see cref="GgufTruncation.IsTruncation"/> recognises (with the shard number) when a prefix ends inside the index, so a caller can read more;
    /// any other <see cref="InvalidDataException"/> means the file is malformed.
    /// </summary>
    public static GgufIndex ParseIndex(IReadOnlyList<(ReadOnlyMemory<byte> Prefix, long FileSize)> shardPrefixes)
    {
        ArgumentNullException.ThrowIfNull(shardPrefixes);
        if (shardPrefixes.Count == 0) throw new ArgumentException("At least one shard prefix is required.", nameof(shardPrefixes));

        var tensors = new List<GgufTensorInfo>();
        ShardParse? first = null;
        var ends = new long[shardPrefixes.Count];
        for (int s = 0; s < shardPrefixes.Count; s++)
        {
            var (prefix, fileSize) = shardPrefixes[s];
            try
            {
                using var pin = prefix.Pin();
                var parsed = ParseShard((byte*)pin.Pointer, prefix.Length, fileSize, s, first?.Alignment ?? DefaultAlignment, tensors);
                first ??= parsed;
                ends[s] = parsed.TensorInfoEndOffset;
            }
            catch (InvalidDataException ex) when (GgufTruncation.IsTruncation(ex, out _))
            {
                throw GgufTruncation.Create(s, ex.Message);
            }
        }

        var metadata = first!.Value.Metadata!;
        InjectDerivedMetadata(metadata, tensors);
        return new GgufIndex(new GgufHeader(GgufMagic, first.Value.Version, (ulong)tensors.Count, first.Value.KvCount), metadata, tensors, ends);
    }

    private static void ParseTensorInfos(
        ref GgufBinaryReader reader,
        ulong tensorCount,
        int alignment,
        int shardIndex,
        long fileSize,
        out long dataStartOffset,
        out long tensorInfoEndOffset,
        List<GgufTensorInfo> result)
    {
        var tensors = new GgufTensorInfo[(int)tensorCount];
        for (ulong i = 0; i < tensorCount; i++)
        {
            var tName  = reader.ReadGgufString();
            var nDims  = reader.ReadUInt32();
            var dims   = new long[nDims];
            for (uint d = 0; d < nDims; d++)
                dims[d] = (long)reader.ReadUInt64();
            uint rawType = reader.ReadUInt32();
            if (!DTypeInfo.TryFromGgufType(rawType, out var dtype))
                throw GgufUnsupported.Create(rawType,
                    $"Tensor '{tName}' uses unsupported GGML type ID {rawType}. " +
                    "This OpenTail.Stingray build follows current llama.cpp GGUF type IDs and rejects " +
                    "retired or newer tensor formats before model load.");
            var offset = reader.ReadUInt64();

            // Validate the geometry here, at the only place the raw header is still in scope.
            // GGML's own limit is 4 dimensions; a larger rank means the file is malformed (or
            // hostile) and every later size computation derived from it would be meaningless.
            if (nDims > 8)
                throw new InvalidDataException(
                    $"Tensor '{tName}' declares {nDims} dimensions; GGUF supports at most 8.");
            for (uint d = 0; d < nDims; d++)
                if (dims[d] <= 0)
                    throw new InvalidDataException(
                        $"Tensor '{tName}' has non-positive dimension {d} ({dims[d]}).");

            tensors[i] = new GgufTensorInfo(tName, (int)nDims, dims, dtype, offset, shardIndex);
        }

        tensorInfoEndOffset = reader.Position;
        dataStartOffset = AlignUp(reader.Position, alignment);

        // Second pass: every tensor's byte range must lie inside this shard. dataStartOffset is only
        // known once the whole info block has been read, so this cannot fold into the loop above.
        //
        // Doing it HERE, once, is what lets the accessors stay cheap: after this, a GgufTensorInfo
        // that came from this file is known to describe a range inside it, so GetTensorData and
        // GetTensorDataPtr re-check only what they cannot assume (that the caller passed a tensor
        // from this model at all).
        foreach (var tensor in tensors)
        {
            long byteSize;
            try
            {
                byteSize = tensor.ByteSize; // checked; throws on a wrapped element count
            }
            catch (OverflowException ex)
            {
                throw new InvalidDataException(
                    $"Tensor '{tensor.Name}' has dimensions whose product overflows a 64-bit size.", ex);
            }

            if (byteSize < 0)
                throw new InvalidDataException($"Tensor '{tensor.Name}' has a negative byte size ({byteSize}).");
            // DataOffset is unsigned on the wire but indexed as a signed offset. Reject anything that
            // would not survive the conversion BEFORE casting, rather than casting to a negative
            // number that then compares as "below the file size" and passes.
            if (tensor.DataOffset > long.MaxValue)
                throw new InvalidDataException(
                    $"Tensor '{tensor.Name}' has data offset {tensor.DataOffset}, which exceeds the addressable range.");

            long absolute;
            try
            {
                absolute = checked(dataStartOffset + (long)tensor.DataOffset);
                if (checked(absolute + byteSize) > fileSize)
                    throw new InvalidDataException(
                        $"Tensor '{tensor.Name}' data (offset={absolute}, size={byteSize}) exceeds shard " +
                        $"{shardIndex} file size ({fileSize}).");
            }
            catch (OverflowException ex)
            {
                throw new InvalidDataException(
                    $"Tensor '{tensor.Name}' data offset/size overflows when resolved against shard {shardIndex}.", ex);
            }
        }

        result.AddRange(tensors);
    }

    private static long AlignUp(long value, int alignment) =>
        (value + alignment - 1) / alignment * alignment;

    /// <summary>
    /// Pointer-based reader for parsing GGUF binary data from a memory-mapped region.
    /// </summary>
    private ref struct GgufBinaryReader
    {
        private readonly byte* _base;
        private readonly long _length;
        private long _pos;

        public GgufBinaryReader(byte* basePtr, long length)
        {
            _base = basePtr;
            _length = length;
            _pos = 0;
        }

        public long Position => _pos;

        public byte ReadByte()
        {
            EnsureAvailable(1);
            return _base[_pos++];
        }

        public uint ReadUInt32()
        {
            EnsureAvailable(4);
            var value = BinaryPrimitives.ReadUInt32LittleEndian(new ReadOnlySpan<byte>(_base + _pos, 4));
            _pos += 4;
            return value;
        }

        public int ReadInt32()
        {
            EnsureAvailable(4);
            var value = BinaryPrimitives.ReadInt32LittleEndian(new ReadOnlySpan<byte>(_base + _pos, 4));
            _pos += 4;
            return value;
        }

        public ulong ReadUInt64()
        {
            EnsureAvailable(8);
            var value = BinaryPrimitives.ReadUInt64LittleEndian(new ReadOnlySpan<byte>(_base + _pos, 8));
            _pos += 8;
            return value;
        }

        public long ReadInt64()
        {
            EnsureAvailable(8);
            var value = BinaryPrimitives.ReadInt64LittleEndian(new ReadOnlySpan<byte>(_base + _pos, 8));
            _pos += 8;
            return value;
        }

        public ushort ReadUInt16()
        {
            EnsureAvailable(2);
            var value = BinaryPrimitives.ReadUInt16LittleEndian(new ReadOnlySpan<byte>(_base + _pos, 2));
            _pos += 2;
            return value;
        }

        public short ReadInt16()
        {
            EnsureAvailable(2);
            var value = BinaryPrimitives.ReadInt16LittleEndian(new ReadOnlySpan<byte>(_base + _pos, 2));
            _pos += 2;
            return value;
        }

        public float ReadFloat32()
        {
            EnsureAvailable(4);
            var value = BinaryPrimitives.ReadSingleLittleEndian(new ReadOnlySpan<byte>(_base + _pos, 4));
            _pos += 4;
            return value;
        }

        public double ReadFloat64()
        {
            EnsureAvailable(8);
            var value = BinaryPrimitives.ReadDoubleLittleEndian(new ReadOnlySpan<byte>(_base + _pos, 8));
            _pos += 8;
            return value;
        }

        public bool ReadBool() => ReadByte() != 0;

        /// <summary>
        /// Reads a GGUF string: uint64 length + UTF-8 bytes (no null terminator).
        /// </summary>
        public string ReadGgufString()
        {
            var length = ReadUInt64();
            if (length > int.MaxValue)
                throw new InvalidDataException($"GGUF string length {length} exceeds maximum.");
            var len = (int)length;
            EnsureAvailable(len);
            var value = Encoding.UTF8.GetString(new ReadOnlySpan<byte>(_base + _pos, len));
            _pos += len;
            return value;
        }

        /// <summary>
        /// Reads a typed GGUF metadata value.
        /// </summary>
        public object ReadGgufValue(GgufValueType type) => type switch
        {
            GgufValueType.UInt8   => ReadByte(),
            GgufValueType.Int8    => (sbyte)ReadByte(),
            GgufValueType.UInt16  => ReadUInt16(),
            GgufValueType.Int16   => ReadInt16(),
            GgufValueType.UInt32  => ReadUInt32(),
            GgufValueType.Int32   => ReadInt32(),
            GgufValueType.Float32 => ReadFloat32(),
            GgufValueType.Bool    => ReadBool(),
            GgufValueType.String  => ReadGgufString(),
            GgufValueType.UInt64  => ReadUInt64(),
            GgufValueType.Int64   => ReadInt64(),
            GgufValueType.Float64 => ReadFloat64(),
            GgufValueType.Array   => ReadGgufArray(),
            _ => throw new InvalidDataException($"Unknown GGUF value type: {type}")
        };

        private object[] ReadGgufArray()
        {
            var elementType = (GgufValueType)ReadUInt32();
            var count = ReadUInt64();
            if (count > int.MaxValue)
                throw new InvalidDataException($"GGUF array length {count} exceeds maximum.");
            // Every element occupies at least one byte, so a count above the bytes left cannot be honest; refuse before sizing the array from it.
            RequirePlausibleCount(count, minBytesPerEntry: 1, "array elements");
            var array = new object[(int)count];
            for (ulong i = 0; i < count; i++)
                array[i] = ReadGgufValue(elementType);
            return array;
        }

        /// <summary>
        /// Skips over a GGUF metadata value without allocating. Used when parsing redundant
        /// metadata in shard files 1+.
        /// </summary>
        public void SkipGgufValue(GgufValueType type)
        {
            switch (type)
            {
                case GgufValueType.UInt8:
                case GgufValueType.Int8:
                case GgufValueType.Bool:  SkipBytes(1); break;
                case GgufValueType.UInt16:
                case GgufValueType.Int16: SkipBytes(2); break;
                case GgufValueType.UInt32:
                case GgufValueType.Int32:
                case GgufValueType.Float32: SkipBytes(4); break;
                case GgufValueType.UInt64:
                case GgufValueType.Int64:
                case GgufValueType.Float64: SkipBytes(8); break;
                case GgufValueType.String:
                {
                    var len = ReadUInt64();
                    // Reject before casting: a length above long.MaxValue casts to a NEGATIVE
                    // advance, which moves the position backwards instead of failing.
                    if (len > long.MaxValue)
                        throw new InvalidDataException($"GGUF string length {len} exceeds the addressable range.");
                    SkipBytes((long)len);
                    break;
                }
                case GgufValueType.Array:
                {
                    var elemType = (GgufValueType)ReadUInt32();
                    var count    = ReadUInt64();
                    for (ulong i = 0; i < count; i++)
                        SkipGgufValue(elemType);
                    break;
                }
                default:
                    throw new InvalidDataException($"Unknown GGUF value type to skip: {type}");
            }
        }

        /// <summary>
        /// Bounds check for the next <paramref name="bytes"/> bytes.
        ///
        /// <para>The negative/overflow cases are not theoretical. Lengths on the wire are
        /// <c>ulong</c> and get cast to <c>long</c> to advance <c>_pos</c>, so a hostile length can
        /// drive the position negative or wrap it. A plain <c>_pos + bytes &gt; _length</c> test
        /// silently PASSES for a negative position and would then read below the mapping.</para>
        /// </summary>
        /// <summary>Throws when <paramref name="count"/> entries of at least <paramref name="minBytesPerEntry"/> bytes could not fit in what is left of the buffer.</summary>
        public void RequirePlausibleCount(ulong count, int minBytesPerEntry, string what)
        {
            long remaining = _length - _pos;
            if (remaining < 0 || count > (ulong)remaining / (ulong)minBytesPerEntry)
                throw GgufTruncation.Create(0,
                    $"Unexpected end of GGUF data at offset {_pos}: {count} {what} declared but only {Math.Max(remaining, 0)} bytes remain.");
        }

        private void EnsureAvailable(long bytes)
        {
            if (bytes < 0 || _pos < 0 || _pos > _length || _pos + bytes < 0 || _pos + bytes > _length)
                throw GgufTruncation.Create(0,
                    $"Unexpected end of GGUF data at offset {_pos} (need {bytes} bytes, {_length - _pos} available).");
        }

        /// <summary>
        /// Advances the position by <paramref name="bytes"/>, validating the result. Skipping is a
        /// read that discards its result, so it needs the same bounds check a read would get —
        /// without it a skip can park <c>_pos</c> out of range or negative, and the NEXT read's
        /// check then evaluates against a nonsense position.
        /// </summary>
        private void SkipBytes(long bytes)
        {
            EnsureAvailable(bytes);
            _pos += bytes;
        }
    }
}
