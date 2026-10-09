namespace OpenTail.Stingray.Core;

/// <summary>
/// GGUF metadata value type tags, matching the GGUF spec.
/// </summary>
public enum GgufValueType : uint
{
    UInt8   = 0,
    Int8    = 1,
    UInt16  = 2,
    Int16   = 3,
    UInt32  = 4,
    Int32   = 5,
    Float32 = 6,
    Bool    = 7,
    String  = 8,
    Array   = 9,
    UInt64  = 10,
    Int64   = 11,
    Float64 = 12,
}

/// <summary>
/// Parsed GGUF file header.
/// </summary>
public readonly record struct GgufHeader(
    uint Magic,
    uint Version,
    ulong TensorCount,
    ulong MetadataKvCount);

/// <summary>
/// Descriptor for a single tensor in a GGUF file.
/// </summary>
public readonly record struct GgufTensorInfo(
    string Name,
    int NDimensions,
    long[] Dimensions,
    DType DType,
    ulong DataOffset,
    int ShardIndex = 0)
{
    /// <summary>
    /// Total number of elements in the tensor.
    ///
    /// <para>Multiplication is <c>checked</c> and the dimension count is clamped to the array that
    /// was actually read. A malformed header can otherwise wrap the product to a small or negative
    /// value that then passes every downstream size check while the tensor's real extent is
    /// enormous — the bounds check would be validating a number that bears no relation to the
    /// bytes about to be read. Overflow has to be an error, not a smaller number.</para>
    /// </summary>
    public long ElementCount
    {
        get
        {
            if (Dimensions.Length == 0) return 0;
            long count = 1;
            int rank = Math.Min(NDimensions, Dimensions.Length);
            checked
            {
                for (int i = 0; i < rank; i++)
                    count *= Dimensions[i];
            }
            return count;
        }
    }

    /// <summary>Total byte size of the tensor data.</summary>
    public long ByteSize => DTypeInfo.ByteSize(ElementCount, DType);
}

/// <summary>
/// The index of a GGUF: header, metadata and every tensor's name, shape and type, without any tensor data. Produced by
/// <see cref="GgufModel.ParseIndex"/> from the start of each shard, so a model can be inspected from a few MB read remotely.
/// </summary>
public sealed record GgufIndex(
    GgufHeader Header,
    IReadOnlyDictionary<string, object> Metadata,
    IReadOnlyList<GgufTensorInfo> Tensors,
    // Offset in each shard where its tensor-info array ends, i.e. how many bytes of the shard the index occupies.
    IReadOnlyList<long> TensorInfoEndOffsets);

/// <summary>
/// Marks and reads "the buffer ended inside the index". The loader always threw <see cref="InvalidDataException"/> for this and callers catch
/// that type (which is sealed, so it cannot be subclassed); the marker lives in <see cref="Exception.Data"/> so those callers are unaffected, while a
/// caller reading a prefix can tell "read more" from "malformed" and learn which shard ran out.
/// </summary>
public static class GgufTruncation
{
    private const string Key = "opentail.gguf.truncated_shard";

    public static InvalidDataException Create(int shard, string message)
    {
        var ex = new InvalidDataException(message);
        ex.Data[Key] = shard;
        return ex;
    }

    public static bool IsTruncation(Exception ex, out int shard)
    {
        if (ex is InvalidDataException && ex.Data[Key] is int s) { shard = s; return true; }
        shard = -1;
        return false;
    }
}

/// <summary>
/// Marks and reads "this file uses a tensor storage type this build does not know". Same mechanism as <see cref="GgufTruncation"/>: the loader keeps throwing
/// the sealed <see cref="InvalidDataException"/>, and the type id travels in <see cref="Exception.Data"/> so a caller can tell an unsupported FORMAT
/// (a verdict about the checkpoint) from a corrupt FILE.
/// </summary>
public static class GgufUnsupported
{
    private const string Key = "opentail.gguf.unsupported_type";

    public static InvalidDataException Create(uint typeId, string message)
    {
        var ex = new InvalidDataException(message);
        ex.Data[Key] = typeId;
        return ex;
    }

    public static bool IsUnsupportedType(Exception ex, out uint typeId)
    {
        if (ex is InvalidDataException && ex.Data[Key] is uint t) { typeId = t; return true; }
        typeId = 0;
        return false;
    }
}