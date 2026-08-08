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
