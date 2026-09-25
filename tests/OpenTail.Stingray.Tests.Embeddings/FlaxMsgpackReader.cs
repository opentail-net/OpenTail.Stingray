using System.Buffers.Binary;

namespace OpenTail.Stingray.Tests.Embeddings;

/// <summary>
/// Minimal msgpack reader for <c>flax.serialization.msgpack_serialize</c> checkpoints (<c>flax_model.msgpack</c>),
/// used as a test oracle: nested maps of parameters whose leaves are numpy arrays stored as msgpack ext type 1
/// holding a packed <c>(shape, dtype_name, raw_bytes)</c> tuple. Only little-endian float32 leaves are decoded;
/// chunked (&gt;1 GB) arrays are not supported.
/// </summary>
internal sealed class FlaxMsgpackReader
{
    private readonly byte[] _b;
    private int _p;

    private FlaxMsgpackReader(byte[] bytes) { _b = bytes; }

    /// <summary>Flattened "a/b/c" path → (shape, float32 values).</summary>
    public static Dictionary<string, (int[] Shape, float[] Data)> ReadParams(string path)
    {
        var r = new FlaxMsgpackReader(File.ReadAllBytes(path));
        var result = new Dictionary<string, (int[], float[])>(StringComparer.Ordinal);
        Flatten(r.Read(), "", result);
        return result;
    }

    private static void Flatten(object? node, string prefix, Dictionary<string, (int[], float[])> into)
    {
        switch (node)
        {
            case Dictionary<string, object?> map:
                foreach (var (k, v) in map) Flatten(v, prefix.Length == 0 ? k : prefix + "/" + k, into);
                break;
            case (int[] shape, float[] data):
                into[prefix] = (shape, data);
                break;
        }
    }

    private object? Read()
    {
        byte t = _b[_p++];
        if (t <= 0x7f) return (long)t;
        if (t >= 0xe0) return (long)(sbyte)t;
        if ((t & 0xf0) == 0x80) return ReadMap(t & 0x0f);
        if ((t & 0xf0) == 0x90) return ReadArray(t & 0x0f);
        if ((t & 0xe0) == 0xa0) return ReadStr(t & 0x1f);
        return t switch
        {
            0xc0 => null,
            0xc2 => false,
            0xc3 => true,
            0xc4 => ReadBin(_b[_p++]),
            0xc5 => ReadBin(U16()),
            0xc6 => ReadBin((int)U32()),
            0xc7 => ReadExt(_b[_p++]),
            0xc8 => ReadExt(U16()),
            0xc9 => ReadExt((int)U32()),
            0xca => (double)BinaryPrimitives.ReadSingleBigEndian(Take(4)),
            0xcb => BinaryPrimitives.ReadDoubleBigEndian(Take(8)),
            0xcc => (long)_b[_p++],
            0xcd => (long)U16(),
            0xce => (long)U32(),
            0xcf => (long)BinaryPrimitives.ReadUInt64BigEndian(Take(8)),
            0xd0 => (long)(sbyte)_b[_p++],
            0xd1 => (long)BinaryPrimitives.ReadInt16BigEndian(Take(2)),
            0xd2 => (long)BinaryPrimitives.ReadInt32BigEndian(Take(4)),
            0xd3 => BinaryPrimitives.ReadInt64BigEndian(Take(8)),
            0xd4 => ReadExt(1),
            0xd5 => ReadExt(2),
            0xd6 => ReadExt(4),
            0xd7 => ReadExt(8),
            0xd8 => ReadExt(16),
            0xd9 => ReadStr(_b[_p++]),
            0xda => ReadStr(U16()),
            0xdb => ReadStr((int)U32()),
            0xdc => ReadArray(U16()),
            0xdd => ReadArray((int)U32()),
            0xde => ReadMap(U16()),
            0xdf => ReadMap((int)U32()),
            _ => throw new InvalidDataException($"msgpack type 0x{t:x2} at {_p - 1}"),
        };
    }

    private ReadOnlySpan<byte> Take(int n) { var s = _b.AsSpan(_p, n); _p += n; return s; }
    private int U16() => BinaryPrimitives.ReadUInt16BigEndian(Take(2));
    private uint U32() => BinaryPrimitives.ReadUInt32BigEndian(Take(4));
    private string ReadStr(int n) => Encoding.UTF8.GetString(Take(n));
    private byte[] ReadBin(int n) => Take(n).ToArray();

    private object?[] ReadArray(int n)
    {
        var a = new object?[n];
        for (int i = 0; i < n; i++) a[i] = Read();
        return a;
    }

    private Dictionary<string, object?> ReadMap(int n)
    {
        var m = new Dictionary<string, object?>(n, StringComparer.Ordinal);
        for (int i = 0; i < n; i++)
        {
            var key = Read();
            m[Convert.ToString(key, System.Globalization.CultureInfo.InvariantCulture) ?? ""] = Read();
        }
        return m;
    }

    private object? ReadExt(int n)
    {
        sbyte code = (sbyte)_b[_p++];
        byte[] payload = Take(n).ToArray();
        if (code != 1) return null; // only ndarray leaves matter here
        var inner = new FlaxMsgpackReader(payload);
        var tuple = (object?[])inner.Read()!;
        var shape = ((object?[])tuple[0]!).Select(v => Convert.ToInt32(v, System.Globalization.CultureInfo.InvariantCulture)).ToArray();
        string dtype = (string)tuple[1]!;
        var raw = (byte[])tuple[2]!;
        if (dtype != "float32") return null;
        var data = new float[raw.Length / 4];
        Buffer.BlockCopy(raw, 0, data, 0, raw.Length);
        return (shape, data);
    }
}
