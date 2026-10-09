namespace OpenTail.Stingray.Tests.Core;

/// <summary>Builds small valid GGUF byte arrays for tests that must always run (no model file needed).</summary>
internal static class SyntheticGguf
{
    internal sealed record T(string Name, long[] Dims, uint Type = 0 /* F32 */, ulong Offset = 0);
    internal delegate void ValueWriter(BinaryWriter w);

    internal static void Str(BinaryWriter w, string s) { var b = Encoding.UTF8.GetBytes(s); w.Write((ulong)b.Length); w.Write(b); }

    /// <summary>Header, metadata (written in order), tensor infos, padding to 32, then <paramref name="dataBytes"/> of zeros.</summary>
    internal static byte[] Build(uint version, IEnumerable<(string Key, uint Type, ValueWriter Value)> kvs, IEnumerable<T> tensors, int dataBytes = 4096)
    {
        var kvList = kvs.ToList(); var tList = tensors.ToList();
        using var ms = new MemoryStream(); using var w = new BinaryWriter(ms);
        w.Write(0x46554747u); w.Write(version); w.Write((ulong)tList.Count); w.Write((ulong)kvList.Count);
        foreach (var (key, type, value) in kvList) { Str(w, key); w.Write(type); value(w); }
        foreach (var t in tList) { Str(w, t.Name); w.Write((uint)t.Dims.Length); foreach (var d in t.Dims) w.Write((ulong)d); w.Write(t.Type); w.Write(t.Offset); }
        while (ms.Length % 32 != 0) w.Write((byte)0);
        w.Write(new byte[dataBytes]);
        return ms.ToArray();
    }

    internal static (string, uint, ValueWriter) KvStr(string k, string v) => (k, 8, w => Str(w, v));
    internal static (string, uint, ValueWriter) KvU32(string k, uint v) => (k, 4, w => w.Write(v));
    internal static (string, uint, ValueWriter) KvStrArray(string k, params string[] items) =>
        (k, 9, w => { w.Write(8u); w.Write((ulong)items.Length); foreach (var s in items) Str(w, s); });

    /// <summary>An array of <paramref name="count"/> strings of <paramref name="each"/> chars: makes the metadata (and so the index) as large as a test needs.</summary>
    internal static (string, uint, ValueWriter) KvBigStrArray(string k, int count, int each) =>
        (k, 9, w => { w.Write(8u); w.Write((ulong)count); var s = new string('x', each); for (int i = 0; i < count; i++) Str(w, s); });

    internal static byte[] Sample(uint version = 3) => Build(version,
    [
        KvStr("general.architecture", "llama"), KvU32("llama.block_count", 2),
        KvStrArray("tokenizer.ggml.tokens", "a", "b", "c"),
    ],
    [
        new("token_embd.weight", [8, 3], 0, 0), new("blk.0.attn_q.weight", [8, 8], 0, 96), new("blk.0.attn_q_norm.weight", [8], 0, 352),
        new("blk.1.attn_q.weight", [8, 8], 1, 384),
    ], dataBytes: 1024);
}
