namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// <see cref="GgufModel.ParseIndex"/> reads a model's index from the START of its file(s), so it can be inspected from a few MB fetched by HTTP Range.
/// It shares its parser with <see cref="GgufModel.Open"/>; these tests pin that the two agree and that a short or hostile buffer is handled safely.
/// Synthetic GGUFs only, so they always run (a test that silently needs a model file proves nothing).
/// </summary>
public sealed class GgufIndexTests : IDisposable
{
    private readonly List<string> _files = [];
    public void Dispose() { foreach (var f in _files) try { File.Delete(f); } catch { } }

    private static byte[] Sample(uint version = 3) => SyntheticGguf.Sample(version);
    private static byte[] Build(uint version, IEnumerable<(string Key, uint Type, SyntheticGguf.ValueWriter Value)> kvs, IEnumerable<SyntheticGguf.T> tensors, int dataBytes = 4096) =>
        SyntheticGguf.Build(version, kvs, tensors, dataBytes);
    private static (string, uint, SyntheticGguf.ValueWriter) KvStr(string k, string v) => SyntheticGguf.KvStr(k, v);
    private static (string, uint, SyntheticGguf.ValueWriter) KvU32(string k, uint v) => SyntheticGguf.KvU32(k, v);
    private static SyntheticGguf.T T(string name, long[] dims, uint type = 0, ulong offset = 0) => new(name, dims, type, offset);
    private string Write(byte[] bytes) { string p = Path.Combine(Path.GetTempPath(), "gguf-index-" + Guid.NewGuid().ToString("N") + ".gguf"); File.WriteAllBytes(p, bytes); _files.Add(p); return p; }

    private static int IndexEnd(string path) { using var m = GgufModel.Open(path); return (int)m.Shard0TensorInfoEndOffset; }

    private static void AssertSame(GgufModel model, GgufIndex index)
    {
        Assert.Equal(model.Header, index.Header);
        Assert.Equal(model.Metadata.Keys.Order(), index.Metadata.Keys.Order());
        foreach (var (k, v) in model.Metadata)
        {
            var other = index.Metadata[k];
            if (v is object[] a) Assert.Equal(a, (object[])other); else Assert.Equal(v, other);
        }
        Assert.Equal(model.Tensors.Count, index.Tensors.Count);
        for (int i = 0; i < model.Tensors.Count; i++)
        {
            var a = model.Tensors[i]; var b = index.Tensors[i];
            Assert.Equal((a.Name, a.NDimensions, a.DType, a.DataOffset, a.ShardIndex), (b.Name, b.NDimensions, b.DType, b.DataOffset, b.ShardIndex));
            Assert.Equal(a.Dimensions, b.Dimensions);
        }
        Assert.Equal(model.Shard0TensorInfoEndOffset, index.TensorInfoEndOffsets[0]);
    }

    // ── agreement with the file loader ──
    [Fact]
    public void A_prefix_ending_at_the_index_gives_exactly_what_Open_gives()
    {
        byte[] bytes = Sample(); string path = Write(bytes);
        using var model = GgufModel.Open(path);
        var index = GgufModel.ParseIndex([(bytes.AsMemory(0, IndexEnd(path)), bytes.Length)]);
        AssertSame(model, index);
        Assert.True(index.Metadata.ContainsKey("_opentailllm.has_qk_norm")); // derived metadata is shared
        Assert.Equal(3UL, index.Metadata["llama.vocab_size"]);
    }

    [Fact]
    public void A_longer_prefix_or_the_whole_file_gives_the_same_answer()
    {
        byte[] bytes = Sample(); string path = Write(bytes); using var model = GgufModel.Open(path);
        AssertSame(model, GgufModel.ParseIndex([(bytes, bytes.Length)]));
        AssertSame(model, GgufModel.ParseIndex([(bytes.AsMemory(0, IndexEnd(path) + 100), bytes.Length)]));
    }

    [Fact]
    public void The_real_header_version_is_reported_for_a_v2_file()
    {
        byte[] bytes = Sample(version: 2); string path = Write(bytes);
        using var model = GgufModel.Open(path);
        Assert.Equal(2u, model.Header.Version);
        Assert.Equal(2u, GgufModel.ParseIndex([(bytes, bytes.Length)]).Header.Version);
    }

    // ── truncation: every cut inside the index is "read more", and nothing else is ──
    [Fact]
    public void Every_cut_short_of_the_index_end_is_reported_as_truncation()
    {
        byte[] bytes = Sample(); int end = IndexEnd(Write(bytes));
        for (int cut = 0; cut < end; cut++)
        {
            var ex = Assert.ThrowsAny<InvalidDataException>(() => GgufModel.ParseIndex([(bytes.AsMemory(0, cut), bytes.Length)]));
            Assert.True(GgufTruncation.IsTruncation(ex, out int shard), $"cut at {cut} of {end} was not recognised as truncation: {ex.Message}");
            Assert.Equal(0, shard);
        }
    }

    [Fact]
    public void Open_still_throws_InvalidDataException_for_a_truncated_local_file()
    {
        byte[] bytes = Sample(); int end = IndexEnd(Write(bytes));
        string cutPath = Write(bytes[..(end - 10)]);
        Assert.Throws<InvalidDataException>(() => GgufModel.Open(cutPath));
    }

    [Fact]
    public void Malformed_input_is_not_mistaken_for_truncation()
    {
        byte[] bad = Sample(); bad[0] = (byte)'X';
        var ex = Assert.Throws<InvalidDataException>(() => GgufModel.ParseIndex([(bad, bad.Length)]));
        Assert.False(GgufTruncation.IsTruncation(ex, out _));

        byte[] v9 = Sample(); v9[4] = 9;
        Assert.False(GgufTruncation.IsTruncation(Assert.Throws<InvalidDataException>(() => GgufModel.ParseIndex([(v9, v9.Length)])), out _));
    }

    [Fact]
    public void A_tensor_pointing_past_the_real_file_size_is_rejected_using_the_declared_size_not_the_prefix_length()
    {
        byte[] bytes = Sample(); int end = IndexEnd(Write(bytes));
        // The prefix is short but the declared file is large enough: fine.
        Assert.NotNull(GgufModel.ParseIndex([(bytes.AsMemory(0, end), bytes.Length)]));
        // Declared as a much smaller file than the offsets need: malformed, and not "truncation".
        var ex = Assert.Throws<InvalidDataException>(() => GgufModel.ParseIndex([(bytes.AsMemory(0, end), end + 10)]));
        Assert.False(GgufTruncation.IsTruncation(ex, out _));
    }

    // ── hostile headers ──
    [Fact]
    public void Absurd_counts_are_refused_before_anything_is_allocated_from_them()
    {
        foreach (ulong huge in new[] { (ulong)int.MaxValue, 1UL << 40, ulong.MaxValue })
        {
            using var ms = new MemoryStream(); using var w = new BinaryWriter(ms);
            w.Write(0x46554747u); w.Write(3u); w.Write(huge); w.Write(0UL);          // tensor count
            var ex = Assert.Throws<InvalidDataException>(() => GgufModel.ParseIndex([(ms.ToArray(), 1000)]));
            Assert.True(GgufTruncation.IsTruncation(ex, out _));

            using var ms2 = new MemoryStream(); using var w2 = new BinaryWriter(ms2);
            w2.Write(0x46554747u); w2.Write(3u); w2.Write(0UL); w2.Write(huge);      // metadata count
            Assert.True(GgufTruncation.IsTruncation(Assert.Throws<InvalidDataException>(() => GgufModel.ParseIndex([(ms2.ToArray(), 1000)])), out _));
        }
    }

    [Fact]
    public void An_array_that_claims_more_elements_than_there_are_bytes_is_refused()
    {
        byte[] bytes = Build(3, [KvStr("general.architecture", "llama"), ("x.arr", 9u, w => { w.Write(4u); w.Write((ulong)int.MaxValue); })], []);
        var ex = Assert.Throws<InvalidDataException>(() => GgufModel.ParseIndex([(bytes, bytes.Length)]));
        Assert.True(GgufTruncation.IsTruncation(ex, out _));
    }

    // ── split models ──
    [Fact]
    public void Shards_are_merged_with_shard_numbers_and_a_short_later_shard_names_itself()
    {
        byte[] s0 = Build(3, [KvStr("general.architecture", "llama"), KvU32("split.count", 2)], [T("token_embd.weight", [8, 3], 0, 0)], 256);
        byte[] s1 = Build(3, [KvStr("general.architecture", "llama"), KvU32("split.count", 2)], [T("blk.0.attn_q.weight", [8, 8], 0, 0)], 512);
        var index = GgufModel.ParseIndex([(s0, s0.Length), (s1, s1.Length)]);
        Assert.Equal(["token_embd.weight", "blk.0.attn_q.weight"], index.Tensors.Select(t => t.Name));
        Assert.Equal([0, 1], index.Tensors.Select(t => t.ShardIndex));
        Assert.Equal(2, index.TensorInfoEndOffsets.Count);
        Assert.Equal(2UL, index.Header.TensorCount);

        var ex = Assert.Throws<InvalidDataException>(() => GgufModel.ParseIndex([(s0, s0.Length), (s1.AsMemory(0, 30), s1.Length)]));
        Assert.True(GgufTruncation.IsTruncation(ex, out int shard));
        Assert.Equal(1, shard);
    }

    [Fact]
    public void No_shards_is_an_argument_error()
    {
        Assert.Throws<ArgumentException>(() => GgufModel.ParseIndex([]));
    }
}
