namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// Malformed-header cases for the GGUF parser (findings A22-A25). Each builds a small hostile file and asserts it is
/// refused with <see cref="InvalidDataException"/>: not a stack overflow, an allocation failure, an unrelated exception
/// type, or a silent under-count. Uses <see cref="SyntheticGguf"/> for well-formed scaffolding and raw bytes where the
/// file must be malformed in a way the builder cannot express.
/// </summary>
public sealed class GgufUntrustedInputTests : IDisposable
{
    private const uint F32 = 0;      // GGML_TYPE_F32
    private const uint Q4K = 12;     // GGML_TYPE_Q4_K

    private readonly List<string> _tempDirs = [];

    public void Dispose()
    {
        foreach (var dir in _tempDirs)
        {
            try { Directory.Delete(dir, recursive: true); } catch { }
        }
    }

    // ── A22: metadata arrays nested too deeply ─────────────────────────────────

    [Fact]
    public void Open_ArrayNestedSixtyFourDeep_IsRefused()
    {
        var path = WriteSingleFile(SyntheticGguf.Build(3, [NestedArrayKv("test.nested", depth: 64)], []));

        var ex = Assert.Throws<InvalidDataException>(() => GgufModel.Open(path));
        Assert.Contains("nest deeper than 8 levels", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Open_ShardSkippedArrayNestedSixtyFourDeep_IsRefused()
    {
        // Shard 2 of 2 repeats metadata, which later shards skip (SkipGgufValue) instead of parsing (ReadGgufValue).
        var dir = NewTempDir();
        var first = Path.Combine(dir, "split-00001-of-00002.gguf");
        File.WriteAllBytes(first, SyntheticGguf.Build(3, [], []));
        File.WriteAllBytes(Path.Combine(dir, "split-00002-of-00002.gguf"),
            SyntheticGguf.Build(3, [NestedArrayKv("test.nested", depth: 64)], []));

        var ex = Assert.Throws<InvalidDataException>(() => GgufModel.Open(first));
        Assert.Contains("nest deeper than 8 levels", ex.Message, StringComparison.Ordinal);
    }

    // ── A23: tensor rank checked before it is allocated ────────────────────────

    [Fact]
    public void Open_TensorRankOfUInt32Max_IsRefusedBeforeAllocation()
    {
        // One tensor entry whose rank is 0xFFFFFFFF. No dimension bytes follow, because the parser must stop at the rank.
        using var ms = new MemoryStream();
        using (var w = new BinaryWriter(ms, Encoding.UTF8, leaveOpen: true))
        {
            w.Write(0x46554747u); // "GGUF"
            w.Write(3u);          // version
            w.Write(1UL);         // tensor count
            w.Write(0UL);         // metadata kv count
            SyntheticGguf.Str(w, "bad.rank");
            w.Write(uint.MaxValue);
            w.Write(F32);
            w.Write(0UL);
        }

        var path = WriteSingleFile(ms.ToArray());

        var ex = Assert.Throws<InvalidDataException>(() => GgufModel.Open(path));
        Assert.Contains("bad.rank", ex.Message, StringComparison.Ordinal);
    }

    // ── A24: quantized tensor whose element count is not a whole number of blocks ──

    [Fact]
    public void Open_Q4KTensorWithElementCountNotBlockMultiple_IsRefused()
    {
        // [1, 100] is 100 elements. A Q4_K block holds 256, so this is not a whole number of blocks. Its floored
        // byte size is 0, so the bounds check alone would accept it.
        var path = WriteSingleFile(SyntheticGguf.Build(3, [], [new("q4k.weight", [1, 100], Q4K, 0)], dataBytes: 144));

        var ex = Assert.Throws<InvalidDataException>(() => GgufModel.Open(path));
        Assert.Contains("q4k.weight", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Open_Q4KTensorWhoseRowLengthIsNotBlockMultiple_IsRefusedEvenWhenTheTotalIsABlockMultiple()
    {
        // [100, 256] has 25600 elements (a whole number of Q4_K blocks), but its row length is 100. GGML requires the
        // first dimension to be a block multiple, so the rule is on dims[0], not on the total count.
        var path = WriteSingleFile(SyntheticGguf.Build(3, [], [new("q4k.rows.weight", [100, 256], Q4K, 0)], dataBytes: 14400));

        var ex = Assert.Throws<InvalidDataException>(() => GgufModel.Open(path));
        Assert.Contains("q4k.rows.weight", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Open_F32TensorWithSameShape_Loads()
    {
        // Control: unquantized types have block size 1, so the same shape must stay valid.
        var path = WriteSingleFile(SyntheticGguf.Build(3, [], [new("f32.weight", [1, 100], F32, 0)], dataBytes: 400));

        using var model = GgufModel.Open(path);
        Assert.Equal(400L, model.Tensors[0].ByteSize);
    }

    // ── A25: general.alignment and general.architecture types ──────────────────

    [Fact]
    public void Alignment_Zero_IsRefused()
    {
        var path = WriteMetadataFile(Kv("general.alignment", (uint)GgufValueType.UInt32, w => w.Write(0u)));

        Assert.Throws<InvalidDataException>(() => GgufModel.Open(path));
    }

    [Fact]
    public void Alignment_Negative_IsRefused()
    {
        var path = WriteMetadataFile(Kv("general.alignment", (uint)GgufValueType.Int32, w => w.Write(-1)));

        Assert.Throws<InvalidDataException>(() => GgufModel.Open(path));
    }

    [Fact]
    public void Alignment_NotPowerOfTwo_IsRefused()
    {
        var path = WriteMetadataFile(Kv("general.alignment", (uint)GgufValueType.UInt32, w => w.Write(3u)));

        Assert.Throws<InvalidDataException>(() => GgufModel.Open(path));
    }

    [Fact]
    public void Alignment_TwoToTheFortieth_IsRefused()
    {
        var path = WriteMetadataFile(Kv("general.alignment", (uint)GgufValueType.UInt64, w => w.Write(1UL << 40)));

        Assert.Throws<InvalidDataException>(() => GgufModel.Open(path));
    }

    [Fact]
    public void Alignment_String_IsRefused()
    {
        var path = WriteMetadataFile(SyntheticGguf.KvStr("general.alignment", "x"));

        Assert.Throws<InvalidDataException>(() => GgufModel.Open(path));
    }

    [Fact]
    public void Alignment_PowerOfTwoWithinLimit_Loads()
    {
        // Control: a valid alignment is still accepted and stored as given.
        var path = WriteMetadataFile(Kv("general.alignment", (uint)GgufValueType.UInt32, w => w.Write(64u)));

        using var model = GgufModel.Open(path);
        Assert.Equal(64u, model.Metadata["general.alignment"]);
    }

    [Fact]
    public void Architecture_Integer_IsRefused()
    {
        var path = WriteMetadataFile(Kv("general.architecture", (uint)GgufValueType.UInt32, w => w.Write(42u)));

        Assert.Throws<InvalidDataException>(() => GgufModel.Open(path));
    }

    // ── helpers ────────────────────────────────────────────────────────────────

    /// <summary>
    /// A metadata entry <paramref name="key"/> whose value is an array nested <paramref name="depth"/> levels deep.
    /// Each level writes its element type and a count of 1; the innermost array's element type is uint8, followed by one byte.
    /// </summary>
    private static (string, uint, SyntheticGguf.ValueWriter) NestedArrayKv(string key, int depth) =>
        (key, (uint)GgufValueType.Array, w =>
        {
            for (int level = 0; level < depth; level++)
            {
                bool innermost = level == depth - 1;
                w.Write(innermost ? (uint)GgufValueType.UInt8 : (uint)GgufValueType.Array);
                w.Write(1UL);
            }
            w.Write((byte)42);
        });

    private static (string, uint, SyntheticGguf.ValueWriter) Kv(string key, uint type, Action<BinaryWriter> write) =>
        (key, type, w => write(w));

    private string WriteMetadataFile(params (string, uint, SyntheticGguf.ValueWriter)[] kvs) =>
        WriteSingleFile(SyntheticGguf.Build(3, kvs, []));

    private string WriteSingleFile(byte[] bytes)
    {
        var path = Path.Combine(NewTempDir(), "model.gguf");
        File.WriteAllBytes(path, bytes);
        return path;
    }

    private string NewTempDir()
    {
        var dir = Path.Combine(Path.GetTempPath(), "stingray-gguf-untrusted-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        _tempDirs.Add(dir);
        return dir;
    }
}
