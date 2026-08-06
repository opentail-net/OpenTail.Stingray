using System.Buffers.Binary;
using System.Text;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// Mutation fixtures for the SafeTensors trust boundary. These exercise the parser directly so a
/// malformed file is rejected before package/profile logic or an inference allocation can obscure it.
/// </summary>
public sealed class SafetensorsLoaderHardeningTests
{
    [Fact]
    public void Open_HeaderLengthBeyondFile_IsRejectedBeforeAllocation()
    {
        using var file = TemporaryFile.Create([0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x7F]);

        Assert.Throws<InvalidDataException>(() => SafetensorsLoader.Open(file.Path));
    }

    [Fact]
    public void Open_ShapeElementCountOverflow_IsRejected()
    {
        const string header = """
            {"weight":{"dtype":"F32","shape":[2147483647,2147483647,2147483647],"data_offsets":[0,0]}}
            """;
        using var file = TemporaryFile.CreateSafetensors(header, []);

        Assert.Throws<InvalidDataException>(() => SafetensorsLoader.Open(file.Path));
    }

    [Theory]
    [InlineData("{\"weight\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,8]}}", 4)]
    [InlineData("{\"weight\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,8]}}", 8)]
    [InlineData("{\"weight\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[4,0]}}", 4)]
    public void Open_InvalidTensorOffsetsOrLength_IsRejected(string header, int dataBytes)
    {
        using var file = TemporaryFile.CreateSafetensors(header, new byte[dataBytes]);

        Assert.Throws<InvalidDataException>(() => SafetensorsLoader.Open(file.Path));
    }

    [Fact]
    public void Open_DuplicateTensorName_IsRejected()
    {
        const string header = """
            {"weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}
            """;
        using var file = TemporaryFile.CreateSafetensors(header, new byte[4]);

        Assert.Throws<InvalidDataException>(() => SafetensorsLoader.Open(file.Path));
    }

    [Fact]
    public void Open_OverlappingTensorRanges_AreRejected()
    {
        const string header = """
            {"first":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"second":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}
            """;
        using var file = TemporaryFile.CreateSafetensors(header, new byte[4]);

        Assert.Throws<InvalidDataException>(() => SafetensorsLoader.Open(file.Path));
    }

    [Fact]
    public void OpenDirectory_ShardIndexEscapingRoot_IsRejectedByTheLoader()
    {
        using var package = TemporaryPackage.Create("""{"weight_map":{"weight":"../outside.safetensors"}}""");

        Assert.Throws<InvalidDataException>(() => SafetensorsLoader.OpenDirectory(package.Directory));
    }

    [Fact]
    public void OpenDirectory_ShardIndexMustMatchShardInventory()
    {
        using var package = TemporaryPackage.Create("""{"weight_map":{"different":"model-00001.safetensors"}}""");
        TemporaryFile.WriteSafetensors(Path.Combine(package.Directory, "model-00001.safetensors"),
            """{"weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}""", new byte[4]);

        Assert.Throws<InvalidDataException>(() => SafetensorsLoader.OpenDirectory(package.Directory));
    }

    [Fact]
    public void OpenDirectory_ValidShardedIndex_ExposesEveryTensor()
    {
        using var package = TemporaryPackage.Create(
            """{"weight_map":{"first":"model-00001.safetensors","second":"model-00002.safetensors"}}""");
        TemporaryFile.WriteSafetensors(Path.Combine(package.Directory, "model-00001.safetensors"),
            """{"first":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}""", new byte[4]);
        TemporaryFile.WriteSafetensors(Path.Combine(package.Directory, "model-00002.safetensors"),
            """{"second":{"dtype":"F16","shape":[2],"data_offsets":[0,4]}}""", new byte[4]);

        using var loader = SafetensorsLoader.OpenDirectory(package.Directory);

        Assert.Equal(2, loader.TensorCount);
        Assert.Equal("F32", loader.GetDtype("first"));
        Assert.Equal("F16", loader.GetDtype("second"));
    }

    [Theory]
    [InlineData("{}")]
    [InlineData("[]")]
    [InlineData("{\"weight_map\":[]}")]
    [InlineData("{\"weight_map\":{\"weight\":false}}")]
    public void OpenDirectory_MalformedIndexJson_IsRejected(string indexJson)
    {
        using var package = TemporaryPackage.Create(indexJson);

        Assert.Throws<InvalidDataException>(() => SafetensorsLoader.OpenDirectory(package.Directory));
    }

    // ── FP8 decoding ──────────────────────────────────────────────────────────
    //
    // The loader used to decode FP8 with its own per-byte helpers, which had no case for either
    // format's non-finite encodings. It now delegates to FastVectorTypeConverter. These pin the
    // encodings that were previously wrong, at the level a caller actually sees them.

    [Fact]
    public void ReadF32_Fp8E4M3_DecodesNonFiniteAndMaxFinite()
    {
        // 0x7F = S.1111.111, which E4M3FN reserves for NaN — the old helper returned +480, a value
        // the format cannot represent. 0x7E is the true max finite, 448.
        using var file = TemporaryFile.CreateSafetensors(
            """{"t":{"dtype":"F8_E4M3","shape":[4],"data_offsets":[0,4]}}""",
            [0x7F, 0xFF, 0x7E, 0x00]);

        using var loader = SafetensorsLoader.Open(file.Path);
        float[] values = loader.ReadF32("t");

        Assert.True(float.IsNaN(values[0]));
        Assert.True(float.IsNaN(values[1]));
        Assert.Equal(448f, values[2]);
        Assert.Equal(0f, values[3]);
    }

    [Fact]
    public void ReadF32_Fp8E5M2_DecodesInfinityAndNaN()
    {
        // E5M2 follows IEEE conventions: exponent 31 with mantissa 0 is Inf, otherwise NaN.
        // The old helper returned finite values near 2^16 for both.
        using var file = TemporaryFile.CreateSafetensors(
            """{"t":{"dtype":"F8_E5M2","shape":[4],"data_offsets":[0,4]}}""",
            [0x7C, 0xFC, 0x7D, 0x00]);

        using var loader = SafetensorsLoader.Open(file.Path);
        float[] values = loader.ReadF32("t");

        Assert.True(float.IsPositiveInfinity(values[0]));
        Assert.True(float.IsNegativeInfinity(values[1]));
        Assert.True(float.IsNaN(values[2]));
        Assert.Equal(0f, values[3]);
    }

    private sealed class TemporaryFile : IDisposable
    {
        public string Path { get; }

        private TemporaryFile(string path) => Path = path;

        public static TemporaryFile Create(byte[] bytes)
        {
            string path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), $"opentail-st-hostile-{Guid.NewGuid():N}.safetensors");
            File.WriteAllBytes(path, bytes);
            return new TemporaryFile(path);
        }

        public static TemporaryFile CreateSafetensors(string header, byte[] data)
        {
            string path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), $"opentail-st-hostile-{Guid.NewGuid():N}.safetensors");
            WriteSafetensors(path, header, data);
            return new TemporaryFile(path);
        }

        public static void WriteSafetensors(string path, string header, byte[] data)
        {
            byte[] headerBytes = Encoding.UTF8.GetBytes(header);
            byte[] bytes = new byte[sizeof(ulong) + headerBytes.Length + data.Length];
            BinaryPrimitives.WriteUInt64LittleEndian(bytes, (ulong)headerBytes.Length);
            headerBytes.CopyTo(bytes, sizeof(ulong));
            data.CopyTo(bytes, sizeof(ulong) + headerBytes.Length);
            File.WriteAllBytes(path, bytes);
        }

        public void Dispose()
        {
            try { File.Delete(Path); }
            catch { }
        }
    }

    private sealed class TemporaryPackage : IDisposable
    {
        public string Directory { get; }

        private TemporaryPackage(string directory) => Directory = directory;

        public static TemporaryPackage Create(string indexJson)
        {
            string directory = Path.Combine(Path.GetTempPath(), $"opentail-st-index-{Guid.NewGuid():N}");
            System.IO.Directory.CreateDirectory(directory);
            File.WriteAllText(Path.Combine(directory, "model.safetensors.index.json"), indexJson);
            return new TemporaryPackage(directory);
        }

        public void Dispose()
        {
            try { System.IO.Directory.Delete(Directory, recursive: true); }
            catch { }
        }
    }
}
