using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Core;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio.Fast;

public sealed class AudioCppEmbeddedFilesTests : IDisposable
{
    private readonly List<string> _tempFiles = [];

    public void Dispose()
    {
        foreach (var file in _tempFiles)
        {
            try { if (File.Exists(file)) File.Delete(file); } catch { }
        }
    }

    private string CreateGguf(Dictionary<string, object> metadata)
    {
        string path = Path.Combine(Path.GetTempPath(), $"audiocpp_test_{Guid.NewGuid():N}.gguf");
        _tempFiles.Add(path);

        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream);

        // Header: Magic "GGUF" (0x46554747), Version 3, 0 tensors, metadata count
        writer.Write((byte)'G');
        writer.Write((byte)'G');
        writer.Write((byte)'U');
        writer.Write((byte)'F');
        writer.Write((uint)3);
        writer.Write((ulong)0); // tensor count = 0
        writer.Write((ulong)metadata.Count);

        foreach (var (key, value) in metadata)
        {
            WriteGgufString(writer, key);
            WriteGgufValue(writer, value);
        }

        return path;
    }

    private static void WriteGgufString(BinaryWriter writer, string s)
    {
        byte[] utf8 = Encoding.UTF8.GetBytes(s);
        writer.Write((ulong)utf8.Length);
        writer.Write(utf8);
    }

    private static void WriteGgufValue(BinaryWriter writer, object value)
    {
        switch (value)
        {
            case uint u32:
                writer.Write((uint)GgufValueType.UInt32);
                writer.Write(u32);
                break;
            case int i32:
                writer.Write((uint)GgufValueType.Int32);
                writer.Write(i32);
                break;
            case ulong u64:
                writer.Write((uint)GgufValueType.UInt64);
                writer.Write(u64);
                break;
            case string s:
                writer.Write((uint)GgufValueType.String);
                WriteGgufString(writer, s);
                break;
            case string[] strArr:
                writer.Write((uint)GgufValueType.Array);
                writer.Write((uint)GgufValueType.String);
                writer.Write((ulong)strArr.Length);
                foreach (var s in strArr) WriteGgufString(writer, s);
                break;
            case ulong[] u64Arr:
                writer.Write((uint)GgufValueType.Array);
                writer.Write((uint)GgufValueType.UInt64);
                writer.Write((ulong)u64Arr.Length);
                foreach (var u in u64Arr) writer.Write(u);
                break;
            case byte[] bArr:
                writer.Write((uint)GgufValueType.Array);
                writer.Write((uint)GgufValueType.UInt8);
                writer.Write((ulong)bArr.Length);
                writer.Write(bArr);
                break;
            default:
                throw new NotSupportedException($"Type {value.GetType()} not supported in test writer.");
        }
    }

    [Fact]
    public void ReadAll_EmptyMetadata_ReturnsEmptyDictionary()
    {
        var meta = new Dictionary<string, object>
        {
            ["general.architecture"] = "test"
        };
        string path = CreateGguf(meta);
        using var model = GgufModel.Open(path);

        var dict = AudioCppEmbeddedFiles.ReadAll(model);
        Assert.NotNull(dict);
        Assert.Empty(dict);

        var names = AudioCppEmbeddedFiles.Names(model);
        Assert.Empty(names);

        Assert.False(AudioCppEmbeddedFiles.TryGet(model, "any_file.txt", out var bytes));
        Assert.Null(bytes);

        Assert.Throws<FileNotFoundException>(() => AudioCppEmbeddedFiles.Get(model, "any_file.txt"));
    }

    [Fact]
    public void ReadAll_ValidEmbeddedFiles_ExtractsFilesAccuratelyAndCaseInsensitive()
    {
        byte[] file1 = Encoding.UTF8.GetBytes("{\"model\": \"test_tokenizer\"}");
        byte[] file2 = Encoding.UTF8.GetBytes("prompt: Emily text recording reference.");
        byte[] file3 = [0x01, 0x02, 0x03, 0x04];

        byte[] allData = new byte[file1.Length + file2.Length + file3.Length];
        Buffer.BlockCopy(file1, 0, allData, 0, file1.Length);
        Buffer.BlockCopy(file2, 0, allData, file1.Length, file2.Length);
        Buffer.BlockCopy(file3, 0, allData, file1.Length + file2.Length, file3.Length);

        string[] names = ["tokenizer.json", "prompts/emily.txt", "data.bin"];
        ulong[] offsets = [0, (ulong)file1.Length, (ulong)(file1.Length + file2.Length), (ulong)allData.Length];

        var meta = new Dictionary<string, object>
        {
            ["general.architecture"] = "test",
            ["audiocpp.embedded_files.names"] = names,
            ["audiocpp.embedded_files.offsets"] = offsets,
            ["audiocpp.embedded_files.data"] = allData,
        };

        string path = CreateGguf(meta);
        using var model = GgufModel.Open(path);

        var dict = AudioCppEmbeddedFiles.ReadAll(model);
        Assert.Equal(3, dict.Count);

        // Verify content
        Assert.True(AudioCppEmbeddedFiles.TryGet(model, "tokenizer.json", out var outBytes1));
        Assert.Equal(file1, outBytes1);

        // Case-insensitivity check
        Assert.True(AudioCppEmbeddedFiles.TryGet(model, "TOKENIZER.JSON", out var outBytesUpper));
        Assert.Equal(file1, outBytesUpper);

        // Slash-style path lookup
        byte[] outBytes2 = AudioCppEmbeddedFiles.Get(model, "prompts/emily.txt");
        Assert.Equal(file2, outBytes2);

        byte[] outBytes3 = AudioCppEmbeddedFiles.Get(model, "DATA.BIN");
        Assert.Equal(file3, outBytes3);

        // Names property check
        var extractedNames = AudioCppEmbeddedFiles.Names(model);
        Assert.Equal(3, extractedNames.Count);
        Assert.Contains("tokenizer.json", extractedNames);
        Assert.Contains("prompts/emily.txt", extractedNames);
        Assert.Contains("data.bin", extractedNames);
    }

    [Fact]
    public void ReadAll_OffsetsLengthEqualsNamesLength_ExtractsLastFileToEnd()
    {
        byte[] file1 = "first_file_content"u8.ToArray();
        byte[] file2 = "second_file_content_until_end"u8.ToArray();

        byte[] allData = new byte[file1.Length + file2.Length];
        Buffer.BlockCopy(file1, 0, allData, 0, file1.Length);
        Buffer.BlockCopy(file2, 0, allData, file1.Length, file2.Length);

        // Offsets length == names length (no sentinel at end)
        string[] names = ["file1.txt", "file2.txt"];
        ulong[] offsets = [0, (ulong)file1.Length];

        var meta = new Dictionary<string, object>
        {
            ["general.architecture"] = "test",
            ["audiocpp.embedded_files.names"] = names,
            ["audiocpp.embedded_files.offsets"] = offsets,
            ["audiocpp.embedded_files.data"] = allData,
        };

        string path = CreateGguf(meta);
        using var model = GgufModel.Open(path);

        var dict = AudioCppEmbeddedFiles.ReadAll(model);
        Assert.Equal(2, dict.Count);
        Assert.Equal(file1, dict["file1.txt"]);
        Assert.Equal(file2, dict["file2.txt"]);
    }

    [Fact]
    public void ReadAll_MissingDataKey_ThrowsInvalidDataException()
    {
        var meta = new Dictionary<string, object>
        {
            ["general.architecture"] = "test",
            ["audiocpp.embedded_files.names"] = new string[] { "file1.txt" },
            ["audiocpp.embedded_files.offsets"] = new ulong[] { 0, 10 },
            // missing audiocpp.embedded_files.data
        };

        string path = CreateGguf(meta);
        using var model = GgufModel.Open(path);

        Assert.Throws<InvalidDataException>(() => AudioCppEmbeddedFiles.ReadAll(model));
    }

    [Fact]
    public void ReadAll_NonMonotonicOffsets_ThrowsInvalidDataException()
    {
        byte[] data = new byte[50];
        var meta = new Dictionary<string, object>
        {
            ["general.architecture"] = "test",
            ["audiocpp.embedded_files.names"] = new string[] { "file1.txt", "file2.txt" },
            ["audiocpp.embedded_files.offsets"] = new ulong[] { 20, 10, 30 }, // 10 < 20
            ["audiocpp.embedded_files.data"] = data,
        };

        string path = CreateGguf(meta);
        using var model = GgufModel.Open(path);

        Assert.Throws<InvalidDataException>(() => AudioCppEmbeddedFiles.ReadAll(model));
    }

    [Fact]
    public void ReadAll_OutOfBoundsOffset_ThrowsInvalidDataException()
    {
        byte[] data = new byte[20];
        var meta = new Dictionary<string, object>
        {
            ["general.architecture"] = "test",
            ["audiocpp.embedded_files.names"] = new string[] { "file1.txt" },
            ["audiocpp.embedded_files.offsets"] = new ulong[] { 0, 50 }, // 50 > 20
            ["audiocpp.embedded_files.data"] = data,
        };

        string path = CreateGguf(meta);
        using var model = GgufModel.Open(path);

        Assert.Throws<InvalidDataException>(() => AudioCppEmbeddedFiles.ReadAll(model));
    }

    [Fact]
    public void ReadAll_CachesResultPerModelInstance()
    {
        byte[] file = "cached_content"u8.ToArray();
        var meta = new Dictionary<string, object>
        {
            ["general.architecture"] = "test",
            ["audiocpp.embedded_files.names"] = new string[] { "file.txt" },
            ["audiocpp.embedded_files.offsets"] = new ulong[] { 0, (ulong)file.Length },
            ["audiocpp.embedded_files.data"] = file,
        };

        string path = CreateGguf(meta);
        using var model = GgufModel.Open(path);

        var first = AudioCppEmbeddedFiles.ReadAll(model);
        var second = AudioCppEmbeddedFiles.ReadAll(model);

        Assert.Same(first, second);
    }
}
