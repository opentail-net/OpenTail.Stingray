using System.Buffers.Binary;
using System.Text;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine.Verification;

namespace OpenTail.Stingray.Tests.Cli;

public sealed class VerifyGoldensCommandTests : IDisposable
{
    private readonly string _tempDir = Path.Combine(Path.GetTempPath(), "ot-verify-goldens-" + Guid.NewGuid().ToString("N"));

    public VerifyGoldensCommandTests()
    {
        Directory.CreateDirectory(_tempDir);
        Environment.SetEnvironmentVariable("STINGRAY_MODEL_DIRS", _tempDir);
    }

    public void Dispose()
    {
        Environment.SetEnvironmentVariable("STINGRAY_MODEL_DIRS", null);
        try { if (Directory.Exists(_tempDir)) Directory.Delete(_tempDir, recursive: true); } catch (IOException) { }
    }

    private (int Exit, string Output) Run(params string[] args) => Run(CancellationToken.None, args);

    private (int Exit, string Output) Run(CancellationToken cancellation, params string[] args)
    {
        var command = (ICommand)new VerifyGoldensCommand();
        var originalOut = Console.Out;
        var originalErr = Console.Error;
        var writer = new StringWriter();
        Console.SetOut(writer);
        Console.SetError(writer);
        try { return (command.Run(args, cancellation), writer.ToString()); }
        finally
        {
            Console.SetOut(originalOut);
            Console.SetError(originalErr);
        }
    }

    private string CreateMinimalGguf(string fileName, string architecture)
    {
        string path = Path.Combine(_tempDir, fileName);
        using var fs = File.Create(path);
        using var writer = new GgufWriter(fs);
        writer.WriteHeader(3, 0, 1);
        writer.WriteMetadataKv("general.architecture", GgufValueType.String, architecture);
        writer.PadToAlignment(32);
        return path;
    }

    [Fact]
    public void MissingDirectoryFailsGracefully()
    {
        var (exit, output) = Run("-d", Path.Combine(_tempDir, "nonexistent"));
        Assert.Equal(1, exit);
        Assert.Contains("Goldens directory not found", output, StringComparison.Ordinal);
    }

    [Fact]
    public void EmptyDirectoryReportsNoGoldens_InNormalMode()
    {
        var (exit, output) = Run("-d", _tempDir);
        Assert.Equal(0, exit);
        Assert.Contains("No golden files found", output, StringComparison.Ordinal);
    }

    [Fact]
    public void EmptyDirectory_InStrictMode_Fails()
    {
        var (exit, output) = Run("-d", _tempDir, "--strict");
        Assert.Equal(1, exit);
        Assert.Contains("No golden files found", output, StringComparison.Ordinal);
    }

    [Fact]
    public void FilterSelectingNoGoldens_InStrictMode_Fails()
    {
        var golden = new GoldenFile
        {
            Architecture = "synthetic-arch",
            Model = new GoldenModel { FileName = "nonexistent.gguf" }
        };
        golden.Save(Path.Combine(_tempDir, "some-arch.golden.json"));

        var (exit, output) = Run("-d", _tempDir, "-g", "non-matching-filter", "--strict");
        Assert.Equal(1, exit);
        Assert.Contains("No golden files found", output, StringComparison.Ordinal);
    }

    [Fact]
    public void MissingModelSkipsCleanlyAndWritesBaseline()
    {
        var golden = new GoldenFile
        {
            Architecture = "synthetic-arch",
            Model = new GoldenModel
            {
                FileName = "nonexistent-model-file-12345.gguf",
                Source = "test-owner/test-model"
            },
            Cases =
            [
                new GoldenCase
                {
                    Name = "c1",
                    PromptTokens = [1, 2, 3],
                    Tokens = [4, 5, 6]
                }
            ]
        };

        string goldenPath = Path.Combine(_tempDir, "synthetic.golden.json");
        golden.Save(goldenPath);

        string baselinePath = Path.Combine(_tempDir, "baseline.json");

        var (exit, output) = Run("-d", _tempDir, "--baseline", baselinePath);
        Assert.Equal(0, exit);
        Assert.Contains("synthetic.golden.json", output, StringComparison.Ordinal);
        Assert.Contains("SKIPPED", output, StringComparison.Ordinal);
        Assert.True(File.Exists(baselinePath));

        var baseline = GoldenBaselineFile.Load(baselinePath);
        Assert.Single(baseline.Entries);
        var entry = baseline.Entries[0];
        Assert.Equal("synthetic-arch", entry.Architecture);
        Assert.Equal("Skipped", entry.Verdict);
        Assert.Equal("Missing", entry.PinStatus);
        Assert.False(entry.Passed);
        Assert.NotNull(entry.GoldenSha256);
    }

    [Fact]
    public void StrictModeFailsOnMissingModel()
    {
        var golden = new GoldenFile
        {
            Architecture = "synthetic-arch",
            Model = new GoldenModel { FileName = "nonexistent-model-file-54321.gguf" }
        };

        golden.Save(Path.Combine(_tempDir, "strict-test.golden.json"));
        var (exit, _) = Run("-d", _tempDir, "--strict");
        Assert.Equal(1, exit);
    }

    [Fact]
    public void DiffOptionReportsComparison()
    {
        var golden = new GoldenFile
        {
            Architecture = "diff-arch",
            Model = new GoldenModel { FileName = "nonexistent-model-diff.gguf" }
        };
        golden.Save(Path.Combine(_tempDir, "diff-test.golden.json"));

        var priorBaseline = new GoldenBaselineFile
        {
            Entries =
            [
                new GoldenBaselineEntry
                {
                    Architecture = "diff-arch",
                    GoldenFile = "diff-test.golden.json",
                    ModelFile = "nonexistent-model-diff.gguf",
                    Verdict = "Exact",
                    Passed = true,
                    DecodeTokensPerSecond = 50.0
                }
            ]
        };
        string baselinePath = Path.Combine(_tempDir, "prior-baseline.json");
        priorBaseline.Save(baselinePath);

        var (exit, output) = Run("-d", _tempDir, "--diff", baselinePath);
        Assert.Equal(0, exit);
        Assert.Contains("Baseline Diff Comparison", output, StringComparison.Ordinal);
        Assert.Contains("diff-arch", output, StringComparison.Ordinal);
    }

    [Fact]
    public void DiffOption_MissingFile_ReturnsNonZero()
    {
        var golden = new GoldenFile
        {
            Architecture = "diff-arch",
            Model = new GoldenModel { FileName = "nonexistent.gguf" }
        };
        golden.Save(Path.Combine(_tempDir, "diff-missing.golden.json"));

        var (exit, output) = Run("-d", _tempDir, "--diff", Path.Combine(_tempDir, "nonexistent-baseline.json"));
        Assert.Equal(1, exit);
        Assert.Contains("Diff baseline file not found", output, StringComparison.Ordinal);
    }

    [Fact]
    public void DiffOption_MalformedBaseline_ReturnsNonZero()
    {
        var golden = new GoldenFile
        {
            Architecture = "diff-arch",
            Model = new GoldenModel { FileName = "nonexistent.gguf" }
        };
        golden.Save(Path.Combine(_tempDir, "diff-malformed.golden.json"));

        string badBaseline = Path.Combine(_tempDir, "bad-baseline.json");
        File.WriteAllText(badBaseline, "{ invalid json content");

        var (exit, output) = Run("-d", _tempDir, "--diff", badBaseline);
        Assert.Equal(1, exit);
        Assert.Contains("Failed to compute baseline diff", output, StringComparison.Ordinal);
    }

    [Fact]
    public void BaselineOption_InvalidOutputPath_ReturnsNonZero()
    {
        var golden = new GoldenFile
        {
            Architecture = "diff-arch",
            Model = new GoldenModel { FileName = "nonexistent.gguf" }
        };
        golden.Save(Path.Combine(_tempDir, "baseline-err.golden.json"));

        // Invalid path (directory that doesn't exist under invalid root)
        string invalidPath = "Z:\\nonexistent_drive_12345\\baseline.json";
        var (exit, output) = Run("-d", _tempDir, "--baseline", invalidPath);
        Assert.Equal(1, exit);
        Assert.Contains("Error writing baseline", output, StringComparison.Ordinal);
    }

    [Fact]
    public void Cancellation_ReturnsNonZero_AndDoesNotWriteBaseline()
    {
        var golden = new GoldenFile
        {
            Architecture = "cancel-arch",
            Model = new GoldenModel { FileName = "nonexistent.gguf" }
        };
        golden.Save(Path.Combine(_tempDir, "cancel.golden.json"));

        using var cts = new CancellationTokenSource();
        cts.Cancel(); // Pre-cancelled

        string baselineOut = Path.Combine(_tempDir, "cancel-baseline.json");
        var (exit, output) = Run(cts.Token, "-d", _tempDir, "--baseline", baselineOut);
        Assert.Equal(1, exit);
        Assert.Contains("Run was cancelled", output, StringComparison.Ordinal);
        Assert.False(File.Exists(baselineOut));
    }

    [Fact]
    public void ArchitectureMismatch_FailsAndRejectsModel()
    {
        // Checkpoint declares "llama", golden expects "mistral"
        string modelFileName = "mismatch-model.gguf";
        CreateMinimalGguf(modelFileName, "llama");

        var golden = new GoldenFile
        {
            Architecture = "mistral",
            Model = new GoldenModel { FileName = modelFileName }
        };
        golden.Save(Path.Combine(_tempDir, "arch-mismatch.golden.json"));

        var (exit, output) = Run("-d", _tempDir);
        Assert.Equal(1, exit);
        Assert.Contains("Architecture mismatch", output, StringComparison.Ordinal);
        Assert.Contains("ArchMismatch", output, StringComparison.Ordinal);
    }

    [Fact]
    public void PinStatus_MismatchAndNotRecorded_PassInNormalMode_FailInStrictMode()
    {
        // Create model declaring "llama"
        string modelFileName = "pin-test-model.gguf";
        CreateMinimalGguf(modelFileName, "llama");

        // Golden 1: Mismatch sha256
        var goldenMismatch = new GoldenFile
        {
            Architecture = "llama",
            Model = new GoldenModel { FileName = modelFileName, Sha256 = "1111222233334444555566667777888899990000111122223333444455556666" }
        };
        goldenMismatch.Save(Path.Combine(_tempDir, "mismatch.golden.json"));

        // Golden 2: NotRecorded sha256
        var goldenNotRecorded = new GoldenFile
        {
            Architecture = "llama",
            Model = new GoldenModel { FileName = modelFileName, Sha256 = null }
        };
        goldenNotRecorded.Save(Path.Combine(_tempDir, "not-recorded.golden.json"));

        // Normal mode: unpinned models run without failing on pin status alone
        var (exitNormal, outNormal) = Run("-d", _tempDir, "-g", "mismatch");
        Assert.Contains("Mismatch", outNormal, StringComparison.Ordinal);

        // Strict mode on Mismatch: fails
        var (exitStrict1, outStrict1) = Run("-d", _tempDir, "-g", "mismatch", "--strict");
        Assert.Equal(1, exitStrict1);

        // Strict mode on NotRecorded: fails
        var (exitStrict2, outStrict2) = Run("-d", _tempDir, "-g", "not-recorded", "--strict");
        Assert.Equal(1, exitStrict2);
    }

    [Fact]
    public void HyperparameterGuardFailure_RejectsBeforeExecution_AndSetsGuardFail()
    {
        string modelFileName = "guard-fail-model.gguf";
        CreateMinimalGguf(modelFileName, "llama");

        var golden = new GoldenFile
        {
            Architecture = "llama",
            Model = new GoldenModel { FileName = modelFileName },
            ExpectedHyperparameters = new Dictionary<string, string>
            {
                ["ropeDim"] = "999" // Mismatch against default/empty hyperparams
            }
        };
        golden.Save(Path.Combine(_tempDir, "guard-fail.golden.json"));

        var (exit, output) = Run("-d", _tempDir);
        Assert.Equal(1, exit);
        Assert.Contains("Guard failure", output, StringComparison.Ordinal);
        Assert.Contains("GuardFail", output, StringComparison.Ordinal);
    }

    private sealed class GgufWriter(Stream stream) : IDisposable
    {
        private long _bytesWritten;

        public void WriteHeader(uint version, ulong tensorCount, ulong metadataKvCount)
        {
            WriteUInt32(0x46554747); // "GGUF" magic
            WriteUInt32(version);
            WriteUInt64(tensorCount);
            WriteUInt64(metadataKvCount);
        }

        public void WriteMetadataKv(string key, GgufValueType type, object value)
        {
            WriteGgufString(key);
            WriteUInt32((uint)type);
            WriteGgufValue(type, value);
        }

        public void PadToAlignment(int alignment)
        {
            var remainder = _bytesWritten % alignment;
            if (remainder != 0)
            {
                var padding = alignment - (int)remainder;
                Span<byte> zeros = stackalloc byte[padding];
                zeros.Clear();
                stream.Write(zeros);
                _bytesWritten += padding;
            }
        }

        private void WriteGgufString(string s)
        {
            var bytes = Encoding.UTF8.GetBytes(s);
            WriteUInt64((ulong)bytes.Length);
            stream.Write(bytes);
            _bytesWritten += bytes.Length;
        }

        private void WriteGgufValue(GgufValueType type, object value)
        {
            switch (type)
            {
                case GgufValueType.UInt8:   WriteByte((byte)value); break;
                case GgufValueType.Int8:    WriteByte((byte)(sbyte)value); break;
                case GgufValueType.UInt16:  WriteUInt16((ushort)value); break;
                case GgufValueType.Int16:   WriteInt16((short)value); break;
                case GgufValueType.UInt32:  WriteUInt32((uint)value); break;
                case GgufValueType.Int32:   WriteInt32((int)value); break;
                case GgufValueType.Float32: WriteFloat32((float)value); break;
                case GgufValueType.Bool:    WriteByte((bool)value ? (byte)1 : (byte)0); break;
                case GgufValueType.String:  WriteGgufString((string)value); break;
                case GgufValueType.UInt64:  WriteUInt64((ulong)value); break;
                case GgufValueType.Int64:   WriteInt64((long)value); break;
                case GgufValueType.Float64: WriteFloat64((double)value); break;
                default: throw new NotSupportedException($"Unsupported GGUF value type: {type}");
            }
        }

        private void WriteByte(byte v) { stream.WriteByte(v); _bytesWritten += 1; }

        private void WriteUInt16(ushort v)
        {
            Span<byte> buf = stackalloc byte[2];
            BinaryPrimitives.WriteUInt16LittleEndian(buf, v);
            stream.Write(buf);
            _bytesWritten += 2;
        }

        private void WriteInt16(short v)
        {
            Span<byte> buf = stackalloc byte[2];
            BinaryPrimitives.WriteInt16LittleEndian(buf, v);
            stream.Write(buf);
            _bytesWritten += 2;
        }

        private void WriteUInt32(uint v)
        {
            Span<byte> buf = stackalloc byte[4];
            BinaryPrimitives.WriteUInt32LittleEndian(buf, v);
            stream.Write(buf);
            _bytesWritten += 4;
        }

        private void WriteInt32(int v)
        {
            Span<byte> buf = stackalloc byte[4];
            BinaryPrimitives.WriteInt32LittleEndian(buf, v);
            stream.Write(buf);
            _bytesWritten += 4;
        }

        private void WriteFloat32(float v)
        {
            Span<byte> buf = stackalloc byte[4];
            BinaryPrimitives.WriteSingleLittleEndian(buf, v);
            stream.Write(buf);
            _bytesWritten += 4;
        }

        private void WriteUInt64(ulong v)
        {
            Span<byte> buf = stackalloc byte[8];
            BinaryPrimitives.WriteUInt64LittleEndian(buf, v);
            stream.Write(buf);
            _bytesWritten += 8;
        }

        private void WriteInt64(long v)
        {
            Span<byte> buf = stackalloc byte[8];
            BinaryPrimitives.WriteInt64LittleEndian(buf, v);
            stream.Write(buf);
            _bytesWritten += 8;
        }

        private void WriteFloat64(double v)
        {
            Span<byte> buf = stackalloc byte[8];
            BinaryPrimitives.WriteDoubleLittleEndian(buf, v);
            stream.Write(buf);
            _bytesWritten += 8;
        }

        public void Dispose() { }
    }
}

