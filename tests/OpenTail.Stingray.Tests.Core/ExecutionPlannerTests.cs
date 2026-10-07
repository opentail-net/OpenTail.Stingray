#nullable enable

using System.Buffers.Binary;
using System.Text;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Engine.Packaging;
using OpenTail.Stingray.Engine.Planning;

namespace OpenTail.Stingray.Tests.Core;

public sealed class ExecutionPlannerTests : IDisposable
{
    private readonly List<string> _tempFiles = [];
    private readonly List<string> _tempDirs = [];

    public void Dispose()
    {
        foreach (var file in _tempFiles)
        {
            try { if (File.Exists(file)) File.Delete(file); } catch { /* best effort */ }
        }
        foreach (var dir in _tempDirs)
        {
            try { if (Directory.Exists(dir)) Directory.Delete(dir, true); } catch { /* best effort */ }
        }
    }

    private string CreateTempFile(string extension = ".gguf")
    {
        var path = Path.Combine(Path.GetTempPath(), $"stingray_test_{Guid.NewGuid():N}{extension}");
        _tempFiles.Add(path);
        return path;
    }

    [Fact]
    public void Plan_EndToEnd_GeneratesImmutablePlanWithoutReopeningFiles()
    {
        var metadata = new Dictionary<string, (GgufValueType, object)>
        {
            ["general.architecture"] = (GgufValueType.String, "llama"),
            ["llama.block_count"] = (GgufValueType.UInt32, 4u),
            ["llama.context_length"] = (GgufValueType.UInt32, 2048u),
            ["llama.embedding_length"] = (GgufValueType.UInt32, 64u),
            ["llama.feed_forward_length"] = (GgufValueType.UInt32, 128u),
            ["llama.attention.head_count"] = (GgufValueType.UInt32, 2u),
            ["llama.attention.head_count_kv"] = (GgufValueType.UInt32, 2u),
            ["llama.vocab_size"] = (GgufValueType.UInt32, 100u)
        };

        var tensors = new (string name, long[] dims, DType dtype, byte[] data)[]
        {
            ("token_embd.weight", [64, 100], DType.Float32, new byte[64 * 100 * 4]),
            ("output.weight", [64, 100], DType.Float32, new byte[64 * 100 * 4]),
            ("blk.0.attn_q.weight", [64, 64], DType.Float32, new byte[64 * 64 * 4]),
            ("blk.1.attn_q.weight", [64, 64], DType.Float32, new byte[64 * 64 * 4]),
            ("blk.2.attn_q.weight", [64, 64], DType.Float32, new byte[64 * 64 * 4]),
            ("blk.3.attn_q.weight", [64, 64], DType.Float32, new byte[64 * 64 * 4])
        };

        string filePath = CreateGguf(metadata, tensors);
        var pkg = LooseGgufModelPackage.Open(filePath);
        var desc = ModelDescription.FromPackage(pkg);

        var capabilities = new BackendCapabilities(
            CudaAvailable: false,
            VulkanAvailable: true,
            HardwareProfile: new HardwareProfile(
                VramBytes: 8L * 1024 * 1024 * 1024,
                RamBytes: 32L * 1024 * 1024 * 1024,
                CpuCores: 8,
                EstPcieBandwidthGBps: 16.0,
                HasAvx512: false),
            VulkanDeviceName: "Simulated Vulkan GPU",
            RecommendedThreadCount: 8);

        var req = new ExecutionRequest
        {
            Goal = "balanced",
            PinnedContextSize = 1024,
            PinnedKvDtype = "fp32"
        };

        var plan = ExecutionPlanner.Plan(desc, req, capabilities);

        Assert.Equal(2, plan.SchemaVersion);
        Assert.Equal("vulkan", plan.Backend);
        Assert.Equal("vulkan", plan.SelectedBackend);
        Assert.Equal(ForwardPassKind.VulkanDense, plan.ForwardPassKind);
        Assert.Equal(4, plan.GpuLayers);
        Assert.Equal(0, plan.CpuLayers);
        Assert.Equal(1024, plan.ContextSize);
        Assert.Equal("float32", plan.KvDtype);

        Assert.NotNull(plan.BackendPlan);
        Assert.NotNull(plan.Placement);
        Assert.NotNull(plan.State);
        Assert.NotNull(plan.Batching);
        Assert.NotNull(plan.Speculation);
        Assert.NotNull(plan.Modality);
        Assert.NotNull(plan.Memory);
        Assert.NotNull(plan.Provenance);

        // Validation passes
        ExecutionPlanValidator.Validate(plan);
    }

    [Fact]
    public void Plan_CandidateEvaluation_RoutesSafeTensorsToCpu()
    {
        using var testPkg = SafetensorsTextModelPackageTests.TestPackage.Create();
        var pkg = SafeTensorsModelPackage.Open(testPkg.Directory);
        var desc = ModelDescription.FromPackage(pkg);

        // Even when GPU is available, SafeTensors candidate evaluation resolves to CPU
        var capabilities = new BackendCapabilities(
            CudaAvailable: true,
            VulkanAvailable: true,
            HardwareProfile: new HardwareProfile(
                VramBytes: 8L * 1024 * 1024 * 1024,
                RamBytes: 32L * 1024 * 1024 * 1024,
                CpuCores: 8,
                EstPcieBandwidthGBps: 16.0,
                HasAvx512: false));

        var plan = ExecutionPlanner.Plan(desc, new ExecutionRequest(), capabilities);

        Assert.Equal("cpu", plan.Backend);
        Assert.Equal(ForwardPassKind.SafeTensorsCpu, plan.ForwardPassKind);
        Assert.Equal(0, plan.GpuLayers);
        Assert.Equal(1, plan.CpuLayers);
    }

    [Fact]
    public void ExecutionPlanBuilder_Build_DelegatesCorrectly()
    {
        var metadata = new Dictionary<string, (GgufValueType, object)>
        {
            ["general.architecture"] = (GgufValueType.String, "llama"),
            ["llama.block_count"] = (GgufValueType.UInt32, 2u),
            ["llama.context_length"] = (GgufValueType.UInt32, 1024u),
            ["llama.embedding_length"] = (GgufValueType.UInt32, 64u),
            ["llama.feed_forward_length"] = (GgufValueType.UInt32, 128u),
            ["llama.attention.head_count"] = (GgufValueType.UInt32, 2u),
            ["llama.attention.head_count_kv"] = (GgufValueType.UInt32, 2u),
            ["llama.vocab_size"] = (GgufValueType.UInt32, 100u)
        };

        var tensors = new (string name, long[] dims, DType dtype, byte[] data)[]
        {
            ("token_embd.weight", [64, 100], DType.Float32, new byte[64 * 100 * 4]),
            ("output.weight", [64, 100], DType.Float32, new byte[64 * 100 * 4]),
            ("blk.0.attn_q.weight", [64, 64], DType.Float32, new byte[64 * 64 * 4]),
            ("blk.1.attn_q.weight", [64, 64], DType.Float32, new byte[64 * 64 * 4])
        };

        string filePath = CreateGguf(metadata, tensors);
        var plan = ExecutionPlanBuilder.Build(
            modelPath: filePath,
            goal: "quality",
            pinBackend: "cpu",
            pinGpuLayers: 0,
            pinContextSize: 512,
            noGpuProbe: true);

        Assert.Equal(2, plan.SchemaVersion);
        Assert.Equal("cpu", plan.Backend);
        Assert.Equal(0, plan.GpuLayers);
        Assert.Equal(2, plan.CpuLayers);
        Assert.Equal(512, plan.ContextSize);
        Assert.Equal(ForwardPassKind.CpuDense, plan.ForwardPassKind);
    }

    private ExecutionPlan PlanTinyModel(ExecutionRequest request, bool vulkan = true)
    {
        var metadata = new Dictionary<string, (GgufValueType, object)>
        {
            ["general.architecture"] = (GgufValueType.String, "llama"),
            ["llama.block_count"] = (GgufValueType.UInt32, 2u),
            ["llama.context_length"] = (GgufValueType.UInt32, 2048u),
            ["llama.embedding_length"] = (GgufValueType.UInt32, 64u),
            ["llama.feed_forward_length"] = (GgufValueType.UInt32, 128u),
            ["llama.attention.head_count"] = (GgufValueType.UInt32, 2u),
            ["llama.attention.head_count_kv"] = (GgufValueType.UInt32, 2u),
            ["llama.vocab_size"] = (GgufValueType.UInt32, 100u)
        };
        var tensors = new (string name, long[] dims, DType dtype, byte[] data)[]
        {
            ("token_embd.weight", [64, 100], DType.Float32, new byte[64 * 100 * 4]),
            ("output.weight", [64, 100], DType.Float32, new byte[64 * 100 * 4]),
            ("blk.0.attn_q.weight", [64, 64], DType.Float32, new byte[64 * 64 * 4]),
            ("blk.1.attn_q.weight", [64, 64], DType.Float32, new byte[64 * 64 * 4]),
        };
        var desc = ModelDescription.FromPackage(LooseGgufModelPackage.Open(CreateGguf(metadata, tensors)));
        var caps = new BackendCapabilities(
            CudaAvailable: false,
            VulkanAvailable: vulkan,
            HardwareProfile: new HardwareProfile(8L << 30, 32L << 30, 8, 16.0, false),
            VulkanDeviceName: vulkan ? "Simulated Vulkan GPU" : null,
            RecommendedThreadCount: 8);
        return ExecutionPlanner.Plan(desc, request, caps);
    }

    [Fact]
    public void Plan_CarriesRequestedDeviceIndex_ForGpuBackend()
    {
        var plan = PlanTinyModel(new ExecutionRequest { PinnedContextSize = 512, DeviceIndex = 2 });

        Assert.Equal("vulkan", plan.Backend);
        Assert.Equal(2, plan.BackendPlan!.DeviceIndex);
    }

    [Fact]
    public void Plan_DeviceIndexIsZero_ForCpuBackend()
    {
        var plan = PlanTinyModel(new ExecutionRequest { PinnedBackend = "cpu", PinnedGpuLayers = 0, PinnedContextSize = 512, DeviceIndex = 2 });

        Assert.Equal("cpu", plan.Backend);
        Assert.Equal(0, plan.BackendPlan!.DeviceIndex);
    }

    [Fact]
    public void Plan_RecordsMoeExecutionChoices_AndSurvivesAotJsonRoundTrip()
    {
        var plan = PlanTinyModel(new ExecutionRequest
        {
            PinnedContextSize = 512,
            CpuMoe = true,
            GpuMoePrefill = false,
            MoeWarmPin = 4,
            MoeWarmPinAfter = 16,
            MoePredictPrefetch = false,
            ExpertStatsPath = "stats.json",
        });

        Assert.NotNull(plan.Moe);
        Assert.Equal(true, plan.Moe!.CpuMoe);
        Assert.Equal(false, plan.Moe.GpuMoePrefill);
        Assert.Equal(4, plan.Moe.WarmPin);
        Assert.Equal(16, plan.Moe.WarmPinAfter);
        Assert.Equal(false, plan.Moe.PredictPrefetch);
        Assert.Equal("stats.json", plan.Moe.ExpertStatsPath);

        string json = System.Text.Json.JsonSerializer.Serialize(plan, ExecutionPlanJsonContext.Default.ExecutionPlan);
        var back = System.Text.Json.JsonSerializer.Deserialize(json, ExecutionPlanJsonContext.Default.ExecutionPlan)!;
        Assert.Equal(plan.Moe, back.Moe);
        Assert.Equal(plan.BackendPlan!.DeviceIndex, back.BackendPlan!.DeviceIndex);
    }

    [Fact]
    public void Plan_UnspecifiedMoeChoices_StayNull()
    {
        var plan = PlanTinyModel(new ExecutionRequest { PinnedContextSize = 512 });

        Assert.Null(plan.Moe!.CpuMoe);
        Assert.Null(plan.Moe.GpuMoePrefill);
        Assert.Null(plan.Moe.PredictPrefetch);
    }

    [Theory]
    [InlineData("src/OpenTail.Stingray.Engine/Runtime/RuntimeInstance.cs")]
    [InlineData("src/OpenTail.Stingray.Server/InferenceEngineLoader.cs")]
    public void RuntimeLayer_DoesNotMakePolicyDecisions(string relativePath)
    {
        // THE PLANNER DECIDES, THE PLAN RECORDS, THE RUNTIME OBEYS: once an ExecutionPlan exists, neither the
        // runtime nor the server loader may call the placement/selection policy again.
        string? dir = AppContext.BaseDirectory;
        while (dir is not null && !File.Exists(Path.Combine(dir, "OpenTail.Stingray.slnx")))
            dir = Path.GetDirectoryName(dir);
        Assert.NotNull(dir);

        string text = File.ReadAllText(Path.Combine(dir!, relativePath));
        Assert.DoesNotContain("TierPlanner.Plan(", text);
        Assert.DoesNotContain("DSparkPlacementPlanner.Plan(", text);
        Assert.DoesNotContain("ForwardPassSelection.Select", text);
    }

    private string CreateGguf(
        Dictionary<string, (GgufValueType type, object value)> metadata,
        (string name, long[] dims, DType dtype, byte[] data)[] tensors)
    {
        var path = CreateTempFile();
        using var fs = File.Create(path);
        using var writer = new GgufWriter(fs);

        writer.WriteHeader(3, (ulong)tensors.Length, (ulong)metadata.Count);
        foreach (var (key, (type, value)) in metadata)
            writer.WriteMetadataKv(key, type, value);

        ulong offset = 0;
        foreach (var (name, dims, dtype, data) in tensors)
        {
            writer.WriteTensorInfo(name, dims, dtype, offset);
            offset += (ulong)data.Length;
        }

        writer.PadToAlignment(32);
        foreach (var (_, _, _, data) in tensors)
            fs.Write(data);

        return path;
    }

    private sealed class GgufWriter(Stream stream) : IDisposable
    {
        private long _bytesWritten;

        public void WriteHeader(uint version, ulong tensorCount, ulong metadataKvCount)
        {
            WriteUInt32(0x46554747);
            WriteUInt32(version);
            WriteUInt64(tensorCount);
            WriteUInt64(metadataKvCount);
        }

        public void WriteTensorInfo(string name, long[] dims, DType dtype, ulong offset)
        {
            WriteGgufString(name);
            WriteUInt32((uint)dims.Length);
            foreach (var dim in dims)
                WriteUInt64((ulong)dim);
            WriteUInt32((uint)dtype);
            WriteUInt64(offset);
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
