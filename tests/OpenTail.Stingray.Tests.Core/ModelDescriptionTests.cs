#nullable enable

using System.Buffers.Binary;
using System.Text;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Engine.Packaging;
using OpenTail.Stingray.Engine.Planning;

namespace OpenTail.Stingray.Tests.Core;

public sealed class ModelDescriptionTests : IDisposable
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

    private string CreateTempDir()
    {
        var path = Path.Combine(Path.GetTempPath(), $"stingray_test_{Guid.NewGuid():N}");
        Directory.CreateDirectory(path);
        _tempDirs.Add(path);
        return path;
    }

    [Fact]
    public void FromPackage_ExtractsPlanningFactsAndClosesFile()
    {
        var metadata = new Dictionary<string, (GgufValueType, object)>
        {
            ["general.architecture"] = (GgufValueType.String, "llama"),
            ["general.name"] = (GgufValueType.String, "TestLlama-1B"),
            ["llama.block_count"] = (GgufValueType.UInt32, 2u),
            ["llama.context_length"] = (GgufValueType.UInt32, 512u),
            ["llama.embedding_length"] = (GgufValueType.UInt32, 64u),
            ["llama.feed_forward_length"] = (GgufValueType.UInt32, 128u),
            ["llama.attention.head_count"] = (GgufValueType.UInt32, 2u),
            ["llama.attention.head_count_kv"] = (GgufValueType.UInt32, 2u),
            ["llama.rope.freq_base"] = (GgufValueType.Float32, 10_000f),
            ["llama.vocab_size"] = (GgufValueType.UInt32, 100u)
        };

        var tensors = new (string name, long[] dims, DType dtype, byte[] data)[]
        {
            ("token_embd.weight", [64, 100], DType.Float32, new byte[64 * 100 * 4]),
            ("output.weight", [64, 100], DType.Float32, new byte[64 * 100 * 4]),
            ("blk.0.attn_q.weight", [64, 64], DType.Float32, new byte[64 * 64 * 4]),
            ("blk.0.attn_k.weight", [64, 64], DType.Float32, new byte[64 * 64 * 4]),
            ("blk.0.attn_v.weight", [64, 64], DType.Float32, new byte[64 * 64 * 4]),
            ("blk.0.attn_output.weight", [64, 64], DType.Float32, new byte[64 * 64 * 4]),
            ("blk.0.attn_norm.weight", [64], DType.Float32, new byte[64 * 4]),
            ("blk.0.ffn_gate.weight", [64, 128], DType.Float32, new byte[64 * 128 * 4]),
            ("blk.0.ffn_up.weight", [64, 128], DType.Float32, new byte[64 * 128 * 4]),
            ("blk.0.ffn_down.weight", [128, 64], DType.Float32, new byte[128 * 64 * 4]),
            ("blk.0.ffn_norm.weight", [64], DType.Float32, new byte[64 * 4]),
            ("blk.1.attn_q.weight", [64, 64], DType.Float32, new byte[64 * 64 * 4]),
            ("blk.1.attn_k.weight", [64, 64], DType.Float32, new byte[64 * 64 * 4]),
            ("blk.1.attn_v.weight", [64, 64], DType.Float32, new byte[64 * 64 * 4]),
            ("blk.1.attn_output.weight", [64, 64], DType.Float32, new byte[64 * 64 * 4]),
            ("blk.1.attn_norm.weight", [64], DType.Float32, new byte[64 * 4]),
            ("blk.1.ffn_gate.weight", [64, 128], DType.Float32, new byte[64 * 128 * 4]),
            ("blk.1.ffn_up.weight", [64, 128], DType.Float32, new byte[64 * 128 * 4]),
            ("blk.1.ffn_down.weight", [128, 64], DType.Float32, new byte[128 * 64 * 4]),
            ("blk.1.ffn_norm.weight", [64], DType.Float32, new byte[64 * 4])
        };

        string filePath = CreateGguf(metadata, tensors);
        var package = LooseGgufModelPackage.Open(filePath);

        // 1. One-time inspection creates snapshot
        var desc = ModelDescription.FromPackage(package);

        // 2. Inspection completed: file handle must be released (able to open with write share)
        using (var fs = File.Open(filePath, FileMode.Open, FileAccess.ReadWrite, FileShare.None))
        {
            Assert.True(fs.Length > 0);
        }

        // 3. Verify semantic facts
        Assert.Equal("llama", desc.Semantics.Architecture);
        Assert.Equal(ForwardPassFamily.Dense, desc.Semantics.Family);
        Assert.Equal("TestLlama-1B", desc.Semantics.ModelName);
        Assert.Equal(ModelFormat.Gguf, desc.Semantics.Format);

        // 4. Verify planning facts
        Assert.Equal(2, desc.PlanningFacts.NumLayers);
        Assert.Equal(512, desc.PlanningFacts.ContextLength);
        Assert.Equal(64, desc.PlanningFacts.EmbeddingDim);
        Assert.Equal(32, desc.PlanningFacts.HeadDim);
        Assert.Equal(2, desc.PlanningFacts.NumHeads);
        Assert.Equal(2, desc.PlanningFacts.NumKvHeads);
        Assert.Equal(10_000f, desc.PlanningFacts.RopeTheta);
        Assert.Equal(2, desc.PlanningFacts.PerLayerGpuWeightBytes.Length);
        Assert.Equal(2, desc.PlanningFacts.PerLayerCpuWeightBytes.Length);
        Assert.False(desc.PlanningFacts.IsMoE);
        Assert.False(desc.PlanningFacts.IsHybridSsm);

        // 5. Verify capabilities & resources
        Assert.True(desc.Capabilities.SupportsContinuousBatching);
        Assert.True(desc.Capabilities.SupportsSpeculation);
        Assert.Equal("kv_cache", desc.Capabilities.StateModel);
        Assert.Equal(2, desc.Resources.LayerCount);
        Assert.Equal(512, desc.Resources.ContextLimit);
    }

    [Fact]
    public void FromPackage_HybridGdn_ExtractsRecurrentFacts()
    {
        var metadata = new Dictionary<string, (GgufValueType, object)>
        {
            ["general.architecture"] = (GgufValueType.String, "qwen2"),
            ["_opentailllm.is_hybrid_ssm"] = (GgufValueType.Bool, true),
            ["qwen2.block_count"] = (GgufValueType.UInt32, 4u),
            ["qwen2.context_length"] = (GgufValueType.UInt32, 2048u),
            ["qwen2.embedding_length"] = (GgufValueType.UInt32, 128u),
            ["qwen2.feed_forward_length"] = (GgufValueType.UInt32, 256u),
            ["qwen2.attention.head_count"] = (GgufValueType.UInt32, 2u),
            ["qwen2.attention.head_count_kv"] = (GgufValueType.UInt32, 2u)
        };

        var tensors = new (string name, long[] dims, DType dtype, byte[] data)[]
        {
            ("token_embd.weight", [128, 100], DType.Float32, new byte[128 * 100 * 4]),
            ("output.weight", [128, 100], DType.Float32, new byte[128 * 100 * 4])
        };

        string filePath = CreateGguf(metadata, tensors);
        var package = LooseGgufModelPackage.Open(filePath);
        var desc = ModelDescription.FromPackage(package);

        Assert.True(desc.PlanningFacts.IsHybridSsm);
        Assert.True(desc.PlanningFacts.HasHybridGdnLayers);
        Assert.True(desc.PlanningFacts.HasCpuHybridGdnPass);
        Assert.False(desc.Capabilities.SupportsSpeculation); // GDN state destructive update
        Assert.Equal("recurrent_gdn", desc.Capabilities.StateModel);
    }

    [Fact]
    public void CreateForwardPassRequest_SatisfiesForwardPassSelectionWithoutDiskAccess()
    {
        var facts = new ModelPlanningFacts
        {
            NumLayers = 4,
            ContextLength = 2048,
            EmbeddingDim = 256,
            HeadDim = 64,
            NumHeads = 4,
            NumKvHeads = 4,
            IntermediateDim = 512,
            VocabSize = 1000,
            IsMoE = false,
            IsHybridSsm = false,
            HasHybridGdnLayers = false,
            HasCpuHybridGdnPass = false,
            IsGemma4 = false,
            HasLayerHeadDim = false,
            HasMlaTensors = false
        };

        var semantics = new ModelSemanticDescription(
            Architecture: "llama",
            Family: ForwardPassFamily.Dense,
            ModelName: "PureSyntheticLlama",
            ParameterCount: 1_000_000,
            Format: ModelFormat.Gguf);

        var capabilities = new ModelCapabilitySummary(
            SupportsContinuousBatching: true,
            SupportsSpeculation: true,
            SupportsVision: false,
            SupportsEmbeddingInput: true,
            SupportsTools: true,
            StateModel: "kv_cache");

        var resources = new ModelResourceSummary(
            TotalWeightBytes: 4_000_000,
            EstimatedContextBytes: 1_000_000,
            PrimaryDType: DType.Float32,
            LayerCount: 4,
            ContextLimit: 2048,
            EmbeddingBytes: 256 * 1000 * 4,
            OutputBytes: 256 * 1000 * 4);

        var identity = ModelPackageIdentity.CreateProvisional(ModelFormat.Gguf, [new(ModelPackageRoles.PrimaryWeights, null, "in-memory-dummy.gguf")]);
        var desc = new ModelDescription(identity, semantics, capabilities, resources, facts);

        // 1. Evaluate CPU request
        var cpuReq = desc.CreateForwardPassRequest(
            frontend: ForwardPassFrontend.Cli,
            backend: ForwardPassBackend.Cpu,
            gpuLayers: 0);

        var cpuDecision = ForwardPassSelection.Select(cpuReq);
        Assert.Null(cpuDecision.Refusal);
        Assert.Equal(ForwardPassKind.CpuDense, cpuDecision.Kind);

        // 2. Evaluate Full CUDA Offload
        var cudaReq = desc.CreateForwardPassRequest(
            frontend: ForwardPassFrontend.Cli,
            backend: ForwardPassBackend.Cuda,
            gpuLayers: -1,
            plannedGpuLayers: 4,
            cudaAvailable: true);

        var cudaDecision = ForwardPassSelection.Select(cudaReq);
        Assert.Null(cudaDecision.Refusal);
        Assert.Equal(ForwardPassKind.CudaDense, cudaDecision.Kind);

        // 3. Evaluate Partial Vulkan Offload
        var vulkanHybridReq = desc.CreateForwardPassRequest(
            frontend: ForwardPassFrontend.Cli,
            backend: ForwardPassBackend.Vulkan,
            gpuLayers: 2,
            plannedGpuLayers: 2);

        var vulkanHybridDecision = ForwardPassSelection.Select(vulkanHybridReq);
        Assert.Null(vulkanHybridDecision.Refusal);
        Assert.Equal(ForwardPassKind.VulkanHybrid, vulkanHybridDecision.Kind);
    }

    [Fact]
    public void FromPackage_SafeTensors_ExtractsCpuFacts()
    {
        using var testPkg = SafetensorsTextModelPackageTests.TestPackage.Create();
        var package = SafeTensorsModelPackage.Open(testPkg.Directory);
        var desc = ModelDescription.FromPackage(package);

        Assert.Equal(ModelFormat.SafeTensors, desc.Semantics.Format);
        Assert.Equal(ForwardPassFamily.Dense, desc.Semantics.Family);
        Assert.False(desc.Capabilities.SupportsContinuousBatching);
        Assert.False(desc.Capabilities.SupportsSpeculation);

        var req = desc.CreateForwardPassRequest(frontend: ForwardPassFrontend.Cli);
        var decision = ForwardPassSelection.Select(req);
        Assert.Null(decision.Refusal);
        Assert.Equal(ForwardPassKind.SafeTensorsCpu, decision.Kind);
    }

    [Fact]
    public void MoEAndMlaFacts_AccuratelyCaptured()
    {
        var facts = new ModelPlanningFacts
        {
            NumLayers = 16,
            ContextLength = 4096,
            EmbeddingDim = 2048,
            HeadDim = 128,
            NumHeads = 16,
            NumKvHeads = 16,
            IntermediateDim = 4096,
            VocabSize = 32000,
            IsMoE = true,
            NumExperts = 8,
            NumActiveExperts = 2,
            ExpertIntermediateDim = 1024,
            HasSharedExpert = true,
            KvLoraRank = 512,
            HasMlaTensors = true,
            IsHybridSsm = false,
            HasHybridGdnLayers = false,
            HasCpuHybridGdnPass = false,
            IsGemma4 = false,
            HasLayerHeadDim = false
        };

        var semantics = new ModelSemanticDescription(
            Architecture: "deepseek2",
            Family: ForwardPassFamily.DeepSeek2Mla,
            ModelName: "DeepSeek2-Synthetic",
            ParameterCount: 16_000_000_000,
            Format: ModelFormat.Gguf);

        var capabilities = new ModelCapabilitySummary(
            SupportsContinuousBatching: true,
            SupportsSpeculation: true,
            SupportsVision: false,
            SupportsEmbeddingInput: true,
            SupportsTools: true,
            StateModel: "kv_cache");

        var resources = new ModelResourceSummary(
            TotalWeightBytes: 8_000_000_000,
            EstimatedContextBytes: 100_000_000,
            PrimaryDType: DType.Q4_K,
            LayerCount: 16,
            ContextLimit: 4096,
            EmbeddingBytes: 100_000,
            OutputBytes: 100_000);

        var identity = ModelPackageIdentity.CreateProvisional(ModelFormat.Gguf, [new(ModelPackageRoles.PrimaryWeights, null, "deepseek.gguf")]);
        var desc = new ModelDescription(identity, semantics, capabilities, resources, facts);

        var req = desc.CreateForwardPassRequest(
            frontend: ForwardPassFrontend.Cli,
            backend: ForwardPassBackend.Vulkan,
            gpuLayers: -1);

        Assert.True(req.IsMoE);
        Assert.Equal(512, req.KvLoraRank);
        Assert.True(req.HasMlaTensors);

        var decision = ForwardPassSelection.Select(req);
        Assert.Null(decision.Refusal);
        Assert.Equal(ForwardPassKind.DeepSeek2Vulkan, decision.Kind);
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
            WriteUInt32(0x46554747); // "GGUF" magic
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
