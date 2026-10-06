#nullable enable

using System.Collections.Immutable;
using System.Text;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Engine.Packaging;
using OpenTail.Stingray.Engine.Planning;
using OpenTail.Stingray.Engine.Runtime;
using Xunit;

namespace OpenTail.Stingray.Tests.Core;

public sealed class FrontendPlanParityTests : IDisposable
{
    private readonly List<string> _tempFiles = [];

    public void Dispose()
    {
        foreach (var file in _tempFiles)
        {
            try { if (File.Exists(file)) File.Delete(file); } catch { /* best effort */ }
        }
    }

    private string CreateTestGguf(int layers = 2, int contextLength = 2048)
    {
        var metadata = new Dictionary<string, (GgufValueType, object)>
        {
            ["general.architecture"] = (GgufValueType.String, "llama"),
            ["llama.block_count"] = (GgufValueType.UInt32, (uint)layers),
            ["llama.context_length"] = (GgufValueType.UInt32, (uint)contextLength),
            ["llama.embedding_length"] = (GgufValueType.UInt32, 64u),
            ["llama.feed_forward_length"] = (GgufValueType.UInt32, 128u),
            ["llama.attention.head_count"] = (GgufValueType.UInt32, 2u),
            ["llama.attention.head_count_kv"] = (GgufValueType.UInt32, 2u),
            ["llama.vocab_size"] = (GgufValueType.UInt32, 100u),
            ["tokenizer.ggml.model"] = (GgufValueType.String, "llama"),
            ["tokenizer.ggml.tokens"] = (GgufValueType.Array, new string[] { "<unk>", "<s>", "</s>", "foo", "bar" })
        };

        var tensors = new List<(string name, long[] dims, DType dtype, byte[] data)>
        {
            ("token_embd.weight", [64, 100], DType.Float32, new byte[64 * 100 * 4]),
            ("output_norm.weight", [64], DType.Float32, new byte[64 * 4]),
            ("output.weight", [64, 100], DType.Float32, new byte[64 * 100 * 4])
        };

        for (int i = 0; i < layers; i++)
        {
            tensors.Add(($"blk.{i}.attn_norm.weight", [64], DType.Float32, new byte[64 * 4]));
            tensors.Add(($"blk.{i}.attn_q.weight", [64, 64], DType.Float32, new byte[64 * 64 * 4]));
            tensors.Add(($"blk.{i}.attn_k.weight", [64, 64], DType.Float32, new byte[64 * 64 * 4]));
            tensors.Add(($"blk.{i}.attn_v.weight", [64, 64], DType.Float32, new byte[64 * 64 * 4]));
            tensors.Add(($"blk.{i}.attn_output.weight", [64, 64], DType.Float32, new byte[64 * 64 * 4]));
            tensors.Add(($"blk.{i}.ffn_norm.weight", [64], DType.Float32, new byte[64 * 4]));
            tensors.Add(($"blk.{i}.ffn_gate.weight", [64, 128], DType.Float32, new byte[64 * 128 * 4]));
            tensors.Add(($"blk.{i}.ffn_up.weight", [64, 128], DType.Float32, new byte[64 * 128 * 4]));
            tensors.Add(($"blk.{i}.ffn_down.weight", [128, 64], DType.Float32, new byte[128 * 64 * 4]));
        }

        var path = Path.Combine(Path.GetTempPath(), $"stingray_parity_test_{Guid.NewGuid():N}.gguf");
        _tempFiles.Add(path);

        using var fs = File.Create(path);
        using var writer = new GgufWriter(fs);

        writer.WriteHeader(3, (ulong)tensors.Count, (ulong)metadata.Count);
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

    [Fact]
    public void FrontendParity_PlannerGeneratesIdenticalPlan_ForSameModelAndIntent()
    {
        string modelPath = CreateTestGguf(layers: 2, contextLength: 2048);

        // Path A: ModelContext planning
        using var model = Model.Load(new ModelParams(modelPath) { Backend = "cpu", GpuLayerCount = 0 });
        using var ctx = (ModelContext)model.CreateContext(new ContextParams
        {
            ContextSize = 1024,
            ThreadCount = 4
        });
        var planCtx = ctx.ExecutionPlan;

        // Path B: CLI ExecutionPlanBuilder facade
        var planCli = ExecutionPlanBuilder.Build(
            modelPath: modelPath,
            goal: "balanced",
            pinBackend: "cpu",
            pinGpuLayers: 0,
            pinContextSize: 1024,
            noGpuProbe: true);

        // Path C: Server / direct ExecutionPlanner
        var pkg = LooseGgufModelPackage.Open(modelPath);
        var desc = ModelDescription.FromPackage(pkg);
        var caps = BackendCapabilities.Detect(noGpuProbe: true);
        var req = new ExecutionRequest
        {
            ModelPath = modelPath,
            Goal = "balanced",
            PinnedBackend = "cpu",
            PinnedGpuLayers = 0,
            PinnedContextSize = 1024,
            ThreadCount = 4,
            NoGpuProbe = true
        };
        var planServer = ExecutionPlanner.Plan(desc, req, caps);

        // All frontends must produce Schema v2 plans
        Assert.Equal(2, planCtx.SchemaVersion);
        Assert.Equal(2, planCli.SchemaVersion);
        Assert.Equal(2, planServer.SchemaVersion);

        // Architectural parity across all frontends
        Assert.Equal(planCtx.Backend, planCli.Backend);
        Assert.Equal(planCli.Backend, planServer.Backend);

        Assert.Equal(planCtx.ForwardPassKind, planCli.ForwardPassKind);
        Assert.Equal(planCli.ForwardPassKind, planServer.ForwardPassKind);

        Assert.Equal(planCtx.ContextSize, planCli.ContextSize);
        Assert.Equal(planCli.ContextSize, planServer.ContextSize);

        Assert.Equal(planCtx.GpuLayers, planCli.GpuLayers);
        Assert.Equal(planCli.GpuLayers, planServer.GpuLayers);

        Assert.Equal(planCtx.CpuLayers, planCli.CpuLayers);
        Assert.Equal(planCli.CpuLayers, planServer.CpuLayers);

        Assert.Equal(planCtx.KvDtype, planCli.KvDtype);
        Assert.Equal(planCli.KvDtype, planServer.KvDtype);

        Assert.True(planCtx.IsExecutable);
        Assert.True(planCli.IsExecutable);
        Assert.True(planServer.IsExecutable);
    }

    [Fact]
    public void FrontendParity_PlanInvariants_AreImmutableAndContainZeroAuto()
    {
        string modelPath = CreateTestGguf();
        var pkg = LooseGgufModelPackage.Open(modelPath);
        var desc = ModelDescription.FromPackage(pkg);
        var caps = BackendCapabilities.Detect(noGpuProbe: true);

        var req = new ExecutionRequest
        {
            ModelPath = modelPath,
            Goal = "balanced",
            PinnedBackend = "auto",
            PinnedContextSize = 512,
            PinnedKvDtype = "auto",
            TurboQuantMode = "auto",
            NoGpuProbe = true
        };

        var plan = ExecutionPlanner.Plan(desc, req, caps);

        // Strong immutability
        Assert.IsType<ImmutableArray<ExecutionPlanDecisionDetail>>(plan.Decisions);
        Assert.IsType<ImmutableArray<string>>(plan.Warnings);
        if (plan.PlanDecisions.HasValue)
        {
            Assert.IsType<ImmutableArray<ExecutionPlanDecision>>(plan.PlanDecisions.Value);
        }

        // Zero "auto" in resolved execution plan
        Assert.NotEqual("auto", plan.Backend, StringComparer.OrdinalIgnoreCase);
        Assert.NotEqual("auto", plan.KvDtype, StringComparer.OrdinalIgnoreCase);
        Assert.NotEqual("auto", plan.SelectedBackend, StringComparer.OrdinalIgnoreCase);

        // Invariants pass strongly typed validation
        ExecutionPlanValidator.Validate(plan);
    }

    [Fact]
    public void FrontendParity_RuntimeInstance_RespectsPlanContract()
    {
        string modelPath = CreateTestGguf(layers: 2, contextLength: 1024);
        using var model = Model.Load(new ModelParams(modelPath) { Backend = "cpu", GpuLayerCount = 0 });

        var pkg = LooseGgufModelPackage.Open(modelPath);
        var desc = ModelDescription.FromPackage(pkg);
        var caps = BackendCapabilities.Detect(noGpuProbe: true);
        var req = new ExecutionRequest
        {
            ModelPath = modelPath,
            PinnedBackend = "cpu",
            PinnedGpuLayers = 0,
            PinnedContextSize = 512,
            NoGpuProbe = true
        };

        var plan = ExecutionPlanner.Plan(desc, req, caps);
        using var instance = RuntimeInstance.Create(plan, model);

        // RuntimeInstance faithfully preserves the exact plan reference
        Assert.Same(plan, instance.Plan);
        Assert.Same(model, instance.Model);
        Assert.NotNull(instance.Engine);
        Assert.NotNull(instance.ForwardPass);
        Assert.NotNull(instance.Tokenizer);
        Assert.NotNull(instance.CpuBackend);
        Assert.Null(instance.CudaBackend);
        Assert.Null(instance.VulkanBackend);

        // Forward pass matches plan contract
        Assert.Equal(ForwardPassKind.CpuDense, plan.ForwardPassKind);
    }

    [Fact]
    public void FrontendParity_PlanNotExecutable_WhenArchitectureMismatch()
    {
        string modelPath = CreateTestGguf();
        using var model = Model.Load(new ModelParams(modelPath) { Backend = "cpu", GpuLayerCount = 0 });

        var plan = ExecutionPlan.CreateSynthesized(
            architecture: "qwen2",
            decision: new ForwardPassDecision(ForwardPassKind.CpuDense, "cpu"),
            backend: ForwardPassBackend.Cpu,
            contextSize: 512,
            gpuLayers: 0);

        var ex = Assert.Throws<PlanNotExecutableException>(() => RuntimeInstance.Create(plan, model));
        Assert.Contains("does not match", ex.Message);
        Assert.Same(plan, ex.Plan);
    }

    [Fact]
    public void FrontendParity_PlanNotExecutable_WhenFormatMismatch()
    {
        string modelPath = CreateTestGguf();
        using var model = Model.Load(new ModelParams(modelPath) { Backend = "cpu", GpuLayerCount = 0 });

        var plan = ExecutionPlan.CreateSynthesized(
            architecture: "llama",
            decision: new ForwardPassDecision(ForwardPassKind.CpuDense, "cpu"),
            backend: ForwardPassBackend.Cpu,
            contextSize: 512,
            gpuLayers: 0) with
        {
            ModelFormat = ModelFormat.SafeTensors
        };

        var ex = Assert.Throws<PlanNotExecutableException>(() => RuntimeInstance.Create(plan, model));
        Assert.Contains("SafeTensors", ex.Message);
        Assert.Same(plan, ex.Plan);
    }

    [Fact]
    public void FrontendParity_PlanNotExecutable_WhenMarkedNonExecutable()
    {
        string modelPath = CreateTestGguf();
        using var model = Model.Load(new ModelParams(modelPath) { Backend = "cpu", GpuLayerCount = 0 });

        var plan = ExecutionPlan.CreateSynthesized(
            architecture: "llama",
            decision: new ForwardPassDecision(ForwardPassKind.CpuDense, "cpu"),
            backend: ForwardPassBackend.Cpu,
            contextSize: 512,
            gpuLayers: 0) with
        {
            IsExecutable = false
        };

        var ex = Assert.Throws<PlanNotExecutableException>(() => RuntimeInstance.Create(plan, model));
        Assert.Contains("non-executable", ex.Message);
        Assert.Same(plan, ex.Plan);
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
                case GgufValueType.UInt8:
                    stream.WriteByte((byte)value);
                    _bytesWritten++;
                    break;
                case GgufValueType.UInt32:
                    WriteUInt32((uint)value);
                    break;
                case GgufValueType.Int32:
                    WriteInt32((int)value);
                    break;
                case GgufValueType.Float32:
                    WriteFloat32((float)value);
                    break;
                case GgufValueType.String:
                    WriteGgufString((string)value);
                    break;
                case GgufValueType.Array:
                    if (value is string[] strArr)
                    {
                        WriteUInt32((uint)GgufValueType.String);
                        WriteUInt64((ulong)strArr.Length);
                        foreach (var s in strArr)
                            WriteGgufString(s);
                    }
                    break;
                default:
                    throw new NotSupportedException($"Unsupported GGUF type in test writer: {type}");
            }
        }

        private void WriteUInt32(uint value)
        {
            Span<byte> buf = stackalloc byte[4];
            System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(buf, value);
            stream.Write(buf);
            _bytesWritten += 4;
        }

        private void WriteInt32(int value)
        {
            Span<byte> buf = stackalloc byte[4];
            System.Buffers.Binary.BinaryPrimitives.WriteInt32LittleEndian(buf, value);
            stream.Write(buf);
            _bytesWritten += 4;
        }

        private void WriteFloat32(float value)
        {
            Span<byte> buf = stackalloc byte[4];
            System.Buffers.Binary.BinaryPrimitives.WriteSingleLittleEndian(buf, value);
            stream.Write(buf);
            _bytesWritten += 4;
        }

        private void WriteUInt64(ulong value)
        {
            Span<byte> buf = stackalloc byte[8];
            System.Buffers.Binary.BinaryPrimitives.WriteUInt64LittleEndian(buf, value);
            stream.Write(buf);
            _bytesWritten += 8;
        }

        public void Dispose() { }
    }
}
