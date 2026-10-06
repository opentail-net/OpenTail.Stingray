#nullable enable

using System.Buffers.Binary;
using System.Text;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Engine.Planning;

namespace OpenTail.Stingray.Tests.Core;

public sealed class ModelContextPlanTests : IDisposable
{
    private readonly List<string> _tempFiles = [];

    public void Dispose()
    {
        foreach (var file in _tempFiles)
        {
            try { if (File.Exists(file)) File.Delete(file); } catch { /* best effort */ }
        }
    }

    private string CreateTestGguf(int layers = 1, int contextLength = 1024)
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

        var tensors = new (string name, long[] dims, DType dtype, byte[] data)[]
        {
            ("token_embd.weight", [64, 100], DType.Float32, new byte[64 * 100 * 4]),
            ("output_norm.weight", [64], DType.Float32, new byte[64 * 4]),
            ("output.weight", [64, 100], DType.Float32, new byte[64 * 100 * 4]),
            ("blk.0.attn_norm.weight", [64], DType.Float32, new byte[64 * 4]),
            ("blk.0.attn_q.weight", [64, 64], DType.Float32, new byte[64 * 64 * 4]),
            ("blk.0.attn_k.weight", [64, 64], DType.Float32, new byte[64 * 64 * 4]),
            ("blk.0.attn_v.weight", [64, 64], DType.Float32, new byte[64 * 64 * 4]),
            ("blk.0.attn_output.weight", [64, 64], DType.Float32, new byte[64 * 64 * 4]),
            ("blk.0.ffn_norm.weight", [64], DType.Float32, new byte[64 * 4]),
            ("blk.0.ffn_gate.weight", [64, 128], DType.Float32, new byte[64 * 128 * 4]),
            ("blk.0.ffn_up.weight", [64, 128], DType.Float32, new byte[64 * 128 * 4]),
            ("blk.0.ffn_down.weight", [128, 64], DType.Float32, new byte[128 * 64 * 4])
        };

        var path = Path.Combine(Path.GetTempPath(), $"stingray_ctx_test_{Guid.NewGuid():N}.gguf");
        _tempFiles.Add(path);

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

    [Fact]
    public void ModelContext_ExecutionPlan_LazilyPopulatedAndMatchesParameters()
    {
        string modelPath = CreateTestGguf();
        using var model = Model.Load(new ModelParams(modelPath) { Backend = "cpu", GpuLayerCount = 0 });
        using var ctx = (ModelContext)model.CreateContext(new ContextParams
        {
            ContextSize = 512,
            ThreadCount = 4,
            FlashAttention = true
        });

        var plan = ctx.ExecutionPlan;

        Assert.NotNull(plan);
        Assert.Equal(2, plan.SchemaVersion);
        Assert.Equal("cpu", plan.Backend);
        Assert.Equal(512, plan.ContextSize);
        Assert.True(plan.IsExecutable);
        Assert.NotNull(plan.BackendPlan);
        Assert.Equal(4, plan.BackendPlan.ThreadCount);
        Assert.Equal(ForwardPassKind.CpuDense, plan.ForwardPassKind);
    }

    [Fact]
    public void ModelContext_Engine_InstantiatesFromPlanAndPreservesPlanInstance()
    {
        string modelPath = CreateTestGguf();
        using var model = Model.Load(new ModelParams(modelPath) { Backend = "cpu", GpuLayerCount = 0 });
        using var ctx = (ModelContext)model.CreateContext(new ContextParams { ContextSize = 512 });

        var engine = ctx.Engine;

        Assert.NotNull(engine);
        Assert.IsType<InferenceEngine>(engine);
        Assert.NotNull(ctx.ExecutionPlan);
        Assert.Equal(512, ctx.ExecutionPlan.ContextSize);
    }

    [Fact]
    public void ModelContext_ContinuousBatching_CreatesPlanWithContinuousBatchingMode()
    {
        string modelPath = CreateTestGguf();
        using var model = Model.Load(new ModelParams(modelPath) { Backend = "cpu", GpuLayerCount = 0 });
        using var ctx = (ModelContext)model.CreateContext(new ContextParams { ContextSize = 512, BatchSize = 6 });

        var batchEngine = ctx.CreateContinuousBatchingEngine(maxBatchSize: 6);

        Assert.NotNull(batchEngine);
        Assert.IsType<ContinuousBatchingEngine>(batchEngine);
        Assert.Equal(BatchingMode.Continuous, ctx.ExecutionPlan.Batching?.Mode);
        Assert.Equal(6, ctx.ExecutionPlan.Batching?.MaxBatchSize);
    }

    [Fact]
    public void ModelContext_SetEngine_ReplacesEngineAndSetsFlag()
    {
        string modelPath = CreateTestGguf();
        using var model = Model.Load(new ModelParams(modelPath) { Backend = "cpu", GpuLayerCount = 0 });
        using var ctx = (ModelContext)model.CreateContext(new ContextParams { ContextSize = 512 });

        var defaultEngine = ctx.Engine;
        Assert.False(ctx.HasExplicitEngine);

        var stubEngine = new StubEngine();
        ctx.SetEngine(stubEngine);

        Assert.True(ctx.HasExplicitEngine);
        Assert.Same(stubEngine, ctx.Engine);
    }

    [Fact]
    public void ModelContext_Dispose_UnregistersAndDisposesRuntime()
    {
        string modelPath = CreateTestGguf();
        var model = Model.Load(new ModelParams(modelPath) { Backend = "cpu", GpuLayerCount = 0 });
        var ctx = (ModelContext)model.CreateContext(new ContextParams { ContextSize = 512 });

        _ = ctx.Engine;
        Assert.Equal(1, model.ActiveContextCount);

        ctx.Dispose();
        Assert.Equal(0, model.ActiveContextCount);

        Assert.Throws<ObjectDisposedException>(() => _ = ctx.Engine);
        Assert.Throws<ObjectDisposedException>(() => _ = ctx.ExecutionPlan);

        model.Dispose();
    }

    private sealed class StubEngine : IInferenceEngine, IDisposable
    {
        public bool IsDisposed { get; private set; }
        public string ModelId => "stub-model";
        public int QueueDepth => 0;
        public int ActiveRequests => 0;
        public bool PrefixCacheEnabled => false;
        public long PrefillTokensReused => 0;

        public async IAsyncEnumerable<GenerateChunk> GenerateChunksAsync(
            string prompt,
            SamplingParams sp,
            [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct = default,
            string? canonicalHistoryPrefix = null)
        {
            await Task.Yield();
            yield break;
        }

        public void Reset() { }

        public void Dispose()
        {
            IsDisposed = true;
        }
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
                case GgufValueType.UInt32:
                    WriteUInt32((uint)value);
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
                    throw new NotSupportedException($"Type {type} not supported in test writer");
            }
        }

        private void WriteUInt32(uint value)
        {
            Span<byte> buf = stackalloc byte[4];
            BinaryPrimitives.WriteUInt32LittleEndian(buf, value);
            stream.Write(buf);
            _bytesWritten += 4;
        }

        private void WriteUInt64(ulong value)
        {
            Span<byte> buf = stackalloc byte[8];
            BinaryPrimitives.WriteUInt64LittleEndian(buf, value);
            stream.Write(buf);
            _bytesWritten += 8;
        }

        public void Dispose() => stream.Dispose();
    }
}
