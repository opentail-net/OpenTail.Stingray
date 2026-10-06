#nullable enable

using System.Collections.Immutable;
using System.Text.Json;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Engine.Packaging;
using OpenTail.Stingray.Engine.Planning;

namespace OpenTail.Stingray.Tests.Core;

public sealed class ExecutionPlanSchemaV2Tests
{
    private static ExecutionPlan CreateSamplePlan(
        ForwardPassKind kind = ForwardPassKind.VulkanDense,
        ForwardPassBackend backend = ForwardPassBackend.Vulkan,
        int gpuLayers = 32,
        int cpuLayers = 0,
        int contextLength = 4096,
        BatchingMode batchingMode = BatchingMode.Sequential,
        int maxBatchSize = 1,
        SpeculationMode specMode = SpeculationMode.None,
        bool turboQuant = false)
    {
        var identity = ModelPackageIdentity.CreateProvisional(
            ModelFormat.Gguf,
            [new ModelPackageComponentIdentity(ModelPackageRoles.PrimaryWeights, null, "models/test.gguf")]);

        var backendPlan = new BackendPlan(
            Backend: backend,
            DeviceName: "Test Vulkan Device",
            DeviceIndex: 0,
            CudaAvailable: false,
            VulkanAvailable: true,
            ThreadCount: 8);

        var placement = new PlacementPlan(
            GpuLayers: gpuLayers,
            CpuLayers: cpuLayers,
            TotalLayers: gpuLayers + cpuLayers,
            GpuWeightBytes: 4_000_000_000,
            CpuWeightBytes: 0);

        var state = new StatePlan(
            StateModel: "kv_cache",
            ContextLength: contextLength,
            KvDType: DType.Float32,
            TurboQuant: turboQuant);

        var batching = new BatchingPlan(
            Mode: batchingMode,
            MaxBatchSize: maxBatchSize,
            MaxConcurrentSessions: maxBatchSize);

        var speculation = new SpeculationPlan(
            Mode: specMode);

        var modality = new ModalityPlan(
            SupportsVision: false,
            SupportsEmbeddingInput: true);

        var memory = new MemoryPlan(
            EstimatedVramMb: 4500.0,
            EstimatedRamMb: 500.0);

        var provenance = new PlanProvenance(
            CreatedAtUtc: "2026-10-06T20:00:00Z",
            PlannerVersion: "2.0.0",
            PrimaryModelPath: "models/test.gguf",
            TargetArchitecture: "llama",
            Goal: "balanced");

        return ExecutionPlan.CreateV2(
            packageIdentity: identity,
            forwardPassKind: kind,
            backendPlan: backendPlan,
            placement: placement,
            state: state,
            batching: batching,
            speculation: speculation,
            modality: modality,
            memory: memory,
            provenance: provenance,
            decisions: ImmutableArray<ExecutionPlanDecisionDetail>.Empty,
            warnings: ImmutableArray<string>.Empty);
    }

    [Fact]
    public void CreateV2_ProducesValidSchemaV2WithSubPlans()
    {
        var plan = CreateSamplePlan();

        Assert.Equal(2, plan.SchemaVersion);
        Assert.Equal("models/test.gguf", plan.ModelPath);
        Assert.Equal("vulkan", plan.Backend);
        Assert.Equal("vulkan", plan.SelectedBackend);
        Assert.Equal(32, plan.GpuLayers);
        Assert.Equal(0, plan.CpuLayers);
        Assert.Equal(32, plan.TotalLayers);
        Assert.Equal(4096, plan.ContextSize);
        Assert.Equal("float32", plan.KvDtype);
        Assert.Equal(4500.0, plan.EstimatedVramMb);
        Assert.Equal(500.0, plan.EstimatedRamMb);

        Assert.NotNull(plan.BackendPlan);
        Assert.NotNull(plan.Placement);
        Assert.NotNull(plan.State);
        Assert.NotNull(plan.Batching);
        Assert.NotNull(plan.Speculation);
        Assert.NotNull(plan.Modality);
        Assert.NotNull(plan.Memory);
        Assert.NotNull(plan.Provenance);

        // NativeAOT JSON round-trip
        string json = JsonSerializer.Serialize(plan, ExecutionPlanJsonContext.Default.ExecutionPlan);
        Assert.Contains("\"schema_version\": 2", json);
        Assert.Contains("\"backend_plan\"", json);
        Assert.Contains("\"placement_plan\"", json);

        var deserialized = JsonSerializer.Deserialize(json, ExecutionPlanJsonContext.Default.ExecutionPlan);
        Assert.NotNull(deserialized);
        Assert.Equal(2, deserialized.SchemaVersion);
        Assert.Equal(plan.ModelPath, deserialized.ModelPath);
        Assert.Equal(plan.Backend, deserialized.Backend);
    }

    [Fact]
    public void Immutability_CollectionsCannotBeMutated()
    {
        var mutableDecisions = new List<ExecutionPlanDecisionDetail>
        {
            new("CODE1", "val1", "reason1", "src1")
        };
        var mutableWarnings = new List<string> { "warn1" };

        var plan = new ExecutionPlan(
            SchemaVersion: 1,
            ModelPath: "test.gguf",
            Goal: "auto",
            Backend: "cpu",
            GpuLayers: 0,
            TotalLayers: 10,
            ContextSize: 1024,
            KvDtype: "fp32",
            EstimatedVramMb: 0,
            EstimatedRamMb: 100,
            Decisions: mutableDecisions,
            Warnings: mutableWarnings);

        // Mutate external lists
        mutableDecisions.Add(new("CODE2", "val2", "reason2", "src2"));
        mutableWarnings.Add("warn2");

        // Plan's immutable snapshot is isolated
        Assert.Single(plan.Decisions);
        Assert.Single(plan.Warnings);
        Assert.Equal("CODE1", plan.Decisions[0].Code);
    }

    [Fact]
    public void ExecutionPlanValidator_PassesValidPlan()
    {
        var plan = CreateSamplePlan();
        ExecutionPlanValidator.Validate(plan);
    }

    [Fact]
    public void ExecutionPlanValidator_RejectsAutoBackend()
    {
        var plan = CreateSamplePlan() with { Backend = "auto", SelectedBackend = "auto" };
        var ex = Assert.Throws<InvalidOperationException>(() => ExecutionPlanValidator.Validate(plan));
        Assert.Contains("must be resolved and cannot remain 'auto'", ex.Message);
    }

    [Fact]
    public void ExecutionPlanValidator_RejectsNegativeLayers()
    {
        var plan = CreateSamplePlan() with { GpuLayers = -1 };
        var ex = Assert.Throws<InvalidOperationException>(() => ExecutionPlanValidator.Validate(plan));
        Assert.Contains("cannot be negative", ex.Message);
    }

    [Fact]
    public void ExecutionPlanValidator_RejectsBackendMismatch()
    {
        // VulkanDense pass paired with CUDA backend
        var plan = CreateSamplePlan(kind: ForwardPassKind.VulkanDense, backend: ForwardPassBackend.Cuda);
        var ex = Assert.Throws<InvalidOperationException>(() => ExecutionPlanValidator.Validate(plan));
        Assert.Contains("requires Vulkan backend", ex.Message);
    }

    [Fact]
    public void ExecutionPlanValidator_RejectsUnsupportedContinuousBatching()
    {
        var plan = CreateSamplePlan(
            kind: ForwardPassKind.RwkvCpu,
            backend: ForwardPassBackend.Cpu,
            batchingMode: BatchingMode.Continuous,
            maxBatchSize: 4);

        var ex = Assert.Throws<InvalidOperationException>(() => ExecutionPlanValidator.Validate(plan));
        Assert.Contains("does not support continuous batching", ex.Message);
    }

    [Fact]
    public void ExecutionPlanValidator_RejectsUnsupportedSpeculationAndTurboQuant()
    {
        // Speculation with hybrid GDN
        var specPlan = CreateSamplePlan(
            kind: ForwardPassKind.CpuHybridGdn,
            backend: ForwardPassBackend.Cpu,
            specMode: SpeculationMode.DraftModel);

        var exSpec = Assert.Throws<InvalidOperationException>(() => ExecutionPlanValidator.Validate(specPlan));
        Assert.Contains("does not support speculative decoding", exSpec.Message);

        // TurboQuant with RWKV
        var tqPlan = CreateSamplePlan(
            kind: ForwardPassKind.RwkvCpu,
            backend: ForwardPassBackend.Cpu,
            turboQuant: true);

        var exTq = Assert.Throws<InvalidOperationException>(() => ExecutionPlanValidator.Validate(tqPlan));
        Assert.Contains("does not support TurboQuant", exTq.Message);
    }
}
