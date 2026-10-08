using System.Collections.Immutable;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Engine.Packaging;
using Xunit;

namespace OpenTail.Stingray.Tests.Core;

public sealed class PlanDrivenArchitectureLoadContextTests
{
    private sealed unsafe class DummyTensorSource : IModelTensorSource
    {
        public IReadOnlyList<GgufTensorInfo> Tensors => [];
        public IReadOnlyDictionary<string, object> Metadata => new Dictionary<string, object>();
        public GgufTensorInfo? FindTensor(string name) => null;
        public ReadOnlySpan<byte> GetTensorData(GgufTensorInfo tensor) => [];
        public byte* GetTensorDataPtr(GgufTensorInfo tensor) => null;
        public void Dispose() { }
    }

    [Fact]
    public void ArchitectureLoadContext_ForwardsPropertiesDirectlyFromPlan()
    {
        var placement = new LayerPlacement(
            GpuLayers: 12,
            CpuLayers: 20,
            GpuWeightBytes: 4000000,
            GpuKvBytes: 0,
            RecommendedCtxSize: 2048,
            ExpertCacheBudgetBytes: 100000,
            MoeRoutedExpertBytes: 200000,
            CpuWeightBytes: 8000000);

        var plan = ExecutionPlan.CreateSynthesized(
            architecture: "llama",
            decision: new ForwardPassDecision(ForwardPassKind.CpuDense, "cpu"),
            backend: ForwardPassBackend.Cpu,
            contextSize: 2048,
            gpuLayers: 12,
            placement: placement,
            turboQuant: true,
            turboQuantMode: "auto",
            headDim: 128,
            tqQuantizer: TqQuantizer.KVarN,
            flashAttention: false,
            kvDtype: DType.Float32,
            prefillDequantCacheBytes: 65536,
            preferBatchingOverAutoSnapKv: true,
            modelPath: "test-model.gguf",
            totalLayers: 32);

        var tensorSource = new DummyTensorSource();
        var probe = new ArchitectureProbe
        {
            Architecture = "llama",
            TensorSource = tensorSource,
        };

        var loadCtx = new ArchitectureLoadContext
        {
            Probe = probe,
            Hyperparams = new ModelHyperparams { EmbeddingDim = 4096, NumLayers = 32, HeadDim = 128, VocabSize = 32000 },
            Plan = plan,
        };

        // Assert direct forwarders from Plan
        Assert.Equal(ForwardPassKind.CpuDense, loadCtx.ForwardPassKind);
        Assert.Equal(ForwardPassKind.CpuDense, loadCtx.Decision.Kind);
        Assert.Equal(ForwardPassBackend.Cpu, loadCtx.Backend);
        Assert.Equal(2048, loadCtx.ContextSize);
        Assert.Equal(12, loadCtx.GpuLayers);

        Assert.NotNull(loadCtx.Placement);
        Assert.Equal(12, loadCtx.Placement.GpuLayers);
        Assert.Equal(20, loadCtx.Placement.CpuLayers);
        Assert.Equal(4000000, loadCtx.Placement.GpuWeightBytes);
        Assert.Equal(8000000, loadCtx.Placement.CpuWeightBytes);
        Assert.Equal(100000, loadCtx.Placement.ExpertCacheBudgetBytes);
        Assert.Equal(200000, loadCtx.Placement.MoeRoutedExpertBytes);

        Assert.True(loadCtx.TurboQuant);
        Assert.Equal("auto", loadCtx.TurboQuantMode);
        Assert.Equal(128, loadCtx.HeadDim);
        Assert.Equal(TqQuantizer.KVarN, loadCtx.TqQuantizer);
        Assert.False(loadCtx.FlashAttention);
        Assert.Equal(DType.Float32, loadCtx.KvDType);
        Assert.Equal(65536, loadCtx.PrefillDequantCacheBytes);
        Assert.True(loadCtx.PreferBatchingOverAutoSnapKv);
    }

    [Fact]
    public void CommonForwardPassFactory_BranchesOnPlanForwardPassKindDirectly()
    {
        var plan = ExecutionPlan.CreateSynthesized(
            architecture: "llama",
            decision: new ForwardPassDecision(ForwardPassKind.RwkvCpu, "cpu"),
            backend: ForwardPassBackend.Cpu,
            contextSize: 512,
            gpuLayers: 0,
            headDim: 64);

        var tensorSource = new DummyTensorSource();
        var probe = new ArchitectureProbe
        {
            Architecture = "llama",
            TensorSource = tensorSource,
        };

        var loadCtx = new ArchitectureLoadContext
        {
            Probe = probe,
            Hyperparams = new ModelHyperparams { EmbeddingDim = 128, NumLayers = 2, HeadDim = 64, VocabSize = 256, NumHeads = 2, NumKvHeads = 2 },
            Plan = plan,
        };

        var ex = Assert.Throws<InvalidOperationException>(() => CommonForwardPassFactory.CreateDense(loadCtx));
        Assert.Contains("Unsupported forward pass kind 'RwkvCpu' for dense architecture.", ex.Message);
    }

    [Fact]
    public void ArchitectureDescriptor_ConstructForwardPass_PassesPlanDrivenContext()
    {
        ExecutionPlan? capturedPlan = null;
        var desc = new ArchitectureDescriptor
        {
            Id = "plan_test_arch",
            Status = AdmissionStatus.Admitted,
            EvidenceDoc = "docs/STATUS.md",
            StatusExemption = "exempt",
            CreateForwardPass = ctx =>
            {
                capturedPlan = ctx.Plan;
                return new MockForwardPass();
            },
        };
        desc.Validate();

        var plan = ExecutionPlan.CreateSynthesized(
            architecture: "plan_test_arch",
            decision: new ForwardPassDecision(ForwardPassKind.CpuDense, "cpu"),
            backend: ForwardPassBackend.Cpu,
            contextSize: 256,
            gpuLayers: 4,
            headDim: 64);

        var tensorSource = new DummyTensorSource();
        var probe = new ArchitectureProbe
        {
            Architecture = "plan_test_arch",
            TensorSource = tensorSource,
        };

        var loadCtx = new ArchitectureLoadContext
        {
            Probe = probe,
            Hyperparams = new ModelHyperparams { EmbeddingDim = 128, NumLayers = 2, HeadDim = 64, VocabSize = 256, NumHeads = 2, NumKvHeads = 2 },
            Plan = plan,
        };

        var fwd = desc.ConstructForwardPass(loadCtx);
        Assert.NotNull(fwd);
        Assert.Same(plan, capturedPlan);
        Assert.Equal(256, loadCtx.ContextSize);
        Assert.Equal(4, loadCtx.GpuLayers);
    }

    private sealed class MockForwardPass : IForwardPass
    {
        public int VocabSize => 256;
        public int MaxSeqLen => 512;
        public ReadOnlySpan<float> Forward(int token, int position) => ReadOnlySpan<float>.Empty;
        public ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, int startPos = 0) => ReadOnlySpan<float>.Empty;
        public void TruncateTo(int length) { }
        public void ResetCache() { }
        public void Dispose() { }
    }
}
