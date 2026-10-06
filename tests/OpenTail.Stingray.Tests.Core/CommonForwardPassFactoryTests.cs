using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using Xunit;

namespace OpenTail.Stingray.Tests.Core;

public sealed class CommonForwardPassFactoryTests
{
    [Fact]
    public void CreateDense_ThrowsOnNullContext()
    {
        Assert.Throws<ArgumentNullException>(() => CommonForwardPassFactory.CreateDense(null!));
    }

    [Fact]
    public void CreateHybridGdn_ThrowsOnNullContext()
    {
        Assert.Throws<ArgumentNullException>(() => CommonForwardPassFactory.CreateHybridGdn(null!));
    }

    [Fact]
    public void CreateDense_ThrowsOnUnsupportedKind()
    {
        var source = new FakeModelTensorSource();
        var hp = new ModelHyperparams { EmbeddingDim = 128, NumLayers = 2, VocabSize = 100, HeadDim = 64 };
        var plan = ExecutionPlan.CreateSynthesized(
            architecture: "custom",
            decision: new ForwardPassDecision(ForwardPassKind.RwkvCpu, null),
            backend: ForwardPassBackend.Cpu,
            contextSize: 512,
            gpuLayers: 0,
            headDim: 64);

        var loadCtx = new ArchitectureLoadContext
        {
            Probe = new ArchitectureProbe
            {
                Architecture = "custom",
                TensorSource = source,
                Hyperparams = hp,
            },
            Plan = plan,
        };

        var ex = Assert.Throws<InvalidOperationException>(() => CommonForwardPassFactory.CreateDense(loadCtx));
        Assert.Contains("Unsupported forward pass kind", ex.Message);
    }

    [Fact]
    public void CreateHybridGdn_ThrowsOnUnsupportedKind()
    {
        var source = new FakeModelTensorSource();
        var hp = new ModelHyperparams { EmbeddingDim = 128, NumLayers = 2, VocabSize = 100, HeadDim = 64 };
        var plan = ExecutionPlan.CreateSynthesized(
            architecture: "custom",
            decision: new ForwardPassDecision(ForwardPassKind.CpuDense, null),
            backend: ForwardPassBackend.Cpu,
            contextSize: 512,
            gpuLayers: 0,
            headDim: 64);

        var loadCtx = new ArchitectureLoadContext
        {
            Probe = new ArchitectureProbe
            {
                Architecture = "custom",
                TensorSource = source,
                Hyperparams = hp,
            },
            Plan = plan,
        };

        var ex = Assert.Throws<InvalidOperationException>(() => CommonForwardPassFactory.CreateHybridGdn(loadCtx));
        Assert.Contains("Unsupported forward pass kind", ex.Message);
    }

    private sealed unsafe class FakeModelTensorSource : IModelTensorSource
    {
        public IReadOnlyList<GgufTensorInfo> Tensors => [];
        public IReadOnlyDictionary<string, object> Metadata => new Dictionary<string, object>();
        public GgufTensorInfo? FindTensor(string name) => null;
        public ReadOnlySpan<byte> GetTensorData(GgufTensorInfo tensor) => [];
        public byte* GetTensorDataPtr(GgufTensorInfo tensor) => null;
    }
}
