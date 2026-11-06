using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Mlx;
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
    public void CreateDense_MlxUnavailable_RefusesBeforeLoadingModel()
    {
        Assert.SkipUnless(!MlxBackend.IsAvailable(), "This check is for hosts without a working MLX/Metal runtime.");
        string? originalFlag = Environment.GetEnvironmentVariable("STINGRAY_MLX_DENSE");
        Environment.SetEnvironmentVariable("STINGRAY_MLX_DENSE", "1");
        try
        {
            var source = new FakeModelTensorSource();
            var hp = new ModelHyperparams { EmbeddingDim = 128, NumLayers = 2, VocabSize = 100, HeadDim = 64 };
            var plan = ExecutionPlan.CreateSynthesized(
                architecture: "llama",
                decision: new ForwardPassDecision(ForwardPassKind.MlxDense, null),
                backend: ForwardPassBackend.Mlx,
                contextSize: 512,
                gpuLayers: 0,
                headDim: 64);
            var loadCtx = new ArchitectureLoadContext
            {
                Probe = new ArchitectureProbe { Architecture = "llama", TensorSource = source },
                Hyperparams = hp,
                Plan = plan,
            };

            var ex = Assert.Throws<NotSupportedException>(() => CommonForwardPassFactory.CreateDense(loadCtx));
            Assert.Contains("MLX backend is unavailable", ex.Message);
        }
        finally
        {
            Environment.SetEnvironmentVariable("STINGRAY_MLX_DENSE", originalFlag);
        }
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
            },
            Hyperparams = hp,
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
            },
            Hyperparams = hp,
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
