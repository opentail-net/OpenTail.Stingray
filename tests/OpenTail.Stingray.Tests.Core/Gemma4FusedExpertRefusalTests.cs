using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using Xunit;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// The stock Gemma 4 26B-A4B GGUF (fused <c>ffn_gate_up_exps</c>) used to die in the dense forward pass with an unhandled "Missing tensor: blk.0.ffn_gate_exps.weight"
/// while the registry called the family admitted. It is now refused with a message that names the layout (checked 2026-11-01 against the real file).
/// </summary>
public sealed class Gemma4FusedExpertRefusalTests
{
    [Fact]
    public void FusedExpertLayout_IsRefusedWithAClearMessage()
    {
        var source = new FakeSource(Gemma4Architecture.FusedExpertTensor);

        var ex = Assert.Throws<NotSupportedException>(() => Gemma4Architecture.RejectFusedExpertLayout(source));
        Assert.Contains("fused expert tensors", ex.Message);
        Assert.Contains("ffn_gate_up_exps", ex.Message);
    }

    [Fact]
    public void DenseLayout_PassesTheCheck()
    {
        Gemma4Architecture.RejectFusedExpertLayout(new FakeSource(null));
    }

    [Fact]
    public void DescriptorRoutesThroughTheRefusal()
    {
        var ctx = new ArchitectureLoadContext
        {
            Probe = new ArchitectureProbe { Architecture = "gemma4", TensorSource = new FakeSource(Gemma4Architecture.FusedExpertTensor) },
            Hyperparams = new ModelHyperparams { EmbeddingDim = 128, NumLayers = 2, VocabSize = 100, HeadDim = 64 },
            Plan = ExecutionPlan.CreateSynthesized(architecture: "gemma4", decision: new ForwardPassDecision(ForwardPassKind.CpuDense, null), backend: ForwardPassBackend.Cpu, contextSize: 512, gpuLayers: 0, headDim: 64),
        };

        Assert.Throws<NotSupportedException>(() => Gemma4Architecture.Descriptor.CreateForwardPass!(ctx));
    }

    private sealed unsafe class FakeSource(string? tensorName) : IModelTensorSource
    {
        public IReadOnlyList<GgufTensorInfo> Tensors => [];
        public IReadOnlyDictionary<string, object> Metadata => new Dictionary<string, object>();
        public GgufTensorInfo? FindTensor(string name) =>
            name == tensorName ? new GgufTensorInfo(name, 1, [1], DType.Float32, 0) : null;
        public ReadOnlySpan<byte> GetTensorData(GgufTensorInfo tensor) => [];
        public byte* GetTensorDataPtr(GgufTensorInfo tensor) => null;
    }
}
