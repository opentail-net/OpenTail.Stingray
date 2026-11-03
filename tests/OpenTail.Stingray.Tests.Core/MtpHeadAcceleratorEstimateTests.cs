using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Engine.Planning;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// The Vulkan hybrid-GDN pass uploads the MTP/NEXTN head's weights and a head-local F32 K/V cache whenever the checkpoint has one, so the
/// planned accelerator figure must include them (and only then).
/// </summary>
public sealed class MtpHeadAcceleratorEstimateTests
{
    private static ModelPlanningFacts Facts(long headBytes = 100, bool hybrid = true) => new()
    {
        NumLayers = 4, ContextLength = 2048, EmbeddingDim = 128, HeadDim = 64, NumHeads = 2, NumKvHeads = 2, IntermediateDim = 256, VocabSize = 100,
        IsHybridSsm = hybrid, MtpHeadGpuBytes = headBytes,
    };

    [Fact]
    public void VulkanHybridWithAHead_CountsWeightsPlusF32KvForTheContext()
    {
        long kv = 2L * 1000 * 2 * 64 * sizeof(float);

        Assert.Equal(100 + kv, ExecutionPlanner.MtpHeadAcceleratorBytes(Facts(), ForwardPassBackend.Vulkan, gpuLayers: 8, contextSize: 1000));
    }

    [Theory]
    [InlineData(ForwardPassBackend.Cpu, 8, 100, true)]
    [InlineData(ForwardPassBackend.Cuda, 8, 100, true)]
    [InlineData(ForwardPassBackend.Vulkan, 0, 100, true)]
    [InlineData(ForwardPassBackend.Vulkan, 8, 0, true)]
    [InlineData(ForwardPassBackend.Vulkan, 8, 100, false)]
    public void ZeroUnlessAVulkanHybridPlanPutsLayersOnTheGpuAndTheModelHasAHead(ForwardPassBackend backend, int gpuLayers, long headBytes, bool hybrid)
    {
        Assert.Equal(0, ExecutionPlanner.MtpHeadAcceleratorBytes(Facts(headBytes, hybrid), backend, gpuLayers, contextSize: 1000));
    }
}
