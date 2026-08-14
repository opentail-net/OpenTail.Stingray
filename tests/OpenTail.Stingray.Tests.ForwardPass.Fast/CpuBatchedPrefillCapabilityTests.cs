using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>Locks the load-time exclusions published by the CPU prefill diagnostic.</summary>
public sealed class CpuBatchedPrefillCapabilityTests
{
    [Fact]
    public void Evaluate_AdmitsOrdinaryDenseCpuModels()
    {
        var capability = CpuBatchedPrefillCapability.Evaluate(
            turboQuantEnabled: false, isMoe: false, moeBatchedPrefillSupported: false,
            perLayerHeadDimUnsupported: false);

        Assert.True(capability.Available);
        Assert.Contains("ordinary multi-token", capability.Detail, StringComparison.Ordinal);
    }

    [Fact]
    public void Evaluate_ExplainsEveryModelLevelSequentialFallback()
    {
        var moe = CpuBatchedPrefillCapability.Evaluate(false, true, false, false);
        var perLayer = CpuBatchedPrefillCapability.Evaluate(false, false, false, true);
        var turboQuant = CpuBatchedPrefillCapability.Evaluate(true, false, false, false);

        Assert.False(moe.Available);
        Assert.Contains("MoE", moe.Detail, StringComparison.Ordinal);
        Assert.False(perLayer.Available);
        Assert.Contains("per-layer-head-dimension", perLayer.Detail, StringComparison.Ordinal);
        Assert.False(turboQuant.Available);
        Assert.Contains("TurboQuant", turboQuant.Detail, StringComparison.Ordinal);
    }
}
