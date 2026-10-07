namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>
/// <see cref="WarmPinConfig"/> is pure: the override comes from an instance's <see cref="MoePlan"/>, never from a static
/// environment read, so these assertions are deterministic and two configurations can coexist.
/// </summary>
public sealed class WarmPinConfigTests
{
    [Fact]
    public void ResolvePerLayer_CacheHoldsFullSet_DisablesWarmPin() =>
        Assert.Equal(0, WarmPinConfig.ResolvePerLayer(0, numLayers: 8, numExperts: 4, numActiveExperts: 2, slotCapacity: 32));

    [Fact]
    public void ResolvePerLayer_TightCache_AutoEnablesAtActiveExperts() =>
        Assert.Equal(8, WarmPinConfig.ResolvePerLayer(0, numLayers: 8, numExperts: 64, numActiveExperts: 8, slotCapacity: 16));

    [Fact]
    public void ResolvePerLayer_DenseRoute_FallsBackToOne() =>
        Assert.Equal(1, WarmPinConfig.ResolvePerLayer(0, numLayers: 4, numExperts: 8, numActiveExperts: 0, slotCapacity: 4));

    [Fact]
    public void ResolvePerLayer_CapsAtNumExperts() =>
        Assert.Equal(8, WarmPinConfig.ResolvePerLayer(0, numLayers: 4, numExperts: 8, numActiveExperts: 16, slotCapacity: 4));

    [Fact]
    public void TwoPlans_WithDifferentOverrides_DoNotInterfere()
    {
        var a = new MoePlan(true, WarmPin: 3, WarmPinAfter: 100);
        var b = new MoePlan(true, WarmPin: 7, WarmPinAfter: 900);

        Assert.Equal(3, WarmPinConfig.ResolvePerLayer(a, 8, 64, 8, 16));
        Assert.Equal(7, WarmPinConfig.ResolvePerLayer(b, 8, 64, 8, 16));
        Assert.Equal(3, WarmPinConfig.ResolvePerLayer(a, 8, 64, 8, 16)); // B did not change A
        Assert.Equal(100, WarmPinConfig.ResolveAfterAccesses(a));
        Assert.Equal(900, WarmPinConfig.ResolveAfterAccesses(b));
        Assert.Equal(WarmPinConfig.DefaultAfterAccesses, WarmPinConfig.ResolveAfterAccesses(null));
    }
}
