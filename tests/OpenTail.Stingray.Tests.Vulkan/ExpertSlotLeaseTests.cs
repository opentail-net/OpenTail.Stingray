namespace OpenTail.Stingray.Tests.Vulkan;

/// <summary>
/// A lease pins an expert slot until <see cref="ExpertSlotManager.ReleaseLeases"/> (streamed-residency plan, defect 3): a slot
/// a forward pass has looked up must not be evicted, and its GPU tensors freed, by promotions that happen before the GPU has
/// used it. Real OLMoE weights and a real Vulkan device; skips visibly without either. Mutation: replacing
/// <c>TryGetCachedLeased</c> with <c>TryGetCached</c> in the first step must fail the first assertion.
/// </summary>
public sealed class ExpertSlotLeaseTests : HeavyTestBase
{
    private const string Olmoe = "OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf";
    private const int Capacity = 8;

    [Fact]
    public void LeasedSlot_SurvivesEvictionPressure_UntilReleased()
    {
        string? path = OpenTail.Stingray.Engine.Verification.ModelLocator.FindDeep(Olmoe);
        Assert.SkipWhen(path is null, $"{Olmoe} not present");
        VulkanBackend? gpu;
        try { gpu = new VulkanBackend(); } catch { gpu = null; }
        Assert.SkipWhen(gpu is null, "no Vulkan device available on this host");
        using var _gpu = gpu;

        using var model = GgufModel.Open(path!);
        var hp = ArchitectureModelResolver.ResolveHyperparams(model);
        var dtypes = new Dictionary<nint, DType>();
        using var manager = new ExpertSlotManager(gpu!, model, hp, Capacity, dtypes);

        // The slot is looked up once (one access), so it is the least-frequently-used resident. The cache is frequency-aware
        // (ExpertCache's victim is the least-accessed probationary entry), so the churn below looks every other expert up
        // three times: without the lease the slot is the victim, and the first assertion can fail.
        manager.Preload(0, 0);
        Assert.True(manager.TryGetCachedLeased(0, 0, out _), "the leased lookup must hit a preloaded expert");

        PromoteDistinct(manager, layer: 1, count: 2 * Capacity);
        // Checked once, after the churn: each check is itself an access, so checking inside the loop would change the outcome.
        Assert.True(manager.TryGetCached(0, 0, out _), "a leased slot was evicted before its lease was released");

        manager.ReleaseLeases();
        // The same kind of pressure, with no lease, must be able to evict it: otherwise the first assertion proves nothing.
        PromoteDistinct(manager, layer: 2, count: 4 * Capacity);
        Assert.False(manager.TryGetCached(0, 0, out _), "a released slot should be evictable under the same pressure");
    }

    /// <summary>Loads <paramref name="count"/> distinct experts of one layer and looks each one up three times. A hit promotes an
    /// entry to the protected segment, which is where eviction bites once it is full.</summary>
    private static void PromoteDistinct(ExpertSlotManager manager, int layer, int count)
    {
        for (int e = 0; e < count; e++)
        {
            manager.Preload(layer, e);
            for (int hit = 0; hit < 3; hit++)
                manager.TryGetCached(layer, e, out _);
        }
    }
}
