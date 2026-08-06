
namespace OpenTail.Stingray.Tests.Pipeline;

public sealed class MemoryHierarchyTests
{
    private static OpenTail.Stingray.Pipeline.TierConfig MakeTier(string name) =>
        new(name, CapacityBytes: 1L << 30);

    [Fact]
    public void TierConfig_StoresNameAndCapacity()
    {
        var tier = new OpenTail.Stingray.Pipeline.TierConfig("gpu", CapacityBytes: 8L << 30);
        Assert.Equal("gpu", tier.Name);
        Assert.Equal(8L << 30, tier.CapacityBytes);
        Assert.Null(tier.MmapPath);
    }

    [Fact]
    public void TierConfig_MmapPath_DefaultsToNullButCanBeSet()
    {
        var tier = new OpenTail.Stingray.Pipeline.TierConfig("nvme", CapacityBytes: 1L << 40, MmapPath: "/mnt/nvme/cache");
        Assert.Equal("/mnt/nvme/cache", tier.MmapPath);
    }

    [Fact]
    public void TierConfig_RecordEquality_ComparesByValue()
    {
        var a = new OpenTail.Stingray.Pipeline.TierConfig("cpu", CapacityBytes: 16L << 30);
        var b = new OpenTail.Stingray.Pipeline.TierConfig("cpu", CapacityBytes: 16L << 30);
        Assert.Equal(a, b);
    }

    [Fact]
    public async System.Threading.Tasks.Task MemoryHierarchy_Construction_DoesNotThrow()
    {
        await using var hierarchy = new OpenTail.Stingray.Pipeline.MemoryHierarchy(
            MakeTier("gpu"), MakeTier("cpu"), MakeTier("nvme"));
        Assert.NotNull(hierarchy);
    }

    // ── Not-yet-implemented surface ─────────────────────────────────────────
    // MemoryHierarchy's promotion/eviction logic (issue tracked in source TODOs) is not
    // implemented yet. These tests pin down today's actual contract — an explicit
    // NotImplementedException, not a silent no-op or a different exception type — so a
    // future implementation change is a deliberate, visible test update rather than a
    // surprise.

    [Fact]
    public async System.Threading.Tasks.Task PromoteToGpuAsync_NotYetImplemented_ThrowsNotImplementedException()
    {
        await using var hierarchy = new OpenTail.Stingray.Pipeline.MemoryHierarchy(
            MakeTier("gpu"), MakeTier("cpu"), MakeTier("nvme"));
        await Assert.ThrowsAsync<System.NotImplementedException>(
            async () => await hierarchy.PromoteToGpuAsync("layer0.attn.q_proj"));
    }

    [Fact]
    public async System.Threading.Tasks.Task EvictFromGpuAsync_NotYetImplemented_ThrowsNotImplementedException()
    {
        await using var hierarchy = new OpenTail.Stingray.Pipeline.MemoryHierarchy(
            MakeTier("gpu"), MakeTier("cpu"), MakeTier("nvme"));
        await Assert.ThrowsAsync<System.NotImplementedException>(
            async () => await hierarchy.EvictFromGpuAsync());
    }

    [Fact]
    public async System.Threading.Tasks.Task DisposeAsync_CompletesWithoutThrowing()
    {
        var hierarchy = new OpenTail.Stingray.Pipeline.MemoryHierarchy(
            MakeTier("gpu"), MakeTier("cpu"), MakeTier("nvme"));
        await hierarchy.DisposeAsync();
    }
}
