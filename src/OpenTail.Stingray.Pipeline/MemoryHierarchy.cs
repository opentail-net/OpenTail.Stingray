using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Pipeline;

/// <summary>
/// Three-tier memory hierarchy: GPU VRAM → CPU RAM → NVMe. Intended to manage tensor placement,
/// eviction, and asynchronous prefetching so hot weights stay on the fastest available tier.
///
/// <para><b>Unimplemented scaffolding — internal on purpose.</b> Both operations below throw
/// <see cref="NotImplementedException"/>. It was public and bundled into the shipped
/// <c>OpenTail.Stingray</c> meta-package, so the package advertised a working tier manager that
/// cannot do anything: any caller reaching for it got an exception at the first call, having read
/// documentation that promised promotion and eviction.</para>
///
/// <para>The production three-tier MoE offload path does NOT go through this type. It is
/// <c>ExpertSlotManager</c>/<c>CudaExpertSlotManager</c> plus <c>MoEPrefetcher</c> in
/// OpenTail.Stingray.Engine, over this assembly's <c>SlruCache</c>/<c>ExpertCache</c> — those are
/// implemented, used, and remain public. Make this type public again only when it does something.</para>
/// </summary>
internal sealed class MemoryHierarchy : IAsyncDisposable
{
    private readonly TierConfig _gpu;
    private readonly TierConfig _cpu;
    private readonly TierConfig _nvme;

    public MemoryHierarchy(TierConfig gpu, TierConfig cpu, TierConfig nvme)
    {
        _gpu = gpu;
        _cpu = cpu;
        _nvme = nvme;
    }

    /// <summary>Ensure <paramref name="tensorName"/> is resident on the GPU tier.</summary>
    public ValueTask<Tensor> PromoteToGpuAsync(string tensorName, CancellationToken ct = default)
    {
        // TODO: move tensor up the hierarchy, evicting LRU tensors as needed
        throw new NotImplementedException();
    }

    /// <summary>Evict the least-recently-used tensor from GPU to CPU tier.</summary>
    public ValueTask EvictFromGpuAsync(CancellationToken ct = default)
    {
        // TODO: LRU eviction with dirty-tracking
        throw new NotImplementedException();
    }

    public ValueTask DisposeAsync() => ValueTask.CompletedTask;
}

internal sealed record TierConfig(string Name, long CapacityBytes, string? MmapPath = null);
