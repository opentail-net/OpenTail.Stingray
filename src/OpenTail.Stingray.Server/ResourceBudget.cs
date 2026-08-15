namespace OpenTail.Stingray.Server;

/// <summary>
/// Point-in-time resource snapshot (docs/032-multi-model-inference-runtime-plan.md §"Resource
/// admission"). Host and accelerator memory are kept as separate figures deliberately — never
/// collapse them into one number. <see cref="AcceleratorMemoryAvailableBytes"/> is <c>null</c>
/// until a later phase wires in real VRAM accounting; treat <c>null</c> as "unknown", not zero.
/// </summary>
public readonly record struct ResourceSnapshot(
    long HostMemoryTotalBytes,
    long HostMemoryAvailableBytes,
    long? AcceleratorMemoryAvailableBytes,
    long ResidentModelBytes);

/// <summary>
/// Reports current resource availability. This is the observation half of docs/032's admission
/// design only — <see cref="GetCurrent"/> is read-only and has no say in whether an acquisition
/// proceeds. Wiring a snapshot into an actual admit/evict/queue decision inside
/// <see cref="ModelRuntimeManager.AcquireAsync"/> is separate, not-yet-built work; introducing
/// that here would risk the existing single-model path for no tested benefit.
/// </summary>
public interface IResourceBudget
{
    ResourceSnapshot GetCurrent();
}

/// <summary>
/// <see cref="IResourceBudget"/> backed by <see cref="GC.GetGCMemoryInfo"/> — portable across
/// Windows/Linux/macOS with no P/Invoke, since the runtime already queries the OS for this
/// internally to size the GC heap. <see cref="GCMemoryInfo.TotalAvailableMemoryBytes"/> is the
/// memory limit the GC believes it can use (a solid proxy for total physical/container memory);
/// subtracting <see cref="GCMemoryInfo.MemoryLoadBytes"/> (current system-wide physical memory in
/// use, as the GC observes it) gives an approximate free-memory figure. This is deliberately an
/// estimate, not an exact accounting — see docs/032 §"Resource admission" on why Phase 1/2's
/// admission story stays conservative rather than exact.
/// </summary>
public sealed class HostResourceBudget(IModelRuntimeManager modelRuntimes) : IResourceBudget
{
    public ResourceSnapshot GetCurrent()
    {
        var gc = GC.GetGCMemoryInfo();
        long total = gc.TotalAvailableMemoryBytes;
        long available = Math.Max(0, total - gc.MemoryLoadBytes);

        long residentBytes = 0;
        foreach (var stats in modelRuntimes.Snapshot())
            residentBytes += stats.EstimatedModelBytes;

        return new ResourceSnapshot(
            HostMemoryTotalBytes: total,
            HostMemoryAvailableBytes: available,
            AcceleratorMemoryAvailableBytes: null, // not wired yet — see class doc
            ResidentModelBytes: residentBytes);
    }
}
