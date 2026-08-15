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
/// Thrown by <see cref="ModelRuntimeManager.AcquireAsync"/> when a cold load's candidate model
/// still doesn't pass <see cref="IResourceBudget.EstimateAdmission"/> even after evicting every
/// idle/evictable resident runtime — the documented hard-failure last resort in docs/032's
/// eviction hierarchy. Only reachable when <see cref="IModelRuntimeManager.ResourceBudget"/> is
/// explicitly set; the default (<c>null</c>) never throws this.
/// </summary>
public sealed class InsufficientResourcesException(ModelId model, long candidateModelBytes, ResourceSnapshot snapshot)
    : InvalidOperationException(
        $"Cannot admit model '{model}' (~{candidateModelBytes:N0} bytes estimated) — insufficient " +
        $"host memory even after evicting idle resident models (available: " +
        $"{snapshot.HostMemoryAvailableBytes:N0} bytes, currently resident: " +
        $"{snapshot.ResidentModelBytes:N0} bytes).")
{
    public ModelId Model { get; } = model;
    public long CandidateModelBytes { get; } = candidateModelBytes;
    public ResourceSnapshot Snapshot { get; } = snapshot;
}

/// <summary>Result of <see cref="IResourceBudget.EstimateAdmission"/>.</summary>
public enum ResourceAdmission
{
    Allowed,
    InsufficientHostMemory,
}

/// <summary>
/// Reports current resource availability and, given a candidate's estimated weight, whether it
/// looks admissible. Still purely advisory — nothing calls <see cref="EstimateAdmission"/> from
/// <see cref="ModelRuntimeManager.AcquireAsync"/> yet. Wiring this into an actual
/// admit/evict/queue decision is separate, not-yet-built work; introducing that here would risk
/// the existing single-model path for no tested benefit.
/// </summary>
public interface IResourceBudget
{
    ResourceSnapshot GetCurrent();

    /// <summary>
    /// Conservative host-memory-only admission estimate for a candidate model of
    /// <paramref name="candidateModelBytes"/> resident weight. Deliberately narrower than
    /// docs/032's full <c>ResourceAvailability EstimateAdmission(ModelRuntimeSpec, InferenceWorkEstimate)</c>
    /// sketch — KV/session-aware estimation needs types that don't exist yet; this is the
    /// weight-only slice of that seam.
    /// </summary>
    ResourceAdmission EstimateAdmission(long candidateModelBytes);
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
public sealed class HostResourceBudget(IModelRuntimeManager modelRuntimes, double safetyMarginMultiplier = 1.25)
    : IResourceBudget
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

    /// <summary>
    /// Requires <paramref name="candidateModelBytes"/> * <paramref name="safetyMarginMultiplier"/>
    /// (default 1.25 — 25% headroom) to fit in currently available host memory. The margin exists
    /// because the raw weight estimate covers neither KV/workspace allocations nor runtime
    /// overhead (docs/032 §"Resource admission": "Phase 1 admission is an estimate, not an exact
    /// allocator"); biasing toward under-admitting is the deliberate, conservative default rather
    /// than trying to model every real allocation.
    /// </summary>
    public ResourceAdmission EstimateAdmission(long candidateModelBytes)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(candidateModelBytes);
        long required = (long)(candidateModelBytes * safetyMarginMultiplier);
        return required <= GetCurrent().HostMemoryAvailableBytes
            ? ResourceAdmission.Allowed
            : ResourceAdmission.InsufficientHostMemory;
    }
}
