using OpenTail.Stingray.Vulkan;

namespace OpenTail.Stingray.Server;

/// <summary>
/// Point-in-time resource snapshot (docs/032-multi-model-inference-runtime-plan.md §"Resource
/// admission"). Host and accelerator memory are kept as separate figures deliberately — never
/// collapse them into one number. <see cref="AcceleratorMemoryAvailableBytes"/> is <c>null</c>
/// when no accelerator was detected (or querying it isn't supported/enabled) — treat <c>null</c>
/// as "unknown", not zero. When non-null, it reports total DEVICE_LOCAL VRAM <em>capacity</em>
/// (see <see cref="HostResourceBudget"/>'s class doc) — <see cref="EstimatedAcceleratorResidentBytes"/>
/// is the separate, self-tracked figure <see cref="IResourceBudget.EstimateAdmission"/> subtracts
/// from it to get a true "available for a new candidate" number, since Vulkan has no live
/// "free VRAM right now" query the way host memory's <see cref="HostMemoryAvailableBytes"/> does.
/// </summary>
public readonly record struct ResourceSnapshot(
    long HostMemoryTotalBytes,
    long HostMemoryAvailableBytes,
    long? AcceleratorMemoryAvailableBytes,
    long ResidentModelBytes,
    long EstimatedAcceleratorResidentBytes);

/// <summary>
/// Thrown by <see cref="ModelRuntimeManager.AcquireAsync"/> when a cold load's candidate model
/// still doesn't pass <see cref="IResourceBudget.EstimateAdmission"/> even after evicting every
/// idle/evictable resident runtime — the documented hard-failure last resort in docs/032's
/// eviction hierarchy. Only reachable when <see cref="IModelRuntimeManager.ResourceBudget"/> is
/// explicitly set; the default (<c>null</c>) never throws this.
/// </summary>
public sealed class InsufficientResourcesException : InvalidOperationException
{
    public ModelId Model { get; }
    public long CandidateModelBytes { get; }
    public long CandidateAcceleratorBytes { get; }
    public ResourceAdmission Reason { get; }
    public ResourceSnapshot Snapshot { get; }

    public InsufficientResourcesException(
        ModelId model, long candidateModelBytes, long candidateAcceleratorBytes,
        ResourceAdmission reason, ResourceSnapshot snapshot)
        : base(BuildMessage(model, candidateModelBytes, candidateAcceleratorBytes, reason, snapshot))
    {
        Model = model;
        CandidateModelBytes = candidateModelBytes;
        CandidateAcceleratorBytes = candidateAcceleratorBytes;
        Reason = reason;
        Snapshot = snapshot;
    }

    private static string BuildMessage(
        ModelId model, long candidateModelBytes, long candidateAcceleratorBytes,
        ResourceAdmission reason, ResourceSnapshot snapshot) => reason switch
    {
        ResourceAdmission.InsufficientAcceleratorMemory =>
            $"Cannot admit model '{model}' (~{candidateAcceleratorBytes:N0} accelerator bytes " +
            $"estimated) — insufficient VRAM even after evicting idle resident models " +
            $"(capacity: {snapshot.AcceleratorMemoryAvailableBytes:N0} bytes, currently " +
            $"accelerator-resident: {snapshot.EstimatedAcceleratorResidentBytes:N0} bytes).",
        _ =>
            $"Cannot admit model '{model}' (~{candidateModelBytes:N0} bytes estimated) — insufficient " +
            $"host memory even after evicting idle resident models (available: " +
            $"{snapshot.HostMemoryAvailableBytes:N0} bytes, currently resident: " +
            $"{snapshot.ResidentModelBytes:N0} bytes).",
    };
}

/// <summary>Result of <see cref="IResourceBudget.EstimateAdmission"/>.</summary>
public enum ResourceAdmission
{
    Allowed,
    InsufficientHostMemory,
    InsufficientAcceleratorMemory,
}

/// <summary>
/// Reports current resource availability and, given a candidate's estimated weight, whether it
/// looks admissible. Still purely advisory — nothing calls <see cref="EstimateAdmission"/> from
/// <see cref="ModelRuntimeManager.AcquireAsync"/> unless <see cref="IModelRuntimeManager.ResourceBudget"/>
/// is explicitly set (off by default).
/// </summary>
public interface IResourceBudget
{
    ResourceSnapshot GetCurrent();

    /// <summary>
    /// Conservative admission estimate for a candidate model of <paramref name="candidateModelBytes"/>
    /// resident weight and, optionally, <paramref name="candidateAcceleratorBytes"/> of expected
    /// accelerator (VRAM) footprint. Deliberately narrower than docs/032's full
    /// <c>ResourceAvailability EstimateAdmission(ModelRuntimeSpec, InferenceWorkEstimate)</c>
    /// sketch — KV/session-aware estimation needs types that don't exist yet; this is the
    /// weight-only slice of that seam. <paramref name="candidateAcceleratorBytes"/> defaults to 0
    /// ("not expected to be accelerator-resident") — a caller that doesn't know or care leaves
    /// accelerator admission untouched, exactly matching this parameter's absence before it existed.
    /// </summary>
    ResourceAdmission EstimateAdmission(long candidateModelBytes, long candidateAcceleratorBytes = 0);
}

/// <summary>
/// <see cref="IResourceBudget"/> backed by <see cref="GC.GetGCMemoryInfo"/> for host memory —
/// portable across Windows/Linux/macOS with no P/Invoke, since the runtime already queries the OS
/// for this internally to size the GC heap. <see cref="GCMemoryInfo.TotalAvailableMemoryBytes"/>
/// is the memory limit the GC believes it can use (a solid proxy for total physical/container
/// memory); subtracting <see cref="GCMemoryInfo.MemoryLoadBytes"/> (current system-wide physical
/// memory in use, as the GC observes it) gives an approximate free-memory figure. This is
/// deliberately an estimate, not an exact accounting — see docs/032 §"Resource admission" on why
/// Phase 1/2's admission story stays conservative rather than exact.
///
/// <para>Accelerator memory is reported when a Vulkan device is present, via a real
/// <see cref="VulkanBackend.VramBytes"/> query (device-local heap capacity — the exact figure
/// <c>TierPlanner</c> already budgets against for model placement). Unlike host memory, Vulkan
/// has no portable "free VRAM right now" query without the <c>VK_EXT_memory_budget</c> extension,
/// so <see cref="ResourceSnapshot.AcceleratorMemoryAvailableBytes"/> is total capacity, not
/// capacity-minus-current-usage — <see cref="EstimateAdmission"/> does that subtraction itself
/// using <see cref="ModelRuntime.AcceleratorResidentBytesEstimate"/>, summed live across resident
/// runtimes on every call rather than cached, so it can never go stale relative to eviction/new
/// loads. The capacity probe constructs a throwaway <see cref="VulkanBackend"/> purely to read
/// this static hardware property and disposes it immediately — same one-shot pattern
/// <c>DoctorCommand</c>'s Vulkan smoke test already uses — then caches that one result
/// process-wide (<see cref="Lazy{T}"/>) since VRAM capacity cannot change during a process's
/// lifetime and repeated device probes would be wasteful. No Vulkan device (or a probe failure)
/// leaves <see cref="ResourceSnapshot.AcceleratorMemoryAvailableBytes"/> <c>null</c> —
/// "unknown", not zero, matching the same convention host memory already follows on probe
/// failure paths elsewhere in this codebase. A candidate with a nonzero
/// <c>candidateAcceleratorBytes</c> is never rejected on accelerator grounds when capacity is
/// unknown — there is no positive evidence it won't fit, so admission falls through to the host
/// check alone rather than blocking on missing data.</para>
/// </summary>
public sealed class HostResourceBudget(IModelRuntimeManager modelRuntimes, double safetyMarginMultiplier = 1.25)
    : IResourceBudget
{
    private static readonly Lazy<long?> s_acceleratorVramCapacityBytes = new(ProbeAcceleratorVramCapacity);

    private static long? ProbeAcceleratorVramCapacity()
    {
        try
        {
            using var vk = new VulkanBackend();
            return (long)vk.VramBytes;
        }
        catch (Exception ex) when (ex is not OutOfMemoryException)
        {
            // No Vulkan device, driver/SDK unavailable, or init failed for any other reason —
            // accelerator memory stays "unknown", exactly like DoctorCommand's own Vulkan probe
            // treats this as a non-fatal "not available" rather than surfacing the exception.
            return null;
        }
    }

    public ResourceSnapshot GetCurrent()
    {
        var gc = GC.GetGCMemoryInfo();
        long total = gc.TotalAvailableMemoryBytes;
        long available = Math.Max(0, total - gc.MemoryLoadBytes);

        long residentBytes = 0;
        long acceleratorResidentBytes = 0;
        foreach (var stats in modelRuntimes.Snapshot())
        {
            residentBytes += stats.EstimatedModelBytes;
            acceleratorResidentBytes += stats.AcceleratorResidentBytesEstimate;
        }

        return new ResourceSnapshot(
            HostMemoryTotalBytes: total,
            HostMemoryAvailableBytes: available,
            AcceleratorMemoryAvailableBytes: s_acceleratorVramCapacityBytes.Value,
            ResidentModelBytes: residentBytes,
            EstimatedAcceleratorResidentBytes: acceleratorResidentBytes);
    }

    /// <summary>
    /// Requires <paramref name="candidateModelBytes"/> * <paramref name="safetyMarginMultiplier"/>
    /// (default 1.25 — 25% headroom) to fit in currently available host memory, and — only when
    /// <paramref name="candidateAcceleratorBytes"/> is positive AND accelerator capacity is known
    /// — the same margin applied to <paramref name="candidateAcceleratorBytes"/> to fit in
    /// capacity minus what's already accelerator-resident. The margin exists because the raw
    /// weight estimate covers neither KV/workspace allocations nor runtime overhead (docs/032
    /// §"Resource admission": "Phase 1 admission is an estimate, not an exact allocator");
    /// biasing toward under-admitting is the deliberate, conservative default rather than trying
    /// to model every real allocation. Host is checked first — an accelerator-bound candidate
    /// that already fails on host memory is reported as a host failure, not silently overwritten
    /// by a later accelerator check.
    /// </summary>
    public ResourceAdmission EstimateAdmission(long candidateModelBytes, long candidateAcceleratorBytes = 0)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(candidateModelBytes);
        ArgumentOutOfRangeException.ThrowIfNegative(candidateAcceleratorBytes);

        var snapshot = GetCurrent();

        long requiredHost = (long)(candidateModelBytes * safetyMarginMultiplier);
        if (requiredHost > snapshot.HostMemoryAvailableBytes)
            return ResourceAdmission.InsufficientHostMemory;

        if (candidateAcceleratorBytes > 0 && snapshot.AcceleratorMemoryAvailableBytes is { } capacity)
        {
            long requiredAccel = (long)(candidateAcceleratorBytes * safetyMarginMultiplier);
            long availableAccel = Math.Max(0, capacity - snapshot.EstimatedAcceleratorResidentBytes);
            if (requiredAccel > availableAccel)
                return ResourceAdmission.InsufficientAcceleratorMemory;
        }

        return ResourceAdmission.Allowed;
    }
}
