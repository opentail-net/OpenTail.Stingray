using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Server;

/// <summary>
/// Bridges the CPU-prefill receipt created during lazy model load to diagnostic endpoints.
/// Like <see cref="TokenizerRelay"/>, this has no inference-loop role.
/// </summary>
public sealed class CpuPrefillRuntimeReceiptRelay
{
    private CpuBatchedPrefillCapability? _capability;

    /// <summary>Model-level regular CPU batched-prefill capability, or null for non-CPU/custom engines.</summary>
    public CpuBatchedPrefillCapability? Capability => Volatile.Read(ref _capability);

    internal void Set(CpuBatchedPrefillCapability? capability) =>
        Interlocked.CompareExchange(ref _capability, capability, null);
}
