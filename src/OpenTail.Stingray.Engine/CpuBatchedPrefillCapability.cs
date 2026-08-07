namespace OpenTail.Stingray.Engine;

/// <summary>
/// Model-level receipt for the regular CPU batched-prefill trunk.
/// It deliberately describes only conditions known at load time. A one-token prompt and an
/// all-control-token prompt still take sequential execution, and individual weight routes may
/// choose a non-Q8 kernel; callers must not present this as a per-request kernel trace.
/// </summary>
public sealed record CpuBatchedPrefillCapability(bool Available, string Detail)
{
    /// <summary>Evaluates the supported load-time exclusions of <see cref="ForwardPass.Prefill"/>.</summary>
    public static CpuBatchedPrefillCapability Evaluate(
        bool turboQuantEnabled,
        bool isMoe,
        bool moeBatchedPrefillSupported,
        bool perLayerHeadDimUnsupported)
    {
        if (isMoe && !moeBatchedPrefillSupported)
            return new(false, "This MoE configuration uses sequential prefill because its batched trunk is unsupported.");
        if (perLayerHeadDimUnsupported)
            return new(false,
                "This per-layer-head-dimension model uses sequential prefill because the CPU batched trunk lacks its required attention features.");
        if (turboQuantEnabled)
            return new(false, "TurboQuant uses its dedicated CPU prefill implementation, not the regular batched-prefill trunk.");

        return new(true,
            "The regular CPU batched-prefill trunk is available for ordinary multi-token prompts; control-only prompts still use sequential F32.");
    }
}
