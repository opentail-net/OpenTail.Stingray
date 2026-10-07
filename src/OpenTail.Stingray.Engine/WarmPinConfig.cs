namespace OpenTail.Stingray.Engine;

/// <summary>
/// Pure resolution of the warm-pinning configuration. The inputs come from <see cref="MoePlan"/>
/// (<c>WarmPin</c> / <c>WarmPinAfter</c>, set from <c>--moe-warmpin[-after]</c> or the inherited
/// <c>STINGRAY_MOE_WARMPIN[_AFTER]</c> via the request bridge); nothing here reads the process environment,
/// so two runtime instances with different settings cannot affect each other.
/// <list type="bullet">
///   <item><c>perLayerOverride</c> — hottest experts to pin per layer. <c>0</c>/null leaves warm-pinning to the
///     auto-enable rule; a positive value forces it.</item>
///   <item><c>afterAccesses</c> — expert accesses observed before the warm set is chosen (default 512).</item>
/// </list>
/// With no override, <see cref="ResolvePerLayer"/> auto-enables warm-pinning at <c>NumActiveExperts</c> per layer
/// whenever the SLRU slot capacity is smaller than the total expert count — the regime where eviction churn happens.
/// Shared by <see cref="ExpertSlotManager"/> (Vulkan) and <see cref="CudaExpertSlotManager"/>.
/// </summary>
internal static class WarmPinConfig
{
    public const long DefaultAfterAccesses = 512;

    public static long ResolveAfterAccesses(MoePlan? moe) =>
        moe?.WarmPinAfter is int after && after > 0 ? after : DefaultAfterAccesses;

    public static int ResolvePerLayer(MoePlan? moe, int numLayers, int numExperts, int numActiveExperts, int slotCapacity) =>
        ResolvePerLayer(moe?.WarmPin ?? 0, numLayers, numExperts, numActiveExperts, slotCapacity);

    public static int ResolvePerLayer(int perLayerOverride, int numLayers, int numExperts, int numActiveExperts, int slotCapacity)
    {
        if (perLayerOverride > 0) return perLayerOverride;
        long total = (long)numLayers * numExperts;
        if (slotCapacity >= total) return 0;
        // numActiveExperts can be 0 on dense models routed through this path — fall
        // back to 1 so the auto-enable still does something useful, but never more
        // than numExperts (a layer cannot have more pinned experts than it has).
        int perLayer = numActiveExperts > 0 ? numActiveExperts : 1;
        return Math.Min(perLayer, numExperts);
    }
}
