namespace OpenTail.Stingray.Server;

/// <summary>
/// The backend and device whose memory a runtime occupies, the key accelerator admission accounts under (plan 6 G2). Two runtimes only compete for
/// the same VRAM when their targets are equal.
/// </summary>
/// <param name="Backend">"cuda" or "vulkan".</param>
/// <param name="DeviceIndex">Vulkan: the index the plan asked for, -1 for "the backend's default pick" (two runtimes both on -1 share a device; -1
/// and an explicit index are counted separately, which can only under-count). CUDA: always 0, because a process is pinned to one CUDA device
/// (<c>GpuDeviceSelection.PinCudaDevice</c>) and the pinned device is the only one it can see.</param>
public readonly record struct AcceleratorTarget(string Backend, int DeviceIndex)
{
    /// <summary>The target for a backend name as <see cref="ServerRuntimeResolution.Backend"/> spells it; null for "cpu" or anything that is not an accelerator.</summary>
    public static AcceleratorTarget? From(string? backend, int? deviceIndex) => backend switch
    {
        "cuda" => new AcceleratorTarget("cuda", 0),
        "vulkan" => new AcceleratorTarget("vulkan", deviceIndex is >= 0 ? deviceIndex.Value : -1),
        _ => null,
    };

    public override string ToString() => DeviceIndex < 0 ? Backend : $"{Backend}:{DeviceIndex}";
}
