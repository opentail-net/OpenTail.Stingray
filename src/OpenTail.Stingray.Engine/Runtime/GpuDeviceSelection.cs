#nullable enable

namespace OpenTail.Stingray.Engine.Runtime;

/// <summary>
/// Applies a concrete GPU device index. CUDA has no per-call device argument here: the process is pinned via
/// <c>CUDA_VISIBLE_DEVICES</c>, which the driver reads at first init (cuInit), so the pin must happen before
/// the first CUDA call — including the capability probe. Vulkan takes the index explicitly.
/// </summary>
public static class GpuDeviceSelection
{
    /// <summary>Pins CUDA to <paramref name="deviceIndex"/> unless the caller already constrained the visible set. No-op for index &lt; 0.</summary>
    public static void PinCudaDevice(int deviceIndex)
    {
        if (deviceIndex < 0) return;
        if (Environment.GetEnvironmentVariable("CUDA_VISIBLE_DEVICES") is null)
            Environment.SetEnvironmentVariable("CUDA_VISIBLE_DEVICES",
                deviceIndex.ToString(System.Globalization.CultureInfo.InvariantCulture));
    }
}
