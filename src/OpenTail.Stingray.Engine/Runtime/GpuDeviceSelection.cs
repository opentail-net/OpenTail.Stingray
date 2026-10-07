#nullable enable

namespace OpenTail.Stingray.Engine.Runtime;

/// <summary>
/// Applies a concrete GPU device index. Vulkan takes the index explicitly per backend instance, so Vulkan runtimes are
/// fully isolated. CUDA has no per-call device argument here: the process is pinned through <c>CUDA_VISIBLE_DEVICES</c>,
/// which the driver reads once, at first init (cuInit) — so the pin must happen before the first CUDA call (including the
/// capability probe), and it is a PROCESS-level constraint: after CUDA has initialised, one process cannot host runtimes
/// on different physical CUDA devices. This type refuses that instead of pretending otherwise.
/// </summary>
public static class GpuDeviceSelection
{
    private const string Var = "CUDA_VISIBLE_DEVICES";
    private static readonly object s_gate = new();
    private static int s_pinnedCudaDevice = -1;

    /// <summary>
    /// The CUDA device this process was pinned to by <see cref="PinCudaDevice"/>, or -1. Only reported while the environment
    /// still shows the value that pin wrote (if something reset it, the pin is no longer a claim about this process).
    /// </summary>
    public static int PinnedCudaDevice
    {
        get
        {
            lock (s_gate) return StillPinned() ? s_pinnedCudaDevice : -1;
        }
    }

    private static bool StillPinned() =>
        s_pinnedCudaDevice >= 0 &&
        Environment.GetEnvironmentVariable(Var) == s_pinnedCudaDevice.ToString(System.Globalization.CultureInfo.InvariantCulture);

    /// <summary>
    /// Pins CUDA to <paramref name="deviceIndex"/> unless the caller already constrained the visible set themselves (then the
    /// index is relative to that set and the variable is not rewritten). No-op for index &lt; 0. Throws when this process
    /// already pinned a different device: CUDA cannot be re-mapped once initialised.
    /// </summary>
    public static void PinCudaDevice(int deviceIndex)
    {
        if (deviceIndex < 0) return;
        lock (s_gate)
        {
            if (StillPinned() && s_pinnedCudaDevice != deviceIndex)
                throw new InvalidOperationException(
                    $"This process is already pinned to CUDA device {s_pinnedCudaDevice}; it cannot also host a runtime on " +
                    $"device {deviceIndex} (CUDA_VISIBLE_DEVICES is read once, at driver initialisation). Use one process per CUDA device.");
            if (Environment.GetEnvironmentVariable(Var) is null)
            {
                Environment.SetEnvironmentVariable(Var, deviceIndex.ToString(System.Globalization.CultureInfo.InvariantCulture));
                s_pinnedCudaDevice = deviceIndex;
            }
        }
    }
}
