namespace OpenTail.Stingray.Mlx;

/// <summary>
/// Availability probe for the experimental MLX dense-Llama route. The engine planner/factory can
/// select and construct that route when explicitly enabled. This check only verifies that mlx-c
/// reports an available Metal device; it is not evidence that native numerical inference or full
/// hardware correctness has been verified. See docs/4-performance/gpu/2026-11-06-mlx-metal-backend-plan.md.
/// </summary>
public static class MlxBackend
{
    /// <summary>
    /// True when <c>libmlxc</c> is present AND it reports a working Metal device. Mirrors
    /// <c>CudaBackend.IsAvailable()</c>'s try/catch shape: a missing native library throws
    /// <see cref="DllNotFoundException"/> from the very first P/Invoke call, which this catches and
    /// turns into <c>false</c> rather than letting it propagate. On this development machine (no Mac,
    /// no <c>libmlxc</c>) this has always returned <c>false</c> -- that is the expected, honest result
    /// here, not a bug.
    /// </summary>
    public static bool IsAvailable()
    {
        return IsAvailable(ProbeNativeAvailability);
    }

    internal static bool IsAvailable(Func<bool> probe)
    {
        try
        {
            return probe();
        }
        catch (DllNotFoundException)
        {
            return false;
        }
        catch (EntryPointNotFoundException)
        {
            return false;
        }
    }

    private static bool ProbeNativeAvailability()
    {
        if (MlxInterop.MetalIsAvailable(out bool metalAvailable) != 0 || !metalAvailable)
            return false;

        var device = MlxInterop.DeviceNewType(MlxInterop.MlxGpu, 0);
        try
        {
            return MlxInterop.DeviceIsAvailable(out bool deviceAvailable, device) == 0 && deviceAvailable;
        }
        finally
        {
            MlxInterop.DeviceFree(device);
        }
    }
}
