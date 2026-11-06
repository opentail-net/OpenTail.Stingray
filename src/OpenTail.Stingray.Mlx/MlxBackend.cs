namespace OpenTail.Stingray.Mlx;

/// <summary>
/// Skeleton for an MLX (Apple Silicon, <c>mlx-c</c>/Metal) compute backend -- see
/// docs/4-performance/gpu/2026-11-06-mlx-metal-backend-plan.md. <b>Does not implement
/// <c>IComputeBackend</c> yet</b>: that is the much larger remaining piece (every kernel
/// <c>CudaForwardPass</c>/<c>CudaBackend</c> have today, written against an API this class cannot
/// exercise -- no Mac exists on this development machine). This class exists so the project, the
/// flag, and an honest availability check have somewhere to live before that work starts.
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
        try
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
        catch (DllNotFoundException)
        {
            return false;
        }
        catch (EntryPointNotFoundException)
        {
            return false;
        }
    }
}
