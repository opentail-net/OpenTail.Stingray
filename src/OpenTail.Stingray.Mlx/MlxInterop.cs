namespace OpenTail.Stingray.Mlx;

/// <summary>
/// P/Invoke bindings for <c>mlx-c</c> (<c>libmlxc</c>), Apple's MLX array framework's C API.
/// Source-generated <c>LibraryImport</c>, matching this project's existing CUDA/NVRTC interop style.
///
/// <para><b>Never run on real hardware</b> -- this development machine has no Mac at all, so nothing
/// here has been exercised against the real native library. The entry points, struct layouts and
/// calling convention below were copied from the real, vendored BSD-licensed TensorSharp source
/// (<c>C:\Git-Public\example\TensorSharp\TensorSharp.Backends.MLX\MlxNative.cs</c>, zhongkaifu/TensorSharp)
/// rather than guessed, so at least the signatures are a real reference rather than invented symbol
/// names -- but a correct-looking signature is not evidence it links or behaves correctly. See
/// docs/4-performance/gpu/2026-11-06-mlx-metal-backend-plan.md for scope and the "done" bar this
/// feature is held to instead of the project's usual real-hardware bar.</para>
///
/// <para>.NET's <c>LibraryImport</c>-generated P/Invoke resolves the native library lazily, at the
/// first call into it -- not at build time -- so this compiles cleanly on Windows/Linux CI with no
/// <c>libmlxc</c> present, the same way <c>CuBlasInterop</c> already compiles here with no CUDA
/// runtime installed. <see cref="MlxBackend.IsAvailable"/> is the only thing that actually calls into
/// this class in this skeleton, and it fails gracefully (try/catch, mirroring
/// <c>CudaBackend.IsAvailable()</c>) rather than crashing when the library is absent.</para>
/// </summary>
internal static partial class MlxInterop
{
    private const string LibraryName = "mlxc";

    /// <summary>MLX device type for Metal/GPU (value 1, per the real mlx-c headers).</summary>
    internal const int MlxGpu = 1;

    [StructLayout(LayoutKind.Sequential)]
    internal readonly struct MlxDevice
    {
        public readonly nint Ctx;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal readonly struct MlxArray
    {
        public readonly nint Ctx;
        public bool IsValid => Ctx != nint.Zero;
    }

    [LibraryImport(LibraryName, EntryPoint = "mlx_metal_is_available")]
    [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
    internal static partial int MetalIsAvailable([MarshalAs(UnmanagedType.I1)] out bool result);

    [LibraryImport(LibraryName, EntryPoint = "mlx_device_new_type")]
    [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
    internal static partial MlxDevice DeviceNewType(int type, int index);

    [LibraryImport(LibraryName, EntryPoint = "mlx_device_free")]
    [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
    internal static partial int DeviceFree(MlxDevice device);

    [LibraryImport(LibraryName, EntryPoint = "mlx_device_is_available")]
    [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
    internal static partial int DeviceIsAvailable([MarshalAs(UnmanagedType.I1)] out bool available, MlxDevice device);

    [LibraryImport(LibraryName, EntryPoint = "mlx_set_default_device")]
    [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
    internal static partial int SetDefaultDevice(MlxDevice device);

    [LibraryImport(LibraryName, EntryPoint = "mlx_array_new")]
    [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
    internal static partial MlxArray ArrayNew();

    [LibraryImport(LibraryName, EntryPoint = "mlx_array_free")]
    [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
    internal static partial int ArrayFree(MlxArray array);

    // ── Streams, array-from-data, and fast RMSNorm ─────────────────────────
    // Ported from MlxNative.cs's DefaultStream()/FastRmsNorm() (lines ~6560, ~3915, and the
    // LibraryImport declarations ~8800-8854), chosen as the first real op to port end to end
    // (docs/4-performance/gpu/2026-11-06-mlx-metal-backend-plan.md, "Figure out one port well").

    [StructLayout(LayoutKind.Sequential)]
    internal readonly struct MlxStream
    {
        public readonly nint Ctx;
    }

    /// <summary>mlx-c's float32 dtype tag (value 10, per MlxNative.cs's DType -> mlx dtype switch).</summary>
    internal const int MlxFloat32 = 10;

    [LibraryImport(LibraryName, EntryPoint = "mlx_get_default_stream")]
    [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
    internal static partial int GetDefaultStream(out MlxStream stream, MlxDevice device);

    [LibraryImport(LibraryName, EntryPoint = "mlx_stream_free")]
    [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
    internal static partial int StreamFree(MlxStream stream);

    /// <summary><paramref name="shape"/>/<paramref name="dim"/> describe <paramref name="data"/>'s
    /// layout; <paramref name="dtype"/> is an mlx-c dtype tag (<see cref="MlxFloat32"/> for this port).
    /// MLX copies the data in (TensorSharp's own comment on the "managed" variant notes this one does
    /// not take ownership of <paramref name="data"/>), so the caller's buffer can be freed/reused
    /// right after this call returns.</summary>
    [LibraryImport(LibraryName, EntryPoint = "mlx_array_new_data")]
    [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
    internal static partial MlxArray ArrayNewData(nint data, int[] shape, int dim, int dtype);

    [LibraryImport(LibraryName, EntryPoint = "mlx_array_data_float32")]
    [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
    internal static partial nint ArrayDataFloat32(MlxArray array);

    /// <summary>MLX's fused RMSNorm primitive: <c>x / sqrt(mean(x^2) + eps) * weight</c> in one
    /// dispatch. <paramref name="weight"/> may be an invalid (default) <see cref="MlxArray"/> for an
    /// unweighted norm -- mlx-c treats a null/invalid array argument there as "no weight" the same way
    /// TensorSharp's own call site does (<c>GetOptionalView</c> in MlxBasicOps.cs's RmsNorm).</summary>
    [LibraryImport(LibraryName, EntryPoint = "mlx_fast_rms_norm")]
    [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
    internal static partial int FastRmsNorm(out MlxArray result, MlxArray input, MlxArray weight, float eps, MlxStream stream);
}
