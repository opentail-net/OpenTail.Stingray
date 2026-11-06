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

    [LibraryImport(LibraryName, EntryPoint = "mlx_array_new_int")]
    [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
    internal static partial MlxArray ArrayNewInt(int value);

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

    [LibraryImport(LibraryName, EntryPoint = "mlx_array_data_int32")]
    [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
    internal static partial nint ArrayDataInt32(MlxArray array);

    /// <summary>MLX's fused RMSNorm primitive: <c>x / sqrt(mean(x^2) + eps) * weight</c> in one
    /// dispatch. <paramref name="weight"/> may be an invalid (default) <see cref="MlxArray"/> for an
    /// unweighted norm -- mlx-c treats a null/invalid array argument there as "no weight" the same way
    /// TensorSharp's own call site does (<c>GetOptionalView</c> in MlxBasicOps.cs's RmsNorm).</summary>
    [LibraryImport(LibraryName, EntryPoint = "mlx_fast_rms_norm")]
    [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
    internal static partial int FastRmsNorm(out MlxArray result, MlxArray input, MlxArray weight, float eps, MlxStream stream);

    // ── Elementwise, matmul and attention (ported from TensorSharp's TensorSharp.Backends.MLX) ───

    [LibraryImport(LibraryName, EntryPoint = "mlx_sigmoid")]
    [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
    internal static partial int Sigmoid(out MlxArray result, MlxArray input, MlxStream stream);

    [LibraryImport(LibraryName, EntryPoint = "mlx_multiply")]
    [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
    internal static partial int Multiply(out MlxArray result, MlxArray lhs, MlxArray rhs, MlxStream stream);

    [LibraryImport(LibraryName, EntryPoint = "mlx_matmul")]
    [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
    internal static partial int MatMul(out MlxArray result, MlxArray a, MlxArray b, MlxStream stream);

    /// <summary><c>result = alpha*(m1@m2) + beta*src</c>.</summary>
    [LibraryImport(LibraryName, EntryPoint = "mlx_addmm")]
    [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
    internal static partial int AddMm(out MlxArray result, MlxArray src, MlxArray m1, MlxArray m2, float alpha, float beta, MlxStream stream);

    [LibraryImport(LibraryName, EntryPoint = "mlx_softmax_axis")]
    [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
    internal static partial int SoftmaxAxis(out MlxArray result, MlxArray input, int axis, [MarshalAs(UnmanagedType.I1)] bool precise, MlxStream stream);

    [LibraryImport(LibraryName, EntryPoint = "mlx_fast_scaled_dot_product_attention")]
    [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
    internal static partial int FastScaledDotProductAttention(out MlxArray result, MlxArray query, MlxArray key, MlxArray value, float scale, nint maskMode, MlxArray mask, MlxArray sinks, [MarshalAs(UnmanagedType.I1)] bool forceFused, MlxStream stream);

    // ── RoPE ────────────────────────────────────────────────────────────────

    /// <summary>mlx-c's int32 dtype tag (value 7, per MlxNative.cs's DType -> mlx dtype switch).</summary>
    internal const int MlxInt32 = 7;

    [StructLayout(LayoutKind.Sequential)]
    internal readonly struct MlxOptionalFloat
    {
        private readonly float value;
        [MarshalAs(UnmanagedType.I1)]
        private readonly bool hasValue;
        private MlxOptionalFloat(float value, bool hasValue) { this.value = value; this.hasValue = hasValue; }
        public static MlxOptionalFloat Some(float value) => new(value, true);
    }

    [LibraryImport(LibraryName, EntryPoint = "mlx_fast_rope_dynamic")]
    [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
    internal static partial int FastRopeDynamic(out MlxArray result, MlxArray input, int dims, [MarshalAs(UnmanagedType.I1)] bool traditional, MlxOptionalFloat baseValue, float scale, MlxArray offsets, MlxArray freqs, MlxStream stream);

    // ── Offset causal mask (continuing a multi-token prefill into an already-populated KV cache) --
    // ported from TensorSharp's SlidingWindowMask (MlxFusedOps.cs ~2076), minus the window term: just
    // the "notFuture" (distance >= 0) half of that function's mask construction.

    [LibraryImport(LibraryName, EntryPoint = "mlx_arange")]
    [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
    internal static partial int Arange(out MlxArray result, double start, double stop, double step, int dtype, MlxStream stream);

    [LibraryImport(LibraryName, EntryPoint = "mlx_subtract")]
    [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
    internal static partial int Subtract(out MlxArray result, MlxArray lhs, MlxArray rhs, MlxStream stream);

    [LibraryImport(LibraryName, EntryPoint = "mlx_greater")]
    [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
    internal static partial int Greater(out MlxArray result, MlxArray lhs, MlxArray rhs, MlxStream stream);

    // ── Reshape/transpose, for multi-head split/concat (TensorSharp uses a custom Metal kernel,
    // FlatToHeadFirst, here instead -- composed from these two basic ops; see MlxOps.SplitHeads/
    // MergeHeads) ────────────────────────────────────────────────────────────

    [LibraryImport(LibraryName, EntryPoint = "mlx_reshape")]
    [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
    internal static partial int Reshape(out MlxArray result, MlxArray array, int[] shape, nuint shapeCount, MlxStream stream);

    [LibraryImport(LibraryName, EntryPoint = "mlx_transpose_axes")]
    [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
    internal static partial int TransposeAxes(out MlxArray result, MlxArray array, int[] axes, nuint axesCount, MlxStream stream);

    // ── MoE router top-K + softmax (ported from TensorSharp's TryMoeRouterTopKSoftmax,
    // MlxFusedOps.cs ~102-182: negate, argpartition for top-K, slice/reshape/cast the indices,
    // gather the top-K logits, softmax them) -- MoE/hybrid/vision architectures item,
    // docs/4-performance/gpu/2026-11-06-mlx-metal-backend-plan.md. This is the router/gating
    // step only: which experts fire and their normalized weights. The expert FFN gather/compute/
    // scatter-add that consumes this output is a separate, larger, not-yet-ported piece.

    [LibraryImport(LibraryName, EntryPoint = "mlx_negative")]
    [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
    internal static partial int Negative(out MlxArray result, MlxArray input, MlxStream stream);

    [LibraryImport(LibraryName, EntryPoint = "mlx_argpartition_axis")]
    [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
    internal static partial int ArgPartitionAxis(out MlxArray result, MlxArray input, int kth, int axis, MlxStream stream);

    [LibraryImport(LibraryName, EntryPoint = "mlx_slice")]
    [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
    internal static partial int Slice(out MlxArray result, MlxArray input, int[] starts, nuint startCount, int[] stops, nuint stopCount, int[] strides, nuint strideCount, MlxStream stream);

    [LibraryImport(LibraryName, EntryPoint = "mlx_astype")]
    [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
    internal static partial int Astype(out MlxArray result, MlxArray array, int dtype, MlxStream stream);

    /// <summary>mlx-c's uint32 dtype tag (value 6, per MlxNative.cs's DType -> mlx dtype switch;
    /// <c>mlx_argpartition_axis</c>'s real output dtype, which the top-K index path casts down to
    /// <see cref="MlxInt32"/> afterwards, matching TensorSharp's own cast).</summary>
    internal const int MlxUInt32 = 6;

    [LibraryImport(LibraryName, EntryPoint = "mlx_take_axis")]
    [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
    internal static partial int TakeAxis(out MlxArray result, MlxArray input, MlxArray indices, int axis, MlxStream stream);
}
