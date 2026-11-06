namespace OpenTail.Stingray.Mlx;

/// <summary>
/// The first real op ported end to end from TensorSharp's MLX backend, as a template for the ones
/// after it (docs/4-performance/gpu/2026-11-06-mlx-metal-backend-plan.md, "Figure out one port well,
/// then follow example"). <b>Never run on real hardware</b> -- no Mac exists on this development
/// machine, so nothing below has been exercised against the real native library; it is a faithful
/// translation of working source, not independently verified logic.
/// </summary>
public static class MlxOps
{
    /// <summary>
    /// RMSNorm over the last axis: <c>x / sqrt(mean(x^2, axis=-1) + eps) * weight</c>, computed by
    /// MLX's fused <c>mlx_fast_rms_norm</c> primitive in one dispatch.
    ///
    /// <para><b>Ported from</b> TensorSharp's <c>MlxBasicOps.RmsNorm</c> (calls
    /// <c>MlxNative.FastRmsNorm</c>) and <c>MlxNative.FastRmsNorm</c> itself (wraps
    /// <c>mlx_fast_rms_norm</c>, obtaining the stream via its cached <c>DefaultStream()</c>) --
    /// <c>C:\Git-Public\example\TensorSharp\TensorSharp.Backends.MLX\MlxBasicOps.cs</c> lines
    /// ~893-919 and <c>MlxNative.cs</c> lines ~3915-3923 and ~6560-6582. Simplified relative to
    /// that source in two ways, both named so a later port of a harder op knows what was dropped:
    /// (1) no stream caching -- a device and stream are created and freed on every call here, where
    /// TensorSharp caches the stream handle per device to avoid a device_new/free round trip per op;
    /// a perf-sensitive port should restore that. (2) no beta (bias) parameter or F16/BF16/quantized
    /// input paths -- TensorSharp's <c>RmsNorm</c> falls back to a generic path for those; this port
    /// covers only the fp32, weight-only case <c>CudaForwardPass</c>'s RMSNorm calls actually need.</para>
    /// </summary>
    /// <param name="input">Row-major <paramref name="rows"/> x <paramref name="hiddenDim"/> activations.</param>
    /// <param name="weight">The norm's learned scale, length <paramref name="hiddenDim"/>, or <c>null</c> for an unweighted norm.</param>
    public static float[] RmsNorm(ReadOnlySpan<float> input, int rows, int hiddenDim, ReadOnlySpan<float> weight, float eps)
    {
        if (input.Length != rows * hiddenDim)
            throw new ArgumentException($"input has {input.Length} elements, expected rows*hiddenDim = {rows * hiddenDim}.", nameof(input));
        if (!weight.IsEmpty && weight.Length != hiddenDim)
            throw new ArgumentException($"weight has {weight.Length} elements, expected hiddenDim = {hiddenDim}.", nameof(weight));

        var device = MlxInterop.DeviceNewType(MlxInterop.MlxGpu, 0);
        MlxInterop.MlxStream stream = default;
        MlxInterop.MlxArray inputArray = default, weightArray = default, resultArray = default;
        try
        {
            Check(MlxInterop.GetDefaultStream(out stream, device), "mlx_get_default_stream");

            int[] inputShape = [rows, hiddenDim];
            inputArray = UploadFloat32(input, inputShape);
            if (!weight.IsEmpty)
                weightArray = UploadFloat32(weight, [hiddenDim]);

            Check(MlxInterop.FastRmsNorm(out resultArray, inputArray, weightArray, eps, stream), "mlx_fast_rms_norm");

            var output = new float[input.Length];
            DownloadFloat32(resultArray, output);
            return output;
        }
        finally
        {
            if (resultArray.IsValid) MlxInterop.ArrayFree(resultArray);
            if (weightArray.IsValid) MlxInterop.ArrayFree(weightArray);
            if (inputArray.IsValid) MlxInterop.ArrayFree(inputArray);
            if (stream.Ctx != nint.Zero) MlxInterop.StreamFree(stream);
            MlxInterop.DeviceFree(device);
        }
    }

    /// <summary>
    /// SiLU (Swish): <c>x * sigmoid(x)</c>, elementwise. Ported from TensorSharp's primitives
    /// (<c>mlx_sigmoid</c> + <c>mlx_multiply</c>) rather than a fused kernel -- TensorSharp fuses this
    /// into its gate/up/SiLU MoE kernels for performance, which this port does not attempt yet.
    /// </summary>
    public static float[] Silu(ReadOnlySpan<float> input)
    {
        var device = MlxInterop.DeviceNewType(MlxInterop.MlxGpu, 0);
        MlxInterop.MlxStream stream = default;
        MlxInterop.MlxArray x = default, sig = default, result = default;
        try
        {
            Check(MlxInterop.GetDefaultStream(out stream, device), "mlx_get_default_stream");
            x = UploadFloat32(input, [input.Length]);
            Check(MlxInterop.Sigmoid(out sig, x, stream), "mlx_sigmoid");
            Check(MlxInterop.Multiply(out result, x, sig, stream), "mlx_multiply");

            var output = new float[input.Length];
            DownloadFloat32(result, output);
            return output;
        }
        finally
        {
            if (result.IsValid) MlxInterop.ArrayFree(result);
            if (sig.IsValid) MlxInterop.ArrayFree(sig);
            if (x.IsValid) MlxInterop.ArrayFree(x);
            if (stream.Ctx != nint.Zero) MlxInterop.StreamFree(stream);
            MlxInterop.DeviceFree(device);
        }
    }

    /// <summary>
    /// Row-major <paramref name="m"/> x <paramref name="k"/> times <paramref name="k"/> x
    /// <paramref name="n"/> matmul (<c>mlx_matmul</c>), e.g. a linear projection's activation-times-
    /// weight. No bias -- TensorSharp's <c>mlx_addmm</c> (<c>result = alpha*A@B + beta*src</c>) is the
    /// next natural port for a biased projection, not ported yet.
    /// </summary>
    public static float[] MatMul(ReadOnlySpan<float> a, int m, int k, ReadOnlySpan<float> b, int n)
    {
        if (a.Length != m * k) throw new ArgumentException($"a has {a.Length} elements, expected m*k = {m * k}.", nameof(a));
        if (b.Length != k * n) throw new ArgumentException($"b has {b.Length} elements, expected k*n = {k * n}.", nameof(b));

        var device = MlxInterop.DeviceNewType(MlxInterop.MlxGpu, 0);
        MlxInterop.MlxStream stream = default;
        MlxInterop.MlxArray aArray = default, bArray = default, result = default;
        try
        {
            Check(MlxInterop.GetDefaultStream(out stream, device), "mlx_get_default_stream");
            aArray = UploadFloat32(a, [m, k]);
            bArray = UploadFloat32(b, [k, n]);
            Check(MlxInterop.MatMul(out result, aArray, bArray, stream), "mlx_matmul");

            var output = new float[m * n];
            DownloadFloat32(result, output);
            return output;
        }
        finally
        {
            if (result.IsValid) MlxInterop.ArrayFree(result);
            if (bArray.IsValid) MlxInterop.ArrayFree(bArray);
            if (aArray.IsValid) MlxInterop.ArrayFree(aArray);
            if (stream.Ctx != nint.Zero) MlxInterop.StreamFree(stream);
            MlxInterop.DeviceFree(device);
        }
    }

    private static unsafe MlxInterop.MlxArray UploadFloat32(ReadOnlySpan<float> data, int[] shape)
    {
        fixed (float* ptr = data)
            return MlxInterop.ArrayNewData((nint)ptr, shape, shape.Length, MlxInterop.MlxFloat32);
    }

    private static unsafe void DownloadFloat32(MlxInterop.MlxArray array, Span<float> destination)
    {
        nint src = MlxInterop.ArrayDataFloat32(array);
        if (src == nint.Zero)
            throw new InvalidOperationException("mlx_array_data_float32 returned null.");
        new ReadOnlySpan<float>((void*)src, destination.Length).CopyTo(destination);
    }

    private static void Check(int returnCode, string call)
    {
        if (returnCode != 0)
            throw new InvalidOperationException($"{call} failed with mlx-c return code {returnCode}.");
    }
}
