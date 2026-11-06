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
    /// <summary>Elementwise multiply of two same-length arrays (<c>mlx_multiply</c>).</summary>
    public static float[] Multiply(ReadOnlySpan<float> lhs, ReadOnlySpan<float> rhs)
    {
        if (lhs.Length != rhs.Length) throw new ArgumentException($"lhs has {lhs.Length} elements, rhs has {rhs.Length}; they must match.", nameof(rhs));

        var device = MlxInterop.DeviceNewType(MlxInterop.MlxGpu, 0);
        MlxInterop.MlxStream stream = default;
        MlxInterop.MlxArray a = default, b = default, result = default;
        try
        {
            Check(MlxInterop.GetDefaultStream(out stream, device), "mlx_get_default_stream");
            a = UploadFloat32(lhs, [lhs.Length]);
            b = UploadFloat32(rhs, [rhs.Length]);
            Check(MlxInterop.Multiply(out result, a, b, stream), "mlx_multiply");

            var output = new float[lhs.Length];
            DownloadFloat32(result, output);
            return output;
        }
        finally
        {
            if (result.IsValid) MlxInterop.ArrayFree(result);
            if (b.IsValid) MlxInterop.ArrayFree(b);
            if (a.IsValid) MlxInterop.ArrayFree(a);
            if (stream.Ctx != nint.Zero) MlxInterop.StreamFree(stream);
            MlxInterop.DeviceFree(device);
        }
    }

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

    /// <summary>
    /// Biased linear projection: <c>bias + a@b</c> (<c>mlx_addmm</c> with alpha=1, beta=1), e.g. a
    /// transformer projection with a bias term. <paramref name="bias"/> broadcasts over rows, length
    /// <paramref name="n"/>.
    /// </summary>
    public static float[] AddMm(ReadOnlySpan<float> bias, ReadOnlySpan<float> a, int m, int k, ReadOnlySpan<float> b, int n)
    {
        if (bias.Length != n) throw new ArgumentException($"bias has {bias.Length} elements, expected n = {n}.", nameof(bias));
        if (a.Length != m * k) throw new ArgumentException($"a has {a.Length} elements, expected m*k = {m * k}.", nameof(a));
        if (b.Length != k * n) throw new ArgumentException($"b has {b.Length} elements, expected k*n = {k * n}.", nameof(b));

        var device = MlxInterop.DeviceNewType(MlxInterop.MlxGpu, 0);
        MlxInterop.MlxStream stream = default;
        MlxInterop.MlxArray biasArray = default, aArray = default, bArray = default, result = default;
        try
        {
            Check(MlxInterop.GetDefaultStream(out stream, device), "mlx_get_default_stream");
            biasArray = UploadFloat32(bias, [n]);
            aArray = UploadFloat32(a, [m, k]);
            bArray = UploadFloat32(b, [k, n]);
            Check(MlxInterop.AddMm(out result, biasArray, aArray, bArray, alpha: 1f, beta: 1f, stream), "mlx_addmm");

            var output = new float[m * n];
            DownloadFloat32(result, output);
            return output;
        }
        finally
        {
            if (result.IsValid) MlxInterop.ArrayFree(result);
            if (bArray.IsValid) MlxInterop.ArrayFree(bArray);
            if (aArray.IsValid) MlxInterop.ArrayFree(aArray);
            if (biasArray.IsValid) MlxInterop.ArrayFree(biasArray);
            if (stream.Ctx != nint.Zero) MlxInterop.StreamFree(stream);
            MlxInterop.DeviceFree(device);
        }
    }

    /// <summary>Softmax over the last axis of a row-major <paramref name="rows"/> x <paramref name="cols"/> matrix.</summary>
    public static float[] Softmax(ReadOnlySpan<float> input, int rows, int cols)
    {
        if (input.Length != rows * cols) throw new ArgumentException($"input has {input.Length} elements, expected rows*cols = {rows * cols}.", nameof(input));

        var device = MlxInterop.DeviceNewType(MlxInterop.MlxGpu, 0);
        MlxInterop.MlxStream stream = default;
        MlxInterop.MlxArray x = default, result = default;
        try
        {
            Check(MlxInterop.GetDefaultStream(out stream, device), "mlx_get_default_stream");
            x = UploadFloat32(input, [rows, cols]);
            Check(MlxInterop.SoftmaxAxis(out result, x, axis: -1, precise: true, stream), "mlx_softmax_axis");

            var output = new float[input.Length];
            DownloadFloat32(result, output);
            return output;
        }
        finally
        {
            if (result.IsValid) MlxInterop.ArrayFree(result);
            if (x.IsValid) MlxInterop.ArrayFree(x);
            if (stream.Ctx != nint.Zero) MlxInterop.StreamFree(stream);
            MlxInterop.DeviceFree(device);
        }
    }

    /// <summary>
    /// Scaled dot-product attention over one sequence: <paramref name="query"/>/<paramref name="key"/>/
    /// <paramref name="value"/> are row-major <c>[numHeads, seqLen, headDim]</c>. <paramref name="causal"/>
    /// selects MLX's built-in <c>"causal"</c> mask string (TensorSharp passes <c>string.Empty</c> for a
    /// single-token decode step and <c>"causal"</c> for a multi-token prefill -- no mask array, no
    /// attention sinks, no forced-fused flag; those are TensorSharp's gpt-oss/chunked-prefill paths,
    /// not ported here).
    /// </summary>
    public static float[] Attention(ReadOnlySpan<float> query, ReadOnlySpan<float> key, ReadOnlySpan<float> value,
        int numHeads, int seqLen, int headDim, float scale, bool causal)
    {
        int expected = numHeads * seqLen * headDim;
        if (query.Length != expected) throw new ArgumentException($"query has {query.Length} elements, expected numHeads*seqLen*headDim = {expected}.", nameof(query));
        if (key.Length != expected) throw new ArgumentException($"key has {key.Length} elements, expected numHeads*seqLen*headDim = {expected}.", nameof(key));
        if (value.Length != expected) throw new ArgumentException($"value has {value.Length} elements, expected numHeads*seqLen*headDim = {expected}.", nameof(value));

        var device = MlxInterop.DeviceNewType(MlxInterop.MlxGpu, 0);
        MlxInterop.MlxStream stream = default;
        MlxInterop.MlxArray q = default, k = default, v = default, result = default;
        nint maskModePtr = nint.Zero;
        try
        {
            Check(MlxInterop.GetDefaultStream(out stream, device), "mlx_get_default_stream");
            int[] shape = [numHeads, seqLen, headDim];
            q = UploadFloat32(query, shape);
            k = UploadFloat32(key, shape);
            v = UploadFloat32(value, shape);
            maskModePtr = Marshal.StringToHGlobalAnsi(causal ? "causal" : string.Empty);

            Check(MlxInterop.FastScaledDotProductAttention(out result, q, k, v, scale, maskModePtr, mask: default, sinks: default, forceFused: false, stream),
                "mlx_fast_scaled_dot_product_attention");

            var output = new float[expected];
            DownloadFloat32(result, output);
            return output;
        }
        finally
        {
            if (result.IsValid) MlxInterop.ArrayFree(result);
            if (v.IsValid) MlxInterop.ArrayFree(v);
            if (k.IsValid) MlxInterop.ArrayFree(k);
            if (q.IsValid) MlxInterop.ArrayFree(q);
            if (maskModePtr != nint.Zero) Marshal.FreeHGlobal(maskModePtr);
            if (stream.Ctx != nint.Zero) MlxInterop.StreamFree(stream);
            MlxInterop.DeviceFree(device);
        }
    }

    /// <summary>
    /// RoPE (base-frequency form, no YaRN/scaled-frequency variant) over the last
    /// <paramref name="dims"/> features of a row-major <c>[seqLen, headDim]</c> tensor, one head at a
    /// time -- ported from TensorSharp's <c>FastRopeDynamic</c> (the <c>freqs=default</c> overload;
    /// its <c>FastRopeDynamicWithFreqs</c> sibling for YaRN/scaled rope is not ported).
    /// <paramref name="positions"/> has length <paramref name="seqLen"/>.
    /// </summary>
    public static float[] Rope(ReadOnlySpan<float> input, int seqLen, int headDim, int dims,
        bool traditional, float baseValue, float scale, ReadOnlySpan<int> positions)
    {
        if (input.Length != seqLen * headDim) throw new ArgumentException($"input has {input.Length} elements, expected seqLen*headDim = {seqLen * headDim}.", nameof(input));
        if (positions.Length != seqLen) throw new ArgumentException($"positions has {positions.Length} elements, expected seqLen = {seqLen}.", nameof(positions));
        if (dims <= 0 || (dims & 1) != 0) throw new ArgumentOutOfRangeException(nameof(dims), dims, "RoPE dims must be a positive even number.");

        var device = MlxInterop.DeviceNewType(MlxInterop.MlxGpu, 0);
        MlxInterop.MlxStream stream = default;
        MlxInterop.MlxArray x = default, offsets = default, result = default;
        try
        {
            Check(MlxInterop.GetDefaultStream(out stream, device), "mlx_get_default_stream");
            x = UploadFloat32(input, [seqLen, headDim]);
            offsets = UploadInt32(positions, [seqLen]);

            Check(MlxInterop.FastRopeDynamic(out result, x, dims, traditional, MlxInterop.MlxOptionalFloat.Some(baseValue), scale, offsets, freqs: default, stream),
                "mlx_fast_rope_dynamic");

            var output = new float[input.Length];
            DownloadFloat32(result, output);
            return output;
        }
        finally
        {
            if (result.IsValid) MlxInterop.ArrayFree(result);
            if (offsets.IsValid) MlxInterop.ArrayFree(offsets);
            if (x.IsValid) MlxInterop.ArrayFree(x);
            if (stream.Ctx != nint.Zero) MlxInterop.StreamFree(stream);
            MlxInterop.DeviceFree(device);
        }
    }

    /// <summary>
    /// Flat projection output [seqLen, numHeads*headDim] -> head-first [numHeads, seqLen, headDim],
    /// the layout <see cref="Attention"/> needs. TensorSharp does this with a custom Metal kernel
    /// (<c>MlxNative.FlatToHeadFirst</c>, compiled from an embedded shader source); this port
    /// composes it from two basic ops instead (<c>mlx_reshape</c> then <c>mlx_transpose_axes</c>) --
    /// faithful to the underlying reshape, not to TensorSharp's specific fused kernel.
    /// </summary>
    public static float[] SplitHeads(ReadOnlySpan<float> flat, int seqLen, int numHeads, int headDim)
    {
        int expected = seqLen * numHeads * headDim;
        if (flat.Length != expected) throw new ArgumentException($"flat has {flat.Length} elements, expected seqLen*numHeads*headDim = {expected}.", nameof(flat));

        var device = MlxInterop.DeviceNewType(MlxInterop.MlxGpu, 0);
        MlxInterop.MlxStream stream = default;
        MlxInterop.MlxArray x = default, reshaped = default, result = default;
        try
        {
            Check(MlxInterop.GetDefaultStream(out stream, device), "mlx_get_default_stream");
            x = UploadFloat32(flat, [seqLen, numHeads * headDim]);
            Check(MlxInterop.Reshape(out reshaped, x, [seqLen, numHeads, headDim], 3, stream), "mlx_reshape");
            Check(MlxInterop.TransposeAxes(out result, reshaped, [1, 0, 2], 3, stream), "mlx_transpose_axes");

            var output = new float[expected];
            DownloadFloat32(result, output);
            return output;
        }
        finally
        {
            if (result.IsValid) MlxInterop.ArrayFree(result);
            if (reshaped.IsValid) MlxInterop.ArrayFree(reshaped);
            if (x.IsValid) MlxInterop.ArrayFree(x);
            if (stream.Ctx != nint.Zero) MlxInterop.StreamFree(stream);
            MlxInterop.DeviceFree(device);
        }
    }

    /// <summary>The inverse of <see cref="SplitHeads"/>: head-first [numHeads, seqLen, headDim] -> flat [seqLen, numHeads*headDim].</summary>
    public static float[] MergeHeads(ReadOnlySpan<float> headFirst, int numHeads, int seqLen, int headDim)
    {
        int expected = numHeads * seqLen * headDim;
        if (headFirst.Length != expected) throw new ArgumentException($"headFirst has {headFirst.Length} elements, expected numHeads*seqLen*headDim = {expected}.", nameof(headFirst));

        var device = MlxInterop.DeviceNewType(MlxInterop.MlxGpu, 0);
        MlxInterop.MlxStream stream = default;
        MlxInterop.MlxArray x = default, transposed = default, result = default;
        try
        {
            Check(MlxInterop.GetDefaultStream(out stream, device), "mlx_get_default_stream");
            x = UploadFloat32(headFirst, [numHeads, seqLen, headDim]);
            Check(MlxInterop.TransposeAxes(out transposed, x, [1, 0, 2], 3, stream), "mlx_transpose_axes");
            Check(MlxInterop.Reshape(out result, transposed, [seqLen, numHeads * headDim], 2, stream), "mlx_reshape");

            var output = new float[expected];
            DownloadFloat32(result, output);
            return output;
        }
        finally
        {
            if (result.IsValid) MlxInterop.ArrayFree(result);
            if (transposed.IsValid) MlxInterop.ArrayFree(transposed);
            if (x.IsValid) MlxInterop.ArrayFree(x);
            if (stream.Ctx != nint.Zero) MlxInterop.StreamFree(stream);
            MlxInterop.DeviceFree(device);
        }
    }

    private static unsafe MlxInterop.MlxArray UploadInt32(ReadOnlySpan<int> data, int[] shape)
    {
        fixed (int* ptr = data)
            return MlxInterop.ArrayNewData((nint)ptr, shape, shape.Length, MlxInterop.MlxInt32);
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
