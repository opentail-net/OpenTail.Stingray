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
    // ── Cached default stream (TensorSharp perf optimizations item,
    // docs/4-performance/gpu/2026-11-06-mlx-metal-backend-plan.md) ──────────────────────────────
    // Ported from TensorSharp's MlxNative.DefaultStream() (MlxNative.cs ~6560-6582): the default
    // MLX stream for a device is a stable handle MLX itself owns, so recomputing it on every op (a
    // device_new + device_free round trip per call, which every op below did until this port) is
    // wasted work -- cache it once per device index instead. This is the first of this port's own
    // "no stream caching" simplifications (named explicitly in RmsNorm's doc comment below) to be
    // closed; unlike every correctness-affecting op in this file, this change cannot be verified by
    // any test on this machine (no mlx-c to actually measure a round trip against) -- it is a
    // structural change justified by reading the real TensorSharp comment, not a measured one.
    private static MlxInterop.MlxStream _cachedStream;
    private static int _cachedStreamDeviceIndex = -1;
    private static readonly object _streamCacheLock = new();

    private static MlxInterop.MlxStream GetCachedStream(int deviceIndex = 0)
    {
        lock (_streamCacheLock)
        {
            if (_cachedStreamDeviceIndex == deviceIndex)
                return _cachedStream;

            var device = MlxInterop.DeviceNewType(MlxInterop.MlxGpu, deviceIndex);
            try
            {
                Check(MlxInterop.GetDefaultStream(out var stream, device), "mlx_get_default_stream");
                _cachedStream = stream;
                _cachedStreamDeviceIndex = deviceIndex;
                return stream;
            }
            finally
            {
                MlxInterop.DeviceFree(device);
            }
        }
    }

    /// <summary>Elementwise multiply of two same-length arrays (<c>mlx_multiply</c>).</summary>
    public static float[] Multiply(ReadOnlySpan<float> lhs, ReadOnlySpan<float> rhs)
    {
        if (lhs.Length != rhs.Length) throw new ArgumentException($"lhs has {lhs.Length} elements, rhs has {rhs.Length}; they must match.", nameof(rhs));

        var stream = GetCachedStream();
        MlxInterop.MlxArray a = default, b = default, result = default;
        try
        {
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
    /// that source in one remaining way: no beta (bias) parameter or F16/BF16/quantized input paths
    /// -- TensorSharp's <c>RmsNorm</c> falls back to a generic path for those; this port covers only
    /// the fp32, weight-only case <c>CudaForwardPass</c>'s RMSNorm calls actually need. (Stream
    /// caching, named here as missing in an earlier version of this port, is now done -- see
    /// <see cref="GetCachedStream"/>.)</para>
    /// </summary>
    /// <param name="input">Row-major <paramref name="rows"/> x <paramref name="hiddenDim"/> activations.</param>
    /// <param name="weight">The norm's learned scale, length <paramref name="hiddenDim"/>, or <c>null</c> for an unweighted norm.</param>
    public static float[] RmsNorm(ReadOnlySpan<float> input, int rows, int hiddenDim, ReadOnlySpan<float> weight, float eps)
    {
        if (input.Length != rows * hiddenDim)
            throw new ArgumentException($"input has {input.Length} elements, expected rows*hiddenDim = {rows * hiddenDim}.", nameof(input));
        if (!weight.IsEmpty && weight.Length != hiddenDim)
            throw new ArgumentException($"weight has {weight.Length} elements, expected hiddenDim = {hiddenDim}.", nameof(weight));

        var stream = GetCachedStream();
        MlxInterop.MlxArray inputArray = default, weightArray = default, resultArray = default;
        try
        {
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
        }
    }

    /// <summary>
    /// SiLU (Swish): <c>x * sigmoid(x)</c>, elementwise. Ported from TensorSharp's primitives
    /// (<c>mlx_sigmoid</c> + <c>mlx_multiply</c>) rather than a fused kernel -- TensorSharp fuses this
    /// into its gate/up/SiLU MoE kernels for performance, which this port does not attempt yet.
    /// </summary>
    public static float[] Silu(ReadOnlySpan<float> input)
    {
        var stream = GetCachedStream();
        MlxInterop.MlxArray x = default, sig = default, result = default;
        try
        {
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

        var stream = GetCachedStream();
        MlxInterop.MlxArray aArray = default, bArray = default, result = default;
        try
        {
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

        var stream = GetCachedStream();
        MlxInterop.MlxArray biasArray = default, aArray = default, bArray = default, result = default;
        try
        {
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
        }
    }

    /// <summary>Softmax over the last axis of a row-major <paramref name="rows"/> x <paramref name="cols"/> matrix.</summary>
    public static float[] Softmax(ReadOnlySpan<float> input, int rows, int cols)
    {
        if (input.Length != rows * cols) throw new ArgumentException($"input has {input.Length} elements, expected rows*cols = {rows * cols}.", nameof(input));

        var stream = GetCachedStream();
        MlxInterop.MlxArray x = default, result = default;
        try
        {
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
        }
    }

    /// <summary>
    /// Scaled dot-product attention: <paramref name="query"/> is row-major
    /// <c>[numQueryHeads, querySeqLen, headDim]</c>; <paramref name="key"/>/<paramref name="value"/>
    /// are row-major <c>[numKvHeads, kvSeqLen, headDim]</c> -- <paramref name="kvSeqLen"/> equal to
    /// <paramref name="querySeqLen"/> for a prefill, or the KV cache's full length for an incremental
    /// decode step (<see cref="MlxKvCache"/>), where <paramref name="querySeqLen"/> is 1.
    /// <paramref name="numKvHeads"/> may be less than <paramref name="numQueryHeads"/> (GQA):
    /// <c>mlx_fast_scaled_dot_product_attention</c> broadcasts internally when given fewer KV heads
    /// than query heads -- confirmed from TensorSharp's own real call site
    /// (<c>MlxFusedOps.cs</c>, its KV cache is explicitly sized <c>numKVHeads</c> while <c>q</c> is
    /// sized the full <c>numHeads</c>, with no repeat/broadcast step anywhere in that code path), not
    /// a guess. <paramref name="causal"/> selects MLX's built-in <c>"causal"</c> mask string
    /// (TensorSharp passes <c>string.Empty</c> for a single-token decode step and <c>"causal"</c> for
    /// a multi-token prefill -- no mask array, no attention sinks, no forced-fused flag; those are
    /// TensorSharp's gpt-oss/chunked-prefill paths, not ported here). <paramref name="queryOffset"/>
    /// &gt; 0 (continuing a prefill into an already-populated <see cref="MlxKvCache"/>, so the new
    /// queries' absolute positions start partway through the key range) builds a real offset causal
    /// mask instead of using the built-in string -- ported from TensorSharp's <c>SlidingWindowMask</c>
    /// (<c>MlxFusedOps.cs</c> ~2076: <c>arange</c>+<c>reshape</c> the query and key absolute
    /// positions, <c>subtract</c> to get their pairwise distance, <c>greater than -1</c> for
    /// "not in the future" -- this port omits that function's window term, since it is building a
    /// plain offset causal mask, not a sliding-window one). The mask is built and consumed entirely
    /// device-side within this call -- never downloaded to host and re-uploaded, since mlx-c's mask
    /// array is boolean and this port has no host-side bool-array path.
    /// </summary>
    public static float[] Attention(ReadOnlySpan<float> query, ReadOnlySpan<float> key, ReadOnlySpan<float> value,
        int numQueryHeads, int numKvHeads, int querySeqLen, int kvSeqLen, int headDim, float scale, bool causal, int queryOffset = 0)
    {
        int queryExpected = numQueryHeads * querySeqLen * headDim;
        int kvExpected = numKvHeads * kvSeqLen * headDim;
        if (query.Length != queryExpected) throw new ArgumentException($"query has {query.Length} elements, expected numQueryHeads*querySeqLen*headDim = {queryExpected}.", nameof(query));
        if (key.Length != kvExpected) throw new ArgumentException($"key has {key.Length} elements, expected numKvHeads*kvSeqLen*headDim = {kvExpected}.", nameof(key));
        if (value.Length != kvExpected) throw new ArgumentException($"value has {value.Length} elements, expected numKvHeads*kvSeqLen*headDim = {kvExpected}.", nameof(value));
        if (queryOffset < 0) throw new ArgumentOutOfRangeException(nameof(queryOffset), queryOffset, "queryOffset must be >= 0.");
        if (queryOffset > 0 && !causal) throw new ArgumentException("queryOffset > 0 only makes sense for causal attention.", nameof(queryOffset));

        var stream = GetCachedStream();
        MlxInterop.MlxArray q = default, k = default, v = default, result = default;
        MlxInterop.MlxArray rows = default, rowsColumn = default, cols = default, colsRow = default, distance = default, offsetMask = default;
        nint maskModePtr = nint.Zero;
        try
        {
            q = UploadFloat32(query, [numQueryHeads, querySeqLen, headDim]);
            k = UploadFloat32(key, [numKvHeads, kvSeqLen, headDim]);
            v = UploadFloat32(value, [numKvHeads, kvSeqLen, headDim]);

            MlxInterop.MlxArray maskArray = default;
            if (queryOffset > 0)
            {
                Check(MlxInterop.Arange(out rows, queryOffset, queryOffset + querySeqLen, 1, MlxInterop.MlxInt32, stream), "mlx_arange (rows)");
                Check(MlxInterop.Reshape(out rowsColumn, rows, [querySeqLen, 1], 2, stream), "mlx_reshape (rowsColumn)");
                Check(MlxInterop.Arange(out cols, 0, kvSeqLen, 1, MlxInterop.MlxInt32, stream), "mlx_arange (cols)");
                Check(MlxInterop.Reshape(out colsRow, cols, [1, kvSeqLen], 2, stream), "mlx_reshape (colsRow)");
                Check(MlxInterop.Subtract(out distance, rowsColumn, colsRow, stream), "mlx_subtract");
                var minusOne = MlxInterop.ArrayNewInt(-1);
                try { Check(MlxInterop.Greater(out offsetMask, distance, minusOne, stream), "mlx_greater"); }
                finally { MlxInterop.ArrayFree(minusOne); }
                maskArray = offsetMask;
            }
            maskModePtr = Marshal.StringToHGlobalAnsi(causal && queryOffset == 0 ? "causal" : string.Empty);

            Check(MlxInterop.FastScaledDotProductAttention(out result, q, k, v, scale, maskModePtr, mask: maskArray, sinks: default, forceFused: false, stream),
                "mlx_fast_scaled_dot_product_attention");

            var output = new float[queryExpected];
            DownloadFloat32(result, output);
            return output;
        }
        finally
        {
            if (result.IsValid) MlxInterop.ArrayFree(result);
            if (offsetMask.IsValid) MlxInterop.ArrayFree(offsetMask);
            if (distance.IsValid) MlxInterop.ArrayFree(distance);
            if (colsRow.IsValid) MlxInterop.ArrayFree(colsRow);
            if (cols.IsValid) MlxInterop.ArrayFree(cols);
            if (rowsColumn.IsValid) MlxInterop.ArrayFree(rowsColumn);
            if (rows.IsValid) MlxInterop.ArrayFree(rows);
            if (v.IsValid) MlxInterop.ArrayFree(v);
            if (k.IsValid) MlxInterop.ArrayFree(k);
            if (q.IsValid) MlxInterop.ArrayFree(q);
            if (maskModePtr != nint.Zero) Marshal.FreeHGlobal(maskModePtr);
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

        var stream = GetCachedStream();
        MlxInterop.MlxArray x = default, offsets = default, result = default;
        try
        {
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

        var stream = GetCachedStream();
        MlxInterop.MlxArray x = default, reshaped = default, result = default;
        try
        {
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
        }
    }

    /// <summary>The inverse of <see cref="SplitHeads"/>: head-first [numHeads, seqLen, headDim] -> flat [seqLen, numHeads*headDim].</summary>
    public static float[] MergeHeads(ReadOnlySpan<float> headFirst, int numHeads, int seqLen, int headDim)
    {
        int expected = numHeads * seqLen * headDim;
        if (headFirst.Length != expected) throw new ArgumentException($"headFirst has {headFirst.Length} elements, expected numHeads*seqLen*headDim = {expected}.", nameof(headFirst));

        var stream = GetCachedStream();
        MlxInterop.MlxArray x = default, transposed = default, result = default;
        try
        {
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
        }
    }

    /// <summary>
    /// MoE router top-K + normalized softmax over a single token's router logits: returns the
    /// indices of the <paramref name="k"/> largest logits and softmax(those <paramref name="k"/>
    /// logits) -- normalized within just the selected experts, matching the common "renormalize
    /// top-K probabilities" MoE gating convention.
    ///
    /// <para><b>Ported from</b> TensorSharp's <c>MlxFusedOps.TryMoeRouterTopKSoftmax</c>
    /// (<c>C:\Git-Public\example\TensorSharp\TensorSharp.Backends.MLX\MlxFusedOps.cs</c> lines
    /// ~102-182): negate the logits so <c>mlx_argpartition_axis</c>'s "k smallest" gives the top-K
    /// largest, slice out the first <paramref name="k"/> partitioned indices, reshape/cast them to
    /// int32, gather (<c>mlx_take_axis</c>) the corresponding logits, then softmax those <paramref
    /// name="k"/> values. This is the router/gating step only -- it decides which experts fire and
    /// their relative weights; the expert FFN gather/compute/scatter-add that consumes this output
    /// (TensorSharp's <c>TryGatherRows</c>/<c>TryScatterAddWeightedRows</c>/<c>TryMoeBiasWeightedSum</c>)
    /// is a separate, larger, not-yet-ported piece (docs/4-performance/gpu/2026-11-06-mlx-metal-
    /// backend-plan.md).</para>
    ///
    /// <para>Unlike TensorSharp's version (which writes results in place into caller-owned MLX-backed
    /// tensors to avoid a host round trip on the decode hot path), this port is a plain host-in/
    /// host-out function -- consistent with every other op in this file; the device-resident,
    /// round-trip-free variant is one of the un-ported performance optimizations.</para>
    /// </summary>
    /// <param name="routerLogits">One token's logits, length <paramref name="numExperts"/>.</param>
    /// <param name="numExperts">Total number of experts (<paramref name="routerLogits"/>'s length).</param>
    /// <param name="k">Number of experts to route to; must satisfy <c>0 &lt; k &lt; numExperts</c>.</param>
    public static (int[] TopIndices, float[] RouteWeights) MoeRouterTopKSoftmax(ReadOnlySpan<float> routerLogits, int numExperts, int k)
    {
        if (routerLogits.Length != numExperts) throw new ArgumentException($"routerLogits has {routerLogits.Length} elements, expected numExperts = {numExperts}.", nameof(routerLogits));
        if (k <= 0 || k >= numExperts) throw new ArgumentOutOfRangeException(nameof(k), k, "k must satisfy 0 < k < numExperts.");

        var stream = GetCachedStream();
        MlxInterop.MlxArray logits = default, neg = default, partIdx = default, topIdx2d = default,
            topIdx1d = default, topIdx1dInt = default, topLogits = default, softmaxed = default;
        try
        {
            logits = UploadFloat32(routerLogits, [1, numExperts]);

            Check(MlxInterop.Negative(out neg, logits, stream), "mlx_negative");
            Check(MlxInterop.ArgPartitionAxis(out partIdx, neg, kth: k - 1, axis: 1, stream), "mlx_argpartition_axis");
            Check(MlxInterop.Slice(out topIdx2d, partIdx, starts: [0, 0], startCount: 2, stops: [1, k], stopCount: 2, strides: [1, 1], strideCount: 2, stream), "mlx_slice");
            Check(MlxInterop.Reshape(out topIdx1d, topIdx2d, [k], 1, stream), "mlx_reshape");
            Check(MlxInterop.Astype(out topIdx1dInt, topIdx1d, MlxInterop.MlxInt32, stream), "mlx_astype");
            Check(MlxInterop.TakeAxis(out topLogits, logits, topIdx1dInt, axis: 1, stream), "mlx_take_axis");
            Check(MlxInterop.SoftmaxAxis(out softmaxed, topLogits, axis: -1, precise: true, stream), "mlx_softmax_axis");

            var topIndices = new int[k];
            DownloadInt32(topIdx1dInt, topIndices);
            var routeWeights = new float[k];
            DownloadFloat32(softmaxed, routeWeights);
            return (topIndices, routeWeights);
        }
        finally
        {
            if (softmaxed.IsValid) MlxInterop.ArrayFree(softmaxed);
            if (topLogits.IsValid) MlxInterop.ArrayFree(topLogits);
            if (topIdx1dInt.IsValid) MlxInterop.ArrayFree(topIdx1dInt);
            if (topIdx1d.IsValid) MlxInterop.ArrayFree(topIdx1d);
            if (topIdx2d.IsValid) MlxInterop.ArrayFree(topIdx2d);
            if (partIdx.IsValid) MlxInterop.ArrayFree(partIdx);
            if (neg.IsValid) MlxInterop.ArrayFree(neg);
            if (logits.IsValid) MlxInterop.ArrayFree(logits);
        }
    }

    /// <summary>
    /// Gathers rows of a row-major <paramref name="srcRows"/> x <paramref name="cols"/> matrix at
    /// <paramref name="indices"/>, producing <c>indices.Length</c> x <paramref name="cols"/> rows in
    /// that order -- e.g. collecting the tokens routed to one MoE expert before running its FFN.
    ///
    /// <para><b>Ported from</b> TensorSharp's <c>MlxFusedOps.TryGatherRows</c>
    /// (<c>MlxFusedOps.cs</c> ~291-333): <c>mlx_take_axis</c> along axis 0, then <c>mlx_contiguous</c>
    /// so the result is a flat, directly-downloadable buffer (TensorSharp calls this before handing
    /// the array to its device-resident caller; this port's own <c>DownloadFloat32</c> needs the same
    /// contiguity).</para>
    /// </summary>
    public static float[] GatherRows(ReadOnlySpan<float> src, int srcRows, int cols, ReadOnlySpan<int> indices)
    {
        if (src.Length != srcRows * cols) throw new ArgumentException($"src has {src.Length} elements, expected srcRows*cols = {srcRows * cols}.", nameof(src));

        var stream = GetCachedStream();
        MlxInterop.MlxArray srcArray = default, indicesArray = default, gathered = default, contiguous = default;
        try
        {
            srcArray = UploadFloat32(src, [srcRows, cols]);
            indicesArray = UploadInt32(indices, [indices.Length]);
            Check(MlxInterop.TakeAxis(out gathered, srcArray, indicesArray, axis: 0, stream), "mlx_take_axis");
            Check(MlxInterop.Contiguous(out contiguous, gathered, allowColMajor: false, stream), "mlx_contiguous");

            var output = new float[indices.Length * cols];
            DownloadFloat32(contiguous, output);
            return output;
        }
        finally
        {
            if (contiguous.IsValid) MlxInterop.ArrayFree(contiguous);
            if (gathered.IsValid) MlxInterop.ArrayFree(gathered);
            if (indicesArray.IsValid) MlxInterop.ArrayFree(indicesArray);
            if (srcArray.IsValid) MlxInterop.ArrayFree(srcArray);
        }
    }

    /// <summary>
    /// Gemma 4 vision encoder's padded self-attention: <paramref name="query"/>/<paramref name="key"/>/
    /// <paramref name="value"/> are row-major <c>[batches, tokens, heads, 64]</c> (head dim fixed at
    /// 64, matching the source's own constraint). No scaling (<c>scale = 1</c>) and no causal mask --
    /// the vision encoder's self-attention is bidirectional over all patch tokens.
    ///
    /// <para><b>Ported from</b> TensorSharp's <c>Gemma4VisionOps.TrySelfAttention</c>
    /// (<c>C:\Git-Public\example\TensorSharp\TensorSharp.Backends.MLX\MlxGemma4VisionOps.cs</c>
    /// ~112-160): transpose <c>[batches, tokens, heads, 64]</c> to head-major
    /// <c>[batches, heads, tokens, 64]</c>, zero-pad the head dimension to 80 (192 for <c>tokens &lt;=
    /// 8</c>) so MLX's fused SDPA kernel takes its F32 SIMD path instead of narrowing 64/96/128/256-
    /// wide heads to TF32 (the padding is exact zeros and the scale stays 1, so this changes nothing
    /// mathematically), run the fused attention, crop back to the real 64 channels, transpose back.
    /// Unlike every other op in this port, this one is built entirely from basic composed ops in
    /// TensorSharp's own source too -- it is not a custom-Metal-kernel function being reimplemented,
    /// just a faithful translation of real composed logic.</para>
    /// </summary>
    public static float[] Gemma4VisionSelfAttention(ReadOnlySpan<float> query, ReadOnlySpan<float> key, ReadOnlySpan<float> value,
        int batches, int tokens, int heads)
    {
        const int headDim = 64;
        int expected = batches * tokens * heads * headDim;
        if (query.Length != expected) throw new ArgumentException($"query has {query.Length} elements, expected batches*tokens*heads*64 = {expected}.", nameof(query));
        if (key.Length != expected) throw new ArgumentException($"key has {key.Length} elements, expected batches*tokens*heads*64 = {expected}.", nameof(key));
        if (value.Length != expected) throw new ArgumentException($"value has {value.Length} elements, expected batches*tokens*heads*64 = {expected}.", nameof(value));

        var stream = GetCachedStream();
        MlxInterop.MlxArray q = default, k = default, v = default, qh = default, kh = default, vh = default,
            padding = default, qp = default, kp = default, vp = default, attention = default,
            cropped = default, restored = default, output = default;
        nint maskModePtr = nint.Zero;
        try
        {
            int[] shape = [batches, tokens, heads, headDim];
            q = UploadFloat32(query, shape);
            k = UploadFloat32(key, shape);
            v = UploadFloat32(value, shape);

            int[] transposeAxes = [0, 2, 1, 3];
            Check(MlxInterop.TransposeAxes(out qh, q, transposeAxes, 4, stream), "mlx_transpose_axes (q)");
            Check(MlxInterop.TransposeAxes(out kh, k, transposeAxes, 4, stream), "mlx_transpose_axes (k)");
            Check(MlxInterop.TransposeAxes(out vh, v, transposeAxes, 4, stream), "mlx_transpose_axes (v)");

            int paddedHead = tokens <= 8 ? 192 : 80;
            padding = UploadFullZero([batches, heads, tokens, paddedHead - headDim], stream);
            qp = ConcatenateLastAxis(qh, padding, stream);
            kp = ConcatenateLastAxis(kh, padding, stream);
            vp = ConcatenateLastAxis(vh, padding, stream);

            maskModePtr = Marshal.StringToHGlobalAnsi(string.Empty);
            Check(MlxInterop.FastScaledDotProductAttention(out attention, qp, kp, vp, scale: 1f, maskModePtr, mask: default, sinks: default, forceFused: true, stream),
                "mlx_fast_scaled_dot_product_attention");

            Check(MlxInterop.Slice(out cropped, attention,
                starts: [0, 0, 0, 0], startCount: 4,
                stops: [batches, heads, tokens, headDim], stopCount: 4,
                strides: [1, 1, 1, 1], strideCount: 4, stream), "mlx_slice");
            Check(MlxInterop.TransposeAxes(out restored, cropped, transposeAxes, 4, stream), "mlx_transpose_axes (restored)");
            Check(MlxInterop.Contiguous(out output, restored, allowColMajor: false, stream), "mlx_contiguous");

            var result = new float[expected];
            DownloadFloat32(output, result);
            return result;
        }
        finally
        {
            if (output.IsValid) MlxInterop.ArrayFree(output);
            if (restored.IsValid) MlxInterop.ArrayFree(restored);
            if (cropped.IsValid) MlxInterop.ArrayFree(cropped);
            if (attention.IsValid) MlxInterop.ArrayFree(attention);
            if (vp.IsValid) MlxInterop.ArrayFree(vp);
            if (kp.IsValid) MlxInterop.ArrayFree(kp);
            if (qp.IsValid) MlxInterop.ArrayFree(qp);
            if (padding.IsValid) MlxInterop.ArrayFree(padding);
            if (vh.IsValid) MlxInterop.ArrayFree(vh);
            if (kh.IsValid) MlxInterop.ArrayFree(kh);
            if (qh.IsValid) MlxInterop.ArrayFree(qh);
            if (v.IsValid) MlxInterop.ArrayFree(v);
            if (k.IsValid) MlxInterop.ArrayFree(k);
            if (q.IsValid) MlxInterop.ArrayFree(q);
            if (maskModePtr != nint.Zero) Marshal.FreeHGlobal(maskModePtr);
        }
    }

    private static MlxInterop.MlxArray UploadFullZero(int[] shape, MlxInterop.MlxStream stream)
    {
        var scalar = MlxInterop.ArrayNewFloat32(0f);
        try
        {
            Check(MlxInterop.Full(out var result, shape, (nuint)shape.Length, scalar, MlxInterop.MlxFloat32, stream), "mlx_full");
            return result;
        }
        finally
        {
            MlxInterop.ArrayFree(scalar);
        }
    }

    private static MlxInterop.MlxArray ConcatenateLastAxis(MlxInterop.MlxArray first, MlxInterop.MlxArray second, MlxInterop.MlxStream stream)
    {
        var vector = MlxInterop.VectorArrayNew();
        try
        {
            Check(MlxInterop.VectorArrayAppendValue(vector, first), "mlx_vector_array_append_value (first)");
            Check(MlxInterop.VectorArrayAppendValue(vector, second), "mlx_vector_array_append_value (second)");
            Check(MlxInterop.ConcatenateAxis(out var result, vector, axis: 3, stream), "mlx_concatenate_axis");
            return result;
        }
        finally
        {
            if (vector.IsValid) MlxInterop.VectorArrayFree(vector);
        }
    }

    private static unsafe void DownloadInt32(MlxInterop.MlxArray array, Span<int> destination)
    {
        nint src = MlxInterop.ArrayDataInt32(array);
        if (src == nint.Zero)
            throw new InvalidOperationException("mlx_array_data_int32 returned null.");
        new ReadOnlySpan<int>((void*)src, destination.Length).CopyTo(destination);
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
