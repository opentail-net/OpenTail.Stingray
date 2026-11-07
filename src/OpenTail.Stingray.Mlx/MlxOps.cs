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
                DownloadFloat32(result, output, stream);
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
            DownloadFloat32(resultArray, output, stream);
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
            DownloadFloat32(result, output, stream);
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
            DownloadFloat32(result, output, stream);
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
            DownloadFloat32(result, output, stream);
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
            DownloadFloat32(result, output, stream);
            return output;
        }
        finally
        {
            if (result.IsValid) MlxInterop.ArrayFree(result);
            if (x.IsValid) MlxInterop.ArrayFree(x);
        }
    }

    /// <summary>
    /// Scaled dot-product attention: the host buffers are row-major
    /// <c>[numQueryHeads, querySeqLen, headDim]</c> and <c>[numKvHeads, kvSeqLen, headDim]</c>;
    /// MLX receives them as <c>[1, heads, sequenceLength, headDim]</c>. <paramref name="kvSeqLen"/> is equal to
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
        if (numQueryHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numQueryHeads));
        if (numKvHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numKvHeads));
        if (numQueryHeads % numKvHeads != 0) throw new ArgumentException("numQueryHeads must be divisible by numKvHeads for grouped-query attention.", nameof(numKvHeads));
        if (querySeqLen <= 0) throw new ArgumentOutOfRangeException(nameof(querySeqLen));
        if (kvSeqLen <= 0) throw new ArgumentOutOfRangeException(nameof(kvSeqLen));
        if (headDim <= 0) throw new ArgumentOutOfRangeException(nameof(headDim));
        int queryExpected = numQueryHeads * querySeqLen * headDim;
        int kvExpected = numKvHeads * kvSeqLen * headDim;
        if (query.Length != queryExpected) throw new ArgumentException($"query has {query.Length} elements, expected numQueryHeads*querySeqLen*headDim = {queryExpected}.", nameof(query));
        if (key.Length != kvExpected) throw new ArgumentException($"key has {key.Length} elements, expected numKvHeads*kvSeqLen*headDim = {kvExpected}.", nameof(key));
        if (value.Length != kvExpected) throw new ArgumentException($"value has {value.Length} elements, expected numKvHeads*kvSeqLen*headDim = {kvExpected}.", nameof(value));
        if (queryOffset < 0) throw new ArgumentOutOfRangeException(nameof(queryOffset), queryOffset, "queryOffset must be >= 0.");
        if (queryOffset > 0 && !causal) throw new ArgumentException("queryOffset > 0 only makes sense for causal attention.", nameof(queryOffset));
        if (queryOffset > 0 && queryOffset + querySeqLen > kvSeqLen)
            throw new ArgumentException("queryOffset + querySeqLen cannot exceed kvSeqLen.", nameof(queryOffset));

        var stream = GetCachedStream();
        MlxInterop.MlxArray q = default, k = default, v = default, result = default;
        MlxInterop.MlxArray rows = default, rowsColumn = default, cols = default, colsRow = default, distance = default, offsetMask2d = default, offsetMask = default;
        nint maskModePtr = nint.Zero;
        try
        {
            q = UploadFloat32(query, AttentionTensorShape(numQueryHeads, querySeqLen, headDim));
            k = UploadFloat32(key, AttentionTensorShape(numKvHeads, kvSeqLen, headDim));
            v = UploadFloat32(value, AttentionTensorShape(numKvHeads, kvSeqLen, headDim));

            MlxInterop.MlxArray maskArray = default;
            if (queryOffset > 0)
            {
                Check(MlxInterop.Arange(out rows, queryOffset, queryOffset + querySeqLen, 1, MlxInterop.MlxInt32, stream), "mlx_arange (rows)");
                Check(MlxInterop.Reshape(out rowsColumn, rows, [querySeqLen, 1], 2, stream), "mlx_reshape (rowsColumn)");
                Check(MlxInterop.Arange(out cols, 0, kvSeqLen, 1, MlxInterop.MlxInt32, stream), "mlx_arange (cols)");
                Check(MlxInterop.Reshape(out colsRow, cols, [1, kvSeqLen], 2, stream), "mlx_reshape (colsRow)");
                Check(MlxInterop.Subtract(out distance, rowsColumn, colsRow, stream), "mlx_subtract");
                var minusOne = MlxInterop.ArrayNewInt(-1);
                try { Check(MlxInterop.Greater(out offsetMask2d, distance, minusOne, stream), "mlx_greater"); }
                finally { MlxInterop.ArrayFree(minusOne); }
                Check(MlxInterop.Reshape(out offsetMask, offsetMask2d, AttentionMaskShape(querySeqLen, kvSeqLen), 4, stream), "mlx_reshape (offsetMask)");
                maskArray = offsetMask;
            }
            maskModePtr = Marshal.StringToHGlobalAnsi(causal && queryOffset == 0 ? "causal" : string.Empty);

            Check(MlxInterop.FastScaledDotProductAttention(out result, q, k, v, scale, maskModePtr, mask: maskArray, sinks: default, forceFused: false, stream),
                "mlx_fast_scaled_dot_product_attention");

            var output = new float[queryExpected];
            DownloadFloat32(result, output, stream);
            return output;
        }
        finally
        {
            if (result.IsValid) MlxInterop.ArrayFree(result);
            if (offsetMask.IsValid) MlxInterop.ArrayFree(offsetMask);
            if (offsetMask2d.IsValid) MlxInterop.ArrayFree(offsetMask2d);
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
    /// <paramref name="dims"/> features of a row-major host <c>[seqLen, headDim]</c> tensor, one head at a
    /// time. MLX receives each contiguous run as <c>[1, runLength, headDim]</c> and a single offset
    /// for the batch; non-contiguous positions are processed as separate runs -- ported from TensorSharp's
    /// <c>FastRopeDynamic</c> (the <c>freqs=default</c> overload;
    /// its <c>FastRopeDynamicWithFreqs</c> sibling for YaRN/scaled rope is not ported).
    /// <paramref name="positions"/> has length <paramref name="seqLen"/>.
    /// </summary>
    public static float[] Rope(ReadOnlySpan<float> input, int seqLen, int headDim, int dims,
        bool traditional, float baseValue, float scale, ReadOnlySpan<int> positions)
    {
        if (seqLen <= 0) throw new ArgumentOutOfRangeException(nameof(seqLen));
        if (headDim <= 0) throw new ArgumentOutOfRangeException(nameof(headDim));
        if (input.Length != seqLen * headDim) throw new ArgumentException($"input has {input.Length} elements, expected seqLen*headDim = {seqLen * headDim}.", nameof(input));
        if (positions.Length != seqLen) throw new ArgumentException($"positions has {positions.Length} elements, expected seqLen = {seqLen}.", nameof(positions));
        if (dims <= 0 || (dims & 1) != 0) throw new ArgumentOutOfRangeException(nameof(dims), dims, "RoPE dims must be a positive even number.");
        if (dims > headDim) throw new ArgumentOutOfRangeException(nameof(dims), dims, "RoPE dims cannot exceed headDim.");
        for (int i = 0; i < positions.Length; i++)
            if (positions[i] < 0) throw new ArgumentOutOfRangeException(nameof(positions), positions[i], "RoPE positions must be non-negative.");

        var stream = GetCachedStream();
        var output = new float[input.Length];
        int runStart = 0;
        while (runStart < seqLen)
        {
            int runLength = 1;
            while (runStart + runLength < seqLen && positions[runStart + runLength] == positions[runStart] + runLength)
                runLength++;

            MlxInterop.MlxArray x = default, offsets = default, result = default;
            try
            {
                x = UploadFloat32(input.Slice(runStart * headDim, runLength * headDim), RopeTensorShape(runLength, headDim));
                offsets = UploadInt32([positions[runStart]], [1]);

                Check(MlxInterop.FastRopeDynamic(out result, x, dims, traditional, MlxInterop.MlxOptionalFloat.Some(baseValue), scale, offsets, freqs: default, stream),
                    "mlx_fast_rope_dynamic");

                DownloadFloat32(result, output.AsSpan(runStart * headDim, runLength * headDim), stream);
            }
            finally
            {
                if (result.IsValid) MlxInterop.ArrayFree(result);
                if (offsets.IsValid) MlxInterop.ArrayFree(offsets);
                if (x.IsValid) MlxInterop.ArrayFree(x);
            }

            runStart += runLength;
        }
        return output;
    }

    internal static int[] AttentionTensorShape(int heads, int sequenceLength, int headDim) => [1, heads, sequenceLength, headDim];

    internal static int[] AttentionMaskShape(int querySequenceLength, int keySequenceLength) => [1, 1, querySequenceLength, keySequenceLength];

    internal static int[] RopeTensorShape(int sequenceLength, int headDim) => [1, sequenceLength, headDim];

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
        MlxInterop.MlxArray x = default, reshaped = default, result = default, contiguous = default;
        try
        {
            x = UploadFloat32(flat, [seqLen, numHeads * headDim]);
            Check(MlxInterop.Reshape(out reshaped, x, [seqLen, numHeads, headDim], 3, stream), "mlx_reshape");
            Check(MlxInterop.TransposeAxes(out result, reshaped, [1, 0, 2], 3, stream), "mlx_transpose_axes");
            Check(MlxInterop.Contiguous(out contiguous, result, allowColMajor: false, stream), "mlx_contiguous");

            var output = new float[expected];
            DownloadFloat32(contiguous, output, stream);
            return output;
        }
        finally
        {
            if (contiguous.IsValid) MlxInterop.ArrayFree(contiguous);
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
        MlxInterop.MlxArray x = default, transposed = default, result = default, contiguous = default;
        try
        {
            x = UploadFloat32(headFirst, [numHeads, seqLen, headDim]);
            Check(MlxInterop.TransposeAxes(out transposed, x, [1, 0, 2], 3, stream), "mlx_transpose_axes");
            Check(MlxInterop.Reshape(out result, transposed, [seqLen, numHeads * headDim], 2, stream), "mlx_reshape");
            Check(MlxInterop.Contiguous(out contiguous, result, allowColMajor: false, stream), "mlx_contiguous");

            var output = new float[expected];
            DownloadFloat32(contiguous, output, stream);
            return output;
        }
        finally
        {
            if (contiguous.IsValid) MlxInterop.ArrayFree(contiguous);
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
            DownloadInt32(topIdx1dInt, topIndices, stream);
            var routeWeights = new float[k];
            DownloadFloat32(softmaxed, routeWeights, stream);
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
            DownloadFloat32(contiguous, output, stream);
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
            DownloadFloat32(output, result, stream);
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

    /// <summary>
    /// MoE combine: <c>output[row] = inPlace[row] + weights[k] * rows[k]</c> for the unique <c>k</c>
    /// with <c>indices[k] == row</c> (0 contribution if no such <c>k</c>); <paramref name="indices"/>
    /// must be sorted and unique, matching the real kernel's binary-search assumption.
    ///
    /// <para><b>Ported from</b> TensorSharp's <c>MlxFusedOps.TryScatterAddWeightedRows</c>
    /// (<c>MlxFusedOps.cs</c> ~335-392), whose real implementation is a custom fused Metal kernel
    /// (<c>ScatterAddWeightedRowsSource</c> in <c>MlxNative.cs</c> ~1475-1495: per output element, a
    /// binary search over <paramref name="indices"/> for a match, add the matching weighted row if
    /// found). Per the owner's "stick to C#" instruction (no new Metal kernels), this composes the
    /// same result from basic ops instead: a one-hot selection matrix built via comparison (no
    /// <c>mlx_equal</c> exists in this port's real-signature inventory, so equality is derived as
    /// <c>1 - (greater(diff,0) + greater(-diff,0))</c>, i.e. "not greater either way"), scaled by
    /// <paramref name="weights"/>, then a single matmul against <paramref name="rows"/> --
    /// mathematically the scatter-add this kernel performs, since each row has at most one match.</para>
    /// </summary>
    public static float[] ScatterAddWeightedRows(ReadOnlySpan<float> inPlace, int seqLen, int hiddenDim,
        ReadOnlySpan<float> rows, ReadOnlySpan<int> indices, ReadOnlySpan<float> weights)
    {
        if (inPlace.Length != seqLen * hiddenDim) throw new ArgumentException($"inPlace has {inPlace.Length} elements, expected seqLen*hiddenDim = {seqLen * hiddenDim}.", nameof(inPlace));
        int batchSize = indices.Length;
        if (weights.Length != batchSize) throw new ArgumentException($"weights has {weights.Length} elements, expected indices.Length = {batchSize}.", nameof(weights));
        if (rows.Length != batchSize * hiddenDim) throw new ArgumentException($"rows has {rows.Length} elements, expected indices.Length*hiddenDim = {batchSize * hiddenDim}.", nameof(rows));
        if (batchSize == 0) return inPlace.ToArray();

        var stream = GetCachedStream();
        MlxInterop.MlxArray inPlaceArray = default, rowsArray = default, indicesIntArray = default, weightsArray = default,
            rowIdx = default, rowIdxCol = default, indicesF = default, indicesRow = default, diff = default, negDiff = default,
            gt1 = default, gt1f = default, gt2 = default, gt2f = default, notEqualF = default, equalF = default,
            weightedMask = default, combined = default, output = default;
        MlxInterop.MlxArray zero = default, one = default;
        try
        {
            inPlaceArray = UploadFloat32(inPlace, [seqLen, hiddenDim]);
            rowsArray = UploadFloat32(rows, [batchSize, hiddenDim]);
            indicesIntArray = UploadInt32(indices, [batchSize]);
            weightsArray = UploadFloat32(weights, [1, batchSize]);
            zero = MlxInterop.ArrayNewFloat32(0f);
            one = MlxInterop.ArrayNewFloat32(1f);

            Check(MlxInterop.Arange(out rowIdx, 0, seqLen, 1, MlxInterop.MlxFloat32, stream), "mlx_arange");
            Check(MlxInterop.Reshape(out rowIdxCol, rowIdx, [seqLen, 1], 2, stream), "mlx_reshape (rowIdxCol)");
            Check(MlxInterop.Astype(out indicesF, indicesIntArray, MlxInterop.MlxFloat32, stream), "mlx_astype (indicesF)");
            Check(MlxInterop.Reshape(out indicesRow, indicesF, [1, batchSize], 2, stream), "mlx_reshape (indicesRow)");

            Check(MlxInterop.Subtract(out diff, rowIdxCol, indicesRow, stream), "mlx_subtract");
            Check(MlxInterop.Negative(out negDiff, diff, stream), "mlx_negative");
            Check(MlxInterop.Greater(out gt1, diff, zero, stream), "mlx_greater (diff>0)");
            Check(MlxInterop.Astype(out gt1f, gt1, MlxInterop.MlxFloat32, stream), "mlx_astype (gt1f)");
            Check(MlxInterop.Greater(out gt2, negDiff, zero, stream), "mlx_greater (diff<0)");
            Check(MlxInterop.Astype(out gt2f, gt2, MlxInterop.MlxFloat32, stream), "mlx_astype (gt2f)");
            Check(MlxInterop.Add(out notEqualF, gt1f, gt2f, stream), "mlx_add (notEqualF)");
            Check(MlxInterop.Subtract(out equalF, one, notEqualF, stream), "mlx_subtract (equalF)");

            Check(MlxInterop.Multiply(out weightedMask, equalF, weightsArray, stream), "mlx_multiply (weightedMask)");
            Check(MlxInterop.MatMul(out combined, weightedMask, rowsArray, stream), "mlx_matmul (combined)");
            Check(MlxInterop.Add(out output, inPlaceArray, combined, stream), "mlx_add (output)");

            var result = new float[seqLen * hiddenDim];
            DownloadFloat32(output, result, stream);
            return result;
        }
        finally
        {
            if (output.IsValid) MlxInterop.ArrayFree(output);
            if (combined.IsValid) MlxInterop.ArrayFree(combined);
            if (weightedMask.IsValid) MlxInterop.ArrayFree(weightedMask);
            if (equalF.IsValid) MlxInterop.ArrayFree(equalF);
            if (notEqualF.IsValid) MlxInterop.ArrayFree(notEqualF);
            if (gt2f.IsValid) MlxInterop.ArrayFree(gt2f);
            if (gt2.IsValid) MlxInterop.ArrayFree(gt2);
            if (gt1f.IsValid) MlxInterop.ArrayFree(gt1f);
            if (gt1.IsValid) MlxInterop.ArrayFree(gt1);
            if (negDiff.IsValid) MlxInterop.ArrayFree(negDiff);
            if (diff.IsValid) MlxInterop.ArrayFree(diff);
            if (indicesRow.IsValid) MlxInterop.ArrayFree(indicesRow);
            if (indicesF.IsValid) MlxInterop.ArrayFree(indicesF);
            if (rowIdxCol.IsValid) MlxInterop.ArrayFree(rowIdxCol);
            if (rowIdx.IsValid) MlxInterop.ArrayFree(rowIdx);
            if (one.IsValid) MlxInterop.ArrayFree(one);
            if (zero.IsValid) MlxInterop.ArrayFree(zero);
            if (weightsArray.IsValid) MlxInterop.ArrayFree(weightsArray);
            if (indicesIntArray.IsValid) MlxInterop.ArrayFree(indicesIntArray);
            if (rowsArray.IsValid) MlxInterop.ArrayFree(rowsArray);
            if (inPlaceArray.IsValid) MlxInterop.ArrayFree(inPlaceArray);
        }
    }

    /// <summary>
    /// GPT-OSS "swiglu_oai" clamped SwiGLU over separate gate/up matrices, with each row's gate/up
    /// bias gathered by its expert id: <c>g = gate[r] + gateBias[experts[r]]; u = up[r] +
    /// upBias[experts[r]]; out[r] = min(g,limit) * sigmoid(alpha*min(g,limit)) *
    /// (clamp(u,-limit,limit) + 1)</c>.
    ///
    /// <para><b>Ported from</b> TensorSharp's <c>MlxFusedOps.TrySwiGluOaiGatherBias</c>
    /// (<c>MlxFusedOps.cs</c> ~541-602), a custom fused Metal kernel in the real source. Composed here
    /// from basic ops instead: <c>mlx_take_axis</c> to gather each row's bias by expert id, then
    /// <c>min(a,b) = -max(-a,-b)</c> (no <c>mlx_minimum</c> in this port's real-signature inventory, but
    /// <c>mlx_maximum</c> is) to get both the clamp and the <c>min(g,limit)</c> term.</para>
    /// </summary>
    public static float[] SwiGluOaiGatherBias(ReadOnlySpan<float> gate, ReadOnlySpan<float> up, int rows, int dim,
        ReadOnlySpan<float> gateBias, ReadOnlySpan<float> upBias, int numExperts, ReadOnlySpan<int> expertIndices,
        float alpha, float limit)
    {
        if (gate.Length != rows * dim) throw new ArgumentException($"gate has {gate.Length} elements, expected rows*dim = {rows * dim}.", nameof(gate));
        if (up.Length != rows * dim) throw new ArgumentException($"up has {up.Length} elements, expected rows*dim = {rows * dim}.", nameof(up));
        if (gateBias.Length != numExperts * dim) throw new ArgumentException($"gateBias has {gateBias.Length} elements, expected numExperts*dim = {numExperts * dim}.", nameof(gateBias));
        if (upBias.Length != numExperts * dim) throw new ArgumentException($"upBias has {upBias.Length} elements, expected numExperts*dim = {numExperts * dim}.", nameof(upBias));
        if (expertIndices.Length != rows) throw new ArgumentException($"expertIndices has {expertIndices.Length} elements, expected rows = {rows}.", nameof(expertIndices));

        var stream = GetCachedStream();
        MlxInterop.MlxArray gateArray = default, upArray = default, gateBiasArray = default, upBiasArray = default,
            expertIndicesArray = default, gatheredGateBias = default, gatheredUpBias = default, g = default, u = default,
            negLimit = default, negG = default, maxNegGNegLimit = default, gMin = default, alphaScalar = default,
            sigArg = default, sig = default, uMaxed = default, negUMaxed = default, maxNegUMaxedNegLimit = default,
            uClamped = default, one = default, onePlusU = default, part1 = default, result = default;
        try
        {
            gateArray = UploadFloat32(gate, [rows, dim]);
            upArray = UploadFloat32(up, [rows, dim]);
            gateBiasArray = UploadFloat32(gateBias, [numExperts, dim]);
            upBiasArray = UploadFloat32(upBias, [numExperts, dim]);
            expertIndicesArray = UploadInt32(expertIndices, [rows]);
            negLimit = MlxInterop.ArrayNewFloat32(-limit);
            alphaScalar = MlxInterop.ArrayNewFloat32(alpha);
            one = MlxInterop.ArrayNewFloat32(1f);

            Check(MlxInterop.TakeAxis(out gatheredGateBias, gateBiasArray, expertIndicesArray, axis: 0, stream), "mlx_take_axis (gateBias)");
            Check(MlxInterop.TakeAxis(out gatheredUpBias, upBiasArray, expertIndicesArray, axis: 0, stream), "mlx_take_axis (upBias)");
            Check(MlxInterop.Add(out g, gateArray, gatheredGateBias, stream), "mlx_add (g)");
            Check(MlxInterop.Add(out u, upArray, gatheredUpBias, stream), "mlx_add (u)");

            Check(MlxInterop.Negative(out negG, g, stream), "mlx_negative (negG)");
            Check(MlxInterop.Maximum(out maxNegGNegLimit, negG, negLimit, stream), "mlx_maximum (gMin)");
            Check(MlxInterop.Negative(out gMin, maxNegGNegLimit, stream), "mlx_negative (gMin)");

            Check(MlxInterop.Multiply(out sigArg, gMin, alphaScalar, stream), "mlx_multiply (sigArg)");
            Check(MlxInterop.Sigmoid(out sig, sigArg, stream), "mlx_sigmoid");

            Check(MlxInterop.Maximum(out uMaxed, u, negLimit, stream), "mlx_maximum (uMaxed)");
            Check(MlxInterop.Negative(out negUMaxed, uMaxed, stream), "mlx_negative (negUMaxed)");
            Check(MlxInterop.Maximum(out maxNegUMaxedNegLimit, negUMaxed, negLimit, stream), "mlx_maximum (uClamped)");
            Check(MlxInterop.Negative(out uClamped, maxNegUMaxedNegLimit, stream), "mlx_negative (uClamped)");
            Check(MlxInterop.Add(out onePlusU, uClamped, one, stream), "mlx_add (onePlusU)");

            Check(MlxInterop.Multiply(out part1, gMin, sig, stream), "mlx_multiply (part1)");
            Check(MlxInterop.Multiply(out result, part1, onePlusU, stream), "mlx_multiply (result)");

            var output = new float[rows * dim];
            DownloadFloat32(result, output, stream);
            return output;
        }
        finally
        {
            if (result.IsValid) MlxInterop.ArrayFree(result);
            if (part1.IsValid) MlxInterop.ArrayFree(part1);
            if (onePlusU.IsValid) MlxInterop.ArrayFree(onePlusU);
            if (one.IsValid) MlxInterop.ArrayFree(one);
            if (uClamped.IsValid) MlxInterop.ArrayFree(uClamped);
            if (maxNegUMaxedNegLimit.IsValid) MlxInterop.ArrayFree(maxNegUMaxedNegLimit);
            if (negUMaxed.IsValid) MlxInterop.ArrayFree(negUMaxed);
            if (uMaxed.IsValid) MlxInterop.ArrayFree(uMaxed);
            if (sig.IsValid) MlxInterop.ArrayFree(sig);
            if (sigArg.IsValid) MlxInterop.ArrayFree(sigArg);
            if (alphaScalar.IsValid) MlxInterop.ArrayFree(alphaScalar);
            if (gMin.IsValid) MlxInterop.ArrayFree(gMin);
            if (maxNegGNegLimit.IsValid) MlxInterop.ArrayFree(maxNegGNegLimit);
            if (negG.IsValid) MlxInterop.ArrayFree(negG);
            if (negLimit.IsValid) MlxInterop.ArrayFree(negLimit);
            if (u.IsValid) MlxInterop.ArrayFree(u);
            if (g.IsValid) MlxInterop.ArrayFree(g);
            if (gatheredUpBias.IsValid) MlxInterop.ArrayFree(gatheredUpBias);
            if (gatheredGateBias.IsValid) MlxInterop.ArrayFree(gatheredGateBias);
            if (expertIndicesArray.IsValid) MlxInterop.ArrayFree(expertIndicesArray);
            if (upBiasArray.IsValid) MlxInterop.ArrayFree(upBiasArray);
            if (gateBiasArray.IsValid) MlxInterop.ArrayFree(gateBiasArray);
            if (upArray.IsValid) MlxInterop.ArrayFree(upArray);
            if (gateArray.IsValid) MlxInterop.ArrayFree(gateArray);
        }
    }

    /// <summary>
    /// Routing-weighted MoE combine: <c>output[n,d] = sum over k in [0,expertsPerToken) of
    /// routeWeights[n*K+k] * (downRows[invOrder[n*K+k],d] + downBias[expertsSorted[invOrder[n*K+k]],d])</c>
    /// (the bias term omitted when <paramref name="downBias"/> is empty). <paramref name="downRows"/>
    /// holds the <c>N*K</c> expert down-projection rows in expert-sorted pair order; <paramref
    /// name="invOrder"/> maps original pair <c>p = n*K+k</c> to its sorted row.
    ///
    /// <para><b>Ported from</b> TensorSharp's <c>MlxFusedOps.TryMoeBiasWeightedSum</c>
    /// (<c>MlxFusedOps.cs</c> ~614-680), a custom fused Metal kernel in the real source. Composed here
    /// from basic ops: gather the down-projection rows and (by a further expert-id gather) their bias
    /// rows at <paramref name="invOrder"/>, scale by <paramref name="routeWeights"/>, then sum the
    /// <paramref name="expertsPerToken"/> terms per token -- done via a batched matmul against a
    /// <c>[1, expertsPerToken]</c> ones row (no <c>mlx_sum</c> in this port's real-signature inventory,
    /// but <c>mlx_matmul</c> broadcasting its leading batch dim is the same reduction).</para>
    /// </summary>
    public static float[] MoeBiasWeightedSum(ReadOnlySpan<float> downRows, int tokens, int expertsPerToken, int dim,
        ReadOnlySpan<float> downBias, int numExperts, ReadOnlySpan<int> expertsSorted, ReadOnlySpan<int> invOrder,
        ReadOnlySpan<float> routeWeights)
    {
        int pairs = checked(tokens * expertsPerToken);
        if (downRows.Length != pairs * dim) throw new ArgumentException($"downRows has {downRows.Length} elements, expected tokens*expertsPerToken*dim = {pairs * dim}.", nameof(downRows));
        bool hasBias = !downBias.IsEmpty;
        if (hasBias && downBias.Length != numExperts * dim) throw new ArgumentException($"downBias has {downBias.Length} elements, expected numExperts*dim = {numExperts * dim}.", nameof(downBias));
        if (expertsSorted.Length != pairs) throw new ArgumentException($"expertsSorted has {expertsSorted.Length} elements, expected pairs = {pairs}.", nameof(expertsSorted));
        if (invOrder.Length != pairs) throw new ArgumentException($"invOrder has {invOrder.Length} elements, expected pairs = {pairs}.", nameof(invOrder));
        if (routeWeights.Length != pairs) throw new ArgumentException($"routeWeights has {routeWeights.Length} elements, expected pairs = {pairs}.", nameof(routeWeights));

        var stream = GetCachedStream();
        MlxInterop.MlxArray downRowsArray = default, downBiasArray = default, expertsSortedArray = default, invOrderArray = default,
            routeWeightsArray = default, gatheredDown = default, gatheredExpertsForBias = default, gatheredBias = default,
            combined = default, weightsCol = default, weighted = default, weighted3d = default, ones = default,
            summedBatched = default, result = default;
        try
        {
            downRowsArray = UploadFloat32(downRows, [pairs, dim]);
            if (hasBias) downBiasArray = UploadFloat32(downBias, [numExperts, dim]);
            expertsSortedArray = UploadInt32(expertsSorted, [pairs]);
            invOrderArray = UploadInt32(invOrder, [pairs]);
            routeWeightsArray = UploadFloat32(routeWeights, [pairs, 1]);

            Check(MlxInterop.TakeAxis(out gatheredDown, downRowsArray, invOrderArray, axis: 0, stream), "mlx_take_axis (down)");
            if (hasBias)
            {
                Check(MlxInterop.TakeAxis(out gatheredExpertsForBias, expertsSortedArray, invOrderArray, axis: 0, stream), "mlx_take_axis (experts)");
                Check(MlxInterop.TakeAxis(out gatheredBias, downBiasArray, gatheredExpertsForBias, axis: 0, stream), "mlx_take_axis (bias)");
                Check(MlxInterop.Add(out combined, gatheredDown, gatheredBias, stream), "mlx_add (combined)");
            }
            else
            {
                combined = gatheredDown;
            }

            Check(MlxInterop.Multiply(out weighted, combined, routeWeightsArray, stream), "mlx_multiply (weighted)");
            Check(MlxInterop.Reshape(out weighted3d, weighted, [tokens, expertsPerToken, dim], 3, stream), "mlx_reshape (weighted3d)");

            var onesData = new float[expertsPerToken];
            Array.Fill(onesData, 1f);
            ones = UploadFloat32(onesData, [1, 1, expertsPerToken]);
            Check(MlxInterop.MatMul(out summedBatched, ones, weighted3d, stream), "mlx_matmul (summedBatched)");
            Check(MlxInterop.Reshape(out result, summedBatched, [tokens, dim], 2, stream), "mlx_reshape (result)");

            var output = new float[tokens * dim];
            DownloadFloat32(result, output, stream);
            return output;
        }
        finally
        {
            if (result.IsValid) MlxInterop.ArrayFree(result);
            if (summedBatched.IsValid) MlxInterop.ArrayFree(summedBatched);
            if (ones.IsValid) MlxInterop.ArrayFree(ones);
            if (weighted3d.IsValid) MlxInterop.ArrayFree(weighted3d);
            if (weighted.IsValid) MlxInterop.ArrayFree(weighted);
            if (weightsCol.IsValid) MlxInterop.ArrayFree(weightsCol);
            if (hasBias && combined.IsValid) MlxInterop.ArrayFree(combined);
            if (gatheredBias.IsValid) MlxInterop.ArrayFree(gatheredBias);
            if (gatheredExpertsForBias.IsValid) MlxInterop.ArrayFree(gatheredExpertsForBias);
            if (gatheredDown.IsValid) MlxInterop.ArrayFree(gatheredDown);
            if (routeWeightsArray.IsValid) MlxInterop.ArrayFree(routeWeightsArray);
            if (invOrderArray.IsValid) MlxInterop.ArrayFree(invOrderArray);
            if (expertsSortedArray.IsValid) MlxInterop.ArrayFree(expertsSortedArray);
            if (downBiasArray.IsValid) MlxInterop.ArrayFree(downBiasArray);
            if (downRowsArray.IsValid) MlxInterop.ArrayFree(downRowsArray);
        }
    }

    /// <summary>
    /// Adds the Gemma 4 vision encoder's learned X/Y position tables to the patch hidden states:
    /// for patch row-major index <c>patch = py*patchesX+px</c> and channel <c>c</c>,
    /// <c>y[patch,c] = hidden[patch,c] + positions[0,px,c] + positions[1,py,c]</c>.
    ///
    /// <para><b>Ported from</b> TensorSharp's <c>MlxGemma4VisionOps.TryAddPosition2D</c>
    /// (<c>MlxGemma4VisionOps.cs</c> ~28-53), a custom fused Metal kernel
    /// (<c>Gemma4VisionPosition2D</c> in <c>MlxNative.cs</c>) in the real source, with a
    /// vector/scalar-channel split that only matters for SIMD width, not the result -- this port
    /// always does <c>value + xv + yv</c>, mathematically identical to both of that kernel's branches.
    /// Composed here via broadcasting add: reshape <paramref name="hidden"/> to
    /// <c>[patchesY, patchesX, channels]</c>, slice the X table to <c>[1, patchesX, channels]</c> and
    /// the Y table to <c>[patchesY, 1, channels]</c> so each broadcasts over the other's axis.</para>
    /// </summary>
    public static float[] Gemma4VisionAddPosition2D(ReadOnlySpan<float> hidden, ReadOnlySpan<float> positions,
        int patchesX, int patchesY, int channels, int maxPositions)
    {
        int rows = checked(patchesX * patchesY);
        if (hidden.Length != rows * channels) throw new ArgumentException($"hidden has {hidden.Length} elements, expected patchesX*patchesY*channels = {rows * channels}.", nameof(hidden));
        if (positions.Length != 2 * maxPositions * channels) throw new ArgumentException($"positions has {positions.Length} elements, expected 2*maxPositions*channels = {2 * maxPositions * channels}.", nameof(positions));
        if (maxPositions < patchesX || maxPositions < patchesY) throw new ArgumentException("maxPositions must be >= both patchesX and patchesY.", nameof(maxPositions));

        var stream = GetCachedStream();
        MlxInterop.MlxArray hidden3 = default, positionsArray = default, posXSlice = default, posYSlice = default,
            posYReshaped = default, sum1 = default, sum2 = default, result = default;
        try
        {
            var hiddenArray = UploadFloat32(hidden, [rows, channels]);
            try
            {
                Check(MlxInterop.Reshape(out hidden3, hiddenArray, [patchesY, patchesX, channels], 3, stream), "mlx_reshape (hidden3)");
            }
            finally
            {
                MlxInterop.ArrayFree(hiddenArray);
            }
            positionsArray = UploadFloat32(positions, [2, maxPositions, channels]);

            Check(MlxInterop.Slice(out posXSlice, positionsArray, starts: [0, 0, 0], startCount: 3,
                stops: [1, patchesX, channels], stopCount: 3, strides: [1, 1, 1], strideCount: 3, stream), "mlx_slice (posX)");
            Check(MlxInterop.Slice(out posYSlice, positionsArray, starts: [1, 0, 0], startCount: 3,
                stops: [2, patchesY, channels], stopCount: 3, strides: [1, 1, 1], strideCount: 3, stream), "mlx_slice (posY)");
            Check(MlxInterop.Reshape(out posYReshaped, posYSlice, [patchesY, 1, channels], 3, stream), "mlx_reshape (posYReshaped)");

            Check(MlxInterop.Add(out sum1, hidden3, posXSlice, stream), "mlx_add (sum1)");
            Check(MlxInterop.Add(out sum2, sum1, posYReshaped, stream), "mlx_add (sum2)");
            Check(MlxInterop.Reshape(out result, sum2, [rows, channels], 2, stream), "mlx_reshape (result)");

            var output = new float[rows * channels];
            DownloadFloat32(result, output, stream);
            return output;
        }
        finally
        {
            if (result.IsValid) MlxInterop.ArrayFree(result);
            if (sum2.IsValid) MlxInterop.ArrayFree(sum2);
            if (sum1.IsValid) MlxInterop.ArrayFree(sum1);
            if (posYReshaped.IsValid) MlxInterop.ArrayFree(posYReshaped);
            if (posYSlice.IsValid) MlxInterop.ArrayFree(posYSlice);
            if (posXSlice.IsValid) MlxInterop.ArrayFree(posXSlice);
            if (positionsArray.IsValid) MlxInterop.ArrayFree(positionsArray);
            if (hidden3.IsValid) MlxInterop.ArrayFree(hidden3);
        }
    }

    /// <summary>
    /// Non-overlapping 3x3 average pooling over the Gemma 4 vision encoder's patch grid: for merged
    /// row-major index <c>merged = my*mergedX+mx</c> (<c>mergedX = patchesX/3</c>, <c>mergedY =
    /// patchesY/3</c>), <c>y[merged,c] = mean of input[(my*3+ky)*patchesX + mx*3+kx, c]</c> over
    /// <c>ky,kx in [0,3)</c>. Incomplete trailing rows/columns are cropped, matching the real kernel.
    ///
    /// <para><b>Ported from</b> TensorSharp's <c>MlxGemma4VisionOps.TryPool3x3</c>
    /// (<c>MlxGemma4VisionOps.cs</c> ~61-77), a custom fused Metal kernel
    /// (<c>Gemma4VisionPool3x3</c> in <c>MlxNative.cs</c>) in the real source, which accumulates the 9
    /// terms in a fixed <c>ky</c>-outer/<c>kx</c>-inner order. Composed here via two strided-slice
    /// passes instead (pool the 3 columns of each box, then pool the 3 resulting rows) -- the same 9
    /// terms summed in a different grouping, mathematically equal up to float addition order.</para>
    /// </summary>
    public static float[] Gemma4VisionPool3x3(ReadOnlySpan<float> input, int patchesX, int patchesY, int channels)
    {
        if (patchesX < 3 || patchesY < 3) throw new ArgumentException("patchesX and patchesY must each be >= 3.", nameof(patchesX));
        int rows = checked(patchesX * patchesY);
        if (input.Length != rows * channels) throw new ArgumentException($"input has {input.Length} elements, expected patchesX*patchesY*channels = {rows * channels}.", nameof(input));
        int mergedX = patchesX / 3, mergedY = patchesY / 3;

        var stream = GetCachedStream();
        MlxInterop.MlxArray reshaped = default, sliceX0 = default, sliceX1 = default, sliceX2 = default,
            sumX01 = default, sumX = default, sliceY0 = default, sliceY1 = default, sliceY2 = default,
            sumY01 = default, sumXY = default, oneNinth = default, scaled = default, result = default;
        try
        {
            var inputArray = UploadFloat32(input, [rows, channels]);
            try
            {
                Check(MlxInterop.Reshape(out reshaped, inputArray, [patchesY, patchesX, channels], 3, stream), "mlx_reshape (reshaped)");
            }
            finally
            {
                MlxInterop.ArrayFree(inputArray);
            }

            Check(MlxInterop.Slice(out sliceX0, reshaped, [0, 0, 0], 3, [patchesY, mergedX * 3, channels], 3, [1, 3, 1], 3, stream), "mlx_slice (x0)");
            Check(MlxInterop.Slice(out sliceX1, reshaped, [0, 1, 0], 3, [patchesY, mergedX * 3, channels], 3, [1, 3, 1], 3, stream), "mlx_slice (x1)");
            Check(MlxInterop.Slice(out sliceX2, reshaped, [0, 2, 0], 3, [patchesY, mergedX * 3, channels], 3, [1, 3, 1], 3, stream), "mlx_slice (x2)");
            Check(MlxInterop.Add(out sumX01, sliceX0, sliceX1, stream), "mlx_add (sumX01)");
            Check(MlxInterop.Add(out sumX, sumX01, sliceX2, stream), "mlx_add (sumX)");

            Check(MlxInterop.Slice(out sliceY0, sumX, [0, 0, 0], 3, [mergedY * 3, mergedX, channels], 3, [3, 1, 1], 3, stream), "mlx_slice (y0)");
            Check(MlxInterop.Slice(out sliceY1, sumX, [1, 0, 0], 3, [mergedY * 3, mergedX, channels], 3, [3, 1, 1], 3, stream), "mlx_slice (y1)");
            Check(MlxInterop.Slice(out sliceY2, sumX, [2, 0, 0], 3, [mergedY * 3, mergedX, channels], 3, [3, 1, 1], 3, stream), "mlx_slice (y2)");
            Check(MlxInterop.Add(out sumY01, sliceY0, sliceY1, stream), "mlx_add (sumY01)");
            Check(MlxInterop.Add(out sumXY, sumY01, sliceY2, stream), "mlx_add (sumXY)");

            oneNinth = MlxInterop.ArrayNewFloat32(1f / 9f);
            Check(MlxInterop.Multiply(out scaled, sumXY, oneNinth, stream), "mlx_multiply (scaled)");
            Check(MlxInterop.Reshape(out result, scaled, [mergedY * mergedX, channels], 2, stream), "mlx_reshape (result)");

            var output = new float[mergedY * mergedX * channels];
            DownloadFloat32(result, output, stream);
            return output;
        }
        finally
        {
            if (result.IsValid) MlxInterop.ArrayFree(result);
            if (scaled.IsValid) MlxInterop.ArrayFree(scaled);
            if (oneNinth.IsValid) MlxInterop.ArrayFree(oneNinth);
            if (sumXY.IsValid) MlxInterop.ArrayFree(sumXY);
            if (sumY01.IsValid) MlxInterop.ArrayFree(sumY01);
            if (sliceY2.IsValid) MlxInterop.ArrayFree(sliceY2);
            if (sliceY1.IsValid) MlxInterop.ArrayFree(sliceY1);
            if (sliceY0.IsValid) MlxInterop.ArrayFree(sliceY0);
            if (sumX.IsValid) MlxInterop.ArrayFree(sumX);
            if (sumX01.IsValid) MlxInterop.ArrayFree(sumX01);
            if (sliceX2.IsValid) MlxInterop.ArrayFree(sliceX2);
            if (sliceX1.IsValid) MlxInterop.ArrayFree(sliceX1);
            if (sliceX0.IsValid) MlxInterop.ArrayFree(sliceX0);
            if (reshaped.IsValid) MlxInterop.ArrayFree(reshaped);
        }
    }

    /// <summary>
    /// Gemma 4 vision's 2D RoPE: each head's <paramref name="headDim"/> channels split into an X half
    /// and a Y half (each <c>headDim/2</c> wide), each half further split into rotate-half pairs
    /// (<c>headDim/4</c>-wide first/second). <paramref name="cosine"/>/<paramref name="sine"/> are
    /// <c>[tokens, headDim/2]</c>: columns <c>[0, headDim/4)</c> rotate the X half, columns
    /// <c>[headDim/4, headDim/2)</c> rotate the Y half.
    ///
    /// <para><b>Ported from</b> TensorSharp's <c>MlxGemma4VisionOps.TryRope2D</c>
    /// (<c>MlxGemma4VisionOps.cs</c> ~164-183), a custom fused Metal kernel
    /// (<c>Gemma4VisionRope2D</c> in <c>MlxNative.cs</c>) in the real source. Composed here as two
    /// ordinary rotate-half RoPE applications (slice the half, split first/second quarter, <c>first*cos
    /// - second*sin</c> / <c>first*sin + second*cos</c>, concatenate) using the matching quarter of the
    /// cos/sin table for each half.</para>
    /// </summary>
    public static float[] Gemma4VisionRope2D(ReadOnlySpan<float> data, ReadOnlySpan<float> cosine, ReadOnlySpan<float> sine,
        int tokens, int heads, int headDim)
    {
        if (headDim <= 0 || headDim % 4 != 0) throw new ArgumentException("headDim must be a positive multiple of 4.", nameof(headDim));
        int halfWidth = headDim / 2, quarter = headDim / 4;
        if (data.Length != tokens * heads * headDim) throw new ArgumentException($"data has {data.Length} elements, expected tokens*heads*headDim = {tokens * heads * headDim}.", nameof(data));
        if (cosine.Length != tokens * halfWidth) throw new ArgumentException($"cosine has {cosine.Length} elements, expected tokens*headDim/2 = {tokens * halfWidth}.", nameof(cosine));
        if (sine.Length != tokens * halfWidth) throw new ArgumentException($"sine has {sine.Length} elements, expected tokens*headDim/2 = {tokens * halfWidth}.", nameof(sine));

        var stream = GetCachedStream();
        MlxInterop.MlxArray reshaped = default, cosArray = default, sinArray = default, result = default;
        var toFree = new List<MlxInterop.MlxArray>();
        try
        {
            var dataArray = UploadFloat32(data, [tokens, heads * headDim]);
            try
            {
                Check(MlxInterop.Reshape(out reshaped, dataArray, [tokens, heads, headDim], 3, stream), "mlx_reshape (reshaped)");
            }
            finally
            {
                MlxInterop.ArrayFree(dataArray);
            }
            cosArray = UploadFloat32(cosine, [tokens, halfWidth]);
            sinArray = UploadFloat32(sine, [tokens, halfWidth]);

            MlxInterop.MlxArray xRotated = RotateHalf(reshaped, cosArray, sinArray, tokens, heads, halfWidth, quarter, halfStart: 0, quarterStart: 0, stream, toFree);
            MlxInterop.MlxArray yRotated = RotateHalf(reshaped, cosArray, sinArray, tokens, heads, halfWidth, quarter, halfStart: halfWidth, quarterStart: quarter, stream, toFree);

            MlxInterop.MlxArray result3d = ConcatenateAxis(xRotated, yRotated, axis: 2, stream);
            toFree.Add(result3d);
            Check(MlxInterop.Reshape(out result, result3d, [tokens, heads * headDim], 2, stream), "mlx_reshape (result)");

            var output = new float[tokens * heads * headDim];
            DownloadFloat32(result, output, stream);
            return output;
        }
        finally
        {
            if (result.IsValid) MlxInterop.ArrayFree(result);
            for (int i = toFree.Count - 1; i >= 0; i--)
                if (toFree[i].IsValid) MlxInterop.ArrayFree(toFree[i]);
            if (sinArray.IsValid) MlxInterop.ArrayFree(sinArray);
            if (cosArray.IsValid) MlxInterop.ArrayFree(cosArray);
            if (reshaped.IsValid) MlxInterop.ArrayFree(reshaped);
        }
    }

    /// <summary>One rotate-half RoPE application over a <c>halfWidth</c>-wide slice of the head
    /// dimension starting at <paramref name="halfStart"/>, using the <paramref name="quarter"/>-wide
    /// cos/sin columns starting at <paramref name="quarterStart"/>. Every intermediate array is
    /// appended to <paramref name="toFree"/> for the caller to free after downloading the final
    /// result (this helper's own arrays must outlive its return value's consumers).</summary>
    private static MlxInterop.MlxArray RotateHalf(MlxInterop.MlxArray reshaped, MlxInterop.MlxArray cosArray, MlxInterop.MlxArray sinArray,
        int tokens, int heads, int halfWidth, int quarter, int halfStart, int quarterStart, MlxInterop.MlxStream stream,
        List<MlxInterop.MlxArray> toFree)
    {
        Check(MlxInterop.Slice(out var half, reshaped, [0, 0, halfStart], 3, [tokens, heads, halfStart + halfWidth], 3, [1, 1, 1], 3, stream), "mlx_slice (half)");
        toFree.Add(half);
        Check(MlxInterop.Slice(out var first, half, [0, 0, 0], 3, [tokens, heads, quarter], 3, [1, 1, 1], 3, stream), "mlx_slice (first)");
        toFree.Add(first);
        Check(MlxInterop.Slice(out var second, half, [0, 0, quarter], 3, [tokens, heads, halfWidth], 3, [1, 1, 1], 3, stream), "mlx_slice (second)");
        toFree.Add(second);

        Check(MlxInterop.Slice(out var cosSlice, cosArray, [0, quarterStart], 2, [tokens, quarterStart + quarter], 2, [1, 1], 2, stream), "mlx_slice (cos)");
        toFree.Add(cosSlice);
        Check(MlxInterop.Slice(out var sinSlice, sinArray, [0, quarterStart], 2, [tokens, quarterStart + quarter], 2, [1, 1], 2, stream), "mlx_slice (sin)");
        toFree.Add(sinSlice);
        Check(MlxInterop.Reshape(out var cosReshaped, cosSlice, [tokens, 1, quarter], 3, stream), "mlx_reshape (cos)");
        toFree.Add(cosReshaped);
        Check(MlxInterop.Reshape(out var sinReshaped, sinSlice, [tokens, 1, quarter], 3, stream), "mlx_reshape (sin)");
        toFree.Add(sinReshaped);

        Check(MlxInterop.Multiply(out var firstCos, first, cosReshaped, stream), "mlx_multiply (firstCos)");
        toFree.Add(firstCos);
        Check(MlxInterop.Multiply(out var secondSin, second, sinReshaped, stream), "mlx_multiply (secondSin)");
        toFree.Add(secondSin);
        Check(MlxInterop.Subtract(out var rotatedFirst, firstCos, secondSin, stream), "mlx_subtract (rotatedFirst)");
        toFree.Add(rotatedFirst);

        Check(MlxInterop.Multiply(out var firstSin, first, sinReshaped, stream), "mlx_multiply (firstSin)");
        toFree.Add(firstSin);
        Check(MlxInterop.Multiply(out var secondCos, second, cosReshaped, stream), "mlx_multiply (secondCos)");
        toFree.Add(secondCos);
        Check(MlxInterop.Add(out var rotatedSecond, firstSin, secondCos, stream), "mlx_add (rotatedSecond)");
        toFree.Add(rotatedSecond);

        var rotated = ConcatenateAxis(rotatedFirst, rotatedSecond, axis: 2, stream);
        toFree.Add(rotated);
        return rotated;
    }

    private static MlxInterop.MlxArray ConcatenateAxis(MlxInterop.MlxArray first, MlxInterop.MlxArray second, int axis, MlxInterop.MlxStream stream)
    {
        var vector = MlxInterop.VectorArrayNew();
        try
        {
            Check(MlxInterop.VectorArrayAppendValue(vector, first), "mlx_vector_array_append_value (first)");
            Check(MlxInterop.VectorArrayAppendValue(vector, second), "mlx_vector_array_append_value (second)");
            Check(MlxInterop.ConcatenateAxis(out var result, vector, axis, stream), "mlx_concatenate_axis");
            return result;
        }
        finally
        {
            if (vector.IsValid) MlxInterop.VectorArrayFree(vector);
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

    private static unsafe void DownloadInt32(MlxInterop.MlxArray array, Span<int> destination, MlxInterop.MlxStream stream)
    {
        Check(MlxInterop.Contiguous(out var contiguous, array, allowColMajor: false, stream), "mlx_contiguous");
        try
        {
            Check(MlxInterop.ArrayEval(contiguous, stream), "mlx_array_eval");
            nint src = MlxInterop.ArrayDataInt32(contiguous);
            if (src == nint.Zero)
                throw new InvalidOperationException("mlx_array_data_int32 returned null.");
            new ReadOnlySpan<int>((void*)src, destination.Length).CopyTo(destination);
        }
        finally
        {
            MlxInterop.ArrayFree(contiguous);
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

    private static unsafe void DownloadFloat32(MlxInterop.MlxArray array, Span<float> destination, MlxInterop.MlxStream stream)
    {
        Check(MlxInterop.Contiguous(out var contiguous, array, allowColMajor: false, stream), "mlx_contiguous");
        try
        {
            Check(MlxInterop.ArrayEval(contiguous, stream), "mlx_array_eval");
            nint src = MlxInterop.ArrayDataFloat32(contiguous);
            if (src == nint.Zero)
                throw new InvalidOperationException("mlx_array_data_float32 returned null.");
            new ReadOnlySpan<float>((void*)src, destination.Length).CopyTo(destination);
        }
        finally
        {
            MlxInterop.ArrayFree(contiguous);
        }
    }

    private static void Check(int returnCode, string call)
    {
        if (returnCode != 0)
            throw new InvalidOperationException($"{call} failed with mlx-c return code {returnCode}.");
    }
}
