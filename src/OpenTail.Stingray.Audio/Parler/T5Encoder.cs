
namespace OpenTail.Stingray.Audio.Parler;

/// <summary>
/// Real T5 encoder forward pass, transcribed directly from the real `transformers` Python
/// package's `modeling_t5.py` (`T5Attention`/`T5LayerNorm`/`T5DenseGatedActDense`), fetched from
/// the locally-installed `transformers` package, not re-derived from memory -- see
/// <see cref="T5EncoderWeights"/>'s doc comment and docs/audio-review-progress.md's Parler-TTS
/// section for the full derivation.
///
/// <para><b>Three real, easy-to-get-wrong T5-specific quirks, confirmed from source, do not
/// "fix" any of these back to a standard-transformer assumption</b>:
/// (1) Attention scores are NOT scaled by <c>1/sqrt(head_dim)</c> -- T5 omits this scaling
/// entirely (confirmed: `scores = torch.matmul(query_states, key_states.transpose(3, 2))`, no
/// division anywhere in the real source).
/// (2) `T5LayerNorm` is a pure RMSNorm variant: <c>x * rsqrt(mean(x^2) + eps) * weight</c> -- NO
/// bias, NO mean-subtraction (confirmed from the real class's own doc comment: "No bias and no
/// subtraction of mean").
/// (3) The relative position bias is computed ONCE, using ONLY block 0's
/// <c>relative_attention_bias</c> table, and the SAME bias tensor is reused/added into every
/// subsequent layer's attention scores -- it is NOT recomputed per layer (confirmed: only block
/// 0 has a real `relative_attention_bias.weight` tensor in the checkpoint; `compute_bias` is
/// called once and the result threaded through as `position_bias` in the real source).</para>
///
/// <para><b>DRY/perf pass (2026-08-31)</b>: rewritten from a jagged `float[][]` (one array per
/// token) representation with per-row `Parallel.For(0, t, i => weight.MatVec(x[i]))` calls to a
/// flat `float[]` (row-major [t, DModel]) representation using `CfmLinearWeight.MatMul`'s
/// existing batched (all `t` rows in one call) path instead -- same real math, same real weights,
/// just data layout. Per CLAUDE.md rule 7, only keep if measurably faster: this real CPU
/// architecture change is also what unblocks the Vulkan `--backend` option (docs/052 §5's
/// "T5Encoder's real blocker isn't weight format, it's call shape" note) for a later pass.</para>
/// </summary>
public static class T5Encoder
{
    /// <summary>Runs the full T5 encoder. `tokenIds` -&gt; embed -&gt; 24x T5 layer (self-attn + gated-GELU FFN) -&gt; final RMSNorm. Returns [t][DModel] (public API unchanged; internally batched flat, see class doc comment).
    /// <paramref name="backend"/>, when supplied, routes the Q/K/V/O and FFN projections through
    /// <see cref="CfmLinearWeight.GpuMatMul"/> (--backend vulkan; only viable now that the encoder
    /// is batched -- see docs/052-vulkan-backend-for-tts-engines-plan.md §5).</summary>
    public static float[][] Forward(T5EncoderWeights w, int[] tokenIds, IComputeBackend? backend = null)
    {
        int t = tokenIds.Length;
        int dim = T5EncoderWeights.DModel;
        var x = new float[t * dim];
        for (int i = 0; i < t; i++)
            Array.Copy(w.SharedEmbedding, (long)tokenIds[i] * dim, x, (long)i * dim, dim);

        var positionBias = ComputeRelativePositionBias(w, t);

        foreach (var layer in w.Layers)
            x = T5Layer(x, layer, t, positionBias, backend);

        var flatOut = new float[t * dim];
        Parallel.For(0, t, i => T5LayerNorm(x.AsSpan(i * dim, dim), w.FinalLayerNormWeight, flatOut.AsSpan(i * dim, dim)));

        var output = new float[t][];
        for (int i = 0; i < t; i++)
        {
            output[i] = new float[dim];
            Array.Copy(flatOut, i * dim, output[i], 0, dim);
        }
        return output;
    }

    /// <summary>
    /// GPU-resident forward pass, mirroring FLUX's own `T5GpuWeights`/`T5GpuWorkspace`/encode-loop
    /// structure exactly (duplicated as <see cref="ParlerT5GpuWeights"/>/
    /// <see cref="ParlerT5GpuWorkspace"/> due to a circular-project-reference constraint -- see
    /// those classes' own doc comments) -- Parler's text encoder is a real, standard T5 encoder
    /// using the exact same tensor names/math conventions (unscaled attention + additive
    /// relative-position bias, RMSNorm, gated-GELU FFN) as FLUX's own T5-XXL, whose GPU residency
    /// is already done and proven (docs/071); only the dims differ (T5-Large:
    /// dim=1024/heads=16/headDim=64/ffDim=2816 vs T5-XXL's 4096/64/64/10240).
    /// <paramref name="gpuWeights"/> must be built with a `getWeight` delegate reading
    /// `text_encoder.{name}` from the same real loader `T5EncoderWeights` was constructed from
    /// (see docs/080's Parler entry for the exact call-site pattern), and
    /// <paramref name="gpuWorkspace"/> with this call's own flat relative-position-bias array
    /// (<see cref="ComputeRelativePositionBiasFlat"/>).
    /// </summary>
    public static float[] EncodeGpu(T5EncoderWeights w, int[] tokenIds, ParlerT5GpuWeights gpuWeights, ParlerT5GpuWorkspace gpuWorkspace, IVisionOpsBackend backend)
    {
        int t = tokenIds.Length;
        int dim = T5EncoderWeights.DModel;
        int ffDim = T5EncoderWeights.DFf;

        var xHost = new float[t * dim];
        for (int i = 0; i < t; i++)
            Array.Copy(w.SharedEmbedding, (long)tokenIds[i] * dim, xHost, (long)i * dim, dim);

        var imageOps = (IImageOpsBackend)backend;
        using (var xInit = backend.Upload(xHost, TensorShape.D2(t, dim), exact: true))
        {
            imageOps.ScaleInPlace(gpuWorkspace.X, 0f);
            imageOps.AddInPlace(gpuWorkspace.X, xInit);
        }

        for (int i = 0; i < T5EncoderWeights.NumLayers; i++)
        {
            var lw = gpuWeights.Layers[i];
            backend.RmsNormBatched(gpuWorkspace.XNorm, gpuWorkspace.X, lw.LayerNorm0Weight, dim, t, eps: 1e-6f);

            imageOps.Sgemm(gpuWorkspace.Q, gpuWorkspace.XNorm, lw.QWeight, t, dim, dim);
            imageOps.Sgemm(gpuWorkspace.K, gpuWorkspace.XNorm, lw.KWeight, t, dim, dim);
            imageOps.Sgemm(gpuWorkspace.V, gpuWorkspace.XNorm, lw.VWeight, t, dim, dim);

            backend.T5MultiHeadAttentionRelBias(gpuWorkspace.AttnOut, gpuWorkspace.Q, gpuWorkspace.K, gpuWorkspace.V, gpuWorkspace.RelPosBias, t, t, T5EncoderWeights.NumHeads, T5EncoderWeights.DKv);

            imageOps.Sgemm(gpuWorkspace.XNorm, gpuWorkspace.AttnOut, lw.OWeight, t, dim, dim);
            imageOps.AddInPlace(gpuWorkspace.X, gpuWorkspace.XNorm);

            backend.RmsNormBatched(gpuWorkspace.XNorm, gpuWorkspace.X, lw.LayerNorm1Weight, dim, t, eps: 1e-6f);
            imageOps.Sgemm(gpuWorkspace.Gate, gpuWorkspace.XNorm, lw.Wi0Weight, t, dim, ffDim);
            imageOps.Sgemm(gpuWorkspace.Val, gpuWorkspace.XNorm, lw.Wi1Weight, t, dim, ffDim);
            imageOps.GeluTanhMul(gpuWorkspace.Gate, gpuWorkspace.Val);
            imageOps.Sgemm(gpuWorkspace.FfOut, gpuWorkspace.Gate, lw.WoWight, t, ffDim, dim);
            imageOps.AddInPlace(gpuWorkspace.X, gpuWorkspace.FfOut);
        }

        backend.RmsNormBatched(gpuWorkspace.X, gpuWorkspace.X, gpuWeights.FinalLayerNormWeight, dim, t, eps: 1e-6f);

        var result = new float[t * dim];
        backend.Download(gpuWorkspace.X, result);
        return result;
    }

    /// <summary>Same real relative-position-bucket formula as <see cref="ComputeRelativePositionBias"/>
    /// (reused unchanged, not re-derived), flattened to the `[heads, seq, seq]` layout
    /// <see cref="Diffusion.T5GpuWorkspace"/> expects instead of that method's own per-head jagged
    /// `float[][,]` shape.</summary>
    public static float[] ComputeRelativePositionBiasFlat(T5EncoderWeights w, int t)
    {
        var jagged = ComputeRelativePositionBias(w, t);
        var flat = new float[T5EncoderWeights.NumHeads * t * t];
        for (int h = 0; h < T5EncoderWeights.NumHeads; h++)
            for (int i = 0; i < t; i++)
                for (int j = 0; j < t; j++)
                    flat[(h * t + i) * t + j] = jagged[h][i, j];
        return flat;
    }

    /// <summary>Test-only entry point exposing one CPU layer's output directly, for isolating
    /// GPU-vs-CPU discrepancies to a single layer instead of the full 24-layer stack (mirrors
    /// F5TTS's own single-block debugging pattern, see docs/080's Parler entry).</summary>
    public static float[] RunSingleLayerForTest(float[] x, T5LayerWeights lw, int t, float[][,] positionBias) =>
        T5Layer(x, lw, t, positionBias, backend: null);

    /// <summary>Test-only: replays just T5Layer's attention sub-layer (norm -> self-attn -> residual),
    /// stopping before the FFN sub-layer, to isolate a GPU-vs-CPU discrepancy to attention vs FFN.</summary>
    public static float[] RunAttentionOnlyForTest(float[] x, T5LayerWeights lw, int t, float[][,] positionBias)
    {
        int dim = T5EncoderWeights.DModel;
        var normed1 = new float[t * dim];
        Parallel.For(0, t, i => T5LayerNorm(x.AsSpan(i * dim, dim), lw.SelfAttnLayerNormWeight, normed1.AsSpan(i * dim, dim)));
        var attnOut = SelfAttention(normed1, lw, t, positionBias, backend: null);
        var afterAttn = new float[t * dim];
        TensorPrimitives.Add(x, attnOut, afterAttn);
        return afterAttn;
    }

    private static float[] T5Layer(float[] x, T5LayerWeights lw, int t, float[][,] positionBias, IComputeBackend? backend = null)
    {
        int dim = T5EncoderWeights.DModel;
        var normed1 = new float[t * dim];
        Parallel.For(0, t, i => T5LayerNorm(x.AsSpan(i * dim, dim), lw.SelfAttnLayerNormWeight, normed1.AsSpan(i * dim, dim)));

        var attnOut = SelfAttention(normed1, lw, t, positionBias, backend);

        var afterAttn = new float[t * dim];
        TensorPrimitives.Add(x, attnOut, afterAttn);

        var normed2 = new float[t * dim];
        Parallel.For(0, t, i => T5LayerNorm(afterAttn.AsSpan(i * dim, dim), lw.FfnLayerNormWeight, normed2.AsSpan(i * dim, dim)));

        var ffnOut = GatedFfn(normed2, lw, t, backend);

        var output = new float[t * dim];
        TensorPrimitives.Add(afterAttn, ffnOut, output);
        return output;
    }

    /// <summary>Real T5 self-attention: NO 1/sqrt(headDim) scaling, plus the shared relative position bias added to raw scores before softmax.</summary>
    private static unsafe float[] SelfAttention(float[] x, T5LayerWeights lw, int t, float[][,] positionBias, IComputeBackend? backend = null)
    {
        int dim = T5EncoderWeights.DModel;
        int nHeads = T5EncoderWeights.NumHeads;
        int dKv = T5EncoderWeights.DKv;
        int qkvDim = nHeads * dKv; // 1024, equals DModel for this config

        var q = new float[t * qkvDim];
        var k = new float[t * qkvDim];
        var v = new float[t * qkvDim];
        fixed (float* xp = x, qp = q, kp = k, vp = v)
        {
            if (backend is not null)
            {
                lw.SelfAttnQWeight.GpuMatMul(backend, xp, t, qp);
                lw.SelfAttnKWeight.GpuMatMul(backend, xp, t, kp);
                lw.SelfAttnVWeight.GpuMatMul(backend, xp, t, vp);
            }
            else
            {
                lw.SelfAttnQWeight.MatMul(xp, t, qp);
                lw.SelfAttnKWeight.MatMul(xp, t, kp);
                lw.SelfAttnVWeight.MatMul(xp, t, vp);
            }
        }

        var context = new float[t * qkvDim];

        Parallel.For(0, nHeads, h =>
        {
            int off = h * dKv;
            var scores = new float[t];
            for (int i = 0; i < t; i++)
            {
                for (int j = 0; j < t; j++)
                {
                    float dot = 0f;
                    for (int d = 0; d < dKv; d++) dot += q[i * qkvDim + off + d] * k[j * qkvDim + off + d];
                    scores[j] = dot + positionBias[h][i, j]; // NO scaling -- real T5 quirk
                }
                SoftmaxInPlace(scores);

                var ctxSpan = context.AsSpan(i * qkvDim + off, dKv);
                for (int j = 0; j < t; j++)
                {
                    float s = scores[j];
                    var vSpan = v.AsSpan(j * qkvDim + off, dKv);
                    for (int d = 0; d < dKv; d++) ctxSpan[d] += s * vSpan[d];
                }
            }
        });

        var output = new float[t * dim];
        fixed (float* cp = context, op = output)
        {
            if (backend is not null) lw.SelfAttnOWeight.GpuMatMul(backend, cp, t, op);
            else lw.SelfAttnOWeight.MatMul(cp, t, op);
        }
        return output;
    }

    /// <summary>Real T5DenseGatedActDense: `wo(gelu_new(wi_0(x)) * wi_1(x))`, no biases anywhere.</summary>
    private static unsafe float[] GatedFfn(float[] x, T5LayerWeights lw, int t, IComputeBackend? backend = null)
    {
        int dim = T5EncoderWeights.DModel;
        int ff = T5EncoderWeights.DFf;
        var gate = new float[t * ff];
        var up = new float[t * ff];
        fixed (float* xp = x, gp = gate, up_ = up)
        {
            if (backend is not null)
            {
                lw.FfnWi0Weight.GpuMatMul(backend, xp, t, gp);
                lw.FfnWi1Weight.GpuMatMul(backend, xp, t, up_);
            }
            else
            {
                lw.FfnWi0Weight.MatMul(xp, t, gp);
                lw.FfnWi1Weight.MatMul(xp, t, up_);
            }
        }

        for (int i = 0; i < gate.Length; i++)
            gate[i] = GeluNew(gate[i]) * up[i];

        var output = new float[t * dim];
        fixed (float* gp = gate, op = output)
        {
            if (backend is not null) lw.FfnWoWeight.GpuMatMul(backend, gp, t, op);
            else lw.FfnWoWeight.MatMul(gp, t, op);
        }
        return output;
    }

    /// <summary>Real "gelu_new" (tanh approximation), matching Parler's real `dense_act_fn=gelu_new` config.</summary>
    private static float GeluNew(float x) =>
        0.5f * x * (1f + MathF.Tanh(0.7978845608f * (x + 0.044715f * x * x * x)));

    /// <summary>Real T5 relative position bucketing + bias lookup, computed once for the whole sequence and shared across all layers. Bidirectional (encoder, not decoder) -- confirmed from the real `compute_bias`/`_relative_position_bucket` source.</summary>
    private static float[][,] ComputeRelativePositionBias(T5EncoderWeights w, int t)
    {
        var bias = new float[T5EncoderWeights.NumHeads][,];
        for (int h = 0; h < T5EncoderWeights.NumHeads; h++) bias[h] = new float[t, t];

        for (int qi = 0; qi < t; qi++)
        {
            for (int kj = 0; kj < t; kj++)
            {
                int relPos = kj - qi;
                int bucket = RelativePositionBucket(relPos, bidirectional: true,
                    T5EncoderWeights.RelativeAttentionNumBuckets, T5EncoderWeights.RelativeAttentionMaxDistance);
                for (int h = 0; h < T5EncoderWeights.NumHeads; h++)
                    bias[h][qi, kj] = w.RelativeAttentionBias[bucket * T5EncoderWeights.NumHeads + h];
            }
        }
        return bias;
    }

    /// <summary>Real `_relative_position_bucket`, transcribed exactly from `transformers/models/t5/modeling_t5.py`.</summary>
    private static int RelativePositionBucket(int relativePosition, bool bidirectional, int numBuckets, int maxDistance)
    {
        int relativeBuckets = 0;
        if (bidirectional)
        {
            numBuckets /= 2;
            relativeBuckets += relativePosition > 0 ? numBuckets : 0;
            relativePosition = Math.Abs(relativePosition);
        }
        else
        {
            relativePosition = -Math.Min(relativePosition, 0);
        }

        int maxExact = numBuckets / 2;
        bool isSmall = relativePosition < maxExact;

        int relativePositionIfLarge = maxExact + (int)(
            MathF.Log(relativePosition / (float)maxExact)
            / MathF.Log(maxDistance / (float)maxExact)
            * (numBuckets - maxExact));
        relativePositionIfLarge = Math.Min(relativePositionIfLarge, numBuckets - 1);

        relativeBuckets += isSmall ? relativePosition : relativePositionIfLarge;
        return relativeBuckets;
    }

    /// <summary>Real T5LayerNorm: pure RMSNorm, NO bias, NO mean-subtraction.</summary>
    private static void T5LayerNorm(ReadOnlySpan<float> x, float[] weight, Span<float> output, float eps = 1e-6f)
    {
        int n = x.Length;
        float sumSq = 0f;
        for (int i = 0; i < n; i++) sumSq += x[i] * x[i];
        float invRms = 1f / MathF.Sqrt(sumSq / n + eps);
        for (int i = 0; i < n; i++) output[i] = x[i] * invRms * weight[i];
    }

    private static void SoftmaxInPlace(float[] scores)
    {
        float max = float.NegativeInfinity;
        for (int i = 0; i < scores.Length; i++) if (scores[i] > max) max = scores[i];
        float sum = 0f;
        for (int i = 0; i < scores.Length; i++)
        {
            float e = MathF.Exp(scores[i] - max);
            scores[i] = e;
            sum += e;
        }
        float invSum = 1f / sum;
        for (int i = 0; i < scores.Length; i++) scores[i] *= invSum;
    }
}
