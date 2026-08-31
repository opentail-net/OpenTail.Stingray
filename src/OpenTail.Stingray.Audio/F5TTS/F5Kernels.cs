using CoreTensor = OpenTail.Stingray.Core.Tensor;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Audio.F5TTS;

/// <summary>
/// Low-level math kernels for F5-TTS's DiT, all operating on SEQUENCE-MAJOR (channel-last)
/// arrays [T, D] (row-major, x[t*D+d]) -- matching the reference PyTorch's native `b n d` tensor
/// layout directly, unlike the VITS-family pipelines' channel-first [D, T] convention (a
/// deliberate choice per-family, not an inconsistency: DiT/Transformer math is naturally
/// per-timestep-row, whereas VITS's convs are naturally per-channel-row).
/// </summary>
public static class F5Kernels
{
    /// <summary>Per-timestep Linear: y[t,o] = bias[o] + sum_i weight[o*inDim+i]*x[t,i]. weight is torch's native [outDim,inDim] row-major (no transpose quirk -- safetensors, not ONNX).</summary>
    public static float[] Linear(float[] x, int t, int inDim, float[] weight, float[]? bias, int outDim)
    {
        var y = new float[t * outDim];
        unsafe
        {
            fixed (float* xp = x, wp = weight, yp = y, bp = bias)
            {
                float* xpLocal = xp;
                float* wpLocal = wp;
                float* ypLocal = yp;
                float* bpLocal = bp;

                if (t == 1)
                {
                    Parallel.For(0, outDim, o =>
                    {
                        float b = bpLocal != null ? bpLocal[o] : 0f;
                        ypLocal[o] = b + OpenTail.Stingray.Cpu.SimdKernels.DotF32(wpLocal + o * inDim, xpLocal, inDim);
                    });
                }
                else
                {
                    Parallel.For(0, t, ti =>
                    {
                        float* xRow = xpLocal + ti * inDim;
                        float* yRow = ypLocal + ti * outDim;
                        for (int o = 0; o < outDim; o++)
                        {
                            float b = bpLocal != null ? bpLocal[o] : 0f;
                            yRow[o] = b + OpenTail.Stingray.Cpu.SimdKernels.DotF32(wpLocal + o * inDim, xRow, inDim);
                        }
                    });
                }
            }
        }
        return y;
    }

    // Per-weight-array persistent GPU tensor cache for the --backend vulkan path (see docs/052-
    // vulkan-backend-for-tts-engines-plan.md). Keyed by the weight array's own reference identity
    // (each DiT block's weight array is a distinct, stable instance for the pipeline's lifetime),
    // so weights upload once and every subsequent call reuses the cached GPU tensor -- same
    // convention as CfmLinearWeight.GpuMatMul (Primitives), just keyed externally since F5Kernels
    // operates on raw float[] rather than a dedicated weight-wrapper type.
    private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<float[], CoreTensor> GpuWeightCache = new();

    /// <summary>GPU-dispatched equivalent of <see cref="Linear"/> via an <see cref="IComputeBackend"/>
    /// (--backend vulkan): y[t,o] = bias[o] + weight[o,:] . x[t,:]. Weight uploaded once (cached by
    /// array identity) and reused; bias added on CPU after download (negligible next to the matmul,
    /// no fused bias-add on the generic backend surface).</summary>
    public static unsafe float[] LinearGpu(IComputeBackend backend, float[] x, int t, int inDim, float[] weight, float[]? bias, int outDim)
    {
        if (!GpuWeightCache.TryGetValue(weight, out var wGpu))
        {
            wGpu = backend.Upload(weight, TensorShape.D1(weight.Length), exact: true);
            GpuWeightCache.Add(weight, wGpu);
        }

        var xGpu = backend.Upload(x, TensorShape.D1(t * inDim));
        var cGpu = backend.Allocate(TensorShape.D1(t * outDim));
        var y = new float[t * outDim];
        try
        {
            backend.Sgemm(cGpu, xGpu, wGpu, t, inDim, outDim);
            backend.Synchronize();
            backend.Download(cGpu, y);
        }
        finally
        {
            backend.Free(xGpu);
            backend.Free(cGpu);
        }

        if (bias is not null)
        {
            fixed (float* yp = y, bp = bias)
            {
                for (int row = 0; row < t; row++)
                {
                    float* yRow = yp + row * outDim;
                    for (int o = 0; o < outDim; o++) yRow[o] += bp[o];
                }
            }
        }
        return y;
    }

    /// <summary>Per-timestep Linear with Q8_0 weights: y[t,o] = bias[o] + MatVecQ8_0(weight, x[t]).</summary>
    public static unsafe float[] LinearQ8_0(float[] x, int t, int inDim, byte[] q8Weight, float[]? bias, int outDim)
    {
        var y = new float[t * outDim];
        fixed (float* xp = x, yp = y, bp = bias)
        fixed (byte* wp = q8Weight)
        {
            float* xpLocal = xp;
            float* ypLocal = yp;
            byte* wpLocal = wp;
            float* bpLocal = bp;

            if (t == 1)
            {
                OpenTail.Stingray.Cpu.SimdKernels.MatVecQ8_0(ypLocal, wpLocal, xpLocal, outDim, inDim);
                if (bpLocal != null)
                {
                    var ySpan = new Span<float>(ypLocal, outDim);
                    var bSpan = new ReadOnlySpan<float>(bpLocal, outDim);
                    System.Numerics.Tensors.TensorPrimitives.Add(ySpan, bSpan, ySpan);
                }
            }
            else
            {
                Parallel.For(0, t, ti =>
                {
                    float* xRow = xpLocal + ti * inDim;
                    float* yRow = ypLocal + ti * outDim;
                    OpenTail.Stingray.Cpu.SimdKernels.MatVecQ8_0(yRow, wpLocal, xRow, outDim, inDim);
                    if (bpLocal != null)
                    {
                        var ySpan = new Span<float>(yRow, outDim);
                        var bSpan = new ReadOnlySpan<float>(bpLocal, outDim);
                        System.Numerics.Tensors.TensorPrimitives.Add(ySpan, bSpan, ySpan);
                    }
                });
            }
        }
        return y;
    }

    /// <summary>Per-timestep affine LayerNorm over the last (channel) dim.</summary>
    public static float[] LayerNorm(float[] x, int t, int dim, float[] gamma, float[] beta, float eps = 1e-6f)
    {
        var y = new float[t * dim];
        Parallel.For(0, t, ti =>
        {
            int off = ti * dim;
            double mean = 0;
            for (int d = 0; d < dim; d++) mean += x[off + d];
            mean /= dim;
            double var = 0;
            for (int d = 0; d < dim; d++) { double diff = x[off + d] - mean; var += diff * diff; }
            var /= dim;
            float invStd = (float)(1.0 / Math.Sqrt(var + eps));
            for (int d = 0; d < dim; d++)
                y[off + d] = (float)((x[off + d] - mean) * invStd) * gamma[d] + beta[d];
        });
        return y;
    }

    /// <summary>Per-timestep non-affine LayerNorm (elementwise_affine=False), for AdaLN modulation.</summary>
    public static unsafe float[] LayerNormNoAffine(float[] x, int t, int dim, float eps = 1e-6f)
    {
        var y = new float[t * dim];
        fixed (float* xp = x, yp = y)
        {
            float* xpLocal = xp;
            float* ypLocal = yp;
            Parallel.For(0, t, ti =>
            {
                int off = ti * dim;
                OpenTail.Stingray.Cpu.SimdKernels.PureLayerNorm(ypLocal + off, xpLocal + off, dim, eps);
            });
        }
        return y;
    }

    /// <summary>AdaLN-Zero scale/shift modulation, `dst = src * (1 + modulation[scaleOffset..]) + modulation[shiftOffset..]`,
    /// reading scale/shift as two `dim`-length slices out of a single combined `modulation` buffer
    /// (the real chunked-Linear-output layout every AdaLN variant in this codebase's DiT family
    /// uses -- F5-TTS's per-block 6-way chunk, its own `norm_out`'s 2-way chunk, and CosyVoice3's
    /// tensor-for-tensor-identical DiT, all share this exact op; was hand-duplicated three times
    /// -- twice as a private method, once inlined -- until extracted here).</summary>
    public static unsafe void ApplyAffineModulationSlice(float[] dst, float[] src, float[] modulation, int scaleOffset, int shiftOffset, int t, int dim)
    {
        int vecSize = System.Numerics.Vector<float>.Count;
        fixed (float* dp = dst, sp = src, mp = modulation)
        {
            float* dpLocal = dp;
            float* spLocal = sp;
            float* scpLocal = mp + scaleOffset;
            float* shpLocal = mp + shiftOffset;
            Parallel.For(0, t, ti =>
            {
                int off = ti * dim;
                float* dRow = dpLocal + off;
                float* sRow = spLocal + off;
                int d = 0;
                for (; d <= dim - vecSize; d += vecSize)
                {
                    var vs = new System.Numerics.Vector<float>(new ReadOnlySpan<float>(sRow + d, vecSize));
                    var vScale = new System.Numerics.Vector<float>(new ReadOnlySpan<float>(scpLocal + d, vecSize));
                    var vShift = new System.Numerics.Vector<float>(new ReadOnlySpan<float>(shpLocal + d, vecSize));
                    var vr = vs * (System.Numerics.Vector<float>.One + vScale) + vShift;
                    vr.CopyTo(new Span<float>(dRow + d, vecSize));
                }
                for (; d < dim; d++) dRow[d] = sRow[d] * (1f + scpLocal[d]) + shpLocal[d];
            });
        }
    }

    /// <summary>AdaLN-Zero gated residual add, `dst = residual + modulation[gateOffset..] * update`
    /// (see <see cref="ApplyAffineModulationSlice"/>'s doc comment -- same sharing rationale).</summary>
    public static unsafe void ApplyGatedResidualSlice(float[] dst, float[] residual, float[] modulation, int gateOffset, float[] update, int t, int dim)
    {
        int vecSize = System.Numerics.Vector<float>.Count;
        fixed (float* dp = dst, rp = residual, mp = modulation, up = update)
        {
            float* dpLocal = dp;
            float* rpLocal = rp;
            float* gpLocal = mp + gateOffset;
            float* upLocal = up;
            Parallel.For(0, t, ti =>
            {
                int off = ti * dim;
                float* dRow = dpLocal + off;
                float* rRow = rpLocal + off;
                float* uRow = upLocal + off;
                int d = 0;
                for (; d <= dim - vecSize; d += vecSize)
                {
                    var vr = new System.Numerics.Vector<float>(new ReadOnlySpan<float>(rRow + d, vecSize));
                    var vg = new System.Numerics.Vector<float>(new ReadOnlySpan<float>(gpLocal + d, vecSize));
                    var vu = new System.Numerics.Vector<float>(new ReadOnlySpan<float>(uRow + d, vecSize));
                    var vRes = vr + vg * vu;
                    vRes.CopyTo(new Span<float>(dRow + d, vecSize));
                }
                for (; d < dim; d++) dRow[d] = rRow[d] + gpLocal[d] * uRow[d];
            });
        }
    }

    public static float SiLU(float x) => x / (1f + MathF.Exp(-x));

    public static float Mish(float x) => x * MathF.Tanh(MathF.Log(1f + MathF.Exp(x)));

    /// <summary>Exact (erf-based) GELU -- nn.GELU()'s default approximate='none'. Used by ConvNeXtV2Block.</summary>
    public static float GeluExact(float x) => 0.5f * x * (1f + Erf(x / 1.4142135f));

    /// <summary>tanh-approximated GELU -- nn.GELU(approximate='tanh'). Used by the DiT block's FeedForward.</summary>
    public static float GeluTanh(float x) =>
        0.5f * x * (1f + MathF.Tanh(0.7978845608f * (x + 0.044715f * x * x * x)));

    // Abramowitz-Stegun erf approximation (max error ~1.5e-7), sufficient for float32 GELU.
    private static float Erf(float x)
    {
        float sign = x < 0 ? -1f : 1f;
        x = MathF.Abs(x);
        const float a1 = 0.254829592f, a2 = -0.284496736f, a3 = 1.421413741f, a4 = -1.453152027f, a5 = 1.061405429f, p = 0.3275911f;
        float tt = 1f / (1f + p * x);
        float y = 1f - (((((a5 * tt + a4) * tt) + a3) * tt + a2) * tt + a1) * tt * MathF.Exp(-x * x);
        return sign * y;
    }

    /// <summary>Standard (non-grouped) "same"-padded Conv1d, sequence-major [T,inCh]-&gt;[T,outCh]. weight is [outCh,inCh,kernel].</summary>
    public static float[] Conv1dSamePad(float[] x, int t, int inCh, float[] weight, float[] bias, int outCh, int kernel)
    {
        int pad = kernel / 2;
        var y = new float[t * outCh];
        Parallel.For(0, outCh, oc =>
        {
            int wBase = oc * inCh * kernel;
            for (int ti = 0; ti < t; ti++)
            {
                float sum = bias[oc];
                for (int ic = 0; ic < inCh; ic++)
                {
                    int wcBase = wBase + ic * kernel;
                    for (int k = 0; k < kernel; k++)
                    {
                        int src = ti - pad + k;
                        if ((uint)src < (uint)t) sum += weight[wcBase + k] * x[src * inCh + ic];
                    }
                }
                y[ti * outCh + oc] = sum;
            }
        });
        return y;
    }

    /// <summary>Depthwise (groups=dim) "same"-padded Conv1d, sequence-major [T,dim] in/out. weight is [dim,1,kernel].</summary>
    public static float[] DepthwiseConv1dSamePad(float[] x, int t, int dim, float[] weight, float[] bias, int kernel)
    {
        int pad = kernel / 2;
        var y = new float[t * dim];
        Parallel.For(0, dim, d =>
        {
            int wBase = d * kernel;
            for (int ti = 0; ti < t; ti++)
            {
                float sum = bias[d];
                for (int k = 0; k < kernel; k++)
                {
                    int src = ti - pad + k;
                    if ((uint)src < (uint)t) sum += weight[wBase + k] * x[src * dim + d];
                }
                y[ti * dim + d] = sum;
            }
        });
        return y;
    }

    /// <summary>Grouped "same"-padded Conv1d, sequence-major [T,dim] in/out. weight is [dim, dim/groups, kernel].</summary>
    public static float[] GroupedConv1dSamePad(float[] x, int t, int dim, float[] weight, float[] bias, int kernel, int groups)
    {
        int pad = kernel / 2;
        int inPerGroup = dim / groups;
        int outPerGroup = dim / groups; // groups divide both in/out channels equally here
        var y = new float[t * dim];
        Parallel.For(0, dim, oc =>
        {
            int group = oc / outPerGroup;
            int inBase = group * inPerGroup;
            int wBase = oc * inPerGroup * kernel;
            for (int ti = 0; ti < t; ti++)
            {
                float sum = bias[oc];
                for (int ic = 0; ic < inPerGroup; ic++)
                {
                    int wcBase = wBase + ic * kernel;
                    int srcCh = inBase + ic;
                    for (int k = 0; k < kernel; k++)
                    {
                        int src = ti - pad + k;
                        if ((uint)src < (uint)t) sum += weight[wcBase + k] * x[src * dim + srcCh];
                    }
                }
                y[ti * dim + oc] = sum;
            }
        });
        return y;
    }

    /// <summary>
    /// GRN (Global Response Normalization): Gx[c] = L2 norm of x[:,c] ACROSS THE SEQUENCE dim
    /// (not per-timestep -- confirmed from modules.py's `torch.norm(x, p=2, dim=1)` where dim=1
    /// is the sequence axis at that point in ConvNeXtV2Block.forward, since x is still `b n d`).
    /// Nx[c] = Gx[c] / (mean_c(Gx) + eps). out[t,c] = gamma[c]*(x[t,c]*Nx[c]) + beta[c] + x[t,c].
    /// </summary>
    public static float[] Grn(float[] x, int t, int dim, float[] gamma, float[] beta, float eps = 1e-6f)
    {
        var gx = new float[dim];
        for (int c = 0; c < dim; c++)
        {
            double sumSq = 0;
            for (int ti = 0; ti < t; ti++) { float v = x[ti * dim + c]; sumSq += (double)v * v; }
            gx[c] = (float)Math.Sqrt(sumSq);
        }
        double meanGx = 0;
        for (int c = 0; c < dim; c++) meanGx += gx[c];
        meanGx /= dim;
        float invMean = (float)(1.0 / (meanGx + eps));

        var y = new float[t * dim];
        for (int ti = 0; ti < t; ti++)
        {
            int off = ti * dim;
            for (int c = 0; c < dim; c++)
            {
                float nx = gx[c] * invMean;
                y[off + c] = gamma[c] * (x[off + c] * nx) + beta[c] + x[off + c];
            }
        }
        return y;
    }

    public static void SoftmaxInPlace(float[] scores, int offset, int length) =>
        SoftmaxInPlace(scores.AsSpan(offset, length));

    public static void SoftmaxInPlace(Span<float> scores)
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

    /// <summary>x_transformers-convention RoPE, applied in place to a [t, heads*headDim] tensor. Shared by F5-TTS's DiT and CosyVoice3's DiT (tensor-for-tensor identical architecture, see CosyVoice3DiTModel's doc comment) -- was hand-duplicated in both files until extracted here.
    /// <paramref name="numRopeHeads"/> is the real `AttnProcessor.pe_attn_head` config (`modules.py`): when not null/less than `heads`, RoPE is applied ONLY to the first `numRopeHeads` heads (real `query[:, :pn, :, :] = apply_rotary_pos_emb(...)`), leaving the rest unrotated -- NOT a uniform apply-to-all-heads default. Confirmed real per-checkpoint: `F5TTS_Base`'s own `F5TTS_Base.yaml` sets `pe_attn_head: 1` (only head 0 gets RoPE); defaults to `heads` (apply to every head) for checkpoints that don't set this (e.g. `F5TTS_v1_Base`'s `pe_attn_head: null`, and CosyVoice3's own real config).</summary>
    public static void ApplyRotary(float[] x, int t, int heads, int headDim, float[] rotaryCos, float[] rotarySin, int? numRopeHeads = null)
    {
        int dim = heads * headDim;
        int halfHead = headDim / 2;
        int ropeHeads = numRopeHeads ?? heads;
        for (int ti = 0; ti < t; ti++)
        {
            int angleBase = ti * halfHead;
            for (int h = 0; h < ropeHeads; h++)
            {
                int hOff = ti * dim + h * headDim;
                for (int k = 0; k < halfHead; k++)
                {
                    float cos = rotaryCos[angleBase + k];
                    float sin = rotarySin[angleBase + k];
                    float x0 = x[hOff + 2 * k];
                    float x1 = x[hOff + 2 * k + 1];
                    x[hOff + 2 * k] = x0 * cos - x1 * sin;
                    x[hOff + 2 * k + 1] = x1 * cos + x0 * sin;
                }
            }
        }
    }

    /// <summary>
    /// Full non-causal multi-head self-attention given already-projected, already-rotary-applied
    /// q/k/v [t, heads*headDim]. Returns context [t, heads*headDim] (pre output-projection).
    /// Shared by F5-TTS's DiT and CosyVoice3's DiT -- was hand-duplicated in both files (including
    /// a scalar, non-SIMD QK/AV inner loop parallelized only over `heads`) until extracted here.
    /// </summary>
    public static unsafe float[] MultiHeadSelfAttention(float[] q, float[] k, float[] v, int t, int heads, int headDim)
    {
        int dim = heads * headDim;
        float scale = 1f / MathF.Sqrt(headDim);
        var context = new float[t * dim];

        fixed (float* qp = q, kp = k, vp = v, ctxp = context)
        {
            float* qBase = qp;
            float* kBase = kp;
            float* vBase = vp;
            float* ctxBase = ctxp;

            Parallel.For(0, heads * t, idx =>
            {
                int h = idx / t;
                int i = idx - h * t;
                int hOff = h * headDim;
                float* qRow = qBase + i * dim + hOff;

                var scores = new float[t];
                for (int j = 0; j < t; j++)
                {
                    scores[j] = OpenTail.Stingray.Cpu.SimdKernels.DotF32(qRow, kBase + j * dim + hOff, headDim) * scale;
                }

                SoftmaxInPlace(scores.AsSpan());

                float* cRow = ctxBase + i * dim + hOff;
                for (int d = 0; d < headDim; d++) cRow[d] = 0f;

                for (int j = 0; j < t; j++)
                {
                    float p = scores[j];
                    if (p == 0f) continue;
                    float* vRow = vBase + j * dim + hOff;
                    for (int d = 0; d < headDim; d++)
                        cRow[d] += p * vRow[d];
                }
            });
        }

        return context;
    }
}
