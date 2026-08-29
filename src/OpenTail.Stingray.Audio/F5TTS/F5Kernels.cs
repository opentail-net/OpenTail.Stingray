
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
                    // Single-row fast path (modulation linear layers): run sequentially without threadpool dispatch overhead
                    for (int o = 0; o < outDim; o++)
                    {
                        float b = bpLocal is null ? 0f : bpLocal[o];
                        ypLocal[o] = b + OpenTail.Stingray.Cpu.SimdKernels.DotF32(wpLocal + o * inDim, xpLocal, inDim);
                    }
                }
                else
                {
                    // Multi-row path: parallelize across timesteps ti; xRow stays hot in L1 cache across all output channels
                    Parallel.For(0, t, ti =>
                    {
                        float* xRow = xpLocal + ti * inDim;
                        float* yRow = ypLocal + ti * outDim;
                        for (int o = 0; o < outDim; o++)
                        {
                            float b = bpLocal is null ? 0f : bpLocal[o];
                            yRow[o] = b + OpenTail.Stingray.Cpu.SimdKernels.DotF32(wpLocal + o * inDim, xRow, inDim);
                        }
                    });
                }
            }
        }
        return y;
    }

    /// <summary>Per-timestep affine LayerNorm over the last (channel) dim.</summary>
    public static float[] LayerNorm(float[] x, int t, int dim, float[] gamma, float[] beta, float eps = 1e-6f)
    {
        var y = new float[t * dim];
        void ComputeRow(int ti)
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
        }

        if (t <= 4)
        {
            for (int ti = 0; ti < t; ti++) ComputeRow(ti);
        }
        else
        {
            Parallel.For(0, t, ComputeRow);
        }
        return y;
    }

    /// <summary>Per-timestep non-affine LayerNorm (elementwise_affine=False), for AdaLN modulation.</summary>
    public static float[] LayerNormNoAffine(float[] x, int t, int dim, float eps = 1e-6f)
    {
        var y = new float[t * dim];
        void ComputeRow(int ti)
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
                y[off + d] = (float)((x[off + d] - mean) * invStd);
        }

        if (t <= 4)
        {
            for (int ti = 0; ti < t; ti++) ComputeRow(ti);
        }
        else
        {
            Parallel.For(0, t, ComputeRow);
        }
        return y;
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

    public static void SoftmaxInPlace(float[] scores, int offset, int length)
    {
        float max = float.NegativeInfinity;
        for (int i = 0; i < length; i++) if (scores[offset + i] > max) max = scores[offset + i];
        float sum = 0f;
        for (int i = 0; i < length; i++)
        {
            float e = MathF.Exp(scores[offset + i] - max);
            scores[offset + i] = e;
            sum += e;
        }
        float invSum = 1f / sum;
        for (int i = 0; i < length; i++) scores[offset + i] *= invSum;
    }

    /// <summary>x_transformers-convention RoPE, applied in place to a [t, heads*headDim] tensor. Shared by F5-TTS's DiT and CosyVoice3's DiT (tensor-for-tensor identical architecture, see CosyVoice3DiTModel's doc comment) -- was hand-duplicated in both files until extracted here.</summary>
    public static void ApplyRotary(float[] x, int t, int heads, int headDim, float[] rotaryCos, float[] rotarySin)
    {
        int dim = heads * headDim;
        int halfHead = headDim / 2;
        for (int ti = 0; ti < t; ti++)
        {
            int angleBase = ti * halfHead;
            for (int h = 0; h < heads; h++)
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
    /// Parallelizes over `t` instead of `heads`: heads is a small, fixed constant (e.g. 16) while
    /// t (frame count) scales with utterance length and is usually far larger, so this gives much
    /// more parallel width; the QK dot and the AV accumulation both use SIMD (DotF32/
    /// TensorPrimitives.MultiplyAdd) instead of scalar loops, following the pattern already
    /// verified in Primitives/CfmUNetKernels.cs.
    /// </summary>
    public static float[] MultiHeadSelfAttention(float[] q, float[] k, float[] v, int t, int heads, int headDim)
    {
        int dim = heads * headDim;
        float scale = 1f / MathF.Sqrt(headDim);
        var context = new float[t * dim];

        unsafe
        {
            fixed (float* qp = q, kp = k, vp = v, ctxP = context)
            {
                float* qpLocal = qp;
                float* kpLocal = kp;
                float* vpLocal = vp;
                float* ctxPLocal = ctxP;

                for (int h = 0; h < heads; h++)
                {
                    int hOff = h * headDim;
                    Parallel.For(0, t, i =>
                    {
                        var scores = new float[t];
                        float* qRow = qpLocal + i * dim + hOff;
                        for (int j = 0; j < t; j++)
                        {
                            scores[j] = OpenTail.Stingray.Cpu.SimdKernels.DotF32(qRow, kpLocal + j * dim + hOff, headDim) * scale;
                        }
                        SoftmaxInPlace(scores, 0, t);

                        float* cRow = ctxPLocal + i * dim + hOff;
                        for (int d = 0; d < headDim; d++) cRow[d] = 0f;

                        for (int j = 0; j < t; j++)
                        {
                            float p = scores[j];
                            if (p == 0f) continue;
                            float* vRow = vpLocal + j * dim + hOff;
                            for (int d = 0; d < headDim; d++)
                            {
                                cRow[d] += p * vRow[d];
                            }
                        }
                    });
                }
            }
        }

        return context;
    }
}
