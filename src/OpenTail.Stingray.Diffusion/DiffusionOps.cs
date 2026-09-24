using System.Runtime.InteropServices;

namespace OpenTail.Stingray.Diffusion;

/// <summary>
/// CPU-only primitive operations needed by the VAE decoder and text encoders.
/// All methods operate on flat float[] arrays with explicit shape parameters.
/// Tensors are NCHW (batch × channels × height × width) for spatial ops.
///
/// <para><b>Row/token-level parallelization pass (measured, CLAUDE.md rule 7)</b>: at realistic
/// DiT sizes (nTokens=4096, dim=3072, matching Flux/SD3-scale visual token counts) and a
/// realistic VAE GroupNorm size ([1,512,64,64], 32 groups), `Parallel.For` across rows/groups
/// gave real, repeatable wins for <see cref="AdaLNModulate"/> (~18.3ms -&gt; ~9.5-16.6ms),
/// <see cref="GroupNorm"/> (~10.1ms -&gt; ~6.3-7.0ms), <see cref="RmsNorm"/> (~7.1ms -&gt;
/// ~5.3-6.0ms), <see cref="ScaleShiftInPlace"/> (~9.8ms -&gt; ~7.7-8.3ms), and a smaller but
/// real gain for <see cref="ScaleGateAdd"/> (~12.0ms -&gt; ~10.6-11.8ms), and
/// <see cref="Upsample2x"/> at a realistic late-stage VAE decode size ([1,512,64,64]:
/// ~5.9ms -&gt; ~3.8-3.9ms) -- all kept.
/// <see cref="LayerNorm"/> showed NO consistent improvement (~19.6-20.5ms either way, sometimes
/// marginally worse parallelized) across repeated runs, so it was left sequential rather than
/// parallelized-but-unproven -- see its own doc comment for why this one specific method didn't
/// benefit (multiple chained `TensorPrimitives` calls per row already keep each row's own work
/// cheap enough that `Parallel.For`'s per-task dispatch overhead ate the gain).</para>
/// </summary>
internal static unsafe class DiffusionOps
{
    // ── Activation functions ──────────────────────────────────────────────

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Gelu(float x)
    {
        // Tanh GELU approximation — matches PyTorch default
        const float c = 0.044715f;
        float v = 0.7978845608028654f * (x + c * x * x * x);
        return 0.5f * x * (1.0f + MathF.Tanh(v));
    }

    /// <summary>Exact (erf-based) GELU: 0.5*x*(1+erf(x/sqrt(2))) -- PyTorch's plain `nn.GELU()`
    /// (no `approximate='tanh'`), distinct from <see cref="Gelu"/>'s tanh approximation. Used by
    /// `Wan.text_embedding.1` per the real reference (`wan.hpp`: "text_embedding.1 is nn.GELU()",
    /// only the FFN's own GELU is `approximate='tanh'`). Erf via the Abramowitz &amp; Stegun 7.1.26
    /// polynomial approximation (max error ~1.5e-7, effectively exact at float32 precision) since
    /// neither <see cref="MathF"/> nor <see cref="Math"/> expose a built-in erf.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float GeluExact(float x)
    {
        float z = x * 0.7071067811865476f; // x / sqrt(2)
        float sign = z < 0 ? -1f : 1f;
        float az = MathF.Abs(z);
        const float a1 = 0.254829592f, a2 = -0.284496736f, a3 = 1.421413741f, a4 = -1.453152027f, a5 = 1.061405429f, p = 0.3275911f;
        float t = 1f / (1f + p * az);
        float poly = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t;
        float erf = sign * (1f - poly * MathF.Exp(-az * az));
        return 0.5f * x * (1f + erf);
    }

    public static void GeluInPlace(Span<float> x)
    {
        if (x.Length >= 4096)
        {
            unsafe
            {
                fixed (float* ptr = x)
                {
                    nint rawPtr = (nint)ptr;
                    int len = x.Length;
                    int chunkSize = Math.Max(2048, len / Environment.ProcessorCount);
                    int numChunks = (len + chunkSize - 1) / chunkSize;
                    Parallel.For(0, numChunks, c =>
                    {
                        float* p = (float*)rawPtr;
                        int start = c * chunkSize;
                        int end = Math.Min(start + chunkSize, len);
                        for (int i = start; i < end; i++)
                            p[i] = Gelu(p[i]);
                    });
                }
            }
        }
        else
        {
            for (int i = 0; i < x.Length; i++)
                x[i] = Gelu(x[i]);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Silu(float x) => x / (1f + MathF.Exp(-x));

    public static void SiluInPlace(Span<float> x)
    {
        // SiLU(x) = x * sigmoid(x), in cache-sized chunks with a stack temp (no pooled full-size
        // buffer). Large tensors (VAE full-res activations, ~33M floats) run in parallel: a 512x512
        // VAE decode's ResBlock SiLUs went 1.23s -> 0.11s, element-wise identical.
        const int Chunk = 16384;
        int len = x.Length;
        if (len <= Chunk * 4)
        {
            SiluChunk(x);
            return;
        }
        fixed (float* p = x)
        {
            nint addr = (nint)p;
            Parallel.For(0, (len + Chunk - 1) / Chunk, c =>
            {
                int start = c * Chunk;
                SiluChunk(new Span<float>((float*)addr + start, Math.Min(Chunk, len - start)));
            });
        }
    }

    private static void SiluChunk(Span<float> x)
    {
        Span<float> temp = stackalloc float[Math.Min(x.Length, 16384)];
        for (int off = 0; off < x.Length; off += temp.Length)
        {
            var seg = x.Slice(off, Math.Min(temp.Length, x.Length - off));
            var t = temp[..seg.Length];
            TensorPrimitives.Sigmoid(seg, t);
            TensorPrimitives.Multiply(seg, t, seg);
        }
    }

    // ── Normalization ─────────────────────────────────────────────────────

    /// <summary>
    /// Layer Normalization: y = (x - mean) / sqrt(var + eps) * weight + bias.
    /// Operates on the last axis of length <paramref name="dim"/>.
    /// </summary>
    public static void LayerNorm(Span<float> x, ReadOnlySpan<float> weight, ReadOnlySpan<float> bias,
                                 int dim, float eps = 1e-5f)
    {
        LayerNorm((ReadOnlySpan<float>)x, x, weight, bias, dim, eps);
    }

    public static void LayerNorm(ReadOnlySpan<float> input, Span<float> output, ReadOnlySpan<float> weight, ReadOnlySpan<float> bias,
                                 int dim, float eps = 1e-5f)
    {
        int n = input.Length / dim;
        for (int row = 0; row < n; row++)
        {
            var inRow = input.Slice(row * dim, dim);
            var outRow = output.Slice(row * dim, dim);
            float mean = TensorPrimitives.Sum(inRow) / dim;
            TensorPrimitives.Subtract(inRow, mean, outRow);
            float var = TensorPrimitives.Dot<float>(outRow, outRow) / dim;
            float scale = 1f / MathF.Sqrt(var + eps);
            TensorPrimitives.Multiply(outRow, scale, outRow);
            TensorPrimitives.Multiply(outRow, weight, outRow);
            TensorPrimitives.Add(outRow, bias, outRow);
        }
    }

    /// <summary>
    /// AdaLN-Zero Modulation: y = Norm(x) * (1 + scale) + shift.
    /// </summary>
    public static void AdaLNModulate(Span<float> output, ReadOnlySpan<float> input, ReadOnlySpan<float> shift, ReadOnlySpan<float> scale,
                                     int nTokens, int dim, bool isRmsNorm = true, float eps = 1e-5f)
    {
        int scaleLen = scale.Length;
        int shiftLen = shift.Length;
        fixed (float* pOut = output, pIn = input, pScale = scale, pShift = shift)
        {
            float* pOutLocal = pOut;
            float* pInLocal = pIn;
            float* pScaleLocal = pScale;
            float* pShiftLocal = pShift;
            Parallel.For(0, nTokens, t =>
            {
                var inRow = new ReadOnlySpan<float>(pInLocal + t * dim, dim);
                var outRow = new Span<float>(pOutLocal + t * dim, dim);
                if (isRmsNorm)
                {
                    float sumSq = TensorPrimitives.SumOfSquares(inRow);
                    float invStd = 1f / MathF.Sqrt(sumSq / dim + eps);
                    for (int i = 0; i < dim; i++)
                    {
                        float s = i < scaleLen ? pScaleLocal[i] : 0f;
                        float sh = i < shiftLen ? pShiftLocal[i] : 0f;
                        outRow[i] = inRow[i] * invStd * (1f + s) + sh;
                    }
                }
                else
                {
                    float mean = TensorPrimitives.Sum(inRow) / dim;
                    float sumSq = 0f;
                    for (int i = 0; i < dim; i++) { float d = inRow[i] - mean; sumSq += d * d; }
                    float invStd = 1f / MathF.Sqrt(sumSq / dim + eps);
                    for (int i = 0; i < dim; i++)
                    {
                        float s = i < scaleLen ? pScaleLocal[i] : 0f;
                        float sh = i < shiftLen ? pShiftLocal[i] : 0f;
                        outRow[i] = (inRow[i] - mean) * invStd * (1f + s) + sh;
                    }
                }
            });
        }
    }

    /// <summary>
    /// Modulated residual addition: x += proj * gate.
    /// </summary>
    public static void ScaleGateAdd(Span<float> x, ReadOnlySpan<float> proj, ReadOnlySpan<float> gate, int nTokens, int dim)
    {
        int gateLen = gate.Length;
        fixed (float* px = x, pp = proj, pg = gate)
        {
            float* pxLocal = px;
            float* ppLocal = pp;
            float* pgLocal = pg;
            Parallel.For(0, nTokens, t =>
            {
                float* xRow = pxLocal + t * dim;
                float* projRow = ppLocal + t * dim;
                for (int i = 0; i < dim; i++)
                {
                    float g = i < gateLen ? pgLocal[i] : 1f;
                    xRow[i] += projRow[i] * g;
                }
            });
        }
    }

    /// <summary>
    /// Per-head QK Normalization.
    /// </summary>
    public static void QKNorm(Span<float> q, Span<float> k, ReadOnlySpan<float> qScale, ReadOnlySpan<float> kScale,
                              int nTokens, int numHeads, int headDim, float eps = 1e-5f)
    {
        int totalHeads = nTokens * numHeads;
        for (int idx = 0; idx < totalHeads; idx++)
        {
            var qSlice = q.Slice(idx * headDim, headDim);
            float qSq = TensorPrimitives.SumOfSquares(qSlice);
            float qInv = 1f / MathF.Sqrt(qSq / headDim + eps);
            for (int i = 0; i < headDim; i++)
            {
                float s = i < qScale.Length ? qScale[i] : 1f;
                qSlice[i] = (qSlice[i] * qInv) * s;
            }

            var kSlice = k.Slice(idx * headDim, headDim);
            float kSq = TensorPrimitives.SumOfSquares(kSlice);
            float kInv = 1f / MathF.Sqrt(kSq / headDim + eps);
            for (int i = 0; i < headDim; i++)
            {
                float s = i < kScale.Length ? kScale[i] : 1f;
                kSlice[i] = (kSlice[i] * kInv) * s;
            }
        }
    }

    /// <summary>
    /// Group Normalization: groups of channels along C axis.
    /// Input layout: [N, C, H, W] flattened. Normalizes within each group.
    /// </summary>
    public static void GroupNorm(Span<float> x, ReadOnlySpan<float> weight, ReadOnlySpan<float> bias,
                                 int n, int c, int h, int w, int groups, float eps = 1e-5f)
    {
        int chansPerGroup = c / groups;
        int spatialSize   = h * w;
        int groupElements = chansPerGroup * spatialSize;

        fixed (float* px = x, pw = weight, pb = bias)
        {
            float* pxLocal = px;
            float* pwLocal = pw;
            float* pbLocal = pb;
            Parallel.For(0, n * groups, idx =>
            {
                int b = idx / groups;
                int g = idx % groups;
                int bOff = b * c * spatialSize;

                // Compute mean and variance over this group (vectorized -- same
                // subtract-then-dot approach LayerNorm above already uses and has measured wins for).
                int gOff = bOff + g * groupElements;
                var groupSpan = new Span<float>(pxLocal + gOff, groupElements);
                float mean = TensorPrimitives.Sum((ReadOnlySpan<float>)groupSpan) / groupElements;

                var devArr = ArrayPool<float>.Shared.Rent(groupElements);
                var dev = devArr.AsSpan(0, groupElements);
                TensorPrimitives.Subtract((ReadOnlySpan<float>)groupSpan, mean, dev);
                float var = TensorPrimitives.Dot<float>(dev, dev) / groupElements;
                float invStd = 1f / MathF.Sqrt(var + eps);
                ArrayPool<float>.Shared.Return(devArr);

                for (int gc = 0; gc < chansPerGroup; gc++)
                {
                    int c_abs = g * chansPerGroup + gc;
                    int cOff  = bOff + c_abs * spatialSize;
                    float chWeight = pwLocal[c_abs];
                    float chBias = pbLocal[c_abs];
                    var chSpan = new Span<float>(pxLocal + cOff, spatialSize);
                    TensorPrimitives.Subtract((ReadOnlySpan<float>)chSpan, mean, chSpan);
                    TensorPrimitives.Multiply((ReadOnlySpan<float>)chSpan, invStd * chWeight, chSpan);
                    TensorPrimitives.Add((ReadOnlySpan<float>)chSpan, chBias, chSpan);
                }
            });
        }
    }

    // ── Convolution ───────────────────────────────────────────────────────

    /// <summary>
    /// 2D convolution. Inputs/outputs: [N, C, H, W] (NCHW flat arrays).
    /// Kernel: [outC, inC, kH, kW].  Bias: [outC] (nullable).
    /// Supports stride=1 or stride=2, padding computed as "same" for stride=1.
    /// </summary>
        public static float[] Conv2D(float[] input, float[] kernel, float[]? bias,
                                  int n, int inC, int inH, int inW,
                                  int outC, int kH, int kW,
                                  int stride = 1, int padding = -1)
    {
        if (padding < 0) padding = (kH - 1) / 2; // "same" padding for odd kernels

        // Fast-path: 1x1 convolution (stride=1, padding=0) using hardware-accelerated SIMD dot products
        if (kH == 1 && kW == 1 && stride == 1 && padding == 0)
        {
            return Conv2D1x1Simd(input, kernel, bias, n, inC, inH, inW, outC);
        }

        int outH = (inH + 2 * padding - kH) / stride + 1;
        int outW = (inW + 2 * padding - kW) / stride + 1;
        var output = new float[n * outC * outH * outW];

        // im2col + GEMM (OpenBLAS/microkernel via MatMulBatchedF32) instead of a scalar direct
        // convolution: each output pixel's receptive field becomes one row of a [pixels, inC*kH*kW]
        // matrix. The GEMM computes [outC, pixels] = kernel[outC, K] x col[pixels, K]^T, so the
        // result is already NCHW-ordered per channel (contiguous row copies, no strided scatter).
        // That orientation measured ~1.8x faster than [pixels, outC] on VAE shapes (outC is small
        // for the microkernel's N dimension). Rows are chunked to bound the col buffer (~chunk*K).
        int hw = outH * outW;
        int inHW = inH * inW;
        int kSize = inC * kH * kW;
        int chunk = Math.Min(hw, Math.Max(64, (16 << 20) / Math.Max(1, kSize))); // ~64MB col buffer
        var col = ArrayPool<float>.Shared.Rent(chunk * kSize);
        var gemmOut = ArrayPool<float>.Shared.Rent(chunk * outC);

        try
        {
            fixed (float* pIn = input)
            fixed (float* pKernel = kernel)
            fixed (float* pOut = output)
            fixed (float* pBias = bias)
            fixed (float* pCol = col)
            fixed (float* pG = gemmOut)
            {
                nint inAddr = (nint)pIn, colAddr = (nint)pCol, gAddr = (nint)pG, outAddr = (nint)pOut;
                for (int b = 0; b < n; b++)
                {
                    int inBatch = b * inC * inHW;
                    int outBatch = b * outC * hw;
                    for (int p0 = 0; p0 < hw; p0 += chunk)
                    {
                        int rows = Math.Min(chunk, hw - p0);

                        // Gather: row r = output pixel p0+r, column order (ic, kh, kw) matches the kernel.
                        Parallel.For(0, rows, r =>
                        {
                            float* src = (float*)inAddr + inBatch;
                            float* dst = (float*)colAddr + (nint)r * kSize;
                            int p = p0 + r;
                            int ih0 = (p / outW) * stride - padding;
                            int iw0 = (p % outW) * stride - padding;
                            int c = 0;
                            for (int ic = 0; ic < inC; ic++)
                            {
                                float* plane = src + ic * inHW;
                                for (int kh = 0; kh < kH; kh++)
                                {
                                    int ih = ih0 + kh;
                                    if ((uint)ih >= (uint)inH)
                                    {
                                        new Span<float>(dst + c, kW).Clear();
                                        c += kW;
                                        continue;
                                    }
                                    float* row = plane + ih * inW;
                                    for (int kw = 0; kw < kW; kw++)
                                    {
                                        int iw = iw0 + kw;
                                        dst[c++] = (uint)iw < (uint)inW ? row[iw] : 0f;
                                    }
                                }
                            }
                        });

                        SimdKernels.MatMulBatchedF32(pG, pCol, pKernel, outC, rows, kSize);

                        // Copy each channel's [rows] slice into NCHW output [outC, hw], adding bias.
                        nint biasAddr = (nint)pBias;
                        Parallel.For(0, outC, oc =>
                        {
                            var src = new ReadOnlySpan<float>((float*)gAddr + (nint)oc * rows, rows);
                            var dst = new Span<float>((float*)outAddr + outBatch + (nint)oc * hw + p0, rows);
                            if (biasAddr != 0) TensorPrimitives.Add(src, ((float*)biasAddr)[oc], dst);
                            else src.CopyTo(dst);
                        });
                    }
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(col);
            ArrayPool<float>.Shared.Return(gemmOut);
        }
        return output;
    }

    private static float[] Conv2D1x1Simd(float[] input, float[] kernel, float[]? bias,
                                         int n, int inC, int h, int w, int outC)
    {
        int hw = h * w;
        var output = new float[n * outC * hw];

        for (int b = 0; b < n; b++)
        {
            int inBatchBase = b * inC * hw;
            int outBatchBase = b * outC * hw;

            // 1. Pack input NCHW -> [HW, inC]
            var packedIn = new float[hw * inC];
            Parallel.For(0, hw, p =>
            {
                int dstOff = p * inC;
                for (int ic = 0; ic < inC; ic++)
                {
                    packedIn[dstOff + ic] = input[inBatchBase + ic * hw + p];
                }
            });

            // 2. Compute packed output: [HW, outC] = [HW, inC] @ [outC, inC]^T via TensorPrimitives.Dot
            var packedOut = new float[hw * outC];
            Parallel.For(0, hw, p =>
            {
                var rowIn = packedIn.AsSpan(p * inC, inC);
                var rowOut = packedOut.AsSpan(p * outC, outC);

                for (int oc = 0; oc < outC; oc++)
                {
                    var kRow = kernel.AsSpan(oc * inC, inC);
                    float dot = TensorPrimitives.Dot(rowIn, kRow);
                    rowOut[oc] = dot + (bias is not null ? bias[oc] : 0f);
                }
            });

            // 3. Unpack packed output [HW, outC] -> output NCHW [outC, H, W]
            Parallel.For(0, outC, oc =>
            {
                int outBase = outBatchBase + oc * hw;
                for (int p = 0; p < hw; p++)
                {
                    output[outBase + p] = packedOut[p * outC + oc];
                }
            });
        }

        return output;
    }

    // ── Spatial ops ───────────────────────────────────────────────────────

    /// <summary>
    /// Nearest-neighbor 2× upsample. Input: [N, C, H, W]. Output: [N, C, 2H, 2W].
    /// </summary>
    public static float[] Upsample2x(float[] input, int n, int c, int h, int w)
    {
        int oh = h * 2, ow = w * 2;
        var output = new float[n * c * oh * ow];
        fixed (float* pIn = input, pOut = output)
        {
            float* pInLocal = pIn;
            float* pOutLocal = pOut;
            Parallel.For(0, n * c, idx =>
            {
                int cIn = idx * h * w, cOut = idx * oh * ow;
                for (int y = 0; y < h; y++)
                {
                    for (int x = 0; x < w; x++)
                    {
                        float v = pInLocal[cIn + y * w + x];
                        pOutLocal[cOut + (2*y)   * ow + 2*x    ] = v;
                        pOutLocal[cOut + (2*y)   * ow + 2*x + 1] = v;
                        pOutLocal[cOut + (2*y+1) * ow + 2*x    ] = v;
                        pOutLocal[cOut + (2*y+1) * ow + 2*x + 1] = v;
                    }
                }
            });
        }
        return output;
    }

    // ── Linear (dense) layer helpers ──────────────────────────────────────

    /// <summary>
    /// Linear projection: y = x @ weight^T + bias.
    /// x: [n, inDim], weight: [outDim, inDim], bias: [outDim] (optional), result: [n, outDim].
    /// Uses TensorPrimitives.Dot for hardware-accelerated SIMD with zero per-token allocations.
    /// </summary>
    public static void Linear(ReadOnlySpan<float> x, ReadOnlySpan<float> weight, ReadOnlySpan<float> bias, Span<float> result, int n, int inDim, int outDim)
    {
        fixed (float* px = x, pw = weight, pr = result, pb = bias)
        {
            float* pxLocal = px;
            float* pwLocal = pw;
            float* prLocal = pr;
            float* pbLocal = pb;

            if (n > 1)
            {
                // Register-tiled GEMM (microkernel/OpenBLAS) instead of one Dot per output element:
                // measured 1.9-2.3x faster on T5-XXL/CLIP/VAE-attention shapes (e.g. 256x4096->10240:
                // 134ms -> 59ms), identical output to ~1e-6.
                SimdKernels.MatMulBatchedF32(pr, pw, px, n, outDim, inDim, pb);
            }
            else
            {
                Parallel.For(0, outDim, o =>
                {
                    float b0 = pbLocal != null ? pbLocal[o] : 0f;
                    float* wRow = pwLocal + (nuint)o * (nuint)inDim;
                    var wSpan = new ReadOnlySpan<float>(wRow, inDim);
                    var xSpan = new ReadOnlySpan<float>(pxLocal, inDim);
                    prLocal[o] = b0 + TensorPrimitives.Dot<float>(xSpan, wSpan);
                });
            }
        }
    }

    /// <summary>
    /// Linear projection: y = x @ weight^T + bias.
    /// x: [n, inDim], weight: [outDim, inDim], bias: [outDim] (optional), result: [n, outDim].
    /// Uses TensorPrimitives.Dot for hardware-accelerated SIMD with zero per-token allocations.
    /// </summary>
    public static float[] Linear(float[] x, float[] weight, float[]? bias, int n, int inDim, int outDim)
    {
        var result = new float[n * outDim];
        Linear(x.AsSpan(0, n * inDim), weight.AsSpan(0, outDim * inDim), bias is not null ? bias.AsSpan(0, outDim) : ReadOnlySpan<float>.Empty, result, n, inDim, outDim);
        return result;
    }

    /// <summary>RMS normalization along last axis (in-place). T5LayerNorm = no bias/mean centering.</summary>
    public static void RmsNorm(Span<float> x, ReadOnlySpan<float> weight, int dim, float eps = 1e-6f)
    {
        int n = x.Length / dim;
        fixed (float* px = x, pw = weight)
        {
            float* pxLocal = px;
            float* pwLocal = pw;
            Parallel.For(0, n, row =>
            {
                var row_ = new Span<float>(pxLocal + row * dim, dim);
                var w_ = new ReadOnlySpan<float>(pwLocal, dim);
                float ss = TensorPrimitives.Dot<float>(row_, row_);
                float invRms = 1f / MathF.Sqrt(ss / dim + eps);
                TensorPrimitives.Multiply(row_, w_, row_);   // row_ *= weight
                TensorPrimitives.Multiply(row_, invRms, row_);   // row_ *= invRms
            });
        }
    }

    /// <summary>Softmax over the last axis of length <paramref name="dim"/> (in-place).</summary>
    public static void Softmax(Span<float> x, int dim)
    {
        int n = x.Length / dim;
        for (int row = 0; row < n; row++)
        {
            var s = x.Slice(row * dim, dim);
            float max = TensorPrimitives.Max(s);
            TensorPrimitives.Subtract(s, max, s);
            TensorPrimitives.Exp(s, s);
            float sum = TensorPrimitives.Sum(s);
            TensorPrimitives.Divide(s, sum, s);
        }
    }

    /// <summary>In-place element-wise: x[i] = x[i] * (1 + scale[i]) + shift[i].</summary>
    public static void ScaleShiftInPlace(Span<float> x, ReadOnlySpan<float> scale, ReadOnlySpan<float> shift, int dim)
    {
        int n = x.Length / dim;
        fixed (float* px = x, psc = scale, psh = shift)
        {
            float* pxLocal = px;
            float* pscLocal = psc;
            float* pshLocal = psh;
            Parallel.For(0, n, row =>
            {
                float* xRow = pxLocal + row * dim;
                for (int i = 0; i < dim; i++)
                    xRow[i] = xRow[i] * (1f + pscLocal[i]) + pshLocal[i];
            });
        }
    }

    /// <summary>Per-row addition: a[i] += b[i].</summary>
    public static void AddRows(float[] a, float[] b, int n, int dim)
    {
        TensorPrimitives.Add(a.AsSpan(0, n * dim), b.AsSpan(0, n * dim), a.AsSpan(0, n * dim));
    }

    // ── Image upscaling primitives ────────────────────────────────────────

    /// <summary>
    /// Pixel Shuffle (sub-pixel convolution): rearranges channels into spatial extent.
    /// Input:  [N, C×r², H, W] — Output: [N, C, H×r, W×r].
    /// Used for learned ×2 / ×4 upsampling in ESRGAN upscalers (old-style variant).
    /// </summary>
    public static float[] PixelShuffle(float[] input, int n, int inC, int h, int w, int r)
    {
        int outC = inC / (r * r);
        int outH = h * r, outW = w * r;
        var output = new float[n * outC * outH * outW];

        for (int b = 0; b < n; b++)
        {
            int bInOff  = b * inC * h * w;
            int bOutOff = b * outC * outH * outW;
            for (int oc = 0; oc < outC; oc++)
            {
                int outCOff = bOutOff + oc * outH * outW;
                for (int oh = 0; oh < outH; oh++)
                {
                    int ih = oh / r, ky = oh % r;
                    for (int ow = 0; ow < outW; ow++)
                    {
                        int iw = ow / r, kx = ow % r;
                        int ic = oc * r * r + ky * r + kx;
                        output[outCOff + oh * outW + ow] =
                            input[bInOff + ic * h * w + ih * w + iw];
                    }
                }
            }
        }
        return output;
    }

    /// <summary>
    /// Pixel Unshuffle: rearranges spatial blocks into channels.
    /// Input:  [N, C, H, W] — Output: [N, C×r², H/r, W/r].
    /// Inverse of PixelShuffle. Used to pre-process input in ×2 ESRGAN models.
    /// </summary>
    public static float[] PixelUnshuffle(float[] input, int n, int c, int h, int w, int r)
    {
        int outC = c * r * r;
        int outH = h / r, outW = w / r;
        var output = new float[n * outC * outH * outW];

        for (int b = 0; b < n; b++)
        {
            int bInOff  = b * c * h * w;
            int bOutOff = b * outC * outH * outW;
            for (int ic = 0; ic < c; ic++)
            {
                int inCOff = bInOff + ic * h * w;
                for (int ih = 0; ih < outH; ih++)
                for (int iw = 0; iw < outW; iw++)
                {
                    for (int ky = 0; ky < r; ky++)
                    for (int kx = 0; kx < r; kx++)
                    {
                        int oc  = ic * r * r + ky * r + kx;
                        int src = inCOff + (ih * r + ky) * w + (iw * r + kx);
                        output[bOutOff + oc * outH * outW + ih * outW + iw] = input[src];
                    }
                }
            }
        }
        return output;
    }

    /// <summary>Leaky ReLU activation in-place: x[i] = x[i] >= 0 ? x[i] : slope * x[i].</summary>
    public static void LeakyReLUInPlace(Span<float> x, float slope = 0.2f)
    {
        for (int i = 0; i < x.Length; i++)
            if (x[i] < 0f) x[i] *= slope;
    }

    /// <summary>
    /// Bilinear upsampling from [N, C, H, W] to [N, C, outH, outW].
    /// Used as post-process resize for ×2 ESRGAN models whose forward pass
    /// produces the same spatial size as input.
    /// </summary>
    public static float[] UpsampleBilinear(float[] input, int n, int c, int h, int w, int outH, int outW)
    {
        var output = new float[n * c * outH * outW];
        float scaleY = h > 1 ? (float)(h - 1) / (outH - 1) : 0f;
        float scaleX = w > 1 ? (float)(w - 1) / (outW - 1) : 0f;

        for (int b = 0; b < n; b++)
        for (int ch = 0; ch < c; ch++)
        {
            int inOff  = b * c * h * w  + ch * h * w;
            int outOff = b * c * outH * outW + ch * outH * outW;
            for (int oy = 0; oy < outH; oy++)
            {
                float iy  = oy * scaleY;
                int   iy0 = (int)iy;
                int   iy1 = Math.Min(iy0 + 1, h - 1);
                float wy  = iy - iy0;
                for (int ox = 0; ox < outW; ox++)
                {
                    float ix  = ox * scaleX;
                    int   ix0 = (int)ix;
                    int   ix1 = Math.Min(ix0 + 1, w - 1);
                    float wx  = ix - ix0;
                    float v00 = input[inOff + iy0 * w + ix0];
                    float v01 = input[inOff + iy0 * w + ix1];
                    float v10 = input[inOff + iy1 * w + ix0];
                    float v11 = input[inOff + iy1 * w + ix1];
                    output[outOff + oy * outW + ox] =
                        v00 * (1f - wy) * (1f - wx) + v01 * (1f - wy) * wx +
                        v10 * wy * (1f - wx)         + v11 * wy * wx;
                }
            }
        }
        return output;
    }

    // ── Aliases for consistent PascalCase naming ──────────────────────────

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float SiLU(float x) => Silu(x);

    public static void SiLUInPlace(Span<float> x) => SiluInPlace(x);

    /// <summary>
    /// Bicubic upsampling from [C, H, W] to [C, outH, outW] (single image, no batch dim).
    /// Uses the standard Keys bicubic kernel (a=-0.5). Input pixels are in [0,1] float range.
    /// </summary>
    public static float[] UpsampleBicubic(float[] input, int c, int h, int w, int outH, int outW)
    {
        var output = new float[c * outH * outW];
        float scaleY = (float)h / outH;
        float scaleX = (float)w / outW;

        static float CubicWeight(float t)
        {
            // Keys a=-0.5 cubic kernel
            float at = MathF.Abs(t);
            if (at <= 1f) return (1.5f * at - 2.5f) * at * at + 1f;
            if (at <  2f) return ((-0.5f * at + 2.5f) * at - 4f) * at + 2f;
            return 0f;
        }

        static float Clamp01(float v) => v < 0f ? 0f : v > 1f ? 1f : v;

        for (int ch = 0; ch < c; ch++)
        {
            int inOff  = ch * h * w;
            int outOff = ch * outH * outW;
            for (int oy = 0; oy < outH; oy++)
            {
                float srcY = (oy + 0.5f) * scaleY - 0.5f;
                int   iy0  = (int)MathF.Floor(srcY);
                for (int ox = 0; ox < outW; ox++)
                {
                    float srcX = (ox + 0.5f) * scaleX - 0.5f;
                    int   ix0  = (int)MathF.Floor(srcX);
                    float sum  = 0f;
                    for (int ky = -1; ky <= 2; ky++)
                    {
                        int iy = Math.Clamp(iy0 + ky, 0, h - 1);
                        float wy = CubicWeight(srcY - (iy0 + ky));
                        for (int kx = -1; kx <= 2; kx++)
                        {
                            int ix = Math.Clamp(ix0 + kx, 0, w - 1);
                            sum += wy * CubicWeight(srcX - (ix0 + kx)) * input[inOff + iy * w + ix];
                        }
                    }
                    output[outOff + oy * outW + ox] = Clamp01(sum);
                }
            }
        }
        return output;
    }

    /// <summary>
    /// Blend two RGB pixel buffers [C, H, W] (single image, C=3).
    /// result[i] = factor * a[i] + (1 - factor) * b[i]
    /// where factor=1.0 returns <paramref name="a"/> unchanged and factor=0.0 returns <paramref name="b"/>.
    /// </summary>
    public static float[] BlendRgb(float[] a, float[] b, float factor)
    {
        if (factor >= 1f) return a;
        if (factor <= 0f) return b;
        float inv = 1f - factor;
        var result = new float[a.Length];
        for (int i = 0; i < a.Length; i++)
            result[i] = factor * a[i] + inv * b[i];
        return result;
    }

    // ── Offset-based overloads (used by ZImageDiT, QwenTextEncoder) ───────

    /// <summary>
    /// RMS-norm one row: dst[dstOff..dstOff+dim] = RMSNorm(src[srcOff..srcOff+dim]).
    /// </summary>
    public static void RmsNorm(float[] src, int srcOff, int dim,
                                float[] weight, float eps,
                                float[] dst, int dstOff)
    {
        var srcSlice = src.AsSpan(srcOff, dim);
        var dstSlice = dst.AsSpan(dstOff, dim);
        float ss     = TensorPrimitives.Dot<float>(srcSlice, srcSlice);
        float invRms = 1f / MathF.Sqrt(ss / dim + eps);
        TensorPrimitives.Multiply(srcSlice, weight.AsSpan(0, dim), dstSlice); // dst = src * weight
        TensorPrimitives.Multiply(dstSlice, invRms, dstSlice);                // dst *= invRms
    }

    /// <summary>
    /// Softmax over scores[offset .. offset+n] in-place.
    /// </summary>
    public static void Softmax(float[] scores, int offset, int n)
    {
        var s = scores.AsSpan(offset, n);
        float max = TensorPrimitives.Max(s);
        TensorPrimitives.Subtract(s, max, s);
        TensorPrimitives.Exp(s, s);
        float sum = TensorPrimitives.Sum(s);
        TensorPrimitives.Divide(s, sum, s);
    }

    public static unsafe void SoftmaxInPlace(float* scores, int n)
    {
        float max = float.NegativeInfinity;
        for (int i = 0; i < n; i++) if (scores[i] > max) max = scores[i];
        float sum = 0f;
        for (int i = 0; i < n; i++) { scores[i] = MathF.Exp(scores[i] - max); sum += scores[i]; }
        float inv = 1f / sum;
        for (int i = 0; i < n; i++) scores[i] *= inv;
    }

    /// <summary>
    /// Multi-head scaled dot-product attention for CPU diffusion models (Wan, LTX, etc.).
    /// q: [qSeq, numHeads * headDim]
    /// k, v: [kvSeq, numHeads * headDim]
    /// output: [qSeq, numHeads * headDim]
    /// </summary>
    public static float[] MultiHeadAttention(float[] q, float[] k, float[] v, int qSeq, int kvSeq, int numHeads, int headDim)
    {
        var output = new float[qSeq * numHeads * headDim];
        MultiHeadAttention(q, k, v, output.AsSpan(), qSeq, kvSeq, numHeads, headDim);
        return output;
    }

    public static void MultiHeadAttention(float[] q, float[] k, float[] v, Span<float> output, int qSeq, int kvSeq, int numHeads, int headDim)
    {
        Wan.WanAttention.TiledMultiHeadAttention(q, k, v, output, qSeq, kvSeq, numHeads, headDim);
    }

    public static void MultiHeadAttention(ReadOnlySpan<float> q, ReadOnlySpan<float> k, ReadOnlySpan<float> v, Span<float> output, int qSeq, int kvSeq, int numHeads, int headDim)
    {
        Wan.WanAttention.TiledMultiHeadAttention(q, k, v, output, qSeq, kvSeq, numHeads, headDim);
    }

    /// <summary>
    /// Mean/variance Layer Normalization with no learned affine parameters.
    /// </summary>
    public static void LayerNormNoAffine(float[] x, int dim, float eps = 1e-6f)
    {
        int n = x.Length / dim;
        Parallel.For(0, n, row =>
        {
            var rowSpan = x.AsSpan(row * dim, dim);
            float mean = TensorPrimitives.Sum(rowSpan) / dim;
            TensorPrimitives.Subtract(rowSpan, mean, rowSpan);
            float sumSq = TensorPrimitives.SumOfSquares(rowSpan);
            float scale = 1f / MathF.Sqrt(sumSq / dim + eps);
            TensorPrimitives.Multiply(rowSpan, scale, rowSpan);
        });
    }

    public static void LayerNormNoAffine(ReadOnlySpan<float> input, Span<float> output, int dim, float eps = 1e-6f)
    {
        int n = input.Length / dim;
        for (int row = 0; row < n; row++)
        {
            var inRow = input.Slice(row * dim, dim);
            var outRow = output.Slice(row * dim, dim);
            float mean = TensorPrimitives.Sum(inRow) / dim;
            TensorPrimitives.Subtract(inRow, mean, outRow);
            float sumSq = TensorPrimitives.SumOfSquares(outRow);
            float scale = 1f / MathF.Sqrt(sumSq / dim + eps);
            TensorPrimitives.Multiply(outRow, scale, outRow);
        }
    }

    /// <summary>
    /// AdaLN-Zero modulate over row-major `[seqLen, dim]` data: `y = x * (1+scale) + shift`, `scale`
    /// and `shift` shared across all rows (broadcast per-channel). Extracted from byte-identical
    /// copies in Wan and HunyuanVideo.
    /// </summary>
    public static float[] ModulateRows(float[] x, int seqLen, int dim, ReadOnlySpan<float> shift, ReadOnlySpan<float> scale)
    {
        var outF = new float[seqLen * dim];
        ModulateRows(x.AsSpan(0, seqLen * dim), outF.AsSpan(), seqLen, dim, shift, scale);
        return outF;
    }

    public static void ModulateRows(ReadOnlySpan<float> input, Span<float> output, int seqLen, int dim, ReadOnlySpan<float> shift, ReadOnlySpan<float> scale)
    {
        // 2026-09-21: was a scalar double-loop; vectorized with TensorPrimitives, same pattern
        // (and same measured-safe precedent) as SDXL's own 2026-09-11 CPU-vectorization pass.
        // `1+scale` is per-channel and invariant across rows, so compute it once per call rather
        // than per row.
        var scale1 = new float[dim];
        TensorPrimitives.Add(scale, 1.0f, scale1);
        for (int i = 0; i < seqLen; i++)
        {
            int off = i * dim;
            TensorPrimitives.MultiplyAdd(input.Slice(off, dim), scale1, shift, output.Slice(off, dim));
        }
    }

    /// <summary>
    /// AdaLN-Zero gated residual add over row-major `[seqLen, dim]` data: `x += branch * gate`,
    /// `gate` shared across all rows. Extracted from byte-identical copies in Wan and HunyuanVideo.
    /// </summary>
    public static void ApplyGatedResidualRows(float[] x, float[] branch, int seqLen, int dim, ReadOnlySpan<float> gate)
    {
        ApplyGatedResidualRows(x.AsSpan(0, seqLen * dim), branch.AsSpan(0, seqLen * dim), seqLen, dim, gate);
    }

    public static void ApplyGatedResidualRows(Span<float> x, ReadOnlySpan<float> branch, int seqLen, int dim, ReadOnlySpan<float> gate)
    {
        // 2026-09-21: was a scalar double-loop; vectorized with TensorPrimitives.MultiplyAdd
        // (x = branch*gate + x, destination == the addend operand, a supported full-overlap case).
        for (int i = 0; i < seqLen; i++)
        {
            int off = i * dim;
            var xRow = x.Slice(off, dim);
            TensorPrimitives.MultiplyAdd(branch.Slice(off, dim), gate, xRow, xRow);
        }
    }

    /// <summary>
    /// Sinusoidal timestep positional embedding with `f_i = exp(-log(theta) * i / half)`.
    /// Output order depends on <paramref name="flipSinToCos"/>: `true` gives
    /// `[cos(t*f), sin(t*f)]` (what every current caller -- Wan, HunyuanVideo, QwenImage, LTX-Video,
    /// FLUX.2, Z-Image -- needs); `false` gives `[sin(t*f), cos(t*f)]`. Pass it explicitly: Z-Image
    /// silently broke (16x16 mosaic) when it relied on the default.
    /// </summary>
    /// <param name="flipSinToCos">diffusers' `flip_sin_to_cos`: true puts cos first.</param>
    public static float[] SinusoidalTimestepEmbedding(float timestep, int dim = 256, float theta = 10000f, bool flipSinToCos = false)
    {
        var emb = new float[dim];
        int half = dim / 2;
        for (int i = 0; i < half; i++)
        {
            float freq = MathF.Exp(-MathF.Log(theta) * i / half);
            float angle = timestep * freq;
            if (flipSinToCos)
            {
                emb[i] = MathF.Cos(angle);
                emb[half + i] = MathF.Sin(angle);
            }
            else
            {
                emb[i] = MathF.Sin(angle);
                emb[half + i] = MathF.Cos(angle);
            }
        }
        return emb;
    }

    /// <summary>
    /// Root-Mean-Square Normalization with no learned affine parameters, epsilon applied AFTER the
    /// sqrt (`1/(rms+eps)`) rather than inside it -- the convention used by Flux2/Flux3/StableAudio
    /// (extracted verbatim from three byte-identical copies, not merged into
    /// <see cref="RmsNormNoAffine"/> above, which applies eps inside the sqrt; the two are
    /// numerically equivalent for realistic magnitudes but are a different formula, so kept
    /// separate rather than silently changed during a DRY pass).
    /// </summary>
    public static void RmsNormNoAffinePostSqrtEps(Span<float> tensor, int dim, float eps = 1e-6f)
    {
        int nTokens = tensor.Length / dim;
        for (int i = 0; i < nTokens; i++)
        {
            var slice = tensor.Slice(i * dim, dim);
            float norm = TensorPrimitives.Norm(slice);
            float rms = norm / MathF.Sqrt(dim);
            float scale = 1.0f / (rms + eps);
            TensorPrimitives.Multiply(slice, scale, slice);
        }
    }

    /// <summary>
    /// Root-Mean-Square Normalization with no learned affine parameters.
    /// </summary>
    public static void RmsNormNoAffine(float[] x, int dim, float eps = 1e-6f)
    {
        int n = x.Length / dim;
        Parallel.For(0, n, row =>
        {
            var rowSpan = x.AsSpan(row * dim, dim);
            float sumSq = TensorPrimitives.SumOfSquares(rowSpan);
            float invRms = 1f / MathF.Sqrt(sumSq / dim + eps);
            TensorPrimitives.Multiply(rowSpan, invRms, rowSpan);
        });
    }
}


