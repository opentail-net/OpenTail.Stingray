using System.Runtime.InteropServices;

namespace OpenTail.Stingray.Diffusion;

/// <summary>
/// CPU-only primitive operations needed by the VAE decoder and text encoders.
/// All methods operate on flat float[] arrays with explicit shape parameters.
/// Tensors are NCHW (batch × channels × height × width) for spatial ops.
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

    public static void GeluInPlace(Span<float> x)
    {
        for (int i = 0; i < x.Length; i++)
            x[i] = Gelu(x[i]);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Silu(float x) => x / (1f + MathF.Exp(-x));

    public static void SiluInPlace(Span<float> x)
    {
        // SiLU(x) = x * sigmoid(x). Pool a temp buffer to avoid heap pressure.
        var tempArr = ArrayPool<float>.Shared.Rent(x.Length);
        var temp    = tempArr.AsSpan(0, x.Length);
        TensorPrimitives.Sigmoid(x, temp);   // temp = sigmoid(x)
        TensorPrimitives.Multiply(x, temp, x);  // x   *= temp  → SiLU
        ArrayPool<float>.Shared.Return(tempArr);
    }

    // ── Normalization ─────────────────────────────────────────────────────

    /// <summary>
    /// Layer Normalization: y = (x - mean) / sqrt(var + eps) * weight + bias.
    /// Operates on the last axis of length <paramref name="dim"/>.
    /// </summary>
    public static void LayerNorm(Span<float> x, ReadOnlySpan<float> weight, ReadOnlySpan<float> bias,
                                 int dim, float eps = 1e-5f)
    {
        int n = x.Length / dim;
        for (int row = 0; row < n; row++)
        {
            var row_ = x.Slice(row * dim, dim);
            float mean = TensorPrimitives.Sum(row_) / dim;
            // Shift so we can compute variance as dot(shifted, shifted)
            TensorPrimitives.Subtract(row_, mean, row_);
            float var  = TensorPrimitives.Dot<float>(row_, row_) / dim;
            float scale = 1f / MathF.Sqrt(var + eps);
            TensorPrimitives.Multiply(row_, scale, row_);
            TensorPrimitives.Multiply(row_, weight, row_);
            TensorPrimitives.Add(row_, bias, row_);
        }
    }

    /// <summary>
    /// AdaLN-Zero Modulation: y = Norm(x) * (1 + scale) + shift.
    /// </summary>
    public static void AdaLNModulate(Span<float> output, ReadOnlySpan<float> input, ReadOnlySpan<float> shift, ReadOnlySpan<float> scale,
                                     int nTokens, int dim, bool isRmsNorm = true, float eps = 1e-5f)
    {
        for (int t = 0; t < nTokens; t++)
        {
            var inRow = input.Slice(t * dim, dim);
            var outRow = output.Slice(t * dim, dim);
            if (isRmsNorm)
            {
                float sumSq = TensorPrimitives.SumOfSquares(inRow);
                float invStd = 1f / MathF.Sqrt(sumSq / dim + eps);
                for (int i = 0; i < dim; i++)
                {
                    float s = i < scale.Length ? scale[i] : 0f;
                    float sh = i < shift.Length ? shift[i] : 0f;
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
                    float s = i < scale.Length ? scale[i] : 0f;
                    float sh = i < shift.Length ? shift[i] : 0f;
                    outRow[i] = (inRow[i] - mean) * invStd * (1f + s) + sh;
                }
            }
        }
    }

    /// <summary>
    /// Modulated residual addition: x += proj * gate.
    /// </summary>
    public static void ScaleGateAdd(Span<float> x, ReadOnlySpan<float> proj, ReadOnlySpan<float> gate, int nTokens, int dim)
    {
        for (int t = 0; t < nTokens; t++)
        {
            var xRow = x.Slice(t * dim, dim);
            var projRow = proj.Slice(t * dim, dim);
            for (int i = 0; i < dim; i++)
            {
                float g = i < gate.Length ? gate[i] : 1f;
                xRow[i] += projRow[i] * g;
            }
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

        for (int b = 0; b < n; b++)
        {
            int bOff = b * c * spatialSize;
            for (int g = 0; g < groups; g++)
            {
                // Compute mean and variance over this group
                int gOff = bOff + g * groupElements;
                float mean = 0f;
                for (int i = 0; i < groupElements; i++) mean += x[gOff + i];
                mean /= groupElements;

                float var = 0f;
                for (int i = 0; i < groupElements; i++)
                { float d = x[gOff + i] - mean; var += d * d; }
                float invStd = 1f / MathF.Sqrt(var / groupElements + eps);

                for (int gc = 0; gc < chansPerGroup; gc++)
                {
                    int c_abs = g * chansPerGroup + gc;
                    int cOff  = bOff + c_abs * spatialSize;
                    for (int s = 0; s < spatialSize; s++)
                    {
                        float v = (x[cOff + s] - mean) * invStd;
                        x[cOff + s] = v * weight[c_abs] + bias[c_abs];
                    }
                }
            }
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

        int hw = outH * outW;
        int inHW = inH * inW;

        fixed (float* pIn = input)
        fixed (float* pKernel = kernel)
        fixed (float* pOut = output)
        fixed (float* pBias = bias)
        {
            var inPtr = pIn;
            var kPtr = pKernel;
            var outPtr = pOut;
            var bPtr = pBias;

            Parallel.For(0, n * outC, bo =>
            {
                int b  = bo / outC;
                int oc = bo % outC;

                float biasVal = bPtr != null ? bPtr[oc] : 0f;
                int outBase = b * outC * hw + oc * hw;
                int kBase0  = oc * inC * kH * kW;
                int inBatch = b * inC * inHW;

                for (int oh = 0; oh < outH; oh++)
                {
                    int ih0 = oh * stride - padding;
                    for (int ow2 = 0; ow2 < outW; ow2++)
                    {
                        int iw0 = ow2 * stride - padding;
                        float sum = biasVal;
                        for (int ic = 0; ic < inC; ic++)
                        {
                            int kBase = kBase0 + ic * kH * kW;
                            int inBase = inBatch + ic * inHW;
                            for (int kh = 0; kh < kH; kh++)
                            {
                                int ih = ih0 + kh;
                                if ((uint)ih >= (uint)inH) continue;
                                int inRow = inBase + ih * inW;
                                int kRow  = kBase + kh * kW;
                                for (int kw = 0; kw < kW; kw++)
                                {
                                    int iw = iw0 + kw;
                                    if ((uint)iw >= (uint)inW) continue;
                                    sum += inPtr[inRow + iw] * kPtr[kRow + kw];
                                }
                            }
                        }
                        outPtr[outBase + oh * outW + ow2] = sum;
                    }
                }
            });
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
        for (int b = 0; b < n; b++)
        {
            int bIn = b * c * h * w, bOut = b * c * oh * ow;
            for (int ch = 0; ch < c; ch++)
            {
                int cIn = bIn + ch * h * w, cOut = bOut + ch * oh * ow;
                for (int y = 0; y < h; y++)
                {
                    for (int x = 0; x < w; x++)
                    {
                        float v = input[cIn + y * w + x];
                        output[cOut + (2*y)   * ow + 2*x    ] = v;
                        output[cOut + (2*y)   * ow + 2*x + 1] = v;
                        output[cOut + (2*y+1) * ow + 2*x    ] = v;
                        output[cOut + (2*y+1) * ow + 2*x + 1] = v;
                    }
                }
            }
        }
        return output;
    }

    // ── Linear (dense) layer helpers ──────────────────────────────────────

    /// <summary>
    /// Dense layer for float32 weights (VAE decoder): out[i] = sum_j(weight[i,j] * x[j]) + bias[i].
    /// weight: [outDim, inDim], result: [n, outDim].
    /// Uses TensorPrimitives.Dot for hardware-accelerated SIMD — no GCHandle, no unsafe.
    /// For large quantized GGUF weights use SimdKernels.MatMulBatched via IWeightLoader.TryGetRaw.
    /// </summary>
    public static float[] Linear(float[] x, float[] weight, float[]? bias, int n, int inDim, int outDim)
    {
        var result = new float[n * outDim];

        Parallel.For(0, outDim, o =>
        {
            float b0   = bias is not null ? bias[o] : 0f;
            var   wRow = weight.AsSpan(o * inDim, inDim);
            for (int b = 0; b < n; b++)
            {
                var xRow = x.AsSpan(b * inDim, inDim);
                result[b * outDim + o] = b0 + TensorPrimitives.Dot<float>(xRow, wRow);
            }
        });

        return result;
    }

    /// <summary>RMS normalization along last axis (in-place). T5LayerNorm = no bias/mean centering.</summary>
    public static void RmsNorm(Span<float> x, ReadOnlySpan<float> weight, int dim, float eps = 1e-6f)
    {
        int n = x.Length / dim;
        for (int row = 0; row < n; row++)
        {
            var row_ = x.Slice(row * dim, dim);
            float ss     = TensorPrimitives.Dot<float>(row_, row_);
            float invRms = 1f / MathF.Sqrt(ss / dim + eps);
            TensorPrimitives.Multiply(row_, weight, row_);   // row_ *= weight
            TensorPrimitives.Multiply(row_, invRms, row_);   // row_ *= invRms
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
        for (int row = 0; row < n; row++)
        {
            int off = row * dim;
            for (int i = 0; i < dim; i++)
                x[off + i] = x[off + i] * (1f + scale[i]) + shift[i];
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
}


