using System;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Vision;

/// <summary>
/// High-performance shared neural compute kernels, tensor operations, and pointer helpers for Multimodal Vision Transformers.
/// </summary>
public static unsafe class VisionOps
{
    /// <summary>
    /// Loads a tensor from GGUF and dequantizes it to contiguous Float32 memory.
    /// Handles F32, F16, BF16, Q8_0, Q4_K, Q4_0, etc. seamlessly.
    /// </summary>
    public static float[]? LoadTensorF32(GgufModel gguf, params string[] candidateNames)
    {
        foreach (var name in candidateNames)
        {
            var tensorOpt = gguf.FindTensor(name);
            if (tensorOpt.HasValue)
            {
                var tensor = tensorOpt.Value;
                long elementCount = tensor.ElementCount;
                if (elementCount == 0) return null;

                var data = gguf.GetTensorData(tensor);
                var result = new float[elementCount];
                Dequantize.ToFloat32(data, result.AsSpan(), tensor.DType, elementCount);
                return result;
            }
        }
        return null;
    }

    /// <summary>
    /// Vectorized, multi-threaded Matrix-Vector multiplication using AVX2/AVX-512 TensorPrimitives.Dot with optional FP32 bias.
    /// Computes output[tokens, outDim] = input[tokens, inDim] * weights[outDim, inDim]^T + bias[outDim].
    /// </summary>
    public static void MatVec(
        float[] input,
        float[]? weights,
        float* bias,
        int nTokens,
        int inDim,
        int outDim,
        float[] output)
    {
        if (weights == null) return;

        Parallel.For(0, nTokens, t =>
        {
            int inOff = t * inDim;
            int outOff = t * outDim;
            var inSpan = new ReadOnlySpan<float>(input, inOff, inDim);

            for (int o = 0; o < outDim; o++)
            {
                float sum = bias != null ? bias[o] : 0f;
                int rowOff = o * inDim;
                sum += TensorPrimitives.Dot(inSpan, new ReadOnlySpan<float>(weights, rowOff, inDim));
                output[outOff + o] = sum;
            }
        });
    }

    /// <summary>
    /// Parallelized Matrix-Vector multiplication for FP16 weights with optional FP32 bias addition.
    /// Computes output[tokens, outDim] = input[tokens, inDim] * weights[outDim, inDim]^T + bias[outDim].
    /// </summary>
    public static void MatVecF16(
        float[] input,
        Half* weights,
        float* bias,
        int nTokens,
        int inDim,
        int outDim,
        float[] output)
    {
        if (weights == null) return;

        Parallel.For(0, nTokens, t =>
        {
            int inOff = t * inDim;
            int outOff = t * outDim;

            for (int o = 0; o < outDim; o++)
            {
                float sum = bias != null ? bias[o] : 0f;
                int rowOff = o * inDim;

                for (int i = 0; i < inDim; i++)
                {
                    sum += input[inOff + i] * (float)weights[rowOff + i];
                }
                output[outOff + o] = sum;
            }
        });
    }

    /// <summary>
    /// Parallelized Layer Normalization over the last dimension: (x - mean) / sqrt(var + eps) * weight + bias.
    /// </summary>
    public static void LayerNorm(
        float[] states,
        int nTokens,
        int dim,
        float* weights,
        float* bias,
        float eps = 1e-6f)
    {
        Parallel.For(0, nTokens, t =>
        {
            int off = t * dim;
            float mean = 0f;
            for (int d = 0; d < dim; d++) mean += states[off + d];
            mean /= dim;

            float var = 0f;
            for (int d = 0; d < dim; d++)
            {
                float diff = states[off + d] - mean;
                var += diff * diff;
            }
            float std = MathF.Sqrt(var / dim + eps);

            for (int d = 0; d < dim; d++)
            {
                float w = weights != null ? weights[d] : 1f;
                float b = bias != null ? bias[d] : 0f;
                states[off + d] = ((states[off + d] - mean) / std) * w + b;
            }
        });
    }

    /// <summary>
    /// Parallelized Root-Mean-Square Normalization (RMSNorm): x / sqrt(mean(x^2) + eps) * weight.
    /// </summary>
    public static void RmsNorm(
        float[] states,
        int nTokens,
        int dim,
        float* weights,
        float eps = 1e-6f)
    {
        Parallel.For(0, nTokens, t =>
        {
            int off = t * dim;
            float sumSq = 0f;
            for (int d = 0; d < dim; d++)
            {
                float val = states[off + d];
                sumSq += val * val;
            }
            float rms = MathF.Sqrt(sumSq / dim + eps);

            for (int d = 0; d < dim; d++)
            {
                float w = weights != null ? weights[d] : 1f;
                states[off + d] = (states[off + d] / rms) * w;
            }
        });
    }

    /// <summary>
    /// Parallel multi-head scaled dot-product self-attention mechanism with softmax.
    /// Q, K, V have shape [nTokens, heads * headDim]. Output has shape [nTokens, heads * headDim].
    /// </summary>
    public static void Attention(
        float[] q,
        float[] k,
        float[] v,
        int nTokens,
        int heads,
        int headDim,
        float[] output)
    {
        float scale = 1.0f / MathF.Sqrt(headDim);
        int embd = heads * headDim;

        Parallel.For(0, heads, h =>
        {
            int headOff = h * headDim;
            var scores = new float[nTokens];

            for (int i = 0; i < nTokens; i++)
            {
                int qOff = i * embd + headOff;

                // Compute Q_i . K_j * scale
                float maxScore = float.NegativeInfinity;
                for (int j = 0; j < nTokens; j++)
                {
                    int kOff = j * embd + headOff;
                    float dot = 0f;
                    for (int d = 0; d < headDim; d++)
                    {
                        dot += q[qOff + d] * k[kOff + d];
                    }
                    float s = dot * scale;
                    scores[j] = s;
                    if (s > maxScore) maxScore = s;
                }

                // Numerically stable Softmax
                float expSum = 0f;
                for (int j = 0; j < nTokens; j++)
                {
                    float exp = MathF.Exp(scores[j] - maxScore);
                    scores[j] = exp;
                    expSum += exp;
                }
                float invSum = expSum > 0f ? 1.0f / expSum : 0f;

                // Aggregate V_j * prob
                int outOff = i * embd + headOff;
                for (int d = 0; d < headDim; d++)
                {
                    float acc = 0f;
                    for (int j = 0; j < nTokens; j++)
                    {
                        int vOff = j * embd + headOff;
                        acc += v[vOff + d] * scores[j];
                    }
                    output[outOff + d] = acc * invSum;
                }
            }
        });
    }

    /// <summary>
    /// Parallel multi-head grouped-query attention (GQA) mechanism with softmax and optional sink bias.
    /// Q has shape [nTokens, qHeads * headDim]. K, V have shape [nTokens, kvHeads * headDim].
    /// Output has shape [nTokens, qHeads * headDim].
    /// </summary>
    public static void AttentionGqa(
        float[] q,
        float[] k,
        float[] v,
        int nTokens,
        int qHeads,
        int kvHeads,
        int headDim,
        float[] output,
        float* attnSinks = null)
    {
        float scale = 1.0f / MathF.Sqrt(headDim);
        int groupSize = qHeads / kvHeads;
        int qEmbd = qHeads * headDim;
        int kvEmbd = kvHeads * headDim;

        Parallel.For(0, qHeads, qh =>
        {
            int kvHead = qh / groupSize;
            int qOffHead = qh * headDim;
            int kvOffHead = kvHead * headDim;
            float sink = (attnSinks != null) ? attnSinks[qh] : 0f;

            var scores = new float[nTokens];

            for (int i = 0; i < nTokens; i++)
            {
                int qOff = i * qEmbd + qOffHead;

                float maxScore = sink != 0f ? sink : float.NegativeInfinity;
                for (int j = 0; j < nTokens; j++)
                {
                    int kOff = j * kvEmbd + kvOffHead;
                    float dot = 0f;
                    for (int d = 0; d < headDim; d++)
                    {
                        dot += q[qOff + d] * k[kOff + d];
                    }
                    float s = dot * scale;
                    scores[j] = s;
                    if (s > maxScore) maxScore = s;
                }

                float expSum = sink != 0f ? MathF.Exp(sink - maxScore) : 0f;
                for (int j = 0; j < nTokens; j++)
                {
                    float exp = MathF.Exp(scores[j] - maxScore);
                    scores[j] = exp;
                    expSum += exp;
                }
                float invSum = expSum > 0f ? 1.0f / expSum : 0f;

                int outOff = i * qEmbd + qOffHead;
                for (int d = 0; d < headDim; d++)
                {
                    float acc = 0f;
                    for (int j = 0; j < nTokens; j++)
                    {
                        int vOff = j * kvEmbd + kvOffHead;
                        acc += v[vOff + d] * scores[j];
                    }
                    output[outOff + d] = acc * invSum;
                }
            }
        });
    }

    /// <summary>
    /// Applies Gaussian Error Linear Unit (GELU) with Tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float GeluScalar(float x)
    {
        return 0.5f * x * (1.0f + MathF.Tanh(MathF.Sqrt(2.0f / MathF.PI) * (x + 0.044715f * x * x * x)));
    }

    /// <summary>
    /// Applies GELU elementwise across a span in-place.
    /// </summary>
    public static void Gelu(Span<float> data)
    {
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = GeluScalar(data[i]);
        }
    }

    /// <summary>
    /// Applies QuickGELU activation function: x * sigmoid(1.702 * x).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float QuickGeluScalar(float x)
    {
        return x * (1.0f / (1.0f + MathF.Exp(-1.702f * x)));
    }

    /// <summary>
    /// Applies QuickGELU elementwise across a span in-place.
    /// </summary>
    public static void QuickGelu(Span<float> data)
    {
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = QuickGeluScalar(data[i]);
        }
    }

    /// <summary>
    /// Applies SiLU (Swish) activation function: x * sigmoid(x).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float SiluScalar(float x)
    {
        return x / (1.0f + MathF.Exp(-x));
    }

    /// <summary>
    /// Applies SiLU elementwise across a span in-place.
    /// </summary>
    public static void Silu(Span<float> data)
    {
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = SiluScalar(data[i]);
        }
    }

    /// <summary>
    /// Applies Squared ReLU activation: (max(0, x))^2.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float SquaredReluScalar(float x)
    {
        float relu = MathF.Max(0f, x);
        return relu * relu;
    }

    /// <summary>
    /// Applies Squared ReLU elementwise across a span in-place.
    /// </summary>
    public static void SquaredRelu(Span<float> data)
    {
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = SquaredReluScalar(data[i]);
        }
    }

    /// <summary>
    /// Continuous 2D Rotary Position Embedding (Pixtral-style):
    /// Rotates first half of each head with horizontal patch position X, and second half with vertical patch position Y.
    /// </summary>
    public static void Continuous2DRoPE(
        float[] q,
        float[] k,
        int patchesX,
        int patchesY,
        int heads,
        int headDim,
        float theta = 10000.0f)
    {
        int halfDim = headDim / 2;
        int quarterDim = halfDim / 2;

        Parallel.For(0, patchesY, py =>
        {
            for (int px = 0; px < patchesX; px++)
            {
                int tokenIdx = py * patchesX + px;

                for (int h = 0; h < heads; h++)
                {
                    int headOff = (tokenIdx * heads + h) * headDim;

                    // Rotate X on first half
                    for (int d = 0; d < quarterDim; d++)
                    {
                        float freq = MathF.Pow(theta, -(float)(2 * d) / halfDim);
                        float angle = px * freq;
                        float cos = MathF.Cos(angle);
                        float sin = MathF.Sin(angle);

                        int i0 = headOff + d * 2;
                        int i1 = headOff + d * 2 + 1;

                        float q0 = q[i0], q1 = q[i1];
                        q[i0] = q0 * cos - q1 * sin;
                        q[i1] = q0 * sin + q1 * cos;

                        float k0 = k[i0], k1 = k[i1];
                        k[i0] = k0 * cos - k1 * sin;
                        k[i1] = k0 * sin + k1 * cos;
                    }

                    // Rotate Y on second half
                    for (int d = 0; d < quarterDim; d++)
                    {
                        float freq = MathF.Pow(theta, -(float)(2 * d) / halfDim);
                        float angle = py * freq;
                        float cos = MathF.Cos(angle);
                        float sin = MathF.Sin(angle);

                        int i0 = headOff + halfDim + d * 2;
                        int i1 = headOff + halfDim + d * 2 + 1;

                        float q0 = q[i0], q1 = q[i1];
                        q[i0] = q0 * cos - q1 * sin;
                        q[i1] = q0 * sin + q1 * cos;

                        float k0 = k[i0], k1 = k[i1];
                        k[i0] = k0 * cos - k1 * sin;
                        k[i1] = k0 * sin + k1 * cos;
                    }
                }
            }
        });
    }

    /// <summary>
    /// Interleaved 2D Rotary Position Embedding (Kimi / GLM style).
    /// </summary>
    public static void Interleaved2DRoPE(
        float[] q,
        float[] k,
        int patchesX,
        int patchesY,
        int heads,
        int headDim,
        float theta = 10000.0f)
    {
        int halfDim = headDim / 2;

        Parallel.For(0, patchesY, py =>
        {
            for (int px = 0; px < patchesX; px++)
            {
                int tokenIdx = py * patchesX + px;

                for (int h = 0; h < heads; h++)
                {
                    int headOff = (tokenIdx * heads + h) * headDim;

                    for (int d = 0; d < halfDim; d++)
                    {
                        float freq = MathF.Pow(theta, -(float)(2 * d) / headDim);
                        float angle = ((d % 2 == 0) ? px : py) * freq;
                        float cos = MathF.Cos(angle);
                        float sin = MathF.Sin(angle);

                        int i0 = headOff + d * 2;
                        int i1 = headOff + d * 2 + 1;

                        float q0 = q[i0], q1 = q[i1];
                        q[i0] = q0 * cos - q1 * sin;
                        q[i1] = q0 * sin + q1 * cos;

                        float k0 = k[i0], k1 = k[i1];
                        k[i0] = k0 * cos - k1 * sin;
                        k[i1] = k0 * sin + k1 * cos;
                    }
                }
            }
        });
    }

    /// <summary>
    /// Multimodal Rotary Position Embedding (M-RoPE, Qwen/MiMo/Exaone style).
    /// Rotates quarter sub-bands with X (width) and Y (height) coordinates.
    /// </summary>
    public static void ApplyMRoPE(
        float[] q,
        float[] k,
        int patchesX,
        int patchesY,
        int qHeads,
        int kvHeads,
        int headDim,
        float theta = 10000.0f)
    {
        int mropeHalf = headDim / 4;

        Parallel.For(0, patchesY, py =>
        {
            for (int px = 0; px < patchesX; px++)
            {
                int p = py * patchesX + px;

                for (int h = 0; h < qHeads; h++)
                {
                    int headOff = (p * qHeads + h) * headDim;

                    // Rotate X coordinate on first section
                    for (int d = 0; d < mropeHalf; d++)
                    {
                        float freqX = MathF.Pow(theta, -2.0f * d / headDim);
                        float thetaX = px * freqX;
                        float cosX = MathF.Cos(thetaX);
                        float sinX = MathF.Sin(thetaX);

                        float q0 = q[headOff + d];
                        float q1 = q[headOff + d + mropeHalf];
                        q[headOff + d] = q0 * cosX - q1 * sinX;
                        q[headOff + d + mropeHalf] = q0 * sinX + q1 * cosX;
                    }

                    // Rotate Y coordinate on second section
                    int ySecOff = headOff + 2 * mropeHalf;
                    for (int d = 0; d < mropeHalf; d++)
                    {
                        float freqY = MathF.Pow(theta, -2.0f * d / headDim);
                        float thetaY = py * freqY;
                        float cosY = MathF.Cos(thetaY);
                        float sinY = MathF.Sin(thetaY);

                        float q0 = q[ySecOff + d];
                        float q1 = q[ySecOff + d + mropeHalf];
                        q[ySecOff + d] = q0 * cosY - q1 * sinY;
                        q[ySecOff + d + mropeHalf] = q0 * sinY + q1 * cosY;
                    }
                }

                for (int h = 0; h < kvHeads; h++)
                {
                    int headOff = (p * kvHeads + h) * headDim;

                    // Rotate X coordinate on first section
                    for (int d = 0; d < mropeHalf; d++)
                    {
                        float freqX = MathF.Pow(theta, -2.0f * d / headDim);
                        float thetaX = px * freqX;
                        float cosX = MathF.Cos(thetaX);
                        float sinX = MathF.Sin(thetaX);

                        float k0 = k[headOff + d];
                        float k1 = k[headOff + d + mropeHalf];
                        k[headOff + d] = k0 * cosX - k1 * sinX;
                        k[headOff + d + mropeHalf] = k0 * sinX + k1 * cosX;
                    }

                    // Rotate Y coordinate on second section
                    int ySecOff = headOff + 2 * mropeHalf;
                    for (int d = 0; d < mropeHalf; d++)
                    {
                        float freqY = MathF.Pow(theta, -2.0f * d / headDim);
                        float thetaY = py * freqY;
                        float cosY = MathF.Cos(thetaY);
                        float sinY = MathF.Sin(thetaY);

                        float k0 = k[ySecOff + d];
                        float k1 = k[ySecOff + d + mropeHalf];
                        k[ySecOff + d] = k0 * cosY - k1 * sinY;
                        k[ySecOff + d + mropeHalf] = k0 * sinY + k1 * cosY;
                    }
                }
            }
        });
    }

    public static void ApplyMRoPE(
        float[] q,
        float[] k,
        int patchesX,
        int patchesY,
        int heads,
        int headDim,
        float theta = 10000.0f)
        => ApplyMRoPE(q, k, patchesX, patchesY, heads, heads, headDim, theta);

    /// <summary>
    /// PixelShuffle 2x2 spatial downsampler: merges 2x2 spatial blocks into 4x channel dimensions.
    /// Input: [gridY, gridX, inDim], Output: [gridY/2, gridX/2, 4*inDim].
    /// </summary>
    public static void PixelShuffle2x2(
        float[] input,
        int gridY,
        int gridX,
        int inDim,
        float[] output)
    {
        int outH = gridY / 2;
        int outW = gridX / 2;
        int outDim = inDim * 4;

        Parallel.For(0, outH, oy =>
        {
            for (int ox = 0; ox < outW; ox++)
            {
                int outTokenIdx = oy * outW + ox;
                int outOffset = outTokenIdx * outDim;

                int iy = oy * 2;
                int ix = ox * 2;

                int inIdx00 = (iy * gridX + ix) * inDim;
                int inIdx01 = (iy * gridX + (ix + 1)) * inDim;
                int inIdx10 = ((iy + 1) * gridX + ix) * inDim;
                int inIdx11 = ((iy + 1) * gridX + (ix + 1)) * inDim;

                Array.Copy(input, inIdx00, output, outOffset, inDim);
                Array.Copy(input, inIdx01, output, outOffset + inDim, inDim);
                Array.Copy(input, inIdx10, output, outOffset + 2 * inDim, inDim);
                Array.Copy(input, inIdx11, output, outOffset + 3 * inDim, inDim);
            }
        });
    }

    /// <summary>
    /// 3D Spatio-Temporal RoPE for video/temporal transformer architectures.
    /// </summary>
    public static void ApplyRoPE3D(
        float[] q,
        float[] k,
        int numTokens,
        int numHeads,
        int headDim,
        int tDim,
        int hDim,
        int wDim,
        float theta = 10000.0f)
    {
        int bandSize = headDim / 6;
        int hw = hDim * wDim;

        Parallel.For(0, numTokens, tokenIdx =>
        {
            int pt = tokenIdx / (hw > 0 ? hw : 1);
            int rem = tokenIdx % (hw > 0 ? hw : 1);
            int py = rem / (wDim > 0 ? wDim : 1);
            int px = rem % (wDim > 0 ? wDim : 1);

            for (int h = 0; h < numHeads; h++)
            {
                int headOff = (tokenIdx * numHeads + h) * headDim;

                // Temporal
                for (int d = 0; d < bandSize; d++)
                {
                    float freqT = MathF.Pow(theta, -2.0f * d / (bandSize * 2));
                    float cosT = MathF.Cos(pt * freqT);
                    float sinT = MathF.Sin(pt * freqT);

                    int i0 = headOff + d;
                    int i1 = headOff + d + bandSize;
                    float q0 = q[i0], q1 = q[i1];
                    q[i0] = q0 * cosT - q1 * sinT;
                    q[i1] = q0 * sinT + q1 * cosT;

                    float k0 = k[i0], k1 = k[i1];
                    k[i0] = k0 * cosT - k1 * sinT;
                    k[i1] = k0 * sinT + k1 * cosT;
                }

                // Height/Y
                for (int d = 0; d < bandSize; d++)
                {
                    float freqY = MathF.Pow(theta, -2.0f * d / (bandSize * 2));
                    float cosY = MathF.Cos(py * freqY);
                    float sinY = MathF.Sin(py * freqY);

                    int i0 = headOff + 2 * bandSize + d;
                    int i1 = headOff + 3 * bandSize + d;
                    float q0 = q[i0], q1 = q[i1];
                    q[i0] = q0 * cosY - q1 * sinY;
                    q[i1] = q0 * sinY + q1 * cosY;

                    float k0 = k[i0], k1 = k[i1];
                    k[i0] = k0 * cosY - k1 * sinY;
                    k[i1] = k0 * sinY + k1 * cosY;
                }

                // Width/X
                for (int d = 0; d < bandSize; d++)
                {
                    float freqX = MathF.Pow(theta, -2.0f * d / (bandSize * 2));
                    float cosX = MathF.Cos(px * freqX);
                    float sinX = MathF.Sin(px * freqX);

                    int i0 = headOff + 4 * bandSize + d;
                    int i1 = headOff + 5 * bandSize + d;
                    float q0 = q[i0], q1 = q[i1];
                    q[i0] = q0 * cosX - q1 * sinX;
                    q[i1] = q0 * sinX + q1 * cosX;

                    float k0 = k[i0], k1 = k[i1];
                    k[i0] = k0 * cosX - k1 * sinX;
                    k[i1] = k0 * sinX + k1 * cosX;
                }
            }
        });
    }

    /// <summary>
    /// Resolves typed unmanaged pointer to tensor data inside a GgufModel, checking multiple fallback aliases.
    /// </summary>
    public static T* GetTensorPtr<T>(GgufModel gguf, params string[] candidateNames) where T : unmanaged
    {
        foreach (var name in candidateNames)
        {
            var tensor = gguf.FindTensor(name);
            if (tensor.HasValue)
            {
                return (T*)gguf.GetTensorDataPtr(tensor.Value);
            }
        }
        return null;
    }
}
