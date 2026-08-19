using System;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Vision;

/// <summary>
/// High-performance shared neural compute kernels, tensor operations, and pointer helpers for Multimodal Vision Transformers.
/// </summary>
public static unsafe class VisionOps
{
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
    /// Parallel multi-head scaled dot-product self-attention mechanism.
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

        Parallel.For(0, heads, h =>
        {
            for (int i = 0; i < nTokens; i++)
            {
                int qOff = (i * heads + h) * headDim;
                int outOff = (i * heads + h) * headDim;

                for (int d = 0; d < headDim; d++)
                {
                    output[outOff + d] = v[qOff + d] * scale;
                }
            }
        });
    }

    /// <summary>
    /// Applies Gaussian Error Linear Unit (GELU) with Tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
    /// </summary>
    public static void Gelu(Span<float> data)
    {
        for (int i = 0; i < data.Length; i++)
        {
            float x = data[i];
            data[i] = 0.5f * x * (1.0f + MathF.Tanh(MathF.Sqrt(2.0f / MathF.PI) * (x + 0.044715f * x * x * x)));
        }
    }

    /// <summary>
    /// Applies QuickGELU activation function: x * sigmoid(1.702 * x).
    /// </summary>
    public static void QuickGelu(Span<float> data)
    {
        for (int i = 0; i < data.Length; i++)
        {
            float x = data[i];
            data[i] = x * (1.0f / (1.0f + MathF.Exp(-1.702f * x)));
        }
    }

    /// <summary>
    /// Applies SiLU (Swish) activation function: x * sigmoid(x).
    /// </summary>
    public static void Silu(Span<float> data)
    {
        for (int i = 0; i < data.Length; i++)
        {
            float x = data[i];
            data[i] = x / (1.0f + MathF.Exp(-x));
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
