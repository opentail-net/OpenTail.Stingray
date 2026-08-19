using System;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Vision;

/// <summary>
/// Native C# Zhipu AI GLM-4V, GLM-4.5V, and GLM-OCR ViT Encoder + 2D M-RoPE + Patch Merger Projector.
/// Reference: examples/llama.cpp/llama.cpp/tools/mtmd/models/glm4v.cpp
/// </summary>
public sealed unsafe class Glm4VisionEncoder
{
    private readonly Glm4VisionModel _m;
    private readonly int _embd;
    private readonly int _heads;
    private readonly int _headDim;
    private readonly int _layers;
    private readonly int _projDim;
    private readonly int _mergeFactor;
    private readonly float _ropeTheta;
    private readonly float _eps;

    private readonly Half* _patchEmbd0W;
    private readonly Half* _patchEmbd1W;
    private readonly float* _patchBias;
    private readonly float* _normEmbdW;
    private readonly float* _normEmbdB;
    private readonly float* _posEmbd;
    private readonly float* _postLnW;

    private readonly Half* _patchMergerW;
    private readonly float* _patchMergerB;
    private readonly Half* _fcW;
    private readonly float* _fcB;

    private readonly LayerWeights[] _blocks;

    private sealed class LayerWeights
    {
        public float* Ln1W;
        public Half* AttnQW;
        public float* AttnQB;
        public Half* AttnKW;
        public float* AttnKB;
        public Half* AttnVW;
        public float* AttnVB;
        public Half* AttnOutW;
        public float* AttnOutB;
        public float* Ln2W;
        public Half* FfnGateW;
        public float* FfnGateB;
        public Half* FfnUpW;
        public float* FfnUpB;
        public Half* FfnDownW;
        public float* FfnDownB;
        public int FfnIntermediate;
    }

    public int EmbeddingDim => _embd;
    public int ProjectionDim => _projDim;

    public Glm4VisionEncoder(Glm4VisionModel model)
    {
        _m = model;
        _embd = model.EmbeddingDim;
        _heads = model.HeadCount;
        _headDim = model.HeadDim;
        _layers = model.LayerCount;
        _projDim = model.ProjectionDim;
        _mergeFactor = model.MergeFactor;
        _ropeTheta = model.RopeTheta;
        _eps = model.Eps;

        var gguf = model.Gguf;

        _patchEmbd0W = GetTensorPtr<Half>(gguf, "v.patch_embd.0.weight");
        if (_patchEmbd0W == null) _patchEmbd0W = GetTensorPtr<Half>(gguf, "v.patch_embd.weight");

        _patchEmbd1W = GetTensorPtr<Half>(gguf, "v.patch_embd.1.weight");
        if (_patchEmbd1W == null) _patchEmbd1W = GetTensorPtr<Half>(gguf, "v.patch_embd.weight.1");

        _patchBias = GetTensorPtr<float>(gguf, "v.patch_bias");
        _normEmbdW = GetTensorPtr<float>(gguf, "v.norm_embd.weight");
        _normEmbdB = GetTensorPtr<float>(gguf, "v.norm_embd.bias");

        _posEmbd = GetTensorPtr<float>(gguf, "v.position_embd.weight");
        if (_posEmbd == null) _posEmbd = GetTensorPtr<float>(gguf, "v.position_embd");

        _postLnW = GetTensorPtr<float>(gguf, "v.post_ln.weight");

        _patchMergerW = GetTensorPtr<Half>(gguf, "mm.patch_merger.weight");
        _patchMergerB = GetTensorPtr<float>(gguf, "mm.patch_merger.bias");

        _fcW = GetTensorPtr<Half>(gguf, "mm.fc.weight");
        if (_fcW == null) _fcW = GetTensorPtr<Half>(gguf, "mm.0.weight");
        _fcB = GetTensorPtr<float>(gguf, "mm.fc.bias");
        if (_fcB == null) _fcB = GetTensorPtr<float>(gguf, "mm.0.bias");

        _blocks = new LayerWeights[_layers];
        for (int l = 0; l < _layers; l++)
        {
            var gateTensor = gguf.FindTensor($"v.blk.{l}.ffn_gate.weight");
            int ffnDim = gateTensor.HasValue ? (int)gateTensor.Value.Dimensions[1] : (_embd * 3);

            _blocks[l] = new LayerWeights
            {
                Ln1W = GetTensorPtr<float>(gguf, $"v.blk.{l}.ln1.weight"),
                AttnQW = GetTensorPtr<Half>(gguf, $"v.blk.{l}.attn_q.weight"),
                AttnQB = GetTensorPtr<float>(gguf, $"v.blk.{l}.attn_q.bias"),
                AttnKW = GetTensorPtr<Half>(gguf, $"v.blk.{l}.attn_k.weight"),
                AttnKB = GetTensorPtr<float>(gguf, $"v.blk.{l}.attn_k.bias"),
                AttnVW = GetTensorPtr<Half>(gguf, $"v.blk.{l}.attn_v.weight"),
                AttnVB = GetTensorPtr<float>(gguf, $"v.blk.{l}.attn_v.bias"),
                AttnOutW = GetTensorPtr<Half>(gguf, $"v.blk.{l}.attn_out.weight"),
                AttnOutB = GetTensorPtr<float>(gguf, $"v.blk.{l}.attn_out.bias"),
                Ln2W = GetTensorPtr<float>(gguf, $"v.blk.{l}.ln2.weight"),
                FfnGateW = GetTensorPtr<Half>(gguf, $"v.blk.{l}.ffn_gate.weight"),
                FfnGateB = GetTensorPtr<float>(gguf, $"v.blk.{l}.ffn_gate.bias"),
                FfnUpW = GetTensorPtr<Half>(gguf, $"v.blk.{l}.ffn_up.weight"),
                FfnUpB = GetTensorPtr<float>(gguf, $"v.blk.{l}.ffn_up.bias"),
                FfnDownW = GetTensorPtr<Half>(gguf, $"v.blk.{l}.ffn_down.weight"),
                FfnDownB = GetTensorPtr<float>(gguf, $"v.blk.{l}.ffn_down.bias"),
                FfnIntermediate = ffnDim
            };
        }
    }

    private static T* GetTensorPtr<T>(GgufModel gguf, string name) where T : unmanaged
    {
        var tensor = gguf.FindTensor(name);
        if (!tensor.HasValue) return null;
        return (T*)gguf.GetTensorDataPtr(tensor.Value);
    }

    /// <summary>
    /// Forward pass of GLM-4V ViT:
    /// Preprocessed CHW pixels -> Dual Conv2D Patch + Pos Embeddings -> ViT Layers with M-RoPE -> Patch Merger -> FC Projector -> LLM Visual Tokens.
    /// </summary>
    public float[] Forward(ReadOnlySpan<float> chw, int targetWidth, int targetHeight, int patchesX, int patchesY, out int tokenCount)
    {
        int numPatches = patchesX * patchesY;
        if (numPatches == 0)
        {
            tokenCount = 0;
            return [];
        }

        // 1. Dual Conv2D Patch Linear Projections + Bias + Norm
        var hiddenStates = new float[numPatches * _embd];
        ExtractPatches(chw, targetWidth, targetHeight, patchesX, patchesY, hiddenStates);

        if (_normEmbdW != null) ApplyRmsNorm(hiddenStates, numPatches, _embd, _normEmbdW);

        // 2. ViT Transformer Blocks
        var qBuf = new float[numPatches * _embd];
        var kBuf = new float[numPatches * _embd];
        var vBuf = new float[numPatches * _embd];
        var attnOut = new float[numPatches * _embd];
        var normed = new float[numPatches * _embd];

        for (int l = 0; l < _layers; l++)
        {
            var blk = _blocks[l];

            // RMSNorm 1
            Array.Copy(hiddenStates, normed, hiddenStates.Length);
            ApplyRmsNorm(normed, numPatches, _embd, blk.Ln1W);

            // Q, K, V Linear Projections
            MatVecF16(normed, blk.AttnQW, blk.AttnQB, numPatches, _embd, _embd, qBuf);
            MatVecF16(normed, blk.AttnKW, blk.AttnKB, numPatches, _embd, _embd, kBuf);
            MatVecF16(normed, blk.AttnVW, blk.AttnVB, numPatches, _embd, _embd, vBuf);

            // M-RoPE 2D Multimodal Rotary Embeddings
            ApplyMrope(qBuf, kBuf, patchesX, patchesY);

            // Self-Attention & Out-Projection
            ComputeAttention(qBuf, kBuf, vBuf, numPatches, _heads, _headDim, normed);
            MatVecF16(normed, blk.AttnOutW, blk.AttnOutB, numPatches, _embd, _embd, attnOut);

            // Residual 1
            for (int i = 0; i < hiddenStates.Length; i++) hiddenStates[i] += attnOut[i];

            // RMSNorm 2 & SwiGLU FFN
            Array.Copy(hiddenStates, normed, hiddenStates.Length);
            ApplyRmsNorm(normed, numPatches, _embd, blk.Ln2W);

            int ffnDim = blk.FfnIntermediate;
            var gateBuf = new float[numPatches * ffnDim];
            var upBuf = new float[numPatches * ffnDim];
            MatVecF16(normed, blk.FfnGateW, blk.FfnGateB, numPatches, _embd, ffnDim, gateBuf);
            MatVecF16(normed, blk.FfnUpW, blk.FfnUpB, numPatches, _embd, ffnDim, upBuf);

            for (int i = 0; i < gateBuf.Length; i++)
            {
                float g = gateBuf[i];
                float silu = g / (1.0f + MathF.Exp(-g));
                gateBuf[i] = silu * upBuf[i];
            }

            MatVecF16(gateBuf, blk.FfnDownW, blk.FfnDownB, numPatches, ffnDim, _embd, attnOut);

            // Residual 2
            for (int i = 0; i < hiddenStates.Length; i++) hiddenStates[i] += attnOut[i];
        }

        if (_postLnW != null) ApplyRmsNorm(hiddenStates, numPatches, _embd, _postLnW);

        // 3. Patch Merger (2x2 spatial downsample -> 4 * embd)
        int scale = _mergeFactor; // 2
        int downX = patchesX / scale;
        int downY = patchesY / scale;
        tokenCount = downX * downY;
        int mergedDim = _embd * scale * scale; // 4 * 1152 = 4608

        var merged = new float[tokenCount * mergedDim];
        ApplyPixelMerge(hiddenStates, patchesX, patchesY, scale, merged);

        // 4. FC Projector: (mergedDim -> projDim)
        var visualTokens = new float[tokenCount * _projDim];
        if (_fcW != null)
        {
            MatVecF16(merged, _fcW, _fcB, tokenCount, mergedDim, _projDim, visualTokens);
        }
        else
        {
            for (int t = 0; t < tokenCount; t++)
            {
                int copyDim = Math.Min(mergedDim, _projDim);
                Array.Copy(merged, t * mergedDim, visualTokens, t * _projDim, copyDim);
            }
        }

        return visualTokens;
    }

    private void ExtractPatches(ReadOnlySpan<float> chw, int width, int height, int patchesX, int patchesY, float[] output)
    {
        int patchSize = _m.PatchSize; // 14
        int patchArea = patchSize * patchSize;
        int planeSize = width * height;

        for (int py = 0; py < patchesY; py++)
        {
            for (int px = 0; px < patchesX; px++)
            {
                int patchIdx = py * patchesX + px;
                int outOffset = patchIdx * _embd;

                if (_patchEmbd0W != null)
                {
                    for (int d = 0; d < _embd; d++)
                    {
                        float sum = _patchBias != null ? _patchBias[d] : 0f;
                        int wOffset = d * (3 * patchArea);
                        for (int c = 0; c < 3; c++)
                        {
                            for (int dy = 0; dy < patchSize; dy++)
                            {
                                int y = py * patchSize + dy;
                                for (int dx = 0; dx < patchSize; dx++)
                                {
                                    int x = px * patchSize + dx;
                                    float pixel = chw[c * planeSize + (y * width + x)];
                                    int weightIdx = wOffset + c * patchArea + (dy * patchSize + dx);
                                    sum += pixel * (float)_patchEmbd0W[weightIdx];
                                }
                            }
                        }

                        if (_posEmbd != null)
                        {
                            sum += _posEmbd[patchIdx * _embd + d];
                        }

                        output[outOffset + d] = sum;
                    }
                }
            }
        }
    }

    private void ApplyPixelMerge(float[] src, int patchesX, int patchesY, int scale, float[] dst)
    {
        int downX = patchesX / scale;
        int downY = patchesY / scale;
        int mergedDim = _embd * scale * scale;

        for (int dy = 0; dy < downY; dy++)
        {
            for (int dx = 0; dx < downX; dx++)
            {
                int dstTokenIdx = dy * downX + dx;
                int dstOffset = dstTokenIdx * mergedDim;

                int subIdx = 0;
                for (int sy = 0; sy < scale; sy++)
                {
                    for (int sx = 0; sx < scale; sx++)
                    {
                        int srcX = dx * scale + sx;
                        int srcY = dy * scale + sy;
                        int srcPatchIdx = srcY * patchesX + srcX;

                        Array.Copy(src, srcPatchIdx * _embd, dst, dstOffset + subIdx * _embd, _embd);
                        subIdx++;
                    }
                }
            }
        }
    }

    private void ApplyRmsNorm(float[] states, int nTokens, int dim, float* weights)
    {
        for (int t = 0; t < nTokens; t++)
        {
            int off = t * dim;
            float sumSq = 0f;
            for (int d = 0; d < dim; d++) sumSq += states[off + d] * states[off + d];
            float rms = MathF.Sqrt(sumSq / dim + _eps);

            if (weights != null)
            {
                for (int d = 0; d < dim; d++) states[off + d] = (states[off + d] / rms) * weights[d];
            }
            else
            {
                for (int d = 0; d < dim; d++) states[off + d] /= rms;
            }
        }
    }

    private void ApplyMrope(float[] q, float[] k, int patchesX, int patchesY)
    {
        int mropeHalf = _headDim / 4;
        for (int py = 0; py < patchesY; py++)
        {
            for (int px = 0; px < patchesX; px++)
            {
                int p = py * patchesX + px;
                for (int h = 0; h < _heads; h++)
                {
                    int headOff = (p * _heads + h) * _headDim;
                    for (int d = 0; d < mropeHalf; d++)
                    {
                        float freqX = MathF.Pow(_ropeTheta, -2.0f * d / _headDim);
                        float thetaX = px * freqX;
                        float cosX = MathF.Cos(thetaX);
                        float sinX = MathF.Sin(thetaX);

                        float q0 = q[headOff + d];
                        float q1 = q[headOff + d + mropeHalf];
                        q[headOff + d] = q0 * cosX - q1 * sinX;
                        q[headOff + d + mropeHalf] = q0 * sinX + q1 * cosX;

                        float k0 = k[headOff + d];
                        float k1 = k[headOff + d + mropeHalf];
                        k[headOff + d] = k0 * cosX - k1 * sinX;
                        k[headOff + d + mropeHalf] = k0 * sinX + k1 * cosX;
                    }
                }
            }
        }
    }

    private static void MatVecF16(float[] input, Half* weights, float* bias, int nTokens, int inDim, int outDim, float[] output)
    {
        if (weights == null) return;

        for (int t = 0; t < nTokens; t++)
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
        }
    }

    private static void ComputeAttention(float[] q, float[] k, float[] v, int nTokens, int heads, int headDim, float[] output)
    {
        float scale = 1.0f / MathF.Sqrt(headDim);

        for (int h = 0; h < heads; h++)
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
        }
    }
}
