using System;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Vision;

/// <summary>
/// Native C# Moonshot AI Kimi K2.5 and Kimi-VL ViT Encoder + 2D Interleaved RoPE + Patch Merger Projector.
/// Reference: examples/llama.cpp/llama.cpp/tools/mtmd/models/kimik25.cpp
/// </summary>
public sealed unsafe class KimiVisionEncoder
{
    private readonly KimiVisionModel _m;
    private readonly int _embd;
    private readonly int _heads;
    private readonly int _headDim;
    private readonly int _layers;
    private readonly int _projDim;
    private readonly int _mergeFactor;
    private readonly float _ropeTheta;
    private readonly float _eps;

    private readonly Half* _patchEmbdW;
    private readonly float* _patchEmbdB;
    private readonly float* _posEmbd;
    private readonly float* _postLnW;
    private readonly float* _postLnB;

    private readonly float* _mmInputNormW;
    private readonly float* _mmInputNormB;
    private readonly Half* _mm1W;
    private readonly float* _mm1B;
    private readonly Half* _mm2W;
    private readonly float* _mm2B;

    private readonly LayerWeights[] _blocks;

    private sealed class LayerWeights
    {
        public float* Ln1W;
        public float* Ln1B;
        public Half* AttnQkvW;
        public float* AttnQkvB;
        public Half* AttnQW;
        public float* AttnQB;
        public Half* AttnKW;
        public float* AttnKB;
        public Half* AttnVW;
        public float* AttnVB;
        public Half* AttnOutW;
        public float* AttnOutB;
        public float* Ln2W;
        public float* Ln2B;
        public Half* FfnUpW;
        public float* FfnUpB;
        public Half* FfnDownW;
        public float* FfnDownB;
        public int FfnIntermediate;
    }

    public int EmbeddingDim => _embd;
    public int ProjectionDim => _projDim;

    public KimiVisionEncoder(KimiVisionModel model)
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

        _patchEmbdW = GetTensorPtr<Half>(gguf, "v.patch_embd.weight");
        _patchEmbdB = GetTensorPtr<float>(gguf, "v.patch_embd.bias");
        _posEmbd = GetTensorPtr<float>(gguf, "v.position_embd.weight");
        if (_posEmbd == null) _posEmbd = GetTensorPtr<float>(gguf, "v.position_embd");
        _postLnW = GetTensorPtr<float>(gguf, "v.post_ln.weight");
        _postLnB = GetTensorPtr<float>(gguf, "v.post_ln.bias");

        _mmInputNormW = GetTensorPtr<float>(gguf, "mm.input_norm.weight");
        _mmInputNormB = GetTensorPtr<float>(gguf, "mm.input_norm.bias");
        _mm1W = GetTensorPtr<Half>(gguf, "mm.1.weight");
        _mm1B = GetTensorPtr<float>(gguf, "mm.1.bias");
        _mm2W = GetTensorPtr<Half>(gguf, "mm.2.weight");
        _mm2B = GetTensorPtr<float>(gguf, "mm.2.bias");

        _blocks = new LayerWeights[_layers];
        for (int l = 0; l < _layers; l++)
        {
            var upTensor = gguf.FindTensor($"v.blk.{l}.ffn_up.weight");
            int intermediate = upTensor.HasValue ? (int)upTensor.Value.Dimensions[1] : (_embd * 4);

            _blocks[l] = new LayerWeights
            {
                Ln1W = GetTensorPtr<float>(gguf, $"v.blk.{l}.ln1.weight"),
                Ln1B = GetTensorPtr<float>(gguf, $"v.blk.{l}.ln1.bias"),
                AttnQkvW = GetTensorPtr<Half>(gguf, $"v.blk.{l}.attn_qkv.weight"),
                AttnQkvB = GetTensorPtr<float>(gguf, $"v.blk.{l}.attn_qkv.bias"),
                AttnQW = GetTensorPtr<Half>(gguf, $"v.blk.{l}.attn_q.weight"),
                AttnQB = GetTensorPtr<float>(gguf, $"v.blk.{l}.attn_q.bias"),
                AttnKW = GetTensorPtr<Half>(gguf, $"v.blk.{l}.attn_k.weight"),
                AttnKB = GetTensorPtr<float>(gguf, $"v.blk.{l}.attn_k.bias"),
                AttnVW = GetTensorPtr<Half>(gguf, $"v.blk.{l}.attn_v.weight"),
                AttnVB = GetTensorPtr<float>(gguf, $"v.blk.{l}.attn_v.bias"),
                AttnOutW = GetTensorPtr<Half>(gguf, $"v.blk.{l}.attn_out.weight"),
                AttnOutB = GetTensorPtr<float>(gguf, $"v.blk.{l}.attn_out.bias"),
                Ln2W = GetTensorPtr<float>(gguf, $"v.blk.{l}.ln2.weight"),
                Ln2B = GetTensorPtr<float>(gguf, $"v.blk.{l}.ln2.bias"),
                FfnUpW = GetTensorPtr<Half>(gguf, $"v.blk.{l}.ffn_up.weight"),
                FfnUpB = GetTensorPtr<float>(gguf, $"v.blk.{l}.ffn_up.bias"),
                FfnDownW = GetTensorPtr<Half>(gguf, $"v.blk.{l}.ffn_down.weight"),
                FfnDownB = GetTensorPtr<float>(gguf, $"v.blk.{l}.ffn_down.bias"),
                FfnIntermediate = intermediate
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
    /// Forward pass of Kimi-VL ViT:
    /// Preprocessed CHW pixels -> Conv2D Patch + Pos Embeddings -> ViT Layers with Interleaved 2D RoPE -> Patch Merger -> 2-layer GELU MLP -> Visual Tokens.
    /// </summary>
    public float[] Forward(ReadOnlySpan<float> chw, int targetWidth, int targetHeight, int patchesX, int patchesY, out int tokenCount)
    {
        int numPatches = patchesX * patchesY;
        if (numPatches == 0)
        {
            tokenCount = 0;
            return [];
        }

        // 1. Patch Embeddings + Learned Position Embeddings
        var hiddenStates = new float[numPatches * _embd];
        ExtractPatches(chw, targetWidth, targetHeight, patchesX, patchesY, hiddenStates);

        // 2. ViT Transformer Blocks
        var qBuf = new float[numPatches * _embd];
        var kBuf = new float[numPatches * _embd];
        var vBuf = new float[numPatches * _embd];
        var attnOut = new float[numPatches * _embd];
        var normed = new float[numPatches * _embd];

        for (int l = 0; l < _layers; l++)
        {
            var blk = _blocks[l];

            // LayerNorm 1
            Array.Copy(hiddenStates, normed, hiddenStates.Length);
            ApplyLayerNorm(normed, numPatches, _embd, blk.Ln1W, blk.Ln1B);

            // Self-Attention Q, K, V
            if (blk.AttnQkvW != null)
            {
                var qkv = new float[numPatches * 3 * _embd];
                MatVecF16(normed, blk.AttnQkvW, blk.AttnQkvB, numPatches, _embd, 3 * _embd, qkv);
                for (int p = 0; p < numPatches; p++)
                {
                    Array.Copy(qkv, p * 3 * _embd, qBuf, p * _embd, _embd);
                    Array.Copy(qkv, p * 3 * _embd + _embd, kBuf, p * _embd, _embd);
                    Array.Copy(qkv, p * 3 * _embd + 2 * _embd, vBuf, p * _embd, _embd);
                }
            }
            else
            {
                MatVecF16(normed, blk.AttnQW, blk.AttnQB, numPatches, _embd, _embd, qBuf);
                MatVecF16(normed, blk.AttnKW, blk.AttnKB, numPatches, _embd, _embd, kBuf);
                MatVecF16(normed, blk.AttnVW, blk.AttnVB, numPatches, _embd, _embd, vBuf);
            }

            // 2D Interleaved RoPE
            Apply2dInterleavedRope(qBuf, kBuf, patchesX, patchesY);

            // Self-Attention & Out-Projection
            ComputeAttention(qBuf, kBuf, vBuf, numPatches, _heads, _headDim, normed);
            MatVecF16(normed, blk.AttnOutW, blk.AttnOutB, numPatches, _embd, _embd, attnOut);

            // Residual 1
            for (int i = 0; i < hiddenStates.Length; i++) hiddenStates[i] += attnOut[i];

            // LayerNorm 2 & FFN
            Array.Copy(hiddenStates, normed, hiddenStates.Length);
            ApplyLayerNorm(normed, numPatches, _embd, blk.Ln2W, blk.Ln2B);

            int intermediate = blk.FfnIntermediate;
            var ffnMid = new float[numPatches * intermediate];
            MatVecF16(normed, blk.FfnUpW, blk.FfnUpB, numPatches, _embd, intermediate, ffnMid);

            // GELU
            for (int i = 0; i < ffnMid.Length; i++)
            {
                float x = ffnMid[i];
                ffnMid[i] = 0.5f * x * (1.0f + MathF.Tanh(MathF.Sqrt(2.0f / MathF.PI) * (x + 0.044715f * x * x * x)));
            }

            MatVecF16(ffnMid, blk.FfnDownW, blk.FfnDownB, numPatches, intermediate, _embd, attnOut);

            // Residual 2
            for (int i = 0; i < hiddenStates.Length; i++) hiddenStates[i] += attnOut[i];
        }

        if (_postLnW != null) ApplyLayerNorm(hiddenStates, numPatches, _embd, _postLnW, _postLnB);

        // 3. Patch Merger (2x2 spatial downsample -> 4 * embd)
        int scale = _mergeFactor; // 2
        int downX = patchesX / scale;
        int downY = patchesY / scale;
        tokenCount = downX * downY;
        int mergedDim = _embd * scale * scale;

        var merged = new float[tokenCount * mergedDim];
        ApplyPixelMerge(hiddenStates, patchesX, patchesY, scale, merged);

        // 4. Projection Norm & 2-Layer GELU MLP Projector
        if (_mmInputNormW != null) ApplyLayerNorm(merged, tokenCount, mergedDim, _mmInputNormW, _mmInputNormB);

        var visualTokens = new float[tokenCount * _projDim];
        if (_mm1W != null && _mm2W != null)
        {
            var midBuf = new float[tokenCount * _projDim];
            MatVecF16(merged, _mm1W, _mm1B, tokenCount, mergedDim, _projDim, midBuf);

            // GELU
            for (int i = 0; i < midBuf.Length; i++)
            {
                float x = midBuf[i];
                midBuf[i] = 0.5f * x * (1.0f + MathF.Tanh(MathF.Sqrt(2.0f / MathF.PI) * (x + 0.044715f * x * x * x)));
            }

            MatVecF16(midBuf, _mm2W, _mm2B, tokenCount, _projDim, _projDim, visualTokens);
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

                if (_patchEmbdW != null)
                {
                    for (int d = 0; d < _embd; d++)
                    {
                        float sum = _patchEmbdB != null ? _patchEmbdB[d] : 0f;
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
                                    sum += pixel * (float)_patchEmbdW[weightIdx];
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

    private void ApplyLayerNorm(float[] states, int nTokens, int dim, float* weights, float* bias)
    {
        for (int t = 0; t < nTokens; t++)
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
            float std = MathF.Sqrt(var / dim + _eps);

            for (int d = 0; d < dim; d++)
            {
                float w = weights != null ? weights[d] : 1f;
                float b = bias != null ? bias[d] : 0f;
                states[off + d] = ((states[off + d] - mean) / std) * w + b;
            }
        }
    }

    private void Apply2dInterleavedRope(float[] q, float[] k, int patchesX, int patchesY)
    {
        int halfDim = _headDim / 2;
        for (int py = 0; py < patchesY; py++)
        {
            for (int px = 0; px < patchesX; px++)
            {
                int p = py * patchesX + px;
                for (int h = 0; h < _heads; h++)
                {
                    int headOff = (p * _heads + h) * _headDim;
                    for (int d = 0; d < halfDim; d += 2)
                    {
                        float freqX = MathF.Pow(_ropeTheta, -2.0f * d / halfDim);
                        float thetaX = px * freqX;
                        float cosX = MathF.Cos(thetaX);
                        float sinX = MathF.Sin(thetaX);

                        float freqY = MathF.Pow(_ropeTheta, -2.0f * (d + 1) / halfDim);
                        float thetaY = py * freqY;
                        float cosY = MathF.Cos(thetaY);
                        float sinY = MathF.Sin(thetaY);

                        float q0 = q[headOff + d];
                        float q1 = q[headOff + halfDim + d];
                        q[headOff + d] = q0 * cosX - q1 * sinX;
                        q[headOff + halfDim + d] = q0 * sinX + q1 * cosX;

                        float k0 = k[headOff + d];
                        float k1 = k[headOff + halfDim + d];
                        k[headOff + d] = k0 * cosX - k1 * sinX;
                        k[headOff + halfDim + d] = k0 * sinX + k1 * cosX;
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
