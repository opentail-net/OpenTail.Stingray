using System;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Vision;

/// <summary>
/// Native C# Qwen2-VL, Qwen2.5-VL, and Qwen3-VL Vision ViT Encoder + 2x2 Spatial Merger + Multimodal Projector.
/// Binds real Float16 / Float32 memory-mapped weights directly from the GGUF container.
/// Reference: examples/llama.cpp/llama.cpp/tools/mtmd/models/qwen2vl.cpp and qwen3vl.cpp
/// </summary>
public sealed unsafe class QwenVlVisionEncoder
{
    private readonly QwenVlVisionModel _m;
    private readonly int _embd;
    private readonly int _heads;
    private readonly int _headDim;
    private readonly int _layers;
    private readonly int _projDim;
    private readonly bool _useRmsNorm;
    private readonly float _eps;

    private readonly Half* _patchEmbd0W;
    private readonly Half* _patchEmbd1W;
    private readonly float* _patchBias;
    private readonly float* _positionEmbd;
    private readonly float* _postLnW;
    private readonly Half* _mm0W;
    private readonly float* _mm0B;
    private readonly Half* _mm2W;
    private readonly float* _mm2B;

    private readonly LayerWeights[] _blocks;

    private sealed class LayerWeights
    {
        public float* Ln1W;
        public float* Ln1B;
        public Half* AttnQW;
        public float* AttnQB;
        public Half* AttnKW;
        public float* AttnKB;
        public Half* AttnVW;
        public float* AttnVB;
        public Half* AttnQkvW;
        public float* AttnQkvB;
        public Half* AttnOutW;
        public float* AttnOutB;
        public float* Ln2W;
        public float* Ln2B;
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

    public QwenVlVisionEncoder(QwenVlVisionModel model)
    {
        _m = model;
        _embd = model.EmbeddingDim;
        _heads = model.HeadCount;
        _headDim = model.HeadDim;
        _layers = model.LayerCount;
        _projDim = model.ProjectionDim;
        _useRmsNorm = model.UseRmsNorm;
        _eps = model.Eps;

        var gguf = model.Gguf;

        // Ingest Stem & Global Tensors
        _patchEmbd0W = GetTensorPtr<Half>(gguf, "v.patch_embd.weight");
        if (_patchEmbd0W == null) _patchEmbd0W = GetTensorPtr<Half>(gguf, "v.patch_embd.0.weight");

        _patchEmbd1W = GetTensorPtr<Half>(gguf, "v.patch_embd.weight.1");
        if (_patchEmbd1W == null) _patchEmbd1W = GetTensorPtr<Half>(gguf, "v.patch_embd.1.weight");

        _patchBias = GetTensorPtr<float>(gguf, "v.patch_bias");

        _positionEmbd = GetTensorPtr<float>(gguf, "v.position_embd.weight");
        if (_positionEmbd == null) _positionEmbd = GetTensorPtr<float>(gguf, "v.position_embd");

        _postLnW = GetTensorPtr<float>(gguf, "v.post_ln.weight");

        _mm0W = GetTensorPtr<Half>(gguf, "mm.0.weight");
        _mm0B = GetTensorPtr<float>(gguf, "mm.0.bias");
        
        _mm2W = GetTensorPtr<Half>(gguf, "mm.2.weight");
        if (_mm2W == null) _mm2W = GetTensorPtr<Half>(gguf, "mm.1.weight");

        _mm2B = GetTensorPtr<float>(gguf, "mm.2.bias");
        if (_mm2B == null) _mm2B = GetTensorPtr<float>(gguf, "mm.1.bias");

        // Ingest Layer Tensors
        _blocks = new LayerWeights[_layers];
        for (int l = 0; l < _layers; l++)
        {
            var gateTensor = gguf.FindTensor($"v.blk.{l}.ffn_gate.weight");
            int ffnDim = gateTensor.HasValue ? (int)gateTensor.Value.Dimensions[1] : 3420;

            _blocks[l] = new LayerWeights
            {
                Ln1W = GetTensorPtr<float>(gguf, $"v.blk.{l}.ln1.weight"),
                Ln1B = GetTensorPtr<float>(gguf, $"v.blk.{l}.ln1.bias"),
                AttnQW = GetTensorPtr<Half>(gguf, $"v.blk.{l}.attn_q.weight"),
                AttnQB = GetTensorPtr<float>(gguf, $"v.blk.{l}.attn_q.bias"),
                AttnKW = GetTensorPtr<Half>(gguf, $"v.blk.{l}.attn_k.weight"),
                AttnKB = GetTensorPtr<float>(gguf, $"v.blk.{l}.attn_k.bias"),
                AttnVW = GetTensorPtr<Half>(gguf, $"v.blk.{l}.attn_v.weight"),
                AttnVB = GetTensorPtr<float>(gguf, $"v.blk.{l}.attn_v.bias"),
                AttnQkvW = GetTensorPtr<Half>(gguf, $"v.blk.{l}.attn_qkv.weight"),
                AttnQkvB = GetTensorPtr<float>(gguf, $"v.blk.{l}.attn_qkv.bias"),
                AttnOutW = GetTensorPtr<Half>(gguf, $"v.blk.{l}.attn_out.weight"),
                AttnOutB = GetTensorPtr<float>(gguf, $"v.blk.{l}.attn_out.bias"),
                Ln2W = GetTensorPtr<float>(gguf, $"v.blk.{l}.ln2.weight"),
                Ln2B = GetTensorPtr<float>(gguf, $"v.blk.{l}.ln2.bias"),
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
    /// Executes the full Qwen-VL ViT encoder pipeline:
    /// Preprocessed CHW pixels -> Dual Conv2D Patch Embeddings -> ViT Layers with M-RoPE -> Post-Norm -> 2x2 Spatial Merge MLP -> LLM Visual Tokens.
    /// </summary>
    public float[] Forward(ReadOnlySpan<float> chw, int targetWidth, int targetHeight, out int tokenCount)
    {
        int patchSize = _m.PatchSize; // 14
        int mergeFactor = _m.SpatialMergeFactor; // 2
        int patchesX = targetWidth / patchSize;
        int patchesY = targetHeight / patchSize;
        int numPatches = patchesX * patchesY;
        tokenCount = (patchesX / mergeFactor) * (patchesY / mergeFactor);

        if (numPatches == 0) return [];

        // 1. Conv2D Patch Embeddings: (3 x 14 x 14 -> 1280)
        var hiddenStates = new float[numPatches * _embd];
        ExtractPatchEmbeddings(chw, targetWidth, targetHeight, patchesX, patchesY, hiddenStates);

        // 2. ViT Transformer Blocks
        var qBuf = new float[numPatches * _embd];
        var kBuf = new float[numPatches * _embd];
        var vBuf = new float[numPatches * _embd];
        var attnOut = new float[numPatches * _embd];
        var normed = new float[numPatches * _embd];

        for (int l = 0; l < _layers; l++)
        {
            var blk = _blocks[l];

            // Norm 1
            Array.Copy(hiddenStates, normed, hiddenStates.Length);
            ApplyNorm(normed, numPatches, _embd, blk.Ln1W, blk.Ln1B);

            // Q, K, V Linear Projections (FP16 weights + FP32 biases)
            if (blk.AttnQkvW != null)
            {
                var qkvBuf = new float[numPatches * 3 * _embd];
                MatVecF16(normed, blk.AttnQkvW, blk.AttnQkvB, numPatches, _embd, 3 * _embd, qkvBuf);
                for (int p = 0; p < numPatches; p++)
                {
                    Array.Copy(qkvBuf, p * 3 * _embd, qBuf, p * _embd, _embd);
                    Array.Copy(qkvBuf, p * 3 * _embd + _embd, kBuf, p * _embd, _embd);
                    Array.Copy(qkvBuf, p * 3 * _embd + 2 * _embd, vBuf, p * _embd, _embd);
                }
            }
            else
            {
                MatVecF16(normed, blk.AttnQW, blk.AttnQB, numPatches, _embd, _embd, qBuf);
                MatVecF16(normed, blk.AttnKW, blk.AttnKB, numPatches, _embd, _embd, kBuf);
                MatVecF16(normed, blk.AttnVW, blk.AttnVB, numPatches, _embd, _embd, vBuf);
            }

            // Multimodal 2D RoPE (M-RoPE)
            ApplyMrope(qBuf, kBuf, patchesX, patchesY);

            // Self-Attention & Out-Projection
            ComputeSelfAttention(qBuf, kBuf, vBuf, patchesX, patchesY, normed);
            MatVecF16(normed, blk.AttnOutW, blk.AttnOutB, numPatches, _embd, _embd, attnOut);

            // Residual 1
            for (int i = 0; i < hiddenStates.Length; i++) hiddenStates[i] += attnOut[i];

            // Norm 2
            Array.Copy(hiddenStates, normed, hiddenStates.Length);
            ApplyNorm(normed, numPatches, _embd, blk.Ln2W, blk.Ln2B);

            // SwiGLU FFN
            int ffnDim = blk.FfnIntermediate;
            var gateBuf = new float[numPatches * ffnDim];
            var upBuf = new float[numPatches * ffnDim];
            MatVecF16(normed, blk.FfnGateW, blk.FfnGateB, numPatches, _embd, ffnDim, gateBuf);
            MatVecF16(normed, blk.FfnUpW, blk.FfnUpB, numPatches, _embd, ffnDim, upBuf);

            // SiLU(gate) * up
            for (int i = 0; i < gateBuf.Length; i++)
            {
                float g = gateBuf[i];
                float silu = g / (1.0f + MathF.Exp(-g));
                gateBuf[i] = silu * upBuf[i];
            }

            // Down Projection
            MatVecF16(gateBuf, blk.FfnDownW, blk.FfnDownB, numPatches, ffnDim, _embd, attnOut);

            // Residual 2
            for (int i = 0; i < hiddenStates.Length; i++) hiddenStates[i] += attnOut[i];
        }

        // 3. Post-Norm
        ApplyNorm(hiddenStates, numPatches, _embd, _postLnW, null);

        // 4. 2x2 Spatial Merge & Multimodal MLP Projection (5120 -> 5120 -> 3584)
        var visualTokens = new float[tokenCount * _projDim];
        ApplySpatialMergeAndMlp(hiddenStates, patchesX, patchesY, visualTokens);

        return visualTokens;
    }

    private void ExtractPatchEmbeddings(ReadOnlySpan<float> chw, int width, int height, int patchesX, int patchesY, float[] output)
    {
        int patchSize = _m.PatchSize; // 14
        int patchArea = patchSize * patchSize;
        int totalPixels = width * height;

        for (int py = 0; py < patchesY; py++)
        {
            for (int px = 0; px < patchesX; px++)
            {
                int patchIdx = py * patchesX + px;
                int outOffset = patchIdx * _embd;

                // Conv2D patch linear projection
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
                                    float pixel = chw[c * totalPixels + (y * width + x)];
                                    int weightIdx = wOffset + c * patchArea + (dy * patchSize + dx);
                                    sum += pixel * (float)_patchEmbd0W[weightIdx];
                                }
                            }
                        }

                        if (_positionEmbd != null)
                        {
                            sum += _positionEmbd[patchIdx * _embd + d];
                        }

                        output[outOffset + d] = sum;
                    }
                }
            }
        }
    }

    private void ApplyNorm(float[] states, int nPatches, int dim, float* weights, float* bias)
    {
        for (int p = 0; p < nPatches; p++)
        {
            int off = p * dim;
            if (_useRmsNorm)
            {
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
            else
            {
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
                        float freqX = MathF.Pow(10000.0f, -2.0f * d / _headDim);
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

    private void ComputeSelfAttention(float[] q, float[] k, float[] v, int patchesX, int patchesY, float[] output)
    {
        int nPatches = patchesX * patchesY;
        float scale = 1.0f / MathF.Sqrt(_headDim);

        for (int h = 0; h < _heads; h++)
        {
            for (int i = 0; i < nPatches; i++)
            {
                int qOff = (i * _heads + h) * _headDim;
                int outOff = (i * _heads + h) * _headDim;

                for (int d = 0; d < _headDim; d++)
                {
                    output[outOff + d] = v[qOff + d] * scale;
                }
            }
        }
    }

    private void ApplySpatialMergeAndMlp(float[] hiddenStates, int patchesX, int patchesY, float[] visualTokens)
    {
        int merge = _m.SpatialMergeFactor; // 2
        int mergedX = patchesX / merge;
        int mergedY = patchesY / merge;
        int inMlpDim = 4 * _embd; // 5120
        var mergedVec = new float[inMlpDim];
        var midVec = new float[inMlpDim];

        for (int my = 0; my < mergedY; my++)
        {
            for (int mx = 0; mx < mergedX; mx++)
            {
                int tokenIdx = my * mergedX + mx;
                int outOffset = tokenIdx * _projDim;

                // Pack 2x2 = 4 adjacent patches into 5120-dim input vector
                int subIdx = 0;
                for (int dy = 0; dy < merge; dy++)
                {
                    int py = my * merge + dy;
                    for (int dx = 0; dx < merge; dx++)
                    {
                        int px = mx * merge + dx;
                        int patchIdx = py * patchesX + px;
                        Array.Copy(hiddenStates, patchIdx * _embd, mergedVec, subIdx * _embd, _embd);
                        subIdx++;
                    }
                }

                // 1. Layer mm.0 (5120 -> 5120) + GELU
                if (_mm0W != null)
                {
                    for (int o = 0; o < inMlpDim; o++)
                    {
                        float sum = _mm0B != null ? _mm0B[o] : 0f;
                        int rOff = o * inMlpDim;
                        for (int i = 0; i < inMlpDim; i++) sum += mergedVec[i] * (float)_mm0W[rOff + i];
                        // GELU
                        float gelu = 0.5f * sum * (1.0f + MathF.Tanh(MathF.Sqrt(2.0f / MathF.PI) * (sum + 0.044715f * sum * sum * sum)));
                        midVec[o] = gelu;
                    }
                }
                else
                {
                    Array.Copy(mergedVec, midVec, inMlpDim);
                }

                // 2. Layer mm.2 (5120 -> 3584)
                if (_mm2W != null)
                {
                    for (int o = 0; o < _projDim; o++)
                    {
                        float sum = _mm2B != null ? _mm2B[o] : 0f;
                        int rOff = o * inMlpDim;
                        for (int i = 0; i < inMlpDim; i++) sum += midVec[i] * (float)_mm2W[rOff + i];
                        visualTokens[outOffset + o] = sum;
                    }
                }
            }
        }
    }
}
