using System;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Vision;

/// <summary>
/// Native C# LLaVA-1.5 / LLaVA-NeXT / LLaVA-OneVision Vision ViT Encoder + 2-layer GELU MLP Projector.
/// Reference: examples/llama.cpp/llama.cpp/tools/mtmd/models/llava.cpp
/// </summary>
public sealed unsafe class LlavaVisionEncoder
{
    private readonly LlavaVisionModel _m;
    private readonly int _embd;
    private readonly int _heads;
    private readonly int _headDim;
    private readonly int _layers;
    private readonly int _projDim;
    private readonly float _eps;

    private readonly float[] _patchEmbdWF32;
    private readonly float* _patchEmbdB;
    private readonly float* _clsEmbd;
    private readonly float[] _posEmbdF32;
    private readonly float* _preLnW;
    private readonly float* _preLnB;
    private readonly float* _postLnW;
    private readonly float* _postLnB;

    private readonly VisionTensorRef _mlp0W;
    private readonly float* _mlp0B;
    private readonly VisionTensorRef _mlp2W;
    private readonly float* _mlp2B;

    private readonly LayerWeights[] _blocks;

    private sealed class LayerWeights
    {
        public float* Ln1W;
        public float* Ln1B;
        public VisionTensorRef AttnQkvW;
        public float* AttnQkvB;
        public VisionTensorRef AttnQW;
        public float* AttnQB;
        public VisionTensorRef AttnKW;
        public float* AttnKB;
        public VisionTensorRef AttnVW;
        public float* AttnVB;
        public VisionTensorRef AttnOutW;
        public float* AttnOutB;
        public float* Ln2W;
        public float* Ln2B;
        public VisionTensorRef FfnUpW;
        public float* FfnUpB;
        public VisionTensorRef FfnDownW;
        public float* FfnDownB;
        public int FfnIntermediate;
    }

    public int EmbeddingDim => _embd;
    public int ProjectionDim => _projDim;

    public LlavaVisionEncoder(LlavaVisionModel model)
    {
        _m = model;
        _embd = model.EmbeddingDim;
        _heads = model.HeadCount;
        _headDim = model.HeadDim;
        _layers = model.LayerCount;
        _projDim = model.ProjectionDim;
        _eps = model.Eps;

        var gguf = model.Gguf;

        _patchEmbdWF32 = VisionOps.DequantizeToFloat32(VisionOps.GetTensor(gguf, "v.patch_embd.weight"));
        _patchEmbdB = VisionOps.GetTensorPtr<float>(gguf, "v.patch_embd.bias");
        _clsEmbd = VisionOps.GetTensorPtr<float>(gguf, "v.class_embd", "v.cls_embd");
        _posEmbdF32 = VisionOps.DequantizeToFloat32(VisionOps.GetTensor(gguf, "v.position_embd.weight", "v.position_embd"));
        _preLnW = VisionOps.GetTensorPtr<float>(gguf, "v.pre_ln.weight");
        _preLnB = VisionOps.GetTensorPtr<float>(gguf, "v.pre_ln.bias");
        _postLnW = VisionOps.GetTensorPtr<float>(gguf, "v.post_ln.weight");
        _postLnB = VisionOps.GetTensorPtr<float>(gguf, "v.post_ln.bias");

        _mlp0W = VisionOps.GetTensor(gguf, "mm.0.weight");
        _mlp0B = VisionOps.GetTensorPtr<float>(gguf, "mm.0.bias");
        _mlp2W = VisionOps.GetTensor(gguf, "mm.2.weight");
        _mlp2B = VisionOps.GetTensorPtr<float>(gguf, "mm.2.bias");

        _blocks = new LayerWeights[_layers];
        for (int l = 0; l < _layers; l++)
        {
            var upTensor = gguf.FindTensor($"v.blk.{l}.ffn_up.weight");
            int intermediate = upTensor.HasValue ? (int)upTensor.Value.Dimensions[1] : (_embd * 4);

            _blocks[l] = new LayerWeights
            {
                Ln1W = VisionOps.GetTensorPtr<float>(gguf, $"v.blk.{l}.ln1.weight"),
                Ln1B = VisionOps.GetTensorPtr<float>(gguf, $"v.blk.{l}.ln1.bias"),
                AttnQkvW = VisionOps.GetTensor(gguf, $"v.blk.{l}.attn_qkv.weight"),
                AttnQkvB = VisionOps.GetTensorPtr<float>(gguf, $"v.blk.{l}.attn_qkv.bias"),
                AttnQW = VisionOps.GetTensor(gguf, $"v.blk.{l}.attn_q.weight"),
                AttnQB = VisionOps.GetTensorPtr<float>(gguf, $"v.blk.{l}.attn_q.bias"),
                AttnKW = VisionOps.GetTensor(gguf, $"v.blk.{l}.attn_k.weight"),
                AttnKB = VisionOps.GetTensorPtr<float>(gguf, $"v.blk.{l}.attn_k.bias"),
                AttnVW = VisionOps.GetTensor(gguf, $"v.blk.{l}.attn_v.weight"),
                AttnVB = VisionOps.GetTensorPtr<float>(gguf, $"v.blk.{l}.attn_v.bias"),
                AttnOutW = VisionOps.GetTensor(gguf, $"v.blk.{l}.attn_out.weight"),
                AttnOutB = VisionOps.GetTensorPtr<float>(gguf, $"v.blk.{l}.attn_out.bias"),
                Ln2W = VisionOps.GetTensorPtr<float>(gguf, $"v.blk.{l}.ln2.weight"),
                Ln2B = VisionOps.GetTensorPtr<float>(gguf, $"v.blk.{l}.ln2.bias"),
                FfnUpW = VisionOps.GetTensor(gguf, $"v.blk.{l}.ffn_up.weight"),
                FfnUpB = VisionOps.GetTensorPtr<float>(gguf, $"v.blk.{l}.ffn_up.bias"),
                FfnDownW = VisionOps.GetTensor(gguf, $"v.blk.{l}.ffn_down.weight"),
                FfnDownB = VisionOps.GetTensorPtr<float>(gguf, $"v.blk.{l}.ffn_down.bias"),
                FfnIntermediate = intermediate
            };
        }
    }

    public float[] Forward(ReadOnlySpan<float> chw, int targetWidth, int targetHeight, int patchesX, int patchesY, out int tokenCount)
    {
        int numPatches = patchesX * patchesY;
        int totalTokensIn = numPatches + 1; // + CLS

        var hiddenStates = new float[totalTokensIn * _embd];
        fixed (float* chwPtr = chw)
        {
            ExtractPatchesWithCls(chwPtr, targetWidth, targetHeight, patchesX, patchesY, hiddenStates);
        }

        if (_preLnW != null) VisionOps.LayerNorm(hiddenStates, totalTokensIn, _embd, _preLnW, _preLnB, _eps);

        var qBuf = new float[totalTokensIn * _embd];
        var kBuf = new float[totalTokensIn * _embd];
        var vBuf = new float[totalTokensIn * _embd];
        var attnOut = new float[totalTokensIn * _embd];
        var normed = new float[totalTokensIn * _embd];
        var qkv = new float[totalTokensIn * 3 * _embd];

        int maxIntermediate = 0;
        for (int l = 0; l < _layers; l++)
        {
            if (_blocks[l].FfnIntermediate > maxIntermediate) maxIntermediate = _blocks[l].FfnIntermediate;
        }
        var ffnMid = new float[totalTokensIn * maxIntermediate];

        for (int l = 0; l < _layers; l++)
        {
            var blk = _blocks[l];

            Array.Copy(hiddenStates, normed, hiddenStates.Length);
            VisionOps.LayerNorm(normed, totalTokensIn, _embd, blk.Ln1W, blk.Ln1B, _eps);

            if (blk.AttnQkvW.IsValid)
            {
                VisionOps.MatVecAny(normed, blk.AttnQkvW, blk.AttnQkvB, totalTokensIn, _embd, 3 * _embd, qkv);
                for (int p = 0; p < totalTokensIn; p++)
                {
                    Array.Copy(qkv, p * 3 * _embd, qBuf, p * _embd, _embd);
                    Array.Copy(qkv, p * 3 * _embd + _embd, kBuf, p * _embd, _embd);
                    Array.Copy(qkv, p * 3 * _embd + 2 * _embd, vBuf, p * _embd, _embd);
                }
            }
            else
            {
                VisionOps.MatVecAny(normed, blk.AttnQW, blk.AttnQB, totalTokensIn, _embd, _embd, qBuf);
                VisionOps.MatVecAny(normed, blk.AttnKW, blk.AttnKB, totalTokensIn, _embd, _embd, kBuf);
                VisionOps.MatVecAny(normed, blk.AttnVW, blk.AttnVB, totalTokensIn, _embd, _embd, vBuf);
            }

            VisionOps.Attention(qBuf, kBuf, vBuf, totalTokensIn, _heads, _headDim, normed);
            VisionOps.MatVecAny(normed, blk.AttnOutW, blk.AttnOutB, totalTokensIn, _embd, _embd, attnOut);

            for (int i = 0; i < hiddenStates.Length; i++) hiddenStates[i] += attnOut[i];

            Array.Copy(hiddenStates, normed, hiddenStates.Length);
            VisionOps.LayerNorm(normed, totalTokensIn, _embd, blk.Ln2W, blk.Ln2B, _eps);

            int intermediate = blk.FfnIntermediate;
            VisionOps.MatVecAny(normed, blk.FfnUpW, blk.FfnUpB, totalTokensIn, _embd, intermediate, ffnMid);
            VisionOps.QuickGelu(ffnMid.AsSpan(0, totalTokensIn * intermediate));
            VisionOps.MatVecAny(ffnMid, blk.FfnDownW, blk.FfnDownB, totalTokensIn, intermediate, _embd, attnOut);

            for (int i = 0; i < hiddenStates.Length; i++) hiddenStates[i] += attnOut[i];
        }

        if (_postLnW != null) VisionOps.LayerNorm(hiddenStates, totalTokensIn, _embd, _postLnW, _postLnB, _eps);

        // Strip CLS token
        tokenCount = numPatches;
        var patchEmbeddings = new float[numPatches * _embd];
        Array.Copy(hiddenStates, _embd, patchEmbeddings, 0, numPatches * _embd);

        // 2-layer GELU MLP Projector
        var visualTokens = new float[tokenCount * _projDim];
        if (_mlp0W.IsValid && _mlp2W.IsValid)
        {
            var midBuf = new float[tokenCount * _projDim];
            VisionOps.MatVecAny(patchEmbeddings, _mlp0W, _mlp0B, tokenCount, _embd, _projDim, midBuf);
            VisionOps.Gelu(midBuf);
            VisionOps.MatVecAny(midBuf, _mlp2W, _mlp2B, tokenCount, _projDim, _projDim, visualTokens);
        }
        else
        {
            for (int t = 0; t < tokenCount; t++)
            {
                int copyDim = Math.Min(_embd, _projDim);
                Array.Copy(patchEmbeddings, t * _embd, visualTokens, t * _projDim, copyDim);
            }
        }

        return visualTokens;
    }

    private void ExtractPatchesWithCls(float* chw, int width, int height, int patchesX, int patchesY, float[] output)
    {
        int patchSize = _m.PatchSize;
        int patchArea = patchSize * patchSize;
        int planeSize = width * height;

        if (_clsEmbd != null)
        {
            for (int d = 0; d < _embd; d++) output[d] = _clsEmbd[d];
        }
        if (_posEmbdF32.Length > 0)
        {
            for (int d = 0; d < _embd; d++) output[d] += _posEmbdF32[d];
        }

        Parallel.For(0, patchesY, py =>
        {
            for (int px = 0; px < patchesX; px++)
            {
                int patchIdx = py * patchesX + px;
                int tokenIdx = patchIdx + 1;
                int outOffset = tokenIdx * _embd;

                if (_patchEmbdWF32.Length > 0)
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
                                    sum += pixel * _patchEmbdWF32[weightIdx];
                                }
                            }
                        }

                        if (_posEmbdF32.Length > 0)
                        {
                            sum += _posEmbdF32[tokenIdx * _embd + d];
                        }

                        output[outOffset + d] = sum;
                    }
                }
            }
        });
    }
}
