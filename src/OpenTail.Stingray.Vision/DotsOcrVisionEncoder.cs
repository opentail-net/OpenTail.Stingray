using System;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Vision;

/// <summary>
/// Native C# Dots-OCR and PaddleOCR-VL Vision ViT Encoder + 2D M-RoPE + Patch Merger + GELU MLP Projector.
/// Reference: examples/llama.cpp/llama.cpp/tools/mtmd/models/dotsocr.cpp
/// </summary>
public sealed unsafe class DotsOcrVisionEncoder
{
    private readonly DotsOcrVisionModel _m;
    private readonly int _embd;
    private readonly int _heads;
    private readonly int _headDim;
    private readonly int _layers;
    private readonly int _projDim;
    private readonly int _mergeFactor;
    private readonly float _eps;

    private readonly float[] _patchEmbdWF32;
    private readonly float* _patchEmbdB;
    private readonly float[] _posEmbdF32;
    private readonly float* _preLnW;
    private readonly float* _postLnW;

    private readonly float* _inputNormW;
    private readonly float* _inputNormB;
    private readonly VisionTensorRef _mlp0W;
    private readonly float* _mlp0B;
    private readonly VisionTensorRef _mlp2W;
    private readonly float* _mlp2B;

    private readonly LayerWeights[] _blocks;

    private sealed class LayerWeights
    {
        public float* Ln1W;
        public float* Ln1B;
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

    public DotsOcrVisionEncoder(DotsOcrVisionModel model)
    {
        _m = model;
        _embd = model.EmbeddingDim;
        _heads = model.HeadCount;
        _headDim = model.HeadDim;
        _layers = model.LayerCount;
        _projDim = model.ProjectionDim;
        _mergeFactor = model.MergeFactor;
        _eps = model.Eps;

        var gguf = model.Gguf;

        _patchEmbdWF32 = VisionOps.DequantizeToFloat32(VisionOps.GetTensor(gguf, "v.patch_embd.weight"));
        _patchEmbdB = VisionOps.GetTensorPtr<float>(gguf, "v.patch_embd.bias");
        _posEmbdF32 = VisionOps.DequantizeToFloat32(VisionOps.GetTensor(gguf, "v.position_embd.weight", "v.position_embd"));
        _preLnW = VisionOps.GetTensorPtr<float>(gguf, "v.pre_ln.weight");
        _postLnW = VisionOps.GetTensorPtr<float>(gguf, "v.post_ln.weight");

        _inputNormW = VisionOps.GetTensorPtr<float>(gguf, "mm.input_norm.weight", "mm.0.weight");
        _inputNormB = VisionOps.GetTensorPtr<float>(gguf, "mm.input_norm.bias", "mm.0.bias");
        _mlp0W = VisionOps.GetTensor(gguf, "mm.1.weight", "mm.0.weight");
        _mlp0B = VisionOps.GetTensorPtr<float>(gguf, "mm.1.bias", "mm.0.bias");
        _mlp2W = VisionOps.GetTensor(gguf, "mm.2.weight", "mm.3.weight");
        _mlp2B = VisionOps.GetTensorPtr<float>(gguf, "mm.2.bias", "mm.3.bias");

        _blocks = new LayerWeights[_layers];
        for (int l = 0; l < _layers; l++)
        {
            var upTensor = gguf.FindTensor($"v.blk.{l}.ffn_up.weight");
            int intermediate = upTensor.HasValue ? (int)upTensor.Value.Dimensions[1] : (_embd * 4);

            _blocks[l] = new LayerWeights
            {
                Ln1W = VisionOps.GetTensorPtr<float>(gguf, $"v.blk.{l}.ln1.weight"),
                Ln1B = VisionOps.GetTensorPtr<float>(gguf, $"v.blk.{l}.ln1.bias"),
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
        var hiddenStates = new float[numPatches * _embd];

        fixed (float* chwPtr = chw)
        {
            ExtractPatches(chwPtr, targetWidth, targetHeight, patchesX, patchesY, hiddenStates);
        }

        if (_preLnW != null) VisionOps.RmsNorm(hiddenStates, numPatches, _embd, _preLnW, _eps);

        var qBuf = new float[numPatches * _embd];
        var kBuf = new float[numPatches * _embd];
        var vBuf = new float[numPatches * _embd];
        var attnOut = new float[numPatches * _embd];
        var normed = new float[numPatches * _embd];

        int maxIntermediate = 0;
        for (int l = 0; l < _layers; l++)
        {
            if (_blocks[l].FfnIntermediate > maxIntermediate) maxIntermediate = _blocks[l].FfnIntermediate;
        }
        var ffnMid = new float[numPatches * maxIntermediate];

        for (int l = 0; l < _layers; l++)
        {
            var blk = _blocks[l];

            Array.Copy(hiddenStates, normed, hiddenStates.Length);
            VisionOps.RmsNorm(normed, numPatches, _embd, blk.Ln1W, _eps);

            VisionOps.MatVecAny(normed, blk.AttnQW, blk.AttnQB, numPatches, _embd, _embd, qBuf);
            VisionOps.MatVecAny(normed, blk.AttnKW, blk.AttnKB, numPatches, _embd, _embd, kBuf);
            VisionOps.MatVecAny(normed, blk.AttnVW, blk.AttnVB, numPatches, _embd, _embd, vBuf);

            // 2D M-RoPE
            VisionOps.Interleaved2DRoPE(qBuf, kBuf, patchesX, patchesY, _heads, _headDim);

            VisionOps.Attention(qBuf, kBuf, vBuf, numPatches, _heads, _headDim, normed);
            VisionOps.MatVecAny(normed, blk.AttnOutW, blk.AttnOutB, numPatches, _embd, _embd, attnOut);

            for (int i = 0; i < hiddenStates.Length; i++) hiddenStates[i] += attnOut[i];

            Array.Copy(hiddenStates, normed, hiddenStates.Length);
            VisionOps.RmsNorm(normed, numPatches, _embd, blk.Ln2W, _eps);

            int intermediate = blk.FfnIntermediate;
            VisionOps.MatVecAny(normed, blk.FfnUpW, blk.FfnUpB, numPatches, _embd, intermediate, ffnMid);
            VisionOps.Gelu(ffnMid.AsSpan(0, numPatches * intermediate));
            VisionOps.MatVecAny(ffnMid, blk.FfnDownW, blk.FfnDownB, numPatches, intermediate, _embd, attnOut);

            for (int i = 0; i < hiddenStates.Length; i++) hiddenStates[i] += attnOut[i];
        }

        if (_postLnW != null) VisionOps.RmsNorm(hiddenStates, numPatches, _embd, _postLnW, _eps);

        // LayerNorm input norm
        if (_inputNormW != null)
        {
            VisionOps.LayerNorm(hiddenStates, numPatches, _embd, _inputNormW, _inputNormB, _eps);
        }

        // Patch merge 2x2
        int downH = patchesY / 2;
        int downW = patchesX / 2;
        tokenCount = downH * downW;
        int mergedDim = _embd * 4;

        var merged = new float[tokenCount * mergedDim];
        VisionOps.PixelShuffle2x2(hiddenStates, patchesY, patchesX, _embd, merged);

        // 2-layer GELU MLP Projector
        var visualTokens = new float[tokenCount * _projDim];
        if (_mlp0W.IsValid && _mlp2W.IsValid)
        {
            var midBuf = new float[tokenCount * mergedDim];
            VisionOps.MatVecAny(merged, _mlp0W, _mlp0B, tokenCount, mergedDim, mergedDim, midBuf);
            VisionOps.Gelu(midBuf);
            VisionOps.MatVecAny(midBuf, _mlp2W, _mlp2B, tokenCount, mergedDim, _projDim, visualTokens);
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

    private void ExtractPatches(float* chw, int width, int height, int patchesX, int patchesY, float[] output)
    {
        int patchSize = _m.PatchSize;
        int patchArea = patchSize * patchSize;
        int planeSize = width * height;

        Parallel.For(0, patchesY, py =>
        {
            for (int px = 0; px < patchesX; px++)
            {
                int patchIdx = py * patchesX + px;
                int outOffset = patchIdx * _embd;

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

                        if (_posEmbdF32.Length > 0 && patchIdx < 729)
                        {
                            sum += _posEmbdF32[patchIdx * _embd + d];
                        }

                        output[outOffset + d] = sum;
                    }
                }
            }
        });
    }
}
