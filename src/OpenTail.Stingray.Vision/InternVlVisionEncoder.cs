using System;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Vision;

/// <summary>
/// Native C# InternVL 2.5 / 3 / 4 Vision ViT Encoder + PixelShuffle + GELU MLP Projector.
/// Reference: examples/llama.cpp/llama.cpp/tools/mtmd/models/internvl.cpp
/// </summary>
public sealed unsafe class InternVlVisionEncoder
{
    private readonly InternVlVisionModel _m;
    private readonly int _embd;
    private readonly int _heads;
    private readonly int _headDim;
    private readonly int _layers;
    private readonly int _projDim;
    private readonly bool _useRmsNorm;
    private readonly float _eps;

    private readonly Half* _patchEmbdW;
    private readonly float* _patchEmbdB;
    private readonly float* _clsEmbd;
    private readonly float* _posEmbd;
    private readonly float* _preLnW;
    private readonly float* _preLnB;
    private readonly float* _postLnW;
    private readonly float* _postLnB;

    private readonly float* _mm0LnW;
    private readonly float* _mm0LnB;
    private readonly Half* _mlp1W;
    private readonly float* _mlp1B;
    private readonly Half* _mlp3W;
    private readonly float* _mlp3B;

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
        public Half* FfnGateW;
        public Half* FfnUpW;
        public float* FfnUpB;
        public Half* FfnDownW;
        public float* FfnDownB;
        public int FfnIntermediate;
    }

    public int EmbeddingDim => _embd;
    public int ProjectionDim => _projDim;

    public InternVlVisionEncoder(InternVlVisionModel model)
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

        _patchEmbdW = VisionOps.GetTensorPtr<Half>(gguf, "v.patch_embd.weight");
        _patchEmbdB = VisionOps.GetTensorPtr<float>(gguf, "v.patch_embd.bias");
        _clsEmbd = VisionOps.GetTensorPtr<float>(gguf, "v.class_embd", "v.cls_embd");
        _posEmbd = VisionOps.GetTensorPtr<float>(gguf, "v.position_embd.weight", "v.position_embd");
        _preLnW = VisionOps.GetTensorPtr<float>(gguf, "v.pre_ln.weight");
        _preLnB = VisionOps.GetTensorPtr<float>(gguf, "v.pre_ln.bias");
        _postLnW = VisionOps.GetTensorPtr<float>(gguf, "v.post_ln.weight");
        _postLnB = VisionOps.GetTensorPtr<float>(gguf, "v.post_ln.bias");

        _mm0LnW = VisionOps.GetTensorPtr<float>(gguf, "mm.0.weight");
        _mm0LnB = VisionOps.GetTensorPtr<float>(gguf, "mm.0.bias");
        _mlp1W = VisionOps.GetTensorPtr<Half>(gguf, "mm.1.weight");
        _mlp1B = VisionOps.GetTensorPtr<float>(gguf, "mm.1.bias");
        _mlp3W = VisionOps.GetTensorPtr<Half>(gguf, "mm.3.weight");
        _mlp3B = VisionOps.GetTensorPtr<float>(gguf, "mm.3.bias");

        _blocks = new LayerWeights[_layers];
        for (int l = 0; l < _layers; l++)
        {
            var upTensor = gguf.FindTensor($"v.blk.{l}.ffn_up.weight");
            int intermediate = upTensor.HasValue ? (int)upTensor.Value.Dimensions[1] : (_embd * 4);

            _blocks[l] = new LayerWeights
            {
                Ln1W = VisionOps.GetTensorPtr<float>(gguf, $"v.blk.{l}.ln1.weight"),
                Ln1B = VisionOps.GetTensorPtr<float>(gguf, $"v.blk.{l}.ln1.bias"),
                AttnQkvW = VisionOps.GetTensorPtr<Half>(gguf, $"v.blk.{l}.attn_qkv.weight"),
                AttnQkvB = VisionOps.GetTensorPtr<float>(gguf, $"v.blk.{l}.attn_qkv.bias"),
                AttnQW = VisionOps.GetTensorPtr<Half>(gguf, $"v.blk.{l}.attn_q.weight"),
                AttnQB = VisionOps.GetTensorPtr<float>(gguf, $"v.blk.{l}.attn_q.bias"),
                AttnKW = VisionOps.GetTensorPtr<Half>(gguf, $"v.blk.{l}.attn_k.weight"),
                AttnKB = VisionOps.GetTensorPtr<float>(gguf, $"v.blk.{l}.attn_k.bias"),
                AttnVW = VisionOps.GetTensorPtr<Half>(gguf, $"v.blk.{l}.attn_v.weight"),
                AttnVB = VisionOps.GetTensorPtr<float>(gguf, $"v.blk.{l}.attn_v.bias"),
                AttnOutW = VisionOps.GetTensorPtr<Half>(gguf, $"v.blk.{l}.attn_out.weight"),
                AttnOutB = VisionOps.GetTensorPtr<float>(gguf, $"v.blk.{l}.attn_out.bias"),
                Ln2W = VisionOps.GetTensorPtr<float>(gguf, $"v.blk.{l}.ln2.weight"),
                Ln2B = VisionOps.GetTensorPtr<float>(gguf, $"v.blk.{l}.ln2.bias"),
                FfnGateW = VisionOps.GetTensorPtr<Half>(gguf, $"v.blk.{l}.ffn_gate.weight"),
                FfnUpW = VisionOps.GetTensorPtr<Half>(gguf, $"v.blk.{l}.ffn_up.weight"),
                FfnUpB = VisionOps.GetTensorPtr<float>(gguf, $"v.blk.{l}.ffn_up.bias"),
                FfnDownW = VisionOps.GetTensorPtr<Half>(gguf, $"v.blk.{l}.ffn_down.weight"),
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

        if (_preLnW != null)
        {
            if (_useRmsNorm) VisionOps.RmsNorm(hiddenStates, totalTokensIn, _embd, _preLnW, _eps);
            else VisionOps.LayerNorm(hiddenStates, totalTokensIn, _embd, _preLnW, _preLnB, _eps);
        }

        var qBuf = new float[totalTokensIn * _embd];
        var kBuf = new float[totalTokensIn * _embd];
        var vBuf = new float[totalTokensIn * _embd];
        var attnOut = new float[totalTokensIn * _embd];
        var normed = new float[totalTokensIn * _embd];

        for (int l = 0; l < _layers; l++)
        {
            var blk = _blocks[l];

            Array.Copy(hiddenStates, normed, hiddenStates.Length);
            if (_useRmsNorm) VisionOps.RmsNorm(normed, totalTokensIn, _embd, blk.Ln1W, _eps);
            else VisionOps.LayerNorm(normed, totalTokensIn, _embd, blk.Ln1W, blk.Ln1B, _eps);

            if (blk.AttnQkvW != null)
            {
                var qkv = new float[totalTokensIn * 3 * _embd];
                VisionOps.MatVecF16(normed, blk.AttnQkvW, blk.AttnQkvB, totalTokensIn, _embd, 3 * _embd, qkv);
                for (int p = 0; p < totalTokensIn; p++)
                {
                    Array.Copy(qkv, p * 3 * _embd, qBuf, p * _embd, _embd);
                    Array.Copy(qkv, p * 3 * _embd + _embd, kBuf, p * _embd, _embd);
                    Array.Copy(qkv, p * 3 * _embd + 2 * _embd, vBuf, p * _embd, _embd);
                }
            }
            else
            {
                VisionOps.MatVecF16(normed, blk.AttnQW, blk.AttnQB, totalTokensIn, _embd, _embd, qBuf);
                VisionOps.MatVecF16(normed, blk.AttnKW, blk.AttnKB, totalTokensIn, _embd, _embd, kBuf);
                VisionOps.MatVecF16(normed, blk.AttnVW, blk.AttnVB, totalTokensIn, _embd, _embd, vBuf);
            }

            VisionOps.Attention(qBuf, kBuf, vBuf, totalTokensIn, _heads, _headDim, normed);
            VisionOps.MatVecF16(normed, blk.AttnOutW, blk.AttnOutB, totalTokensIn, _embd, _embd, attnOut);

            for (int i = 0; i < hiddenStates.Length; i++) hiddenStates[i] += attnOut[i];

            Array.Copy(hiddenStates, normed, hiddenStates.Length);
            if (_useRmsNorm) VisionOps.RmsNorm(normed, totalTokensIn, _embd, blk.Ln2W, _eps);
            else VisionOps.LayerNorm(normed, totalTokensIn, _embd, blk.Ln2W, blk.Ln2B, _eps);

            int intermediate = blk.FfnIntermediate;
            var ffnMid = new float[totalTokensIn * intermediate];

            if (blk.FfnGateW != null)
            {
                var gateBuf = new float[totalTokensIn * intermediate];
                VisionOps.MatVecF16(normed, blk.FfnGateW, null, totalTokensIn, _embd, intermediate, gateBuf);
                VisionOps.MatVecF16(normed, blk.FfnUpW, blk.FfnUpB, totalTokensIn, _embd, intermediate, ffnMid);
                for (int i = 0; i < ffnMid.Length; i++)
                {
                    float g = gateBuf[i];
                    float silu = g / (1.0f + MathF.Exp(-g));
                    ffnMid[i] = silu * ffnMid[i];
                }
            }
            else
            {
                VisionOps.MatVecF16(normed, blk.FfnUpW, blk.FfnUpB, totalTokensIn, _embd, intermediate, ffnMid);
                VisionOps.Gelu(ffnMid);
            }

            VisionOps.MatVecF16(ffnMid, blk.FfnDownW, blk.FfnDownB, totalTokensIn, intermediate, _embd, attnOut);

            for (int i = 0; i < hiddenStates.Length; i++) hiddenStates[i] += attnOut[i];
        }

        if (_postLnW != null)
        {
            if (_useRmsNorm) VisionOps.RmsNorm(hiddenStates, totalTokensIn, _embd, _postLnW, _eps);
            else VisionOps.LayerNorm(hiddenStates, totalTokensIn, _embd, _postLnW, _postLnB, _eps);
        }

        // Strip CLS token (token 0)
        var patchEmbeddings = new float[numPatches * _embd];
        Array.Copy(hiddenStates, _embd, patchEmbeddings, 0, numPatches * _embd);

        // PixelShuffle 2x2: (H/2) * (W/2) tokens with 4 * _embd dim
        int downH = patchesY / 2;
        int downW = patchesX / 2;
        tokenCount = downH * downW;
        int shuffledDim = _embd * 4;

        var shuffled = new float[tokenCount * shuffledDim];
        VisionOps.PixelShuffle2x2(patchEmbeddings, patchesY, patchesX, _embd, shuffled);

        // Projector: LayerNorm (mm.0) + MLP1 (mm.1) + GELU + MLP3 (mm.3)
        var visualTokens = new float[tokenCount * _projDim];
        if (_mlp1W != null && _mlp3W != null)
        {
            if (_mm0LnW != null) VisionOps.LayerNorm(shuffled, tokenCount, shuffledDim, _mm0LnW, _mm0LnB, _eps);

            var midBuf = new float[tokenCount * _projDim];
            VisionOps.MatVecF16(shuffled, _mlp1W, _mlp1B, tokenCount, shuffledDim, _projDim, midBuf);
            VisionOps.Gelu(midBuf);
            VisionOps.MatVecF16(midBuf, _mlp3W, _mlp3B, tokenCount, _projDim, _projDim, visualTokens);
        }
        else
        {
            for (int t = 0; t < tokenCount; t++)
            {
                int copyDim = Math.Min(shuffledDim, _projDim);
                Array.Copy(shuffled, t * shuffledDim, visualTokens, t * _projDim, copyDim);
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
        if (_posEmbd != null)
        {
            for (int d = 0; d < _embd; d++) output[d] += _posEmbd[d];
        }

        Parallel.For(0, patchesY, py =>
        {
            for (int px = 0; px < patchesX; px++)
            {
                int patchIdx = py * patchesX + px;
                int tokenIdx = patchIdx + 1;
                int outOffset = tokenIdx * _embd;

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
                            sum += _posEmbd[tokenIdx * _embd + d];
                        }

                        output[outOffset + d] = sum;
                    }
                }
            }
        });
    }
}
