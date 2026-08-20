using System;
using System.Threading.Tasks;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Vision;

/// <summary>
/// Native C# CogVLM and CogAgent Post-Norm ViT Vision Encoder + SwiGLU MLP Projector.
/// Reference: examples/llama.cpp/llama.cpp/tools/mtmd/models/cogvlm.cpp
/// </summary>
public sealed unsafe class CogVlmVisionEncoder
{
    private readonly CogVlmVisionModel _m;
    private readonly int _embd;
    private readonly int _heads;
    private readonly int _headDim;
    private readonly int _layers;
    private readonly int _projDim;
    private readonly float _eps;

    private readonly float[] _patchEmbdWF32;
    private readonly float* _patchEmbdB;
    private readonly float[] _clsEmbdF32;
    private readonly float[] _posEmbdF32;

    private readonly VisionTensorRef _modelProjW;
    private readonly float* _modelProjB;
    private readonly float* _postFcNormW;
    private readonly float* _postFcNormB;
    private readonly VisionTensorRef _hTo4hW;
    private readonly float* _hTo4hB;
    private readonly VisionTensorRef _gateW;
    private readonly float* _gateB;
    private readonly VisionTensorRef _toHW;
    private readonly float* _toHB;

    private readonly LayerWeights[] _blocks;

    private sealed class LayerWeights
    {
        public VisionTensorRef QkvW;
        public float* QkvB;
        public VisionTensorRef AttnOutW;
        public float* AttnOutB;
        public float* Ln1W;
        public float* Ln1B;
        public VisionTensorRef FfnUpW;
        public float* FfnUpB;
        public VisionTensorRef FfnGateW;
        public float* FfnGateB;
        public VisionTensorRef FfnDownW;
        public float* FfnDownB;
        public float* Ln2W;
        public float* Ln2B;
        public int FfnIntermediate;
    }

    public int EmbeddingDim => _embd;
    public int ProjectionDim => _projDim;

    public CogVlmVisionEncoder(CogVlmVisionModel model)
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
        _clsEmbdF32 = VisionOps.DequantizeToFloat32(VisionOps.GetTensor(gguf, "v.class_embd.weight", "v.class_embd"));
        _posEmbdF32 = VisionOps.DequantizeToFloat32(VisionOps.GetTensor(gguf, "v.position_embd.weight", "v.position_embd"));

        _modelProjW = VisionOps.GetTensor(gguf, "mm.model_proj.weight", "mm.0.weight");
        _modelProjB = VisionOps.GetTensorPtr<float>(gguf, "mm.model_proj.bias", "mm.0.bias");
        _postFcNormW = VisionOps.GetTensorPtr<float>(gguf, "mm.post_fc_norm.weight");
        _postFcNormB = VisionOps.GetTensorPtr<float>(gguf, "mm.post_fc_norm.bias");

        _hTo4hW = VisionOps.GetTensor(gguf, "mm.h_to_4h.weight", "mm.1.weight");
        _hTo4hB = VisionOps.GetTensorPtr<float>(gguf, "mm.h_to_4h.bias", "mm.1.bias");
        _gateW = VisionOps.GetTensor(gguf, "mm.gate.weight", "mm.2.weight");
        _gateB = VisionOps.GetTensorPtr<float>(gguf, "mm.gate.bias", "mm.2.bias");
        _toHW = VisionOps.GetTensor(gguf, "mm.4h_to_h.weight", "mm.3.weight");
        _toHB = VisionOps.GetTensorPtr<float>(gguf, "mm.4h_to_h.bias", "mm.3.bias");

        _blocks = new LayerWeights[_layers];
        for (int l = 0; l < _layers; l++)
        {
            var upTensor = gguf.FindTensor($"v.blk.{l}.ffn_up.weight");
            int intermediate = upTensor.HasValue ? (int)upTensor.Value.Dimensions[1] : (_embd * 4);

            _blocks[l] = new LayerWeights
            {
                QkvW = VisionOps.GetTensor(gguf, $"v.blk.{l}.qkv.weight"),
                QkvB = VisionOps.GetTensorPtr<float>(gguf, $"v.blk.{l}.qkv.bias"),
                AttnOutW = VisionOps.GetTensor(gguf, $"v.blk.{l}.attn_out.weight"),
                AttnOutB = VisionOps.GetTensorPtr<float>(gguf, $"v.blk.{l}.attn_out.bias"),
                Ln1W = VisionOps.GetTensorPtr<float>(gguf, $"v.blk.{l}.ln1.weight"),
                Ln1B = VisionOps.GetTensorPtr<float>(gguf, $"v.blk.{l}.ln1.bias"),
                FfnUpW = VisionOps.GetTensor(gguf, $"v.blk.{l}.ffn_up.weight"),
                FfnUpB = VisionOps.GetTensorPtr<float>(gguf, $"v.blk.{l}.ffn_up.bias"),
                FfnGateW = VisionOps.GetTensor(gguf, $"v.blk.{l}.ffn_gate.weight"),
                FfnGateB = VisionOps.GetTensorPtr<float>(gguf, $"v.blk.{l}.ffn_gate.bias"),
                FfnDownW = VisionOps.GetTensor(gguf, $"v.blk.{l}.ffn_down.weight"),
                FfnDownB = VisionOps.GetTensorPtr<float>(gguf, $"v.blk.{l}.ffn_down.bias"),
                Ln2W = VisionOps.GetTensorPtr<float>(gguf, $"v.blk.{l}.ln2.weight"),
                Ln2B = VisionOps.GetTensorPtr<float>(gguf, $"v.blk.{l}.ln2.bias"),
                FfnIntermediate = intermediate
            };
        }
    }

    public float[] Forward(float[] chw, int targetWidth, int targetHeight, int patchesX, int patchesY, out int tokenCount)
    {
        var img = new CogVlmPreprocessedImage(chw, targetWidth, targetHeight, patchesX, patchesY);
        var result = Encode(img);
        tokenCount = patchesX * patchesY;
        return result;
    }

    public float[] Encode(CogVlmPreprocessedImage img)
    {
        int numPatches = img.PatchesX * img.PatchesY;
        int totalTokens = numPatches + 1; // +1 for [CLS] token

        var hiddenStates = new float[totalTokens * _embd];

        // Class embedding token at index 0
        if (_clsEmbdF32.Length > 0)
        {
            for (int d = 0; d < _embd; d++) hiddenStates[d] = _clsEmbdF32[d];
        }

        fixed (float* chwPtr = img.Chw)
        {
            ExtractPatches(chwPtr, img.TargetWidth, img.TargetHeight, img.PatchesX, img.PatchesY, hiddenStates, 1);
        }

        if (_posEmbdF32.Length > 0)
        {
            for (int i = 0; i < hiddenStates.Length; i++) hiddenStates[i] += _posEmbdF32[i % _embd];
        }

        var qkvBuf = new float[totalTokens * _embd * 3];
        var qBuf = new float[totalTokens * _embd];
        var kBuf = new float[totalTokens * _embd];
        var vBuf = new float[totalTokens * _embd];
        var attnOut = new float[totalTokens * _embd];
        var normBuf = new float[totalTokens * _embd];

        int maxIntermediate = 0;
        for (int l = 0; l < _layers; l++)
        {
            if (_blocks[l].FfnIntermediate > maxIntermediate) maxIntermediate = _blocks[l].FfnIntermediate;
        }
        var ffnMid = new float[totalTokens * maxIntermediate];
        var gateBuf = new float[totalTokens * maxIntermediate];

        for (int l = 0; l < _layers; l++)
        {
            var blk = _blocks[l];

            // 1. QKV Self-Attention
            VisionOps.MatVecAny(hiddenStates, blk.QkvW, blk.QkvB, totalTokens, _embd, _embd * 3, qkvBuf);

            // Split QKV
            Parallel.For(0, totalTokens, t =>
            {
                int srcOff = t * (_embd * 3);
                int dstOff = t * _embd;
                Array.Copy(qkvBuf, srcOff, qBuf, dstOff, _embd);
                Array.Copy(qkvBuf, srcOff + _embd, kBuf, dstOff, _embd);
                Array.Copy(qkvBuf, srcOff + _embd * 2, vBuf, dstOff, _embd);
            });

            VisionOps.Attention(qBuf, kBuf, vBuf, totalTokens, _heads, _headDim, normBuf);
            VisionOps.MatVecAny(normBuf, blk.AttnOutW, blk.AttnOutB, totalTokens, _embd, _embd, attnOut);

            // Post-LN1
            VisionOps.LayerNorm(attnOut, totalTokens, _embd, blk.Ln1W, blk.Ln1B, _eps);
            for (int i = 0; i < hiddenStates.Length; i++) hiddenStates[i] += attnOut[i];

            // 2. FFN
            int intermediate = blk.FfnIntermediate;
            int midCount = totalTokens * intermediate;
            VisionOps.MatVecAny(hiddenStates, blk.FfnUpW, blk.FfnUpB, totalTokens, _embd, intermediate, ffnMid);

            if (blk.FfnGateW.IsValid)
            {
                VisionOps.MatVecAny(hiddenStates, blk.FfnGateW, blk.FfnGateB, totalTokens, _embd, intermediate, gateBuf);
                VisionOps.Silu(gateBuf.AsSpan(0, midCount));
                for (int i = 0; i < midCount; i++) ffnMid[i] *= gateBuf[i];
            }
            else
            {
                VisionOps.Gelu(ffnMid.AsSpan(0, midCount));
            }

            VisionOps.MatVecAny(ffnMid, blk.FfnDownW, blk.FfnDownB, totalTokens, intermediate, _embd, attnOut);

            // Post-LN2
            VisionOps.LayerNorm(attnOut, totalTokens, _embd, blk.Ln2W, blk.Ln2B, _eps);
            for (int i = 0; i < hiddenStates.Length; i++) hiddenStates[i] += attnOut[i];
        }

        // Drop [CLS] token and project remaining numPatches tokens
        var patchOnly = new float[numPatches * _embd];
        Array.Copy(hiddenStates, _embd, patchOnly, 0, numPatches * _embd);

        var projOut = new float[numPatches * _projDim];
        if (_modelProjW.IsValid)
        {
            VisionOps.MatVecAny(patchOnly, _modelProjW, _modelProjB, numPatches, _embd, _projDim, projOut);
            if (_postFcNormW != null)
            {
                VisionOps.LayerNorm(projOut, numPatches, _projDim, _postFcNormW, _postFcNormB, 1e-5f);
            }
            VisionOps.Gelu(projOut);

            // SwiGLU MLP
            if (_gateW.IsValid && _hTo4hW.IsValid && _toHW.IsValid)
            {
                int midDim = _projDim * 2;
                var h4h = new float[numPatches * midDim];
                var gate = new float[numPatches * midDim];
                VisionOps.MatVecAny(projOut, _hTo4hW, _hTo4hB, numPatches, _projDim, midDim, h4h);
                VisionOps.MatVecAny(projOut, _gateW, _gateB, numPatches, _projDim, midDim, gate);
                VisionOps.Silu(gate);
                for (int i = 0; i < h4h.Length; i++) h4h[i] *= gate[i];

                VisionOps.MatVecAny(h4h, _toHW, _toHB, numPatches, midDim, _projDim, projOut);
            }
        }
        else
        {
            for (int t = 0; t < numPatches; t++)
            {
                int copyDim = Math.Min(_embd, _projDim);
                Array.Copy(patchOnly, t * _embd, projOut, t * _projDim, copyDim);
            }
        }

        return projOut;
    }

    private void ExtractPatches(float* chw, int width, int height, int patchesX, int patchesY, float[] output, int tokenOffset)
    {
        int patchSize = _m.PatchSize;
        int patchArea = patchSize * patchSize;
        int planeSize = width * height;

        Parallel.For(0, patchesY, py =>
        {
            for (int px = 0; px < patchesX; px++)
            {
                int patchIdx = py * patchesX + px;
                int outOffset = (tokenOffset + patchIdx) * _embd;

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

                        output[outOffset + d] = sum;
                    }
                }
            }
        });
    }
}
