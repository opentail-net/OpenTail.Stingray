using System;
using System.Threading.Tasks;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Vision;

/// <summary>
/// Native C# IBM Granite 4 Vision SigLIP Vision Tower + WindowQFormer Spatial Projector.
/// Reference: examples/llama.cpp/llama.cpp/tools/mtmd/models/granite4-vision.cpp
/// </summary>
public sealed unsafe class Granite4VisionEncoder
{
    private readonly Granite4VisionModel _m;
    private readonly int _embd;
    private readonly int _heads;
    private readonly int _headDim;
    private readonly int _layers;
    private readonly int _projDim;
    private readonly float _eps;

    private readonly float[] _patchEmbdWF32;
    private readonly float[]? _patchEmbdB;
    private readonly float[] _posEmbdF32;
    private readonly float[]? _postLnW;
    private readonly float[]? _postLnB;

    private readonly float[]? _projNormW;
    private readonly float[]? _projNormB;
    private readonly VisionTensorRef _projW;
    private readonly float[]? _projB;

    private readonly LayerWeights[] _blocks;

    private sealed class LayerWeights
    {
        public float[]? Ln1W;
        public float[]? Ln1B;
        public VisionTensorRef AttnQW;
        public float[]? AttnQB;
        public VisionTensorRef AttnKW;
        public float[]? AttnKB;
        public VisionTensorRef AttnVW;
        public float[]? AttnVB;
        public VisionTensorRef AttnOutW;
        public float[]? AttnOutB;
        public float[]? Ln2W;
        public float[]? Ln2B;
        public VisionTensorRef FfnUpW;
        public float[]? FfnUpB;
        public VisionTensorRef FfnDownW;
        public float[]? FfnDownB;
        public int FfnIntermediate;
    }

    public int EmbeddingDim => _embd;
    public int ProjectionDim => _projDim;

    public Granite4VisionEncoder(Granite4VisionModel model)
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
        _patchEmbdB = VisionOps.GetTensorArray(gguf, "v.patch_embd.bias");
        _posEmbdF32 = VisionOps.DequantizeToFloat32(VisionOps.GetTensor(gguf, "v.position_embd.weight", "v.position_embd"));
        _postLnW = VisionOps.GetTensorArray(gguf, "v.post_ln.weight");
        _postLnB = VisionOps.GetTensorArray(gguf, "v.post_ln.bias");

        _projNormW = VisionOps.GetTensorArray(gguf, "mm.proj_norm.weight", "mm.0.weight");
        _projNormB = VisionOps.GetTensorArray(gguf, "mm.proj_norm.bias", "mm.0.bias");
        _projW = VisionOps.GetTensor(gguf, "mm.proj.weight", "mm.1.weight");
        _projB = VisionOps.GetTensorArray(gguf, "mm.proj.bias", "mm.1.bias");

        _blocks = new LayerWeights[_layers];
        for (int l = 0; l < _layers; l++)
        {
            var upTensor = gguf.FindTensor($"v.blk.{l}.ffn_up.weight");
            int intermediate = upTensor.HasValue ? (int)upTensor.Value.Dimensions[1] : (_embd * 4);

            _blocks[l] = new LayerWeights
            {
                Ln1W = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.ln1.weight"),
                Ln1B = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.ln1.bias"),
                AttnQW = VisionOps.GetTensor(gguf, $"v.blk.{l}.attn_q.weight"),
                AttnQB = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.attn_q.bias"),
                AttnKW = VisionOps.GetTensor(gguf, $"v.blk.{l}.attn_k.weight"),
                AttnKB = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.attn_k.bias"),
                AttnVW = VisionOps.GetTensor(gguf, $"v.blk.{l}.attn_v.weight"),
                AttnVB = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.attn_v.bias"),
                AttnOutW = VisionOps.GetTensor(gguf, $"v.blk.{l}.attn_out.weight"),
                AttnOutB = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.attn_out.bias"),
                Ln2W = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.ln2.weight"),
                Ln2B = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.ln2.bias"),
                FfnUpW = VisionOps.GetTensor(gguf, $"v.blk.{l}.ffn_up.weight"),
                FfnUpB = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.ffn_up.bias"),
                FfnDownW = VisionOps.GetTensor(gguf, $"v.blk.{l}.ffn_down.weight"),
                FfnDownB = VisionOps.GetTensorArray(gguf, $"v.blk.{l}.ffn_down.bias"),
                FfnIntermediate = intermediate
            };
        }
    }

    public float[] Forward(float[] chw, int targetWidth, int targetHeight, int patchesX, int patchesY, out int tokenCount)
    {
        var img = new Granite4PreprocessedImage(chw, targetWidth, targetHeight, patchesX, patchesY);
        var result = Encode(img);
        tokenCount = patchesX * patchesY;
        return result;
    }

    public float[] Encode(Granite4PreprocessedImage img)
    {
        int numPatches = img.PatchesX * img.PatchesY;
        var hiddenStates = new float[numPatches * _embd];

        fixed (float* chwPtr = img.Chw)
        {
            ExtractPatches(chwPtr, img.TargetWidth, img.TargetHeight, img.PatchesX, img.PatchesY, hiddenStates);
        }

        if (_posEmbdF32.Length > 0)
        {
            for (int i = 0; i < hiddenStates.Length; i++) hiddenStates[i] += _posEmbdF32[i % _embd];
        }

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

            fixed (float* ln1W = blk.Ln1W, ln1B = blk.Ln1B, attnQB = blk.AttnQB, attnKB = blk.AttnKB,
                   attnVB = blk.AttnVB, attnOutB = blk.AttnOutB, ln2W = blk.Ln2W, ln2B = blk.Ln2B,
                   ffnUpB = blk.FfnUpB, ffnDownB = blk.FfnDownB)
            {
                Array.Copy(hiddenStates, normed, hiddenStates.Length);
                VisionOps.LayerNorm(normed, numPatches, _embd, ln1W, ln1B, _eps);

                VisionOps.MatVecAny(normed, blk.AttnQW, attnQB, numPatches, _embd, _embd, qBuf);
                VisionOps.MatVecAny(normed, blk.AttnKW, attnKB, numPatches, _embd, _embd, kBuf);
                VisionOps.MatVecAny(normed, blk.AttnVW, attnVB, numPatches, _embd, _embd, vBuf);

                VisionOps.Attention(qBuf, kBuf, vBuf, numPatches, _heads, _headDim, normed);
                VisionOps.MatVecAny(normed, blk.AttnOutW, attnOutB, numPatches, _embd, _embd, attnOut);

                for (int i = 0; i < hiddenStates.Length; i++) hiddenStates[i] += attnOut[i];

                Array.Copy(hiddenStates, normed, hiddenStates.Length);
                VisionOps.LayerNorm(normed, numPatches, _embd, ln2W, ln2B, _eps);

                int intermediate = blk.FfnIntermediate;
                VisionOps.MatVecAny(normed, blk.FfnUpW, ffnUpB, numPatches, _embd, intermediate, ffnMid);
                VisionOps.Gelu(ffnMid.AsSpan(0, numPatches * intermediate));
                VisionOps.MatVecAny(ffnMid, blk.FfnDownW, ffnDownB, numPatches, intermediate, _embd, attnOut);

                for (int i = 0; i < hiddenStates.Length; i++) hiddenStates[i] += attnOut[i];
            }
        }

        if (_postLnW != null)
        {
            fixed (float* postLnW = _postLnW, postLnB = _postLnB)
            {
                VisionOps.LayerNorm(hiddenStates, numPatches, _embd, postLnW, postLnB, _eps);
            }
        }

        // Projector: WindowQFormer / MLP downsampler
        var projOut = new float[numPatches * _projDim];
        if (_projW.IsValid)
        {
            fixed (float* projNormW = _projNormW, projNormB = _projNormB, projB = _projB)
            {
                if (projNormW != null)
                {
                    VisionOps.LayerNorm(hiddenStates, numPatches, _embd, projNormW, projNormB, 1e-5f);
                }
                VisionOps.MatVecAny(hiddenStates, _projW, projB, numPatches, _embd, _projDim, projOut);
            }
        }
        else
        {
            for (int t = 0; t < numPatches; t++)
            {
                int copyDim = Math.Min(_embd, _projDim);
                Array.Copy(hiddenStates, t * _embd, projOut, t * _projDim, copyDim);
            }
        }

        return projOut;
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

                        output[outOffset + d] = sum;
                    }
                }
            }
        });
    }
}
