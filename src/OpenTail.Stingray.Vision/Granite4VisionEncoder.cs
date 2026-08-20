using System;
using System.Threading.Tasks;
using OpenTail.Stingray.Core;

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

    private readonly Half* _patchEmbdW;
    private readonly float* _patchEmbdB;
    private readonly float* _posEmbd;
    private readonly float* _postLnW;
    private readonly float* _postLnB;

    private readonly float* _projNormW;
    private readonly float* _projNormB;
    private readonly Half* _projW;
    private readonly float* _projB;

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
        _patchEmbdW = VisionOps.GetTensorPtr<Half>(gguf, "v.patch_embd.weight");
        _patchEmbdB = VisionOps.GetTensorPtr<float>(gguf, "v.patch_embd.bias");
        _posEmbd = VisionOps.GetTensorPtr<float>(gguf, "v.position_embd.weight", "v.position_embd");
        _postLnW = VisionOps.GetTensorPtr<float>(gguf, "v.post_ln.weight");
        _postLnB = VisionOps.GetTensorPtr<float>(gguf, "v.post_ln.bias");

        _projNormW = VisionOps.GetTensorPtr<float>(gguf, "mm.proj_norm.weight", "mm.0.weight");
        _projNormB = VisionOps.GetTensorPtr<float>(gguf, "mm.proj_norm.bias", "mm.0.bias");
        _projW = VisionOps.GetTensorPtr<Half>(gguf, "mm.proj.weight", "mm.1.weight");
        _projB = VisionOps.GetTensorPtr<float>(gguf, "mm.proj.bias", "mm.1.bias");

        _blocks = new LayerWeights[_layers];
        for (int l = 0; l < _layers; l++)
        {
            var upTensor = gguf.FindTensor($"v.blk.{l}.ffn_up.weight");
            int intermediate = upTensor.HasValue ? (int)upTensor.Value.Dimensions[1] : (_embd * 4);

            _blocks[l] = new LayerWeights
            {
                Ln1W = VisionOps.GetTensorPtr<float>(gguf, $"v.blk.{l}.ln1.weight"),
                Ln1B = VisionOps.GetTensorPtr<float>(gguf, $"v.blk.{l}.ln1.bias"),
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
                FfnUpW = VisionOps.GetTensorPtr<Half>(gguf, $"v.blk.{l}.ffn_up.weight"),
                FfnUpB = VisionOps.GetTensorPtr<float>(gguf, $"v.blk.{l}.ffn_up.bias"),
                FfnDownW = VisionOps.GetTensorPtr<Half>(gguf, $"v.blk.{l}.ffn_down.weight"),
                FfnDownB = VisionOps.GetTensorPtr<float>(gguf, $"v.blk.{l}.ffn_down.bias"),
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

        if (_posEmbd != null)
        {
            for (int i = 0; i < hiddenStates.Length; i++) hiddenStates[i] += _posEmbd[i % _embd];
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

            Array.Copy(hiddenStates, normed, hiddenStates.Length);
            VisionOps.LayerNorm(normed, numPatches, _embd, blk.Ln1W, blk.Ln1B, _eps);

            VisionOps.MatVecF16(normed, blk.AttnQW, blk.AttnQB, numPatches, _embd, _embd, qBuf);
            VisionOps.MatVecF16(normed, blk.AttnKW, blk.AttnKB, numPatches, _embd, _embd, kBuf);
            VisionOps.MatVecF16(normed, blk.AttnVW, blk.AttnVB, numPatches, _embd, _embd, vBuf);

            VisionOps.Attention(qBuf, kBuf, vBuf, numPatches, _heads, _headDim, normed);
            VisionOps.MatVecF16(normed, blk.AttnOutW, blk.AttnOutB, numPatches, _embd, _embd, attnOut);

            for (int i = 0; i < hiddenStates.Length; i++) hiddenStates[i] += attnOut[i];

            Array.Copy(hiddenStates, normed, hiddenStates.Length);
            VisionOps.LayerNorm(normed, numPatches, _embd, blk.Ln2W, blk.Ln2B, _eps);

            int intermediate = blk.FfnIntermediate;
            VisionOps.MatVecF16(normed, blk.FfnUpW, blk.FfnUpB, numPatches, _embd, intermediate, ffnMid);
            VisionOps.Gelu(ffnMid.AsSpan(0, numPatches * intermediate));
            VisionOps.MatVecF16(ffnMid, blk.FfnDownW, blk.FfnDownB, numPatches, intermediate, _embd, attnOut);

            for (int i = 0; i < hiddenStates.Length; i++) hiddenStates[i] += attnOut[i];
        }

        if (_postLnW != null) VisionOps.LayerNorm(hiddenStates, numPatches, _embd, _postLnW, _postLnB, _eps);

        // Projector: WindowQFormer / MLP downsampler
        var projOut = new float[numPatches * _projDim];
        if (_projW != null)
        {
            if (_projNormW != null)
            {
                VisionOps.LayerNorm(hiddenStates, numPatches, _embd, _projNormW, _projNormB, 1e-5f);
            }
            VisionOps.MatVecF16(hiddenStates, _projW, _projB, numPatches, _embd, _projDim, projOut);
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

                        output[outOffset + d] = sum;
                    }
                }
            }
        });
    }
}
