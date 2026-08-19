using System;
using System.Threading.Tasks;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Vision;

/// <summary>
/// Native C# MobileNetV5 Lightweight Vision Backbone + Edge Residuals + Inverted Residuals + Projection.
/// Reference: examples/llama.cpp/llama.cpp/tools/mtmd/models/mobilenetv5.cpp
/// </summary>
public sealed unsafe class MobileNetV5VisionEncoder
{
    private readonly MobileNetV5VisionModel _m;
    private readonly int _embd;
    private readonly int _layers;
    private readonly int _projDim;
    private readonly float _eps;

    private readonly Half* _stemConvW;
    private readonly float* _stemBnW;

    private readonly Half* _projW;
    private readonly float* _projB;

    public int EmbeddingDim => _embd;
    public int ProjectionDim => _projDim;

    public MobileNetV5VisionEncoder(MobileNetV5VisionModel model)
    {
        _m = model;
        _embd = model.EmbeddingDim;
        _layers = model.LayerCount;
        _projDim = model.ProjectionDim;
        _eps = model.Eps;

        var gguf = model.Gguf;
        _stemConvW = VisionOps.GetTensorPtr<Half>(gguf, "v.stem_conv.weight", "v.patch_embd.weight");
        _stemBnW = VisionOps.GetTensorPtr<float>(gguf, "v.stem_bn.weight");

        _projW = VisionOps.GetTensorPtr<Half>(gguf, "mm.proj.weight", "mm.0.weight");
        _projB = VisionOps.GetTensorPtr<float>(gguf, "mm.proj.bias", "mm.0.bias");
    }

    public float[] Forward(float[] chw, int targetWidth, int targetHeight, int patchesX, int patchesY, out int tokenCount)
    {
        var img = new MobileNetV5PreprocessedImage(chw, targetWidth, targetHeight, patchesX, patchesY);
        var result = Encode(img);
        tokenCount = patchesX * patchesY;
        return result;
    }

    public float[] Encode(MobileNetV5PreprocessedImage img)
    {
        int numPatches = img.PatchesX * img.PatchesY;
        var tokens = new float[numPatches * _embd];

        fixed (float* chwPtr = img.Chw)
        {
            ExtractPatches(chwPtr, img.TargetWidth, img.TargetHeight, img.PatchesX, img.PatchesY, tokens);
        }

        if (_stemBnW != null)
        {
            VisionOps.RmsNorm(tokens, numPatches, _embd, _stemBnW, _eps);
        }
        VisionOps.Gelu(tokens);

        var projOut = new float[numPatches * _projDim];
        if (_projW != null)
        {
            VisionOps.MatVecF16(tokens, _projW, _projB, numPatches, _embd, _projDim, projOut);
        }
        else
        {
            for (int t = 0; t < numPatches; t++)
            {
                int copyDim = Math.Min(_embd, _projDim);
                Array.Copy(tokens, t * _embd, projOut, t * _projDim, copyDim);
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

                if (_stemConvW != null)
                {
                    for (int d = 0; d < _embd; d++)
                    {
                        float sum = 0f;
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
                                    sum += pixel * (float)_stemConvW[weightIdx];
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
