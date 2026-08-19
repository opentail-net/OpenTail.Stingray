using System;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Vision;

/// <summary>
/// Model container for Step-3 VL multimodal vision (Kuaishou/StepFun).
/// Architecture: SigLIP ViT + 2D RoPE + two Conv2D downsamplers + linear projector.
/// Reference: examples/llama.cpp/llama.cpp/tools/mtmd/models/step3vl.cpp
/// </summary>
public sealed class Step3VlVisionModel : IDisposable
{
    public GgufModel Gguf { get; }
    public string ProjectorType { get; }
    public int PatchSize { get; }
    public int ImageSize { get; }
    public int EmbeddingDim { get; }
    public int ProjectionDim { get; }
    public int LayerCount { get; }
    public int HeadCount { get; }
    public int HeadDim { get; }
    public float RopeTheta { get; }
    public float Eps { get; }

    private bool _disposed;

    private Step3VlVisionModel(
        GgufModel gguf, string projectorType,
        int patchSize, int imageSize,
        int embeddingDim, int projectionDim,
        int layerCount, int headCount, int headDim,
        float ropeTheta, float eps)
    {
        Gguf = gguf; ProjectorType = projectorType;
        PatchSize = patchSize; ImageSize = imageSize;
        EmbeddingDim = embeddingDim; ProjectionDim = projectionDim;
        LayerCount = layerCount; HeadCount = headCount; HeadDim = headDim;
        RopeTheta = ropeTheta; Eps = eps;
    }

    public static Step3VlVisionModel Open(string path) => FromGguf(GgufModel.Open(path));

    public static Step3VlVisionModel FromGguf(GgufModel gguf)
    {
        string projType = "step3vl";
        if (gguf.Metadata.TryGetValue("clip.vision.projector_type", out var ptObj) && ptObj is string ptStr)
            projType = ptStr.Trim().ToLowerInvariant();

        int patchSize     = GetInt(gguf, "clip.vision.patch_size", 14);
        int imageSize     = GetInt(gguf, "clip.vision.image_size", 1024);
        int embeddingDim  = GetInt(gguf, "clip.vision.embedding_length", 1536);
        int projectionDim = GetInt(gguf, "clip.vision.projection_dim", 0);
        int layerCount    = GetInt(gguf, "clip.vision.block_count", 40);
        int headCount     = GetInt(gguf, "clip.vision.attention.head_count", 16);
        int headDim       = headCount > 0 ? embeddingDim / headCount : 96;
        float ropeTheta   = GetFloat(gguf, "clip.vision.rope.freq_base", 10000.0f);
        float eps         = GetFloat(gguf, "clip.vision.attention.layer_norm_epsilon", 1e-6f);

        if (projectionDim <= 0)
        {
            var t = gguf.FindTensor("mm.model_proj.weight");
            projectionDim = t.HasValue ? (int)t.Value.Dimensions[1] : 4096;
        }

        return new Step3VlVisionModel(
            gguf, projType, patchSize, imageSize, embeddingDim, projectionDim,
            layerCount, headCount, headDim, ropeTheta, eps);
    }

    private static int GetInt(GgufModel gguf, string key, int def)
    {
        if (gguf.Metadata.TryGetValue(key, out var v))
        {
            if (v is int i)   return i;
            if (v is uint u)  return (int)u;
            if (v is long l)  return (int)l;
            if (v is ulong ul) return (int)ul;
        }
        return def;
    }

    private static float GetFloat(GgufModel gguf, string key, float def)
    {
        if (gguf.Metadata.TryGetValue(key, out var v))
        {
            if (v is float f)  return f;
            if (v is double d) return (float)d;
        }
        return def;
    }

    public void Dispose()
    {
        if (!_disposed) { Gguf.Dispose(); _disposed = true; }
    }
}
