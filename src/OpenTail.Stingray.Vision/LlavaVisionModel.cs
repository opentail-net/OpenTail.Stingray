using System;
using System.IO;
using System.Linq;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Vision;

/// <summary>
/// Model container and metadata descriptor for LLaVA 1.5, 1.6 (NeXT), and LLaVA-OneVision models.
/// </summary>
public sealed class LlavaVisionModel : IDisposable
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
    public bool HasClassEmbedding { get; }
    public float Eps { get; }

    private bool _disposed;

    private LlavaVisionModel(
        GgufModel gguf,
        string projectorType,
        int patchSize,
        int imageSize,
        int embeddingDim,
        int projectionDim,
        int layerCount,
        int headCount,
        int headDim,
        bool hasClassEmbedding,
        float eps)
    {
        Gguf = gguf;
        ProjectorType = projectorType;
        PatchSize = patchSize;
        ImageSize = imageSize;
        EmbeddingDim = embeddingDim;
        ProjectionDim = projectionDim;
        LayerCount = layerCount;
        HeadCount = headCount;
        HeadDim = headDim;
        HasClassEmbedding = hasClassEmbedding;
        Eps = eps;
    }

    public static LlavaVisionModel Open(string path)
    {
        var gguf = GgufModel.Open(path);
        return FromGguf(gguf);
    }

    public static LlavaVisionModel FromGguf(GgufModel gguf)
    {
        string projType = "llava";
        if (gguf.Metadata.TryGetValue("clip.vision.projector_type", out var ptObj) && ptObj is string ptStr)
            projType = ptStr.Trim().ToLowerInvariant();
        else if (gguf.Metadata.TryGetValue("clip.projector_type", out var ptObj2) && ptObj2 is string ptStr2)
            projType = ptStr2.Trim().ToLowerInvariant();

        int patchSize = GetInt(gguf, "clip.vision.patch_size", 14);
        int imageSize = GetInt(gguf, "clip.vision.image_size", 336);
        int embeddingDim = GetInt(gguf, "clip.vision.embedding_length", 1024);
        int projectionDim = GetInt(gguf, "clip.vision.projection_dim", 0);
        int layerCount = GetInt(gguf, "clip.vision.block_count", 24);
        int headCount = GetInt(gguf, "clip.vision.attention.head_count", 16);
        int headDim = headCount > 0 ? embeddingDim / headCount : 64;
        bool hasClassEmbd = gguf.FindTensor("v.class_embd").HasValue || gguf.FindTensor("v.cls_embd").HasValue;
        float eps = GetFloat(gguf, "clip.vision.attention.layer_norm_epsilon", 1e-5f);

        if (projectionDim <= 0)
        {
            var pTensor = gguf.FindTensor("mm.2.weight") ?? gguf.FindTensor("mm.1.weight") ?? gguf.FindTensor("mm.0.weight");
            if (pTensor.HasValue)
            {
                projectionDim = (int)pTensor.Value.Dimensions[1];
            }
            else
            {
                projectionDim = 4096;
            }
        }

        return new LlavaVisionModel(
            gguf,
            projType,
            patchSize,
            imageSize,
            embeddingDim,
            projectionDim,
            layerCount,
            headCount,
            headDim,
            hasClassEmbd,
            eps);
    }

    private static int GetInt(GgufModel gguf, string key, int defaultValue)
    {
        if (gguf.Metadata.TryGetValue(key, out var val))
        {
            if (val is int i) return i;
            if (val is uint u) return (int)u;
            if (val is long l) return (int)l;
            if (val is ulong ul) return (int)ul;
        }
        return defaultValue;
    }

    private static float GetFloat(GgufModel gguf, string key, float defaultValue)
    {
        if (gguf.Metadata.TryGetValue(key, out var val))
        {
            if (val is float f) return f;
            if (val is double d) return (float)d;
        }
        return defaultValue;
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            Gguf.Dispose();
            _disposed = true;
        }
    }
}
