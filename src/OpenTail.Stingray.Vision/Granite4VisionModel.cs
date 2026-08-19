using System;
using System.IO;
using System.Linq;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Vision;

/// <summary>
/// Model container and metadata parser for IBM Granite 4 Vision models.
/// Reference: examples/llama.cpp/llama.cpp/tools/mtmd/models/granite4-vision.cpp
/// </summary>
public sealed class Granite4VisionModel : IDisposable
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
    public float Eps { get; }

    private bool _disposed;

    private Granite4VisionModel(
        GgufModel gguf,
        string projectorType,
        int patchSize,
        int imageSize,
        int embeddingDim,
        int projectionDim,
        int layerCount,
        int headCount,
        int headDim,
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
        Eps = eps;
    }

    public static Granite4VisionModel Open(string path)
    {
        var gguf = GgufModel.Open(path);
        return FromGguf(gguf);
    }

    public static Granite4VisionModel FromGguf(GgufModel gguf)
    {
        string projType = "granite4-vision";
        if (gguf.Metadata.TryGetValue("clip.vision.projector_type", out var ptObj) && ptObj is string ptStr)
            projType = ptStr.Trim().ToLowerInvariant();

        int patchSize = GetInt(gguf, "clip.vision.patch_size", 14);
        int imageSize = GetInt(gguf, "clip.vision.image_size", 384);
        int embeddingDim = GetInt(gguf, "clip.vision.embedding_length", 1152);
        int projectionDim = GetInt(gguf, "clip.vision.projection_dim", 2048);
        int layerCount = GetInt(gguf, "clip.vision.block_count", 27);
        int headCount = GetInt(gguf, "clip.vision.attention.head_count", 16);
        int headDim = headCount > 0 ? embeddingDim / headCount : 72;
        float eps = GetFloat(gguf, "clip.vision.attention.layer_norm_epsilon", 1e-6f);

        return new Granite4VisionModel(
            gguf,
            projType,
            patchSize,
            imageSize,
            embeddingDim,
            projectionDim,
            layerCount,
            headCount,
            headDim,
            eps);
    }

    private static int GetInt(GgufModel g, string key, int fallback)
    {
        if (g.Metadata.TryGetValue(key, out var val))
        {
            if (val is int i) return i;
            if (val is uint u) return (int)u;
            if (val is long l) return (int)l;
            if (val is ulong ul) return (int)ul;
        }
        return fallback;
    }

    private static float GetFloat(GgufModel g, string key, float fallback)
    {
        if (g.Metadata.TryGetValue(key, out var val))
        {
            if (val is float f) return f;
            if (val is double d) return (float)d;
        }
        return fallback;
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
