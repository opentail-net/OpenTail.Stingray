using System;
using System.IO;
using System.Linq;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Vision;

/// <summary>
/// Model container and metadata parser for MobileNetV5 vision backbone models.
/// Reference: examples/llama.cpp/llama.cpp/tools/mtmd/models/mobilenetv5.cpp
/// </summary>
public sealed class MobileNetV5VisionModel : IDisposable
{
    public GgufModel Gguf { get; }
    public string ProjectorType { get; }
    public int PatchSize { get; }
    public int ImageSize { get; }
    public int EmbeddingDim { get; }
    public int ProjectionDim { get; }
    public int LayerCount { get; }
    public float Eps { get; }

    private bool _disposed;

    private MobileNetV5VisionModel(
        GgufModel gguf,
        string projectorType,
        int patchSize,
        int imageSize,
        int embeddingDim,
        int projectionDim,
        int layerCount,
        float eps)
    {
        Gguf = gguf;
        ProjectorType = projectorType;
        PatchSize = patchSize;
        ImageSize = imageSize;
        EmbeddingDim = embeddingDim;
        ProjectionDim = projectionDim;
        LayerCount = layerCount;
        Eps = eps;
    }

    public static MobileNetV5VisionModel Open(string path)
    {
        var gguf = GgufModel.Open(path);
        return FromGguf(gguf);
    }

    public static MobileNetV5VisionModel FromGguf(GgufModel gguf)
    {
        string projType = "mobilenetv5";
        if (gguf.Metadata.TryGetValue("clip.vision.projector_type", out var ptObj) && ptObj is string ptStr)
            projType = ptStr.Trim().ToLowerInvariant();

        int patchSize = GetInt(gguf, "clip.vision.patch_size", 16);
        int imageSize = GetInt(gguf, "clip.vision.image_size", 224);
        int embeddingDim = GetInt(gguf, "clip.vision.embedding_length", 512);
        int projectionDim = GetInt(gguf, "clip.vision.projection_dim", 2048);
        int layerCount = GetInt(gguf, "clip.vision.block_count", 16);
        float eps = GetFloat(gguf, "clip.vision.attention.layer_norm_epsilon", 1e-5f);

        return new MobileNetV5VisionModel(
            gguf,
            projType,
            patchSize,
            imageSize,
            embeddingDim,
            projectionDim,
            layerCount,
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
