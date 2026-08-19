using System;
using System.IO;
using System.Linq;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Vision;

/// <summary>
/// Model container and metadata parser for CogVLM and CogAgent vision models.
/// Reference: examples/llama.cpp/llama.cpp/tools/mtmd/models/cogvlm.cpp
/// </summary>
public sealed class CogVlmVisionModel : IDisposable
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

    private CogVlmVisionModel(
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

    public static CogVlmVisionModel Open(string path)
    {
        var gguf = GgufModel.Open(path);
        return FromGguf(gguf);
    }

    public static CogVlmVisionModel FromGguf(GgufModel gguf)
    {
        string projType = "cogvlm";
        if (gguf.Metadata.TryGetValue("clip.vision.projector_type", out var ptObj) && ptObj is string ptStr)
            projType = ptStr.Trim().ToLowerInvariant();

        int patchSize = GetInt(gguf, "clip.vision.patch_size", 14);
        int imageSize = GetInt(gguf, "clip.vision.image_size", 490);
        int embeddingDim = GetInt(gguf, "clip.vision.embedding_length", 1024);
        int projectionDim = GetInt(gguf, "clip.vision.projection_dim", 4096);
        int layerCount = GetInt(gguf, "clip.vision.block_count", 24);
        int headCount = GetInt(gguf, "clip.vision.attention.head_count", 16);
        int headDim = headCount > 0 ? embeddingDim / headCount : 64;
        float eps = GetFloat(gguf, "clip.vision.attention.layer_norm_epsilon", 1e-5f);

        return new CogVlmVisionModel(
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
