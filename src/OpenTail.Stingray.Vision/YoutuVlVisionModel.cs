using System;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Vision;

/// <summary>
/// Model container for YoutuVL (Tencent YouTu Lab).
/// Architecture: Qwen2.5-VL-shaped ViT â Conv3D-as-linear patch embed (2 frames),
/// 2D M-RoPE, optional window attention, VLPatchMerger (RMSNorm + 2x2 spatial merge + 2-layer GELU MLP).
/// Reference: examples/llama.cpp/llama.cpp/tools/mtmd/models/youtuvl.cpp
/// </summary>
public sealed class YoutuVlVisionModel : IDisposable
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
    /// <summary>Spatial merge factor (always 2 for 2Ã2 merge).</summary>
    public int SpatialMergeFactor { get; }
    public float Eps { get; }

    private bool _disposed;

    private YoutuVlVisionModel(
        GgufModel gguf, string projectorType,
        int patchSize, int imageSize,
        int embeddingDim, int projectionDim,
        int layerCount, int headCount, int headDim,
        int spatialMergeFactor, float eps)
    {
        Gguf = gguf; ProjectorType = projectorType;
        PatchSize = patchSize; ImageSize = imageSize;
        EmbeddingDim = embeddingDim; ProjectionDim = projectionDim;
        LayerCount = layerCount; HeadCount = headCount; HeadDim = headDim;
        SpatialMergeFactor = spatialMergeFactor; Eps = eps;
    }

    public static YoutuVlVisionModel Open(string path) => FromGguf(GgufModel.Open(path));

    public static YoutuVlVisionModel FromGguf(GgufModel gguf)
    {
        string projType = "youtuvl";
        if (gguf.Metadata.TryGetValue("clip.vision.projector_type", out var ptObj) && ptObj is string ptStr)
            projType = ptStr.Trim().ToLowerInvariant();

        int patchSize     = GetInt(gguf, "clip.vision.patch_size", 14);
        int imageSize     = GetInt(gguf, "clip.vision.image_size", 980);
        int embeddingDim  = GetInt(gguf, "clip.vision.embedding_length", 1280);
        int projectionDim = GetInt(gguf, "clip.vision.projection_dim", 0);
        int layerCount    = GetInt(gguf, "clip.vision.block_count", 32);
        int headCount     = GetInt(gguf, "clip.vision.attention.head_count", 16);
        int headDim       = headCount > 0 ? embeddingDim / headCount : 80;
        int mergeFactor   = GetInt(gguf, "clip.vision.merge_factor", 2);
        if (mergeFactor <= 0) mergeFactor = 2;
        float eps = GetFloat(gguf, "clip.vision.attention.layer_norm_epsilon", 1e-6f);

        if (projectionDim <= 0)
        {
            var t = gguf.FindTensor("mm.1.weight") ?? gguf.FindTensor("mm.0.weight");
            projectionDim = t.HasValue ? (int)t.Value.Dimensions[1] : 4096;
        }

        return new YoutuVlVisionModel(
            gguf, projType, patchSize, imageSize, embeddingDim, projectionDim,
            layerCount, headCount, headDim, mergeFactor, eps);
    }

    private static int GetInt(GgufModel gguf, string key, int def)
    {
        if (gguf.Metadata.TryGetValue(key, out var v))
        {
            if (v is int i)    return i;
            if (v is uint u)   return (int)u;
            if (v is long l)   return (int)l;
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
