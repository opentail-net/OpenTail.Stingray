using System;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Vision;

/// <summary>
/// Model container for LG AI Research EXAONE 4.5 Vision.
/// Architecture: Qwen2.5-VL-shaped ViT with GQA (32Q/8KV heads), optional window attention,
/// dual Conv2D patch embedding (temporal merge), 2D M-RoPE, SwiGLU MLP,
/// 4-patch spatial merge projector.
/// Reference: examples/llama.cpp/llama.cpp/tools/mtmd/models/exaone4_5.cpp
/// </summary>
public sealed class Exaone4VisionModel : IDisposable
{
    public GgufModel Gguf { get; }
    public string ProjectorType { get; }
    public int PatchSize { get; }
    public int ImageSize { get; }
    public int EmbeddingDim { get; }
    public int ProjectionDim { get; }
    public int LayerCount { get; }
    public int HeadCount { get; }
    public int KvHeadCount { get; }
    public int HeadDim { get; }
    public float RopeTheta { get; }
    /// <summary>Window attention repeating pattern size (0 = no window attn).</summary>
    public int WindowAttnPattern { get; }
    public float Eps { get; }

    private bool _disposed;

    private Exaone4VisionModel(
        GgufModel gguf, string projectorType,
        int patchSize, int imageSize,
        int embeddingDim, int projectionDim,
        int layerCount, int headCount, int kvHeadCount, int headDim,
        float ropeTheta, int windowAttnPattern, float eps)
    {
        Gguf = gguf; ProjectorType = projectorType;
        PatchSize = patchSize; ImageSize = imageSize;
        EmbeddingDim = embeddingDim; ProjectionDim = projectionDim;
        LayerCount = layerCount; HeadCount = headCount;
        KvHeadCount = kvHeadCount; HeadDim = headDim;
        RopeTheta = ropeTheta; WindowAttnPattern = windowAttnPattern; Eps = eps;
    }

    public static Exaone4VisionModel Open(string path) => FromGguf(GgufModel.Open(path));

    public static Exaone4VisionModel FromGguf(GgufModel gguf)
    {
        string projType = "exaone4";
        if (gguf.Metadata.TryGetValue("clip.vision.projector_type", out var ptObj) && ptObj is string ptStr)
            projType = ptStr.Trim().ToLowerInvariant();

        int patchSize     = GetInt(gguf, "clip.vision.patch_size", 14);
        int imageSize     = GetInt(gguf, "clip.vision.image_size", 980);
        int embeddingDim  = GetInt(gguf, "clip.vision.embedding_length", 1280);
        int projectionDim = GetInt(gguf, "clip.vision.projection_dim", 0);
        int layerCount    = GetInt(gguf, "clip.vision.block_count", 32);
        int headCount     = GetInt(gguf, "clip.vision.attention.head_count", 32);
        int kvHeadCount   = GetInt(gguf, "clip.vision.attention.head_count_kv", 8);
        if (kvHeadCount <= 0) kvHeadCount = headCount;
        int headDim       = headCount > 0 ? embeddingDim / headCount : 40;
        float ropeTheta   = GetFloat(gguf, "clip.vision.rope.freq_base", 10000.0f);
        int waPattern     = GetInt(gguf, "clip.vision.n_wa_pattern", 0);
        float eps         = GetFloat(gguf, "clip.vision.attention.layer_norm_epsilon", 1e-6f);

        if (projectionDim <= 0)
        {
            var t = gguf.FindTensor("mm.1.weight") ?? gguf.FindTensor("mm.0.weight");
            projectionDim = t.HasValue ? (int)t.Value.Dimensions[1] : 4096;
        }

        return new Exaone4VisionModel(
            gguf, projType, patchSize, imageSize, embeddingDim, projectionDim,
            layerCount, headCount, kvHeadCount, headDim, ropeTheta, waPattern, eps);
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
