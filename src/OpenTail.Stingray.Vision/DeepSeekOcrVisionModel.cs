using System;
using System.IO;
using System.Linq;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Vision;

/// <summary>
/// Model container and metadata descriptor for DeepSeek-OCR and DeepSeek-OCR2 multimodal vision models.
/// Reference: examples/llama.cpp/llama.cpp/tools/mtmd/models/deepseekocr.cpp
/// </summary>
public sealed class DeepSeekOcrVisionModel : IDisposable
{
    public GgufModel Gguf { get; }
    public string ProjectorType { get; }
    public int PatchSize { get; }
    public int ImageSize { get; }
    public int EmbeddingDim { get; }
    public int SamEmbeddingDim { get; }
    public int ProjectionDim { get; }
    public int LayerCount { get; }
    public int HeadCount { get; }
    public int HeadDim { get; }
    public int WindowSize { get; }
    public bool IsV2 { get; }
    public float Eps { get; }

    private bool _disposed;

    private DeepSeekOcrVisionModel(
        GgufModel gguf,
        string projectorType,
        int patchSize,
        int imageSize,
        int embeddingDim,
        int samEmbeddingDim,
        int projectionDim,
        int layerCount,
        int headCount,
        int headDim,
        int windowSize,
        bool isV2,
        float eps)
    {
        Gguf = gguf;
        ProjectorType = projectorType;
        PatchSize = patchSize;
        ImageSize = imageSize;
        EmbeddingDim = embeddingDim;
        SamEmbeddingDim = samEmbeddingDim;
        ProjectionDim = projectionDim;
        LayerCount = layerCount;
        HeadCount = headCount;
        HeadDim = headDim;
        WindowSize = windowSize;
        IsV2 = isV2;
        Eps = eps;
    }

    public static DeepSeekOcrVisionModel Open(string path)
    {
        var gguf = GgufModel.Open(path);
        return FromGguf(gguf);
    }

    public static DeepSeekOcrVisionModel FromGguf(GgufModel gguf)
    {
        string projType = "deepseekocr";
        if (gguf.Metadata.TryGetValue("clip.vision.projector_type", out var ptObj) && ptObj is string ptStr)
            projType = ptStr.Trim().ToLowerInvariant();
        else if (gguf.Metadata.TryGetValue("clip.projector_type", out var ptObj2) && ptObj2 is string ptStr2)
            projType = ptStr2.Trim().ToLowerInvariant();

        bool isV2 = projType.Contains("deepseekocr2") || projType.Contains("deepseek_ocr2") || gguf.Tensors.Any(t => t.Name.Contains("resample_query"));

        int patchSize = GetInt(gguf, "clip.vision.patch_size", 16);
        int imageSize = GetInt(gguf, "clip.vision.image_size", 1024);
        int embeddingDim = GetInt(gguf, "clip.vision.embedding_length", 1024);
        int samEmbeddingDim = GetInt(gguf, "clip.vision.sam.embedding_length", GetInt(gguf, "clip.vision.sam_embedding_length", 1024));
        int projectionDim = GetInt(gguf, "clip.vision.projection_dim", 0);
        int layerCount = GetInt(gguf, "clip.vision.block_count", 24);
        int headCount = GetInt(gguf, "clip.vision.attention.head_count", 16);
        int headDim = headCount > 0 ? embeddingDim / headCount : 64;
        int windowSize = GetInt(gguf, "clip.vision.window_size", GetInt(gguf, "clip.vision.attn_window_size", 16));
        float eps = GetFloat(gguf, "clip.vision.attention.layer_norm_epsilon", 1e-6f);

        if (projectionDim <= 0)
        {
            var pTensor = gguf.FindTensor("mm.model.fc.weight") ?? gguf.FindTensor("mm.fc.weight") ?? gguf.FindTensor("mm.2.weight") ?? gguf.FindTensor("mm.0.weight");
            if (pTensor.HasValue)
            {
                projectionDim = (int)pTensor.Value.Dimensions[1];
            }
            else
            {
                projectionDim = 1280;
            }
        }

        return new DeepSeekOcrVisionModel(
            gguf,
            projType,
            patchSize,
            imageSize,
            embeddingDim,
            samEmbeddingDim,
            projectionDim,
            layerCount,
            headCount,
            headDim,
            windowSize,
            isV2,
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
