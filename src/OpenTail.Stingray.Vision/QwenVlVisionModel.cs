using System;
using System.IO;
using System.Linq;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Vision;

/// <summary>
/// Parser and container for Qwen2-VL, Qwen2.5-VL, and Qwen3-VL Vision mmproj GGUF models.
/// </summary>
public sealed class QwenVlVisionModel : IDisposable
{
    public GgufModel Gguf { get; }
    public string ProjectorType { get; }
    public int PatchSize { get; } = 14;
    public int SpatialMergeFactor { get; } = 2; // 2x2 = 4 patches per merged token
    public int EmbeddingDim { get; } = 1280;
    public int ProjectionDim { get; } = 3584; // LLM hidden dim (e.g. 3584 for 7B, 2048 for 3B/2B)
    public int HeadCount { get; } = 16;
    public int LayerCount { get; } = 32;
    public int HeadDim => EmbeddingDim / HeadCount;
    public bool UseRmsNorm { get; } = true;
    public float Eps { get; } = 1e-6f;

    private bool _disposed;

    private QwenVlVisionModel(GgufModel gguf, string projectorType)
    {
        Gguf = gguf;
        ProjectorType = projectorType;

        // Ingest architecture metadata
        if (gguf.Metadata.TryGetValue("clip.vision.embedding_length", out var el) && el is int elInt)
            EmbeddingDim = elInt;
        else if (gguf.Metadata.TryGetValue("clip.embedding_length", out var el2) && el2 is int el2Int)
            EmbeddingDim = el2Int;

        if (gguf.Metadata.TryGetValue("clip.vision.projection_dim", out var pd) && pd is int pdInt)
            ProjectionDim = pdInt;
        else if (gguf.Metadata.TryGetValue("clip.projection_dim", out var pd2) && pd2 is int pd2Int)
            ProjectionDim = pd2Int;

        if (gguf.Metadata.TryGetValue("clip.vision.attention.head_count", out var hc) && hc is int hcInt)
            HeadCount = hcInt;
        else if (gguf.Metadata.TryGetValue("clip.attention.head_count", out var hc2) && hc2 is int hc2Int)
            HeadCount = hc2Int;

        if (gguf.Metadata.TryGetValue("clip.vision.block_count", out var bc) && bc is int bcInt)
            LayerCount = bcInt;
        else if (gguf.Metadata.TryGetValue("clip.block_count", out var bc2) && bc2 is int bc2Int)
            LayerCount = bc2Int;

        if (projectorType.Contains("qwen2vl", StringComparison.OrdinalIgnoreCase))
            UseRmsNorm = false;
        else
            UseRmsNorm = true;
    }

    public static QwenVlVisionModel Open(string path)
    {
        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
            throw new FileNotFoundException($"Qwen-VL vision model file not found: {path}");

        var gguf = GgufModel.Open(path);

        string projType = "qwen2.5vl";
        if (gguf.Metadata.TryGetValue("clip.vision.projector_type", out var ptObj) && ptObj is string ptStr)
            projType = ptStr.Trim().ToLowerInvariant();
        else if (gguf.Metadata.TryGetValue("clip.projector_type", out var ptObj2) && ptObj2 is string ptStr2)
            projType = ptStr2.Trim().ToLowerInvariant();

        return new QwenVlVisionModel(gguf, projType);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        Gguf.Dispose();
    }
}
