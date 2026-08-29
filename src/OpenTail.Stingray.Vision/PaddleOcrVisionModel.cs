
namespace OpenTail.Stingray.Vision;

/// <summary>
/// Model container and metadata parser for PaddleOCR (PP-OCR) vision models.
/// Reference: examples/llama.cpp/llama.cpp/tools/mtmd/models/paddleocr.cpp
/// </summary>
public sealed class PaddleOcrVisionModel : IDisposable
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
    public int MergeFactor { get; }
    public float Eps { get; }

    private bool _disposed;

    private PaddleOcrVisionModel(
        GgufModel gguf,
        string projectorType,
        int patchSize,
        int imageSize,
        int embeddingDim,
        int projectionDim,
        int layerCount,
        int headCount,
        int headDim,
        int mergeFactor,
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
        MergeFactor = mergeFactor;
        Eps = eps;
    }

    public static PaddleOcrVisionModel Open(string path)
    {
        var gguf = GgufModel.Open(path);
        return FromGguf(gguf);
    }

    public static PaddleOcrVisionModel FromGguf(GgufModel gguf)
    {
        string projType = "paddleocr";
        if (gguf.Metadata.TryGetValue("clip.vision.projector_type", out var ptObj) && ptObj is string ptStr)
            projType = ptStr.Trim().ToLowerInvariant();
        else if (gguf.Metadata.TryGetValue("clip.projector_type", out var ptObj2) && ptObj2 is string ptStr2)
            projType = ptStr2.Trim().ToLowerInvariant();

        int patchSize = GetInt(gguf, "clip.vision.patch_size", 14);
        int imageSize = GetInt(gguf, "clip.vision.image_size", 768);
        int embeddingDim = GetInt(gguf, "clip.vision.embedding_length", 1024);
        int projectionDim = GetInt(gguf, "clip.vision.projection_dim", 0);
        int layerCount = GetInt(gguf, "clip.vision.block_count", 24);
        int headCount = GetInt(gguf, "clip.vision.attention.head_count", 16);
        int headDim = headCount > 0 ? embeddingDim / headCount : 64;
        int mergeFactor = GetInt(gguf, "clip.vision.n_merge", 2);
        float eps = GetFloat(gguf, "clip.vision.attention.layer_norm_epsilon", 1e-6f);

        if (projectionDim <= 0)
        {
            var pTensor = gguf.FindTensor("mm.2.weight") ?? gguf.FindTensor("mm.fc.weight") ?? gguf.FindTensor("mm.0.weight");
            if (pTensor.HasValue)
            {
                projectionDim = (int)pTensor.Value.Dimensions[1];
            }
            else
            {
                projectionDim = 2048;
            }
        }

        return new PaddleOcrVisionModel(
            gguf,
            projType,
            patchSize,
            imageSize,
            embeddingDim,
            projectionDim,
            layerCount,
            headCount,
            headDim,
            mergeFactor,
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
