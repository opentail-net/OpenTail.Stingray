
namespace OpenTail.Stingray.Vision;

/// <summary>
/// Model container and metadata descriptor for Moonshot AI Kimi K2.5 and Kimi-VL multimodal vision models.
/// </summary>
public sealed class KimiVisionModel : IDisposable
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
    public float RopeTheta { get; }
    public float Eps { get; }

    private bool _disposed;

    private KimiVisionModel(
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
        float ropeTheta,
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
        RopeTheta = ropeTheta;
        Eps = eps;
    }

    public static KimiVisionModel Open(string path)
    {
        var gguf = GgufModel.Open(path);
        return FromGguf(gguf);
    }

    public static KimiVisionModel FromGguf(GgufModel gguf)
    {
        string projType = "kimik25";
        if (gguf.Metadata.TryGetValue("clip.vision.projector_type", out var ptObj) && ptObj is string ptStr)
            projType = ptStr.Trim().ToLowerInvariant();
        else if (gguf.Metadata.TryGetValue("clip.projector_type", out var ptObj2) && ptObj2 is string ptStr2)
            projType = ptStr2.Trim().ToLowerInvariant();

        int patchSize = GetInt(gguf, "clip.vision.patch_size", 14);
        int imageSize = GetInt(gguf, "clip.vision.image_size", 896);
        int embeddingDim = GetInt(gguf, "clip.vision.embedding_length", 1152);
        int projectionDim = GetInt(gguf, "clip.vision.projection_dim", 0);
        int layerCount = GetInt(gguf, "clip.vision.block_count", 27);
        int headCount = GetInt(gguf, "clip.vision.attention.head_count", 16);
        int headDim = headCount > 0 ? embeddingDim / headCount : 72;
        int mergeFactor = GetInt(gguf, "clip.vision.merge_factor", 2);
        if (mergeFactor <= 0) mergeFactor = 2;
        float ropeTheta = GetFloat(gguf, "clip.vision.rope.freq_base", 10000.0f);
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

        return new KimiVisionModel(
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
            ropeTheta,
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
