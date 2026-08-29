
namespace OpenTail.Stingray.Vision;

/// <summary>
/// Model container and metadata descriptor for MiniCPM-V 2.6 multimodal vision models.
/// </summary>
public sealed class MiniCpmVisionModel : IDisposable
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
    public int ResamplerQueryCount { get; }
    public float Eps { get; }

    private bool _disposed;

    private MiniCpmVisionModel(
        GgufModel gguf,
        string projectorType,
        int patchSize,
        int imageSize,
        int embeddingDim,
        int projectionDim,
        int layerCount,
        int headCount,
        int headDim,
        int resamplerQueryCount,
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
        ResamplerQueryCount = resamplerQueryCount;
        Eps = eps;
    }

    public static MiniCpmVisionModel Open(string path)
    {
        var gguf = GgufModel.Open(path);
        return FromGguf(gguf);
    }

    public static MiniCpmVisionModel FromGguf(GgufModel gguf)
    {
        string projType = "minicpmv";
        if (gguf.Metadata.TryGetValue("clip.vision.projector_type", out var ptObj) && ptObj is string ptStr)
            projType = ptStr.Trim().ToLowerInvariant();
        else if (gguf.Metadata.TryGetValue("clip.projector_type", out var ptObj2) && ptObj2 is string ptStr2)
            projType = ptStr2.Trim().ToLowerInvariant();

        int patchSize = GetInt(gguf, "clip.vision.patch_size", 14);
        int imageSize = GetInt(gguf, "clip.vision.image_size", 448);
        int embeddingDim = GetInt(gguf, "clip.vision.embedding_length", 1152);
        int projectionDim = GetInt(gguf, "clip.vision.projection_dim", 0);
        int layerCount = GetInt(gguf, "clip.vision.block_count", 27);
        int headCount = GetInt(gguf, "clip.vision.attention.head_count", 16);
        int headDim = headCount > 0 ? embeddingDim / headCount : 72;
        int queryCount = GetInt(gguf, "clip.vision.resampler_query_num", 64);
        float eps = GetFloat(gguf, "clip.vision.attention.layer_norm_epsilon", 1e-6f);

        // Fallback projectionDim from resampler.query or resampler.proj.weight
        if (projectionDim <= 0)
        {
            var qTensor = gguf.FindTensor("resampler.query") ?? gguf.FindTensor("mm.model.query");
            if (qTensor.HasValue)
            {
                projectionDim = (int)qTensor.Value.Dimensions[0];
                if (qTensor.Value.NDimensions > 1) queryCount = (int)qTensor.Value.Dimensions[1];
            }
            else
            {
                projectionDim = 3584;
            }
        }

        return new MiniCpmVisionModel(
            gguf,
            projType,
            patchSize,
            imageSize,
            embeddingDim,
            projectionDim,
            layerCount,
            headCount,
            headDim,
            queryCount,
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
