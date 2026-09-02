
namespace OpenTail.Stingray.Vision;

/// <summary>
/// Model container for Tencent HunyuanVL multimodal vision.
/// Architecture: SigLIP/CLIP ViT + RMSNorm + Conv2D perceiver projector + LLM-wrap tokens.
/// Reference: examples/llama.cpp/llama.cpp/tools/mtmd/models/hunyuanvl.cpp
/// </summary>
public sealed class HunyuanVlVisionModel : IDisposable
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
    /// <summary>Spatial merge factor (conv2d stride, default 2).</summary>
    public int NMerge { get; }
    public float Eps { get; }

    private bool _disposed;

    private HunyuanVlVisionModel(
        GgufModel gguf,
        string projectorType,
        int patchSize,
        int imageSize,
        int embeddingDim,
        int projectionDim,
        int layerCount,
        int headCount,
        int headDim,
        int nMerge,
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
        NMerge = nMerge;
        Eps = eps;
    }

    public static HunyuanVlVisionModel Open(string path)
    {
        var gguf = GgufModel.Open(path);
        return FromGguf(gguf);
    }

    public static HunyuanVlVisionModel FromGguf(GgufModel gguf)
    {
        string projType = "hunyuanvl";
        if (gguf.Metadata.TryGetValue("clip.vision.projector_type", out var ptObj) && ptObj is string ptStr)
            projType = ptStr.Trim().ToLowerInvariant();

        int patchSize     = GetInt(gguf, "clip.vision.patch_size", 14);
        int imageSize     = GetInt(gguf, "clip.vision.image_size", 378);
        int embeddingDim  = GetInt(gguf, "clip.vision.embedding_length", 1152);
        int projectionDim = GetInt(gguf, "clip.vision.projection_dim", 0);
        int layerCount    = GetInt(gguf, "clip.vision.block_count", 27);
        int headCount     = GetInt(gguf, "clip.vision.attention.head_count", 16);
        int headDim       = headCount > 0 ? embeddingDim / headCount : 72;
        int nMerge        = GetInt(gguf, "clip.vision.n_merge", 2);
        if (nMerge <= 0) nMerge = 2;
        float eps = GetFloat(gguf, "clip.vision.attention.layer_norm_epsilon", 1e-6f);

        if (projectionDim <= 0)
        {
            // Infer from mm.model.fc weight dimensions (real GGUF tensor name).
            var pTensor = gguf.FindTensor("mm.model.fc.weight");
            projectionDim = pTensor.HasValue ? (int)pTensor.Value.Dimensions[1] : 4096;
        }

        return new HunyuanVlVisionModel(
            gguf, projType, patchSize, imageSize, embeddingDim, projectionDim,
            layerCount, headCount, headDim, nMerge, eps);
    }

    private static int GetInt(GgufModel gguf, string key, int defaultValue)
    {
        if (gguf.Metadata.TryGetValue(key, out var val))
        {
            if (val is int i)   return i;
            if (val is uint u)  return (int)u;
            if (val is long l)  return (int)l;
            if (val is ulong ul) return (int)ul;
        }
        return defaultValue;
    }

    private static float GetFloat(GgufModel gguf, string key, float defaultValue)
    {
        if (gguf.Metadata.TryGetValue(key, out var val))
        {
            if (val is float f)  return f;
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
