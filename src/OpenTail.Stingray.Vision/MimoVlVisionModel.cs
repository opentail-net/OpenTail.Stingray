
namespace OpenTail.Stingray.Vision;

/// <summary>
/// Model container for MiMo-VL (ByteDance MiMo-V2.5).
/// Architecture: Qwen2.5-VL-shaped ViT with GQA (32Q/8KV), per-head attention sinks,
/// row/col window attention pattern, dual Conv2D temporal patch embed, 2D M-RoPE,
/// SwiGLU MLP (with biases), post-LayerNorm, 4-patch spatial merge GELU MLP projector.
/// Reference: examples/llama.cpp/llama.cpp/tools/mtmd/models/mimovl.cpp
/// </summary>
public sealed class MimoVlVisionModel : IDisposable
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
    public int NMerge { get; }
    public float Eps { get; }

    private bool _disposed;

    private MimoVlVisionModel(
        GgufModel gguf, string projectorType,
        int patchSize, int imageSize,
        int embeddingDim, int projectionDim,
        int layerCount, int headCount, int kvHeadCount, int headDim,
        int nMerge, float eps)
    {
        Gguf = gguf; ProjectorType = projectorType;
        PatchSize = patchSize; ImageSize = imageSize;
        EmbeddingDim = embeddingDim; ProjectionDim = projectionDim;
        LayerCount = layerCount; HeadCount = headCount;
        KvHeadCount = kvHeadCount; HeadDim = headDim;
        NMerge = nMerge; Eps = eps;
    }

    public static MimoVlVisionModel Open(string path) => FromGguf(GgufModel.Open(path));

    public static MimoVlVisionModel FromGguf(GgufModel gguf)
    {
        string projType = "mimovl";
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
        int nMerge        = GetInt(gguf, "clip.vision.n_merge", 2);
        if (nMerge <= 0) nMerge = 2;
        float eps = GetFloat(gguf, "clip.vision.attention.layer_norm_epsilon", 1e-6f);

        // head_dim derived from fused QKV weight (rows = (nQ + 2*nKV)*headDim)
        int headDim = headCount > 0 ? embeddingDim / headCount : 40;
        var qkvTensor = gguf.FindTensor("v.blk.0.attn_qkv.weight");
        if (qkvTensor.HasValue)
        {
            long qkvRows = qkvTensor.Value.Dimensions[1];
            headDim = (int)(qkvRows / (headCount + 2 * kvHeadCount));
        }

        if (projectionDim <= 0)
        {
            var t = gguf.FindTensor("mm.1.weight") ?? gguf.FindTensor("mm.0.weight");
            projectionDim = t.HasValue ? (int)t.Value.Dimensions[1] : 4096;
        }

        return new MimoVlVisionModel(
            gguf, projType, patchSize, imageSize, embeddingDim, projectionDim,
            layerCount, headCount, kvHeadCount, headDim, nMerge, eps);
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
