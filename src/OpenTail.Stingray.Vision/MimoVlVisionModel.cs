
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
    /// <summary>Window attention repeating pattern size (0 = no window attn). Local checkpoints
    /// under this class's name are actually the shared Qwen2.5VL-merger projector type (confirmed
    /// via `clip.projector_type=qwen2.5vl_merger` metadata, not the more elaborate real MIMOVL
    /// row/col-banded-sink projector this class's own doc comment describes) -- so windowing here
    /// uses the same simple n_wa_pattern spatial-window mechanism as Exaone4/QwenVl, gated on
    /// whichever of the two real reference graphs the checkpoint's own metadata actually selects.
    /// Real semantics (qwen2vl.cpp, shared): layer il gets FULL attention only when
    /// (il+1) % WindowAttnPattern == 0.</summary>
    public int WindowAttnPattern { get; }
    /// <summary>Window size in pixels (clip.vision.window_size, real default 112 if unset).</summary>
    public int WindowSize { get; }

    private bool _disposed;

    private MimoVlVisionModel(
        GgufModel gguf, string projectorType,
        int patchSize, int imageSize,
        int embeddingDim, int projectionDim,
        int layerCount, int headCount, int kvHeadCount, int headDim,
        int nMerge, float eps, int windowAttnPattern, int windowSize)
    {
        Gguf = gguf; ProjectorType = projectorType;
        PatchSize = patchSize; ImageSize = imageSize;
        EmbeddingDim = embeddingDim; ProjectionDim = projectionDim;
        LayerCount = layerCount; HeadCount = headCount;
        KvHeadCount = kvHeadCount; HeadDim = headDim;
        NMerge = nMerge; Eps = eps;
        WindowAttnPattern = windowAttnPattern; WindowSize = windowSize;
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
        // Real bug: this default was hardcoded to 8 (a GQA assumption), so any checkpoint with no
        // clip.vision.attention.head_count_kv metadata (confirmed absent in both real local
        // checkpoints under this class's name -- they are plain MHA, all Q/K/V share head_count)
        // silently got kvHeadCount=8 against a real attn_k/v.weight sized for the full head_count
        // (e.g. 16), causing MatVec to read only the first 8 heads' worth of a [1280,1280] weight
        // as if it were [640,1280] -- garbage K/V for every layer. Default to headCount instead,
        // matching the real reference's own kv_head_count-not-set-means-MHA convention.
        int kvHeadCount   = GetInt(gguf, "clip.vision.attention.head_count_kv", headCount);
        if (kvHeadCount <= 0) kvHeadCount = headCount;
        int nMerge        = GetInt(gguf, "clip.vision.n_merge", 2);
        if (nMerge <= 0) nMerge = 2;
        float eps = GetFloat(gguf, "clip.vision.attention.layer_norm_epsilon", 1e-6f);
        int windowAttnPattern = GetInt(gguf, "clip.vision.n_wa_pattern", 0);
        int windowSize = GetInt(gguf, "clip.vision.window_size", 112);

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
            layerCount, headCount, kvHeadCount, headDim, nMerge, eps, windowAttnPattern, windowSize);
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
