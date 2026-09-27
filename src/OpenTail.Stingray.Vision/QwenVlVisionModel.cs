
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
    /// <summary>Window attention repeating pattern size (0 = no window attn, real default for
    /// Qwen2.5-VL if unset is required=true in the real reference -- only Qwen2.5-VL of this
    /// family uses windowing; Qwen2-VL/Qwen3-VL do not). Real semantics (qwen2vl.cpp): layer il
    /// gets FULL attention only when (il+1) % WindowAttnPattern == 0.</summary>
    public int WindowAttnPattern { get; }
    /// <summary>Window size in pixels (clip.vision.window_size, real default 112 if unset).</summary>
    public int WindowSize { get; } = 112;
    public int MergeRatio => SpatialMergeFactor;

    /// <summary>Qwen3-VL (<c>qwen3vl_merger</c>): learned position grid resized to the image, GELU FFN, and
    /// deepstack branches whose outputs are appended to each token (llama.cpp tools/mtmd/models/qwen3vl.cpp).</summary>
    public bool IsQwen3Vl { get; }

    /// <summary>Vision layers with a deepstack branch (<c>v.deepstack.{il}.*</c>), in layer order.</summary>
    public IReadOnlyList<int> DeepstackLayers => Enumerable.Range(0, LayerCount)
        .Where(l => Gguf.FindTensor($"v.deepstack.{l}.fc1.weight").HasValue).ToArray();

    /// <summary>Width of one output token: the projection plus one projection-sized slice per deepstack layer.</summary>
    public int OutputTokenDim => ProjectionDim * (1 + DeepstackLayers.Count);

    private bool _disposed;

    private QwenVlVisionModel(GgufModel gguf, string projectorType)
    {
        Gguf = gguf;
        ProjectorType = projectorType;

        // Ingest architecture metadata. GGUF u32 values box as uint, so read them through GetIntWiden: an `is int`
        // check silently kept the defaults (ProjectionDim 3584, found 2026-09-27 on Qwen3-VL-2B whose projection is 2048).
        EmbeddingDim = GetIntWiden(gguf, "clip.vision.embedding_length", GetIntWiden(gguf, "clip.embedding_length", EmbeddingDim));
        ProjectionDim = GetIntWiden(gguf, "clip.vision.projection_dim", GetIntWiden(gguf, "clip.projection_dim", ProjectionDim));
        HeadCount = GetIntWiden(gguf, "clip.vision.attention.head_count", GetIntWiden(gguf, "clip.attention.head_count", HeadCount));
        LayerCount = GetIntWiden(gguf, "clip.vision.block_count", GetIntWiden(gguf, "clip.block_count", LayerCount));

        IsQwen3Vl = projectorType.Contains("qwen3vl", StringComparison.OrdinalIgnoreCase);
        PatchSize = GetIntWiden(gguf, "clip.vision.patch_size", PatchSize);
        SpatialMergeFactor = GetIntWiden(gguf, "clip.vision.spatial_merge_size", SpatialMergeFactor);

        // Qwen2-VL and Qwen3-VL use LayerNorm with bias; Qwen2.5-VL uses RMSNorm.
        if (projectorType.Contains("qwen2vl", StringComparison.OrdinalIgnoreCase) || IsQwen3Vl)
            UseRmsNorm = false;
        else
            UseRmsNorm = true;

        WindowAttnPattern = GetIntWiden(gguf, "clip.vision.n_wa_pattern", 0);
        WindowSize = GetIntWiden(gguf, "clip.vision.window_size", 112);
    }

    private static int GetIntWiden(GgufModel gguf, string key, int def)
    {
        if (gguf.Metadata.TryGetValue(key, out var v))
        {
            if (v is int i) return i;
            if (v is uint u) return (int)u;
            if (v is long l) return (int)l;
            if (v is ulong ul) return (int)ul;
        }
        return def;
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
