
namespace OpenTail.Stingray.Vision;

/// <summary>
/// Model container and metadata parser for IBM Granite 4 Vision models.
/// Reference: examples/llama.cpp/llama.cpp/tools/mtmd/models/granite4-vision.cpp
/// </summary>
public sealed class Granite4VisionModel : IDisposable
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
    public float Eps { get; }

    /// <summary>
    /// Per-WindowQFormer-block index into the SigLIP tower's saved intermediate layer outputs
    /// (clip.vision.feature_layer, e.g. [26,20,14,8,26,26,26,26] for the real 3B checkpoint).
    /// Length is the number of projector blocks K.
    /// </summary>
    public int[] FeatureLayers { get; }

    /// <summary>
    /// Per-block downsampling mode selector (clip.vision.projector.spatial_offsets):
    /// -1 selects average-pool downsampling (interp_down), >=0 selects the indexed
    /// 2x2-checkerboard-phase gather (spatial_idx) with that phase offset.
    /// </summary>
    public int[] ProjSpatialOffsets { get; }

    /// <summary>Window side for the raster-to-window gather (clip.vision.projector.window_side).</summary>
    public int WindowSide { get; }

    /// <summary>Query-window side for the QFormer's learned query grid (clip.vision.projector.query_side).</summary>
    public int QuerySide { get; }

    private bool _disposed;

    private Granite4VisionModel(
        GgufModel gguf,
        string projectorType,
        int patchSize,
        int imageSize,
        int embeddingDim,
        int projectionDim,
        int layerCount,
        int headCount,
        int headDim,
        float eps,
        int[] featureLayers,
        int[] projSpatialOffsets,
        int windowSide,
        int querySide)
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
        Eps = eps;
        FeatureLayers = featureLayers;
        ProjSpatialOffsets = projSpatialOffsets;
        WindowSide = windowSide;
        QuerySide = querySide;
    }

    public static Granite4VisionModel Open(string path)
    {
        var gguf = GgufModel.Open(path);
        return FromGguf(gguf);
    }

    public static Granite4VisionModel FromGguf(GgufModel gguf)
    {
        string projType = "granite4-vision";
        if (gguf.Metadata.TryGetValue("clip.vision.projector_type", out var ptObj) && ptObj is string ptStr)
            projType = ptStr.Trim().ToLowerInvariant();

        int patchSize = GetInt(gguf, "clip.vision.patch_size", 14);
        int imageSize = GetInt(gguf, "clip.vision.image_size", 384);
        int embeddingDim = GetInt(gguf, "clip.vision.embedding_length", 1152);
        int projectionDim = GetInt(gguf, "clip.vision.projection_dim", 2048);
        int layerCount = GetInt(gguf, "clip.vision.block_count", 27);
        int headCount = GetInt(gguf, "clip.vision.attention.head_count", 16);
        int headDim = headCount > 0 ? embeddingDim / headCount : 72;
        float eps = GetFloat(gguf, "clip.vision.attention.layer_norm_epsilon", 1e-6f);

        // clip.vision.feature_layer: which SigLIP intermediate layer each WindowQFormer block
        // reads from (KEY_FEATURE_LAYERS = "clip.%s.feature_layer" with prefix "vision", see
        // examples/llama.cpp/llama.cpp/tools/mtmd/clip-impl.h line 47). Length is K, the number
        // of projector blocks. Falls back to a single block reading the final layer if absent
        // (keeps this constructible against non-standard/older exports rather than throwing).
        int[] featureLayers = GetIntArray(gguf, "clip.vision.feature_layer", [layerCount - 1]);
        int[] projSpatialOffsets = GetIntArray(gguf, "clip.vision.projector.spatial_offsets", new int[featureLayers.Length]);
        int windowSide = GetInt(gguf, "clip.vision.projector.window_side", 8);
        int querySide = GetInt(gguf, "clip.vision.projector.query_side", 4);
        if (projSpatialOffsets.Length != featureLayers.Length)
        {
            var fill = new int[featureLayers.Length];
            Array.Fill(fill, -1);
            projSpatialOffsets = fill;
        }

        return new Granite4VisionModel(
            gguf,
            projType,
            patchSize,
            imageSize,
            embeddingDim,
            projectionDim,
            layerCount,
            headCount,
            headDim,
            eps,
            featureLayers,
            projSpatialOffsets,
            windowSide,
            querySide);
    }

    private static int[] GetIntArray(GgufModel g, string key, int[] fallback)
    {
        if (g.Metadata.TryGetValue(key, out var val) && val is object[] arr)
        {
            var result = new int[arr.Length];
            for (int i = 0; i < arr.Length; i++) result[i] = Convert.ToInt32(arr[i]);
            return result;
        }
        return fallback;
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
