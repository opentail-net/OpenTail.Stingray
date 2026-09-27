namespace OpenTail.Stingray.Vision;

/// <summary>
/// Unified abstraction for multimodal vision embedders that turn preprocessed
/// image pixels into LLM soft-token embeddings for decoder injection.
/// </summary>
public interface IVisionEmbedder : IDisposable
{
    /// <summary>Canonical projector type name (e.g., "gemma4uv", "gemma4v", "gemma3", "llama4").</summary>
    string ProjectorType { get; }

    /// <summary>Dimensionality of the produced soft tokens (must match LLM embedding dim).</summary>
    int EmbeddingDim { get; }

    /// <summary>Native/standard input image width for this model.</summary>
    int ImageWidth { get; }

    /// <summary>Native/standard input image height for this model.</summary>
    int ImageHeight { get; }

    /// <summary>Special text token / sequence placed immediately before the image soft tokens.</summary>
    string ImageOpenMarker { get; }

    /// <summary>Special text token / sequence placed immediately after the image soft tokens.</summary>
    string ImageCloseMarker { get; }

    /// <summary>Placeholder token in user text prompt that gets replaced by the image tokens.</summary>
    string PlaceholderMarker { get; }

    /// <summary>
    /// Projects planar RGB pixel bytes into a contiguous block of soft-token vectors (total length: tokenCount * EmbeddingDim).
    /// </summary>
    float[] EmbedImage(ReadOnlySpan<byte> rgb, int width, int height, out int tokenCount);

    /// <summary>
    /// Token grid (columns x rows) of the most recent <see cref="EmbedImage"/> call, for text decoders that
    /// give image tokens 2D M-RoPE positions (Qwen2-VL family, PaddleOCR-VL). (0, 0) when not tracked.
    /// </summary>
    (int Width, int Height) LastTokenGrid => (0, 0);

    /// <summary>
    /// For images that expand to several views wrapped in their own marker tokens (Step3-VL: crops as
    /// <c>&lt;patch_start&gt; .. &lt;patch_end&gt;</c> with <c>&lt;patch_newline&gt;</c> between rows, then the overview
    /// in <c>&lt;im_start&gt; .. &lt;im_end&gt;</c>): the full ordered sequence from the most recent
    /// <see cref="EmbedImage"/> call. When non-null the caller feeds exactly this and ignores
    /// <see cref="ImageOpenMarker"/>/<see cref="ImageCloseMarker"/>. Null for single-block embedders.
    /// </summary>
    IReadOnlyList<VisionSegment>? LastSegments => null;

    /// <summary>
    /// Loads an image file from disk, preprocesses it, and runs the vision encoder.
    /// </summary>
    float[] EmbedImageFile(string filePath, out int tokenCount);
}

/// <summary>One element of <see cref="IVisionEmbedder.LastSegments"/>: either a special token (by its text)
/// or a run of <paramref name="Count"/> soft tokens starting at soft-token index <paramref name="Start"/>
/// of the array returned by <see cref="IVisionEmbedder.EmbedImage"/>.</summary>
public readonly record struct VisionSegment(string? Token, int Start, int Count);
