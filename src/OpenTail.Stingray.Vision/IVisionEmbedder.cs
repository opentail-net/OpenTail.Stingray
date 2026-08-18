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
    /// Loads an image file from disk, preprocesses it, and runs the vision encoder.
    /// </summary>
    float[] EmbedImageFile(string filePath, out int tokenCount);
}
