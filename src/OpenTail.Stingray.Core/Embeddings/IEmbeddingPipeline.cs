namespace OpenTail.Stingray.Core.Embeddings;

/// <summary>
/// Common interface for text embedding models producing dense semantic vectors.
/// </summary>
public interface IEmbeddingPipeline : IDisposable
{
    string ModelName { get; }
    int EmbeddingDimensions { get; }
    PoolingType DefaultPooling { get; }

    /// <summary>
    /// Generates dense embedding vectors for the requested input texts.
    /// </summary>
    EmbeddingResult Embed(EmbeddingRequest request);
}

public sealed record EmbeddingRequest
{
    public required IReadOnlyList<string> Inputs { get; init; }
    public string? Model { get; init; }
    public int? Dimensions { get; init; }
    public bool Normalize { get; init; } = true;
    public PoolingType? Pooling { get; init; }
    public string EncodingFormat { get; init; } = "float";
}

public sealed record EmbeddingData
{
    public int Index { get; init; }
    public required float[] Vector { get; init; }
}

public sealed class EmbeddingResult
{
    public string Model { get; }
    public IReadOnlyList<EmbeddingData> Data { get; }
    public int PromptTokens { get; }
    public int TotalTokens { get; }

    public EmbeddingResult(
        string model,
        IReadOnlyList<EmbeddingData> data,
        int promptTokens,
        int totalTokens)
    {
        Model = model;
        Data = data;
        PromptTokens = promptTokens;
        TotalTokens = totalTokens;
    }
}
