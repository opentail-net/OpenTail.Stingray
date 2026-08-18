namespace OpenTail.Stingray.Core.Embeddings;

/// <summary>
/// Common interface for cross-encoder reranking models that score query-document relevance.
/// </summary>
public interface IRerankerPipeline : IDisposable
{
    string ModelName { get; }

    /// <summary>
    /// Scores and ranks candidate documents by relevance to the query.
    /// </summary>
    RerankResult Rerank(RerankRequest request);
}

public sealed record RerankRequest
{
    public required string Query { get; init; }
    public required IReadOnlyList<string> Documents { get; init; }
    public int? TopN { get; init; }
    public string? Model { get; init; }
    public bool ReturnDocuments { get; init; } = true;
}

public sealed record RerankDocumentResult
{
    public int Index { get; init; }
    public float RelevanceScore { get; init; }
    public string? Document { get; init; }
}

public sealed class RerankResult
{
    public string Model { get; }
    public IReadOnlyList<RerankDocumentResult> Results { get; }
    public int TotalTokens { get; }

    public RerankResult(
        string model,
        IReadOnlyList<RerankDocumentResult> results,
        int totalTokens)
    {
        Model = model;
        Results = results;
        TotalTokens = totalTokens;
    }
}
