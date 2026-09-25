namespace OpenTail.Stingray.Server.Endpoints;

/// <summary>
/// Picks the model path for <c>/v1/embeddings</c> and <c>/v1/rerank</c>: the request's <c>model</c> when it names
/// a checkpoint directory or GGUF file on disk, else the server default from the environment
/// (<c>STINGRAY_EMBEDDING_MODEL</c> / <c>STINGRAY_RERANK_MODEL</c>). Null means nothing real is configured; the
/// endpoints then answer 404 instead of producing synthetic vectors or scores.
/// </summary>
internal static class EncoderModelResolver
{
    public const string EmbeddingEnv = "STINGRAY_EMBEDDING_MODEL";
    public const string RerankEnv = "STINGRAY_RERANK_MODEL";

    public static string? Resolve(string? requested, string envVar)
    {
        if (!string.IsNullOrWhiteSpace(requested) && (Directory.Exists(requested) || File.Exists(requested)))
            return requested;
        string? fallback = Environment.GetEnvironmentVariable(envVar);
        return !string.IsNullOrWhiteSpace(fallback) && (Directory.Exists(fallback) || File.Exists(fallback)) ? fallback : null;
    }

    public static string NotFoundJson(string? requested, string envVar)
    {
        string message = $"No model at '{requested}' and {envVar} is not set to an existing checkpoint directory or GGUF file.";
        return "{\"error\":{\"message\":\"" + JsonEncodedText.Encode(message).Value + "\",\"type\":\"model_not_found\"}}";
    }
}
