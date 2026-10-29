namespace OpenTail.Stingray.Server.Endpoints;

/// <summary>
/// Picks the model path for <c>/v1/embeddings</c> and <c>/v1/rerank</c>: the server default from the environment
/// (<c>STINGRAY_EMBEDDING_MODEL</c> / <c>STINGRAY_RERANK_MODEL</c>). Null means nothing real is configured; the
/// endpoints then answer 404 instead of producing synthetic vectors or scores.
/// <para>
/// A request's <c>model</c> is treated as a filesystem path only when the operator opts in with
/// <c>STINGRAY_ENCODER_REQUEST_PATHS=1</c>. Without it, the <c>model</c> field is a name like any OpenAI client
/// sends and the configured default answers. The opt-in exists because a request path is loaded and kept resident
/// for the life of the process: any client could otherwise make the server load any GGUF or checkpoint on its disk
/// (each one staying in memory), and probe which paths exist from the differing responses.
/// </para>
/// </summary>
internal static class EncoderModelResolver
{
    public const string EmbeddingEnv = "STINGRAY_EMBEDDING_MODEL";
    public const string RerankEnv = "STINGRAY_RERANK_MODEL";
    public const string RequestPathsEnv = "STINGRAY_ENCODER_REQUEST_PATHS";

    /// <summary>True when the operator allows requests to name a model path (<c>1</c> or <c>true</c>).</summary>
    public static bool RequestPathsAllowed
    {
        get
        {
            string? raw = Environment.GetEnvironmentVariable(RequestPathsEnv)?.Trim();
            return string.Equals(raw, "1", StringComparison.Ordinal)
                || string.Equals(raw, "true", StringComparison.OrdinalIgnoreCase);
        }
    }

    public static string? Resolve(string? requested, string envVar)
    {
        if (RequestPathsAllowed && !string.IsNullOrWhiteSpace(requested)
            && (Directory.Exists(requested) || File.Exists(requested)))
            return requested;
        string? fallback = Environment.GetEnvironmentVariable(envVar);
        return !string.IsNullOrWhiteSpace(fallback) && (Directory.Exists(fallback) || File.Exists(fallback)) ? fallback : null;
    }

    public static string NotFoundJson(string? requested, string envVar)
    {
        string message = RequestPathsAllowed
            ? $"No model at '{requested}' and {envVar} is not set to an existing checkpoint directory or GGUF file."
            : $"No model for '{requested}': {envVar} is not set to an existing checkpoint directory or GGUF file " +
              $"(request model paths are disabled; set {RequestPathsEnv}=1 to allow them).";
        return "{\"error\":{\"message\":\"" + JsonEncodedText.Encode(message).Value + "\",\"type\":\"model_not_found\"}}";
    }
}
