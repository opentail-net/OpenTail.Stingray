using OpenTail.Stingray.Core.Embeddings;

namespace OpenTail.Stingray.Server.Endpoints;

public static class OpenAiEmbeddingEndpoints
{
    public static IEndpointRouteBuilder MapOpenAiEmbeddingEndpoints(this IEndpointRouteBuilder app)
    {
        // OpenAI: POST /v1/embeddings {"model","input": string | string[]}
        app.MapPost("/v1/embeddings", async (HttpContext ctx) =>
        {
            var parsed = await ReadRequestAsync(ctx, openAiShape: true);
            if (parsed is null) return;
            var (req, texts) = parsed.Value;

            var result = await RunAsync(ctx, req, texts);
            if (result is null) return;

            var responseObj = new EmbeddingApiResponse
            {
                Object = "list",
                Model = result.Model,
                Data = result.Data.Select(d => new EmbeddingItemResponse
                {
                    Object = "embedding",
                    Index = d.Index,
                    Embedding = d.Vector
                }).ToList(),
                Usage = new EmbeddingUsageResponse
                {
                    PromptTokens = result.PromptTokens,
                    TotalTokens = result.TotalTokens
                }
            };

            ctx.Response.StatusCode = 200;
            ctx.Response.ContentType = "application/json";
            await JsonSerializer.SerializeAsync(ctx.Response.Body, responseObj, cancellationToken: ctx.RequestAborted);
        });

        // Ollama: POST /api/embed {"model","input": string | string[]} -> {"model","embeddings":[[...]],...}
        app.MapPost("/api/embed", async (HttpContext ctx) =>
        {
            var parsed = await ReadRequestAsync(ctx, openAiShape: false);
            if (parsed is null) return;
            var (req, texts) = parsed.Value;

            long started = System.Diagnostics.Stopwatch.GetTimestamp();
            var result = await RunAsync(ctx, req, texts);
            if (result is null) return;

            var responseObj = new OllamaEmbedResponse
            {
                Model = result.Model,
                Embeddings = result.Data.OrderBy(d => d.Index).Select(d => d.Vector).ToList(),
                TotalDuration = (long)(System.Diagnostics.Stopwatch.GetElapsedTime(started).TotalMilliseconds * 1_000_000d),
                PromptEvalCount = result.PromptTokens
            };

            ctx.Response.StatusCode = 200;
            ctx.Response.ContentType = "application/json";
            await JsonSerializer.SerializeAsync(ctx.Response.Body, responseObj, cancellationToken: ctx.RequestAborted);
        });

        return app;
    }

    /// <summary>
    /// Parses the <c>input</c> field: a string or an array of strings (OpenAI and Ollama both allow either).
    /// Empty strings, empty arrays and non-string array items (token-id arrays are not supported) are rejected,
    /// never silently dropped.
    /// </summary>
    public static bool TryParseInput(JsonElement input, out List<string> texts, out string error)
    {
        texts = [];
        error = "";
        switch (input.ValueKind)
        {
            case JsonValueKind.String:
                texts.Add(input.GetString() ?? "");
                break;
            case JsonValueKind.Array:
                foreach (var item in input.EnumerateArray())
                {
                    if (item.ValueKind != JsonValueKind.String)
                    {
                        error = "'input' array items must be strings (token-id arrays are not supported).";
                        return false;
                    }
                    texts.Add(item.GetString() ?? "");
                }
                break;
            default:
                error = "'input' field is required and must be a string or an array of strings.";
                return false;
        }

        if (texts.Count == 0) { error = "'input' must contain at least one string."; return false; }
        if (texts.Any(string.IsNullOrEmpty)) { error = "'input' must not contain empty strings."; return false; }
        return true;
    }

    private static async Task<(EmbeddingApiRequest Req, List<string> Texts)?> ReadRequestAsync(HttpContext ctx, bool openAiShape)
    {
        EmbeddingApiRequest? req;
        try
        {
            req = await JsonSerializer.DeserializeAsync<EmbeddingApiRequest>(ctx.Request.Body, cancellationToken: ctx.RequestAborted);
        }
        catch (JsonException)
        {
            await WriteErrorAsync(ctx, 400, "Invalid JSON request body", openAiShape);
            return null;
        }

        if (req is null)
        {
            await WriteErrorAsync(ctx, 400, "Request body is required.", openAiShape);
            return null;
        }

        // `inputs` is a non-standard alias kept for existing callers.
        JsonElement input = req.Input.ValueKind != JsonValueKind.Undefined ? req.Input : req.Inputs;
        if (!TryParseInput(input, out var texts, out var error))
        {
            await WriteErrorAsync(ctx, 400, error, openAiShape);
            return null;
        }
        return (req, texts);
    }

    private static async Task<EmbeddingResult?> RunAsync(HttpContext ctx, EmbeddingApiRequest req, List<string> texts)
    {
        string? modelPath = EncoderModelResolver.Resolve(req.Model, EncoderModelResolver.EmbeddingEnv);
        if (modelPath is null)
        {
            ctx.Response.StatusCode = 404;
            ctx.Response.ContentType = "application/json";
            await ctx.Response.WriteAsync(EncoderModelResolver.NotFoundJson(req.Model, EncoderModelResolver.EmbeddingEnv));
            return null;
        }

        var engine = Engine.Encoders.EncoderPipelineFactory.GetSharedEmbedding(modelPath);

        var embedReq = new EmbeddingRequest
        {
            Inputs = texts,
            Model = modelPath,
            Dimensions = req.Dimensions,
            Normalize = true,
            EncodingFormat = req.EncodingFormat ?? "float"
        };

        lock (engine) return engine.Embed(embedReq); // the GGUF forward pass is not re-entrant
    }

    private static async Task WriteErrorAsync(HttpContext ctx, int status, string message, bool openAiShape)
    {
        ctx.Response.StatusCode = status;
        ctx.Response.ContentType = "application/json";
        string escaped = JsonSerializer.Serialize(message);
        await ctx.Response.WriteAsync(openAiShape
            ? $"{{\"error\":{{\"message\":{escaped},\"type\":\"invalid_request_error\"}}}}"
            : $"{{\"error\":{escaped}}}");
    }
}

public sealed record EmbeddingApiRequest
{
    [JsonPropertyName("model")]
    public string? Model { get; init; }

    /// <summary>A string or an array of strings; parsed by <see cref="OpenAiEmbeddingEndpoints.TryParseInput"/>.</summary>
    [JsonPropertyName("input")]
    public JsonElement Input { get; init; }

    [JsonPropertyName("inputs")]
    public JsonElement Inputs { get; init; }

    [JsonPropertyName("dimensions")]
    public int? Dimensions { get; init; }

    [JsonPropertyName("encoding_format")]
    public string? EncodingFormat { get; init; } = "float";

    [JsonPropertyName("user")]
    public string? User { get; init; }
}

public sealed record EmbeddingApiResponse
{
    [JsonPropertyName("object")]
    public string Object { get; init; } = "list";

    [JsonPropertyName("model")]
    public string Model { get; init; } = "text-embedding-3-small";

    [JsonPropertyName("data")]
    public List<EmbeddingItemResponse> Data { get; init; } = [];

    [JsonPropertyName("usage")]
    public EmbeddingUsageResponse Usage { get; init; } = new();
}

public sealed record EmbeddingItemResponse
{
    [JsonPropertyName("object")]
    public string Object { get; init; } = "embedding";

    [JsonPropertyName("index")]
    public int Index { get; init; }

    [JsonPropertyName("embedding")]
    public float[] Embedding { get; init; } = [];
}

public sealed record EmbeddingUsageResponse
{
    [JsonPropertyName("prompt_tokens")]
    public int PromptTokens { get; init; }

    [JsonPropertyName("total_tokens")]
    public int TotalTokens { get; init; }
}

public sealed record OllamaEmbedResponse
{
    [JsonPropertyName("model")]
    public string Model { get; init; } = "";

    [JsonPropertyName("embeddings")]
    public List<float[]> Embeddings { get; init; } = [];

    [JsonPropertyName("total_duration")]
    public long TotalDuration { get; init; }

    [JsonPropertyName("prompt_eval_count")]
    public int PromptEvalCount { get; init; }
}
