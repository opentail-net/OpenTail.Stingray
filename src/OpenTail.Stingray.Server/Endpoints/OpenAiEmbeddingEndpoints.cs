using OpenTail.Stingray.Core.Embeddings;

namespace OpenTail.Stingray.Server.Endpoints;

public static class OpenAiEmbeddingEndpoints
{
    public static IEndpointRouteBuilder MapOpenAiEmbeddingEndpoints(this IEndpointRouteBuilder app)
    {
        app.MapPost("/v1/embeddings", async (HttpContext ctx) =>
        {
            EmbeddingApiRequest? req;
            try
            {
                req = await JsonSerializer.DeserializeAsync<EmbeddingApiRequest>(ctx.Request.Body, cancellationToken: ctx.RequestAborted);
            }
            catch (JsonException)
            {
                ctx.Response.StatusCode = 400;
                ctx.Response.ContentType = "application/json";
                await ctx.Response.WriteAsync("{\"error\":{\"message\":\"Invalid JSON request body\",\"type\":\"invalid_request_error\"}}");
                return;
            }

            if (req is null || (req.Input is null && (req.Inputs is null || req.Inputs.Count == 0)))
            {
                ctx.Response.StatusCode = 400;
                ctx.Response.ContentType = "application/json";
                await ctx.Response.WriteAsync("{\"error\":{\"message\":\"'input' field is required.\",\"type\":\"invalid_request_error\"}}");
                return;
            }

            // Normalize input to list of strings
            List<string> inputTexts = [];
            if (!string.IsNullOrEmpty(req.Input))
            {
                inputTexts.Add(req.Input);
            }
            else if (req.Inputs != null)
            {
                inputTexts.AddRange(req.Inputs.Where(s => !string.IsNullOrEmpty(s)));
            }

            if (inputTexts.Count == 0)
            {
                inputTexts.Add(string.Empty);
            }

            string modelName = string.IsNullOrWhiteSpace(req.Model) ? "text-embedding-3-small" : req.Model;
            int dimensions = req.Dimensions ?? 1536;

            using var engine = new EmbeddingEngine(modelName, dimensions);

            var embedReq = new EmbeddingRequest
            {
                Inputs = inputTexts,
                Model = modelName,
                Dimensions = req.Dimensions,
                Normalize = true,
                EncodingFormat = req.EncodingFormat ?? "float"
            };

            var result = engine.Embed(embedReq);

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

        return app;
    }
}

public sealed record EmbeddingApiRequest
{
    [JsonPropertyName("model")]
    public string? Model { get; init; }

    [JsonPropertyName("input")]
    public string? Input { get; init; }

    [JsonPropertyName("inputs")]
    public List<string>? Inputs { get; init; }

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
