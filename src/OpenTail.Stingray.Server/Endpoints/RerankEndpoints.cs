using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;
using OpenTail.Stingray.Core.Embeddings;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Server.Endpoints;

public static class RerankEndpoints
{
    public static IEndpointRouteBuilder MapRerankEndpoints(this IEndpointRouteBuilder app)
    {
        app.MapPost("/v1/rerank", async (HttpContext ctx) =>
        {
            RerankApiRequest? req;
            try
            {
                req = await JsonSerializer.DeserializeAsync<RerankApiRequest>(ctx.Request.Body, cancellationToken: ctx.RequestAborted);
            }
            catch (JsonException)
            {
                ctx.Response.StatusCode = 400;
                ctx.Response.ContentType = "application/json";
                await ctx.Response.WriteAsync("{\"error\":{\"message\":\"Invalid JSON request body\",\"type\":\"invalid_request_error\"}}");
                return;
            }

            if (req is null || string.IsNullOrWhiteSpace(req.Query))
            {
                ctx.Response.StatusCode = 400;
                ctx.Response.ContentType = "application/json";
                await ctx.Response.WriteAsync("{\"error\":{\"message\":\"'query' field is required.\",\"type\":\"invalid_request_error\"}}");
                return;
            }

            if (req.Documents is null || req.Documents.Count == 0)
            {
                ctx.Response.StatusCode = 400;
                ctx.Response.ContentType = "application/json";
                await ctx.Response.WriteAsync("{\"error\":{\"message\":\"'documents' list cannot be empty.\",\"type\":\"invalid_request_error\"}}");
                return;
            }

            string modelName = string.IsNullOrWhiteSpace(req.Model) ? "bge-reranker-large" : req.Model;

            using var engine = new EmbeddingEngine(modelName);

            var rerankReq = new RerankRequest
            {
                Query = req.Query,
                Documents = req.Documents,
                TopN = req.TopN,
                Model = modelName,
                ReturnDocuments = req.ReturnDocuments
            };

            var result = engine.Rerank(rerankReq);

            var responseObj = new RerankApiResponse
            {
                Id = $"rerank-{Guid.NewGuid():N}",
                Results = result.Results.Select(r => new RerankItemResponse
                {
                    Index = r.Index,
                    RelevanceScore = r.RelevanceScore,
                    Document = r.Document
                }).ToList(),
                Meta = new RerankMetaResponse
                {
                    Tokens = new RerankTokensMeta
                    {
                        InputTokens = result.TotalTokens,
                        OutputTokens = result.Results.Count
                    }
                }
            };

            ctx.Response.StatusCode = 200;
            ctx.Response.ContentType = "application/json";
            await JsonSerializer.SerializeAsync(ctx.Response.Body, responseObj, cancellationToken: ctx.RequestAborted);
        });

        return app;
    }
}

public sealed record RerankApiRequest
{
    [JsonPropertyName("model")]
    public string? Model { get; init; }

    [JsonPropertyName("query")]
    public string? Query { get; init; }

    [JsonPropertyName("documents")]
    public List<string>? Documents { get; init; }

    [JsonPropertyName("top_n")]
    public int? TopN { get; init; }

    [JsonPropertyName("return_documents")]
    public bool ReturnDocuments { get; init; } = true;
}

public sealed record RerankApiResponse
{
    [JsonPropertyName("id")]
    public string Id { get; init; } = string.Empty;

    [JsonPropertyName("results")]
    public List<RerankItemResponse> Results { get; init; } = [];

    [JsonPropertyName("meta")]
    public RerankMetaResponse Meta { get; init; } = new();
}

public sealed record RerankItemResponse
{
    [JsonPropertyName("index")]
    public int Index { get; init; }

    [JsonPropertyName("relevance_score")]
    public float RelevanceScore { get; init; }

    [JsonPropertyName("document")]
    public string? Document { get; init; }
}

public sealed record RerankMetaResponse
{
    [JsonPropertyName("tokens")]
    public RerankTokensMeta Tokens { get; init; } = new();
}

public sealed record RerankTokensMeta
{
    [JsonPropertyName("input_tokens")]
    public int InputTokens { get; init; }

    [JsonPropertyName("output_tokens")]
    public int OutputTokens { get; init; }
}
