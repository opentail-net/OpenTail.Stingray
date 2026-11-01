namespace OpenTail.Stingray.Server.Endpoints;

/// <summary>
/// OpenAI Image API routes:
///   POST /v1/images/generations
///   POST /v1/images/edits
///
/// <para><b>Not implemented.</b> Until 2026-11-01 these routes answered 200 with a colour gradient seeded from the prompt's hash, encoded as a
/// PNG, and the server README said they served SD 1.5, SDXL, SD 3/3.5, FLUX and Z-Image. No diffusion pipeline was ever called from here, and
/// no test covered the routes (found in the parameter audit of the other endpoints). A client that read the 200 believed it had a generated
/// image. They now validate the request the way the API defines it and then answer <c>501 Not Implemented</c> with an error that says what
/// to use instead. Connecting the real pipelines (the same ones <c>stingray image</c> runs) is tracked in plan 3.</para>
/// </summary>
public static class OpenAiImageEndpoints
{
    /// <summary>The message both routes answer with once the request itself is valid.</summary>
    internal const string NotImplementedMessage =
        "The HTTP image API is not connected to a diffusion pipeline in this build. Generate images with the command line instead: `stingray image` " +
        "(see docs/WHAT-YOU-CAN-DO.md); this route used to return a placeholder gradient and no longer pretends to.";

    public sealed class ImageGenerationRequest
    {
        [JsonPropertyName("prompt")]
        public string? Prompt { get; set; }

        [JsonPropertyName("model")]
        public string? Model { get; set; }

        [JsonPropertyName("n")]
        public int N { get; set; } = 1;

        [JsonPropertyName("quality")]
        public string? Quality { get; set; } = "standard";

        [JsonPropertyName("response_format")]
        public string? ResponseFormat { get; set; } = "b64_json";

        [JsonPropertyName("size")]
        public string? Size { get; set; } = "1024x1024";

        [JsonPropertyName("style")]
        public string? Style { get; set; } = "vivid";

        [JsonPropertyName("user")]
        public string? User { get; set; }
    }

    public sealed class ImageResponseItem
    {
        [JsonPropertyName("b64_json")]
        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public string? B64Json { get; set; }

        [JsonPropertyName("url")]
        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public string? Url { get; set; }

        [JsonPropertyName("revised_prompt")]
        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public string? RevisedPrompt { get; set; }
    }

    public sealed class ImageApiResponse
    {
        [JsonPropertyName("created")]
        public long Created { get; set; }

        [JsonPropertyName("data")]
        public List<ImageResponseItem> Data { get; set; } = new();
    }

    public static IEndpointRouteBuilder MapOpenAiImageEndpoints(this IEndpointRouteBuilder app)
    {
        // POST /v1/images/generations
        app.MapPost("/v1/images/generations", async (HttpContext ctx) =>
        {
            ImageGenerationRequest? req;
            try
            {
                req = await JsonSerializer.DeserializeAsync<ImageGenerationRequest>(ctx.Request.Body, cancellationToken: ctx.RequestAborted);
            }
            catch (JsonException)
            {
                await WriteErrorAsync(ctx, 400, "Invalid JSON request body");
                return;
            }

            if (req is null || string.IsNullOrWhiteSpace(req.Prompt))
            {
                await WriteErrorAsync(ctx, 400, "'prompt' field is required.");
                return;
            }

            await WriteErrorAsync(ctx, 501, NotImplementedMessage);
        }).WithOperationMetrics("image_generation", "openai.images_generations");

        // POST /v1/images/edits
        app.MapPost("/v1/images/edits", async (HttpContext ctx) =>
        {
            if (!ctx.Request.HasFormContentType)
            {
                await WriteErrorAsync(ctx, 400, "Multipart form data required for image edits");
                return;
            }

            var form = await ctx.Request.ReadFormAsync(ctx.RequestAborted);
            if (string.IsNullOrWhiteSpace(form["prompt"].ToString()))
            {
                await WriteErrorAsync(ctx, 400, "'prompt' form field is required");
                return;
            }

            await WriteErrorAsync(ctx, 501, NotImplementedMessage);
        }).WithOperationMetrics("image_edit", "openai.images_edits");

        return app;
    }

    private static async Task WriteErrorAsync(HttpContext ctx, int status, string message)
    {
        ctx.Response.StatusCode = status;
        ctx.Response.ContentType = "application/json";
        await ctx.Response.WriteAsync(
            JsonSerializer.Serialize(
                new ErrorResponse(status == 501 ? "not_implemented" : "invalid_request_error", message),
                OpenTailStingrayJsonContext.Default.ErrorResponse),
            ctx.RequestAborted);
    }
}
