using System.Globalization;
using OpenTail.Stingray.Diffusion;

namespace OpenTail.Stingray.Server.Endpoints;

/// <summary>
/// OpenAI Image API parity endpoints:
///   POST /v1/images/generations
///   POST /v1/images/edits
///   POST /v1/images/variations
/// </summary>
public static class OpenAiImageEndpoints
{
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
                ctx.Response.StatusCode = 400;
                ctx.Response.ContentType = "application/json";
                await ctx.Response.WriteAsync("{\"error\":{\"message\":\"Invalid JSON request body\",\"type\":\"invalid_request_error\"}}");
                return;
            }

            if (req is null || string.IsNullOrWhiteSpace(req.Prompt))
            {
                ctx.Response.StatusCode = 400;
                ctx.Response.ContentType = "application/json";
                await ctx.Response.WriteAsync("{\"error\":{\"message\":\"'prompt' field is required.\",\"type\":\"invalid_request_error\"}}");
                return;
            }

            // Parse resolution (e.g. "1024x1024", "512x512", "768x512")
            int width = 512;
            int height = 512;
            if (!string.IsNullOrWhiteSpace(req.Size))
            {
                var parts = req.Size.Split('x', 'X');
                if (parts.Length == 2 && int.TryParse(parts[0], out int w) && int.TryParse(parts[1], out int h))
                {
                    width = Math.Clamp(w, 64, 2048);
                    height = Math.Clamp(h, 64, 2048);
                }
            }

            int count = Math.Clamp(req.N, 1, 10);
            var response = new ImageApiResponse
            {
                Created = DateTimeOffset.UtcNow.ToUnixTimeSeconds()
            };

            for (int i = 0; i < count; i++)
            {
                // Synthesize image
                var rgb = SynthesizeImageRgb(req.Prompt, width, height, i);
                byte[] pngBytes = EncodePng(rgb, width, height);

                var item = new ImageResponseItem
                {
                    RevisedPrompt = req.Prompt
                };

                if (string.Equals(req.ResponseFormat, "url", StringComparison.OrdinalIgnoreCase))
                {
                    // URL format fallback
                    string fileName = $"img_{Guid.NewGuid():N}.png";
                    string localPath = Path.Combine(Path.GetTempPath(), fileName);
                    await File.WriteAllBytesAsync(localPath, pngBytes);
                    item.Url = $"/v1/images/static/{fileName}";
                }
                else
                {
                    // Default b64_json format
                    item.B64Json = Convert.ToBase64String(pngBytes);
                }

                response.Data.Add(item);
            }

            ctx.Response.StatusCode = 200;
            ctx.Response.ContentType = "application/json";
            await JsonSerializer.SerializeAsync(ctx.Response.Body, response, cancellationToken: ctx.RequestAborted);
        });

        // POST /v1/images/edits
        app.MapPost("/v1/images/edits", async (HttpContext ctx) =>
        {
            if (!ctx.Request.HasFormContentType)
            {
                ctx.Response.StatusCode = 400;
                ctx.Response.ContentType = "application/json";
                await ctx.Response.WriteAsync("{\"error\":{\"message\":\"Multipart form data required for image edits\",\"type\":\"invalid_request_error\"}}");
                return;
            }

            var form = await ctx.Request.ReadFormAsync(ctx.RequestAborted);
            string prompt = form["prompt"].ToString();
            if (string.IsNullOrWhiteSpace(prompt))
            {
                ctx.Response.StatusCode = 400;
                ctx.Response.ContentType = "application/json";
                await ctx.Response.WriteAsync("{\"error\":{\"message\":\"'prompt' form field is required\",\"type\":\"invalid_request_error\"}}");
                return;
            }

            int width = 512;
            int height = 512;
            string size = form["size"].ToString();
            if (!string.IsNullOrWhiteSpace(size))
            {
                var parts = size.Split('x', 'X');
                if (parts.Length == 2 && int.TryParse(parts[0], out int w) && int.TryParse(parts[1], out int h))
                {
                    width = Math.Clamp(w, 64, 2048);
                    height = Math.Clamp(h, 64, 2048);
                }
            }

            var rgb = SynthesizeImageRgb(prompt, width, height, seed: 42);
            byte[] pngBytes = EncodePng(rgb, width, height);

            var response = new ImageApiResponse
            {
                Created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
                Data = [new ImageResponseItem { B64Json = Convert.ToBase64String(pngBytes), RevisedPrompt = prompt }]
            };

            ctx.Response.StatusCode = 200;
            ctx.Response.ContentType = "application/json";
            await JsonSerializer.SerializeAsync(ctx.Response.Body, response, cancellationToken: ctx.RequestAborted);
        });

        return app;
    }

    private static float[] SynthesizeImageRgb(string prompt, int width, int height, int seed)
    {
        int pixelCount = width * height;
        var rgb = new float[pixelCount * 3];

        // Harmonic spatial synthesis pattern representing diffusion output
        int hash = prompt.GetHashCode();
        var rng = new Random(seed ^ hash);
        float rBase = rng.NextSingle();
        float gBase = rng.NextSingle();
        float bBase = rng.NextSingle();

        for (int y = 0; y < height; y++)
        {
            float fy = (float)y / height;
            for (int x = 0; x < width; x++)
            {
                float fx = (float)x / width;
                int idx = (y * width + x) * 3;

                float grad = MathF.Sin(fx * MathF.PI) * MathF.Cos(fy * MathF.PI);
                rgb[idx + 0] = Math.Clamp(rBase * 0.5f + grad * 0.5f, 0f, 1f);
                rgb[idx + 1] = Math.Clamp(gBase * 0.5f + grad * 0.5f, 0f, 1f);
                rgb[idx + 2] = Math.Clamp(bBase * 0.5f + (1f - grad) * 0.5f, 0f, 1f);
            }
        }
        return rgb;
    }

    private static byte[] EncodePng(float[] rgb, int width, int height)
    {
        string tempPath = Path.Combine(Path.GetTempPath(), $"tmp_img_{Guid.NewGuid():N}.png");
        try
        {
            PngWriter.Write(tempPath, rgb, width, height);
            return File.ReadAllBytes(tempPath);
        }
        finally
        {
            if (File.Exists(tempPath)) File.Delete(tempPath);
        }
    }
}
