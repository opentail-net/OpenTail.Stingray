using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;
using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Audio.Kokoro;

namespace OpenTail.Stingray.Server.Endpoints;

public static class OpenAiAudioEndpoints
{
    public static IEndpointRouteBuilder MapOpenAiAudioEndpoints(this IEndpointRouteBuilder app)
    {
        app.MapPost("/v1/audio/speech", async (HttpContext ctx) =>
        {
            SpeechRequest? req;
            try
            {
                req = await JsonSerializer.DeserializeAsync<SpeechRequest>(ctx.Request.Body, cancellationToken: ctx.RequestAborted);
            }
            catch (JsonException)
            {
                ctx.Response.StatusCode = 400;
                ctx.Response.ContentType = "application/json";
                await ctx.Response.WriteAsync("{\"error\":{\"message\":\"Invalid JSON request body\",\"type\":\"invalid_request_error\"}}");
                return;
            }

            if (req is null || string.IsNullOrWhiteSpace(req.Input))
            {
                ctx.Response.StatusCode = 400;
                ctx.Response.ContentType = "application/json";
                await ctx.Response.WriteAsync("{\"error\":{\"message\":\"'input' text field is required.\",\"type\":\"invalid_request_error\"}}");
                return;
            }

            string voice = string.IsNullOrWhiteSpace(req.Voice) ? "af_heart" : req.Voice;
            float speed = req.Speed > 0 ? req.Speed : 1.0f;

            using var pipeline = new KokoroPipeline();
            var audioReq = new AudioGenerationRequest
            {
                Text = req.Input,
                Voice = voice,
                Speed = speed
            };

            var result = pipeline.Generate(audioReq);
            byte[] wavBytes = result.ToWavBytes();

            ctx.Response.StatusCode = 200;
            ctx.Response.ContentType = "audio/wav";
            ctx.Response.Headers.ContentLength = wavBytes.Length;
            await ctx.Response.Body.WriteAsync(wavBytes, ctx.RequestAborted);
        });

        return app;
    }
}

public sealed record SpeechRequest
{
    [JsonPropertyName("model")]
    public string? Model { get; init; }

    [JsonPropertyName("input")]
    public string? Input { get; init; }

    [JsonPropertyName("voice")]
    public string? Voice { get; init; }

    [JsonPropertyName("response_format")]
    public string? ResponseFormat { get; init; }

    [JsonPropertyName("speed")]
    public float Speed { get; init; } = 1.0f;
}
