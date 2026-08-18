using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;
using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Audio.Chatterbox;
using OpenTail.Stingray.Audio.F5TTS;
using OpenTail.Stingray.Audio.Kokoro;
using OpenTail.Stingray.Audio.MeloTTS;
using OpenTail.Stingray.Audio.Piper;
using OpenTail.Stingray.Audio.Whisper;

namespace OpenTail.Stingray.Server.Endpoints;

public static class OpenAiAudioEndpoints
{
    public static IEndpointRouteBuilder MapOpenAiAudioEndpoints(this IEndpointRouteBuilder app)
    {
        // 1. Text-to-Speech (TTS) Endpoint
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

            bool isMelo = req.Model != null && req.Model.Contains("melo", StringComparison.OrdinalIgnoreCase);
            bool isChatterbox = req.Model != null && req.Model.Contains("chatter", StringComparison.OrdinalIgnoreCase);
            bool isF5 = req.Model != null && req.Model.Contains("f5", StringComparison.OrdinalIgnoreCase);
            bool isPiper = req.Model != null && (req.Model.Contains("piper", StringComparison.OrdinalIgnoreCase) || req.Model.Contains("vits", StringComparison.OrdinalIgnoreCase));

            ITextToSpeechPipeline pipeline = isMelo
                ? new MeloPipeline()
                : (isChatterbox
                    ? new ChatterboxPipeline()
                    : (isF5
                        ? new F5TtsPipeline()
                        : (isPiper ? new PiperPipeline() : new KokoroPipeline())));

            byte[] wavBytes;

            using (pipeline)
            {
                var audioReq = new AudioGenerationRequest
                {
                    Text = req.Input,
                    Voice = voice,
                    Speed = speed
                };

                var result = pipeline.Generate(audioReq);
                wavBytes = result.ToWavBytes();
            }

            ctx.Response.StatusCode = 200;
            ctx.Response.ContentType = "audio/wav";
            ctx.Response.Headers.ContentLength = wavBytes.Length;
            await ctx.Response.Body.WriteAsync(wavBytes, ctx.RequestAborted);
        });

        // 2. Speech-to-Text (ASR) Transcriptions Endpoint
        app.MapPost("/v1/audio/transcriptions", async (HttpContext ctx) =>
        {
            await HandleSpeechToTextAsync(ctx, SpeechTask.Transcribe);
        });

        // 3. Speech-to-Text (ASR) Translations Endpoint
        app.MapPost("/v1/audio/translations", async (HttpContext ctx) =>
        {
            await HandleSpeechToTextAsync(ctx, SpeechTask.Translate);
        });

        return app;
    }

    private static async Task HandleSpeechToTextAsync(HttpContext ctx, SpeechTask task)
    {
        float[]? samples = null;
        int sampleRate = 16000;
        string? language = null;
        float temperature = 0.0f;
        string? responseFormat = "json";

        if (ctx.Request.HasFormContentType)
        {
            var form = await ctx.Request.ReadFormAsync(ctx.RequestAborted);
            var file = form.Files.GetFile("file");

            if (file != null && file.Length > 0)
            {
                using var stream = file.OpenReadStream();
                var wavData = WavReader.ReadWav(stream);
                samples = wavData.Samples;
                sampleRate = wavData.SampleRate;
            }

            language = form["language"].FirstOrDefault();
            responseFormat = form["response_format"].FirstOrDefault() ?? "json";
            if (float.TryParse(form["temperature"].FirstOrDefault(), out float temp))
            {
                temperature = temp;
            }
        }
        else
        {
            // Direct WAV stream upload
            using var ms = new MemoryStream();
            await ctx.Request.Body.CopyToAsync(ms, ctx.RequestAborted);
            ms.Position = 0;
            if (ms.Length > 0)
            {
                var wavData = WavReader.ReadWav(ms);
                samples = wavData.Samples;
                sampleRate = wavData.SampleRate;
            }
        }

        if (samples == null || samples.Length == 0)
        {
            ctx.Response.StatusCode = 400;
            ctx.Response.ContentType = "application/json";
            await ctx.Response.WriteAsync("{\"error\":{\"message\":\"Audio file or payload is required.\",\"type\":\"invalid_request_error\"}}");
            return;
        }

        using var pipeline = new WhisperPipeline();
        var req = new SpeechToTextRequest
        {
            AudioSamples = samples,
            SampleRate = sampleRate,
            Language = language,
            Task = task,
            Temperature = temperature,
            EnableTimestamps = true
        };

        var result = pipeline.Transcribe(req);

        if (responseFormat.Equals("text", StringComparison.OrdinalIgnoreCase))
        {
            ctx.Response.StatusCode = 200;
            ctx.Response.ContentType = "text/plain";
            await ctx.Response.WriteAsync(result.Text);
            return;
        }

        var responseObj = new TranscriptionResponse
        {
            Text = result.Text,
            Language = result.Language,
            Duration = (float)result.Duration.TotalSeconds,
            Segments = result.Segments.Select(s => new SegmentResponse
            {
                Id = s.Id,
                Start = (float)s.Start.TotalSeconds,
                End = (float)s.End.TotalSeconds,
                Text = s.Text
            }).ToList()
        };

        ctx.Response.StatusCode = 200;
        ctx.Response.ContentType = "application/json";
        await JsonSerializer.SerializeAsync(ctx.Response.Body, responseObj, cancellationToken: ctx.RequestAborted);
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

public sealed record TranscriptionResponse
{
    [JsonPropertyName("text")]
    public string Text { get; init; } = string.Empty;

    [JsonPropertyName("language")]
    public string Language { get; init; } = "en";

    [JsonPropertyName("duration")]
    public float Duration { get; init; }

    [JsonPropertyName("segments")]
    public List<SegmentResponse> Segments { get; init; } = [];
}

public sealed record SegmentResponse
{
    [JsonPropertyName("id")]
    public int Id { get; init; }

    [JsonPropertyName("start")]
    public float Start { get; init; }

    [JsonPropertyName("end")]
    public float End { get; init; }

    [JsonPropertyName("text")]
    public string Text { get; init; } = string.Empty;
}
