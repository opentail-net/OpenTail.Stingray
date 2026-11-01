using System.Globalization;
using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Audio.Chatterbox;
using OpenTail.Stingray.Audio.F5TTS;
using OpenTail.Stingray.Audio.Kokoro;
using OpenTail.Stingray.Audio.MeloTTS;
using OpenTail.Stingray.Audio.Piper;
using OpenTail.Stingray.Audio.Whisper;
using OpenTail.Stingray.Core.Catalog;

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
                await ctx.Response.WriteAsync(JsonSerializer.Serialize(new ErrorResponse("invalid_request_error", "Invalid JSON request body"), OpenTailStingrayJsonContext.Default.ErrorResponse), ctx.RequestAborted);
                return;
            }

            if (req is null || string.IsNullOrWhiteSpace(req.Input))
            {
                ctx.Response.StatusCode = 400;
                ctx.Response.ContentType = "application/json";
                await ctx.Response.WriteAsync(JsonSerializer.Serialize(new ErrorResponse("invalid_request_error", "'input' text field is required."), OpenTailStingrayJsonContext.Default.ErrorResponse), ctx.RequestAborted);
                return;
            }

            // This endpoint always answers with WAV. A client that asks for another container (mp3, opus, aac, flac, pcm) would receive WAV bytes
            // labelled audio/wav; refuse that instead of answering with a format the client did not ask for. Omitted or "wav" is accepted.
            if (!string.IsNullOrEmpty(req.ResponseFormat) && !string.Equals(req.ResponseFormat, "wav", StringComparison.OrdinalIgnoreCase))
            {
                ctx.Response.StatusCode = 400;
                ctx.Response.ContentType = "application/json";
                await ctx.Response.WriteAsync(
                    JsonSerializer.Serialize(
                        new ErrorResponse("invalid_request_error", $"'response_format' must be 'wav' (or omitted): this server only produces WAV audio (got '{req.ResponseFormat}')."),
                        OpenTailStingrayJsonContext.Default.ErrorResponse),
                    ctx.RequestAborted);
                return;
            }

            string voice = string.IsNullOrWhiteSpace(req.Voice) ? "af_heart" : req.Voice;
            if (RequestParameterValidation.ValidateSpeechSpeed(req.Speed) is { } speedError)
            {
                ctx.Response.StatusCode = 400;
                ctx.Response.ContentType = "application/json";
                await ctx.Response.WriteAsync(
                    JsonSerializer.Serialize(new ErrorResponse("invalid_request_error", speedError), OpenTailStingrayJsonContext.Default.ErrorResponse),
                    ctx.RequestAborted);
                return;
            }
            float speed = req.Speed;

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
                ctx.Items[OperationMetricsFilterExtensions.AudioSecondsItem] = result.Duration.TotalSeconds; // output audio, for opentail.audio.* metrics
            }

            ctx.Response.StatusCode = 200;
            ctx.Response.ContentType = "audio/wav";
            ctx.Response.Headers.ContentLength = wavBytes.Length;
            await ctx.Response.Body.WriteAsync(wavBytes, ctx.RequestAborted);
        }).WithOperationMetrics("speech", "openai.audio_speech");

        // 2. Speech-to-Text (ASR) Transcriptions Endpoint
        app.MapPost("/v1/audio/transcriptions", async (HttpContext ctx) =>
        {
            await HandleSpeechToTextAsync(ctx, SpeechTask.Transcribe);
        }).WithOperationMetrics("transcription", "openai.audio_transcriptions");

        // 3. Speech-to-Text (ASR) Translations Endpoint
        app.MapPost("/v1/audio/translations", async (HttpContext ctx) =>
        {
            await HandleSpeechToTextAsync(ctx, SpeechTask.Translate);
        }).WithOperationMetrics("translation", "openai.audio_translations");

        return app;
    }

    // OpenAI's own id for its transcription model, and the catalog's task name (ModelCatalog.Find accepts it for the task default).
    private static readonly string[] s_transcriptionAliases = ["whisper-1", "transcribe"];

    /// <summary>
    /// Null when <paramref name="model"/> is omitted (the route keeps its default pipeline), is an alias, or names a catalog entry of the
    /// transcribe task; otherwise a message that names the field and lists the valid ids.
    /// </summary>
    internal static string? ValidateTranscriptionModel(string? model)
    {
        if (string.IsNullOrWhiteSpace(model)) return null;
        var ids = ModelCatalog.Entries.Where(e => e.Task.Equals("transcribe", StringComparison.OrdinalIgnoreCase)).Select(e => e.Id).ToList();
        ids.AddRange(s_transcriptionAliases);
        if (ids.Contains(model, StringComparer.OrdinalIgnoreCase)) return null;
        return $"model must be one of {string.Join(", ", ids)} (got '{model}')";
    }

    private static readonly HashSet<string> s_speechToTextFormats = new(StringComparer.OrdinalIgnoreCase) { "json", "text", "srt", "verbose_json", "vtt" };

    private static async Task HandleSpeechToTextAsync(HttpContext ctx, SpeechTask task)
    {
        float[]? samples = null;
        int sampleRate = 16000;
        string? language = null;
        string? prompt = null;
        string? model = null;
        float temperature = 0.0f;
        string responseFormat = "json";
        bool useVad = false;
        bool stream = false;

        // Client mistakes are 400s that name the problem, not unhandled exceptions: a body that is not WAV audio used to surface as a 500
        // (InvalidDataException), and an unparseable temperature, an unknown response_format or a non-boolean flag was silently replaced by a default.
        string? invalid = null;

        if (ctx.Request.HasFormContentType)
        {
            var form = await ctx.Request.ReadFormAsync(ctx.RequestAborted);
            var file = form.Files.GetFile("file");

            if (file != null && file.Length > 0)
            {
                using var fileStream = file.OpenReadStream();
                try
                {
                    var wavData = WavReader.ReadWav(fileStream);
                    samples = wavData.Samples;
                    sampleRate = wavData.SampleRate;
                }
                catch (Exception ex) when (ex is InvalidDataException or EndOfStreamException) { invalid = $"'file' is not a valid WAV file: {ex.Message}"; }
            }

            language = form["language"].FirstOrDefault();
            model = form["model"].FirstOrDefault();
            prompt = form["prompt"].FirstOrDefault() ?? form["initial_prompt"].FirstOrDefault();
            responseFormat = form["response_format"].FirstOrDefault() ?? "json";

            string? tempText = form["temperature"].FirstOrDefault();
            if (tempText is not null)
            {
                if (float.TryParse(tempText, NumberStyles.Float, CultureInfo.InvariantCulture, out float temp) && temp >= 0f && temp <= 1f)
                    temperature = temp;
                else
                    invalid ??= $"temperature must be a number between 0 and 1 (got '{tempText}')";
            }

            string? vadText = form["vad"].FirstOrDefault() ?? form["use_vad"].FirstOrDefault();
            if (vadText is not null)
            {
                if (bool.TryParse(vadText, out bool vadFlag)) useVad = vadFlag;
                else invalid ??= $"vad must be true or false (got '{vadText}')";
            }

            string? streamText = form["stream"].FirstOrDefault();
            if (streamText is not null)
            {
                if (bool.TryParse(streamText, out bool streamFlag)) stream = streamFlag;
                else invalid ??= $"stream must be true or false (got '{streamText}')";
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
                try
                {
                    var wavData = WavReader.ReadWav(ms);
                    samples = wavData.Samples;
                    sampleRate = wavData.SampleRate;
                }
                catch (Exception ex) when (ex is InvalidDataException or EndOfStreamException) { invalid = $"the request body is not a valid WAV file: {ex.Message}"; }
            }

            if (ctx.Request.Query.TryGetValue("language", out var langVal)) language = langVal.ToString();
            if (ctx.Request.Query.TryGetValue("model", out var modelVal)) model = modelVal.ToString();
            if (ctx.Request.Query.TryGetValue("prompt", out var promptVal)) prompt = promptVal.ToString();
            if (ctx.Request.Query.TryGetValue("response_format", out var formatVal)) responseFormat = formatVal.ToString();
            if (ctx.Request.Query.TryGetValue("vad", out var vadVal))
            {
                if (bool.TryParse(vadVal, out bool v)) useVad = v;
                else invalid ??= $"vad must be true or false (got '{vadVal}')";
            }
            if (ctx.Request.Query.TryGetValue("stream", out var streamVal))
            {
                if (bool.TryParse(streamVal, out bool s)) stream = s;
                else invalid ??= $"stream must be true or false (got '{streamVal}')";
            }
        }

        invalid ??= RequestParameterValidation.ValidateLanguageCode(language);
        invalid ??= ValidateTranscriptionModel(model);
        if (invalid is null && !s_speechToTextFormats.Contains(responseFormat))
            invalid = $"response_format must be one of json, text, srt, verbose_json, vtt (got '{responseFormat}')";
        if (invalid is not null)
        {
            ctx.Response.StatusCode = 400;
            ctx.Response.ContentType = "application/json";
            await ctx.Response.WriteAsync(
                JsonSerializer.Serialize(new ErrorResponse("invalid_request_error", invalid), OpenTailStingrayJsonContext.Default.ErrorResponse),
                ctx.RequestAborted);
            return;
        }
        if (samples == null || samples.Length == 0)
        {
            ctx.Response.StatusCode = 400;
            ctx.Response.ContentType = "application/json";
            await ctx.Response.WriteAsync(JsonSerializer.Serialize(new ErrorResponse("invalid_request_error", "Audio file or payload is required."), OpenTailStingrayJsonContext.Default.ErrorResponse), ctx.RequestAborted);
            return;
        }

        ctx.Items[OperationMetricsFilterExtensions.AudioSecondsItem] = (double)samples.Length / sampleRate; // input audio, for opentail.audio.* metrics
        using var pipeline = new WhisperPipeline();
        var req = new SpeechToTextRequest
        {
            AudioSamples = samples,
            SampleRate = sampleRate,
            Language = language,
            InitialPrompt = prompt,
            Task = task,
            Temperature = temperature,
            UseVad = useVad,
            EnableTimestamps = true
        };

        if (stream)
        {
            ctx.Response.StatusCode = 200;
            ctx.Response.ContentType = "text/event-stream";
            ctx.Response.Headers.CacheControl = "no-cache";

            async IAsyncEnumerable<ReadOnlyMemory<float>> AudioChunkGenerator()
            {
                int chunkSize = 16000 * 2; // 2 seconds
                for (int i = 0; i < samples.Length; i += chunkSize)
                {
                    int len = Math.Min(chunkSize, samples.Length - i);
                    yield return samples.AsMemory(i, len);
                    await Task.Yield();
                }
            }

            await foreach (var seg in pipeline.TranscribeStreamAsync(AudioChunkGenerator(), req, ctx.RequestAborted))
            {
                var segDto = new SegmentResponse
                {
                    Id = seg.Id,
                    Start = (float)seg.Start.TotalSeconds,
                    End = (float)seg.End.TotalSeconds,
                    Text = seg.Text
                };
                string jsonLine = JsonSerializer.Serialize(segDto);
                await ctx.Response.WriteAsync($"data: {jsonLine}\n\n", ctx.RequestAborted);
                await ctx.Response.Body.FlushAsync(ctx.RequestAborted);
            }

            await ctx.Response.WriteAsync("data: [DONE]\n\n", ctx.RequestAborted);
            return;
        }

        var result = pipeline.Transcribe(req);

        // Subtitle and text response formats
        switch (responseFormat.ToLowerInvariant())
        {
            case "text":
                ctx.Response.StatusCode = 200;
                ctx.Response.ContentType = "text/plain; charset=utf-8";
                await ctx.Response.WriteAsync(result.Text);
                return;

            case "srt":
                ctx.Response.StatusCode = 200;
                ctx.Response.ContentType = "application/x-subrip; charset=utf-8";
                await ctx.Response.WriteAsync(FormatSrt(result.Segments));
                return;

            case "vtt":
                ctx.Response.StatusCode = 200;
                ctx.Response.ContentType = "text/vtt; charset=utf-8";
                await ctx.Response.WriteAsync(FormatVtt(result.Segments));
                return;

            case "verbose_json":
                var verboseObj = new VerboseTranscriptionResponse
                {
                    Task = task == SpeechTask.Translate ? "translate" : "transcribe",
                    Language = result.Language,
                    Duration = (float)result.Duration.TotalSeconds,
                    Text = result.Text,
                    Segments = result.Segments.Select(s => new VerboseSegmentResponse
                    {
                        Id = s.Id,
                        Start = (float)s.Start.TotalSeconds,
                        End = (float)s.End.TotalSeconds,
                        Text = s.Text,
                        Tokens = s.Tokens,
                        AvgLogprob = -0.15f,
                        CompressionRatio = 1.0f,
                        NoSpeechProb = 0.01f
                    }).ToList()
                };
                ctx.Response.StatusCode = 200;
                ctx.Response.ContentType = "application/json";
                await JsonSerializer.SerializeAsync(ctx.Response.Body, verboseObj, cancellationToken: ctx.RequestAborted);
                return;

            default: // "json"
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
                return;
        }
    }

    /// <summary>
    /// Formats speech segments into SubRip (.srt) subtitle format.
    /// Index\nhh:mm:ss,fff --> hh:mm:ss,fff\nText\n\n
    /// </summary>
    public static string FormatSrt(IReadOnlyList<SpeechSegment> segments)
    {
        var sb = new StringBuilder();
        for (int i = 0; i < segments.Count; i++)
        {
            var seg = segments[i];
            sb.AppendLine((i + 1).ToString(CultureInfo.InvariantCulture));
            sb.Append(FormatTimestampSrt(seg.Start));
            sb.Append(" --> ");
            sb.AppendLine(FormatTimestampSrt(seg.End));
            sb.AppendLine(seg.Text.Trim());
            sb.AppendLine();
        }
        return sb.ToString();
    }

    /// <summary>
    /// Formats speech segments into WebVTT (.vtt) subtitle format.
    /// WEBVTT\n\n00:00:00.000 --> 00:00:00.000\nText\n\n
    /// </summary>
    public static string FormatVtt(IReadOnlyList<SpeechSegment> segments)
    {
        var sb = new StringBuilder();
        sb.AppendLine("WEBVTT");
        sb.AppendLine();

        for (int i = 0; i < segments.Count; i++)
        {
            var seg = segments[i];
            sb.Append(FormatTimestampVtt(seg.Start));
            sb.Append(" --> ");
            sb.AppendLine(FormatTimestampVtt(seg.End));
            sb.AppendLine(seg.Text.Trim());
            sb.AppendLine();
        }
        return sb.ToString();
    }

    private static string FormatTimestampSrt(TimeSpan t)
    {
        return $"{(int)t.TotalHours:00}:{t.Minutes:00}:{t.Seconds:00},{t.Milliseconds:000}";
    }

    private static string FormatTimestampVtt(TimeSpan t)
    {
        return $"{(int)t.TotalHours:00}:{t.Minutes:00}:{t.Seconds:00}.{t.Milliseconds:000}";
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

public sealed record VerboseTranscriptionResponse
{
    [JsonPropertyName("task")]
    public string Task { get; init; } = "transcribe";

    [JsonPropertyName("language")]
    public string Language { get; init; } = "en";

    [JsonPropertyName("duration")]
    public float Duration { get; init; }

    [JsonPropertyName("text")]
    public string Text { get; init; } = string.Empty;

    [JsonPropertyName("segments")]
    public List<VerboseSegmentResponse> Segments { get; init; } = [];
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

public sealed record VerboseSegmentResponse
{
    [JsonPropertyName("id")]
    public int Id { get; init; }

    [JsonPropertyName("start")]
    public float Start { get; init; }

    [JsonPropertyName("end")]
    public float End { get; init; }

    [JsonPropertyName("text")]
    public string Text { get; init; } = string.Empty;

    [JsonPropertyName("tokens")]
    public int[] Tokens { get; init; } = [];

    [JsonPropertyName("avg_logprob")]
    public float AvgLogprob { get; init; } = 0.0f;

    [JsonPropertyName("compression_ratio")]
    public float CompressionRatio { get; init; } = 1.0f;

    [JsonPropertyName("no_speech_prob")]
    public float NoSpeechProb { get; init; } = 0.0f;
}
