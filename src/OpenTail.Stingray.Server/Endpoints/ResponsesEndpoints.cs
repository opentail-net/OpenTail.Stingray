

namespace OpenTail.Stingray.Server.Endpoints;

/// <summary>
/// OpenAI Responses API: POST /v1/responses
/// Supports both streaming (SSE) and non-streaming responses.
/// Input can be a plain string or an array of message objects.
/// </summary>
public static class ResponsesEndpoints
{
    public static IEndpointRouteBuilder MapResponsesEndpoints(this IEndpointRouteBuilder app)
    {
        app.MapPost("/v1/responses", HandleCreateResponse).WithConcurrencyLimit();
        return app;
    }

    private static async Task HandleCreateResponse(
        HttpContext ctx,
        IInferenceService inferenceService,
        ServerMetrics metrics,
        IOptions<OpenTailStingrayServerOptions> options)
    {
        var opts = options.Value;
        ResponsesRequest? req;
        try
        {
            req = await ctx.Request.ReadFromJsonAsync(OpenTailStingrayJsonContext.Default.ResponsesRequest, ctx.RequestAborted);
        }
        catch
        {
            await WriteBadRequestAsync(ctx, "The request body is not valid JSON.");
            return;
        }

        if (req is null)
        {
            await WriteBadRequestAsync(ctx, "The request body must be a JSON object.");
            return;
        }
        if (RequestParameterValidation.ValidateResponses(req) is { } invalid)
        {
            await WriteBadRequestAsync(ctx, invalid);
            return;
        }

        // docs/032 Phase 7: resolve + acquire the requested model — but ONLY in multi-model mode.
        // Single-model mode keeps resolving IInferenceEngine/ChatTemplateRenderer directly from DI,
        // exactly as before Phase 7 existed — see OpenAiEndpoints.HandleChatCompletion's identical
        // branch for why.
        IInferenceEngine engine;
        ChatTemplateRenderer chatTemplate;
        ModelRuntimeHandle? handle = null;
        if (inferenceService.IsMultiModel)
        {
            ModelId resolvedModelId;
            try
            {
                resolvedModelId = inferenceService.ResolveModel(req.Model);
            }
            catch (ModelNotFoundException)
            {
                ctx.Response.StatusCode = 404;
                return;
            }

            try
            {
                handle = await inferenceService.AcquireAsync(resolvedModelId, ctx.RequestAborted);
            }
            catch (InsufficientResourcesException)
            {
                ctx.Response.StatusCode = 503;
                return;
            }

            engine = handle.Runtime.Loaded.Engine;
            chatTemplate = new ChatTemplateRenderer(handle.Runtime.Loaded.Architecture, handle.Runtime.Loaded.ChatTemplate);
            chatTemplate.Configure(handle.Runtime.Loaded.Architecture, handle.Runtime.Loaded.ChatTemplate,
                handle.Runtime.Loaded.ToolBoundaryStopTokenIds, handle.Runtime.Loaded.Grammar);
        }
        else
        {
            engine = ctx.RequestServices.GetRequiredService<IInferenceEngine>();
            chatTemplate = ctx.RequestServices.GetRequiredService<ChatTemplateRenderer>();
        }
        using var handleGuard = handle;

        metrics.RecordRequest();

        var messages = BuildMessageList(req);
        // Apply the same model-family thinking default as the chat endpoints: Gemma 4 defaults
        // reasoning off (ChatTemplateRenderer.ModelDefaultsThinkingOff), and STINGRAY_NO_THINKING
        // forces it off globally. /v1/responses has no per-request thinking flag, so the family
        // default + the server kill-switch are the whole decision.
        bool enableThinking = !chatTemplate.ModelDefaultsThinkingOff && !opts.DisableThinking;
        string prompt;
        try
        {
            prompt = chatTemplate.Format(messages, enableThinking);
        }
        catch (ChatTemplateException)
        {
            // The template rejected the message list (e.g. non-alternating roles) — bad input, not a server fault.
            await WriteBadRequestAsync(ctx, "The chat template rejected the message list (for example roles that do not alternate).");
            return;
        }

        var sp = SamplingParamsBuilder.Build(opts,
            temperature: req.Temperature,
            topP:        req.TopP,
            maxTokens:   req.MaxOutputTokens,
            thinkingDisabled: !enableThinking);

        var responseId = $"resp_{Guid.NewGuid():N}";
        var itemId = $"msg_{Guid.NewGuid():N}";
        long createdAt = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        string modelId = engine.ModelId;
        using var timing = metrics.BeginServingRequest("openai.responses", engine.ModelId, ctx.RequestAborted);

        if (req.Stream == true)
        {
            ctx.Response.ContentType = "text/event-stream";
            ctx.Response.Headers.CacheControl = "no-cache";
            ctx.Response.Headers.Connection = "keep-alive";

            // response.created
            var createdEvent = new RespCreatedEvent("response.created",
                new RespObject(responseId, "response", createdAt, modelId, "in_progress",
                    [], new RespUsage(0, 0, 0)));
            await WriteEvent(ctx.Response, "response.created",
                JsonSerializer.Serialize(createdEvent, OpenTailStingrayJsonContext.Default.RespCreatedEvent));

            // response.output_item.added
            var itemAddedEvent = new RespOutputItemAddedEvent("response.output_item.added", 0,
                new RespOutputItem(itemId, "message", "assistant", []));
            await WriteEvent(ctx.Response, "response.output_item.added",
                JsonSerializer.Serialize(itemAddedEvent, OpenTailStingrayJsonContext.Default.RespOutputItemAddedEvent));

            // response.content_part.added
            var partAddedEvent = new RespContentPartAddedEvent("response.content_part.added",
                itemId, 0, 0, new RespContentPart("output_text", ""));
            await WriteEvent(ctx.Response, "response.content_part.added",
                JsonSerializer.Serialize(partAddedEvent, OpenTailStingrayJsonContext.Default.RespContentPartAddedEvent));

            var sb = new StringBuilder();
            long tokenCount = 0;
            int promptTokens = 0;
            await foreach (var chunk in engine.GenerateChunksAsync(prompt, sp, ctx.RequestAborted))
            {
                // The engine reports the prompt size out of band, as a usage chunk; only text chunks are the answer.
                if (chunk.Kind == GenerateChunkKind.Usage) { promptTokens = chunk.PromptTokens; continue; }
                if (chunk.Kind != GenerateChunkKind.Text) continue;
                string token = chunk.Text;
                timing.MarkAdmitted();
                timing.MarkFirstToken();
                tokenCount++;
                sb.Append(token);
                var deltaEvent = new RespOutputTextDeltaEvent("response.output_text.delta",
                    itemId, 0, 0, token);
                await WriteEvent(ctx.Response, "response.output_text.delta",
                    JsonSerializer.Serialize(deltaEvent, OpenTailStingrayJsonContext.Default.RespOutputTextDeltaEvent));
            }

            string fullText = sb.ToString();

            // response.output_text.done
            var textDoneEvent = new RespOutputTextDoneEvent("response.output_text.done",
                itemId, 0, 0, fullText);
            await WriteEvent(ctx.Response, "response.output_text.done",
                JsonSerializer.Serialize(textDoneEvent, OpenTailStingrayJsonContext.Default.RespOutputTextDoneEvent));

            // response.output_item.done
            var finalItem = new RespOutputItem(itemId, "message", "assistant",
                [new RespContentPart("output_text", fullText)]);
            var itemDoneEvent = new RespOutputItemDoneEvent("response.output_item.done", 0, finalItem);
            await WriteEvent(ctx.Response, "response.output_item.done",
                JsonSerializer.Serialize(itemDoneEvent, OpenTailStingrayJsonContext.Default.RespOutputItemDoneEvent));

            // response.completed
            var usage = new RespUsage(promptTokens, (int)tokenCount, promptTokens + (int)tokenCount);
            var completedResp = new RespObject(responseId, "response", createdAt, modelId, "completed",
                [finalItem], usage);
            var completedEvent = new RespCompletedEvent("response.completed", completedResp);
            await WriteEvent(ctx.Response, "response.completed",
                JsonSerializer.Serialize(completedEvent, OpenTailStingrayJsonContext.Default.RespCompletedEvent));

            timing.MarkGenerationComplete();
            await ctx.Response.Body.FlushAsync(ctx.RequestAborted);
            timing.SetOutputTokens(tokenCount);
            metrics.RecordTokens(tokenCount);
        }
        else
        {
            var sb = new StringBuilder();
            long tokenCount = 0;
            int promptTokens = 0;
            await foreach (var chunk in engine.GenerateChunksAsync(prompt, sp, ctx.RequestAborted))
            {
                if (chunk.Kind == GenerateChunkKind.Usage) { promptTokens = chunk.PromptTokens; continue; }
                if (chunk.Kind != GenerateChunkKind.Text) continue;
                timing.MarkAdmitted();
                timing.MarkFirstToken();
                tokenCount++;
                sb.Append(chunk.Text);
            }

            string text = sb.ToString();
            var usage = new RespUsage(promptTokens, (int)tokenCount, promptTokens + (int)tokenCount);
            var item = new RespOutputItem(itemId, "message", "assistant",
                [new RespContentPart("output_text", text)]);
            timing.MarkGenerationComplete();
            var response = new RespObject(responseId, "response", createdAt, modelId, "completed",
                [item], usage, req.OpentailTiming == true
                    ? OpenTailResponseTiming.From(timing)
                    : null);

            ctx.Response.ContentType = "application/json";
            await ctx.Response.WriteAsync(
                JsonSerializer.Serialize(response, OpenTailStingrayJsonContext.Default.RespObject),
                ctx.RequestAborted);

            timing.SetOutputTokens(tokenCount);
            metrics.RecordTokens(tokenCount);
        }
    }

    private static List<(string role, string content)> BuildMessageList(ResponsesRequest req)
    {
        var list = new List<(string, string)>();

        // instructions → system message
        if (!string.IsNullOrEmpty(req.Instructions))
            list.Add(("system", req.Instructions));

        if (req.Input.HasValue)
        {
            var input = req.Input.Value;
            if (input.ValueKind == JsonValueKind.String)
            {
                list.Add(("user", input.GetString() ?? ""));
            }
            else if (input.ValueKind == JsonValueKind.Array)
            {
                foreach (var item in input.EnumerateArray())
                {
                    string role = "user";
                    string content = "";
                    if (item.TryGetProperty("role", out var roleProp))
                        role = roleProp.GetString() ?? "user";
                    if (item.TryGetProperty("content", out var contentProp))
                    {
                        if (contentProp.ValueKind == JsonValueKind.String)
                            content = contentProp.GetString() ?? "";
                        else if (contentProp.ValueKind == JsonValueKind.Array)
                        {
                            // content is array of {type, text} parts — concatenate text parts
                            var sb = new StringBuilder();
                            foreach (var part in contentProp.EnumerateArray())
                                if (part.TryGetProperty("text", out var textProp))
                                    sb.Append(textProp.GetString());
                            content = sb.ToString();
                        }
                    }
                    if (role == "assistant")
                        content = ChatTemplate.ScrubAssistantThinking(content);
                    list.Add((role, content));
                }
            }
        }

        if (list.Count == 0)
            list.Add(("user", ""));

        return list;
    }

    /// <summary>A 400 with the same error body the chat endpoints send (the flat type and message fields plus the nested error object), so a client can read why.</summary>
    private static async Task WriteBadRequestAsync(HttpContext ctx, string message)
    {
        ctx.Response.StatusCode = 400;
        ctx.Response.ContentType = "application/json";
        await ctx.Response.WriteAsync(
            JsonSerializer.Serialize(new ErrorResponse("invalid_request_error", message), OpenTailStingrayJsonContext.Default.ErrorResponse),
            ctx.RequestAborted);
    }

    private static async Task WriteEvent(HttpResponse response, string eventType, string data)
    {
        await response.WriteAsync($"event: {eventType}\ndata: {data}\n\n",
            response.HttpContext.RequestAborted);
        await response.Body.FlushAsync(response.HttpContext.RequestAborted);
    }
}

// ── Request / Response types ──────────────────────────────────────────────────

public sealed record ResponsesRequest(
    string? Model,
    JsonElement? Input,
    string? Instructions,
    int? MaxOutputTokens,
    float? Temperature,
    float? TopP,
    bool? Stream,
    [property: System.Text.Json.Serialization.JsonPropertyName("opentail_timing")] bool? OpentailTiming = null,
    [property: System.Text.Json.Serialization.JsonPropertyName("tools")] JsonElement? Tools = null,
    [property: System.Text.Json.Serialization.JsonPropertyName("previous_response_id")] string? PreviousResponseId = null);

public sealed record RespObject(
    string Id,
    string Object,
    long CreatedAt,
    string Model,
    string Status,
    RespOutputItem[] Output,
    RespUsage Usage,
    [property: System.Text.Json.Serialization.JsonPropertyName("opentail_timing")] OpenTailResponseTiming? OpenTailTiming = null);

public sealed record RespOutputItem(
    string Id,
    string Type,
    string Role,
    RespContentPart[] Content);

public sealed record RespContentPart(string Type, string Text);

public sealed record RespUsage(int InputTokens, int OutputTokens, int TotalTokens);

// Streaming event types

public sealed record RespCreatedEvent(string Type, RespObject Response);

public sealed record RespOutputItemAddedEvent(string Type, int OutputIndex, RespOutputItem Item);

public sealed record RespContentPartAddedEvent(
    string Type, string ItemId, int OutputIndex, int ContentIndex, RespContentPart Part);

public sealed record RespOutputTextDeltaEvent(
    string Type, string ItemId, int OutputIndex, int ContentIndex, string Delta);

public sealed record RespOutputTextDoneEvent(
    string Type, string ItemId, int OutputIndex, int ContentIndex, string Text);

public sealed record RespOutputItemDoneEvent(string Type, int OutputIndex, RespOutputItem Item);

public sealed record RespCompletedEvent(string Type, RespObject Response);
