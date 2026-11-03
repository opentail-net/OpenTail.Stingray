
namespace OpenTail.Stingray.Server.Endpoints;

/// <summary>
/// Range and support checks for the sampling fields of the OpenAI chat and Anthropic messages requests
/// (docs/reference/request-parameter-audit-2026-10-31.md, findings F1 and F3). A field the server cannot honour, or a value outside
/// the range the API defines, is refused with a 400 that names the field, instead of being passed to the engine or dropped.
/// Unknown JSON properties are NOT checked here: clients send vendor fields and those must keep working.
/// </summary>
internal static class RequestParameterValidation
{
    /// <summary>Returns the error message for the first invalid field, or null when the request is acceptable.</summary>
    public static string? ValidateOpenAi(ChatCompletionRequest r) =>
        PositiveInt("max_tokens", r.MaxTokens)
        ?? PositiveInt("max_completion_tokens", r.MaxCompletionTokens)
        ?? Range("temperature", r.Temperature, 0f, 2f)
        ?? Range("top_p", r.TopP, 0f, 1f)
        ?? NonNegativeInt("top_k", r.TopK)
        ?? Range("presence_penalty", r.PresencePenalty, -2f, 2f)
        ?? Range("frequency_penalty", r.FrequencyPenalty, -2f, 2f)
        ?? (r.N is { } n && n != 1 ? $"n must be 1: this server returns exactly one choice (got {n})" : null)
        ?? ParseOpenAiStops(r.Stop, out _);

    /// <summary>
    /// <c>/v1/responses</c>: ranges for the three sampling fields it reads, and the two request features it does not implement, refused with a
    /// 400 that names them instead of being dropped (docs/reference/request-parameter-audit-other-endpoints-2026-11-01.md). An empty tools array is accepted.
    /// </summary>
    public static string? ValidateResponses(ResponsesRequest r) =>
        PositiveInt("max_output_tokens", r.MaxOutputTokens)
        ?? Range("temperature", r.Temperature, 0f, 2f)
        ?? Range("top_p", r.TopP, 0f, 1f)
        ?? (r.Tools is { ValueKind: System.Text.Json.JsonValueKind.Array } tools && tools.GetArrayLength() > 0
            ? "tools are not supported on /v1/responses: this endpoint answers with text only (use /v1/chat/completions for tool calls)" : null)
        ?? (!string.IsNullOrEmpty(r.PreviousResponseId)
            ? "previous_response_id is not supported: /v1/responses is stateless here, send the whole conversation in 'input'" : null);

    /// <summary>A session turn's sampling fields, with the same ranges as the chat endpoints (the session route accepted any value before).</summary>
    public static string? ValidateSessionTurn(SessionTurnRequest r) =>
        Range("temperature", r.Temperature, 0f, 2f)
        ?? Range("top_p", r.TopP, 0f, 1f);

    public static string? ValidateAnthropic(AnthropicMessageRequest r) =>
        PositiveInt("max_tokens", r.MaxTokens)
        ?? Range("temperature", r.Temperature, 0f, 1f)
        ?? Range("top_p", r.TopP, 0f, 1f)
        ?? NonNegativeInt("top_k", r.TopK)
        ?? StopSequenceList("stop_sequences", r.StopSequences, MaxAnthropicStopSequences)
        ?? AnthropicToolChoice(r.ToolChoice);

    /// <summary>Anthropic <c>tool_choice</c>: auto and none are honoured; any and tool would force tool use, which is not supported here.</summary>
    private static string? AnthropicToolChoice(System.Text.Json.JsonElement? toolChoice)
    {
        if (toolChoice is not { } tc || tc.ValueKind is System.Text.Json.JsonValueKind.Null or System.Text.Json.JsonValueKind.Undefined) return null;
        if (tc.ValueKind != System.Text.Json.JsonValueKind.Object
            || !tc.TryGetProperty("type", out var type) || type.ValueKind != System.Text.Json.JsonValueKind.String)
            return "tool_choice must be an object with a string type (auto, none, any or tool)";
        return type.GetString() switch
        {
            "auto" or "none" => null,
            "any" or "tool" => $"tool_choice type '{type.GetString()}' is not supported on /v1/messages: forced tool use is not implemented here (auto and none are)",
            var other => $"tool_choice type '{other}' is not one of auto, none, any or tool",
        };
    }

    /// <summary>`tool_choice: {type: none}`: the model is not offered the declared tools (nothing in the prompt invites a tool call).</summary>
    public static AnthropicMessageRequest ApplyAnthropicToolChoice(AnthropicMessageRequest r) =>
        r.ToolChoice is { ValueKind: System.Text.Json.JsonValueKind.Object } tc
        && tc.TryGetProperty("type", out var type) && type.ValueKind == System.Text.Json.JsonValueKind.String && type.GetString() == "none"
            ? r with { Tools = null }
            : r;

    private const int MaxOpenAiStops = 4;
    private const int MaxAnthropicStopSequences = 16;

    /// <summary>`stop` is one string or an array of up to 4. Returns the error message, or null and the strings (null when absent).</summary>
    public static string? ParseOpenAiStops(System.Text.Json.JsonElement? stop, out string[]? stops)
    {
        stops = null;
        if (stop is not { } e || e.ValueKind is System.Text.Json.JsonValueKind.Null or System.Text.Json.JsonValueKind.Undefined) return null;
        string[] list;
        if (e.ValueKind == System.Text.Json.JsonValueKind.String) list = [e.GetString() ?? string.Empty];
        else if (e.ValueKind == System.Text.Json.JsonValueKind.Array)
        {
            var items = new List<string>();
            foreach (var item in e.EnumerateArray())
            {
                if (item.ValueKind != System.Text.Json.JsonValueKind.String) return "stop must be a string or an array of strings";
                items.Add(item.GetString() ?? string.Empty);
            }
            list = [.. items];
        }
        else return "stop must be a string or an array of strings";
        if (StopSequenceList("stop", list, MaxOpenAiStops) is { } error) return error;
        stops = list.Length == 0 ? null : list;
        return null;
    }

    private static string? StopSequenceList(string name, string[]? list, int max)
    {
        if (list is null) return null;
        if (list.Length > max) return $"{name} accepts at most {max} strings (got {list.Length})";
        foreach (string s in list)
            if (string.IsNullOrEmpty(s)) return $"{name} must not contain an empty string";
        return null;
    }

    /// <summary>`/v1/audio/speech` `speed`: OpenAI's documented range; a non-positive or NaN value used to be replaced by 1.0 silently.</summary>
    public static string? ValidateSpeechSpeed(float speed) => Range("speed", speed, 0.25f, 4.0f);

    /// <summary>`/v1/rerank` `top_n` must be at least 1 when given.</summary>
    public static string? ValidateRerankTopN(int? topN) => PositiveInt("top_n", topN);

    /// <summary>Spoken-language code for transcription: two or three letters (en, de, haw); not checked against Whisper's list.</summary>
    public static string? ValidateLanguageCode(string? language) =>
        string.IsNullOrEmpty(language) || (language.Length is 2 or 3 && language.All(char.IsAsciiLetter))
            ? null
            : $"language must be a two or three letter code such as 'en' (got '{language}')";

    private static string? PositiveInt(string name, int? value) =>
        value is { } v && v < 1 ? $"{name} must be at least 1 (got {v})" : null;

    private static string? NonNegativeInt(string name, int? value) =>
        value is { } v && v < 0 ? $"{name} must not be negative (got {v})" : null;

    private static string? Range(string name, float? value, float min, float max) =>
        value is { } v && !(v >= min && v <= max)   // also rejects NaN
            ? $"{name} must be between {min:0.##} and {max:0.##} (got {v})"
            : null;
}
