
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

    public static string? ValidateAnthropic(AnthropicMessageRequest r) =>
        PositiveInt("max_tokens", r.MaxTokens)
        ?? Range("temperature", r.Temperature, 0f, 1f)
        ?? Range("top_p", r.TopP, 0f, 1f)
        ?? NonNegativeInt("top_k", r.TopK)
        ?? StopSequenceList("stop_sequences", r.StopSequences, MaxAnthropicStopSequences);

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

    private static string? PositiveInt(string name, int? value) =>
        value is { } v && v < 1 ? $"{name} must be at least 1 (got {v})" : null;

    private static string? NonNegativeInt(string name, int? value) =>
        value is { } v && v < 0 ? $"{name} must not be negative (got {v})" : null;

    private static string? Range(string name, float? value, float min, float max) =>
        value is { } v && !(v >= min && v <= max)   // also rejects NaN
            ? $"{name} must be between {min:0.##} and {max:0.##} (got {v})"
            : null;
}
