
namespace OpenTail.Stingray.Server.Endpoints;

/// <summary>
/// Opt-in per-response timing detail (§8 Phase 2 item 3 of the QoL plan). Answers "why was this
/// one response slow?" without requiring a separate <c>/status</c> poll timed around the request.
///
/// Opt-in, not always-on: a client that sets <c>opentail_timing: true</c> on its request gets the
/// field back; every other client sees exactly the stock OpenAI/Anthropic response shape it had
/// before this existed. That is the "without breaking protocol compatibility" requirement made
/// concrete — an always-on extension field risks a strict client that rejects unknown JSON
/// properties, however rare that is in practice; an opt-in field cannot regress anyone who never
/// asks for it.
/// </summary>
public sealed record OpenTailResponseTiming(
    [property: JsonPropertyName("time_to_first_token_ms")] double? TimeToFirstTokenMs,
    [property: JsonPropertyName("total_ms")] double TotalMs,
    // Engine-side breakdown (present when the engine supplied per-request stats). Counts and
    // durations only; omitted, not null, when unavailable so the original two fields are unchanged.
    [property: JsonPropertyName("prompt_tokens"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] int? PromptTokens = null,
    [property: JsonPropertyName("reused_prompt_tokens"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] int? ReusedPromptTokens = null,
    [property: JsonPropertyName("generated_tokens"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] int? GeneratedTokens = null,
    [property: JsonPropertyName("encode_ms"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] long? EncodeMs = null,
    [property: JsonPropertyName("prefill_ms"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] long? PrefillMs = null,
    [property: JsonPropertyName("decode_ms"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] long? DecodeMs = null)
{
    /// <summary>Builds the extension from the request's timing and, when available, the engine's stats.</summary>
    public static OpenTailResponseTiming From(ServingRequestTiming timing)
    {
        var s = timing.EngineStats;
        return s is null
            ? new OpenTailResponseTiming(timing.TimeToFirstTokenMs, timing.ElapsedMs)
            : new OpenTailResponseTiming(timing.TimeToFirstTokenMs, timing.ElapsedMs,
                s.PromptTokens, s.ReusedTokens, s.OutputTokens, s.EncodeMs, s.PrefillMs, s.DecodeMs);
    }
}
