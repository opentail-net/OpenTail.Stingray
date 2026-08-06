using System.Text.Json.Serialization;

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
    [property: JsonPropertyName("total_ms")] double TotalMs);
