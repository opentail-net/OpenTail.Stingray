using System.Collections.Frozen;

namespace OpenTail.Stingray.Core;

/// <summary>
/// The closed set of attribute KEYS that OpenTelemetry metrics, spans, span events and span links may carry.
/// Plan: <c>docs/3-product-and-runtime/2026-10-12-opentelemetry-observability-plan.md</c>, section 7 (cardinality and privacy).
///
/// <para>Section 7 forbids prompt and completion text, tool arguments and results, the canonical history prefix, image and
/// audio bytes, request, session and user identifiers, client-supplied <c>user</c> and <c>metadata</c> fields, file paths, raw
/// model paths, and exception messages. A key on this list is a name, not a value: each key here carries a closed enum, a
/// count, a duration, a dimension, or a catalogue model id. Adding a key is a review decision, and the allow-list tests
/// fail until it is made here.</para>
///
/// <para>Values are checked separately (the canary tests). This list checks names only.</para>
/// </summary>
public static class TelemetryAttributePolicy
{
    public static readonly FrozenSet<string> AllowedAttributeKeys =
        new[]
        {
            // Operation and model identity (section 7, "allowed metric attributes").
            "gen_ai.operation.name",
            "gen_ai.request.model",
            "opentail.endpoint",
            "opentail.backend",
            "opentail.backend.requested",
            "opentail.architecture",
            "opentail.forward_pass",
            // Outcome and error classification (closed enums).
            "outcome",
            "error.type",
            "opentail.reason",
            "opentail.fallback.reason",
            // Closed-enum breakdowns (section 6, item 6).
            "opentail.sampling.path",
            "opentail.speculative.mode",
            "opentail.profile.phase",
            // Token counts and timings (numeric values only).
            "gen_ai.usage.input_tokens",
            "gen_ai.usage.output_tokens",
            "opentail.prompt_tokens",
            "opentail.output_tokens",
            "opentail.reused_tokens",
            "opentail.time_to_first_token_ms",
        }.ToFrozenSet(System.StringComparer.Ordinal);
}
