using System.Diagnostics;
using System.Diagnostics.Metrics;

namespace OpenTail.Stingray.Engine;

/// <summary>
/// Engine-level OpenTelemetry-compatible instruments (plan: docs/3-product-and-runtime/
/// 2026-10-12-opentelemetry-observability-plan.md, Phase 1). Built only on the in-box
/// <c>System.Diagnostics</c> Meter / ActivitySource, so the library references no OpenTelemetry
/// package; a host adds the SDK and calls <c>AddMeter(EngineTelemetry.MeterName)</c>.
/// One sample per request; nothing per token. No content, paths or identifiers are recorded.
/// </summary>
public static class EngineTelemetry
{
    public const string MeterName = "OpenTail.Stingray.Engine";
    public const string ActivitySourceName = "OpenTail.Stingray.Engine";

    /// <summary>Closed set of values for the <c>outcome</c> attribute.</summary>
    public const string OutcomeSuccess = "success", OutcomeError = "error", OutcomeCancelled = "cancelled";

    private static readonly Meter s_meter = new(MeterName, "1.0.0");
    internal static readonly ActivitySource Source = new(ActivitySourceName, "1.0.0");

    private static readonly Counter<long> s_requests = s_meter.CreateCounter<long>(
        "opentail.engine.requests", "{request}", "Generation requests handled by the engine, by outcome.");
    private static readonly Histogram<double> s_prefillDuration = s_meter.CreateHistogram<double>(
        "opentail.engine.prefill.duration", "s", "Prompt processing time (new prompt tokens only).");
    private static readonly Histogram<long> s_prefillTokens = s_meter.CreateHistogram<long>(
        "opentail.engine.prefill.tokens", "{token}", "New prompt tokens processed (reused prefix tokens excluded).");
    private static readonly Histogram<double> s_decodeDuration = s_meter.CreateHistogram<double>(
        "opentail.engine.decode.duration", "s", "Time from first generated token to end of generation.");
    private static readonly Histogram<long> s_decodeTokens = s_meter.CreateHistogram<long>(
        "opentail.engine.decode.tokens", "{token}", "Tokens generated.");
    private static readonly Histogram<double> s_firstToken = s_meter.CreateHistogram<double>(
        "opentail.engine.first_token.duration", "s", "Engine request start to first generated token.");
    private static readonly Histogram<double> s_total = s_meter.CreateHistogram<double>(
        "opentail.engine.generation.duration", "s", "Engine request start to generation complete.");

    private static readonly Histogram<double> s_modelLoad = s_meter.CreateHistogram<double>(
        "opentail.engine.model.load.duration", "s", "Time to build a runtime for a model (weights mapped, backend and forward pass created).");

    /// <summary>Records one model load. Architecture family is the model identity (never a path or file name).</summary>
    internal static void RecordModelLoad(string architecture, string forwardPass, string outcome, double seconds)
    {
        try
        {
            if (!s_modelLoad.Enabled) return;
            s_modelLoad.Record(seconds, new TagList
            {
                { "opentail.architecture", architecture },
                { "opentail.forward_pass", forwardPass },
                { "outcome", outcome },
            });
        }
        catch
        {
        }
    }

    /// <summary>True when anything listens; callers skip building the record otherwise.</summary>
    internal static bool Enabled =>
        s_requests.Enabled || s_prefillDuration.Enabled || s_decodeDuration.Enabled
        || s_firstToken.Enabled || s_total.Enabled;

    /// <summary>
    /// Records one finished request. Never throws: telemetry must not break inference.
    /// Times are in seconds; <paramref name="firstTokenSeconds"/> is negative when no token was produced.
    /// </summary>
    internal static void RecordGeneration(
        string forwardPass, string outcome,
        int newPromptTokens, double prefillSeconds,
        int decodeTokens, double firstTokenSeconds, double totalSeconds)
    {
        try
        {
            var tags = new TagList
            {
                { "opentail.forward_pass", forwardPass },
                { "outcome", outcome },
            };
            s_requests.Add(1, tags);
            if (newPromptTokens > 0 || prefillSeconds > 0)
            {
                s_prefillDuration.Record(prefillSeconds, tags);
                s_prefillTokens.Record(newPromptTokens, tags);
            }
            s_total.Record(totalSeconds, tags);
            if (firstTokenSeconds >= 0)
            {
                s_firstToken.Record(firstTokenSeconds, tags);
                s_decodeDuration.Record(Math.Max(0, totalSeconds - firstTokenSeconds), tags);
                s_decodeTokens.Record(decodeTokens, tags);
            }
        }
        catch
        {
            // Listener faults are contained; inference must never see them.
        }
    }
}
