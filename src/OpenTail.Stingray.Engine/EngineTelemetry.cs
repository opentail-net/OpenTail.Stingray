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

    // Tier 3 (explicit diagnostic mode): these exist ONLY when a profile switch is on (STINGRAY_MOE_PHASE_TIMING=1 or
    // STINGRAY_PROFILE_GPU_SPLIT=1). The profilers perturb what they measure, so these numbers are labelled as such
    // and must never feed a benchmark comparison. Observable, so reading costs nothing unless collected.
    // The Vulkan switch is read from the environment here, and VulkanBackend is touched only inside the guarded callbacks,
    // so this class never triggers Vulkan's type initialiser on a machine without Vulkan.
    private static readonly bool s_vulkanProfile = Environment.GetEnvironmentVariable("STINGRAY_PROFILE_GPU_SPLIT") == "1";
    private static readonly ObservableCounter<double>? s_profilePhase =
        MoePhaseTiming.Enabled || s_vulkanProfile
            ? s_meter.CreateObservableCounter(
                "opentail.profile.phase.duration", ProfileMeasurements, "s",
                "DIAGNOSTIC (perturbed by profiling, never use for benchmarks): accumulated seconds per profiled phase.")
            : null;
    private static readonly ObservableCounter<long>? s_profileDispatches =
        s_vulkanProfile
            ? s_meter.CreateObservableCounter(
                "opentail.profile.vulkan.dispatches", VulkanDispatches,
                "{submit}", "DIAGNOSTIC: Vulkan queue submits (each a fence wait) since the last profile reset.")
            : null;

    private static long VulkanDispatches()
    {
        try { return OpenTail.Stingray.Vulkan.VulkanBackend.GpuProfileSnapshot().Dispatches; }
        catch { return 0; }
    }

    private static IEnumerable<Measurement<double>> ProfileMeasurements()
    {
        var list = new List<Measurement<double>>();
        try
        {
            if (MoePhaseTiming.Enabled)
                foreach (var (phase, seconds) in MoePhaseTiming.Snapshot())
                    list.Add(new Measurement<double>(seconds, new KeyValuePair<string, object?>("opentail.profile.phase", phase)));
            if (s_vulkanProfile)
            {
                var (submitWaitMs, stagingMs, _) = OpenTail.Stingray.Vulkan.VulkanBackend.GpuProfileSnapshot();
                list.Add(new Measurement<double>(submitWaitMs / 1000.0, new KeyValuePair<string, object?>("opentail.profile.phase", "vulkan.submit_wait")));
                list.Add(new Measurement<double>(stagingMs / 1000.0, new KeyValuePair<string, object?>("opentail.profile.phase", "vulkan.staging_copy")));
            }
        }
        catch
        {
        }
        return list;
    }

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

    // Tier 2: one histogram sample per continuous-batching decode ITERATION (never per token), only when a listener is attached.
    // One iteration serves many sequences, so this is cheap; it is held to the same overhead budget as everything else.
    private static readonly Histogram<long> s_batchSize = s_meter.CreateHistogram<long>(
        "opentail.engine.batch.size", "{sequence}", "Sequences advanced together in one decode iteration.",
        advice: new InstrumentAdvice<long> { HistogramBucketBoundaries = [1, 2, 4, 8, 16, 32, 64] });
    private static readonly Histogram<double> s_iterationDuration = s_meter.CreateHistogram<double>(
        "opentail.engine.iteration.duration", "s", "Wall time of the batched forward step of one decode iteration (results processing excluded).");

    internal static bool IterationEnabled => s_batchSize.Enabled || s_iterationDuration.Enabled;

    /// <summary>Records one decode iteration. <paramref name="argmaxPath"/> is true when the on-device argmax tail was used. Never throws.</summary>
    internal static void RecordIteration(int sequences, long startTicks, bool argmaxPath)
    {
        try
        {
            var tags = new TagList { { "opentail.sampling.path", argmaxPath ? "argmax" : "full_logits" } };
            if (s_batchSize.Enabled) s_batchSize.Record(sequences, tags);
            if (s_iterationDuration.Enabled) s_iterationDuration.Record(Stopwatch.GetElapsedTime(startTicks).TotalSeconds, tags);
        }
        catch
        {
        }
    }

    // Speculative decoding. Definitions (per verify step of the mode named in opentail.speculative.mode):
    //   mtp: one verify step = one drafts event; draft_tokens = drafts proposed to the verifier in that step;
    //        accepted_tokens = how many of those the verifier accepted (the verifier's own corrected token is NOT counted).
    // Only modes whose events carry both numbers are exported. Draft-model, prompt-lookup and DSpark are not exported
    // until each has the same written definition. Acceptance rate = rate(accepted_tokens) / rate(draft_tokens).
    private static readonly Counter<long> s_specSteps = s_meter.CreateCounter<long>(
        "opentail.engine.speculative.drafts", "{step}", "Speculative verify steps (one per drafting round).");
    private static readonly Counter<long> s_specDraftTokens = s_meter.CreateCounter<long>(
        "opentail.engine.speculative.draft_tokens", "{token}", "Draft tokens proposed to the verifier.");
    private static readonly Counter<long> s_specAccepted = s_meter.CreateCounter<long>(
        "opentail.engine.speculative.accepted_tokens", "{token}", "Draft tokens the verifier accepted.");

    /// <summary>Records one speculative verify step. <paramref name="mode"/> is a closed label (currently only <c>mtp</c>). Never throws.</summary>
    internal static void RecordSpeculativeStep(string mode, long proposed, long accepted)
    {
        try
        {
            if (!s_specSteps.Enabled && !s_specDraftTokens.Enabled && !s_specAccepted.Enabled) return;
            var tags = new TagList { { "opentail.speculative.mode", mode } };
            s_specSteps.Add(1, tags);
            s_specDraftTokens.Add(proposed, tags);
            s_specAccepted.Add(accepted, tags);
        }
        catch
        {
        }
    }

    /// <summary>
    /// Carries the <c>model.load</c> span's context out of a loader to whoever started it, so requests that waited for a
    /// shared load can link to it. Set <see cref="LoadHolder"/> on the loading flow before building the runtime; the load
    /// span fills <see cref="Context"/> when it starts. Contexts only, never content.
    /// </summary>
    public sealed class LoadContextHolder
    {
        /// <summary>The load span's context, or default when no span listener was attached.</summary>
        public ActivityContext Context { get; set; }
    }

    /// <summary>Ambient holder for the load currently being built on this flow (see <see cref="LoadContextHolder"/>).</summary>
    public static readonly AsyncLocal<LoadContextHolder?> LoadHolder = new();

    /// <summary>
    /// Starts a span without letting a faulty <see cref="ActivityListener"/> throw into the caller. .NET invokes the listener's
    /// <c>ActivityStarted</c> inside <c>Start()</c>, so a listener that throws would otherwise fail the request or the model load it
    /// merely observes (plan acceptance: a telemetry failure cannot break inference). On failure the caller's
    /// <see cref="Activity.Current"/> is put back and no span is recorded. Null with no listener, and allocation-free.
    /// </summary>
    internal static Activity? TryStartActivity(string name, ActivityKind kind, ActivityContext parent)
    {
        if (!Source.HasListeners()) return null;
        var saved = Activity.Current;
        try
        {
            return Source.StartActivity(name, kind, parent);
        }
        catch
        {
            Activity.Current = saved;
            return null;
        }
    }

    /// <summary>Starts the <c>model.load</c> span as a new root (a load is shared, so it is never a child of one request). Null with no listener.</summary>
    internal static ModelLoadScope StartModelLoad()
    {
        if (!Source.HasListeners()) return default;
        // StartActivity inherits Activity.Current whatever parent context is passed, so clear it for the
        // start and restore the caller's context when the scope ends.
        var saved = Activity.Current;
        Activity.Current = null;
        var activity = TryStartActivity("model.load", ActivityKind.Internal, default);
        if (activity is not null && LoadHolder.Value is { } holder) holder.Context = activity.Context;
        return new ModelLoadScope(activity, saved, swapped: true);
    }

    /// <summary>Owns the <c>model.load</c> span and puts the caller's <see cref="Activity.Current"/> back on dispose.</summary>
    internal readonly struct ModelLoadScope : IDisposable
    {
        private readonly Activity? _saved;
        private readonly bool _swapped;

        public ModelLoadScope(Activity? activity, Activity? saved, bool swapped)
        {
            Activity = activity;
            _saved = saved;
            _swapped = swapped;
        }

        public Activity? Activity { get; }

        public void Dispose()
        {
            // A listener's ActivityStopped runs inside Dispose; it must not throw out of the load it observes.
            try { Activity?.Dispose(); }
            catch { }
            finally
            {
                if (_swapped) System.Diagnostics.Activity.Current = _saved;
            }
        }
    }

    private static readonly Histogram<double> s_gpuLayerShare = s_meter.CreateHistogram<double>(
        "opentail.engine.placement.gpu_layer_share", "1", "Share of the model's layers placed on the GPU at load (0 = all CPU). Selecting a GPU does not mean the whole model runs on it.");
    private static readonly Counter<long> s_backendFallbacks = s_meter.CreateCounter<long>(
        "opentail.engine.backend.fallbacks", "{load}", "Loads that ran on a different backend than the one explicitly requested; reason is a closed category.");

    /// <summary>Selected backend as a closed label (auto/cpu/cuda/vulkan, or unknown).</summary>
    internal static string BackendLabel(ExecutionPlan plan) =>
        plan.BackendPlan is { } b ? b.Backend.ToString().ToLowerInvariant() : "unknown";

    /// <summary>
    /// Records, once per successful load, where the model was placed and whether an explicitly requested backend was
    /// not honoured. Reasons are a closed set: <c>unavailable</c> (the requested GPU backend is not available) or <c>other</c>.
    /// </summary>
    internal static void RecordPlacement(ExecutionPlan plan)
    {
        try
        {
            string backend = BackendLabel(plan);
            if (plan.Placement is { TotalLayers: > 0 } p && s_gpuLayerShare.Enabled)
                s_gpuLayerShare.Record((double)p.GpuLayers / p.TotalLayers, new TagList { { "opentail.backend", backend } });

            if (!s_backendFallbacks.Enabled || plan.BackendPlan is not { } bp) return;
            string requested = plan.Request?.Backend?.Trim().ToLowerInvariant() ?? "auto";
            if (requested is "" or "auto" or "hybrid" || requested == backend) return;
            string reason = (requested == "cuda" && !bp.CudaAvailable) || (requested == "vulkan" && !bp.VulkanAvailable)
                ? "unavailable" : "other";
            s_backendFallbacks.Add(1, new TagList
            {
                { "opentail.backend.requested", requested is "cpu" or "cuda" or "vulkan" ? requested : "other" },
                { "opentail.backend", backend },
                { "opentail.fallback.reason", reason },
            });
        }
        catch
        {
        }
    }

    /// <summary>Records one model load. Architecture family is the model identity (never a path or file name).</summary>
    internal static void RecordModelLoad(Activity? activity, string architecture, string forwardPass, string outcome, double seconds, string backend = "unknown")
    {
        try
        {
            if (activity is not null)
            {
                activity.SetTag("opentail.architecture", architecture);
                activity.SetTag("opentail.forward_pass", forwardPass);
                activity.SetTag("opentail.backend", backend);
                activity.SetTag("outcome", outcome);
                if (outcome == OutcomeError) activity.SetStatus(ActivityStatusCode.Error);
            }
            if (!s_modelLoad.Enabled) return;
            s_modelLoad.Record(seconds, new TagList
            {
                { "opentail.architecture", architecture },
                { "opentail.forward_pass", forwardPass },
                { "opentail.backend", backend },
                { "outcome", outcome },
            });
        }
        catch
        {
        }
    }

    /// <summary>
    /// Records a prefill or decode measurement reported by an event source that has no request object (the CLI run
    /// command). A negative token count means "not part of this call". Never throws.
    /// </summary>
    internal static void RecordPrefillAndDecode(string forwardPass, int newPromptTokens, double prefillSeconds, int decodeTokens, double decodeSeconds)
    {
        try
        {
            var tags = new TagList { { "opentail.forward_pass", forwardPass }, { "outcome", OutcomeSuccess } };
            if (newPromptTokens >= 0)
            {
                s_prefillDuration.Record(prefillSeconds, tags);
                s_prefillTokens.Record(newPromptTokens, tags);
            }
            if (decodeTokens >= 0)
            {
                s_requests.Add(1, tags);
                s_decodeDuration.Record(decodeSeconds, tags);
                s_decodeTokens.Record(decodeTokens, tags);
            }
        }
        catch
        {
        }
    }

    /// <summary>Starts and ends one synthetic span so a self-test can prove spans reach a listener. Public surface: the CLI self-test command.</summary>
    public static void RunSelfTestSpan()
    {
        using var activity = Source.StartActivity("opentail.engine.selftest");
        activity?.SetTag("outcome", OutcomeSuccess);
    }

    /// <summary>
    /// Records a finished continuous-batching request from its stats snapshot (millisecond resolution) and, when a span
    /// listener is attached, emits the generation span after the fact with its real start time and duration.
    /// Never throws; restores <see cref="Activity.Current"/>.
    /// </summary>
    internal static void RecordBatchedRequest(GenerationStats s, string outcome, ActivityContext parent, DateTimeOffset startedAt)
    {
        const string ForwardPass = "continuous_batching";
        try
        {
            if (Enabled)
                RecordGeneration(ForwardPass, outcome, s.NewPromptTokens, s.PrefillMs / 1000.0, s.OutputTokens,
                    s.FirstTokenMs >= 0 ? s.FirstTokenMs / 1000.0 : -1, s.TotalMs / 1000.0);
            if (!Source.HasListeners()) return;
            var saved = Activity.Current;
            try
            {
                var activity = Source.StartActivity("opentail.engine.generate", ActivityKind.Internal, parent,
                    tags: null, links: null, startTime: startedAt);
                if (activity is null) return;
                activity.SetEndTime(startedAt.UtcDateTime + TimeSpan.FromMilliseconds(s.TotalMs));
                EndActivity(activity, outcome, s.PromptTokens, s.ReusedTokens, s.OutputTokens);
            }
            finally
            {
                Activity.Current = saved;
            }
        }
        catch
        {
        }
    }

    /// <summary>Tags and ends the generation span (no-op for null). Never throws; counts only, no content.</summary>
    internal static void EndActivity(Activity? activity, string outcome, int promptTokens, int reusedTokens, int decodeTokens)
    {
        if (activity is null) return;
        try
        {
            activity.SetTag("outcome", outcome);
            activity.SetTag("opentail.prompt_tokens", promptTokens);
            activity.SetTag("opentail.reused_tokens", reusedTokens);
            activity.SetTag("opentail.output_tokens", decodeTokens);
            if (outcome == OutcomeError) activity.SetStatus(ActivityStatusCode.Error);
        }
        catch
        {
        }
        finally
        {
            // ActivityStopped runs inside Dispose; a throwing listener must not escape into the request's own finally block.
            try { activity.Dispose(); }
            catch { }
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
