using System.Diagnostics;
using System.Diagnostics.Metrics;

namespace OpenTail.Stingray.Server;

/// <summary>
/// In-process counters scraped by <c>/metrics</c> and exposed to System.Diagnostics.Metrics listeners.
/// Registered as a singleton by <see cref="ServiceCollectionExtensions.AddOpenTailStingray"/>.
/// </summary>
public sealed class ServerMetrics
{
    /// <summary>Name of the <see cref="ActivitySource"/> for server request spans; add it to a tracer provider to export them.</summary>
    public const string ActivitySourceName = "OpenTail.Stingray.Server";
    internal static readonly ActivitySource Activities = new(ActivitySourceName, "1.0.0");

    private static readonly Meter s_meter = new("OpenTail.Stingray.Server", "1.0.0");
    private static readonly Counter<long> s_requestsCounter = s_meter.CreateCounter<long>("opentailllm.requests", "requests", "Total admitted inference requests");
    private static readonly Counter<long> s_tokensCounter = s_meter.CreateCounter<long>("opentailllm.tokens", "tokens", "Total tokens generated");
    private static readonly Counter<long> s_rejectionsCounter = s_meter.CreateCounter<long>("opentailllm.overload_rejections", "requests", "Total requests rejected due to overload");
    private static readonly Histogram<double> s_ttftHistogram = s_meter.CreateHistogram<double>("opentailllm.ttft_ms", "ms", "Time to first token latency");
    private static readonly Histogram<double> s_itlHistogram = s_meter.CreateHistogram<double>("opentailllm.itl_ms", "ms", "Inter-token generation duration");

    // GenAI-convention server metrics (Development status upstream; see the observability plan's mapping table).
    // Seconds, tagged with a closed endpoint value. Added beside the legacy opentailllm.* instruments, which stay for one release.
    private static readonly InstrumentAdvice<double> s_latencyAdvice = new()
    {
        HistogramBucketBoundaries = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60],
    };
    private static readonly InstrumentAdvice<double> s_perTokenAdvice = new()
    {
        HistogramBucketBoundaries = [0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1],
    };
    private static readonly Histogram<double> s_genAiRequestDuration = s_meter.CreateHistogram<double>(
        "gen_ai.server.request.duration", "s", "Server operation duration, from the start of serving (after parsing and template rendering) to the response being complete", advice: s_latencyAdvice);
    private static readonly Histogram<double> s_genAiTtft = s_meter.CreateHistogram<double>(
        "gen_ai.server.time_to_first_token", "s", "Time from the start of serving to the first generated token", advice: s_latencyAdvice);
    private static readonly Histogram<double> s_genAiTpot = s_meter.CreateHistogram<double>(
        "gen_ai.server.time_per_output_token", "s", "Decode time per output token after the first, one sample per request", advice: s_perTokenAdvice);

    private static readonly Counter<long> s_rejected = s_meter.CreateCounter<long>(
        "opentail.requests.rejected", "{request}", "Requests rejected before admission, by reason");
    private static readonly UpDownCounter<long> s_active = s_meter.CreateUpDownCounter<long>(
        "opentail.requests.active", "{request}", "Generation requests currently being served");
    private static readonly Histogram<double> s_queueWait = s_meter.CreateHistogram<double>(
        "opentail.queue.wait", "s", "Time from the start of serving until the engine admits the request", advice: s_latencyAdvice);

    private readonly Stopwatch _uptime = Stopwatch.StartNew();
    private long _totalRequests;
    private long _totalTokens;
    private long _overloadRejections;
    private readonly LatencyHistogram _queueLatency = new();
    private readonly LatencyHistogram _timeToFirstToken = new();
    private readonly LatencyHistogram _generationDuration = new();
    private readonly LatencyHistogram _requestDuration = new();

    /// <summary>Wall-clock time since the metrics instance was constructed.</summary>
    public TimeSpan Uptime => _uptime.Elapsed;

    /// <summary>Lifetime count of inference requests admitted to the engine.</summary>
    public long TotalRequests => Interlocked.Read(ref _totalRequests);

    /// <summary>Generation requests rejected before entering the engine because the bounded queue was full.</summary>
    public long OverloadRejections => Interlocked.Read(ref _overloadRejections);

    /// <summary>Lifetime count of tokens emitted by the engine (text + reasoning).</summary>
    public long TotalTokens => Interlocked.Read(ref _totalTokens);

    /// <summary>Increments <see cref="TotalRequests"/> by one. Called once per inbound HTTP request.</summary>
    public void RecordRequest()
    {
        Interlocked.Increment(ref _totalRequests);
        s_requestsCounter.Add(1);
    }

    /// <summary>Records one request rejected by the server's bounded admission queue.</summary>
    public void RecordOverloadRejection()
    {
        Interlocked.Increment(ref _overloadRejections);
        s_rejectionsCounter.Add(1);
        s_rejected.Add(1, new KeyValuePair<string, object?>("opentail.reason", "overload"));
    }

    /// <summary>Adds <paramref name="count"/> to <see cref="TotalTokens"/>.</summary>
    public void RecordTokens(long count)
    {
        Interlocked.Add(ref _totalTokens, count);
        s_tokensCounter.Add(count);
    }

    /// <summary>Starts latency accounting at engine-submission time for one generation request.</summary>
    /// <param name="endpoint">Closed set of endpoint names (for example <c>openai.chat_completions</c>); never client-supplied text.</param>
    /// <param name="model">The server-side resolved model id (never the client-supplied string). Path-like or over-long values are reported as <c>other</c>.</param>
    /// <param name="requestAborted">Used only to tell a cancelled request from one that ended without finishing generation.</param>
    public ServingRequestTiming BeginServingRequest(string endpoint = "unknown", string? model = null, CancellationToken requestAborted = default) =>
        new(this, new ServingTags(endpoint, SanitizeModel(model)), requestAborted);

    internal static string SanitizeModel(string? model) =>
        string.IsNullOrEmpty(model) ? "unknown"
        : model.Length > 64 || model.AsSpan().IndexOfAny('/', '\\', ':') >= 0 ? "other"
        : model;

    internal bool OperationMetricsEnabled => s_genAiRequestDuration.Enabled;

    /// <summary>Records <c>gen_ai.server.request.duration</c> for a non-chat operation. Operation/endpoint are fixed strings; outcome is success, client_error, error or cancelled.</summary>
    internal void RecordOperation(string operation, string endpoint, TimeSpan elapsed, string outcome)
    {
        if (!s_genAiRequestDuration.Enabled) return;
        s_genAiRequestDuration.Record(elapsed.TotalSeconds, new TagList
        {
            { "gen_ai.operation.name", operation },
            { "gen_ai.request.model", "unknown" },
            { "opentail.endpoint", endpoint },
            { "outcome", outcome },
        });
    }

    private static TagList EndpointTags(ServingTags endpoint) => new()
    {
        { "gen_ai.operation.name", "chat" },
        { "gen_ai.request.model", endpoint.Model },
        { "opentail.endpoint", endpoint.Endpoint },
    };

    internal void RecordQueueLatency(TimeSpan elapsed, ServingTags endpoint)
    {
        _queueLatency.Observe(elapsed);
        if (s_queueWait.Enabled) s_queueWait.Record(elapsed.TotalSeconds, EndpointTags(endpoint));
    }
    internal bool AddActive(long delta, ServingTags endpoint)
    {
        if (!s_active.Enabled) return false;
        s_active.Add(delta, new KeyValuePair<string, object?>("opentail.endpoint", endpoint.Endpoint));
        return true;
    }
    internal void RecordTimeToFirstToken(TimeSpan elapsed, ServingTags endpoint)
    {
        _timeToFirstToken.Observe(elapsed);
        s_ttftHistogram.Record(elapsed.TotalMilliseconds);
        if (s_genAiTtft.Enabled) s_genAiTtft.Record(elapsed.TotalSeconds, EndpointTags(endpoint));
    }
    internal void RecordRequestDurationAndRate(TimeSpan total, ServingTags endpoint, TimeSpan decode, long outputTokens, string outcome)
    {
        if (s_genAiRequestDuration.Enabled)
        {
            var tags = EndpointTags(endpoint);
            tags.Add("outcome", outcome);
            s_genAiRequestDuration.Record(total.TotalSeconds, tags);
        }
        if (outputTokens > 1 && decode > TimeSpan.Zero && s_genAiTpot.Enabled)
            s_genAiTpot.Record(decode.TotalSeconds / (outputTokens - 1), EndpointTags(endpoint));
    }
    internal void RecordGenerationDuration(TimeSpan elapsed)
    {
        _generationDuration.Observe(elapsed);
        s_itlHistogram.Record(elapsed.TotalMilliseconds);
    }
    internal void RecordRequestDuration(TimeSpan elapsed) => _requestDuration.Observe(elapsed);

    /// <summary>Time from engine submission until the request is admitted.</summary>
    public ServerLatencySummary QueueLatencySummary => _queueLatency.Snapshot();

    /// <summary>Time from engine submission until the first generated token.</summary>
    public ServerLatencySummary TimeToFirstTokenSummary => _timeToFirstToken.Snapshot();

    /// <summary>Time from the first generated token until generation completes.</summary>
    public ServerLatencySummary GenerationDurationSummary => _generationDuration.Snapshot();

    /// <summary>Generation and response-delivery duration per request.</summary>
    public ServerLatencySummary RequestDurationSummary => _requestDuration.Snapshot();

    /// <summary>Renders bounded Prometheus histograms for serving latency.</summary>
    public string RenderServingLatencyMetrics() =>
        _queueLatency.Render("opentail_llm_queue_latency_seconds",
            "Time from engine submission until the request is admitted") +
        _timeToFirstToken.Render("opentail_llm_time_to_first_token_seconds",
            "Time from engine submission until the first generated token") +
        _generationDuration.Render("opentail_llm_generation_duration_seconds",
            "Time from the first generated token until generation completes") +
        _requestDuration.Render("opentail_llm_request_generation_duration_seconds",
            "Generation and response-delivery duration per request");
}

/// <summary>Per-request serving stopwatch. It is intentionally endpoint-owned, so it starts after
/// request parsing/template rendering and includes generation plus response serialization.</summary>
public sealed class ServingRequestTiming : IDisposable
{
    private readonly ServerMetrics _metrics;
    private readonly long _started = Stopwatch.GetTimestamp();

    /// <summary>The engine's per-request stats, taken from the final Stop chunk; null if the engine did not supply them.</summary>
    public OpenTail.Stingray.Engine.GenerationStats? EngineStats { get; set; }
    private long _firstToken;
    private long _generationEnd;
    private bool _admitted;
    private bool _completed;

    private readonly ServingTags _endpoint;
    private long _outputTokens;

    private readonly CancellationToken _requestAborted;

    // The operation span: started with the timing (so it matches gen_ai.server.request.duration's boundaries),
    // parented by the ASP.NET Core request span through Activity.Current. Null, and free, with no listener.
    private readonly Activity? _activity;

    internal ServingRequestTiming(ServerMetrics metrics, ServingTags endpoint, CancellationToken requestAborted = default)
    {
        _requestAborted = requestAborted;
        if (ServerMetrics.Activities.HasListeners())
        {
            _activity = ServerMetrics.Activities.StartActivity("chat " + endpoint.Model, ActivityKind.Internal);
            _activity?.SetTag("gen_ai.operation.name", "chat");
            _activity?.SetTag("gen_ai.request.model", endpoint.Model);
            _activity?.SetTag("opentail.endpoint", endpoint.Endpoint);
        }
        _metrics = metrics;
        _endpoint = endpoint;

        _activeCounted = metrics.AddActive(1, endpoint);
    }

    // Instrument.Enabled can flip mid-request; decrement only if the increment was actually recorded.
    private readonly bool _activeCounted;

    /// <summary>Output token count for this request, used for time per output token. Call before <see cref="Complete"/>.</summary>
    public void SetOutputTokens(long count) => Interlocked.Exchange(ref _outputTokens, count);

    /// <summary>Marks the first engine event (the usage/admission chunk on supported engines).</summary>
    public void MarkAdmitted()
    {
        if (_admitted) return;
        _admitted = true;
        _metrics.RecordQueueLatency(Stopwatch.GetElapsedTime(_started), _endpoint);
    }

    /// <summary>Marks the first generated text or reasoning token.</summary>
    public void MarkFirstToken()
    {
        if (Interlocked.CompareExchange(ref _firstToken, Stopwatch.GetTimestamp(), 0) != 0) return;
        _metrics.RecordTimeToFirstToken(Stopwatch.GetElapsedTime(_started), _endpoint);
    }

    /// <summary>
    /// Milliseconds from request start to the first generated token, or null before
    /// <see cref="MarkFirstToken"/> has fired. Endpoint-owned so a single call site can read it
    /// for a per-response extension field without duplicating the stopwatch.
    /// </summary>
    public double? TimeToFirstTokenMs
    {
        get
        {
            long first = Interlocked.Read(ref _firstToken);
            return first == 0 ? null : Stopwatch.GetElapsedTime(_started, first).TotalMilliseconds;
        }
    }

    /// <summary>Milliseconds from request start to now. Meaningful once the request has completed.</summary>
    public double ElapsedMs => Stopwatch.GetElapsedTime(_started).TotalMilliseconds;

    /// <summary>
    /// Marks the moment the last token has been produced by the engine — i.e. immediately after
    /// the token-generating <c>await foreach</c> loop finishes, before any response
    /// serialization/delivery work (JSON body write, trailing SSE frames, tool-call parsing).
    /// Endpoints call this explicitly because <see cref="Complete"/> fires from a <c>using</c>
    /// block at method exit, which on the non-streaming path is AFTER the full response has been
    /// written to the client — recording generation duration there means a slow client or a large
    /// response inflates the reported decode/inter-token rate with delivery time that has nothing
    /// to do with the engine. Safe to call at most once; a cancelled/errored request that never
    /// reaches this call still gets a generation-duration sample from <see cref="Complete"/>'s
    /// fallback, using dispose time as before.
    /// </summary>
    public void MarkGenerationComplete()
    {
        if (Interlocked.CompareExchange(ref _generationEnd, Stopwatch.GetTimestamp(), 0) != 0) return;
        long first = Interlocked.Read(ref _firstToken);
        if (first != 0)
            _metrics.RecordGenerationDuration(Stopwatch.GetElapsedTime(first, Interlocked.Read(ref _generationEnd)));
    }

    /// <summary>Marks a normal generation completion.</summary>
    public void Complete()
    {
        if (_completed) return;
        _completed = true;
        if (_activeCounted) _metrics.AddActive(-1, _endpoint);
        long end = Stopwatch.GetTimestamp();
        _metrics.RecordRequestDuration(Stopwatch.GetElapsedTime(_started, end));
        long firstToken = Interlocked.Read(ref _firstToken);
        long genEnd = Interlocked.Read(ref _generationEnd);
        string outcome = genEnd != 0 ? "success" : _requestAborted.IsCancellationRequested ? "cancelled" : "incomplete";
        _metrics.RecordRequestDurationAndRate(
            Stopwatch.GetElapsedTime(_started, end), _endpoint,
            firstToken != 0 ? Stopwatch.GetElapsedTime(firstToken, genEnd != 0 ? genEnd : end) : TimeSpan.Zero,
            Interlocked.Read(ref _outputTokens),
            outcome);
        if (_activity is not null)
        {
            try
            {
                _activity.SetTag("outcome", outcome);
                _activity.SetTag("gen_ai.usage.output_tokens", Interlocked.Read(ref _outputTokens));
                if (EngineStats is { } s)
                {
                    _activity.SetTag("gen_ai.usage.input_tokens", s.PromptTokens);
                    _activity.SetTag("opentail.reused_tokens", s.ReusedTokens);
                }
                if (firstToken != 0)
                    _activity.SetTag("opentail.time_to_first_token_ms", Stopwatch.GetElapsedTime(_started, firstToken).TotalMilliseconds);
            }
            catch
            {
                // Telemetry must never break serving.
            }
            finally
            {
                _activity.Dispose();
            }
        }
        // MarkGenerationComplete already recorded the accurate (delivery-excluded) sample on the
        // normal path. Only fall back to dispose-time here when it was never called — a
        // cancellation or exception before the generation loop finished — so those requests still
        // contribute a sample instead of silently vanishing from the histogram.
        if (Interlocked.Read(ref _generationEnd) == 0)
        {
            long first = Interlocked.Read(ref _firstToken);
            if (first != 0) _metrics.RecordGenerationDuration(Stopwatch.GetElapsedTime(first, end));
        }
    }

    public void Dispose() => Complete();
}

/// <summary>Fixed-bucket, lock-free Prometheus histogram. Bounded storage avoids per-model or
/// per-route labels, which would be dangerous under arbitrary client-supplied model names.</summary>
internal sealed class LatencyHistogram
{
    private static readonly double[] Bounds = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60];
    private readonly long[] _buckets = new long[Bounds.Length];
    private long _count;
    private long _sumTicks;

    public void Observe(TimeSpan elapsed)
    {
        double seconds = Math.Max(0, elapsed.TotalSeconds);
        for (int i = 0; i < Bounds.Length; i++)
            if (seconds <= Bounds[i]) Interlocked.Increment(ref _buckets[i]);
        Interlocked.Increment(ref _count);
        Interlocked.Add(ref _sumTicks, elapsed.Ticks);
    }

    /// <summary>
    /// Reads the histogram into a JSON-friendly summary. Called at the <c>/status</c> HTTP
    /// boundary only — the per-call array allocation is deliberate there and would not be
    /// acceptable anywhere near a decode step.
    /// </summary>
    public ServerLatencySummary Snapshot()
    {
        long count = Interlocked.Read(ref _count);
        double seconds = Interlocked.Read(ref _sumTicks) / (double)TimeSpan.TicksPerSecond;
        long[] cumulative = new long[Bounds.Length];
        for (int i = 0; i < Bounds.Length; i++) cumulative[i] = Interlocked.Read(ref _buckets[i]);

        return new ServerLatencySummary(
            Count: count,
            TotalSeconds: seconds,
            MeanMs: count == 0 ? 0 : seconds * 1000.0 / count,
            P50Ms: Quantile(cumulative, count, 0.50),
            P95Ms: Quantile(cumulative, count, 0.95),
            P99Ms: Quantile(cumulative, count, 0.99));
    }

    /// <summary>
    /// Smallest bucket bound whose cumulative count covers the quantile, in milliseconds. Buckets
    /// are already cumulative (<see cref="Observe"/> increments every bound the sample fits under),
    /// so this is a single scan. Returns null when there are no samples, or when the quantile lies
    /// beyond the largest bound — reporting the top bound in that case would understate the tail.
    /// </summary>
    private static double? Quantile(long[] cumulative, long count, double q)
    {
        if (count == 0) return null;
        double target = q * count;
        for (int i = 0; i < Bounds.Length; i++)
            if (cumulative[i] >= target) return Bounds[i] * 1000.0;
        return null;
    }

    public string Render(string name, string help)
    {
        var sb = new System.Text.StringBuilder();
        sb.Append("# HELP ").Append(name).Append(' ').Append(help).Append('\n');
        sb.Append("# TYPE ").Append(name).Append(" histogram\n");
        for (int i = 0; i < Bounds.Length; i++)
            sb.Append(name).Append("_bucket{le=\"").Append(Bounds[i].ToString(System.Globalization.CultureInfo.InvariantCulture))
                .Append("\"} ").Append(Interlocked.Read(ref _buckets[i])).Append('\n');
        sb.Append(name).Append("_bucket{le=\"+Inf\"} ").Append(Interlocked.Read(ref _count)).Append('\n');
        sb.Append(name).Append("_sum ").Append((Interlocked.Read(ref _sumTicks) / (double)TimeSpan.TicksPerSecond)
            .ToString("F6", System.Globalization.CultureInfo.InvariantCulture)).Append('\n');
        sb.Append(name).Append("_count ").Append(Interlocked.Read(ref _count)).Append('\n');
        return sb.ToString();
    }
}

/// <summary>Closed-vocabulary attribute values for one serving request (endpoint name and server-resolved model id).</summary>
internal readonly record struct ServingTags(string Endpoint, string Model);
