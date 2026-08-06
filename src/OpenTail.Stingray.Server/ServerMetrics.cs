using System.Diagnostics;
using System.Diagnostics.Metrics;

namespace OpenTail.Stingray.Server;

/// <summary>
/// In-process counters scraped by <c>/metrics</c> and exposed to System.Diagnostics.Metrics listeners.
/// Registered as a singleton by <see cref="ServiceCollectionExtensions.AddOpenTailStingray"/>.
/// </summary>
public sealed class ServerMetrics
{
    private static readonly Meter s_meter = new("OpenTail.Stingray.Server", "1.0.0");
    private static readonly Counter<long> s_requestsCounter = s_meter.CreateCounter<long>("opentailllm.requests", "requests", "Total admitted inference requests");
    private static readonly Counter<long> s_tokensCounter = s_meter.CreateCounter<long>("opentailllm.tokens", "tokens", "Total tokens generated");
    private static readonly Counter<long> s_rejectionsCounter = s_meter.CreateCounter<long>("opentailllm.overload_rejections", "requests", "Total requests rejected due to overload");
    private static readonly Histogram<double> s_ttftHistogram = s_meter.CreateHistogram<double>("opentailllm.ttft_ms", "ms", "Time to first token latency");
    private static readonly Histogram<double> s_itlHistogram = s_meter.CreateHistogram<double>("opentailllm.itl_ms", "ms", "Inter-token generation duration");

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
    }

    /// <summary>Adds <paramref name="count"/> to <see cref="TotalTokens"/>.</summary>
    public void RecordTokens(long count)
    {
        Interlocked.Add(ref _totalTokens, count);
        s_tokensCounter.Add(count);
    }

    /// <summary>Starts latency accounting at engine-submission time for one generation request.</summary>
    public ServingRequestTiming BeginServingRequest() => new(this);

    internal void RecordQueueLatency(TimeSpan elapsed) => _queueLatency.Observe(elapsed);
    internal void RecordTimeToFirstToken(TimeSpan elapsed)
    {
        _timeToFirstToken.Observe(elapsed);
        s_ttftHistogram.Record(elapsed.TotalMilliseconds);
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
    private long _firstToken;
    private long _generationEnd;
    private bool _admitted;
    private bool _completed;

    internal ServingRequestTiming(ServerMetrics metrics) => _metrics = metrics;

    /// <summary>Marks the first engine event (the usage/admission chunk on supported engines).</summary>
    public void MarkAdmitted()
    {
        if (_admitted) return;
        _admitted = true;
        _metrics.RecordQueueLatency(Stopwatch.GetElapsedTime(_started));
    }

    /// <summary>Marks the first generated text or reasoning token.</summary>
    public void MarkFirstToken()
    {
        if (Interlocked.CompareExchange(ref _firstToken, Stopwatch.GetTimestamp(), 0) != 0) return;
        _metrics.RecordTimeToFirstToken(Stopwatch.GetElapsedTime(_started));
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
        long end = Stopwatch.GetTimestamp();
        _metrics.RecordRequestDuration(Stopwatch.GetElapsedTime(_started, end));
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
