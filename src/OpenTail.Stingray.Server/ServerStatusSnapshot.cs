using System.Text.Json.Serialization;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Server;

/// <summary>
/// Versioned, bounded runtime status document (§6.5 / §7.4 of the QoL plan). It answers
/// "what is loaded, where is it running, how busy is it, and why is it slow?" from state the
/// engine and <see cref="ServerMetrics"/> already keep.
///
/// It is assembled per request at the HTTP boundary — never in a decode, prefill, or batching
/// loop — so it adds nothing to the inner loop's register/port/instruction budget (§5.6).
/// It deliberately carries no prompt text, generated text, token IDs, credentials, or
/// filesystem paths, so it is safe to poll and safe to paste into a support thread.
/// </summary>
public sealed record ServerStatusSnapshot(
    int SchemaVersion,
    string Status,
    string Model,
    string Architecture,
    long UptimeSeconds,
    ServerStatusPlacement Placement,
    ServerStatusTraffic Traffic,
    ServerStatusLatency Latency,
    ServerStatusCache Cache,
    ServerStatusMemory Memory,
    ServerStatusBatching? Batching,
    ServerStatusConfiguration Configuration,
    string[] Warnings)
{
    /// <summary>Schema version of the emitted document. Bump on any breaking field change.</summary>
    public const int CurrentSchemaVersion = 1;

    /// <summary>Upper bound on the derived warning list, so the document stays cheap to poll.</summary>
    public const int MaxWarnings = 8;

    /// <summary>
    /// Builds the snapshot from the bound server options, the loaded engine, and the process's
    /// serving counters.
    /// </summary>
    /// <param name="admissionCapacity">
    /// The bounded active-plus-waiting generation ceiling, or 0 when admission control is off.
    /// Passed in rather than resolved here because the gate that owns it is internal.
    /// </param>
    public static ServerStatusSnapshot Create(
        OpenTailStingrayServerOptions options,
        IInferenceEngine engine,
        ServerMetrics metrics,
        int admissionCapacity,
        ServerEnvironmentOverrideReceipt? environmentOverrides = null,
        CpuBatchedPrefillCapability? cpuBatchedPrefill = null)
    {
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(engine);
        ArgumentNullException.ThrowIfNull(metrics);

        double uptimeSeconds = metrics.Uptime.TotalSeconds;
        long totalTokens = metrics.TotalTokens;
        long overloadRejections = metrics.OverloadRejections;
        int active = engine.ActiveRequests;
        int queued = engine.QueueDepth;

        var batching = engine as IContinuousBatchingObservability;
        // Same expression as ServerCompatibilitySnapshot, deliberately: two documents from the
        // same process disagreeing about whether batching is on would be worse than either being
        // slightly conservative. MaxBatchSize is the built-in loader's load-time contract, so a
        // custom engine factory without batching observability still reports the configured shape.
        bool continuousBatching = options.MaxBatchSize > 1
            || batching is { IsContinuousBatching: true };

        var traffic = new ServerStatusTraffic(
            RequestsTotal: metrics.TotalRequests,
            TokensGeneratedTotal: totalTokens,
            TokensPerSecond: uptimeSeconds > 0 ? totalTokens / uptimeSeconds : 0,
            ActiveRequests: active,
            QueuedRequests: queued,
            AdmissionCapacity: admissionCapacity,
            OverloadRejectionsTotal: overloadRejections);

        var generation = metrics.GenerationDurationSummary;
        var latency = new ServerStatusLatency(
            Queue: metrics.QueueLatencySummary,
            TimeToFirstToken: metrics.TimeToFirstTokenSummary,
            Generation: generation,
            Request: metrics.RequestDurationSummary,
            // Generation timing starts AT the first token, so its span covers tokens 2..N of each
            // completed request. Dividing by (tokens - completed requests) is therefore the mean
            // gap between consecutive tokens rather than a per-token average that double-counts
            // the first one. Null until at least one request produced a second token.
            InterTokenMeanMs: generation.Count > 0 && totalTokens > generation.Count
                ? generation.TotalSeconds * 1000.0 / (totalTokens - generation.Count)
                : null);

        long? prefixHits = batching?.PrefixCacheHits;
        long? prefixMisses = batching?.PrefixCacheMisses;
        double? hitRate = prefixHits is { } h && prefixMisses is { } m && h + m > 0
            ? h * 100.0 / (h + m)
            : null;

        var cache = new ServerStatusCache(
            PrefixCacheEnabled: engine.PrefixCacheEnabled,
            PrefillTokensReusedTotal: engine.PrefillTokensReused,
            Entries: batching?.PrefixCacheEntries,
            Hits: prefixHits,
            Misses: prefixMisses,
            Evictions: batching?.PrefixCacheEvictions,
            HitRatePercent: hitRate,
            UsedBytes: batching?.PrefixCacheUsedBytes,
            CapacityBytes: Bounded(batching?.PrefixCacheBudgetBytes));

        var gcInfo = GC.GetGCMemoryInfo();
        var memory = new ServerStatusMemory(
            ManagedHeapBytes: GC.GetTotalMemory(forceFullCollection: false),
            GcCommittedBytes: gcInfo.TotalCommittedBytes,
            WorkingSetBytes: Environment.WorkingSet,
            HostAvailableBytes: gcInfo.TotalAvailableMemoryBytes,
            PrefixCacheBytes: batching?.PrefixCacheUsedBytes,
            // Weights, KV pages and expert slots live in native/device allocations the server
            // boundary cannot see. Reporting a managed-heap number as if it were model memory
            // would be worse than reporting nothing, so the device breakdown stays absent until
            // the engine exposes a placement-memory surface.
            DeviceVramBytes: null,
            DeviceVramFreeBytes: null);

        long? kvBudget = Bounded(batching?.KvTokenBudget);
        // The section exists only when a live engine is actually supplying these counters. An
        // all-zero section on an engine that reports nothing would read as "batching is on and
        // idle", which is a different claim from "this engine exposes no batching counters".
        var batchingStatus = batching is { IsContinuousBatching: true }
            ? new ServerStatusBatching(
                PrefillChunkTokens: batching.PrefillChunkTokens,
                CommittedKvTokens: batching.CommittedKvTokens,
                KvTokenBudget: kvBudget,
                KvOccupancyPercent: kvBudget is > 0
                    ? batching.CommittedKvTokens * 100.0 / kvBudget.Value
                    : null)
            : null;

        bool saturated = admissionCapacity > 0 && active + queued >= admissionCapacity;

        return new ServerStatusSnapshot(
            SchemaVersion: CurrentSchemaVersion,
            Status: saturated ? "saturated" : "ok",
            Model: engine.ModelId,
            Architecture: options.Architecture,
            UptimeSeconds: (long)uptimeSeconds,
            Placement: new ServerStatusPlacement(
                Backend: options.Backend.ToString().ToLowerInvariant(),
                GpuLayers: options.NGpuLayers,
                ContextSize: options.ContextSize,
                MaxBatchSize: options.MaxBatchSize,
                ContinuousBatching: continuousBatching,
                SpecType: options.SpecType.ToString().ToLowerInvariant()),
            Traffic: traffic,
            Latency: latency,
            Cache: cache,
            Memory: memory,
            Batching: batchingStatus,
            Configuration: new ServerStatusConfiguration([.. (environmentOverrides?.Names ?? [])],
                SimdKernels.Q8PrefillEnabled,
                cpuBatchedPrefill is null ? null : new ServerStatusCpuBatchedPrefill(
                    cpuBatchedPrefill.Available, cpuBatchedPrefill.Detail),
                new ServerStatusBoundConfiguration(
                    MaxQueuedRequests: options.MaxQueuedRequests,
                    MaxConcurrentRequests: options.MaxConcurrentRequests,
                    CpuThreads: options.CpuThreads,
                    PrefillChunkTokens: options.PrefillChunkTokens,
                    PrefillDequantCacheMb: options.PrefillDequantCacheMb,
                    KvBudgetMb: options.KvBudgetMb,
                    PrefixCacheMb: options.PrefixCacheMb,
                    TurboQuant: options.TurboQuant,
                    KvType: options.KvType,
                    ToolGrammarRequested: options.ToolGrammar,
                    SessionsEnabled: options.EnableSessions,
                    SessionPersistenceConfigured: !string.IsNullOrWhiteSpace(options.SessionStorageDirectory))),
            Warnings: BuildWarnings(engine, traffic, cache, batchingStatus, saturated));
    }

    /// <summary>
    /// The engine reports "no budget configured" as <see cref="long.MaxValue"/>. A percentage of
    /// an unlimited budget is meaningless and the sentinel reads as a real number over the wire,
    /// so unlimited is published as <c>null</c> instead.
    /// </summary>
    private static long? Bounded(long? value) =>
        value is null or long.MaxValue ? null : value;

    private static string[] BuildWarnings(
        IInferenceEngine engine,
        ServerStatusTraffic traffic,
        ServerStatusCache cache,
        ServerStatusBatching? batching,
        bool saturated)
    {
        var warnings = new List<string>(MaxWarnings);

        if (saturated)
            warnings.Add(
                $"Admission capacity reached: {traffic.ActiveRequests} active plus " +
                $"{traffic.QueuedRequests} queued against a ceiling of {traffic.AdmissionCapacity}. " +
                "New requests will be rejected until one completes.");

        if (traffic.OverloadRejectionsTotal > 0)
            warnings.Add(
                $"{traffic.OverloadRejectionsTotal} request(s) were rejected because the bounded " +
                "inference queue was full. Raise STINGRAY_MAX_QUEUED_REQUESTS or reduce client concurrency.");

        if (!engine.PrefixCacheEnabled)
            warnings.Add(
                "Prefix-cache reuse is unavailable for this forward-pass family, so every request " +
                "re-prefills its whole prompt. Multi-turn latency will scale with conversation length.");

        if (batching is { KvOccupancyPercent: >= 90 })
            warnings.Add(
                $"KV admission budget is {batching.KvOccupancyPercent:F0}% committed. " +
                "Further requests will wait for capacity rather than start prefill.");

        if (cache is { UsedBytes: { } used, CapacityBytes: { } capacity } && capacity > 0
            && used >= capacity * 0.95)
            warnings.Add(
                "The prefix cache is at capacity; snapshots are being evicted and reuse will degrade. " +
                "Raise STINGRAY_PREFIX_CACHE_MB to retain more.");

        if (warnings.Count > MaxWarnings)
            warnings.RemoveRange(MaxWarnings, warnings.Count - MaxWarnings);
        return [.. warnings];
    }
}

/// <summary>Where the loaded model is running and the load-time shape of the serving path.</summary>
public sealed record ServerStatusPlacement(
    string Backend,
    int GpuLayers,
    int ContextSize,
    int MaxBatchSize,
    bool ContinuousBatching,
    string SpecType);

/// <summary>Lifetime request/token counters plus the instantaneous queue picture.</summary>
public sealed record ServerStatusTraffic(
    long RequestsTotal,
    long TokensGeneratedTotal,
    double TokensPerSecond,
    int ActiveRequests,
    int QueuedRequests,
    int AdmissionCapacity,
    long OverloadRejectionsTotal);

/// <summary>Non-sensitive configuration provenance. Values and paths are deliberately omitted.</summary>
public sealed record ServerStatusConfiguration(
    IReadOnlyList<string> EnvironmentOverrides,
    bool CpuQ8PrefillEnabled,
    ServerStatusCpuBatchedPrefill? CpuBatchedPrefill,
    ServerStatusBoundConfiguration Bound);

/// <summary>Load-time availability of the regular CPU batched-prefill trunk for the loaded model.</summary>
public sealed record ServerStatusCpuBatchedPrefill(bool Available, string Detail);

/// <summary>
/// Non-sensitive server options after host binding and legacy environment override resolution.
/// Paths, sampling defaults, credentials, and delegate hooks are intentionally excluded.
/// A value such as <c>CpuThreads = 0</c> retains its configured meaning (automatic), rather than
/// pretending the status boundary can report a later kernel-level choice as a bound option.
/// </summary>
public sealed record ServerStatusBoundConfiguration(
    int MaxQueuedRequests,
    int? MaxConcurrentRequests,
    int CpuThreads,
    int PrefillChunkTokens,
    long? PrefillDequantCacheMb,
    long KvBudgetMb,
    long PrefixCacheMb,
    bool TurboQuant,
    string? KvType,
    bool ToolGrammarRequested,
    bool SessionsEnabled,
    bool SessionPersistenceConfigured);

/// <summary>Serving-latency summaries rendered from the same bounded histograms as <c>/metrics</c>.</summary>
public sealed record ServerStatusLatency(
    ServerLatencySummary Queue,
    ServerLatencySummary TimeToFirstToken,
    ServerLatencySummary Generation,
    ServerLatencySummary Request,
    double? InterTokenMeanMs);

/// <summary>
/// A bounded-histogram summary. Percentiles are bucket upper bounds, so they are conservative
/// estimates rather than exact order statistics; <c>null</c> means the quantile falls above the
/// largest bucket (or no samples were recorded).
/// </summary>
public sealed record ServerLatencySummary(
    [property: JsonPropertyName("count")] long Count,
    [property: JsonPropertyName("total_seconds")] double TotalSeconds,
    [property: JsonPropertyName("mean_ms")] double MeanMs,
    [property: JsonPropertyName("p50_ms")] double? P50Ms,
    [property: JsonPropertyName("p95_ms")] double? P95Ms,
    [property: JsonPropertyName("p99_ms")] double? P99Ms);

/// <summary>
/// Continuation/prefix-cache reuse. <see cref="PrefixCacheEnabled"/> is a capability;
/// <see cref="PrefillTokensReusedTotal"/> is the evidence that it is actually saving work.
/// The remaining fields are null on engines without continuous-batching observability.
/// </summary>
public sealed record ServerStatusCache(
    bool PrefixCacheEnabled,
    long PrefillTokensReusedTotal,
    int? Entries,
    long? Hits,
    long? Misses,
    long? Evictions,
    double? HitRatePercent,
    long? UsedBytes,
    long? CapacityBytes);

/// <summary>
/// Host memory as the process can observe it. Device memory is intentionally absent rather
/// than estimated — see the comment at its assignment site.
/// </summary>
public sealed record ServerStatusMemory(
    long ManagedHeapBytes,
    long GcCommittedBytes,
    long WorkingSetBytes,
    long HostAvailableBytes,
    long? PrefixCacheBytes,
    long? DeviceVramBytes,
    long? DeviceVramFreeBytes);

/// <summary>Continuous-batching admission state. Null when the engine is the single-user path.</summary>
public sealed record ServerStatusBatching(
    int PrefillChunkTokens,
    long CommittedKvTokens,
    long? KvTokenBudget,
    double? KvOccupancyPercent);
