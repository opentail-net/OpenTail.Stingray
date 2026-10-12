using System.Diagnostics.Metrics;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Server;

/// <summary>
/// Observable instruments: values the server and engine already keep, read only when a listener
/// collects (no work on the request path, no work at all with no listener). Model residency comes
/// from <see cref="IModelRuntimeManager.GetStats"/>; queue and cache state from the DI engine's
/// <see cref="IContinuousBatchingObservability"/>. In multi-model mode the engine gauges describe
/// the DI singleton engine only, the same limitation as <c>/health</c> and <c>/metrics</c>.
/// A read that throws yields no measurement; telemetry never breaks serving.
/// </summary>
internal sealed class ServerObservableMetrics : IDisposable
{
    private readonly Meter _meter = new("OpenTail.Stingray.Server", "1.0.0");
    private readonly IServiceProvider _services;
    private readonly IDisposable _eventBridge;

    public ServerObservableMetrics(IServiceProvider services)
    {
        _services = services;
        // Speculative-decoding (MTP) steps are reported on the engine event bus, not per request; turn them into the
        // opentail.engine.speculative.* counters. With no meter listener the bridge returns immediately.
        _eventBridge = OpenTail.Stingray.Core.EngineEvents.Subscribe(new EngineEventsTelemetryBridge("server"));

        // Residency (ModelRuntimeManager). Cumulative counts are observable counters over the existing totals.
        Gauge("opentail.models.resident", "{model}", "Models currently resident", m => m.ResidentModels);
        Gauge("opentail.models.pending_loads", "{load}", "Model loads in flight", m => m.PendingLoads);
        Gauge("opentail.models.estimated_bytes", "By", "Estimated bytes of resident model weights (an estimate, not measured memory)", m => m.EstimatedResidentModelBytes);
        Gauge("opentail.admission.queued", "{request}", "Requests waiting for model admission", m => m.QueuedAdmissions);
        Counter("opentail.models.loads", "{load}", "Model loads completed", m => m.ModelLoads);
        Counter("opentail.models.load_failures", "{load}", "Model loads that failed", m => m.ModelLoadFailures);
        Counter("opentail.models.evictions", "{model}", "Models evicted", m => m.ModelEvictions);
        Counter("opentail.models.admission_rejections", "{request}", "Requests rejected by model admission", m => m.AdmissionRejects);

        // Engine queue, batching and prefix cache (DI singleton engine).
        EngineGauge("opentail.engine.queue.depth", "{request}", "Requests waiting to start generation", e => e.QueueDepth);
        EngineGauge("opentail.engine.sequences.active", "{request}", "Requests currently generating", e => e.ActiveRequests);
        BatchGauge("opentail.engine.kv_cache.committed_tokens", "{token}", "KV tokens reserved by admitted requests", b => b.CommittedKvTokens);
        BatchGauge("opentail.engine.kv_cache.budget_tokens", "{token}", "KV-token admission budget; long.MaxValue means unlimited", b => b.KvTokenBudget);
        BatchGauge("opentail.engine.prefix_cache.entries", "{entry}", "Prefix snapshots retained", b => b.PrefixCacheEntries);
        BatchGauge("opentail.engine.prefix_cache.estimated_bytes", "By", "Engine-accounted bytes of retained prefix snapshots (an estimate)", b => b.PrefixCacheUsedBytes);
        BatchCounter("opentail.engine.prefix_cache.hits", "{lookup}", "Prefix-cache lookups that found a reusable prefix", b => b.PrefixCacheHits);
        BatchCounter("opentail.engine.prefix_cache.misses", "{lookup}", "Prefix-cache lookups with no reusable prefix", b => b.PrefixCacheMisses);
        BatchCounter("opentail.engine.prefix_cache.evictions", "{entry}", "Prefix snapshots evicted to honour capacity", b => b.PrefixCacheEvictions);
    }

    private void Gauge(string name, string unit, string description, Func<MultiModelRuntimeStats, long> read) =>
        _meter.CreateObservableGauge(name, () => Read(() => read(Manager().GetStats())), unit, description);

    private void Counter(string name, string unit, string description, Func<MultiModelRuntimeStats, long> read) =>
        _meter.CreateObservableCounter(name, () => Read(() => read(Manager().GetStats())), unit, description);

    private void EngineGauge(string name, string unit, string description, Func<IInferenceEngine, long> read) =>
        _meter.CreateObservableGauge(name, () => Read(() => read(_services.GetRequiredService<IInferenceEngine>())), unit, description);

    private void BatchGauge(string name, string unit, string description, Func<IContinuousBatchingObservability, long> read) =>
        _meter.CreateObservableGauge(name, () => ReadBatch(read), unit, description);

    private void BatchCounter(string name, string unit, string description, Func<IContinuousBatchingObservability, long> read) =>
        _meter.CreateObservableCounter(name, () => ReadBatch(read), unit, description);

    private IModelRuntimeManager Manager() => _services.GetRequiredService<IModelRuntimeManager>();

    private static IEnumerable<Measurement<long>> Read(Func<long> read)
    {
        long value;
        try { value = read(); }
        catch { return []; }
        return [new Measurement<long>(value)];
    }

    private IEnumerable<Measurement<long>> ReadBatch(Func<IContinuousBatchingObservability, long> read)
    {
        try
        {
            return _services.GetService<IInferenceEngine>() is IContinuousBatchingObservability { IsContinuousBatching: true } b
                ? [new Measurement<long>(read(b))]
                : [];
        }
        catch
        {
            return [];
        }
    }

    public void Dispose()
    {
        _eventBridge.Dispose();
        _meter.Dispose();
    }
}

/// <summary>Forces <see cref="ServerObservableMetrics"/> to be created when the host starts.</summary>
internal sealed class ServerObservableMetricsStarter : IHostedService
{
    // The constructor parameter is the point: resolving it creates the observable instruments.
    public ServerObservableMetricsStarter(ServerObservableMetrics observable) => Observable = observable;
    internal ServerObservableMetrics Observable { get; }
    public Task StartAsync(CancellationToken cancellationToken) => Task.CompletedTask;
    public Task StopAsync(CancellationToken cancellationToken) => Task.CompletedTask;
}
