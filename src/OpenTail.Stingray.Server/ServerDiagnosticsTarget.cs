namespace OpenTail.Stingray.Server;

/// <summary>
/// What the diagnostics surfaces (<c>/health</c>, <c>/metrics</c>, <c>/status</c>, <c>/capabilities</c>
/// and the observable engine gauges) describe.
/// <list type="bullet">
/// <item>Single-model (<see cref="OpenTailStingrayServerOptions.Models"/> empty): the DI
/// <see cref="IInferenceEngine"/> and the relays its load populated. Resolving it loads the model,
/// exactly as before.</item>
/// <item>Multi-model: a read-only view over <see cref="IModelRuntimeManager"/>. It never loads, acquires
/// or pins a model. <see cref="Model"/> is the default alias (the one a request without <c>model</c>
/// is routed to), <see cref="ConfiguredModels"/> lists every alias, traffic counters are summed over
/// the resident models, and per-model facts (architecture, resolved runtime, sessions, image input)
/// describe the default model: resolved values when it is resident, its configuration otherwise.</item>
/// </list>
/// </summary>
internal sealed record ServerDiagnosticsTarget(
    IInferenceEngine Engine,
    OpenTailStingrayServerOptions Options,
    string? Architecture,
    ServerRuntimeResolution? RuntimeResolution,
    CpuBatchedPrefillCapability? CpuBatchedPrefill,
    IServerSessionRuntime? Sessions,
    IReadOnlyList<string>? ConfiguredModels)
{
    public string Model => Engine.ModelId;

    public static ServerDiagnosticsTarget Resolve(IServiceProvider services)
    {
        ArgumentNullException.ThrowIfNull(services);
        var options = services.GetRequiredService<IOptions<OpenTailStingrayServerOptions>>().Value;

        if (options.Models.Count == 0)
        {
            return new ServerDiagnosticsTarget(
                services.GetRequiredService<IInferenceEngine>(),
                options,
                services.GetService<ServerLoadedArchitectureRelay>()?.Architecture,
                services.GetService<ServerRuntimeResolutionRelay>()?.Resolution,
                services.GetService<CpuPrefillRuntimeReceiptRelay>()?.Capability,
                services.GetService<IServerSessionRuntime>(),
                ConfiguredModels: null);
        }

        var inference = services.GetRequiredService<IInferenceService>();
        var manager = services.GetRequiredService<IModelRuntimeManager>();
        string defaultAlias = inference.AvailableModelAliases[0];
        var defaultId = inference.ResolveModel(null);
        var defaultOptions = ServiceCollectionExtensions.ForNamedModel(options, options.Models[0]);

        // Observation only: TryGetResident takes no handle, so this never extends residency.
        LoadedEngine? loaded = manager.TryGetResident(defaultId, out var runtime)
            && runtime is { State: not ModelRuntimeState.Disposed }
                ? runtime.Loaded
                : null;

        IServerSessionRuntime sessions = loaded is null
            ? new FixedSessionRuntime(null, null, options.EnableSessions
                ? "Multi-model deployment: session support is per model and is known once the default model has loaded."
                : "Sessions are disabled by server configuration.")
            : new FixedSessionRuntime(loaded.SessionRuntime, loaded.ColdSessionRuntime, options.EnableSessions
                ? "The default model's engine did not expose the CPU-dense session runtime."
                : "Sessions are disabled by server configuration.");

        return new ServerDiagnosticsTarget(
            new ResidentModelsDiagnosticsEngine(manager, defaultAlias, loaded?.Engine),
            defaultOptions,
            string.IsNullOrWhiteSpace(loaded?.Architecture) ? null : loaded.Architecture,
            loaded?.RuntimeResolution,
            loaded?.CpuBatchedPrefill,
            sessions,
            inference.AvailableModelAliases);
    }

    private sealed class FixedSessionRuntime(HotSessionRuntime? runtime, ColdSessionRuntime? coldRuntime, string reason)
        : IServerSessionRuntime
    {
        public HotSessionRuntime? Runtime => runtime;
        public ColdSessionRuntime? ColdRuntime => coldRuntime;
        public string? UnavailabilityReason => runtime is null ? reason : null;
    }
}

/// <summary>
/// Read-only multi-model view for diagnostics: queue and request counters summed over every resident
/// model, capabilities of the default model. It is not an inference engine; generation throws.
/// A runtime evicted between the snapshot and the read is skipped rather than reported.
/// </summary>
internal sealed class ResidentModelsDiagnosticsEngine(
    IModelRuntimeManager manager, string defaultAlias, IInferenceEngine? defaultEngine) : IInferenceEngine
{
    public string ModelId => defaultAlias;
    public int QueueDepth => (int)Sum(e => e.QueueDepth);
    public int ActiveRequests => (int)Sum(e => e.ActiveRequests);
    public long PrefillTokensReused => Sum(e => e.PrefillTokensReused);

    /// <summary>False when any resident model cannot reuse prefixes; true when none is resident.</summary>
    public bool PrefixCacheEnabled
    {
        get
        {
            foreach (var engine in ResidentEngines())
            {
                try
                {
                    if (!engine.PrefixCacheEnabled) return false;
                }
                catch (ObjectDisposedException)
                {
                    // Evicted mid-read.
                }
            }
            return true;
        }
    }

    public bool SupportsImageInput
    {
        get
        {
            try { return defaultEngine?.SupportsImageInput ?? false; }
            catch (ObjectDisposedException) { return false; }
        }
    }

    public IAsyncEnumerable<GenerateChunk> GenerateChunksAsync(
        string prompt, SamplingParams sp, CancellationToken ct = default, string? canonicalHistoryPrefix = null) =>
        throw new NotSupportedException("The multi-model diagnostics view cannot generate; acquire a model through IInferenceService.");

    private long Sum(Func<IInferenceEngine, long> read)
    {
        long total = 0;
        foreach (var engine in ResidentEngines())
        {
            try { total += read(engine); }
            catch (ObjectDisposedException) { /* evicted mid-read */ }
        }
        return total;
    }

    private List<IInferenceEngine> ResidentEngines()
    {
        var engines = new List<IInferenceEngine>();
        foreach (var stats in manager.Snapshot())
        {
            if (stats.State == ModelRuntimeState.Loading) continue;
            if (manager.TryGetResident(stats.ModelId, out var runtime)
                && runtime is { State: not ModelRuntimeState.Disposed })
                engines.Add(runtime.Engine);
        }
        return engines;
    }
}
