using Microsoft.AspNetCore.Http.Json;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Server;

/// <summary>
/// DI registration entry points for the OpenTail.Stingray HTTP API.
/// </summary>
public static class ServiceCollectionExtensions
{
    /// <summary>Default configuration section name bound to <see cref="OpenTailStingrayServerOptions"/>.</summary>
    public const string DefaultConfigurationSection = "OpenTail.Stingray";

    /// <summary>
    /// Registers the OpenTail.Stingray engine, chat-template renderer, metrics counters, and
    /// JSON source-gen context. The engine itself is constructed lazily on first request,
    /// so the call returns immediately even when <see cref="OpenTailStingrayServerOptions.ModelPath"/>
    /// points at a multi-gigabyte GGUF file.
    /// </summary>
    /// <param name="services">Service collection to mutate.</param>
    /// <param name="configure">
    /// Optional inline configuration. Runs after any prior <c>Configure&lt;OpenTailStingrayServerOptions&gt;</c>
    /// call (e.g. binding from <see cref="IConfiguration"/>) so callers can override individual fields.
    /// </param>
    public static IServiceCollection AddOpenTailStingray(
        this IServiceCollection services,
        Action<OpenTailStingrayServerOptions>? configure = null)
    {
        services.AddOptions<OpenTailStingrayServerOptions>();
        if (configure is not null)
            services.Configure(configure);

        // TryAdd: a test or downstream module may already have registered a fake/replacement
        // for any of these services. We never overwrite an existing registration here.
        services.TryAddSingleton<ServerMetrics>();
        services.TryAddSingleton<SessionRuntimeRelay>(sp => new SessionRuntimeRelay(
            sp.GetRequiredService<IOptions<OpenTailStingrayServerOptions>>().Value.EnableSessions
                ? "The engine has not been loaded yet, or it does not expose the CPU-dense session runtime."
                : "Sessions are disabled by server configuration."));
        services.TryAddSingleton<IServerSessionRuntime>(sp => sp.GetRequiredService<SessionRuntimeRelay>());
        // Multi-model session routing (docs/032 Phase 7 follow-up) — unused, harmless to
        // construct, in single-model deployments.
        services.TryAddSingleton<SessionModelRegistry>();
        services.TryAddSingleton<CpuPrefillRuntimeReceiptRelay>();
        services.TryAddSingleton<ServerRuntimeResolutionRelay>();

        // Request admission gate. Resolved lazily so the options object is fully bound
        // (config + the host's inline Configure) by first construction. A positive
        // MaxQueuedRequests creates a bounded active+waiting policy around the engine's FIFO
        // admission path; an explicitly configured legacy MaxConcurrentRequests takes precedence.
        services.TryAddSingleton(sp => new Endpoints.RequestConcurrencyGate(
            sp.GetRequiredService<IOptions<OpenTailStingrayServerOptions>>().Value));
        services.TryAddSingleton<ChatTemplateRenderer>(sp =>
        {
            var opts = sp.GetRequiredService<IOptions<OpenTailStingrayServerOptions>>().Value;
            return new ChatTemplateRenderer(opts.Architecture);
        });

        // Multi-model runtime manager (docs/032-multi-model-inference-runtime-plan.md). Registered
        // even though today's DI wiring below only ever acquires the one configured model — this
        // is Phase 1's abstraction seam, not a behavior change. Lazily constructed like everything
        // else here; constructing it does not itself invoke the loader.
        services.TryAddSingleton<IModelRuntimeManager>(sp =>
        {
            var opts = sp.GetRequiredService<IOptions<OpenTailStingrayServerOptions>>().Value;

            // Models empty (the default): today's exact single-model loader, unchanged — the id
            // argument is discarded because there's only ever one possible model. Models non-empty
            // (docs/032 Phase 7): dispatch each cold load to whichever NamedModelOptions entry's
            // canonicalized ModelPath matches the requested id.
            Func<ModelId, LoadedEngine> loader = opts.Models.Count == 0
                ? (_ => (opts.EngineFactory ?? (s => InferenceEngineLoader.Load(opts)))(sp))
                : (id => LoadNamedModel(opts, id, sp));

            var manager = new ModelRuntimeManager(
                loader, opts.ModelResidencyMode ?? ResolveResidencyModeFromEnvironment());

            // Off unless explicitly opted into (see OpenTailStingrayServerOptions.EnableResourceAdmission
            // doc for why this isn't a new default) — leaving it null preserves today's exact behavior.
            if (opts.EnableResourceAdmission ?? ResolveEnableResourceAdmissionFromEnvironment())
                manager.ResourceBudget = new HostResourceBudget(manager);

            return manager;
        });

        // docs/032 Phase 7 — model resolution + discovery for the multi-model chat-style
        // endpoints. Cheap to construct (no loading); safe to register unconditionally even in
        // single-model deployments, where it's a thin, side-effect-free wrapper.
        services.TryAddSingleton<IInferenceService, InferenceService>();

        services.TryAddSingleton<IInferenceEngine>(sp =>
        {
            var opts = sp.GetRequiredService<IOptions<OpenTailStingrayServerOptions>>().Value;
            var manager = sp.GetRequiredService<IModelRuntimeManager>();
            var modelId = ModelId.Canonicalize(opts.ModelPath ?? "engine-factory-model");

            // Acquire through the runtime manager (single-flight, state-tracked load) but release
            // our own lease immediately — DI's own IInferenceEngine singleton registration below
            // remains the sole owner of this engine's disposal lifecycle, so IsPinned keeps the
            // manager's shutdown cleanup (ModelRuntimeManager.Dispose) from ever touching it and
            // racing that ownership.
            using var handle = manager.AcquireAsync(modelId, CancellationToken.None).GetAwaiter().GetResult();
            handle.Runtime.IsPinned = true;
            var loaded = handle.Runtime.Loaded;

            // Hand the single-user engine the host's logger so its per-request perf trace
            // (Debug level) flows through the configured logging pipeline rather than stderr.
            if (loaded.Engine is InferenceEngine ie)
                ie.Logger = sp.GetService<ILoggerFactory>()?.CreateLogger("OpenTail.Stingray.Engine");

            // Reconfigure the renderer with the model's actual arch + Jinja template now
            // that we have them. Done here rather than as a separate DI registration so
            // resolving ChatTemplateRenderer doesn't transitively trigger model loading
            // — important for tests that override IInferenceEngine but expect the
            // renderer to use the safe fallback path.
            sp.GetRequiredService<ChatTemplateRenderer>().Configure(
                loaded.Architecture, loaded.ChatTemplate, loaded.ToolBoundaryStopTokenIds, loaded.Grammar);

            // Stash the tokenizer so /tokenize and /detokenize endpoints can reach it
            // without re-opening the model. Uses a relay so the singleton registration
            // below can be resolved before IInferenceEngine is fully constructed.
            if (loaded.Tokenizer is not null)
                sp.GetRequiredService<TokenizerRelay>().Set(loaded.Tokenizer);

            sp.GetRequiredService<SessionRuntimeRelay>().Set(
                loaded.SessionRuntime,
                loaded.ColdSessionRuntime,
                opts.EnableSessions
                    ? "The loaded engine did not expose the CPU-dense session runtime."
                    : "Sessions are disabled by server configuration.");
            sp.GetRequiredService<CpuPrefillRuntimeReceiptRelay>().Set(loaded.CpuBatchedPrefill);
            sp.GetRequiredService<ServerRuntimeResolutionRelay>().Set(loaded.RuntimeResolution);

            return loaded.Engine;
        });

        // TokenizerRelay bridges the lazy engine load (which produces a tokenizer) to the
        // endpoints (which need to resolve ITokenizer from DI). The relay is populated when
        // IInferenceEngine is first resolved.
        services.TryAddSingleton<TokenizerRelay>();
        services.TryAddSingleton<OpenTail.Stingray.Core.ITokenizer>(sp =>
            sp.GetRequiredService<TokenizerRelay>().Tokenizer
            ?? throw new InvalidOperationException(
                "ITokenizer is not available yet — IInferenceEngine must be resolved first."));

        // Wire the source-gen JSON context into ASP.NET Core's JSON pipeline so
        // POST bodies and SSE deltas are AOT-compatible.
        services.Configure<JsonOptions>(o =>
            o.SerializerOptions.TypeInfoResolverChain.Insert(0, OpenTailStingrayJsonContext.Default));

        return services;
    }

    /// <summary>
    /// Convenience overload that binds <see cref="OpenTailStingrayServerOptions"/> from the supplied
    /// <see cref="IConfiguration"/> section before applying <paramref name="configure"/>.
    /// </summary>
    /// <param name="services">Service collection to mutate.</param>
    /// <param name="configuration">
    /// Configuration root (or sub-section) holding a <see cref="DefaultConfigurationSection"/>
    /// child. Pass <c>builder.Configuration</c> for the typical case.
    /// </param>
    /// <param name="configure">Optional inline tweaks applied after the configuration bind.</param>
    public static IServiceCollection AddOpenTailStingray(
        this IServiceCollection services,
        IConfiguration configuration,
        Action<OpenTailStingrayServerOptions>? configure = null)
    {
        services.Configure<OpenTailStingrayServerOptions>(configuration.GetSection(DefaultConfigurationSection));
        return services.AddOpenTailStingray(configure);
    }

    /// <summary>
    /// Loads one entry of <see cref="OpenTailStingrayServerOptions.Models"/> — the
    /// <see cref="IModelRuntimeManager"/> loader used in multi-model mode (docs/032 Phase 7).
    /// Derives a per-model <see cref="OpenTailStingrayServerOptions"/> via
    /// <see cref="OpenTailStingrayServerOptions.Clone"/> so every knob besides the ones
    /// <see cref="NamedModelOptions"/> exposes (TurboQuant, KV type, MoE tuning, speculative
    /// decoding, sampling defaults, ...) stays shared/global across all configured models rather
    /// than needing to be repeated per entry — deliberately narrower than a fully independent
    /// per-model options surface, matching the scope <see cref="NamedModelOptions"/>'s own doc
    /// comment states.
    /// </summary>
    private static LoadedEngine LoadNamedModel(OpenTailStingrayServerOptions opts, ModelId id, IServiceProvider sp)
    {
        NamedModelOptions? named = null;
        foreach (var candidate in opts.Models)
        {
            if (ModelId.Canonicalize(candidate.ModelPath).Equals(id)) { named = candidate; break; }
        }
        if (named is null)
            throw new InvalidOperationException($"No configured model in OpenTailStingrayServerOptions.Models matches '{id}'.");

        var perModelOpts = opts.Clone();
        perModelOpts.ModelPath = named.ModelPath;
        perModelOpts.MmprojPath = named.MmprojPath;
        perModelOpts.Architecture = named.Architecture ?? opts.Architecture;
        perModelOpts.Backend = named.Backend ?? opts.Backend;
        perModelOpts.NGpuLayers = named.NGpuLayers ?? opts.NGpuLayers;
        perModelOpts.ContextSize = named.ContextSize ?? opts.ContextSize;

        // named.EngineFactory (per-model, knows which entry it's building) takes precedence over
        // the top-level EngineFactory (which has no model identity to dispatch on — see its doc).
        return (named.EngineFactory ?? opts.EngineFactory ?? (s => InferenceEngineLoader.Load(perModelOpts)))(sp);
    }

    /// <summary>
    /// Resolves <see cref="OpenTailStingrayServerOptions.ModelResidencyMode"/>'s environment-variable
    /// fallback, <c>STINGRAY_MODEL_RESIDENCY_MODE</c> (<c>single</c>/<c>singleslot</c> or
    /// <c>multi</c>/<c>multislot</c>, case-insensitive). Mirrors the existing
    /// options-property-first-then-env-var convention used by <c>InferenceEngineLoader</c> (e.g.
    /// <c>STINGRAY_DSPARK_MODEL</c>) rather than routing through ASP.NET configuration binding,
    /// so this works the same whether or not the host also binds <c>IConfiguration</c>.
    /// </summary>
    private static ModelResidencyMode ResolveResidencyModeFromEnvironment()
    {
        var raw = Environment.GetEnvironmentVariable("STINGRAY_MODEL_RESIDENCY_MODE");
        if (string.IsNullOrWhiteSpace(raw)) return ModelResidencyMode.MultiSlot;
        return raw.Trim().ToLowerInvariant() switch
        {
            "single" or "singleslot" or "single-slot" => ModelResidencyMode.SingleSlot,
            "multi" or "multislot" or "multi-slot" => ModelResidencyMode.MultiSlot,
            _ => throw new InvalidOperationException(
                $"Unknown STINGRAY_MODEL_RESIDENCY_MODE '{raw}'. Expected 'single' or 'multi'."),
        };
    }

    /// <summary>
    /// Resolves <see cref="OpenTailStingrayServerOptions.EnableResourceAdmission"/>'s
    /// environment-variable fallback, <c>STINGRAY_ENABLE_RESOURCE_ADMISSION</c> (<c>1</c>/<c>true</c>,
    /// case-insensitive). Unset/anything else → off, matching the option's own documented default.
    /// </summary>
    private static bool ResolveEnableResourceAdmissionFromEnvironment()
    {
        var raw = Environment.GetEnvironmentVariable("STINGRAY_ENABLE_RESOURCE_ADMISSION");
        return string.Equals(raw, "1", StringComparison.Ordinal)
            || string.Equals(raw, "true", StringComparison.OrdinalIgnoreCase);
    }
}
