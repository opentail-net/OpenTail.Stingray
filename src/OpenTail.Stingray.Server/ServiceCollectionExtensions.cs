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

        services.TryAddSingleton<IInferenceEngine>(sp =>
        {
            var opts = sp.GetRequiredService<IOptions<OpenTailStingrayServerOptions>>().Value;
            var loaded = (opts.EngineFactory ?? (s => InferenceEngineLoader.Load(opts)))(sp);

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
}
