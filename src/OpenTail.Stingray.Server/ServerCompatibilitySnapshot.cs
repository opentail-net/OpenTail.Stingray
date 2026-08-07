using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Server;

/// <summary>
/// Read-only, versioned description of the server's protocol surface and the small subset of
/// resolved configuration that changes protocol behaviour. It is deliberately assembled at a
/// diagnostics boundary, never in an inference loop.
/// </summary>
public sealed record ServerCompatibilitySnapshot(
    int SchemaVersion,
    string Model,
    string Architecture,
    ServerApiSurface Api,
    ServerRuntimeCapabilities Runtime,
    ServerEffectiveCompatibilityConfiguration Configuration)
{
    /// <summary>Creates the snapshot from the fully bound server options and loaded engine.</summary>
    public static ServerCompatibilitySnapshot Create(
        OpenTailStingrayServerOptions options,
        IInferenceEngine engine,
        IServerSessionRuntime? sessions = null)
    {
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(engine);

        // MaxBatchSize is the load-time selection contract for the built-in loader. Include it
        // as well as the live engine interface so a test/downstream EngineFactory can still
        // expose the configuration that makes tool grammar ineligible.
        bool continuousBatching = options.MaxBatchSize > 1
            || engine is IContinuousBatchingObservability { IsContinuousBatching: true };
        var toolGrammar = options.ToolGrammar
            ? continuousBatching
                ? new ServerFeatureAvailability(false,
                    "Tool-argument grammar is not available with continuous batching.")
                : new ServerFeatureAvailability(true,
                    "Requires a supported model family and a request containing tools.")
            : new ServerFeatureAvailability(false, "Disabled by server configuration.");

        // Sessions have three intentionally distinct states: disabled, hot-only, and persisted
        // CPU-dense GGUF. Do not infer persistence from EnableSessions alone; storage must have
        // been constructed for the actually loaded engine.
        var sessionRestart = sessions?.ColdRuntime is not null
            ? new ServerFeatureAvailability(true,
                "CPU-dense GGUF session state is persisted after completed turns and restored on demand. "
                + "The in-memory operation-result ledger is not durable.")
            : sessions?.Runtime is not null
                ? new ServerFeatureAvailability(false,
                    "Named sessions are active only in memory. Configure SessionStorageDirectory "
                    + "to enable the proven CPU-dense GGUF restart-continuation lane.")
                : new ServerFeatureAvailability(false,
                    (sessions?.UnavailabilityReason ?? "Sessions are disabled by server configuration.")
                    + " Restart continuation is unavailable.");

        return new ServerCompatibilitySnapshot(
            SchemaVersion: 1,
            Model: engine.ModelId,
            Architecture: options.Architecture,
            Api: new ServerApiSurface(
                OpenAiChatCompletions: true,
                OpenAiResponses: true,
                AnthropicMessages: true,
                OpenAiModels: true,
                Health: true,
                Metrics: true,
                SessionLifecycle: sessions?.Runtime is not null),
            Runtime: new ServerRuntimeCapabilities(
                ContinuousBatching: continuousBatching,
                ImageInput: engine.SupportsImageInput,
                ToolGrammar: toolGrammar,
                OutputConstraintConfigured: options.OutputConstraintFactory is not null,
                SessionRestartContinuation: sessionRestart),
            Configuration: new ServerEffectiveCompatibilityConfiguration(
                Backend: options.Backend.ToString().ToLowerInvariant(),
                GpuLayers: options.NGpuLayers,
                ContextSize: options.ContextSize,
                MaxBatchSize: options.MaxBatchSize,
                DisableThinking: options.DisableThinking,
                PreserveThinking: options.PreserveThinking,
                ToolGrammarRequested: options.ToolGrammar,
                SpecType: options.SpecType.ToString().ToLowerInvariant()));
    }
}

/// <summary>HTTP APIs supplied by <see cref="EndpointRouteBuilderExtensions.MapOpenTailStingray"/>.</summary>
public sealed record ServerApiSurface(
    bool OpenAiChatCompletions,
    bool OpenAiResponses,
    bool AnthropicMessages,
    bool OpenAiModels,
    bool Health,
    bool Metrics,
    bool SessionLifecycle);

/// <summary>Runtime capabilities relevant to client compatibility and request eligibility.</summary>
public sealed record ServerRuntimeCapabilities(
    bool ContinuousBatching,
    bool ImageInput,
    ServerFeatureAvailability ToolGrammar,
    bool OutputConstraintConfigured,
    ServerFeatureAvailability SessionRestartContinuation);

/// <summary>Whether a capability is currently usable, with a user-facing reason when it is conditional or unavailable.</summary>
public sealed record ServerFeatureAvailability(bool Available, string? Detail);

/// <summary>
/// Effective, non-secret configuration values that can change protocol or compatibility behaviour.
/// This intentionally excludes paths, credentials, delegates, and all inference-loop state.
/// </summary>
public sealed record ServerEffectiveCompatibilityConfiguration(
    string Backend,
    int GpuLayers,
    int ContextSize,
    int MaxBatchSize,
    bool DisableThinking,
    bool PreserveThinking,
    bool ToolGrammarRequested,
    string SpecType);
