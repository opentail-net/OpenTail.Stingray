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
        IInferenceEngine engine)
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

        // Two different things, and the report must not conflate them: the named-session LIFECYCLE
        // (POST/GET/DELETE /v1/sessions) is served whenever EnableSessions is set, while restart
        // CONTINUATION — state surviving a host restart — is not implemented at all. So the flag
        // stays false in both configurations and only the detail moves.
        //
        // Saying "not exposed by the server yet" once those routes are live is a false denial: a
        // client can call them today. Flipping the flag to true would be the opposite error, and
        // the one this report exists to avoid — promising durability that does not exist. Session
        // state is in-process only; turn submission and durable restart wiring are still to come.
        var sessionRestart = options.EnableSessions
            ? new ServerFeatureAvailability(false,
                "Named-session lifecycle endpoints are served, but restart continuation is not: "
                + "session state is held in-process and does not survive a host restart.")
            : new ServerFeatureAvailability(false,
                "Disabled by server configuration; no session endpoints are served and no session "
                + "state is retained across a restart.");

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
                Metrics: true),
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
    bool Metrics);

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
