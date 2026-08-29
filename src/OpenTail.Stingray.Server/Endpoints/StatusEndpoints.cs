
namespace OpenTail.Stingray.Server.Endpoints;

/// <summary>
/// Maps the versioned runtime status document (§7.4 of the QoL plan). It is separate from
/// <c>/metrics</c> on purpose: Prometheus exposition is a monitoring protocol whose stability
/// matters to scrapers, while <c>/status</c> is a human- and CLI-facing snapshot that may gain
/// fields behind its schema version.
/// </summary>
public static class StatusEndpoints
{
    /// <summary>Maps <c>GET /status</c>.</summary>
    public static IEndpointRouteBuilder MapStatusEndpoints(this IEndpointRouteBuilder app)
    {
        app.MapGet("/status", (
                IInferenceEngine engine,
                ServerMetrics metrics,
                RequestConcurrencyGate admissionGate,
                IOptions<OpenTailStingrayServerOptions> options,
                IServiceProvider services) =>
            Results.Json(
                ServerStatusSnapshot.Create(options.Value, engine, metrics, admissionGate.Limit,
                    services.GetService<ServerEnvironmentOverrideReceipt>(),
                    services.GetService<CpuPrefillRuntimeReceiptRelay>()?.Capability,
                    services.GetService<ServerRuntimeResolutionRelay>()?.Resolution),
                OpenTailStingrayJsonContext.Default.ServerStatusSnapshot));
        return app;
    }
}
