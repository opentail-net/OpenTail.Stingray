
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
                ServerMetrics metrics,
                RequestConcurrencyGate admissionGate,
                IServiceProvider services) =>
        {
            // Never loads or pins a model in a multi-model deployment: see ServerDiagnosticsTarget.
            var target = ServerDiagnosticsTarget.Resolve(services);
            return Results.Json(
                ServerStatusSnapshot.Create(target.Options, target.Engine, metrics, admissionGate.Limit,
                    services.GetService<ServerEnvironmentOverrideReceipt>(),
                    target.CpuBatchedPrefill,
                    target.RuntimeResolution,
                    target.Architecture,
                    target.ConfiguredModels),
                OpenTailStingrayJsonContext.Default.ServerStatusSnapshot);
        });
        return app;
    }
}
