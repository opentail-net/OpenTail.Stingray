
namespace OpenTail.Stingray.Server.Endpoints;

/// <summary>
/// Read-only diagnostics for a host that maps OpenTail's complete protocol surface through
/// <see cref="EndpointRouteBuilderExtensions.MapOpenTailStingray"/>.
/// </summary>
public static class CompatibilityEndpoints
{
    /// <summary>
    /// Maps <c>GET /capabilities</c>. It is intentionally separate from health/metrics: a host
    /// mapping only a subset of OpenTail's APIs must not return a snapshot claiming routes it
    /// did not map.
    /// </summary>
    public static IEndpointRouteBuilder MapCompatibilityEndpoints(this IEndpointRouteBuilder app)
    {
        app.MapGet("/capabilities", (IInferenceEngine engine, IServerSessionRuntime sessions,
                IOptions<OpenTailStingrayServerOptions> options) =>
            Results.Json(
                ServerCompatibilitySnapshot.Create(options.Value, engine, sessions),
                OpenTailStingrayJsonContext.Default.ServerCompatibilitySnapshot));
        return app;
    }
}
