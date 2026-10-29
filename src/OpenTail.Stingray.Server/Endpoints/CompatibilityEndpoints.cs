
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
        app.MapGet("/capabilities", (IServiceProvider services, IEnumerable<EndpointDataSource> endpointSources) =>
        {
            // Never loads or pins a model in a multi-model deployment: see ServerDiagnosticsTarget.
            var target = ServerDiagnosticsTarget.Resolve(services);
            return Results.Json(
                ServerCompatibilitySnapshot.Create(target.Options, target.Engine, target.Sessions,
                    MappedRoutePatterns(endpointSources), target.Architecture, target.ConfiguredModels),
                OpenTailStingrayJsonContext.Default.ServerCompatibilitySnapshot);
        });
        return app;
    }

    /// <summary>Route patterns registered on this host, so the snapshot reports only mapped APIs.</summary>
    private static HashSet<string> MappedRoutePatterns(IEnumerable<EndpointDataSource> sources) =>
        sources.SelectMany(source => source.Endpoints)
            .OfType<RouteEndpoint>()
            .Select(endpoint => endpoint.RoutePattern.RawText ?? string.Empty)
            .ToHashSet(StringComparer.Ordinal);
}
