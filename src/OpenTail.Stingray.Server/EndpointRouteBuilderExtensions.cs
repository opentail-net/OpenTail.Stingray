using Microsoft.AspNetCore.Routing;
using OpenTail.Stingray.Server.Endpoints;

namespace OpenTail.Stingray.Server;

/// <summary>
/// Composite map-endpoints extension for hosts that want every OpenTail.Stingray HTTP API
/// in one call. Individual <c>Map…Endpoints()</c> extensions remain available for hosts
/// that want only a subset (e.g. <c>MapOpenAiEndpoints()</c> alone behind an auth filter).
/// </summary>
public static class EndpointRouteBuilderExtensions
{
    /// <summary>
    /// Maps the OpenAI chat completions + models endpoints, the Anthropic <c>/v1/messages</c>
    /// endpoint, the OpenAI Responses endpoint, and the <c>/health</c> + <c>/metrics</c>
    /// observability endpoints onto <paramref name="endpoints"/>.
    /// </summary>
    public static IEndpointRouteBuilder MapOpenTailStingray(this IEndpointRouteBuilder endpoints)
    {
        endpoints.MapOpenAiEndpoints();
        endpoints.MapAnthropicEndpoints();
        endpoints.MapResponsesEndpoints();
        endpoints.MapHealthEndpoints();
        endpoints.MapStatusEndpoints();
        endpoints.MapCompatibilityEndpoints();
        endpoints.MapLlamaCompatEndpoints();
        return endpoints;
    }
}
