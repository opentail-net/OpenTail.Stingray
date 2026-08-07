using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Sessions;

namespace OpenTail.Stingray.Server.Endpoints;

/// <summary>
/// Minimal named-session lifecycle endpoints. Hosts that expose these routes should attach their
/// authentication/tenant policy to the returned route group or individual endpoint builders.
/// The first server lane is deliberately hot-only and CPU-dense; this API does not claim that a
/// session survives a process restart.
/// </summary>
public static class SessionEndpoints
{
    /// <summary>Maps create, inspect, and delete operations under <c>/v1/sessions</c>.</summary>
    public static IEndpointRouteBuilder MapSessionEndpoints(this IEndpointRouteBuilder app)
    {
        app.MapPost("/v1/sessions", Create);
        app.MapGet("/v1/sessions/{sessionId:guid}", Get);
        app.MapDelete("/v1/sessions/{sessionId:guid}", Delete);
        return app;
    }

    private static IResult Create(IInferenceEngine _, IServerSessionRuntime sessions)
    {
        if (sessions.Runtime is not { } runtime)
            return Unavailable(sessions);

        var session = runtime.Create();
        return Results.Json(ToResponse(runtime.GetSessionSnapshot(session.SessionId)),
            OpenTailStingrayJsonContext.Default.SessionResponse, statusCode: StatusCodes.Status201Created);
    }

    private static IResult Get(Guid sessionId, IInferenceEngine _, IServerSessionRuntime sessions)
    {
        if (sessions.Runtime is not { } runtime)
            return Unavailable(sessions);

        try
        {
            return Results.Json(ToResponse(runtime.GetSessionSnapshot(new SessionId(sessionId))),
                OpenTailStingrayJsonContext.Default.SessionResponse);
        }
        catch (SessionNotFoundException)
        {
            return Results.NotFound();
        }
    }

    private static IResult Delete(Guid sessionId, IInferenceEngine _, IServerSessionRuntime sessions)
    {
        if (sessions.Runtime is not { } runtime)
            return Unavailable(sessions);

        return runtime.Delete(new SessionId(sessionId)) ? Results.NoContent() : Results.NotFound();
    }

    private static IResult Unavailable(IServerSessionRuntime sessions) =>
        Results.Json(new ErrorResponse("session_unavailable",
                sessions.UnavailabilityReason ?? "Sessions are not available for the loaded engine."),
            OpenTailStingrayJsonContext.Default.ErrorResponse,
            statusCode: StatusCodes.Status409Conflict);

    private static SessionResponse ToResponse(SessionSnapshot snapshot) => new(
        snapshot.SessionId.ToString(), snapshot.CommittedRevision.Value,
        snapshot.DurableRevision?.Value, snapshot.CurrentFencingEpoch,
        snapshot.Operations.Count);
}

/// <summary>Bounded session metadata returned by the lifecycle endpoints.</summary>
public sealed record SessionResponse(
    string Id,
    long CommittedRevision,
    long? DurableRevision,
    long FencingEpoch,
    int OperationCount);
