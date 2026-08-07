using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;
using Microsoft.Extensions.Options;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Sessions;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json.Serialization;

namespace OpenTail.Stingray.Server.Endpoints;

/// <summary>
/// Minimal named-session lifecycle endpoints. Hosts that expose these routes should attach their
/// authentication/tenant policy to the returned route group or individual endpoint builders.
/// The first server lane is deliberately hot-only and CPU-dense; this API does not claim that a
/// session survives a process restart.
///
/// <para>Every handler takes an unused <see cref="IInferenceEngine"/> parameter. That is not an
/// oversight: resolving it makes these routes share the engine's construction and failure
/// behaviour with the chat endpoints, so a host with no usable engine fails the same way here as
/// everywhere else instead of serving session routes over a runtime that was never brought up.
/// Discard the value with <c>_</c> rather than deleting the parameter.</para>
/// </summary>
public static class SessionEndpoints
{
    /// <summary>Maps create, inspect, and delete operations under <c>/v1/sessions</c>.</summary>
    public static IEndpointRouteBuilder MapSessionEndpoints(this IEndpointRouteBuilder app)
    {
        app.MapPost("/v1/sessions", Create);
        app.MapGet("/v1/sessions/{sessionId:guid}", Get);
        app.MapPost("/v1/sessions/{sessionId:guid}/turns", RunTurn);
        app.MapDelete("/v1/sessions/{sessionId:guid}", Delete);
        return app;
    }

    private static IResult Create(IInferenceEngine _, IServerSessionRuntime sessions)
    {
        if (sessions.Runtime is not { } runtime)
            return Unavailable(sessions);

        var session = sessions.ColdRuntime?.Create() ?? runtime.Create();
        return Results.Json(ToResponse(runtime.GetSessionSnapshot(session.SessionId), session.CommittedRevision),
            OpenTailStingrayJsonContext.Default.SessionResponse, statusCode: StatusCodes.Status201Created);
    }

    private static IResult Get(Guid sessionId, IInferenceEngine engine, IServerSessionRuntime sessions)
    {
        if (sessions.Runtime is not { } runtime)
            return Unavailable(sessions);

        try
        {
            var id = new SessionId(sessionId);
            var session = sessions.ColdRuntime?.Open(id) ?? runtime.Open(id);
            return Results.Json(ToResponse(runtime.GetSessionSnapshot(id), session.CommittedRevision),
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

        var id = new SessionId(sessionId);
        bool deleted = sessions.ColdRuntime?.Delete(id) ?? runtime.Delete(id);
        return deleted ? Results.NoContent() : Results.NotFound();
    }

    private static async Task<IResult> RunTurn(
        Guid sessionId,
        SessionTurnRequest request,
        IInferenceEngine engine,
        IServerSessionRuntime sessions,
        IOptions<OpenTailStingrayServerOptions> options,
        CancellationToken cancellationToken)
    {
        if (sessions.Runtime is not { } runtime)
            return Unavailable(sessions);
        if (string.IsNullOrEmpty(request.AppendPrompt))
            return Results.BadRequest(new ErrorResponse("invalid_request_error", "append_prompt must not be empty."));
        if (request.ExpectedRevision is not { } expectedRevision || expectedRevision < 0)
            return Results.BadRequest(new ErrorResponse("invalid_request_error", "expected_revision must be a non-negative integer."));
        if (request.MaxTokens is <= 0)
            return Results.BadRequest(new ErrorResponse("invalid_request_error", "max_tokens must be positive."));

        var sampling = SamplingParamsBuilder.Build(options.Value,
            temperature: request.Temperature,
            topP: request.TopP,
            maxTokens: request.MaxTokens,
            maxThinking: null,
            presencePenalty: null,
            frequencyPenalty: null,
            logitBias: null,
            thinkingDisabled: options.Value.DisableThinking);
        // Hash what will actually be EXECUTED, not what was requested: sampling defaults come from
        // server options, so digesting the raw request values would give two turns the same digest
        // while they ran with different parameters. Length-prefix the prompt so it cannot be
        // confused with the fields that follow it.
        byte[] canonical = SHA256.HashData(Encoding.UTF8.GetBytes(
            $"{request.AppendPrompt.Length}:{request.AppendPrompt}"
            + $"|{expectedRevision}|{sampling.MaxNewTokens}|{sampling.Temperature}|{sampling.TopP}"));
        var digest = SessionRequestDigest.FromCanonicalValue(Convert.ToHexString(canonical));

        // An idempotency key is only useful if it is STABLE ACROSS RETRIES. Defaulting to
        // Guid.NewGuid() made every retry a distinct operation, so a client that timed out and
        // retried would silently run the turn twice — the endpoint would advertise idempotency
        // while providing none to exactly the callers who need it most. Absent a client-supplied
        // key, derive one from the canonical digest instead: identical content at the same
        // expected revision is then recognised as a replay without the caller doing anything.
        //
        // A client that wants two deliberately separate turns with identical text can still get
        // them by sending its own operation_id, which is the case that genuinely needs an opinion
        // from the caller.
        var operationId = new SessionOperationId(
            request.OperationId ?? new Guid(canonical.AsSpan(0, 16)));

        try
        {
            var id = new SessionId(sessionId);
            var session = sessions.ColdRuntime?.Open(id) ?? runtime.Open(id);
            var outcome = await session.RunTurnAsync(
                request.AppendPrompt, sampling, new SessionRevision(expectedRevision), operationId, digest, cancellationToken);
            var snapshot = runtime.GetSessionSnapshot(id);
            if (sessions.ColdRuntime is not null && outcome.Operation.State == SessionOperationState.Completed)
                sessions.ColdRuntime.EvictToDisk(session, engine.ModelId);
            string text = string.Concat(outcome.Chunks.Where(c => c.Kind == GenerateChunkKind.Text).Select(c => c.Text));
            string thinking = string.Concat(outcome.Chunks.Where(c => c.Kind == GenerateChunkKind.Thinking).Select(c => c.Text));
            return Results.Json(new SessionTurnResponse(
                    ToResponse(snapshot, session.CommittedRevision), operationId.ToString(),
                    outcome.Operation.State.ToString().ToLowerInvariant(), text, thinking,
                    outcome.IsIdempotentReplay),
                OpenTailStingrayJsonContext.Default.SessionTurnResponse);
        }
        catch (SessionNotFoundException)
        {
            return Results.NotFound();
        }
        catch (SessionRevisionConflictException ex)
        {
            return Results.Conflict(new ErrorResponse("session_revision_conflict",
                $"Expected revision {ex.ExpectedRevision.Value}, but current revision is {ex.ActualRevision.Value}."));
        }
        catch (SessionOperationConflictException ex)
        {
            return Results.Conflict(new ErrorResponse("session_operation_conflict", ex.Message));
        }
    }

    private static IResult Unavailable(IServerSessionRuntime sessions) =>
        Results.Json(new ErrorResponse("session_unavailable",
                sessions.UnavailabilityReason ?? "Sessions are not available for the loaded engine."),
            OpenTailStingrayJsonContext.Default.ErrorResponse,
            statusCode: StatusCodes.Status409Conflict);

    /// <summary>
    /// Builds the wire response. <paramref name="committedRevision"/> is taken from the SESSION, not
    /// from <paramref name="snapshot"/>, because those are two different counters and only one of
    /// them is the concurrency token.
    ///
    /// <para><c>HotSession.CommittedRevision</c> is the accepted-position count and is what
    /// <c>RunTurnAsync</c> compares <c>expected_revision</c> against; the store snapshot carries a
    /// separate per-turn counter. Reporting the snapshot's meant the API handed clients a number
    /// that conflict detection would always reject — one turn returned revision 1 while a turn sent
    /// at revision 1 was refused with "current revision is 6", so a client echoing back what it was
    /// just told could never make a second call.</para>
    /// </summary>
    private static SessionResponse ToResponse(SessionSnapshot snapshot, SessionRevision committedRevision) => new(
        snapshot.SessionId.ToString(), committedRevision.Value,
        snapshot.DurableRevision?.Value, snapshot.CurrentFencingEpoch,
        snapshot.Operations.Count);
}

/// <summary>Bounded session metadata returned by the lifecycle endpoints.</summary>
public sealed record SessionResponse(
    [property: JsonPropertyName("id")] string Id,
    [property: JsonPropertyName("committed_revision")] long CommittedRevision,
    [property: JsonPropertyName("durable_revision")] long? DurableRevision,
    [property: JsonPropertyName("fencing_epoch")] long FencingEpoch,
    [property: JsonPropertyName("operation_count")] int OperationCount);

/// <summary>Append-only turn input. Reusing an operation id requires exactly the same request.</summary>
// Wire names are stated EXPLICITLY, as on every other request DTO in this server, and not left
// to a naming policy. Output goes through OpenTailStingrayJsonContext (snake_case), but minimal-API
// input binding uses the ambient JsonOptions, whose Web defaults are camelCase — the source-gen
// context supplies metadata there, not its compile-time naming policy. Without these attributes the
// endpoint demanded camelCase requests while answering in snake_case, so a client that mirrored the
// response it just received got a 400.
public sealed record SessionTurnRequest(
    [property: JsonPropertyName("append_prompt")] string? AppendPrompt,
    [property: JsonPropertyName("expected_revision")] long? ExpectedRevision,
    [property: JsonPropertyName("operation_id")] Guid? OperationId = null,
    [property: JsonPropertyName("max_tokens")] int? MaxTokens = null,
    [property: JsonPropertyName("temperature")] float? Temperature = null,
    [property: JsonPropertyName("top_p")] float? TopP = null);

/// <summary>Completed or cancelled hot-session turn result.</summary>
public sealed record SessionTurnResponse(
    [property: JsonPropertyName("session")] SessionResponse Session,
    [property: JsonPropertyName("operation_id")] string OperationId,
    [property: JsonPropertyName("state")] string State,
    [property: JsonPropertyName("text")] string Text,
    [property: JsonPropertyName("thinking")] string Thinking,
    [property: JsonPropertyName("idempotent_replay")] bool IdempotentReplay);
