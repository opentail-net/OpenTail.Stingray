using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;
using Microsoft.Extensions.Options;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Sessions;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;

namespace OpenTail.Stingray.Server.Endpoints;

/// <summary>
/// Minimal named-session lifecycle endpoints. Hosts that expose these routes should attach their
/// authentication/tenant policy to the returned route group or individual endpoint builders.
/// This CPU-dense lane can restore a completed session when durable storage is configured. Its
/// bounded completed-operation/idempotency ledger travels with the persisted session, so a client
/// can retrieve or replay a retained response after restart; it is not an archival transcript.
///
/// <para>Every handler takes an unused <see cref="IInferenceEngine"/> parameter. That is not an
/// oversight: resolving it makes these routes share the engine's construction and failure
/// behaviour with the chat endpoints, so a host with no usable engine fails the same way here as
/// everywhere else instead of serving session routes over a runtime that was never brought up.
/// Discard the value with <c>_</c> rather than deleting the parameter.</para>
/// </summary>
public static partial class SessionEndpoints
{
    /// <summary>Maps create, inspect, and delete operations under <c>/v1/sessions</c>.</summary>
    public static IEndpointRouteBuilder MapSessionEndpoints(this IEndpointRouteBuilder app)
    {
        app.MapPost("/v1/sessions", Create);
        app.MapGet("/v1/sessions/{sessionId:guid}", Get);
        app.MapGet("/v1/sessions/{sessionId:guid}/operations/{operationId:guid}", GetOperation);
        // The turns route generates, exactly like /v1/chat/completions and /v1/messages, so it takes
        // the same bounded-admission filter. Without it MaxQueuedRequests bounded only the three
        // stateless chat routes while any number of named sessions could enqueue prompts in
        // parallel — the queue limit was route-shaped rather than engine-shaped.
        app.MapPost("/v1/sessions/{sessionId:guid}/turns", RunTurn).WithConcurrencyLimit();
        app.MapDelete("/v1/sessions/{sessionId:guid}", Delete);
        return app;
    }

    private static IResult Create(IInferenceEngine _, IServerSessionRuntime sessions)
    {
        if (sessions.Runtime is not { } runtime)
            return Unavailable(sessions);

        var session = sessions.ColdRuntime?.Create() ?? runtime.Create();
        return Results.Json(ToResponse(runtime.GetSessionSnapshot(session.SessionId)),
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
            return Results.Json(ToResponse(runtime.GetSessionSnapshot(id)),
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

    /// <summary>
    /// Returns the result retained for a completed or in-flight operation. This is the reconnect
    /// path for a client which lost the response to a turn. With configured durable CPU-dense
    /// storage, bounded completed records survive restart; hot-only sessions retain them in RAM.
    /// </summary>
    private static IResult GetOperation(
        Guid sessionId,
        Guid operationId,
        IInferenceEngine _,
        IServerSessionRuntime sessions)
    {
        if (sessions.Runtime is not { } runtime)
            return Unavailable(sessions);

        try
        {
            var id = new SessionId(sessionId);
            var session = sessions.ColdRuntime?.Open(id) ?? runtime.Open(id);
            var operation = runtime.GetOperation(id, new SessionOperationId(operationId));
            string text = string.Concat((operation.ResultChunks ?? []).Where(c => c.Kind == GenerateChunkKind.Text).Select(c => c.Text));
            string thinking = string.Concat((operation.ResultChunks ?? []).Where(c => c.Kind == GenerateChunkKind.Thinking).Select(c => c.Text));
            return Results.Json(new SessionOperationResponse(
                    ToResponse(runtime.GetSessionSnapshot(id)),
                    operation.OperationId.ToString(), operation.State.ToString().ToLowerInvariant(),
                    operation.CommittedRevision?.Value, operation.CreatedAt, operation.CompletedAt,
                    RedactFilesystemPaths(operation.FailureReason), text, thinking),
                OpenTailStingrayJsonContext.Default.SessionOperationResponse);
        }
        catch (SessionNotFoundException)
        {
            return Results.NotFound();
        }
        catch (KeyNotFoundException)
        {
            // A bounded durable ledger may prune old or oversized records. Do not claim an
            // unknown operation was retained merely because its session could be restored.
            return Results.NotFound();
        }
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
                    ToResponse(snapshot), operationId.ToString(),
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


    // Windows drive paths, UNC shares, and multi-segment POSIX paths. Deliberately narrow: this
    // redacts locations, not ordinary prose, so "max_tokens must be positive" survives intact.
    [GeneratedRegex(@"[A-Za-z]:\\[^\s""';)]*|\\\\[^\s""';)]*|/(?:[A-Za-z0-9_.\-]+/)+[A-Za-z0-9_.\-]*",
        RegexOptions.NonBacktracking)]
    private static partial Regex FilesystemPathToken();

    /// <summary>
    /// Strips filesystem locations out of an operation failure reason before it goes on the wire.
    ///
    /// <para>The reason is an exception message — <c>HotSession.RunTurnAsync</c> stores
    /// <c>ex.Message</c> from a general <c>catch</c>, so an IOException touching cold storage, the
    /// model file, or an mmap carries a full path. That is fine in a server log and not fine in an
    /// HTTP response: a path discloses a username, a project name, or an unreleased model name,
    /// which is precisely what <c>DiagnosticSurfaceRedactionTests</c> exists to prevent on the
    /// other diagnostic surfaces. Keep the sentence, drop the location.</para>
    /// </summary>
    private static string? RedactFilesystemPaths(string? reason) =>
        string.IsNullOrEmpty(reason) ? reason : FilesystemPathToken().Replace(reason, "[path]");

    private static IResult Unavailable(IServerSessionRuntime sessions) =>
        Results.Json(new ErrorResponse("session_unavailable",
                sessions.UnavailabilityReason ?? "Sessions are not available for the loaded engine."),
            OpenTailStingrayJsonContext.Default.ErrorResponse,
            statusCode: StatusCodes.Status409Conflict);

    /// <summary>
    /// Builds the wire response. <c>committed_revision</c> comes from the SNAPSHOT, because the
    /// store's revision is the one <c>RunTurnAsync</c> compares <c>expected_revision</c> against —
    /// publishing anything else hands the client a token that conflict detection rejects.
    ///
    /// <para>This previously published <c>HotSession.CommittedRevision</c>, the cursor's
    /// accepted-position count. The two diverge as soon as a turn accepts more than one position, so
    /// a session would advertise <c>committed_revision: 6</c> and then answer a turn carrying 6 with
    /// <c>409 "Expected revision 6, but current revision is 1"</c> — the only workable client
    /// pattern (read the value, echo it back) failed on its second turn. That property is now named
    /// <c>AcceptedPositionCount</c> so it cannot be mistaken for a concurrency token again, and the
    /// durable manifest records the store's counter too (manifest v3), so the live and restored
    /// lanes finally agree by construction rather than by coincidence.</para>
    /// </summary>
    private static SessionResponse ToResponse(SessionSnapshot snapshot) => new(
        snapshot.SessionId.ToString(), snapshot.CommittedRevision.Value,
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

/// <summary>Hot-runtime operation state and the retained generated result, when available.</summary>
public sealed record SessionOperationResponse(
    [property: JsonPropertyName("session")] SessionResponse Session,
    [property: JsonPropertyName("operation_id")] string OperationId,
    [property: JsonPropertyName("state")] string State,
    // This is the store's operation revision, deliberately distinct from session.committed_revision
    // (the latter is the optimistic-concurrency token clients must send with their next turn).
    [property: JsonPropertyName("operation_revision")] long? OperationRevision,
    [property: JsonPropertyName("created_at")] DateTimeOffset CreatedAt,
    [property: JsonPropertyName("completed_at")] DateTimeOffset? CompletedAt,
    [property: JsonPropertyName("failure_reason")] string? FailureReason,
    [property: JsonPropertyName("text")] string Text,
    [property: JsonPropertyName("thinking")] string Thinking);
