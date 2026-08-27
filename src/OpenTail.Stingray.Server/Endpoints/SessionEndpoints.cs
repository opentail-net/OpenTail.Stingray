using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Options;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Sessions;
using System.Collections.Immutable;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
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
/// <para>Single-model mode (the default): every handler forces the one configured
/// <see cref="IInferenceEngine"/> to load by resolving it explicitly before checking
/// <see cref="IServerSessionRuntime"/>. That is not an oversight — it makes these routes share the
/// engine's construction and failure behaviour with the chat endpoints, so a host with no usable
/// engine fails the same way here as everywhere else instead of serving session routes over a
/// runtime that was never brought up.</para>
///
/// <para>Multi-model mode (docs/032-multi-model-inference-runtime-plan.md Phase 7 follow-up):
/// a session is created against one resolved model and holds a real <see cref="ModelRuntimeHandle"/>
/// for its entire lifetime via <see cref="SessionModelRegistry"/> — closing the Phase 4 gap where
/// a live session's model runtime had no lease keeping it resident. The single configured
/// <see cref="IInferenceEngine"/> singleton is never resolved in this mode (it would try to
/// eagerly load <c>OpenTailStingrayServerOptions.ModelPath</c>, which may not even be set) — see
/// <see cref="TryResolveExistingSession"/> and <see cref="Create"/>, mirroring the branch
/// <c>OpenAiEndpoints.HandleChatCompletion</c> already established.</para>
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
        // Session-scoped tool authorization (docs/051 item #1): attach/detach/list the skills
        // this session's tool calls are checked against, and validate a candidate call. This
        // endpoint does not parse tool calls out of generated text itself -- the raw append-prompt
        // turn API has no chat template/adapter to do that with (unlike the OpenAI/Anthropic-compat
        // routes, which already detect calls from their own chunk stream). A caller that has its
        // own way of recognising a candidate call in this session's output asks here whether it's
        // authorized, instead of Stingray inventing a new tool-call wire syntax for raw sessions.
        app.MapPost("/v1/sessions/{sessionId:guid}/skills", AttachSkill);
        app.MapGet("/v1/sessions/{sessionId:guid}/skills", ListSkills);
        app.MapDelete("/v1/sessions/{sessionId:guid}/skills/{skillName}", DetachSkill);
        app.MapPost("/v1/sessions/{sessionId:guid}/tool-calls/validate", ValidateToolCall);
        // docs/051 item #2: checkpoint/rollback to any earlier point, not just the last turn (that
        // undo already happens internally on RunTurnAsync's own failure paths). In-memory only,
        // like every other HotSession capability — a checkpoint token is opaque and valid only
        // while the same HotSession instance still holds the retained cache it was taken from.
        app.MapPost("/v1/sessions/{sessionId:guid}/checkpoints", CreateCheckpoint);
        app.MapPost("/v1/sessions/{sessionId:guid}/rollback", Rollback);
        // docs/051 item #3: fork lineage/aggregate-metrics observability for HotSessionRuntime.Fork.
        app.MapGet("/v1/sessions/{sessionId:guid}/tree", GetTree);
        // docs/051 item #4: explicit proactive suspend, for a caller that knows a session is going
        // idle for a while and wants to free its cache now rather than wait for HotSessionRuntime's
        // own pressure-driven idle reclaim to get to it. Resume is a no-op endpoint offered purely
        // for symmetry — HotSession has no separate "resumed" state; the next turn just re-prefills.
        app.MapPost("/v1/sessions/{sessionId:guid}/suspend", Suspend);
        app.MapPost("/v1/sessions/{sessionId:guid}/resume", Resume);
        return app;
    }

    /// <summary>The (runtime, optional cold runtime, engine model id) needed to operate an
    /// <em>existing</em> session — resolved once per request by <see cref="TryResolveExistingSession"/>
    /// so every handler below shares the same single-vs-multi-model branch instead of repeating it.</summary>
    private readonly record struct SessionRoute(HotSessionRuntime Runtime, ColdSessionRuntime? ColdRuntime, string EngineModelId);

    /// <summary>
    /// Resolves the route for an existing session. Single-model mode forces the configured engine
    /// to load and checks <see cref="IServerSessionRuntime"/> exactly as every handler did before
    /// multi-model mode existed. Multi-model mode looks up the <see cref="SessionModelRegistry"/>
    /// binding created by <see cref="Create"/> — a miss means "no such live session" (404), a
    /// materially different outcome from "sessions are unavailable" (409), since the feature works
    /// fine here, this particular id just isn't bound to anything.
    /// </summary>
    /// <returns>The <see cref="IResult"/> to return immediately, or <c>null</c> when
    /// <paramref name="route"/> was resolved successfully and the caller should proceed.</returns>
    private static IResult? TryResolveExistingSession(
        HttpContext ctx, Guid sessionId, IInferenceService inferenceService, out SessionRoute route)
    {
        if (inferenceService.IsMultiModel)
        {
            var registry = ctx.RequestServices.GetRequiredService<SessionModelRegistry>();
            if (!registry.TryGet(new SessionId(sessionId), out var handle))
            {
                route = default;
                return Results.NotFound();
            }
            var loaded = handle.Runtime.Loaded;
            // SessionRuntime is guaranteed non-null here — Create only ever binds a session after
            // confirming it, and a binding's model identity never changes for its lifetime.
            route = new SessionRoute(loaded.SessionRuntime!, loaded.ColdSessionRuntime, loaded.Engine.ModelId);
            return null;
        }

        var engine = ctx.RequestServices.GetRequiredService<IInferenceEngine>();
        var sessions = ctx.RequestServices.GetRequiredService<IServerSessionRuntime>();
        if (sessions.Runtime is not { } runtime)
        {
            route = default;
            return Unavailable(sessions);
        }
        route = new SessionRoute(runtime, sessions.ColdRuntime, engine.ModelId);
        return null;
    }

    private static async Task<IResult> Create(
        HttpContext ctx, IInferenceService inferenceService, SessionCreateRequest? request = null)
    {
        if (inferenceService.IsMultiModel)
        {
            ModelId modelId;
            try
            {
                modelId = inferenceService.ResolveModel(request?.Model);
            }
            catch (ModelNotFoundException ex)
            {
                return Results.Json(new ErrorResponse("invalid_request_error", ex.Message),
                    OpenTailStingrayJsonContext.Default.ErrorResponse, statusCode: StatusCodes.Status404NotFound);
            }

            ModelRuntimeHandle handle;
            try
            {
                handle = await inferenceService.AcquireAsync(modelId, ctx.RequestAborted);
            }
            catch (InsufficientResourcesException ex)
            {
                return Results.Json(new ErrorResponse("server_error", ex.Message),
                    OpenTailStingrayJsonContext.Default.ErrorResponse, statusCode: StatusCodes.Status503ServiceUnavailable);
            }

            if (handle.Runtime.Loaded.SessionRuntime is not { } runtime)
            {
                // Not keeping the handle — release it immediately rather than leaking a residency
                // claim on a model this request is about to fail against.
                handle.Dispose();
                return Results.Json(new ErrorResponse("session_unavailable",
                        $"The loaded engine for model '{modelId}' does not expose the CPU-dense session runtime."),
                    OpenTailStingrayJsonContext.Default.ErrorResponse, statusCode: StatusCodes.Status409Conflict);
            }

            var coldRuntime = handle.Runtime.Loaded.ColdSessionRuntime;
            var session = coldRuntime?.Create() ?? runtime.Create();
            // The handle is now owned by the registry for this session's whole lifetime — not
            // disposed here (docs/032 Phase 4 gap fix: this is what keeps the model resident).
            ctx.RequestServices.GetRequiredService<SessionModelRegistry>().Bind(session.SessionId, handle);
            return Results.Json(ToResponse(runtime.GetSessionSnapshot(session.SessionId), session),
                OpenTailStingrayJsonContext.Default.SessionResponse, statusCode: StatusCodes.Status201Created);
        }

        // Single-model mode: byte-identical to before multi-model session routing existed.
        _ = ctx.RequestServices.GetRequiredService<IInferenceEngine>();
        var sessions = ctx.RequestServices.GetRequiredService<IServerSessionRuntime>();
        if (sessions.Runtime is not { } singleRuntime)
            return Unavailable(sessions);

        var singleSession = sessions.ColdRuntime?.Create() ?? singleRuntime.Create();
        return Results.Json(ToResponse(singleRuntime.GetSessionSnapshot(singleSession.SessionId), singleSession),
            OpenTailStingrayJsonContext.Default.SessionResponse, statusCode: StatusCodes.Status201Created);
    }

    private static IResult Get(Guid sessionId, HttpContext ctx, IInferenceService inferenceService)
    {
        if (TryResolveExistingSession(ctx, sessionId, inferenceService, out var route) is { } error)
            return error;

        try
        {
            var id = new SessionId(sessionId);
            var session = route.ColdRuntime?.Open(id) ?? route.Runtime.Open(id);
            return Results.Json(ToResponse(route.Runtime.GetSessionSnapshot(id), session),
                OpenTailStingrayJsonContext.Default.SessionResponse);
        }
        catch (SessionNotFoundException)
        {
            return Results.NotFound();
        }
    }

    private static IResult Delete(Guid sessionId, HttpContext ctx, IInferenceService inferenceService)
    {
        if (TryResolveExistingSession(ctx, sessionId, inferenceService, out var route) is { } error)
            return error;

        var id = new SessionId(sessionId);
        bool deleted = route.ColdRuntime?.Delete(id) ?? route.Runtime.Delete(id);
        // Release the binding regardless of `deleted` — a session the registry knows about but
        // the runtime no longer does (shouldn't happen, but never leave a dangling residency
        // claim on the strength of an assumption) must not permanently pin its model.
        if (inferenceService.IsMultiModel)
            ctx.RequestServices.GetRequiredService<SessionModelRegistry>().Release(id);
        return deleted ? Results.NoContent() : Results.NotFound();
    }

    private static IResult AttachSkill(
        Guid sessionId, SessionAttachSkillRequest request, HttpContext ctx, IInferenceService inferenceService)
    {
        if (TryResolveExistingSession(ctx, sessionId, inferenceService, out var route) is { } error)
            return error;
        if (string.IsNullOrEmpty(request.Name))
            return Results.BadRequest(new ErrorResponse("invalid_request_error", "name must not be empty."));

        try
        {
            var id = new SessionId(sessionId);
            var session = route.ColdRuntime?.Open(id) ?? route.Runtime.Open(id);
            var wireSkill = new WireSkill(request.Name, request.Description, request.Instructions, request.Tools);
            session.AttachSkill(wireSkill.ToCoreSkill());
            return Results.Json(ToSkillResponse(session.AttachedSkills.Last(s => s.Name == request.Name)),
                OpenTailStingrayJsonContext.Default.SessionSkillResponse, statusCode: StatusCodes.Status201Created);
        }
        catch (SessionNotFoundException)
        {
            return Results.NotFound();
        }
    }

    private static IResult ListSkills(Guid sessionId, HttpContext ctx, IInferenceService inferenceService)
    {
        if (TryResolveExistingSession(ctx, sessionId, inferenceService, out var route) is { } error)
            return error;

        try
        {
            var id = new SessionId(sessionId);
            var session = route.ColdRuntime?.Open(id) ?? route.Runtime.Open(id);
            return Results.Json(
                new SessionSkillsResponse(session.AttachedSkills.Select(ToSkillResponse).ToArray()),
                OpenTailStingrayJsonContext.Default.SessionSkillsResponse);
        }
        catch (SessionNotFoundException)
        {
            return Results.NotFound();
        }
    }

    private static IResult DetachSkill(
        Guid sessionId, string skillName, HttpContext ctx, IInferenceService inferenceService)
    {
        if (TryResolveExistingSession(ctx, sessionId, inferenceService, out var route) is { } error)
            return error;

        try
        {
            var id = new SessionId(sessionId);
            var session = route.ColdRuntime?.Open(id) ?? route.Runtime.Open(id);
            return session.DetachSkill(skillName) ? Results.NoContent() : Results.NotFound();
        }
        catch (SessionNotFoundException)
        {
            return Results.NotFound();
        }
    }

    private static IResult ValidateToolCall(
        Guid sessionId, SessionValidateToolCallRequest request, HttpContext ctx, IInferenceService inferenceService)
    {
        if (TryResolveExistingSession(ctx, sessionId, inferenceService, out var route) is { } error)
            return error;
        if (string.IsNullOrEmpty(request.Name))
            return Results.BadRequest(new ErrorResponse("invalid_request_error", "name must not be empty."));

        try
        {
            var id = new SessionId(sessionId);
            var session = route.ColdRuntime?.Open(id) ?? route.Runtime.Open(id);
            var call = new OpenTail.Stingray.Core.Tools.ToolCall(Guid.NewGuid().ToString(), request.Name, request.Arguments);
            return Results.Json(new SessionValidateToolCallResponse(session.ValidateToolCall(call)),
                OpenTailStingrayJsonContext.Default.SessionValidateToolCallResponse);
        }
        catch (SessionNotFoundException)
        {
            return Results.NotFound();
        }
    }

    private static SessionSkillResponse ToSkillResponse(ISkill skill) =>
        new(skill.Name, skill.Description, skill.Tools.Select(t => t.Name).ToArray());

    private static IResult CreateCheckpoint(Guid sessionId, HttpContext ctx, IInferenceService inferenceService)
    {
        if (TryResolveExistingSession(ctx, sessionId, inferenceService, out var route) is { } error)
            return error;

        try
        {
            var id = new SessionId(sessionId);
            var session = route.ColdRuntime?.Open(id) ?? route.Runtime.Open(id);
            var checkpoint = session.CreateCheckpoint();
            string token = Convert.ToBase64String(SessionCursorCodec.Encode(checkpoint.Cursor));
            return Results.Json(new SessionCheckpointResponse(token, checkpoint.CommittedRevision.Value),
                OpenTailStingrayJsonContext.Default.SessionCheckpointResponse, statusCode: StatusCodes.Status201Created);
        }
        catch (SessionNotFoundException)
        {
            return Results.NotFound();
        }
    }

    private static async Task<IResult> Rollback(
        Guid sessionId, SessionRollbackRequest request, HttpContext ctx, IInferenceService inferenceService)
    {
        if (TryResolveExistingSession(ctx, sessionId, inferenceService, out var route) is { } error)
            return error;
        if (string.IsNullOrEmpty(request.CheckpointToken))
            return Results.BadRequest(new ErrorResponse("invalid_request_error", "checkpoint_token must not be empty."));

        SessionCursor cursor;
        try
        {
            cursor = SessionCursorCodec.Decode(Convert.FromBase64String(request.CheckpointToken));
        }
        catch (Exception ex) when (ex is FormatException or SessionCursorFormatException)
        {
            return Results.BadRequest(new ErrorResponse("invalid_request_error",
                $"checkpoint_token is not a valid checkpoint: {ex.Message}"));
        }

        try
        {
            var id = new SessionId(sessionId);
            var session = route.ColdRuntime?.Open(id) ?? route.Runtime.Open(id);
            var checkpoint = new HotSessionCheckpoint(id, cursor, new SessionRevision(request.CommittedRevision));
            await session.RollbackAsync(checkpoint);
            return Results.Json(ToResponse(route.Runtime.GetSessionSnapshot(id), session),
                OpenTailStingrayJsonContext.Default.SessionResponse);
        }
        catch (SessionNotFoundException)
        {
            return Results.NotFound();
        }
        // Turn in progress, or the retained cache can no longer rewind that far (e.g. evicted
        // since the checkpoint was taken) — both are "this rollback cannot happen right now",
        // not a malformed request.
        catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException)
        {
            return Results.Json(new ErrorResponse("session_rollback_conflict", ex.Message),
                OpenTailStingrayJsonContext.Default.ErrorResponse, statusCode: StatusCodes.Status409Conflict);
        }
    }

    private static IResult GetTree(Guid sessionId, HttpContext ctx, IInferenceService inferenceService)
    {
        if (TryResolveExistingSession(ctx, sessionId, inferenceService, out var route) is { } error)
            return error;

        try
        {
            var id = new SessionId(sessionId);
            var session = route.ColdRuntime?.Open(id) ?? route.Runtime.Open(id);
            var tree = session.Tree;
            return Results.Json(new SessionTreeResponse(
                    tree.RootId.ToString(), tree.ParentId?.ToString(),
                    tree.Children.Select(c => c.ToString()).ToArray(),
                    ToMetricsResponse(tree.CumulativeTreeMetrics)),
                OpenTailStingrayJsonContext.Default.SessionTreeResponse);
        }
        catch (SessionNotFoundException)
        {
            return Results.NotFound();
        }
    }

    private static async Task<IResult> Suspend(Guid sessionId, HttpContext ctx, IInferenceService inferenceService)
    {
        if (TryResolveExistingSession(ctx, sessionId, inferenceService, out var route) is { } error)
            return error;

        try
        {
            var id = new SessionId(sessionId);
            var session = route.ColdRuntime?.Open(id) ?? route.Runtime.Open(id);
            await session.SuspendAsync(ctx.RequestAborted);
            return Results.Json(ToResponse(route.Runtime.GetSessionSnapshot(id), session),
                OpenTailStingrayJsonContext.Default.SessionResponse);
        }
        catch (SessionNotFoundException)
        {
            return Results.NotFound();
        }
        catch (InvalidOperationException ex)
        {
            return Results.Json(new ErrorResponse("session_suspend_conflict", ex.Message),
                OpenTailStingrayJsonContext.Default.ErrorResponse, statusCode: StatusCodes.Status409Conflict);
        }
    }

    private static async Task<IResult> Resume(Guid sessionId, HttpContext ctx, IInferenceService inferenceService)
    {
        if (TryResolveExistingSession(ctx, sessionId, inferenceService, out var route) is { } error)
            return error;

        try
        {
            var id = new SessionId(sessionId);
            var session = route.ColdRuntime?.Open(id) ?? route.Runtime.Open(id);
            await session.ResumeAsync(ctx.RequestAborted);
            return Results.Json(ToResponse(route.Runtime.GetSessionSnapshot(id), session),
                OpenTailStingrayJsonContext.Default.SessionResponse);
        }
        catch (SessionNotFoundException)
        {
            return Results.NotFound();
        }
    }

    /// <summary>
    /// Returns the result retained for a completed or in-flight operation. This is the reconnect
    /// path for a client which lost the response to a turn. With configured durable CPU-dense
    /// storage, bounded completed records survive restart; hot-only sessions retain them in RAM.
    /// </summary>
    private static IResult GetOperation(
        Guid sessionId,
        Guid operationId,
        HttpContext ctx,
        IInferenceService inferenceService)
    {
        if (TryResolveExistingSession(ctx, sessionId, inferenceService, out var route) is { } error)
            return error;

        try
        {
            var id = new SessionId(sessionId);
            var session = route.ColdRuntime?.Open(id) ?? route.Runtime.Open(id);
            var operation = route.Runtime.GetOperation(id, new SessionOperationId(operationId));
            var resultChunks = (operation.ResultChunks ?? []).ToImmutableArray();
            string text = string.Concat(resultChunks.Where(c => c.Kind == GenerateChunkKind.Text).Select(c => c.Text));
            string thinking = string.Concat(resultChunks.Where(c => c.Kind == GenerateChunkKind.Thinking).Select(c => c.Text));
            var (finishReason, toolCalls) = HotSessionTurnResult.DescribeOutcome(
                resultChunks, cancelled: false, failed: operation.State == SessionOperationState.Failed);
            return Results.Json(new SessionOperationResponse(
                    ToResponse(route.Runtime.GetSessionSnapshot(id), session),
                    operation.OperationId.ToString(), operation.State.ToString().ToLowerInvariant(),
                    operation.CommittedRevision?.Value, operation.CreatedAt, operation.CompletedAt,
                    RedactFilesystemPaths(operation.FailureReason), text, thinking,
                    finishReason.ToString().ToLowerInvariant(), ToToolCallResponses(toolCalls)),
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
        HttpContext ctx,
        IInferenceService inferenceService,
        IOptions<OpenTailStingrayServerOptions> options,
        CancellationToken cancellationToken)
    {
        if (TryResolveExistingSession(ctx, sessionId, inferenceService, out var route) is { } error)
            return error;
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
            thinkingDisabled: options.Value.DisableThinking,
            allowedChoices: request.AllowedChoices);
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
            var session = route.ColdRuntime?.Open(id) ?? route.Runtime.Open(id);
            var outcome = await session.RunTurnAsync(
                request.AppendPrompt, sampling, new SessionRevision(expectedRevision), operationId, digest, cancellationToken);
            var snapshot = route.Runtime.GetSessionSnapshot(id);
            if (route.ColdRuntime is not null && outcome.Operation.State == SessionOperationState.Completed)
                route.ColdRuntime.EvictToDisk(session, route.EngineModelId);
            string text = string.Concat(outcome.Chunks.Where(c => c.Kind == GenerateChunkKind.Text).Select(c => c.Text));
            string thinking = string.Concat(outcome.Chunks.Where(c => c.Kind == GenerateChunkKind.Thinking).Select(c => c.Text));
            return Results.Json(new SessionTurnResponse(
                    ToResponse(snapshot, session), operationId.ToString(),
                    outcome.Operation.State.ToString().ToLowerInvariant(), text, thinking,
                    outcome.IsIdempotentReplay,
                    outcome.FinishReason.ToString().ToLowerInvariant(), ToToolCallResponses(outcome.ToolCalls)),
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
    private static SessionResponse ToResponse(SessionSnapshot snapshot, HotSession session) => new(
        snapshot.SessionId.ToString(), snapshot.CommittedRevision.Value,
        snapshot.DurableRevision?.Value, snapshot.CurrentFencingEpoch,
        snapshot.Operations.Count, ToMetricsResponse(session.Metrics), session.IsSuspended,
        ToMetadataResponse(session.Metadata));

    private static SessionMetricsResponse ToMetricsResponse(ISessionMetrics m) => new(
        m.PromptTokens, m.GeneratedTokens, m.TotalPrefillTime.TotalSeconds,
        m.TotalGenerationTime.TotalSeconds, m.TokensPerSecond, m.KvPagesHeld);

    // Metadata values are arbitrary host-supplied objects (ISessionMetadata.Set(string, object?))
    // with no NativeAOT-safe general serialization — stringified for the wire rather than
    // reflection-serialized. A caller that needs a structured value back should store it as a
    // pre-serialized JSON string and parse it client-side.
    private static Dictionary<string, string?>? ToMetadataResponse(ISessionMetadata metadata)
    {
        var entries = metadata.GetEntries();
        if (entries.Count == 0) return null;
        var result = new Dictionary<string, string?>(entries.Count, StringComparer.Ordinal);
        foreach (var (key, value) in entries) result[key] = value?.ToString();
        return result;
    }

    private static SessionToolCallResponse[]? ToToolCallResponses(
        IReadOnlyList<OpenTail.Stingray.Core.Tools.ToolCall> calls) =>
        calls.Count == 0 ? null : [.. calls.Select(c => new SessionToolCallResponse(c.Id, c.Name, c.Arguments))];
}

/// <summary>Optional session-creation body. <c>model</c> is meaningful only in multi-model mode
/// (matches an <see cref="NamedModelOptions.Alias"/>, case-insensitively; null/unmatched-empty
/// resolves to the first configured model, same contract as the chat-style endpoints'
/// <c>model</c> field) — single-model mode ignores it entirely, exactly like every other field
/// here would be ignored if a client sent one before this type existed.</summary>
public sealed record SessionCreateRequest(
    [property: JsonPropertyName("model")] string? Model = null);

/// <summary>Bounded session metadata returned by the lifecycle endpoints.</summary>
public sealed record SessionResponse(
    [property: JsonPropertyName("id")] string Id,
    [property: JsonPropertyName("committed_revision")] long CommittedRevision,
    [property: JsonPropertyName("durable_revision")] long? DurableRevision,
    [property: JsonPropertyName("fencing_epoch")] long FencingEpoch,
    [property: JsonPropertyName("operation_count")] int OperationCount,
    [property: JsonPropertyName("metrics")] SessionMetricsResponse Metrics,
    // True when the session currently holds no retained KV cache — never prefilled, evicted by
    // an explicit POST .../suspend, or reclaimed by HotSessionRuntime's own idle pressure.
    [property: JsonPropertyName("is_suspended")] bool IsSuspended = false,
    // Arbitrary host-supplied ISessionMetadata entries, stringified — see ToMetadataResponse.
    // Null (omitted) rather than an empty object when nothing has been set.
    [property: JsonPropertyName("metadata")] Dictionary<string, string?>? Metadata = null);

/// <summary>Per-session inference/KV usage statistics — <see cref="OpenTail.Stingray.Sessions.ISessionMetrics"/>.</summary>
public sealed record SessionMetricsResponse(
    [property: JsonPropertyName("prompt_tokens")] long PromptTokens,
    [property: JsonPropertyName("generated_tokens")] long GeneratedTokens,
    [property: JsonPropertyName("total_prefill_seconds")] double TotalPrefillSeconds,
    [property: JsonPropertyName("total_generation_seconds")] double TotalGenerationSeconds,
    [property: JsonPropertyName("tokens_per_second")] double TokensPerSecond,
    [property: JsonPropertyName("kv_pages_held")] int KvPagesHeld);

/// <summary>One structured tool call the model emitted during a turn.</summary>
public sealed record SessionToolCallResponse(
    [property: JsonPropertyName("id")] string Id,
    [property: JsonPropertyName("name")] string Name,
    [property: JsonPropertyName("arguments")] JsonElement Arguments);

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
    [property: JsonPropertyName("top_p")] float? TopP = null,
    // Constrains generation to one of these literal strings (SamplingParams.AllowedChoices),
    // e.g. ["yes", "no"] for a forced classification turn. Absent/empty means unconstrained.
    [property: JsonPropertyName("allowed_choices")] string[]? AllowedChoices = null);

/// <summary>Completed or cancelled hot-session turn result.</summary>
public sealed record SessionTurnResponse(
    [property: JsonPropertyName("session")] SessionResponse Session,
    [property: JsonPropertyName("operation_id")] string OperationId,
    [property: JsonPropertyName("state")] string State,
    [property: JsonPropertyName("text")] string Text,
    [property: JsonPropertyName("thinking")] string Thinking,
    [property: JsonPropertyName("idempotent_replay")] bool IdempotentReplay,
    // Why generation stopped (HotSessionTurnResult.FinishReason) — "completed", "max_tokens",
    // "tool_call", "context_limit", "cancelled", or "failed".
    [property: JsonPropertyName("finish_reason")] string FinishReason,
    [property: JsonPropertyName("tool_calls")] SessionToolCallResponse[]? ToolCalls = null);

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
    [property: JsonPropertyName("thinking")] string Thinking,
    [property: JsonPropertyName("finish_reason")] string FinishReason,
    [property: JsonPropertyName("tool_calls")] SessionToolCallResponse[]? ToolCalls = null);

/// <summary>
/// Attaches a native <see cref="OpenTail.Stingray.Core.Skill"/> to the session. <c>Tools</c> are
/// authorized immediately (<see cref="OpenTail.Stingray.Sessions.HotSession.ValidateToolCall"/>) —
/// Stingray does not execute them itself. <c>Instructions</c>, if any, are queued and prepended to
/// the append-prompt text of the session's NEXT turn only (see
/// <see cref="OpenTail.Stingray.Sessions.HotSession.AttachSkill"/>) — a turn already committed to
/// the KV cache cannot be retroactively rewritten to have "seen" a skill attached afterward.
/// </summary>
public sealed record SessionAttachSkillRequest(
    [property: JsonPropertyName("name")] string Name,
    [property: JsonPropertyName("description")] string? Description = null,
    [property: JsonPropertyName("instructions")] WireSkillInstruction[]? Instructions = null,
    [property: JsonPropertyName("tools")] WireSkillTool[]? Tools = null);

/// <summary>One skill currently attached to the session.</summary>
public sealed record SessionSkillResponse(
    [property: JsonPropertyName("name")] string Name,
    [property: JsonPropertyName("description")] string? Description,
    [property: JsonPropertyName("tools")] string[] Tools);

/// <summary>Every skill currently attached to the session.</summary>
public sealed record SessionSkillsResponse(
    [property: JsonPropertyName("skills")] SessionSkillResponse[] Skills);

/// <summary>
/// A candidate tool call to check against the session's attached skills (and
/// <see cref="OpenTail.Stingray.Sessions.HotSession.ToolProvider"/>, if set). The raw append-prompt
/// turn API has no chat-template adapter to detect calls in generated text itself, so the caller —
/// which does its own detection against this session's output — asks here whether a given call is
/// authorized before acting on it.
/// </summary>
public sealed record SessionValidateToolCallRequest(
    [property: JsonPropertyName("name")] string Name,
    [property: JsonPropertyName("arguments")] JsonElement Arguments = default);

/// <summary>Whether a candidate tool call names a tool this session currently authorizes.</summary>
public sealed record SessionValidateToolCallResponse(
    [property: JsonPropertyName("authorized")] bool Authorized);

/// <summary>
/// An opaque checkpoint a later <c>POST /v1/sessions/{id}/rollback</c> can restore. In-memory
/// only, like every other HotSession capability — valid only while the same <c>HotSession</c>
/// instance still holds the retained cache it was taken from (echoed back to the SAME session,
/// after which it may fail with 409 if the cache has since been evicted).
/// </summary>
public sealed record SessionCheckpointResponse(
    [property: JsonPropertyName("checkpoint_token")] string CheckpointToken,
    [property: JsonPropertyName("committed_revision")] long CommittedRevision);

/// <summary>Both fields from a <see cref="SessionCheckpointResponse"/>, echoed back verbatim.</summary>
public sealed record SessionRollbackRequest(
    [property: JsonPropertyName("checkpoint_token")] string CheckpointToken,
    [property: JsonPropertyName("committed_revision")] long CommittedRevision);

/// <summary>Fork lineage and aggregated subtree metrics — <see cref="OpenTail.Stingray.Sessions.ISessionTree"/>.</summary>
public sealed record SessionTreeResponse(
    [property: JsonPropertyName("root_id")] string RootId,
    [property: JsonPropertyName("parent_id")] string? ParentId,
    [property: JsonPropertyName("children")] string[] Children,
    [property: JsonPropertyName("cumulative_metrics")] SessionMetricsResponse CumulativeMetrics);
