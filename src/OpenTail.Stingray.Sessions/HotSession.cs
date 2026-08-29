
namespace OpenTail.Stingray.Sessions;

/// <summary>
/// Outcome of one in-memory hot session turn. <see cref="FinishReason"/> and <see cref="ToolCalls"/>
/// are derived from <see cref="Chunks"/> the same way <c>GenerationStream</c> used to derive
/// <c>GenerationResult</c> from an <see cref="IInferenceSession"/>'s chunk stream.
/// </summary>
public sealed record HotSessionTurnResult(
    SessionOperationSnapshot Operation,
    SessionCursor Cursor,
    ImmutableArray<GenerateChunk> Chunks,
    bool IsIdempotentReplay,
    FinishReason FinishReason,
    IReadOnlyList<OpenTail.Stingray.Core.Tools.ToolCall> ToolCalls)
{
    /// <summary>
    /// Derives <see cref="FinishReason"/> and <see cref="ToolCalls"/> from a completed chunk
    /// sequence. <paramref name="cancelled"/>/<paramref name="failed"/> take priority over whatever
    /// the chunks themselves indicate, since a turn that didn't reach its own <c>Stop</c>/<c>ToolCall</c>
    /// chunk was interrupted before the engine could report why.
    /// </summary>
    public static (FinishReason Reason, IReadOnlyList<OpenTail.Stingray.Core.Tools.ToolCall> ToolCalls) DescribeOutcome(
        ImmutableArray<GenerateChunk> chunks, bool cancelled, bool failed)
    {
        var toolCalls = new List<OpenTail.Stingray.Core.Tools.ToolCall>();
        FinishReason? reason = null;
        foreach (var chunk in chunks)
        {
            if (chunk.Kind == GenerateChunkKind.ToolCall)
            {
                reason = OpenTail.Stingray.Sessions.FinishReason.ToolCall;
                if (chunk.ToolCalls is not null) toolCalls.AddRange(chunk.ToolCalls);
            }
            else if (chunk.Kind == GenerateChunkKind.Stop)
            {
                if (chunk.TruncatedByResourceBudget)
                    reason = OpenTail.Stingray.Sessions.FinishReason.ContextLimit;
                else if (chunk.TruncatedByMaxTokens)
                    reason = OpenTail.Stingray.Sessions.FinishReason.MaxTokens;
                else
                    reason ??= OpenTail.Stingray.Sessions.FinishReason.Completed;

                if (chunk.ToolCalls is not null) toolCalls.AddRange(chunk.ToolCalls);
            }
            else if (chunk.ToolCalls is not null)
            {
                toolCalls.AddRange(chunk.ToolCalls);
            }
        }

        if (cancelled) return (OpenTail.Stingray.Sessions.FinishReason.Cancelled, toolCalls);
        if (failed) return (OpenTail.Stingray.Sessions.FinishReason.Failed, toolCalls);
        return (reason ?? OpenTail.Stingray.Sessions.FinishReason.Completed, toolCalls);
    }
}

/// <summary>
/// Point-in-time snapshot of a <see cref="HotSession"/>'s cursor and committed revision, captured
/// by <see cref="HotSession.CreateCheckpoint"/> and restored by <see cref="HotSession.RollbackAsync"/>.
/// In-memory only, like every other <see cref="HotSession"/> capability — it does not survive the
/// session's retained cache being evicted or the process restarting.
/// </summary>
public sealed record HotSessionCheckpoint(SessionId SessionId, SessionCursor Cursor, SessionRevision CommittedRevision);

/// <summary>
/// Couples one retained CPU sequence state to the in-memory revision/lease ledger. This is the
/// Milestone 1 hot path: it has no durability or cross-process recovery guarantee. Callers submit
/// append-only prompt suffixes; the authoritative execution cursor is updated only after the
/// operation reaches <see cref="SessionOperationState.Completed"/>.
/// </summary>
public sealed class HotSession : IDisposable
{
    private readonly object _cursorGate = new();
    private readonly SemaphoreSlim _turnGate = new(1, 1);
    private readonly ContinuousBatchingEngine _engine;
    private readonly ITokenizer _tokenizer;
    private readonly InMemorySessionStore _store;
    private readonly SessionResourceBudget _resources;
    private readonly long _kvBytesPerToken;
    private readonly int _maxSequenceLength;
    private readonly int _maxCapturedOutputChunks;
    private readonly string _modelKey;
    private readonly Action<SessionId, HotSession>? _onDisposed;
    private readonly Func<SessionId, long, long>? _reclaimIdleBytes;
    private readonly RetainedSequenceState _state = new();
    private readonly object _lifecycleGate = new();
    private readonly SessionMetrics _metrics;
    private readonly SessionMetadata _metadata = new();
    private readonly List<OpenTail.Stingray.Core.ISkill> _attachedSkills = [];
    private readonly List<string> _pendingInstructionPreamble = [];
    private readonly HotSessionRuntime? _runtime;
    private SessionCursor _cursor = new([], 0, 0, 0, 0, StateCoverage.Full);
    private bool _disposed;

    /// <summary>Direct parent this session was forked from (see <see cref="HotSessionRuntime.Fork"/>), or null if it is a root session.</summary>
    public SessionId? ParentSessionId { get; init; }

    /// <summary>Read-only topology surface exposing this session's lineage and aggregated subtree metrics.</summary>
    public ISessionTree Tree => new HotSessionTree(this, _runtime);

    internal HotSession(
        ContinuousBatchingEngine engine,
        ITokenizer tokenizer,
        InMemorySessionStore store,
        SessionResourceBudget resources,
        long kvBytesPerToken,
        int maxSequenceLength,
        int maxCapturedOutputChunks,
        SessionId sessionId,
        Action<SessionId, HotSession>? onDisposed = null,
        string? modelKey = null,
        Func<SessionId, long, long>? reclaimIdleBytes = null,
        HotSessionRuntime? runtime = null)
    {
        _runtime = runtime;
        _engine = engine;
        _tokenizer = tokenizer;
        _store = store;
        _resources = resources;
        _kvBytesPerToken = kvBytesPerToken;
        _maxSequenceLength = maxSequenceLength;
        _maxCapturedOutputChunks = maxCapturedOutputChunks;
        SessionId = sessionId;
        _onDisposed = onDisposed;
        _modelKey = modelKey ?? engine.ModelId;
        _reclaimIdleBytes = reclaimIdleBytes;
        // Page granularity is not queryable from RetainedSequenceState's backend-opaque cache handle
        // (see its own doc comment); KvPageSize.Default (32 tokens/page) is the same fixed constant
        // every concrete backend (CpuKvCache, CudaSequenceKvCache) already uses, so this reports the
        // same page count a real backend query would.
        _metrics = new SessionMetrics(() => KvPageMath.GetRequiredPageCount(_state.MaterializedPosition, KvPageSize.Default.Tokens));
    }

    /// <summary>Read-only metrics surface exposing per-session inference and physical KV page usage statistics.</summary>
    public ISessionMetrics Metrics => _metrics;

    /// <summary>Application-level metadata bag associated with this session (see <see cref="ISessionMetadata"/>).</summary>
    public ISessionMetadata Metadata => _metadata;

    /// <summary>
    /// Fine-tuned LoRA adapter intended to apply during this session's generation.
    /// <para><b>Not yet wired to inference.</b> <see cref="ContinuousBatchingEngine"/>'s batched
    /// forward pass (<see cref="IBatchedForwardPass.BatchForwardMulti"/>/<c>PrefillPackedMulti</c>)
    /// amortizes one shared set of model weights across every sequence in a batch in a single
    /// matmul; applying a different LoRA delta per row needs new batched-kernel support that does
    /// not exist yet (tracked separately — see docs/032). This property exists so the setting has
    /// somewhere to live and survives the port; setting it currently has no effect on generation,
    /// matching <c>InferenceSession.ActiveLora</c>'s own behavior before this port (it was never
    /// wired to its forward pass there either).</para>
    /// </summary>
    public OpenTail.Stingray.Core.Lora.LoraAdapter? ActiveLora { get; set; }

    /// <summary>Tool catalog consulted by <see cref="ValidateToolCall"/> alongside <see cref="AttachedSkills"/>.</summary>
    public OpenTail.Stingray.Core.Tools.IToolProvider? ToolProvider { get; set; }

    /// <summary>Context passed to <see cref="ToolProvider"/> when listing permitted tools.</summary>
    public OpenTail.Stingray.Core.Tools.InferenceToolContext? ToolContext { get; set; }

    /// <summary>
    /// Optional tool-call detector run by a caller against this session's generated tokens.
    /// <see cref="HotSession"/> does not invoke this itself — chunk-based tool-call detection
    /// already happens independently at the Server layer (see docs/032) — this exists so callers
    /// that still want session-scoped detection have somewhere to store the delegate.
    /// </summary>
    public Func<IReadOnlyList<int>, IReadOnlyList<OpenTail.Stingray.Core.Tools.ToolCall>?>? ToolCallParser { get; set; }

    /// <summary>Read-only list of skills currently attached to the session.</summary>
    public IReadOnlyList<OpenTail.Stingray.Core.ISkill> AttachedSkills => _attachedSkills;

    /// <summary>
    /// Attaches a native declarative skill package to the session. A skill's <c>Tools</c> are
    /// authorized immediately (<see cref="ValidateToolCall"/>). A skill's <c>Instructions</c> are
    /// queued and prepended to the append-prompt text of the NEXT <see cref="RunTurnAsync"/> call
    /// only, then cleared — deliberately not retroactive: a turn already committed to the KV cache
    /// cannot be rewritten to have "seen" a skill attached afterward, so this only ever affects
    /// what the session generates going forward, never its existing history.
    /// </summary>
    public void AttachSkill(OpenTail.Stingray.Core.ISkill skill)
    {
        ArgumentNullException.ThrowIfNull(skill);
        _attachedSkills.Add(skill);
        foreach (var instruction in skill.Instructions)
        {
            if (!string.IsNullOrEmpty(instruction.Content))
                _pendingInstructionPreamble.Add(instruction.Content);
        }
    }

    /// <summary>Detaches a skill from the session by name. Returns true if a matching skill was found and removed.</summary>
    public bool DetachSkill(string skillName)
    {
        ArgumentNullException.ThrowIfNull(skillName);
        for (int i = 0; i < _attachedSkills.Count; i++)
        {
            if (string.Equals(_attachedSkills[i].Name, skillName, StringComparison.OrdinalIgnoreCase))
            {
                _attachedSkills.RemoveAt(i);
                return true;
            }
        }
        return false;
    }

    /// <summary>
    /// Whether <paramref name="call"/> names a tool exposed by an attached skill or by
    /// <see cref="ToolProvider"/> — the same two-source check <c>InferenceSession.ValidateToolCall</c>
    /// used to perform.
    /// </summary>
    public bool ValidateToolCall(OpenTail.Stingray.Core.Tools.ToolCall call)
    {
        ArgumentNullException.ThrowIfNull(call);
        foreach (var skill in _attachedSkills)
        {
            foreach (var tool in skill.Tools)
            {
                if (string.Equals(tool.Name, call.Name, StringComparison.OrdinalIgnoreCase))
                    return true;
            }
        }

        if (ToolProvider is null) return false;

        foreach (var t in ToolProvider.GetTools(ToolContext))
        {
            if (string.Equals(t.Name, call.Name, StringComparison.OrdinalIgnoreCase))
                return true;
        }
        return false;
    }

    /// <summary>
    /// Fires once per generated token after a turn commits, mirroring
    /// <c>IInferenceSession.OnTokenGenerated</c>'s per-token notification — but after the whole
    /// turn rather than mid-stream, since <see cref="RunTurnAsync"/> only publishes tokens once
    /// they're part of a committed <see cref="HotSessionTurnResult"/>. A caller that needs true
    /// mid-generation notification should consume the chunk stream the Server layer already
    /// exposes instead (see docs/032). Exceptions thrown by a listener are isolated exactly like
    /// the old event did — one bad listener must never break the turn that already committed.
    /// </summary>
    public event Action<int, string>? OnTokenGenerated;

    /// <summary>Whether this session holds a retained cache and has no turn queued or active —
    /// i.e. whether <see cref="EvictRetainedCacheIfIdle"/> could free anything right now. Racy by
    /// nature (see <see cref="RetainedSequenceState.IsReclaimable"/>); a caller iterating
    /// candidates should call the evict method directly rather than branching on this first.</summary>
    internal bool IsIdle => _state.IsReclaimable;

    /// <summary>Whether this session currently holds no retained cache — either never prefilled, or
    /// evicted by <see cref="SuspendAsync"/>/<see cref="HotSessionRuntime"/>'s own idle reclaim.</summary>
    public bool IsSuspended => !_state.HasRetainedState;

    /// <summary>
    /// Explicit counterpart to <c>InferenceSession.SuspendAsync</c>: releases this session's
    /// retained KV cache now rather than waiting for <see cref="HotSessionRuntime"/>'s own
    /// pressure-driven idle reclaim (docs/028 Phase 1) to get to it. The cursor is unaffected —
    /// this only frees the physical cache, exactly like <see cref="EvictRetainedCacheIfIdle"/>,
    /// which this wraps. Throws if a turn is currently queued or active on this session (unlike the
    /// internal reclaim path, which silently no-ops instead, since an explicit caller-requested
    /// suspend should surface that it didn't happen rather than pretend it did).
    /// </summary>
    public Task SuspendAsync(CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        cancellationToken.ThrowIfCancellationRequested();
        if (_state.IsInUse)
            throw new InvalidOperationException("Cannot suspend a session with a turn currently queued or active.");
        EvictRetainedCacheIfIdle();
        return Task.CompletedTask;
    }

    /// <summary>
    /// Explicit counterpart to <c>InferenceSession.ResumeAsync</c>. Unlike the old architecture,
    /// <see cref="HotSession"/> has no separate "resumed" state to transition into — a suspended
    /// session's next <see cref="RunTurnAsync"/> call re-prefills from <see cref="Cursor"/>
    /// automatically, exactly like a session that was never prefilled in the first place. This
    /// method exists purely for API symmetry with the old explicit suspend/resume pair; it performs
    /// no work.
    /// </summary>
    public Task ResumeAsync(CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        cancellationToken.ThrowIfCancellationRequested();
        return Task.CompletedTask;
    }

    /// <summary>
    /// Drops this session's retained cache if it is currently idle, reconciling
    /// <see cref="SessionResourceBudget"/>'s accounting to match. The session itself is
    /// unaffected — its next turn simply starts cold instead of resuming hot. Returns the number
    /// of bytes freed (0 if the session was in use or had nothing retained).
    /// </summary>
    internal long EvictRetainedCacheIfIdle()
    {
        int freedPositions = _state.EvictIfIdle();
        if (freedPositions == 0) return 0;
        _resources.SetResidentBytes(SessionId, 0);
        return checked(freedPositions * _kvBytesPerToken);
    }

    /// <summary>
    /// docs/028 Phase 2, source side: attempts to fork up to <paramref name="maxPrefixLength"/>
    /// leading positions of this session's retained cache for a new sibling to seed from. Requires
    /// this session to be idle at the moment of the call (enforced atomically inside
    /// <see cref="RetainedSequenceState.TryForkSharedPrefix"/>, not by a separate check here — see
    /// that method for why); returns null on any reason sharing isn't possible right now, which a
    /// caller must always tolerate by falling back to an ordinary cold session.
    /// </summary>
    internal (ISequenceKvCache Cache, int Length)? TryForkSharedPrefixCache(int maxPrefixLength) =>
        _engine.TryForkSharedPrefix(_state, maxPrefixLength);

    /// <summary>
    /// docs/028 Phase 2, destination side: seeds this brand-new session's retained state and
    /// cursor from a prefix forked off a sibling (<see cref="TryForkSharedPrefixCache"/>), so this
    /// session's first <see cref="RunTurnAsync"/> call only has to prefill whatever comes after
    /// <paramref name="prefixTokens"/> — the same append-only contract every later turn already
    /// follows, just starting from a non-empty cursor instead of an empty one. Only valid before
    /// this session's first turn; see <see cref="RetainedSequenceState.SeedWithForkedCache"/>.
    /// </summary>
    internal void SeedFromSharedPrefix(ImmutableArray<int> prefixTokens, ISequenceKvCache forkedCache)
    {
        _state.SeedWithForkedCache(forkedCache, prefixTokens.Length);
        var log = ImmutableArray.Create<ExecutionSegment>(new TokenSegment(prefixTokens));
        var seededCursor = new SessionCursor(log, prefixTokens.Length, prefixTokens.Length,
            prefixTokens.Length, prefixTokens.Length, StateCoverage.Full);
        lock (_cursorGate) _cursor = seededCursor;
        _resources.SetResidentBytes(SessionId, checked((long)prefixTokens.Length * _kvBytesPerToken));
    }

    /// <summary>
    /// Test-only hook invoked immediately before the turn is committed, so a test can fault the
    /// turn at the one point where <see cref="CompensateUncommittedTurn"/> executes its full body.
    /// Always null in production. See the call site for why a seam is required at all.
    /// </summary>
    internal Action? FaultBeforeCommitForTests { get; set; }

    public SessionId SessionId { get; }

    /// <summary>
    /// The cursor's accepted-position count, exposed under its own name.
    ///
    /// <para><b>This is not the optimistic-concurrency token</b>, and it used to be called
    /// <c>CommittedRevision</c>, which is exactly how it came to be published as one. The token is
    /// <c>GetSessionSnapshot(id).CommittedRevision</c> — the store's per-turn counter, advanced by
    /// <c>.Next()</c> per completed turn, and the value <see cref="RunTurnAsync"/> validates
    /// <c>expectedRevision</c> against. The two diverge the moment a turn accepts more than one
    /// position, and publishing this one made the server advertise a revision it would then
    /// reject.</para>
    ///
    /// <para>The snapshot route also works for cold-restored sessions; this property does not,
    /// because a restored session is not in the hot store.</para>
    /// </summary>
    public long AcceptedPositionCount => Cursor.AcceptedPositionCount;

    public SessionCursor Cursor
    {
        get { lock (_cursorGate) return _cursor; }
    }

    /// <summary>
    /// Explains whether a requested execution history is an exact append to this hot session.
    /// The result does not perform a rewind or replay; <see cref="RunTurnAsync"/> remains
    /// append-only in this milestone.
    /// </summary>
    public SessionContinuationDiagnostic DiagnoseContinuation(ImmutableArray<ExecutionSegment> target) =>
        ExecutionReconciler.Diagnose(Cursor, target);

    /// <summary>
    /// Captures the session's current cursor and committed revision so a later
    /// <see cref="RollbackAsync"/> can restore exactly this point — the <see cref="HotSession"/>
    /// counterpart to <c>InferenceSession.CreateCheckpoint</c>/<c>.Rollback</c>. Unlike the internal
    /// single-turn undo <see cref="RunTurnAsync"/> uses on its own failure paths, this can restore
    /// to ANY earlier checkpoint, not just the immediately preceding turn, the same way the old
    /// capability did. Must be called between turns (no active <see cref="RunTurnAsync"/> call on
    /// this session).
    /// </summary>
    public HotSessionCheckpoint CreateCheckpoint()
    {
        ThrowIfDisposed();
        var cursor = Cursor;
        var revision = _store.Open(SessionId).CommittedRevision;
        return new HotSessionCheckpoint(SessionId, cursor, revision);
    }

    /// <summary>
    /// Restores this session to a previously captured <see cref="HotSessionCheckpoint"/>: rewinds
    /// the retained KV cache to the checkpoint's materialized position, resets the cursor, and
    /// resets the store's revision counter so a subsequent <see cref="RunTurnAsync"/> call can
    /// supply the checkpoint's revision as <c>expectedRevision</c>. Throws if the checkpoint belongs
    /// to a different session, a turn is currently active, or the retained cache cannot rewind that
    /// far (e.g. it was evicted since the checkpoint was taken — the same "best-effort, not durable"
    /// character every other in-memory-only HotSession capability has).
    /// </summary>
    public async Task RollbackAsync(HotSessionCheckpoint checkpoint)
    {
        ArgumentNullException.ThrowIfNull(checkpoint);
        if (checkpoint.SessionId != SessionId)
            throw new ArgumentException(
                $"Checkpoint belongs to session '{checkpoint.SessionId}', not '{SessionId}'.", nameof(checkpoint));

        await _turnGate.WaitAsync().ConfigureAwait(false);
        try
        {
            ThrowIfDisposed();
            _state.RollbackTo(checkpoint.Cursor.MaterializedPositionCount);
            lock (_cursorGate) _cursor = checkpoint.Cursor;
            _resources.SetResidentBytes(SessionId, CurrentResidentBytes());
            _store.SetRevision(SessionId, checkpoint.CommittedRevision);
        }
        finally
        {
            _turnGate.Release();
        }
    }

    public async Task<HotSessionTurnResult> RunTurnAsync(
        string appendPrompt,
        SamplingParams sampling,
        SessionRevision expectedRevision,
        SessionOperationId operationId,
        SessionRequestDigest requestDigest,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(appendPrompt);
        ThrowIfDisposed();
        // Any skill instructions queued since the last turn (see AttachSkill) prepend to exactly
        // this call's text, once, then clear — a raw append-only session has no separate "system"
        // slot, so this is the only place instruction text can enter the model's input at all.
        string effectivePrompt = appendPrompt;
        if (_pendingInstructionPreamble.Count > 0)
        {
            effectivePrompt = string.Join("\n\n", _pendingInstructionPreamble) + "\n\n" + appendPrompt;
            _pendingInstructionPreamble.Clear();
        }
        var appendTokens = _tokenizer.Encode(effectivePrompt).ToImmutableArray();
        if (appendTokens.IsDefaultOrEmpty)
            throw new ArgumentException("An append-only session turn requires at least one input token.", nameof(appendPrompt));

        await _turnGate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            ThrowIfDisposed();
            var lease = _store.AcquireLease(SessionId);
            var operation = _store.Begin(lease, expectedRevision, operationId, requestDigest);
            if (operation.State != SessionOperationState.Accepted)
            {
                var replayedChunks = operation.ResultChunks?.ToImmutableArray() ?? [];
                var (replayReason, replayToolCalls) = HotSessionTurnResult.DescribeOutcome(
                    replayedChunks, cancelled: false, failed: operation.State == SessionOperationState.Failed);
                return new HotSessionTurnResult(operation, Cursor, replayedChunks, IsIdempotentReplay: true,
                    replayReason, replayToolCalls);
            }

            var priorCursor = Cursor;
            int projectedPositions = checked(priorCursor.MaterializedPositionCount + appendTokens.Length
                + Math.Max(0, sampling.MaxNewTokens));
            if (projectedPositions > _maxSequenceLength)
            {
                var ex = new ArgumentOutOfRangeException(nameof(appendPrompt),
                    $"Projected sequence length {projectedPositions} exceeds backend maximum {_maxSequenceLength}.");
                _store.Fail(lease, operationId, ex.Message);
                throw ex;
            }
            if (sampling.MaxNewTokens > _maxCapturedOutputChunks - 4)
            {
                var ex = new ArgumentOutOfRangeException(nameof(sampling),
                    $"MaxNewTokens exceeds the configured captured-output limit of {_maxCapturedOutputChunks} chunks.");
                _store.Fail(lease, operationId, ex.Message);
                throw ex;
            }

            SessionResourceBudget.SessionResourceReservation reservation;
            int initialPositions = checked(priorCursor.MaterializedPositionCount + appendTokens.Length
                + (sampling.MaxNewTokens > 0 ? 1 : 0));
            long neededBytes = checked((long)initialPositions * _kvBytesPerToken);
            try
            {
                reservation = _resources.Reserve(SessionId, neededBytes, _modelKey);
            }
            catch (SessionResourceBudgetExceededException ex)
            {
                // The budget is exhausted, but not necessarily by work that's actually running:
                // idle, retained sibling sessions hold committed bytes SessionResourceBudget has
                // no way to reclaim on its own (docs/028 Phase 1). Evict idle siblings once, then
                // retry exactly once before failing for real.
                if (_reclaimIdleBytes is null || _reclaimIdleBytes(SessionId, neededBytes) <= 0)
                {
                    _store.Fail(lease, operationId, ex.Message);
                    throw;
                }
                try
                {
                    reservation = _resources.Reserve(SessionId, neededBytes, _modelKey);
                }
                catch (SessionResourceBudgetExceededException retryEx)
                {
                    _store.Fail(lease, operationId, retryEx.Message);
                    throw;
                }
            }

            using (reservation)
            {
                _store.Transition(lease, operationId, SessionOperationState.Accepted, SessionOperationState.Prefilling);
                _store.Transition(lease, operationId, SessionOperationState.Prefilling, SessionOperationState.Generating);
                var chunks = ImmutableArray.CreateBuilder<GenerateChunk>();
                bool generationCompleted = false;
                bool resourcesFinalized = false;
                bool cursorPublished = false;
                bool operationCommitted = false;
                try
                {
                    var turnClock = System.Diagnostics.Stopwatch.StartNew();
                    TimeSpan prefillElapsed = TimeSpan.Zero;
                    bool firstChunkSeen = false;
                    await foreach (var chunk in _engine.GenerateRetainedChunksAsync(
                        effectivePrompt, sampling, _state, cancellationToken,
                        targetPositions => reservation.TryRenew(checked((long)targetPositions * _kvBytesPerToken))).ConfigureAwait(false))
                    {
                        if (!firstChunkSeen)
                        {
                            // The engine performs prefill before emitting its first chunk, so time to
                            // first chunk is the closest approximation to prefill latency available
                            // from this call site without deeper engine instrumentation.
                            prefillElapsed = turnClock.Elapsed;
                            firstChunkSeen = true;
                        }
                        chunks.Add(chunk);
                    }
                    generationCompleted = true;
                    TimeSpan generationElapsed = turnClock.Elapsed - prefillElapsed;

                    var outcome = _state.LastTurn
                        ?? throw new InvalidOperationException("The retained engine completed without a retained turn outcome.");
                    // Only successfully committed turns count toward ISessionMetrics (its own doc
                    // says "successfully committed") -- the cancel/fail branches below deliberately
                    // do not record partial metrics.
                    _metrics.AddPromptTokens(appendTokens.Length, prefillElapsed);
                    _metrics.AddGeneratedTokens(outcome.GeneratedTokenIds.Length, generationElapsed);
                    var nextCursor = BuildNextCursor(appendTokens, outcome);
                    _store.Transition(lease, operationId, SessionOperationState.Generating, SessionOperationState.CommitPrepared);
                    var resultChunks = chunks.ToImmutable();
                    reservation.Complete(CurrentResidentBytes());
                    resourcesFinalized = true;
                    lock (_cursorGate) _cursor = nextCursor;
                    cursorPublished = true;
                    // Test-only fault injection. Null in every production path, and deliberately
                    // sited here — after generationCompleted, cursorPublished and resourcesFinalized
                    // are all set — because that is the only state in which CompensateUncommittedTurn
                    // runs its full body, including RollbackLastTurn.
                    //
                    // The seam exists because the compensation path was otherwise UNREACHABLE from a
                    // test: every fault point between generation and commit is a concrete type
                    // (InMemorySessionStore, the reservation), and a token cancelled during
                    // generation throws while generationCompleted is still false, which skips the
                    // rollback branch. So the recovery path that runs only after a turn has already
                    // failed had never executed. One nulled-out delegate buys real coverage of it;
                    // see docs/sessions-release-gate-matrix.md.
                    FaultBeforeCommitForTests?.Invoke();
                    // This is deliberately the final stateful operation. If an earlier step
                    // faults, the catch path can still restore cache, cursor and accounting.
                    var completed = _store.Complete(lease, operationId, resultChunks);
                    operationCommitted = true;
                    var (reason, toolCalls) = HotSessionTurnResult.DescribeOutcome(resultChunks, cancelled: false, failed: false);
                    NotifyTokenListeners(outcome.GeneratedTokenIds);
                    return new HotSessionTurnResult(completed, nextCursor, resultChunks, IsIdempotentReplay: false, reason, toolCalls);
                }
                catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
                {
                    await WaitForStateReleaseAsync().ConfigureAwait(false);
                    if (operationCommitted) throw;
                    CompensateUncommittedTurn(generationCompleted, cursorPublished, priorCursor, resourcesFinalized);
                    var cancelled = _store.Cancel(lease, operationId);
                    var cancelledChunks = chunks.ToImmutable();
                    var (cancelReason, cancelToolCalls) = HotSessionTurnResult.DescribeOutcome(cancelledChunks, cancelled: true, failed: false);
                    return new HotSessionTurnResult(cancelled, Cursor, cancelledChunks, IsIdempotentReplay: false, cancelReason, cancelToolCalls);
                }
                catch (Exception ex)
                {
                    await WaitForStateReleaseAsync().ConfigureAwait(false);
                    if (operationCommitted) throw;
                    CompensateUncommittedTurn(generationCompleted, cursorPublished, priorCursor, resourcesFinalized);
                    var failed = _store.Fail(lease, operationId, ex.Message);
                    var failedChunks = chunks.ToImmutable();
                    var (failReason, failToolCalls) = HotSessionTurnResult.DescribeOutcome(failedChunks, cancelled: false, failed: true);
                    return new HotSessionTurnResult(failed, Cursor, failedChunks, IsIdempotentReplay: false, failReason, failToolCalls);
                }
            }
        }
        finally
        {
            _turnGate.Release();
        }
    }

    private SessionCursor BuildNextCursor(ImmutableArray<int> appendTokens, RetainedTurnOutcome outcome)
    {
        var prior = Cursor;
        if (outcome.TurnStartPosition != prior.MaterializedPositionCount)
            throw new InvalidOperationException(
                "Retained cache turn start does not match the committed execution cursor.");

        var log = prior.ExecutionLog.Add(new TokenSegment(appendTokens));
        if (!outcome.GeneratedTokenIds.IsDefaultOrEmpty)
            log = log.Add(new TokenSegment(outcome.GeneratedTokenIds));
        int accepted = checked(prior.AcceptedPositionCount + appendTokens.Length + outcome.GeneratedTokenIds.Length);
        if (outcome.MaterializedPosition != accepted)
            throw new InvalidOperationException(
                "Retained cache materialized position does not match the exact accepted execution log.");
        return new SessionCursor(log, accepted, accepted, accepted, accepted, StateCoverage.Full);
    }

    private void NotifyTokenListeners(ImmutableArray<int> generatedTokenIds)
    {
        var handler = OnTokenGenerated;
        if (handler is null || generatedTokenIds.IsDefaultOrEmpty) return;
        foreach (int tokenId in generatedTokenIds)
        {
            try { handler(tokenId, _tokenizer.Decode([tokenId])); }
            catch { /* Isolate host listener exceptions, matching InferenceSession's own behavior. */ }
        }
    }

    private async Task WaitForStateReleaseAsync()
    {
        while (_state.IsInUse)
            await Task.Delay(1).ConfigureAwait(false);
    }

    private long CurrentResidentBytes() =>
        _state.HasRetainedState ? checked((long)_state.MaterializedPosition * _kvBytesPerToken) : 0;

    private void CompensateUncommittedTurn(
        bool generationCompleted,
        bool cursorPublished,
        SessionCursor priorCursor,
        bool resourcesFinalized)
    {
        if (generationCompleted)
        {
            try { _state.RollbackLastTurn(); }
            catch { /* A failed rollback discards the cache and allows a fresh state on the next turn. */ }
        }
        if (cursorPublished)
            lock (_cursorGate) _cursor = priorCursor;
        if (resourcesFinalized)
            _resources.SetResidentBytes(SessionId, CurrentResidentBytes());
    }

    private void ThrowIfDisposed()
    {
        lock (_lifecycleGate)
            ObjectDisposedException.ThrowIf(_disposed, this);
    }

    public void Dispose()
    {
        lock (_lifecycleGate)
        {
            if (_disposed) return;
            _disposed = true;
        }
        _turnGate.WaitAsync().GetAwaiter().GetResult();
        try
        {
            _state.Dispose();
            _resources.Remove(SessionId);
            _store.Delete(SessionId);
            _onDisposed?.Invoke(SessionId, this);
        }
        finally
        {
            _turnGate.Release();
            _turnGate.Dispose();
        }
    }

    /// <summary>Exports canonical active session state into a versioned binary envelope (Milestone 2).</summary>
    public byte[] ExportState(string modelFingerprint = "default", SessionCursorCodecLimits? limits = null,
        ModelFormat modelFormat = ModelFormat.Gguf)
    {
        ThrowIfDisposed();
        var abi = new SessionStateABI(modelFingerprint, _kvBytesPerToken, _maxSequenceLength, modelFormat);
        var compatKey = SessionStateCodec.ComputeCompatibilityKey(abi);
        var cursorEnvelope = new SessionCursorEnvelope(Cursor, []);
        var payloadHash = StatePayloadHash.Compute(Encoding.UTF8.GetBytes(Cursor.InputIdentity.Value.Value));

        var envelope = new SessionStateEnvelope(SessionId, cursorEnvelope, abi, compatKey, payloadHash, []);
        return SessionStateCodec.Encode(envelope, limits);
    }

    /// <summary>
    /// Physical KV pages for this session, or <c>null</c> when the cache cannot export them.
    /// </summary>
    /// <remarks>
    /// <para><b>Deliberately NOT carried inside <see cref="ExportState"/>.</b> KV rode in a
    /// <c>KvCacheData</c> optional section briefly and could not work there:
    /// <c>SessionCursorCodecLimits.MaxPayloadBytes</c> caps an envelope at <b>4 MB</b>, while one
    /// block of a 24-layer 2048-wide cache is 6 MB — so roughly <b>11 tokens</b> exhausted the limit.
    /// <c>SessionCursorCodec</c> also states outright that it is not a KV-cache codec, and stuffing
    /// pages through it was a layering violation as well as a size failure.</para>
    ///
    /// <para>KV is bulk binary with its own framing and belongs in segment packs, which is what
    /// <c>SegmentPackStore</c> and the manifest's block list were built for. Callers export the
    /// cursor envelope and the pages separately and store them side by side.</para>
    /// </remarks>
    public byte[]? ExportKvBytes()
    {
        ThrowIfDisposed();
        return _state.ExportKvBytes();
    }

    /// <summary>True while a turn holds this session's retained state.</summary>
    internal bool IsInUse => _state.IsInUse;

    /// <summary>
    /// Restores cursor state and seeds the store's revision.
    ///
    /// <para><paramref name="committedRevision"/> is the persisted turn counter from the manifest and
    /// is the value to seed with. Seeding from <c>cursor.AcceptedPositionCount</c> instead — which is
    /// what this used to do unconditionally, and still does when no persisted revision is supplied —
    /// made a restored session's revision a POSITION count while a live session's was a TURN count.
    /// The two lanes then meant different things by <c>committed_revision</c>, which is precisely the
    /// disagreement the manifest-v3 change exists to end.</para>
    /// </summary>
    internal void RestoreCursor(SessionCursor cursor, SessionRevision? committedRevision = null)
    {
        lock (_cursorGate)
        {
            _cursor = cursor;
        }
        _state.SetMaterializedPosition(cursor.MaterializedPositionCount);
        _store.SetRevision(SessionId, committedRevision ?? new SessionRevision(cursor.AcceptedPositionCount));
    }

    internal void RestoreKvBytes(byte[] kvBytes)
    {
        _state.RestoreKvBytes(kvBytes);
    }

    internal void RestoreCompletedOperations(IReadOnlyCollection<SessionOperationSnapshot> operations) =>
        _store.RestoreCompletedOperations(SessionId, operations);
}

/// <summary>Factory for hot sessions sharing one engine and one in-memory operation ledger.</summary>
public sealed class HotSessionRuntime
{
    private readonly ContinuousBatchingEngine _engine;
    private readonly ITokenizer _tokenizer;
    private readonly InMemorySessionStore _store;
    private readonly SessionResourceBudget _resources;
    private readonly HotSessionRuntimeOptions _options;
    private readonly ConcurrentDictionary<SessionId, HotSession> _sessions = [];

    public HotSessionRuntime(ContinuousBatchingEngine engine, ITokenizer tokenizer, HotSessionRuntimeOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(engine);
        ArgumentNullException.ThrowIfNull(tokenizer);
        _engine = engine;
        _tokenizer = tokenizer;
        _options = options ?? new HotSessionRuntimeOptions();
        _store = new InMemorySessionStore(_options.MaxCompletedOperationRecords,
            _options.CompletedOperationRetention);
        _resources = new SessionResourceBudget(_options);
    }

    public long ResidentBytes => _resources.ResidentBytes;

    public void SetModelBudget(string modelKey, long maxBytes) => _resources.SetModelBudget(modelKey, maxBytes);

    public long GetModelResidentBytes(string modelKey) => _resources.GetModelResidentBytes(modelKey);

    public HotSession Create(SessionId? sessionId = null, string? modelKey = null) =>
        Create(sessionId, modelKey, parentSessionId: null);

    private HotSession Create(SessionId? sessionId, string? modelKey, SessionId? parentSessionId)
    {
        var snapshot = _store.Create(sessionId);
        var session = new HotSession(_engine, _tokenizer, _store, _resources,
            _engine.KvBytesPerToken, _engine.MaxSequenceLength,
            _options.MaxCapturedOutputChunks, snapshot.SessionId, RemoveDisposedSession, modelKey,
            ReclaimIdleBytes, runtime: this)
        { ParentSessionId = parentSessionId };
        if (!_sessions.TryAdd(snapshot.SessionId, session))
        {
            session.Dispose();
            throw new InvalidOperationException($"Session '{snapshot.SessionId}' is already active.");
        }
        return session;
    }

    public HotSession Create(SessionAddress address) => Create(address.ToSessionId(), address.ModelFingerprint);

    /// <summary>Looks up an active session without throwing, for internal topology queries (<see cref="HotSessionTree"/>).</summary>
    internal bool TryGetSession(SessionId sessionId, out HotSession session) => _sessions.TryGetValue(sessionId, out session!);

    /// <summary>Active child session IDs whose <see cref="HotSession.ParentSessionId"/> is <paramref name="parentId"/>.</summary>
    internal IReadOnlyList<SessionId> GetChildSessionIds(SessionId parentId)
    {
        List<SessionId>? children = null;
        foreach (var (id, session) in _sessions)
        {
            if (session.ParentSessionId == parentId)
                (children ??= []).Add(id);
        }
        return (IReadOnlyList<SessionId>?)children ?? [];
    }

    /// <summary>
    /// Aggregates <see cref="ISessionMetrics"/> across <paramref name="rootId"/> and every currently
    /// active descendant in its fork subtree — a point-in-time computation over <see cref="_sessions"/>,
    /// not a live-tracked total, matching <see cref="ISessionTree.CumulativeTreeMetrics"/>'s own
    /// "snapshot" wording.
    /// </summary>
    internal SessionMetricsSnapshot AggregateTreeMetrics(SessionId rootId)
    {
        long promptTokens = 0, generatedTokens = 0, prefillTicks = 0, generationTicks = 0;
        int kvPagesHeld = 0;

        void Accumulate(SessionId id)
        {
            if (!_sessions.TryGetValue(id, out var session)) return;
            var m = session.Metrics;
            promptTokens += m.PromptTokens;
            generatedTokens += m.GeneratedTokens;
            prefillTicks += m.TotalPrefillTime.Ticks;
            generationTicks += m.TotalGenerationTime.Ticks;
            kvPagesHeld += m.KvPagesHeld;
            foreach (var childId in GetChildSessionIds(id)) Accumulate(childId);
        }
        Accumulate(rootId);

        var generationTime = TimeSpan.FromTicks(generationTicks);
        double tokensPerSecond = generatedTokens > 0 && generationTime.TotalSeconds > 0
            ? generatedTokens / generationTime.TotalSeconds : 0.0;
        return new SessionMetricsSnapshot(promptTokens, generatedTokens,
            TimeSpan.FromTicks(prefillTicks), generationTime, tokensPerSecond, kvPagesHeld);
    }

    /// <summary>
    /// docs/028 Phase 2: creates a new session pre-seeded from the longest matching prefix among
    /// currently idle sibling sessions on this runtime, when the active backend supports prefix
    /// forking (<see cref="HotSession.TryForkSharedPrefixCache"/>) — e.g. many chat sessions
    /// opening with an identical system prompt reuse its already-computed KV pages instead of
    /// each re-prefilling it from nothing.
    ///
    /// <para>Best-effort: any reason sharing doesn't happen (no idle sibling shares a prefix, the
    /// backend doesn't support forking, the seeded size is rejected by <see cref="SessionResourceBudget"/>)
    /// silently falls back to an ordinary cold session, exactly what <see cref="Create(SessionId?, string?)"/>
    /// alone would have produced — this must never be the reason session creation itself fails.</para>
    ///
    /// <para>Returns the session together with how many leading tokens of
    /// <paramref name="desiredPromptTokens"/> were actually seeded (0 if none). The caller's first
    /// <see cref="HotSession.RunTurnAsync"/> call must submit only the remaining suffix as its
    /// append prompt — the same append-only contract every later turn already follows. Token-level
    /// prefix matching is only guaranteed correct when <paramref name="desiredPromptTokens"/> was
    /// produced by encoding the shared text in isolation (e.g. every session's first turn is
    /// literally the same system-prompt string) — BPE merge boundaries mean re-encoding a longer,
    /// merged string is not guaranteed to reproduce the same leading token IDs.</para>
    /// </summary>
    public (HotSession Session, int SeededPrefixLength) CreateWithSharedPrefixHint(
        ImmutableArray<int> desiredPromptTokens, SessionId? sessionId = null, string? modelKey = null)
    {
        var session = Create(sessionId, modelKey);
        if (desiredPromptTokens.IsDefaultOrEmpty) return (session, 0);

        HotSession? bestSibling = null;
        int bestMatch = 0;
        foreach (var (id, sibling) in _sessions)
        {
            if (id == session.SessionId || !sibling.IsIdle) continue;
            int match = CommonPrefixLength(sibling.Cursor.ExecutionLog, desiredPromptTokens);
            if (match > bestMatch)
            {
                bestMatch = match;
                bestSibling = sibling;
            }
        }
        if (bestSibling is null || bestMatch == 0) return (session, 0);

        var forked = bestSibling.TryForkSharedPrefixCache(bestMatch);
        if (forked is null) return (session, 0);

        try
        {
            session.SeedFromSharedPrefix(desiredPromptTokens[..forked.Value.Length], forked.Value.Cache);
            return (session, forked.Value.Length);
        }
        catch
        {
            forked.Value.Cache.Dispose();
            return (session, 0);
        }
    }

    /// <summary>
    /// Longest run of leading tokens <paramref name="desired"/> shares with <paramref name="log"/>,
    /// stopping at the first mismatch or the first non-token (atomic) segment — a fork can only
    /// share KV positions that came from actual token execution.
    /// </summary>
    private static int CommonPrefixLength(ImmutableArray<ExecutionSegment> log, ImmutableArray<int> desired)
    {
        int matched = 0;
        foreach (var segment in log)
        {
            if (matched >= desired.Length) break;
            if (segment is not TokenSegment tokenSegment) break;
            foreach (var token in tokenSegment.TokenIds)
            {
                if (matched >= desired.Length || token != desired[matched]) return matched;
                matched++;
            }
        }
        return matched;
    }

    /// <summary>
    /// docs/028 Phase 3: creates <paramref name="count"/> independent branches from
    /// <paramref name="parent"/>'s current retained state, sharing its physical KV pages
    /// zero-copy (the same <see cref="HotSession.TryForkSharedPrefixCache"/> ref-counted
    /// mechanism Phase 2's cross-session sharing uses) instead of re-prefilling or copying
    /// tensors. Copy-on-write happens automatically at the existing cache layer the moment a
    /// branch's own generation writes to a still-shared page — nothing here needs to know about
    /// that.
    ///
    /// <para><b>Page-aligned boundary.</b> The underlying fork can only share whole pages, at
    /// whatever block size the active backend reports — deliberately not assumed here, the same
    /// way <see cref="CreateWithSharedPrefixHint"/> defers to it rather than hard-coding a
    /// concrete cache type's alignment into this layer. A parent not currently sitting on that
    /// boundary shares everything up to its last full page; a branch's returned cursor reflects
    /// that (possibly shorter) shared length, not the parent's exact current position. The
    /// caller's next <see cref="RunTurnAsync"/> call on a branch supplies whatever text covers the
    /// remaining tail — the same "encode the shared portion in isolation" contract
    /// <see cref="CreateWithSharedPrefixHint"/> already documents, not a new caveat. The shared
    /// prefix also stops at the first non-token (atomic) execution segment in the parent's
    /// history, for the same reason <see cref="CommonPrefixLength"/> does: a fork can only share
    /// KV positions that came from actual token execution.</para>
    ///
    /// <para><b>Atomic</b>, unlike <see cref="CreateWithSharedPrefixHint"/>'s best-effort
    /// fallback: this call either produces all <paramref name="count"/> branches genuinely sharing
    /// the parent's state, or none. <paramref name="parent"/> must be idle (no queued or active
    /// turn); if it becomes busy partway through this call (a caller racing its own
    /// <see cref="RunTurnAsync"/> against its own <see cref="Fork"/> call on the same session is a
    /// caller bug, not something to paper over) or the active backend doesn't support prefix
    /// forking at all, every branch already created by this call is torn down — its reservation
    /// and pages released — before the exception propagates. A caller never has to distinguish
    /// "some branches exist" from "the call failed."</para>
    /// </summary>
    public IReadOnlyList<HotSession> Fork(HotSession parent, int count)
    {
        ArgumentNullException.ThrowIfNull(parent);
        if (count < 1)
            throw new ArgumentOutOfRangeException(nameof(count), "Fork count must be at least 1.");

        // The cap here is the token-only prefix length (stops at any atomic segment) -- a
        // session-layer concept the engine-level fork has no way to know about. The engine still
        // floors this to its own backend-reported block size internally; that authoritative,
        // possibly-smaller result (forked.Length below) is what actually gets shared, not this cap.
        var (tokens, tokenOnlyLength) = TokenOnlyPrefix(
            parent.Cursor.ExecutionLog, parent.Cursor.MaterializedPositionCount);

        var branches = new List<HotSession>(count);
        try
        {
            ImmutableArray<int>? sharedTokens = null;
            for (int i = 0; i < count; i++)
            {
                var branch = Create(sessionId: null, modelKey: null, parentSessionId: parent.SessionId);
                branches.Add(branch);
                foreach (var kv in parent.Metadata.GetEntries()) branch.Metadata.Set(kv.Key, kv.Value);
                if (tokenOnlyLength == 0) continue;

                var forked = parent.TryForkSharedPrefixCache(tokenOnlyLength)
                    ?? throw new InvalidOperationException(
                        "Cannot fork session: it is not idle, or the active backend does not support prefix forking.");
                sharedTokens ??= tokens[..forked.Length];
                branch.SeedFromSharedPrefix(sharedTokens.Value, forked.Cache);
            }
            return branches;
        }
        catch
        {
            foreach (var branch in branches) Delete(branch.SessionId);
            throw;
        }
    }

    /// <summary>
    /// The leading run of <paramref name="log"/> that is plain token execution, capped at
    /// <paramref name="maxLength"/> positions — stops at the first non-token (atomic) segment
    /// even if that lands short of the cap. See <see cref="Fork"/> for why this matters: a shared
    /// prefix can only cover positions that came from actual token execution.
    /// </summary>
    private static (ImmutableArray<int> Tokens, int Length) TokenOnlyPrefix(
        ImmutableArray<ExecutionSegment> log, int maxLength)
    {
        var tokens = ImmutableArray.CreateBuilder<int>();
        foreach (var segment in log)
        {
            if (tokens.Count >= maxLength) break;
            if (segment is not TokenSegment tokenSegment) break;
            foreach (var token in tokenSegment.TokenIds)
            {
                if (tokens.Count >= maxLength) break;
                tokens.Add(token);
            }
        }
        return (tokens.ToImmutable(), tokens.Count);
    }

    /// <summary>
    /// docs/028 Phase 1's eviction policy: evict idle sibling sessions (oldest-created first, an
    /// arbitrary but stable and starvation-free order — <see cref="ConcurrentDictionary{TKey,TValue}"/>
    /// enumeration order is not insertion order, so this does not currently guarantee that; true
    /// LRU-by-last-use would need each session to track its own last-active timestamp, deferred
    /// until a real workload shows FIFO-by-creation is the wrong policy) until at least
    /// <paramref name="neededBytes"/> has been freed or every session has been tried. Returns the
    /// actual total reclaimed, which may be less than requested — the caller (the requesting
    /// session's own reservation retry) decides whether that's enough.
    /// </summary>
    private long ReclaimIdleBytes(SessionId requester, long neededBytes)
    {
        long reclaimed = 0;
        foreach (var (id, session) in _sessions)
        {
            if (reclaimed >= neededBytes) break;
            if (id == requester) continue;
            reclaimed += session.EvictRetainedCacheIfIdle();
        }
        return reclaimed;
    }

    /// <summary>Opens an active in-memory hot session. Cold restoration is a later milestone.</summary>
    public HotSession Open(SessionId sessionId) =>
        _sessions.TryGetValue(sessionId, out var session)
            ? session
            : throw new SessionNotFoundException(sessionId);

    /// <summary>Opens an active in-memory hot session by multi-model routing address.</summary>
    public HotSession Open(SessionAddress address) => Open(address.ToSessionId());

    /// <summary>Imports a versioned binary session state envelope into an active hot session (Milestone 2).</summary>
    public HotSession ImportState(ReadOnlySpan<byte> stateBytes, string expectedModelFingerprint = "default",
        SessionCursorCodecLimits? limits = null, ModelFormat expectedModelFormat = ModelFormat.Gguf,
        SessionRevision? committedRevision = null)
    {
        var envelope = SessionStateCodec.Decode(stateBytes, limits);
        if (envelope.Abi.ModelFingerprint != expectedModelFingerprint)
            throw new SessionCursorFormatException($"Model fingerprint mismatch. Expected '{expectedModelFingerprint}', got '{envelope.Abi.ModelFingerprint}'.");
        if (envelope.Abi.ModelFormat != expectedModelFormat)
            throw new SessionCursorFormatException($"Model format mismatch. Expected '{expectedModelFormat}', got '{envelope.Abi.ModelFormat}'.");
        if (envelope.Abi.KvBytesPerToken != _engine.KvBytesPerToken)
            throw new SessionCursorFormatException($"KvBytesPerToken mismatch. Expected {_engine.KvBytesPerToken}, got {envelope.Abi.KvBytesPerToken}.");
        if (envelope.Abi.MaxSequenceLength != _engine.MaxSequenceLength)
            throw new SessionCursorFormatException($"MaxSequenceLength mismatch. Expected {_engine.MaxSequenceLength}, got {envelope.Abi.MaxSequenceLength}.");

        var session = Create(envelope.SessionId);
        session.RestoreCursor(envelope.CursorEnvelope.Cursor, committedRevision);
        return session;
    }

    /// <summary>
    /// Imports a cursor envelope together with out-of-band KV pages. The pages are staged on the
    /// session and loaded into a real cache when the engine next admits it.
    /// </summary>
    public HotSession ImportState(ReadOnlySpan<byte> stateBytes, ReadOnlySpan<byte> kvStateBytes,
        string expectedModelFingerprint = "default", SessionCursorCodecLimits? limits = null,
        ModelFormat expectedModelFormat = ModelFormat.Gguf, SessionRevision? committedRevision = null)
    {
        var session = ImportState(stateBytes, expectedModelFingerprint, limits, expectedModelFormat, committedRevision);
        if (kvStateBytes.Length > 0) session.RestoreKvBytes(kvStateBytes.ToArray());
        return session;
    }

    /// <summary>Disposes and removes an active hot session and its in-memory ledger entry.</summary>
    public bool Delete(SessionId sessionId)
    {
        if (!_sessions.TryRemove(sessionId, out var session)) return false;
        session.Dispose();
        return true;
    }

    /// <summary>Retrieves a point-in-time snapshot of an in-memory session.</summary>
    public SessionSnapshot GetSessionSnapshot(SessionId sessionId) => _store.Open(sessionId);

    /// <summary>
    /// Retrieves an operation record (including result chunks) for a completed or in-flight
    /// operation — the detachable-result path (§6.5): a caller that disconnected mid-turn can come
    /// back for the result instead of losing it.
    /// </summary>
    /// <exception cref="SessionNotFoundException">The session is not active in memory.</exception>
    /// <exception cref="KeyNotFoundException">
    /// No such operation. NOTE this does not distinguish "never existed" from "completed and since
    /// pruned by the retention policy" — <c>Open</c> prunes expired records, so a result fetched
    /// too late is indistinguishable from a typo'd id. Callers that need that distinction cannot
    /// get it from this API today; see the retention options on
    /// <see cref="HotSessionRuntimeOptions"/> for the window they have.
    /// </exception>
    public SessionOperationSnapshot GetOperation(SessionId sessionId, SessionOperationId operationId)
    {
        var snapshot = _store.Open(sessionId);
        foreach (var op in snapshot.Operations)
        {
            if (op.OperationId == operationId)
                return op;
        }
        throw new KeyNotFoundException(
            $"Operation '{operationId}' does not exist in session '{sessionId}' — it was never "
            + "started, or it completed and has since been pruned by the retention policy.");
    }

    /// <summary>
    /// Async form of <see cref="GetOperation"/>, matching the plan's <c>GetOperationAsync</c>
    /// surface. Retrieval is a synchronous in-memory lookup today; the async shape exists so the
    /// durable-store milestone can make it genuinely asynchronous without a breaking change.
    /// </summary>
    public Task<SessionOperationSnapshot> GetOperationAsync(
        SessionId sessionId, SessionOperationId operationId, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return Task.FromResult(GetOperation(sessionId, operationId));
    }

    private void RemoveDisposedSession(SessionId sessionId, HotSession disposed)
    {
        if (_sessions.TryGetValue(sessionId, out var current) && ReferenceEquals(current, disposed))
            _sessions.TryRemove(sessionId, out _);
    }
}

/// <summary>
/// <see cref="ISessionTree"/> for a <see cref="HotSession"/>, backed by its owning
/// <see cref="HotSessionRuntime"/>'s <c>_sessions</c> table. Computed on demand from each
/// property access rather than tracked incrementally — cheap given fork trees are small and
/// short-lived, and it means a session leaving the runtime is reflected immediately with no
/// separate teardown bookkeeping to keep in sync.
/// </summary>
internal sealed class HotSessionTree(HotSession self, HotSessionRuntime? runtime) : ISessionTree
{
    public SessionId RootId
    {
        get
        {
            if (runtime is null) return self.SessionId;
            var current = self;
            while (current.ParentSessionId is { } parentId && runtime.TryGetSession(parentId, out var parent))
                current = parent;
            return current.SessionId;
        }
    }

    public SessionId? ParentId => self.ParentSessionId;

    public IReadOnlyList<SessionId> Children => runtime?.GetChildSessionIds(self.SessionId) ?? [];

    public ISessionMetrics CumulativeTreeMetrics =>
        runtime is null ? self.Metrics : new SessionMetricsView(runtime.AggregateTreeMetrics(self.SessionId));

    /// <summary>Adapts an immutable <see cref="SessionMetricsSnapshot"/> to the live <see cref="ISessionMetrics"/> read surface.</summary>
    private sealed class SessionMetricsView(SessionMetricsSnapshot snapshot) : ISessionMetrics
    {
        public long PromptTokens => snapshot.PromptTokens;
        public long GeneratedTokens => snapshot.GeneratedTokens;
        public TimeSpan TotalPrefillTime => snapshot.TotalPrefillTime;
        public TimeSpan TotalGenerationTime => snapshot.TotalGenerationTime;
        public double TokensPerSecond => snapshot.TokensPerSecond;
        public int KvPagesHeld => snapshot.KvPagesHeld;
    }
}
