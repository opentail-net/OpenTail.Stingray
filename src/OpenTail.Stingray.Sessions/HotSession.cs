using System.Collections.Immutable;
using System.Collections.Concurrent;
using System.Text;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Sessions;

/// <summary>Outcome of one in-memory hot session turn.</summary>
public sealed record HotSessionTurnResult(
    SessionOperationSnapshot Operation,
    SessionCursor Cursor,
    ImmutableArray<GenerateChunk> Chunks,
    bool IsIdempotentReplay);

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
    private readonly RetainedSequenceState _state = new();
    private readonly object _lifecycleGate = new();
    private SessionCursor _cursor = new([], 0, 0, 0, 0, StateCoverage.Full);
    private bool _disposed;

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
        string? modelKey = null)
    {
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
        var appendTokens = _tokenizer.Encode(appendPrompt).ToImmutableArray();
        if (appendTokens.IsDefaultOrEmpty)
            throw new ArgumentException("An append-only session turn requires at least one input token.", nameof(appendPrompt));

        await _turnGate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            ThrowIfDisposed();
            var lease = _store.AcquireLease(SessionId);
            var operation = _store.Begin(lease, expectedRevision, operationId, requestDigest);
            if (operation.State != SessionOperationState.Accepted)
                return new HotSessionTurnResult(operation, Cursor,
                    operation.ResultChunks?.ToImmutableArray() ?? [], IsIdempotentReplay: true);

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
            try
            {
                int initialPositions = checked(priorCursor.MaterializedPositionCount + appendTokens.Length
                    + (sampling.MaxNewTokens > 0 ? 1 : 0));
                reservation = _resources.Reserve(SessionId, checked((long)initialPositions * _kvBytesPerToken), _modelKey);
            }
            catch (SessionResourceBudgetExceededException ex)
            {
                _store.Fail(lease, operationId, ex.Message);
                throw;
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
                    await foreach (var chunk in _engine.GenerateRetainedChunksAsync(
                        appendPrompt, sampling, _state, cancellationToken,
                        targetPositions => reservation.TryRenew(checked((long)targetPositions * _kvBytesPerToken))).ConfigureAwait(false))
                        chunks.Add(chunk);
                    generationCompleted = true;

                    var outcome = _state.LastTurn
                        ?? throw new InvalidOperationException("The retained engine completed without a retained turn outcome.");
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
                    return new HotSessionTurnResult(completed, nextCursor, resultChunks, IsIdempotentReplay: false);
                }
                catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
                {
                    await WaitForStateReleaseAsync().ConfigureAwait(false);
                    if (operationCommitted) throw;
                    CompensateUncommittedTurn(generationCompleted, cursorPublished, priorCursor, resourcesFinalized);
                    var cancelled = _store.Cancel(lease, operationId);
                    return new HotSessionTurnResult(cancelled, Cursor, chunks.ToImmutable(), IsIdempotentReplay: false);
                }
                catch (Exception ex)
                {
                    await WaitForStateReleaseAsync().ConfigureAwait(false);
                    if (operationCommitted) throw;
                    CompensateUncommittedTurn(generationCompleted, cursorPublished, priorCursor, resourcesFinalized);
                    var failed = _store.Fail(lease, operationId, ex.Message);
                    return new HotSessionTurnResult(failed, Cursor, chunks.ToImmutable(), IsIdempotentReplay: false);
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

    public HotSession Create(SessionId? sessionId = null, string? modelKey = null)
    {
        var snapshot = _store.Create(sessionId);
        var session = new HotSession(_engine, _tokenizer, _store, _resources,
            _engine.KvBytesPerToken, _engine.MaxSequenceLength,
            _options.MaxCapturedOutputChunks, snapshot.SessionId, RemoveDisposedSession, modelKey);
        if (!_sessions.TryAdd(snapshot.SessionId, session))
        {
            session.Dispose();
            throw new InvalidOperationException($"Session '{snapshot.SessionId}' is already active.");
        }
        return session;
    }

    public HotSession Create(SessionAddress address) => Create(address.ToSessionId(), address.ModelFingerprint);

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
