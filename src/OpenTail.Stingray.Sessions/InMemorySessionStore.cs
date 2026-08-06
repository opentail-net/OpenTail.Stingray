using System.Collections.Concurrent;

namespace OpenTail.Stingray.Sessions;

/// <summary>
/// Thread-safe hot transaction core for session revision, idempotency and fencing semantics.
/// It deliberately stores no model cache and offers no durability guarantee; a later state owner
/// attaches retained inference state only after these transitions are accepted.
/// </summary>
public sealed class InMemorySessionStore
{
    private sealed class Entry(SessionId id)
    {
        public readonly object Gate = new();
        public readonly SessionId Id = id;
        public SessionRevision Revision = SessionRevision.Initial;
        public long FencingEpoch;
        public readonly Dictionary<SessionOperationId, SessionOperationSnapshot> Operations = [];
    }

    private readonly ConcurrentDictionary<SessionId, Entry> _entries = [];
    private readonly int _maxCompletedOperationRecords;
    private readonly TimeSpan _completedOperationRetention;
    private readonly TimeProvider _clock;

    public InMemorySessionStore(
        int maxCompletedOperationRecords = 256,
        TimeSpan? completedOperationRetention = null,
        TimeProvider? clock = null)
    {
        if (maxCompletedOperationRecords <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxCompletedOperationRecords));
        _maxCompletedOperationRecords = maxCompletedOperationRecords;
        _completedOperationRetention = completedOperationRetention ?? TimeSpan.FromHours(1);
        if (_completedOperationRetention <= TimeSpan.Zero)
            throw new ArgumentOutOfRangeException(nameof(completedOperationRetention));
        _clock = clock ?? TimeProvider.System;
    }

    /// <summary>Creates an empty session; supplying an existing identifier is rejected.</summary>
    public SessionSnapshot Create(SessionId? sessionId = null)
    {
        var entry = new Entry(sessionId ?? SessionId.New());
        if (!_entries.TryAdd(entry.Id, entry))
            throw new InvalidOperationException($"Session '{entry.Id}' already exists.");
        return Snapshot(entry);
    }

    /// <summary>Returns an immutable point-in-time view of a session.</summary>
    public SessionSnapshot Open(SessionId sessionId) => Snapshot(GetEntry(sessionId));

    /// <summary>Updates committed revision during cursor restoration upon cold state import.</summary>
    public void SetRevision(SessionId sessionId, SessionRevision revision)
    {
        var entry = GetEntry(sessionId);
        lock (entry.Gate)
        {
            entry.Revision = revision;
        }
    }

    /// <summary>Deletes a session and all in-memory operation records.</summary>
    public bool Delete(SessionId sessionId) => _entries.TryRemove(sessionId, out _);

    /// <summary>Acquires the next fencing epoch for the one generation writer.</summary>
    public SessionLease AcquireLease(SessionId sessionId)
    {
        var entry = GetEntry(sessionId);
        lock (entry.Gate)
        {
            entry.FencingEpoch = checked(entry.FencingEpoch + 1);
            return new SessionLease(sessionId, entry.FencingEpoch);
        }
    }

    /// <summary>
    /// Starts an operation or returns its existing record when the operation id and request digest
    /// match. A new operation must use the current lease and expected committed revision.
    /// </summary>
    public SessionOperationSnapshot Begin(
        SessionLease lease,
        SessionRevision expectedRevision,
        SessionOperationId operationId,
        SessionRequestDigest requestDigest)
    {
        var entry = GetEntry(lease.SessionId);
        lock (entry.Gate)
        {
            EnsureCurrentLease(entry, lease);
            PruneCompletedOperations(entry, _clock.GetUtcNow());
            if (entry.Operations.TryGetValue(operationId, out var prior))
            {
                if (prior.RequestDigest != requestDigest)
                    throw new SessionOperationConflictException(operationId);
                return prior;
            }
            if (entry.Revision != expectedRevision)
                throw new SessionRevisionConflictException(expectedRevision, entry.Revision);

            var accepted = new SessionOperationSnapshot(
                entry.Id, operationId, requestDigest, SessionOperationState.Accepted, lease.FencingEpoch,
                CommittedRevision: null, _clock.GetUtcNow(), CompletedAt: null, FailureReason: null,
                ResultChunks: null);
            entry.Operations.Add(operationId, accepted);
            return accepted;
        }
    }

    /// <summary>Moves a live operation through one non-terminal execution state.</summary>
    public SessionOperationSnapshot Transition(
        SessionLease lease,
        SessionOperationId operationId,
        SessionOperationState expected,
        SessionOperationState next)
    {
        if (next is SessionOperationState.Completed or SessionOperationState.Cancelled or SessionOperationState.Failed)
            throw new ArgumentOutOfRangeException(nameof(next), "Use Complete, Cancel or Fail for terminal states.");
        var entry = GetEntry(lease.SessionId);
        lock (entry.Gate)
        {
            EnsureCurrentLease(entry, lease);
            var operation = GetOperation(entry, operationId);
            if (operation.State != expected)
                throw new SessionOperationStateException(expected, operation.State);
            return Replace(entry, operation with { State = next });
        }
    }

    /// <summary>Atomically advances the committed revision and records successful completion.</summary>
    public SessionOperationSnapshot Complete(
        SessionLease lease,
        SessionOperationId operationId,
        IReadOnlyList<OpenTail.Stingray.Engine.GenerateChunk>? resultChunks = null)
    {
        var entry = GetEntry(lease.SessionId);
        lock (entry.Gate)
        {
            EnsureCurrentLease(entry, lease);
            var operation = GetOperation(entry, operationId);
            if (operation.State != SessionOperationState.CommitPrepared)
                throw new SessionOperationStateException(SessionOperationState.CommitPrepared, operation.State);

            entry.Revision = entry.Revision.Next();
            var completed = Replace(entry, operation with
            {
                State = SessionOperationState.Completed,
                CommittedRevision = entry.Revision,
                CompletedAt = _clock.GetUtcNow(),
                ResultChunks = resultChunks?.ToArray(),
            });
            PruneCompletedOperations(entry, _clock.GetUtcNow());
            return completed;
        }
    }

    /// <summary>Records a terminal cancellation without advancing the committed revision.</summary>
    public SessionOperationSnapshot Cancel(SessionLease lease, SessionOperationId operationId)
        => FinishUncommitted(lease, operationId, SessionOperationState.Cancelled, failureReason: null);

    /// <summary>Records a terminal failure without advancing the committed revision.</summary>
    public SessionOperationSnapshot Fail(SessionLease lease, SessionOperationId operationId, string failureReason)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(failureReason);
        return FinishUncommitted(lease, operationId, SessionOperationState.Failed, failureReason);
    }

    private SessionOperationSnapshot FinishUncommitted(
        SessionLease lease, SessionOperationId operationId, SessionOperationState terminal, string? failureReason)
    {
        var entry = GetEntry(lease.SessionId);
        lock (entry.Gate)
        {
            EnsureCurrentLease(entry, lease);
            var operation = GetOperation(entry, operationId);
            if (operation.State is SessionOperationState.Completed or SessionOperationState.Cancelled or SessionOperationState.Failed)
                throw new SessionOperationStateException(SessionOperationState.Generating, operation.State);
            var finished = Replace(entry, operation with
            {
                State = terminal,
                CompletedAt = _clock.GetUtcNow(),
                FailureReason = failureReason,
            });
            PruneCompletedOperations(entry, _clock.GetUtcNow());
            return finished;
        }
    }

    private static SessionOperationSnapshot Replace(Entry entry, SessionOperationSnapshot operation)
    {
        entry.Operations[operation.OperationId] = operation;
        return operation;
    }

    private Entry GetEntry(SessionId sessionId) =>
        _entries.TryGetValue(sessionId, out var entry) ? entry : throw new SessionNotFoundException(sessionId);

    private static SessionOperationSnapshot GetOperation(Entry entry, SessionOperationId operationId) =>
        entry.Operations.TryGetValue(operationId, out var operation)
            ? operation
            : throw new KeyNotFoundException($"Operation '{operationId}' does not exist in session '{entry.Id}'.");

    private static void EnsureCurrentLease(Entry entry, SessionLease lease)
    {
        if (lease.FencingEpoch != entry.FencingEpoch)
            throw new SessionFencedException(lease.FencingEpoch, entry.FencingEpoch);
    }

    private void PruneCompletedOperations(Entry entry, DateTimeOffset now)
    {
        var completed = entry.Operations.Values
            .Where(operation => operation.State is SessionOperationState.Completed
                or SessionOperationState.Cancelled or SessionOperationState.Failed)
            .OrderBy(operation => operation.CompletedAt)
            .ToList();

        foreach (var expired in completed.Where(operation => operation.CompletedAt <= now - _completedOperationRetention))
            entry.Operations.Remove(expired.OperationId);

        completed = entry.Operations.Values
            .Where(operation => operation.State is SessionOperationState.Completed
                or SessionOperationState.Cancelled or SessionOperationState.Failed)
            .OrderBy(operation => operation.CompletedAt)
            .ToList();
        int excess = completed.Count - _maxCompletedOperationRecords;
        for (int i = 0; i < excess; i++)
            entry.Operations.Remove(completed[i].OperationId);
    }

    private SessionSnapshot Snapshot(Entry entry)
    {
        lock (entry.Gate)
        {
            PruneCompletedOperations(entry, _clock.GetUtcNow());
            return new SessionSnapshot(
                entry.Id,
                entry.Revision,
                DurableRevision: null,
                entry.FencingEpoch,
                entry.Operations.Values.OrderBy(x => x.CreatedAt).ToArray());
        }
    }
}
