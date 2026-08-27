using System.Collections.Immutable;

namespace OpenTail.Stingray.Engine;

/// <summary>Exact token outcome of the most recently committed retained turn.</summary>
public sealed record RetainedTurnOutcome(
    int TurnStartPosition,
    int MaterializedPosition,
    ImmutableArray<int> GeneratedTokenIds);

/// <summary>
/// Owns one hot, backend-private sequence cache between calls to
/// <see cref="ContinuousBatchingEngine.GenerateRetainedChunksAsync"/>.
///
/// <para>The handle is intentionally not a persistence format and exposes no concrete cache
/// type. It is single-writer: while a turn is queued or executing, another turn against the same
/// handle is rejected. Successful turns retain their materialized state; cancelled and failed
/// turns restore the turn-start boundary on backends that implement
/// <see cref="IRewindableSequenceKvCache"/>.</para>
/// </summary>
public sealed class RetainedSequenceState : IDisposable
{
    private readonly object _gate = new();
    private ISequenceKvCache? _cache;
    private bool _inUse;
    private bool _disposed;
    private int _materializedPosition;
    private RetainedTurnOutcome? _lastTurn;

    /// <summary>Whether this handle currently owns a reusable hot cache.</summary>
    public bool HasRetainedState
    {
        get { lock (_gate) return _cache is not null; }
    }

    /// <summary>
    /// Whether this handle can be reclaimed right now: it holds a retained cache and has no turn
    /// queued or active. Callers must still race-check under their own lock before acting, since
    /// this can go stale the instant it's read.
    /// </summary>
    public bool IsReclaimable
    {
        get { lock (_gate) return !_inUse && _cache is not null; }
    }

    /// <summary>
    /// Drops the retained cache, freeing its resources, without disposing this handle — the next
    /// turn against this session simply starts cold (a fresh prefill) instead of resuming hot.
    /// No-op if a turn is currently queued or active (mirrors <see cref="Dispose"/>'s own
    /// in-use guard: an active lease already took ownership of <c>_cache</c>, so there is nothing
    /// here to evict). Returns the materialized position that was dropped, so a caller can size
    /// how much it just reclaimed.
    /// </summary>
    internal int EvictIfIdle()
    {
        lock (_gate)
        {
            if (_inUse || _cache is null) return 0;
            int freed = _materializedPosition;
            _cache.Dispose();
            _cache = null;
            _materializedPosition = 0;
            _lastTurn = null;
            return freed;
        }
    }

    /// <summary>
    /// docs/028 Phase 2: attempts to fork a shareable, page-aligned leading prefix off this
    /// handle's retained cache for another (sibling) session to seed from, using the same
    /// capture-then-fork primitives <see cref="ContinuousBatchingEngine"/>'s own cross-request
    /// prefix cache already relies on (<see cref="IPrefixCacheableBatchedForwardPass"/>) — this is
    /// a second caller of that mechanism, not a new one. Runs entirely under this handle's own
    /// gate so it cannot race a concurrent <see cref="Reserve"/>: either this sees the cache while
    /// genuinely idle and captures it, or a racing reservation wins the gate first and this
    /// correctly reports no match. Returns null (never throws) on any reason forking isn't
    /// possible right now — in use, no retained cache, nothing left after page-alignment — since
    /// this is always a best-effort optimization a caller must tolerate failing.
    /// </summary>
    internal (ISequenceKvCache Cache, int Length)? TryForkSharedPrefix(
        IPrefixCacheableBatchedForwardPass prefixFwd, int maxPrefixLength)
    {
        ArgumentNullException.ThrowIfNull(prefixFwd);
        lock (_gate)
        {
            if (_inUse || _cache is null) return null;
            int aligned = Math.Min(maxPrefixLength, _materializedPosition);
            aligned -= aligned % prefixFwd.PrefixCacheBlockSize;
            if (aligned <= 0) return null;

            var snapshot = prefixFwd.CapturePrefix(_cache, aligned);
            try
            {
                return (prefixFwd.ForkPrefix(snapshot), aligned);
            }
            finally
            {
                // The snapshot only exists to seed the fork above (mirrors CapturePrefix's other
                // caller, ContinuousBatchingEngine.RetainPrefix, except that one keeps its snapshot
                // alive for repeated future reuse in the engine's own LRU list -- this call site
                // needs exactly one fork, so nothing is served by keeping it around). The fork holds
                // its own independent ref-count on the underlying pages (see PagedKvCache.Dispose),
                // so releasing the snapshot here does not affect the fork just returned.
                snapshot.Dispose();
            }
        }
    }

    /// <summary>
    /// Seeds a fresh, never-used handle with an externally forked cache, as if a turn had already
    /// materialized <paramref name="materializedPosition"/> positions — the counterpart to
    /// <see cref="TryForkSharedPrefix"/> on the destination session. Used to pre-warm a brand-new
    /// session from another session's shared prefix (docs/028 Phase 2) rather than the byte-import
    /// path (<see cref="RestoreKvBytes"/>). Throws if this handle already has retained state or a
    /// turn in flight — seeding only makes sense before the handle's first use, and a caller
    /// racing this against a first real turn on the same handle is a caller bug, not something to
    /// silently ignore the way the best-effort source side does.
    /// </summary>
    internal void SeedWithForkedCache(ISequenceKvCache cache, int materializedPosition)
    {
        ArgumentNullException.ThrowIfNull(cache);
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (_inUse || _cache is not null)
                throw new InvalidOperationException(
                    "A retained sequence state can only be seeded from a shared prefix before its first use.");
            _cache = cache;
            _materializedPosition = materializedPosition;
            _lastTurn = null;
        }
    }

    /// <summary>Whether a caller currently has this state queued or executing in the batcher.</summary>
    public bool IsInUse
    {
        get { lock (_gate) return _inUse; }
    }

    /// <summary>Absolute position materialized by the retained state.</summary>
    public int MaterializedPosition
    {
        get { lock (_gate) return _materializedPosition; }
    }

    internal void SetMaterializedPosition(int position)
    {
        lock (_gate)
        {
            _materializedPosition = position;
        }
    }

    private byte[]? _exportedKvBytes;

    internal void RestoreKvBytes(byte[] kvBytes)
    {
        lock (_gate)
        {
            _exportedKvBytes = kvBytes;
        }
    }

    internal byte[]? GetExportedKvBytes()
    {
        lock (_gate)
        {
            return _exportedKvBytes;
        }
    }

    public byte[]? ExportKvBytes()
    {
        lock (_gate)
        {
            if (_cache is IPersistableSequenceKvCache persistable)
            {
                return persistable.ExportKvState();
            }
            return _exportedKvBytes;
        }
    }

    /// <summary>Exact generated token IDs from the last successful retained turn, if any.</summary>
    public RetainedTurnOutcome? LastTurn
    {
        get { lock (_gate) return _lastTurn; }
    }

    internal RetainedSequenceLease Reserve()
    {
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (_inUse)
                throw new InvalidOperationException(
                    "A retained sequence state already has a queued or active generation turn.");

            _inUse = true;
            _lastTurn = null;
            var cache = _cache;
            _cache = null;
            return new RetainedSequenceLease(cache, _materializedPosition);
        }
    }

    internal void Complete(ISequenceKvCache cache, int turnStartPosition, IReadOnlyList<int>? generatedTokenIds = null)
    {
        ArgumentNullException.ThrowIfNull(cache);
        lock (_gate)
        {
            if (!_inUse)
                throw new InvalidOperationException("Retained sequence state completed without an active lease.");

            _inUse = false;
            if (_disposed)
            {
                cache.Dispose();
                return;
            }

            if (cache is not IRewindableSequenceKvCache rewindable)
            {
                cache.Dispose();
                throw new NotSupportedException(
                    $"{cache.GetType().Name} does not provide the retained-session rollback capability.");
            }

            _cache = cache;
            _materializedPosition = rewindable.LogicalPosition;
            _lastTurn = new RetainedTurnOutcome(
                turnStartPosition,
                _materializedPosition,
                generatedTokenIds is null ? [] : [.. generatedTokenIds]);
        }
    }

    internal void RollbackAndComplete(ISequenceKvCache cache, int turnStartPosition)
    {
        ArgumentNullException.ThrowIfNull(cache);
        try
        {
            if (cache is not IRewindableSequenceKvCache rewindable
                || !rewindable.CanRewindTo(turnStartPosition))
                throw new NotSupportedException(
                    $"{cache.GetType().Name} cannot exactly roll back to position {turnStartPosition}.");

            rewindable.RewindTo(turnStartPosition);
            Complete(cache, turnStartPosition);
        }
        catch
        {
            Fail(cache);
            throw;
        }
    }

    internal void Fail(ISequenceKvCache? cache)
    {
        lock (_gate)
        {
            _inUse = false;
            _lastTurn = null;
            cache?.Dispose();
            // A failed cache is not reusable. Its logical position must not survive the cache it
            // described, otherwise a later fresh cache would be incorrectly treated as though it
            // had already materialized this many tokens.
            _materializedPosition = 0;
        }
    }

    /// <summary>
    /// Restores the retained cache to an arbitrary earlier materialized position (a checkpoint),
    /// not just the last turn's own start boundary (contrast <see cref="RollbackLastTurn"/>).
    /// Requires the handle to be idle and hold a rewindable retained cache; throws otherwise.
    /// </summary>
    internal void RollbackTo(int position)
    {
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (_inUse)
                throw new InvalidOperationException("Cannot roll back retained state while a turn is active.");
            if (_cache is not IRewindableSequenceKvCache rewindable || !rewindable.CanRewindTo(position))
                throw new NotSupportedException(
                    $"The retained cache cannot be restored to position {position}.");

            rewindable.RewindTo(position);
            _materializedPosition = position;
            _lastTurn = null;
        }
    }

    /// <summary>Restores the last completed turn's cache to its pre-turn boundary.</summary>
    internal void RollbackLastTurn()
    {
        ISequenceKvCache? cache;
        int turnStart;
        lock (_gate)
        {
            if (_inUse)
                throw new InvalidOperationException("Cannot roll back retained state while a turn is active.");
            cache = _cache;
            turnStart = _lastTurn?.TurnStartPosition
                ?? throw new InvalidOperationException("There is no completed retained turn to roll back.");
            _cache = null;
        }

        try
        {
            if (cache is not IRewindableSequenceKvCache rewindable || !rewindable.CanRewindTo(turnStart))
                throw new NotSupportedException("The retained cache cannot be restored to its last committed boundary.");
            rewindable.RewindTo(turnStart);
            lock (_gate)
            {
                if (_disposed) cache.Dispose();
                else
                {
                    _cache = cache;
                    _materializedPosition = turnStart;
                    _lastTurn = null;
                }
            }
        }
        catch
        {
            cache?.Dispose();
            lock (_gate)
            {
                _cache = null;
                _materializedPosition = 0;
                _lastTurn = null;
            }
            throw;
        }
    }

    /// <inheritdoc />
    public void Dispose()
    {
        lock (_gate)
        {
            if (_disposed) return;
            _disposed = true;
            if (_inUse) return;
            _cache?.Dispose();
            _cache = null;
            _lastTurn = null;
        }
    }
}

internal readonly record struct RetainedSequenceLease(ISequenceKvCache? Cache, int TurnStartPosition);
