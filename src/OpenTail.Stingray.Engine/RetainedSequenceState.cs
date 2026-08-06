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
