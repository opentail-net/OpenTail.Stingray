using System.Diagnostics;

namespace OpenTail.Stingray.Engine;

/// <summary>
/// Execution-level mediation between several models that compete for one compute domain (multi-model plan 032, Phase 6).
/// An engine takes one turn per batcher iteration (a prefill chunk plus one decode step) and gives it back before the next, so
/// the gate decides, at iteration boundaries only, which model runs next. Nothing is interrupted part-way and no KV state is
/// checkpointed: a forward pass is never preempted, it just waits for its next turn.
/// </summary>
public interface IExecutionTurnGate
{
    /// <summary>
    /// Waits for <paramref name="participant"/>'s turn. Dispose the returned lease to end the turn. Cancelling
    /// <paramref name="cancellationToken"/> while queued removes the caller from the queue; once the turn has been granted it is
    /// too late to cancel and the lease must still be disposed.
    /// </summary>
    ValueTask<IDisposable> AcquireTurnAsync(string participant, CancellationToken cancellationToken = default);
}

/// <summary>Counters of a <see cref="FairExecutionGate"/>; a snapshot, not a live view.</summary>
/// <param name="Grants">Turns granted so far.</param>
/// <param name="ContendedGrants">Turns granted to a caller that had to wait for them.</param>
/// <param name="Waiting">Callers queued right now.</param>
/// <param name="MaxWait">Longest time any caller waited for a turn.</param>
public readonly record struct ExecutionGateStats(long Grants, long ContendedGrants, int Waiting, TimeSpan MaxWait);

/// <summary>
/// One turn at a time, handed to the longest-waiting caller when a turn ends. A participant that re-acquires immediately queues
/// behind everyone already waiting, so with N busy participants each gets exactly one turn in every N: no participant can starve
/// another, and the worst wait is the sum of the other participants' single iterations. With nobody waiting, acquiring is a
/// lock and a flag, and completes synchronously.
/// </summary>
/// <remarks>
/// Deliberately simple policy (no weights, priorities or time quanta): the unit of service is one engine iteration, which is the
/// finest grain at which a forward pass can be paused without checkpointing. A gate covers one compute domain; models on
/// different devices should not share a gate.
/// </remarks>
public sealed class FairExecutionGate : IExecutionTurnGate
{
    private readonly object _lock = new();
    private readonly LinkedList<Waiter> _waiters = new();
    private bool _held;
    private long _grants;
    private long _contendedGrants;
    private long _maxWaitTicks;

    public ValueTask<IDisposable> AcquireTurnAsync(string participant, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(participant);
        cancellationToken.ThrowIfCancellationRequested();

        Waiter waiter;
        lock (_lock)
        {
            if (!_held)
            {
                _held = true;
                _grants++;
                return new ValueTask<IDisposable>(new Lease(this));
            }
            waiter = new Waiter(Stopwatch.GetTimestamp());
            waiter.Node = _waiters.AddLast(waiter);
        }

        if (cancellationToken.CanBeCanceled)
            waiter.Registration = cancellationToken.Register(static state =>
            {
                var (gate, queued) = ((FairExecutionGate, Waiter))state!;
                gate.Cancel(queued);
            }, (this, waiter));
        return new ValueTask<IDisposable>(waiter.Turn.Task);
    }

    public ExecutionGateStats GetStats()
    {
        lock (_lock)
            return new ExecutionGateStats(_grants, _contendedGrants, _waiters.Count, Stopwatch.GetElapsedTime(0, _maxWaitTicks));
    }

    private void Release()
    {
        Waiter? next = null;
        lock (_lock)
        {
            if (_waiters.First is { } node)
            {
                _waiters.RemoveFirst();
                next = node.Value;
                _grants++;
                _contendedGrants++;
                long waited = Stopwatch.GetTimestamp() - next.EnqueuedAt;
                if (waited > _maxWaitTicks) _maxWaitTicks = waited;
                // The turn passes straight to the next waiter: _held stays true, so a caller arriving now cannot jump the queue.
            }
            else
            {
                _held = false;
            }
        }
        if (next is null) return;
        next.Registration.Dispose();
        next.Turn.TrySetResult(new Lease(this));
    }

    private void Cancel(Waiter waiter)
    {
        bool removed;
        lock (_lock)
        {
            removed = waiter.Node!.List is not null;
            if (removed) _waiters.Remove(waiter.Node);
        }
        if (removed) waiter.Turn.TrySetCanceled();
    }

    private sealed class Waiter(long enqueuedAt)
    {
        public long EnqueuedAt { get; } = enqueuedAt;
        public LinkedListNode<Waiter>? Node { get; set; }
        public CancellationTokenRegistration Registration { get; set; }
        public TaskCompletionSource<IDisposable> Turn { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
    }

    private sealed class Lease(FairExecutionGate gate) : IDisposable
    {
        private int _released;
        public void Dispose()
        {
            if (Interlocked.Exchange(ref _released, 1) == 0) gate.Release();
        }
    }
}
