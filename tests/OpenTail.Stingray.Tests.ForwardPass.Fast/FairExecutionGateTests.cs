
namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>
/// Execution-level fairness for several resident models sharing one compute domain (multi-model plan 032, Phase 6, slice 5).
/// Everything here is in memory: participants are plain tasks, so the ordering guarantees are asserted exactly, not statistically.
/// </summary>
public sealed class FairExecutionGateTests
{
    [Fact]
    public async Task Uncontended_AcquireCompletesSynchronously_AndCountsNoContention()
    {
        var gate = new FairExecutionGate();

        for (int i = 0; i < 3; i++)
        {
            var pending = gate.AcquireTurnAsync("a");
            Assert.True(pending.IsCompletedSuccessfully);   // no waiting, no allocation of a continuation
            (await pending).Dispose();
        }

        var stats = gate.GetStats();
        Assert.Equal(3, stats.Grants);
        Assert.Equal(0, stats.ContendedGrants);
        Assert.Equal(0, stats.Waiting);
    }

    [Fact]
    public async Task Waiters_AreServedInArrivalOrder()
    {
        var gate = new FairExecutionGate();
        var first = await gate.AcquireTurnAsync("first");

        var order = new List<string>();
        var b = Task.Run(async () => { using var _ = await gate.AcquireTurnAsync("b"); lock (order) order.Add("b"); });
        await WaitUntil(() => gate.GetStats().Waiting == 1);
        var c = Task.Run(async () => { using var _ = await gate.AcquireTurnAsync("c"); lock (order) order.Add("c"); });
        await WaitUntil(() => gate.GetStats().Waiting == 2);
        var d = Task.Run(async () => { using var _ = await gate.AcquireTurnAsync("d"); lock (order) order.Add("d"); });
        await WaitUntil(() => gate.GetStats().Waiting == 3);

        first.Dispose();
        await Task.WhenAll(b, c, d).WaitAsync(TimeSpan.FromSeconds(10));

        Assert.Equal(["b", "c", "d"], order);
        Assert.Equal(3, gate.GetStats().ContendedGrants);
    }

    [Fact]
    public async Task ABusyParticipant_CannotStarveOthers_StrictRoundRobin()
    {
        // Three participants each want 30 turns back to back. A greedy one (re-acquiring the instant it releases) must still
        // alternate with the rest: while anyone else waits, no participant gets two turns in a row.
        // Driven on one thread, so the order is exact. Completing a waiter's task is synchronous in the gate (only its
        // continuations are deferred), which lets the test see who was granted the turn by polling IsCompleted.
        var gate = new FairExecutionGate();
        var holder = await gate.AcquireTurnAsync("test");
        var pending = new Dictionary<string, Task<IDisposable>>
        {
            ["a"] = gate.AcquireTurnAsync("a").AsTask(),
            ["b"] = gate.AcquireTurnAsync("b").AsTask(),
            ["c"] = gate.AcquireTurnAsync("c").AsTask(),
        };
        Assert.Equal(3, gate.GetStats().Waiting);
        holder.Dispose();

        var log = new List<string>();
        for (int turn = 0; turn < 90; turn++)
        {
            var granted = pending.Where(p => p.Value.IsCompleted).ToList();
            Assert.Single(granted);   // exactly one participant holds the turn at any time
            string name = granted[0].Key;
            log.Add(name);
            (await granted[0].Value).Dispose();   // already completed, so this does not suspend
            // Greedy: the participant asks again the instant it is done, and must go to the back of the queue.
            pending[name] = gate.AcquireTurnAsync(name).AsTask();
        }

        for (int i = 0; i < log.Count; i++)
            Assert.True(log[i] == "abc"[i % 3].ToString(), $"turn {i} went to {log[i]}: {string.Join("", log)}");
    }

    [Fact]
    public async Task ACancelledWaiter_LeavesTheQueue_AndTheTurnPassesToTheNext()
    {
        var gate = new FairExecutionGate();
        var holder = await gate.AcquireTurnAsync("holder");

        using var cts = new CancellationTokenSource();
        var cancelled = gate.AcquireTurnAsync("cancelled", cts.Token).AsTask();
        await WaitUntil(() => gate.GetStats().Waiting == 1);
        var next = gate.AcquireTurnAsync("next").AsTask();
        await WaitUntil(() => gate.GetStats().Waiting == 2);

        cts.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => cancelled);
        Assert.Equal(1, gate.GetStats().Waiting);

        holder.Dispose();
        (await next.WaitAsync(TimeSpan.FromSeconds(10))).Dispose();

        Assert.Equal(0, gate.GetStats().Waiting);
        // The gate is free again: a fresh acquire is immediate.
        Assert.True(gate.AcquireTurnAsync("again").IsCompletedSuccessfully);
    }

    [Fact]
    public async Task AlreadyCancelledToken_ThrowsWithoutQueueing()
    {
        var gate = new FairExecutionGate();
        using var cts = new CancellationTokenSource();
        cts.Cancel();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () => await gate.AcquireTurnAsync("a", cts.Token));

        Assert.Equal(0, gate.GetStats().Grants);
        Assert.True(gate.AcquireTurnAsync("b").IsCompletedSuccessfully);   // not left held
    }

    [Fact]
    public async Task DisposingALeaseTwice_ReleasesOnce()
    {
        var gate = new FairExecutionGate();
        var lease = await gate.AcquireTurnAsync("a");
        var waiter = gate.AcquireTurnAsync("b").AsTask();
        await WaitUntil(() => gate.GetStats().Waiting == 1);

        lease.Dispose();
        lease.Dispose();   // must not hand the turn on a second time
        var b = await waiter.WaitAsync(TimeSpan.FromSeconds(10));

        var c = gate.AcquireTurnAsync("c");
        Assert.False(c.IsCompleted);   // b still holds it
        b.Dispose();
        (await c.AsTask().WaitAsync(TimeSpan.FromSeconds(10))).Dispose();
    }

    [Fact]
    public async Task MaxWait_ReportsTheLongestQueuedWait()
    {
        var gate = new FairExecutionGate();
        var holder = await gate.AcquireTurnAsync("holder");
        var waiter = gate.AcquireTurnAsync("waiter").AsTask();
        await WaitUntil(() => gate.GetStats().Waiting == 1);

        await Task.Delay(120);
        holder.Dispose();
        (await waiter.WaitAsync(TimeSpan.FromSeconds(10))).Dispose();

        Assert.True(gate.GetStats().MaxWait >= TimeSpan.FromMilliseconds(100), $"max wait {gate.GetStats().MaxWait}");
    }

    private static async Task WaitUntil(Func<bool> condition)
    {
        var deadline = DateTime.UtcNow.AddSeconds(10);
        while (!condition())
        {
            if (DateTime.UtcNow > deadline) throw new TimeoutException("condition not reached");
            await Task.Delay(1);
        }
    }
}
