using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// Phase 2 queue limit (Qwen 3.8 plan): with <c>maxQueuedRequests</c> set, a request that arrives while the waiting queue
/// is full is refused rather than held, and a cancelled waiting request frees its slot at once. Synthetic.
/// </summary>
public sealed class ContinuousBatchingQueueLimitTests
{
    [Fact]
    public async Task QueueFull_RefusesTheRequest_AndCancelledWaitingRequestFreesItsSlot()
    {
        var tok = new ContinuousBatchingCancellationMemoryTests.ByteTokenizer();
        using var engine = new ContinuousBatchingEngine(
            new ContinuousBatchingCancellationMemoryTests.CountingForwardPass(), tok, "queue-limit",
            maxBatchSize: 1, maxQueuedRequests: 1);
        // Long enough that the first request keeps the only batch slot until it is cancelled.
        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 4000 };
        using var overall = new CancellationTokenSource(TimeSpan.FromSeconds(60));

        // A takes the batch slot.
        using var cancelA = new CancellationTokenSource();
        var a = RunUntilCancelled(engine, sp, cancelA.Token);
        await WaitUntilAsync(() => engine.CommittedKvTokens > 0);
        Assert.Equal(0, engine.QueueDepth);

        // B waits: the queue is now full.
        using var cancelB = new CancellationTokenSource();
        var b = RunUntilCancelled(engine, sp, cancelB.Token);
        await WaitUntilAsync(() => engine.QueueDepth == 1);

        // C is refused at once, and its own count is undone.
        await Assert.ThrowsAsync<EngineQueueFullException>(async () =>
        {
            await foreach (var _ in engine.GenerateChunksAsync("Q", sp, overall.Token)) { }
        });
        Assert.Equal(1, engine.QueueDepth);

        // Cancelling B frees its slot now, even though A still holds the batch slot.
        cancelB.Cancel();
        await WaitUntilAsync(() => engine.QueueDepth == 0);
        await b;

        // With the slot free, a new request is accepted and waits.
        using var cancelD = new CancellationTokenSource();
        var d = RunUntilCancelled(engine, sp, cancelD.Token);
        await WaitUntilAsync(() => engine.QueueDepth == 1);
        Assert.False(d.IsFaulted);

        cancelD.Cancel();
        cancelA.Cancel();
        await Task.WhenAll(a, d);
        await WaitUntilAsync(() => engine.QueueDepth == 0 && engine.CommittedKvTokens == 0);
        Assert.Equal(0, engine.QueueDepth);
        Assert.Equal(0, engine.CommittedKvTokens);
    }

    [Fact]
    public void QueueLimit_BelowOne_IsRejectedAtConstruction()
    {
        var tok = new ContinuousBatchingCancellationMemoryTests.ByteTokenizer();
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new ContinuousBatchingEngine(
                new ContinuousBatchingCancellationMemoryTests.CountingForwardPass(), tok, "bad-limit", maxQueuedRequests: 0));
    }

    /// <summary>Consumes a request until it is cancelled, which is the expected end for these long-running requests.</summary>
    private static async Task RunUntilCancelled(ContinuousBatchingEngine engine, SamplingParams sp, CancellationToken ct)
    {
        try
        {
            await foreach (var _ in engine.GenerateChunksAsync("Q", sp, ct)) { }
        }
        catch (OperationCanceledException) { }
    }

    /// <summary>A poll rather than a fixed sleep, so a slow transition passes and a stuck one fails at the timeout.</summary>
    private static async Task WaitUntilAsync(Func<bool> condition)
    {
        var deadline = DateTime.UtcNow + TimeSpan.FromSeconds(10);
        while (!condition() && DateTime.UtcNow < deadline)
            await Task.Delay(5);
    }
}
