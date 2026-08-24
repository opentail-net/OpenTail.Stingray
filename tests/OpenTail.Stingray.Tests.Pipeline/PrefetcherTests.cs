
namespace OpenTail.Stingray.Tests.Pipeline;

public sealed class PrefetcherTests
{
    private static OpenTail.Stingray.Pipeline.MemoryHierarchy MakeHierarchy() => new(
        new OpenTail.Stingray.Pipeline.TierConfig("gpu", CapacityBytes: 1L << 30),
        new OpenTail.Stingray.Pipeline.TierConfig("cpu", CapacityBytes: 1L << 30),
        new OpenTail.Stingray.Pipeline.TierConfig("nvme", CapacityBytes: 1L << 30));

    [Fact]
    public void PrefetchRequest_StoresTensorNameAndDefaultsPriorityToZero()
    {
        var req = new OpenTail.Stingray.Pipeline.PrefetchRequest("layer3.mlp.gate_proj");
        Assert.Equal("layer3.mlp.gate_proj", req.TensorName);
        Assert.Equal(0, req.Priority);
    }

    [Fact]
    public void PrefetchRequest_Priority_CanBeSetExplicitly()
    {
        var req = new OpenTail.Stingray.Pipeline.PrefetchRequest("layer3.mlp.gate_proj", Priority: 5);
        Assert.Equal(5, req.Priority);
    }

    [Fact]
    public async System.Threading.Tasks.Task EnqueueAsync_WithinQueueDepth_CompletesSynchronously()
    {
        // Nothing has been enqueued yet so the background worker is idle waiting on the
        // channel; a write within the bounded capacity must not block. This test only asserts
        // that -- it does not care what Dispose() does with the item afterwards.
        //
        // Dispose() itself races the background worker: MemoryHierarchy.PromoteToGpuAsync is
        // deliberately unimplemented scaffolding (see MemoryHierarchy's own doc comment), and
        // whether the worker manages to dequeue-and-fault on "w" before Dispose()'s
        // cts.Cancel() lands is non-deterministic (Dispose_AfterProcessingRequest_
        // SurfacesUnderlyingFault below documents and relies on exactly that fault when the
        // worker DOES win; Dispose_WhenWorkerIdle_CompletesWithoutThrowing documents the clean
        // shutdown when it doesn't). Tolerate either outcome here rather than assert one.
        await using var memory = MakeHierarchy();
        var prefetcher = new OpenTail.Stingray.Pipeline.Prefetcher(memory, queueDepth: 4);
        var task = prefetcher.EnqueueAsync(new OpenTail.Stingray.Pipeline.PrefetchRequest("w"));
        Assert.True(task.IsCompletedSuccessfully);
        try
        {
            prefetcher.Dispose();
        }
        catch (System.NotImplementedException)
        {
            // Expected when the background worker wins the race and reaches the
            // deliberately-unimplemented PromoteToGpuAsync before cancellation takes effect.
        }
    }

    // ── Dispose is clean on the expected shutdown path ──────────────────────
    // Prefetcher.Dispose() cancels the CTS to stop the background worker's
    // `await foreach (... ReadAllAsync(_cts.Token))` loop. RunAsync catches that
    // cancellation as its normal shutdown signal, so a caller doing
    // `using var prefetcher = ...;` sees no exception even though the worker never
    // dequeued anything.

    [Fact]
    public async System.Threading.Tasks.Task Dispose_WhenWorkerIdle_CompletesWithoutThrowing()
    {
        await using var memory = MakeHierarchy();
        using var prefetcher = new OpenTail.Stingray.Pipeline.Prefetcher(memory, queueDepth: 4);
    }

    [Fact]
    public async System.Threading.Tasks.Task Dispose_AfterProcessingRequest_SurfacesUnderlyingFault()
    {
        // MemoryHierarchy.PromoteToGpuAsync is not implemented yet (see
        // MemoryHierarchyTests), so once the worker actually dequeues and processes a
        // request it faults with NotImplementedException — a real fault, distinct from
        // the expected-shutdown cancellation above, and Dispose() must still surface it
        // rather than swallow it alongside the cancellation case.
        await using var memory = MakeHierarchy();
        var prefetcher = new OpenTail.Stingray.Pipeline.Prefetcher(memory, queueDepth: 4);
        await prefetcher.EnqueueAsync(new OpenTail.Stingray.Pipeline.PrefetchRequest("w"));

        // Enqueue only proves channel admission; it does not mean the worker has dequeued the
        // item. Wait for the actual worker fault rather than relying on a timing delay.
        await Assert.ThrowsAsync<System.NotImplementedException>(async () => await prefetcher.Completion);

        // Dispose() awaits the faulted worker via GetAwaiter().GetResult(), which unwraps and
        // rethrows the original exception directly (unlike .Wait()/.Result, which wrap it in an
        // AggregateException) — that's the whole reason to prefer GetAwaiter().GetResult() here.
        Assert.Throws<System.NotImplementedException>(prefetcher.Dispose);
    }
}
