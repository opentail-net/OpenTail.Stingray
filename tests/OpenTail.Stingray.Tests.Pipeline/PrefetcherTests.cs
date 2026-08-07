
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
        // channel; a write within the bounded capacity must not block.
        await using var memory = MakeHierarchy();
        var prefetcher = new OpenTail.Stingray.Pipeline.Prefetcher(memory, queueDepth: 4);
        var task = prefetcher.EnqueueAsync(new OpenTail.Stingray.Pipeline.PrefetchRequest("w"));
        Assert.True(task.IsCompletedSuccessfully);
        prefetcher.Dispose();
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

        var ex = Assert.Throws<System.AggregateException>(prefetcher.Dispose);
        Assert.IsType<System.NotImplementedException>(ex.InnerException);
    }
}
