using System.Runtime.CompilerServices;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Server;

namespace OpenTail.Stingray.Tests.Server.Fast;

/// <summary>
/// Phase 1 coverage for <see cref="ModelRuntimeManager"/>
/// (docs/032-multi-model-inference-runtime-plan.md). No real model/GGUF involved — the loader
/// passed to the manager is a caller-supplied delegate, exactly like
/// <see cref="OpenTailStingrayServerOptions.EngineFactory"/> already is for the rest of the
/// server test suite.
/// </summary>
public sealed class ModelRuntimeManagerTests
{
    private static ModelId Id(string name) => new(name);

    private sealed class DisposableFakeEngine : IInferenceEngine, IDisposable
    {
        public string ModelId { get; }
        public int QueueDepth => 0;
        public int ActiveRequests => 0;
        public bool PrefixCacheEnabled => false;
        public long PrefillTokensReused => 0;
        public bool Disposed { get; private set; }

        public DisposableFakeEngine(string modelId) => ModelId = modelId;

        public void Dispose() => Disposed = true;

        public async IAsyncEnumerable<GenerateChunk> GenerateChunksAsync(
            string prompt, SamplingParams sp, [EnumeratorCancellation] CancellationToken ct = default,
            string? canonicalHistoryPrefix = null)
        {
            await Task.Yield();
            yield return new GenerateChunk(GenerateChunkKind.Text, "ok");
        }
    }

    private static LoadedEngine Load(string modelId) =>
        new(new DisposableFakeEngine(modelId), "qwen2", null);

    // ── Lifecycle ────────────────────────────────────────────────────────────

    [Fact]
    public async Task Acquire_RepeatedlyForSameModel_SharesOneRuntime()
    {
        int loadCount = 0;
        using var manager = new ModelRuntimeManager(id => { loadCount++; return Load(id.Value); });

        using var h1 = await manager.AcquireAsync(Id("a"));
        using var h2 = await manager.AcquireAsync(Id("a"));

        Assert.Same(h1.Runtime, h2.Runtime);
        Assert.Equal(1, loadCount);
        Assert.Equal(2, h1.Runtime.HandleCount);
    }

    [Fact]
    public async Task Acquire_DifferentModels_ProduceIndependentRuntimes()
    {
        using var manager = new ModelRuntimeManager(id => Load(id.Value));

        using var ha = await manager.AcquireAsync(Id("a"));
        using var hb = await manager.AcquireAsync(Id("b"));

        Assert.NotSame(ha.Runtime, hb.Runtime);
        Assert.NotSame(ha.Runtime.Engine, hb.Runtime.Engine);
    }

    [Fact]
    public async Task Acquire_ConcurrentColdRequests_SingleFlightToOnePhysicalLoad()
    {
        int loadCount = 0;
        var gate = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var manager = new ModelRuntimeManager(id =>
        {
            Interlocked.Increment(ref loadCount);
            gate.Task.GetAwaiter().GetResult(); // block every concurrent loader call the same way
            return Load(id.Value);
        });

        var tasks = Enumerable.Range(0, 20).Select(_ => manager.AcquireAsync(Id("cold")).AsTask()).ToArray();
        await Task.Delay(50); // let all 20 callers reach the manager and observe the in-flight load
        gate.SetResult();
        var handles = await Task.WhenAll(tasks);

        Assert.Equal(1, loadCount);
        Assert.All(handles, h => Assert.Same(handles[0].Runtime, h.Runtime));
        foreach (var h in handles) h.Dispose();
    }

    [Fact]
    public async Task Acquire_LoadFailure_DoesNotPoisonFutureAcquisitions()
    {
        int attempt = 0;
        using var manager = new ModelRuntimeManager(id =>
        {
            attempt++;
            if (attempt == 1) throw new InvalidOperationException("boom");
            return Load(id.Value);
        });

        await Assert.ThrowsAsync<InvalidOperationException>(async () => await manager.AcquireAsync(Id("flaky")));

        using var handle = await manager.AcquireAsync(Id("flaky"));
        Assert.Equal(2, attempt);
        Assert.NotNull(handle.Runtime);
    }

    [Fact]
    public async Task HandleDispose_IsIdempotent_AndReleasingDoesNotDisposeTheRuntime()
    {
        using var manager = new ModelRuntimeManager(id => Load(id.Value));
        var handle = await manager.AcquireAsync(Id("a"));
        var runtime = handle.Runtime;

        handle.Dispose();
        handle.Dispose(); // must not throw or double-release

        Assert.Equal(0, runtime.HandleCount);
        Assert.Equal(ModelRuntimeState.Ready, runtime.State);
        Assert.False(((DisposableFakeEngine)runtime.Engine).Disposed);
    }

    [Fact]
    public async Task ManagerDispose_DisposesUnpinnedResidentRuntimes_ButNeverPinnedOnes()
    {
        var manager = new ModelRuntimeManager(id => Load(id.Value));
        using var pinned = await manager.AcquireAsync(Id("pinned"));
        pinned.Runtime.IsPinned = true;
        using (var unpinned = await manager.AcquireAsync(Id("unpinned")))
        {
            var unpinnedEngine = (DisposableFakeEngine)unpinned.Runtime.Engine;
            var pinnedEngine = (DisposableFakeEngine)pinned.Runtime.Engine;

            manager.Dispose();

            Assert.True(unpinnedEngine.Disposed);
            Assert.False(pinnedEngine.Disposed);
        }
    }

    // ── SingleSlot residency ─────────────────────────────────────────────────

    [Fact]
    public async Task SingleSlot_AcquiringNewModel_EvictsIdleOtherModel()
    {
        using var manager = new ModelRuntimeManager(id => Load(id.Value), ModelResidencyMode.SingleSlot);

        var ha = await manager.AcquireAsync(Id("a"));
        var engineA = (DisposableFakeEngine)ha.Runtime.Engine;
        ha.Dispose(); // idle now — evictable

        using var hb = await manager.AcquireAsync(Id("b"));

        Assert.True(engineA.Disposed);
        Assert.False(manager.TryGetResident(Id("a"), out _));
        Assert.True(manager.TryGetResident(Id("b"), out _));
    }

    [Fact]
    public async Task SingleSlot_AcquiringNewModel_WaitsForBusyOtherModel_NeverLoadsAlongsideIt()
    {
        using var manager = new ModelRuntimeManager(id => Load(id.Value), ModelResidencyMode.SingleSlot);

        using var ha = await manager.AcquireAsync(Id("a")); // stays held — "busy"
        var acquireB = manager.AcquireAsync(Id("b")).AsTask();

        var completed = await Task.WhenAny(acquireB, Task.Delay(200));
        Assert.NotSame(acquireB, completed); // must still be waiting, not loaded alongside A

        ha.Dispose(); // A goes idle
        using var hb = await acquireB.WaitAsync(TimeSpan.FromSeconds(5));
        Assert.Equal(Id("b"), hb.Runtime.Id);
    }

    [Fact]
    public async Task ResidencyMode_ChangedAtRuntime_UnblocksAWaitingAcquisitionImmediately()
    {
        using var manager = new ModelRuntimeManager(id => Load(id.Value), ModelResidencyMode.SingleSlot);

        using var ha = await manager.AcquireAsync(Id("a")); // busy — blocks a SingleSlot acquire of "b"
        var acquireB = manager.AcquireAsync(Id("b")).AsTask();

        var completedEarly = await Task.WhenAny(acquireB, Task.Delay(100));
        Assert.NotSame(acquireB, completedEarly);

        manager.ResidencyMode = ModelResidencyMode.MultiSlot; // the runtime "ask to be single or multi later" toggle

        using var hb = await acquireB.WaitAsync(TimeSpan.FromSeconds(5));
        Assert.Equal(Id("b"), hb.Runtime.Id);
        // Both resident at once now — MultiSlot never evicted "a" to make room.
        Assert.True(manager.TryGetResident(Id("a"), out _));
    }

    // ── Cross-model concurrency / no global lock ────────────────────────────

    [Fact]
    public async Task Acquire_DoesNotBlockOnAnotherModelsInFlightLoad()
    {
        var gateA = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var manager = new ModelRuntimeManager(id =>
        {
            if (id.Value == "slow-a") gateA.Task.GetAwaiter().GetResult();
            return Load(id.Value);
        });

        var acquireA = manager.AcquireAsync(Id("slow-a")).AsTask(); // never released in this test
        await Task.Delay(30); // ensure A's load has genuinely started

        using var hb = await manager.AcquireAsync(Id("fast-b")).AsTask().WaitAsync(TimeSpan.FromSeconds(5));
        Assert.Equal(Id("fast-b"), hb.Runtime.Id);

        gateA.SetResult();
        using var ha = await acquireA;
        Assert.Equal(Id("slow-a"), ha.Runtime.Id);
    }

    [Fact]
    public async Task Cancellation_OfOneWaiter_DoesNotCancelTheSharedLoadForOthers()
    {
        var gate = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var manager = new ModelRuntimeManager(id =>
        {
            gate.Task.GetAwaiter().GetResult();
            return Load(id.Value);
        });

        using var cts = new CancellationTokenSource();
        var cancelledWaiter = manager.AcquireAsync(Id("shared"), cts.Token).AsTask();
        var survivingWaiter = manager.AcquireAsync(Id("shared")).AsTask();

        await Task.Delay(30);
        cts.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () => await cancelledWaiter);

        gate.SetResult();
        using var handle = await survivingWaiter.WaitAsync(TimeSpan.FromSeconds(5));
        Assert.Equal(Id("shared"), handle.Runtime.Id);
    }
}
