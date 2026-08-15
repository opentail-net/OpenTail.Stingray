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

    private sealed class FakeResourceBudget(Func<long, ResourceAdmission> estimateAdmission) : IResourceBudget
    {
        public int EstimateAdmissionCallCount { get; private set; }
        public ResourceSnapshot GetCurrent() => new(0, 0, null, 0);
        public ResourceAdmission EstimateAdmission(long candidateModelBytes)
        {
            EstimateAdmissionCallCount++;
            return estimateAdmission(candidateModelBytes);
        }
    }

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
    public async Task Acquire_100ConcurrentColdRequests_ResultIn1PhysicalLoad_100LogicalUsers()
    {
        // Literal Phase 2 acceptance bar (docs/032 §"Implementation phases", Phase 2): "100
        // concurrent requests for one cold model result in 1 physical model load, 100 logical
        // users." Built during Phase 1 already (single-flight was load-bearing for AcquireAsync
        // to be correct at all, not deferrable) — this pins the exact numbers down explicitly.
        const int concurrentCallers = 100;
        int loadCount = 0;
        var gate = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var manager = new ModelRuntimeManager(id =>
        {
            Interlocked.Increment(ref loadCount);
            gate.Task.GetAwaiter().GetResult(); // block every concurrent loader call the same way
            return Load(id.Value);
        });

        var tasks = Enumerable.Range(0, concurrentCallers)
            .Select(_ => manager.AcquireAsync(Id("cold")).AsTask()).ToArray();
        await Task.Delay(75); // let all 100 callers reach the manager and observe the in-flight load
        gate.SetResult();
        var handles = await Task.WhenAll(tasks);

        var runtime = handles[0].Runtime; // captured before disposal — Handle.Runtime throws once disposed
        Assert.Equal(1, loadCount); // 1 physical load ...
        Assert.All(handles, h => Assert.Same(runtime, h.Runtime));
        Assert.Equal(concurrentCallers, runtime.HandleCount); // ... 100 logical users

        // Safe disposal at scale: releasing all 100 handles must land exactly back at zero, with
        // no over-release exception (the codebase-wide convention ModelRuntime.OnHandleReleased
        // enforces) and without physically disposing the still-resident, still-usable runtime.
        foreach (var h in handles) h.Dispose();
        Assert.Equal(0, runtime.HandleCount);
        Assert.Equal(ModelRuntimeState.Ready, runtime.State);
        Assert.False(((DisposableFakeEngine)runtime.Engine).Disposed);
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

    // ── Phase 3 slice: resource observability (not wired into admission yet) ───

    [Fact]
    public void HostResourceBudget_ReportsSaneHostMemoryFigures()
    {
        using var manager = new ModelRuntimeManager(id => Load(id.Value));
        var budget = new HostResourceBudget(manager);

        var snapshot = budget.GetCurrent();

        Assert.True(snapshot.HostMemoryTotalBytes > 0);
        Assert.True(snapshot.HostMemoryAvailableBytes >= 0);
        Assert.True(snapshot.HostMemoryAvailableBytes <= snapshot.HostMemoryTotalBytes);
        Assert.Null(snapshot.AcceleratorMemoryAvailableBytes); // not wired yet — see class doc
    }

    [Fact]
    public async Task HostResourceBudget_ResidentModelBytes_SumsAcrossResidentRuntimes()
    {
        using var manager = new ModelRuntimeManager(id => Load(id.Value), estimateBytes: id => id.Value == "a" ? 100 : 250);
        var budget = new HostResourceBudget(manager);

        using var ha = await manager.AcquireAsync(Id("a"));
        using var hb = await manager.AcquireAsync(Id("b"));

        Assert.Equal(350, budget.GetCurrent().ResidentModelBytes);
    }

    [Fact]
    public void EstimateAdmission_CandidateFarBelowAvailable_IsAllowed()
    {
        using var manager = new ModelRuntimeManager(id => Load(id.Value));
        var budget = new HostResourceBudget(manager);
        long available = budget.GetCurrent().HostMemoryAvailableBytes;

        // Deterministic regardless of the test machine's actual RAM: a candidate at 1% of
        // whatever's currently available must clear the 25% safety margin comfortably.
        Assert.Equal(ResourceAdmission.Allowed, budget.EstimateAdmission(available / 100));
    }

    [Fact]
    public void EstimateAdmission_CandidateFarAboveAnyRealMachine_IsInsufficientHostMemory()
    {
        using var manager = new ModelRuntimeManager(id => Load(id.Value));
        var budget = new HostResourceBudget(manager);

        Assert.Equal(ResourceAdmission.InsufficientHostMemory, budget.EstimateAdmission(long.MaxValue / 4));
    }

    [Fact]
    public void EstimateAdmission_CandidateExactlyAtAvailable_IsRejectedBySafetyMargin()
    {
        using var manager = new ModelRuntimeManager(id => Load(id.Value));
        var budget = new HostResourceBudget(manager);
        long available = budget.GetCurrent().HostMemoryAvailableBytes;

        // A candidate sized exactly at "currently available" still needs the 25% margin on top —
        // proves the margin is actually applied, not just documented.
        Assert.Equal(ResourceAdmission.InsufficientHostMemory, budget.EstimateAdmission(available));
    }

    // ── Phase 3 slice: wiring admission into AcquireAsync ───────────────────

    [Fact]
    public async Task AcquireAsync_NoResourceBudgetSet_AdmissionNeverConsulted_ExistingBehaviorUnchanged()
    {
        using var manager = new ModelRuntimeManager(id => Load(id.Value));
        Assert.Null(manager.ResourceBudget); // off by default

        using var handle = await manager.AcquireAsync(Id("a"));
        Assert.Equal(Id("a"), handle.Runtime.Id);
    }

    [Fact]
    public async Task AcquireAsync_ResourceBudgetAllows_LoadsNormallyWithoutEvictingAnything()
    {
        using var manager = new ModelRuntimeManager(id => Load(id.Value))
        {
            ResourceBudget = new FakeResourceBudget(_ => ResourceAdmission.Allowed),
        };

        using var ha = await manager.AcquireAsync(Id("a"));
        ha.Dispose(); // idle, but nothing should evict it since nothing else was admission-gated
        using var hb = await manager.AcquireAsync(Id("b"));

        Assert.True(manager.TryGetResident(Id("a"), out _)); // still resident — never touched
        Assert.True(manager.TryGetResident(Id("b"), out _));
    }

    [Fact]
    public async Task AcquireAsync_InsufficientButEvictionFreesRoom_SucceedsAfterEvictingIdleOther()
    {
        using var manager = new ModelRuntimeManager(id => Load(id.Value))
        {
            ResourceBudget = new FakeResourceBudget(_ => ResourceAdmission.Allowed),
        };

        var ha = await manager.AcquireAsync(Id("a"));
        var engineA = (DisposableFakeEngine)ha.Runtime.Engine;
        ha.Dispose(); // idle — evictable

        // Swapped in fresh right before "b" so its own call count starts at zero: first check
        // (before eviction) insufficient, second (after evicting idle "a") fits.
        int calls = 0;
        manager.ResourceBudget = new FakeResourceBudget(_ => ++calls == 1
            ? ResourceAdmission.InsufficientHostMemory
            : ResourceAdmission.Allowed);

        using var hb = await manager.AcquireAsync(Id("b"));

        Assert.True(engineA.Disposed);
        Assert.False(manager.TryGetResident(Id("a"), out _));
        Assert.True(manager.TryGetResident(Id("b"), out _));
    }

    [Fact]
    public async Task AcquireAsync_StillInsufficientAfterEviction_ThrowsAndLeavesNoPoisonedState()
    {
        using var manager = new ModelRuntimeManager(id => Load(id.Value))
        {
            ResourceBudget = new FakeResourceBudget(_ => ResourceAdmission.InsufficientHostMemory),
        };

        var ex = await Assert.ThrowsAsync<InsufficientResourcesException>(
            async () => await manager.AcquireAsync(Id("too-big")));
        Assert.Equal(Id("too-big"), ex.Model);
        Assert.False(manager.TryGetResident(Id("too-big"), out _));

        // Not poisoned: a later acquisition (once the budget allows it) still works cleanly.
        manager.ResourceBudget = new FakeResourceBudget(_ => ResourceAdmission.Allowed);
        using var handle = await manager.AcquireAsync(Id("too-big"));
        Assert.Equal(Id("too-big"), handle.Runtime.Id);
    }

    [Fact]
    public async Task AcquireAsync_AlreadyResidentModel_BypassesAdmissionEntirely()
    {
        using var manager = new ModelRuntimeManager(id => Load(id.Value))
        {
            ResourceBudget = new FakeResourceBudget(_ => ResourceAdmission.Allowed),
        };
        using var first = await manager.AcquireAsync(Id("a"));

        var budget = new FakeResourceBudget(_ => ResourceAdmission.InsufficientHostMemory);
        manager.ResourceBudget = budget;

        // Re-acquiring an ALREADY-resident model must hit the fast path, never the admission
        // gate — admission only governs starting a brand-new physical load.
        using var second = await manager.AcquireAsync(Id("a"));
        Assert.Same(first.Runtime, second.Runtime);
        Assert.Equal(0, budget.EstimateAdmissionCallCount);
    }
}
