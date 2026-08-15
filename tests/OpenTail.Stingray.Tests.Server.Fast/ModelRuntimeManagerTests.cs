using System.Diagnostics;
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
        public long LastCandidateAcceleratorBytes { get; private set; }
        public ResourceSnapshot GetCurrent() => new(0, 0, null, 0, 0);
        public ResourceAdmission EstimateAdmission(long candidateModelBytes, long candidateAcceleratorBytes = 0)
        {
            EstimateAdmissionCallCount++;
            LastCandidateAcceleratorBytes = candidateAcceleratorBytes;
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
    }

    [Fact]
    public void HostResourceBudget_AcceleratorMemory_IsNullOrARealPositiveVramFigure()
    {
        // Portable across dev machines / CI: no Vulkan device -> null ("unknown"); a Vulkan
        // device present -> a real, positive VRAM capacity figure. Never a fabricated or zero
        // placeholder either way — asserting a specific outcome here would make this test
        // hardware-dependent, which the rest of this ("Fast", no-real-GPU-device) suite avoids.
        using var manager = new ModelRuntimeManager(id => Load(id.Value));
        var budget = new HostResourceBudget(manager);

        var snapshot = budget.GetCurrent();

        Assert.True(snapshot.AcceleratorMemoryAvailableBytes is null or > 0);
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

    [Fact]
    public void EstimateAdmission_ZeroAcceleratorCandidate_NeverConsultsAcceleratorCapacity()
    {
        // The default (0 = "not expected to be accelerator-resident") must behave identically to
        // before this parameter existed, on ANY machine — with or without a real accelerator.
        using var manager = new ModelRuntimeManager(id => Load(id.Value));
        var budget = new HostResourceBudget(manager);
        long available = budget.GetCurrent().HostMemoryAvailableBytes;

        Assert.Equal(ResourceAdmission.Allowed, budget.EstimateAdmission(available / 100, candidateAcceleratorBytes: 0));
        Assert.Equal(ResourceAdmission.Allowed, budget.EstimateAdmission(available / 100));
    }

    [Fact]
    public void EstimateAdmission_AcceleratorCandidateFarBelowCapacity_IsAllowed()
    {
        using var manager = new ModelRuntimeManager(id => Load(id.Value));
        var budget = new HostResourceBudget(manager);
        var snapshot = budget.GetCurrent();
        Assert.SkipUnless(snapshot.AcceleratorMemoryAvailableBytes is not null,
            "No accelerator detected in this environment — nothing to measure against.");

        // Deterministic regardless of the real GPU's actual capacity: 1% of measured capacity,
        // host bytes negligible, must clear the 25% safety margin comfortably.
        long candidateAccel = snapshot.AcceleratorMemoryAvailableBytes!.Value / 100;
        Assert.Equal(ResourceAdmission.Allowed, budget.EstimateAdmission(1024, candidateAccel));
    }

    [Fact]
    public void EstimateAdmission_AcceleratorCandidateFarAboveAnyRealGpu_IsInsufficientAcceleratorMemory()
    {
        using var manager = new ModelRuntimeManager(id => Load(id.Value));
        var budget = new HostResourceBudget(manager);
        Assert.SkipUnless(budget.GetCurrent().AcceleratorMemoryAvailableBytes is not null,
            "No accelerator detected in this environment — nothing to measure against.");

        Assert.Equal(ResourceAdmission.InsufficientAcceleratorMemory,
            budget.EstimateAdmission(1024, long.MaxValue / 4));
    }

    [Fact]
    public void EstimateAdmission_AcceleratorCandidateExactlyAtCapacity_IsRejectedBySafetyMargin()
    {
        using var manager = new ModelRuntimeManager(id => Load(id.Value));
        var budget = new HostResourceBudget(manager);
        var snapshot = budget.GetCurrent();
        Assert.SkipUnless(snapshot.AcceleratorMemoryAvailableBytes is not null,
            "No accelerator detected in this environment — nothing to measure against.");

        // Sized exactly at capacity (with nothing else resident, so capacity == available) still
        // needs the margin on top — proves the margin actually applies to the accelerator branch
        // too, not just the host one.
        long capacity = snapshot.AcceleratorMemoryAvailableBytes!.Value;
        Assert.Equal(ResourceAdmission.InsufficientAcceleratorMemory,
            budget.EstimateAdmission(1024, capacity));
    }

    [Fact]
    public void EstimateAdmission_HostInsufficient_ReportsHostReason_EvenIfAcceleratorAlsoInsufficient()
    {
        using var manager = new ModelRuntimeManager(id => Load(id.Value));
        var budget = new HostResourceBudget(manager);

        // Both dimensions are hopelessly oversized — host is checked first, so that's the
        // reported reason regardless of the accelerator figure also failing.
        Assert.Equal(ResourceAdmission.InsufficientHostMemory,
            budget.EstimateAdmission(long.MaxValue / 4, long.MaxValue / 4));
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

    // ── Phase 3 slice: aggregate activity counters (GetStats) ───────────────

    [Fact]
    public async Task GetStats_TracksLoadsAndResidentActiveCounts()
    {
        using var manager = new ModelRuntimeManager(id => Load(id.Value));

        using var ha = await manager.AcquireAsync(Id("a"));
        using (await manager.AcquireAsync(Id("b"))) { } // acquire then immediately release — idle

        var stats = manager.GetStats();
        Assert.Equal(2, stats.ModelLoads);
        Assert.Equal(0, stats.ModelLoadFailures);
        Assert.Equal(2, stats.ResidentModels);
        Assert.Equal(1, stats.ActiveModels); // only "a" still has a live handle
        Assert.Equal(2, stats.KnownModels);
        Assert.Equal(0, stats.PendingLoads);
        Assert.True(stats.EstimatedResidentModelBytes >= 0);
    }

    [Fact]
    public async Task GetStats_TracksLoadFailures_WithoutCountingThemAsSuccesses()
    {
        using var manager = new ModelRuntimeManager(id =>
            id.Value == "bad" ? throw new InvalidOperationException("boom") : Load(id.Value));

        await Assert.ThrowsAsync<InvalidOperationException>(async () => await manager.AcquireAsync(Id("bad")));
        using var ok = await manager.AcquireAsync(Id("good"));

        var stats = manager.GetStats();
        Assert.Equal(1, stats.ModelLoadFailures);
        Assert.Equal(1, stats.ModelLoads);
        Assert.Equal(1, stats.ResidentModels); // only the successful one is resident
    }

    [Fact]
    public async Task GetStats_TracksEvictions_FromSingleSlotResidencyEnforcement()
    {
        using var manager = new ModelRuntimeManager(id => Load(id.Value), ModelResidencyMode.SingleSlot);

        var ha = await manager.AcquireAsync(Id("a"));
        ha.Dispose(); // idle — evictable
        using var hb = await manager.AcquireAsync(Id("b")); // evicts "a" to enforce single-slot

        Assert.Equal(1, manager.GetStats().ModelEvictions);
    }

    [Fact]
    public async Task GetStats_TracksResidencyPressureAndAdmissionRejects_Distinctly()
    {
        using var manager = new ModelRuntimeManager(id => Load(id.Value));

        // Pressure that eviction resolves: pressure event fires, but it's not a hard reject.
        using (var ha = await manager.AcquireAsync(Id("a")))
        {
            // released immediately below — idle before "b" is requested
        }
        int calls = 0;
        manager.ResourceBudget = new FakeResourceBudget(_ => ++calls == 1
            ? ResourceAdmission.InsufficientHostMemory
            : ResourceAdmission.Allowed);
        using (await manager.AcquireAsync(Id("b"))) { }

        var afterResolvedPressure = manager.GetStats();
        Assert.Equal(1, afterResolvedPressure.ResidencyPressureEvents);
        Assert.Equal(0, afterResolvedPressure.AdmissionRejects);

        // Pressure that eviction CANNOT resolve (nothing evictable): hard reject.
        manager.ResourceBudget = new FakeResourceBudget(_ => ResourceAdmission.InsufficientHostMemory);
        await Assert.ThrowsAsync<InsufficientResourcesException>(async () => await manager.AcquireAsync(Id("c")));

        var afterHardReject = manager.GetStats();
        Assert.Equal(2, afterHardReject.ResidencyPressureEvents);
        Assert.Equal(1, afterHardReject.AdmissionRejects);
    }

    [Fact]
    public async Task GetStats_TracksPendingLoads_WhileALoadIsStillInFlight()
    {
        var gate = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var manager = new ModelRuntimeManager(id =>
        {
            gate.Task.GetAwaiter().GetResult();
            return Load(id.Value);
        });

        var acquiring = manager.AcquireAsync(Id("slow")).AsTask();
        await Task.Delay(30); // let the load genuinely start

        var midFlight = manager.GetStats();
        Assert.Equal(1, midFlight.PendingLoads);
        Assert.Equal(0, midFlight.ResidentModels);
        Assert.Equal(1, midFlight.KnownModels); // known even though not yet resident

        gate.SetResult();
        using var handle = await acquiring;
        Assert.Equal(0, manager.GetStats().PendingLoads);
    }

    // ── Pending-load visibility in Snapshot() ───────────────────────────────

    [Fact]
    public async Task Snapshot_ShowsAModelMidLoad_WithLoadingStateAndPreLoadEstimate()
    {
        var gate = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var manager = new ModelRuntimeManager(
            id => { gate.Task.GetAwaiter().GetResult(); return Load(id.Value); },
            estimateBytes: _ => 12345);

        var before = DateTimeOffset.UtcNow;
        var acquiring = manager.AcquireAsync(Id("slow")).AsTask();
        await Task.Delay(30); // let the load genuinely start

        var entry = Assert.Single(manager.Snapshot());
        Assert.Equal(Id("slow"), entry.ModelId);
        Assert.Equal(ModelRuntimeState.Loading, entry.State);
        Assert.Equal(12345, entry.EstimatedModelBytes); // pre-load estimate, not from a real ModelRuntime
        Assert.Equal(0, entry.HandleCount);
        Assert.Equal(0, entry.ActiveRequests);
        Assert.False(entry.Pinned);
        Assert.True(entry.LastUsed >= before); // reports load-start time, not a stale default

        gate.SetResult();
        using var handle = await acquiring;

        // Once loaded, the SAME model now appears as a real resident entry instead.
        var resident = Assert.Single(manager.Snapshot());
        Assert.Equal(ModelRuntimeState.Ready, resident.State);
    }

    [Fact]
    public async Task Snapshot_IncludesBothResidentAndPendingEntriesSimultaneously()
    {
        var gate = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var manager = new ModelRuntimeManager(id =>
            id.Value == "loading" ? Gate(id) : Load(id.Value));

        LoadedEngine Gate(ModelId id)
        {
            gate.Task.GetAwaiter().GetResult();
            return Load(id.Value);
        }

        using var ready = await manager.AcquireAsync(Id("ready"));
        var stillLoading = manager.AcquireAsync(Id("loading")).AsTask();
        await Task.Delay(30);

        var snapshot = manager.Snapshot();
        Assert.Equal(2, snapshot.Count);
        Assert.Contains(snapshot, s => s.ModelId.Equals(Id("ready")) && s.State == ModelRuntimeState.Ready);
        Assert.Contains(snapshot, s => s.ModelId.Equals(Id("loading")) && s.State == ModelRuntimeState.Loading);

        gate.SetResult();
        using var handle = await stillLoading;
    }

    // ── Phase 3 slice: per-runtime accelerator (GPU) residency tracking ─────

    private static LoadedEngine LoadOnBackend(string modelId, string backend) =>
        new(new DisposableFakeEngine(modelId), "qwen2", null,
            RuntimeResolution: new ServerRuntimeResolution(backend, "fake", "gguf", 512));

    [Fact]
    public async Task IsAcceleratorResident_TrueForCudaAndVulkan_FalseForCpu()
    {
        using var manager = new ModelRuntimeManager(id => id.Value switch
        {
            "cpu-model" => LoadOnBackend(id.Value, "cpu"),
            "cuda-model" => LoadOnBackend(id.Value, "cuda"),
            "vulkan-model" => LoadOnBackend(id.Value, "vulkan"),
            _ => throw new InvalidOperationException(),
        });

        using var cpu = await manager.AcquireAsync(Id("cpu-model"));
        using var cuda = await manager.AcquireAsync(Id("cuda-model"));
        using var vulkan = await manager.AcquireAsync(Id("vulkan-model"));

        Assert.False(cpu.Runtime.IsAcceleratorResident);
        Assert.True(cuda.Runtime.IsAcceleratorResident);
        Assert.True(vulkan.Runtime.IsAcceleratorResident);
    }

    [Fact]
    public async Task IsAcceleratorResident_FalseWhenRuntimeResolutionIsUnavailable()
    {
        // e.g. a caller-supplied EngineFactory that never set RuntimeResolution — must not
        // crash or default to "resident" just because the signal is missing.
        using var manager = new ModelRuntimeManager(id => Load(id.Value));
        using var handle = await manager.AcquireAsync(Id("a"));

        Assert.False(handle.Runtime.IsAcceleratorResident);
        Assert.Equal(0, handle.Runtime.AcceleratorResidentBytesEstimate);
    }

    [Fact]
    public async Task AcceleratorResidentBytesEstimate_EqualsModelBytesOnlyWhenAcceleratorResident()
    {
        using var manager = new ModelRuntimeManager(
            id => id.Value == "gpu" ? LoadOnBackend(id.Value, "vulkan") : LoadOnBackend(id.Value, "cpu"),
            estimateBytes: _ => 999);

        using var gpu = await manager.AcquireAsync(Id("gpu"));
        using var cpu = await manager.AcquireAsync(Id("cpu"));

        Assert.Equal(999, gpu.Runtime.AcceleratorResidentBytesEstimate);
        Assert.Equal(0, cpu.Runtime.AcceleratorResidentBytesEstimate);
    }

    [Fact]
    public async Task GetStats_EstimatedAcceleratorResidentBytes_SumsOnlyAcceleratorResidentRuntimes()
    {
        using var manager = new ModelRuntimeManager(
            id => id.Value == "gpu" ? LoadOnBackend(id.Value, "cuda") : LoadOnBackend(id.Value, "cpu"),
            estimateBytes: id => id.Value == "gpu" ? 500 : 300);

        using var gpu = await manager.AcquireAsync(Id("gpu"));
        using var cpu = await manager.AcquireAsync(Id("cpu"));

        Assert.Equal(500, manager.GetStats().EstimatedAcceleratorResidentBytes);
    }

    // ── Phase 3 slice: wiring the accelerator dimension into admission ──────

    [Fact]
    public async Task AcquireAsync_NoAcceleratorEstimatorSupplied_AlwaysPassesZeroToEstimateAdmission()
    {
        var budget = new FakeResourceBudget(_ => ResourceAdmission.Allowed);
        using var manager = new ModelRuntimeManager(id => Load(id.Value)) { ResourceBudget = budget };

        using var handle = await manager.AcquireAsync(Id("a"));

        Assert.Equal(0, budget.LastCandidateAcceleratorBytes);
    }

    [Fact]
    public async Task AcquireAsync_AcceleratorEstimatorSupplied_ValueReachesEstimateAdmission()
    {
        var budget = new FakeResourceBudget(_ => ResourceAdmission.Allowed);
        using var manager = new ModelRuntimeManager(
            id => Load(id.Value), estimateAcceleratorBytes: _ => 12345)
        { ResourceBudget = budget };

        using var handle = await manager.AcquireAsync(Id("a"));

        Assert.Equal(12345, budget.LastCandidateAcceleratorBytes);
    }

    [Fact]
    public async Task AcquireAsync_InsufficientAcceleratorMemory_ThrowsWithCorrectReasonAndCandidateBytes()
    {
        using var manager = new ModelRuntimeManager(
            id => Load(id.Value), estimateAcceleratorBytes: _ => 777)
        {
            ResourceBudget = new FakeResourceBudget(_ => ResourceAdmission.InsufficientAcceleratorMemory),
        };

        var ex = await Assert.ThrowsAsync<InsufficientResourcesException>(
            async () => await manager.AcquireAsync(Id("a")));

        Assert.Equal(ResourceAdmission.InsufficientAcceleratorMemory, ex.Reason);
        Assert.Equal(777, ex.CandidateAcceleratorBytes);
        Assert.Contains("VRAM", ex.Message);
    }

    [Fact]
    public async Task AcquireAsync_InsufficientHostMemory_ExceptionMessageMentionsHostNotVram()
    {
        using var manager = new ModelRuntimeManager(id => Load(id.Value))
        {
            ResourceBudget = new FakeResourceBudget(_ => ResourceAdmission.InsufficientHostMemory),
        };

        var ex = await Assert.ThrowsAsync<InsufficientResourcesException>(
            async () => await manager.AcquireAsync(Id("a")));

        Assert.Equal(ResourceAdmission.InsufficientHostMemory, ex.Reason);
        Assert.Contains("host memory", ex.Message);
        Assert.DoesNotContain("VRAM", ex.Message);
    }

    // ── Phase 6 slice: bounded queue as the alternative to hard failure ─────

    [Fact]
    public async Task AcquireAsync_NoAdmissionWaitTimeoutSet_StillThrowsImmediately_ExistingBehaviorUnchanged()
    {
        using var manager = new ModelRuntimeManager(id => Load(id.Value))
        {
            ResourceBudget = new FakeResourceBudget(_ => ResourceAdmission.InsufficientHostMemory),
        };
        Assert.Null(manager.AdmissionWaitTimeout); // off by default

        var sw = Stopwatch.StartNew();
        await Assert.ThrowsAsync<InsufficientResourcesException>(async () => await manager.AcquireAsync(Id("a")));
        Assert.True(sw.Elapsed < TimeSpan.FromSeconds(1)); // no waiting happened at all
    }

    [Fact]
    public async Task AcquireAsync_AdmissionWaitTimeoutSet_QueuesThenSucceedsOnceResourcesFreeUp()
    {
        using var manager = new ModelRuntimeManager(id => Load(id.Value))
        {
            AdmissionWaitTimeout = TimeSpan.FromSeconds(5),
        };

        var handleA = await manager.AcquireAsync(Id("a")); // held open, unrelated to the fake below

        // First attempt (both the pre- and post-eviction checks inside one TryEnsureAdmissibleLocked
        // call) fails; the retry after the queue wakes succeeds.
        int calls = 0;
        manager.ResourceBudget = new FakeResourceBudget(_ => ++calls <= 2
            ? ResourceAdmission.InsufficientHostMemory
            : ResourceAdmission.Allowed);

        var acquireB = manager.AcquireAsync(Id("b")).AsTask();
        await Task.Delay(50); // let it genuinely enter the queue wait

        var early = await Task.WhenAny(acquireB, Task.Delay(200));
        Assert.NotSame(acquireB, early); // still queued, not already resolved

        handleA.Dispose(); // any residency-change notification wakes a queued waiter for a re-check

        using var handleB = await acquireB.WaitAsync(TimeSpan.FromSeconds(5));
        Assert.Equal(Id("b"), handleB.Runtime.Id);

        // Queuing-then-succeeding must never be counted as a hard rejection — only a caller that
        // actually gives up (immediately, or after a queue timeout) counts.
        Assert.Equal(0, manager.GetStats().AdmissionRejects);
        Assert.True(manager.GetStats().ResidencyPressureEvents >= 1);
    }

    [Fact]
    public async Task AcquireAsync_AdmissionWaitTimeoutSet_ElapsesAndThrowsOriginalReason()
    {
        using var manager = new ModelRuntimeManager(id => Load(id.Value))
        {
            AdmissionWaitTimeout = TimeSpan.FromMilliseconds(150),
            ResourceBudget = new FakeResourceBudget(_ => ResourceAdmission.InsufficientAcceleratorMemory),
        };

        var sw = Stopwatch.StartNew();
        var ex = await Assert.ThrowsAsync<InsufficientResourcesException>(
            async () => await manager.AcquireAsync(Id("a")));
        sw.Stop();

        Assert.Equal(ResourceAdmission.InsufficientAcceleratorMemory, ex.Reason);
        Assert.True(sw.Elapsed >= TimeSpan.FromMilliseconds(120)); // genuinely waited out the timeout
        Assert.Equal(1, manager.GetStats().AdmissionRejects); // counted exactly once
    }

    [Fact]
    public async Task AcquireAsync_AdmissionWaitTimeoutSet_CallerCancellationPropagatesAsOperationCanceled()
    {
        using var manager = new ModelRuntimeManager(id => Load(id.Value))
        {
            AdmissionWaitTimeout = TimeSpan.FromSeconds(30),
            ResourceBudget = new FakeResourceBudget(_ => ResourceAdmission.InsufficientHostMemory),
        };

        using var cts = new CancellationTokenSource();
        var acquiring = manager.AcquireAsync(Id("a"), cts.Token).AsTask();
        await Task.Delay(30); // let it genuinely enter the queue wait
        cts.Cancel();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () => await acquiring);
        // The rejection counter is specifically for gave-up-with-a-resource-reason outcomes —
        // a caller cancelling its own wait is a different, already-covered case (invariant 9).
        Assert.Equal(0, manager.GetStats().AdmissionRejects);
    }

    [Fact]
    public async Task AcquireAsync_QueueAtCapacity_RejectsImmediatelyRatherThanWaiting()
    {
        using var manager = new ModelRuntimeManager(id => Load(id.Value))
        {
            AdmissionWaitTimeout = TimeSpan.FromSeconds(30), // long enough a real wait would stall this test
            MaxQueuedAdmissions = 1,
            ResourceBudget = new FakeResourceBudget(_ => ResourceAdmission.InsufficientHostMemory),
        };

        using var cts = new CancellationTokenSource();
        var firstQueued = manager.AcquireAsync(Id("a"), cts.Token).AsTask();
        await Task.Delay(50); // let it genuinely occupy the one queue slot

        var sw = Stopwatch.StartNew();
        await Assert.ThrowsAsync<InsufficientResourcesException>(async () => await manager.AcquireAsync(Id("b")));
        sw.Stop();
        Assert.True(sw.Elapsed < TimeSpan.FromSeconds(1)); // rejected immediately, never entered the wait

        Assert.Equal(1, manager.GetStats().AdmissionQueueOverflows);
        Assert.Equal(1, manager.GetStats().QueuedAdmissions); // still just "a"

        cts.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () => await firstQueued);
        Assert.Equal(0, manager.GetStats().QueuedAdmissions); // released back to zero on cleanup
    }

    [Fact]
    public async Task AcquireAsync_QueuedAdmission_OldestQueuedAdmissionAgeTracksElapsedTime()
    {
        using var manager = new ModelRuntimeManager(id => Load(id.Value))
        {
            AdmissionWaitTimeout = TimeSpan.FromSeconds(30),
            ResourceBudget = new FakeResourceBudget(_ => ResourceAdmission.InsufficientHostMemory),
        };

        Assert.Null(manager.GetStats().OldestQueuedAdmissionAge); // nothing queued yet

        using var cts = new CancellationTokenSource();
        var queued = manager.AcquireAsync(Id("a"), cts.Token).AsTask();
        await Task.Delay(150);

        var age = manager.GetStats().OldestQueuedAdmissionAge;
        Assert.NotNull(age);
        Assert.True(age!.Value >= TimeSpan.FromMilliseconds(100));

        cts.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () => await queued);
        Assert.Null(manager.GetStats().OldestQueuedAdmissionAge); // cleared once the queue empties
    }

    // ── Phase 6 slice: admission-fairness (oldest-admissible wake, not broadcast re-race) ──

    [Fact]
    public async Task AcquireAsync_TwoEquallyAdmissibleQueuedWaiters_OlderAlwaysWinsOverNewer()
    {
        // Both candidates fail admission identically until "freedUp" flips — proving this exercises
        // the resource-admission queue (WakeOldestAdmissibleQueuedWaiter), not SingleSlot's separate,
        // unchanged broadcast-wait path. Once both are equally admissible, only ONE waiter is woken
        // per residency-change event (see WakeOldestAdmissibleQueuedWaiter's doc), and it must always
        // be the older of the two — never the newer one, regardless of how a broadcast/re-race
        // implementation might have let them compete for the lock.
        using var manager = new ModelRuntimeManager(id => Load(id.Value))
        {
            AdmissionWaitTimeout = TimeSpan.FromSeconds(5),
        };

        using var handle = await manager.AcquireAsync(Id("holder")); // no budget set yet — always succeeds

        bool freedUp = false;
        manager.ResourceBudget = new FakeResourceBudget(
            _ => freedUp ? ResourceAdmission.Allowed : ResourceAdmission.InsufficientHostMemory);

        var older = manager.AcquireAsync(Id("older")).AsTask();
        await Task.Delay(60); // "older" genuinely queues first
        var newer = manager.AcquireAsync(Id("newer")).AsTask();
        await Task.Delay(60); // "newer" queues second, strictly after "older"

        freedUp = true;
        handle.Dispose(); // triggers the fairness scan — both are now admissible; only one is woken

        var firstDone = await Task.WhenAny(older, newer).WaitAsync(TimeSpan.FromSeconds(5));
        Assert.Same(older, firstDone);
        using var olderHandle = await older;
        Assert.Equal(Id("older"), olderHandle.Runtime.Id);

        // "newer" is still queued — only one waiter is woken per notify event, and "older" won it.
        var stillWaiting = await Task.WhenAny(newer, Task.Delay(200));
        Assert.NotSame(newer, stillWaiting);

        // Not permanently stuck, just correctly ordered behind "older" — a second notify lets it
        // through too.
        olderHandle.Dispose();
        using var newerHandle = await newer.WaitAsync(TimeSpan.FromSeconds(5));
        Assert.Equal(Id("newer"), newerHandle.Runtime.Id);
    }

    [Fact]
    public async Task AcquireAsync_OversizedOldestWaiter_DoesNotBlockSmallerAdmissibleWaiterBehindIt()
    {
        // "big" can never fit (>= 1e9 bytes always rejected); "small" fits once resources free up.
        // Both start out queued identically (gated by the same "freedUp" flag) so this proves the
        // scan actually evaluates every candidate rather than "small" merely succeeding outright on
        // its own first attempt. A strict head-only FIFO would let "big" (the oldest) permanently
        // block "small" from ever being woken — oldest-ADMISSIBLE selection must skip "big" instead.
        using var manager = new ModelRuntimeManager(
            id => Load(id.Value),
            estimateBytes: id => id.Value == "big" ? 1_000_000_000L : 1_000L)
        {
            AdmissionWaitTimeout = TimeSpan.FromSeconds(5),
        };

        using var handle = await manager.AcquireAsync(Id("holder")); // no budget set yet — always succeeds

        bool freedUp = false;
        manager.ResourceBudget = new FakeResourceBudget(bytes =>
            !freedUp ? ResourceAdmission.InsufficientHostMemory
            : bytes >= 1_000_000_000L ? ResourceAdmission.InsufficientHostMemory
            : ResourceAdmission.Allowed);

        using var bigCts = new CancellationTokenSource();
        var big = manager.AcquireAsync(Id("big"), bigCts.Token).AsTask();
        await Task.Delay(60); // "big" genuinely queues first — it's the oldest
        var small = manager.AcquireAsync(Id("small")).AsTask();
        await Task.Delay(60); // "small" queues second, strictly newer than "big"

        freedUp = true;
        var sw = Stopwatch.StartNew();
        handle.Dispose(); // triggers the fairness scan

        using var smallHandle = await small.WaitAsync(TimeSpan.FromSeconds(2));
        Assert.True(sw.Elapsed < TimeSpan.FromSeconds(1), $"small took {sw.Elapsed} to resolve");
        Assert.Equal(Id("small"), smallHandle.Runtime.Id);

        // "big" never becomes admissible — it stays queued rather than blocking "small".
        var stillWaiting = await Task.WhenAny(big, Task.Delay(200));
        Assert.NotSame(big, stillWaiting);
        Assert.Equal(1, manager.GetStats().QueuedAdmissions); // just "big"

        bigCts.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () => await big);
    }
}
