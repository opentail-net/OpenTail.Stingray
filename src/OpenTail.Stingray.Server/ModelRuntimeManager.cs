namespace OpenTail.Stingray.Server;

/// <summary>
/// Production model-residency manager (docs/032-multi-model-inference-runtime-plan.md
/// §"IModelRuntimeManager"). Owns model identity, single-flight loading, residency, and eviction
/// for <see cref="ModelRuntime"/> instances — the successor to
/// <see cref="OpenTail.Stingray.Core.SharedModelCache"/> for production use, not a second
/// ownership system next to it.
/// </summary>
public interface IModelRuntimeManager
{
    /// <summary>
    /// Runtime-mutable residency policy (docs/032 §"Single-slot mode"). Changing this wakes any
    /// acquisition currently blocked waiting on the previous policy.
    /// </summary>
    ModelResidencyMode ResidencyMode { get; set; }

    /// <summary>
    /// Optional resource-pressure gate on <em>new</em> loads (docs/032 §"Resource admission").
    /// <c>null</c> (default) disables admission checking entirely — every acquisition behaves
    /// exactly as before this existed. When set, a cold acquisition first evicts idle/evictable
    /// resident runtimes if needed and re-checks; if the candidate still doesn't fit, it throws
    /// <see cref="InsufficientResourcesException"/> rather than overcommitting host memory.
    /// Already-resident acquisitions are never affected — admission only gates starting a new
    /// physical load.
    /// </summary>
    IResourceBudget? ResourceBudget { get; set; }

    /// <summary>
    /// Returns a lease on the resident runtime for <paramref name="model"/>, loading it first if
    /// necessary. Concurrent cold acquisitions for the same <paramref name="model"/> single-flight
    /// onto one physical load. The returned handle must be disposed by the caller when the work
    /// requiring residency is done (see <see cref="ModelRuntimeHandle"/>).
    /// </summary>
    ValueTask<ModelRuntimeHandle> AcquireAsync(ModelId model, CancellationToken cancellationToken = default);

    /// <summary>Non-blocking lookup of an already-resident runtime. Does not acquire a handle —
    /// only use this for observability; acquiring a real lease requires <see cref="AcquireAsync"/>.</summary>
    bool TryGetResident(ModelId model, out ModelRuntime? runtime);

    /// <summary>Point-in-time snapshot of every currently resident OR currently-loading model.
    /// A model mid-load appears with <see cref="ModelRuntimeState.Loading"/>, an estimated
    /// (pre-load) <c>EstimatedModelBytes</c>, zeroed <c>HandleCount</c>/<c>ActiveRequests</c>
    /// (nothing can hold a handle to a runtime that doesn't exist yet), and <c>LastUsed</c> set
    /// to when the load started rather than "last used".</summary>
    IReadOnlyList<ModelRuntimeStats> Snapshot();

    /// <summary>System-wide activity snapshot (docs/032 §"Metrics that matter") — resident/active
    /// counts plus cumulative load/eviction/admission-reject counters since construction.</summary>
    MultiModelRuntimeStats GetStats();
}

/// <inheritdoc cref="IModelRuntimeManager"/>
public sealed class ModelRuntimeManager : IModelRuntimeManager, IDisposable
{
    /// <summary>A load in flight, plus when it started — the latter exists purely so
    /// <see cref="ModelRuntimeManager.Snapshot"/> can report something meaningful for
    /// <c>LastUsed</c> on a model that doesn't have a <see cref="ModelRuntime"/> yet.</summary>
    private sealed class PendingLoad
    {
        public required TaskCompletionSource<ModelRuntime> Tcs { get; init; }
        public required DateTimeOffset StartedAt { get; init; }
    }

    private readonly Func<ModelId, LoadedEngine> _loader;
    private readonly Func<ModelId, long> _estimateBytes;
    private readonly object _lock = new();
    private readonly Dictionary<ModelId, ModelRuntime> _resident = new();
    private readonly Dictionary<ModelId, PendingLoad> _pendingLoads = new();
    private TaskCompletionSource _residencyChanged = new(TaskCreationOptions.RunContinuationsAsynchronously);
    private ModelResidencyMode _residencyMode;
    private bool _disposed;

    // Cumulative activity counters (docs/032 §"Metrics that matter") — never reset, read via
    // GetStats(). Incremented at the exact points the corresponding event actually happens, not
    // inferred after the fact.
    private long _modelLoads;
    private long _modelLoadFailures;
    private long _modelEvictions;
    private long _admissionRejects;
    private long _residencyPressureEvents;

    /// <summary>Off by default (see interface doc) — deliberately not passed by the server DI
    /// wiring yet, so production behavior for the existing single-model path is unchanged until
    /// this is explicitly opted into.</summary>
    public IResourceBudget? ResourceBudget { get; set; }

    public ModelResidencyMode ResidencyMode
    {
        get => _residencyMode;
        set
        {
            _residencyMode = value;
            // A mode flip must be able to unblock anyone already waiting under the old policy
            // (docs/032: "ask to be single or multi later" is a runtime toggle, not a restart).
            NotifyResidencyChanged();
        }
    }

    public ModelRuntimeManager(
        Func<ModelId, LoadedEngine> loader,
        ModelResidencyMode residencyMode = ModelResidencyMode.MultiSlot,
        Func<ModelId, long>? estimateBytes = null)
    {
        _loader = loader ?? throw new ArgumentNullException(nameof(loader));
        _estimateBytes = estimateBytes ?? DefaultEstimateBytes;
        _residencyMode = residencyMode;
    }

    public bool TryGetResident(ModelId model, out ModelRuntime? runtime)
    {
        lock (_lock) return _resident.TryGetValue(model, out runtime);
    }

    public IReadOnlyList<ModelRuntimeStats> Snapshot()
    {
        lock (_lock)
        {
            var list = new List<ModelRuntimeStats>(_resident.Count + _pendingLoads.Count);
            foreach (var rt in _resident.Values)
                list.Add(new ModelRuntimeStats(rt.Id, rt.State, rt.EstimatedModelBytes, rt.HandleCount,
                    rt.ActiveRequests, rt.IsPinned, rt.LastUsed));
            // A model mid-load has no ModelRuntime yet (that's only created once RunLoad
            // succeeds) — without this, it's invisible to every observer between "acquisition
            // started" and "acquisition finished", which for a large GGUF can be a long window.
            foreach (var (id, pending) in _pendingLoads)
                list.Add(new ModelRuntimeStats(id, ModelRuntimeState.Loading, _estimateBytes(id),
                    HandleCount: 0, ActiveRequests: 0, Pinned: false, pending.StartedAt));
            return list;
        }
    }

    public MultiModelRuntimeStats GetStats()
    {
        lock (_lock)
        {
            int active = 0;
            long estimatedBytes = 0;
            foreach (var rt in _resident.Values)
            {
                if (rt.HandleCount > 0) active++;
                estimatedBytes += rt.EstimatedModelBytes;
            }
            int pendingLoads = _pendingLoads.Count;

            return new MultiModelRuntimeStats(
                KnownModels: _resident.Count + pendingLoads,
                ResidentModels: _resident.Count,
                ActiveModels: active,
                EstimatedResidentModelBytes: estimatedBytes,
                PendingLoads: pendingLoads,
                ModelLoads: Interlocked.Read(ref _modelLoads),
                ModelLoadFailures: Interlocked.Read(ref _modelLoadFailures),
                ModelEvictions: Interlocked.Read(ref _modelEvictions),
                AdmissionRejects: Interlocked.Read(ref _admissionRejects),
                ResidencyPressureEvents: Interlocked.Read(ref _residencyPressureEvents));
        }
    }

    public async ValueTask<ModelRuntimeHandle> AcquireAsync(ModelId model, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        while (true)
        {
            Task<ModelRuntime>? pendingLoad = null;
            Task? waitForResidencyChange = null;

            lock (_lock)
            {
                // Fast path: already resident. Handle creation happens while still holding the
                // lock (via the ctor's OnHandleAcquired) so this can never race the eviction path
                // below, which also mutates _resident only under this same lock — the
                // acquire-vs-evict linearization point docs/032 calls for.
                if (_resident.TryGetValue(model, out var resident) && resident.State != ModelRuntimeState.Disposed)
                    return new ModelRuntimeHandle(this, resident);

                if (_pendingLoads.TryGetValue(model, out var pending))
                {
                    pendingLoad = pending.Tcs.Task;
                }
                else if (_residencyMode == ModelResidencyMode.SingleSlot && HasBlockingOtherResident(model))
                {
                    // A different model is resident and busy (live handles, or pinned). Never
                    // evict active work to satisfy single-slot residency (invariant 10) — wait
                    // for it to free up instead of loading a second model alongside it.
                    waitForResidencyChange = _residencyChanged.Task;
                }
                else
                {
                    if (_residencyMode == ModelResidencyMode.SingleSlot)
                        EvictIdleOthersLocked(keep: model);

                    if (ResourceBudget is { } budget)
                        EnsureAdmissibleLocked(model, budget);

                    var newTcs = new TaskCompletionSource<ModelRuntime>(TaskCreationOptions.RunContinuationsAsynchronously);
                    _pendingLoads[model] = new PendingLoad { Tcs = newTcs, StartedAt = DateTimeOffset.UtcNow };
                    pendingLoad = newTcs.Task;
                    _ = Task.Run(() => RunLoad(model, newTcs));
                }
            }

            if (waitForResidencyChange is not null)
            {
                // Cancelling THIS wait only stops this caller from waiting — it never touches
                // the shared load/residency state, so other waiters on the same thing are
                // unaffected (invariant 9).
                await waitForResidencyChange.WaitAsync(cancellationToken).ConfigureAwait(false);
                continue;
            }

            var runtime = await pendingLoad!.WaitAsync(cancellationToken).ConfigureAwait(false);

            lock (_lock)
            {
                // Re-check under the lock: nothing evicts a Ready runtime in Phase 1, but this
                // guards the same race for whenever resource-pressure eviction (Phase 3) lands,
                // without needing to touch this method again.
                if (_resident.TryGetValue(model, out var rt) && ReferenceEquals(rt, runtime) && rt.State != ModelRuntimeState.Disposed)
                    return new ModelRuntimeHandle(this, runtime);
            }
            // Lost the race to eviction between load completion and re-acquiring the lock; retry.
        }
    }

    /// <summary>Called by <see cref="ModelRuntimeHandle.Dispose"/> whenever a handle is released,
    /// and by the <see cref="ResidencyMode"/> setter — both are events that can unblock a
    /// SingleSlot acquisition that's waiting for another model to go idle.</summary>
    internal void NotifyResidencyChanged()
    {
        var old = Interlocked.Exchange(ref _residencyChanged, new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously));
        old.TrySetResult();
    }

    /// <summary>Caller must hold <see cref="_lock"/>.</summary>
    private bool HasBlockingOtherResident(ModelId requested)
    {
        foreach (var kv in _resident)
        {
            if (kv.Key.Equals(requested)) continue;
            if (kv.Value.HandleCount > 0 || kv.Value.IsPinned) return true;
        }
        return false;
    }

    /// <summary>SingleSlot enforcement: evict every idle, non-pinned resident runtime other than
    /// <paramref name="keep"/>. Caller must hold <see cref="_lock"/>. Busy/pinned runtimes are
    /// never touched here — <see cref="HasBlockingOtherResident"/> already routed those callers
    /// into the wait path instead of reaching this method.</summary>
    private void EvictIdleOthersLocked(ModelId keep)
    {
        List<ModelRuntime>? toDispose = null;
        foreach (var key in _resident.Keys.ToArray())
        {
            if (key.Equals(keep)) continue;
            var rt = _resident[key];
            if (!rt.IsEvictable) continue;
            rt.State = ModelRuntimeState.Evicting;
            _resident.Remove(key);
            (toDispose ??= new List<ModelRuntime>()).Add(rt);
        }
        if (toDispose is null) return;
        Interlocked.Add(ref _modelEvictions, toDispose.Count);
        // Disposing while holding _lock mirrors SharedModelCache.Dispose's existing precedent in
        // this codebase (synchronous disposal under the same lock) rather than introducing a new
        // async-disposal pattern for Phase 1.
        foreach (var rt in toDispose) rt.Dispose();
    }

    /// <summary>Resource-pressure admission gate for a fresh cold load. Caller must hold
    /// <see cref="_lock"/>. Reentrant-safe: <paramref name="budget"/>'s
    /// <c>GetCurrent()</c>/<c>EstimateAdmission</c> calls back into this same manager's
    /// <see cref="Snapshot"/>, which re-enters <see cref="_lock"/> — safe because <c>Monitor</c>
    /// (what <c>lock</c> compiles to) is reentrant per-thread, and this is always called
    /// synchronously from the same thread that already holds it.
    ///
    /// Best-effort, not exact: eviction disposes memory-mapped GGUF handles, which typically
    /// frees host pages promptly, but nothing here waits for the OS to actually reflect it before
    /// re-checking — see docs/032 §"Resource admission" on why Phase 1/3's admission story is
    /// conservative rather than exact. Throws as the documented hard-failure last resort; Phase 6's
    /// queueing alternative isn't built yet.</summary>
    private void EnsureAdmissibleLocked(ModelId model, IResourceBudget budget)
    {
        long candidateBytes = _estimateBytes(model);
        if (budget.EstimateAdmission(candidateBytes) == ResourceAdmission.Allowed)
            return;

        // Doesn't fit as things stand — this is "under pressure" regardless of whether eviction
        // below manages to resolve it, so it's counted separately from AdmissionRejects (which
        // only counts the harder, unresolved case).
        Interlocked.Increment(ref _residencyPressureEvents);

        // Free whatever's safely reclaimable (idle, non-pinned, never the candidate itself) and
        // re-check once before giving up.
        EvictIdleOthersLocked(keep: model);
        if (budget.EstimateAdmission(candidateBytes) == ResourceAdmission.Allowed)
            return;

        Interlocked.Increment(ref _admissionRejects);
        throw new InsufficientResourcesException(model, candidateBytes, budget.GetCurrent());
    }

    private void RunLoad(ModelId model, TaskCompletionSource<ModelRuntime> tcs)
    {
        try
        {
            var loaded = _loader(model);
            var runtime = new ModelRuntime(model, loaded, _estimateBytes(model));
            lock (_lock)
            {
                _resident[model] = runtime;
                _pendingLoads.Remove(model);
            }
            Interlocked.Increment(ref _modelLoads);
            NotifyResidencyChanged();
            tcs.TrySetResult(runtime);
        }
        catch (Exception ex)
        {
            // A failed load must not poison the single-flight table forever — remove the entry
            // so a later acquisition retries instead of awaiting a permanently-faulted task.
            lock (_lock) { _pendingLoads.Remove(model); }
            Interlocked.Increment(ref _modelLoadFailures);
            NotifyResidencyChanged();
            tcs.TrySetException(ex);
        }
    }

    private static long DefaultEstimateBytes(ModelId model)
    {
        try
        {
            if (File.Exists(model.Value)) return new FileInfo(model.Value).Length;
            if (Directory.Exists(model.Value))
            {
                long total = 0;
                foreach (var f in Directory.EnumerateFiles(model.Value, "*", SearchOption.AllDirectories))
                    total += new FileInfo(f).Length;
                return total;
            }
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            // Best-effort estimate — fall through to 0.
        }
        return 0;
    }

    /// <summary>Unconditionally disposes every resident runtime that is NOT <see cref="ModelRuntime.IsPinned"/>,
    /// regardless of outstanding handles. Pinned runtimes are deliberately left untouched — the
    /// caller who pinned them (e.g. the server DI wiring's always-on configured model) owns their
    /// disposal, so this can't race a second disposal of the same underlying engine. Call at
    /// process/host shutdown only, same contract as <see cref="OpenTail.Stingray.Core.SharedModelCache.Dispose"/>.</summary>
    public void Dispose()
    {
        lock (_lock)
        {
            if (_disposed) return;
            _disposed = true;
            foreach (var rt in _resident.Values)
            {
                if (!rt.IsPinned) rt.Dispose();
            }
            _resident.Clear();
        }
    }
}
