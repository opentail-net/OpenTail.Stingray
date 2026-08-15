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
    /// Returns a lease on the resident runtime for <paramref name="model"/>, loading it first if
    /// necessary. Concurrent cold acquisitions for the same <paramref name="model"/> single-flight
    /// onto one physical load. The returned handle must be disposed by the caller when the work
    /// requiring residency is done (see <see cref="ModelRuntimeHandle"/>).
    /// </summary>
    ValueTask<ModelRuntimeHandle> AcquireAsync(ModelId model, CancellationToken cancellationToken = default);

    /// <summary>Non-blocking lookup of an already-resident runtime. Does not acquire a handle —
    /// only use this for observability; acquiring a real lease requires <see cref="AcquireAsync"/>.</summary>
    bool TryGetResident(ModelId model, out ModelRuntime? runtime);

    /// <summary>Point-in-time snapshot of every currently resident runtime.</summary>
    IReadOnlyList<ModelRuntimeStats> Snapshot();
}

/// <inheritdoc cref="IModelRuntimeManager"/>
public sealed class ModelRuntimeManager : IModelRuntimeManager, IDisposable
{
    private readonly Func<ModelId, LoadedEngine> _loader;
    private readonly Func<ModelId, long> _estimateBytes;
    private readonly object _lock = new();
    private readonly Dictionary<ModelId, ModelRuntime> _resident = new();
    private readonly Dictionary<ModelId, TaskCompletionSource<ModelRuntime>> _pendingLoads = new();
    private TaskCompletionSource _residencyChanged = new(TaskCreationOptions.RunContinuationsAsynchronously);
    private ModelResidencyMode _residencyMode;
    private bool _disposed;

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
            var list = new List<ModelRuntimeStats>(_resident.Count);
            foreach (var rt in _resident.Values)
                list.Add(new ModelRuntimeStats(rt.Id, rt.State, rt.EstimatedModelBytes, rt.HandleCount,
                    rt.ActiveRequests, rt.IsPinned, rt.LastUsed));
            return list;
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

                if (_pendingLoads.TryGetValue(model, out var tcs))
                {
                    pendingLoad = tcs.Task;
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

                    var newTcs = new TaskCompletionSource<ModelRuntime>(TaskCreationOptions.RunContinuationsAsynchronously);
                    _pendingLoads[model] = newTcs;
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
        // Disposing while holding _lock mirrors SharedModelCache.Dispose's existing precedent in
        // this codebase (synchronous disposal under the same lock) rather than introducing a new
        // async-disposal pattern for Phase 1.
        foreach (var rt in toDispose) rt.Dispose();
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
            NotifyResidencyChanged();
            tcs.TrySetResult(runtime);
        }
        catch (Exception ex)
        {
            // A failed load must not poison the single-flight table forever — remove the entry
            // so a later acquisition retries instead of awaiting a permanently-faulted task.
            lock (_lock) { _pendingLoads.Remove(model); }
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
