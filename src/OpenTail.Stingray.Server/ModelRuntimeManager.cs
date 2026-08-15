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
    /// Optional bounded wait for resource-pressure admission failures — the "queue" step in
    /// docs/032's eviction hierarchy (unused → idle → drain → KV suspension → queue → hard
    /// failure). <c>null</c> (default): a candidate that fails admission even after evicting idle
    /// resident runtimes throws <see cref="InsufficientResourcesException"/> immediately — today's
    /// exact, unchanged behavior; <see cref="ResourceBudget"/> being set doesn't imply this is too.
    /// When set, that failure instead waits up to this duration for a residency change (a handle
    /// released, another runtime evicted, <see cref="ResidencyMode"/> switched) before retrying
    /// the whole acquisition from scratch — cancelling the caller's own token still stops the wait
    /// immediately regardless (invariant 9 unaffected). This bounds wait time <em>per admission
    /// attempt</em>, not total: a candidate that repeatedly almost-fits can retry across several
    /// timeouts before either succeeding or a wait finally elapses with nothing having changed.
    /// Bounded by wait duration and by <see cref="MaxQueuedAdmissions"/> together — see there for
    /// the depth bound.
    /// </summary>
    TimeSpan? AdmissionWaitTimeout { get; set; }

    /// <summary>
    /// Caps how many callers may simultaneously be queued under <see cref="AdmissionWaitTimeout"/>
    /// (docs/032 invariant 11: "every admission queue... is bounded... overflow is an explicit
    /// rejection, never unbounded growth"). Ignored when <see cref="AdmissionWaitTimeout"/> is
    /// <c>null</c>. A caller that would exceed this limit is rejected immediately — the same
    /// <see cref="InsufficientResourcesException"/> it would have received with queueing off,
    /// not a distinct "queue full" error, since from the caller's perspective the outcome is
    /// identical (it doesn't get the model). Default 16, matching this codebase's existing
    /// <c>OpenTailStingrayServerOptions.MaxQueuedRequests</c> default for the same shape of
    /// "how many callers may wait" question elsewhere in the server.
    /// </summary>
    int MaxQueuedAdmissions { get; set; }

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

    /// <summary>A caller queued under <see cref="AdmissionWaitTimeout"/> (docs/032 Phase 6
    /// "admission fairness" slice). One instance is reused across every retry of the <em>same</em>
    /// <see cref="AcquireAsync"/> call — <see cref="EnqueuedAt"/> is set once, at first queue entry,
    /// and never refreshed, so a waiter that wakes, re-checks, and fails again keeps its original
    /// place in line rather than going to the back. Only <see cref="WakeSignal"/> is replaced per
    /// retry (a fresh, not-yet-completed signal to wait on next).</summary>
    private sealed class AdmissionWaiter
    {
        public required ModelId Model { get; init; }
        public required long CandidateBytes { get; init; }
        public required long CandidateAcceleratorBytes { get; init; }
        public required DateTimeOffset EnqueuedAt { get; init; }
        public TaskCompletionSource WakeSignal { get; set; } = null!;
    }

    private readonly Func<ModelId, LoadedEngine> _loader;
    private readonly Func<ModelId, long> _estimateBytes;
    private readonly Func<ModelId, long> _estimateAcceleratorBytes;
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
    private long _admissionQueueOverflows;

    // Admission-fairness queue (docs/032 Phase 6) — every currently-queued AdmissionWaiter,
    // mutated only while holding _lock. Order in this list is NOT wake order (see
    // WakeOldestAdmissibleQueuedWaiter): each waiter's own AdmissionWaiter.EnqueuedAt is the real
    // ordering key, so a waiter that's re-queued after a failed retry keeps its original position
    // in line regardless of where it lands in this list.
    private readonly List<AdmissionWaiter> _admissionQueue = new();

    /// <summary>Off by default (see interface doc) — deliberately not passed by the server DI
    /// wiring yet, so production behavior for the existing single-model path is unchanged until
    /// this is explicitly opted into.</summary>
    public IResourceBudget? ResourceBudget { get; set; }

    /// <summary>Off by default (see interface doc) — an admission failure hard-fails immediately
    /// unless this is explicitly opted into.</summary>
    public TimeSpan? AdmissionWaitTimeout { get; set; }

    /// <summary>Default 16 (see interface doc). Only meaningful when <see cref="AdmissionWaitTimeout"/>
    /// is set.</summary>
    public int MaxQueuedAdmissions { get; set; } = 16;

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
        Func<ModelId, long>? estimateBytes = null,
        Func<ModelId, long>? estimateAcceleratorBytes = null)
    {
        _loader = loader ?? throw new ArgumentNullException(nameof(loader));
        _estimateBytes = estimateBytes ?? DefaultEstimateBytes;
        // No default estimator: whether a candidate is expected to land on an accelerator is
        // caller-specific policy (backend selection lives in whatever OpenTailStingrayServerOptions
        // the loader closure captures) that this manager has no way to infer on its own. Absent an
        // estimator, every candidate is treated as 0 accelerator bytes — i.e. accelerator admission
        // is never consulted, exactly matching behavior before this parameter existed.
        _estimateAcceleratorBytes = estimateAcceleratorBytes ?? (_ => 0);
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
                    rt.ActiveRequests, rt.IsPinned, rt.LastUsed, rt.IsAcceleratorResident,
                    rt.AcceleratorResidentBytesEstimate));
            // A model mid-load has no ModelRuntime yet (that's only created once RunLoad
            // succeeds) — without this, it's invisible to every observer between "acquisition
            // started" and "acquisition finished", which for a large GGUF can be a long window.
            // Backend isn't known yet either at this point, so accelerator fields stay at their
            // not-yet-resolved default (false / 0) rather than guessing.
            foreach (var (id, pending) in _pendingLoads)
                list.Add(new ModelRuntimeStats(id, ModelRuntimeState.Loading, _estimateBytes(id),
                    HandleCount: 0, ActiveRequests: 0, Pinned: false, pending.StartedAt,
                    IsAcceleratorResident: false, AcceleratorResidentBytesEstimate: 0));
            return list;
        }
    }

    public MultiModelRuntimeStats GetStats()
    {
        lock (_lock)
        {
            int active = 0;
            long estimatedBytes = 0;
            long acceleratorBytes = 0;
            foreach (var rt in _resident.Values)
            {
                if (rt.HandleCount > 0) active++;
                estimatedBytes += rt.EstimatedModelBytes;
                acceleratorBytes += rt.AcceleratorResidentBytesEstimate;
            }
            int pendingLoads = _pendingLoads.Count;

            DateTimeOffset? oldestQueued = null;
            foreach (var waiter in _admissionQueue)
            {
                if (oldestQueued is null || waiter.EnqueuedAt < oldestQueued.Value)
                    oldestQueued = waiter.EnqueuedAt;
            }

            return new MultiModelRuntimeStats(
                KnownModels: _resident.Count + pendingLoads,
                ResidentModels: _resident.Count,
                ActiveModels: active,
                EstimatedResidentModelBytes: estimatedBytes,
                EstimatedAcceleratorResidentBytes: acceleratorBytes,
                PendingLoads: pendingLoads,
                ModelLoads: Interlocked.Read(ref _modelLoads),
                ModelLoadFailures: Interlocked.Read(ref _modelLoadFailures),
                ModelEvictions: Interlocked.Read(ref _modelEvictions),
                AdmissionRejects: Interlocked.Read(ref _admissionRejects),
                ResidencyPressureEvents: Interlocked.Read(ref _residencyPressureEvents),
                QueuedAdmissions: _admissionQueue.Count,
                OldestQueuedAdmissionAge: oldestQueued is { } since ? DateTimeOffset.UtcNow - since : null,
                AdmissionQueueOverflows: Interlocked.Read(ref _admissionQueueOverflows));
        }
    }

    public async ValueTask<ModelRuntimeHandle> AcquireAsync(ModelId model, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        // Persists across retries of this one AcquireAsync call so a waiter that wakes, re-checks,
        // and fails again keeps its original queue position (see AdmissionWaiter's doc) instead of
        // being treated as a brand-new arrival on every retry.
        AdmissionWaiter? myWaiter = null;

        while (true)
        {
            Task<ModelRuntime>? pendingLoad = null;
            Task? waitForResidencyChange = null;
            InsufficientResourcesException? admissionFailure = null;

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

                    if (ResourceBudget is { } budget && !TryEnsureAdmissibleLocked(model, budget, out admissionFailure))
                    {
                        if (AdmissionWaitTimeout is null)
                        {
                            Interlocked.Increment(ref _admissionRejects);
                            throw admissionFailure!;
                        }
                        // A waiter re-queueing after a failed retry (myWaiter already set) is the
                        // same logical wait continuing, not a new arrival — it never counts against
                        // capacity a second time, even if new callers filled the queue in the
                        // meantime, since it already legitimately holds its slot.
                        if (myWaiter is null && _admissionQueue.Count >= MaxQueuedAdmissions)
                        {
                            // Queue is at capacity — explicit rejection, never unbounded growth
                            // (invariant 11). Same exception the caller would see with queueing
                            // off entirely: the outcome ("didn't get the model") is identical.
                            Interlocked.Increment(ref _admissionRejects);
                            Interlocked.Increment(ref _admissionQueueOverflows);
                            throw admissionFailure!;
                        }
                        // Queue step (docs/032 eviction hierarchy): wait for a residency change
                        // and retry the whole acquisition from scratch, rather than failing
                        // immediately. admissionFailure is kept so the wait below can surface the
                        // real reason if the timeout elapses with nothing having changed.
                        myWaiter ??= new AdmissionWaiter
                        {
                            Model = model,
                            CandidateBytes = _estimateBytes(model),
                            CandidateAcceleratorBytes = _estimateAcceleratorBytes(model),
                            EnqueuedAt = DateTimeOffset.UtcNow,
                        };
                        myWaiter.WakeSignal = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
                        _admissionQueue.Add(myWaiter);
                        waitForResidencyChange = myWaiter.WakeSignal.Task;
                    }
                    else
                    {
                        var newTcs = new TaskCompletionSource<ModelRuntime>(TaskCreationOptions.RunContinuationsAsynchronously);
                        _pendingLoads[model] = new PendingLoad { Tcs = newTcs, StartedAt = DateTimeOffset.UtcNow };
                        pendingLoad = newTcs.Task;
                        _ = Task.Run(() => RunLoad(model, newTcs));
                    }
                }
            }

            if (waitForResidencyChange is not null)
            {
                try
                {
                    // Cancelling THIS wait only stops this caller from waiting — it never touches
                    // the shared load/residency state, so other waiters on the same thing are
                    // unaffected (invariant 9).
                    if (admissionFailure is null)
                    {
                        await waitForResidencyChange.WaitAsync(cancellationToken).ConfigureAwait(false);
                    }
                    else
                    {
                        // Bounded (per-attempt) wait: TimeoutException means nothing changed in
                        // time — surface the real admission reason instead of a generic timeout.
                        // A caller cancellation still throws OperationCanceledException straight
                        // through, uncaught here, exactly like the unbounded SingleSlot wait above.
                        try
                        {
                            await waitForResidencyChange.WaitAsync(AdmissionWaitTimeout!.Value, cancellationToken)
                                .ConfigureAwait(false);
                        }
                        catch (TimeoutException)
                        {
                            Interlocked.Increment(ref _admissionRejects);
                            throw admissionFailure;
                        }
                    }
                }
                finally
                {
                    // Only admission-queue waits (never SingleSlot's) ever add to _admissionQueue
                    // in the first place — remove this waiter on every exit path (retry, timeout,
                    // or cancellation) so it never lingers as "queued" after this wait ends. A
                    // successful wake already removed it via WakeOldestAdmissibleQueuedWaiter, so
                    // this is a harmless no-op in that case (List<T>.Remove on a reference type
                    // that's no longer present just returns false).
                    if (admissionFailure is not null)
                    {
                        lock (_lock) { _admissionQueue.Remove(myWaiter!); }
                    }
                }
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
    /// by the <see cref="ResidencyMode"/> setter, and by <see cref="RunLoad"/> on both success and
    /// failure — all events that can unblock a SingleSlot acquisition waiting for another model to
    /// go idle. <paramref name="mayHaveFreedResources"/> additionally gates the admission-fairness
    /// scan (see <see cref="WakeOldestAdmissibleQueuedWaiter"/>): a completed load — success or
    /// failure — never frees host/accelerator budget (it only ever consumes it, or consumes
    /// nothing), so <see cref="RunLoad"/> passes <c>false</c>. Critically, a just-succeeded load's
    /// own <see cref="_resident"/> entry is briefly idle/evictable (no handle exists for it yet —
    /// that's created by the awaiting caller, one lock acquisition later) — if the scan ran here
    /// too, it would see its own freshly-loaded model as "idle" and evict it via
    /// <see cref="EvictIdleOthersLocked"/>, which the caller then reloads, gets evicted again, and
    /// so on forever. Handle release and the mode switch don't have this hazard: neither coincides
    /// with a model being mid-handoff between "just loaded" and "claimed by a handle."</summary>
    internal void NotifyResidencyChanged(bool mayHaveFreedResources = true)
    {
        var old = Interlocked.Exchange(ref _residencyChanged, new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously));
        old.TrySetResult();
        if (mayHaveFreedResources) WakeOldestAdmissibleQueuedWaiter();
    }

    /// <summary>Admission-fairness slice (docs/032 Phase 6): picks at most ONE queued waiter to
    /// wake per residency-change event — deliberately not "wake every admissible waiter in one
    /// pass". <see cref="TryEnsureAdmissibleLocked"/>'s <c>EstimateAdmission</c> only counts
    /// already-<em>resident</em> bytes, never in-flight <see cref="_pendingLoads"/> — waking two
    /// waiters from the same snapshot could let both pass admission against resource headroom that
    /// only actually exists once, since neither's load has registered as resident yet. Waking one
    /// at a time means the next candidate's check (on the NEXT notify, e.g. once the winner
    /// finishes loading and itself calls this) sees a state that already reflects the winner
    /// having claimed its share — conservative, not maximally throughput-optimal, but correct
    /// against that accounting gap.
    ///
    /// Selection is oldest-<em>admissible</em>, not strictly oldest: a candidate that can never fit
    /// (e.g. larger than total capacity) would otherwise permanently head-of-line-block every
    /// smaller, genuinely-serviceable waiter behind it. Every queued candidate is evaluated against
    /// the same post-eviction snapshot (eviction runs once for the whole scan, not once per
    /// candidate — none of the queued models are resident to begin with, so which one eventually
    /// wins doesn't change what's evictable), and the admissible one with the smallest
    /// <see cref="AdmissionWaiter.EnqueuedAt"/> wins. An older waiter can therefore never be
    /// overtaken by a newer one that is <em>also</em> queued — but this says nothing about brand
    /// new callers arriving fresh through <see cref="AcquireAsync"/>: those never consult this
    /// queue at all and can still win the lock/load race against everyone already queued. Fixing
    /// that is a materially bigger change (reserving capacity ahead of an actual load, not just
    /// ahead of a wake) and is explicitly out of scope for this slice.</summary>
    private void WakeOldestAdmissibleQueuedWaiter()
    {
        if (ResourceBudget is not { } budget) return;

        AdmissionWaiter? winner = null;
        lock (_lock)
        {
            if (_admissionQueue.Count == 0) return;

            EvictIdleOthersLocked(keep: null);

            foreach (var candidate in _admissionQueue)
            {
                if (winner is not null && candidate.EnqueuedAt >= winner.EnqueuedAt) continue;
                if (budget.EstimateAdmission(candidate.CandidateBytes, candidate.CandidateAcceleratorBytes)
                    == ResourceAdmission.Allowed)
                {
                    winner = candidate;
                }
            }

            if (winner is not null) _admissionQueue.Remove(winner);
        }

        winner?.WakeSignal.TrySetResult();
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

    /// <summary>SingleSlot/resource-admission enforcement: evict every idle, non-pinned resident
    /// runtime other than <paramref name="keep"/> (or, when <c>null</c>, every idle non-pinned
    /// resident runtime with no exception — used by <see cref="WakeOldestAdmissibleQueuedWaiter"/>,
    /// where none of the candidates being considered are themselves resident, so nothing needs
    /// protecting from this pass). Caller must hold <see cref="_lock"/>. Busy/pinned runtimes are
    /// never touched here — <see cref="HasBlockingOtherResident"/> already routed those callers
    /// into the wait path instead of reaching this method.</summary>
    private void EvictIdleOthersLocked(ModelId? keep)
    {
        List<ModelRuntime>? toDispose = null;
        foreach (var key in _resident.Keys.ToArray())
        {
            if (keep is { } k && key.Equals(k)) continue;
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
    /// conservative rather than exact.
    ///
    /// Returns <c>false</c> (with <paramref name="failure"/> populated, not thrown) rather than
    /// throwing directly — the caller decides whether that's an immediate hard failure or a
    /// bounded queue wait depending on <see cref="AdmissionWaitTimeout"/>, so this method stays
    /// policy-free and doesn't need to know which one applies. Does NOT increment
    /// <see cref="_admissionRejects"/> itself for the same reason: only a caller that actually
    /// gives up (immediately, or after a queue timeout) counts as a rejection — one that
    /// successfully queues-and-retries never should, even though this method returned false for
    /// that same attempt.</summary>
    private bool TryEnsureAdmissibleLocked(ModelId model, IResourceBudget budget, out InsufficientResourcesException? failure)
    {
        long candidateBytes = _estimateBytes(model);
        long candidateAcceleratorBytes = _estimateAcceleratorBytes(model);

        var admission = budget.EstimateAdmission(candidateBytes, candidateAcceleratorBytes);
        if (admission == ResourceAdmission.Allowed)
        {
            failure = null;
            return true;
        }

        // Doesn't fit as things stand — this is "under pressure" regardless of whether eviction
        // below, or a later queue retry, manages to resolve it, so it's counted separately from
        // AdmissionRejects (which only counts the harder, unresolved case).
        Interlocked.Increment(ref _residencyPressureEvents);

        // Free whatever's safely reclaimable (idle, non-pinned, never the candidate itself) and
        // re-check once before giving up.
        EvictIdleOthersLocked(keep: model);
        admission = budget.EstimateAdmission(candidateBytes, candidateAcceleratorBytes);
        if (admission == ResourceAdmission.Allowed)
        {
            failure = null;
            return true;
        }

        failure = new InsufficientResourcesException(
            model, candidateBytes, candidateAcceleratorBytes, admission, budget.GetCurrent());
        return false;
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
            NotifyResidencyChanged(mayHaveFreedResources: false); // a successful load only ever
            // consumes budget — see NotifyResidencyChanged's doc for why the fairness scan must
            // not run here specifically.
            tcs.TrySetResult(runtime);
        }
        catch (Exception ex)
        {
            // A failed load must not poison the single-flight table forever — remove the entry
            // so a later acquisition retries instead of awaiting a permanently-faulted task.
            lock (_lock) { _pendingLoads.Remove(model); }
            Interlocked.Increment(ref _modelLoadFailures);
            NotifyResidencyChanged(mayHaveFreedResources: false); // a failed load never held any
            // budget to begin with (pending loads aren't counted as resident) — nothing freed.
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
