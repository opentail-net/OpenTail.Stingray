using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Server;

/// <summary>
/// Canonical identity for a loadable model, used as the key <see cref="IModelRuntimeManager"/>
/// tracks residency by (docs/032-multi-model-inference-runtime-plan.md, "ModelRuntime"). Phase 1
/// resolves this from the configured model path via <see cref="Canonicalize"/> rather than
/// treating a raw, caller-supplied path string as identity — the indirection is what lets a
/// later phase map multiple path spellings/aliases onto one physical model without redefining
/// identity everywhere it's used. No content hash yet; the boundary just needs to already exist.
/// </summary>
public readonly record struct ModelId(string Value)
{
    /// <summary>
    /// Resolves <paramref name="rawPath"/> to a canonical form: <see cref="Path.GetFullPath"/>
    /// when the path exists on disk (mirrors <c>SharedModelCache.Acquire</c>'s own
    /// canonicalization), otherwise the raw string unchanged — e.g. an <c>EngineFactory</c>-only
    /// deployment with no real path, or a path <see cref="InferenceEngineLoader"/>'s own
    /// multi-directory search will still resolve later. Best-effort: a transient I/O error while
    /// probing falls back to the raw string rather than failing identity resolution.
    /// </summary>
    public static ModelId Canonicalize(string rawPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(rawPath);
        try
        {
            if (File.Exists(rawPath) || Directory.Exists(rawPath))
                return new ModelId(Path.GetFullPath(rawPath));
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or ArgumentException or NotSupportedException)
        {
            // Fall through to the raw string.
        }
        return new ModelId(rawPath);
    }

    public override string ToString() => Value;
}

/// <summary>
/// Lifecycle state of a <see cref="ModelRuntime"/>. Mirrors the full state set from
/// docs/032 §"ModelRuntime" so the public shape doesn't need to change as later phases land;
/// Phase 1 only actively produces <see cref="Loading"/>, <see cref="Ready"/>,
/// <see cref="Evicting"/>, <see cref="Failed"/>, and <see cref="Disposed"/>.
/// <see cref="Unloaded"/>, <see cref="Active"/>, and <see cref="Draining"/> are reserved for the
/// residency/scheduling and drain-on-evict work in later phases.
/// </summary>
public enum ModelRuntimeState
{
    Unloaded,
    Loading,
    Ready,
    Active,
    Draining,
    Evicting,
    Failed,
    Disposed,
}

/// <summary>
/// Residency policy for <see cref="IModelRuntimeManager"/> (docs/032 §"Design premise" /
/// §9 "Single-slot mode"). A runtime-mutable property, not a fixed deployment choice — flipping
/// it via <see cref="IModelRuntimeManager.ResidencyMode"/> takes effect immediately, including
/// waking any acquisition currently blocked under <see cref="SingleSlot"/>. Set the initial value
/// via <see cref="OpenTailStingrayServerOptions.ModelResidencyMode"/> or the
/// <c>STINGRAY_MODEL_RESIDENCY_MODE</c> environment variable (<c>single</c> / <c>multi</c>).
/// </summary>
public enum ModelResidencyMode
{
    /// <summary>
    /// Resource/pressure-driven residency (docs/032 §"Resource admission") — the default. Phase 1
    /// has no resource budget yet (that lands in Phase 3), so today this simply means no forced
    /// eviction: every distinct acquired <see cref="ModelId"/> gets its own resident runtime.
    /// </summary>
    MultiSlot = 0,

    /// <summary>
    /// At most one non-pinned resident model at a time. Acquiring a different model evicts the
    /// current one once it has no outstanding handles; if the current one is busy (live handles
    /// or pinned), the new acquisition waits rather than loading a second model alongside it —
    /// never kills active inference to satisfy residency (docs/032 invariant 10).
    /// </summary>
    SingleSlot = 1,
}

/// <summary>
/// Point-in-time snapshot of a <see cref="ModelRuntime"/> for observability
/// (docs/032 §"Observability"). Deliberately narrower than the plan's full
/// <c>ModelRuntimeStats</c> sketch — Phase 1 has no load-count/eviction-count counters yet.
/// </summary>
public readonly record struct ModelRuntimeStats(
    ModelId ModelId,
    ModelRuntimeState State,
    long EstimatedModelBytes,
    int HandleCount,
    int ActiveRequests,
    bool Pinned,
    DateTimeOffset LastUsed);

/// <summary>
/// System-wide activity snapshot for <see cref="IModelRuntimeManager"/> (docs/032
/// §"Observability"/§"Metrics that matter"). The count fields (<see cref="ModelLoads"/> onward)
/// are cumulative since the manager was constructed, not point-in-time — everything else is a
/// snapshot as of the call. Narrower than the plan's own sketch in one place: <see cref="PendingLoads"/>
/// replaces its <c>PendingRequests</c> name, because what this manager actually tracks is
/// in-flight physical loads, not per-inference-request queue depth (that lives on
/// <c>IInferenceEngine.QueueDepth</c> per resident runtime instead).
/// </summary>
public readonly record struct MultiModelRuntimeStats(
    int KnownModels,
    int ResidentModels,
    int ActiveModels,
    long EstimatedResidentModelBytes,
    int PendingLoads,
    long ModelLoads,
    long ModelLoadFailures,
    long ModelEvictions,
    long AdmissionRejects,
    long ResidencyPressureEvents);

/// <summary>
/// A complete, independently executable model runtime — the production successor to
/// <see cref="OpenTail.Stingray.Core.SharedModelCache"/>'s <see cref="OpenTail.Stingray.Core.ModelHandle"/>,
/// extended with residency state and handle-based lifetime instead of a bare refcount
/// (docs/032 §"New core abstraction: ModelRuntime").
///
/// <see cref="ActiveRequests"/> is manager-side <b>observability</b>, not authoritative session
/// ownership — it reads straight through to <see cref="Engine"/>'s own counter.
/// <see cref="HandleCount"/> (backed by outstanding <see cref="ModelRuntimeHandle"/>s) is the
/// actual eviction gate; nothing infers liveness from timestamps or request counts.
/// </summary>
public sealed class ModelRuntime : IDisposable
{
    private int _handleCount;
    private long _lastUsedTicks;

    public ModelId Id { get; }

    /// <summary>The loaded engine bundle this runtime wraps (tokenizer, chat template, session
    /// runtime, etc.) — exactly what <see cref="InferenceEngineLoader.Load"/> or a caller's
    /// <see cref="OpenTailStingrayServerOptions.EngineFactory"/> produced.</summary>
    public LoadedEngine Loaded { get; }

    public IInferenceEngine Engine => Loaded.Engine;

    /// <summary>Best-effort resident-weight estimate (file/directory size on disk). Not KV, not
    /// workspace, not runtime overhead — see docs/032 §"Resource admission" on why Phase 1
    /// admission is deliberately conservative rather than exact.</summary>
    public long EstimatedModelBytes { get; }

    public ModelRuntimeState State { get; internal set; }

    /// <summary>
    /// When true, this runtime is never evicted by <see cref="IModelRuntimeManager"/> policy AND
    /// is skipped by <see cref="ModelRuntimeManager.Dispose"/> — disposal remains whoever pinned
    /// it's own responsibility. Used by the server DI wiring so the always-on configured model's
    /// lifetime stays owned solely by the <c>IInferenceEngine</c> DI registration, avoiding a
    /// double-dispose race with the manager's own shutdown cleanup.
    /// </summary>
    public bool IsPinned { get; set; }

    public int HandleCount => Volatile.Read(ref _handleCount);

    public int ActiveRequests => Engine.ActiveRequests;

    public DateTimeOffset LastUsed => new(Volatile.Read(ref _lastUsedTicks), TimeSpan.Zero);

    /// <summary>Eviction invariant (docs/032): evictable only when not pinned, no outstanding
    /// handles, and not already mid-transition.</summary>
    internal bool IsEvictable => !IsPinned && HandleCount == 0 && State == ModelRuntimeState.Ready;

    internal ModelRuntime(ModelId id, LoadedEngine loaded, long estimatedModelBytes)
    {
        Id = id;
        Loaded = loaded;
        EstimatedModelBytes = estimatedModelBytes;
        State = ModelRuntimeState.Ready;
        Touch();
    }

    internal void Touch() => Interlocked.Exchange(ref _lastUsedTicks, DateTimeOffset.UtcNow.Ticks);

    internal void OnHandleAcquired()
    {
        Interlocked.Increment(ref _handleCount);
        Touch();
    }

    internal void OnHandleReleased()
    {
        if (Interlocked.Decrement(ref _handleCount) < 0)
            throw new InvalidOperationException($"ModelRuntime '{Id}': handle over-released.");
        Touch();
    }

    /// <summary>Disposes the underlying engine exactly once (idempotent — safe to call from both
    /// eviction and manager teardown paths). Callers holding a live <see cref="ModelRuntimeHandle"/>
    /// must never observe this: <see cref="IModelRuntimeManager"/> only disposes runtimes that are
    /// already <see cref="IsEvictable"/>.</summary>
    public void Dispose()
    {
        if (State == ModelRuntimeState.Disposed) return;
        State = ModelRuntimeState.Disposed;
        (Loaded.Engine as IDisposable)?.Dispose();
    }
}

/// <summary>
/// A caller's lease on a resident <see cref="ModelRuntime"/> — the authoritative residency lease
/// (docs/032 §"ModelRuntimeHandle is the authoritative residency lease"). Holding a handle
/// prevents the runtime from becoming evictable, full stop; disposing it releases exactly this
/// caller's claim and does not itself evict anything. <see cref="IModelRuntimeManager"/> never
/// infers liveness from anything other than outstanding handles.
///
/// Handle lifetime is scoped to the operation that needs residency, not to model selection:
/// acquire → use the session/engine → dispose. Retaining a handle indefinitely as a "currently
/// selected model" marker accidentally pins that runtime forever.
///
/// Class, not a struct, for the same reason as <see cref="OpenTail.Stingray.Core.ModelHandle"/>:
/// ordinary reference semantics plus an idempotent <c>Dispose()</c> guard is the safe shape for a
/// disposal token over shared, mutable, reference-counted state.
/// </summary>
public sealed class ModelRuntimeHandle : IDisposable
{
    private readonly ModelRuntimeManager _manager;
    private ModelRuntime? _runtime;

    public ModelRuntime Runtime => _runtime ?? throw new ObjectDisposedException(nameof(ModelRuntimeHandle));

    internal ModelRuntimeHandle(ModelRuntimeManager manager, ModelRuntime runtime)
    {
        _manager = manager;
        _runtime = runtime;
        runtime.OnHandleAcquired();
    }

    public void Dispose()
    {
        var rt = Interlocked.Exchange(ref _runtime, null);
        if (rt is null) return;
        rt.OnHandleReleased();
        _manager.NotifyResidencyChanged();
    }
}
