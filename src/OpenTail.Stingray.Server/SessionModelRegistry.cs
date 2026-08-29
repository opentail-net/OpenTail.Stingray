using System.Collections.Concurrent;

namespace OpenTail.Stingray.Server;

/// <summary>
/// Tracks which <see cref="ModelId"/> a live multi-model session belongs to
/// (docs/032-multi-model-inference-runtime-plan.md Phase 7 follow-up — closing the Phase 4 gap:
/// "a live but currently-idle <c>HotSession</c> doesn't hold a <see cref="ModelRuntimeHandle"/>").
/// Single-model deployments never touch this type — they keep resolving <see cref="IServerSessionRuntime"/>
/// directly, byte-identical to before this existed.
///
/// <para>The bound <see cref="ModelRuntimeHandle"/> is held — not disposed — for the session's
/// entire lifetime (create → delete), the one legitimate case for retaining a handle past a
/// single operation: docs/032's "scope the handle to the operation, not model selection" guidance
/// is about not pinning a model as a standing "currently selected" marker, not about a session
/// that genuinely needs residency guaranteed across many turns. As long as a binding exists here,
/// <c>ModelRuntime.HandleCount &gt; 0</c>, so the runtime is never evictable — Invariant 2 made
/// real for the multi-model path instead of merely relying on the single-model path's blanket
/// <c>IsPinned</c>.</para>
///
/// <para>In-memory only, deliberately: restoring a durable/cold session's model binding across a
/// process restart would require reverse-mapping a stored model fingerprint back to a currently
/// configured <see cref="NamedModelOptions"/> alias, a materially separate problem left for a
/// dedicated follow-up. A session id this registry doesn't know is treated as "no such session"
/// (404), never a silent wrong-model guess.</para>
/// </summary>
public sealed class SessionModelRegistry
{
    private readonly ConcurrentDictionary<SessionId, ModelRuntimeHandle> _bindings = new();

    /// <summary>Binds a freshly created session to the model runtime its handle was acquired
    /// for. Called exactly once, at session creation. <paramref name="handle"/> is now owned by
    /// this registry — released only via <see cref="Release"/>, never disposed by the caller.</summary>
    public void Bind(SessionId sessionId, ModelRuntimeHandle handle)
    {
        if (!_bindings.TryAdd(sessionId, handle))
            throw new InvalidOperationException($"Session '{sessionId}' is already bound to a model runtime.");
    }

    /// <summary>Looks up the model runtime lease an existing session was created against.</summary>
    public bool TryGet(SessionId sessionId, out ModelRuntimeHandle handle) => _bindings.TryGetValue(sessionId, out handle!);

    /// <summary>Removes and disposes the session's binding, releasing its residency claim so the
    /// model becomes evictable again once no other session references it. A no-op if the session
    /// was never bound (idempotent, safe to call defensively).</summary>
    public void Release(SessionId sessionId)
    {
        if (_bindings.TryRemove(sessionId, out var handle)) handle.Dispose();
    }
}
