namespace OpenTail.Stingray.Core;

/// <summary>
/// Process-wide, reference-counted cache of <see cref="GgufModel"/> instances keyed by
/// canonical model path. Repeated <see cref="Acquire"/> calls for the same path within one
/// process share the underlying mmap instead of each opening (and pre-faulting) their own
/// copy. Currently consumed only by test projects (see docs/025, docs/026) — not wired into
/// the Server/CLI production model-loading path. That's an intentional scope boundary, not an
/// oversight: see docs/027 for why.
///
/// Phase 1 (this file): open once, hand out reference-counted handles, dispose only at
/// <see cref="Dispose"/> (unconditional, process/test-run teardown). No eviction, no capacity
/// limit — see docs/026 for why that's deliberately a separate phase.
/// </summary>
public sealed class SharedModelCache : IDisposable
{
    private readonly Dictionary<string, Entry> _entries = new();
    private readonly object _lock = new();
    private bool _disposed;

    private sealed class Entry
    {
        public required GgufModel Model { get; init; }
        public int RefCount;
    }

    /// <summary>
    /// Returns a handle to the model at <paramref name="path"/>, opening it if this is the
    /// first request for that path in this process. The returned handle must be disposed by
    /// the caller when done — disposing releases this caller's reference; it does not
    /// necessarily unload the model (see docs/026 for when unloading actually happens).
    /// </summary>
    public ModelHandle Acquire(string path)
    {
        string key = Path.GetFullPath(path);
        lock (_lock)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (!_entries.TryGetValue(key, out var entry))
            {
                entry = new Entry { Model = GgufModel.Open(key) };
                _entries[key] = entry;
            }
            entry.RefCount++;
            return new ModelHandle(this, key, entry.Model);
        }
    }

    internal void Release(string key)
    {
        lock (_lock)
        {
            if (_disposed) return;   // Dispose() already tore everything down
            if (!_entries.TryGetValue(key, out var entry)) return;
            entry.RefCount--;
            // Phase 1 deliberately does NOT evict at refcount==0 — see docs/026.
            // A ref count that goes negative is a caller bug (double-release); make it loud.
            if (entry.RefCount < 0)
                throw new InvalidOperationException($"SharedModelCache: over-released '{key}'.");
        }
    }

    /// <summary>Unconditionally disposes every cached model, regardless of outstanding
    /// handles. Call at process/test-run teardown only.</summary>
    public void Dispose()
    {
        lock (_lock)
        {
            if (_disposed) return;
            _disposed = true;
            foreach (var entry in _entries.Values) entry.Model.Dispose();
            _entries.Clear();
        }
    }
}

/// <summary>
/// A caller's reference to a cached model. Dispose releases the reference.
///
/// Class, not a struct: a struct carrying disposal responsibility over a shared, mutable,
/// reference-counted resource is a real hazard (copying the value copies "ownership" too,
/// so two independent Dispose() calls can double-release). A class has ordinary reference
/// semantics — copying the variable copies the reference, not the ownership — and the
/// <c>_disposed</c> flag makes Dispose() idempotent regardless of how many places hold that
/// same reference.
/// </summary>
public sealed class ModelHandle : IDisposable
{
    private readonly SharedModelCache _cache;
    private readonly string _key;
    private bool _disposed;
    public GgufModel Model { get; }

    internal ModelHandle(SharedModelCache cache, string key, GgufModel model)
    {
        _cache = cache; _key = key; Model = model;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _cache.Release(_key);
    }
}
