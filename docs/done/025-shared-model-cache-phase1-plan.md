> **ARCHIVED, 2026-08-15.** Implemented as designed (confirmed against source). No open
> remainder tracked separately from [00-current-work.md](../00-current-work.md).

---

# Shared Model Cache — Phase 1: Extraction & Reference-Counted Reuse

## Status

Implementation-ready. Descoped from `024-multi-model-serving-and-request-scheduling-plan.md`
— see `027-model-cache-scope-decision.md` for why this is deliberately smaller than that
plan's own Phase 1. This document is self-contained; it does not require reading 024 first.

## Problem this phase solves

`Tests.ForwardPass` has ~37 files that call `GgufModel.Open(path)` directly, each opening,
mmap'ing, pre-faulting, and disposing its own instance — including 10 files that all open the
*same* `SmolLM2-1.7B-Instruct-Q4_K_M.gguf`. There is no reuse across call sites. A full-suite
run measured climbing to 59.5 GB working set on a 63 GB machine as a result (details in the
investigation this plan traces back to; not reproduced here).

This phase does exactly one thing: make "the `GgufModel` for path X" a lookup-and-reuse
operation instead of an always-fresh-open operation, so repeated requests for the same model
path within one process share the underlying mmap rather than each paying their own
load/pre-fault cost.

## Explicit non-scope for this phase

No eviction, no capacity limit, no scheduler, no session awareness, no `IInferenceEngine`
bundling. Just: open once, hand out reference-counted handles, dispose the underlying model
when nobody holds a handle *and* the caller explicitly releases it. Capacity/eviction is
Phase 2 (`026-shared-model-cache-phase2-eviction-plan.md`) — deliberately separated so this
phase can be validated (correctness: does reuse actually happen, does nothing get disposed
while still in use) before capacity management is layered on.

## Design

New file: `src/OpenTail.Stingray.Core/SharedModelCache.cs`. Lives next to `GgufModel.cs`
since it's a thin wrapper around it, not a new subsystem — same project, no new project
reference required anywhere it's consumed.

```csharp
namespace OpenTail.Stingray.Core;

/// <summary>
/// Process-wide, reference-counted cache of <see cref="GgufModel"/> instances keyed by
/// canonical model path. Repeated <see cref="Acquire"/> calls for the same path within one
/// process share the underlying mmap instead of each opening (and pre-faulting) their own
/// copy. Currently consumed only by test projects (see docs/025, docs/026) — not wired into
/// the Server/CLI production model-loading path. That's an intentional scope boundary, not an
/// oversight: see docs/027 for why.
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
    /// necessarily unload the model (see Phase 2 for when unloading actually happens).
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

/// <summary>A caller's reference to a cached model. Dispose releases the reference.</summary>
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
```

**Class, not a struct.** An earlier version of this sketch used a `readonly struct`. A struct
carrying disposal responsibility over a shared, mutable, reference-counted resource is a real
hazard: `var a = cache.Acquire(path); var b = a;` produces two independent value copies that
both believe they own one logical acquisition, and `a.Dispose(); b.Dispose();` double-releases.
`SharedModelCache.Release`'s negative-refcount check (below) would catch that loudly rather
than silently corrupting state, but "catch loudly" is a worse property than "impossible by
construction." A class has ordinary reference semantics — copying the variable copies the
reference, not the ownership — and `_disposed` makes `Dispose()` idempotent regardless of how
many places hold that same reference. The small allocation cost of a class here is not
meaningful next to the multi-GB operations this handle guards.

Notes on the shape of this:

- **Why a struct handle, not `IDisposable` on `GgufModel` itself.** `GgufModel.Dispose()`
  today means "destroy this," and multiple call sites already rely on that (idempotent, but
  destructive). Changing its meaning to "release a reference" would be a breaking change to
  every existing direct caller (production Server/CLI included). `ModelHandle` is a new,
  additive type; `GgufModel.Open`/`.Dispose()` behavior is completely unchanged for anyone not
  going through the cache.
- **Why the model itself isn't disposed at refcount 0 yet.** That's Phase 2's job (capacity +
  eviction policy). Phase 1's job is narrower: prove that reuse and reference-counting are
  correct in isolation, without also having to reason about eviction timing at the same time.
  Phase 1 alone already fixes the *redundant-load* half of the problem (10 files opening the
  same SmolLM2 model 10 times becomes 1 open + 9 cache hits); it does not yet fix the *total
  distinct models resident at once* half, which only matters once eviction exists.
- **Thread safety**: a single `lock` around the dictionary, matching the pattern already used
  by `ExpertSlotManager`. `parallelizeTestCollections: false` means test-process contention on
  this lock will be negligible in practice; the lock exists for correctness under any future
  caller, not because current test execution is actually concurrent.

## Test-suite migration

Mechanical, one-line-per-call-site change. Before:

```csharp
using var model = GgufModel.Open(path);
```

After:

```csharp
using var handle = SharedModelCacheFixture.Instance.Acquire(path);
var model = handle.Model;
```

`SharedModelCacheFixture` — a thin static holder for one process-wide `SharedModelCache`
instance, disposed once at test-assembly teardown (an `IDisposable` xUnit assembly fixture, or
equivalent for xunit.v3's fixture model). Exact wiring mechanism (assembly fixture vs. a
`[ModuleInitializer]`-registered static) is an implementation detail to confirm against
xunit.v3's supported fixture scopes when this phase is built — not a design question, a
one-afternoon lookup.

Given the scale (37 files), this migration is a scripted find/replace, not manual editing —
the same approach already used successfully this session for other bulk test-file changes
(e.g., namespace updates across the Cuda/Vulkan project splits). Each of the three test
projects (`Tests.ForwardPass`, `Tests.Cuda`, `Tests.Vulkan`) shares one `SharedModelCache`
instance if they end up sharing a process; if they run as separate processes (current default:
one process per test project invocation), each gets its own cache instance and the benefit is
scoped to reuse *within* a project's run, which is where the measured 59.5 GB problem actually
lives (`Tests.ForwardPass` alone).

## Verification

Re-run the exact 9-file/1-model bisection batch already used earlier in the investigation
(`ContinuousBatchingTests`, `DecodePathParityTests`, `Flash64KvOuterTests`,
`MaxSeqLenContractTests`, `PrefillAttentionParityTests`, `PrefillDecodeSelfConsistencyTests`,
`PrefillPathParityTests`, `Q8PrefillLowMagnitudeInputTests`, `SnapKvTests` — all share
SmolLM2). Measured baseline for this batch: peak 7687 MB, 38 tests, 663s.

Expected after this phase: the "Pre-faulted 1.06 GiB" log line (from `MmapPrefault`, which
fires inside `GgufModel.Open`) should appear once instead of ~9 times across the batch — a
directly observable, log-visible confirmation that reuse is actually happening, independent of
the memory measurement. Peak memory should drop meaningfully (the model itself is now loaded
once, not up to 9 times with staggered disposal), though a full explanation of the exact
before/after numbers belongs in Phase 2's write-up, since Phase 1 alone doesn't bound how many
*distinct* models can pile up in one run — only how many times the *same* model gets reloaded.
