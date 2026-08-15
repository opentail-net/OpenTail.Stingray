---

# Shared Model Cache — Phase 2: Bounded Capacity & Eviction

## Status

Implementation-ready, depends on Phase 1 (`025-shared-model-cache-phase1-plan.md`) being built
first. Descoped from `024-multi-model-serving-and-request-scheduling-plan.md` — see
`027-model-cache-scope-decision.md` for why this is deliberately smaller than that plan's own
Phase 2 (no scheduler, no session awareness, no resource-budget taxonomy).

## Problem this phase solves

Phase 1 fixes redundant reloading of the *same* model path within one process. It does not
bound how many *distinct* models can be resident at once — a full `Tests.ForwardPass` run
touches ~15 different models across ~40 files, and Phase 1 alone would happily keep all ~15
resident simultaneously for the life of the process, since nothing ever gets disposed until
teardown. That's the other half of the original 59.5 GB measurement. This phase adds a
capacity bound and LRU eviction so the cache holds at most N distinct models at a time.

## Design

Extend `SharedModelCache` (Phase 1) to wrap its dictionary in the existing
`SlruCache<TKey,TValue>` primitive (`src/OpenTail.Stingray.Pipeline/SlruCache.cs`) instead of a
plain `Dictionary`, reusing the same primitive `ExpertSlotManager`/`CudaExpertSlotManager`
already use for MoE expert eviction — no new cache implementation.

**A note on how `SlruCache` actually works, checked against source rather than assumed**:
`SlruCache<TKey,TValue>.Put(key, value, out evictedKey, out evictedValue)` has no callback at
all — it's a plain synchronous method that hands the evicted entry back to its caller via `out`
parameters. The `onEvict` callback pattern some readers may expect from `ExpertCache<T>` is
implemented *one layer above* `SlruCache`, in `ExpertCache<T>.Put()` itself, and fires only
after the inner `_slru.Put()` call has fully returned — so `SlruCache` itself has no
callback-reentrancy hazard to worry about; it doesn't have callbacks.

That does *not* mean the original draft's "re-admit an in-use entry by calling `Put` again"
idea was safe, though — it relocates the risk rather than removing it. If a wrapper's own
`Put()` re-admits an evicted-but-in-use entry by calling itself again, and capacity is still
full, that second call can evict a *different* entry, whose own re-admission can evict a
*third*, and so on — a real cascade when several resident models are simultaneously in use,
not a hypothetical. The design below avoids it structurally instead of guarding against it:
**in-use entries that get evicted from the SLRU never go back into the SLRU.** They move to a
small side table that isn't subject to capacity at all, and are only actually disposed once
their reference count drops to zero — checked at `Release()` time, not at eviction time.

```csharp
public sealed class SharedModelCache : IDisposable
{
    private readonly SlruCache<string, Entry> _cache;      // capacity-bounded, LRU-managed
    private readonly Dictionary<string, Entry> _overflow;  // in-use entries evicted from _cache;
                                                             // NOT subject to capacity
    private readonly object _lock = new();
    private bool _disposed;

    private sealed class Entry
    {
        public required string Key { get; init; }
        public required GgufModel Model { get; init; }
        public int RefCount;
    }

    private long _hits, _misses, _evictions, _softCapExceeded;   // see "Observability" below

    private readonly int _capacity;

    public SharedModelCache(int capacity = DefaultCapacity)
    {
        _capacity = capacity;   // stored directly for GetStats() — simpler than deriving it
                                 // back out of SlruCache's own prob/protected split.
        _cache = new SlruCache<string, Entry>(
            probationaryCapacity: Math.Max(1, capacity / 4),
            protectedCapacity: Math.Max(1, capacity - capacity / 4));
    }

    private const int DefaultCapacity = 3;   // see "Choosing a default" below

    public ModelHandle Acquire(string path)
    {
        string key = Path.GetFullPath(path);
        lock (_lock)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);

            if (_overflow.TryGetValue(key, out var overflowEntry))
            {
                overflowEntry.RefCount++;
                return new ModelHandle(this, key, overflowEntry.Model);
            }

            if (_cache.TryGet(key, out var entry))
            {
                _hits++;
                entry.RefCount++;
                return new ModelHandle(this, key, entry.Model);
            }

            _misses++;
            entry = new Entry { Key = key, Model = GgufModel.Open(key) };
            if (_cache.Put(key, entry, out var evictedKey, out var evicted))
                HandleEviction(evictedKey, evicted);   // never re-enters _cache.Put
            entry.RefCount++;
            return new ModelHandle(this, key, entry.Model);
        }
    }

    // Called only from Acquire, immediately after a successful _cache.Put — never called
    // reentrantly, and never itself calls _cache.Put. That's the whole point: no recursion,
    // no cascade, regardless of how many resident entries happen to be in use at once.
    // Always called with _lock already held.
    private void HandleEviction(string evictedKey, Entry evicted)
    {
        if (evicted.RefCount == 0)
        {
            evicted.Model.Dispose();
            _evictions++;
            return;
        }
        // In use: park it outside the SLRU's capacity accounting entirely, rather than
        // fighting the cache to keep it inside a structure sized for exactly `capacity`.
        _overflow[evictedKey] = evicted;
        _softCapExceeded++;
    }

    internal void Release(string key)
    {
        lock (_lock)
        {
            if (_disposed) return;
            // Whichever table currently holds it — resident-and-managed, or overflow.
            if (_overflow.TryGetValue(key, out var overflowEntry))
            {
                if (--overflowEntry.RefCount == 0)
                {
                    _overflow.Remove(key);
                    overflowEntry.Model.Dispose();   // overflow entries are disposed here,
                    _evictions++;                      // not re-admitted to _cache — see note below
                }
                return;
            }
            if (_cache.TryGet(key, out var entry)) entry.RefCount--;
        }
    }

    /// <summary>Immutable snapshot of current counters, taken atomically under the same lock
    /// every mutation goes through — never a live, independently-mutable view. See
    /// "Observability" below for why this is a snapshot rather than an exposed live object.</summary>
    public SharedModelCacheStats GetStats()
    {
        lock (_lock)
        {
            return new SharedModelCacheStats(
                Capacity: _capacity,
                CacheEntries: _cache.Count,
                OverflowEntries: _overflow.Count,
                LoadedModels: _cache.Count + _overflow.Count,
                ActiveReferences: SumRefCounts(),
                Hits: _hits, Misses: _misses,
                Evictions: _evictions, SoftCapExceeded: _softCapExceeded);
        }
    }

    private int SumRefCounts() =>
        _cache.Values.Sum(e => e.RefCount) + _overflow.Values.Sum(e => e.RefCount);

    public void Dispose()
    {
        lock (_lock)
        {
            if (_disposed) return;
            _disposed = true;
            foreach (var entry in _cache.Values) entry.Model.Dispose();
            foreach (var entry in _overflow.Values) entry.Model.Dispose();
            _overflow.Clear();
            _cache.Clear();
        }
    }
}
```

(`ModelHandle` is unchanged from Phase 1.)

A deliberate consequence of this shape: once an in-use entry is evicted into `_overflow`, it
stays out of LRU management for the rest of its life — even after its ref count returns to
zero, it's disposed immediately rather than being given a chance to be reused from cache. This
trades a small amount of reuse opportunity (a model that overflowed and is briefly re-requested
right after its last reference drops would have to be reopened) for a *much* simpler and
provably-non-cascading eviction path. Given the soft cap is expected to engage rarely (see
below), this trade is the right one for this scope — the alternative (re-admitting a
zero-refcount overflow entry back into `_cache`, which itself might now evict something else)
reintroduces exactly the cascade shape this design avoids, for a benefit that only matters in
the rare case the soft cap fires at all.

### What happens when everything resident is in use

If capacity is reached and every resident entry has `RefCount > 0` (all in active use, none
evictable), the newly-admitted entry pushes the previous occupant into `_overflow`, and the
cache temporarily holds more than `capacity` total (resident + overflow) rather than blocking
the caller or throwing. This is a deliberate choice for this scope: introducing blocking (wait
for something to free up) is exactly the scheduling behavior this plan explicitly deferred to
the shelved full design (`024`, §4 — the Model Residency Scheduler). A soft cap that
occasionally exceeds its target under real concurrent pressure — observably, via the counters
below, not just a log line — is the right amount of mechanism for "stop test runs from piling
up 15 resident models," without building admission control to get there.

Given `parallelizeTestCollections: false` and the resulting near-total sequential execution in
the test host, "everything in use simultaneously" should be rare to nonexistent in practice —
tests dispose their handle (a `using` block, same discipline as `GgufModel.Open` already
requires today) before the next test starts. This soft-cap behavior is a safety net for the
edge case, not the expected steady state — which is exactly why it needs to be observable
rather than assumed: if measurement shows it firing often, that's a direct, quantified signal
the simplification isn't adequate, not something to discover by inference from memory graphs.

### Observability

```csharp
public readonly record struct SharedModelCacheStats(
    int Capacity,
    int CacheEntries,       // resident, capacity-managed by the SLRU
    int OverflowEntries,    // resident, in use, evicted from capacity management
    int LoadedModels,       // CacheEntries + OverflowEntries — total physically resident
    int ActiveReferences,   // sum of RefCount across both tables
    long Hits,
    long Misses,
    long Evictions,          // actual disposals
    long SoftCapExceeded);   // times an in-use entry overflowed
```

An immutable snapshot returned by `SharedModelCache.GetStats()`, not a live mutable object
exposed as a property. Two reasons, not one:

- **Correctness under concurrent callers.** Exposing a mutable `Stats` object directly would
  let a reader observe individual counters at different instants relative to concurrent
  `Acquire`/`Release` calls — not truly corrupted (the underlying counters are only ever
  mutated under `_lock`), but not a *consistent* snapshot either, since the reader isn't
  holding that lock while reading multiple fields. `GetStats()` takes the same lock every
  mutation goes through and returns a value-type copy, so every field in one returned
  `SharedModelCacheStats` reflects exactly the same instant.
- **Naming clarity** (a second, independent piece of review feedback): `ResidentModels` in an
  earlier draft was ambiguous between "managed by the capacity-bounded cache" and "physically
  loaded at all" — an overflow entry is resident in memory even though it isn't inside the
  SLRU's own bookkeeping. `CacheEntries` / `OverflowEntries` / `LoadedModels` makes the three
  distinct counts unambiguous.

Call `GetStats()` and log the result at test-run teardown. The signal this exists to catch:
`Capacity = 3, LoadedModels = 7, SoftCapExceeded = 142` is an unambiguous "the soft-cap
simplification is not adequate for this workload" result — a measured trigger to revisit
`024`, rather than a judgment call made from inference.

### Choosing a default capacity

Start at 3. Rationale: `SlruCache`'s own segmentation (25% probationary / 75% protected, per
`ExpertCache<T>`'s existing split) floors each segment at 1, so capacity 3 gives 1 probationary
+ 2 protected — small but non-degenerate. This is a starting point for measurement (§
Verification), not a tuned constant; adjust based on what the full-suite dogfood run in this
phase's verification actually shows.

## Test-suite migration (completing Phase 1's partial wiring)

Phase 1 already migrated call sites to `SharedModelCacheFixture.Instance.Acquire(path)`. No
further test-file changes are needed for this phase — capacity/eviction is entirely internal
to `SharedModelCache`. This phase's work is the cache-internals change above, plus the
verification methodology below.

## Verification

Two measurements, deliberately not relying on any change to test execution order (this phase
adds no scheduler — the point is that bounded-capacity LRU reuse should help regardless of
what order tests happen to run in):

1. **Full, unmodified `Tests.ForwardPass` run**, current default (effectively arbitrary, mostly
   filename-order) test ordering. Record:

   | Metric | Before (measured) | After (target) |
   |---|---|---|
   | Peak working set | 59.5 GB | — (see note) |
   | Physical `GgufModel.Open` calls | ~40 (one per model-loading test file) | ≈ number of distinct models actually used in a given window ≤ capacity, not ≈ 40 |
   | Total suite runtime | ~20+ min (full run) | — |

   Note on peak working set: this phase bounds *distinct resident models* to `capacity`, it
   does not claim to reproduce the exact pre-investigation 59.5 GB number under identical
   conditions — that number was influenced by at least one confirmed concurrent-process
   confound (an unrelated IDE test-host process running at the same time; see the
   investigation this plan traces back to). Treat 59.5 GB as motivating context, not a strict
   regression-test target; the real target is bounding resident-model count, which the
   physical-open-call metric measures directly and unambiguously.

2. **9-file/1-model bisection batch** (same set used in Phase 1's verification) as a
   regression check that Phase 2's capacity/eviction logic didn't reintroduce Phase 1's fixed
   problem — this batch only ever touches one distinct model, so `GetStats().SoftCapExceeded`
   should read exactly 0 and `GetStats().Evictions` should read 0 throughout (capacity 3
   comfortably fits one distinct model), meaning it should show exactly the same "one physical
   open, N cache hits" behavior Phase 1 already validated. If this batch's numbers regress
   relative to Phase 1 alone, or if `SoftCapExceeded`/`Evictions` are nonzero here, the eviction
   logic has a bug — this specific batch should never touch the eviction path at all — not a
   capacity-tuning issue.

If (1) doesn't show physical opens dropping well below the ~40 raw call-site count, check
`GetStats()` before assuming the design has a gap: a high `SoftCapExceeded` count means the
soft cap is genuinely being exercised (a real, measured signal — see "Observability" above, and
`027`'s note on when that becomes a trigger to revisit `024`), whereas a high `Misses` count
with low `SoftCapExceeded` points at a bug in the hit path instead. Either way, the fix is
diagnosing what `GetStats()` shows, not adding scheduling complexity pre-emptively.

## Implementation checklist

Concrete acceptance criteria for whoever builds this (both phases; from two rounds of external
review, condensed):

- [ ] `ModelHandle` is a class, not a struct (Phase 1) — copying the reference must not create
      a second logical acquisition.
- [ ] `Acquire`, `Release`, and eviction (`HandleEviction`) all execute under the same `_lock`
      — no code path mutates `_cache`, `_overflow`, or any `Entry.RefCount` outside it.
- [ ] `HandleEviction` never calls `_cache.Put` (directly or transitively) — grep for this if
      the implementation drifts from the sketch above; it's the one invariant this whole design
      depends on to avoid the cascade.
- [ ] `Dispose()` disposes every model in both `_cache` and `_overflow`, not just one.
- [ ] `GetStats()` takes the same lock as every mutation and returns a value-type snapshot —
      no method or property hands out a live, externally-mutable stats object.
- [ ] Tests exist for: double `Dispose()` on the same `ModelHandle` (must be a no-op, not throw
      or double-release); releasing a handle to a model that has since been evicted into
      `_overflow`, then releasing the last reference (must dispose exactly once); acquiring a
      model that's currently in `_overflow` while another handle to it is still held (ref count
      must go to 2, not re-enter the SLRU); a full-suite run's final `SoftCapExceeded` and
      `Evictions` counts logged and asserted against expected bounds, not just eyeballed.
