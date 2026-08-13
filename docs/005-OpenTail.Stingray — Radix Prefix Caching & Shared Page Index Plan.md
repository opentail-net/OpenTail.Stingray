# OpenTail.Stingray — Consolidated Radix Prefix Caching & Shared Page Index Plan (`Plan 005b Refined`)

## 1. Executive Summary & Purpose

This plan defines the architectural consolidation and evolution of prefix caching in `OpenTail.Stingray`.

All prefix retention is consolidated into a single, unified `IPrefixCacheIndex` powered by a page-aligned Radix Tree (`RadixPrefixTree`) over physical `IKvCache` pages.

---

## 2. Core Architectural Invariants

### 2.1 Model/KV Namespace & Token Radix Path
Prefix tree navigation cleanly separates namespace identity from token branching:
- **`PrefixCacheNamespace`**: `(ModelId, KvConfigHash)` defines the isolated root tree namespace.
- **Token Radix Path**: `ReadOnlySpan<int>` token IDs form the branching radix path within that specific namespace.

```text
PrefixCacheNamespace (ModelId + KvConfigHash)
          │
          ▼
        ROOT
         │
         ├── [1 ── 42 ── 91] (Page 17)
         │
         └── [1 ── 42 ── 33] (Page 19)
```

### 2.2 Atomic Ref-Count Acquisition During `MatchPrefix`
To eliminate lifetime concurrency races between lookup and eviction:
- `MatchPrefix()` invokes `RetainPage()` on every returned physical page **inside the tree lock** before returning the result.
- This guarantees returned physical pages in `PrefixMatchResult` are valid and owned by the caller even if a concurrent `EvictLruEntries()` removes the prefix tree node immediately afterward.

### 2.3 Page Ref-Count Retention & Page Reuse Safety
Physical pages indexed in `RadixPrefixTree` hold an explicit reference count:
1. **Publishing**: `Publish()` calls `RetainPage()` on all full physical pages inserted into the index.
2. **Page Ref-Count Guard**: While indexed, physical pages have `refCount >= 1`, making page ID reuse in `CpuKvCache._freePageIds` impossible.
3. **Eviction**: `EvictLruEntries()` removes tree nodes and calls `ReleasePage()`, allowing physical pages to be freed when all session handles drop.

### 2.4 Immutable & Validated Publishing Rule
Pages enter the global prefix cache **only** after:
- **Full**: Token position fills complete page boundaries ($N \times \text{PageSize}$).
- **Immutable**: Token sequence is committed.
- **Validated**: Forward pass prefill executed cleanly without cancellation or exception.

In-flight, partial, or failed prefill operations MUST NEVER be published. Partial tail pages remain private to the active sequence.

---

## 3. Consolidated API Contracts

```csharp
namespace OpenTail.Stingray.Engine;

public readonly record struct PrefixCacheNamespace(
    string ModelId,
    string KvConfigHash);

public readonly record struct PrefixCacheKey(
    PrefixCacheNamespace Namespace,
    ReadOnlyMemory<int> TokenIds);

public readonly record struct PrefixMatchResult(
    int MatchedTokenCount,
    int MatchedPageCount,
    ReadOnlyMemory<KvPageId> SharedPages);

public interface IPrefixCacheIndex : IDisposable
{
    /// <summary>
    /// Looks up the longest page-aligned prefix match for the given namespace and token sequence.
    /// Atomically retains ref-counts on matched physical pages before returning.
    /// </summary>
    PrefixMatchResult MatchPrefix(PrefixCacheNamespace ns, ReadOnlySpan<int> tokens);

    /// <summary>
    /// Publishes a completed, committed sequence of full physical pages from an IKvSequence into the prefix index.
    /// </summary>
    void Publish(PrefixCacheNamespace ns, IKvSequence sequence, int committedTokenCount);

    /// <summary>
    /// Removes a specific prefix entry and releases its page ref-counts.
    /// </summary>
    void Remove(PrefixCacheNamespace ns, ReadOnlySpan<int> tokens);

    /// <summary>
    /// Evicts unreferenced LRU prefix entries until target memory is reclaimed.
    /// </summary>
    int EvictLruEntries(int targetPagesToFree);
}
```

---

## 4. Mandatory Invariant Test Suite (10 Tests)

1. **Model Isolation Test**: Identical tokens for Model A vs Model B produce a cache miss.
2. **KV Configuration Isolation Test**: Identical tokens under different KV dtypes produce a cache miss.
3. **Longest Page-Aligned Prefix Match Test**: Requests matching 32, 64, or 96 tokens return the longest 96-token match.
4. **Page Reuse Safety Test (Full Lifecycle)**: Allocate page $P$ $\rightarrow$ publish $P$ $\rightarrow$ release session $\rightarrow$ allocate new page (assert $\neq P$) $\rightarrow$ evict prefix $\rightarrow$ allocate new page (assert $= P$).
5. **Arbitrary Release Order Test**: Releasing sessions before prefix cache (or vice versa across multiple sessions A/B/C) keeps page ref-counts clean.
6. **Failed Prefill Safety Test**: Failed or cancelled prefill operations NEVER publish pages to the prefix index.
7. **Copy-on-Write Isolation Test**: Mutating a session sharing prefix pages triggers CoW without altering shared prefix pages.
8. **Partial Page Privacy Test**: Incomplete tail pages (< 32 tokens) remain private to the session.
9. **Prefix Eviction & Page Release Test**: Evicting prefix entries cleanly releases physical page ref-counts.
10. **Atomic Match/Eviction Concurrency Test**: Concurrent `MatchPrefix` and `EvictLruEntries` does not return invalid page references.

---

## 5. Definition of Done

1. `IPrefixCacheIndex` and `RadixPrefixTree` implemented with namespace/token-radix path separation and atomic ref-count matching.
2. Consolidated `ContinuousBatchingEngine` and `InferenceRuntime` to use unified `IPrefixCacheIndex`.
3. All 10 mandatory invariant tests passing 100%.
4. Entire solution builds cleanly with **0 Warnings, 0 Errors**.
