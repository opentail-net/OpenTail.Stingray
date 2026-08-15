> **ARCHIVED, 2026-08-15.** Implemented as an `IInferenceSession`/`InferenceSession`-era
> Sessions-layer feature (confirmed against source, not just this document's own claim). That
> whole lineage is being superseded by the `HotSession` architecture — see
> [028](028-inference-session-to-hotsession-migration-plan.md) for current migration status
> (Phases 1-3 done) and [030](../030-delete-inferencesession-todo.md) for the still-open
> deletion of the legacy `InferenceSession`/`InferenceRuntime` types once HotSession fully
> replaces them. Carried forward: nothing beyond what 028/030 already track in
> [00-current-work.md](../00-current-work.md).

---

# Implementation Plan — Plan 008: Zero-Copy Session Multiverse & Parallel Branching

## Objective

Add native zero-copy session branching to Stingray.

An `IInferenceSession` should be able to create one or more independent child sessions from its current state without copying the existing KV tensors:

```csharp
var branch = session.Fork();
```

and:

```csharp
var branches = session.ForkMany(4);
```

Each branch initially shares the parent's physical KV pages through the existing paged Copy-on-Write (`CoW`) infrastructure.

As branches diverge, only branch-specific KV pages consume additional physical memory.

The feature must build on the existing:

- `CpuKvSequence`
- physical KV pages
- page reference counting
- Copy-on-Write
- `TruncateTo`
- session token history
- transactional rollback
- session lifecycle/state machinery
- `KvMemoryGovernor`

**Do not introduce another KV sharing mechanism.**

---

# 1. Core Design Principle

A fork is a **logical session clone**, not a KV memory clone.

Given:

```text
Parent
Tokens: [0 ... 3999]
KV:     [Page 10, Page 11, Page 12, Page 13]
```

calling:

```csharp
var branch = session.Fork();
```

should produce:

```text
Parent  ───────┐
               ├── Page 10
Branch ────────┤── Page 11
               ├── Page 12
               └── Page 13
```

The physical pages are shared.

Reference counts increase accordingly.

No KV tensor data should be copied.

---

# 2. Public Session API

Add the smallest useful API.

Prefer:

```csharp
IInferenceSession Fork();
```

and:

```csharp
IReadOnlyList<IInferenceSession> ForkMany(int count);
```

If the existing API conventions favour asynchronous lifecycle methods, `Fork()` itself should nevertheless remain synchronous if it performs no model computation or I/O.

The operation should be effectively:

```text
capture session state
retain shared KV pages
create child session
return child
```

No model forward pass is required.

---

# 3. Fork Semantics

A child must inherit the parent's current logical state:

- model identity
- model/KV configuration
- token history
- sampling configuration where appropriate
- KV sequence
- relevant session metadata required for continuation

The child must **not** inherit mutable lifecycle state such as:

- `Disposed`
- `Suspended`
- `Generating`
- active generation operations
- cancellation state
- pending tool execution
- pending asynchronous operations

A newly-created branch should start in the normal ready/usable state.

The exact state copied should follow the existing `InferenceSession` architecture rather than introducing a second state model.

---

# 4. Fork Safety

Do not allow arbitrary forking while the parent is in the middle of a state mutation.

At minimum, reject or safely prevent:

```text
Fork while GenerateAsync is mutating KV
Fork while Prefill is running
Fork while AppendAsync is mutating the sequence
Fork while TruncateTo is running
Fork while ResumeAsync is reconstructing KV
Fork while DisposeAsync is releasing KV
```

Use the existing session synchronization/state guards.

**Do not add a new locking architecture if the existing session gate already provides the necessary serialization.**

A clean rule is:

> A fork captures a stable session checkpoint.

---

# 5. Atomic Fork Snapshot

The fork operation should capture the logical state and KV sequence as one consistent snapshot.

Conceptually:

```csharp
lock (sessionGate)
{
    EnsureForkable();

    var tokenSnapshot = _tokenHistory.Snapshot();
    var kvSnapshot = _kvSequence.Fork();

    return CreateChild(tokenSnapshot, kvSnapshot);
}
```

The exact implementation may differ.

The important invariant is:

```text
Token history position
        ==
KV logical position
        ==
forward-pass/session checkpoint
```

at the instant the branch is created.

Do not permit a branch to observe half of a parent mutation.

---

# 6. Physical KV Fork

Extend `CpuKvSequence` only if necessary.

The desired operation is conceptually:

```csharp
CpuKvSequence Fork()
```

which:

1. Creates a new logical sequence.
2. Copies the page-ID references.
3. Calls `RetainPage()` for every shared physical page.
4. Preserves the logical token length.
5. Does NOT copy page contents.

For example:

```text
Parent pages:

[42, 43, 44, 45]

Fork:

Child pages:

[42, 43, 44, 45]

Reference counts:

42: 2
43: 2
44: 2
45: 2
```

The operation must be transactional.

If retaining page references fails part-way through:

```text
Retain 42 ✓
Retain 43 ✓
Retain 44 ✗
```

release everything already retained before throwing.

No partially-created child may escape.

---

# 7. Copy-on-Write After Fork

This feature must rely on the existing CoW implementation.

Example:

```text
Parent:
    [42,43,44]

Branch:
    [42,43,44]
```

Branch generates enough tokens to require a new page:

```text
Parent:
    [42,43,44]

Branch:
    [42,43,44,51]
```

No parent KV contents may change.

If mutation of an existing shared page is ever required, the existing CoW mechanism must allocate a private page before mutation.

**Do not implement branch-specific copying in `Fork()`.**

Fork only establishes sharing.

---

# 8. Token History Isolation

The child must receive an independent token-history representation.

Do not allow:

```csharp
parent._tokenHistory.Add(...)
```

to modify the child's history.

Likewise:

```csharp
child._tokenHistory.Add(...)
```

must never modify the parent's history.

A practical approach is:

```text
KV pages:
    shared / CoW

Token history:
    logical copy / immutable snapshot
```

Token history is comparatively small compared with KV tensors, so copying it is acceptable.

The "zero-copy" claim specifically applies to **physical KV memory**.

---

# 9. Tool State Isolation

If the parent has an outstanding tool call or tool continuation, the fork must not accidentally share mutable tool execution state.

Prefer:

```text
Parent:
    ToolCall A
    Tool execution state A

Branch:
    independent tool-call state
```

If the current session model does not store mutable tool execution state, document that the branch inherits only the completed logical conversation state.

Do not copy active asynchronous tool operations into a branch.

---

# 10. Sampling State

This requires particular care.

If sampling uses mutable RNG state, two branches should not accidentally advance the same RNG instance.

Each branch should have independent sampling/RNG state.

A suitable approach is:

```text
Parent RNG state
       │
       ├── Branch A RNG
       ├── Branch B RNG
       ├── Branch C RNG
       └── Branch D RNG
```

The exact seed derivation is implementation-dependent.

A simple deterministic approach is acceptable, for example deriving branch seeds from:

```text
parent seed + branch index
```

Do not introduce a complex random-stream framework.

The important invariant is:

> Generating branch A must not alter branch B's RNG state.

---

# 11. `ForkMany`

Implement:

```csharp
IReadOnlyList<IInferenceSession> ForkMany(int count)
```

as a convenience operation over the same underlying mechanism.

Validate:

```text
count > 0
```

and apply a sensible configurable or documented maximum if the existing architecture has a branch limit.

Do not silently create thousands of branches.

For example:

```csharp
var branches = session.ForkMany(4);
```

should create four children that all share the parent's existing physical KV pages.

---

# 12. ForkMany Must Be Transactional

If creating branch 3 of 4 fails:

```text
Branch 1 ✓
Branch 2 ✓
Branch 3 ✗
Branch 4 not attempted
```

the already-created branches must be disposed/released before the exception escapes.

The parent must remain completely unchanged.

This is important because a large `ForkMany()` operation may fail due to memory/resource limits.

Conceptually:

```csharp
var created = new List<IInferenceSession>();

try
{
    for (...)
        created.Add(Fork());
}
catch
{
    foreach (var branch in created)
        await branch.DisposeAsync();

    throw;
}
```

The actual implementation may use a more efficient synchronous cleanup if appropriate.

---

# 13. Branch Disposal

Disposing a branch must release only its references.

Example:

```text
Parent ── Page 42
Branch ── Page 42

RefCount = 2
```

After:

```csharp
await branch.DisposeAsync();
```

the result must be:

```text
Parent ── Page 42

RefCount = 1
```

The physical page must NOT be returned to the free pool until all references have gone.

This must use the existing reference-counting implementation.

---

# 14. Parent Disposal Before Branch

The reverse ordering must also work:

```text
Parent ── Page 42
Branch ── Page 42

Dispose(parent)

Branch ── Page 42
RefCount = 1
```

The branch must remain completely usable.

This should be explicitly tested.

---

# 15. Branch Disposal Before Parent

Likewise:

```text
Dispose(branch)

Parent ── Page 42
RefCount = 1
```

The parent remains valid.

This should also be tested.

---

# 16. Interaction With Prefix Cache

A forked session may contain pages which are also retained by `RadixPrefixTree`.

Reference counting must remain correct across all owners.

For example:

```text
Prefix cache ──┐
Parent ────────┼── Page 42
Branch ────────┘

RefCount = 3
```

Disposing the branch must only remove its reference.

The fork implementation must not know about prefix-cache ownership.

This is deliberately an ownership concern of `IKvCache`.

---

# 17. Interaction With Memory Governor

The existing `KvMemoryGovernor` should treat branches as ordinary sessions.

Do not add a separate branch-specific eviction mechanism.

However, branches are likely to be highly disposable and may become idle quickly.

Therefore the governor should naturally be able to reclaim them through the existing:

```text
idle time
+
physical page footprint
+
pressure
```

policy.

Do not modify the governor unless the existing session enumeration cannot see branches.

---

# 18. Suspended Parent / Branches

Do not allow ambiguous fork semantics from a suspended session.

Prefer:

```text
Suspended session
    ↓
Fork()
    ↓
reject
```

with a clear exception indicating that the session must be resumed first.

Likewise, don't allow a branch to be forked while it is itself suspended.

This avoids accidentally creating a complex "fork a snapshot of a snapshot" state machine.

It can be added later if genuinely useful.

---

# 19. Session Identity

Every branch needs a unique session identifier.

Useful metadata:

```text
ParentSessionId
BranchId
```

but keep this lightweight.

For example:

```csharp
public Guid SessionId { get; }
public Guid? ParentSessionId { get; }
```

This is useful for debugging MCTS/tree searches without requiring a separate branch-management subsystem.

A branch should not automatically form a permanent parent/child ownership relationship.

The parent can be disposed while the branch remains alive.

---

# 20. Branching Extension Helpers

If appropriate, provide:

```csharp
public static class SessionBranchingExtensions
{
    public static IInferenceSession Fork(
        this IInferenceSession session);

    public static IReadOnlyList<IInferenceSession> ForkMany(
        this IInferenceSession session,
        int count);
}
```

However:

**Prefer adding methods directly to `IInferenceSession` if that interface is already the canonical public session abstraction.**

Use an extension class only if changing the public interface would create unnecessary compatibility/API churn.

The coding agent is explicitly allowed to choose the cleaner existing project convention.

---

# 21. Optional `GenerateBranchesAsync`

Do NOT initially build a large "branch execution framework".

Once `ForkMany()` exists, callers can naturally write:

```csharp
var branches = session.ForkMany(4);

var results = await Task.WhenAll(
    branches.Select(b => b.GenerateAsync(options)));
```

This gives the desired parallel execution capability without introducing an orchestration layer into Stingray.

If the existing API makes this awkward, a tiny helper may be added later.

For this plan, **Fork/ForkMany are the core feature.**

---

# 22. Example Usage

### Best-of-N

```csharp
var branches = session.ForkMany(4);

var candidates = await Task.WhenAll(
    branches.Select(async branch =>
    {
        try
        {
            return await branch.GenerateAsync(sampling);
        }
        finally
        {
            await branch.DisposeAsync();
        }
    }));
```

All four branches initially share the prompt KV.

Only newly-generated KV consumes additional physical pages.

---

### Tree-of-Thought

Conceptually:

```text
                    Root
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
        Path A      Path B      Path C
          │           │
       Fork A1     Fork B1
       Fork A2     Fork B2
```

Every child can share all ancestor KV pages until it diverges.

This is precisely the workload the paged CoW architecture is designed to support.

---

# 23. Memory Accounting

Do not claim literally "zero memory" for a fork.

The correct claim is:

> **Zero additional physical KV tensor memory at the moment of the fork.**

There is still small logical overhead for:

- session object
- token-history representation
- page-ID references
- metadata
- RNG state

For example:

```text
4,000-token prompt
4 branches

Before fork:
    4,000 tokens of physical KV

Immediately after fork:
    still approximately 4,000 tokens of physical KV
    + small logical session/page-reference overhead

After each branch generates 50 new tokens:
    approximately:
        4,000 shared
      + 4 × 50 branch-specific
      = 4,200 token-equivalents of physical KV
```

This is the correct way to document the advantage.

---

# 24. Mandatory Invariant Tests

Add:

### `SessionForkTests.cs`

#### Test 1 — ForkSharesPhysicalPages

Verify:

```text
Parent pages == Branch pages
```

and no new physical KV buffers are allocated.

---

#### Test 2 — ForkRetainsPageReferences

Verify every shared physical page has its reference count incremented.

---

#### Test 3 — BranchGenerationDoesNotCorruptParent

Fork → generate on branch → verify parent's:

- token history
- KV contents
- logical length

remain unchanged.

---

#### Test 4 — ParentGenerationDoesNotCorruptBranch

Fork → generate on parent → verify branch remains unchanged.

---

#### Test 5 — BranchDisposalReleasesOnlyBranchReferences

Dispose branch and verify parent's pages remain live.

---

#### Test 6 — ParentDisposalDoesNotInvalidateBranch

Dispose parent and continue generating on branch.

This is a particularly important ownership test.

---

#### Test 7 — ForkManySharesOnePhysicalPrompt

Create four branches and verify that all four initially reference the same physical pages.

---

#### Test 8 — ForkManyIsTransactional

Force branch creation failure part-way through and verify all previously-created branches are cleaned up.

---

#### Test 9 — TokenHistoryIsIndependent

Mutate parent and branch independently and verify their histories diverge correctly.

---

#### Test 10 — SamplingStateIsIndependent

Verify generating one branch does not mutate another branch's RNG state.

---

#### Test 11 — SharedPageCopyOnWrite

Fork → force a branch mutation requiring CoW → verify the parent still sees the original physical page/content.

---

#### Test 12 — ArbitraryDisposalOrder

Create:

```text
Parent
 ├── A
 ├── B
 └── C
```

Dispose in several different orders and verify all physical page reference counts eventually return to the correct baseline.

---

#### Test 13 — PrefixCacheOwnershipCoexists

If practical with the existing test infrastructure, retain a shared page through the prefix cache, fork a session, then dispose the branch/session/cache in different orders and verify the page is not prematurely reused.

---

#### Test 14 — ForkRejectedDuringUnsafeMutation

Attempt to fork while the session is actively generating/prefilling and verify the operation is rejected or safely serialized according to the existing session-state contract.

---

#### Test 15 — SuspendedSessionCannotFork

Verify a suspended session cannot be forked.

---

# 25. Concurrency Tests

Add at least one test where several branches generate concurrently:

```csharp
var branches = session.ForkMany(4);

await Task.WhenAll(
    branches.Select(b => b.GenerateAsync(...)));
```

Verify:

- no page corruption
- no reference-count leaks
- no parent mutation
- no cross-branch token-history corruption
- all branches can be disposed cleanly

Do not require deterministic generated text unless the existing sampling infrastructure guarantees it.

The test is primarily checking **state isolation and memory correctness**.

---

# 26. Performance Smoke Test

Add a lightweight test/benchmark showing that:

```text
Fork()
```

does not scale with the number of KV tokens in the session.

For example, compare:

```text
1,000 tokens
10,000 tokens
50,000 tokens
```

The fork operation should remain approximately constant-time with respect to KV tensor copying because **no KV tensor data is copied**.

Do not turn this into a performance-tuning project.

The purpose is simply to catch an accidental implementation that copies KV buffers.

---

# 27. Non-Goals

Do NOT implement:

- MCTS itself
- Tree-of-Thought orchestration
- Best-of-N ranking
- branch scheduling
- branch scoring
- branch pruning
- speculative search algorithms
- a new scheduler
- distributed branching
- GPU-specific branching code
- a branch persistence format

Those belong above Stingray.

Stingray should provide the primitive:

> **"Give me another independent inference session starting from this exact state, sharing physical KV wherever possible."**

OpenTail or another caller can decide what to do with those branches.

---

# 28. Definition of Done

Plan 008 is complete when:

1. `IInferenceSession.Fork()` exists or an equivalent canonical API is provided.
2. `ForkMany(count)` exists.
3. Forking shares physical KV pages rather than copying tensors.
4. Physical page reference counts are correctly retained.
5. Branch token history is independent.
6. Branch sampling state is independent.
7. CoW protects the parent and sibling branches.
8. Parent and branches can be disposed in arbitrary order.
9. Parent can be disposed while a branch remains usable.
10. `ForkMany()` is transactional.
11. Unsafe session states cannot be forked incorrectly.
12. Suspended sessions cannot be forked.
13. Existing prefix-cache ownership remains correct.
14. Memory governor naturally sees branches as ordinary sessions.
15. Concurrent branch generation is safe.
16. Full test suite passes.
17. Release build passes.

## Architectural invariant

The most important invariant is:

> **Forking creates a new logical inference state while initially creating zero additional physical KV tensor storage.**

The implementation must preserve this property without introducing a second KV-sharing mechanism.

The coding agent is explicitly permitted to improve individual API names, helper placement, locking details, or snapshot implementation if a better approach fits the existing Stingray architecture. The above behaviour and ownership invariants are the contract; the exact internal implementation is not.