> **ARCHIVED, 2026-08-15.** Implemented as an `IInferenceSession`/`InferenceSession`-era
> Sessions-layer feature (confirmed against source, not just this document's own claim). That
> whole lineage is being superseded by the `HotSession` architecture — see
> [028](028-inference-session-to-hotsession-migration-plan.md) for current migration status
> (Phases 1-3 done) and [030](../030-delete-inferencesession-todo.md) for the still-open
> deletion of the legacy `InferenceSession`/`InferenceRuntime` types once HotSession fully
> replaces them. Carried forward: nothing beyond what 028/030 already track in
> [00-current-work.md](../00-current-work.md).

---

# Implementation Plan — Zero-Copy Session Multiverse / Parallel Branching (`Plan 008`)

## Objective

Add native zero-copy session branching to OpenTail.Stingray.

The feature allows an existing `IInferenceSession` to create multiple independent child branches from the exact current inference state without re-prefilling the prompt and without copying the physical KV pages.

Example:

```csharp
var branches = await session.GenerateBranchesAsync(
    count: 4,
    options);
```

The resulting branches must:

- share the parent's existing physical KV prefix pages;
- have independent logical KV sequence state;
- use copy-on-write when a branch modifies shared pages;
- preserve independent token histories;
- be independently resumable/disposable;
- never corrupt the parent or sibling branches;
- integrate with existing session suspension/resumption;
- work with existing sampling/speculative-generation infrastructure.

The initial implementation should focus on **correct zero-copy branching**, not a new scheduling system.

---

# 1. What "zero-copy" means

The feature must not do this:

```text
Parent KV
   ↓
copy KV to Branch A
copy KV to Branch B
copy KV to Branch C
copy KV to Branch D
```

Instead:

```text
                    Parent
                      │
                      ▼
                 physical pages
                /      |      \
               /       |       \
          Branch A  Branch B  Branch C
               \       |       /
                \      |      /
                  shared prefix
```

Each logical sequence points at the same physical pages.

Physical pages are retained using the existing page ownership/ref-count machinery.

When a branch writes to a shared page:

```text
Shared Page 42
 refCount = 5

Branch B writes
       ↓
      CoW
       ↓
Branch B → Page 91

Parent/A/C/D → Page 42
```

No other branch is affected.

---

# 2. Important architectural constraint

**Do not introduce a new KV cache implementation.**

The implementation must use the existing:

- `IKvCache`;
- `CpuKvCache`;
- `CpuKvSequence`;
- physical page allocation;
- page reference counting;
- CoW behaviour;
- session lifecycle.

The purpose of this feature is to expose the existing physical CoW capability through a higher-level session API.

If the current `CpuKvSequence` or equivalent sequence object already has a clone/fork operation, reuse it.

If it does not, add the smallest primitive required there rather than implementing page copying in `SessionBranchingExtensions`.

---

# 3. Public API

Add a branching API to `IInferenceSession`.

Possible shape:

```csharp
Task<IReadOnlyList<IInferenceSession>> GenerateBranchesAsync(
    int count,
    GenerationOptions? options = null,
    CancellationToken cancellationToken = default);
```

However, the coding agent should inspect the current session API before committing to this exact signature.

An alternative that may fit the architecture better is:

```csharp
Task<IReadOnlyList<InferenceBranchResult>> GenerateBranchesAsync(
    int count,
    GenerationOptions options,
    CancellationToken cancellationToken = default);
```

where each result contains:

```csharp
public interface IInferenceBranch
{
    IInferenceSession Session { get; }

    IReadOnlyList<int> GeneratedTokens { get; }

    GenerationResult Result { get; }
}
```

**The existing API conventions should win.**

The important contract is:

> Create N independent child inference states from one existing session state without re-prefilling the shared prefix.

---

# 4. Separate "Fork" from "Generate"

Internally, implement this as two concepts:

```text
Fork()
  ↓
N zero-copy child sessions
  ↓
Generate()
  ↓
N independent completions
```

Do not make the low-level fork operation dependent on generation.

This is important because future features such as:

- Best-of-N;
- beam search;
- MCTS;
- self-consistency;
- tree search;
- candidate ranking;

may want to fork sessions without immediately generating.

A useful internal primitive is therefore conceptually:

```csharp
IInferenceSession Fork();
```

or:

```csharp
IInferenceSession CreateBranch();
```

Then:

```csharp
GenerateBranchesAsync()
```

becomes a convenience operation built on top.

---

# 5. Fork semantics

When a session is forked:

```text
Parent:
tokens = [1, 2, 3, 4, 5]
KV     = pages [10, 11]

Fork × 3:

Parent  → [10, 11]
BranchA → [10, 11]
BranchB → [10, 11]
BranchC → [10, 11]
```

All four logical sequences initially refer to the same physical pages.

Every physical page referenced by the new branch must acquire the appropriate reference/ownership count.

The fork operation must be **O(number of referenced pages)** in metadata/refcount work, not O(number of KV elements).

It must never copy the actual KV tensors.

---

# 6. Parent remains completely usable

After:

```csharp
var branches = session.CreateBranches(4);
```

the parent session must remain valid.

The following must be safe:

```text
Parent generates
Branch A generates
Branch B generates
Branch C is disposed
Branch D generates
```

in arbitrary order.

The parent must not become dependent on the lifetime of its children.

---

# 7. Branch independence

Each branch gets independent:

- token cursor;
- generated-token history;
- sampling state;
- RNG state;
- logical KV sequence;
- session metadata;
- generation cancellation;
- constraint state, if active;
- speculative decoding state, if active.

Do not share mutable generation state between branches.

In particular:

```text
Parent RNG
Branch A RNG
Branch B RNG
Branch C RNG
```

must not accidentally become one shared mutable RNG.

If deterministic branching is desired, derive independent RNG streams/seeds from the parent's state.

---

# 8. RNG semantics

Define this explicitly.

A sensible initial rule is:

```text
parent seed
    ↓
branch seed = deterministic derivation(parent seed, branch index)
```

For example:

```csharp
branchSeed = Hash(parentSeed, branchIndex);
```

The exact derivation is not important.

The important invariant is:

> Branches must not contend over one shared mutable random-number generator.

If the caller explicitly supplies a seed, preserve reproducibility.

---

# 9. Generation options

Each branch should begin with a copy of the parent's generation configuration, with caller-supplied options overriding where appropriate.

Do not share mutable `GenerationOptions` objects.

Conceptually:

```text
Parent options
      │
      ├── clone → Branch A
      ├── clone → Branch B
      ├── clone → Branch C
      └── clone → Branch D
```

Constraints are especially important here.

If the parent has an active `ConstraintEngine`, each branch must receive an **independent constraint state** representing the same current logical position.

Do not let Branch A's JSON parser state affect Branch B.

---

# 10. Speculative decoding compatibility

The existing `SpeculativeDecoder` must continue to work on branches.

A branch should be able to perform:

```text
Fork
  ↓
SpeculativeDecoder
  ↓
draft tokens
  ↓
verification
  ↓
CoW as needed
```

Do not implement a separate speculative mechanism for branches.

Reuse the existing decoder.

If the current speculative decoder owns mutable state, that state must be branch-local.

---

# 11. Prompt-lookup speculation compatibility

The existing prompt-lookup speculation must also work.

A branch inherits the same prompt/token history, so it should naturally be able to use the existing prompt lookup mechanism.

No prompt should be re-prefilled merely because the session was forked.

---

# 12. Prefix-cache interaction

The new Plan 005b prefix cache must remain completely independent of branch lifetime.

Consider:

```text
PrefixCache
     │
     ▼
Page 42
     ▲
     │
 ┌───┼──────────────┐
 │   │              │
Parent A    Branch B    Branch C
```

Disposing Branch B must release **only Branch B's reference**.

If the prefix cache owns Page 42, Page 42 remains alive.

Likewise, if the parent remains alive, the page remains alive.

Never make branch disposal call directly into prefix-cache eviction.

---

# 13. CoW invariant

This is the central correctness invariant.

Given:

```text
Parent → Page 42
Branch A → Page 42
Branch B → Page 42
```

and Branch A performs an operation requiring modification:

```text
Branch A
   ↓
write Page 42
```

the result must be:

```text
Parent  → Page 42
Branch B → Page 42
Branch A → Page 91
```

The contents of Page 42 must remain unchanged.

The CoW operation must occur at the existing `CpuKvSequence`/KV ownership layer.

Do not implement manual tensor copying in the session layer.

---

# 14. Branch generation

The basic convenience API should conceptually perform:

```csharp
var branches = session.CreateBranches(count);

await Task.WhenAll(
    branches.Select(branch =>
        branch.GenerateAsync(options)));
```

However:

> **Do not assume `Task.WhenAll` means actual model-level parallel execution.**

The implementation should reuse Stingray's existing execution model.

If the existing `ContinuousBatchingEngine` naturally supports concurrent sequences, let it batch them.

If it serialises execution, the feature is still valid: the important property is **zero-copy branching and zero prompt re-prefill**.

Do not introduce a scheduler merely to make this feature "parallel".

---

# 15. Avoid accidental serialisation

At the same time, don't accidentally put:

```csharp
lock (_session)
{
    await GenerateAsync(...);
}
```

around the entire branch-generation operation.

The parent session should not become a global lock protecting all children.

Use existing inference-engine concurrency rules.

The implementation should allow the underlying engine to process independent branch sequences according to its existing capabilities.

---

# 16. Branch count validation

Validate:

```text
count >= 1
```

and apply a reasonable upper bound if the existing runtime has one.

Do not silently create thousands of branches.

Possible behaviour:

```text
count = 0 → ArgumentOutOfRangeException
count < 0 → ArgumentOutOfRangeException
```

A count of 1 should still work and should have the same semantics as a single fork.

---

# 17. Fork from any valid session state

A branch should be possible from:

- newly completed prefill;
- partially generated session;
- resumed session;
- session containing shared prefix-cache pages.

The implementation should not assume the session is at token position zero or at the end of a prompt.

Example:

```text
Prompt
  ↓
Generate 10 tokens
  ↓
Fork × 4
  ↓
four alternative continuations
```

This is one of the main use cases.

---

# 18. Forking a suspended session

Define behaviour explicitly.

Preferred behaviour:

```text
Suspended session
       ↓
Fork()
       ↓
child branches inherit logical state
```

but the implementation may need to resume or materialise the necessary logical KV state.

Do **not** accidentally turn fork into a mechanism for permanently pinning every suspended page.

If the current suspension architecture makes direct fork-from-suspended unsafe, return a clear unsupported/error result rather than corrupting state.

The coding agent should inspect the existing `SessionState.Suspended` implementation and choose the safest architecture.

---

# 19. Branch disposal

Branches must be disposable independently.

Example:

```text
Parent
 ├── A
 ├── B
 ├── C
 └── D

Dispose B
Dispose D
Generate A
Generate C
Dispose Parent
Generate? → A/C remain valid if their lifecycle contract permits
```

Whether branches are allowed to outlive the parent should follow the existing session ownership model.

If the architecture requires parent ownership, enforce that explicitly.

What must never happen is a dangling physical-page reference.

---

# 20. Ref-count lifecycle

The intended lifecycle is:

```text
Parent owns Page P
        │
        ▼
Fork × 4
        │
        ├── Branch A retains P
        ├── Branch B retains P
        ├── Branch C retains P
        └── Branch D retains P
        │
        ▼
Parent releases P
        │
        ▼
P remains alive
        │
        ├── A releases
        ├── B releases
        ├── C releases
        └── D releases
        │
        ▼
P finally reusable
```

Add assertions/tests around this lifecycle.

---

# 21. Do not copy token history unnecessarily

Token IDs are cheap compared with KV tensors, but still avoid unnecessary repeated allocations if the current session architecture supports efficient persistent token history.

The branch may use:

```text
Parent token history
        +
branch-local suffix
```

rather than immediately copying the entire token array.

However, **correctness is more important than micro-optimising token-history storage**.

Do not build a persistent-vector data structure solely for this feature.

---

# 22. Result type

The generated branch result should expose enough information for Best-of-N callers.

For example:

```csharp
public sealed record BranchGenerationResult
{
    public int BranchIndex { get; init; }

    public IInferenceSession Session { get; init; } = default!;

    public IReadOnlyList<int> GeneratedTokens { get; init; } = default!;

    public GenerationResult Generation { get; init; } = default!;
}
```

The exact type should follow existing result conventions.

At minimum the caller needs:

- branch identity/index;
- generated output;
- generation metadata;
- access to the resulting branch/session if continued generation is desired.

---

# 23. Future Best-of-N compatibility

Do not implement ranking yet.

But ensure this works naturally:

```csharp
var branches = await session.GenerateBranchesAsync(8, options);

var best = branches
    .OrderByDescending(x => Score(x))
    .First();
```

The API should not force the caller to discard branch sessions immediately.

That enables future:

```text
Generate 8
    ↓
score
    ↓
keep best
    ↓
continue best
```

without another prefill.

---

# 24. Future MCTS compatibility

Likewise, don't implement MCTS.

But the low-level fork primitive should allow:

```text
root
 ├── fork A
 │    ├── fork A1
 │    └── fork A2
 │
 ├── fork B
 │    ├── fork B1
 │    └── fork B2
 │
 └── fork C
```

This is why `CreateBranch()` should remain conceptually independent from `GenerateBranchesAsync()`.

---

# 25. Thread safety

Do not assume an individual `IInferenceSession` is already safe for simultaneous operations.

Define the contract clearly.

Recommended:

> The parent session must not be concurrently mutated by `GenerateBranchesAsync()` while another operation is modifying the parent.

The fork operation should take a consistent snapshot of:

- token position;
- logical KV sequence;
- generation state;
- relevant session metadata.

After that, branches are independent.

If the existing session has an operation lock, use it only around the **fork snapshot**, not around all subsequent branch generation.

---

# 26. Snapshot boundary

Conceptually:

```text
Parent
  │
  ├── acquire session state lock
  │
  ├── snapshot logical state
  ├── retain physical pages
  ├── create child sequences
  │
  └── release lock
          │
          ▼
      branches run independently
```

This is the correct concurrency boundary.

---

# 27. Tests

Create:

```text
SessionBranchingTests.cs
```

with at least the following mandatory tests.

### Test 1 — ForkCreatesIndependentBranches

Fork a session into four branches.

Verify:

- four branches exist;
- all have identical initial token position;
- all reference the same physical prefix pages;
- no KV tensor data was copied.

---

### Test 2 — ForkDoesNotRePrefill

Instrument the forward/prefill path.

Verify:

```text
initial prompt prefill = 1
fork × 4
additional prompt prefills = 0
```

This is one of the feature's key promises.

---

### Test 3 — BranchGenerationIsIndependent

Generate different continuations on A/B/C/D.

Verify their token histories diverge independently.

---

### Test 4 — CopyOnWriteProtectsSiblings

Start with shared Page P.

Modify Branch A.

Verify:

```text
A → new page
B → original page
C → original page
Parent → original page
```

and verify original page contents remain unchanged.

---

### Test 5 — ParentUnaffectedByBranch

Generate from a branch and verify parent token history and KV state remain unchanged.

---

### Test 6 — ArbitraryBranchDisposal

Dispose branches in random order.

Verify:

- remaining branches remain valid;
- parent remains valid;
- page refcounts remain correct.

---

### Test 7 — ParentDisposalDoesNotCorruptChildren

Where the session lifecycle contract permits child survival, dispose parent first and verify branches remain valid.

If the architecture intentionally forbids this, assert the documented ownership rule instead.

---

### Test 8 — SharedPrefixCachePagesRemainValid

Fork from a session whose prefix contains pages retained by `RadixPrefixTree`.

Dispose/suspend branches and verify prefix-cache pages remain valid.

---

### Test 9 — BranchesCanBeNested

```text
Root
 ↓
A/B/C
 ↓
A1/A2
```

Verify nested branching works and refcounts remain correct.

---

### Test 10 — BranchRngIndependence

Verify branches have independent RNG state.

Two branches must not interfere with one another's sampling sequence.

---

### Test 11 — BranchConstraintsAreIndependent

If JSON/schema constraints are implemented:

```text
Parent
 ↓
A/B
```

modify/advance A's constraint state.

Verify B's constraint state is unchanged.

---

### Test 12 — SpeculativeDecodingWorksAfterFork

Fork, then enable existing speculative decoding.

Verify generation succeeds and KV correctness remains intact.

---

### Test 13 — PromptLookupWorksAfterFork

Fork from a session with usable prompt history.

Verify existing prompt-lookup speculation continues to work.

---

### Test 14 — CancellationDoesNotCorruptSiblings

Cancel Branch A.

Verify B/C/D continue successfully.

---

### Test 15 — ConcurrentBranchesRemainSafe

Generate multiple branches concurrently using the existing engine.

Verify:

- no page corruption;
- no refcount leaks;
- no invalid page reuse;
- all branch results are internally consistent.

---

# 28. Physical allocation test

Add a particularly useful instrumentation test.

Capture:

```text
physical pages allocated before fork
physical pages allocated immediately after fork
```

Expected:

```text
before = N
after fork = N
```

or only the small amount of metadata required by the implementation.

Then generate divergent branches.

Expected physical allocation grows only as CoW requires:

```text
Fork:
    +0 KV pages

Branch A writes:
    +X pages

Branch B writes:
    +Y pages
```

This proves the actual zero-copy property rather than merely testing equivalent results.

---

# 29. Refcount invariant test

At every stage verify:

```text
page.refCount == number of live owners
```

where observable through the existing cache/test instrumentation.

Particularly test:

```text
Parent
 + 4 branches
 + prefix cache
```

then release owners in arbitrary order.

No page may become reusable while any owner remains.

---

# 30. Error handling

If branch creation fails partway through:

```text
requested = 8
created = 5
failure
```

the implementation must clean up the five already-created branches and release their page references.

No partial fork may leak physical pages.

Prefer an atomic-looking public operation:

```text
success → all requested branches returned
failure → no leaked branches
```

unless existing API conventions dictate another result model.

---

# 31. Metrics

Add lightweight metrics if the existing metrics infrastructure supports them:

```text
SessionBranches.Created
SessionBranches.Completed
SessionBranches.Cancelled
SessionBranches.ForkFailures
SessionBranches.CoWPages
SessionBranches.SharedPages
```

The particularly useful metric is:

```text
CoWPages
```

because it demonstrates whether the multiverse is actually sharing KV effectively.

Do not create a new telemetry framework.

---

# 32. Performance / complexity expectation

This feature is **not a performance-tuning project**.

The desired complexity is:

```text
Fork:
O(number of referenced KV pages)
```

with:

```text
O(1) physical KV copying
```

meaning **zero KV tensor copying**.

Branch generation itself remains governed by the existing inference engine.

Do not promise that four branches automatically produce four-times throughput.

The feature's guaranteed benefit is:

> **No repeated prompt prefill and no physical copying of the shared KV prefix.**

Actual generation parallelism depends on the existing execution engine.

---

# 33. API ergonomics

The ideal user experience should be something like:

```csharp
var branches = await session.GenerateBranchesAsync(
    count: 8,
    options: new GenerationOptions
    {
        MaxTokens = 128,
        Temperature = 0.8f
    });

var best = branches
    .MaxBy(Score);
```

and:

```csharp
var child = session.CreateBranch();

var result = await child.GenerateAsync(...);
```

The second API is important because it makes the feature useful beyond Best-of-N.

---

# 34. Do not overbuild

Do NOT add:

- beam search;
- MCTS;
- candidate scoring;
- branch ranking;
- scheduler v2;
- special branch scheduler;
- new KV cache;
- new tensor-sharing layer;
- distributed branching.

Those can be future consumers of the branching primitive.

The feature is simply:

> **Fork the logical inference state while sharing physical KV pages.**

---

# 35. Acceptance criteria

The implementation is complete when:

- [ ] `IInferenceSession` exposes a native branch/fork capability.
- [ ] Branch creation performs no KV tensor copying.
- [ ] Existing physical CoW infrastructure is reused.
- [ ] Parent and children share existing physical prefix pages.
- [ ] Every branch receives independent logical KV state.
- [ ] Branch mutation triggers existing CoW behaviour.
- [ ] Parent remains unaffected.
- [ ] Siblings remain unaffected.
- [ ] Page reference counts remain correct.
- [ ] Branch disposal releases only branch-owned references.
- [ ] Prefix-cache page ownership remains independent.
- [ ] Branch RNG state is independent.
- [ ] Constraint state is independent where applicable.
- [ ] Existing speculative decoding works after branching.
- [ ] Existing prompt-lookup speculation works after branching.
- [ ] Nested branching works.
- [ ] Branch cancellation does not corrupt siblings.
- [ ] Partial branch creation cannot leak pages.
- [ ] Existing continuous-batching/inference execution is reused.
- [ ] No new scheduler is introduced.
- [ ] Unit/integration tests pass.
- [ ] Full solution test suite passes.
- [ ] Release build passes.

---

# 36. Definition of done

The final architecture should look like:

```text
                         Parent Session
                              │
                       logical KV sequence
                              │
                       physical KV pages
                              │
                  ┌───────────┼───────────┐
                  │           │           │
                  ▼           ▼           ▼
              Branch A    Branch B    Branch C
                  │           │           │
              shared       shared       shared
               pages        pages        pages
                  │           │           │
               generate     generate     generate
                  │           │           │
               CoW only    CoW only    CoW only
             when needed  when needed  when needed
```

The key invariant is:

> **Forking creates independent logical inference states over shared physical KV pages. No prompt re-prefill and no KV tensor copy occurs during the fork. Physical divergence occurs only when a branch actually needs to modify shared state.**

---

## Implementation flexibility

The API signatures and C# snippets in this plan are **conceptual guidance, not mandatory code**.

The coding agent should inspect the existing:

- `IInferenceSession`;
- `InferenceSession`;
- `CpuKvSequence`;
- `CpuKvCache`;
- `IKvCache`;
- `InferenceRuntime`;
- `ContinuousBatchingEngine`;
- session suspension/resumption;
- speculative decoder;
- prefix cache;

before deciding the exact implementation.

If the existing code provides a cleaner primitive — particularly an existing sequence clone/fork operation or page-retention mechanism — **use that instead of adding another abstraction**.

The implementation should be as small as possible, but **correctness and ownership semantics take precedence over achieving an arbitrary "~80 lines" target**.

The "~80 lines" estimate is therefore not an acceptance criterion.