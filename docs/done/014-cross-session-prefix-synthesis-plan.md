# Implementation Plan — Real-Time Cross-Session Prefix Synthesis (`Plan 014`)

## Objective

Implement a **zero-persistence, background cross-session prefix synthesis service** for OpenTail.Stingray.

The feature automatically discovers identical prompt prefixes already present in active sessions and publishes their existing physical KV pages into the existing `IPrefixCacheIndex` / `RadixPrefixTree`.

The next session containing the same prefix can therefore reuse the already-computed KV pages without re-prefilling those tokens.

### Core concept

Today:

```text
Session A
    │
    ├── System prompt ──→ KV pages
    └── User prompt   ──→ KV pages

Session B
    │
    ├── Same system prompt ──→ compute KV again
    └── Different user prompt
```

After Plan 014:

```text
Session A ──┐
            │
            ├── identical committed prefix
            │
Session B ──┘
            │
            ▼
     RadixPrefixTree
            │
            ▼
      shared KV pages
            │
       ┌────┴────┐
       ▼         ▼
   Session A  Session B
```

No disk persistence is introduced.

No prefix data is written to disk.

No cross-process shared memory is introduced.

Everything exists only for the lifetime of the running `InferenceRuntime`.

On process restart:

```text
Process exits
     ↓
RAM released
     ↓
RadixPrefixTree disappears
     ↓
next process starts empty
```

---

# 1. Architectural principle

**Do not implement another prefix cache.**

Plan 005b already provides:

- `IPrefixCacheIndex`;
- `RadixPrefixTree`;
- `PrefixCacheNamespace`;
- token-based radix matching;
- page-aligned prefix entries;
- physical page reference counting;
- atomic retention during lookup;
- immutable publishing;
- LRU eviction;
- model/KV configuration isolation.

Plan 014 should consume that infrastructure.

The new architecture should therefore be:

```text
                 InferenceRuntime
                       │
             ┌─────────┴─────────┐
             │                   │
             ▼                   ▼
       Active Sessions       PrefixIndex
             │                   │
             │                   │
             └───────┬───────────┘
                     │
                     ▼
       CrossSessionPrefixSynthesizer
                     │
             background scanner
                     │
                     ▼
             Publish matching
              committed pages
```

The synthesizer is a **producer of prefix-cache entries**, not a replacement for `RadixPrefixTree`.

---

# 2. New component

Add a new component in the appropriate engine/runtime layer, for example:

```text
CrossSessionPrefixSynthesizer.cs
```

Potential interface:

```csharp
public interface ICrossSessionPrefixSynthesizer : IAsyncDisposable
{
    Task StartAsync(CancellationToken cancellationToken);
    Task StopAsync(CancellationToken cancellationToken);
}
```

The exact interface may differ if the existing runtime has a better hosted/background-service abstraction.

The coding agent should inspect the repository first and use existing lifecycle patterns where possible.

---

# 3. Responsibilities

`CrossSessionPrefixSynthesizer` should do only four things:

1. discover active sessions;
2. identify reusable committed prefixes;
3. identify the physical KV pages backing those prefixes;
4. publish them to `IPrefixCacheIndex`.

It should **not**:

- perform model inference;
- mutate session token history;
- mutate existing KV pages;
- create a second cache;
- persist anything to disk;
- implement its own eviction;
- implement its own reference counting;
- bypass COW;
- attach pages to sessions itself.

The existing cache/session infrastructure remains authoritative.

---

# 4. Active-session discovery

The synthesizer needs a safe way to enumerate active sessions.

Prefer an existing session registry if Stingray already has one.

If not, introduce a small abstraction:

```csharp
public interface IActiveSessionRegistry
{
    IReadOnlyCollection<IInferenceSession> GetActiveSessions();
}
```

Avoid allowing the synthesizer to reach into private runtime collections.

The registry should provide a **snapshot**, not an enumeration that remains locked while synthesis occurs.

For example:

```text
lock runtime registry
      ↓
copy session references
      ↓
release lock
      ↓
scan snapshot
```

This prevents the background scanner from blocking normal inference.

---

# 5. Session state requirements

Only sessions that have a stable, committed prefix may participate.

The synthesizer must ignore:

- disposed sessions;
- failed sessions;
- cancelled sessions;
- sessions currently rolling back;
- sessions with invalid/incomplete KV state;
- speculative/uncommitted tokens;
- partially written physical pages.

A useful conceptual state is:

```text
Session
 ├── TokenHistory
 ├── CommittedTokenCount
 ├── KV sequence/page table
 └── lifecycle state
```

The synthesizer should work from the **committed boundary**, never from speculative state.

---

# 6. What constitutes a synthesizable prefix?

The safest initial rule is:

> A synthesizable prefix is a contiguous sequence of committed tokens beginning at token 0 and ending on a physical KV page boundary.

For example, with:

```text
PageSize = 16
```

valid candidates are:

```text
16 tokens
32 tokens
48 tokens
64 tokens
...
```

but not:

```text
7 tokens
25 tokens
37 tokens
```

The incomplete tail page remains session-private.

This follows the invariant established in Plan 005b.

---

# 7. Prefixes should come from the beginning of the session

Do not attempt arbitrary internal subsequence synthesis in this plan.

Good:

```text
[system][tools][instructions][user...]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
shared prefix
```

Not:

```text
[system][tools][instructions][user]
             ^^^^^^^^^^^^^
             arbitrary middle
```

The RadixPrefixTree is a **prefix cache**, not a general KV substring cache.

This keeps the feature safe and makes the relationship with normal prompt prefill explicit.

---

# 8. Namespace identity

Every synthesized prefix must be published under the correct:

```text
PrefixCacheNamespace
```

containing at least:

```text
ModelId
KvConfigHash
```

Do not infer namespace identity from token IDs.

Identical token IDs under different:

- models;
- KV dtypes;
- KV layouts;
- model revisions;
- incompatible cache configurations

must never share pages.

This is already enforced by Plan 005b and must remain an invariant.

---

# 9. Detecting matching sessions

The synthesizer does not need to compare every session with every other session naively.

A simple first implementation can group sessions by:

```text
PrefixCacheNamespace
```

then compare their committed token prefixes.

Conceptually:

```text
sessions
   │
   ├── Model A / KV config X
   │       ├── Session 1
   │       ├── Session 2
   │       └── Session 3
   │
   └── Model B / KV config Y
           ├── Session 4
           └── Session 5
```

Only sessions in the same namespace are candidates for sharing.

---

# 10. Prefer existing RadixPrefixTree matching

Do not build a second prefix-comparison tree inside the synthesizer.

The synthesizer can ask:

```text
Does this session's committed prefix already exist?
```

through:

```csharp
PrefixIndex.MatchPrefix(...)
```

If the prefix is already represented, there is nothing to synthesize.

The main new operation is therefore:

```text
discover → identify reusable pages → Publish()
```

---

# 11. Efficient discovery strategy

The synthesizer should preferably use a **candidate prefix fingerprint** to avoid repeatedly comparing complete token arrays.

For example:

```text
namespace
+
prefix token count
+
lightweight token hash
```

could identify potential duplicates.

But this is an optimisation, not a correctness mechanism.

If hashing is introduced:

```text
hash match
    ↓
exact token comparison
    ↓
confirmed identical prefix
```

Never treat a hash collision as a valid match.

If the current session/token infrastructure makes direct comparison cheap enough, a hash index is unnecessary for the first implementation.

---

# 12. How synthesis actually publishes pages

Suppose:

```text
Session A:

Tokens:
[1 2 3 4 5 6 7 8 ...]
         ↓
Pages:
P10 P11 P12
```

and Session B has:

```text
[1 2 3 4 5 6 7 8 ...]
```

The synthesizer identifies:

```text
Token prefix = identical
Physical pages = P10 P11 P12
```

Then:

```text
PrefixIndex.Publish(
    namespace,
    tokens,
    [P10, P11, P12]);
```

`RadixPrefixTree.Publish()` remains responsible for:

```text
RetainPage(P10)
RetainPage(P11)
RetainPage(P12)
```

inside its existing publishing/locking semantics.

The synthesizer must **not manually manipulate page ref-counts** unless the existing API explicitly requires it.

---

# 13. Important reference-count rule

The synthesizer must never publish a page without retaining it.

The ownership model should remain:

```text
Session owns page
       +
PrefixIndex owns page
       +
other sessions may own page
```

For example:

```text
Before:

Page 42
refCount = 1
Session A owns it

After publishing:

Page 42
refCount = 2

Session A
    │
    └── ownership #1

PrefixIndex
    │
    └── ownership #2
```

If Session A disappears:

```text
refCount = 1
```

The page remains alive because the prefix cache owns it.

If the prefix is evicted:

```text
refCount = 0
```

and the page can return to the free pool.

---

# 14. Critical race: session disposal

The synthesizer operates in the background, so this race must be explicitly handled.

Potential sequence:

```text
Synthesizer reads Session A
        ↓
Session A is disposed
        ↓
its pages are released
        ↓
Synthesizer attempts Publish()
```

This must not result in publishing stale/reused page IDs.

Therefore the synthesizer needs an atomic way to obtain a **stable page ownership snapshot** from the session.

Prefer an existing session/KV API if available.

Conceptually:

```csharp
using var snapshot = session.AcquireCommittedKvSnapshot();

if (!snapshot.IsValid)
    return;

PrefixIndex.Publish(
    namespace,
    snapshot.TokenIds,
    snapshot.FullPages);
```

The snapshot should hold enough ownership to prevent the physical pages disappearing while they are being published.

If the existing `CpuKvSequence`/session API already provides equivalent guarantees, use that instead of adding another abstraction.

---

# 15. Atomic publication

The ideal sequence is:

```text
1. Acquire stable committed session snapshot
2. Validate page alignment
3. Validate token/page correspondence
4. PrefixIndex.Publish()
5. PrefixIndex retains pages
6. Release temporary snapshot ownership
```

This creates the ownership transition:

```text
session snapshot
      ↓
PrefixIndex ownership
```

without an unsafe gap.

---

# 16. Do not publish partial pages

For:

```text
PageSize = 16
CommittedTokens = 42
```

only:

```text
16
32
```

are synthesizable.

The final:

```text
tokens 33–42
```

remain private.

The prefix cache therefore never references an incompletely populated physical page.

---

# 17. Minimum prefix length

Avoid synthesizing tiny prefixes.

For example, configure:

```text
MinimumSynthesizedTokens = 1–2 pages
```

or use an existing prefix-cache threshold.

The reason is simple:

```text
tiny prefix
   ↓
little memory saved
   ↓
cache metadata/refcount overhead
```

System prompts and tool declarations are generally much longer than a single page anyway.

Make the threshold configurable.

---

# 18. Background scan interval

Do not scan continuously in a tight loop.

Use a configurable interval, conceptually:

```text
ScanInterval = 250–1000 ms
```

The exact default should be chosen after considering existing runtime scheduling.

A simpler initial implementation might use:

```text
PeriodicTimer
```

with:

```csharp
while (await timer.WaitForNextTickAsync(ct))
{
    await SynthesizeOnceAsync(ct);
}
```

Do not create a custom scheduler for this feature.

---

# 19. Work budget

The synthesizer must not monopolise CPU while inference is running.

Give each scan a bounded budget.

For example:

```text
MaxSessionsPerScan
MaxPrefixesPerScan
MaxWorkMilliseconds
```

The exact controls can be kept minimal initially.

The important invariant is:

> Background synthesis must be opportunistic and must never block foreground inference.

---

# 20. Backpressure

If there are hundreds or thousands of active sessions, the scanner must not repeatedly process every session in full.

Use a lightweight dirty/recent-session strategy if the existing architecture supports it.

For example:

```text
new/changed session
       ↓
mark prefix dirty
       ↓
background synthesizer
       ↓
process once
       ↓
clear dirty state
```

This is preferable to repeatedly rescanning unchanged sessions.

If implementing dirty tracking adds substantial complexity, begin with bounded periodic scanning and leave dirty tracking as a future optimisation.

---

# 21. Trigger on useful events

If the session architecture already exposes events such as:

```text
SessionCreated
PromptCommitted
PrefillCompleted
SessionDisposed
```

the synthesizer can use them.

The ideal trigger is:

```text
successful prefill / committed prefix
              ↓
notify synthesizer
              ↓
candidate becomes available
```

This can make sharing happen almost immediately.

However, the event should enqueue work rather than perform synthesis synchronously inside the inference path.

---

# 22. Recommended hybrid design

A particularly good design would be:

```text
Successful prefill
      │
      ▼
"prefix available" notification
      │
      ▼
bounded background queue
      │
      ▼
CrossSessionPrefixSynthesizer
      │
      ▼
RadixPrefixTree.Publish()
```

This means there is no need to scan the whole runtime after every token.

If no suitable event infrastructure exists, use the periodic scanner.

---

# 23. Do not synthesise during active mutation

A session that is currently being extended should not be used as an unstable source.

Prefer a source snapshot taken after:

```text
prefill committed
```

or after a stable committed generation boundary.

The synthesizer does not need to react to every generated token.

For example:

```text
System prompt + tools + instructions
        ↓
prefill complete
        ↓
eligible for synthesis
```

is sufficient.

---

# 24. What prefixes are especially valuable?

The synthesizer should naturally discover things like:

```text
System prompt
+
Tool definitions
+
Agent instructions
+
Safety/instruction headers
+
Common RAG scaffolding
```

Example:

```text
Session A:
[System][Tools A][User A]

Session B:
[System][Tools A][User B]

Session C:
[System][Tools A][User C]
```

The first successful session can seed:

```text
[System][Tools A]
```

for the others.

The user-specific portion remains private.

---

# 25. Do not require semantic understanding

The synthesizer does not need to know that something is:

```text
system prompt
tool declaration
RAG document
```

It operates purely on exact token prefixes.

This is important.

Do not introduce an LLM, string classifier, semantic similarity model, or JSON parser.

Exact token equality is safer and cheaper.

---

# 26. Cross-user isolation

"Cross-session" does **not** mean bypassing application security.

The cache must only share KV for **exactly identical reusable token prefixes under the same model/KV namespace**.

It must not:

- infer similarity;
- share near-matches;
- share user-specific content merely because it looks similar;
- cross namespace boundaries;
- bypass application/session permissions.

For exact identical tokens, the physical KV content is identical by construction.

---

# 27. Sensitive/user-specific prefixes

The safest initial design is to make synthesis opt-in/eligible only for the **leading reusable prefix**.

Do not try to identify sensitive information heuristically.

If the application has metadata indicating:

```text
PrefixSharingAllowed = false
```

for a session, the synthesizer must honour it.

If such metadata does not currently exist, do not invent a complicated policy system in this plan.

The core engine should simply provide an explicit eligibility flag if needed.

---

# 28. COW interaction

No special COW implementation belongs in Plan 014.

The existing COW mechanism should already guarantee:

```text
Shared page
   │
   ├── Session A reads
   ├── Session B reads
   │
   └── mutation attempted
            ↓
          CoW
            ↓
      private physical page
```

The synthesizer should only publish immutable completed pages.

If a test fails here, fix the underlying COW/refcount implementation rather than adding special-case logic to the synthesizer.

---

# 29. Eviction interaction

The synthesizer must not implement eviction.

The existing:

```text
IPrefixCacheIndex.EvictLruEntries()
```

remains responsible for cache eviction.

After publication:

```text
PrefixIndex
    │
    └── page reference
```

is treated exactly like any other prefix-cache entry.

The memory governor can therefore eventually reclaim these pages through the existing prefix-cache mechanisms.

---

# 30. Duplicate publication

If two sessions discover the same prefix simultaneously:

```text
Session A ──┐
            ├── Publish(prefix X)
Session B ──┘
```

the index must remain correct.

The synthesizer does not need to solve this itself.

`RadixPrefixTree.Publish()` should be idempotent or safely handle an already-existing prefix.

Possible behaviour:

```text
existing entry
      ↓
do not create duplicate ownership
```

Avoid multiplying reference counts simply because the same logical cache entry was discovered twice.

The coding agent should inspect the existing `Publish()` semantics and add an invariant test if necessary.

---

# 31. Metrics

Add lightweight metrics.

For example:

```text
SynthesisScans
SessionsScanned
CandidatePrefixes
DuplicatePrefixes
PublishedPrefixes
PublishedPages
SkippedPartialPages
SkippedUnstableSessions
```

Most useful:

```text
PublishedPages
```

and:

```text
SynthesisHits
```

These let us establish whether the feature actually helps.

Do not introduce a new metrics framework if Stingray already has one.

---

# 32. Observability example

A useful debug/diagnostic event might say:

```text
Prefix synthesis:
namespace=ModelA/KVConfig42
tokens=512
pages=32
sourceSession=abc
result=published
```

Avoid logging actual prompt/token contents.

Diagnostics should identify:

- session ID if safe;
- namespace/model identifier;
- token/page counts;
- result.

Never dump system prompts or tool definitions into logs merely for diagnostics.

---

# 33. Tests

Add:

```text
CrossSessionPrefixSynthesisTests.cs
```

with the following mandatory tests.

### Test 1 — IdenticalPrefixIsSynthesized

Session A and B have:

```text
A B C D E F
```

at the beginning.

After synthesis:

```text
PrefixIndex.MatchPrefix(B)
```

returns the shared pages.

---

### Test 2 — DifferentPrefixIsNotSynthesized

```text
Session A: A B C
Session B: A B X
```

must not produce a shared prefix beyond the common page-aligned boundary.

---

### Test 3 — NamespaceIsolation

Identical tokens but:

```text
Model A
Model B
```

must not share.

---

### Test 4 — KvConfigIsolation

Identical tokens/model but different KV configuration must not share.

---

### Test 5 — PartialPageNeverPublished

For a sequence with:

```text
PageSize * N + partialTail
```

only full pages are published.

---

### Test 6 — FailedSessionNeverPublished

A failed/cancelled prefill must not become a synthesis source.

---

### Test 7 — SessionDisposalRace

Dispose the source session while synthesis is attempting to publish.

Verify:

- no stale page IDs are published;
- no invalid reference counts;
- no page corruption.

---

### Test 8 — RefCountRetention

Before publication:

```text
refCount = N
```

After publication:

```text
refCount = N + 1
```

and after eviction:

```text
refCount = N
```

Use the existing cache APIs rather than reaching into private implementation details unless the test infrastructure already does so.

---

### Test 9 — COWIsolation

Two sessions share synthesized pages.

Mutate one session.

Verify:

```text
Session A → private page
Session B → original shared page
```

and contents remain correct.

---

### Test 10 — DuplicatePublication

Two sessions attempt to publish the same logical prefix.

Verify the RadixPrefixTree contains one logical cache entry and reference counts remain correct.

---

### Test 11 — RestartCleanSlate

Create a runtime, synthesize prefixes, dispose it, create a new runtime.

Verify the new runtime has no entries from the old runtime.

This proves the zero-persistence requirement.

---

### Test 12 — BackgroundCancellation

Start the synthesizer, cancel its token, and verify it terminates cleanly without leaking session/page references.

---

### Test 13 — BoundedScan

Create many sessions and verify one synthesis pass respects its configured work/session budget.

---

### Test 14 — NoForegroundBlocking

Ensure synthesis does not hold the session registry or prefix-tree lock while performing expensive session enumeration/comparison.

---

# 34. Integration test

Add an end-to-end test:

```text
Create Session A
       ↓
prefill common system/tool prefix
       ↓
commit
       ↓
background synthesizer runs
       ↓
RadixPrefixTree contains prefix
       ↓
Create Session B
       ↓
prompt begins with same prefix
       ↓
MatchPrefix()
       ↓
attach shared pages
       ↓
prefill only remaining tokens
```

Verify that Session B receives the correct output and no KV corruption occurs.

---

# 35. Important physical-page invariant

The implementation must preserve this invariant:

```text
A physical page may only be reused by CpuKvCache
when its reference count is zero.
```

Therefore:

```text
Session page
     +
Prefix cache page reference
```

must keep the page alive.

The synthesizer must never manipulate page IDs directly in a way that bypasses `CpuKvCache` ownership rules.

---

# 36. Lifetime model

The intended ownership model is:

```text
┌────────────────────────────────────┐
│          CpuKvCache                │
│                                    │
│ Page 42                            │
│   ref=3                            │
└──────────────┬─────────────────────┘
               │
       ┌───────┼────────┐
       │       │        │
       ▼       ▼        ▼
   Session A Session B PrefixIndex
```

If:

```text
Session A disposed
```

then:

```text
ref=2
```

If:

```text
Session B disposed
```

then:

```text
ref=1
```

If:

```text
PrefixIndex evicts
```

then:

```text
ref=0
```

Only then may:

```text
CpuKvCache
```

return the page to its free pool.

---

# 37. Recommended implementation sequence

### Phase 1 — Repository inspection

Before changing code, inspect:

- `IPrefixCacheIndex`;
- `RadixPrefixTree`;
- `InferenceRuntime`;
- `InferenceSession`;
- session registry;
- `CpuKvCache`;
- `CpuKvSequence`;
- page ownership/ref-count APIs;
- COW implementation;
- existing lifecycle/background-service patterns.

Do not assume the conceptual API in this plan exactly matches the current code.

---

### Phase 2 — Stable session snapshot

Implement or reuse a mechanism that exposes:

```text
committed token prefix
+
physical page references
+
namespace
```

with safe lifetime semantics.

This is the most important integration point.

---

### Phase 3 — Synthesizer

Implement:

```text
CrossSessionPrefixSynthesizer
```

with:

- bounded background scanning;
- namespace grouping;
- exact token matching;
- page-aligned candidate selection;
- `PrefixIndex.Publish()`;
- cancellation;
- lightweight metrics.

---

### Phase 4 — Runtime integration

Instantiate the synthesizer alongside:

```text
InferenceRuntime
PrefixIndex
```

Start it with runtime startup.

Stop/dispose it with runtime shutdown.

The runtime owns the synthesizer lifetime.

---

### Phase 5 — Event integration

If existing lifecycle events make it easy, trigger synthesis after successful committed prefill.

Otherwise retain the periodic scanner.

Do not redesign session lifecycle merely to support this feature.

---

### Phase 6 — Tests

Implement the invariant suite above.

Run:

```bash
dotnet test tests/OpenTail.Stingray.Tests.ForwardPass/OpenTail.Stingray.Tests.ForwardPass.csproj
```

Then:

```bash
dotnet test OpenTail.Stingray.slnx
```

---

### Phase 7 — Release verification

Build:

```bash
dotnet build OpenTail.Stingray.slnx -c Release
```

Verify there are no:

- leaked page references;
- stale page IDs;
- cross-namespace matches;
- background tasks surviving runtime disposal;
- partial-page publications.

---

# 38. Acceptance criteria

Plan 014 is complete when:

- [ ] Cross-session synthesis runs in the background.
- [ ] It is completely in-memory.
- [ ] It writes no disk state.
- [ ] Runtime restart starts with an empty synthesis/cache state.
- [ ] It discovers identical committed prompt prefixes.
- [ ] It only considers page-aligned complete KV pages.
- [ ] It uses the existing `IPrefixCacheIndex`.
- [ ] It uses the existing `RadixPrefixTree`.
- [ ] It does not implement another prefix cache.
- [ ] It respects `PrefixCacheNamespace`.
- [ ] Different models cannot share pages.
- [ ] Different KV configurations cannot share pages.
- [ ] Failed/cancelled sessions cannot publish pages.
- [ ] Speculative/uncommitted tokens cannot be published.
- [ ] Session disposal cannot cause stale page publication.
- [ ] Prefix publication retains physical pages correctly.
- [ ] Existing COW isolation remains intact.
- [ ] Existing LRU eviction remains responsible for reclaiming cache pages.
- [ ] Duplicate discovery does not corrupt ref-counts.
- [ ] Background work is bounded.
- [ ] Foreground inference is not blocked by synthesis.
- [ ] Cancellation is clean.
- [ ] Metrics expose synthesis activity.
- [ ] All new invariant tests pass.
- [ ] Full solution tests pass.
- [ ] Release build passes.

---

# 39. Non-goals

Do **not** add these as part of Plan 014:

- persistent prefix cache;
- disk-backed KV cache;
- distributed prefix sharing;
- cross-process prefix sharing;
- semantic similarity;
- fuzzy prefix matching;
- automatic prompt rewriting;
- token deduplication;
- a second Radix tree;
- a second LRU cache;
- a second reference-counting mechanism;
- a new scheduler;
- model inference to identify reusable prefixes;
- prompt classification;
- arbitrary substring KV sharing.

Those are separate features and would unnecessarily increase the risk of this change.

---

# 40. Architectural result

After Plan 014:

```text
                         InferenceRuntime
                                │
              ┌─────────────────┼──────────────────┐
              │                 │                  │
              ▼                 ▼                  ▼
        Active Sessions    PrefixIndex       Memory Governor
              │                 │
              │                 │
              └────────┐        │
                       │        │
                       ▼        │
           CrossSessionPrefix   │
              Synthesizer       │
                       │        │
                       └────────┤
                                ▼
                       RadixPrefixTree
                                │
                                ▼
                         Physical KV Pages
                                │
                       ┌────────┴────────┐
                       ▼                 ▼
                   Session A         Session B
```

The key property is that **synthesis is merely an automatic producer of entries for the prefix cache you have already built**.

The existing invariants remain the security and correctness boundary:

```text
Exact token prefix
       +
same ModelId
       +
same KvConfigHash
       +
complete committed pages
       +
reference-count retention
       +
COW isolation
       ↓
SAFE SHARED KV
```

That is the part I would be particularly firm about with the coding AI. **Do not let it "improve" the idea into semantic prefix matching or a second cache.** Exact token equality + existing RadixPrefixTree + existing page ownership is what makes this feature powerful *without making it scary*.

### Implementation flexibility

The API names, class names, scan intervals, thresholds, and code fragments above are **guidance, not requirements**.

The coding agent should inspect the current Stingray implementation first and adapt the design to the abstractions already present. If the existing code provides a cleaner way to obtain a stable committed KV/page snapshot, use that.

Likewise, if the current `RadixPrefixTree.Publish()` already handles duplicate entries, do not add duplicate-detection machinery to the synthesizer.

**Prefer the smallest implementation that preserves the stated invariants. Better concepts are explicitly allowed where repository inspection shows a cleaner integration point.**