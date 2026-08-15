# OpenTail.Stingray — Paged KV Cache Architecture & Implementation Plan

## Objective

Refactor the existing KV-cache implementation in OpenTail.Stingray into a backend-independent, page-oriented KV-cache abstraction.

The goal is **not** to immediately rewrite every backend.

The goal is to establish a clean KV-cache contract that allows the runtime to support:

- continuous batching
- paged KV storage
- prefix sharing
- session persistence
- session fork/branching
- KV eviction
- KV quantisation
- long-context serving
- speculative decoding
- CPU/CUDA/Vulkan implementations
- future memory-pressure-aware scheduling

without making `ContinuousBatchingEngine` understand backend-specific KV memory details.

The refactor must preserve existing inference behaviour and performance before introducing new capabilities.

---

# 1. Current architecture

The repository already contains:

- `ISequenceKvCache`
- backend-specific KV implementations
- `ContinuousBatchingEngine`
- `IBatchedForwardPass`
- `KvBytesPerToken`
- prefix-cache support
- retained sequence/session state
- KV-budget admission control
- CPU/CUDA/Vulkan backends

The current `ContinuousBatchingEngine` already treats KV cache as an object associated with an individual sequence.

The engine also currently performs responsibilities related to:

- sequence admission
- KV reservation
- prefix-cache management
- sequence retirement
- retained sequence handling
- prefill
- decode
- batching

The new architecture must **not duplicate those responsibilities**.

The KV layer should own physical KV storage.

The scheduler should own scheduling.

The model/forward-pass layer should own computation.

---

# 2. Target architecture

Move toward this conceptual architecture:

```text
                         Inference Scheduler
                                │
                                │ logical sequence
                                ▼
                         IKvCacheManager
                                │
                     ┌──────────┼──────────┐
                     │          │          │
                     ▼          ▼          ▼
                 Sequence A  Sequence B  Sequence C
                     │          │          │
                     └──────────┼──────────┘
                                │
                         logical KV pages
                                │
                  ┌─────────────┼─────────────┐
                  ▼             ▼             ▼
              Page 17        Page 42        Page 91
                  │             │             │
             physical KV storage/backend
                  │
          ┌───────┼──────────────┐
          ▼       ▼              ▼
         CPU     CUDA          Vulkan
```

The important distinction is:

```text
Logical sequence
       ≠
Physical memory
```

A sequence should reference a list/table of KV pages.

The page table determines where its KV data physically lives.

---

# 3. Core design principle

The inference engine must never need to know:

- where KV memory physically resides
- whether memory is contiguous
- whether pages are shared
- whether pages are quantised
- whether pages are CPU or GPU memory
- how pages are allocated
- how pages are freed
- how pages are copied
- how pages are evicted

The engine should only know:

```text
sequence
position
page
logical token range
```

The KV implementation owns everything else.

---

# 4. Define the logical model

Introduce the following concepts.

## 4.1 KvPage

A KV page represents a fixed number of logical tokens.

Example:

```csharp
public readonly record struct KvPageId(int Value);
```

A page should have a fixed capacity:

```csharp
public readonly record struct KvPageSize(int Tokens);
```

Initially use:

```text
32 tokens/page
```

as the default.

Do not make page size configurable throughout the entire engine yet.

Define it once in the KV configuration.

A later implementation can benchmark:

```text
16
32
64
128
```

but the initial implementation should use one stable value.

---

# 5. Page ownership

Every physical page has exactly one of these states:

```text
Free
Owned
Shared
Evicting
Evicted
```

Do not expose mutable state directly to the scheduler.

The KV manager owns page lifecycle.

A page may be shared by multiple logical sequences.

For example:

```text
Prompt:
A B C D E F G H

Sequence 1:
A B C D E F G H X Y

Sequence 2:
A B C D E F G H P Q
```

Both sequences can reference:

```text
Page 0 -> A B C D E F G H
```

but must have separate writable pages after the shared prefix:

```text
Sequence 1
  Page 0 [shared]
  Page 1 [owned]

Sequence 2
  Page 0 [shared]
  Page 2 [owned]
```

Never modify a shared page.

Use copy-on-write if modification is required.

---

# 6. Target interfaces

Do not immediately delete `ISequenceKvCache`.

Introduce the new abstraction alongside it.

The target API should look approximately like this:

```csharp
public interface IKvCache
{
    IKvSequence AllocateSequence(KvSequenceOptions options);

    void ReleaseSequence(IKvSequence sequence);

    KvCacheStatistics GetStatistics();
}
```

Then:

```csharp
public interface IKvSequence
{
    long SequenceId { get; }

    int TokenCount { get; }

    int Capacity { get; }

    int PageCount { get; }

    ReadOnlySpan<KvPageId> Pages { get; }

    void Append(int tokenCount);

    IKvSequence Fork();

    IKvSequence ForkAt(int tokenCount);

    void Release();

    void Clear();
}
```

Do not copy this API blindly if existing Stingray conventions require different naming.

The important semantics are more important than the exact spelling.

---

# 7. Separate manager from sequence

Do not make `IKvSequence` responsible for global allocation.

The correct relationship is:

```text
IKvCache
    owns global page pool

IKvSequence
    owns logical page table

KvPage
    represents physical allocation
```

For example:

```text
IKvCache
 ├── PagePool
 │    ├── Page 0
 │    ├── Page 1
 │    ├── Page 2
 │    └── ...
 │
 ├── Sequence 1
 │    └── [0, 4, 8]
 │
 └── Sequence 2
      └── [0, 7, 9]
```

---

# 8. Page table

Each sequence should maintain a logical page table.

Conceptually:

```csharp
internal sealed class KvPageTable
{
    private KvPageId[] _pages;

    public int Count { get; }

    public KvPageId this[int pageIndex] => _pages[pageIndex];
}
```

Do not expose the mutable backing array.

The forward pass should receive a read-only representation.

---

# 9. Token-to-page mapping

For:

```text
PageSize = 32
```

mapping is:

```text
pageIndex = tokenPosition / 32
offset    = tokenPosition % 32
```

Example:

```text
token 0  -> page 0 offset 0
token 31 -> page 0 offset 31
token 32 -> page 1 offset 0
token 63 -> page 1 offset 31
token 64 -> page 2 offset 0
```

This mapping must be centralized.

Create:

```csharp
internal static class KvPageMath
{
    public static int GetPageIndex(int tokenPosition, int pageSize);

    public static int GetPageOffset(int tokenPosition, int pageSize);

    public static int GetRequiredPageCount(int tokenCount, int pageSize);
}
```

Add unit tests for boundary conditions.

---

# 10. KV layout

Do not redesign the actual K/V tensor layout in the first implementation.

The first version should preserve the existing backend tensor representation.

The page abstraction should initially be an **allocation/indexing abstraction**.

For example:

```text
Existing:
Layer → contiguous K/V storage

First paged implementation:
Layer → page allocations

Future:
Layer → backend-specific paged kernel layout
```

This distinction is extremely important.

Do not combine:

1. page abstraction
2. new GPU memory layout
3. new attention kernel

into one change.

That would make debugging unnecessarily difficult.

---

# 11. CPU implementation

Create:

```text
CpuKvCache
CpuKvSequence
CpuKvPagePool
```

The first implementation may use managed/unmanaged allocations depending on the existing Stingray memory model.

Prefer the existing allocation strategy where possible.

The page pool should allocate pages large enough for:

```text
all layers
K
V
page token capacity
head dimensions
dtype
```

Do not allocate one tiny managed object per layer/token.

Avoid object-heavy designs.

The hot path must remain contiguous where possible.

---

# 12. Recommended CPU physical representation

Conceptually:

```text
CpuKvPage
 ├── K storage
 └── V storage
```

or, preferably for locality:

```text
CpuKvPage
 ├── K layer storage
 └── V layer storage
```

The exact tensor layout must follow the existing attention kernels.

Do not invent a layout before examining the current CPU attention implementation.

The implementation must document:

```text
[token]
[layer]
[head]
[dimension]
```

ordering.

---

# 13. CUDA implementation

Do not initially rewrite CUDA kernels.

Create:

```text
CudaKvCache
CudaKvSequence
CudaKvPagePool
```

but initially make the implementation capable of adapting the existing contiguous cache.

The first CUDA milestone may internally allocate one contiguous region and expose page descriptors.

Example:

```text
CudaKvPageId
    ↓
offset into large CUDA allocation
```

This establishes the API without immediately requiring a new paged-attention kernel.

Only after the abstraction passes correctness tests should CUDA receive a genuinely paged physical layout.

---

# 14. Vulkan implementation

Apply the same principle.

Create:

```text
VulkanKvCache
VulkanKvSequence
VulkanKvPagePool
```

Initially preserve the existing Vulkan buffer layout if possible.

Represent pages as offsets/ranges into existing buffers.

Do not rewrite the attention shader in the same commit as the interface migration.

---

# 15. Quantised KV cache

Do not make quantisation part of `IKvSequence`.

It is a property of the physical cache.

For example:

```csharp
KvCacheOptions
{
    DType = KvCacheDType.F32
}
```

Later:

```text
F32
F16
BF16
Q8_0
Q6
Q4
```

The sequence should not care.

It should still say:

```text
I contain 4096 logical KV tokens.
```

The physical cache determines how many bytes that consumes.

---

# 16. Required capacity APIs

The scheduler needs accurate capacity information.

Expose:

```csharp
public interface IKvCache
{
    long CapacityBytes { get; }

    long UsedBytes { get; }

    long FreeBytes { get; }

    long BytesPerToken { get; }

    int PageSizeTokens { get; }

    int TotalPages { get; }

    int FreePages { get; }

    int UsedPages { get; }
}
```

Do not make the scheduler calculate these values itself.

---

# 17. Admission control

Move physical-memory decisions into the KV manager.

The scheduler should ask:

```csharp
var reservation = kvCache.TryReserve(
    sequenceId,
    requiredTokens);
```

Conceptually:

```csharp
public interface IKvReservation : IDisposable
{
    long ReservedTokens { get; }

    long ReservedBytes { get; }

    bool TryGrow(int additionalTokens);
}
```

The scheduler owns the reservation.

The cache owns physical allocation.

This is important because:

```text
reservation
≠
allocation
```

A request may reserve capacity before it actually consumes pages.

This preserves the existing Stingray admission-control behaviour.

---

# 18. Reservation semantics

Reservations must prevent overcommit.

Example:

```text
KV capacity = 1000 tokens

Request A reserves 400
Request B reserves 400
Request C requests 300

C must wait.

400 + 400 + 300 > 1000
```

Even if only 200 tokens have physically been written so far, the scheduler must not admit C if the remaining requested capacity cannot be guaranteed.

This is especially important for continuous batching.

---

# 19. Growth

Sequences grow during generation.

A sequence must be able to request another page without blocking the entire scheduler indefinitely.

Preferred model:

```text
TryGrow()
    ↓
available page?
    ├── yes → allocate
    └── no  → scheduler handles backpressure
```

Do not silently evict another active sequence.

Eviction must be an explicit policy decision.

---

# 20. Forking

Forking is a first-class requirement.

Implement:

```csharp
IKvSequence ForkAt(int tokenCount);
```

Example:

```text
Original:
Page 0
Page 1
Page 2
Page 3

Fork at token 96
```

If page size = 32:

```text
Original:
Page 0
Page 1
Page 2
Page 3

Child:
Page 0
Page 1
Page 2
```

Pages 0–2 become shared.

Page 3 remains original-only.

Reference counts:

```text
Page 0 refcount = 2
Page 1 refcount = 2
Page 2 refcount = 2
Page 3 refcount = 1
```

---

# 21. Copy-on-write

If a child tries to append to a partially shared page:

```text
Shared page:
A B C D E F
```

the implementation must not overwrite the shared page.

Allocate a private page:

```text
Original:
Page 0 [shared]
Page 1

Child:
Page 0 [private copy]
Page 2
```

However, prefer page-aligned forks wherever possible.

Expose:

```csharp
ForkAt(tokenCount)
```

but optimise:

```text
ForkAt(page boundary)
```

as the zero-copy case.

---

# 22. Prefix sharing

Prefix sharing should eventually be implemented as a separate index over immutable page sequences.

Conceptually:

```text
PrefixCache
    │
    ├── token hash
    ├── page IDs
    ├── token length
    ├── refcount
    └── last-used timestamp
```

Do not make the global prefix cache part of `IKvSequence`.

A sequence can reference cached pages.

The cache manager owns them.

---

# 23. Prefix identity

A prefix must not be identified only by text.

The cache key must include enough information to ensure identical tokens actually imply identical KV state.

At minimum consider:

```text
model identity
tokenizer/model revision
architecture
KV dtype
RoPE configuration
relevant attention configuration
token sequence
```

For a first implementation, the key may simply be:

```text
model instance + token sequence
```

because a cache must never be shared between incompatible model states.

Document this invariant explicitly.

---

# 24. Page immutability

A page becomes immutable when it is shared.

Use this rule:

```text
refcount == 1
    → writable

refcount > 1
    → immutable
```

Never mutate a shared page.

This rule should be enforced as close to the physical cache as practical.

Do not rely only on caller discipline.

---

# 25. Eviction

Do not implement arbitrary eviction first.

Start with:

```text
LRU eviction of unreferenced immutable prefix pages.
```

Never evict:

- active sequence pages
- writable pages
- pages referenced by a retained session
- pages under an outstanding reservation

Only pages with:

```text
refcount == 0
```

may be physically released.

---

# 26. Session interaction

Sessions should retain a logical sequence handle rather than owning raw backend memory.

Bad:

```text
Session
    └── raw CPU K/V arrays
```

Preferred:

```text
Session
    └── IKvSequence
           └── page table
                  └── physical pages
```

This makes session persistence and fork possible without exposing backend details.

---

# 27. Session persistence

Do not attempt to serialize raw physical pages as the first step.

The first persistent-session design should distinguish:

```text
Logical session state
```

from:

```text
Physical KV residency
```

A persisted session can contain:

```text
model identity
token IDs
sampling state
conversation metadata
KV configuration
```

The KV pages can be reconstructed.

Later, an optional fast-path can serialize compatible physical pages.

---

# 28. Forward-pass API

The current forward pass must gradually stop accepting "an arbitrary sequence cache" and instead consume an explicit KV view.

Introduce something conceptually like:

```csharp
public readonly ref struct KvSequenceView
{
    public int TokenCount { get; }

    public int PageSize { get; }

    public ReadOnlySpan<KvPageId> Pages { get; }
}
```

The forward pass should receive this view.

It should not own the page table.

---

# 29. Batched forward pass

For continuous batching:

```text
Sequence A → pages [1,2,3]
Sequence B → pages [4,5]
Sequence C → pages [7,8,9]
```

the batch forward pass receives:

```text
KvBatchView
    Sequence 0 → page table
    Sequence 1 → page table
    Sequence 2 → page table
```

Do not concatenate the logical KV sequences merely to make the API easier.

The backend needs to understand the mapping.

---

# 30. Important first milestone

The first implementation does **not** need true paged attention.

Milestone 1 should be:

```text
Existing contiguous KV implementation
        ↓
IKvCache abstraction
        ↓
IKvSequence abstraction
        ↓
page metadata
        ↓
existing physical storage
```

Everything must continue producing the same logits.

This is an architectural migration.

Not yet a performance optimisation.

---

# 31. Correctness invariants

Add tests for all of these.

## Invariant 1 — sequential equivalence

```text
Generate 100 tokens normally
```

must equal:

```text
Generate 50
then continue 50
```

under deterministic sampling.

---

## Invariant 2 — batch equivalence

For deterministic generation:

```text
batch size = 1
```

must produce the same tokens as:

```text
batch size = N
```

for every sequence.

---

## Invariant 3 — fork equivalence

Given:

```text
Prompt → A B C D
```

fork after D.

Then:

```text
Parent → X
Child  → Y
```

must produce the same result as two independently evaluated sequences that both received:

```text
A B C D
```

before their divergent tokens.

---

## Invariant 4 — prefix-cache equivalence

```text
without prefix cache
```

and:

```text
with prefix cache
```

must produce identical logits/tokens.

---

## Invariant 5 — page boundary equivalence

Test generation where sequence lengths are:

```text
1
31
32
33
63
64
65
95
96
97
```

with page size 32.

---

## Invariant 6 — fork at boundaries

Test:

```text
ForkAt(0)
ForkAt(1)
ForkAt(31)
ForkAt(32)
ForkAt(33)
```

---

## Invariant 7 — release

After releasing a sequence:

```text
used pages decreases
```

and:

```text
no subsequent sequence can observe the old contents.
```

---

## Invariant 8 — sharing

After:

```text
parent.Fork()
```

shared pages must have:

```text
refcount == 2
```

After child release:

```text
refcount == 1
```

After parent release:

```text
page becomes reclaimable
```

---

# 32. Add allocation/leak tests

Run a stress test:

```text
allocate
generate
fork
generate
release child
fork again
release
```

for thousands of iterations.

At the end:

```text
UsedPages == 0
UsedBytes == 0
```

unless intentionally retained by a prefix cache.

Also run the same test with:

```text
prefix cache enabled
prefix cache disabled
```

---

# 33. Threading model

The initial implementation should use the existing Stingray model:

```text
scheduler/batcher thread
        │
        ▼
KV ownership mutations
```

Do not introduce arbitrary concurrent mutation of page tables.

Prefer:

```text
single owner for page-table mutation
atomic reference counts for shared physical pages
```

If physical reference counting requires `Interlocked`, use it.

Avoid global locks in the decode hot path.

---

# 34. No allocation in decode hot path

After warm-up, generation should not allocate:

- managed arrays
- LINQ objects
- page-table wrapper objects
- per-token page descriptors

where avoidable.

Page growth occurs only at page boundaries.

Pre-size page tables based on expected maximum sequence length where practical.

---

# 35. Statistics

Expose cache statistics:

```csharp
public readonly record struct KvCacheStatistics(
    long CapacityBytes,
    long UsedBytes,
    long FreeBytes,
    long ReservedBytes,
    int TotalPages,
    int UsedPages,
    int FreePages,
    int SharedPages,
    long Allocations,
    long Releases,
    long Forks,
    long CopyOnWriteCopies,
    long Evictions);
```

Also expose:

```text
peak used bytes
peak pages
allocation failures
```

These are essential for debugging long-context workloads.

---

# 36. Diagnostics

Add:

```bash
stingray inspect-kv
```

or equivalent diagnostic functionality later.

Example:

```text
KV CACHE
────────────────────────────
Backend       CPU
Page size     32 tokens
Capacity      8.0 GB
Used          5.2 GB
Reserved      6.4 GB

Pages
  Total       262144
  Used        166400
  Free         95744
  Shared        18432

Sequences
  Active             8
  Retained           4

Prefix cache
  Entries          127
  Hit rate        81.2%
  Tokens reused  48122

Copy-on-write
  Copies            12
```

---

# 37. Migration of ContinuousBatchingEngine

Do this carefully.

Do not rewrite the entire class.

Current conceptual flow:

```text
PendingRequest
      ↓
ActiveSeq
      ↓
ISequenceKvCache
      ↓
ForwardPass
```

Change to:

```text
PendingRequest
      ↓
ActiveSeq
      ↓
IKvSequence
      ↓
KvSequenceView
      ↓
ForwardPass
```

The scheduler should retain:

```text
logical token position
reservation
sampling state
request state
```

The KV layer should retain:

```text
physical page allocation
page table
reference counts
physical memory
```

---

# 38. Existing `KvBytesPerToken`

Keep `KvBytesPerToken`.

But change its role.

It should become a backend/cache property:

```csharp
IKvCache.BytesPerToken
```

rather than something the engine needs to derive from tensor details.

The existing admission-control calculation can then become:

```csharp
requiredBytes =
    requestedTokens * kvCache.BytesPerToken;
```

The engine does not need to know:

```text
layers
heads
head dimension
dtype
```

---

# 39. Existing KV budget

Preserve current semantics.

If:

```text
kvBudgetBytes == 0
```

means auto:

```text
half available memory
```

retain that behaviour initially.

But move the resulting capacity into:

```text
IKvCache
```

or:

```text
KvCacheOptions
```

Do not change user-visible defaults during the refactor.

---

# 40. Existing prefix cache

Do not immediately delete the existing prefix cache.

Wrap/migrate it.

The current prefix-cache implementation can initially become:

```text
PrefixCache
      ↓
IKvCache.AcquireSharedPrefix(...)
```

The objective is to move ownership of physical KV pages underneath it.

Eventually:

```text
ContinuousBatchingEngine
      ↓
PrefixCache
      ↓
IKvCache
      ↓
PagePool
```

rather than:

```text
ContinuousBatchingEngine
      ↓
PrefixCache
      ↓
raw ISequenceKvCache details
```

---

# 41. Backward compatibility

Do not remove:

```csharp
ISequenceKvCache
```

until the new implementation has passed all existing tests.

Initially provide an adapter:

```csharp
PagedKvSequenceAdapter : ISequenceKvCache
```

or:

```csharp
SequenceKvCacheAdapter
```

depending on naming conventions.

This permits incremental migration.

---

# 42. Adapter strategy

The adapter should translate:

```text
ISequenceKvCache
        ↕
IKvSequence
```

without copying the entire KV cache.

Do not implement the adapter by:

```text
read every K/V value
copy into new cache
```

That defeats the purpose.

The adapter should reference the same physical storage where possible.

---

# 43. Implementation phases

## Phase 0 — inventory

Before writing code, inspect:

```text
ISequenceKvCache
all implementations
IBatchedForwardPass
ForwardPass
attention kernels
ContinuousBatchingEngine
prefix cache
session code
SnapKV
TurboQuant KV
CPU KV
CUDA KV
Vulkan KV
```

Produce a dependency map.

Do not modify code during this phase.

---

## Phase 1 — introduce contracts

Add:

```text
IKvCache
IKvSequence
IKvReservation
KvPageId
KvCacheOptions
KvCacheStatistics
KvPageMath
KvSequenceView
```

No behavioural change.

Compile.

Run all tests.

---

## Phase 2 — adapter

Implement:

```text
LegacySequenceKvCacheAdapter
```

so existing caches can participate in the new API.

Run the entire existing test suite.

---

## Phase 3 — CPU paged metadata

Implement:

```text
CpuKvCache
CpuKvSequence
CpuKvPagePool
```

but initially preserve the existing physical layout if necessary.

Validate:

```text
same logits
same tokens
same performance within reasonable noise
```

---

## Phase 4 — page ownership

Implement:

```text
refcount
Fork()
ForkAt()
Release()
CopyOnWrite()
```

Add exhaustive tests.

---

## Phase 5 — prefix sharing

Move existing prefix-cache functionality onto the new page abstraction.

Validate:

```text
cached == uncached
```

---

## Phase 6 — scheduler integration

Modify `ContinuousBatchingEngine` so that it requests:

```text
reservation
sequence
page growth
release
```

from `IKvCache`.

Remove physical-memory logic from the engine where it is now redundant.

---

## Phase 7 — CUDA

Implement the abstraction for CUDA.

Initially preserve contiguous physical allocation if required.

Do not change CUDA kernels until correctness is established.

---

## Phase 8 — Vulkan

Same approach.

---

## Phase 9 — true paged kernels

Only after the abstraction is stable:

```text
CPU paged attention
CUDA paged attention
Vulkan paged attention
```

can be implemented.

This is a separate performance project.

---

# 44. Performance acceptance criteria

The abstraction must not cause a significant regression.

For CPU batch size 1:

```text
Target:
< 3% decode regression
< 5% prefill regression
```

during the migration phase.

If the abstraction causes more than this:

**stop and profile before proceeding.**

Do not accept an abstraction that permanently damages the hot path.

---

# 45. Benchmark matrix

Run at least:

```text
Model:
  SmolLM2-1.7B
  Qwen3-8B

Backend:
  CPU
  CUDA (if available)
  Vulkan (if available)

Batch:
  1
  2
  4
  8

Context:
  512
  2048
  8192

Generation:
  128
  512
```

Measure:

```text
TTFT
prefill tok/s
decode tok/s
peak memory
KV bytes
allocation count
page count
batch utilisation
```

---

# 46. Do not implement yet

The following are explicitly **out of scope for the first implementation**:

- new attention kernels
- new quantisation formats
- automatic GPU/CPU KV migration
- distributed KV cache
- disk-backed KV cache
- compression of arbitrary pages
- remote KV cache
- multi-GPU KV
- speculative decoding
- model architecture changes

The abstraction should make those features possible later.

Do not implement them as part of this task.

---

# 47. Architectural invariants

The final architecture must obey these rules.

### Rule 1

The scheduler knows about:

```text
logical tokens
reservations
sequence lifecycle
```

but not physical KV layout.

### Rule 2

The KV cache knows about:

```text
pages
memory
ownership
reference counts
capacity
```

but not sampling or model semantics.

### Rule 3

The forward pass knows about:

```text
K/V access
tensor layout
attention
```

but not request scheduling.

### Rule 4

A shared page is immutable.

### Rule 5

Physical memory is released only when no logical sequence references it.

### Rule 6

Prefix cache entries reference KV pages; they do not own independent copies unless explicitly materialised.

### Rule 7

Forking must be O(number of page references), not O(number of KV elements).

### Rule 8

A cache hit must never alter model output.

### Rule 9

Batch size must not alter deterministic model output.

### Rule 10

The CPU/CUDA/Vulkan implementations share the same logical KV contract.

---

# 48. Final target architecture

The intended result is:

```text
                         ┌─────────────────────┐
                         │ ContinuousBatching  │
                         │      Scheduler      │
                         └──────────┬──────────┘
                                    │
                          logical sequence
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │     IKvCache        │
                         │                     │
                         │ allocation          │
                         │ reservations        │
                         │ page ownership      │
                         │ sharing             │
                         │ eviction            │
                         └──────────┬──────────┘
                                    │
                          ┌─────────┴─────────┐
                          │                   │
                    IKvSequence         Prefix Cache
                          │                   │
                    page table          shared pages
                          │                   │
              ┌───────────┼───────────────────┘
              │
       ┌──────┼──────┬─────────┐
       ▼      ▼      ▼         ▼
      CPU    CUDA  Vulkan   Future backend
       │      │      │
       ▼      ▼      ▼
   physical KV storage
```

The key architectural property is:

```text
Logical KV state
        ↓
Page abstraction
        ↓
Backend physical storage
```

rather than:

```text
Scheduler
   ↓
CPU/CUDA/Vulkan-specific KV implementation
```

---

# 49. Definition of done

This task is complete only when all of the following are true:

- [ ] `IKvCache` exists.
- [ ] `IKvSequence` exists.
- [ ] Page identity is explicit.
- [ ] Page size is explicit.
- [ ] Logical token → page mapping is centralized.
- [ ] Reservations are separated from physical allocation.
- [ ] Sequence release returns pages correctly.
- [ ] Forking is implemented.
- [ ] Shared pages use reference counting.
- [ ] Copy-on-write is implemented for partial shared pages.
- [ ] Existing prefix caching works through the new abstraction.
- [ ] Continuous batching works unchanged from the user's perspective.
- [ ] CPU backend works.
- [ ] CUDA backend is migrated without requiring a kernel rewrite.
- [ ] Vulkan backend is migrated without requiring a kernel rewrite.
- [ ] Existing session functionality continues to work.
- [ ] Existing TurboQuant/SnapKV paths are either correctly adapted or explicitly isolated.
- [ ] No memory leaks occur under stress.
- [ ] Deterministic batch=1 and batch=N tests pass.
- [ ] Deterministic cached and uncached tests pass.
- [ ] Forked and independently evaluated sequences produce equivalent output.
- [ ] Page-boundary tests pass.
- [ ] Existing benchmark performance has no material regression.
- [ ] `KvCacheStatistics` exposes sufficient information to diagnose memory behaviour.
- [ ] No backend-specific KV implementation leaks into scheduler code.
- [ ] Existing public APIs are preserved until migration is complete.
- [ ] All existing tests pass.
- [ ] New KV tests pass.
- [ ] Release build passes.
- [ ] NativeAOT build passes.

---

# 50. Most important instruction to the implementation AI

**Do not perform a giant rewrite.**

This work must be performed as a sequence of small, compilable, testable migrations.

At every phase:

```text
change
→ compile
→ unit tests
→ inference correctness test
→ benchmark
→ inspect diff
→ continue
```

If a proposed change requires simultaneously rewriting:

```text
ContinuousBatchingEngine
+ CPU attention
+ CUDA attention
+ Vulkan attention
+ prefix cache
+ sessions
```

then stop.

Break the change into smaller architectural steps.

The first objective is not maximum performance.

The first objective is to establish a **correct, backend-independent logical KV model** that can subsequently support paged storage and advanced scheduling without requiring another architectural rewrite.

**Do not claim completion because the interfaces exist.**

Completion means that real inference has been migrated onto the new abstraction and the correctness invariants above have been demonstrated.