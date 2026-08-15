
---

# Model Residency Management & Request Scheduling — Design Plan

## Status

Draft, revised after an independent conceptual review (external AI reviewer, full review
preserved in project history alongside this doc). The review's five mandatory changes are
incorporated below; see "Changes from the reviewed draft" for what moved and why. Not yet
scoped into implementation with sign-off — still pre-code.

## Origin

This design did not start as a server feature request. It started as an investigation into
why `Tests.ForwardPass` was climbing to ~59.5 GB of memory on a 63 GB machine during a full
run. That investigation ruled out several hypotheses in order — a GC-scheduling/collection-
timing problem, an mmap refcount leak, a missing `Dispose()` call, background-process
contention — and none of them held up under direct, controlled measurement. What was never
ruled out, because it isn't a bug: the test suite loads ~15 different multi-gigabyte models
across ~40 files, in roughly filename order, with no coordination over which model is "hot" at
any point in time. Model A's tests run, then B's, then A's again, then C's, then B's — each
transition pays a full load-and-page-in cost, and because tests execute sequentially but
nothing enforces disposal-to-actual-release ordering, a de-facto overlap of "still winding
down" and "just starting up" model footprints can occur.

The test suite isn't the problem. It's a good demonstration of a real gap in Stingray itself:
**the engine has no concept of more than one model, and therefore no way to manage residency
or schedule work across models.** This plan closes that gap, with the test suite's current
thrashing behavior serving as both the motivating case and, later, a regression benchmark.

## What Stingray does today (confirmed by direct code inspection)

- `AddOpenTailStingray` registers `IInferenceEngine` as a **lazily-built singleton**
  (`ServiceCollectionExtensions.cs`). `InferenceEngineLoader.Load` resolves exactly one
  `STINGRAY_MODEL` path, opens exactly one `GgufModel`, and wraps it in exactly one engine
  instance, held for the process's entire lifetime.
- `IInferenceEngine.GenerateChunksAsync` has no model parameter. Model identity (`ModelId`) is
  a fixed, read-only property set once at construction. There is no per-request model
  resolution anywhere in the interface.
- `/v1/models` literally returns a one-element list built from the single loaded `ModelId`.
- There is no registry, pool, swap, or eviction mechanism for whole models anywhere in the
  codebase. Grepping for `SwapModel`, `ModelRegistry`, `MultiModel`, `HotSwap` etc. across
  `src/` returns nothing.
- The one *sub-model* precedent that does exist — `ExpertSlotManager` /
  `CudaExpertSlotManager` — manages an SLRU cache of individual MoE **expert tensors** inside
  one already-loaded model. Right shape of idea, wrong granularity for this problem, built on
  a genuinely reusable primitive: `SlruCache<TKey,TValue>`
  (`src/OpenTail.Stingray.Pipeline/ExpertCache.cs`), a segmented-LRU cache (25% probationary /
  75% protected, optional frequency-aware eviction, optional pinning) generic over key/value
  type. This plan reuses that primitive rather than inventing a new one.
- Stingray also has real session/KV infrastructure (`OpenTail.Stingray.Sessions` —
  transactional, revisioned hot-session orchestration). Any design that can evict a model out
  from under a live session is incompatible with that existing subsystem, not just theoretically
  unsafe — see "Session and KV affinity" below.

## What vLLM does (checked directly against `examples/vllm`, not assumed)

Worth stating plainly because it changes the shape of this plan: vLLM does not solve this
either, at this granularity. It is explicitly single-model-per-process (`docs/usage/faq.md`:
"serve multiple models on a single port... that is not currently supported... run multiple
instances of the server... and have another layer to route"). Its `ModelRegistry` is a static
table mapping architecture strings (`"LlamaForCausalLM"`) to Python classes to instantiate —
not a registry of loaded weight instances. Its scheduler (`vllm/v1/core/sched/scheduler.py`)
is plain continuous-batching FCFS/priority-preemption with zero resource-affinity grouping,
because it never has more than one model to schedule against.

The closest thing vLLM has to this problem is LoRA adapter serving: instead of grouping
requests by adapter over time, it keeps a bounded number of adapters resident simultaneously
in fixed GPU slots (LRU eviction, pinning for hot ones) and uses batched heterogeneous kernels
(Punica/SGMV) so *one* physical batch step can serve requests targeting *different* adapters
at once. That works because LoRA adapters are tiny deltas (tens of MB); it doesn't transfer to
swapping entire multi-GB base models.

This combination — whole-GGUF-model residency management with model-affinity request
scheduling, in a single local-inference process — doesn't appear to exist in vLLM. That's a
conclusion from one specific comparison, not a claim that nothing like it exists anywhere; it
should be read as "we didn't find prior art in the one codebase we checked," not as a
verified survey of the field.

## Goals

1. Allow one Stingray server process to serve requests against more than one model, loading
   and evicting whole models on demand, bounded by available memory.
2. Minimize model residency transitions while keeping per-request latency bounded — not simply
   "batch requests by model" (which is necessary but not sufficient: a naive version of it can
   starve a low-traffic model indefinitely behind a high-traffic one).
3. Make the swap/evict decision **resource-aware** — host RAM, mapped-vs-resident memory, KV
   allocation, and (eventually) accelerator/VRAM budget are different quantities that can each
   independently gate admission; no single number is sufficient.
4. **Never evict a model with live inference or session state.** Residency and correctness are
   linked, not independent concerns — this is a first-class goal, not an implementation detail.
5. Zero behavior change for the existing single-model deployment (`STINGRAY_MODEL=...`). This
   must degenerate to exactly today's behavior when only one model is ever requested.
6. Reuse the existing `SlruCache<TKey,TValue>` primitive rather than build a new cache.
7. Prove it out on the exact workload that motivated it: point the test suite's model loading
   at the same manager and measure the thrashing going away, not just assert it.

## Non-goals (for this plan)

- Cross-machine / distributed model routing. This is one-process, one-machine.
- Continuous-batching internals. `ContinuousBatchingEngine` keeps doing exactly what it does
  today for whichever model is currently receiving requests; this plan only decides *which*
  model that is at any given moment.
- **Concurrent execution of multiple models.** This plan assumes one *active* model at a time
  (multiple models may be *resident*, but only one is being driven for inference at once). A
  machine with enough spare RAM/VRAM to run two models simultaneously is a real future
  direction, but it's a different scheduling problem (parallel execution, not residency
  turn-taking) and this design deliberately doesn't take it on. Naming reflects this: the
  scheduler below is a **Model Residency Scheduler**, not a "multi-model scheduler" — it
  decides which model is *resident and active*, not how to run several at once. This keeps the
  door open to a later "concurrent execution" layer without a redesign of this one.
- LoRA-style intra-batch heterogeneity (serving two different models in one physical batch
  step). Not feasible for whole models the way it is for adapter deltas; out of scope.

---

## Design

### 1. `ModelRuntime` — the unit of residency

A new type in `OpenTail.Stingray.Engine`. What `InferenceEngineLoader.Load` builds today for
the single-model case — a `GgufModel`, a forward pass, and an engine — becomes a reusable,
independently-loadable bundle, with an explicit lifecycle instead of an implicit one:

```csharp
public sealed class ModelRuntime : IDisposable
{
    public ModelId Id { get; }
    public GgufModel Model { get; }
    public IForwardPass ForwardPass { get; }
    public IInferenceEngine Engine { get; }
    public long EstimatedResidentBytes { get; }

    public ModelRuntimeState State { get; }        // see below
    public int ActiveRequests { get; }              // in-flight generations against this runtime
    public int ActiveSessions { get; }              // live Sessions bound to this runtime
    public bool IsPinned { get; }

    public bool CanEvict =>
        State == ModelRuntimeState.Ready &&
        ActiveRequests == 0 &&
        ActiveSessions == 0 &&
        !IsPinned;
}

public enum ModelRuntimeState { Loading, Ready, Draining, Evicting, Failed, Disposed }
```

`CanEvict` is the load-bearing addition here relative to the first draft, which implicitly
treated "this model is LRU-cold" as sufficient grounds to dispose it. It isn't: a model can be
cold on *new* request arrivals while still serving an in-flight generation, a streaming
response, an active session, or speculative-decode state. Eviction must check readiness
(`Ready`, not mid-load or already-draining), liveness (zero active requests *and* zero active
sessions), and the pin flag — all three, every time, not just recency.

### 2. `ModelRuntimeManager` — residency, loading, safe eviction

Structurally a sibling of `ExpertSlotManager`, reusing `SlruCache<TKey,TValue>` unmodified,
keyed by canonical model path instead of `(layer, expertId)`:

```csharp
public sealed class ModelRuntimeManager : IDisposable
{
    private readonly SlruCache<string, ModelRuntime> _cache;
    private readonly Dictionary<string, Task<ModelRuntime>> _inFlightLoads = new();
    private readonly object _lock = new();

    public ValueTask<ModelRuntime> GetOrLoadAsync(string modelPath, CancellationToken ct) { ... }
    public bool TryGetResident(string modelPath, out ModelRuntime runtime) { ... }
    public void Pin(string modelPath) { ... }
    public void Dispose() { ... }   // drains; disposes every runtime with CanEvict-equivalent checks
}
```

Two corrections relative to the first draft, both real gaps rather than style points:

- **Async, not synchronous.** Loading a 20–50 GB GGUF is not a fast operation. A synchronous
  `GetOrLoad` inside the scheduler blocks every other request behind one model's load. This is
  `GetOrLoadAsync`.
- **Single-flight coalescing.** If three requests for the same cold model arrive close
  together, they must share one load, not trigger three. `_inFlightLoads` holds one
  `Task<ModelRuntime>` per model path currently loading; concurrent callers await the same
  task instead of racing independent loads.

**Capacity is not a slot count.** MoE experts within one model are roughly uniform in size, so
`ExpertCache<T>`'s plain integer `capacity` works there. Whole models range from ~1 GB
(SmolLM2) to 50+ GB — a fixed slot count doesn't map to a memory guarantee, and mmap'd file
size is not the same thing as resident memory, which is itself not the whole picture once KV
cache and (for GPU-resident runtimes) VRAM are counted. `SlruCache` itself stays a plain
integer-capacity cache (small, e.g. 2–4 slots — "how many whole models can plausibly
coexist"), unmodified. Memory-awareness is not folded into the cache primitive; it lives in
the admission layer below, which the manager consults before actually committing a load.

Eviction policy: plain LRU, not frequency-aware. Expert access is skewed by design (routing
concentrates on a few hot experts), which is why `ExpertCache` supports frequency-aware
eviction there. Whole-model access in a serving context is closer to genuine recency —
"whichever model got requests most recently is most likely to get more soon." Pinning (already
in `SlruCache`) covers "always keep model X hot regardless of recency" for free.

### 3. Resource accounting — a real budget abstraction, not a single heuristic

The first draft proposed extending `MmapPrefault.ShouldRun`'s RAM check into the long-term
admission authority. That's insufficient: Stingray has host RAM, native (non-GC) allocations,
mmap'd-but-not-necessarily-resident model pages, KV cache, pinned buffers, and — for
Cuda/Vulkan-hybrid runtimes — a second, independent resource axis in VRAM, which can be full
while host RAM is not, or vice versa. A single `TotalAvailableMemoryBytes` check can't
represent that.

```csharp
public interface IResourceBudget
{
    ResourceHeadroom HostMemory { get; }
    ResourceHeadroom? AcceleratorMemory { get; }   // null when the runtime is CPU-only
}

public enum AdmissionResult { Allowed, AllowedWithEviction, InsufficientMemory, RetryLater }

public interface IResourceAdmissionController
{
    AdmissionResult CanLoad(ModelRuntimeSpec candidate, IReadOnlyList<ModelRuntime> resident);
}
```

`MmapPrefault.ShouldRun`'s existing calculation (`GC.GetGCMemoryInfo().TotalAvailableMemoryBytes`
vs. 80% threshold) is not thrown away — it's the seed for `HostMemory`'s headroom estimate, and
keeps doing its existing, narrower job (should this *specific* load eagerly pre-fault) exactly
as today. It stops being asked to *also* be the multi-model admission authority; that's now an
explicit, separately-testable component with a real return type (`AllowedWithEviction` vs.
`InsufficientMemory` vs. `RetryLater` are meaningfully different outcomes for a caller, and a
bare bool wasn't going to capture that).

### 4. Model Residency Scheduler

Sits in front of `ModelRuntimeManager`. One pending-request queue per known model. Exposes a
small info surface so the scheduling policy can evolve independently of the residency
mechanics underneath it:

```csharp
public readonly record struct ModelSchedulingInfo(
    ModelId ModelId,
    int PendingRequests,
    TimeSpan OldestRequestAge,
    int ActiveRequests,
    bool IsResident,
    bool IsPinned,
    TimeSpan EstimatedLoadCost);
```

Mechanics:

- The scheduler tracks one **active** model at a time (see "Non-goals" — no concurrent
  execution in this design) and keeps feeding that model's queue into its engine
  (`ContinuousBatchingEngine`'s own batching behavior is unchanged) for as long as there's work
  for it, rather than round-robining per request.
- **Switch trigger** — either of:
  - The active model's queue empties.
  - A **service quantum** expires: the active model has now served N tokens (or M
    milliseconds) continuously, *and* at least one other model has pending work. This bounds
    how long a high-traffic model can monopolize the runtime even if its own queue never truly
    empties — the first draft only had an SLA-based override reacting to another model's
    staleness, which is necessary but not sufficient on its own (see below).
  - A non-active model's `OldestRequestAge` exceeds a configurable SLA threshold — a hard
    latency guarantee, independent of throughput or quantum accounting.
- On switch, the scheduler calls `ModelRuntimeManager.GetOrLoadAsync(nextModel)`, which may
  evict the previously-active model (only if `CanEvict`) if capacity requires it.
- This is not the classic disk-elevator/SCAN algorithm — there's no linear "direction" across
  models to scan in. It's closer to a cooperative, quantum-bounded gang scheduler with a hard
  SLA override. Worth naming precisely in code/comments so nobody goes looking for
  elevator-specific literature that doesn't apply.
- The scoring formula for "which model should be active next" does not need to be sophisticated
  on day one — `ModelSchedulingInfo` above is deliberately just data, not a scored ranking, so
  a simple initial policy (oldest-queue-first, subject to the quantum and SLA rules) can later
  be replaced by something weighing `PendingRequests`, `EstimatedLoadCost`, etc. without
  touching `ModelRuntimeManager` or the queues themselves.

### 5. Session and KV affinity

This was the largest omission in the first draft, and it's the direct justification for
`ActiveSessions` on `ModelRuntime` in §1 — restated here as its own section because it's a
correctness boundary, not an implementation detail of eviction.

A session (`OpenTail.Stingray.Sessions`) carries model affinity as a fundamental property: its
KV state is only meaningful against the `ModelRuntime` it was built on. Two ways to handle a
model going cold while it still has live sessions:

- **Pin-on-session (Phase 1, this plan).** A model with `ActiveSessions > 0` is never a
  candidate for eviction — `CanEvict` already encodes this. Simple, safe, and requires no new
  session-lifecycle machinery: sessions already know which model they're bound to; the runtime
  manager just needs to be told when a session starts and ends.
- **Suspend/resume across eviction (explicitly future work, not this plan).** Snapshot a
  session's KV state, allow its model to evict, reload the model later, and restore the KV
  snapshot. Given Stingray's existing session/KV infrastructure this is a genuinely valuable
  future direction — it turns "resident models" into a soft cache even for session-bound
  traffic — but it's materially more complex (KV serialization, restore-time validation that
  the reloaded model is byte-identical to the one the snapshot was taken against, etc.) and is
  explicitly deferred so it doesn't block or complicate this plan's Phase 1–5 rollout.

### 6. API layering — router above engine, not inside it

The first draft proposed adding a model-id parameter to `IInferenceEngine.GenerateChunksAsync`
directly. That conflates two responsibilities that should stay separate: *deciding which
model* handles a request, and *executing* a request against an already-selected model.
`IInferenceEngine` stays exactly what it is today — "I execute against one model" — which is
also what keeps single-model backward compatibility trivial to reason about (§ below). A new
interface sits above it:

```csharp
public interface IInferenceService
{
    IAsyncEnumerable<string> GenerateAsync(
        ModelId modelId, string prompt, SamplingParams sp, CancellationToken ct);
}
```

```
        HTTP / API
             │
             ▼
    IInferenceService              — model selection, request admission
             │
             ▼
  Model Residency Scheduler        — per-model queues, quantum, SLA
             │
             ▼
    ModelRuntimeManager            — residency, async load, safe eviction
             │
      ┌──────┴──────┐
      ▼             ▼
 ModelRuntime A  ModelRuntime B
      │             │
      ▼             ▼
 IInferenceEngine IInferenceEngine  — unchanged: "I execute against one model"
      │             │
 ContinuousBatching ContinuousBatching
      │             │
    KV A           KV B
```

Existing single-model callers (CLI, current Server wiring) keep talking to `IInferenceEngine`
directly and are entirely unaffected; `IInferenceService` is additive.

### 7. Test-suite application (the dogfood, and the fix)

Give the test host process a shared, process-wide `ModelRuntimeManager` instead of each test
file independently calling `GgufModel.Open`/`Dispose` per test method. Ten of the ~37
model-loading test files in `Tests.ForwardPass` load the *same* SmolLM2 model — under a shared
manager those become cache hits, not ten independent load/prefault/dispose cycles.

**Two tests, not a reordered test suite.** The first draft proposed a custom xUnit test orderer
grouping classes by model. Dropped: it would demonstrate the best case but risks the *ordering*
being what's actually responsible for any improvement, rather than the scheduler. Instead:

- **Adversarial test** — force interleaved execution (`A B A C B A C B...`, i.e. close to
  today's de-facto scattered order) and assert the runtime manager reduces *physical* model
  loads relative to *requested* loads (i.e. cache hits are actually happening despite the bad
  ordering).
- **Grouped test** — force clustered execution (`A A A B B B C C C`) as a sanity ceiling: near-
  zero unnecessary reloads, confirming the mechanism works at all before trusting the
  adversarial result.

Measure and record, before/after:

| Metric | Before | After |
|---|---|---|
| Peak working set | 59.5 GB (measured) | ? |
| Requested model loads | ~40 | ~40 (unchanged — same tests) |
| Physical model loads | ~40 (one per test file, no reuse) | ? (goal: ≈ distinct model count) |
| Model transitions (A→B counted) | ~39 | ? |
| Total suite runtime | 663–668s (9-file slice, unfixed) | ? |

If the adversarial-ordering test doesn't show physical loads collapsing toward the distinct-
model count, that's a signal the design has a real gap, not that the test harness needs
reordering.

---

## Backward compatibility

Every existing single-model deployment path — `STINGRAY_MODEL=...`, the CLI, everything
documented today — must produce byte-identical behavior to the current code. Mechanism:
single-model configuration collapses to slot capacity 1, meaning `GetOrLoadAsync` always hits
after the first call (nothing else was ever loaded to evict), the scheduler only ever has one
non-empty queue (nothing to switch between, no quantum/SLA logic ever engages), and the
admission controller never has cause to return anything but `Allowed`. This must be true by
construction from the general-case code, not maintained as a separate path — a separate path
is a second thing to keep correct and test.

## Phased rollout

1. **`ModelRuntime`** — extract the current `GgufModel` + `ForwardPass` + `IInferenceEngine`
   construction into the reusable bundle with its state machine. No manager, no scheduler yet.
2. **`ModelRuntimeManager`** — residency (`SlruCache`-backed), async loading with single-flight
   coalescing, pinning, and `CanEvict`-gated safe eviction (`ActiveRequests`/`ActiveSessions`
   tracking wired in from the start, not bolted on later — this is where the correctness
   guarantee lives). A lightweight manual check against the test-suite workload is reasonable
   here, once loading/caching actually exists to observe — but the formal before/after
   measurement table (§7) waits for Phase 7, once sessions and the scheduler are also in place.
3. **Resource admission** — `IResourceBudget` / `IResourceAdmissionController`, host-memory
   budget first (seeded from `MmapPrefault`'s existing calculation), accelerator/VRAM budget
   stubbed but present in the interface even before it's populated for CPU-only runtimes.
4. **Model Residency Scheduler** — per-model queues, service quantum, SLA-based preemption,
   `ModelSchedulingInfo` exposed. Server-side only, behind opt-in configuration.
5. **`IInferenceService`** — the new router-layer API, additive to (not replacing)
   `IInferenceEngine`. `/v1/models` gains multi-entry responses once this lands.
6. **Session integration** — enforce live-session-pins-model (§5) explicitly; this is the point
   at which eviction safety is proven against the real session subsystem, not just the
   request-level `ActiveRequests` counter.
7. **Dogfood** — point the test suite's model loading at the full stack (manager + scheduler +
   session pinning) and run the adversarial/grouped test pair from §7, recording the metrics
   table for real.

## Open questions

- Exact default for the service quantum and the SLA preemption threshold — needs real
  measurement of swap cost (load + prefault time) across a representative model-size range,
  not a guessed constant. Likely needs to be a function of model size rather than one fixed
  value, given SmolLM2 (~1 GB) and a 50 GB checkpoint have very different load costs.
  Related question: whether the quantum/SLA constants should be additionally weighted by
  `EstimatedLoadCost` from the start, or left flat until measurement shows it matters.
- Whether `IResourceAdmissionController` needs its own background memory-pressure polling loop,
  or can rely entirely on point-in-time checks at swap decisions (the latter is simpler and
  matches how `MmapPrefault.ShouldRun` already works — recommend starting there).
- Exact shape of the KV-suspend/resume future work in §5 — deliberately not designed here
  beyond "deferred," since designing it properly needs its own pass over the session subsystem.

## Changes from the reviewed draft

For traceability: the external review's five mandatory changes, and where each landed above.

1. *Keep `IInferenceEngine` single-model; put model selection above it.* → §6.
2. *Make model loading asynchronous and single-flight.* → §2.
3. *Make eviction state/session-aware; never evict a model with live state.* → §1 (`CanEvict`), §5.
4. *Introduce proper resource accounting instead of `MmapPrefault.ShouldRun` as the long-term
   authority.* → §3.
5. *Add explicit session→model affinity/KV semantics.* → §5.

Also incorporated: `LoadedModel` renamed `ModelRuntime` with an explicit state machine (§1);
"Multi-Model Scheduler" renamed **Model Residency Scheduler** and scoped explicitly to
single-active-model turn-taking, not concurrent execution (Non-goals, §4); service-quantum
concept added alongside the SLA override so a continuously-busy model can't monopolize
residency indefinitely (§4); `ModelSchedulingInfo` exposed as a stable seam between scheduling
policy and residency mechanics (§4); GPU/VRAM handling promoted from an open question into the
`IResourceBudget` design itself (§3); custom xUnit ordering dropped in favor of an
adversarial/grouped test pair that isolates the scheduler's effect from test ordering (§7);
concrete before/after metrics table added (§7); "genuinely novel ground" softened to reflect
that the conclusion rests on one comparison, not a field survey (vLLM section).
