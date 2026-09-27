---

# Multi-Model Inference Runtime, Residency & Scheduling

## Status

**Phases 1, 2, and 4 implemented. Phase 3 at 9 slices (host + accelerator admission, eviction,
observability). Phase 5's no-lock half proven with real models. Phase 6 at 4 slices (the queue
step of the eviction hierarchy, bounded, observable, fair among queued waiters, and now immune to
a fresh caller jumping that queue — execution-level service quantum/starvation-protection still
open). Phase 7 implemented for the
stateless request surface (chat completions, messages, responses, model listing) and, as a
follow-up, `/v1/sessions/*` — a session now holds a real `ModelRuntimeHandle` for its lifetime via
`SessionModelRegistry`, closing the Phase 4 gap; durable/cold session restore across a process
restart in multi-model mode remains explicitly deferred.** See each phase's own entry in
"Implementation phases" below for exact done/not-done
splits. `ModelId`, `ModelRuntimeState`, `ModelResidencyMode`,
`ModelRuntime`, `ModelRuntimeHandle`, `IModelRuntimeManager`/`ModelRuntimeManager`
(`src/OpenTail.Stingray.Server/ModelRuntime.cs`, `ModelRuntimeManager.cs`), wired into
`ServiceCollectionExtensions.AddOpenTailStingray` so the server's single configured model now
loads through the manager's single-flight path (pinned, so DI retains sole disposal ownership).
`ModelResidencyMode` (`SingleSlot`/`MultiSlot`) is configurable via
`OpenTailStingrayServerOptions.ModelResidencyMode` or `STINGRAY_MODEL_RESIDENCY_MODE`, and is a
runtime-mutable property (not just a boot-time constant) so residency policy can change without a
restart.

Phase 2's canonical-identity/shared-handle/single-flight/safe-disposal/load-failure-cleanup work
(carrying forward `SharedModelCache`'s (`025`/`026`) proven refcount ownership) turned out to be
load-bearing for Phase 1's `AcquireAsync` to be correct at all, so it landed in the same pass
rather than as a separate increment — nothing here reflects a second implementation effort, just
the literal Phase 2 acceptance bar pinned down explicitly in its own test. (`026`'s
capacity-bounded-eviction/overflow-table mechanism is genuinely separate work and remains
Phase 3, "Bounded, resource-aware residency" — not yet built.)

Covered by `tests/OpenTail.Stingray.Tests.Server.Fast/ModelRuntimeManagerTests.cs`: single-flight
(100 concurrent cold requests → 1 physical load, 100 logical users, verified safe mass-release
back to zero with no over-release and no premature disposal), lifecycle, SingleSlot eviction/wait,
live mode-switch, no-global-lock, isolated cancellation. Plus the full existing `Tests.Server.Fast`
suite (271/271) and the real-GGUF `Tests.Server` session-restart acceptance test, both green with
zero behavior change to the single-model path. Phases 3–7 (below) remain design only.

Implementation-ready design. Supersedes the shelved production portion of
`done/024-multi-model-serving-and-request-scheduling-plan.md`. Builds directly on
`done/025-shared-model-cache-phase1-plan.md` and `done/026-shared-model-cache-phase2-eviction-plan.md`
(both implemented — `src/OpenTail.Stingray.Core/SharedModelCache.cs`), and reopens the decision
recorded in `done/027-model-cache-scope-decision.md`.

## Why 027's decision is being revisited, not overturned

027 shelved `024` because "concurrent production requests actually competing for several
different models at once" (**Problem C**) had no demonstrated caller — every entry point (CLI,
`OpenTailStingrayServerOptions`, `InferenceEngineLoader`) was, and still is today, single-model
by construction (`STINGRAY_MODEL` → one `ContinuousBatchingEngine`). That reasoning was correct
*at the time*. The OpenTail product direction has since been clarified: OpenTail is a multiuser
local AI service built around a cheap, tool-capable, near-permanently-resident **sidekick**
model (4–8B) that escalates hard tasks to an occasionally-resident **specialist/reasoner** model
(14–30B), and both may genuinely serve concurrent users at once. That is Problem C, now with a
real caller. The engineering discipline from 027 still applies — this plan does not resurrect
`024`'s full taxonomy (SLA preemption, complex scoring, a new session architecture) wholesale.

## Design premise

**We are not building a model-switching server. We are building a multi-model inference
runtime.** The expected steady-state deployment is 1–2 resident models — a sidekick and,
sometimes, a specialist — but Stingray must make no architectural assumption that `N == 1` or
`N == 2`. Residency is driven by memory pressure, not a configured model count:

- A 16 GB machine naturally runs single-slot (sidekick, or sidekick evicted for reasoner on
  demand).
- A 32–64 GB machine naturally keeps both resident.
- A larger workstation can hold more, without any code change.

Same-model concurrency (many users → one sidekick) is the common case and is **already solved**
by `HotSession` + `ContinuousBatchingEngine` + paged KV — this plan does not touch that path.
The new problem is scheduling and resource accounting *across* independently loaded models, and
letting two different models generate at the same time when hardware permits it.

## Design principles

1. **N is a capability, not the deployment size.** Model instances = N; resource policy
   determines how many stay resident. No `MaxModels` constant driving admission.
2. **Model count is not the resource budget.** Models differ wildly in size, KV needs, and
   accelerator footprint — admission must be resource-based (host RAM, accelerator memory
   tracked separately), not "count < limit".
3. **Same-model concurrency stays owned by existing machinery.** The multi-model scheduler picks
   *which model runtimes* run; it never re-implements per-session token batching.
4. **Different-model concurrency is allowed, not special-cased.** No process-wide inference
   lock. Two `ModelRuntime`s with active work may both be mid-decode simultaneously.
5. **A session's model identity is immutable.** A `HotSession` belongs to exactly one
   `ModelRuntime` for its lifetime; switching models means a different session.
6. **Model residency and KV residency are separate concerns**, owned by separate subsystems
   that don't reach into each other (model manager never touches KV internals; KV governor never
   knows about models).

## New core abstraction: `ModelRuntime`

The fundamental unit is a complete, independently executable model runtime — the production
successor to `SharedModelCache`'s `ModelHandle`, extended with residency state and load/session
accounting instead of a bare refcount.

```csharp
public sealed class ModelRuntime : IDisposable
{
    public ModelId Id { get; }                    // canonical: resolved model path today
    public GgufModel Model { get; }
    public IInferenceEngine Engine { get; }        // wraps the existing ContinuousBatchingEngine
    public long EstimatedModelBytes { get; }
    public ModelRuntimeState State { get; }
    public int ActiveRequests { get; }
    public int ActiveSessions { get; }
    public bool IsPinned { get; }
}

public enum ModelRuntimeState
{
    Unloaded, Loading, Ready, Active, Draining, Evicting, Failed, Disposed
}
```

`ActiveRequests`/`ActiveSessions` are **manager-side accounting/observability, not authoritative
session ownership**. `HotSession` (and whatever session registry sits above it) remains the
single source of truth for a session's lifecycle; `ModelRuntime` only counts references it has
itself been handed via `ModelRuntimeHandle`. Do not let these counters become a second place
that decides whether a session is alive — that's exactly the kind of double-bookkeeping that
produces the eviction hazard the invariant below guards against.

`ModelId` is deliberately an abstraction even though the Phase 1 implementation resolves it as a
canonical model path. Do not treat a raw, user-supplied path string as identity — resolve it to
a canonical form once, at acquisition time, so that later work (aliases, multiple paths
resolving to one physical model) doesn't require redefining identity everywhere it's used. A
content hash is not needed yet; the boundary just needs to already exist.

**Eviction invariant**: a runtime is evictable only when `State` permits it AND
`ActiveRequests == 0` AND `ActiveSessions == 0` AND `!IsPinned`. LRU recency alone is never
sufficient — this is the exact live-session-eviction hazard `024`'s review caught, and it still
applies.

## `IModelRuntimeManager`

```csharp
public interface IModelRuntimeManager
{
    ValueTask<ModelRuntimeHandle> AcquireAsync(ModelId model, CancellationToken ct = default);
    bool TryGetResident(ModelId model, out ModelRuntime runtime);
}
```

Owns model identity, single-flight loading, residency, reference tracking, eviction, and
disposal — the production replacement for `SharedModelCache`, not a second ownership system
next to it. `SharedModelCache.Acquire(string path)` today does plain refcounting with no
eviction-vs-in-use distinction beyond what `026` already added via `SlruCache` + overflow table;
`IModelRuntimeManager` is that same mechanism plus the state machine and scheduling seam needed
for concurrent, resource-aware production use.

### `ModelRuntimeHandle` is the authoritative residency lease

This is the mechanism that makes eviction safe, so its contract must be explicit rather than
implied by `ActiveRequests`:

- Holding a `ModelRuntimeHandle` prevents the runtime it was acquired for from becoming
  evictable, full stop.
- Disposing the handle releases exactly that caller's residency claim; it does not itself evict
  anything.
- The manager must never infer liveness from request timestamps, LRU position, or any other
  proxy — only from outstanding handles.

Mental model: `Acquire → handle → "you now hold this runtime" → Generate → Dispose handle →
runtime may become evictable`. `ActiveRequests`/`ActiveSessions` (above) are observability
derived from outstanding handles, not an independent gate.

**Handle lifetime is scoped to the operation, not to model selection.** A handle should normally
be held for the duration of the work that actually requires residency — acquire, create/use a
session, generate, dispose — not retained indefinitely as a marker of "this is the currently
selected model":

```
Acquire → handle → create/use session → generation → handle.Dispose()
```

A caller that retains a handle past the operation that needed it (e.g. as a long-lived
"currently selected sidekick" reference) accidentally pins that runtime forever, defeating
eviction entirely. `IInferenceService.GenerateAsync` (below) should acquire and dispose its own
handle around each call rather than exposing handle lifetime to OpenTail.

### Acquire-vs-evict linearization

The manager needs a single linearization point around acquisition and eviction so this race
cannot happen:

```
A: evict starts
B:                 Acquire()
A: dispose()
B:                 gets a handle to a runtime that's being/been disposed
```

Concretely: a physical model/engine instance is disposed exactly once, and no caller can obtain
a usable handle once disposal has begun — `AcquireAsync` and the eviction path must serialize
through the same lock/state-check, not race against independent reads of `State`.

### Single-flight loading (mandatory)

Five concurrent requests for a cold model must produce one physical `GgufModel` load, all
callers awaiting the same `Task<ModelRuntime>`. A failed load must not leave that task
permanently faulted in the single-flight table — waiters observe the failure, then the entry is
cleared so a later request can retry (bounded retry/backoff, not silent poisoning forever).

## Resource admission

```csharp
public interface IResourceBudget
{
    ResourceSnapshot GetCurrent();
    ResourceAvailability EstimateAdmission(ModelRuntimeSpec model, InferenceWorkEstimate work);
}
```

Track host memory, accelerator memory, model residency, and KV memory as **separate** figures —
never collapsed into one number (20 GB free host RAM says nothing about 1 GB free VRAM).
Residency policy is admission-driven, not a fixed cap:

```
new model request → resource admission → allowed | evict-then-load | queue/reject
```

A configurable soft residency budget may exist as a safety ceiling, but actual resource pressure
is the primary mechanism — this is what makes the same binary behave as single-slot on 16 GB and
multi-resident on 64 GB with no configuration change.

**Phase 1 admission is an estimate, not an exact allocator.** Real usage includes model weights,
KV, temporary inference/workspace allocations, runtime overhead, OS headroom, mmap'd-GGUF
behavior, memory-bandwidth contention, and compute saturation — modeling all of that precisely
is its own project. `EstimateAdmission` should be conservative (built-in safety margin, biased
toward under-admitting) rather than attempt to predict every runtime allocation exactly. Model +
KV memory is the Phase 1 estimate; the rest stays headroom, not something the formula claims to
account for.

## Scheduling — two levels, not one

- **Level 1 (new, this plan): model runtime.** Which model runtimes get to execute right now?
  Exposed via a stable seam so the initial policy can stay simple without a rewrite later:

  ```csharp
  public readonly record struct ModelSchedulingInfo(
      ModelId ModelId, int PendingRequests, int ActiveRequests, int ActiveSessions,
      bool IsResident, bool IsPinned, TimeSpan OldestRequestAge, TimeSpan EstimatedLoadCost);
  ```

  Initial policy: never starve a model indefinitely; favour resident + interactive work; respect
  request age; don't evict an actively useful model to save a small amount of memory; don't let
  one busy model monopolise all execution capacity. No scoring engine in Phase 1 — the seam
  matters more than the formula.

- **Level 2 (existing, unchanged): session/batch.** Which sessions within one model's next
  decode batch run? Owned entirely by `ContinuousBatchingEngine` as today.

A continuously busy model does not get unconditional service — apply a service quantum, but
**only when the underlying resource is actually contended**. When CPU/GPU headroom exists, two
runtimes run fully concurrently; the quantum is a fairness mechanism for contention, not a
scheduling ceremony imposed by default.

**The model scheduler is an admission/eligibility gate, not a global token arbiter.** It decides
*which runtimes are allowed to have work in flight*; it does not sit in the token-generation path
serializing them. Once two runtimes are admitted and the execution backend has headroom, their
inference tasks execute concurrently — actual overlap (Model A decode running while Model B
decode is also running), not `A → B → A → B` alternation. This distinction matters because it's
easy to satisfy "two models can be resident" with an implementation that quietly serializes all
inference behind one semaphore and still calls itself a scheduler — see the concurrency test
below, which exists specifically to catch that.

## Eviction hierarchy (ownership stays split)

```
Model residency  →  Model runtime eviction   (new: IModelRuntimeManager)
KV residency     →  Session suspension       (existing: KV memory governor)
```

**Phase 1 eviction rule is deliberately narrow**: evict only when `ActiveRequests == 0` AND no
live `ModelRuntimeHandle`s remain. That's it — no draining. "Runtime whose active sessions can
safely drain" is real future work, but it's explicitly **deferred past Phase 1**, not built into
the first implementation: draining active sessions to reclaim memory is exactly the kind of
session/KV-lifetime complexity `027` shelved `024` to avoid, and getting it wrong reintroduces
the live-session-eviction hazard this plan otherwise protects against. Full preferred order,
for reference and later phases: unused runtime → idle/evictable runtime → (deferred) runtime
whose active sessions can safely drain → KV/session suspension → queue → hard failure. Phase 1
keeps model residency and KV residency as *separate, explicitly-owned* mechanisms rather than one
coordinator reaching into both — a later policy layer can coordinate them once both halves are
proven independently.

## Request/cancellation semantics

A request is always in one of: queued, loading (joined single-flight), running, completed,
cancelled, rejected — never silently dropped into an unbounded queue. Cancellation propagates
request → scheduler → runtime → session → generation, but cancelling *one* waiter on a shared
cold-load must not cancel the load for other waiters still attached to it.

**Queues are bounded and cancellable from Phase 1, not deferred to Phase 6.** The full fairness
policy (per-model pending queues, service quantum, starvation protection) is Phase 6 work, but
the baseline safety property is not a scheduling refinement — it's an invariant: every queue
(per-model and global) has a capacity, every queued request carries a deadline or honours its
`CancellationToken`, and queue overflow is an explicit, immediate rejection rather than unbounded
growth. Without this, an overloaded machine stays "correct" (no corruption, no crash) while
silently accumulating an unbounded backlog — that's a real Phase 1 requirement, not sophisticated
backpressure. Phase 6 refines *which* queued request runs next; it doesn't introduce the queue's
existence or its bound.

## OpenTail-facing API

OpenTail should not see the scheduler/admission machinery — just:

```csharp
public interface IInferenceService
{
    IAsyncEnumerable<InferenceChunk> GenerateAsync(
        InferenceRequest request, CancellationToken ct = default);
}

public sealed record InferenceRequest
{
    public required ModelId Model { get; init; }
    public required SessionId Session { get; init; }
    // existing generation parameters...
}
```

`request → ModelId → ModelRuntime → HotSession → continuous batch`, fully opaque to the caller
(whether the model was already resident, just loaded, sharing a batch, or executing alongside
another model is entirely Stingray's concern). If only one model is ever requested,
`STINGRAY_MODEL` behaviour must be observably unchanged from today.

## Explicit non-goals

Distributed/multi-machine inference, model ensembles or voting, automatic "best model"
selection/routing intelligence, LoRA adapter scheduling, cross-model physical batching, model
quantization/conversion, changes to speculative decoding, KV-cache implementation,
`HotSession`, or the KV memory governor, GPU optimization beyond resource accounting, and
background/speculative model preloading. All out of scope here.

## Implementation phases

1. ✅ **Production `ModelRuntime` abstraction.** `ModelRuntime`, `ModelRuntimeHandle`,
   `ModelRuntimeManager`, `ModelRuntimeState`. Move the existing single-model load path
   (`InferenceEngineLoader`) behind it. *Acceptance: existing single-model server tests stay
   green, now routed through the new abstraction.*
2. ✅ **Shared residency & single-flight loading.** Carry `SharedModelCache`'s proven
   refcount/overflow-table ownership (`025`/`026`) into `ModelRuntimeManager`: canonical
   identity, shared handles, async single-flight, safe disposal, load-failure cleanup.
   *Acceptance: 100 concurrent requests for one cold model → 1 physical load, 100 logical
   users.* (Landed alongside Phase 1 — see Status above.)
3. 🔶 **Bounded, resource-aware residency.** Resident tracking, memory estimates, `IResourceBudget`,
   admission, safe eviction — host RAM first, accelerator accounting added through the same
   abstraction rather than a parallel policy. *Acceptance: two models coexist when resources
   permit; a third triggers safe eviction/queueing.*
   — **Slice 1 done:** `IResourceBudget`/`ResourceSnapshot`/`HostResourceBudget`
   (`src/OpenTail.Stingray.Server/ResourceBudget.cs`) — host memory (via `GC.GetGCMemoryInfo`,
   portable, no P/Invoke) plus resident-model-bytes totals, purely observational.
   **Slice 2 done:** `IResourceBudget.EstimateAdmission(long candidateModelBytes)` — a
   weight-only, conservative (25% safety-margin) admission *check*.
   **Slice 3 done:** wired into `AcquireAsync` via the new `IModelRuntimeManager.ResourceBudget`
   property — `null` by default (feature off, zero behavior change; the server DI wiring still
   doesn't set it). When set, a cold load first checks admission, evicts idle/evictable resident
   runtimes and re-checks once if it doesn't fit, and throws `InsufficientResourcesException` as
   the documented hard-failure last resort if it still doesn't. Already-resident acquisitions
   always bypass the gate entirely (fast path returns before admission is ever consulted).
   **Slice 4 done:** `OpenTailStingrayServerOptions.EnableResourceAdmission` (bool?, default
   `null`/off) / `STINGRAY_ENABLE_RESOURCE_ADMISSION` — the only way to actually reach this
   machinery from the real server, since Slice 3's wiring lives on the manager and nothing set it
   in DI until this. Explicitly opt-in rather than a new default: flipping it on changes an
   existing deployment's startup behavior (a load that always succeeded can now throw
   `InsufficientResourcesException` on a machine sized close to the wire), so that's the
   operator's call. Verified default-off leaves the real single-model DI path byte-for-byte
   unchanged, including a live real-GGUF end-to-end rerun.
   **Slice 5 done (docs/032 §"Metrics that matter"):** `MultiModelRuntimeStats`/`GetStats()` —
   cumulative `ModelLoads`/`ModelLoadFailures`/`ModelEvictions`/`AdmissionRejects`/
   `ResidencyPressureEvents` counters incremented at their exact real event sites, plus a
   point-in-time `ResidentModels`/`ActiveModels`/`PendingLoads`/`KnownModels`/
   `EstimatedResidentModelBytes` snapshot. `ResidencyPressureEvents` (pressure detected) is
   counted separately from `AdmissionRejects` (pressure eviction couldn't resolve) — the two
   answer different operational questions. Purely additive/observational, no behavior change.
   **Slice 6 done:** `Snapshot()` now includes models currently mid-load
   (`ModelRuntimeState.Loading`, pre-load `EstimatedModelBytes` estimate, zeroed
   `HandleCount`/`ActiveRequests` since no `ModelRuntime` exists yet, `LastUsed` reporting when
   the load started) alongside resident entries — previously a model between "acquisition
   started" and "acquisition finished" was invisible to every observer. `TryGetResident`
   deliberately left unchanged: it promises a *usable* `ModelRuntime`, and a loading model
   genuinely doesn't have one yet, so returning `false` for it is correct, not a gap.
   **Slice 7 done:** `ResourceSnapshot.AcceleratorMemoryAvailableBytes` now reports real Vulkan
   VRAM capacity when a device is present (`HostResourceBudget`, a throwaway `VulkanBackend`
   probe read once and cached process-wide via `Lazy<T>` — VRAM capacity can't change during a
   process's lifetime, and repeated device probes would be wasteful). Verified against real
   hardware in this environment (not assumed): a temporary strict assertion confirmed a real,
   sane, non-zero figure (~15.7 GiB) before being replaced with the portable
   `is null or > 0` check the permanent test uses, so CI/dev machines without a GPU still pass
   honestly. Deliberately scoped narrow, twice over: (1) this reports total DEVICE_LOCAL
   *capacity*, not capacity-minus-current-usage — Vulkan has no portable "free VRAM right now"
   query without the `VK_EXT_memory_budget` extension, so this stays consistent with the same
   conservative-estimate posture host-memory admission already has; (2) `EstimateAdmission`
   itself is still host-memory-only — wiring an accelerator dimension into the actual admission
   decision needs per-runtime GPU-residency tracking (which runtime actually consumed how much
   VRAM) that doesn't exist yet, since `ModelRuntime` doesn't currently know which backend a
   given `LoadedEngine` runs on. CUDA was not attempted: no CUDA device in this environment to
   verify against, and shipping an unverified native-memory-query path isn't an acceptable
   trade for a "small, safe, tested" slice.
   **Slice 8 done:** `ModelRuntime.IsAcceleratorResident`/`AcceleratorResidentBytesEstimate` —
   per-runtime GPU-residency, derived from `LoadedEngine.RuntimeResolution.Backend`
   ("cpu"/"cuda"/"vulkan"), which `InferenceEngineLoader.DescribeRuntime` already computes
   correctly for every dispatch branch, so this needed zero new loader plumbing. Deliberately
   didn't thread the per-branch `LayerPlacement.GpuWeightBytes` the loader computes internally
   through instead — that would mean touching the many backend-dispatch branches inside
   `InferenceEngineLoader.BuildForwardPass` (CUDA/Vulkan × full/hybrid/hybrid-GDN), real risk in
   a performance-critical, multi-path method, for a precision gain not needed yet. Reuses
   `EstimatedModelBytes` instead: **exact for full GPU offload**, an **overestimate for
   hybrid/partial offload** (documented explicitly in the property's own doc comment) — the safe
   direction for an eventual admission check, so it's usable now, with the honest gap flagged for
   later. `ModelRuntimeStats`/`MultiModelRuntimeStats` extended to expose both the per-runtime and
   aggregate figures. Verified against real Vulkan hardware end-to-end, not just fakes:
   `tests/OpenTail.Stingray.Tests.Server/AcceleratorResidencyTests.cs` loads an actual Qwen3-0.6B
   GGUF on the real device and confirms residency + a positive byte estimate come out of the
   genuine `DescribeRuntime` dispatch path (plus a CPU-loaded counterpart confirming `false`/0).
   **Slice 9 done:** `EstimateAdmission` now actually consults accelerator memory.
   `IResourceBudget.EstimateAdmission` grew a `candidateAcceleratorBytes` parameter (default 0 —
   "not expected to be accelerator-resident", so a caller that never opts in sees byte-identical
   behavior to before this parameter existed). `ModelRuntimeManager` grew a matching
   `estimateAcceleratorBytes` constructor delegate mirroring `estimateBytes` exactly (no default
   estimator — inferring GPU intent isn't the manager's job, it's whatever backend policy the
   loader closure already captures); absent, every candidate is treated as 0 accelerator bytes.
   `HostResourceBudget.EstimateAdmission` checks host memory first (an accelerator-bound candidate
   that already fails on host is reported as a host failure, never silently overwritten), then —
   only when the candidate declares nonzero accelerator need AND capacity is known — applies the
   same safety margin to `capacity - EstimatedAcceleratorResidentBytes` (that subtraction computed
   fresh on every call from a live `Snapshot()` sum, never cached, so it can't go stale relative to
   eviction/new loads). A candidate with real accelerator need but *unknown* capacity is never
   blocked on that alone — no positive evidence it won't fit, so it falls through to the host check.
   `InsufficientResourcesException` restructured to carry the actual `ResourceAdmission` `Reason`
   plus `CandidateAcceleratorBytes`, with a message that correctly names VRAM vs. host memory
   instead of always assuming host. `ResourceSnapshot` gained `EstimatedAcceleratorResidentBytes`
   for symmetry with `ResidentModelBytes`. Verified twice over: deterministic fake-based tests for
   the wiring itself (estimator reaches `EstimateAdmission`, exception carries the right
   reason/bytes, host-vs-accelerator precedence), and — the strongest proof — a full real-hardware
   round trip in `AcceleratorResidencyTests.cs`: a reasonable accelerator estimate loads normally
   on the real Vulkan device with admission ON, and an absurd one (`long.MaxValue / 4`) is rejected
   with `InsufficientAcceleratorMemory` *before* `InferenceEngineLoader.Load` ever runs — confirmed
   via `Assert.Empty(manager.Snapshot())`, i.e. no wasted real GPU load attempt. 304/304 fast,
   8/8 real-model, full solution build clean.
   **Slice 10 done: byte-exact hybrid/partial-offload accelerator accounting**, closing the one
   gap Slice 8 left open. `InferenceEngineLoader.BuildForwardPass` already asked `TierPlanner` for
   a precise per-layer `LayerPlacement` on the CUDA/Vulkan *partial*-offload hybrid paths
   (`CudaHybridForwardPass`/`HybridForwardPass`) — the real number existed, it was just discarded
   after construction. `BuildForwardPass` now also returns `GpuWeightBytesExact` (`long?`): the
   real `planForHybrid.GpuWeightBytes` on those two branches, `null` everywhere else (CPU-only;
   full dense GPU offload, where the whole-file estimate is already exact by construction; and
   hybrid-GDN, whose placement is driven by `hp.LayerTypes` rather than `TierPlanner` and still
   isn't attempted — genuinely separate work, not a discarded-but-available number like the fix
   above). Threaded through a new `LoadedEngine.GpuWeightBytesExact` field;
   `ModelRuntime.AcceleratorResidentBytesEstimate` now prefers it over `EstimatedModelBytes` when
   present, falling back exactly as before when not.
   **Real bug found and fixed along the way, not before it shipped** — a second instance of the
   exact double-free class Phase 5 found and fixed in `ForwardPass.Dispose()`
   (`docs/done/bugstofix-resolved-2026-08.md` → `ForwardPass.cs:6047`), this time in four *other*
   forward-pass types that turned out to share the same missing-idempotency-guard gap:
   `GpuForwardPass`, `CudaForwardPass`, `CudaHybridForwardPass`, and `HybridForwardPass`.
   `InferenceEngineLoader` adds each to both `_fwd` and its engine's `_owned[]` array, and
   `InferenceEngine.DisposeCore` disposes `_fwd` explicitly and then every item in `_owned` —
   without an idempotency guard, every native/VRAM free in `Dispose()` ran twice. Caught by this
   slice's own new real-hardware test (`AcceleratorResidencyTests.RealVulkanModel_PartialOffload_...`,
   the first test in this codebase to construct-then-dispose a genuinely partial-offload
   `HybridForwardPass` through `ModelRuntimeManager`): a native access violation
   (`0xC0000005` in `NativeMemory.Free`) during `manager.Dispose()`, not a managed exception —
   exactly the same crash signature Phase 5's writeup describes. `GpuForwardPass`/`CudaForwardPass`
   had silently lacked the guard the whole time too (full-offload real-model tests had run and
   "passed" repeatedly without visibly crashing — a double-`NativeMemory.Free`/VRAM-free is
   undefined behavior, not a guaranteed-immediate fault, so its absence had gone undetected). Fixed
   all four with the same established `_disposed`-guard pattern
   (`if (_disposed) return; _disposed = true;` as the first statement in `Dispose()`), verified
   both by the new test passing cleanly and by rerunning the full existing accelerator/cross-model
   real-model suites. 5/5 `AcceleratorResidencyTests`, 12/12 `Tests.Server` real-model, full
   solution build clean.
   **Not yet done:** the hybrid-GDN full-offload branches (CUDA/Vulkan) still fall back to the
   file-size estimate rather than a `TierPlanner`-precise figure — their placement isn't
   `TierPlanner`-driven, so no ready-made precise number exists there the way it did for the two
   branches this slice fixed; computing one would mean new placement logic, not just threading an
   existing value through. A CUDA equivalent of the VRAM-capacity probe is parked, not planned —
   no CUDA hardware in this environment to build or verify it against; revisit if that changes.
4. ✅ **Multi-session model execution.** Wire each runtime to `HotSession` + continuous batching.
   *Acceptance: N sessions on one runtime behave exactly as today's same-model concurrency.*
   Proven with a real model, not fakes:
   `tests/OpenTail.Stingray.Tests.Server/SessionRestartPersistenceTests.cs`'s
   `ConcurrentSessions_RealCpuGguf_ContinuousBatchingKeepsSessionsIndependent` runs 3 genuinely
   concurrent sessions (distinct low-perplexity prompts, greedy decoding) against one engine
   loaded through `ModelRuntimeManager.AcquireAsync`, and checks each session's answer is
   correct and uncontaminated by the others — not just that nothing throws.
   **Known gap, deliberately deferred to Phase 7, not built now:** a `HotSession` doesn't hold a
   `ModelRuntimeHandle` for its lifetime. Today this is safe only because the server's one engine
   is always `IsPinned` (never evicted regardless of handle count) — a live-but-idle session
   would otherwise look evictable (`HandleCount == 0`) even though it could resume generating at
   any moment, which is exactly the "eviction destroys live session state" hazard `024`'s review
   flagged and docs/032 §15 requires never happen. Fixing this now would mean inventing
   session-to-model handle plumbing with no real caller — Phase 7 is what actually defines how a
   session gets bound to a specific (non-pinned) model, and the fix belongs there, built against
   that real API rather than guessed at ahead of it.
5. ✅ **Cross-model concurrent execution** *(no-lock half; model-level resource scheduling is
   still Phase 6)*. `ModelRuntimeManager` never held a lock across a load or generation call to
   begin with, so there was no serialization to remove — what this phase actually needed was
   *proof*, with real models, not just the fake-loader tests in
   `ModelRuntimeManagerTests.cs`.
   *Acceptance: two independent models demonstrably overlap execution, not turn-take.*
   `tests/OpenTail.Stingray.Tests.Server/CrossModelConcurrencyTests.cs`'s
   `TwoRealModels_GenerateConcurrently_OverlapRatherThanSerialize` loads SmolLM2-1.7B and
   Qwen3-0.6B (two different real GGUFs, different architecture families) and proves genuine
   interleaving — one model's stream produces output before the other's has finished. A first
   attempt asserted total wall-clock time was well below the serial sum instead, and false-failed:
   real CPU-bound models genuinely contend for the same cores/memory bandwidth (see "8. CPU-only
   systems" earlier in this doc's own history), so partial slowdown from contention is expected
   physics, not evidence of a lock. The interleaving check is robust to that; a raw timing
   threshold isn't.
   **Real bug found and fixed along the way:** disposing a plain (non-batching) `InferenceEngine`
   after real generation crashed the process natively (heap corruption). Bisection showed neither
   two models nor concurrency nor even generation were required — a single model, load-then-
   immediately-dispose, reproduced it just as reliably, which pointed straight at the real cause:
   `InferenceEngine.DisposeCore` calls `_fwd.Dispose()` explicitly and then disposes every item in
   `_owned`, which also contains that same `ForwardPass` instance — a double-free, since
   `ForwardPass.Dispose()` had no idempotency guard (unlike every other disposal type in this
   codebase). Fixed with that same established `_disposed`-guard pattern. Full writeup and
   verification (four test suites rerun green, including the largest one touching this file):
   `docs/done/bugstofix-resolved-2026-08.md` → `ForwardPass.cs:6047`. Regression guard:
   `CrossModelConcurrencyTests.cs`'s `SingleRealModel_DisposeAfterGeneration_DoesNotCorruptTheNativeHeap`.
6. 🔶 **Fair scheduling & admission.** Per-model pending queues, request age, service quantum,
   starvation protection, cancellation, queue limits — deliberately simple policy.
   **Slice 1 done: the "queue" step itself** (docs/032's own eviction hierarchy: unused → idle →
   drain → KV suspension → queue → hard failure — this was the one step still unbuilt).
   `IModelRuntimeManager.AdmissionWaitTimeout` (`TimeSpan?`, default `null` — an admission
   failure still hard-fails immediately unless explicitly opted into, same discipline as every
   other slice). When set, a candidate that fails admission (even after eviction) waits, bounded
   by this duration and always cancellable via the caller's own token, for a residency change —
   a handle released, a runtime evicted elsewhere, `ResidencyMode` switched — then retries the
   *whole* acquisition from scratch rather than failing immediately. Built by reusing the exact
   wait-and-retry mechanism already proven for `SingleSlot` mode's blocking-wait path
   (`_residencyChanged`), not new machinery — the only new logic is the timeout wrapped around
   that same wait, and routing `TryEnsureAdmissibleLocked`'s failure through either "throw now"
   or "queue" depending on whether a timeout is configured. `EnsureAdmissibleLocked` (threw
   directly) became `TryEnsureAdmissibleLocked` (returns `bool` + `out` exception) so the *caller*
   decides which of those two policies applies, keeping the admission-check method itself
   policy-free. `AdmissionRejects` only increments on an actual give-up (immediate hard-fail, or
   a queue timeout) — a caller that queues and then successfully retries is correctly never
   counted as a rejection, verified explicitly by test. Honest scope limits, stated in the
   property's own doc comment: bounded *per admission attempt*, not in total (a candidate that
   repeatedly almost-fits can retry across several timeouts); and bounded by wait duration only,
   not yet also by a queue-depth limit (no cap on how many callers can simultaneously queue).
   Verified: 4 new tests covering queue-then-succeed, queue-then-timeout (with the real elapsed
   duration measured, not just the exception type), cancellation during a queued wait producing
   `OperationCanceledException` not a resource exception, and the pre-existing hard-fail-immediately
   behavior staying byte-identical when the timeout isn't configured. 308/308 fast, 8/8 real-model,
   full solution build clean — this touched the core `AcquireAsync` loop directly, so the full
   regression suite mattered more than usual here.
   **Slice 2 done: queue-depth bound + observability.** `IModelRuntimeManager.MaxQueuedAdmissions`
   (`int`, default 16 — matching the existing `OpenTailStingrayServerOptions.MaxQueuedRequests`
   precedent elsewhere in the codebase). A candidate that would otherwise queue is instead
   rejected immediately (counted as both `AdmissionRejects` and a new `AdmissionQueueOverflows`)
   once `_queuedAdmissions` is already at the cap — an unbounded number of simultaneous waiters
   was the one remaining way this feature could turn a resource-pressure moment into unbounded
   memory/task growth, the same "every queue has a capacity" invariant the plan already requires
   elsewhere (see the eviction-hierarchy note above). `MultiModelRuntimeStats` gained three
   fields: `QueuedAdmissions` (current waiter count), `OldestQueuedAdmissionAge` (`TimeSpan?`,
   how long the longest-waiting candidate has been queued — `null` when nothing is queued), and
   `AdmissionQueueOverflows` (cumulative rejection-due-to-full-queue count). Bookkeeping is
   `lock`-protected alongside the existing admission state; the wait path increments on entry and
   decrements in a `finally` so a timeout, a cancellation, or a successful retry all correctly
   free the slot. Verified: 2 new tests (queue-at-capacity rejects immediately rather than
   waiting; `OldestQueuedAdmissionAge` tracks real elapsed time via `Stopwatch`, not just
   presence/absence). 310/310 fast, full solution build clean; the 8/8 real-model suite re-run
   after this slice too since it again touched `AcquireAsync` directly.
   **Slice 3 done: admission fairness (oldest-admissible wake, not broadcast re-race).** Replaced
   the shared-broadcast-then-everyone-races wake with a real per-waiter ordered queue.
   `_admissionQueue` (`List<AdmissionWaiter>`) holds one `AdmissionWaiter` per currently-queued
   caller — `ModelId`, its candidate byte estimates cached at first queue entry, `EnqueuedAt`, and
   its own `TaskCompletionSource`. `WakeOldestAdmissibleQueuedWaiter` (called from
   `NotifyResidencyChanged`) evaluates every queued candidate against the current
   `ResourceBudget`, oldest-`EnqueuedAt`-first, and wakes **exactly one**: the admissible candidate
   with the smallest `EnqueuedAt`, not just the physically-oldest entry. This is deliberately
   *oldest-admissible*, not strict head-of-line FIFO — a strict-FIFO design would let one
   oversized, permanently-unfittable oldest waiter block every smaller, genuinely-serviceable
   waiter behind it forever, which `032`'s own resource-based (not count-based) admission model
   makes a real risk, not a hypothetical one. A waiter that's woken, retries via the unchanged
   `TryEnsureAdmissibleLocked`, and still fails re-queues *itself* (same `AdmissionWaiter`
   instance, `EnqueuedAt` untouched) rather than being treated as a new arrival, so a repeatedly-
   almost-fitting request never loses its place in line. `EvictIdleOthersLocked`'s `keep`
   parameter became nullable (`ModelId?`) so the scan can evict every idle resident runtime with
   no exception — safe because none of the queued candidates are themselves resident.
   **Found and fixed during this slice, not before it shipped:** wiring the scan into
   `NotifyResidencyChanged()` unconditionally created a real, deterministic infinite-load bug —
   `RunLoad`'s own success path already called `NotifyResidencyChanged()` (pre-existing, for
   SingleSlot's benefit), and a model is briefly idle/evictable in the window between `RunLoad`
   registering it as resident and the awaiting caller actually constructing its `ModelRuntimeHandle`
   one lock acquisition later. With the scan wired in, that freshly-loaded-but-not-yet-handled
   model looked "idle" to `EvictIdleOthersLocked` and got evicted by its own completion event,
   which made the caller reload it, which evicted it again — observed directly via a test
   diagnostic as `ModelLoads` reaching 1.6M in under 2 seconds. Fixed by giving
   `NotifyResidencyChanged` a `mayHaveFreedResources` parameter (default `true`) and having both of
   `RunLoad`'s call sites (success and failure) pass `false`: a completed load never frees
   host/accelerator budget — it only ever consumes it or consumes nothing — so it has no business
   triggering the fairness scan in the first place; only a handle release or a `ResidencyMode`
   switch can actually free something. Verified: 2 new tests (an older queued waiter always wins
   the wake over a newer, equally-admissible one — proven with a real elapsed-time bound, not just
   ordering; an oversized oldest waiter never blocks a smaller admissible waiter queued behind it).
   312/312 fast, full solution build clean, 8/8 real-model re-run again since this is the third
   slice in a row to touch `AcquireAsync`/`NotifyResidencyChanged` directly.
   **Not yet done:** request age is now both observable (`OldestQueuedAdmissionAge`) *and* the
   actual scheduling key (oldest-admissible wins) — that part of "deliberately simple policy" is
   done. Still open: service quantum and starvation protection for *execution*, not admission —
   i.e. once a model is loaded and generating, nothing yet bounds how long it monopolizes
   contended compute before another resident model gets a turn. That's a materially different,
   bigger problem (would need an execution-mediation layer that doesn't exist: can generation be
   interrupted/checkpointed, what happens to in-flight KV, does preemption wreck batching
   efficiency) and is explicitly left for a real Level-1 scheduler phase, not folded into this
   admission-queue work.
   **Slice 4 done: closed the fresh-caller-jumps-the-queue race.** The one gap Slice 3 left open
   — "a brand-new caller arriving fresh through `AcquireAsync` never consults `_admissionQueue`
   at all and can still win the lock/load race against everyone already queued, since nothing
   reserves a queued winner's resources ahead of its actual retry" — is fixed. `WakeOldestAdmissibleQueuedWaiter`
   now reserves the winner's load atomically with picking it: still inside the same `_lock` scope
   that selects the winner and removes it from `_admissionQueue`, it registers a `PendingLoad` and
   kicks off `RunLoad` immediately (factored into a new `StartLoadLocked` helper, shared with
   `AcquireAsync`'s own cold-load path) rather than merely completing the winner's `WakeSignal` and
   trusting it to re-win the admission check on its own time. This closes the actual race: the
   winner's `WakeSignal` is a `TaskCompletionSource` with `RunContinuationsAsynchronously`, so its
   continuation never resumes synchronously — without reserving first, a fresh caller racing in
   through `AcquireAsync` during that gap could see the just-freed budget before the winner's retry
   ever ran, admit itself, and leave the winner to lose the race it had already won. Since
   `Snapshot()` (Slice 6, Phase 3) already counts in-flight pending loads as consumed capacity,
   reserving the winner's `PendingLoad` before releasing the lock is sufficient — any caller
   evaluated afterward, whether already-queued or brand new, sees accounting that already reflects
   the winner's claim. Skips the reservation if the model is somehow already resident/pending by
   the time it's picked (e.g. another waiter for the same model already started it), so
   single-flight is never violated by a duplicate load.
   Verified: `AcquireAsync_WokenWaiter_ReservesLoadAtomically_FreshCallerCannotStealTheSlot` — a
   new `CapacityFakeResourceBudget` derives admission from the manager's own `Snapshot()` (like
   the real `HostResourceBudget` does) rather than an external test flag, so the race is exercised
   through the same accounting production code uses. Confirmed as a genuine regression guard, not
   just a plausible-looking test: manually reverting the reservation line reproduces the failure
   (the woken waiter times out, starved by a fresh caller repeatedly re-stealing the slot) before
   being restored. 328/328 fast, full solution build clean.
7. 🔶 **OpenTail server integration.** Expose model identity through `IInferenceService`.
   *Acceptance: Users A/B → sidekick, Users C/D → reasoner, with same-model batching,
   cross-model concurrency, residency, and session isolation all verified together.*
   **Done: multi-model config + routing for the stateless request surface.**
   `OpenTailStingrayServerOptions.Models` (`IReadOnlyList<NamedModelOptions>`, empty by default —
   today's exact single-model behavior, unchanged byte-for-byte) opts a deployment into N named
   models (`Alias` + `ModelPath`, with `MmprojPath`/`Architecture`/`Backend`/`NGpuLayers`/`ContextSize`
   as per-model overrides of the shared/global settings — everything else, TurboQuant/KV
   type/MoE/spec-decode/sampling defaults, stays one shared config across all models, deliberately
   narrower than a fully independent per-model surface). Unlike the single-model path (eagerly
   loaded and pinned at startup), multi-model entries load lazily on first request and are never
   pinned — ordinary residency/eviction/admission applies, matching the design premise that
   residency is memory-pressure-driven, not a fixed deployment size.
   `IInferenceService` (`src/OpenTail.Stingray.Server/InferenceService.cs`) is deliberately narrower
   than the plan's own `GenerateAsync(InferenceRequest, ct)` sketch: `ResolveModel(string?
   requestedModel)` maps the OpenAI/Anthropic/Responses request's `model` field to a `ModelId`
   (case-insensitive alias match; single-model mode ignores the field entirely, exactly like
   today), and `AcquireAsync` is a thin passthrough to `IModelRuntimeManager.AcquireAsync` — an
   endpoint gets back the real `ModelRuntimeHandle` and reads `handle.Runtime.Loaded` for
   everything prompt-building needs (chat template, tokenizer, grammar, tool-boundary tokens)
   rather than the API wrapping generation itself, since that's the shape prompt-building already
   needed. All three chat-style endpoints (`OpenAiEndpoints.HandleChatCompletion`,
   `AnthropicEndpoints.HandleMessages`, `ResponsesEndpoints.HandleCreateResponse`) branch on
   `IInferenceService.IsMultiModel`: single-model mode keeps resolving `IInferenceEngine`/
   `ChatTemplateRenderer` directly from DI exactly as before Phase 7 existed; multi-model mode
   resolves + acquires the requested model per request (handle held for the request's whole
   duration — acquire → build prompt → generate → dispose, per the handle-lifetime contract) and
   builds a fresh per-request `ChatTemplateRenderer` from the acquired runtime's `Loaded` bundle
   (cheap, stateless-after-construction — reusing the shared DI singleton across concurrent
   different-model requests would race). `/v1/models` similarly branches: multi-model mode lists
   every configured alias; single-model mode keeps reading the injected `IInferenceEngine.ModelId`
   directly, since that string (a bare filename) is not the same value
   `IInferenceService.AvailableModelAliases` reports in single-model mode (a canonicalized full
   path) — the two are different strings, so unifying them would have silently changed
   `/v1/models`' single-model output.
   **Two real bugs found and fixed while building this, not before it shipped:**
   (1) An unconditional-branch first draft routed *every* request (including single-model ones)
   through `IInferenceService.AcquireAsync`, which broke ~79 of the ~320 fast tests: the
   established, pervasive test convention (`services.AddSingleton<IInferenceEngine>(fake)`,
   used throughout the whole endpoint test suite) replaces `IInferenceEngine` directly, bypassing
   `IModelRuntimeManager`'s loader entirely — routing unconditionally through the manager silently
   stopped honouring that replacement and tried to cold-load a real (nonexistent) GGUF instead.
   Fixed by the `IsMultiModel` branch above, confirmed by re-running the full fast suite (not just
   a clean build) before moving on. (2) The first real-model acceptance-test run failed two
   assertions with an *empty* answer where a real model should have produced text — not a routing
   bug: Qwen3 (the reasoner stand-in) defaults reasoning on, and an 8–48 token budget was being
   consumed entirely by `<think>` content before any answer text, exactly the omission the test
   was meant to catch cross-talk with, not thinking behavior. Fixed by setting
   `enable_thinking:false` explicitly in the acceptance test's requests.
   **Verified:** `tests/OpenTail.Stingray.Tests.Server.Fast/InferenceServiceTests.cs` (model
   resolution/discovery, fakes only) and `MultiModelEndpointTests.cs` (routing through the real
   HTTP surface, fakes only) — 327/327 fast suite green throughout. Real-model acceptance:
   `tests/OpenTail.Stingray.Tests.Server/MultiModelHttpAcceptanceTests.cs` — two real GGUFs
   (SmolLM2-1.7B, Qwen3-0.6B, standing in for sidekick/reasoner) driven through the *actual* HTTP
   endpoints (not `ModelRuntimeManager` directly, unlike Phase 5's `CrossModelConcurrencyTests`,
   which never exercised this phase's routing at all): four concurrent users split across the two
   models each get the right model's correct answer with the right `response.model`, each model
   loads exactly once despite two concurrent requests per model (single-flight, invariant 3),
   residency is observable (`GetStats().ResidentModels == 2`); a second test proves genuine
   cross-model overlap through real SSE streaming (the same interleaving-timestamp proof Phase 5
   used at the engine layer, now through the HTTP/SSE pipeline, catching a hypothetical
   `RequestConcurrencyGate`-level serialization that Phase 5's own test couldn't see). 10/10
   real-model tests, full solution build clean.
   **Follow-up done: `/v1/sessions/*` is now multi-model-aware**, closing the Phase 4 "known gap"
   (`HotSession` not holding a `ModelRuntimeHandle` for its lifetime) for real rather than merely
   documenting it. The original deferral note above worried about turning several global DI
   singletons into per-model registries — that concern turned out to be smaller than it looked:
   `LoadedEngine` (the per-`ModelRuntime` bundle both the single-model loader and `LoadNamedModel`
   already produce) already carries `SessionRuntime`/`ColdSessionRuntime` per model, so
   `IModelRuntimeManager` itself already *is* the per-model registry. The only genuinely missing
   piece was routing a session's later calls back to the model it was created against, and holding
   a handle for the session's whole life instead of per-request.
   New: `SessionModelRegistry` (`src/OpenTail.Stingray.Server/SessionModelRegistry.cs`) —
   `ConcurrentDictionary<SessionId, ModelRuntimeHandle>`. `Bind` is called once at session
   creation with the handle `IInferenceService.AcquireAsync` returned; that handle is held, not
   disposed, until `Release` runs on session delete — the one legitimate case for retaining a
   handle past a single operation (docs/032's "scope the handle to the operation" guidance is
   about not pinning a model as a standing "currently selected" marker, not about a session that
   genuinely needs residency guaranteed across many turns). As long as a binding exists,
   `ModelRuntime.HandleCount > 0`, so the runtime is never evictable — Invariant 2 made real for
   the multi-model path instead of relying on the single-model path's blanket `IsPinned`.
   `SessionEndpoints.cs` now branches on `IInferenceService.IsMultiModel`, mirroring
   `OpenAiEndpoints.HandleChatCompletion`'s established split: every handler used to take an
   unconditional `IInferenceEngine _` parameter specifically to force the one configured engine to
   load; that forced-load must not happen in multi-model mode (it would try to eagerly load
   `OpenTailStingrayServerOptions.ModelPath`, which may not even be set), so `IInferenceEngine` is
   now resolved explicitly only inside the single-model branch, exactly like the chat endpoint
   already does. `Create` gained an optional JSON body, `SessionCreateRequest(string? Model)` —
   same `model` field contract chat/messages/responses already use (null/unmatched-empty resolves
   to the first configured model); a client posting no body at all (today's exact behavior) is
   unaffected. `Get`/`Delete`/`GetOperation`/`RunTurn` resolve an existing session's route via a
   small shared helper, `TryResolveExistingSession`: multi-model mode looks up the
   `SessionModelRegistry` binding (a miss means "no such live session" → 404, a materially
   different outcome from "sessions unavailable" → 409, since the feature works fine here, this id
   just isn't bound to anything); single-model mode is byte-identical to before this existed.
   `Delete` additionally releases the binding on success, letting the model become evictable again
   once no other session references it.
   **Explicitly still out of scope, documented rather than silently dropped: durable/cold session
   restore across a process restart in multi-model mode.** `SessionModelRegistry` is in-memory
   only. Restoring a durable session's model binding after a restart would require reverse-mapping
   the manifest's stored model fingerprint (`ColdSessionRuntime.EvictToDisk`'s
   `modelFingerprint` argument) back to a currently-configured `NamedModelOptions` alias — a
   genuinely separate problem (what if that alias was renamed or removed from config since the
   manifest was written?) from "route a live session's calls to the model it was created against."
   After a restart, a multi-model call on a session id the registry doesn't know returns 404, same
   as any other unknown session id — never a silent wrong-model guess. Single-model mode's
   durable/cold restore is completely untouched. Also still not built: per-model overrides for
   TurboQuant/KV-type/MoE/spec-decode/sampling (all stay global across every configured model).
   **Verified:** fast suite (fakes only) — `MultiModelSessionEndpointTests.cs` (routing through
   the real HTTP surface: create+turn routes to the right model, case-insensitive alias, two
   sessions on different models don't cross-talk, unknown alias/session id → 404/404, delete
   releases and a subsequent get is 404) and two new tests in `ModelRuntimeManagerTests.cs`
   proving the actual invariant directly — `SessionModelRegistry_BoundHandle_KeepsModelResident_UntilSessionReleased`
   uses the same `SingleSlot`-mode eviction/wait behavior the file's own residency tests already
   use as proof (a bound session's model is waited-for, never evicted, exactly like an "active"
   model already isn't) and `SessionModelRegistry_Release_IsIdempotent_AndSafeWhenNeverBound`. The
   complete pre-existing `SessionEndpointTests.cs` suite (single-model) passes unmodified —
   byte-for-byte parity confirmed, not assumed. 338/338 fast suite green. Real-model acceptance:
   `tests/OpenTail.Stingray.Tests.Server/MultiModelSessionHttpAcceptanceTests.cs` — the same two
   real GGUFs `MultiModelHttpAcceptanceTests` uses, one session created per model via
   `POST /v1/sessions {"model": "..."}`, a real turn run on each with a distinguishing low-perplexity
   prompt, correct/isolated answers confirmed, `GetStats()` shows exactly one load and one resident
   runtime per model, and deleting one session leaves the other's still fully functional. 11/11
   real-model tests, full solution build clean.

## Test matrix

- **Lifecycle** — single load; repeated acquisition shares one runtime; concurrent cold
  acquisitions single-flight; load failure cleans up; disposal is idempotent.
- **Residency** — one/two models coexist; third triggers admission logic; idle model evicts;
  active model never evicts; pinned model never evicts.
- **Same-model concurrency** — 1 model / 10 sessions still uses the existing batching path
  correctly (regression guard, not new behavior).
- **Cross-model concurrency** — Model A + Model B generate concurrently with no cross-talk in
  model state, KV, session state, sampling, or tokenizer state.
- **Shared vs. independent runtime identity** — User A acquires Model A, User B acquires Model
  A: assert the same underlying runtime is shared (not two loads). User A acquires Model A, User
  B acquires Model B: assert distinct runtimes, independent engines, independent KV/session
  state. Model A loads, goes idle, evicts, then is requested again: assert a genuinely new
  physical runtime with no stale session/KV ownership carried over from the evicted instance.
- **Behavioral proof of no global lock** (not just an assertion that two runtimes *exist*) —
  Model A is made to block mid-generation (e.g. a controllable delay), Model B is then requested;
  assert Model B *begins* execution before Model A's blocked generation completes. This is the
  test that catches an implementation which satisfies every other test while secretly
  serializing all inference behind one semaphore.
- **Memory pressure** — ample / tight / insufficient memory, each with deterministic behavior.
- **Eviction race** — candidate selected → new request arrives → candidate becomes active before
  eviction completes → runtime must not be disposed.
- **Load/evict race** — Model A loading while Model B causes pressure while Model A's request is
  cancelled, and permutations thereof.
- **Cancellation** — cancelling one waiter never cancels another request sharing the same load.
- **Queue bound** — flood a model's admission queue past its configured capacity: assert
  overflow is rejected explicitly (not silently queued) and that queued requests respond to
  cancellation rather than waiting indefinitely.
- **End-to-end acceptance**: Model A = sidekick, Model B = reasoner; Users 1,2 → A, Users 3,4 →
  B, concurrently. Verify: A and B each load exactly once; each serves multiple sessions via
  continuous batching; A and B execute concurrently when resources permit; KV stays
  session-isolated; neither runtime evicts while active; pressure is observable; B becomes
  evictable once idle; on a constrained-memory config the same workload degrades to safe
  queueing/single-slot rather than OOM or corruption.

## Performance acceptance

Not judged on aggregate tokens/sec alone. Measure: one-model/one-user baseline latency;
one-model/many-users throughput+fairness; two-models/one-user-each overlap; two-models/many-users
throughput+latency; cold-second-model load penalty; memory-pressure eviction recovery time;
repeated-switching physical load count. The bar is: *when two models fit, running them
concurrently is materially better than repeatedly swapping them* — not "always run two models."

## Critical invariants

1. A physical model/engine instance is disposed exactly once, and no caller can obtain a usable
   handle to it once disposal has begun (see the acquire-vs-evict linearization point above).
2. A runtime with active requests or sessions cannot be evicted.
3. Concurrent acquisition of one cold model produces one physical load.
4. One session belongs to exactly one model runtime, for its lifetime.
5. One model runtime may serve many sessions concurrently.
6. Different model runtimes may execute concurrently — no global inference lock.
7. Resource admission never relies solely on model count.
8. KV memory ownership stays with the KV/session subsystem; the model manager never reaches into
   KV internals.
9. Cancelling one request never cancels shared work another request still needs.
10. Under insufficient resources, Stingray queues or rejects safely — it never violates
    runtime/session ownership to make room.

**Status of invariant 2, specifically:** both halves are now enforced. "Active requests" via
`ModelRuntime.HandleCount`/`IsEvictable` (Phase 1, tested). "Sessions" via `SessionModelRegistry`
(Phase 7 follow-up, tested) — a multi-model session holds a real `ModelRuntimeHandle` for its
entire lifetime, so its runtime is never evictable between turns. In single-model mode this
remains moot (harmless) rather than newly-enforced, since the one configured engine is still
always pinned regardless.
11. Every admission queue (per-model and global) is bounded and every queued request is
    cancellable; overflow is an explicit rejection, never unbounded growth.

## Final architecture

```
                         OpenTail
                            │
                    IInferenceService
                            │
                    ┌───────┴───────┐
                 Model A          Model B
                 Sidekick         Reasoner
                    │               │
             ┌──────┴──────┐  ┌─────┴─────┐
           HotSession   HotSession    HotSession
                    │               │
             Continuous Batch  Continuous Batch
                    └───────┬───────┘
                     Model Scheduler
                     Resource Admission
                ┌─────────────┴─────────────┐
          Model Residency              KV Governor
                │                           │
           Model Cache                  KV Cache
```

The scheduler doesn't own models; the model cache doesn't own sessions; the KV governor doesn't
own models. Each subsystem keeps exactly one job — same discipline `027` asked for, now applied
to a requirement that actually exists.
