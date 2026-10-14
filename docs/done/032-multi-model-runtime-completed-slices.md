# 032 Multi-model runtime: completed slices of Phases 3 and 6

> **ARCHIVED (2026-10-14).** Slice-level write-ups split verbatim out of [../3-product-and-runtime/032-multi-model-inference-runtime-plan.md](../3-product-and-runtime/032-multi-model-inference-runtime-plan.md). Phases 3 and 6 are still open in the live plan (accelerator accounting for hybrid-GDN, execution-side service quantum and starvation protection); only their finished slices are here.

## Phase 3 slices 1-10 (resource budget, admission, accelerator accounting)

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

## Phase 6 slices 1-3 (admission queue, queue-depth bound, oldest-admissible wake)

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

## Phase 6 slice 4 (fresh-caller race closed)

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
