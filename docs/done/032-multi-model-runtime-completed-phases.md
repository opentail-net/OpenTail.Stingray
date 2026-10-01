# Multi-model runtime — completed phases (1, 2, 4, 5)

Split verbatim on 2026-10-01 out of [../3-product-and-runtime/032-multi-model-inference-runtime-plan.md](../3-product-and-runtime/032-multi-model-inference-runtime-plan.md), whose Phases 3, 6 and 7 remain open. Item numbers match the plan's Phase Checklist.

- [x] 1. **Production `ModelRuntime` abstraction.** `ModelRuntime`, `ModelRuntimeHandle`,
   `ModelRuntimeManager`, `ModelRuntimeState`. Move the existing single-model load path
   (`InferenceEngineLoader`) behind it. *Acceptance: existing single-model server tests stay
   green, now routed through the new abstraction.*
- [x] 2. **Shared residency & single-flight loading.** Carry `SharedModelCache`'s proven
   refcount/overflow-table ownership (`025`/`026`) into `ModelRuntimeManager`: canonical
   identity, shared handles, async single-flight, safe disposal, load-failure cleanup.
   *Acceptance: 100 concurrent requests for one cold model → 1 physical load, 100 logical
   users.* (Landed alongside Phase 1 — see Status above.)

- [x] 4. **Multi-session model execution.** Wire each runtime to `HotSession` + continuous batching.
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
- [x] 5. **Cross-model concurrent execution** *(no-lock half; model-level resource scheduling is
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
