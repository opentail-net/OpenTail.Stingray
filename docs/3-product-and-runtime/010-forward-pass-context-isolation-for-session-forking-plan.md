# 010 — Forward-Pass Context Isolation for Session Forking

## Status

**Current position (2026-10-31): superseded in part; nothing here is scheduled for implementation.**
- Session forking on the **CPU dense** path works and is tested on a real model, sequentially and with both branches generating
  concurrently (`HotSessionForkRealModelTests`). It goes through `HotSessionRuntime.Fork` and the reference-counted KV cache, not
  through `IForwardPass.CreateContext()`.
- `CreateContext()` had no caller in `src/`, so the per-backend work this plan proposes is not needed for forking. **Decided 2026-11-01 (owner): remove it.**
  It is removed from `IForwardPass` (with its two test doubles and the one test that exercised it); see the CHANGELOG.
- Not verified: forking on the **GPU, hybrid and Gated-DeltaNet** passes (GDN cannot rewind, so it needs a state snapshot).
- Everything after the "Original text follows" line below is the original design (written before the above was known), kept as history.

**Thread:** [configuration and sessions](04-quality-of-life-improvements-plan.md).

### Original status (historical, as of the original write-up and the 2026-10-30 check)

Planning only. No implementation in this document. **Checked 2026-10-30:** still planning only; `IForwardPass.CreateContext()` still returns `this` (`src/OpenTail.Stingray.Core/IForwardPass.cs:14`). The guard this section cites, `ThrowIfForwardPassNotIsolated` in `InferenceSession.cs`, is not in the repository, so that reference is stale. Written as the flagged immediate
follow-up to the bug-fix pass that added a fail-loud guard (`InferenceSession.cs`,
`ThrowIfForwardPassNotIsolated`) instead of implementing this — see that pass's plan for
context. This document scopes what implementing the real thing actually requires.

## Correction, 2026-10-31: the premise of this plan no longer holds

Checked in code before any implementation. **Nothing in `src/` calls `IForwardPass.CreateContext()`.** Its only callers are two test
doubles (`tests/OpenTail.Stingray.Tests.ForwardPass.Fast/ForwardPassTests.cs`, `tests/OpenTail.Stingray.Tests.Sessions.Fast/SpeculativeCascadeTests.cs`);
`Model.CreateContext` in the public API is an unrelated method. The `InferenceSession` class this plan is written around does not
exist any more. Session forking is `HotSessionRuntime.Fork`, which forks the engine's reference-counted sequence KV cache:
`HotSession.TryForkSharedPrefixCache` → `ContinuousBatchingEngine.TryForkSharedPrefix` → `RetainedSequenceState.TryForkSharedPrefix` →
`PagedKvCache.ForkSharedPrefix`. A fork therefore does **not** hand the child the parent's forward pass state; parent and child run
through one shared batched forward pass with separate KV sequences.

Evidence on a real model (SmolLM2-1.7B Q4_K_M, CPU dense): `HotSessionForkRealModelTests` showed physical page sharing and
copy-on-write correctness against a precision-consistent replay oracle, but only with the two branches generating one after the
other. Two tests now cover it (2026-10-31): the original sequential one, and
`Fork_TwoBranchesGeneratingConcurrently_DoNotDisturbEachOtherOrTheParent`, which runs both branches in one two-slot batch for 16 tokens
each, asserts the engine really decoded them in a shared batch, and checks both against the same oracle (10 of 10 runs passed).
This is the "concurrency test" item below, done against the path that exists.

What that leaves open, and why the per-backend tiers below are **not** to be implemented as written:
- `CreateContext()` is dead public surface on `IForwardPass`. Implementing it on seven passes would add code nothing calls.
- The real fork path was checked on the CPU dense pass only. Whether `TryForkSharedPrefix` works for GPU, hybrid and GDN passes
  (GDN cannot rewind, so a mid-sequence fork needs a state snapshot) is the live question, not `CreateContext`.
- **Decided 2026-11-01 (owner): `CreateContext()` is removed** from the public interface (it was a default member returning `this`; code that called it
  breaks, an implementer that defined a `CreateContext()` method keeps compiling, an explicit interface implementation would not). This plan stays unimplemented.

Original text follows, unchanged apart from this correction.

## Problem

`IForwardPass.CreateContext()` (`src/OpenTail.Stingray.Core/IForwardPass.cs:14`) exists
so `InferenceSession.Fork()`/`ForkMany()` (the Zero-Copy Session Multiverse feature,
doc `008`) can give a forked child session an execution context that shares the
parent's immutable model weights but owns independent mutable decode state — its own KV
cache, its own position counter, its own scratch buffers. The interface default is
`=> this`, documented as the correct behavior "for stateless forward passes."

No concrete production forward pass is stateless, and none of the seven overrides it:

- `ForwardPass` (`Engine/ForwardPass.cs`) — CPU dense
- `CudaForwardPass` (`Engine/CudaForwardPass.cs`)
- `HybridForwardPass` (`Engine/HybridForwardPass.cs`) — CPU MoE expert offload
- `CudaHybridForwardPass` (`Engine/CudaHybridForwardPass.cs`)
- `HybridGdnForwardPass` (`Engine/HybridGdnForwardPass.cs`) — qwen35moe hybrid GDN
- `CudaHybridGdnForwardPass` (`Engine/CudaHybridGdnForwardPass.cs`)
- `GpuForwardPass` (`Engine/GpuForwardPass.cs`) — Vulkan

So today, forking a session bound to any real model silently hands the child the exact
same forward pass instance as the parent — same KV cache, same position counter.
Concurrent or interleaved generation on the two corrupts both. The fail-loud guard added
in the prior pass turns that from silent corruption into a clear `NotSupportedException`
at fork time; it does not restore the forking capability for real models.

## What a real `CreateContext()` needs, per instance

Looking at `ForwardPass` (the CPU dense case, the most tractable of the seven) as the
concrete example: its fields split cleanly into two categories.

**Safe to share by reference** (immutable, or safe lazily-populated read caches keyed by
tensor name, populated once and never mutated in a way that depends on decode
position):
- `_model` (`IModelTensorSource`) — the mmap'd/loaded weights
- `_hp` (`ModelHyperparams`)
- `_normCache`, `_dequantWeightCache`, `_q4kx8Cache` — lazily populated, keyed by tensor
  name, independent of any particular sequence's decode state. Sharing these across
  parent and child is not just safe, it's the point (that's what makes the fork
  "zero-copy" for weights).

**Must be independent per instance** (decode state — this is exactly the state that
must NOT be shared for two sessions to generate independently):
- `_kvCache` (`PagedKvCache`) — holds the actual K/V history; this is the thing
  `IKvSequence.Fork()` on the *session* side already handles for the CPU dense KV cache
  path, but `ForwardPass` embeds its *own* `PagedKvCache` internally, separate from
  `InferenceSession._kvSequence`. This is the crux of the design question below.
- `_hidden`, `_residual`, `_normBuf`, and the other preallocated scratch buffers —
  cheap to duplicate, no reason to share.
- Any position/length counters mirroring the KV cache's own.

The open design question this raises: `InferenceSession` already has its own
`IKvSequence` abstraction (`CpuKvSequence`/`CudaSequenceKvCache`/etc., forked via
`IKvSequence.Fork()`) that is logically supposed to be the source of truth for a
session's KV state — but `ForwardPass` (and presumably the other six) maintains an
*independent*, internally-owned `PagedKvCache` that the session-level `IKvSequence`
does not actually drive. That's a second, adjacent architectural gap worth confirming
before writing any `CreateContext()` code: does forking at the `IForwardPass` level need
to duplicate a *third* copy of KV state, or should this be the point where
`ForwardPass`'s internal cache is finally unified with the session-level `IKvSequence`
it currently duplicates? Implementing `CreateContext()` naively (duplicate whatever
`PagedKvCache` currently holds) works but leaves that duplication in place; the
alternative (thread the session's own `IKvSequence`/pages into `ForwardPass` so there's
one KV owner, not two) is a larger change but removes a standing inconsistency. This
needs a decision before implementation starts, not during it.

## Per-backend scope, in rollout order (least to most unknown)

- [ ] 1. **`ForwardPass` (CPU dense).** Fully scoped above. A `CreateContext()` here is a
   constructor overload that takes the existing `_model`/`_hp`/weight-cache references
   from the parent and allocates fresh scratch buffers + a fresh `PagedKvCache`. Lowest
   risk, do this one first and validate the pattern against real Golden-style fork tests
   with a real model before touching anything else.

- [ ] 2. **`GpuForwardPass` (Vulkan)** and **`CudaForwardPass`.** Same shape as #1 but the
   "cheap to duplicate" scratch buffers are GPU-resident (device memory allocations,
   possibly through `GpuBufferPool`), and `CudaForwardPass` additionally implements
   `IMultiSlotKvCache` and is thread-affine (`IThreadAffineBackend` — see
   `_fwd.BindToCurrentThread()` calls throughout `ContinuousBatchingEngine`). Needs its
   own investigation pass: what does duplicating a CUDA-resident forward pass cost
   (device memory budget, allocation latency), and does context creation need to happen
   on a specific thread/stream.

- [ ] 3. **`HybridForwardPass` / `CudaHybridForwardPass`** (MoE expert offload). These
   additionally own expert-slot cache state (`ExpertSlotManager`/`CudaExpertSlotManager`
   per the MoE offloading design in the top-level design doc). Need to determine whether
   expert slot caches are safe to share (likely yes — they're an LRU cache over static
   expert weights, analogous to `_dequantWeightCache`) or need their own isolation.

- [ ] 4. **`HybridGdnForwardPass` / `CudaHybridGdnForwardPass`** (qwen35moe hybrid
   Gated-DeltaNet). Highest unknown: GDN state is described elsewhere in this codebase
   (`GdnStateCache`, `SupportsPartialRewind => false` by default) as destructively
   updated per token and NOT arbitrarily rewindable — which is exactly the kind of state
   that's hardest to safely duplicate mid-sequence. This tier needs the most dedicated
   investigation and should not be assumed to follow the same pattern as #1-3 until
   confirmed.

## Testing plan (once implementation starts)

- [ ] Extend `GoldenArchitectureTests`-style coverage with a REAL-model variant of
  `GoldenTest2`/`GoldenTest9`/`GoldenTest10` per backend (currently these only run
  against `StatefulTestForwardPass`, which is why the production gap was invisible —
  see the prior bug-fix pass's verification section for the same lesson applied to bug
  #1).
- [ ] A concurrency test: fork a session bound to a real model, drive parent and child
  generation concurrently, assert neither's token history/logits are affected by the
  other. This is the actual failure mode `CreateContext() => this` produces today and is
  the test that should have existed from the start.
- [ ] Remove the `ThrowIfForwardPassNotIsolated` guard's `NotSupportedException` path for
  each backend only once that backend's `CreateContext()` is implemented and covered by
  the above — the guard should fail closed per-backend as this rolls out, not open all
  at once.

## Non-goals for this document

- No implementation here.
- Not attempting to unify `ForwardPass`'s internal `PagedKvCache` with
  `InferenceSession.IKvSequence` as part of this plan — flagged as an open question for
  whoever picks this up, not decided here.

---

## Research findings (2026-10-31)

Code and outside sources, checked 2026-10-31. Excerpts are in [research/2026-10-31-product-runtime-snippets.md](research/2026-10-31-product-runtime-snippets.md).

**Verified in code**
- `IForwardPass.cs:14` is `IForwardPass CreateContext() => this;`. No `CreateContext()` override exists under `src/OpenTail.Stingray.Engine`, so the doc's claim that none of the engine passes override it holds. Cuda and Vulkan projects were not searched.
- `ThrowIfForwardPassNotIsolated` is absent from `src/`. The "stale reference" note is correct.
- `SupportsPartialRewind` defaults to `false` (`IForwardPass.cs:139`). The GDN passes do not override it, so they inherit `false`. This supports the tier-4 concern.

**Verified outside (web)**
- llama.cpp's recurrent memory module (used for Mamba/RWKV-style state) implements `seq_cp` by adding the destination sequence to the source cell and copying the tail pointer. Its source comments say a recurrent state cannot be partially erased at the end of a sequence, because earlier-token state is not kept. That matches the doc's "not arbitrarily rewindable" premise. Source: [llama-memory-recurrent.cpp (searchfox mirror)](https://searchfox.org/firefox-main/source/third_party/llama.cpp/src/llama-memory-recurrent.cpp). The mirror's version is not pinned, so treat the detail as provisional.
- A newer llama.cpp variant keeps per-sequence recurrent snapshots for rollback, controlled by an `n_rs_seq` parameter (same mirror, provisional). This is a concrete third option for tier 4: snapshot the GDN state at chosen positions, rather than duplicating it mid-sequence or sharing it. The doc does not list it.

**Gaps in the doc**
- The open question (a third KV copy versus unifying with `IKvSequence`) is still unresolved. The llama.cpp finding shows that a shared-cell model is the established pattern for recurrent state, but I did not verify how llama.cpp handles attention KV on fork. Do not treat that as settled.

**Not verified:** whether the CUDA and Vulkan passes can implement `CreateContext()` without changing their device-memory pooling; the test list was not checked against the existing golden tests.

**Next step:** read `GdnStateCache` and decide between snapshot (`n_rs_seq`-style) and copy for tier 4, before any tier-1 implementation.
