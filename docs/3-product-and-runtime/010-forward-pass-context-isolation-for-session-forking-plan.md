# 010 — Forward-Pass Context Isolation for Session Forking

## Status

Planning only. No implementation in this document. Written as the flagged immediate
follow-up to the bug-fix pass that added a fail-loud guard (`InferenceSession.cs`,
`ThrowIfForwardPassNotIsolated`) instead of implementing this — see that pass's plan for
context. This document scopes what implementing the real thing actually requires.

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

1. **`ForwardPass` (CPU dense).** Fully scoped above. A `CreateContext()` here is a
   constructor overload that takes the existing `_model`/`_hp`/weight-cache references
   from the parent and allocates fresh scratch buffers + a fresh `PagedKvCache`. Lowest
   risk, do this one first and validate the pattern against real Golden-style fork tests
   with a real model before touching anything else.

2. **`GpuForwardPass` (Vulkan)** and **`CudaForwardPass`.** Same shape as #1 but the
   "cheap to duplicate" scratch buffers are GPU-resident (device memory allocations,
   possibly through `GpuBufferPool`), and `CudaForwardPass` additionally implements
   `IMultiSlotKvCache` and is thread-affine (`IThreadAffineBackend` — see
   `_fwd.BindToCurrentThread()` calls throughout `ContinuousBatchingEngine`). Needs its
   own investigation pass: what does duplicating a CUDA-resident forward pass cost
   (device memory budget, allocation latency), and does context creation need to happen
   on a specific thread/stream.

3. **`HybridForwardPass` / `CudaHybridForwardPass`** (MoE expert offload). These
   additionally own expert-slot cache state (`ExpertSlotManager`/`CudaExpertSlotManager`
   per the MoE offloading design in the top-level design doc). Need to determine whether
   expert slot caches are safe to share (likely yes — they're an LRU cache over static
   expert weights, analogous to `_dequantWeightCache`) or need their own isolation.

4. **`HybridGdnForwardPass` / `CudaHybridGdnForwardPass`** (qwen35moe hybrid
   Gated-DeltaNet). Highest unknown: GDN state is described elsewhere in this codebase
   (`GdnStateCache`, `SupportsPartialRewind => false` by default) as destructively
   updated per token and NOT arbitrarily rewindable — which is exactly the kind of state
   that's hardest to safely duplicate mid-sequence. This tier needs the most dedicated
   investigation and should not be assumed to follow the same pattern as #1-3 until
   confirmed.

## Testing plan (once implementation starts)

- Extend `GoldenArchitectureTests`-style coverage with a REAL-model variant of
  `GoldenTest2`/`GoldenTest9`/`GoldenTest10` per backend (currently these only run
  against `StatefulTestForwardPass`, which is why the production gap was invisible —
  see the prior bug-fix pass's verification section for the same lesson applied to bug
  #1).
- A concurrency test: fork a session bound to a real model, drive parent and child
  generation concurrently, assert neither's token history/logits are affected by the
  other. This is the actual failure mode `CreateContext() => this` produces today and is
  the test that should have existed from the start.
- Remove the `ThrowIfForwardPassNotIsolated` guard's `NotSupportedException` path for
  each backend only once that backend's `CreateContext()` is implemented and covered by
  the above — the guard should fail closed per-backend as this rolls out, not open all
  at once.

## Non-goals for this document

- No implementation here.
- Not attempting to unify `ForwardPass`'s internal `PagedKvCache` with
  `InferenceSession.IKvSequence` as part of this plan — flagged as an open question for
  whoever picks this up, not decided here.
