# 010-forward-pass-context-isolation-for-session-forking-plan - completed items 2026-11

Ticked items moved from [010-forward-pass-context-isolation-for-session-forking-plan.md](../010-forward-pass-context-isolation-for-session-forking-plan.md) on 2026-11-08. Each keeps its text and nested lines. Items with open follow-up text stay in the source doc.

- [x] *(superseded 2026-11-01: `CreateContext()` removed on the owner's decision, forking does not use it)* 1. **`ForwardPass` (CPU dense).** Fully scoped above. A `CreateContext()` here is a
   constructor overload that takes the existing `_model`/`_hp`/weight-cache references
   from the parent and allocates fresh scratch buffers + a fresh `PagedKvCache`. Lowest
   risk, do this one first and validate the pattern against real Golden-style fork tests
   with a real model before touching anything else.
- [x] *(superseded 2026-11-01, as 1)* 2. **`GpuForwardPass` (Vulkan)** and **`CudaForwardPass`.** Same shape as #1 but the
   "cheap to duplicate" scratch buffers are GPU-resident (device memory allocations,
   possibly through `GpuBufferPool`), and `CudaForwardPass` additionally implements
   `IMultiSlotKvCache` and is thread-affine (`IThreadAffineBackend` — see
   `_fwd.BindToCurrentThread()` calls throughout `ContinuousBatchingEngine`). Needs its
   own investigation pass: what does duplicating a CUDA-resident forward pass cost
   (device memory budget, allocation latency), and does context creation need to happen
   on a specific thread/stream.
- [x] *(superseded 2026-11-01, as 1)* 3. **`HybridForwardPass` / `CudaHybridForwardPass`** (MoE expert offload). These
   additionally own expert-slot cache state (`ExpertSlotManager`/`CudaExpertSlotManager`
   per the MoE offloading design in the top-level design doc). Need to determine whether
   expert slot caches are safe to share (likely yes — they're an LRU cache over static
   expert weights, analogous to `_dequantWeightCache`) or need their own isolation.
- [x] *(superseded 2026-11-01, as 1)* 4. **`HybridGdnForwardPass` / `CudaHybridGdnForwardPass`** (qwen35moe hybrid
   Gated-DeltaNet). Highest unknown: GDN state is described elsewhere in this codebase
   (`GdnStateCache`, `SupportsPartialRewind => false` by default) as destructively
   updated per token and NOT arbitrarily rewindable — which is exactly the kind of state
   that's hardest to safely duplicate mid-sequence. This tier needs the most dedicated
   investigation and should not be assumed to follow the same pattern as #1-3 until
   confirmed.
- [x] *(done for CPU dense: `HotSessionForkRealModelTests` forks on a real model, sequentially and with both branches generating concurrently; GPU, hybrid and Gated-DeltaNet forking stays unverified, see the Status section)* A concurrency test: fork a session bound to a real model, drive parent and child
  generation concurrently, assert neither's token history/logits are affected by the
  other. This is the actual failure mode `CreateContext() => this` produces today and is
  the test that should have existed from the start.
