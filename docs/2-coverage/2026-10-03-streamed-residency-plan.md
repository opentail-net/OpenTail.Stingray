# Streamed residency for oversized MoE (Vulkan hybrid first)

Written 2026-10-03. Part of the TensorSharp / SharpMind selective-port work
(`examples/tensorsharp-parity-selective-port-plan.md`, `examples/sharpmind-parity-selective-port-plan.md`). The
question: how should Stingray run a mixture-of-experts model whose experts do not fit the accelerator budget, and how
do we know it is right?

**Status: decode on the Vulkan hybrid path is fixed and measured; correctness under eviction is tested; prefill,
CUDA, larger-than-RAM and the residency abstraction are not done.** Section 5 lists exactly what is left.

## 1. What measuring found (OLMoE-1B-7B Q4_K_M, `-g 8`, Ryzen 5700G + integrated Radeon, 2026-10-03)

The first measurement of the existing slot cache on an undersized budget showed decode collapsing from about 20 tok/s
(every expert resident) to 2-3 tok/s (4-64 slots), well below plain CPU (23-33 tok/s). Opt-in timers
(`STINGRAY_MOE_TIMING=1`) split a GPU MoE layer-step and found three separate defects, each hidden by the next:

| # | Defect | Evidence | Fix |
|---|---|---|---|
| 1 | The CPU fallback for a missed expert read its input from mapped Vulkan host-visible memory, whose reads are uncached. The quantised matvec reads its input scalar by scalar. | Per missed expert: gate 2.6 ms, up 2.7 ms, down 0.05 ms, for the same bytes. A warm repeat cost the same (so not page faults). After copying the input once into ordinary memory: 0.08 / 0.08 / 0.07 ms. Fallback per expert 5.4 ms -> 0.25 ms. | One 8 KB copy per layer (`HybridForwardPass`, `VulkanHybridGdnForwardPass`) |
| 2 | `ExpertSlotManager.Preload` held the manager lock for the whole upload (allocate + copy about 10 MB per expert), and every lookup from the forward pass takes the same lock. | Cache lookup 14-43 ms per MoE layer-step once defect 1 was gone. Test `ExpertSlotManagerConcurrencyTests`: lookup p99 655 ms (max 2.5 s) with the old code, 0.65 ms with the new. | Claim the key under the lock, upload outside it, publish the finished slot under the lock; a lookup never waits and never sees a half-uploaded slot |
| 3 | A slot returned by a lookup could be evicted and freed by a background prefetch before the GPU used it (hits promoted to the protected segment are demoted again when it fills). | `Tensor handle N not found` with 8 slots and 8 active experts. Latent in the old code; the small-cache regime this work targets is where it shows. | `TryGetCachedLeased` pins the slot until `ReleaseLeases`, which the pass calls only after the next fence-wait |

Before / after, decode tok/s (OLMoE `-g 8`, 32 tokens, prompt "Explain why the sky is blue.", Release build, nothing else
running; "after" is three runs each, no diagnostics enabled):

| Expert cache slots (of 512) | Before (single runs) | After (3 runs) | After, mean |
|---|---:|---|---:|
| 4 | 2.1 - 2.7 | 19.0, 18.4, 18.8 | 18.7 |
| 64 | 2.7 - 2.9 | 19.0, 19.3, 18.0 | 18.8 |
| 256 | 7.0 | 21.0, 19.9, 18.3 | 19.7 |
| 512 (all resident) | 19.7 - 23.5 | 20.9, 15.6, 20.4 | 19.0 |
| CPU only (`-g 0`) | 22.9 | 29.7, 26.5, 25.2 | 27.1 |

Reading it honestly: the cache size no longer matters (a 4-slot cache decodes as fast as an all-resident one), which is
the fix. But on this machine the hybrid path as a whole is slower than the CPU alone (19 against 27 tok/s). The GPU shares
system RAM with the CPU, so there is no bandwidth to gain, and each GPU MoE layer-step costs a submit, a router download
and a second submit that the CPU path does not have. That says nothing about a discrete GPU (project rule 13). The
"before" figures were taken in earlier sessions on the same machine (the CPU row's 22.9 came from a different prompt
length), so compare the shape, not the second decimal; the 8-10x collapse at small caches is far outside that noise.

## 2. Design principles taken from the two reference projects

From **TensorSharp** (`docs/models/qwen38-flash-next.md`, `docs/moe_cpu_offload_benchmark.md`):

- **Decode runs cold experts on the host straight from the memory map** with a thread team woken once per layer. It does
  not upload experts per token. The accelerator holds attention, dense layers and the experts that are hot.
- **Prefill streams, decode does not.** For a chunk of 128 tokens or more, the experts a chunk uses are uploaded to the
  GPU once per layer and run as a batched matmul. Their pages are first faulted in on 16 threads (about 2.4x faster
  prefill on their Mac).
- **Do not wire more than fits.** A wired layer holds all its experts, used or not, and takes page cache from the layers
  that read theirs from disk; past a point more GPU layers made decode slower and prefill collapsed. The engine plans the
  split from the accelerator working set and the RAM left for page cache.
- Things that did not work and should not be retried: `POSIX_FADV_WILLNEED` and `readahead` over an evicted range
  (0.0015 % resident afterwards); page-locking offloaded experts (added 20 s to load, made pages unevictable).

From **SharpMind** (`TransformerWeightsStreaming`, `StreamingLayerLoadRaceTests`, its streaming changelog):

- **A consumer must wait on the load, never infer completion from partial state.** Their race: the check "is the block
  empty" went false once the first tensor landed, so a half-loaded layer was used. Our equivalent is that a slot is
  published only when complete, and a lookup treats an in-flight upload as a miss.
- **"Streaming" can save no memory.** Cached quantised repacks kept the original layer weights alive. Any residency
  claim needs a measured, bounded working set after many tokens, not the existence of a streaming mode.
- **Full-load and streamed-load must agree numerically**, tested across the shapes that broke them: fused QKV, tied and
  untied heads, per-layer norms, MoE, sliding window, multi-turn.

## 3. Done (all on the Vulkan hybrid path, `-g N` on a MoE model)

- [x] `STINGRAY_MOE_TIMING=1` diagnostic (where a layer-step's time goes)
- [x] `STINGRAY_MOE_SLOTS=N` to force a small cache for testing
- [x] Fallback input copy (defect 1); same change applied to `VulkanHybridGdnForwardPass`, which has no local checkpoint
  to run (build-verified only)
- [x] Upload outside the lock (defect 2); `ExpertSlotManagerConcurrencyTests`, shown to fail on the old code
- [x] Slot leases (defect 3); `VulkanHybridOlmoeParityTests` at 8 and 16 slots
- [x] Memory-bounded test (`StreamedExpertMemoryBoundTests`) and a `VulkanBackend.LiveBufferCount` diagnostic
- [x] Eviction correctness: logits within cosine 0.996-0.999 of CPU with no argmax flips on OLMoE (64 experts) at 8 and 16
  slots, and Qwen3-Coder-30B-A3B (128 experts x 4 GPU layers through 16 slots): cosine 0.9982, one argmax flip on a
  near-tie (allowed by the test contract: only a gap above 2 % of the logit range fails)

## 4. Acceptance matrix (SharpMind's, adapted to experts)

| Check | State |
|---|---|
| Logits full-cache vs small-cache vs CPU | done for OLMoE and Qwen3-Coder (cosine, argmax) |
| Greedy tokens identical | not met: GPU hit and CPU miss use different kernels, so small and large caches diverge after a few tokens (same cause as GPU vs CPU); logits are the contract |
| Lookup latency while uploading | done (`ExpertSlotManagerConcurrencyTests`) |
| Concurrent prefetch + lookup safe, dispose with uploads in flight | done (same test) |
| Reload the same expert after eviction gives the same result | covered by the parity runs (constant eviction at 8 slots) |
| Memory bounded after N tokens (GC forced) | done for Vulkan hybrid (`StreamedExpertMemoryBoundTests`: 12 slots for 256 experts; tokens 40 -> 440, about 12,800 expert uploads: live GPU buffers 100 -> 106, private memory 523.6 -> 526.1 MiB). Not run at 1,000+ tokens |
| Several turns / long context identical | not done |
| MoE with shared experts, sigmoid gating | not done on Vulkan (no local checkpoint of that kind) |
| Larger-than-RAM model | not done (GLM-5.3 shards still downloading; arch unverified) |

## 5. Not done, in the order I would do it

1. **Batched MoE prefill on the hybrid/GPU paths.** `HybridForwardPass.Prefill` is a loop of single-token `Forward`
   calls, and `GpuForwardPass` has no batched trunk for MoE. Measured on OLMoE, 631-token prompt: CPU 141.6 tok/s, hybrid
   25.1, full-GPU 26.6. This is the larger remaining cost (plan: [batched MoE prefill](2026-10-03-batched-moe-prefill-plan.md)) and the place TensorSharp gets its 4.5-10x prefill advantage
   (stream the used experts of a chunk once, run batched). On this integrated GPU the CPU path may stay faster; the design
   should pick per chunk size. Needs its own plan and numbers.
2. **Longer leak check** (1,000+ tokens) and a CUDA equivalent; the 440-token Vulkan check is done.
3. **CUDA mirror.** `CudaExpertSlotManager.Preload` also holds its lock across `UploadExpertAsync` and the CUDA hybrid has
   its own lookup-then-record window, so defects 2 and 3 very likely exist there. CUDA pinned memory is not
   write-combined, so defect 1 does not. Not changed: no NVIDIA GPU here to test, and the slot index and pending-event
   bookkeeping make a blind port risky.
4. **Leases for `VulkanHybridGdnForwardPass`** (uses `TryGetCached`, so defect 3 remains possible at tiny capacities).
5. **Residency abstraction** (`TensorResidency`, planner, `IResidencyManager`) only after 1-4 show what it must cover.
   The expert cache is already the concrete case; do not generalise it before the prefill design exists.
6. **Automatic split planning** that accounts for page cache (TensorSharp's lesson), and a per-chunk parallel prefault of
   the experts a prefill is about to touch.
7. **Larger-than-RAM run** (GLM-5.3 Q2_K_XL, 236 GiB, paged, correctness only) once the download completes and the
   architecture is verified.
