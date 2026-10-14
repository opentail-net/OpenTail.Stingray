# Streamed residency for oversized MoE (Vulkan hybrid first): open items only

**Status (2026-10-14):** decode on the Vulkan hybrid path is fixed and measured, and correctness under eviction is tested. Measurements
(the three defects, before/after decode tok/s), the TensorSharp/SharpMind design principles, the done list and the acceptance matrix are
archived verbatim in [../done/2026-10-03-streamed-residency-plan-history.md](../done/2026-10-03-streamed-residency-plan-history.md).
Headline: cache size no longer matters (4 slots decode as fast as all-resident, about 19 tok/s) but on this iGPU the hybrid is slower than
CPU alone (19 vs 27 tok/s); that says nothing about a discrete GPU (CLAUDE.md rule 13).

## Open items, in the order I would do them

1. **Batched MoE prefill on the hybrid/GPU paths.** `HybridForwardPass.Prefill` is a loop of single-token `Forward` calls and `GpuForwardPass`
   has no batched trunk for MoE. OLMoE 631-token prompt: CPU 141.6 tok/s, hybrid 25.1, full-GPU 26.6. Plan: [batched MoE prefill](2026-10-03-batched-moe-prefill-plan.md).
   The design should pick per chunk size (the CPU path may stay faster on this iGPU).
2. **Longer leak check** (1,000+ tokens; the 440-token Vulkan check is done) and a CUDA equivalent.
3. **CUDA mirror.** `CudaExpertSlotManager.Preload` holds its lock across `UploadExpertAsync` and the CUDA hybrid has its own lookup-then-record window, so defects 2 (lock held during upload) and 3 (slot evicted before use) very likely exist there. Defect 1 does not (CUDA pinned memory is not write-combined). Not changed: no NVIDIA GPU here; a blind port is risky.
4. **Leases for `VulkanHybridGdnForwardPass`** (uses `TryGetCached`, so defect 3 remains possible at tiny capacities).
5. **Residency abstraction** (`TensorResidency`, planner, `IResidencyManager`) only after 1-4 show what it must cover.
6. **Automatic split planning** that accounts for page cache (TensorSharp's lesson), and a per-chunk parallel prefault of the experts a prefill is about to touch.
7. **Larger-than-RAM run** (GLM-5.3 Q2_K_XL, 236 GiB, paged, correctness only) once the download completes and the architecture is verified.

## Acceptance-matrix rows still not met

| Check | State |
|---|---|
| Greedy tokens identical, full vs small cache | **not met by design:** GPU hit and CPU miss use different kernels, so tokens diverge after a few; logits (cosine, argmax) are the contract |
| Several turns / long context identical | not done |
| MoE with shared experts, sigmoid gating on Vulkan | not done (no local checkpoint of that kind) |
| Memory bounded at 1,000+ tokens | not run |
| Larger-than-RAM model | not done |
