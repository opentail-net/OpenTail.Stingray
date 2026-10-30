# Workstream: Vulkan GEMM for LLM prefill

**Owner:** this file. **Last verified:** 2026-10-30 against the leads log (`4-performance/2026-10-10-inference-improvement-leads-plan.md`, Lead 2).
**Not the same thread as** the diffusion FP16-tile project in [../4-performance/2026-10-10-vulkan-quantized-gemm-fp16-tile-plan.md](../4-performance/2026-10-10-vulkan-quantized-gemm-fp16-tile-plan.md), which is diffusion work.

## Correct state

- **Diagnosis (2026-10-10):** the batched prefill already exists. `GpuForwardPass` has a weight-amortising batched trunk for dense Q4_K/Q6_K models. The backlog wording "prefills per token" describes `VulkanLayerSplitForwardPass` and the MoE path, not `GpuForwardPass`. The slow part is the matmul kernel: about 0.3 TFLOP/s on a 493-token prompt.
- **Path 3 (2026-10-11), opt-in:** `STINGRAY_VULKAN_MM_PATH=3`. Default is still Path 1.

| Qwen3-4B-Q4_K_M, 493-token prefill, Vulkan iGPU | Prefill |
|---|---|
| Path 1 (default) | 38.4 t/s (12.8 s) |
| Path 3, 128-token chunks, tiled quantised GEMM, sliced flash attention | 71.3 t/s (6.9 s), +86% |
| CPU, same prompt (2026-10-10) | 102.8 t/s |

- Greedy generation identical to Path 1. Weight traffic 64.2 GB to 8.3 GB.
- Sliced flash attention (16-query slices) fixed a fallback that had made Path 3 slower before the fix.

## Open items

- [ ] Larger prefill chunks (256); check scratch buffers and batched attention/KV-append accept them
- [ ] A second dense model under Path 3 (the first measurement is Qwen3-4B only)
- [ ] Decode regression check on Vulkan
- [ ] Decide whether Path 3 becomes the default (the gate in the leads log: at least 2x today's Vulkan prefill, and identical greedy output)
- [ ] Confirm the per-kernel CPU-parity tests (`VulkanSgemmQuantEntryTests`, 6 cases) run in the fast suite

- [ ] K-quants other than Q4_K and Q6_K stay off the batched Vulkan trunk. `GpuForwardPass.ComputeCanBatchedTrunk` accepts only Q4_K and Q6_K weights (checked 2026-10-30). SmolLM2-1.7B Q5_K_M prefills at 22.3 tok/s against 235 for its Q4_K_M file (PerformanceLeague, 2026-10-11 recheck, finding 5). Path 3 (`SgemmQuant`) already covers Q3_K, Q4_K, Q5_K and Q6_K, so the work is to extend the trunk gate; measure per the [measurement rules](../reference/performance-measurement-rules.md).

## Decisions for the owner

- Path 3 default: yes or no, after the open items above.

## Cadence

- Review after each Path 3 measurement, and whenever a Vulkan kernel commit lands.
- Re-read at least every two weeks while any item is open.
- Close when the default decision is recorded and the STATUS and RUNNING rows match it.
