# Completed items split from vulkan-gemm.md (2026-11-01)

Finished items moved verbatim out of [../4-performance/gpu/vulkan-gemm/vulkan-gemm.md](../4-performance/gpu/vulkan-gemm/vulkan-gemm.md). Open items stay there. Each block is preceded by the heading it sat under. Relative links inside the blocks were rebased for this folder.

> from: ### Actionable items checklist
- [x] **Tiled Q6_K shader**: Implement tiled Q6_K kernel for `ffn_down` (1418 of 9914 dispatches) to remove fallback to Path 1. (Done: `SgemmQ6K` exists and `TryMatMulBatchedPath2` accepts Q6_K; this checkbox was stale, checked 2026-10-31.)

> from: ### Checklist (2026-10-30; checked against the progress log)
- [x] Phase 1: small-M kernel (kept, 2026-10-10)

> from: ### Checklist (2026-10-30; checked against the progress log)
- [x] Phase 2(c): chunked image-stream MLP (kept, 2026-10-10; about 2.6 s, 10.7%, faster)

> from: ### Checklist (2026-10-30; checked against the progress log)
- [x] Phase 2(a): 128x128 tile with 256 threads (rejected and reverted, 2026-10-10)

> from: ### Checklist (2026-10-30; checked against the progress log)
- [x] Phase 2 MLP-up follow-ups: stopped, nothing cheap left (2026-10-10)

> from: ### Open items
- [x] Path 3 default decided on 2026-10-11 (gate: at least 2x today's Vulkan prefill, and identical greedy output; recorded in the leads plan, Lead 2)

> from: ### Open items
- [x] K-quants other than Q4_K and Q6_K stay off the batched Vulkan trunk. **Closed 2026-10-31:** `GpuForwardPass.ComputeCanBatchedTrunk` accepts Q3_K and Q5_K under Path 3 (`GpuForwardPass.cs` 1352-1357; the earlier "only Q4_K and Q6_K" note was stale). SmolLM2-1.7B Q5_K_M now prefills 66.8-85.5 t/s against 84.7-86.0 for its Q4_K_M file (503 tokens, `-g -1`, 3 runs each; the Q4_K_M range is from idle re-runs, the lowest Q5_K_M run has no known cause). The 2026-10-11 figure (22.3 vs 235 tok/s, PerformanceLeague finding 5) predates Path 3. Remaining K-quants (Q2_K, IQ types) were not checked.

