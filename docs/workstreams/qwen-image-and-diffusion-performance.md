# Workstream: Qwen Image and diffusion performance

**Owner:** this file. **Last verified:** 2026-10-30.
**Plans:** [../4-performance/2026-10-10-diffusion-samples-and-qwen-image-performance-plan.md](../4-performance/2026-10-10-diffusion-samples-and-qwen-image-performance-plan.md) (samples and Qwen Image plan) and [../4-performance/2026-10-10-vulkan-quantized-gemm-fp16-tile-plan.md](../4-performance/2026-10-10-vulkan-quantized-gemm-fp16-tile-plan.md) (the diffusion FP16-tile GEMM project).

## Correct state

- **Backlog:** Qwen Image has no backlog item in [../00-current-work.md](../00-current-work.md). Items 8 and 9 there are FLUX, not Qwen Image. This file is the only place the thread is tracked.
- **STATUS:** there is no Qwen Image row in [../STATUS.md](../STATUS.md). The documentation-architecture plan describes the DiT forward pass and VAE decode as verified against real weights, and text conditioning integration as still open ([../3-product-and-runtime/2026-10-09-documentation-architecture-plan.md](../3-product-and-runtime/2026-10-09-documentation-architecture-plan.md)).
- **Baseline (2026-10-10, Ryzen 5700G iGPU, Vulkan, Q3_K_S, 60/60 blocks resident):** the measured-results table in the samples plan compares Stingray with `sd-cli --backend vulkan0`.
- **FP16-tile project, phases (see that plan's checklist):** Phase 1 (small-M kernel) kept. Phase 2(c) (chunked image-stream MLP) kept: about 2.6 s (10.7%) faster, gap to the reference about 6% (was about 18%). Phase 2(a) rejected. Phase 2 follow-ups stopped as cheap changes ran out.
- The five-run median for the Phase 1 plus 2(c) checkpoint is not recorded yet. The iGPU timings describe this APU, not a discrete GPU (CLAUDE.md rule 13).

## Open items

- [ ] Remaining image samples, in order: Qwen Image Q3_K_S at 512x512 on the GPU (the first attempt hit the 20-minute CPU cap); Wan 2.2 14B one frame; HunyuanVideo 720 fp8 one frame; FLUX.2-dev with Mistral-Small-3.2-24B at 512x512
- [ ] Qwen Image text conditioning integration, if still open (per the documentation-architecture plan)
- [ ] Five-run median for the Phase 1 plus 2(c) checkpoint
- [ ] FP16-tile phases 3 (new tiled kernel family; carries the numerics decision), 4 and 5
- [ ] Add a STATUS row for Qwen Image once the end-to-end evidence exists

## Decisions for the owner

- The numerics decision that FP16-tile Phase 3 carries.

## Cadence

- Review after each dated measurement; record the numbers in `PerformanceLeague.md`, not only here.
- Re-read at least every two weeks while any item is open.
- Close when the STATUS row exists, the median is recorded, and each FP16-tile phase is kept or rejected in writing.
