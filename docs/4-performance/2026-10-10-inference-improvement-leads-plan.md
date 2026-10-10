# Inference improvement leads (plan, 2026-10-10)

Status: **proposed, not started.** Order of work: first let the sample-image queue finish (Wan 2.2, HunyuanVideo, FLUX.2; see
[the samples and Qwen plan](2026-10-10-diffusion-samples-and-qwen-image-performance-plan.md)), then take the leads below in
order. These leads came from reading the performance league ([PerformanceLeague.md](../../PerformanceLeague.md)), the backlog
([00-current-work.md](../00-current-work.md)) and the vendored reference code in `examples/`. Several league rows date from
September, so **each lead starts by re-measuring today's number** before any code changes.

Ground rules (from CLAUDE.md and the Qwen work): measure before and after with alternating runs and medians; a change is
kept only if it is measurably faster and numerically acceptable; every change keeps an environment-switch rollback; one model
job at a time (shared-memory iGPU); iGPU timings describe this APU, not a discrete GPU (rule 13).

## Deferred on purpose

**Reference per-op timings for the video/large diffusion models (Wan 2.2, HunyuanVideo, FLUX.2) are explicitly deferred until Leads 1 to 5 below are all done** (Leads 1 to 5 themselves wait for the sample images to finish). They are not to be started before that, even after the
sample images exist. The method is ready for that day: run `sd-cli --backend vulkan0`
with `GGML_VK_PERF_LOGGER=1` on the same workload (it prints per-op GPU time and GFLOP/s by shape), compare op by op with our
stage timings, as was done for Qwen Image. Notes already gathered for that day:
- Wan's GPU weights (`WanGpuWeights`) are dequantized to FP32 and uploaded as FP16 for every block, with no residency
  budget; Wan 2.2 14B therefore holds about 28 GB (the live run showed a process near 27 GB). A single 256x256 frame is
  compute-bound, so quantized weights would mainly cut memory (to about 9 GB), not time. Worth doing as a footprint fix.
- HunyuanVideo (fp8) and FLUX.2 (Q4_K plus a 24B Mistral encoder) are the two heaviest runs; stage timings from the sample
  runs will say where their time goes.

## The leads, ranked

### Lead 1: Vulkan raw-quant matvec bandwidth (backlog item 6)
- **Evidence:** decode is bandwidth-bound and reaches only about 13-18 GB/s here. Our M=1 quantized matvec moved about 13 GB/s
  (3.9 MB in 0.3 ms); gpt-oss decode is MXFP4-matvec-bound at about 18 GB/s. A sustained-memory-bandwidth ceiling for this
  APU has not been measured, so "how far from the limit" is unknown.
- **Reference:** ggml's `mul_mat_vec_q3_k/q4_k/q5_k/q6_k.comp` (`examples/stable-diffusion.cpp/ggml/src/ggml-vulkan/vulkan-shaders/`),
  timed with the vendored `test-backend-ops` (`examples/stable-diffusion.cpp/build/bin/test-backend-ops.exe perf`).
- **First step (cheap, no code change):** measure a streaming-read bandwidth ceiling on the device, then time ggml's
  mat-vec for Q3_K/Q4_K at the model shapes and compare effective GB/s with ours (same bytes-read-over-time convention).
- **Upside:** up to about 2x Vulkan decode if the reference is near the memory ceiling; also speeds every M=1 projection
  (Qwen modulation, Wan/FLUX conditioning). **Stop rule:** if the reference is within about 10% of ours, park the lead.

### Lead 2: batched prefill on the Vulkan LLM path (backlog item 5)
- **Evidence:** `VulkanLayerSplitForwardPass` prefills one token at a time; Vulkan prefill lost to CPU prefill on every
  model in the September sweep (Qwen3-0.6B: 23.6 vs 226 t/s). The code today hands long prompts to the CPU
  (`HybridCpuPrefill`) as a workaround.
- **Reference:** ggml's `mul_mm` path, and the quantized GEMM kernels already improved this session (large tile, small-M
  tile, `STINGRAY_VK_QGEMM_SMALLM`).
- **First step:** re-measure today's Vulkan prefill vs CPU prefill on two models (one dense, one MoE) and read what the
  per-token path does per layer; write down which kernels a batched path can reuse.
- **Upside:** large for GPU-resident models, especially where the CPU hand-off cannot apply (model larger than RAM share,
  or when the CPU is busy). **Risk:** correctness across 10+ architectures; gate with the existing prefill-vs-decode logit
  parity tests per architecture. **Stop rule:** a measured win on both models, otherwise park.

### Lead 3: hybrid-GDN (Qwen3.5/3.6) prefill
- **Evidence:** prefill is sequential single-token evaluation; league ratios 0.05x (35B) and 0.16x (27B) against C++,
  the worst of any architecture measured.
- **Reference:** llama.cpp's delta-net / gated-delta-net implementation in `examples/llama.cpp` (chunked prefill form).
- **First step:** re-measure, and confirm whether the existing `VulkanBatchedPrefill` / GDN chunked-prefill work
  (see the env switches `STINGRAY_VULKAN_GDN_CHUNKED_PREFILL`, `STINGRAY_VULKAN_BATCHED_PREFILL`) already changed the
  picture; the league number may be stale.
- **Upside:** several times faster prefill on those models. **Gate:** teacher-forced logit parity against `llama-server`.

### Lead 4: MoE decode gap (OLMoE 0.50x)
- **Evidence:** already investigated in [2026-10-04-moe-decode-gap-investigation.md](2026-10-04-moe-decode-gap-investigation.md);
  that document lists what was tried.
- **First step:** read it, re-measure, and only reopen if a not-yet-tried idea remains (for example the per-expert Q4_K GEMM
  plan in [2026-10-04-q4k-expert-gemm-plan.md](2026-10-04-q4k-expert-gemm-plan.md)). **Upside:** moderate.

### Lead 5: FLUX.2 GPU (backlog item 8)
- **Evidence:** open in the backlog: production-shape GEMM ladder and the `DoubleBlockGpu` row-offset audit. Overlaps with
  the quantized-GEMM work from the Qwen thread, which shares the same kernel family (FLUX.2 uploads raw Q4_K).
- **Reference:** the PyTorch and sd.cpp references already in the repo; the `sd-cli` per-op logger (see Deferred).
- **First step:** after the FLUX.2 sample run, read its stage timings and check whether the kernel-family changes already
  help it (a switch A/B with `STINGRAY_VK_QGEMM_SMALLM`).

## Not worth chasing now
CPU SmolLM2 prefill (about 0.7-0.9x per the notes), Qwen Image non-GEMM ops and VAE (at parity with the reference).

## Order and checkpoints
1. Image queue finishes (images committed, README updated).
2. Leads 1 to 5, in that order, only after step 1 (the images) is complete. Leads 4 and 5 may be parked with a written reason, but the reference timings in step 4 still wait until all five are done or parked.
3. After each lead: log measured numbers (dated) here, update `PerformanceLeague.md` rows, and commit.
4. Only after Leads 1 to 5 are all done: the deferred reference per-op timings for Wan 2.2, HunyuanVideo and FLUX.2 (and the Wan quantized-weights footprint fix, if wanted).
