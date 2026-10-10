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

## Progress log

### Lead 1, first measurement, 2026-10-10 (done): the reference is about 1.8x faster, with large headroom
Method: ggml `test-backend-ops perf -o MUL_MAT -b Vulkan0` (vendored in `examples/stable-diffusion.cpp/build/bin/`) for the
reference; our own standalone benchmark for Stingray (one-row quantized mat-vec, 50 calls per submission so fixed overhead
drops out, effective GB/s = quantized-weight bytes / kernel time). Same GPU (Radeon iGPU, `uma 1, fp16 1, warp 64`).

Sustained-bandwidth ceiling on this machine (ggml, 4096 x 14336 mat-vec): F16 42.6, BF16 44.7, F32 44.1, Q8_0 45.1, Q4_0 41.4
GB/s, so about 44 GB/s is the practical ceiling.

| Type | Ours | ggml | Ours as share of ceiling | ggml as share |
|---|---|---|---|---|
| Q3_K, 4096x14336 | 1596 us, 15.8 GB/s | 868 us, 29.0 GB/s | 36% | 66% |
| Q4_K, 4096x14336 | 1400 us, 23.6 GB/s | 775 us, 42.6 GB/s | 54% | 97% |
| Q5_K, 4096x14336 | 1551 us, 26.0 GB/s | not in the log | 59% | n/a |
| MXFP4 | about 18 GB/s (league doc, gpt-oss) | 35.9 GB/s | about 41% | 82% |

Our rates are flat across the Qwen shapes (3072x3072, 3072x12288, 12288x3072: 15.6-15.9 GB/s Q3_K, 23.5-25.4 Q4_K), so it is the
kernel's inner structure, not a shape effect. Verdict: Lead 1 is confirmed, with up to about 1.8x available on Q3_K/Q4_K
mat-vec and about 2x on MXFP4. Next: compare our `MatVecDqQ3K/Q4K/Q5K` shaders with ggml's `mul_mat_vec_q3_k/q4_k.comp`
(workgroup size, rows per workgroup, vector loads, reduction), try the cheapest structural change, and A/B it.

### Lead 1, correction and the real picture, 2026-10-10: LLM-decode kernels measured
The first table above measured the diffusion-side `MatVecDq*` family. The Vulkan LLM decode path uses a different, older
family (`MatVecQ3K/Q4K/Q5K/Q6K/Q8_0/Q4_0/Mxfp4`, entry `VulkanBackend.MatMul(..., weightDType)` and `MatVecMxfp4`). Measured with
the same method (4096 x 14336 mat-vec, 50 calls per submission, effective GB/s on the quantized bytes) against ggml on the same GPU
(ggml's mat-vec ceiling on this machine is about 44 GB/s):

| Type | Ours | ggml | ggml faster by | Notes |
|---|---|---|---|---|
| Q4_K | 42.2 GB/s | 42.6 | 1.0x | parity: the most common LLM quant needs nothing |
| Q5_K | 35.8 | 42.9 | 1.2x | |
| Q6_K | 36.6 | 45.0 | 1.23x | part of every Q4_K_M / Q5_K_M model (ffn_down, output) |
| Q8_0 | 32.7 | 45.1 | 1.38x | many small models ship as Q8_0 |
| Q4_0 | 31.2 | 41.4 | 1.33x | |
| Q3_K | 16.1 | 29.1 | 1.8x | the Q3_K_S/M/XL checkpoints, for example Qwen3.6-27B |
| MXFP4 | 14.3 | 35.9 | 2.5x | gpt-oss-20B decode is MXFP4 mat-vec-bound |

Impact: any Vulkan LLM using these weight types, in proportion to the bytes they hold; Q4_K-only models are unaffected. Priority by
gap x how many models: MXFP4, then Q8_0, Q3_K, Q4_0, Q6_K, Q5_K. ggml's Q4_K kernel (for reference, reading it): aligned 32-bit
word loads, all block scales decoded with a few packed bit operations, the scale factored out of the dot product (q.y first,
then one multiply per sub-block) instead of dequantizing each weight, and several rows per workgroup so activation loads are
shared. The experiment of decoding 16 weights per lane in our diffusion-side mat-vec was slower (13.8-25 GB/s vs 15.8-26),
so decode amortisation alone does not help; access pattern and ALU per weight are what to change.
