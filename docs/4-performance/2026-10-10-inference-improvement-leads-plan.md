# Inference improvement leads (plan, 2026-10-10)

Status: **in progress; the status board below is the source of truth (the intro text after it predates the work).** Order of work: first let the sample-image queue finish (Wan 2.2, HunyuanVideo, FLUX.2; see
[the samples and Qwen plan](2026-10-10-diffusion-samples-and-qwen-image-performance-plan.md)), then take the leads below in
order. These leads came from reading the performance league ([PerformanceLeague.md](../../PerformanceLeague.md)), the backlog
([00-current-work.md](../00-current-work.md)) and the vendored reference code in `examples/`. Several league rows date from
September, so **each lead starts by re-measuring today's number** before any code changes.

Ground rules (from CLAUDE.md and the Qwen work): measure before and after with alternating runs and medians; a change is
kept only if it is measurably faster and numerically acceptable; every change keeps an environment-switch rollback; one model
job at a time (shared-memory iGPU); iGPU timings describe this APU, not a discrete GPU (rule 13).

## Status board (living checklist; items may be done in any order)

Tick an item when it is done, add the commit and the measured result next to it, and put the supporting numbers in the evidence log at the end of this file.
Order below is "what to look at next" within each group; the groups themselves are ordered by expected payoff. Updated 2026-10-11.

### A. Decode on Vulkan (new, top of the list: the attention win generalises)
- [x] **A0** Diagnose decode: elimination experiment plus the ggml per-op reference showed about 68 ms of a 159 ms token was single-token attention at about 500 context tokens (Qwen3-4B). Evidence log: "Lead 2b".
- [x] **A1** GQA-group split-KV decode attention kernel (fp32 KV, head dim 64/128, 1-4 query heads per KV head), default on, `STINGRAY_VULKAN_DECODE_ATTN_GQA=0` rolls back. Kernel 11.8x faster (132 vs 1,566 us/layer, ggml about 140); Qwen3-4B decode after a 493-token prompt 6.3 to 12.1 t/s (+88%), text identical. Commit `2624089d`.
- [x] **A2** GQA kernel at every context length (`GpuForwardPass.UseSplitKvDecodeFp32`): short-prompt decode (Qwen3-4B, 96 tokens) **10.5-10.6 -> 12.6-12.7 t/s (+20%)**, text identical in both A/B rounds (estimated +10%; attention was a bigger share than extrapolated). Commit in the same change as A3.
- [x] **A3** Head dimension 256 (Gemma family) accepted by `SupportsSplitKvGqa`; parity cases added (16/4/256 group 4, 8/8/256 group 1), 8 of 8 tests pass. Not yet run end to end on a Gemma model.
- [x] **A4** Remaining decode attention call sites: the fp32 path inside `GpuForwardPass` (both sites) and the hybrid Qwen3.5/3.6 engine (`VulkanHybridGdnForwardPass.AttentionAuto`, trunk and MTP head, partial buffers allocated when the shape is supported). Hybrid Qwen3.5-4B (16 heads / 4 KV heads, head dim 256, attention every 4th layer), 493-token prompt, alternating A/B: **decode 9.5 -> 10.9 t/s (+15%)**, text identical. Left: the Gemma-4 per-layer path (`ForwardGemma4`) and the layer-split engine, see A4b.
- [x] **A4b** Gemma-4 per-layer attention (fp32 KV) now uses the GQA split-KV kernel (head dim 256 SWA layers and 512 global layers; `SupportsSplitKvGqa` extended to 512; partial buffers sized by `_maxHeadDim`; `STINGRAY_VULKAN_DECODE_ATTN_GQA=0` rolls back). Gemma-4 E4B Q4_K_M after a 1,953-token prompt, 2 alternating rounds: **decode 5.4/5.4 -> 9.5/9.6 t/s (1.77x)**, its per-token prefill 6.6 -> 9.7 t/s, text identical. Kernel parity cases added for 256 and 512. Caveat: the repo Gemma-4 Vulkan test classes all skipped here (fixtures absent), so the A/B text plus the kernel parity test are the evidence. Gemma-4 prefill is still the per-token path (6-10 t/s at 2k): see new item B8.
- [x] **A5** GQA groups of 5-8: groups above 4 use ceil(G/4) workgroups per KV head (`SupportsSplitKvGqa` now accepts 1-8). 13 of 13 parity cases pass (groups 5, 6, 7, 8; head dim 256 with group 8). End to end, Qwen2.5-VL-7B (28 heads / 4 KV heads, group 7), 493-token prompt, alternating A/B: **decode 6.5 -> 7.5 t/s (+15%)**, text identical; its per-token prefill 7.0 -> 7.6 t/s.
- [ ] **A6** BF16 and Q8_0 KV-cache variants (`AttentionSplitKvPartialBf16`, `AttentionSplitKvPartialQ8`) have the same slow structure: port the design.
- [x] **A7** gpt-oss attention sinks on the GQA split-KV path (sink-aware combine kernel `AttentionSplitKvCombineSinks`: the sink logit joins the max and the denominator; group 8 served by two workgroups per KV head). gpt-oss-20b-MXFP4 after a 1,876-token prompt, 2 alternating rounds: **decode 4.8/4.9 -> 7.8/8.1 t/s (1.65x)**, per-token prefill 6.0 -> 7.8-7.9 t/s, text identical. Parity test vs the sinks kernel (incl. 128 SWA window), GptOssGpuParityTests and shader-drift pass. `STINGRAY_VULKAN_DECODE_ATTN_GQA=0` rolls back.
- [ ] **A8** (partly done) Verify on more models and contexts. Done: Qwen3-4B at about 1,900 tokens of context, alternating A/B: **decode 3.3 -> 10.9-11.0 t/s (3.3x)**, text identical (the gain grows with context, as predicted). Done 2026-10-11: SmolLM2-1.7B (head dim 64, MHA) after a 2,126-token prompt, 2 alternating rounds: **decode 8.3/8.2 -> 17.2/16.7 t/s (2.0x)**, text identical, prefill unchanged (85 t/s). Done 2026-10-11: Qwen3-8B-Q4_K_M after a 2,896-token prompt (SnapKV auto-on, budget 2048), 2 alternating rounds: **decode 2.2/2.2 -> 6.3/5.5 t/s (2.5-2.9x)**, text identical, prefill unchanged (40 t/s). Done 2026-10-11: Gemma-3-4B-Q4_K_M after a 2,885-token prompt: **decode 5.5 -> 10.9/11.2 t/s (2.0x)**, prefill 11.7 -> 37 t/s (the old path also slows prefill); the old-kernel round 2 hit `VK_ERROR_DEVICE_LOST` (baseline path only, one of two runs; the GQA path ran clean twice); text compared only by the first 140 chars. Still to do: a 4k context.
- [~] **A9** Short-context per-layer overhead, measured 2026-10-11 (`STINGRAY_VULKAN_DECODE_PROFILE=1` prints per-token host-record vs GPU-wait time and dispatch/barrier counts). Qwen3-4B decode: **record 1.3 ms of 77.4 ms per token, GPU wait 76.2 ms** (not host-bound); 771 dispatches and 588 barriers per token. Microbenchmark (scratch): a dependent tiny dispatch + barrier costs about 8 us (0.8-1.3 us without the barrier), so all barriers are about 4.7 ms (6%) of the token and fusing 150 dispatches would save roughly 1.5-3 ms (2-4%). The matmul floor is about 57 ms at 44 GB/s, so about 19 ms is non-matmul; most of it is small kernels running with few workgroups, not launch overhead. Verdict: modest ceiling (fusion of QK-norm + RoPE + KV append, add + norm), low priority; the larger lever is the matmul bandwidth gap (C-group).
- [ ] **A10** CUDA decode attention (`llm_attention`, `_bf16`, `_q8_0`) has the identical one-block-per-head, one-thread-per-key structure; needs an NVIDIA machine to measure before porting.
- [ ] **A11** CPU decode attention: league shows CPU decode falling from 0.88x to 0.67x of llama.cpp as context grows (267 to 3,218 tokens); profile first, reference is ggml's CPU flash attention.

### B. Prefill on Vulkan (Lead 2)
- [x] **B0** Diagnose: the batched prefill runs at about 0.3 TFLOP/s (multi-token mat-vec, 16 tokens per dispatch); a larger chunk silently falls back to a slow attention kernel. Evidence log: "Lead 2".
- [x] **B1** Opt-in Path 3 (`STINGRAY_VULKAN_MM_PATH=3`): tiled quantized GEMM incl. a new Q6_K decoder, `SgemmQuant`, and flash attention run in 16-query slices. Qwen3-4B prefill 38.4 to 71.3 t/s (+86%), text identical. Commit `eef12b83`.
- [x] **B2** Path 3 vs Path 1 as an alternating A/B, two rounds (Qwen3-4B-Q4_K_M, Vulkan): 493-token prompt **38.3 / 38.4 -> 72.4 / 71.8 t/s (+88%)**; 1,915-token prompt **27.4 / 27.4 -> 41.2 / 41.4 t/s (+51%)**; decode unchanged (12.0-12.4 and 10.9 t/s in both paths).
- [x] **B3** Path 3 chunk 128 to 256 (default; `STINGRAY_VK_PATH3_CHUNK` overrides, 16-1024). Qwen3-4B 493-token prompt, 2 alternating rounds: 128 = 71.3/71.5, 256 = 77.4/78.0 (+8.6%), 512 = 81.7/81.5 (+14%, single chunk; not made default: scratch grows, unmeasured on large models). Text identical; chunk-split 9/9, batched-prefill 3/3, spec-verify 18/18 at 256.
- [x] **B3b** Causal GQA register-tiled prefill attention (`FlashPrefillGqa128`, head dim 128, fp32 KV; `STINGRAY_VULKAN_FLASH_PREFILL=0` rolls back): a whole chunk runs in one dispatch (64x64 tile, 4x4 per thread) instead of 16-query slices that each re-read K/V. Qwen3-4B 1,915-token prompt, 2 alternating rounds: prefill **43.0/43.1 -> 97.9/99.5 t/s (2.3x)**, text identical, decode unchanged. Parity test `VulkanFlashPrefillGqaTests` (8 cases vs the sliced kernel), shader-drift, batched-prefill, chunk-split, spec-verify, arch-parity (18) and Qwen3-VL suites pass. Head dim 64 added (`FlashPrefillGqa64`): SmolLM2-1.7B 2,126-token prompt prefill 84.5/84.2 -> 153.5/153.3 t/s (1.8x), text identical, arch-parity 18/18. Open: other head dims, bf16/q8 KV and SWA still use the old slices.
- [x] **B4** Second dense model and decode check with Path 3 on (2026-10-11): Mistral-7B-Q4_K_M prefill 22.7 to 46.6 t/s (tokens per dispatch 16 to 115), SmolLM2-1.7B 81.7 to 180.2 t/s, Qwen3-4B 38 to 72; decode unchanged (7.8 and 24.5 to 24.8 t/s), greedy text identical.
- [x] **B5** Path 3 is now the default (2026-10-11; `STINGRAY_VULKAN_MM_PATH=1` rolls back). Evidence: arch-parity 18/18, chunk-split 9/9, spec batch-verify 18/18, layer-split 13/13, hybrid-OLMoE 7/7, Qwen3-VL M-RoPE 2/2, KV dtype 4/4, flash 14/14, SnapKV, batched-prefill 3/3 (Path 3 tolerance widened to 2e-2 like Path 2: logit delta 9.2e-3, argmax identical). DeepSeek2 CPU-parity fails identically on Path 1 and 3 (pre-existing, separate). Still open: Q3_K/Q5_K weights in the batched trunk.
- [x] **B5b** (found by the 2026-10-11 all-checkpoint recheck) K-quants other than Q4_K/Q6_K stay off the batched trunk: SmolLM2-1.7B Q5_K_M prefills at 22.3 tok/s on Vulkan at 555 tokens against 235 tok/s for the Q4_K_M file, because `ComputeCanBatchedTrunk` accepts only Q4_K/Q6_K weights while Path 3 already serves Q3_K/Q5_K. Fix: allow Q3_K/Q5_K in `IsBatchable` when Path 3 is active (MatMulBatched already routes them); then re-run the arch-parity suite. FIXED 2026-10-11: `IsBatchable` accepts Q3_K/Q5_K when Path 3 is active. SmolLM2-1.7B Q5_K_M, 554-token prompt: **prefill 21.9 -> 219.7 tok/s (10x)**, greedy text identical to the per-token path; arch-parity suite 19/19 in 192 s incl. a new Q5_K_M case (worst cosine 0.9987, 0/9 argmax flips).
- [ ] **B6** MoE prefill (currently handed to the CPU) and `VulkanLayerSplitForwardPass` per-token prefill: revisit after B5.
- [x] **B8** Gemma-4 prefill via the CPU-prefill KV handoff (2026-10-11; not a GPU batched trunk). Gemma-4 E4B Q4_K_M, 495-token prompt: **prefill 9.8 -> 56.0 t/s (5.7x)** (CPU batched pass 59 t/s + 62 ms KV upload), decode 9.3-9.5 unchanged, greedy text identical. The handoff now uploads per-layer head dims (256/512) and skips the aliased shared-KV tail layers; `Gemma4CpuPrefillHandoffTests` (default and F32 KV, 24 teacher-forced steps vs the sequential GPU prefill: cosine 0.9995-0.9999, prefill logits 0.9988) is the receipt, family `gemma4` registered for the full-GPU path. Regression check done 2026-10-11 after the `HandoffKv` refactor: `GpuCpuPrefillHandoffTests` 6/6 pass in 345 s on real weights (OLMoE, Qwen1.5-MoE, Phi-3.5-MoE; prefill cosine 0.9995-0.9999, startPos>0 continuation 0.9998). A batched GPU Gemma-4 trunk (for discrete GPUs) is still open.
- [ ] **B9** (new, 2026-10-11) gpt-oss prefill: both paths are per-token. gpt-oss-20b-MXFP4, 514-token prompt: CPU `-g 0` 10.1 t/s, Vulkan `-g -1` 7.8 t/s (about 4 min for 2k tokens). `GptOssGpuForwardPass` has its own per-token loop (router top-k on the CPU, submit per layer) and the generic CPU-prefill handoff does not reach it. Needs a batched MoE prefill (expert-grouped GEMM) or a CPU batched pass for the gpt-oss graph; large.
- [x] **B7** Batched trunk for M-RoPE models while no image is registered (`BatchedTrunkNow`; also requires sections to cover all pairs). Qwen2.5-VL-7B-Q4_K_M, 493-token text prompt: prefill 7.6 to 53.5 t/s (7x), greedy text identical to the per-token path (`STINGRAY_VULKAN_NO_BATCHED_PREFILL=1`), decode unchanged 7.4-7.5; Qwen3-VL M-RoPE parity tests 2/2 (images still take the per-token path).

### C. Mat-vec kernels (Lead 1)
- [x] **C1** MXFP4: one block per lane, packed-LUT decode. 14 to 25 GB/s; gpt-oss decode 6.7 to 7.5 t/s (+12%). Commit `c30e455a`.
- [x] **C2** Q8_0 (33 to 43 GB/s, ggml 45) and Q4_0 (+7%). Qwen3-0.6B-Q8_0 decode +7-10%. Commit `c7bd0f80`.
- [ ] **C3** Q3_K (16 vs ggml 29 GB/s): needs the cooperative scale-cache structure; parked.
- [ ] **C4** Residuals: Q6_K and Q5_K (about 1.2x behind), Q4_0 (about 1.2x), MXFP4 (25 vs 35.9 GB/s: six-word funnel loads, several rows per workgroup).

### D. Qwen Image (earlier thread; the plan for it lives in the other plan file)
- [x] **D1** Small-M quantized GEMM, **D2** chunked MLP: forward 24.5 s to 21.9 s (reference about 20.7 s). Rejected and logged: raster order, split-K, 128x128 tile, unrolled loop.
- [ ] **D3** FP16-tile GEMM project (the only large-upside item left; carries the numerics decision). Plan: [the quantized GEMM plan](2026-10-10-vulkan-quantized-gemm-fp16-tile-plan.md).

### E. Other leads and deferred work
- [x] **E1** Hybrid-GDN (Qwen3.5/3.6) batched prefill, re-measured and fixed 2026-10-11. Cause: the batched trunk was capped at 8 tokens per chunk and ran the dense FFN per row (weights re-streamed per token). Now: chunk = min(128, active matmul path width) (`STINGRAY_VULKAN_HYBRID_PREFILL_CHUNK`), dense-GPU FFN batched across the chunk, and `MatMulBatched` falls back instead of throwing for dtypes Path 3 does not serve. 480-token prompt: **Qwen3.5-4B 11.0 -> 46.1 t/s (4.2x)** (chunk 8: 17.6, 64: 33.6, 128: 46.1, 256: 48.2, 512: 50.8), **Qwen3.5-9B 6.4 -> 35.5 t/s (5.5x)**, decode unchanged, greedy text identical to the sequential path on both. Follow-up: head-dim-256 causal prefill attention kernel (`FlashPrefillGqa256`) used by the hybrid attention layers: 45.4 -> 53.2 t/s on Qwen3.5-4B, text identical; elimination experiments (skip one stage, 480 tokens, 10.4 s baseline): scan costs 2.2 s, per-query attention cost 2.9 s, conv 0; the opt-in chunked scan gives no gain (46.0 vs 46.1). Open: the recurrence scan (about 2 s); MoE hybrid FFN is still per row; the CUDA-oracle hybrid tests cannot run here.
- [ ] **E2** Lead 4: MoE decode gap (OLMoE 0.50x).
- [ ] **E3** Lead 5: FLUX.2 GPU (3,260 s for one 512x512 image).
- [ ] **E4** Lead 6: diffusion-side `MatVecDq*` kernels (parked, would double the effort).
- [ ] **E5** Reference per-op timings for Wan 2.2, HunyuanVideo and FLUX.2: explicitly deferred until the leads above are done or parked.
- [ ] **E6** (later, from the 2026-10-11 all-checkpoint sweep) Text sanity check in `scripts/bench-league-recheck.ps1`: record the first ~80 characters of each run's output, flag empty or degenerate text (repetition, mostly non-text), and keep a baseline file of known-good starts so a later sweep reports "text changed" per model. Today the sweep proves a checkpoint loads and decodes (CPU 94 of 95, Vulkan 81 of 95 at 94 models in), not that its output is right.

## How the wins were found (the playbook; reuse it)

Every win above came from the same loop. The order matters: each step removed a wrong guess before any kernel was written.
1. **Fix the number and the workload.** One model, one prompt, one machine, greedy decoding; quote the prompt length with any decode figure (decode slows with context, which is what hid the attention problem for a while).
2. **Get a reference on the same hardware.** ggml's Vulkan backend is vendored (`examples/stable-diffusion.cpp`, `test-backend-ops`): `GGML_VK_PERF_LOGGER=1` gives per-op GPU time and GFLOP/s by shape, and its F16/Q8_0 mat-vec gives the machine's sustained bandwidth ceiling (about 44 GB/s here).
3. **Measure by elimination.** Temporarily skip a class of work (all matmuls) and time the rest; repeat at a short and a long context. The difference says what that class costs, and what scales with context (here: not the matmuls).
4. **Compare against a floor.** Bytes that must move divided by the ceiling (K/V read at 500 tokens is 147 MB, 3.5 ms) against the measured time (68 ms) showed the kernel, not the memory system, was the problem: about 19x slower than the floor.
5. **Read the kernel next to the reference kernel** for structural causes: uncoalesced reads (lanes a whole KV row apart), idle threads (head-dim of 256 threads working), redundant work (each query head re-reading the K/V it shares with three others), and hidden fallbacks (a chunk above 16 queries silently dropped to a much slower attention kernel).
6. **Rewrite it standalone first**, with a harness that compares the new kernel against the old on random data across the awkward shapes (context lengths around split boundaries, head dims, group sizes, a sliding window) and times it at the real shape. Only then wire it in.
7. **Wire it behind an environment switch**, A/B it with alternating runs, require the greedy text to be identical, add a repository regression test and run the shader-drift gate.
8. **Write down every number, including the experiments that failed** (raster order, split-K, 128x128 tile, unrolled loop, 16-weights-per-lane mat-vec): they are why the next person does not repeat them.

Traps met along the way: documentation drifted from code (the "not implemented" Path 2, the "per-token prefill" description); LLM weights are generic raw tensors whose quantization type travels separately; a standalone benchmark that syncs per call hides small kernels under fixed overhead; run-to-run drift is about 5%, so a single run is a direction, not a number; the device has a 64-wide subgroup and an earlier subgroup-reduction bug, so reductions go through shared memory.

---

## Original lead write-ups and evidence log (chronological; the status board above supersedes the ordering)

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

## The original leads, ranked (history; see the status board for current state)

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

### Lead 6 (deferred follow-on): diffusion-side mat-vec kernels
- **Evidence:** the `MatVecDq*` family (Qwen, FLUX, SD3.5 and other diffusion M=1 projections) trails ggml too: Q3_K 15.8 vs 29.0 GB/s,
  Q4_K 23.6 vs 42.6, Q5_K about 26 vs 42.9 (same 4096x14336 shape, kernel time only, 2026-10-10). A 16-weights-per-lane variant was tried
  and was slower, so decode amortisation alone does not help.
- **Why deferred:** it is a separate body of work that would double the effort, and its end-to-end effect on diffusion is small today
  (Qwen's M=1 products are about 1% of a forward). Taken after Leads 1-5, using whatever structure the LLM-decode work settles on
  (aligned word loads, packed scale decode, scale factored out of the dot product, several rows per workgroup, whole blocks per lane).

## Not worth chasing now
CPU SmolLM2 prefill (about 0.7-0.9x per the notes), Qwen Image non-GEMM ops and VAE (at parity with the reference).

## Original order and checkpoints (superseded by the status board)
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

### Scope note, 2026-10-10: diffusion-side mat-vec kernels are parked
The `MatVecDq*` family (diffusion M=1 projections) also trails ggml (Q3_K 15.8 vs 29 GB/s, Q4_K 23.6 vs 42.6) and could use the same
ggml-style rewrite, but it is a separate body of work that would double the effort and matters little for diffusion speed today
(Qwen's M=1 products are about 1% of a forward). Deferred; revisit after the LLM-decode kernels below.

### Lead 1, MXFP4 first result, 2026-10-10
New `MatVecMxfp4V2` (`Shaders.cs`): one 17-byte block per lane (five unaligned 32-bit loads), the 32 values decoded with a packed 4-bit
magnitude constant (no dynamically indexed array), input read as aligned vec4s, same push constants and dispatch. Standalone, 4096x14336:
**14.1-14.3 GB/s -> 24.3-25.1 GB/s (about 1.75x)**, output sum of |c| identical to 8 digits (summation order differs, first element agrees to
6 digits). ggml reaches 35.9 GB/s, so further room remains. Switch: `STINGRAY_VK_MXFP4_V2=1` (experiment; default flips after end-to-end
gpt-oss parity and speed checks below).

### Lead 1, MXFP4 kept as the default, 2026-10-10
- Real model, gpt-oss-20B (`gpt-oss-20b-MXFP4.gguf`), Vulkan GPU, greedy, 48 tokens, alternating A/B (old kernel / new kernel, two rounds):
  decode **6.7 / 6.7 t/s -> 7.5 / 7.6 t/s (+12-13%)**, prefill 6.7-6.9 -> 7.7 t/s, generated text **identical** in both rounds.
- Correctness gate added: `VulkanMxfp4MatVecTests` compares the Vulkan mat-vec with the CPU MXFP4 dequantizer and a double-precision dot
  (rows 37-70, cols 32 / 96 / 1024 / 2880 / 14336, row offsets 0-11); both the old and new kernels pass; the shader-drift gate passes.
- `STINGRAY_VK_MXFP4_V2=0` restores the old kernel. Remaining gap to ggml on this shape: 25 vs 35.9 GB/s. Ideas not yet tried: load the
  17-byte block as six consecutive words and funnel-shift once instead of five unaligned two-word reads; several rows per workgroup to share
  the activation loads.

### Lead 1, Q8_0 and Q4_0, 2026-10-10: kept
`MatVecQ8_0V2` and `MatVecQ4_0V2` (`Shaders.cs`): one block per lane, block bytes read as aligned or 16-bit-shifted 32-bit words (block starts are even),
Q8_0 bytes converted with `bitfieldExtract`, input read as vec4s, scale read once per block. Same bindings, push constants and dispatch.
- Standalone, 4096x14336 (ggml in brackets): **Q8_0 33.0 -> 43.0-43.3 GB/s (+31%) [45.1]**, Q4_0 31.2-31.4 -> 33.5-33.9 (+7%) [41.4].
  Q8_0 is within 4% of the reference; Q4_0's per-nibble integer decode is ALU-bound and still about 18% behind.
- Real model, Qwen3-0.6B-Q8_0, Vulkan, greedy, 96 tokens, alternating A/B: decode **23.6 / 23.6 -> 25.3 / 25.9 t/s (+7-10%)**, prefill 27.4-27.6 -> 29.6-30.8
  t/s (+8-12%), text identical in both rounds. A 0.6B model is overhead-heavy, so larger Q8_0 models should gain more (not yet measured).
- Correctness gate added: `VulkanQuantMatVecTests` (Q8_0, Q4_0, Q3_K, Q4_K, Q5_K, Q6_K against the CPU dequantizer with a double-precision dot; 16 cases,
  new and old kernels both pass). Rollback: `STINGRAY_VK_MATVEC_V2=0`.
- Remaining Lead 1 order: Q3_K (1.8x behind), Q4_0 (18%), Q6_K (1.23x), Q5_K (1.2x); MXFP4 still 25 vs 35.9 GB/s.

### Lead 1 status, 2026-10-10: the large wins are shipped; the rest is parked with reasons
Shipped: MXFP4 (1.75x kernel, gpt-oss decode +12%) and Q8_0 (+31% kernel, within 4% of ggml; Qwen3-0.6B-Q8_0 decode +7-10%), plus Q4_0 (+7%).
Q4_K is already at parity with ggml (42.2 vs 42.6 GB/s). Parked:
- **Q3_K (1.8x behind):** ggml's kernel uses 16 cooperating threads per block with the 16 sub-block scales decoded cooperatively into shared memory
  (barrier per block); even so it reaches only 29 GB/s. Our kernel issues about a dozen small, partly unaligned loads per 8 elements. A faithful port is a
  restructuring, for an audience limited to Q3_K checkpoints (for example Qwen3.6-27B-UD-Q3_K_XL, which does not fit the GPU budget anyway).
- **Q6_K, Q5_K (about 1.2x behind), Q4_0 (about 1.2x):** each is a modest share of a typical model's bytes (Q6_K is roughly a fifth of a Q4_K_M model), so the
  end-to-end gain is a few percent each.
- **MXFP4 residual (25 vs 35.9 GB/s):** untried ideas are six-word funnel loads and several rows per workgroup.
Revisit if a profile shows one of these types dominating a model we care about. Next: Lead 2 (batched prefill on the Vulkan LLM path).

### Lead 2, first measurement, 2026-10-10 (Vulkan iGPU vs CPU, same build, ~500-token benchmark prompt, 8 generated tokens)

| Model | CPU prefill | Vulkan prefill | CPU decode | Vulkan decode |
|---|---|---|---|---|
| Qwen3-4B-Q4_K_M (dense, 36 layers) | 102.8 t/s | 38.0 t/s (2.7x slower) | 12.2 t/s | 6.0 t/s |
| OLMoE-1B-7B-Q4_K_M (MoE) | 175.0 t/s | 155.7 t/s (near parity) | 30.8 t/s | 14.2 t/s |

Reading: the September finding (dense-model Vulkan prefill much slower than CPU) still holds, while MoE prefill is near parity. Vulkan decode is about half
of CPU decode on both models even though the Q4_K mat-vec kernel is at ggml parity (42.2 vs 42.6 GB/s), so Vulkan decode on this machine looks overhead-bound
(per-token dispatches, syncs, non-mat-vec kernels, lm_head), not mat-vec-bound. Caveat: only 8 decode tokens, which is a noisy basis; re-measure with 48+.
No decode profiler exists for `GpuForwardPass` (`STINGRAY_PROFILE_DECODE` is the hybrid-GDN path's), so the first step of Lead 2 is to add one: per-stage
GPU/host time for a decode token and for a prefill chunk, then decide between batched prefill, fewer syncs, or kernel work.

### Lead 2, diagnosis, 2026-10-10: the batched prefill exists but runs at about 0.3 TFLOP/s; the fix is a prefill-sized tiled quantized GEMM
Correction to the backlog wording: `GpuForwardPass` already has a weight-amortising batched prefill (chunked trunk, `_canBatchedTrunk`: dense models with
Q4_K/Q6_K weights) and records each decode token into one command buffer; "prefills per token" describes `VulkanLayerSplitForwardPass` and the MoE path (which hands
prompts to the CPU). The slow part is the matmul kernel.
- Measured (Qwen3-4B-Q4_K_M, 493-token prompt, Vulkan, `STINGRAY_VULKAN_MM_STATS=1`): Path 1 (default) 38.3 t/s, Path 2 (the existing tiled `MatMulTiledQ4K/Q6K`,
  capped at 16 tokens per dispatch) 38.8 t/s. Weight traffic is 64,197 MiB (130 MiB/token) in both, versus a GEMM floor of about 5 MiB/token.
- That traffic is not the bottleneck: 64 GB at the measured 44 GB/s ceiling is about 1.5 s, but the prefill takes 12.9 s. About 4 TFLOP in 12.9 s is **0.3 TFLOP/s**, so both
  paths are compute/kernel-bound. A tiled quantized GEMM with a 128-row tile (`SgemmQ3K/Q4K/Q5K`, which reach about 1 TFLOP/s at M >= 128 in the Qwen Image work)
  would take the same prefill to roughly 4 s (about 120 t/s), CPU parity (CPU: 102.8 t/s).
- **Design:** (1) add a Q6_K decoder and `SgemmQ6K` (Q4_K_M models keep Q6_K for ffn_down in many layers and the output matrix; without it those layers pin the chunk at 16);
  (2) implement Path 2 for Q3_K/Q4_K/Q5_K/Q6_K by routing to `VulkanBackend.Sgemm` (activations [nTok, K] and outputs [nTok, N] already match its layout);
  (3) raise `Path2MaxTokensPerDispatch` from 16 to a prefill-sized chunk (128-256) and check the scratch buffers and batched attention/KV-append kernels accept it
  (the prefill chunk already follows `MaxTokensPerDispatch`; the speculative-decode verify constant `MaxBatchVerifyK` stays 16); (4) keep Path 1 as the default until the A/B proves Path 2.
- **Gates:** per-kernel CPU-parity tests (extend `VulkanQuantMatVecTests` style to the GEMM), then end-to-end greedy generation identical to Path 1 on Qwen3-4B and one more dense model,
  then the prefill A/B (target: Vulkan prefill at least 2x today's, i.e. at or above CPU) and a Vulkan decode regression check. Rollback: `STINGRAY_VULKAN_MM_PATH=1`.

### Lead 2, Path 3 first result, 2026-10-11 (Qwen3-4B-Q4_K_M, 493-token prompt, Vulkan iGPU, greedy)
Implemented (all opt-in, `STINGRAY_VULKAN_MM_PATH=3`; the default stays Path 1):
- `SgemmQ6K` (Q6_K decoder in the shared tiled GEMM; large and small tile) and `VulkanBackend.SgemmQuant` (Q3_K/Q4_K/Q5_K/Q6_K, explicit weight type for the LLM's generic raw tensors);
  CPU-parity tests `VulkanSgemmQuantEntryTests` (6 cases, M from 1 to 130, N not a tile multiple).
- `VulkanMatMulPath.Path3` (prefill chunk 128) routed in `MatMulBatched`.
- Sliced flash attention in the batched trunk (`GpuForwardPass.RecordBatchedFlashAttention`): the flash kernel holds one register accumulator per query and serves at most
  16 per dispatch, so a 128-token chunk previously fell back to the per-(head, query) `AttentionBatched` kernel, which re-streams the K/V cache for every query.
  Chunks above 16 now run as 16-query slices of the same flash kernel (`basePos` offset per slice; K/V for the whole chunk are already appended).
- Measured by elimination (matmuls skipped, cumulative ms for the whole prefill): Path 1 = 10.6 s matmul + 2.15 s other; Path 3 before the slicing fix = 5.1 s matmul + **14.0 s other**
  (the fallback attention); Path 3 after = 5.1 s matmul + about 1.8 s other.

| Qwen3-4B-Q4_K_M, 493-token prefill, Vulkan | Prefill |
|---|---|
| Path 1 (default) | 38.4 t/s (12.8 s) |
| Path 3 (128-token chunks, tiled quantized GEMM, sliced flash attention) | **71.3 t/s (6.9 s), +86%** |
| CPU, same prompt (2026-10-10) | 102.8 t/s |

Greedy generation is identical to Path 1. Matmuls are now about 0.8 TFLOP/s effective (4 TFLOP in 5.1 s). Weight traffic fell from 64.2 GB to 8.3 GB (16.8 MiB/token).
Next: larger chunks (256), a second dense model, decode-regression check, then decide whether Path 3 earns the default.

### Lead 2b (new): Vulkan decode, where the time goes, 2026-10-11 (Qwen3-4B-Q4_K_M, greedy, GPU)
Decode is a separate cost from prefill, so Path 3 does not touch it. Elimination experiment (single-vector matmuls skipped temporarily, so the elapsed time shows everything else; hook removed again):

| Context | Decode | Matmuls | Everything else (per token) |
|---|---|---|---|
| short (30-token prompt) | 10.8-10.9 t/s (92 ms) | about 72 ms (about 35 GB/s on 2.5 GB, close to the 42-44 GB/s ceiling) | 49 t/s without matmuls, so about 20 ms |
| after a 493-token prompt | 6.3 t/s (159 ms) | about 71 ms (unchanged) | 11.3-11.5 t/s without matmuls, so **about 88 ms** |

- The matmul share is near the memory ceiling and the Q4_K mat-vec is at ggml parity, so little is left there.
- **About 68 ms per token is attention (and KV handling) at about 500 context tokens.** The K/V traffic for this model at that context is about 147 MB per token
  (36 layers x 500 positions x 8 KV heads x 128 x 4 B x 2), about 3.5 ms at 42 GB/s: the single-token attention kernel is roughly 19x slower than the bandwidth floor.
  Target: attention near 10 ms at 500 tokens would lift decode after a long prompt from 6.3 to about 10 t/s (+57%), and the gain grows with context.
- About 20 ms per token (22% at short context) is the rest: per-layer norms, RoPE, residual adds, SwiGLU, KV append, sampling readback, plus dispatch/barrier overhead
  (ggml fuses RMS-norm+mul(+RoPE), add+RMS-norm; we issue them separately).
- Next steps: (1) time our single-token attention kernel standalone against ggml's `flash_attn` at this shape (hsk=128, 32 heads, 8 KV heads, 512 keys, 1 query) via `test-backend-ops perf -o FLASH_ATTN_EXT`;
  (2) read our decode attention kernel (`Shaders.Attention*` and the GQA mapping) for the structural cause (serial softmax, per-head KV re-reads for the 4 query heads sharing a KV head, shared score array limiting occupancy);
  (3) fix, A/B on the long-context decode numbers above, parity via greedy text.

### Lead 2b, reference and diagnosis of the decode attention kernel, 2026-10-11
- Reference (ggml `FLASH_ATTN_EXT`, this GPU, 8 KV heads x 4 query heads (Qwen3-4B's shape), head dim 128, FP16 KV, one query): 1.07 ms per layer at 4,096 keys, 2.15 ms at 8,192, 5.14 ms
  at 16,384, about 0.26 us per key per layer. At about 520 keys that is roughly 5 ms per token over 36 layers; ours is about 68 ms (about 13x behind).
- Split-KV decode (`AttentionSplitKvPartial` + `Combine`) is already default-on above 256 positions, so the 6.3 t/s at 500 tokens already includes it (two splits of 256 keys, 64 workgroups).
  The partial kernel's structure is the cost: (1) QK phase: one key per thread, each looping over its 128 floats with consecutive lanes `kv_dim` floats apart (uncoalesced), `q` re-read from global each iteration;
  (2) value phase: only `head_dim` of 256 threads work, each walking all 256 positions serially behind a global load; (3) the 4 query heads that share a KV head run in separate workgroups and re-read the same K/V.
- Design for a replacement partial kernel (fp32 KV first; combine and partial layout unchanged): one workgroup per (KV head, split) handling the whole GQA group (up to 8 query heads); QK phase with 16 lanes per key
  (each lane 8 dims as two vec4 loads, q for all group heads held in registers, reduction over the 16 lanes), scores for the group in shared memory; value phase with 32 lanes covering the head dim as vec4 and 8 key groups
  walking interleaved positions, accumulators per group head, then a shared-memory reduction over the key groups. Falls back to the existing kernel for head dim other than 64/128 or groups above 8.
- Gates: standalone parity against the existing partial+combine on random data across seq lengths 1-2,048 and several group sizes; greedy text identical on Qwen3-4B; A/B on long-context decode (target: attention near 10 ms at 500 tokens,
  decode after a 493-token prompt from 6.3 to about 10 t/s); then a second model with a different head count. Rollback env switch.

### Lead 2b, decode attention kernel shipped, 2026-10-11 (default on, `STINGRAY_VULKAN_DECODE_ATTN_GQA=0` rolls back)
`AttentionSplitKvPartialGqa` (`Shaders.cs`) and `VulkanBackend.AttentionSplitKvGqa`, wired through `GpuForwardPass.AttentionSplitKvAuto` (both decode call sites; fp32 KV; head dim 64/128; 1-4 query heads per KV head;
anything else keeps the original kernel). One workgroup per (KV head, split) serving the whole GQA group: 8 (or 4) lanes per key with contiguous vec4 K reads and the group's Q in registers, softmax statistics for all group heads
together, value phase as 32 (or 16) vec4 lanes across the head dim times interleaved key groups with a shared-memory reduction. Same partial layout and combine pass.
- Standalone, 32 heads / 8 KV heads / head dim 128, 520 keys, 36 layers per submission: old 56.4 ms (1,566 us/layer) -> **new 4.8 ms (132 us/layer), 11.8x**; ggml's `flash_attn` is about 140 us/layer at 512 keys (FP16 KV).
  Parity vs the old kernel across seq 1-2,048, head dims 64/128, groups 1/2/4: max difference about 1e-7.
- End to end, Qwen3-4B-Q4_K_M, Vulkan, greedy, 64 tokens, alternating A/B: **decode after a 493-token prompt 6.5 / 6.3 -> 12.0 / 12.2 t/s (+88%)**, generated text identical in both rounds; with a 30-token prompt 10.7 -> 10.9 t/s,
  text identical. Vulkan decode at long context is now level with the CPU (12.2 t/s).
- Tests: `VulkanSplitKvGqaAttentionTests` (5 shapes x 9 context/window cases plus the support check); shader-drift gate passes.
- Remaining decode costs at short context (about 20 ms per token outside the matmuls, matmuls near the bandwidth ceiling): fusing norm/RoPE/residual dispatches. bf16/q8 KV caches still use the old kernels.

### Decode attention A2 and A3, 2026-10-11
- A2: the GQA kernel now serves every fp32-KV decode position when the shape is supported (below 257 positions it is a single split plus the combine pass), instead of the original
  one-workgroup-per-head kernel. Qwen3-4B-Q4_K_M, Vulkan, 30-token prompt, 96 greedy tokens, alternating A/B: decode 10.6 / 10.5 -> 12.6 / 12.7 t/s (+20%), text identical.
- A3: `SupportsSplitKvGqa` accepts head dim 256 (the kernel's lane mapping already covered it: 16 lanes per key, 4 key groups in the value phase, 4,096-float shared reduction).
- Not touched yet: the Gemma-4 per-layer path (`ForwardGemma4`, per-layer head dims and shared KV) and the hybrid Qwen3.5/3.6 engine (head dim 256 but GQA groups of 6-8, which needs A5).

### Decode attention A5, 2026-10-11
Groups of 5-8 query heads per KV head are served by `ceil(G/4)` workgroups per KV head (`AttentionSplitKvPartialGqa`: group chunk from `gl_WorkGroupID.x`, `G = min(4, Gt - 4 * chunk)`), reading K/V twice instead of once per query head.
Parity against the original kernel: groups 5, 6, 7, 8 at head dim 128 and group 8 at head dim 256, plus all earlier cases (13 of 13). Qwen2.5-VL-7B-Instruct-Q4_K_M (group 7), Vulkan, 493-token prompt, 48 greedy tokens, alternating A/B:
decode 6.5 / 6.5 -> 7.5 / 7.5 t/s (+15%), text identical in both rounds. Finding while testing: this model prefills at about 7 t/s because the batched trunk excludes Q/K/V-bias models (new item B7).

### Decode attention A4, 2026-10-11
`VulkanHybridGdnForwardPass` (Qwen3.5/3.6 hybrid) now calls `AttentionAuto`, which uses `AttentionSplitKvGqa` when the shape is supported (partial buffers sized `numHeads x ceil(maxSeqLen/256) x headDim`, allocated in the constructor and freed in Dispose, skipped when
`STINGRAY_VULKAN_DECODE_ATTN_GQA=0` or `maxSeqLen` exceeds 65,536). Qwen3.5-4B-Q4_K_M (16 heads, 4 KV heads, key length 256, full attention every 4th layer), Vulkan, 480-token prompt, 48 greedy tokens, alternating A/B:
decode 9.5 / 9.5 -> 10.9 / 10.9 t/s (+15%), prefill unchanged (12.6 t/s: it does not use this kernel), text identical in both rounds.

### A8 partial, B3b finding, 2026-10-11
Qwen3-4B-Q4_K_M, Vulkan, Path 3 prefill (both modes), 1,915-token context, 48 greedy tokens, alternating A/B of the decode kernel: decode 3.3 / 3.3 t/s -> 10.9 / 11.0 t/s (3.3x), text identical.
Prefill at that length with Path 3 was 41.3 t/s (71.3 at 493 tokens): the attention share of prefill grows with context (item B3b).
