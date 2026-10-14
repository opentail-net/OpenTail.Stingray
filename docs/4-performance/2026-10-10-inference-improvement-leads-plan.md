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
> Ticked (`[x]`) items were moved verbatim to [../done/2026-10-10-inference-improvement-leads-closed.md](../done/2026-10-10-inference-improvement-leads-closed.md) on 2026-10-14; only open (`[ ]`) and in-progress (`[~]`) items remain here, with their original ids.

Tick an item when it is done, add the commit and the measured result next to it, and put the supporting numbers in the evidence log at the end of this file.
Order below is "what to look at next" within each group; the groups themselves are ordered by expected payoff. Updated 2026-10-11.

### A. Decode on Vulkan (new, top of the list: the attention win generalises)
- [~] **A6** (2026-10-11) GQA-group split-KV kernels for fp16-packed ("bf16") and block_q8_0 KV caches (`AttentionSplitKvPartialGqaBf16`/`Q8`, `AttentionSplitKvGqa(kvFormat:)`), wired into all four decode sites, same short-context gate as fp32, `STINGRAY_VULKAN_DECODE_ATTN_GQA=0` rolls back. Kernel parity vs CPU SDPA: 8 new cases (groups 2/4/8, head dim 64/128/256, multi-split, tiny tail) plus the 9 existing split-KV tests pass. End to end, Qwen3-4B `--kv-type bf16`, 2,896-token prompt: **decode 6.1 -> 10.6 t/s**, text identical over the first 120 chars (one round only). **Open:** `--kv-type q8_0` end to end: a long-prompt run exits 1 silently even with the GQA kernel off (not yet shown to pre-date this change), so Q8 has kernel parity only; the second bf16 round also died silently. Further model runs paused at the user's request (a peer session reported the loop restarting). The precompiled-shader drift tests fail on a clean HEAD too (`RmsNorm` missing), unrelated.
- [~] **A9** Short-context per-layer overhead, measured 2026-10-11 (`STINGRAY_VULKAN_DECODE_PROFILE=1` prints per-token host-record vs GPU-wait time and dispatch/barrier counts). Qwen3-4B decode: **record 1.3 ms of 77.4 ms per token, GPU wait 76.2 ms** (not host-bound); 771 dispatches and 588 barriers per token. Microbenchmark (scratch): a dependent tiny dispatch + barrier costs about 8 us (0.8-1.3 us without the barrier), so all barriers are about 4.7 ms (6%) of the token and fusing 150 dispatches would save roughly 1.5-3 ms (2-4%). The matmul floor is about 57 ms at 44 GB/s, so about 19 ms is non-matmul; most of it is small kernels running with few workgroups, not launch overhead. Verdict: modest ceiling (fusion of QK-norm + RoPE + KV append, add + norm), low priority; the larger lever is the matmul bandwidth gap (C-group).
- [ ] **A10** CUDA decode attention (`llm_attention`, `_bf16`, `_q8_0`) has the identical one-block-per-head, one-thread-per-key structure; needs an NVIDIA machine to measure before porting.
- [ ] **A11** CPU decode attention: league shows CPU decode falling from 0.88x to 0.67x of llama.cpp as context grows (267 to 3,218 tokens); profile first, reference is ggml's CPU flash attention.

### B. Prefill on Vulkan (Lead 2)
- [ ] **B6** MoE prefill (currently handed to the CPU) and `VulkanLayerSplitForwardPass` per-token prefill: revisit after B5.
- [ ] **B9** (new, 2026-10-11) gpt-oss prefill: both paths are per-token. gpt-oss-20b-MXFP4, 514-token prompt: CPU `-g 0` 10.1 t/s, Vulkan `-g -1` 7.8 t/s (about 4 min for 2k tokens). `GptOssGpuForwardPass` has its own per-token loop (router top-k on the CPU, submit per layer) and the generic CPU-prefill handoff does not reach it. Needs a batched MoE prefill (expert-grouped GEMM) or a CPU batched pass for the gpt-oss graph; large.

### C. Mat-vec kernels (Lead 1)
- [ ] **C3** Q3_K (16 vs ggml 29 GB/s): needs the cooperative scale-cache structure; parked.
- [ ] **C4** Residuals: Q6_K and Q5_K (about 1.2x behind), Q4_0 (about 1.2x), MXFP4 (25 vs 35.9 GB/s: six-word funnel loads, several rows per workgroup).

### D. Qwen Image (earlier thread; the plan for it lives in the other plan file)
- [ ] **D3** FP16-tile GEMM project (the only large-upside item left; carries the numerics decision). Plan: [the quantized GEMM plan](2026-10-10-vulkan-quantized-gemm-fp16-tile-plan.md).

### E. Other leads and deferred work
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

## Not worth chasing now
CPU SmolLM2 prefill (about 0.7-0.9x per the notes), Qwen Image non-GEMM ops and VAE (at parity with the reference).

Closed items (A0-A8, B0-B8 except open ones, C1-C2, D1, E1), the original ranked lead write-ups, the superseded original order and the full chronological evidence log ("Lead 1", "Lead 2", "Lead 2b", decode attention A2-A8) are archived verbatim in [../done/2026-10-10-inference-improvement-leads-closed.md](../done/2026-10-10-inference-improvement-leads-closed.md).
