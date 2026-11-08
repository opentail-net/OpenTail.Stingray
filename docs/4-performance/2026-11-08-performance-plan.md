# Performance plan

Status 2026-11-08. Performance work was paused by the owner on 2026-10-30. This is the one plan for performance. It replaces the pickup guide, the leads plan and the perf sweep, which are now in [done/](../done) as history. Numbers are in [PerformanceLeague.md](../../PerformanceLeague.md).

**Machine.** Ryzen 7 5700G (Zen 3, 8 cores, 16 threads, AVX2 and FMA, no VNNI, no AVX-512), 64 GB DDR4, integrated Radeon that shares system RAM. "Ratio" means Stingray speed divided by reference speed; 1.0x is parity. Default CPU thread count is the number of physical cores (8).

**Rules.** [performance-measurement-rules](../reference/performance-measurement-rules.md) apply. In short: one run at a time; interleaved A/B; three or more runs; equal output before and after; named models; record every result in PerformanceLeague.md. Rule 13: a Vulkan or CUDA loss on this iGPU is not evidence about a discrete GPU, so GPU results here are never performance claims.

**Start rule.** Nothing in Part A starts until the owner unpauses performance (Part B, decision 1). Part C is blocked on hardware or on a checkpoint.

---

## Part A: CPU work, runnable here, in order

Each item says what the item's first step is, and when it is done.

### A1. Gemma 4 E4B prefill under the handoff threshold

- **Why:** a 19-token prompt prefills at 3.8 tok/s, down from 9.3 on 2026-10-11. The 500-token Vulkan prefill is fine (59.3 tok/s), so the batched path works. The 19-token case goes through the per-token path, which is not explained.
- **First step:** check the handoff threshold default (32 tokens) against the 19-token prompt, and time the per-token path on the 2026-10-11 build against today's.
- **Done when:** the drop is explained (threshold, or a regression with its fix), and the result is recorded.

### A2. DeepSeek-V2-Lite Q8_0 prefill at 0.49x

- **Why:** prefill is 28.0 tok/s against llama.cpp's 56.9 (0.49x). Decode is 13.6 against 14.8 (0.92x). The prefill-only gap suggests a batched-prefill problem.
- **Checkpoint:** `K:\_other_models\DeepSeek-V2-Lite-Chat.Q8_0.gguf`.
- **First step:** find the path that prefill takes that decode does not. Note that this checkpoint's greedy output is already known to be wrong, so compare token for token against a run from before the change, not by reading the transcript.
- **Done when:** the prefill ratio is re-measured after the change, the output is equal to the pre-change run, and the result is recorded.

### A3. OLMoE decode at 0.81x

- **Why:** decode 0.81x and prefill about 0.8x of llama.cpp (re-measured 2026-10-30). The old 0.50x figure is stale.
- **Checkpoint:** `F:\_models\OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf`.
- **First step:** re-measure against llama.cpp with the current build. If decode is at or above 0.9x, close this item with the number. If not, profile decode for MoE routing and expert dispatch.
- **Done when:** the item is closed with a number, or a profile names the cost to attack.

### A4. Single-row Q8_K GEMV for decode

- **Why:** Q8_K decode is about 3x faster per thread than Q8_KS, but 5-7% slower at 16 threads, so it is off by default. A single-row GEMV in the repacked GEMM (`RepackedGemm.GemvQ4K8x8Q8K`, exists at `RepackedGemm.cs:249`) could remove that loss.
- **First step:** check whether any decode path calls `GemvQ4K8x8Q8K`. If none does, wire it behind the existing Q8_K decode switch.
- **Done when:** Q8_K decode is measured at 16 threads against Q8_KS, with equal output. Either it wins and the default changes, or the result is recorded as not worth it.

### A5. Row-interleaved Q8_0 repack (prefill)

- **Why:** the survey's top-ranked prefill idea after tinyBLAS failed. Layout used by ik_llama.cpp (`_R4`, `_R8`).
- **First step:** a kernel microbenchmark of the new layout against the current repacked Q8_0 layout, before any engine change. Only continue if the microbenchmark gains.
- **Done when:** the engine A/B is recorded, or the microbenchmark result says not worth it.

### A6. Small-model decode (SmolLM2-135M, Qwen2.5-0.5B)

- **Why:** the two plans disagree. The pickup says about 0.55-0.75x; the sweep says 0.125-0.31x at 135M-500M. Per-token overhead (thread dispatch) is the stated cause. The spin-then-park worker pool is built but opt-in.
- **First step:** re-measure both models against llama.cpp with the current build. Whichever number holds becomes the figure in PerformanceLeague.md.
- **Done when:** the current ratio is recorded, and the spin pool's effect at these sizes is measured or the item is closed as not worth it.

### A7. Allocation leftovers (conditional)

- **Why:** per-call `float[]` allocations in `LinearNoBias` and `RmsNormRow` (sweep 1.5), and a conv allocation and `Parallel.For` per call in the pointwise kernel (sweep 11.3, reverted once because it was not faster warm).
- **First step:** measure whether allocation is a visible cost on the current CPU paths. If it is not, record "not worth it" with the measurement and close both items.
- **Done when:** both are closed with a measurement. If allocation is visible, re-benchmark end to end (three runs, the 14.1 s clip) before closing.

### A8. Audio generation real-time factor (ACE-Step Turbo at 114x, Stable Audio 3 Medium at 97x)

- **Why:** the worst real-time factors in the sweep. ACE-Step Turbo is the worst ratio in the PerformanceLeague TTS section.
- **First step:** check the checkpoints are on disk. Then profile one ACE-Step generation to see where the time goes. Plateau rule: if ACE-Step has no quick win, move to Stable Audio 3 Medium with the same approach.
- **Done when:** a measured change is recorded for one model, or both are recorded as plateaued with the profile.

### A9. Vision-language decode at about 0.48x

- **Why:** InternVL3-2B, Granite-4.0-3B-Vision and Granite-Vision-3.2-2B all decode at about 0.48x. The sweep calls this a real pattern, not noise.
- **First step:** re-measure one checkpoint (Granite-Vision-3.2-2B Q3_K_S) against llama.cpp. The sweep's numbers are from before several later changes.
- **Done when:** the current ratio is recorded, and either a cause is named or the item is closed as plateaued.

### A10. Expert GEMM for CPU MoE prefill, step 0 (the NLL gate)

- **Why:** CPU MoE prefill has no expert GEMM kernel. Step 0 of the expert plan measures the cost of an approximate version and touches no engine code.
- **First step:** run the NLL and greedy-token comparison from the plan's step 0 on OLMoE and the other checkpoints the plan names. Record the result. The decision it informs is in Part B, decision 2.
- **Done when:** step 0 is recorded in the expert plan, with its numbers.

---

## Part B: Decisions

1. **Unpause performance.** Owner. Until then Part A is not started.
2. **Expert GEMM: option A or option B.** Option A (repack each expert into scratch and reuse the repacked GEMM) is faster to build but not bit-identical to the per-token path. Option B (exact tiled GEMM) keeps bit-identical output and needs more code. Decide after A10. Full options: [2026-10-04-q4k-expert-gemm-plan.md](2026-10-04-q4k-expert-gemm-plan.md), section 7.
3. **FP16-tile GEMM for the diffusion and Vulkan paths (the largest upside left, sweep Phase 3 and the diffusion FP16 project).** It carries a numerics decision. Make the decision before any iGPU measurement, because the measurement can only show the iGPU result (rule 13). Owner doc: [gpu/vulkan-gemm/vulkan-gemm.md](gpu/vulkan-gemm/vulkan-gemm.md), Part 2.
4. **Whether MoE prefill on the Vulkan hybrid is worth an iGPU measurement.** The CPU handoff already covers dense `llama`; MoE is still per-token on the GPU path. Decide whether to measure it at all on this machine, given rule 13.

---

## Part C: Blocked

### C1. Need a discrete GPU (rule 13)

- **Vulkan iGPU diffusion timings (sweep and PerformanceLeague rows, 2026-09):** SD 1.5 at 0.47x, FLUX.1-schnell at 0.51x, SD 3.5 Medium at 0.53x, SDXL-Turbo at 0.63x. Owner doc: [diffusion/diffusion-performance.md](diffusion/diffusion-performance.md), 19 open boxes. The leading lever is quantised matmul for the UNet and DiT GEMMs.
- **FLUX.2 GPU** (3,260 s for one 512x512 image) and the remaining image samples (Qwen Image, Wan 2.2, HunyuanVideo, FLUX.2-dev). Same owner doc.
- **Vulkan raw-quant matvec kernels:** Q3_K at 16 against ggml's 29 GB/s, Q6_K and Q5_K about 1.2x behind, MXFP4 at 25 against 35.9 GB/s. Owner doc: [gpu/vulkan-gemm/vulkan-gemm.md](gpu/vulkan-gemm/vulkan-gemm.md). Q3_K needs a cooperative scale-cache structure; parked.
- **Vulkan GEMM phases:** Phase 0 harness (partial), five-run median, Phase 3 tiled kernel family, Phase 4 fusions, Phase 5 propagation. Owner doc.
- **Audio GPU residency:** 31 boxes in [audio/080-tts-asr-gpu-residency-checklist.md](audio/080-tts-asr-gpu-residency-checklist.md). Almost all say "confirmed 100% CPU", which is a finding about this machine. The residency work needs a discrete GPU. MiniMax-Music3 flow transformer, 7 steps in [audio/079-minimax-music3-gpu-residency-plan.md](audio/079-minimax-music3-gpu-residency-plan.md).
- **Vulkan iGPU dispatch overhead for small models (sweep Phase 4):** a real pattern on this machine, but the fix can only be measured on a discrete GPU.
- **SDXL UNet attention batching (sweep Phase 13):** the sweep itself says this needs a discrete GPU to re-test the premise. Do not re-try the two failed attempts.
- **Diffusion-side mat-vec kernels (lead 6, parked)** and the per-op timings for Wan 2.2, HunyuanVideo and FLUX.2 (deferred until the leads above are done).
- **Small-batch decode amortisation for the Phase 2 Qwen3.6 family (sweep 2.3b):** not attempted. It is real architectural work: batching more than one token through a decode path that only handles one token at a time. The sweep declared Phase 2 at a plateau on 2026-09-12. Start only with an owner decision to reopen the plateau.

### C2. Checkpoint not on any drive

- **gpt-oss-20b MXFP4** (per-token prefill on both CPU and Vulkan, 10.1 and 7.8 tok/s): not on any drive. Download only with owner approval for the size.

### C3. Closed or plateaued (no action unless the owner asks)

- Voxtral-Mini-4B (sweep Phase 1): plateau declared 2026-09-12 at 0.221x of the first port's 0.02x.
- Qwen3.6-35B-A3B prefill (row 6): 0.97x on 2026-11-01.
- Gemma family prefill (sweep Phase 6): stale, effectively closed. Gemma 4 uses the batched path. The default int8 prefill is 0.95x for Gemma 3 4B and 0.77x for Gemma 4 E4B (2026-11-01).
- Q5_K_M off the batched Vulkan trunk: closed 2026-10-31.
- Long-context CPU decode: re-measured at 0.93-0.98x on 2026-11-01. The old 0.67x does not reproduce.
- MTP on Qwen3.6-27B UD-Q3_K_XL: fixed 2026-11-03 (about 1.05x of plain).
- One-row Q4_K decode kernel: tried and reverted (3% slower, byte-identical output).
- Weight-stationary prefill loop reorder: tried, 4x slower.
- OpenBLAS: never faster than the in-house kernels.
- Path 3 tiled quantised GEMM for Vulkan prefill: the default since 2026-10-11.
- Decode attention on Vulkan (leads 2b): shipped 2026-10-11.

---

## Owner docs and their status

Four documents hold the detail for the blocked work and are kept as they are. They are owner docs for their threads, and their open boxes are listed in Part C above:

- [diffusion/diffusion-performance.md](diffusion/diffusion-performance.md)
- [gpu/vulkan-gemm/vulkan-gemm.md](gpu/vulkan-gemm/vulkan-gemm.md)
- [audio/080-tts-asr-gpu-residency-checklist.md](audio/080-tts-asr-gpu-residency-checklist.md)
- [audio/079-minimax-music3-gpu-residency-plan.md](audio/079-minimax-music3-gpu-residency-plan.md)

The expert GEMM plan ([2026-10-04-q4k-expert-gemm-plan.md](2026-10-04-q4k-expert-gemm-plan.md)) stays as the detail for Part A10 and Part B decision 2.

## Already tried; do not repeat blind

| Idea | Result |
|---|---|
| llamafile tinyBLAS register tiles (4x2, 2x4) for Q8_0 prefill | No gain; reverted |
| ik_llama.cpp's Q4_K sub-block reduction in the Q8_KS kernels | +3-8%, but output changed; reverted |
| One-row Q4_K decode kernel instead of two-row | 3% slower in all six interleaved pairs; byte-identical output |
| Q8_K decode at all thread counts | 5-7% slower at 16 threads; off by default, used at one thread |
| Weight-stationary loop reorder for prefill | 4x slower |
| OpenBLAS | Never faster than the in-house kernels |
| Naive and tiled GPU attention shaders for SDXL UNet | Both measured and root-caused as failures; do not retry |
| Per-call GPU residency for SDXL attention | Failed twice; see sweep Phase 13 |

## Standing claims that need care

- **ik_llama.cpp as a whole** is slower than llama.cpp for decode on this CPU (3.1 vs 4.7 t/s, one thread). The author's own pp512 table shows it ahead across quants on a Ryzen 7950X, and the ISA there is unclear. Do not generalise either result.
- **Vulkan results on this iGPU** do not describe a discrete GPU (rule 13).
