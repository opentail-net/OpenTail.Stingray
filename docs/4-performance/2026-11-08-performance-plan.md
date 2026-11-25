# Performance plan
> **OpenBLAS: never use it. It must not be loaded, run, measured or enabled in OpenTail.Stingray (owner directive 2026-11-25; see CLAUDE.md). The opt-in gate in BlasInterop.cs stays as a guard.**

> **REOPENED 2026-11-24 by the owner.** Performance work is active again ("fair game, as long as it achieves something"). Part A is open and starts with the items in order; the work is recorded per item here and in [PerformanceLeague-LOG.md](PerformanceLeague-LOG.md). The 2026-11-24 prefill decision closes only prefill speed on the default route, not performance as a whole.

Status 2026-11-08. Performance work was paused by the owner on 2026-10-30; it was reopened on 2026-11-24. This is the one plan for performance. It replaces the pickup guide, the leads plan and the perf sweep, which are now in [done/](../done) as history. Numbers are in [PerformanceLeague.md](../../PerformanceLeague.md).

**Machine.** Ryzen 7 5700G (Zen 3, 8 cores, 16 threads, AVX2 and FMA, no VNNI, no AVX-512), 64 GB DDR4, integrated Radeon that shares system RAM. "Ratio" means Stingray speed divided by reference speed; 1.0x is parity. Default CPU thread count is the number of physical cores (8).

**Rules.** [performance-measurement-rules](../reference/performance-measurement-rules.md) apply. In short: one run at a time; interleaved A/B; three or more runs; equal output before and after; named models; record every result in PerformanceLeague.md. Rule 13: a Vulkan or CUDA loss on this iGPU is not evidence about a discrete GPU, so GPU results here are never performance claims.

**Start rule.** Part A is open: the owner unpaused performance on 2026-11-24 (Part B, decision 1, is done). Part C is blocked on hardware or on a checkpoint.

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
- **Status 2026-11-24 (second change): open, now 0.82x.** The gate and up sweep was the same kind of loop as the down sweep (one row per dot); it now takes Q4_K rows two at a time through the bit-identical 2-row kernel. Interleaved A/B at 8 threads: +3.9% over 20 pairs, text identical in 20 of 20. Final default route, 5 runs: 42.66 t/s, ratio 0.82. The next costs are the non-MoE part of each token: the QKV projections (17.5% of decode) and the output projection with the LM head (13%). Record: [PerformanceLeague-LOG.md](PerformanceLeague-LOG.md), second part of "OLMoE CPU decode".
- **Status 2026-11-24: open, improved to 0.81x.** Re-measured: llama.cpp tg128 at 8 threads 51.96 ± 2.18 t/s (5 runs); Stingray 39.58 before the change (ratio 0.76), 42.00 after (ratio 0.81; 5 runs). Profile: the MoE down sweep was 40% of MoE time and ran at about 23 GB/s against 34 GB/s for the gate and up sweep. Change: the down dots are one parallel region over (expert, row pair) with the 2-row kernel, then the same k-ordered reduction. Text identical in 16 of 16 A/B pairs; interleaved A/B at 8 threads +5.0% (39.16 to 41.11, 10 pairs each); the 16-thread case is a tie. The 0.9 target is not reached. Next cost named: the gate and up sweep (54% of MoE time) and the non-MoE part of each token. Record: [PerformanceLeague-LOG.md](PerformanceLeague-LOG.md), section "OLMoE CPU decode".

### A4. Single-row Q8_K GEMV for decode

- **Why:** Q8_K decode is about 3x faster per thread than Q8_KS, but 5-7% slower at 16 threads, so it is off by default. A single-row GEMV in the repacked GEMM (`RepackedGemm.GemvQ4K8x8Q8K`, exists at `RepackedGemm.cs:249`) could remove that loss.
- **First step:** check whether any decode path calls `GemvQ4K8x8Q8K`. If none does, wire it behind the existing Q8_K decode switch.
- **Done when:** Q8_K decode is measured at 16 threads against Q8_KS, with equal output. Either it wins and the default changes, or the result is recorded as not worth it.

### A5. Row-interleaved Q8_0 repack (prefill)

- **Why:** the survey's top-ranked prefill idea after tinyBLAS failed. Layout used by ik_llama.cpp (`_R4`, `_R8`).
- **First step:** a kernel microbenchmark of the new layout against the current repacked Q8_0 layout, before any engine change. Only continue if the microbenchmark gains.
- **Done when:** the engine A/B is recorded, or the microbenchmark result says not worth it.

### A6. Small-model decode (SmolLM2-135M, Qwen2.5-0.5B)

- **Status 2026-11-24: open, re-measured.** Qwen2.5-0.5B Q4_K_M at 8 threads: Stingray 75.48 t/s against llama.cpp 97.25 ± 1.23, ratio 0.78 (the old 0.125 to 0.31 figure is stale). The opt-in spin pool is slower here (68.7 against 74.7 t/s, six pairs; not adopted). Thread scaling flattens from 4 to 8 threads. The fork-join hypothesis is weak: `Parallel.For` with 8 threads costs about 2 µs empty and about 5 µs of overhead with real work (scratch measurement), which is 4 to 7% of the token, not 14%. Per-kernel decomposition (iteration 54, scratch harness, not in the repo): this checkpoint is not Q4_K-dominated. Attention q, k and o and FFN gate and up are Q5_0; attention v and the LM head (`output.weight`) are Q8_0; FFN down is Q6_K. The matvec calls in a token sum to about 12 ms of the 13.2 ms token. Two costs make up the gap: a per-call overhead of about 5 µs at 8 threads (about 170 calls per token, about 0.8 ms or 6%), and a streaming slope of about 28 to 33 GB/s for the Q5_0 and Q6_K calls rotated past the L3, against 38 to 44 GB/s for the streaming probe. Fork-join is not the cause. The earlier readings of 1.9 GB/s for Q5_0 and 318 µs for a 896-row call did not reproduce on a single buffer and are retracted; the 318 µs figure is a rotation artifact of the harness, since decode cannot take 15 ms in those calls. Next: a 2-row Q5_0 kernel with the same per-row arithmetic, measured by interleaved A/B on the rotated FFN shape, and fusing the gate and up dispatches to cut per-call overhead. Iteration 55: the 2-row harness is bit-identical and about 8% faster single-thread, about 16% faster at 8 threads cache-warm; the rotated minimum does not move (see PerformanceLeague-LOG.md, "2-row Q5_0 dot"). Implemented and measured end to end in iteration 55: bit-identical, text identical in 12 runs, but about 3.5% slower in six interleaved pairs (FFN about +5% in the profile). Reverted and not adopted (PerformanceLeague-LOG.md, "End-to-end A/B: NOT ADOPTED"). Iteration 56, separated: pair dispatch with the 1-row kernel is neutral (+0.3%); the 2-row kernel is the cost in the product (-2.7%; -6.4% with 8 pairs per item). The 2-row kernel stays out (PerformanceLeague-LOG.md, "Q5_0 dispatch versus kernel, separated"). Iteration 57: the arithmetic matches ggml's Q5_0 AVX2 kernel; the difference is the per-block fp16 scale conversion (ggml uses F16C; we use a software conversion). The conversion is about a quarter of the Q5_0 dot (bound), but F16C is not reachable from managed code here, and a vectorised gather conversion is bit-identical and 2.3 times slower. Iteration 58 (PerformanceLeague-LOG.md, "Scale layout for the Q5_0 dot"): a repacked 24-byte block with the float32 scale in place (+9% Q5_0 bytes) is bit-identical to the control and faster in all four scenarios in both runs (single thread about -24 to -64%; the magnitude is not stable on this machine, but the sign is). A separate float array is no better and costs more bytes. Owner decision (q): whether to take the repacked layout (about +23 MB streamed per token and a resident-memory trade for Qwen 0.5B). No product change. Open: the fused Q5_0 gate and up case in MatVecDual (about 1%, not yet measured); the token-level A/B for the repacked Q5_0 layout is done: +4.6% decode (median 61.0 against 58.3 t/s; higher in 8 of 8 interleaved rounds; PerformanceLeague-LOG.md, "Repacked Q5_0 layout: token-level A/B"). No product change; merging the layout and its memory cost are the owner's decision (q). Record: [PerformanceLeague-LOG.md](PerformanceLeague-LOG.md), "Small-model CPU decode".
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

### A11. Q4_K decode row kernel: 2-row versus 1-row on the Q8_KS path (opened 2026-11-24)

- **Why:** `MatVecQ4K`'s parallel fast path dispatches row pairs to `DotQ4K_Q8KS_2Row`, and its single-thread path does the same. The 2026-10-30 findings (pickup guide, now in done/) measured the 2-row kernel about 38% slower per row than the 1-row `DotQ4K_Q8KS` on DRAM-sized matrices, and noted the switch to 1-row as a cheap bit-identical lead, untested end to end. Decode is the path that matters on this machine (8 physical threads by default).
- **Claimed:** each row of `DotQ4K_Q8KS_2Row` is bit-identical to `DotQ4K_Q8KS` (the code comments say so). Not yet tested.
- **First step:** (1) per-row microbenchmark with `tools/kernel-bench-cs q4k-dot` (1-row and 2-row against ggml, DRAM-sized shapes), three or more runs; (2) a bit-identity test of the two kernels on random rows; (3) only if (1) and (2) hold: switch the parallel and single-thread fast paths to 1-row, and measure decode on SmolLM2-1.7B Q4_K_M and Qwen2.5-1.5B Q4_K_M at 8 threads, interleaved A/B, equal greedy output.
- **Done when:** the microbenchmark, the bit-identity test and the end-to-end decode numbers are recorded in PerformanceLeague-LOG.md, with the default changed only if it wins on repeat runs with equal output. Otherwise the result is recorded as not worth it.
- **Status 2026-11-24: closed, not worth changing.** Microbenchmark: 1-row is 16 to 19% faster per row at DRAM sizes, not the 38% the 2026-10-30 notes gave; 2-row is slightly faster in cache. Bit identity holds (`DotQ4K2RowBitIdentityTests`, 3 of 3). End to end on the CPU route (SmolLM2-1.7B Q4_K_M, interleaved A/B, identical text in 6 of 6 pairs): 1-row is about 3% slower at the default thread count and about 9% slower at 4 threads, and no different at 16. The default stays on the 2-row kernel and no switch is kept. Full record: [PerformanceLeague-LOG.md](PerformanceLeague-LOG.md), section "Q4_K row kernel, 1-row versus 2-row".

---

## Part B: Decisions

1. **Unpause performance.** Owner. **Done 2026-11-24: the owner unpaused performance** ("fair game, as long as it achieves something"). Part A is open.
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
| Copy-based CPU expert residency (SharpMind-style: private copies of hot MoE experts) | **Discarded 2026-11-09 (owner decision).** Reason: duplicate memory, since copied experts take private RAM on top of the mapped GGUF. Not the same as in-place pinning of mapped pages (`VirtualLock`), which is not discarded and is untested. See [the residency measurement plan](2026-11-09-moe-expert-residency-measurement-plan.md). |

## Standing claims that need care

- **ik_llama.cpp as a whole** is slower than llama.cpp for decode on this CPU (3.1 vs 4.7 t/s, one thread). The author's own pp512 table shows it ahead across quants on a Ryzen 7950X, and the ISA there is unclear. Do not generalise either result.
- **Vulkan results on this iGPU** do not describe a discrete GPU (rule 13).
