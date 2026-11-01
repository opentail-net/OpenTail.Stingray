# Performance work: where to pick up (paused 2026-10-30)

Performance work was paused on 2026-10-30 at the owner's request. This page is the starting point for resuming it: what
is left, ranked; what was already tried; and how to measure without repeating anything. The numbers are in
[PerformanceLeague.md](../../PerformanceLeague.md); background reading is the
[CPU quantised GEMM survey](2026-10-30-cpu-quantized-gemm-survey.md). The Vulkan prefill work is in
[thread owner](gpu/vulkan-gemm/vulkan-gemm.md).

Machine for every figure: Ryzen 7 5700G (Zen 3, 8 cores / 16 threads, AVX2 + FMA, no VNNI, no AVX-512), DDR4, integrated
Radeon (Vulkan, shares system RAM). "Ratio" below means Stingray speed / reference speed (1.0x = parity, lower is slower).

## State on 2026-10-30

- **Text models, CPU: mixed, not uniformly close to llama.cpp.** The 2026-10-11 recheck (PerformanceLeague, "Findings
  from the 2026-10-11 sweep") estimates against stored llama.cpp numbers: CPU prefill median **0.93x** (15 of 25 comparable
  rows at 0.9x or better), CPU decode median **0.69x** (1 of 23 at 0.9x or better). Mistral-7B decode is the close case:
  8.8 vs 8.6-9.45 t/s at the default thread count (physical cores = 8). Q8_0 prefill is at or above llama.cpp after #25.
  One-thread decode switches to Q8_K automatically and beats llama.cpp (#30). The owner judged decode "close enough".
  The recheck says decode is not close across most of the board.
- **Vulkan LLM prefill is now usually ahead of CPU prefill.** At about 500 tokens (2026-10-11, same run):
  Qwen3-4B 108 vs 88 tok/s, Qwen3-8B 60 vs 50, Mistral-7B 57 vs 48. The gain is the Path 3 tiled quantised GEMM, which
  has been the **default since 2026-10-11** (`STINGRAY_VULKAN_MM_PATH=1` rolls back; `VulkanMatMulPath.cs`). Qwen3-4B
  prefill on a 493-token prompt went from 38.4 to 71.3 tok/s (the Vulkan leads log has the detail).
- **Partially offloaded dense `llama` models (`-g N`) now prefill through the CPU handoff (2026-10-31).** Before, the hybrid refused the handoff for them and prefilled token by token: SmolLM2-1.7B `-g 12` 18 tok/s, Mistral-7B `-g 4` 7.7. With the `llama` receipt: 180 and 44-49 tok/s (3 interleaved pairs each, STATUS row 142 has the parity evidence, the Falcon3 `-g 8` deviation; the one-time 1.9 s first-decode warm-up was moved to model load the same day, PR 73: decode after a handoff went from 7.9 to about 20.5 tok/s at 24 tokens). This is the largest single prefill gain found in the session; other dense architectures on the hybrid have no receipt yet.
- **The big remaining gaps are image diffusion on the iGPU and Voxtral ASR**, each about 2x slower than the C++ reference.
- **Gemma 3 and Gemma 4 prefill is far behind on default F32** (see gap 10 below). Int8 closes most of it.

## Remaining gaps, ranked (latest measurement for each)

The table is the index. The detail for each row lives in its owner plan: rows 1-4 in [the diffusion owner file](diffusion/diffusion-performance.md) (Part 1 umbrella; Parts 2 and 3 for FLUX); row 5 in [perf-sweep Phase 1](perf-sweep-plan.md); row 6 in perf-sweep Phase 2; row 7 in perf-sweep Phase 3; row 10 in perf-sweep Phase 6. Row 11 is owned by the [Vulkan GEMM document](gpu/vulkan-gemm/vulkan-gemm.md) and row 12 by the [batched MoE prefill plan](../2-coverage/2026-10-03-batched-moe-prefill-plan.md). Rows 8 and 9 have no owning plan yet; their evidence is in PerformanceLeague.md.

| # | Area | Workload | Ratio | Measured | Where the time goes / first step |
|---|---|---|---:|---|---|
| 1 | Image diffusion, Vulkan iGPU vs `sd-cli` | SD 1.5 end to end (512x512, 20 steps) | 0.47x (2.14x slower) | 2026-09-16 | UNet denoise GEMMs. sd.cpp uses quantised matmul and wave ops, we use FP16/FP32 tiled GEMMs. **Biggest single lever across all pipelines.** |
| 2 | | FLUX.1-schnell end to end | 0.51x | 2026-09-16 | DiT loop ~1.95x; T5-XXL text encode 0.40x (2.49x slower). See [the diffusion owner file](diffusion/diffusion-performance.md). |
| 3 | | SD 3.5 Medium end to end | 0.53x | 2026-09-16 | MMDiT denoise loop is ~all of the delta. |
| 4 | | SDXL-Turbo end to end | 0.63x | 2026-09-17 | UNet loop 2.32x. The VAE is already faster than C++. |
| 5 | ASR, CPU vs audio.cpp | Voxtral-Mini-4B, 14 s clip | 0.46x | 2026-09-17 | Already 23x faster than the first port; declared a plateau. Profile before touching. |
| 6 | Text, CPU | Qwen3.6-35B-A3B (hybrid GDN + MoE) prefill | 0.97x | 2026-11-01 | Re-measured (was 0.63x on 2026-09-25): 61.45 against 63.45 t/s, Q6_K, 550 tokens, 4 interleaved rounds; MoE 51%, GDN 36%, attention 13% of trunk time. |
| 7 | Text, CPU | Tiny-model decode (SmolLM2-135M, Qwen2.5-0.5B) | ~0.55-0.75x | backlog §4.4a | Per-token overhead dominates (thread dispatch). Spin-then-park worker pool experiment. |
| 8 | Text, CPU | Long-context decode | **Re-measured 2026-11-01: about 0.93-0.98x at ~3.7k context (SmolLM2-1.7B Q4_K_M, 3 runs; llama.cpp tg24 at 3.2k, same day). The 2026-09-10 0.67x does not hold.** Prefill at 3.7k is about 1.4-1.6x llama.cpp (not depth-matched). The cause of the old gap was not investigated and is not known; the 2026-09-10 measurement is no longer reproduced. Taken off the ranked list; re-open only with a matched-depth measurement that shows a gap. | 2026-11-01 (PerformanceLeague, SmolLM2 section) | Depths differ (3.7k vs 3.2k), so the ratio is conservative. |
| 9 | Text, CPU | OLMoE-1B-7B | decode 0.81x, prefill ~0.8x | **2026-10-30** | Re-measured: was 0.50x / 0.69x on 2026-09-10, so the old "OLMoE decode 0.50x" weakness is stale. |
| 10 | Text, CPU | Gemma 3 4B and Gemma 4 E4B prefill | **Re-measured 2026-11-01: the 0.33-0.39x gap was the rollback path, not the default.** The default is int8 batched prefill (ADR-0003, 2026-10-02). Gemma 3 4B default 105.3 t/s = 0.95x of llama-bench pp512 (110.3); rollback (`STINGRAY_CPU_PREFILL_Q8=0`) 43.4 = 0.39x. Gemma 4 E4B default 64.3 = 0.77x of 83.6; rollback 31.0 = 0.37x. Decode 0.89-0.91x for both (no change from the flag). | 2026-11-01 (3 interleaved pairs, 495-token prompt; the 2026-10-02 row is superseded) | The E4B 0.77x is below the Gemma 3 0.95x; the cause is not diagnosed. Decode and prefill are both now close to llama.cpp on CPU. |
| 11 | Text, Vulkan | Q5_K_M models off the batched Vulkan trunk | **Closed 2026-10-31.** SmolLM2-1.7B Q5_K_M prefill 66.8-85.5 t/s vs 84.7-86.0 for its Q4_K_M file (503 tokens, 3 runs each, `-g -1`; the Q4_K_M range is from idle re-runs, the Q5_K_M low run of 66.8 has no known cause) | 2026-10-31 (was 22.3 vs 235 tok/s on 2026-10-11) | The 2026-10-11 figure predates Path 3. `GpuForwardPass.ComputeCanBatchedTrunk` now accepts Q3_K and Q5_K when Path 3 is on (`GpuForwardPass.cs` 1352-1357), and Q5_K_M prefills in the same band as Q4_K_M. The point is "no 10x gap" (about 1.0x), not a precise ratio. Backlog B5b is closed. |
| 12 | Text, Vulkan | Gemma 4 E4B with a prompt under the CPU-prefill handoff threshold (default 32 tokens; the sweep's prompt was 19 tokens) | **3.8 tok/s prefill, re-measured 2026-11-01 (3 runs at 3.7-3.8, plus one more at 3.8), down from 9.3 on 2026-10-11: a real drop, not yet explained.** The 500-token Vulkan prefill is 59.3 t/s (2026-10-11: 55.7), so the batched path is fine; the per-token short-prompt path is what slowed. Decode at 19 tokens is 10.1-10.3 t/s. | 2026-11-01 | Below the threshold (`STINGRAY_HYBRID_CPU_PREFILL_MIN_TOKENS`, default 32, `EngineEnvironment.cs:55`; checked 2026-10-31) the handoff is skipped and the Vulkan path runs per token. The earlier "threshold 19" was the prompt length, not the threshold. Check which path a short Gemma 4 prompt takes before tuning kernels ([batched MoE plan](../2-coverage/2026-10-03-batched-moe-prefill-plan.md), line 220). |

The Vulkan LLM prefill that was behind CPU earlier in the year is no longer behind (see State). Vulkan results on this
iGPU still do not describe a discrete GPU (CLAUDE.md rule 13).

## Already tried; do not repeat blind

| Idea | Result | Record |
|---|---|---|
| llamafile tinyBLAS register tiles (4x2, 2x4) for Q8_0 prefill | No gain; reverted | survey doc, "Tried" |
| ik_llama.cpp's Q4_K sub-block reduction in our Q8_KS kernels | +3-8% only, and output changed; reverted | this page |
| ik_llama.cpp as a whole | Slower than llama.cpp for decode on this CPU (3.1 vs 4.7 t/s, 1 thread). The author's pp512 table (Ryzen 7950X) shows it ahead across quants; that table is labelled AVX2 but the author says AVX-512 ran when available, so the ISA is unclear. Prefill is not measured here. | this page; [ik_llama discussion #164](https://github.com/ikawrakow/ik_llama.cpp/discussions/164) |
| Q8_K decode at all thread counts | 3x per thread, but 5-7% slower at 16 threads, and less accurate (Q8_KS is ~2x more accurate per dot) | PerformanceLeague "Decode activation format"; used at 1 thread only (#30) |
| One-row Q4_K kernel instead of the two-row kernel in `MatVecQ4K` decode (same pair-of-rows scheduling, two `DotQ4K_Q8KS` calls per work item) | **Loses 3% on both models, in all 6 interleaved pairs, with byte-identical output.** SmolLM2-1.7B Q4_K_M decode 30.7-31.2 t/s vs 31.5-31.9 for the two-row kernel; Qwen2.5-1.5B Q4_K_M 33.4 (all three runs) vs 34.3-34.6. 128 greedy tokens, `-g 0`, idle machine, 2026-10-31. Reverted. Only `MatVecQ4K` was tried; `MatVecDual` interleaves two different matrices and is a separate question. | this page |
| Weight-stationary loop reorder for prefill | 4x slower | comment in `SimdKernels.MatMulBatched` |
| OpenBLAS | Never faster than the in-house kernels | docs/done/openblas-elimination-findings-2026-08-20.md |
| MTP self-speculation on Qwen3.6-27B UD-Q3_K_XL | **Loses**: 0.77-0.81x of plain decode at every draft depth (2026-10-08, "NEGATIVE RESULT" row). The Qwen3.8-27B file wins by +12% on the same engine. Cause not diagnosed. | PerformanceLeague MTP rows; [00-current-work](../00-current-work.md) item 20 |
| Path 3 tiled quantised GEMM for Vulkan prefill | **Worked**, now the default (+86% on Qwen3-4B). Listed here so nobody re-proposes it as new. | [thread owner](gpu/vulkan-gemm/vulkan-gemm.md) |

## Untested leads (cheap first)

- ~~Two-row Q4_K decode kernel is slower per row than the one-row kernel~~ **Tried 2026-10-31: the one-row kernel loses end to end** (see "Already tried"). The microbenchmark gap (38% per row, `tools/kernel-bench-cs q4k-dot`) does not carry over to full decode.
- **Single-row Q8_K GEMV for decode** (`RepackedGemm.GemvQ4K8x8Q8K` exists) instead of the N = 1 GEMM. That would remove
  the 5-7% loss that kept Q8_K decode off at high thread counts.
- **Row-interleaved Q8_0 repack** (ik_llama.cpp `_R4`/`_R8` style), the survey's top-ranked prefill idea after tinyBLAS
  failed.
- ~~Confirm four single-run slowdowns from the 2026-10-11 recheck~~ **Closed: none is a regression.** PerformanceLeague
  (finding 4) already holds a 3-run recheck from 2026-10-11, and a second independent 3-run set on 2026-10-31 (CPU, `-g 0`,
  ~500-token prompt, idle machine) agrees: rwkv-6 1.6B Q8_0 prefill 38.6 / 135.3 / 145.7 tok/s (the first run reads the file
  cold from E:, the old 78.8 was not beaten by a regression), Phi-3-mini prefill 33.5 / 37.9 / 36.8 (0.92x of 40.1), granite-vision
  2B decode 25.2 / 25.0 / 24.3 (0.94-0.97x of 25.9), qwen2.5-1.5b decode 33.3 / 33.2 / 33.1 (above the old 27.5). The
  low single-run figures were noise or cold-file first runs. Prompt lengths here were 502-673 tokens, so ratios are indicative.

## How to measure (rules that held up)

- Measurement rules (one run at a time, interleaved A/B, output equality, named models): see [../reference/performance-measurement-rules.md](../reference/performance-measurement-rules.md).
- Reference: `tools/llama.cpp` (b10306): `llama-bench -p N -n M -t 8 -ngl 0 -r 2`. Kernel-level comparison against
  ggml's own `vec_dot`: `DOTNET_TieredCompilation=0 tools/kernel-bench-cs/... q4k-dot <tools/llama.cpp dir>`.
- Profilers: `STINGRAY_PROFILE_PREFILL=1` (CPU prefill stages), `STINGRAY_PROFILE_DECODE=1` (decode stages),
  `STINGRAY_PREFILL_PROFILE=1` (CUDA passes; a different name for the CUDA path).
- Default CPU thread count is the number of physical cores (8 here), not logical processors.

---

## Research findings (2026-10-30)

Checked in code on 2026-10-30. Nothing was measured in this pass.

- **Row 11 (Q5_K off the batched Vulkan trunk) may be stale.** The 22.3 tok/s figure is from 2026-10-11, the same day Path 3 became the default. Path 3's `SgemmQuant` covers Q5_K (leads plan, 2026-10-11 Path 3 entry). Re-measure with `STINGRAY_VULKAN_MM_PATH=3` (the default) before treating this as a gap.
- **Path 2 is live for Q4_K and Q6_K** (`VulkanBackend.cs` 3177-3183: it returns false for other dtypes, for `nTok > TiledQ4KBn`, and for `cols % 256 != 0`). The Vulkan GEMM doc says Path 2 "declines every shape today". That was true when written and is not true now. Path 3 is still the default, so routing is unchanged.
- **Lead "two-row Q4_K decode kernel is ~38% slower":** the two-row structure is visible in `SimdKernels.cs` (`MatVecQ4K` uses "dynamically balanced 2-row chunks", as stated in the `MatVecDual` comment near line 1168). I did not verify the bit-identical claim or the 38% figure. The lead's own rule (measure end to end) still applies.
- **Lead "single-row Q8_K GEMV"** cites `RepackedGemm.GemvQ4K8x8Q8K`. Checked 2026-10-31: `GemvQ4K8x8Q8K` exists (`RepackedGemm.cs:249`, AVX2 and scalar variants, used by the 16-token batched wrapper at line 204), so the citation is right; whether a decode path could call it was not checked.
- **Row 12:** re-checked 2026-10-31: the threshold is 32, and 19 was the sweep's prompt length (row text corrected). **Rows 8-9** were not re-checked.

**Next checks, in order:** re-measure row 11 on Path 3; A/B the one-row `MatVecQ4K` against the two-row version with the existing equality test; confirm `GemvQ4K8x8Q8K` is wired and cited correctly.

---

## Research findings (2026-10-31, deep pass)

Excerpts and external references are in [research/2026-10-31-performance-snippets.md](research/2026-10-31-performance-snippets.md).

- **"ik_llama.cpp as a whole is slower than llama.cpp on this AVX2 CPU" (table row) needs qualifying.** The pickup's evidence is decode (3.1 vs 4.7 t/s, 1 thread). The author's pp512 table is a different workload, and it shows ik_llama faster across quants: Q8_0 1.81x, Q4_K_S 2.49x, Q5_K_S 3.37x ([ik_llama discussion #164](https://github.com/ikawrakow/ik_llama.cpp/discussions/164)). Those numbers are on a Ryzen 7950X, labelled AVX2 but with the author saying AVX-512 was used when available. So the pickup's decode number does not contradict the author's prompt numbers, and the ISA is unclear. Keep the row, but say which workload and which ISA.
- **Row 11 (Q5_K off the batched Vulkan trunk).** No new external source. The internal finding stands: re-measure on Path 3.
- **Lead "Q8_K decode", and the "two-row Q4_K kernel 38% slower" lead.** External references do not settle either. The author's table confirms that Q4_K-family kernels have large headroom on the author's machine, but that is a different CPU.
- **Web research summary:** one external source (ik_llama.cpp) was checked against a row in this doc. The rest of the pickup's numbers are internal measurements.
