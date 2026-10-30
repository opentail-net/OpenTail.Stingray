# Performance work: where to pick up (paused 2026-10-30)

Performance work was paused on 2026-10-30 at the owner's request. This page is the starting point for resuming it: what
is left, ranked; what was already tried; and how to measure without repeating anything. The numbers are in
[PerformanceLeague.md](../../PerformanceLeague.md); background reading is the
[CPU quantised GEMM survey](2026-10-30-cpu-quantized-gemm-survey.md). The Vulkan prefill work is in
[workstreams/vulkan-gemm-llm-prefill.md](../workstreams/vulkan-gemm-llm-prefill.md).

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
- **The big remaining gaps are image diffusion on the iGPU and Voxtral ASR**, each about 2x slower than the C++ reference.
- **Gemma 3 and Gemma 4 prefill is far behind on default F32** (see gap 10 below). Int8 closes most of it.

## Remaining gaps, ranked (latest measurement for each)

The table is the index. The detail for each row lives in its owner plan: rows 1-4 in [diffusion/094](diffusion/094-diffusion-performance-plan.md) (with [069](diffusion/069-flux-vulkan-gemm-perf-handoff.md) and [093](diffusion/093-flux2-gpu-performance-optimization-plan.md) for FLUX); row 5 in [perf-sweep Phase 1](perf-sweep-plan.md); row 6 in perf-sweep Phase 2; row 7 in perf-sweep Phase 3; row 10 in perf-sweep Phase 6. Rows 8, 9, 11 and 12 have no owning plan yet; their evidence is in PerformanceLeague.md.

| # | Area | Workload | Ratio | Measured | Where the time goes / first step |
|---|---|---|---:|---|---|
| 1 | Image diffusion, Vulkan iGPU vs `sd-cli` | SD 1.5 end to end (512x512, 20 steps) | 0.47x (2.14x slower) | 2026-09-16 | UNet denoise GEMMs. sd.cpp uses quantised matmul and wave ops, we use FP16/FP32 tiled GEMMs. **Biggest single lever across all pipelines.** |
| 2 | | FLUX.1-schnell end to end | 0.51x | 2026-09-16 | DiT loop ~1.95x; T5-XXL text encode 0.40x (2.49x slower). See [diffusion/069](diffusion/069-flux-vulkan-gemm-perf-handoff.md), [diffusion/094](diffusion/094-diffusion-performance-plan.md). |
| 3 | | SD 3.5 Medium end to end | 0.53x | 2026-09-16 | MMDiT denoise loop is ~all of the delta. |
| 4 | | SDXL-Turbo end to end | 0.63x | 2026-09-17 | UNet loop 2.32x. The VAE is already faster than C++. |
| 5 | ASR, CPU vs audio.cpp | Voxtral-Mini-4B, 14 s clip | 0.46x | 2026-09-17 | Already 23x faster than the first port; declared a plateau. Profile before touching. |
| 6 | Text, CPU | Qwen3.6-35B-A3B (hybrid GDN + MoE) prefill | 0.63x | 2026-09-25 | GDN recurrence + MoE; see the Qwen3 family rows. |
| 7 | Text, CPU | Tiny-model decode (SmolLM2-135M, Qwen2.5-0.5B) | ~0.55-0.75x | backlog §4.4a | Per-token overhead dominates (thread dispatch). Spin-then-park worker pool experiment. |
| 8 | Text, CPU | Long-context decode | 0.67x at 3.2k (SmolLM2) | 2026-09-10 | Attention cost grows faster than llama.cpp's. **Old: re-measure first.** |
| 9 | Text, CPU | OLMoE-1B-7B | decode 0.81x, prefill ~0.8x | **2026-10-30** | Re-measured: was 0.50x / 0.69x on 2026-09-10, so the old "OLMoE decode 0.50x" weakness is stale. |
| 10 | Text, CPU | Gemma 3 4B and Gemma 4 prefill, **default F32** | 0.33-0.39x | 2026-10-02 (commit `a6d5583e`) | With int8 the same models run at 0.95-0.97x, so the gap is the F32 default, not the kernels. Decide what the default should be for these models before any kernel work. Decode is 0.84-0.85x. |
| 11 | Text, Vulkan | Q5_K_M models off the batched Vulkan trunk | SmolLM2-1.7B Q5_K_M prefill 22.3 tok/s vs 235 for its Q4_K_M file | 2026-10-11 | Only Q4_K and Q6_K take the batched trunk; other K-quants fall back. Backlog B5b. |
| 12 | Text, Vulkan | Gemma 4 with a prompt under the CPU-prefill handoff threshold (19 tokens) | 9.3 tok/s prefill | 2026-10-11 | Below the handoff threshold the Vulkan path runs per token. Check the threshold before tuning kernels. |

The Vulkan LLM prefill that was behind CPU earlier in the year is no longer behind (see State). Vulkan results on this
iGPU still do not describe a discrete GPU (CLAUDE.md rule 13).

## Already tried; do not repeat blind

| Idea | Result | Record |
|---|---|---|
| llamafile tinyBLAS register tiles (4x2, 2x4) for Q8_0 prefill | No gain; reverted | survey doc, "Tried" |
| ik_llama.cpp's Q4_K sub-block reduction in our Q8_KS kernels | +3-8% only, and output changed; reverted | this page |
| ik_llama.cpp as a whole | Slower than llama.cpp on this AVX2 CPU (decode 3.1 vs 4.7 t/s, 1 thread). Its gains are on Zen 4 / AVX-512. | this page |
| Q8_K decode at all thread counts | 3x per thread, but 5-7% slower at 16 threads, and less accurate (Q8_KS is ~2x more accurate per dot) | PerformanceLeague "Decode activation format"; used at 1 thread only (#30) |
| Weight-stationary loop reorder for prefill | 4x slower | comment in `SimdKernels.MatMulBatched` |
| OpenBLAS | Never faster than the in-house kernels | docs/done/openblas-elimination-findings-2026-08-20.md |
| MTP self-speculation on Qwen3.6-27B UD-Q3_K_XL | **Loses**: 0.77-0.81x of plain decode at every draft depth (2026-10-08, "NEGATIVE RESULT" row). The Qwen3.8-27B file wins by +12% on the same engine. Cause not diagnosed. | PerformanceLeague MTP rows; [00-current-work](../00-current-work.md) item 20 |
| Path 3 tiled quantised GEMM for Vulkan prefill | **Worked**, now the default (+86% on Qwen3-4B). Listed here so nobody re-proposes it as new. | [workstreams/vulkan-gemm-llm-prefill.md](../workstreams/vulkan-gemm-llm-prefill.md) |

## Untested leads (cheap first)

- **Two-row Q4_K decode kernel is ~38% slower per row than the one-row kernel** on DRAM-sized matrices
  (`tools/kernel-bench-cs q4k-dot`). They are bit-identical, so switching `MatVecQ4K` / `MatVecDual` to the one-row kernel
  cannot change outputs. Measure end to end before keeping it.
- **Single-row Q8_K GEMV for decode** (`RepackedGemm.GemvQ4K8x8Q8K` exists) instead of the N = 1 GEMM. That would remove
  the 5-7% loss that kept Q8_K decode off at high thread counts.
- **Row-interleaved Q8_0 repack** (ik_llama.cpp `_R4`/`_R8` style), the survey's top-ranked prefill idea after tinyBLAS
  failed.
- **Confirm four single-run slowdowns from the 2026-10-11 recheck before acting** (each was one run; confirm with 3):
  rwkv-6 1.6B prefill 78.8 to 49.1 tok/s (0.62x), Phi-3-mini prefill 40.1 to 27.5 (0.69x; this file already has a closed
  Phi-3 regression note), granite-vision 2B (Q3_K_S) decode 25.9 to 19.5 (0.75x), qwen2.5-1.5b decode 27.5 to 22.8 (0.83x).

## How to measure (rules that held up)

- Measurement rules (one run at a time, interleaved A/B, output equality, named models): see [../reference/performance-measurement-rules.md](../reference/performance-measurement-rules.md).
- Reference: `tools/llama.cpp` (b10306): `llama-bench -p N -n M -t 8 -ngl 0 -r 2`. Kernel-level comparison against
  ggml's own `vec_dot`: `DOTNET_TieredCompilation=0 tools/kernel-bench-cs/... q4k-dot <tools/llama.cpp dir>`.
- Profilers: `STINGRAY_PROFILE_PREFILL=1` (CPU prefill stages), `STINGRAY_PROFILE_DECODE=1` (decode stages),
  `STINGRAY_PREFILL_PROFILE=1` (CUDA passes; a different name for the CUDA path).
- Default CPU thread count is the number of physical cores (8 here), not logical processors.
