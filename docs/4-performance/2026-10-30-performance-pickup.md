# Performance work: where to pick up (paused 2026-10-30)

Performance work was paused on 2026-10-30 at the owner's request. This page is the starting point for resuming it: what
is left, ranked; what was already tried; and how to measure without repeating anything. The numbers are in
[PerformanceLeague.md](../../PerformanceLeague.md); background reading is the
[CPU quantised GEMM survey](2026-10-30-cpu-quantized-gemm-survey.md).

Machine for every figure: Ryzen 7 5700G (Zen 3, 8 cores / 16 threads, AVX2 + FMA, no VNNI, no AVX-512), DDR4, integrated
Radeon (Vulkan, shares system RAM). "Ratio" below means Stingray speed / reference speed (1.0x = parity, lower is slower).

## State on 2026-10-30

- **Text models, CPU, are close to llama.cpp.** Dense 7-12B prefill is at parity. Q8_0 prefill is at or above llama.cpp
  after #25. Decode at the default thread count (physical cores = 8) is within a few percent: Mistral-7B 8.8 vs
  8.6-9.45 t/s. One-thread decode switches to Q8_K automatically and beats llama.cpp (#30). The owner judged decode
  "close enough".
- **The big remaining gaps are image diffusion on the iGPU and Voxtral ASR**, each about 2x slower than the C++ reference.

## Remaining gaps, ranked (latest measurement for each)

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

Not a gap versus a reference, but known: Vulkan LLM prefill loses to CPU prefill on this iGPU (CLAUDE.md rule 13: do not
generalise iGPU results to discrete GPUs).

## Already tried; do not repeat blind

| Idea | Result | Record |
|---|---|---|
| llamafile tinyBLAS register tiles (4x2, 2x4) for Q8_0 prefill | No gain; reverted | survey doc, "Tried" |
| ik_llama.cpp's Q4_K sub-block reduction in our Q8_KS kernels | +3-8% only, and output changed; reverted | this page |
| ik_llama.cpp as a whole | Slower than llama.cpp on this AVX2 CPU (decode 3.1 vs 4.7 t/s, 1 thread). Its gains are on Zen 4 / AVX-512. | this page |
| Q8_K decode at all thread counts | 3x per thread, but 5-7% slower at 16 threads, and less accurate (Q8_KS is ~2x more accurate per dot) | PerformanceLeague "Decode activation format"; used at 1 thread only (#30) |
| Weight-stationary loop reorder for prefill | 4x slower | comment in `SimdKernels.MatMulBatched` |
| OpenBLAS | Never faster than the in-house kernels | docs/done/openblas-elimination-findings-2026-08-20.md |

## Untested leads (cheap first)

- **Two-row Q4_K decode kernel is ~38% slower per row than the one-row kernel** on DRAM-sized matrices
  (`tools/kernel-bench-cs q4k-dot`). They are bit-identical, so switching `MatVecQ4K` / `MatVecDual` to the one-row kernel
  cannot change outputs. Measure end to end before keeping it.
- **Single-row Q8_K GEMV for decode** (`RepackedGemm.GemvQ4K8x8Q8K` exists) instead of the N = 1 GEMM. That would remove
  the 5-7% loss that kept Q8_K decode off at high thread counts.
- **Row-interleaved Q8_0 repack** (ik_llama.cpp `_R4`/`_R8` style), the survey's top-ranked prefill idea after tinyBLAS
  failed.

## How to measure (rules that held up)

- Interleave control and candidate runs (A, B, A, B), at least 2-3 each. Machine load moved llama.cpp's own numbers by
  15-20% between sessions on 2026-10-30, so compare within one session only.
- Save the control build first (copy `src/OpenTail.Stingray.Cli/bin/Release/net10.0` aside), then rebuild the candidate.
- Check output equality (greedy, `--repeat-penalty 1.0`) for every A/B. If numerics change, measure perplexity
  token by token (`stingray perplexity -c 1024 --dump-nll`) against `llama-perplexity` on the same text, and compare
  paired per-token NLL, not just the totals.
- Reference: `tools/llama.cpp` (b10306): `llama-bench -p N -n M -t 8 -ngl 0 -r 2`. Kernel-level comparison against
  ggml's own `vec_dot`: `DOTNET_TieredCompilation=0 tools/kernel-bench-cs/... q4k-dot <tools/llama.cpp dir>`.
- Profilers: `STINGRAY_PROFILE_PREFILL=1` (CPU prefill stages), `STINGRAY_PROFILE_DECODE=1` (decode stages),
  `STINGRAY_PREFILL_PROFILE=1` (CUDA passes; a different name for the CUDA path).
- Default CPU thread count is the number of physical cores (8 here), not logical processors.
