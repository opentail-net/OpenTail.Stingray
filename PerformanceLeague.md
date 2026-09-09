# PerformanceLeague

> **Purpose:** Central store for verified inference performance numbers.
> No optimization rationale, no design discussion — those live in the source documents listed per row.
> **Every row must carry a date-verified.** A number without a date is hearsay.
> **C++ / llama.cpp comparison is mandatory for every inference row.**
> Keep commentary sparse: one short parenthetical per cell if needed; for anything longer, link the source doc.

---

## How to read this file

- **Hardware:** Unless noted, all CPU numbers are from **Ryzen 7 5700G** (Zen 3, 6 physical / 12 logical,
  AVX2 + FMA, no VNNI/AVX-512, measured DRAM ceiling ~36.8 GB/s).
  All Vulkan numbers are from the same machine's **integrated AMD Radeon Graphics** (iGPU, shared DRAM).
  CUDA numbers are from an external NVIDIA GPU (see per-row notes); this box has no CUDA device.
- **llama.cpp reference:** `tools/llama.cpp` b8585-cpu (CPU-only build). GPU comparison uses public
  llama-bench figures where noted.
- **Quant notation:** Q4_K_M = Q4_K mixed sub-block quantization; Q8_0 = 8-bit.
- **Columns:** prefill = prompt-processing t/s; decode = autoregressive generation t/s.
  RTF = wall-clock / audio-seconds (lower is better for TTS/ASR).

---

## 1 — CPU Inference

### SmolLM2 / SmolLM family

| Model | Quant | Prefill (OT) | Prefill (llama.cpp) | OT/LC | Decode (OT) | Decode (llama.cpp) | OT/LC | Date verified | Source |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| SmolLM2-1.7B | Q4_K_M | **34.1 t/s** (warm, JIT-corrected) | 205.0 t/s | 0.17x | **26.5 t/s** | 29.7 t/s | **0.89x** | 2026-07 | perf-loop-progress.md iter 8/11 |
| SmolLM2-1.7B | Q4_K_M | **67.3 t/s** (+Q8 prefill, shipped) | 205.0 t/s | 0.33x | ~26 t/s | 29.7 t/s | ~0.88x | 2026-08 | perf-loop-progress.md iter 38 |
| SmolLM2-1.7B | Q4_K_M | **77.2 t/s** (+Q4Kx8 repack, opt-in) | 205.0 t/s | 0.38x | ~26 t/s | 29.7 t/s | ~0.88x | 2026-08 | perf-loop-progress.md iter 42 |

> **Prefill gap:** llama.cpp 205 t/s uses `block_q4_Kx8` GEMM with integer-domain scale folding and
> 8-row interleave. The OT repack (iter 39-42) begins closing this but is OPT-IN
> (`STINGRAY_Q4KX8_CACHE_MB=<MB>`) pending perplexity gating.
> **Decode:** bandwidth-bound at ~93% DRAM ceiling; the ~1.1x gap is near the hardware floor.

**CPU prefill scaling with context — date verified: 2026-08 (Q8 prefill on, tiled attention, iter 33):**

| Prompt tokens | 267 | 773 | 1621 | 3218 |
|---|---:|---:|---:|---:|
| Prefill OT | ~50 t/s | ~49 t/s | ~49 t/s | ~42 t/s |
| Prefill llama.cpp | 205 t/s | — | — | — |

> Source: perf-loop-progress.md iter 33 (+56% at 3.2k tokens, tiled KV pass).

**CPU decode scaling with context — date verified: 2026-08 (contiguous KV score pass, iter 35):**

| Prompt tokens | 267 | 773 | 1621 | 3218 |
|---|---:|---:|---:|---:|
| Decode OT | 26.3 t/s | 21.0 t/s | 15.0 t/s | 9.5 t/s |

> Source: perf-loop-progress.md iter 35. Contiguous KV score pass (+15-22% at long context) shipped.

---

### Qwen3 family

| Model | Quant | Prefill (OT) | Prefill (llama.cpp) | OT/LC | Decode (OT) | Decode (llama.cpp) | OT/LC | Date verified | Source |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| Qwen3-0.6B | Q8_0 | — | — | — | **47.4 t/s** | — | — | 2026-08-07 | cpu-performance-baseline.md |
| Qwen3-8B | Q4_K_M | not measured | — | — | **6.8 t/s** (34.2 GB/s = 93% DRAM ceil.) | — | — | 2026-08 | cpu-speculative-decoding-findings.md |

> **Qwen3-8B:** At 93% of DRAM ceiling, further decode gains require VNNI-class dot throughput (Zen 4+).

---

### OLMoE family (MoE)

| Model | Quant | Prefill (OT) | Prefill (llama.cpp) | OT/LC | Decode (OT) | Decode (llama.cpp) | OT/LC | Date verified | Source |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| OLMoE-1B-7B | Q4_K_M | 105.6 t/s | — | — | **28.2 t/s** | — | — | 2026-08-07 | cpu-performance-baseline.md |

> Decode measured over 7 tokens (early EOS); treat decode figure as approximate.

---

### Gemma family

| Model | Quant | Prefill (OT) | Prefill (llama.cpp) | OT/LC | Decode (OT) | Decode (llama.cpp) | OT/LC | Date verified | Source |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| Gemma-4-12B | Q4_0 | **3.8 t/s** | — | — | **3.7 t/s** | — | — | 2026-08-07 | cpu-performance-baseline.md |

> Prefill:decode ratio 1.0x = missing batched prefill (`perLayerHdUnsupported` gate). ~5.7x prefill
> penalty beyond what model size explains vs Qwen3-8B. No llama.cpp comparison run on this model.

---

## 2 — Vulkan Inference

> **All Vulkan numbers: integrated AMD Radeon (iGPU), shared DRAM with CPU. NOT representative of a
> discrete GPU.** iGPU bandwidth ceiling measured at 35.5 GB/s.

### SmolLM2 family — Vulkan

| Model | Quant | Metric | Before optimisations | After optimisations | Date verified | Source |
|---|---|---|---:|---:|---|---|
| SmolLM2-1.7B | Q4_K_M | Prefill (43-tok) | 6.55 t/s (per-token loop) | **75.4 t/s** (batched + flash attn) | 2026-08 | perf-loop-progress.md iter 26/29/31 |
| SmolLM2-1.7B | Q4_K_M | Prefill (267-tok) | 6.55 t/s | **53.5 t/s** | 2026-08 | iter 29/31 |
| SmolLM2-1.7B | Q4_K_M | Prefill (3218-tok, default) | 6.4 t/s (SnapKV blocked batched trunk) | **45.9 t/s** | 2026-08 | iter 32 |
| SmolLM2-1.7B | Q4_K_M | Prefill (3218-tok, SnapKV off) | — | **51.7 t/s** | 2026-08 | iter 31 |
| SmolLM2-1.7B | Q4_K_M | Decode (43-tok ctx) | 6.0 t/s (uncoalesced Q4_K matvec) | **24.0 t/s** | 2026-08 | iter 28/28b |
| SmolLM2-1.7B | Q4_K_M | Decode (267-tok ctx) | ~19.7 t/s | **20.2 t/s** | 2026-08 | iter 36 |
| SmolLM2-1.7B | Q4_K_M | Decode (1621-tok ctx) | ~6.0 t/s | **9.7 t/s** | 2026-08 | iter 36 |
| SmolLM2-1.7B | Q4_K_M | Decode (3218-tok ctx) | ~4.1 t/s | **6.4 t/s** | 2026-08 | iter 36 |

**Vulkan vs CPU baseline (SmolLM2-1.7B Q4_K_M, same iGPU machine):**

| Metric | CPU | Vulkan (iGPU) | Date verified | Notes |
|---|---:|---:|---|---|
| Prefill (short prompt, original) | 96.1 t/s | 84.2 t/s | 2026-08-07 | iGPU shares DRAM — not representative of discrete GPU |
| Decode | 23.7 t/s | 24.0 t/s | 2026-08-07 | Dead heat on shared DRAM |

> Source: vulkan-backend-evidence.md. Original per-token Vulkan prefill was 6.55 t/s — ~15x slower
> than CPU. After batching + flash attention the gap inverted at short context.

**Vulkan prefill scaling with context — date verified: 2026-08 (iter 31/32/33):**

| Prompt tokens | 43 | 267 | 773 | 1621 | 3218 (default) |
|---|---:|---:|---:|---:|---:|
| Vulkan prefill OT | ~83 t/s | ~84 t/s | ~78 t/s | ~66 t/s | **45.9 t/s** |
| CPU prefill OT | — | ~50 t/s | ~49 t/s | ~49 t/s | ~42 t/s |

> Source: perf-loop-progress.md iter 31/32/33. No llama.cpp iGPU comparison available.

**Vulkan KV dtype comparison (SmolLM2-1.7B Q4_K_M, 3239-tok, SnapKV off) — date verified: 2026-08 (iter 44/45):**

| KV dtype | Prefill | Decode | Perplexity delta vs fp32 | Date verified |
|---|---:|---:|---|---|
| fp32 | 53.2 t/s | 6.0 t/s | baseline | 2026-08 |
| bf16 | 52.0 t/s (~noise) | **9.4 t/s (+57%)** | +0.023% (negligible) | 2026-08 |
| q8_0 | not measured | ~7.6 t/s | +0.143% | 2026-08 |

> Source: perf-loop-progress.md iter 44/45. bf16 prefill was 2.4x slower before flash variant landed
> (iter 44). bf16 default flip blocked pending --tq and SnapKV compatibility.

**Vulkan Q4_K matvec bandwidth utilisation (single-row, post-optimisation) — date verified: 2026-08 (iter 28b):**

| Shape | Achieved | % of 35.5 GB/s ceiling | Date verified |
|---|---:|---:|---|
| QKV/O 2048x2048 | 19.43 GB/s | 55% | 2026-08 |
| gate/up 8192x2048 | 30.52 GB/s | 86% | 2026-08 |
| down 2048x8192 | 28.98 GB/s | 82% | 2026-08 |
| Q6_K (large shapes) | 31.5–32.3 GB/s | 89–91% | 2026-08 |

> Source: perf-loop-progress.md iter 28b. Q6_K was already at ceiling before any work.

---

## 3 — CUDA Inference

> **No CUDA GPU is present on the development machine. No measured numbers exist.**
> See `GR_performance.md` and `docs/done/gpu-review-log.md` for the code-level audit.

| Model | Quant | Prefill (OT) | Prefill (llama.cpp) | Decode (OT) | Decode (llama.cpp) | Date verified | Notes |
|---|---|---|---|---|---|---|---|
| any | Q4_K | not measured | — | not measured | — | — | No CUDA device on dev machine |
| any | Q6_K / Q5_K | not measured | — | not measured | — | — | Prefill via dequant+cuBLAS; direct MMQ absent — see GR_performance.md §3 |

> **Highest-confidence CUDA opportunity:** Q6_K and Q5_K prefill dequantize full weight matrix to
> FP16 scratch then invoke cuBLAS — a full-weight HBM round-trip per call. Direct MMQ (as llama.cpp
> does) would eliminate this. Requires real NVIDIA hardware to validate.
> **CUDA current state:** tensor-core flash attention, split-KV decode, grouped GQA reuse, CUDA graphs,
> and int8 MMA matmul are implemented. No shape/arch-aware kernel planner. Monolithic NVRTC compile
> confirmed broken for pre-Ampere (sm < 80); CudaDeviceCaps layer added.

---

## 4 — TTS / Audio Inference (CPU)

> No published llama.cpp TTS pipeline for direct comparison. Figures are absolute RTF only.

| Pipeline | Audio produced | Mean wall-clock | RTF | Date verified | Source |
|---|---:|---:|---:|---|---|
| QwenTTS (Talker + Code Predictor, Qwen3 backbone) | 2.16 s | **14.34 s** | **6.64x slower than RT** | 2026-08-29 | tts-performance-baseline-and-plan.md |
| QwenTTS (after Turn 1 optimization) | 2.16 s | **13.998 s** | **6.48x** | 2026-08-29 | tts-performance-baseline-and-plan.md |
| CosyVoice3 (LLM + flow/DiT + HiFT) | 2.44 s | **21.13 s** | **8.66x slower than RT** | 2026-08-29 | tts-performance-baseline-and-plan.md |

> Harness: `tests/OpenTail.Stingray.Tests.Audio/TtsPerformanceBaselineDebugTest.cs`.
> Prompt: `"Hello, I will make some lunch, darling!"`, seed 42, 1 warmup + 3 timed runs.
> Both pipelines CPU-only; neither wired to Vulkan or CUDA yet.

---

## 5 — Speculative Decoding (CPU)

| Target | Draft | Strategy | Decode (baseline) | Decode (speculative) | Delta | Acceptance | Date verified | Source |
|---|---|---|---:|---:|---:|---:|---|---|
| Qwen3-8B Q4_K_M | Qwen3-0.6B Q8_0 | draft-n 4 | 6.8 t/s | 4.3 t/s | **-37%** | 62% | 2026-08 | cpu-speculative-decoding-findings.md |

> Speculative decoding is a **confirmed loss** on this hardware. Root cause: Q4_K dot is ~87%
> compute-bound (not bandwidth-bound as naively assumed). Verifying k tokens costs ~kx the compute
> regardless of dispatch. VNNI (`vpdpbusd`, Zen 4+) is the prerequisite for speculation to pay.
> No llama.cpp comparison run.

---

## 6 — Vision Encoder (CPU)

| Component | Result | Date verified | Source |
|---|---|---|---|
| VisionOps.Attention / AttentionGqa | **>1.2x** over scalar at 1024-token / 16-head ViT-L scale | 2026-08-20 | perf-loop-project-review-progress.md |

> Scalar reference kept in `VisionOpsBenchmarkTests.cs` as a permanent regression baseline.
> No llama.cpp vision encoder comparison measured.

---

## 7 — Known Measurement Gaps

| Area | Reason | Opportunity |
|---|---|---|
| CUDA inference (all models) | No NVIDIA GPU on dev machine | Q6_K/Q5_K direct prefill MMQ — see GR_performance.md §3 |
| Discrete GPU Vulkan | Only iGPU available | All Vulkan numbers are iGPU/shared DRAM; discrete card expected to show much larger prefill wins |
| Gemma-4-12B prefill (batched) | `perLayerHdUnsupported` gate | ~5.7x penalty over Qwen3-8B size-adjusted; see cpu-performance-baseline.md |
| QwenTTS / CosyVoice3 on GPU | Pipelines not yet wired to CUDA/Vulkan | Expected to be the single largest TTS win when wired |
| Qwen3-8B prefill | Not yet measured | Only decode (6.8 t/s) verified |
| CPU bf16/q8 KV cache | PagedKvCache hard-wired fp32 | Vulkan showed +57% decode at no quality cost; no equivalent on CPU path |

---

*Last updated: 2026-09-09.
Source documents crawled: `docs/done/perf-loop-progress.md`, `docs/cpu-performance-baseline.md`,
`docs/tts-performance-baseline-and-plan.md`, `docs/done/cpu-speculative-decoding-findings.md`,
`docs/done/vulkan-backend-evidence.md`, `docs/done/gpu-review-log.md`, `GR_performance.md`,
`docs/perf-loop-project-review-progress.md`.*


---

## 8 — Prior Hardware / Earlier Codebase Benchmarks

> **Provenance:** These numbers were measured on a **different machine from the current dev box**,
> NOT verified on this repository's primary development hardware. Included for reference and
> comparison leverage, but treat with proportionate caution: the codebase has diverged significantly
> since these were recorded.
>
> **Hardware:** AMD Zen 4 (12c/24t) + RTX 4070 Ti (12 GB VRAM), Windows.
> The CPU-only rows (where stated) are from a separate Zen 4 machine; the RTX 4070 Ti is the
> primary GPU bench target. Models are Q4_K_M unless noted.
>
> **Date measured:** 2026-06-16 (CUDA rows via `scripts/bench-allrows-1k.ps1`; warm clock start,
> 1 discarded warm-up run). Gemma 4 E4B q4_0 Vulkan row freshly measured 2026-06-22.
>
> **This project's own CPU measurements on the same models** appear in section 1 for comparison.

### Measured on this project's own machine (Zen 3 / no GPU)

| Model | Quant | Prefill t/s @3K ctx | Decode t/s | Date verified |
|---|---|---:|---:|---|
| SmolLM2-1.7B-Instruct | Q4_K_M | **130.7 – 134.7** | 25.3 – 27.2 | 2026-07 (git 0c171ed) |
| Qwen3-8B | Q4_K_M | **30.5 – 30.9** | 5.6 – 5.8 | 2026-07 (git 0c171ed) |

> These are from the initial README commit (0c171ed, "fork attribution, verified CPU benchmarks").
> The SmolLM2 prefill figure (130-134 t/s) was measured BEFORE the Q8 prefill gate was default-on
> and before the Q4Kx8 repack; sections 1's current 67-77 t/s reflects a different short-context
> prompt length and JIT-correction methodology. See perf-loop-progress.md for reconciliation.

### Zen 4 + RTX 4070 Ti 12 GB machine — all GPU rows

> SmolLM2 GPU numbers were not published at this revision; see section 2 for current iGPU figures.


#### Gemma 4 E4B-it QAT Q4_0 — 5 GB

| Backend | Prefill t/s | Decode t/s | Date verified | Notes |
|---|---:|---:|---|---|
| **CUDA** `-g -1 -c 2048` | **3666** | **100.4** | 2026-06-16 | QAT q4_0: ~1.4x decode vs Q8. Q4_0 int8 tensor-core MMQ + SoA repack. |
| Vulkan `-g -1 -c 2048` | **35** | **39.5** | 2026-06-22 | Per-token prefill (no batched-prefill path); full trunk incl. PLE + shared-KV tail. |

> llama.cpp reference for Gemma 4 E4B not published.

#### Gemma 4 12B-it QAT Q4_0 — 7 GB

| Backend | Prefill t/s | Decode t/s | Date verified | Notes |
|---|---:|---:|---|---|
| **CUDA** `-g -1 -c 2048` | **1714** | **54.1** | 2026-06-16 | Q4_0 int8 tensor-core MMQ. **Within ~6% of llama.cpp** (57 t/s decode). bf16 KV → 128K ctx in 12 GB. |
| Vulkan `-g -1 -c 2048` | **17.0** | **19.1** | 2026-06-16 | Per-token prefill. |
| CPU | **5.0** | **5.1** | 2026-06-16 | 48-layer dense gemma4 on Zen 4 (12c). |

#### Qwen3-8B — Q4_K_M — 5 GB

| Backend | Prefill t/s | Decode t/s | Date verified | Notes |
|---|---:|---:|---|---|
| **CUDA** `-g -1` | (see §1) | (see §1) | 2026-06-16 | Byte-identical to llama.cpp b8585 under greedy (60-token decode). |

#### Qwen3-Coder 30B-A3B (MoE) — Q4_K_M — 17 GB

| Backend | Prefill t/s | Decode t/s | Date verified | Notes |
|---|---:|---:|---|---|
| **CUDA** `-g -1` (hybrid) | **102.6** | **28.0** | 2026-06-16 | Trunk on GPU, routed experts CPU mmap. Batched CPU-MoE 3.5x over per-token (29.4). |
| CPU `--tq` | 19.6 | 22.6 | 2026-06-16 | 3-bit KV; FastScan → 15.5 t/s decode @3.2K ctx. |
| CPU | 19.8 | 22.4 | 2026-06-16 | 128 experts / 8 active. |
| Vulkan `-g -1` (hybrid) | 1.2 | 4.9 | 2026-06-16 | PCIe/expert-stream bound. 29 GPU + 19 CPU layers. |

#### Carnice (Qwen3.6-35B-A3B-MTP finetune, APEX mixed-precision) — 17 GB

| Backend | Prefill t/s | Decode t/s | Date verified | Notes |
|---|---:|---:|---|---|
| **CUDA** `-g -1 --no-thinking` (hybrid) | **522.0** | **26.5** | 2026-06-16 | GPU MoE op-offload + Q3_K cuBLAS + FlashQLA. 3.6x over CPU MoE (144 t/s). 80% MTP acceptance. |
| Vulkan `-g -1 --no-thinking` (hybrid) | 18.4 | 12.2 | 2026-06-16 | 47% acceptance; MTP regresses vs plain decode (~22 t/s). |

#### Qwen3.6-35B-A3B (GDN+MoE) — Q4_K_M — 22 GB

| Backend | Prefill t/s | Decode t/s | Date verified | Notes |
|---|---:|---:|---|---|
| **CUDA** `-g -1` (hybrid) | **475.4** | **24.5** | 2026-06-16 | GPU op-offload + raw-Q8_0 trunk (4.2x over F32 dequant). |
| Vulkan `-g -1` (hybrid) | 17.3 | 22.8 | 2026-06-16 | Decode ~matches CUDA (CPU-expert bound). |
| CPU | **11.3** | 9.3 | 2026-06-16 | FlashQLA GDN prefill 1.35x over per-token. |

#### Qwen3.6-35B-A3B-MTP (GDN+MoE) — Q4_K_M — 22 GB

| Backend | Prefill t/s | Decode t/s | Date verified | Notes |
|---|---:|---:|---|---|
| **CUDA** `-g -1 --no-thinking` (hybrid) | **480.2** | **33.3** | 2026-06-16 | Raw-Q8_0 trunk 4.2x. MTP ~74% accept. Plain decode ~80% of llama.cpp. |
| Vulkan `-g -1 --no-thinking` (hybrid) | 15.4 | 9.3 | 2026-06-16 | 61% accept; MTP regresses vs ~22 t/s plain decode. |
| CPU `--no-thinking` | 9.1 | **8.5** | 2026-06-16 | MoE-MTP batched verify; expert-sequential, ~MTP-off parity. |

#### Qwen3.6-27B-MTP (GDN dense) — Q4_K_M — 16 GB

| Backend | Prefill t/s | Decode t/s | Date verified | Notes |
|---|---:|---:|---|---|
| **CUDA** `-g -1 --no-thinking` (hybrid) | **22.0** | **12.3** | 2026-06-16 | 84% accept; 1.9x over MTP-off (6.5). |
| **CUDA** `-g -1 --no-thinking` Q5_K_M (hybrid) | 9.6 | **5.5** | 2026-06-16 | 98% accept. |
| Vulkan `-g -1 --no-thinking` (hybrid) | 7.6 | **3.9** | 2026-06-16 | MTP slight loss (3.9 vs 4.9 MTP-off). |
| CPU `--no-thinking` | 3.0 | **3.6** | 2026-06-16 | 90% accept; 1.2x over MTP-off (3.0). |
| CPU `--no-thinking` Q5_K_M | 2.8 | **3.5** | 2026-06-16 | ~10% slower. |

#### Llama-4 Scout 17B-16E (MoE) — Q4_K_M — 61 GB

| Backend | Prefill t/s | Decode t/s | Date verified | Notes |
|---|---:|---:|---|---|
| CPU | 2.1 | 4.3 | 2026-06-16 | 48 layers, 17B active; split GGUF (not on bench machine — smoke run only). |
| CUDA `-g -1` (hybrid) | 1.2 | 2.6 | 2026-06-16 | Model dwarfs 12 GB card; CPU-only wins. Not on bench machine — smoke run only. |

---

*Last updated: 2026-09-09. Sources for section 8: `README.md` git history, commit `0c171ed`
("fork attribution, verified CPU benchmarks"), benchmarks as preserved at that commit.
Hardware: Zen 4 + RTX 4070 Ti. This section should be treated as
**indicative, not authoritative** for the current codebase — re-measure on new hardware to confirm.*
