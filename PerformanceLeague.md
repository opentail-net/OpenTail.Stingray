# PerformanceLeague

> **Purpose:** C# vs C++ inference comparison, by model. The **Ratio** column is the primary
> signal — it shows how close this engine is to its llama.cpp / C++ reference equivalent for each scenario.
> A blank ratio (`—`) means no C++ reference has been measured for that scenario yet.
>
> No optimization rationale, no design history — those belong in `docs/done/perf-loop-progress.md`.
> **Every row carries a Performance Check. A number without a date is hearsay.**

---

## How to read this file

- **Ratio** = C# ÷ C++ reference (llama.cpp, whisper.cpp, etc.). 1.0x = parity. Blank (`—`) = no C++ reference measured.
- **Scenario vocabulary:** `prefill` = prompt-processing; `decode` = autoregressive generation.
  Context length tagged where it materially affects the number (e.g. `decode @3.2k ctx`).
- **RTF** (TTS/ASR) = wall-clock ÷ audio-duration. Lower is better; <1.0x = faster than real-time.
- **TTFA** (TTS Streaming) = Time-To-First-Audio latency in seconds.
- **Hardware (dev machine):** Ryzen 7 5700G, Zen 3, 6c/12t, AVX2+FMA, no VNNI/AVX-512,
  DDR4, DRAM ceiling ~36.8 GB/s. Vulkan = same machine's integrated AMD Radeon (iGPU, 35.5 GB/s).
  No CUDA device on dev machine.
- **† Prior hardware:** Zen 4 (12c/24t) + RTX 4070 Ti 12 GB. Numbers from README history
  (commit `0c171ed`, bench script `scripts/bench-allrows-1k.ps1`, 2026-06-16). Treat as
  indicative — codebase has evolved since.
- **llama.cpp reference:** `tools/llama.cpp` b8585-cpu (CPU). GPU llama-bench where noted.
- **Quant notation:** Q4_K_M = Q4_K mixed; Q8_0 = 8-bit activations.

---

## SmolLM2-1.7B-Instruct

| Scenario | Backend | C# (OT, t/s) | C++ (llama.cpp, t/s) | Ratio | Performance Check | Source |
|---|---|---:|---:|---:|---|---|
| prefill (default, ~267 tok) | CPU | 67.3 t/s | 205 t/s | **0.33x** | 2026-08 | perf-loop-progress.md iter 38 |
| prefill (Q4Kx8 repack, opt-in) | CPU | 77.2 t/s | 205 t/s | **0.38x** | 2026-08 | iter 42; `STINGRAY_Q4KX8_CACHE_MB=<MB>` |
| decode (short ctx) | CPU | 26.5 t/s | 29.7 t/s | **0.89x** | 2026-07 | iter 8/11 |
| prefill (43 tok) | Vulkan iGPU | 75.4 t/s | — | — | 2026-08 | iter 26/29/31; no llama.cpp Vulkan ref |
| prefill (267 tok) | Vulkan iGPU | 53.5 t/s | — | — | 2026-08 | iter 29/31 |
| prefill (3.2k tok, default) | Vulkan iGPU | 45.9 t/s | — | — | 2026-08 | iter 32; SnapKV on |
| decode (short ctx) | Vulkan iGPU | 24.0 t/s | — | — | 2026-08 | iter 28/28b |
| decode @3.2k ctx | Vulkan iGPU | 6.4 t/s | — | — | 2026-08 | iter 36 |
| decode (short ctx) | CPU | 26.5 t/s | 29.7 t/s | **0.89x** | 2026-07 | iter 8/11 |
| decode @3.2k ctx | CPU | 9.5 t/s | — | — | 2026-08 | iter 35 |

> **Prefill gap note:** llama.cpp 205 t/s uses `block_q4_Kx8` GEMM with 8-row interleave and
> integer-domain scale folding. The OT Q4Kx8 repack begins closing this but is opt-in pending
> a perplexity gate. Decode is near-parity because it is bandwidth-bound at ~93% of DRAM ceiling.

**CPU prefill context scaling — Performance Check: 2026-09-10 (llama.cpp backfill; OT: Q8 prefill on, tiled KV, iter 33):**

| Prompt tokens | 267 | 773 | 1621 | 3218 |
|---|---:|---:|---:|---:|
| Prefill OT (t/s) | ~50 t/s | ~49 t/s | ~49 t/s | ~42 t/s |
| Prefill llama.cpp (t/s) | 206.95 t/s | 203.98 t/s | 185.20 t/s | 155.92 t/s |
| Ratio | **0.24x** | **0.24x** | **0.26x** | **0.27x** |

> Backfilled 2026-09-10 via `llama-bench.exe -m SmolLM2-1.7B-Instruct-Q4_K_M.gguf -p 267,773,1621,3218 -n 0 -t 6 -ngl 0 -r 3`. The 0.33x default-prefill ratio at the top of this section holds roughly flat (0.24-0.27x) across the full context range — the gap is not context-length-dependent, consistent with the `block_q4_Kx8` GEMM-interleave explanation already given, not a KV-cache-scaling effect.

**CPU decode context scaling — Performance Check: 2026-09-10 (llama.cpp backfill; OT: contiguous KV score pass, iter 35):**

| Prompt tokens | 267 | 773 | 1621 | 3218 |
|---|---:|---:|---:|---:|
| Decode OT (t/s) | 26.3 t/s | 21.0 t/s | 15.0 t/s | 9.5 t/s |
| Decode llama.cpp (t/s) | 29.90 t/s | 23.46 t/s | 20.37 t/s | 14.21 t/s |
| Ratio | **0.88x** | **0.89x** | **0.74x** | **0.67x** |

> Backfilled 2026-09-10 via `llama-bench.exe -m SmolLM2-1.7B-Instruct-Q4_K_M.gguf -p 0 -n 24 -d 267,773,1621,3218 -t 6 -ngl 0 -r 3`. Near-parity at short context (0.88-0.89x) erodes as context grows (0.67x @3.2k) — OT's decode falls off faster than llama.cpp's as KV cache grows, a real, previously-unmeasured gap.

**Vulkan prefill context scaling — Performance Check: 2026-08 (flash attention + SnapKV fix, iter 31-33):**

| Prompt tokens | 43 | 267 | 773 | 1621 | 3218 (default) |
|---|---:|---:|---:|---:|---:|
| Vulkan prefill OT (t/s) | ~83 t/s | ~84 t/s | ~78 t/s | ~66 t/s | **45.9 t/s** |
| CPU prefill OT (t/s) | — | ~50 t/s | ~49 t/s | ~49 t/s | ~42 t/s |

**Vulkan KV dtype breakdown (3239 tok, SnapKV off) — Performance Check: 2026-08 (iter 44/45):**

| KV dtype | Prefill (t/s) | Decode (t/s) | Perplexity delta vs fp32 | Performance Check |
|---|---:|---:|---|---|
| fp32 | 53.2 t/s | 6.0 t/s | baseline | 2026-08 |
| bf16 | 52.0 t/s (noise) | **9.4 t/s (+57%)** | +0.023% (negligible) | 2026-08 |
| q8_0 | not measured | ~7.6 t/s | +0.143% | 2026-08 |

> bf16 default flip blocked pending `--tq` and SnapKV compatibility. Source: iter 44/45.

**Vulkan Q4_K matvec bandwidth — Performance Check: 2026-08 (iter 28b); ceiling = 35.5 GB/s:**

| Shape | Achieved | % of ceiling | Performance Check |
|---|---:|---:|---|
| QKV/O 2048×2048 | 19.43 GB/s | 55% | 2026-08 |
| gate/up 8192×2048 | 30.52 GB/s | 86% | 2026-08 |
| down 2048×8192 | 28.98 GB/s | 82% | 2026-08 |
| Q6_K (large shapes) | 31.5–32.3 GB/s | 89–91% | 2026-08 |

---

## SmolLM2 small variants (135M / 360M)

| Model | Scenario | Backend | C# (OT, t/s) | C++ (llama.cpp, t/s) | Ratio | Performance Check | Source |
|---|---|---|---:|---:|---:|---|---|
| SmolLM2-135M-Instruct Q4_K_M | prefill (554 tok) | CPU | 1000.4 t/s | 1212.10 t/s | **0.83x** | 2026-09-11 | new coverage; stingray CLI + llama-bench, best-of-3 |
| SmolLM2-135M-Instruct Q4_K_M | decode (554 tok prompt, 24 tok gen) | CPU | 39.6 t/s | 316.65 t/s | **0.125x** | 2026-09-11 | new coverage; stingray CLI + llama-bench, best-of-3. Extremely low — the smallest dense model tested anywhere in this doc, confirming the small-model-decode-weakness pattern first seen in the Qwen2.5 family. |
| SmolLM2-360M-Instruct Q4_K_M | prefill (554 tok) | CPU | 369.5 t/s | 379.78 t/s | **0.97x** | 2026-09-11 | new coverage; stingray CLI + llama-bench, best-of-3. Near-parity prefill at this size. |
| SmolLM2-360M-Instruct Q4_K_M | decode (554 tok prompt, 24 tok gen) | CPU | 26.0 t/s | 135.04 t/s | **0.19x** | 2026-09-11 | new coverage; stingray CLI + llama-bench, best-of-3. Also low, same pattern. |

> **Small-model-decode-weakness pattern, now confirmed across two independent architecture
> families** (Qwen2.5 and SmolLM2/Llama): decode ratio at the smallest sizes (0.125-0.31x at
> 135M-500M) is dramatically worse than at 1.5-3B (0.58-0.79x), which is itself worse than the
> 7-8B dense-parity band (0.99-1.08x) found elsewhere in this doc. Three size tiers, three
> distinct decode-ratio bands — a real, size-dependent trend, not noise on one model.

---

## Qwen2.5 family

| Model | Scenario | Backend | C# (OT, t/s) | C++ (llama.cpp, t/s) | Ratio | Performance Check | Source |
|---|---|---|---:|---:|---:|---|---|
| Qwen2.5-0.5B-Instruct Q4_K_M | prefill (503 tok) | CPU | 327.1 t/s | 457.08 t/s | **0.72x** | 2026-09-11 | new coverage; stingray CLI + llama-bench, best-of-3 |
| Qwen2.5-0.5B-Instruct Q4_K_M | decode (503 tok prompt, 24 tok gen) | CPU | 25.8 t/s | 100.05 t/s | **0.26x** | 2026-09-11 | new coverage; stingray CLI + llama-bench, best-of-3. Notably low — the smallest dense model tested at this quant, decode ratio is much worse than its larger siblings below. |
| Qwen2.5-1.5B-Instruct Q4_K_M | prefill (503 tok) | CPU | 181.9 t/s | 242.26 t/s | **0.75x** | 2026-09-11 | new coverage; stingray CLI + llama-bench, best-of-3 |
| Qwen2.5-1.5B-Instruct Q4_K_M | decode (503 tok prompt, 24 tok gen) | CPU | 27.5 t/s | 41.55 t/s | **0.66x** | 2026-09-11 | new coverage; stingray CLI + llama-bench, best-of-3 |
| Qwen2.5-3B-Instruct Q4_K_M | prefill (503 tok) | CPU | 83.0 t/s | 111.96 t/s | **0.74x** | 2026-09-11 | new coverage; stingray CLI + llama-bench, best-of-3 |
| Qwen2.5-3B-Instruct Q4_K_M | decode (503 tok prompt, 24 tok gen) | CPU | 14.7 t/s | 20.76 t/s | **0.71x** | 2026-09-11 | new coverage; stingray CLI + llama-bench, best-of-3 |
| Qwen2.5-Coder-0.5B-Instruct Q4_K_M | prefill (503 tok) | CPU | 334.3 t/s | 307.70 t/s | <span style="color:#16a34a">**1.09x**</span> | 2026-09-11 | new coverage; stingray CLI + llama-bench, best-of-3. **Beats llama.cpp on prefill.** |
| Qwen2.5-Coder-0.5B-Instruct Q4_K_M | decode (503 tok prompt, 24 tok gen) | CPU | 25.3 t/s | 82.87 t/s | **0.31x** | 2026-09-11 | new coverage; stingray CLI + llama-bench, best-of-3 |
| Qwen2.5-Coder-1.5B-Instruct Q4_K_M | prefill (503 tok) | CPU | 170.7 t/s | 242.26 t/s | **0.70x** | 2026-09-11 | new coverage; stingray CLI + llama-bench, best-of-3 |
| Qwen2.5-Coder-1.5B-Instruct Q4_K_M | decode (503 tok prompt, 24 tok gen) | CPU | 24.2 t/s | 41.55 t/s | **0.58x** | 2026-09-11 | new coverage; stingray CLI + llama-bench, best-of-3 |
| Qwen2.5-Coder-3B-Instruct Q4_K_M | prefill (503 tok) | CPU | 87.1 t/s | 109.50 t/s | **0.80x** | 2026-09-11 | new coverage; stingray CLI + llama-bench, best-of-3 |
| Qwen2.5-Coder-3B-Instruct Q4_K_M | decode (503 tok prompt, 24 tok gen) | CPU | 14.7 t/s | 21.48 t/s | **0.68x** | 2026-09-11 | new coverage; stingray CLI + llama-bench, best-of-3 |

> **Pattern**: decode ratio scales up with model size within this family (0.26-0.31x at 0.5B →
> 0.58-0.71x at 1.5-3B) — the smallest models lose proportionally more on decode, the opposite
> of the dense-7-8B-parity pattern found elsewhere in this doc. Prefill stays in a tighter
> 0.70-0.80x band except the 0.5B-Coder outlier, which beats llama.cpp outright.

---

## Qwen3 family

| Model | Scenario | Backend | C# (OT, t/s) | C++ (llama.cpp, t/s) | Ratio | Performance Check | Source |
|---|---|---|---:|---:|---:|---|---|
| Qwen3-0.6B Q8_0 | prefill (493 tok, docs/benchmark-prompt.txt) | CPU | 226.1 t/s | 265.28 t/s | **0.85x** | 2026-09-10 | stingray CLI + llama-bench, best-of-3 |
| Qwen3-0.6B Q8_0 | decode (493 tok prompt, 24 tok gen) | CPU | 41.2 t/s | 62.95 t/s | **0.65x** | 2026-09-10 | stingray CLI + llama-bench, best-of-3 (re-measured at matched prompt length; supersedes the 47.4 t/s figure below) |
| Qwen3-0.6B Q8_0 | prefill (493 tok) | Vulkan iGPU | 23.6 t/s | — | — | 2026-09-10 | new coverage; no llama.cpp Vulkan ref. Much worse than CPU (23.6 vs 226.1 t/s) — at this small size, fixed per-dispatch GPU overhead dominates |
| Qwen3-0.6B Q8_0 | decode (493 tok prompt, 24 tok gen) | Vulkan iGPU | 21.1 t/s | — | — | 2026-09-10 | new coverage; also worse than CPU (21.1 vs 41.2 t/s) — the smallest model tested on Vulkan this pass, and the one with the largest CPU-vs-Vulkan decode gap |
| Qwen3-4B Q4_K_M | prefill (493 tok) | CPU | 61.4 t/s | 84.36 t/s | **0.73x** | 2026-09-10 | stingray CLI + llama-bench, best-of-3 (new coverage) |
| Qwen3-4B Q4_K_M | decode (493 tok prompt, 24 tok gen) | CPU | 10.0 t/s | 16.64 t/s | **0.60x** | 2026-09-10 | stingray CLI + llama-bench, best-of-3 (new coverage) |
| Qwen3-4B Q4_K_M | prefill (493 tok) | Vulkan iGPU | 37.4 t/s | — | — | 2026-09-10 | new coverage; no llama.cpp Vulkan ref. Worse than CPU (37.4 vs 61.4 t/s) |
| Qwen3-4B Q4_K_M | decode (493 tok prompt, 24 tok gen) | Vulkan iGPU | 9.6 t/s | — | — | 2026-09-10 | new coverage; close to CPU decode (9.6 vs 10.0 t/s) |
| Qwen3-0.6B Q8_0 | decode (short ctx, original baseline) | CPU | 47.4 t/s | — | — | 2026-08-07 | cpu-performance-baseline.md (kept for history; see re-measured row above) |
| Qwen3-8B Q4_K_M | decode (short ctx, original) | CPU | 6.8 t/s | — | — | 2026-08 | cpu-speculative-decoding-findings.md |
| Qwen3-8B Q4_K_M | prefill (493 tok) | CPU | 48.3 t/s | 47.28 t/s | **1.02x** | 2026-09-10 | stingray CLI + llama-bench, best-of-3 (backfill) |
| Qwen3-8B Q4_K_M | decode (493 tok prompt, 24 tok gen) | CPU | 6.3 t/s | 7.67 t/s | **0.82x** | 2026-09-10 | stingray CLI + llama-bench, best-of-3 (backfill) |
| Qwen3-8B Q4_K_M | prefill (493 tok) | Vulkan iGPU | 19.7 t/s | — | — | 2026-09-10 | new coverage; no llama.cpp Vulkan ref. Notably *worse* than this same model's CPU prefill (19.7 vs 48.3 t/s) — this iGPU trails CPU on prefill here, consistent with `docs/done/vulkan-backend-evidence.md` |
| Qwen3-8B Q4_K_M | decode (493 tok prompt, 24 tok gen) | Vulkan iGPU | 5.9 t/s | — | — | 2026-09-10 | new coverage; close to CPU decode (5.9 vs 6.3 t/s) — decode is the one path where this iGPU is competitive with CPU |
| Qwen3-Coder 30B-A3B Q4_K_M | prefill | CUDA (†) | 102.6 t/s | — | — | 2026-06-16 | README history 0c171ed |
| Qwen3-Coder 30B-A3B Q4_K_M | decode | CUDA (†) | 28.0 t/s | — | — | 2026-06-16 | README history 0c171ed |
| Qwen3-Coder 30B-A3B Q4_K_M | decode (original) | CPU | 22.4 t/s | — | — | 2026-06-16 | README history 0c171ed |
| Qwen3-Coder 30B-A3B Q4_K_M | decode (`--tq`) | CPU | 22.6 t/s | — | — | 2026-06-16 | README history 0c171ed |
| Qwen3-Coder-30B-A3B-Instruct Q4_K_M | prefill (482 tok) | CPU | 24.7 t/s | 41.76 t/s | **0.59x** | 2026-09-10 | stingray CLI + llama-bench, best-of-3 (backfill; unsloth GGUF, checkpoint differs from the original † row's) |
| Qwen3-Coder-30B-A3B-Instruct Q4_K_M | decode (482 tok prompt, 24 tok gen) | CPU | 7.4 t/s | 6.85 t/s | <span style="color:#16a34a">**1.08x**</span> | 2026-09-10 | stingray CLI + llama-bench, best-of-3 (backfill — beats llama.cpp on decode) |
| Qwen3-Coder-30B-A3B-Instruct Q4_K_M | prefill (482 tok) | Vulkan iGPU | 9.2 t/s | — | — | 2026-09-10 | new coverage; no llama.cpp Vulkan ref. Worse than CPU prefill (9.2 vs 24.7 t/s), the usual iGPU-trails-CPU-on-prefill pattern |
| Qwen3-Coder-30B-A3B-Instruct Q4_K_M | decode (482 tok prompt, 24 tok gen) | Vulkan iGPU | 9.0 t/s | — | — | 2026-09-10 | new coverage; actually *beats* this same model's CPU decode (9.0 vs 7.4 t/s) — the one case this pass where Vulkan decode outperforms CPU decode outright, not just "close" |
| Qwen3.6-35B-A3B Q4_K_M | prefill | CUDA (†) | 475.4 t/s | — | — | 2026-06-16 | README history 0c171ed |
| Qwen3.6-35B-A3B Q4_K_M | decode | CUDA (†) | 24.5 t/s | — | — | 2026-06-16 | README history 0c171ed |
| Qwen3.6-35B-A3B Q4_K_M | decode | Vulkan (†) | 22.8 t/s | — | — | 2026-06-16 | README history 0c171ed |
| Qwen3.6-35B-A3B Q4_K_M | decode (original) | CPU | 9.3 t/s | — | — | 2026-06-16 | README history 0c171ed |
| Qwen3.6-35B-A3B Q6_K (hybrid GDN) | prefill (480 tok) | CPU | 2.9 t/s | 58.60 t/s | **0.05x** | 2026-09-10 | stingray CLI + llama-bench, best-of-3 (backfill; unsloth Q6_K GGUF, different quant/build from the original † row's; same hybrid-GDN architecture as Qwen3.6-27B-MTP) |
| Qwen3.6-35B-A3B Q6_K (hybrid GDN) | decode (480 tok prompt, 24 tok gen) | CPU | 1.8 t/s | 10.59 t/s | **0.17x** | 2026-09-10 | stingray CLI + llama-bench, best-of-3 (backfill) |
| Qwen3.6-35B-A3B Q6_K (hybrid GDN) | prefill (480 tok) | Vulkan iGPU | 2.1 t/s | — | — | 2026-09-10 | new coverage; no llama.cpp Vulkan ref. MoE routed experts run on CPU (mmap) with shared expert on GPU — a split workload. Slightly worse than CPU (2.1 vs 2.9 t/s), unlike the 27B hybrid-GDN case where Vulkan was much worse — this 35B MoE variant's split isn't as costly. |
| Qwen3.6-35B-A3B Q6_K (hybrid GDN) | decode (480 tok prompt, 24 tok gen) | Vulkan iGPU | 2.0 t/s | — | — | 2026-09-10 | new coverage; actually beats CPU decode (2.0 vs 1.8 t/s) — unlike Qwen3.6-27B-MTP where Vulkan lost on both metrics |
| Qwen3.6-35B-A3B-MTP Q4_K_M | prefill | CUDA (†) | 480.2 t/s | — | — | 2026-06-16 | README history 0c171ed |
| Qwen3.6-35B-A3B-MTP Q4_K_M | decode (`--no-thinking`) | CUDA (†) | 33.3 t/s | ~41 t/s (est.) | **~0.81x** | 2026-06-16 | README: "~80% of llama.cpp tg128" |
| Qwen3.6-27B-MTP Q4_K_M | prefill | CUDA (†) | 22.0 t/s | — | — | 2026-06-16 | README history 0c171ed |
| Qwen3.6-27B-MTP Q4_K_M | decode (`--no-thinking`) | CUDA (†) | 12.3 t/s | — | — | 2026-06-16 | README history 0c171ed |
| Qwen3.6-27B-MTP Q4_K_M | decode (`--no-thinking`, original) | CPU | 3.6 t/s | — | — | 2026-06-16 | README history 0c171ed |
| Qwen3.6-27B Q3_K_XL (hybrid GDN) | prefill (480 tok) | CPU | 1.0 t/s | 6.15 t/s | **0.16x** | 2026-09-10 | stingray CLI + llama-bench, best-of-3 (backfill; unsloth Q3_K_XL GGUF, different quant/build from the original † row's; real find — this arch **is** supported, `[HybridGdnForwardPass]`, contrary to the earlier assumption it was unlocatable/gated) |
| Qwen3.6-27B Q3_K_XL (hybrid GDN) | decode (480 tok prompt, 24 tok gen) | CPU | 1.0 t/s | 1.33 t/s | **0.75x** | 2026-09-10 | stingray CLI + llama-bench, best-of-3 (backfill) |
| Qwen3.6-27B Q3_K_XL (hybrid GDN) | prefill (480 tok) | Vulkan iGPU | 0.4 t/s | — | — | 2026-09-10 | new coverage; no llama.cpp Vulkan ref. **Worse than CPU (0.4 vs 1.0 t/s)** — only 37/64 dense-FFN layers fit on GPU (`Dense FFN-on-GPU: uploaded 37/64 layers; 27 stay on CPU`), so this is a split CPU/GPU workload, not full offload; 117.4s just to load. |
| Qwen3.6-27B Q3_K_XL (hybrid GDN) | decode (480 tok prompt, 24 tok gen) | Vulkan iGPU | 0.3 t/s | — | — | 2026-09-10 | new coverage; also worse than CPU (0.3 vs 1.0 t/s) — hybrid-GDN is the one architecture where Vulkan is strictly worse than CPU on both prefill and decode, unlike dense/MoE models where Vulkan only lost on prefill |
| Ornith-1.0-9B Q4_K_M (hybrid GDN, `qwen35` arch) | prefill (480 tok) | CPU | 1.7 t/s | 20.55 t/s | **0.08x** | 2026-09-10 | new coverage; stingray CLI + llama-bench, best-of-2/3. **Third hybrid-GDN size point (9B) confirming the same severe prefill weakness as 27B (0.16x) and 35B (0.05x) — a consistent architectural pattern, not noise on one checkpoint.** |
| Ornith-1.0-9B Q4_K_M (hybrid GDN, `qwen35` arch) | decode (480 tok prompt, 24 tok gen) | CPU | 1.2 t/s | 3.70 t/s | **0.32x** | 2026-09-10 | new coverage; stingray CLI + llama-bench, best-of-2/3. **Caveat, unconfirmed which cause:** output repeats a single token ("giochi giochi giochi..."). The CLI's own warning on this run said greedy decoding (`--temp 0`) on a reasoning-tuned model can loop — so this may just be that expected failure mode rather than a new correctness bug (unlike Qwen3-ASR/FunASR-Nano's degenerate output, which had no such warning attached). Not re-tested with `--temp 0.6` to disambiguate this pass. |
| Ornith-1.0-9B Q4_K_M (hybrid GDN, `qwen35` arch) | prefill (480 tok) | Vulkan iGPU | 7.1 t/s | — | — | 2026-09-10 | new coverage; no llama.cpp Vulkan ref. **All 32/32 dense-FFN layers fit on GPU this time (unlike Qwen3.6-27B's 37/64 partial split) — full offload achieved, and it shows: Vulkan beats CPU by 4x here (7.1 vs 1.7 t/s)**, the opposite of every other hybrid-GDN case measured. Also degenerate output, but a different pattern ("!!!!!!!!" repeated) than CPU's ("giochi") — same underlying instability, different manifestation per backend. |
| Ornith-1.0-9B Q4_K_M (hybrid GDN, `qwen35` arch) | decode (480 tok prompt, 24 tok gen) | Vulkan iGPU | 5.3 t/s | — | — | 2026-09-10 | new coverage; also beats CPU decode by ~4x (5.3 vs 1.2 t/s) — full-GPU-offload hybrid-GDN is a real bright spot when it fits, unlike the partial-offload cases |
| Qwen3.8-27B Q3_K_XL (hybrid GDN, `qwen35` arch) | prefill (522 tok) | CPU | 0.2 t/s | 3.00 t/s | **0.07x** | 2026-09-10 | new coverage; stingray CLI + llama-bench (single run vs best-of-2). **Fourth hybrid-GDN size point confirming the same severe prefill weakness** (27B/9B/35B all in the 0.05-0.16x band). Real, coherent output this time (unlike Ornith) — also surfaced a separate, unrelated finding: 3 real Jinja chat-template rendering gaps logged as warnings (`sysns.text + ...`, `resolved_reasoning_effort not in (...)`, string-concat-in-conditional expressions) — template output may be subtly wrong for this checkpoint's chat format. |
| Qwen3.8-27B Q3_K_XL (hybrid GDN, `qwen35` arch) | decode (522 tok prompt, 24 tok gen) | CPU | 0.2 t/s | 1.55 t/s | **0.13x** | 2026-09-10 | new coverage; stingray CLI + llama-bench (single run vs best-of-2) |
| Qwen3.8-27B Q3_K_XL (hybrid GDN, `qwen35` arch) | Vulkan iGPU | — | — | — | — | 2026-09-10 | **Clean, documented limitation, not a bug**: `VulkanHybridGdnForwardPass` throws `NotSupportedException` — this model's 248320-vocab × 5120-dim embedding/output tensor exceeds Vulkan's 2GB single-GPU-storage-buffer limit; CPU-embedding fallback isn't implemented in v1. Exact error message names the fix ("Reduce ctx size or use HybridGdnForwardPass for CPU-only execution"). |
| Carnice 35B-A3B-MTP (APEX) | prefill (`--no-thinking`) | CUDA (†) | 522.0 t/s | — | — | 2026-06-16 | README history 0c171ed |
| Carnice 35B-A3B-MTP (APEX) | decode (`--no-thinking`) | CUDA (†) | 26.5 t/s | — | — | 2026-06-16 | README history 0c171ed |

> **Qwen3-8B decode:** 6.8 t/s = 34.2 GB/s = 93% of the measured 36.8 GB/s DRAM ceiling.
> Speculative decoding is a confirmed −37% loss on this CPU — see Speculative Decoding section.

---

## OLMoE family

| Model | Scenario | Backend | C# (OT, t/s) | C++ (llama.cpp, t/s) | Ratio | Performance Check | Source |
|---|---|---|---:|---:|---:|---|---|
| OLMoE-1B-7B Q4_K_M | prefill (405-465 tok, original) | CPU | 105.6 t/s | — | — | 2026-08-07 | cpu-performance-baseline.md |
| OLMoE-1B-7B Q4_K_M | decode (original, ~7 tok, early EOS) | CPU | 28.2 t/s | — | — | 2026-08-07 | cpu-performance-baseline.md |
| OLMoE-1B-7B-0924-Instruct Q4_K_M | prefill (515 tok) | CPU | 124.8 t/s | 180.82 t/s | **0.69x** | 2026-09-10 | stingray CLI + llama-bench, best-of-3 (backfill) |
| OLMoE-1B-7B-0924-Instruct Q4_K_M | decode (515 tok prompt, 24 tok gen, no early EOS) | CPU | 25.3 t/s | 50.58 t/s | **0.50x** | 2026-09-10 | stingray CLI + llama-bench, best-of-3 (backfill; full 24-token run, not early-EOS-truncated like the row above) |
| OLMoE-1B-7B-0924-Instruct Q4_K_M | prefill (515 tok) | Vulkan iGPU | 25.0 t/s | — | — | 2026-09-10 | new coverage; no llama.cpp Vulkan ref. Much worse than CPU prefill (25.0 vs 124.8 t/s) — same iGPU-trails-CPU-on-prefill pattern as Qwen3-8B and Gemma-4 |
| OLMoE-1B-7B-0924-Instruct Q4_K_M | decode (515 tok prompt, 24 tok gen) | Vulkan iGPU | 21.1 t/s | — | — | 2026-09-10 | new coverage; close to CPU decode (21.1 vs 25.3 t/s) |

> Original decode measured over ~7 tokens (early EOS); treat as approximate. The re-measured row above forced a full 24-token generation and is the reliable one going forward.

---

## Gemma family

| Model | Scenario | Backend | C# (OT, t/s) | C++ (llama.cpp, t/s) | Ratio | Performance Check | Source |
|---|---|---|---:|---:|---:|---|---|
| Gemma-4-12B Q4_0 | prefill (original) | CPU | 3.8 t/s | — | — | 2026-08-07 | cpu-performance-baseline.md |
| Gemma-4-12B Q4_0 | decode (original) | CPU | 3.7 t/s | — | — | 2026-08-07 | cpu-performance-baseline.md |
| Gemma-4-12B-it Q4_K_M | prefill (501 tok) | CPU | 3.5 t/s | 27.81 t/s | **0.13x** | 2026-09-10 | stingray CLI + llama-bench (quant differs from original Q4_0 row: Q4_K_M unsloth GGUF, not the QAT Q4_0 build; prefill:decode still ~1.0x, same batched-prefill-missing signature as the original row) |
| Gemma-4-12B-it Q4_K_M | decode (501 tok prompt, 24 tok gen) | CPU | 4.2 t/s | 5.69 t/s | **0.74x** | 2026-09-10 | stingray CLI + llama-bench |
| Gemma-4-12B-it Q4_K_M | prefill (501 tok) | Vulkan iGPU | 3.4 t/s | — | — | 2026-09-10 | new coverage; no llama.cpp Vulkan ref. **Real finding: prefill≈decode (3.4 vs 3.6 t/s) — the same missing-batched-prefill signature as CPU, confirming this bug is not CPU-specific.** |
| Gemma-4-12B-it Q4_K_M | decode (501 tok prompt, 24 tok gen) | Vulkan iGPU | 3.6 t/s | — | — | 2026-09-10 | new coverage; see prefill row — Vulkan decode is actually *slower* than CPU decode (3.6 vs 4.2 t/s) on this iGPU, consistent with `docs/done/vulkan-backend-evidence.md`'s finding that this iGPU trails CPU |
| Gemma-3-4B-it Q4_K_M | prefill (490 tok) | CPU | 12.4 t/s | 97.32 t/s | **0.127x** | 2026-09-11 | new coverage; stingray CLI + llama-bench, best-of-3. **Gemma 3 is a genuinely different architecture from Gemma 4 (per README: `gemma3`/SigLIP vs `gemma4uv`/unified) — and shows the SAME prefill≈decode signature (12.4 vs 12.1 t/s) as Gemma 4's missing-batched-prefill bug. Suggests this gap spans the whole Gemma family, not just Gemma 4.** |
| Gemma-3-4B-it Q4_K_M | decode (490 tok prompt, 24 tok gen) | CPU | 12.1 t/s | 15.31 t/s | **0.79x** | 2026-09-11 | new coverage; stingray CLI + llama-bench, best-of-3 |
| Gemma-3-4B-it Q4_K_M | prefill (490 tok) | Vulkan iGPU | 35.4 t/s | — | — | 2026-09-11 | new coverage; no llama.cpp Vulkan ref. **Nuance: unlike Gemma-4, the bug does NOT reproduce on Vulkan here** — prefill:decode is a real 3.3x ratio (35.4 vs 10.8), not the ~1:1 signature seen on CPU. Either Gemma-3 takes a genuinely different Vulkan code path than Gemma-4, or the CPU-side gap has a different root cause than initially assumed. |
| Gemma-3-4B-it Q4_K_M | decode (490 tok prompt, 24 tok gen) | Vulkan iGPU | 10.8 t/s | — | — | 2026-09-11 | new coverage; slightly worse than CPU decode (10.8 vs 12.1 t/s) |
| Gemma4 E4B QAT Q4_0 | prefill | CUDA (†) | 3666 t/s | — | — | 2026-06-16 | README history 0c171ed |
| Gemma4 E4B QAT Q4_0 | decode | CUDA (†) | 100.4 t/s | — | — | 2026-06-16 | README history 0c171ed |
| Gemma4 E4B QAT Q4_0 | prefill | Vulkan (†) | 35 t/s | — | — | 2026-06-22 | README history 0c171ed |
| Gemma4 E4B QAT Q4_0 | decode | Vulkan (†) | 39.5 t/s | — | — | 2026-06-22 | README history 0c171ed |
| Gemma4-E4B-it Q4_K_M | prefill (497 tok) | CPU | 9.8 t/s | 80.50 t/s | **0.12x** | 2026-09-10 | stingray CLI + llama-bench (new CPU coverage, quant differs from QAT Q4_0 rows above) |
| Gemma4-E4B-it Q4_K_M | decode (497 tok prompt, 24 tok gen) | CPU | 9.7 t/s | 13.07 t/s | **0.74x** | 2026-09-10 | stingray CLI + llama-bench (new CPU coverage) |
| Gemma4-E4B-it Q4_K_M | prefill (497 tok) | Vulkan iGPU | 9.1 t/s | — | — | 2026-09-10 | new coverage; no llama.cpp Vulkan ref. **Missing-batched-prefill signature reproduces a third time** (prefill≈decode: 9.1 vs 8.2 t/s) — now confirmed on Gemma-4-12B (CPU+Vulkan) and Gemma4-E4B (CPU+Vulkan), fully architecture-wide across both sizes and both backends. Output is coherent this time (unlike the separate Granite/Vulkan bug). |
| Gemma4-E4B-it Q4_K_M | decode (497 tok prompt, 24 tok gen) | Vulkan iGPU | 8.2 t/s | — | — | 2026-09-10 | new coverage; close to CPU decode (8.2 vs 9.7 t/s) |
| Gemma4 12B QAT Q4_0 | prefill | CUDA (†) | 1714 t/s | — | — | 2026-06-16 | README history 0c171ed |
| Gemma4 12B QAT Q4_0 | decode | CUDA (†) | 54.1 t/s | 57 t/s | **0.95x** | 2026-06-16 | README history 0c171ed |
| Gemma4 12B QAT Q4_0 | prefill | Vulkan (†) | 17.0 t/s | — | — | 2026-06-16 | README history 0c171ed |
| Gemma4 12B QAT Q4_0 | decode | Vulkan (†) | 19.1 t/s | — | — | 2026-06-16 | README history 0c171ed |
| Gemma4 12B QAT Q4_0 | decode | CPU (†, Zen 4) | 5.1 t/s | — | — | 2026-06-16 | README history 0c171ed |

> **Gemma-4-12B (dev machine):** prefill:decode ratio 1.0x is the signature of a missing
> batched-prefill gate (`perLayerHdUnsupported`). ~5.7× prefill penalty vs Qwen3-8B size-adjusted.
> **Gemma4 12B CUDA decode (†):** within ~6% of llama.cpp — best verified parity in the table.

---

## Llama family

| Model | Scenario | Backend | C# (OT, t/s) | C++ (llama.cpp, t/s) | Ratio | Performance Check | Source |
|---|---|---|---:|---:|---:|---|---|
| Llama-4 Scout 17B-16E Q4_K_M | decode | CPU (†, smoke) | 4.3 t/s | — | — | 2026-06-16 | README history 0c171ed; smoke run only |
| Llama-4 Scout 17B-16E Q4_K_M | decode | CUDA (†, smoke) | 2.6 t/s | — | — | 2026-06-16 | README history 0c171ed; smoke run, model dwarfs 12 GB card |
| Mistral-7B-Instruct-v0.3 Q4_K_M | prefill (571 tok) | CPU | 46.4 t/s | 46.27 t/s | <span style="color:#16a34a">**1.00x**</span> | 2026-09-10 | new coverage; stingray CLI + llama-bench, best-of-3. **Real parity — a second dense 7-8B-class model hitting ~1.0x prefill, alongside Qwen3-8B's 1.02x.** |
| Mistral-7B-Instruct-v0.3 Q4_K_M | decode (571 tok prompt, 24 tok gen) | CPU | 6.5 t/s | 9.37 t/s | **0.69x** | 2026-09-10 | new coverage; stingray CLI + llama-bench, best-of-3 |
| Mistral-7B-Instruct-v0.3 Q4_K_M | prefill (571 tok) | Vulkan iGPU | 22.2 t/s | — | — | 2026-09-10 | new coverage; no llama.cpp Vulkan ref. Worse than CPU (22.2 vs 46.4 t/s), the usual pattern |
| Mistral-7B-Instruct-v0.3 Q4_K_M | decode (571 tok prompt, 24 tok gen) | Vulkan iGPU | 6.8 t/s | — | — | 2026-09-10 | new coverage; slightly beats CPU decode (6.8 vs 6.5 t/s) |
| Ministral-8B-Instruct-2410 Q4_K_M | prefill (488 tok) | CPU | 45.2 t/s | 45.79 t/s | <span style="color:#16a34a">**0.99x**</span> | 2026-09-10 | new coverage; stingray CLI + llama-bench, best-of-3. **Third dense 7-8B model at ~1.0x prefill parity** (with Qwen3-8B 1.02x and Mistral-7B 1.00x) — a consistent, real pattern at this size class. |
| Ministral-8B-Instruct-2410 Q4_K_M | decode (488 tok prompt, 24 tok gen) | CPU | 6.1 t/s | 8.61 t/s | **0.71x** | 2026-09-10 | new coverage; stingray CLI + llama-bench, best-of-3 |
| Ministral-8B-Instruct-2410 Q4_K_M | prefill (488 tok) | Vulkan iGPU | 21.9 t/s | — | — | 2026-09-10 | new coverage; no llama.cpp Vulkan ref. Worse than CPU (21.9 vs 45.2 t/s), the usual pattern |
| Ministral-8B-Instruct-2410 Q4_K_M | decode (488 tok prompt, 24 tok gen) | Vulkan iGPU | 6.5 t/s | — | — | 2026-09-10 | new coverage; beats CPU decode (6.5 vs 6.1 t/s) — third 7-8B-class model this pass where Vulkan decode edges out CPU |

---

## DeepSeek family (`deepseek2`, run with `--allow-unverified-arch`)

**Correctness caveat, not a perf gap:** this architecture is explicitly documented in `README.md`
as producing "numerically wrong greedy output" — a real, closed investigation (see
`docs/done/032-deepseek2-mla-yarn-moe-routing-investigation.md`) found and fixed 8+ real bugs but
root-caused the remaining gap as this specific checkpoint's inherent MoE routing-landscape
flatness, not a discoverable code defect — formally accepted, not left open. Confirmed 2026-09-11:
running without `--allow-unverified-arch` gives a clean rejection (not in the supported-architecture
allowlist); with the flag it runs and produces garbled output, matching README's own documented
finding exactly (repro: `Infjs<garbled>icer\nHello\n的多` for a simple "Hello" prompt). The
throughput numbers below are still real and meaningful — this is the same "measure it anyway, note
the correctness caveat" approach used elsewhere in this doc (e.g. Ornith-1.0-9B, Qwen3-ASR).

| Model | Scenario | Backend | C# (OT, t/s) | C++ (llama.cpp, t/s) | Ratio | Performance Check | Source |
|---|---|---|---:|---:|---:|---|---|
| DeepSeek-V2-Lite-Chat Q8_0 | prefill (502 tok) | CPU | 28.0 t/s | 56.90 t/s | **0.49x** | 2026-09-11 | new coverage; stingray CLI (`--allow-unverified-arch`) + llama-bench, best-of-3 |
| DeepSeek-V2-Lite-Chat Q8_0 | decode (502 tok prompt, 24 tok gen) | CPU | 13.6 t/s | 14.83 t/s | <span style="color:#16a34a">**0.92x**</span> | 2026-09-11 | new coverage; stingray CLI (`--allow-unverified-arch`) + llama-bench, best-of-3. **Near-parity decode despite the known correctness gap** — the throughput cost of this architecture's MoE dispatch is small even though the routing itself produces wrong tokens. |

---

## gpt-oss-20B (`gpt-oss`, run with `--allow-unverified-arch`) — genuinely new coverage, not previously benchmarked

Not in `ModelCompatibility`'s supported-architecture allowlist at all (clean rejection without the
flag, listing every currently-supported profile). With `--allow-unverified-arch` it loads and runs,
but produces fully garbled output — expected exactly per the flag's own warning; this is a genuinely
unimplemented architecture, not a bug to chase. (Also surfaced an unrelated real Jinja gap: a
string-concat-inside-conditional expression for `tool_call.content_type`, same class already logged
for Qwen3.8-27B/Gemma-3-4B-it above.) `-MXFP4.gguf` is an exotic 4-bit microscaling quant format;
part of the garbling may also be an unimplemented/incorrect MXFP4 dequant path rather than purely
the unverified architecture, not disambiguated.

| Model | Scenario | Backend | C# (OT, t/s) | Performance Check | Source |
|---|---|---|---:|---|---|
| gpt-oss-20B MXFP4 | prefill (72 tok) | CPU | 12.6 t/s avg (12.6/12.7/12.6 across 3 runs) | 2026-09-11 | new coverage; stingray CLI (`--allow-unverified-arch`), best-of-3. First timing ever recorded for this checkpoint |
| gpt-oss-20B MXFP4 | decode (60 tok gen) | CPU | 14.0 t/s avg (13.8/14.1/14.1 across 3 runs) | 2026-09-11 | same run; garbled output as expected for an unimplemented architecture, not usable as a correctness measurement |

---

## Speculative Decoding (CPU)

| Target | Draft | Scenario | C# (OT, t/s) | C++ (ref, t/s) | Ratio | Acceptance rate | Performance Check | Source |
|---|---|---|---:|---:|---:|---|---|---|
| Qwen3-8B Q4_K_M | Qwen3-0.6B Q8_0 | decode, draft-n 4 | 4.3 t/s (−37%) | — | — | 62% | 2026-08 | cpu-speculative-decoding-findings.md |
| Qwen3-4B Q4_K_M | DSpark block-7 (`dspark_qwen3_4b_block7`) | decode, DSpark | 2.5 t/s (**−75%** vs 10.0 t/s plain baseline) | — | — | 23% (14/60) | 2026-09-10 | new coverage; `--dspark-model` CLI flag, real run (draft 3438ms / verify 5879ms / commit 68ms per step) |

> Speculation is a **confirmed loss** on this hardware, now on two independent target/draft pairs.
> Q4_K dot is ~87% compute-bound (not bandwidth-bound); verifying k tokens costs ~k× compute
> regardless of dispatch. VNNI (`vpdpbusd`, Zen 4+) is the prerequisite for speculation to become
> viable. The DSpark pairing is a worse loss than n-gram/draft-model speculation (−75% vs −37%) and
> its acceptance rate is also lower (23% vs 62%) — the draft head itself may be weaker, or block-7
> confidence-gated speculation costs more per rejected block than simple n-token drafting does;
> not yet disambiguated.

---

## TTS / Audio Synthesis

| Pipeline | Scenario | Backend | C# Wall | C# RTF | C++ Wall | C++ RTF | Ratio | Performance Check | Confirmed Working |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| Piper lessac-medium (ONNX) | text → 2.45s audio | CPU | 0.47s | **0.19×** | — | — | — | 2026-09-09 | 2026-09-03 👂 |
| Piper lessac-medium (ONNX) | text → ~2.2-2.4s audio, cold CLI (`stingray tts -e piper -g cpu/vulkan`) | CPU vs Vulkan iGPU | 1.08s (CPU) vs 1.05s (Vulkan), RTF ~0.46-0.47x both | — | — | — | — | 2026-09-10 | new coverage. CPU and Vulkan give near-identical timing — this pipeline runs through ONNX Runtime, which likely doesn't route through this project's native Vulkan compute backend at all regardless of `-g`, so the two numbers may really be measuring the same CPU execution path twice. Not confirmed either way — flagged as a real question, not a claim. **CLI UX note:** passing `-m <file>.onnx` directly fails with a confusing JSON-parse error; the `.onnx.json` sidecar path must be passed instead (it locates the `.onnx` weights itself). |
| MMS-TTS eng (VITS) | text → 3.65s audio | CPU | 1.38s | **0.38×** | — | — | — | 2026-09-09 | 2026-08-30 🔬 |
| Kokoro-82M (Q8_0 GGUF) | text → 2.93s audio | CPU | 2.60s | **0.89×** | — | — | — | 2026-09-09 | 2026-09-03 👂 |
| Kokoro-82M (Q8_0 GGUF) | text → 2.93s audio, cold CLI (`stingray tts -g cpu`, no warmup) | CPU | ~3.9s (mean of 3) | **~1.33×** | — | — | — | 2026-09-10 | new methodology, not a contradiction of the row above: `stingray tts` timing includes per-call model load (~82M weights), while the 0.89x row above times generation only after a warmup call. Both are real; they measure different things (cold-start-inclusive vs warm generation). |
| Kokoro-82M (Q8_0 GGUF) | text → 2.93s audio, cold CLI (`stingray tts -g vulkan`, no warmup) | Vulkan iGPU | ~3.7s (mean of 3) | **~1.27×** | — | — | — | 2026-09-10 | new coverage, same cold-CLI methodology as the CPU row above — comparable to each other, not to the warm 0.89x row. Vulkan is marginally faster than the cold CPU run here (~1.27x vs ~1.33x), unlike the usual iGPU-trails-CPU pattern seen on LLMs. |
| MeloTTS zh_en (ONNX) | text → 2.74s audio | CPU | 3.11s | **1.14×** | — | — | — | 2026-09-09 | 2026-09-03 👂 |
| MeloTTS zh_en (ONNX) | text → 2.73s audio, cold CLI (`stingray tts -e melo -g cpu/vulkan`) | CPU vs Vulkan iGPU | 4.48s (CPU, 1.64x) vs 4.24s (Vulkan, 1.56x) | — | — | — | — | 2026-09-10 | new coverage. Close numbers again, same ONNX-Runtime-likely-ignores-`-g` caveat as Piper above. |
| QwenTTS 0.6B (Q8_0 GGUF) | text → 2.16s audio | CPU | 6.59s | **3.05×** | 11.38s | 4.07× | <span style="color:#16a34a">**1.33x**</span> | 2026-09-09 | 2026-08-29 👂 |
| CosyVoice3 (DiT + HiFT) | text → 3.00s audio | CPU | 17.20s | **5.73×** | 31.66s | 9.05× | <span style="color:#16a34a">**1.58x**</span> | 2026-09-09 | 2026-09-06 👂 |
| Chatterbox Turbo (Q4_K) | text → 2.52s audio | CPU | 13.72s | **5.45×** | 16.24s | 6.77× | <span style="color:#16a34a">**1.24x**</span> | 2026-09-09 | 2026-08-30 🔬 |
| Chatterbox Turbo (Q4_K) | text → ~2.2-2.3s audio, cold CLI (`stingray tts -e chatterbox -g cpu/vulkan`) | CPU vs Vulkan iGPU | 6.00s (CPU, 2.68x) vs 6.11s (Vulkan, 2.63x) | — | — | — | — | 2026-09-10 | new coverage. **Checked and ruled out a false "Vulkan is 2x faster" claim**: cold CPU RTF here (2.68x) is also much better than the doc's warm 5.45x row — confirms this is the same cold-CLI-vs-warm-benchmark methodology gap seen on Kokoro/MeloTTS, not a real Vulkan speedup. CPU and Vulkan are near-identical to each other here (2.68x vs 2.63x). |
| Parler-TTS Mini v1 | text → 2.81s audio | CPU | 17.36s | **6.18×** | — | — | — | 2026-09-09 | 2026-08-28 👂 |
| FishSpeech S2 Pro (Q4_K) | text → 3.44s audio | CPU | 28.46s | **8.28×** | 36.12s | 10.95× | <span style="color:#16a34a">**1.32x**</span> | 2026-09-09 | 2026-08-29 👂 |
| F5-TTS Base (DiT) | text → 2.77s audio | CPU | 27.25s | **9.82×** | 43.2s (mean of 3: 43.2/43.1/43.2) | 15.6× | <span style="color:#16a34a">**1.59x**</span> | 2026-09-11 (C++ ref, real, first ever) / 2026-08-28 👂 (C#) | C++ ref: real, first-ever working `audio.cpp` F5-TTS comparison — see the FIXED entry in Known Measurement Gaps below for how; `--voice-ref test_s2.wav --reference-text "hello" --text "Hello, I will make some lunch, darling!"` (same real prompt as the original gap-finding repro). OT is faster here, unlike most other TTS rows where C++ wins |
| F5-TTS Base (DiT) | text → 2.77s audio, cold CLI (`stingray tts -e f5tts -g cpu`) | CPU | 31.09s | **11.21×** | — | — | — | 2026-09-10 | new coverage; roughly matches the existing 9.82x row (no big cold-vs-warm gap here, unlike Kokoro/Chatterbox) |
| F5-TTS Base (DiT) | text → 2.77s audio, cold CLI (`stingray tts -e f5tts -g vulkan`) | Vulkan iGPU | 59.58s | **21.48×** | — | — | — | 2026-09-10 | new coverage. **Real, substantial finding — Vulkan is ~2x slower than CPU here**, not a methodology artifact (the CPU number itself is consistent with the existing warm-benchmark row, unlike the Kokoro/Chatterbox cases). Diffusion-style DiT generation seems to be where this iGPU loses badly. |
| F5-TTS Base (Paragraph) | text → 14.5s audio | CPU | 10.20s | **0.70×** | — | — | — | 2026-09-09 | 2026-08-28 👂 |
| CosyVoice2-0.5B | text → 8.00s audio | CPU | 23.20s | **2.90×** | — (no `cosyvoice2` family in `examples/audio.cpp`'s registry — only `cosyvoice3`'s `llm_job` path is implemented there) | — | — | 2026-09-10 | new coverage; `CosyVoice2PerfBaselineDebugTest.cs` (temporary) |
| XTTS-v2 | text → 3.16s audio | CPU | 10.16s | **3.22×** | — (`examples/xtts_inference.cpp` is source-only, never built to an `.exe`) | — | — | 2026-09-10 | new coverage; `XttsPerfBaselineDebugTest.cs` (temporary, uses `b.wav` — the existing `Baseline_Xtts` test's named reference wav isn't present on this machine) |
| MOSS-TTS-Nano (100M) | text → 3.04s audio, 40 frames | CPU | 3.46s | **1.14×** | — (not attempted this pass) | — | — | 2026-09-10 | new coverage; existing `MossTtsGenerateWavDebugTest.cs` already had built-in perf metrics, just run as-is. Near-real-time — the smallest, fastest TTS checkpoint tested this pass. |
| PersonaPlex (7B LLM + Mimi codec, voice prompt) | text → 4.00s audio, 50 frames | CPU | ~1049s (total test 1099s minus ~50s load) | **~262×** | — (not attempted this pass) | — | — | 2026-09-10 | new coverage; existing `PersonaPlexGenerateWavDebugTest.cs`, timed externally (no built-in perf metrics). Real, sensible decoded text ("Hey, let me know if you have any questions.") — working, just very slow (25.5 GiB, largest TTS-class checkpoint tested this pass). |
| Higgs Audio TTS (4B LLM + codec) | text → 2.64s audio | CPU | 36.01s (mean of 3) | **13.64×** | — (not attempted this pass) | — | — | 2026-09-10 | new coverage; `HiggsAudioPerfBaselineDebugTest.cs` (temporary). Real, non-degenerate audio produced. |
| VoxCPM2 (LLM + residual-LM + CFM/DiT + AudioVAE) | text → 3 patches, 4 CFM steps | CPU | 15.8s (single run) | — (audio duration not computed — `maxPatches=3` smoke-test params, not a real full generation) | — | — | 2026-09-10 | new coverage; existing `VoxCpm2GeneratorRealWeightsTests.cs`, timed externally. Not RTF-comparable to other rows — reported wall time only. |
| NeuTTS-2E (speech-code generation only, no codec decode) | text → 32 speech codes | CPU | 6.42s | — (no audio duration — codes not decoded to waveform in this test) | — | — | 2026-09-10 | new coverage; existing `NeuTtsGeneratorRealWeightsTests.cs` already had built-in timing, just run as-is. Not RTF-comparable — reports code-generation time only, not full text-to-waveform. |
| OmniVoice (MaskGIT + acoustic decoder) | text → 3.20s audio, 12 MaskGIT steps | CPU | 43.05s (single run) | **~13.45×** | — (not attempted this pass) | — | — | 2026-09-10 | new coverage; existing `OmniVoiceGenerateWavDebugTest.cs`, timed externally (single run, not best-of-3). Real, full text-to-waveform generation. |
| VibeVoice-TTS 1.5B (LLM + speech-diffusion head) | text → 8.00s audio, 60 tokens | CPU | 101.27s (single run) | **~12.3×** | — (not attempted this pass) | — | — | 2026-09-10 | new coverage; existing `VibeVoiceTtsGenerateWavDebugTest.cs`, timed externally (single run, not best-of-3). Real, full text-to-waveform generation. |
| MusicGen-small (T5 + delayed-pattern + EnCodec) | text → 3.00s audio | CPU | 15.80s | **5.27×** | — (not attempted this pass) | — | — | 2026-09-10 | new coverage; `MusicGenPerfBaselineDebugTest.cs` (temporary). Non-degeneracy checked only — no numeric golden reference exists for this port yet, so treat as "real audio energy produced," not "musically/numerically correct." |
| AudioGen-medium (T5-large + delayed-pattern + EnCodec) | text → 3.00s audio | CPU | 93.48s (mean of 3) | **31.16×** | — (not attempted this pass) | — | — | 2026-09-10 | new coverage; `AudioGenPerfBaselineDebugTest.cs` (temporary). Non-degeneracy checked only, same caveat as MusicGen. |
| Stable Audio 3 Small Music (DiT + `taae_v2` VAE) | text → 6.00s audio, 8-step CFG | CPU | 152.19s (mean of 3) | **25.36×** | — (not attempted this pass) | — | — | 2026-09-10 | new coverage; `StableAudio3SmallMusicPerfBaselineDebugTest.cs` (temporary). 8 steps is a deliberately short smoke-test step count, not the model's recommended full schedule (per `docs/00-current-work.md`'s SA3 entries, real generations use 15-25+ steps) — this RTF would be meaningfully worse at a realistic step count, not better. |
| Stable Audio 3 Medium (differential-attention DiT + SAME-L VAE) | text → 4.00s audio, 15-step CFG | CPU | 387s (mean) | **96.75×** | — (not attempted) | — | — | 2026-09-03 (cited from prior session) | `docs/00-current-work.md`'s Stable Audio 3 Medium entry — ~97s wall-clock per second of audio, real generation timing, not a smoke test |
| ACE-Step Turbo (Qwen3 text encoder + DiT + Oobleck VAE) | text → 2.00s audio, 8-step CFG | CPU | 228.29s (mean of 3) | **114.14×** | — (not attempted) | — | — | 2026-09-10 | new coverage; `AceStepPerfBaselineDebugTest.cs` (temporary). **Worst RTF in this entire doc** — worse even than Voxtral's 85.5x — despite "Turbo" naming implying an 8-step fast schedule. Non-degeneracy checked only, no numeric golden reference exists yet. |

> RTF < 1.0x = faster than real-time. Piper (0.19x = 5.2× real-time), MMS-TTS (0.38x = 2.6× real-time), and Kokoro (0.89x) are faster than real-time on CPU.
> Autoregressive pipelines (QwenTTS, CosyVoice3, Chatterbox, Parler, FishSpeech) are compute-bound on CPU; GPU dispatch is expected to be the largest speedup.
> **Confirmed Working:** 🔬 = Golden-verified against reference; 👂 = Confirmed working by ear / transcription.
> Harness: `scripts/bench-audio.ps1` (`tests/OpenTail.Stingray.Tests.Audio/TtsPerformanceBaselineDebugTest.cs`) & `scripts/bench-cpp.ps1 -Suite Tts`.

---

## Forced Alignment / Audio Timing

| Model | Scenario | Backend | C# Wall | C# RTF | C++ Wall | C++ RTF | Ratio | Performance Check | Confirmed Working |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| Qwen3 Forced Aligner 0.6B | 14.1s audio alignment | CPU | 1.68s | **0.12×** | 1.71s | 0.12× | <span style="color:#16a34a">**1.01x**</span> | 2026-09-09 | 2026-09-02 🔬 |

> RTF < 1.0x = faster than real-time. Qwen3 Forced Aligner runs at **8.4x real-time** speed in C# with full AVX2/FMA vectorized Exact-Erf GELU matching `audio.cpp` C++ reference.
> Harness: `scripts/bench-cpp.ps1 -Suite Align`.

---

## TTS Streaming Latency (TTFA)

| Pipeline | Scenario | Backend | TTFA (C#) | Total Time | C++ (ref) | Ratio | Performance Check | Confirmed Working |
|---|---|---|---:|---:|---:|---:|---|---|
| FishSpeech S2 Pro (Stream) | Streaming TTFA (1-frame) | CPU | **0.634s** | 63.55s | — | — | 2026-09-09 | 2026-08-29 👂 |
| Piper lessac (Stream) | Streaming TTFA (16-frame) | CPU | **0.776s** | 6.47s | — | — | 2026-09-09 | 2026-09-03 👂 |
| MMS-TTS eng (Stream) | Streaming TTFA (16-frame) | CPU | **0.996s** | 16.40s | — | — | 2026-09-09 | 2026-08-30 🔬 |
| Parler-TTS Mini (Stream) | Streaming TTFA (16-frame) | CPU | **1.712s** | 17.40s | — | — | 2026-09-09 | 2026-08-28 👂 |
| QwenTTS 0.6B (Stream) | Streaming TTFA (1-frame) | CPU | **2.728s** | 72.53s | — | — | 2026-09-09 | 2026-08-29 👂 |
| MeloTTS (Stream) | Streaming TTFA | CPU | **6.540s** | 6.55s | — | — | 2026-09-09 | 2026-09-03 👂 |
| Kokoro-82M (Stream) | Streaming TTFA (1-chunk) | CPU | **6.592s** | 6.60s | — | — | 2026-09-09 | 2026-09-03 👂 |
| F5-TTS (Stream) | Streaming TTFA | CPU | **27.034s** | 27.03s | — | — | 2026-09-09 | 2026-08-28 👂 |
| Chatterbox Turbo (Stream) | Streaming TTFA | CPU | **13.669s** | 13.67s | — | — | 2026-09-09 | 2026-08-30 🔬 |

> **Why every C++ (ref) cell here is blank (checked 2026-09-10):** `audiocpp_cli` genuinely supports
> `--mode streaming` as a flag, but every family actually tried through it refuses at runtime:
> `qwen3_tts` → "Qwen3 TTS only supports offline sessions"; `cosyvoice3` → "CosyVoice3 supports
> offline sessions" (both explicit, deliberate refusals, not bugs); `chatterbox_turbo` failed
> earlier on a missing tokenizer asset (`chatterbox_turbo_vocab.json`) before even reaching that
> question. `--metrics` also flatly refuses in streaming mode ("`--metrics` currently supports
> offline mode only"), so even a family that *did* accept `--mode streaming` wouldn't emit the
> same wall/RTF metrics used elsewhere in this doc.
>
> **One real exception found: `examples/s2.cpp/build/bin/s2.exe` (FishSpeech's own binary, not
> `audiocpp_cli`) has a working `--stream-file` path with genuine streaming metrics.** A real run
> (`s2.exe -m s2-pro-q4_k_m.gguf --stream-file`, 2026-09-10) produced:
> `stride=16 frames, holdback=144 frames, ref_encode=49957ms, kv_init=25.8ms, generate=170348ms,
> stream_decode=20419ms, total=190800ms, total_rtf=61.32`. This is real and usable, but it does
> **not** print an explicit "time to first chunk emitted" — computing a TTFA-equivalent from
> `ref_encode + kv_init + (holdback-frames worth of AR generation)` would be an inference, not a
> measurement, and risks silently not matching OT's own TTFA definition (prompt-ingestion → first
> playable chunk). Left this un-computed rather than presenting an inferred number as a real one —
> the CLI itself would need a printed first-chunk timestamp for a trustworthy comparison.

> **TTFA (Time-To-First-Audio):** Latency from prompt ingestion to the first playable audio chunk emitted.
> **Confirmed Working:** 🔬 = Golden-verified against reference; 👂 = Confirmed working by ear / transcription.
> Harness: `scripts/bench-audio.ps1 -Suite Streaming`.

---

## ASR / Speech Recognition (Whisper GGUF)

| Model | Scenario | Backend | C# Wall | C# RTF | C++ Wall | C++ RTF | Ratio | Performance Check | Source |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| Whisper Tiny (39M, HF safetensors) | 14.1s audio transcribe | CPU | 1.36s (mean of 3) | **0.097x** | — (only a 575KB CI-stub `.bin` exists in `examples/whisper.cpp/models`, not the real checkpoint) | — | — | 2026-09-10 | new coverage; `WhisperTinyPerfBaselineDebugTest.cs` (temporary). Correct transcript (matches reference text), 10.3x real-time. |
| Whisper Base (39M) | 14.1s audio transcribe | CPU | 0.84s | **0.070x** | 0.82s | 0.058x | **0.83x** | 2026-09-09 | scripts/bench-cpp.ps1 |
| Whisper Small (244M) | 14.1s audio transcribe | CPU | 2.42s | **0.202x** | 2.36s | 0.168x | **0.83x** | 2026-09-09 | scripts/bench-cpp.ps1 |
| Whisper Medium (769M) | 14.1s audio transcribe | CPU | 6.71s | **0.560x** | 6.93s | 0.492x | **0.88x** | 2026-09-09 | scripts/bench-cpp.ps1 |
| Whisper Large-v3 (1.5B) | 14.1s audio transcribe | CPU | 11.32s | **0.943x** | 12.69s | 0.900x | **0.95x** | 2026-09-09 | scripts/bench-cpp.ps1 |
| Voxtral-Mini-4B-Realtime | 14.1s audio transcribe | CPU | 1203.1s (mean of 3) | **85.5x** | 24.05s | 1.71x | <span style="color:#dc2626">**0.02x**</span> | 2026-09-10 | new coverage; raw building-block harness (`VoxtralAudioEncoder`/`VoxtralTextDecoder`, no CLI pipeline exists yet), `examples/audio.cpp/build/bin/audiocpp_cli.exe --family voxtral_realtime` |
| Qwen3-ASR 0.6B (safetensors) | 14.1s audio transcribe | CPU | 3.16s (mean of 3) | **0.225x** | — (no C++ reference attempted this pass) | — | — | 2026-09-10 | new coverage; `QwenAsrPerfBaselineDebugTest.cs` (temporary). **Caveat:** transcript is degenerate ("aspects" only, not the real reference text) — fast but likely a real correctness bug in this pipeline/request path, not a working transcription. Timing is real; do not read this row as "Qwen3-ASR works." |
| VibeVoice-ASR (7B-class LLM + acoustic/semantic tokenizers) | 14.1s audio transcribe | CPU | 151.19s (mean of 3) | **10.74x** | — (not attempted this pass) | — | — | 2026-09-10 | new coverage; `VibeVoiceAsrPerfBaselineDebugTest.cs` (temporary). Correct transcript (matches reference text). Slow — real, large (9 GiB) LLM-based ASR, not yet CLI-wired. |
| Citrinet-ASR (Jasper-style conv encoder) | ~3.5s LibriSpeech clip transcribe | CPU | 0.872s (single run) | **~0.25×** | — (not attempted this pass) | — | — | 2026-09-10 | new coverage; existing `CitrinetAsrRealWeightsTests.cs` already had built-in timing, just run as-is. **Correct, coherent transcript** ("concord returned to its place amidst the tents") on real LibriSpeech audio — a real, fast, working ASR pipeline, unlike Qwen3-ASR/FunASR-Nano's degenerate output. |
| Nemotron 3.5 ASR Streaming 0.6B | real audio transcribe (duration not measured this pass) | CPU | 38.05s (single run) | — (audio duration unknown, not RTF-comparable) | — | — | 2026-09-10 | new coverage; existing `NemotronAsrEndToEndTests.cs`, timed externally. **Correct, coherent transcript** ("This little work was finished in the year eighteen oh three and intended for immediate publication.") — a second real, working ASR pipeline this pass, alongside Citrinet-ASR. |
| NVIDIA NeMo Parakeet-CTC 0.6B | 14.1s audio transcribe | CPU | 3.56s (mean of 3) | **0.253×** | — (not attempted this pass) | — | — | 2026-09-10 | new coverage; `ParakeetPerfBaselineDebugTest.cs` (temporary). **Correct transcript** (matches reference text, sans punctuation/casing). Third genuinely-working ASR pipeline this pass. Required adding a `models/parakeet-ctc-0.6b-q4_k.gguf` symlink (existing test only checked `models/`, checkpoint lives in `models/_models/`) — its own test was silently no-op'ing before that, per CLAUDE.md rule 12's known pattern. |
| FunASR-Nano (Paraformer-based, `paraformer-q8.gguf`) | synthetic-tone audio, 99 tokens | CPU | 26.24s (single run) | — (not RTF-comparable, synthetic audio not real speech) | — | — | 2026-09-10 | new coverage; existing `FunAsrNanoEndToEndTests.cs`, timed externally. **Caveat, corrected 2026-09-11:** output is degenerate word-salad ("to to to to... a a a at at..."), but the test harness feeds `new Random(0)`-generated synthetic tone, not real speech (confirmed by reading `FunAsrNanoEndToEndTests.cs:106`) — unlike Qwen3-ASR's bug below, which was found on a **real speech** clip. A real speech model given non-speech audio producing garbage is expected, not necessarily a bug — downgraded from "confirmed correctness gap" to "inconclusive pending a real-speech re-test." |
| Silero VAD (ONNX) | 12s synthetic audio, segment detection | CPU | 73.24ms (mean of 8) | **0.0061×** | — (not attempted this pass) | — | — | 2026-09-10 | new coverage; existing `SileroVadPerfBenchTests.cs`, just needed a `models/silero_vad.onnx` symlink to `models/_models/silero_vad.onnx` (added, matching the existing symlink convention). **164x real-time — the fastest pipeline measured anywhere in this doc.** |
| MarbleNet VAD (safetensors, bundled in-repo) | segment detection, single run | CPU | 109ms | — (input duration not computed this pass, not RTF-comparable) | — | — | 2026-09-10 | new coverage; existing `MarbleNetVadRealWeightsTests.cs`, run as-is. Real, plausible segment detected (`[8320-56320] conf=0.978, 3.00s`) — a second real, working VAD model alongside Silero. |
| Orpheus-3B TTS + SNAC vocoder | transformer decode + vocoder, 6 runs | Vulkan iGPU (auto-selected) | transformer 14.4 tok/s mean, vocoder RTF 5.3x mean, total ~10.0s/run | — (not attempted on CPU this pass) | — | — | 2026-09-10 | new coverage; existing `OrpheusFullPipelinePerfBenchTests.cs`, needed `models/orpheus-3b-0.1-ft.Q4_K_M.gguf` + `models/snac-24khz.gguf` symlinks (checkpoints live in `models/_models/`, another instance of the silent-no-op-via-missing-symlink pattern — this one wasn't silent, it printed a clear "GGUFs not found" skip message). Test auto-selected Vulkan GPU without being asked. |
| ~~SenseVoice~~ | ~3.5s LibriSpeech clip transcribe | CPU | 2229ms (single run) | **~0.64×** | — (not attempted this pass) | — | — | 2026-09-11 | **NEW pipeline, implemented and verified this session** (`src/OpenTail.Stingray.Audio/SenseVoice/`, real ONNX forward + real fbank/LFR/CMVN + real CTC decode, per `docs/00-current-work.md`'s ONNX-expansion plan). **Exact correct transcript** on the same real LibriSpeech clip Citrinet-ASR used: `"concord returned to its place amidst the tents"` — a real, fully working ASR pipeline on the first attempt, not just non-crashing. Real language/emotion/event tags also output correctly (`en`/unknown/speech). Was previously 0% implemented (README's prose-only claim, no code at all) |
| FunASR Paraformer (ONNX path) | real Mandarin speech clip transcribe | CPU | 2171ms (single run) | — (audio duration not measured, not RTF-comparable) | — | — | 2026-09-11 | **NEW independent path, implemented and verified this session** (`src/OpenTail.Stingray.Audio/ParaformerOnnx/`, separate from the broken native GGUF path — real single-graph ONNX forward + real fbank/LFR/CMVN + real per-position-argmax decode). Real test clip fetched from the same real HF repo this checkpoint's vocab came from (`csukuangfj/sherpa-onnx-paraformer-zh-small-2024-03-09/test_wavs/0.wav`). **Real, coherent, grammatically valid Mandarin transcript**: `"对我做了介绍啊那么我想说的是呢大家如果对我的研究感兴趣呢嗯"` — no exact ground-truth string to assert against, but unambiguously real, sensible speech content, not garbage. Gives this project a working Paraformer route independent of the confirmed-broken GGUF conversion |

> RTF < 1.0x = faster than real-time. Whisper Base runs at **14.3x real-time** speed on CPU; Small at **5.0x real-time**; Medium at **1.79x real-time**; Large-v3 at **1.06x real-time** (faster than real-time on CPU).
> **Voxtral-Mini-4B is 50x slower than its C++ reference** — the worst ratio anywhere in this doc. Transcript is correct (matches the reference text, modulo the model's own real streaming control markers), so this is a genuine performance gap, not a correctness bug: the 4B dense text decoder currently only has raw building blocks (`VoxtralTextDecoder.Step`/`PrefillWithCache`) wired up for testing, no CLI, and almost certainly no batched/optimized decode path — unlike Whisper, which is a mature, tuned pipeline. Worth a dedicated look given the ratio.
> Harness: `scripts/bench-audio.ps1` (`tests/OpenTail.Stingray.Tests.Audio/WhisperFullPipelinePerfBenchTests.cs`); Voxtral timing via a new temporary `VoxtralPerfBaselineDebugTest.cs`.

---

## Embeddings (CPU)

**History:** an earlier pass in this doc claimed "Qwen3-Embedding-0.6B, 38,900 tok/s" — that was
**wrong**, caught and retracted 2026-09-10 after a second model produced byte-identical output.
Root cause: `stingray embed`'s GGUF path (`EmbeddingEngine.cs:147-162`) is a hash-based synthetic
stub that never loads any model at all — **this is still true and still unfixed**; no GGUF
embedding measurement is possible with this CLI. See Known Measurement Gaps.

**Fixed 2026-09-11 — the ONNX path, in two passes.** `stingray embed -m <file>.onnx` genuinely
invokes ONNX Runtime (confirmed by real, model-specific output dimensions below — not a stub), but
crashed with `Missing Input: token_type_ids` because `EmbedCommand.cs`'s ONNX branch never
constructed that tensor. Fixed by adding an all-zero `token_type_ids` input alongside
`input_ids`/`attention_mask`. Also fixed in the same pass: `stingray embed -o <file>` crashed with
`System.InvalidOperationException: Reflection-based serialization has been disabled` (NativeAOT/
trim violation, `CLAUDE.md` rule 4) — replaced `JsonSerializer.Serialize` with a hand-rolled JSON
writer. **The tokenizer itself was fixed for real later the same day**, per explicit user request
("fix the fake ONNX tokenizer for real. I want it to work properly"): the ONNX branch's
"tokenization" had been mapping each raw character to its char code as a placeholder token id, not
real WordPiece. Wrote `BertWordPieceTokenizer.cs` (a faithful port of HuggingFace's real
`BasicTokenizer`+`WordpieceTokenizer` algorithm, confirmed against each checkpoint's own real
`tokenizer_config.json`), downloaded the real `vocab.txt` (30522 tokens) for all 4 local
checkpoints from their real HuggingFace repos, and wired it in with automatic vocab-file discovery.
**Verified with a real semantic-correctness check**: two paraphrased sentences scored 0.72 cosine
similarity; an unrelated sentence scored only 0.15 against the same reference — the embeddings are
now genuinely semantically meaningful, not just non-crashing. Token counts below reflect the real
tokenizer (12 tokens for the benchmark sentence, vs. the old fake tokenizer's inflated 39).

| Model | Scenario | Backend | C# result | C++ reference | Ratio | Performance Check | Source |
|---|---|---|---|---|---:|---|---|
| all-MiniLM-L6-v2 (quantized ONNX) | 1 text, 12 real tokens (WordPiece), mean pooling, 384-dim (real native dim) | CPU | 26ms (best of 3) | — (not attempted this pass) | — | 2026-09-11 | real WordPiece tokenizer, post-fix; `stingray embed -m all-MiniLM-L6-v2_quantized.onnx -p "Hello, I will make some lunch, darling!"` |
| bge-small-en-v1.5 (quantized ONNX) | 1 text, 12 real tokens, mean pooling, 384-dim | CPU | 28ms (best of 3) | — (not attempted this pass) | — | 2026-09-11 | real WordPiece tokenizer, post-fix |
| bge-base-en-v1.5 (quantized ONNX) | 1 text, 12 real tokens, mean pooling, 768-dim (implied by base-size BERT) | CPU | 30ms (best of 3) | — (not attempted this pass) | — | 2026-09-11 | real WordPiece tokenizer, post-fix |
| bge-large-en-v1.5 (quantized ONNX) | 1 text, 12 real tokens, mean pooling, 1024-dim (real native dim) | CPU | 41ms (best of 3) | — (not attempted this pass) | — | 2026-09-11 | real WordPiece tokenizer, post-fix |

> All four report their real, correct native output dimension (384/384/768/1024) rather than the
> GGUF stub's hardcoded 1536 — direct evidence these are genuine forward passes, not stub output.

---

## Vision-Language Model Text Backbones (CPU, text-only — no image input)

Not a vision-encoding measurement — this is the LLM text backbone underneath a VLM, run in plain
text mode (no `--image`/`--mmproj`), included as new LLM throughput coverage since the checkpoint
was on hand and untested. A real vision-encode measurement would need the image path exercised
separately.

| Model | Scenario | Backend | C# (OT, t/s) | C++ (llama.cpp, t/s) | Ratio | Performance Check | Source |
|---|---|---|---|---:|---:|---|---|
| InternVL3-2B Q4_K_M (Qwen2-1.5B backbone) | prefill (503 tok) | CPU | 131.8 t/s | 168.90 t/s | **0.78x** | 2026-09-10 | new coverage; stingray CLI + llama-bench, best-of-3, text-only |
| InternVL3-2B Q4_K_M (Qwen2-1.5B backbone) | decode (503 tok prompt, 24 tok gen) | CPU | 18.3 t/s | 37.94 t/s | **0.48x** | 2026-09-10 | new coverage; stingray CLI + llama-bench, best-of-3, text-only |
| InternVL3-2B Q4_K_M (Qwen2-1.5B backbone) | prefill (503 tok) | Vulkan iGPU | 23.2 t/s | — | — | 2026-09-10 | new coverage; no llama.cpp Vulkan ref, text-only. Worse than CPU (23.2 vs 131.8 t/s) — small model, dispatch-overhead-dominated, the usual pattern |
| InternVL3-2B Q4_K_M (Qwen2-1.5B backbone) | decode (503 tok prompt, 24 tok gen) | Vulkan iGPU | 21.8 t/s | — | — | 2026-09-10 | new coverage; beats CPU decode (21.8 vs 18.3 t/s) |
| Granite-4.0-3B-Vision Q4_K_M (Granite backbone) | prefill (494 tok) | CPU | 58.4 t/s | 69.82 t/s | **0.84x** | 2026-09-10 | new coverage; stingray CLI + llama-bench, best-of-3, text-only |
| Granite-4.0-3B-Vision Q4_K_M (Granite backbone) | decode (494 tok prompt, 24 tok gen) | CPU | 8.7 t/s | 18.09 t/s | **0.48x** | 2026-09-10 | new coverage; stingray CLI + llama-bench, best-of-3, text-only |
| ~~Granite-4.0-3B-Vision Q4_K_M | prefill (494 tok) | Vulkan iGPU | 41.2 t/s (garbled output)~~ | — | — | 2026-09-10 | **FIXED 2026-09-11** — see below for the corrected re-measurement. Original finding kept struck through, not deleted, per this doc's retraction discipline. |
| Granite-4.0-3B-Vision Q4_K_M (Granite backbone) | prefill (494 tok) | Vulkan iGPU | 44.1 t/s | — | — | 2026-09-11 | **Re-measured post-fix.** Output is now coherent English matching CPU exactly ("OpenTail.Stingray is a high-performance local inference engine designed for running large language models, vision-language models, and") — the Vulkan/Granite correctness bug is fixed, see Known Measurement Gaps for the fix details. Real perf note: 44.1 t/s vs the pre-fix garbled-output run's 41.2 t/s — no meaningful slowdown from the added scale ops. |
| Granite-4.0-3B-Vision Q4_K_M (Granite backbone) | decode (494 tok prompt, 24 tok gen) | Vulkan iGPU | 10.8 t/s | — | — | 2026-09-11 | Re-measured post-fix, coherent output, close to CPU decode (10.8 vs 8.7 t/s) |
| Granite-Vision-3.2-2B Q3_K_S (Granite backbone) | prefill (573 tok) | CPU | 27.0 t/s | 66.50 t/s | **0.41x** | 2026-09-10 | new coverage; stingray CLI + llama-bench, best-of-3, text-only |
| Granite-Vision-3.2-2B Q3_K_S (Granite backbone) | decode (573 tok prompt, 24 tok gen) | CPU | 17.1 t/s | 35.65 t/s | **0.48x** | 2026-09-10 | new coverage; stingray CLI + llama-bench, best-of-3, text-only |
| ~~Granite-Vision-3.2-2B Q3_K_S | prefill+decode | Vulkan iGPU | 4.2/3.8 t/s, 1 token (degenerate)~~ | — | — | 2026-09-10 | **FIXED 2026-09-11** — see below. |
| Granite-Vision-3.2-2B Q3_K_S (Granite backbone) | prefill (573 tok) | Vulkan iGPU | 4.1 t/s | — | — | 2026-09-11 | **Re-measured post-fix.** Coherent output matching CPU exactly ("1. OpenTail.Stingray is a high-performance local inference engine for running large language models, vision-"). Full 24-token generation completed (vs. 1 token before the fix). |
| Granite-Vision-3.2-2B Q3_K_S (Granite backbone) | decode (573 tok prompt, 24 tok gen) | Vulkan iGPU | 4.1 t/s | — | — | 2026-09-11 | Re-measured post-fix. Notably low decode ratio vs CPU (17.1 t/s) — Vulkan is much worse here, unlike the 3B-Vision checkpoint above where Vulkan and CPU were close. Worth another look, but the correctness bug (the more serious issue) is confirmed fixed. |
| dots.ocr Q8_0 (Qwen2-1.5B backbone) | prefill (477 tok) | CPU | 96.3 t/s | 135.37 t/s | **0.71x** | 2026-09-10 | new coverage; stingray CLI + llama-bench, best-of-3, text-only |
| dots.ocr Q8_0 (Qwen2-1.5B backbone) | decode (477 tok prompt, 24 tok gen) | CPU | 20.5 t/s | 25.97 t/s | **0.79x** | 2026-09-10 | new coverage; stingray CLI + llama-bench, best-of-3, text-only |
| dots.ocr Q8_0 (Qwen2-1.5B backbone) | prefill (477 tok) | Vulkan iGPU | 13.8 t/s | — | — | 2026-09-10 | new coverage; no llama.cpp Vulkan ref, text-only. Worse than CPU (13.8 vs 96.3 t/s) |
| dots.ocr Q8_0 (Qwen2-1.5B backbone) | decode (477 tok prompt, 24 tok gen) | Vulkan iGPU | 12.9 t/s | — | — | 2026-09-10 | new coverage; also worse than CPU (12.9 vs 20.5 t/s) — unlike InternVL3-2B, this one loses on both metrics on Vulkan |

---

## Vision-Language Model Real Image Encoding (CPU) — first real `--image` measurements

Unlike the text-backbone-only table above, these runs feed a real image through `--image`/`--mmproj`
and produce a real image-grounded text description — the actual vision-encode path, not just the
LLM backbone. Image used: `docs/diffusion-samples/ltx_test_apple_256.png` (256×256, an existing
repo asset, no new download needed). Prompt: `"Describe this image in one sentence."`, `-n 30`,
`--temp 0` (both sides), CPU only.

| Model | Scenario | Tool | Result | Performance Check | Source |
|---|---|---|---|---|---|
| InternVL3-2B Q4_K_M + mmproj-q8_0 | prefill (294 tok = 256 image + 38 text) | stingray CLI (OT) | 28.1 t/s avg (28.4/28.1/27.8 across 3 runs) | 2026-09-11 | new coverage; real vision-encode path, `--image`/`--mmproj`, best-of-3 |
| InternVL3-2B Q4_K_M + mmproj-q8_0 | decode (23-30 tok gen) | stingray CLI (OT) | 27.4 t/s avg (27.2/27.3/27.6 across 3 runs) | 2026-09-11 | new coverage; combined prefill+decode t/s reported by OT's own `Prefill:`/`Decode:` output lines |
| InternVL3-2B Q4_K_M + mmproj-q8_0 | vision-encoder portion only | `llama-mtmd-cli.exe` (C++ reference) | 2488–3226ms (3 runs: 2904, 2488, 3226ms) | 2026-09-11 | **not directly comparable to the OT numbers above** — see caveat below |
| Granite-4.0-3B-Vision Q4_K_M + mmproj-f16 | prefill (614 tok = 576 image + 38 text) | stingray CLI (OT) | 13.5 t/s avg (13.6/13.6/13.5 across 3 runs) | 2026-09-11 | new coverage; real vision-encode path runs and produces real timing, but see correctness caveat below |
| Granite-4.0-3B-Vision Q4_K_M + mmproj-f16 | decode (10 tok gen) | stingray CLI (OT) | 12.1 t/s avg (10.0/13.2/13.1 across 3 runs) | 2026-09-11 | same caveat |

**Correctness caveat for Granite-4.0-3B-Vision:** all 3 runs produced the same degenerate,
non-image-grounded output — `"This image is a description of the provided text."` — instead of an
actual description of the picture's content (unlike InternVL3-2B above, which correctly described
the real colors/shapes in the same image). The vision encoder does run (576 soft tokens/2560-dim
reported, real non-trivial prefill/decode timing, consistent across 3 runs) but the generated text
suggests the image embeddings aren't being attended to correctly by this checkpoint's backbone, or
its prompt-template wiring for the image placeholder differs from InternVL3's. Logged as a real,
reproducible bug in `docs/00-current-work.md` rather than silently reported as a working
measurement — the timing is real, the correctness is not verified. **ROOT-CAUSED 2026-09-11**: the
projector (`Granite4VisionEncoder.cs`) is a simplified single-linear-layer approximation of the
real reference's multi-block windowed QFormer, and isn't even wired to the real GGUF tensor names —
real inference silently falls back to feeding raw, untrained SigLIP hidden states as "image
tokens." A real fix needs a substantial projector rewrite, not a quick patch. See
`docs/00-current-work.md`'s 2026-09-11 entry for full detail.

| Granite-Vision-3.2-2B Q3_K_S + mmproj-f16 | prefill (785 tok = 729 image + 56 text) | stingray CLI (OT) | 18.3 t/s avg (19.2/17.6/18.1 across 3 runs) | 2026-09-11 | new coverage; real vision path runs, same correctness caveat family as Granite-4.0-3B-Vision above |
| Granite-Vision-3.2-2B Q3_K_S + mmproj-f16 | decode (26 tok gen) | stingray CLI (OT) | 16.1 t/s avg (17.6/15.3/15.4 across 3 runs) | 2026-09-11 | same caveat |

**Correctness caveat for Granite-Vision-3.2-2B:** same failure family as Granite-4.0-3B-Vision —
all 3 runs produced the identical degenerate output (`"The image contains a series of words and
phrases, including 'color,' 'story,' 'identity,' and various combinations thereof."`), completely
unrelated to the actual image content, and identical byte-for-byte across all 3 runs (deterministic
given `--temp 0`, but wrong). Both Granite-family VLMs failing the same way (real timing, wrong/
generic output) while InternVL3-2B works correctly on the identical image/prompt suggests a
**Granite-family-specific** vision-integration bug (e.g. image placeholder token handling or
embedding injection point in the Granite chat template/forward path), not two unrelated one-off
issues. Worth root-causing as a single Granite-vision bug rather than two separate ones.

| dots.ocr Q8_0 + mmproj-Q8_0 | prefill (93 tok = 81 image + 12 text) | stingray CLI (OT) | 21.2 t/s | 2026-09-11 | new coverage; vision encoder runs (81 soft tokens) but decode emits `<\|endofassistant\|>` immediately — 1-token degenerate output, likely a chat-template/stop-condition issue specific to this OCR-focused checkpoint's prompt formatting rather than the vision path itself |
| Gemma-3-4B-it Q4_K_M + mmproj-f16 | prefill (274 tok = 256 image + 18 text) | stingray CLI (OT) | 11.6 t/s avg (11.2/12.4/11.2 across 3 runs) | 2026-09-11 | new coverage; real, working vision-encode — genuinely describes the image ("a distorted, vibrant portrait of a person with a red face and dark hair, rendered in an intensely pixelated style") |
| Gemma-3-4B-it Q4_K_M + mmproj-f16 | decode (27 tok gen) | stingray CLI (OT) | 11.1 t/s avg (12.5/10.4/~10.8 across 3 runs) | 2026-09-11 | same run; a real, second confirmed-working vision-encode checkpoint alongside InternVL3-2B. (Unrelated note: a Jinja chat-template gap was logged for this checkpoint's `<start_of_turn>` role-concat expression — passed through unevaluated, doesn't affect this measurement's validity since output was still correct.) |

**Methodology caveat (same discipline as the earlier Kokoro/Piper cold-CLI-vs-warm-benchmark
note):** `llama-mtmd-cli.exe`'s only timing output is `mtmd batch encoding done in N ms`, which
measures the **vision-encoder-only** portion of the pipeline (turning the 256 image tokens into
embeddings), not full generation. OT's `Prefill:`/`Decode: t/s` figures are a **combined**
prefill+decode throughput number across all 294 tokens (image + text) plus the generated tokens.
These measure different, non-overlapping slices of the same pipeline — there is no valid way to
turn llama-mtmd-cli's single number into a ratio against OT's t/s without fabricating an
apples-to-oranges comparison, so none is given. What both sides *do* independently confirm: this
is a real, working, correctness-verified vision-encode path on both C# (OT) and C++ (llama.cpp)
for this checkpoint — both produced plausible, image-grounded descriptions of the actual picture
content (OT: "a colorful, abstract composition featuring vibrant reds, yellows, and blues with
some black shapes"; llama-mtmd-cli: "a distorted, colorful abstract scene with a prominent red
shape in the center"). This is the first real vision-encode-path measurement recorded anywhere in
this document — prior VLM rows were text-backbone-only (no image input at all).

### Unsupported-architecture vision attempts (`--allow-unverified-arch`), no usable timing

The remaining Phase 1 checkpoints all previously had their text backbone rejected as an unverified
architecture. Retried here with `--allow-unverified-arch` + `--image`/`--mmproj` per the expansion
plan — all either crashed or produced unusable garbled output, so **no valid t/s numbers exist for
these**; each is a real, reproducible gap logged in `docs/00-current-work.md`, not a missing
measurement to chase further right now:

| Checkpoint | Architecture | Result |
|---|---|---|
| Kimi-VL-A3B-thinking Q2_K | `deepseek2` | **Crash**: `Missing tensor: blk.0.attn_kv_b.weight` (also hit by YouTu-VL-4B below — same class). **Corrected 2026-09-11**: this checkpoint's `attn_q.weight` DOES exist (confirmed via `list-tensors`) — it's the "Lite" `q_lora_rank==0` variant, not full-size MLA. The real, narrower gap: this checkpoint uses the split `attn_k_b`/`attn_v_b`/`attn_kv_a_mqa` "absorption" K/V layout, which this codebase's MLA code doesn't read — only the legacy fused `attn_kv_b` tensor is handled |
| MiMo-VL-7B-sft Q2_K | `qwen2vl` | ~~**Crash**: `ArgumentOutOfRangeException`~~ **FIXED 2026-09-11 — now a clean error, not a crash.** Root cause: `RunImagePrompt` used the text backbone's `hp.EmbeddingDim` (4096) as the per-token stride into the vision buffer instead of the vision embedder's own reported width. This checkpoint's mmproj genuinely projects to 3584-dim, not 4096 — a real GGUF-conversion mismatch, not something fixable from this codebase. Now reported as `Error: vision projector (qwen2.5vl_merger) outputs 3584-dim embeddings but the text backbone expects 4096-dim input` instead of crashing |
| Nemotron-Nano-12B-v2-VL Q2_K | `nemotron_h` | **Crash**: `HybridGdnForwardPass dense FFN requires hp.IntermediateDim > 0`. **Root-caused 2026-09-11**: the real metadata key (`nemotron_h.feed_forward_length`) exists but is a per-layer array (0 = pure Mamba/SSM layer, no dense FFN), not the single scalar this codebase's hyperparameter model assumes — a real, moderate feature addition (new per-layer hyperparameter + dispatch logic), not fixed yet |
| DeepSeek-OCR-2 Q4_K_M | `deepseek2-ocr` | **Runs**, real timing (41.9 t/s prefill / 30.9 t/s decode, 4096 image + 7 text tokens) but fully garbled mixed-language output — per the flag's own warning, not usable as a real measurement |
| PaddleOCR-VL-1.6 | `paddleocr` | **Runs**, real timing (51.5 t/s prefill / 39.2 t/s decode) but output is 4 repeated newline-byte tokens — degenerate |
| YouTu-VL-4B Q8_0 | `deepseek2` | **Crash**: `Missing tensor: blk.0.attn_kv_b.weight` — same class as Kimi-VL-A3B-thinking above (split K/V absorption layout, not the fused tensor this codebase reads) |
| Step3-VL-10B Q2_K | (admitted arch, no warning) | ~~**Crash**: identical `ArgumentOutOfRangeException`~~ **FIXED 2026-09-11 — real bug, real fix.** Root cause (distinct from MiMo-VL's, despite the identical crash symptom): `Step3VlVisionEncoder` looked up the final projector tensor under a made-up name (`mm.model_proj.weight`), which never matched this checkpoint's real GGUF tensor (`mm.model.fc.weight`, confirmed against `examples/llama.cpp`'s real `clip.cpp`/`clip-impl.h` reference). The wrong name meant the tensor was never found, silently narrowing the returned buffer while the reported width stayed wide — the exact same width-mismatch failure mode as MiMo-VL's bug, but from a genuine tensor-name typo. Fixed the name; now runs to completion: 10.0/9.9 t/s prefill/decode (3 runs), garbled output as expected for a Q2_K quant on an unverified architecture, but the crash itself is gone |

**Both fixed 2026-09-11.** The identical crash symptom across two unrelated architectures had two
DIFFERENT real root causes — one a genuine upstream checkpoint-conversion defect (MiMo-VL, not
fixable here), one a real bug in this codebase (Step3-VL's tensor-name typo, now fixed). The
underlying design flaw both symptoms shared — `RunImagePrompt` trusting `hp.EmbeddingDim` instead
of validating against the vision embedder's own reported width — is fixed for all architectures
going forward, not just these two.

---

## Vision Encoder (CPU)

| Component | Scenario | Backend | C# result | C++ reference | Ratio | Performance Check | Source |
|---|---|---|---|---|---:|---|---|
| VisionOps.Attention / AttentionGqa | 1024-tok / 16-head ViT-L | CPU | >1.2× over scalar | — | — | 2026-08-20 | perf-loop-project-review-progress.md |

> Scalar reference kept in `VisionOpsBenchmarkTests.cs` as a permanent regression baseline.

---

## Image & Video Diffusion (CPU + Vulkan iGPU) — out of core scope, tried anyway

Different domain from this doc's LLM/TTS/ASR focus (a different pipeline, `OpenTail.Stingray.Diffusion`,
not benchmarked here systematically) — included as real data points since the checkpoints were on
hand and untested.

| Model | Scenario | Backend | C# Wall | Performance Check | Source |
|---|---|---|---:|---|---|
| Z-Image-Turbo (S3-DiT + Qwen3-4B text encoder) | 512×512 image, default steps | CPU | 871.8s (14m32s) | 2026-09-10 | timing citation only — **correctness NOT verified at the time** ("non-trivial" meant "not a degenerate tiny file," not "visually inspected"). Re-tested 2026-09-11: this exact config produces pure visual noise — see the real regression writeup below and in `docs/00-current-work.md` |
| Z-Image-Turbo (S3-DiT + Qwen3-4B text encoder) | 512×512, default steps | Vulkan iGPU | 472.7s (~7.9min) | 2026-09-11 | new Vulkan timing (~1.8x faster than the CPU row above) — but **output is pure visual noise, not coherent**, confirmed by direct visual inspection (unlike the CPU row's unverified "non-trivial" claim). See the regression writeup below |
| ~~**Z-Image-Turbo: real regression, pure noise on both backends**~~ | 256×256/4 steps (the known-good config) AND 512×512/default | CPU + Vulkan | — | 2026-09-11 | **Real, serious, NOT fixed.** Was previously verified working 2026-09-01 (real sample: `docs/diffusion-samples/z-image-turbo_red-apple-on-white-table_CPU-256x256-4steps_GOOD.png`). Re-tested the exact same config today: pure noise on both CPU and Vulkan. Ruled out directly: not resolution-specific (noise at 256×256 and 512×512), not Vulkan-specific (CPU produces near-identical noise, ruling out this session's own `Synchronize()`-removal changes — verified by briefly reverting them and confirming CPU output unaffected), not text-encoder-checkpoint-specific (`Qwen3-4B-Q4_K_M.gguf` vs `Z-Image-AbliteratedV1.Q5_K_M.gguf` produce near-identical noise for the same seed). Real cause not found — needs a dedicated bisect from the 2026-09-01 known-good commit, not a quick fix. README downgraded 🟢→🔴. Full writeup in `docs/00-current-work.md` |
| Wan2.1-T2V-1.3B (DiT + UMT5-XXL text encoder + VAE) | 512×512, 2 video frames, 20 denoising steps | CPU | 4238.7s (70.6 min) | 2026-09-11 | new coverage; `stingray image --video-frames 2` CLI (video generation is routed through the same `image` command). Real, non-trivial 692KB PNG output saved to `docs/diffusion-samples/wan2.1-t2v-1.3b-perfleague-check.png`. **~4.9x slower than Z-Image-Turbo's single image** despite only 2 frames — real cost of the larger UMT5-XXL text encoder plus video-specific DiT attention, not just "more frames." |
| SD3.5-medium (Q4_K_M) | 256×256, 20 steps, seed 42 | CPU | 656.9s (~11m) | 2026-09-02 (pre-fix), see caveat | citation from `docs/057-sd35-performance-handoff.md`; no new run needed per this doc's own backfill note. **Caveat: this timing predates 4 real correctness bugs found and fixed 2026-09-05** (dual-attention norm input, VAE scale/shift, unpatchify channel order, missing positional embedding) — output at measurement time was pure noise, now fixed to produce real coherent (non-photorealistic) image structure at the same resolution/steps. No fresh post-fix timing exists in the source doc; per-step compute cost is architecturally unchanged by these fixes (they were correctness-only, not perf-affecting ops), so this number is likely still representative, but that has NOT been re-measured — flagged as an open item rather than assumed |
| MiniMax-Music3 (full pipeline: condition encoder, RVQ depth decoder, DiT transformer, vocoder) | 200-frame / ~8s audio generation | CPU | 3352.9s (~56 min) | 2026-09-04 | citation from `docs/00-current-work.md`/`docs/066-minimax-music3-future-plan.md`; real weights, real end-to-end pipeline, post-fix (a frame-index off-by-one bug was found and fixed before this run — earlier attempts produced audible "jitter" garbage). Output saved to `docs/diffusion-samples/minimax_music3_v1_folk_verse_200frames.wav`, not yet judged by ear by any session as of the cited doc |
| FLUX.1-schnell (Q4_K_S, real CLIP-L + T5-XXL + VAE) | 512×512, 4 steps, seed 42 | CPU | 1024.4s (~17min) | 2026-09-11 | new coverage post-fix; `stingray image` CLI, real all-component checkpoint download (DiT+CLIP-L+T5-XXL+VAE, ~13GB total). **Major correctness improvement this run**: found and fixed a real `SingleBlock` attn+MLP layout bug (see README/docs/056 for detail) — for the first time ever, a genuinely recognizable red apple and wooden table render correctly in the output (`docs/diffusion-samples/flux-schnell-fix-verify.png`). Still not fully clean: a repeating tiled pattern remains in the background, a second, separable, still-open artifact |
| FLUX.1-schnell (Q4_K_S), same config | 512×512, 4 steps, seed 42 | Vulkan iGPU | 993.5s (~16.6min) | 2026-09-11 | First Vulkan timing for this checkpoint (needed a T5-XXL tokenizer not present on this machine — FLUX.1-schnell's own HF repo is gated; used `google/flan-t5-xxl`'s `tokenizer.json` instead, same underlying T5 SentencePiece vocab as base T5-XXL, just a different fine-tune — verified this is a valid substitute by the fact tokenization produced a correct, on-prompt image). **Real, notable finding: barely any Vulkan speedup here (~3% vs CPU)**, a sharp contrast to SDXL's ~3x Vulkan win — not yet explained; FLUX's architecture is Linear/attention-based (no spatial convolution, confirmed earlier this session via grep), so none of this session's conv-specific optimizations (implicit-GEMM, residency) apply here at all, and it's not yet known whether the bottleneck is the T5-XXL encoder (large, possibly CPU-only like CLIP was before its own fix) or the DiT body itself. Output correctly on-prompt (real apple, matching the prompt) with the same already-documented background-tiling artifact, not a new regression |
| LTX-Video-2B v0.9.1 (Lightricks Video DiT, T5-XXL text encoder + VAE) | 256×256, 1 frame, 25 steps, CFG 3.0 | CPU | 100.5s (timing consistent: 3 runs at ~100s each) | 2026-09-11 | new coverage; `stingray image` CLI (LTX-Video auto-routed by filename), real weights. **Correctness caveat added 2026-09-11 (same day, after further runs):** only 1 of 5 total runs against this checkpoint was actually visually inspected and confirmed coherent (the 181KB PNG, `ltx-video-perfleague-check2.png`) — the timing-only best-of-3 runs were never re-viewed. Two further runs since (one with `--upscaler`, one without, both otherwise-identical default-seed re-runs) both produced pure visual noise, not a coherent image, and use a real, independent scheduler (`RectifiedFlowScheduler`, not the `EulerDiscreteScheduler` fixed elsewhere in this doc — ruled out as the cause). **Real, unresolved finding, updated after a `--seed 42` sweep: only 1 of 6 total runs across this checkpoint has ever produced a coherent image** — the original default-seed run, plus two more default-seed runs and two `--seed 42` runs (visually identical to each other, ruling out simple non-determinism) all produced garbled noise. Timing stays real and consistent across every run. Downgraded from "first real success" to "real timing, correctness genuinely unreliable, not root-caused." |
| Stable Diffusion 1.5 (`v1-5-pruned-emaonly`) | 512×512, 20 steps, Euler | Vulkan iGPU | 659.3s (~11m) | 2026-09-11 | new coverage; `stingray image` CLI, real weights, real coherent 503KB PNG output (a genuine photorealistic table+device scene, not degenerate) saved to `docs/diffusion-samples/sd15-perfleague-check.png`. First timing ever recorded for this checkpoint |
| SD1.5 + RealESRGAN_x4plus upscale (`--upscaler`) | 512×512 base → 2048×2048 (4x), 20 steps | Vulkan iGPU | 781.3s total (~13m) — base gen ~657s + upscaler ~124.5s (real diagnostic breakdown: RRDB body 106.4s, upsample/HR/download 18.1s) | 2026-09-11 | new coverage; first-ever working measurement of `RealESRGAN_x4plus.safetensors`, the `--upscaler` RRDBNet path — real, genuinely sharp 2048×2048 upscale (5.5MB PNG, visually confirmed coherent wood-grain detail, not degenerate), `docs/diffusion-samples/sd15-upscaled-check.png`. The same flag/checkpoint produced pure visual noise when paired with LTX-Video's separately-unstable output (see LTX-Video's own row for that unrelated finding) — this run rules out `--upscaler` itself as a cause, since it works cleanly here on a known-good base image |
| SDXL-Turbo (`sd_xl_turbo_1.0_fp16`) | 512×512, 4 steps, Euler | CPU | 832.3s (~14m) | 2026-09-11 | new coverage; `stingray image` CLI, real weights, real coherent 659KB PNG (genuine apples-on-a-table image). First timing ever recorded for this checkpoint. **Also found and fixed a real `EulerDiscreteScheduler` division-by-zero bug** (steps=1 produced NaN sigmas → solid black 843-byte output on both CPU and Vulkan) — see Known Measurement Gaps below for the fix; steps=1 output quality is still poor post-fix (needs "trailing" timestep spacing, not implemented) but no longer crashes/blackscreens |
| SDXL-Turbo (`sd_xl_turbo_1.0_fp16`) | 512×512, 4 steps, Euler | Vulkan iGPU | 265.2s (~4.4m) | 2026-09-11 | same run; **~3.1x faster than CPU** (265.2s vs 832.3s) — a real, clean Vulkan win for SDXL-class diffusion on this iGPU, unlike the mixed/negative results seen for small-LLM Vulkan prefill elsewhere in this doc |
| SDXL-Turbo (`sd_xl_turbo_1.0_fp16`), real turbo config | 512×512, 4 steps, Euler, `--cfg-scale 0.0` (real recommended usage), trailing timestep spacing | Vulkan iGPU | 161.5s (~2.7m) | 2026-09-11 | **39% faster than the row above** (161.5s vs 265.2s) — a real, measured win from skipping the mathematically-redundant CFG pass at guidance=0 (see "Skip the mathematically-redundant CFG pass" below): `CombineGuidance` reduces to just the uncond prediction at guidance==0, so the cond pass is pure waste and is now skipped entirely, halving the UNet forward-pass count per step (4 total forwards instead of 8). Output re-verified coherent post-change (real, structured room-scene image, not noise) |
| SDXL-Turbo (`sd_xl_turbo_1.0_fp16`), same + redundant-sync removal | 512×512, 4 steps, Euler, `--cfg-scale 0.0`, trailing spacing | Vulkan iGPU | 154.7s (~2.6m) | 2026-09-11 | **further ~4% faster than the row above** (154.7s vs 161.5s) — removed a redundant `vkDeviceWaitIdle()` called before every single `Download()` in `SdxlUNet2DConditionModel`/`VaeDecoder`/`StableDiffusion.UNet2DConditionModel`'s `Conv()`/`Lin()` helpers; traced the real Vulkan call chain and confirmed `Sgemm()`'s own `Dispatch()` already fence-waits for that specific dispatch, and `Download()`'s `CopyBuffer` already does its own submit-and-wait, so the extra full-device idle added no correctness guarantee, only overhead. Output pixel-identical to the pre-change run (same seed), confirming zero behavior change |
| SDXL-Turbo (`sd_xl_turbo_1.0_fp16`), same + real per-stage profiling | 512×512, 4 steps, guidance=0 | Vulkan iGPU | 166.6s total: text-encode 13.3s, step1 34.6s (cold cache), steps2-4 ~22.5s each, VAE decode 50.8s | 2026-09-11 | Added real stage-level timing (wired to `-v`) instead of guessing from steps-vs-wall-time deltas, which conflated `CachedWeightReader`'s one-time cold-cache disk/dequant cost with real per-step compute. **Found VAE decode is the single largest stage** — ~30% of total wall time at 4 steps, a *fixed* cost independent of step count that dominates even more at turbo's real 1-step use case |
| SDXL-Turbo (`sd_xl_turbo_1.0_fp16`), same + fp16 weight upload | 512×512, 4 steps, guidance=0 | Vulkan iGPU | 158.7s (~4.7% faster than the profiling-run row above) | 2026-09-11 | `SdxlUNet2DConditionModel`/`VaeDecoder` always uploaded weights as fp32 regardless of `BestSgemmPrecision` (this backend reports Fp16), silently forcing every matmul onto the slowest full-fp32 Sgemm shader even though this exact checkpoint is fp16 on disk. Switched weight uploads to `UploadHalf` (via `TensorPrimitives.ConvertToHalf`, not a scalar loop — an initial scalar-loop version measurably regressed cold-start weight-load cost, caught and fixed by re-measuring) when the backend prefers fp16. Real gains at every stage, no regression anywhere including cold-start; output pixel-content re-verified unchanged (same seed) |
| SDXL-Turbo (`sd_xl_turbo_1.0_fp16`), same + addEmb caching | 512×512, 4 steps, guidance=0 | Vulkan iGPU | 153.2s (~3.5% faster than the row above) | 2026-09-11 | Found `ComputeTimeAndAddEmbedding` recomputed SDXL's `label_emb` micro-conditioning projection on every denoising step even though the input (`addEmbeds`) is invariant across the whole loop — only the timestep half actually varies per step. Added a reference-keyed cache so the two `label_emb` GPU round-trips run once per `Generate()` call instead of once per step |
| SDXL-Turbo (`sd_xl_turbo_1.0_fp16`), same + CPU vectorization | 512×512, 4 steps, guidance=0 | Vulkan iGPU | 152.9s (roughly flat vs the row above) | 2026-09-11 | Replaced ~10 scalar elementwise-add loops (residual adds, attention/FFN adds, per-channel timestep-bias broadcasts, per-row `Lin()` bias adds, the full-resolution RGB clamp at the end of VAE decode) across `SdxlUNet2DConditionModel`/`StableDiffusion.UNet2DConditionModel`/`VaeDecoder` with `TensorPrimitives.Add`/`Multiply`/`Clamp` (SIMD instead of scalar). Small/flat effect here since GPU dispatch dominates each step on this backend — would matter more on a CPU-only run where these same functions fall back to a pure-CPU path. Output pixel-identical to the pre-change baseline (same seed), confirming zero numerical drift |
| SDXL-Turbo (`sd_xl_turbo_1.0_fp16`), same + parallel im2col | 512×512, 1 step (turbo's real flagship use case), guidance=0 | Vulkan iGPU | 82.9s total (VAE decode 35.59s) vs 97.2s total (VAE decode 49.06s) pre-fix | 2026-09-11 | Added real per-substage VAE profiling (`STINGRAY_PROFILE_VAE=1`) and found the last two up-blocks — the ones running at the full 512×512 output resolution — dominated VAE decode (`up.1` 16.2s, `up.0` 18.0s of 49.1s total). Traced to `Conv()`/`ConvGpu()`'s im2col gather: single-threaded scalar code doing hundreds of millions of boundary-checked gathers per conv layer at this resolution while the GPU sat idle. Each output row's gather writes a disjoint, directly-computable buffer range, so parallelized it with `Parallel.For` across rows (fixed in `VaeDecoder.Im2ColChunk`, `SdxlUNet2DConditionModel.Conv`, `StableDiffusion.UNet2DConditionModel.Conv` — same duplicated pattern in all three). **~27% faster VAE decode, ~15% faster total at 1 step** — the single largest win of this whole perf pass. Output pixel-identical to baseline at both 1 and 4 steps |
| SDXL-Turbo (`sd_xl_turbo_1.0_fp16`), same + parallel write-back | 512×512, 1 step, guidance=0 | Vulkan iGPU | VAE decode 33.25s, total 80.9s | 2026-09-11 | Same class of fix as im2col: `Conv()`/`ConvGpu()`'s post-GEMM transpose+bias-add write-back (`output[oc*hw+absPos] = resultBuf[pos*outC+oc] + bias`) was also single-threaded scalar code at the same scale — each `pos` writes disjoint locations, so parallelized across `pos`. ~6.6% further VAE decode improvement |
| SDXL-Turbo (`sd_xl_turbo_1.0_fp16`), same + vectorized GroupNorm | 512×512, 1 step, guidance=0 | Vulkan iGPU | VAE decode 31.01s, total 77.0s | 2026-09-11 | `DiffusionOps.GroupNorm`'s per-group work was already `Parallel.For`'d, but its variance and normalize/affine inner loops were scalar. Vectorized with `TensorPrimitives.Subtract`/`Dot`/`Multiply`/`Add` (same subtract-then-dot approach `LayerNorm` already uses). GroupNorm is shared by every conv pipeline in this codebase, not SDXL-specific — real win visible in both VAE decode and the UNet's own denoise-step timing. Output pixel-identical to baseline throughout this whole VAE-decode investigation |
| SDXL-Turbo (`sd_xl_turbo_1.0_fp16`), CPU-only path (`--backend cpu`) | 512×512, 1 step, guidance=0 | CPU SIMD | 458.3s total, VAE decode 373.25s (~81% of total) | 2026-09-11 | First real measurement of this checkpoint's CPU-only path (a distinct execution path from the Vulkan GPU path all the above rows target — `ConvBlock` branches to the pure-CPU `DiffusionOps.Conv2D` direct convolution, no im2col/GPU dispatch at all, when no backend is present). ~12x slower than the GPU path's 77.0s. Notably `up.2` (128×128 spatial, 512ch) dominates here (131.6s) rather than the largest-*resolution* stages that dominate on GPU — CPU cost scales with raw channel×spatial FLOPs, not dispatch count. `Conv2D` is already `Parallel.For`'d across output channels; no equivalent single-threaded bottleneck found there (unlike the GPU path's im2col/write-back, which were genuinely unparallelized) |
| ~~GPU-dispatch reduction attempts (two tried, both reverted)~~ | 512×512, 1 step, guidance=0 | Vulkan iGPU | Larger im2col chunks (32M→128M floats): VAE decode 31.01s→34.59s (regression). CPU/GPU Task.Run pipelining of im2col: 31.01s→32.23s/32.90s (flat-to-regression, 2 runs) | 2026-09-11 | Both are real, measured negative results, not assumptions — recorded so neither idea gets retried without re-measuring. Larger chunks: bigger single Upload/Download transfers cost more than they save in reduced dispatch count on this iGPU (no compute/transfer overlap). Task.Run pipelining: `Im2ColChunk`'s own `Parallel.For` already saturates the thread pool, so a background "next chunk" competes for the same workers instead of cleanly overlapping idle fence-wait time |
| SDXL-Turbo, native GPU Conv2d shader (first step) | 512×512, 1 step, guidance=0 | Vulkan iGPU | `norm_out + conv_out` stage: 0.27s (from ~0.5-0.6s baseline) | 2026-09-11 | Found a real, existing, battle-tested native GPU `Conv2d` compute shader already in this codebase (RRDBNet's real, working upscaler path) that does a whole conv in ONE dispatch with zero CPU-side im2col/transpose — but its shared-memory weight buffer was a fixed 2048 floats, sized only for RRDBNet's ≤192-channel convs; using it as-is for SDXL/VAE's up-to-512-channel convs would silently overflow it. Installed the Vulkan SDK (`winget install KhronosGroup.VulkanSDK`, needed for `glslc` to compile shader edits — this machine had none), generalized the shader to loop over channel tiles instead of assuming everything fits one shot (verified byte-identical behavior for RRDBNet's real use case via a controlled A/B test), then wired VaeDecoder's 3 standalone convs (`post_quant_conv`/`conv_in`/`conv_out`) to it as the first, most contained step of a larger planned port. Real, visible win on the one stage touched so far; ResBlock/mid-block/up-block convs (the majority of VAE decode's cost) not yet converted |
| ~~Naive Conv2d shader on ResBlock's larger convs~~ | 512×512, 1 step, guidance=0 | Vulkan iGPU | VAE decode: 32.20s → ~38.3s mean (regression, 3 runs) | 2026-09-11 | Real, measured negative result: extending the *naive* shader (one thread per output pixel, scalar accumulation) to ResBlock's 256-512 channel convs regressed ~20%. Not a CPU-vs-GPU question — both paths already ran on GPU. The naive shader's accumulation doesn't scale like a tiled GEMM does; eliminating CPU-side im2col wasn't the bottleneck here, raw GPU compute efficiency was. Reverted, replaced by the implicit-GEMM shader below |
| **SDXL-Turbo, implicit-GEMM Conv2d shader (VaeDecoder)** | 512×512, 1 step, guidance=0 | Vulkan iGPU | VAE decode: 30.41s → ~25.2s mean (24.96/25.01/25.57s, 3 runs) | 2026-09-11 | **The real fix**: same 16×16 shared-memory-tiled GEMM structure as `SgemmF32` (real GEMM efficiency), but the im2col "A" operand is computed on the fly inside the kernel from the real input tensor instead of read from a materialized CPU-uploaded buffer — combines both prior approaches' wins instead of choosing one. This is the standard "implicit GEMM" convolution pattern production ML frameworks use, not something specific to this codebase. ~17% faster than the best prior VAE-decode result, and faster than both individual approaches tried. Also fused bias-add + NCHW transpose directly into the kernel, eliminating the separate CPU write-back step too. Verified pixel-identical output at 1 and 4 steps across all 3 runs. **This class (`VaeDecoder`) is shared across SDXL/SD1.5/FLUX.1/Z-Image-Turbo's VAE decode**, so this win applies to all of them, not just SDXL |
| SDXL-Turbo, implicit-GEMM extended to SdxlUNet2DConditionModel | 512×512, 4 steps, guidance=0 | Vulkan iGPU | Steady-state denoise step: ~20.0-20.3s → ~19.7-19.8s | 2026-09-11 | Same shader applied to the UNet body itself (routes automatically for every stride=1 conv — the two stride=2 downsample convs correctly fall back to the old path, since implicit-GEMM only supports stride=1). Real but much smaller win than VAE decode's — SDXL's UNet runs at downsampled resolutions (64×64 down to 8×8) vs VAE decode's up-to-512×512, so proportionally less CPU-side overhead existed here to eliminate. Output pixel-identical to baseline |
| **Session-cumulative total, SDXL-Turbo** | 512×512, 4 steps, guidance=0 (real turbo config) | Vulkan iGPU | **265.2s → 128.8s (~51% faster)** | 2026-09-11 | Full session arc: correctness fixes (trailing timestep spacing, real CFG=0 support) + 8 verified perf wins (CFG-skip, redundant-sync removal, fp16 weights, addEmb caching, CPU vectorization, parallel im2col/write-back, vectorized GroupNorm, implicit-GEMM shader ×2) + 3 correctly-rejected negative results (larger GPU chunks, CPU/GPU Task.Run pipelining, naive-shader-on-large-convs), each measured and reverted rather than assumed |
| ~~**CRITICAL: CFG-skip at guidance≤0 used the WRONG embedding (negative prompt, not real prompt)**~~ | 512×512, 4 steps, guidance=0 | Vulkan iGPU | Timing unchanged (132.5s); **content was wrong for the entire session until this fix** | 2026-09-11 | Built a real `stable-diffusion.cpp` (Vulkan backend, from source, this session) specifically to get an independent reference to compare against — and the real reference correctly rendered "a red apple on a wooden table" for the exact prompt/seed/config, while every SDXL-Turbo output this session (including every row above marked "coherent"/"real structured scene") was a generic, PROMPT-UNRELATED room scene. Root cause: the `guidance<=0f` fast-path (added earlier this session, see the row two above) returned the UNCOND (negative/empty-prompt) embedding instead of the COND (real prompt) embedding. Verified against real diffusers source (`pipeline_stable_diffusion_xl.py:1148-1149,1202-1225`): when CFG is off, the real pipeline runs a single forward pass with the REAL prompt embedding, never the negative one — `guidance==0` does not mean "run unconditioned," it means "run the real prompt with no CFG contrast applied." **This means every "coherent, correct-looking" SDXL-Turbo/SD-Turbo image logged in this doc from the CFG-skip commit onward was non-degenerate but NOT prompt-following** — a real, severe correctness bug that "looks plausible" checks alone (rather than an independent reference) failed to catch. Fixed in both `SdxlPipeline.Generate` and `StableDiffusionPipeline.Generate`. Re-verified post-fix: same exact config now produces a correct, clearly on-prompt image (an apple orchard scene with visible red apples on a wooden table) — a complete change in content, confirming the fix. Timing is unaffected (same UNet compute cost regardless of which context is passed) |
| **Real reference comparison: stable-diffusion.cpp (Vulkan, built from source this session)** | 512×512, 4 steps, cfg≈1.0 (no-CFG equivalent), SDXL-Turbo, same prompt/seed | Vulkan (same iGPU) | Text encode 1.86s, denoise (4 steps) 10.45s (~2.6s/step), VAE decode 8.85s, **total 21.18s** | 2026-09-11 | Built via `winget`-installed Vulkan SDK + CMake/Ninja/MSVC (all present on this machine) from the vendored `examples/stable-diffusion.cpp` source (`git submodule update --init` was needed first — `ggml` wasn't checked out). Real, working `sd-cli.exe`, real Vulkan device detected (confirms 32KB shared memory on this iGPU, relevant to this session's shader-buffer-size work). This is the first real, independently-built C++ yardstick this project has had for the diffusion pipeline — not a stand-in estimate |
| **Our implementation vs. the real C++ reference, same config, post-CFG-fix** | 512×512, 4 steps, guidance=0, SDXL-Turbo, same prompt/seed | Vulkan iGPU (ours) vs Vulkan (sd.cpp) | Text encode 11.45s vs 1.86s (**~6.2x slower**); denoise 4 steps ~93s vs 10.45s (**~8.9x slower**, ~20-33s/step vs ~2.6s/step); VAE decode 27.72s vs 8.85s (**~3.1x slower**); **total 132.5s vs 21.18s (~6.3x slower overall**) | 2026-09-11 | **Not close to C++ parity** — a real, honest gap, not a guess. The breakdown is informative though: VAE decode's gap (~3.1x) is meaningfully smaller than the UNet's (~8.9x) or text encode's (~6.2x), which lines up with this session's own work — VAE decode got 3 real optimization passes (parallel im2col/write-back, vectorized GroupNorm, implicit-GEMM), while the UNet only got 1 (implicit-GEMM) and text encode got none. The UNet denoise step and text encode are the real remaining gap to chase, not VAE decode, which is proportionally much closer already |
| SDXL-Turbo, full GPU residency (fused GroupNorm+SiLU, ResBlock stays on GPU throughout) | 512×512, 1 step, guidance=0 | Vulkan iGPU | VAE decode: 19.57s/19.87s (2 runs) vs 25.25s pre-residency baseline (**~22.5% further faster**) | 2026-09-11 | Real next step after implicit-GEMM: added a fused GroupNorm+SiLU GPU shader and restructured `ResBlock` to try a fully GPU-resident path first (`ResBlockGpu`) — one Upload of the block's input, the whole `norm1→silu→conv1→norm2→silu→conv2→(+skip)` chain stays on GPU (explicit `Free()` on every intermediate tensor), one Download of the result. Collapses what was up to 2 separate conv Upload/Download round-trips into 1, with GroupNorm/SiLU now running on GPU instead of CPU too. Falls back cleanly on backends without `GroupNormSilu` (CUDA throws, probed once and cached). Content re-verified correct (not just coherent, given the CFG bug found the same day) and pixel-identical to the post-CFG-fix baseline at 4 steps |
| **Updated vs. C++ reference, post-residency** | 512×512, 4 steps, guidance=0, SDXL-Turbo, same prompt/seed | Vulkan iGPU (ours) vs Vulkan (sd.cpp) | Total 129.5s vs 21.18s (**~6.1x slower**, down from ~6.3x); VAE decode gap alone: ~2.2x slower (down from ~3.1x) | 2026-09-11 | Real, measured narrowing of the gap specifically where this session's optimization work concentrated (VAE decode) — confirms the residency work is closing real ground, not just moving the bottleneck around. UNet denoise (~8.9x) and text encode (~6.2x) remain the dominant gap and haven't been touched by residency work yet |
| CLIP-L/CLIP-G weight caching | 512×512, 4 steps, guidance=0 | Vulkan iGPU | Text encode: 11.45s → 7.63s (**~33% faster**); gap to sd.cpp narrowed ~6.2x → ~4.1x | 2026-09-11 | Found `ClipLEncoder`/`OpenClipGEncoder` called `_st.ReadF32()` directly everywhere with zero caching — a real `file.Seek`+`ReadExactly` disk read under a lock on every call, and `Encode()` runs twice per generation (cond+uncond), re-reading the same ~120/~200 weight tensors both times. Added a simple `Dictionary<string,float[]>` cache matching the existing `CachedWeightReader` pattern. Pure caching fix, no compute-path change — output pixel-identical to the known-correct baseline |
| SpatialTransformer (UNet attention) GPU residency — **scoped, not attempted** | — | — | — | 2026-09-11 | Investigated as the next lever for the UNet's remaining ~8.9x gap: `SpatialTransformer` does the same per-`Lin()`-call Upload/Sgemm/Download round-trip pattern `ResBlock` had before its residency fix — at `depth=10` (the deepest UNet blocks) that's ~100 `Lin()` calls per `SpatialTransformer` invocation. Unlike `ResBlock`, a real fix needs a GPU softmax/attention kernel matching this shape (full bidirectional self+cross attention over up to hw=4096 tokens, no causal mask, no KV cache) — this codebase's existing `Attention`/`AttentionBatched` shaders are LLM-decode-shaped (KV-cache-oriented) and don't directly apply. Building and verifying a new attention kernel correctly carries real risk without more time budget in one sitting; recording as a real, scoped, un-started next step rather than attempting a rushed version |
| SpatialTransformer Q/K/V upload batching (`LinMulti`) | 512×512, 4 steps, guidance=0 | Vulkan iGPU | Total 135.9s → 117.4s/115.6s (2 runs, **~14-15% faster**) | 2026-09-11 | A smaller, safer step short of the full attention-residency rewrite above: self-attention's Q/K/V projections all read the same normed input, and cross-attention's K/V both read the same `context`, but each `Lin()` call previously did its own separate `Upload()` of that identical input. Added `LinMulti()` — uploads the shared input once, reuses it across N projections (still one Sgemm+Download each). Real, measured, consistent win across 2 runs; output pixel-identical to the known-correct baseline |
| **Fresh vs. C++ reference, current state** | 512×512, 4 steps, guidance=0, SDXL-Turbo, same prompt/seed | Vulkan iGPU (ours) vs Vulkan (sd.cpp) | Total ~116.5s (2-run avg) vs 21.18s (**~5.5x slower**, down from ~6.1x/~6.3x earlier this session) | 2026-09-11 | Continued, real narrowing from this session's full arc of fixes (CFG bug, implicit-GEMM, residency, CLIP/T5 caching, Q/K/V batching). UNet denoise remains the largest absolute contributor (~77% of total wall time) and the biggest relative gap — the scoped-but-unattempted attention-residency rewrite above remains the single highest-value remaining lever if pursued with proper verification time |
| ~~Naive GPU attention shader wired into SpatialTransformer~~ | 512×512, 4 steps, guidance=0 | Vulkan iGPU | Denoise step ~19.5s → ~26.8s (**regression**), total 117.4s → 146.8s | 2026-09-11 | Real, measured negative result attempting the attention-residency lever above: added a real `MultiHeadAttention` GPU shader (one thread per query/head pair, online softmax), numerically verified correct against the CPU reference first (`MultiHeadAttentionGpuParityTests`, 4 shapes, all passed) — output stayed pixel-identical to the correct baseline in the real pipeline too, so this is purely a perf regression, not a correctness one. Root cause: each of `qSeq*nHeads` threads independently re-reads the ENTIRE K/V sequence from global memory with zero tiling — for self-attention at `hw=4096` that's ~10+ billion redundant reads per call, far more bandwidth-inefficient than the CPU's cache-friendly SIMD version. Same class of lesson as the earlier naive-Conv2d regression, worse here because O(seq²) attention punishes "no tiling" harder than bounded-kernel convolution did. Reverted; shader+test kept as a correct building block for a future properly-**tiled** (shared-memory-blocked, flash-attention-style) rewrite — a real, much bigger undertaking than anything else in this session's arc |
| ~~Tiled (flash-attention-style) GPU attention shader wired into SpatialTransformer~~ | 512×512, 4 steps, guidance=0 | Vulkan iGPU | Denoise step ~19.5s (CPU baseline) → 52-72s (**far worse regression**); VAE decode (no attention at all) 19.6s baseline → 361s | 2026-09-11 | Attempted the tiled rewrite the naive-shader row above flagged as the next step — followed the row/column-tile + online-softmax technique from ggml-vulkan's real `flash_attn.comp` (reviewed, not copied; rewritten for this codebase's `[seq, numHeads*headDim]` interleaved-head layout vs. ggml's per-head-buffer layout), with a second-opinion design review from an external LLM before implementing (Br=16, Bc=32, 256 threads/workgroup, 16 threads per query row via subgroup-shuffle reduction, sized for this iGPU's 32KB shared-memory budget). Verified numerically correct in isolation first (`MultiHeadAttentionTiledGpuParityTests`, 6 shapes including multi-tile qSeq=4096 and 1024×1024 cases, tolerance 5e-3 to account for reordered float accumulation — all passed, ~1.4s real device time). Wired into `SdxlUNet2DConditionModel.MultiHeadAttention` with the same probe-once/fallback pattern as `VaeDecoder.ResBlockGpu`, then measured against real weights: an even worse regression than the naive shader, not an improvement — despite eliminating the naive shader's "re-read all of K/V per thread" problem, per-dispatch fixed overhead on this iGPU (3 uploads + 1 download per call, many calls per UNet forward) dominates at these problem sizes. The VAE-decode blowup (19.6s→361s) is the more striking anomaly since `VaeDecoder` has no attention calls at all and wasn't touched by this change — likely GPU scheduler/memory contention from queuing many small tiled-attention dispatches ahead of it in the same process, though not root-caused; flagged here rather than assumed. Reverted immediately; shader + parity test (still passing) kept as a *correct* building block, but this is now the second consecutive real-measured failure to turn attention residency into a win on this iGPU — the attention lever looks structurally unfavorable on this specific hardware (shared-memory iGPU, small per-call payloads) rather than a tuning problem, and shouldn't be re-attempted without either much larger batched dispatches (fusing many attention calls into one) or access to a discrete GPU to re-test the premise |

> No C++ reference attempted for either (no vendored image/video-diffusion C++ CLI in this repo).
> `hunyuanvideo` and `ltx-t5` were not attempted — no wired end-to-end CLI/test path was found for
> either, and given Wan2.1's real ~71-minute cost for just 2 frames, both would likely take
> considerably longer still.

---

## CUDA Inference — No Numbers Yet

> **No CUDA GPU on dev machine.** No measured numbers exist for CUDA on this box.
> See `GR_performance.md` and `docs/done/gpu-review-log.md` for the code-level audit.
> All CUDA rows above are from prior hardware (†).
>
> **Highest-confidence open opportunity:** Q6_K and Q5_K prefill currently dequantize full weight
> matrix to FP16 then call cuBLAS — a full-weight HBM round-trip per call. Direct MMQ (as llama.cpp
> does) would eliminate this. Requires real NVIDIA hardware to validate.

---

## Known Measurement Gaps

Rows where the Ratio column is blank and a C++ comparison would be actionable:

| Model | Scenario | Backend | Why blank | Opportunity |
|---|---|---|---|---|
| Piper / Kokoro / MeloTTS / MMS-TTS | text → audio | CPU | Re-verified 2026-09-10: no `.exe` in any of `examples/piper`, `examples/kokoro.cpp`, `examples/MeloTTS.cpp`, `examples/TTS.cpp` build trees — still unbuilt, still blocked on external SDKs (onnxruntime, OpenVINO, cppjieba, espeak-ng) | Build minimal self-contained CLI wrappers for baseline verification (multi-SDK build effort, not attempted this pass) |
| Parler-TTS | text → audio | CPU | `parler` is not a registered family in `examples/audio.cpp`'s model registry at all (checked `--task tts --family <x> --help` family list, 2026-09-10) — no model_spec, no loader | Would need a new audio.cpp family registration, not just a build fix |
| ~~F5-TTS~~ | text → audio | CPU | **FIXED 2026-09-11, two real bugs.** (1) `cpu_graph_compute.h`'s `ggml_graph_compute_with_ctx` resolution only had GCC/Linux code paths (a weak-symbol check, then a `dlopen`-based fallback) — under MSVC (`__GNUC__` undefined, this vendored build's real compiler, confirmed via `build/CMakeCache.txt`), neither path compiled in, so it unconditionally threw regardless of whether the CPU backend was actually available. It was: this build is a plain static link (`GGML_BACKEND_DL=OFF`), so the symbol was directly linkable the whole time — added a `#if !defined(__GNUC__)` branch that just calls it directly. (2) Once past that, hit a second real bug: `ggml_new_object: not enough space in the context's memory pool` — the `ctx_bytes` sizing formula in `runtime.cpp` (two call sites) was consistently a few MB short of what ggml actually needed, and its hard cap (12288MB) was also too low for longer reference-audio clips. Added a flat +64MB safety margin and raised the cap to 16384MB (this machine has 64GB RAM, real headroom). **Verified end-to-end**: real WAV output produced from both a short (65KB output) and a longer (768KB output) reference-audio clip, consistent 43.2s/43.1s/43.2s timing across 3 runs. First-ever real C++ timing comparison for F5-TTS — see the new row above. | Closed |
| All models | any | CUDA | No CUDA device on dev machine | Direct MMQ for Q6K/Q5K prefill |
| All models | any | Vulkan iGPU | No llama.cpp Vulkan reference | Discrete GPU needed for real comparison |
| All audio.cpp TTS/ASR pipelines (QwenTTS, CosyVoice3, Chatterbox, FishSpeech, etc.) | any | Vulkan iGPU (via `audio.cpp`) | Checked 2026-09-10: `audiocpp_cli --backend vulkan` is a real, listed CLI option, but this vendored build reports "Vulkan backend requested but it is not registered in this build" — only CPU is compiled in | Rebuild `audio.cpp` with Vulkan support enabled to get a real C++ Vulkan comparison for any of these pipelines |
| TTS Pipelines (all) | full synthesis | GPU | Pipelines not wired to GPU yet | Expected to be the largest TTS win |
| CPU KV cache | bf16/q8 dtype | CPU | `PagedKvCache` hard-wired fp32 | Vulkan showed +57% decode at no quality cost |
| Llama-4-Scout 17B-16E Q4_K_M | prefill + decode | CPU | Cancelled 2026-09-10 by explicit user instruction (`~93GB` across 2 shards vs. this machine's 64GB total RAM — would never fit; user said "no point in killing the pc"). Partial download deleted. | Not pursuing on this hardware; would need a machine with substantially more RAM |
| Carnice 35B-A3B-MTP (APEX) | prefill/decode | CPU | No locatable public repo for "Carnice APEX" as of 2026-09-10; likely a gated/private checkpoint from the original README-history capture | Needs the original source/access used when the † numbers were first captured |
| Any GGUF embedding model | throughput vs C++ | CPU | Confirmed 2026-09-10: `stingray embed`'s GGUF path is a hash-based synthetic stub (`EmbeddingEngine.cs:147-162`) that never loads real weights — produces identical output for any `-m` path including a nonexistent one. No real GGUF embedding measurement is possible with this CLI today. | Wire a real GGUF forward pass into `EmbeddingEngine` (or route GGUF paths through the same `ForwardPass`/backend machinery the `run`/`image` commands use) before any embedding throughput number can be trusted |
| ~~Any ONNX embedding model (MiniLM, BGE, etc.)~~ | throughput vs C++ | CPU | **FIXED 2026-09-11** — see Embeddings section above for real measurements. Was: crashed with `Missing Input: token_type_ids`. | — |
| (systemic) Any real-weights test using an absolute/relative model-path search rooted at `models/` only | measurement validity | any | Confirmed 2026-09-10 on `ParakeetRealWeightsTests`: it was silently no-op'ing (0.1s runtime, `CLAUDE.md` rule 12's documented pattern) because `parakeet-ctc-0.6b-q4_k.gguf` lives in `models/_models/`, not `models/`, and the test's search helper only checks `models/`. Fixed here by adding a symlink, but the same class of silent-no-op likely affects other tests with the same narrow search pattern — this was found by accident while sweeping for new coverage, not by a systematic audit | A systematic sweep of every `*RealWeightsTests.cs`'s model-search helper against actual `models/_models/` contents would likely surface more silently-skipped tests, per the scope CLAUDE.md rule 12 already flags |
| Qwen3.8-27B | chat-template correctness, not a perf gap | CPU | Found in passing 2026-09-10 while benchmarking: 3 real Jinja chat-template rendering gaps logged as runtime warnings for this checkpoint's template (unsupported string-concat-in-conditional expressions) — value passed through unchanged rather than evaluated, so rendered prompt output may be subtly wrong for this specific chat format | Extend the Jinja subset this project's template engine supports to cover string-concatenation inside conditional/`in` expressions |
| ~~Granite family — Vulkan correctness bug~~ | — | — | **FIXED 2026-09-11.** Root cause found via subagent investigation: `GpuForwardPass.cs`'s `RunStandardLayers`/`RecordBatchedTrunk` never threaded `AttentionScaleOverride`, `ResidualScale`, or `LogitScale` into the Vulkan dispatch path, while the CPU path applied all three (Granite's real, non-default scaling hyperparameters). Fixed by a second subagent: added the same Q-prescale-to-cancel-shader's-hardcoded-scale trick already used for Gemma 4's `AttentionScaleOverride`, plus `ScaleInPlace` calls for `ResidualScale`/`LogitScale` at every relevant call site (9 sites total across single-token decode, batched prefill, and all logit-output paths), gated on non-default values so other architectures are unaffected. Verified 2026-09-11 on both previously-broken checkpoints: both now produce coherent output on Vulkan matching CPU exactly, with no measurable perf regression. Full solution rebuilds clean (0 warnings, `TreatWarningsAsErrors` enabled). | Closed |
| QwenTTS / CosyVoice3 (streaming) | TTFA vs C++ | CPU | Re-verified 2026-09-10 with real runs: both explicitly refuse `--mode streaming` at runtime ("only supports offline sessions") — a deliberate design limit in `audiocpp_cli`, not a missing build/asset | Would need real streaming-session support added to these two families in `audio.cpp` itself |
| ~~Chatterbox Turbo (streaming)~~ | TTFA vs C++ | CPU | **FIXED 2026-09-11 (vocab), but streaming was never the real blocker.** Extracted the real tokenizer.ggml.tokens/merges GGUF metadata into the 3 sidecar files the loader needs; verified with a real end-to-end `--mode offline` run (genuine 126KB WAV, `docs/audio-samples/chatterbox-turbo-audiocpp-vocab-fix-verify.wav`). Once loadable, `--help` shows this build genuinely has NO streaming mode for Chatterbox Turbo at all (`--mode streaming` → "Chatterbox Turbo only supports offline mode") — same deliberate offline-only design limit as QwenTTS/CosyVoice3 below, not a bug the vocab fix could ever have unblocked | Closed as far as this doc can take it — a real TTFA/streaming comparison for this model would need `audio.cpp` itself to add streaming support, same as the QwenTTS/CosyVoice3 row |
| FishSpeech S2 Pro (streaming) | TTFA vs C++ | CPU | `s2.exe --stream-file` genuinely works and prints real streaming metrics (2026-09-10: `stride=16, holdback=144, ref_encode=49957ms, generate=170348ms, total_rtf=61.32`), but doesn't print an explicit first-chunk timestamp — computing a TTFA-equivalent from the other fields would be an inference, not a real measurement | Add a printed first-chunk timestamp to `s2.exe`'s streaming metrics, then it's a real, direct TTFA comparison |
| FunASR Paraformer (GGUF path) | any | CPU | Confirmed 2026-09-11 via direct real test run: `paraformer-q8.gguf` genuinely lacks the `pf.vocab` GGUF metadata `FunAsrWeights.cs` requires — throws `InvalidDataException` on load, not a silent no-op | Needs a fresh, correctly-converted GGUF checkpoint for this model — not fixable from existing local files |
| SenseVoice | any | any | Confirmed 2026-09-11: not a real wired pipeline — only a doc-comment mention in `FunAsrPipeline.cs`, no model spec/config/code path. README's feature-list prose overstates coverage here | Would need a real, dedicated pipeline implementation, not a bug fix |
| Parakeet TDT | any | any | Confirmed 2026-09-11: only `ParakeetCtcDecoder.cs` exists, no TDT-specific decoder anywhere in `Parakeet/`. README's "CTC/TDT" phrasing overstates coverage — only CTC is real | Would need a real TDT decode-head implementation, not a bug fix |
| Granite-4.0-3B-Vision (real image input) | vision-encode correctness | CPU | **Two real bugs found and fixed 2026-09-11, still not fully correct.** (1) The real multi-block windowed QFormer projector (8 blocks, real window gather/scatter, real self+cross attention) was ported from `granite4-vision.cpp`, replacing the old single-linear-layer stand-in — no more crash, soft-token count now architecturally correct (1152 = 8×144). (2) A second pass re-checked eps/index-math/concat-order/downsample-branching (all confirmed clean against the real reference) and found the ACTUAL remaining bug: hardcoded OpenAI CLIP mean/std used instead of the real per-checkpoint SigLIP `[0.5,0.5,0.5]`/`[0.5,0.5,0.5]` values — fixed, now reads real GGUF metadata like the other 3 encoders in this codebase already did. **Real behavioral change verified**: output shifted from generic non-image text to confident, specific (but still wrong) scene descriptions. **Still not correctly grounded** to real image content — needs numeric golden-parity against `llama-mtmd-cli.exe` next | Compare per-block intermediate tensors against a captured reference trace to find the remaining bug |
| Granite-Vision-3.2-2B (real image input) | vision-encode correctness | CPU | Confirmed 2026-09-11: same degenerate-output symptom as Granite-4.0-3B-Vision above, but this checkpoint's mmproj actually routes to a DIFFERENT code path (`LlavaAdapter`, `clip.projector_type: mlp`, not `Granite4Adapter`) per `docs/vl-migration-plan-2026-08-20.md` — not yet confirmed whether it shares the same root cause or is a separate LLaVA-path bug | Root-cause `LlavaAdapter`'s handling of this specific checkpoint separately from the Granite4 finding above |
| ~~SDXL-Turbo at `--steps 1`~~ | any | CPU / Vulkan | **FULLY FIXED 2026-09-11** (three real bugs, sequential). (1) `EulerDiscreteScheduler` divided by zero at `numInferenceSteps == 1`, producing NaN sigmas → solid black 843-byte PNG. Fixed to match diffusers' real `linspace(start,stop,num=1)==[start]` semantics. (2) The scheduler always used "linspace" timestep spacing regardless of checkpoint; SDXL-Turbo's own real `scheduler_config.json` sets `timestep_spacing="trailing"` (confirmed against real diffusers source `scheduling_euler_discrete.py`) — added a `TimestepSpacing` enum with a real "trailing" formula, auto-selected via the existing turbo/schnell/lcm filename heuristic. (3) `--cfg-scale 0.0` was silently discarded (0f was both the CLI's "unset" sentinel and a legitimate value) even though SDXL-Turbo's real recommended usage is `guidance_scale=0.0` — sentinel changed to -1f, all comparisons to `>= 0f`. **Verified real**: 1-step generation went from pure textured noise to a genuinely coherent photoreal room scene (real weights, Vulkan); 20-step generation re-confirmed no regression | none — closed |
| `deepseek2`-tagged VLM checkpoints missing MLA tensors (Kimi-VL-A3B-thinking, YouTu-VL-4B) | any | CPU | Confirmed 2026-09-11: this is a deliberate, documented design limit (see `ForwardPass.cs`'s own comment) — only the "Lite" MLA variant (`q_lora_rank==0`, plain per-head Q projection) is implemented; full-size DeepSeek-V2/V3/R1-style MLA (`q_lora_rank>0`, split `wq_a`/`wq_b` + RMSNorm) is not, so `ResolveTensor` correctly throws rather than silently mis-loading | Implement the full `q_lora_rank>0` MLA path (wq_a projection, RMSNorm, wq_b projection) — a real, substantial feature, not a quick fix |
| ~~MiMo-VL-7B-sft / Step3-VL-10B (real image input)~~ | any | CPU | **FIXED 2026-09-11.** Both crashed identically but for two different real reasons: MiMo-VL's mmproj has a genuine 3584-vs-4096-dim conversion mismatch (now a clean error, not fixable here); Step3-VL's encoder used a wrong tensor name (`mm.model_proj.weight` vs the real `mm.model.fc.weight`), now fixed and running to completion. `RunImagePrompt` also now validates vision/text dim agreement explicitly for every architecture, not just these two. | Closed |

---

## Strengths & weaknesses, after the 2026-09-10 backfill

This section reads across the ratios above; it doesn't replace them.

**Where OT is genuinely strong:**
- **Dense 7-8B models hit real prefill parity, consistently, across three independent architectures**: Qwen3-8B (1.02x), Mistral-7B-Instruct-v0.3 (1.00x), Ministral-8B-Instruct-2410 (0.99x). This is a real, repeatable pattern, not a fluke on one checkpoint — and it's the *opposite* of the SmolLM2 story: a dense 1.7B model prefills at only 0.24-0.27x. Whatever GEMM-shape gap hurts SmolLM2 apparently doesn't dominate at the 7-8B size/shape, on any of the three architectures tried.
- **Whisper family (0.83-0.95x across all four sizes)** and **Qwen3 Forced Aligner (1.01x)** remain the most consistently near-parity subsystem in the whole doc.
- **Short-context decode is close to parity** across most dense LLMs (SmolLM2 0.88-0.89x, Qwen3-0.6B 0.65-0.80x depending on measurement) — the bandwidth-bound decode path is fundamentally sound.

**Where OT clearly trails:**
- **SmolLM2-1.7B prefill sits at 0.24-0.27x flat across all context lengths (267-3218 tok)** — this backfill confirms the gap is a fixed GEMM-shape penalty, not a scaling artifact.
- **Gemma-4 family prefill is catastrophic at both sizes now measured**: 12B at 0.13x, E4B at 0.12x. Both show the same prefill≈decode signature (missing batched prefill, `perLayerHdUnsupported`), now confirmed on two model sizes instead of one.
- **OLMoE decode falls to 0.50x** on a full (non-early-EOS) 24-token run — worse than the approximate 28.2 t/s figure the doc previously carried, which undersold this gap.
- **Decode degrades faster than llama.cpp's as context grows**: SmolLM2 decode ratio drops from 0.88x (short ctx) to 0.67x (3.2k ctx) — a real, newly-measured trend, not previously visible in this doc.
- **Hybrid-GDN architecture (Qwen3.6-27B/35B) has the worst prefill ratios of any dense/MoE architecture measured**: 0.05x (35B) and 0.16x (27B) — worse even than Gemma-4's missing-batched-prefill gap. Decode is comparatively better (0.17x, 0.75x) but still trails every other architecture family in this doc. Not yet root-caused — worth its own investigation given how consistent the pattern is across both hybrid-GDN checkpoints tested.
- **Voxtral-Mini-4B-Realtime ASR is 50x slower than its C++ reference (0.02x)** — the worst *ratio* anywhere in this doc (has a C++ comparison point), on a correct transcript. No CLI/pipeline wiring exists yet, only raw per-token building blocks with no batching at all.
- **ACE-Step Turbo has the worst RTF outright (114x real-time, no C++ comparator exists)** — worse even than Voxtral, despite "Turbo" implying an 8-step fast schedule. Stable Audio 3 Medium (96.75x) and Small Music (25.36x at only 8 of a recommended 15-25+ steps) round out a consistent story: every diffusion-based audio/music pipeline measured this pass is 25-115x real-time on CPU, several orders of magnitude further from real-time than any autoregressive TTS pipeline in this doc.
- **Two independent ASR pipelines produce degenerate output — disambiguated 2026-09-11, not a shared root cause**: Qwen3-ASR 0.6B ("aspects" instead of the real reference sentence, despite running fast at RTF 0.225) is a confirmed real correctness bug — found on **real speech** audio. FunASR-Nano's repetitive word-salad ("to to to... a a a at at at") was found on `Random`-generated synthetic tone audio, not real speech (per its test harness source) — downgraded to inconclusive, since a real ASR model failing on non-speech input isn't necessarily a defect. Qwen3-ASR's bug stands as more urgent than any perf gap in this doc — a wrong-but-fast answer is worse than a slow-but-right one; FunASR-Nano needs a real-speech re-test before its status can be called either way.
- **The `embed` CLI's GGUF path is entirely fake** — worse than a performance gap or even a degenerate-output bug, this is silent total non-function dressed up as a working command. It returns identical, plausible-looking vectors for any `-m` path, including a nonexistent file, because non-`.onnx` model paths fall through to a hash-based synthetic stub that never loads a GGUF at all. This was caught only because a second, unrelated model happened to be tested against it and produced suspiciously identical output — the original single-model measurement in this doc looked completely normal and would have gone unnoticed otherwise. Worth treating as a reminder to cross-check any single-model "it works" claim against a second, different input before trusting it.
- **The Granite architecture had a real Vulkan-specific correctness bug — found AND fixed this session.** Found because the Vulkan sweep happened to cover it (identical prompt/seed gave coherent English on CPU but broken output on Vulkan — garbled multilingual gibberish on one checkpoint, an immediate single-token stop on a second). This is exactly the kind of gap a CPU-only benchmarking pass would never surface. Root-caused via subagent investigation (three Granite-specific scaling hyperparameters read on CPU but never wired into the Vulkan dispatch path) and fixed via a second subagent, verified with real re-runs on both previously-broken checkpoints: both now produce coherent output on Vulkan matching CPU. A genuine example of a benchmarking pass surfacing, and this session then closing, a real production bug — not just measuring one.

**Vulkan iGPU, backfilled across 6 LLMs this pass (Gemma-4-12B, Qwen3-8B, Qwen3-Coder-30B, Qwen3-0.6B/4B, OLMoE, Qwen3.6-27B-MTP):**
- **A consistent pattern across every dense/MoE model**: Vulkan prefill always loses to CPU prefill (worst gap at the smallest model, Qwen3-0.6B: 23.6 vs 226.1 t/s), while Vulkan decode is usually close to CPU decode and in two cases actually *wins* — Qwen3-Coder-30B (9.0 vs 7.4 t/s) and, mildly, nothing else quite matches that margin. This matches `docs/done/vulkan-backend-evidence.md`'s standing finding that this integrated GPU shares system RAM bandwidth with the CPU and has no dedicated-VRAM advantage, so it can only really win where per-token dispatch overhead is amortized well (decode) rather than where raw throughput matters (prefill).
- **Hybrid-GDN's Vulkan performance depends entirely on whether it fully fits in VRAM, not on the architecture itself.** Qwen3.6-27B-MTP loses on *both* prefill and decode on Vulkan (0.4 vs 1.0 t/s, 0.3 vs 1.0 t/s) — only 37 of 64 dense-FFN layers fit in the 16GB placement budget, a split CPU/GPU workload. But Ornith-1.0-9B (same `qwen35` architecture, smaller) fits **all 32/32 layers** on GPU and **beats CPU by ~4x on both metrics** (7.1 vs 1.7 t/s prefill, 5.3 vs 1.2 t/s decode). The earlier framing of hybrid-GDN as "uniformly worse on Vulkan" was an artifact of testing only large checkpoints that don't fit — a smaller one that fits fully tells the opposite story.
- **Confirms the Gemma-4 missing-batched-prefill bug is architectural, not CPU-specific**: Vulkan prefill≈decode (3.4 vs 3.6 t/s) shows the exact same signature as CPU (3.5 vs 4.2 t/s) — whatever's missing in `PrefillCoreAttention` isn't a CPU-only gap, it's a gap in the model's batched-prefill logic entirely, independent of backend.

**Net picture:** OT is closest to parity on bandwidth-bound decode at short context and on two fully-optimized subsystems (Whisper, Forced Aligner). Every prefill-heavy or long-context path is where the gap widens, and it widens further exactly where a known code-level cause already exists (missing batched prefill on Gemma, missing `block_q4_Kx8` GEMM interleave elsewhere) — the newly-measured numbers corroborate rather than contradict the existing diagnoses, just with real magnitudes attached now instead of an isolated single data point.

---

*Last updated: 2026-09-11 (C++ reference backfill pass, extended into a full model-coverage sweep across
CPU and Vulkan iGPU — see `docs/PerformanceLeague-backfill-plan.md` for the checklist and methodology.
By the end of this pass: every TTS/ASR/VAD pipeline subdirectory in `src/OpenTail.Stingray.Audio` (30
total) has at least one real measurement; every dense/MoE/hybrid-GDN LLM checkpoint with CPU coverage
also has a Vulkan iGPU row (or a documented reason it can't); several VLM text backbones, music/audio-gen
pipelines, and image diffusion were added as new-domain coverage. Five real bugs were found and
documented along the way: a fake hash-stub `embed` CLI path, a real ONNX `embed` crash, a Vulkan-specific
Granite correctness divergence (confirmed on two checkpoints), and two independent degenerate-ASR-output
cases (Qwen3-ASR, FunASR-Nano) — plus three previously-silently-no-op'ing tests fixed via missing
`models/` symlinks (Parakeet-CTC, Orpheus, and the pattern flagged as likely affecting others too).*
Source documents: `docs/done/perf-loop-progress.md`, `docs/cpu-performance-baseline.md`,
`docs/tts-performance-baseline-and-plan.md`, `docs/done/cpu-speculative-decoding-findings.md`,
`docs/done/vulkan-backend-evidence.md`, `docs/done/gpu-review-log.md`, `GR_performance.md`,
`docs/perf-loop-project-review-progress.md`, `scripts/bench-audio.ps1`, `scripts/bench-cpp.ps1`, `README.md` git history commit `0c171ed`,
`docs/benchmark-prompt.txt` (2026-09-10 backfill prompt, 493-515 tok depending on tokenizer), `tools/llama.cpp/llama-bench.exe` (b8585-cpu),
`examples/audio.cpp/build/bin/audiocpp_cli.exe` (F5-TTS re-verification run).*

*Reproducibility: All runs can be replicated with `.\scripts\bench-cpp.ps1 -Suite Tts`, `.\scripts\bench-cpp.ps1 -Suite Align`, `.\scripts\bench-cpp.ps1 -Suite Whisper`, or `.\scripts\bench-cpp.ps1 -Suite All`. The 2026-09-10 LLM backfill commands: `tools/llama.cpp/llama-bench.exe -m <gguf> -p <n>[,<n>...] -n <n> -t 6 -ngl 0 -r 3` and `src/OpenTail.Stingray.Cli/bin/Release/net10.0/stingray.exe -m <gguf> -f docs/benchmark-prompt.txt -n 24 -g 0 --temp 0 --single-turn --no-display-prompt`, best-of-3 each.*
