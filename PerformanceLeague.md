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
| F5-TTS Base (DiT) | text → 2.77s audio | CPU | 27.25s | **9.82×** | — | — | — | 2026-09-09 | 2026-08-28 👂 |
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
| FunASR-Nano (Paraformer-based, `paraformer-q8.gguf`) | synthetic-tone audio, 99 tokens | CPU | 26.24s (single run) | — (not RTF-comparable, synthetic audio not real speech) | — | — | 2026-09-10 | new coverage; existing `FunAsrNanoEndToEndTests.cs`, timed externally. **Caveat:** output is degenerate word-salad ("to to to to... a a a at at..."), same class of bug as Qwen3-ASR's degenerate transcript — a second real ASR correctness gap, not a performance finding. |
| Silero VAD (ONNX) | 12s synthetic audio, segment detection | CPU | 73.24ms (mean of 8) | **0.0061×** | — (not attempted this pass) | — | — | 2026-09-10 | new coverage; existing `SileroVadPerfBenchTests.cs`, just needed a `models/silero_vad.onnx` symlink to `models/_models/silero_vad.onnx` (added, matching the existing symlink convention). **164x real-time — the fastest pipeline measured anywhere in this doc.** |
| MarbleNet VAD (safetensors, bundled in-repo) | segment detection, single run | CPU | 109ms | — (input duration not computed this pass, not RTF-comparable) | — | — | 2026-09-10 | new coverage; existing `MarbleNetVadRealWeightsTests.cs`, run as-is. Real, plausible segment detected (`[8320-56320] conf=0.978, 3.00s`) — a second real, working VAD model alongside Silero. |
| Orpheus-3B TTS + SNAC vocoder | transformer decode + vocoder, 6 runs | Vulkan iGPU (auto-selected) | transformer 14.4 tok/s mean, vocoder RTF 5.3x mean, total ~10.0s/run | — (not attempted on CPU this pass) | — | — | 2026-09-10 | new coverage; existing `OrpheusFullPipelinePerfBenchTests.cs`, needed `models/orpheus-3b-0.1-ft.Q4_K_M.gguf` + `models/snac-24khz.gguf` symlinks (checkpoints live in `models/_models/`, another instance of the silent-no-op-via-missing-symlink pattern — this one wasn't silent, it printed a clear "GGUFs not found" skip message). Test auto-selected Vulkan GPU without being asked. |

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

**Fixed 2026-09-11 — the ONNX path.** `stingray embed -m <file>.onnx` genuinely invokes ONNX
Runtime (confirmed by real, model-specific output dimensions below — not a stub), but crashed with
`Missing Input: token_type_ids` because `EmbedCommand.cs`'s ONNX branch never constructed that
tensor. Fixed by adding an all-zero `token_type_ids` input alongside `input_ids`/`attention_mask`
(`OnnxModelSession.Run` already filters to only the inputs a given model actually declares, so this
is safe for models that don't need it too). Also fixed in the same pass, found while testing:
`stingray embed -o <file>` crashed with `System.InvalidOperationException: Reflection-based
serialization has been disabled for this application` (NativeAOT/trim violation, `CLAUDE.md` rule
4) — replaced `JsonSerializer.Serialize` with a small hand-rolled JSON writer for the simple
`List<float[]>` output shape. And a defensive fix for a related but separate issue found while
fixing the crash: the ONNX branch's "tokenization" is not real WordPiece/BPE — it maps each raw
character to its char code as a placeholder token id (no `tokenizer.json`/`vocab.txt` ships
alongside these ONNX checkpoints on this machine) — this inflates apparent token count ~4x vs. real
subword tokenization and was overflowing BERT's 512-position limit on long inputs with an opaque
ONNX broadcast error. Added a defensive truncation with a clear warning instead of a crash; **the
underlying missing-real-tokenizer issue is not fixed**, just contained so it fails gracefully — the
numbers below use a short, single-sentence input specifically to avoid it, so they're not affected.

| Model | Scenario | Backend | C# result | C++ reference | Ratio | Performance Check | Source |
|---|---|---|---|---|---:|---|---|
| all-MiniLM-L6-v2 (quantized ONNX) | 1 text, 39 tok, mean pooling, 384-dim (real native dim) | CPU | 27ms (best of 3) | — (not attempted this pass) | — | 2026-09-11 | new coverage, post-fix; `stingray embed -m all-MiniLM-L6-v2_quantized.onnx -p "Hello, I will make some lunch, darling!"` |
| bge-small-en-v1.5 (quantized ONNX) | 1 text, 39 tok, mean pooling, 384-dim | CPU | 29ms (best of 3) | — (not attempted this pass) | — | 2026-09-11 | new coverage, post-fix |
| bge-base-en-v1.5 (quantized ONNX) | 1 text, 39 tok, mean pooling, 768-dim (implied by base-size BERT) | CPU | 35ms (best of 3) | — (not attempted this pass) | — | 2026-09-11 | new coverage, post-fix |
| bge-large-en-v1.5 (quantized ONNX) | 1 text, 39 tok, mean pooling, 1024-dim (real native dim) | CPU | 56ms (best of 3) | — (not attempted this pass) | — | 2026-09-11 | new coverage, post-fix |

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

## Vision Encoder (CPU)

| Component | Scenario | Backend | C# result | C++ reference | Ratio | Performance Check | Source |
|---|---|---|---|---|---:|---|---|
| VisionOps.Attention / AttentionGqa | 1024-tok / 16-head ViT-L | CPU | >1.2× over scalar | — | — | 2026-08-20 | perf-loop-project-review-progress.md |

> Scalar reference kept in `VisionOpsBenchmarkTests.cs` as a permanent regression baseline.

---

## Image & Video Diffusion (CPU) — out of core scope, tried anyway

Different domain from this doc's LLM/TTS/ASR focus (a different pipeline, `OpenTail.Stingray.Diffusion`,
not benchmarked here systematically) — included as real data points since the checkpoints were on
hand and untested.

| Model | Scenario | Backend | C# Wall | Performance Check | Source |
|---|---|---|---:|---|---|
| Z-Image-Turbo (S3-DiT + Qwen3-4B text encoder) | 512×512 image, default steps | CPU | 871.8s (14m32s) | 2026-09-10 | new coverage; `stingray image` CLI, real non-trivial 733KB PNG output saved to `docs/diffusion-samples/z-image-turbo-perfleague-check.png` |
| Wan2.1-T2V-1.3B (DiT + UMT5-XXL text encoder + VAE) | 512×512, 2 video frames, 20 denoising steps | CPU | 4238.7s (70.6 min) | 2026-09-11 | new coverage; `stingray image --video-frames 2` CLI (video generation is routed through the same `image` command). Real, non-trivial 692KB PNG output saved to `docs/diffusion-samples/wan2.1-t2v-1.3b-perfleague-check.png`. **~4.9x slower than Z-Image-Turbo's single image** despite only 2 frames — real cost of the larger UMT5-XXL text encoder plus video-specific DiT attention, not just "more frames." |

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
| F5-TTS | text → audio | CPU | Re-verified 2026-09-10: `f5_tts` family + `model_specs/f5_tts.json` exist and load, but generation fails with `ggml_graph_compute_with_ctx unavailable (CPU backend not loaded)` — the CPU compute backend genuinely isn't wired for this family's graph exec, confirmed with a real run (`--voice-ref b.wav --reference-text ... --text "Hello, I will make some lunch, darling!"`) | Needs the CPU ggml backend wired for F5-TTS's graph compute path in audio.cpp itself |
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
| Chatterbox Turbo (streaming) | TTFA vs C++ | CPU | Blocked on a missing `models/chatterbox_turbo_vocab.json` tokenizer asset referenced by `model_specs/chatterbox_turbo.json`, before even reaching the streaming-mode question | Locate/regenerate the missing vocab asset, then retry `--mode streaming` |
| FishSpeech S2 Pro (streaming) | TTFA vs C++ | CPU | `s2.exe --stream-file` genuinely works and prints real streaming metrics (2026-09-10: `stride=16, holdback=144, ref_encode=49957ms, generate=170348ms, total_rtf=61.32`), but doesn't print an explicit first-chunk timestamp — computing a TTFA-equivalent from the other fields would be an inference, not a real measurement | Add a printed first-chunk timestamp to `s2.exe`'s streaming metrics, then it's a real, direct TTFA comparison |

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
- **Two independent ASR pipelines produce degenerate output**: Qwen3-ASR 0.6B ("aspects" instead of the real reference sentence, despite running fast at RTF 0.225) and FunASR-Nano (repetitive word-salad — "to to to... a a a at at at"). Both are correctness bugs, not performance ones, and more urgent than any perf gap in this doc — a wrong-but-fast answer is worse than a slow-but-right one. Worth checking whether these share a root cause (e.g. a common sampling/decoding utility both pipelines call into) given how similar the failure mode is.
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
