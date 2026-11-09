# PerformanceLeague

> **Purpose:** C# vs C++ inference comparison, by model. The **Ratio** column is the primary signal: how close this engine is to its llama.cpp or C++ reference for each scenario. A blank ratio (`—`) means no C++ reference has been measured for that scenario.
>
> **This file holds the current numbers only.** Each measurement appears once, with its newest value. Older runs of the same measurement, the root-cause write-ups, the sweep logs and the methodology notes are in **[docs/4-performance/PerformanceLeague-LOG.md](docs/4-performance/PerformanceLeague-LOG.md)**, which is the complete, unedited history of this file (2026-11-03).
>
> **Every row carries a Performance Check date. A number without a date is hearsay.**

---

## How to read this file

- **Ratio** = C# ÷ C++ reference (llama.cpp, whisper.cpp, sd.cpp, audio.cpp). 1.0x = parity. Blank = no C++ reference measured for that row.
- **Same-pass vs cross-date:** a ratio is only quoted from a single session, unless the row says "cross-date", in which case it is indicative. Re-run `scripts/bench-vs-llamacpp.ps1` before quoting a cross-date ratio externally.
- **Scenario vocabulary:** `prefill` = prompt processing; `decode` = autoregressive generation. Context length is tagged where it matters (`decode @3.2k ctx`).
- **RTF** (TTS/ASR) = wall-clock ÷ audio duration. Lower is better; <1.0x is faster than real time.
- **TTFA** (TTS streaming) = time to first audio, in seconds.
- **Hardware (dev machine):** Ryzen 7 5700G, Zen 3, 8c/16t, AVX2+FMA, no VNNI/AVX-512, DDR4 (dual channel, ~51 GB/s theoretical peak). Vulkan = the same machine's integrated AMD Radeon (iGPU, shared system RAM). No CUDA device. **Every Vulkan number is an iGPU number and says nothing about a discrete GPU** (CLAUDE.md rule 13).
- **† Prior hardware:** Zen 4 (12c/24t) + RTX 4070 Ti 12 GB. Numbers from README history (commit `0c171ed`). Indicative only.
- **llama.cpp reference:** `tools/llama.cpp` b8585-cpu, and b10306 for the 2026-10 rows. GPU llama-bench where noted.
- **Quant notation:** Q4_K_M = Q4_K mixed; Q8_0 = 8-bit weights, 8-bit activations on the batched path.

---

## Contents

- [Text generation, CPU vs llama.cpp (ratio rows)](#text-generation-cpu-vs-llamacpp-ratio-rows)
- [Text generation, C# only (CPU and Vulkan iGPU, 2026-10-11 recheck)](#text-generation-c-only-cpu-and-vulkan-igpu-2026-10-11-recheck)
- [Text generation, Vulkan iGPU, SmolLM2-1.7B](#text-generation-vulkan-igpu-smollm2-17b-instruct-q4_k_m-2026-08--2026-09-28)
- [Newly-supported architectures (2026-09-13)](#newly-supported-architectures-2026-09-13-first-persistent-timings)
- [DeepSeek family](#deepseek-family-deepseek2)
- [gpt-oss-20B](#gpt-oss-20b-gpt-oss-mxfp4)
- [EXAONE-4.5-33B](#exaone-45-33b-exaone4-q4_k_m)
- [Speculative decoding (CPU)](#speculative-decoding-cpu)
- [Historical CUDA (†) and prior-hardware rows](#historical-cuda--and-prior-hardware-rows)
- [Text encoders and embeddings](#text-encoders-embeddings--rerank)
- [Vision-language models](#vision-language-model-text-backbones-cpu-and-vulkan-igpu-text-only)
- [TTS, streaming, forced alignment, ASR, RVC](#tts--audio-synthesis)
- [Image and video diffusion](#image--video-diffusion-cpu-and-vulkan-igpu)
- [Sweeps and experiments](#sweeps-and-experiments-cpu-2026-10)
- [Known measurement gaps and C++ head-to-head](#known-measurement-gaps--c-reference-head-to-head)
- [Strengths and weaknesses](#strengths--weaknesses-current)

---

## Text generation, CPU vs llama.cpp (ratio rows)

Ratio = C# ÷ llama.cpp. Each row is the newest C# measurement for its key. **Same-pass** rows (both numbers measured in one session, same prompt) carry one date. **Cross-date** rows pair a newer C# number with an older llama.cpp number; their ratio is indicative only (the file's rule: re-run `scripts/bench-vs-llamacpp.ps1` before quoting one).

| Model | Scenario | C# t/s (CPU) | date | llama.cpp t/s | date | Ratio |
|---|---|---:|---|---:|---|---|
| SmolLM2-1.7B-Instruct Q4_K_M | prefill, 3,698 tok (3 runs 163-188) | 163-188 | 2026-11-01 | 119.4 (pp3218, depth differs) | 2026-11-01 | ~1.4-1.6x (not depth-matched) |
| SmolLM2-1.7B-Instruct Q4_K_M | decode, ~3.7k ctx, 24 tok (best of 3) | 11.8 (mean 11.2) | 2026-11-01 | 12.04 (tg24 @ d3218) | 2026-11-01 | **0.98x** (0.93x mean; depth differs) |
| SmolLM2-1.7B-Instruct Q4_K_M | decode, 267 tok ctx | 26.3 | 2026-09-10 | 29.90 | 2026-09-10 | **0.88x** |
| SmolLM2-1.7B-Instruct Q4_K_M | decode, 773 tok ctx | 21.0 | 2026-09-10 | 23.46 | 2026-09-10 | **0.89x** |
| SmolLM2-1.7B-Instruct Q4_K_M | decode, 1,621 tok ctx | 15.0 | 2026-09-10 | 20.37 | 2026-09-10 | 0.74x |
| SmolLM2-1.7B-Instruct Q4_K_M | decode, 3,218 tok ctx | 9.5 | 2026-09-10 | 14.21 | 2026-09-10 | 0.67x |
| SmolLM2-1.7B-Instruct Q4_K_M | prefill, 991 tok, 6 runs (217-237) | 217-237 | 2026-09-26 | 260.6 (-t 16) | 2026-09-26 | ~0.85x |
| SmolLM2-1.7B-Instruct Q4_K_M | prefill, 735 tok (wiki.test.raw, int8 default) | 201.6 (median 3 runs) | 2026-11-01 | not stored | - | - |
| SmolLM2-135M-Instruct Q4_K_M | prefill, 554 tok | 1000.4 | 2026-09-11 | 1212.10 | 2026-09-11 | 0.83x |
| SmolLM2-135M-Instruct Q4_K_M | decode, 554 tok prompt, 24 tok gen | 39.6 | 2026-09-11 | 316.65 | 2026-09-11 | 0.125x |
| SmolLM2-360M-Instruct Q4_K_M | prefill, 554 tok | 369.5 | 2026-09-11 | 379.78 | 2026-09-11 | 0.97x |
| SmolLM2-360M-Instruct Q4_K_M | decode, 554 tok prompt, 24 tok gen | 26.0 | 2026-09-11 | 135.04 | 2026-09-11 | 0.19x |
| Qwen2.5-0.5B-Instruct Q4_K_M | prefill, 735 tok (int8 default) | 436.8 (median of 3) | 2026-11-01 | 455.6 (stored) | 2026-09-10 | 0.96x |
| Qwen2.5-0.5B-Instruct Q4_K_M | prefill, 661 tok, after int8 Q5_0 batched kernel (spread 269-340) | ~318 | 2026-09-28 | 379 | 2026-09-28 | ~0.84x |
| Qwen2.5-0.5B-Instruct Q4_K_M | decode, 51 tok prompt, 128 tok gen, after kernel (62.9-69.1) | 65.7 | 2026-09-28 | 98 (item 16, different prompt) | earlier | 0.67x (indicative) |
| Qwen2.5-0.5B-Instruct Q4_K_M | decode, 503 tok prompt, 24 tok gen | 58.2 | 2026-10-11 | 100.05 | 2026-09-11 | 0.58x (cross-date) |
| Qwen2.5-0.5B-Instruct Q4_K_M | decode, 59 tok prompt, 128 tok gen, `STINGRAY_CPU_POOL=spin` (opt-in) | 82 (81-84) | 2026-10-02 | 94.01 | 2026-10-02 | 0.87x |
| Qwen2.5-Coder-0.5B-Instruct Q4_K_M | prefill, 503 tok | 334.3 | 2026-09-11 | 307.70 | 2026-09-11 | **1.09x** |
| Qwen2.5-Coder-0.5B-Instruct Q4_K_M | decode, 503 tok prompt, 24 tok gen | 55.0 | 2026-10-11 | 82.87 | 2026-09-11 | 0.66x (cross-date) |
| Qwen2.5-1.5B-Instruct Q4_K_M | prefill, 503 tok | 181.9 | 2026-09-11 | 242.26 | 2026-09-11 | 0.75x |
| Qwen2.5-1.5B-Instruct Q4_K_M | decode, 503 tok prompt, 24 tok gen | 27.5 | 2026-09-11 | 41.55 | 2026-09-11 | 0.66x |
| Qwen2.5-3B-Instruct Q4_K_M | prefill, 503 tok | 83.0 | 2026-09-11 | 111.96 | 2026-09-11 | 0.74x |
| Qwen2.5-3B-Instruct Q4_K_M | decode, 503 tok prompt, 24 tok gen | 14.7 | 2026-09-11 | 20.76 | 2026-09-11 | 0.71x |
| Qwen2.5-Coder-1.5B-Instruct Q4_K_M | prefill, 503 tok | 170.7 | 2026-09-11 | 242.26 | 2026-09-11 | 0.70x |
| Qwen2.5-Coder-1.5B-Instruct Q4_K_M | decode, 503 tok prompt, 24 tok gen | 24.2 | 2026-09-11 | 41.55 | 2026-09-11 | 0.58x |
| Qwen2.5-Coder-3B-Instruct Q4_K_M | prefill, 503 tok | 87.1 | 2026-09-11 | 109.50 | 2026-09-11 | 0.80x |
| Qwen2.5-Coder-3B-Instruct Q4_K_M | decode, 503 tok prompt, 24 tok gen | 14.7 | 2026-09-11 | 21.48 | 2026-09-11 | 0.68x |
| Qwen3-0.6B Q8_0 | prefill, 493 tok | 226.1 | 2026-09-10 | 265.28 | 2026-09-10 | 0.85x |
| Qwen3-0.6B Q8_0 | decode, 493 tok prompt, 24 tok gen | 41.2 | 2026-09-10 | 62.95 | 2026-09-10 | 0.65x |
| Qwen3-4B Q4_K_M | prefill, 493 tok | 61.4 | 2026-09-10 | 84.36 | 2026-09-10 | 0.73x |
| Qwen3-4B Q4_K_M | decode, 493 tok prompt, 24 tok gen | 10.0 | 2026-09-10 | 16.64 | 2026-09-10 | 0.60x |
| Qwen3-8B Q4_K_M | prefill, 493 tok | 48.3 | 2026-09-10 | 47.28 | 2026-09-10 | **1.02x** |
| Qwen3-8B Q4_K_M | decode, 493 tok prompt, 24 tok gen | 6.3 | 2026-09-10 | 7.67 | 2026-09-10 | 0.82x |
| Qwen3-8B Q4_K_M | decode, 24 tok prompt, 128 tok gen, after Q4_K row-pair balancing (3 runs) | 7.7 | 2026-10-01 | 8.85 (8 thr) | 2026-10-01 | **0.87x** |
| Qwen3-Coder-30B-A3B-Instruct Q4_K_M | prefill, 507 tok, after shared MoE batching (3 alternating runs 66.7-69.3) | 66.7-69.3 | 2026-09-28 | 89.27 ± 0.93 | 2026-09-28 | 0.75x |
| Qwen3-Coder-30B-A3B-Instruct Q4_K_M | prefill, 482 tok (backfill) | 24.7 | 2026-09-10 | 41.76 | 2026-09-10 | 0.59x (482-tok prompt, unsloth checkpoint; the 2026-09-28 row is a 507-tok prompt on the same checkpoint) |
| Qwen3-Coder-30B-A3B-Instruct Q4_K_M | decode, 482 tok prompt, 24 tok gen | 7.4 (2026-09-10) → 12.4 (2026-10-11) | 2026-10-11 | 20.4 (llama-bench re-run 2026-10-04; the 6.85 of 2026-09-10 is withdrawn) | 2026-10-04 | 0.61x (cross-date) |
| Qwen3.6-35B-A3B Q6_K (hybrid GDN) | prefill, 515 tok, after all MoE/GDN passes (3 runs 62.3-63.2) | 62.3-63.2 | 2026-09-28 | 65.42 ± 0.38 | 2026-09-28 | ~0.96x |
| Qwen3.6-35B-A3B Q6_K (hybrid GDN) | prefill, 522 tok, after batched MoE and projections | 36.7 | 2026-09-25 | 58.66 ± 0.59 | 2026-09-25 | 0.63x |
| Qwen3.6-35B-A3B Q6_K (hybrid GDN) | decode, 522 tok prompt, 24 tok gen | 2.9 | 2026-09-25 | 11.15 ± 0.05 | 2026-09-25 | 0.26x |
| Qwen3.6-27B Q3_K_XL (hybrid GDN) | prefill, 480 tok (unsloth Q3_K_XL) | 1.0 | 2026-09-10 | 6.15 | 2026-09-10 | 0.16x |
| Qwen3.6-27B Q3_K_XL (hybrid GDN) | decode, 480 tok prompt, 24 tok gen | 1.0 | 2026-09-10 | 1.33 | 2026-09-10 | 0.75x |
| Qwen3.6-27B UD-Q3_K_XL (hybrid GDN) | prefill, 702 tok (int8 default; weakest ratio, different quant and prompt for the reference) | 8.4 (8.3-8.5) | 2026-11-01 | 6.15 | 2026-09-10 | 1.37x (indicative) |
| Qwen3.8-27B Q3_K_XL (hybrid GDN, qwen35) | prefill, 522 tok (earliest, single run) | 0.2 | 2026-09-10 | 3.00 | 2026-09-10 | 0.07x |
| Qwen3.8-27B Q3_K_XL (hybrid GDN, qwen35) | decode, 522 tok prompt (earliest) | 0.2 | 2026-09-10 | 1.55 | 2026-09-10 | 0.13x |
| Qwen3.8-27B UD-Q3_K_XL (hybrid GDN + MTP head) | prefill, 620 tok, after ADR-0002 chunked prefill | 8.8 | 2026-10-02 | 4.36 (pp512) | 2026-10-02 | ~2.0x (lengths differ) |
| Qwen3.8-27B UD-Q3_K_XL | prefill, 568 tok, re-trace (2 runs) | 9.1 | 2026-10-08 | 4.36 (pp512) | 2026-10-02 | ~2.1x (indicative) |
| Qwen3.8-27B UD-Q3_K_XL | decode, 64 tok prompt, 48 tok gen, plain | 2.1-2.2 | 2026-09-25 | 2.5 | 2026-09-25 | ~0.86x |
| Ornith-1.0-9B Q4_K_M (hybrid GDN, qwen35) | prefill, 575 tok, after batched chunked FFN (interleaved x2) | 33.0 (32.6-33.4) | 2026-10-02 | 48.12 (pp512) | 2026-10-02 | 0.69x |
| Ornith-1.0-9B Q4_K_M (hybrid GDN, qwen35) | decode, 575 tok prompt, 32 tok gen | 6.8 (6.4-6.9) | 2026-10-02 | 7.24 (tg64) | 2026-10-02 | 0.94x |
| Ornith-1.0-9B Q4_K_M (hybrid GDN, qwen35) | prefill, 480 tok (earliest; repeated-token output, see LOG) | 1.7 | 2026-09-10 | 20.55 | 2026-09-10 | 0.08x (superseded by the 2026-10-02 row) |
| Gemma-4-12B-it Q4_K_M | prefill, 501 tok, CPU batched (int8 at the time) | 28.5 | 2026-09-16 | 27.81 | 2026-09-16 | 1.02x |
| Gemma-4-12B-it Q4_K_M | prefill, 587 tok, default F32 prefill (int8 opt-in) | 14.5 | 2026-10-02 | 37.55 (pp512) | 2026-10-02 | 0.39x |
| Gemma-4-12B-it Q4_K_M | prefill, 587 tok, `STINGRAY_CPU_PREFILL_Q8=1` | 36.4 | 2026-10-02 | 37.55 (pp512) | 2026-10-02 | 0.97x |
| Gemma-4-12B-it Q4_K_M | decode, 587 tok prompt, 32 tok gen | 4.6 | 2026-10-02 | 5.64 (tg64) | 2026-10-02 | 0.84x |
| Gemma-4-E4B-it Q4_K_M | prefill, 495 tok, default int8 (rollback F32 = 31.0, 0.37x) | 64.3 (61.3-64.3) | 2026-11-01 | 83.62 ± 1.21 (pp512, `-t 8 -r 2`) | 2026-11-01 | 0.77x |
| Gemma-4-E4B-it Q4_K_M | decode, 495 tok prompt, 16 tok gen | 11.0 (10.9-11.0) | 2026-11-01 | 12.32 ± 0.17 (tg16) | 2026-11-01 | 0.89x |
| Gemma-4-E4B-it Q4_K_M | prefill, 583 tok, F32 default (superseded by the 2026-11-01 row) | 32.3 | 2026-10-02 | 98.69 | 2026-10-02 | 0.33x |
| Gemma-4-E4B-it Q4_K_M | decode, 583 tok prompt, 32 tok gen, F32 default | 10.9 | 2026-10-02 | 12.86 | 2026-10-02 | 0.85x |
| Gemma-4-E4B-it Q4_K_M | prefill, 497 tok, CPU batched (superseded by 2026-11-01) | 49.2 | 2026-09-16 | 80.50 | 2026-09-10 | 0.61x |
| Gemma-4-E4B-it Q4_K_M | decode, 497 tok prompt, 24 tok gen | 7.7-9.7 | 2026-09-10 | 13.07 | 2026-09-10 | 0.74x |
| Gemma-3-4B-it Q4_K_M | prefill, 495 tok, default int8 (rollback 43.4 = 0.39x) | 105.3 (100.0-105.3) | 2026-11-01 | 110.32 ± 4.16 (pp512, `-t 8 -r 2`) | 2026-11-01 | **0.95x** |
| Gemma-3-4B-it Q4_K_M | decode, 495 tok prompt, 16 tok gen | 13.2 (13.0-13.2) | 2026-11-01 | 14.69 ± 0.28 (tg16) | 2026-11-01 | **0.90x** |
| Gemma-3-4B-it Q4_K_M | prefill, 590 tok, F32 default (superseded by 2026-11-01) | 44.0-44.7 | 2026-10-02 | 114.02 | 2026-10-02 | 0.39x |
| Gemma-3-4B-it Q4_K_M | decode, 590 tok prompt, 32 tok gen | 13.5 | 2026-10-02 | 16.02 | 2026-10-02 | 0.85x |
| Gemma-3-4B-it Q4_K_M | prefill, 490 tok, CPU batched | 62.2 | 2026-09-16 | 97.32 | 2026-09-16 | 0.64x |
| Gemma-3-4B-it Q4_K_M | decode, 490 tok prompt, 24 tok gen | 9.6-12.1 | 2026-09-11 | 15.31 | 2026-09-11 | 0.79x |
| Mistral-7B-Instruct-v0.3 Q4_K_M | prefill, 571 tok | 46.4 | 2026-09-10 | 46.27 | 2026-09-10 | **1.00x** |
| Mistral-7B-Instruct-v0.3 Q4_K_M | decode, 24 tok prompt, 128 tok gen, after Q4_K row-pair balancing (6 runs) | 8.3 (8.2-8.5) | 2026-10-01 | 9.45 (8 thr) | 2026-10-01 | **0.88x** |
| Mistral-7B-Instruct-v0.3 Q4_K_M | decode, 571 tok prompt, 24 tok gen (earliest; superseded) | 6.5 | 2026-09-10 | 9.37 | 2026-09-10 | 0.69x |
| Ministral-8B-Instruct-2410 Q4_K_M | prefill, 488 tok | 45.2 | 2026-09-10 | 45.79 | 2026-09-10 | **0.99x** |
| Ministral-8B-Instruct-2410 Q4_K_M | decode, 24 tok prompt, 128 tok gen | 7.5 (7.3-7.8) | 2026-10-01 | 8.90 (8 thr) | 2026-10-01 | 0.84x |
| Ministral-8B-Instruct-2410 Q4_K_M | decode, 488 tok prompt, 24 tok gen | 6.1 | 2026-09-10 | 8.61 | 2026-09-10 | 0.71x |
| RWKV7-Goose-World3-1.5B Q8_0 | decode, 128 tok gen, after per-head WKV | 22.2 (22.1-22.3) | 2026-10-02 | 23.36 (8 thr) | 2026-10-01 | 0.95x |
| RWKV7-Goose-World3-1.5B Q8_0 | prefill, 605 tok, after Phase 16 | 76 (69.9-80.3) | 2026-10-02 | 185.10 (pp512, 16 thr) | 2026-10-01 | 0.41x |
| rwkv-6-world-1.6b Q8_0 | decode, 64 tok gen, after Phase 16 | 21.8 (21.5-21.9) | 2026-10-02 | 23.12 (8 thr) | 2026-10-01 | 0.94x |
| rwkv-6-world-1.6b Q8_0 | prefill, 590 tok, after Phase 16 | 78.8 (78.3-79.1) | 2026-10-02 | 171.08 (pp512) | 2026-10-01 | 0.46x |
| OLMoE-1B-7B-0924-Instruct Q4_K_M | prefill, 802 tok (llama-bench pp687; llama.cpp noisy) | 162.5 / 182.6 | 2026-10-30 | 218.41 ± 36.85 | 2026-10-30 | ~0.8x |
| OLMoE-1B-7B-0924-Instruct Q4_K_M | decode, 64 tok gen, short prompt | 43.5 / 43.8 | 2026-10-30 | 53.67 ± 0.34 (tg64, `-t 8`) | 2026-10-30 | **0.81x** |
| OLMoE-1B-7B-0924-Instruct Q4_K_M | prefill, 515 tok (earliest; superseded) | 124.8 | 2026-09-10 | 180.82 | 2026-09-10 | 0.69x |
| OLMoE-1B-7B-0924-Instruct Q4_K_M | decode, 515 tok prompt, 24 tok gen (earliest; superseded) | 25.3 | 2026-09-10 | 50.58 | 2026-09-10 | 0.50x |
| DeepSeek-V2-Lite-Chat Q8_0 | prefill, 502 tok | 28.0 | 2026-09-11 | 56.90 | 2026-09-11 | 0.49x |
| DeepSeek-V2-Lite-Chat Q8_0 | decode, 502 tok prompt, 24 tok gen | 13.6 | 2026-09-11 | 14.83 | 2026-09-11 | **0.92x** |
| DeepSeek-V2-Lite-Chat Q8_0 | decode, 502 tok prompt (2026-10-11 recheck) | 11.7 | 2026-10-11 | 14.83 | 2026-09-11 | 0.79x (cross-date) |

> **Pattern (current):** at 7-12B dense sizes CPU prefill is at or near parity with llama.cpp (Qwen3-8B 1.02x, Mistral-7B 1.00x, Ministral-8B 0.99x, Gemma-3-4B 0.95x). Decode is the gap: tiny models are worst (SmolLM2-135M 0.125x), and most 1-4B rows sit at 0.6-0.9x. Hybrid-GDN checkpoints were the weakest prefill rows (0.05-0.16x on 2026-09-10); the 2026-09-25 and 2026-09-28 batched-prefill work brought the 35B to 0.63-0.96x.

## Text generation, C# only (CPU and Vulkan iGPU, 2026-10-11 recheck)

Engine defaults, one run per row unless noted. Prompt: about 490-580 tokens (`docs/reference/benchmark-prompt.txt`), 16 new tokens, CPU `-g 0`, Vulkan `-g -1`. **No llama.cpp reference was re-measured here.** Rows that have a newer measurement are updated to that measurement and marked with its date. A single-digit percent difference is noise. Rows marked "short" use a 15-40 token prompt, and "2k" uses about 1.9-2.3k tokens. Rule 13 applies to every Vulkan number (iGPU only).

| Model | Backend | Scenario | Prompt tok | Prefill t/s | Decode t/s | Performance Check |
|---|---|---|---:|---:|---:|---|
| SmolLM2-1.7B-Instruct Q4_K_M | cpu | short | 40 | 120.8 | 29.9 | 2026-10-11 |
| SmolLM2-1.7B-Instruct Q4_K_M | cpu | 500 | 555 | 232.9 | 24.5 | 2026-10-11 |
| SmolLM2-1.7B-Instruct Q4_K_M | vulkan | short | 40 | 95.0 | 28.8 | 2026-10-11 |
| SmolLM2-1.7B-Instruct Q4_K_M | vulkan | 500 | 555 | 235.5 | 23.7 | 2026-10-11 |
| SmolLM2-1.7B-Instruct Q4_K_M | vulkan | 2k | 2130 | 153.2 | 17.2 | 2026-10-11 |
| Qwen3-4B-Q4_K_M | cpu | short | 30 | 52.3 | 13.7 | 2026-10-11 |
| Qwen3-4B-Q4_K_M | cpu | 500 | 493 | 103.5 | 12.0 | 2026-10-11 |
| Qwen3-4B-Q4_K_M | vulkan | short | 30 | 40.5 | 12.4 | 2026-10-11 |
| Qwen3-4B-Q4_K_M | vulkan | 500 | 493 | 107.7 | 12.0 | 2026-10-11 |
| Qwen3-4B-Q4_K_M | vulkan | 2k | 1915 | 100.0 | 10.5 | 2026-10-11 |
| Qwen3.5-4B-Q4_K_M | cpu | short | 21 | 26.1 | 11.6 | 2026-10-11 |
| Qwen3.5-4B-Q4_K_M | cpu | 500 | 480 | 59.0 | 10.8 | 2026-10-11 |
| Qwen3.5-4B-Q4_K_M | vulkan | short | 21 | 23.9 | 10.6 | 2026-10-11 |
| Qwen3.5-4B-Q4_K_M | vulkan | 500 | 480 | 62.1 | 10.3 | 2026-10-11 |
| Qwen3.5-4B-Q4_K_M | vulkan | 2k | 1893 | 64.6 | 10.6 | 2026-10-11 |
| gemma-4-E4B-it-Q4_K_M | cpu | 500 | 495 | 64.7 | 10.6 | 2026-11-01 (see ratio table) |
| gemma-4-E4B-it-Q4_K_M | vulkan | short | 19 | 3.7 (3.0-3.8 over 6 runs) | 10.1 | 2026-11-01 (9.3 on 2026-10-11, not explained) |
| gemma-4-E4B-it-Q4_K_M | vulkan | 500 | 495 | 59.3 (38.8-59.3 over 3 runs) | 3.3 (2.9-3.3 over 4 runs) | 2026-11-01 (2026-10-11: 55.7 / 9.0) |
| gemma-4-E4B-it-Q4_K_M | vulkan | 2k | 1959 | 61.1 | 9.3 | 2026-10-11 |
| Mistral-7B-Instruct-v0.3-Q4_K_M | cpu | short | 16 | 21.7 | 7.7 | 2026-10-11 |
| Mistral-7B-Instruct-v0.3-Q4_K_M | cpu | 500 | 577 | 56.6 | 7.0 | 2026-10-11 |
| Mistral-7B-Instruct-v0.3-Q4_K_M | vulkan | short | 16 | 21.5 | 8.1 | 2026-10-11 |
| Mistral-7B-Instruct-v0.3-Q4_K_M | vulkan | 500 | 577 | 56.8 | 7.8 | 2026-10-11 |
| Mistral-7B-Instruct-v0.3-Q4_K_M | vulkan | 2k | 2293 | 43.5 | 7.2 | 2026-10-11 |

Other rows from the 2026-10-11 all-checkpoint sweep (CPU and Vulkan, 500-token prompt, one run each) that have no newer measurement are in the table below. The 2026-11-01 and later rows above replace the matching entries here.

| Model | GB | Arch | Backend | Prompt tok | Prefill t/s | Decode t/s |
|---|---:|---|---|---:|---:|---:|
| SmolLM2-135M-Instruct-Q4_K_M | 0.1 | llama | cpu | 555 | 995.0 | 94.0 |
| SmolLM2-135M-Instruct-Q4_K_M | 0.1 | llama | vulkan | 555 | 65.7 | 58.8 |
| pythia-160m.Q8_0 | 0.16 | gptneox | cpu | 529 | 894.0 | 117.5 |
| pythia-160m.Q8_0 | 0.16 | gptneox | vulkan | 529 | 163.3 | 122.1 |
| gpt2.Q8_0 | 0.17 | gpt2 | cpu | 532 | 896.2 | 116.3 |
| gpt2.Q8_0 | 0.17 | gpt2 | vulkan | 532 | 152.9 | 116.2 |
| openai-gpt2-124M-F16 | 0.24 | gpt2 | cpu | 532 | 205.0 | 90.7 |
| openai-gpt2-124M-F16 | 0.24 | gpt2 | vulkan | 532 | 68.8 | 59.7 |
| SmolLM2-360M-Instruct-Q4_K_M | 0.25 | llama | vulkan | 555 | 30.3 | 28.5 |
| granite-4.0-h-350m-Q8_0 | 0.34 | granitehybrid | cpu | 493 | 285.8 | 61.3 |
| ERNIE-4.5-0.3B-Q8_0 | 0.36 | ernie4_5 | cpu | 532 | 400.8 | 68.3 |
| ERNIE-4.5-0.3B-Q8_0 | 0.36 | ernie4_5 | vulkan | 532 | 46.5 | 43.1 |
| qwen2.5-0.5b-instruct-q4_k_m | 0.46 | qwen2 | vulkan | 503 | 28.4 | 27.1 |
| qwen2.5-coder-0.5b-instruct-q4_k_m | 0.46 | qwen2 | vulkan | 503 | 28.6 | 27.3 |
| jais-family-590m-chat.Q4_K_M | 0.47 | jais | cpu | 610 | 214.5 | 45.5 |
| jais-family-590m-chat.Q4_K_M | 0.47 | jais | vulkan | 610 | 56.5 | 47.8 |
| Qwen3.5-0.8B-Q4_K_M | 0.5 | qwen35 | cpu | 480 | 280.1 | 37.5 |
| Qwen3.5-0.8B-Q4_K_M | 0.5 | qwen35 | vulkan | 480 | 265.2 | 35.9 |
| tencent_Hunyuan-0.5B-Instruct-Q8_0 | 0.54 | hunyuan-dense | cpu | 484 | 241.0 | 46.9 |
| tencent_Hunyuan-0.5B-Instruct-Q8_0 | 0.54 | hunyuan-dense | vulkan | 484 | 34.1 | 31.2 |
| qwen3-embedding-0.6b-q8_0 | 0.6 | qwen3 | cpu | 493 | 234.6 | 43.1 |
| qwen3-embedding-0.6b-q8_0 | 0.6 | qwen3 | vulkan | 493 | 29.4 | 27.1 |
| Qwen3-0.6B-Q8_0 | 0.6 | qwen3 | vulkan | 493 | 30.1 | 27.6 |
| Maincoder-1B-Q4_K_M | 0.63 | maincoder | cpu | 493 | 356.5 | 34.4 |
| Maincoder-1B-Q4_K_M | 0.63 | maincoder | vulkan | 493 | 283.5 | 26.8 |
| Qwen3.5-0.8B-Q8_0 | 0.78 | qwen35 | cpu | 480 | 288.9 | 33.9 |
| Qwen3.5-0.8B-Q8_0 | 0.78 | qwen35 | vulkan | 480 | 59.3 | 20.5 |
| rwkv-6-world-1.6b-Q4_K_S | 0.92 | rwkv6 | cpu | 501 | 96.6 | 28.0 |
| rwkv-6-world-1.6b-Q4_K_S | 0.92 | rwkv6 | vulkan | 501 | 119.8 | 28.2 |
| RWKV7-Goose-World3-1.5B-HF.Q4_K_S | 0.92 | rwkv7 | cpu | 516 | 79.7 | 27.4 |
| RWKV7-Goose-World3-1.5B-HF.Q4_K_S | 0.92 | rwkv7 | vulkan | 516 | 98.6 | 25.9 |
| SmolLM2-1.7B-Instruct-Q4_K_M | 0.98 | llama | vulkan | 555 | 231.8 | 23.8 |
| rwkv7-g1h-1.5b-20260710-ctx10240.i1-Q4_K_M | 1.01 | rwkv7 | cpu | 504 | 84.0 | 27.9 |
| rwkv7-g1h-1.5b-20260710-ctx10240.i1-Q4_K_M | 1.01 | rwkv7 | vulkan | 504 | 107.4 | 28.0 |
| Maincoder-1B-Q8_0 | 1.02 | maincoder | cpu | 493 | 107.2 | 25.0 |
| Maincoder-1B-Q8_0 | 1.02 | maincoder | vulkan | 493 | 18.8 | 17.0 |
| qwen2.5-1.5b-instruct-q4_k_m | 1.04 | qwen2 | vulkan | 503 | 279.2 | 27.3 |
| InternVL3-2B.Q4_K_M | 1.04 | qwen2 | cpu | 503 | 225.9 | 23.8 |
| InternVL3-2B.Q4_K_M | 1.04 | qwen2 | vulkan | 503 | 281.8 | 27.3 |
| qwen2.5-coder-1.5b-instruct-q4_k_m | 1.04 | qwen2 | vulkan | 503 | 282.4 | 27.2 |
| granite-vision-3.2-2b-Q3_K_S | 1.05 | granite | cpu | 534 | 51.8 | 19.5 |
| granite-vision-3.2-2b-Q3_K_S | 1.05 | granite | vulkan | 534 | 14.1 | 13.5 |
| qwen3-embedding-0.6b-f16 | 1.12 | qwen3 | cpu | 493 | 47.2 | 22.5 |
| qwen3-embedding-0.6b-f16 | 1.12 | qwen3 | vulkan | 493 | 15.7 | 15.3 |
| SmolLM2-1.7B-Instruct-Q5_K_M | 1.14 | llama | cpu | 555 | 30.1 | 16.3 |
| SmolLM2-1.7B-Instruct-Q5_K_M | 1.14 | llama | vulkan | 555 | 22.3 | 20.1 |
| LFM2-1.2B-Q8_0 | 1.16 | lfm2 | cpu | 516 | 108.1 | 24.3 |
| OLMo-1B.Q8_0 | 1.17 | olmo | cpu | 529 | 27.5 | 24.7 |
| granite-3.0-1b-a400m-instruct-Q8_0 | 1.32 | granitemoe | cpu | 534 | 261.6 | 45.4 |
| granite-3.0-1b-a400m-instruct-Q8_0 | 1.32 | granitemoe | vulkan | 534 | 34.7 | 32.5 |
| ibm-granite_granite-3.3-2b-instruct-Q4_K_M | 1.44 | granite | cpu | 534 | 120.6 | 17.9 |
| ibm-granite_granite-3.3-2b-instruct-Q4_K_M | 1.44 | granite | vulkan | 534 | 150.7 | 18.7 |
| Qwen3.5-0.8B-BF16 | 1.45 | qwen35 | cpu | 480 | 105.3 | 18.9 |
| Qwen3.5-0.8B-BF16 | 1.45 | qwen35 | vulkan | 480 | 34.4 | 16.1 |
| granite-4.0-h-1b-Q8_0 | 1.45 | granitehybrid | cpu | 493 | 67.3 | 17.4 |
| OLMo-2-0425-1B-Q8_0 | 1.47 | olmo2 | cpu | 471 | 101.9 | 17.3 |
| RWKV7-Goose-World3-1.5B-HF.Q8_0 | 1.58 | rwkv7 | cpu | 516 | 49.7 | 22.6 |
| RWKV7-Goose-World3-1.5B-HF.Q8_0 | 1.58 | rwkv7 | vulkan | 516 | 83.9 | 22.3 |
| gemma-2-2b-it-Q4_K_M | 1.59 | gemma2 | cpu | 491 | 175.2 | 17.4 |
| gemma-2-2b-it-Q4_K_M | 1.59 | gemma2 | vulkan | 491 | 67.0 | 16.1 |
| stablelm-zephyr-3b.Q4_K_M | 1.59 | stablelm | cpu | 526 | 108.4 | 15.8 |
| stablelm-zephyr-3b.Q4_K_M | 1.59 | stablelm | vulkan | 526 | 114.0 | 9.6 |
| rwkv-6-world-1.6b-Q8_0 | 1.63 | rwkv6 | vulkan | 501 | 83.8 | 21.7 |
| Qwen3VL-2B-Instruct-Q8_0 | 1.71 | qwen3vl | cpu | 482 | 74.9 | 18.1 |
| Qwen3VL-2B-Instruct-Q8_0 | 1.71 | qwen3vl | vulkan | 482 | 13.2 | 12.8 |
| dots.ocr-Q8_0 | 1.76 | qwen2 | cpu | 477 | 102.6 | 20.5 |
| dots.ocr-Q8_0 | 1.76 | qwen2 | vulkan | 477 | 21.0 | 20.1 |
| starcoder2-3b.Q4_K_M | 1.76 | starcoder2 | cpu | 550 | 103.2 | 13.0 |
| starcoder2-3b.Q4_K_M | 1.76 | starcoder2 | vulkan | 550 | 119.7 | 13.4 |
| SmolLM3-Q4_K_M | 1.78 | smollm3 | cpu | 705 | 110.7 | 15.5 |
| SmolLM3-Q4_K_M | 1.78 | smollm3 | vulkan | 705 | 138.2 | 15.2 |
| Falcon3-3B-Instruct-Q4_K_M | 1.87 | llama | cpu | 507 | 126.4 | 15.3 |
| Falcon3-3B-Instruct-Q4_K_M | 1.87 | llama | vulkan | 507 | 55.0 | 16.9 |
| TinyLLama-4x1.1B-MoE.Q4_K_M | 1.9 | llama | cpu | 595 | 112.0 | 22.9 |
| TinyLLama-4x1.1B-MoE.Q4_K_M | 1.9 | llama | vulkan | 595 | 25.2 | 24.4 |
| rwkv7-g1h-2.9b-20260710-ctx10240.i1-Q4_K_M | 1.91 | rwkv7 | cpu | 504 | 38.1 | 13.5 |
| rwkv7-g1h-2.9b-20260710-ctx10240.i1-Q4_K_M | 1.91 | rwkv7 | vulkan | 504 | 50.8 | 15.5 |
| qwen2.5-coder-3b-instruct-q4_k_m | 1.96 | qwen2 | vulkan | 503 | 147.5 | 15.1 |
| qwen2.5-3b-instruct-q4_k_m | 1.96 | qwen2 | vulkan | 503 | 148.3 | 15.5 |
| granite-4.0-3b-vision-Q4_K_M | 2.09 | granite | cpu | 471 | 94.1 | 13.3 |
| granite-4.0-3b-vision-Q4_K_M | 2.09 | granite | vulkan | 471 | 119.0 | 14.3 |
| orpheus-3b-0.1-ft.Q4_K_M | 2.2 | llama | cpu | 498 | 102.4 | 9.2 |
| orpheus-3b-0.1-ft.Q4_K_M | 2.2 | llama | vulkan | 498 | 149.1 | 12.5 |
| Phi-3-mini-4k-instruct-Q4_K_M | 2.23 | phi3 | cpu | 588 | 27.5 | 11.8 |
| Phi-3-mini-4k-instruct-Q4_K_M | 2.23 | phi3 | vulkan | 588 | 10.3 | 8.3 |
| phi-2.Q4_K_M | 2.23 | phi3 | cpu | 588 | 27.4 | 9.8 |
| phi-2.Q4_K_M | 2.23 | phi3 | vulkan | 588 | 10.3 | 8.2 |
| Qwen3-4B-Q4_K_M | 2.33 | qwen3 | vulkan | 493 | 108.0 | 12.3 |
| Qwen3.5-4B-Q4_K_M | 2.55 | qwen35 | cpu | 480 | 49.8 | 8.6 |
| Qwen3.5-4B-Q4_K_M | 2.55 | qwen35 | vulkan | 480 | 62.7 | 10.8 |
| Z-Image-AbliteratedV1.Q5_K_M | 2.69 | qwen3 | cpu | 493 | 11.4 | 8.4 |
| Z-Image-AbliteratedV1.Q5_K_M | 2.69 | qwen3 | vulkan | 493 | 10.3 | 10.0 |
| mimo-vl-7b-sft-Q2_K | 2.87 | qwen2vl | cpu | 499 | 21.4 | 10.6 |
| mimo-vl-7b-sft-Q2_K | 2.87 | qwen2vl | vulkan | 499 | 2.2 | 2.2 |
| step3-vl-10b-Q2_K | 3.06 | qwen3 | cpu | 495 | 25.7 | 11.0 |
| step3-vl-10b-Q2_K | 3.06 | qwen3 | vulkan | 495 | 2.0 | 2.0 |
| qwen2.5-7b-instruct-q4_k_m-00001-of-00002 | 3.72 | qwen2 | cpu | 503 | 63.4 | 7.7 |
| qwen2.5-7b-instruct-q4_k_m-00001-of-00002 | 3.72 | qwen2 | vulkan | 503 | 62.8 | 7.4 |
| ggml-model-q4_k | 3.8 | llama | cpu | 587 | 60.0 | 7.0 |
| ggml-model-q4_k | 3.8 | llama | vulkan | 587 | 62.5 | 7.6 |
| OLMoE-1B-7B-0924-Instruct-Q4_K_M | 3.92 | olmoe | vulkan | 516 | 133.1 | 21.8 |
| Mistral-7B-Instruct-v0.3-Q4_K_M | 4.07 | llama | vulkan | 577 | 57.0 | 7.7 |
| xverse-7b-chat-q4_k_m | 4.16 | xverse | cpu | 843 | 51.8 | 6.9 |
| xverse-7b-chat-q4_k_m | 4.16 | xverse | vulkan | 843 | 63.7 | 6.6 |
| ggml-model-Q4_K_M | 4.36 | qwen2 | cpu | 493 | 54.6 | 7.0 |
| ggml-model-Q4_K_M | 4.36 | qwen2 | vulkan | 493 | 62.7 | 7.1 |
| Qwen2.5-VL-7B-Instruct-Q4_K_M | 4.36 | qwen2vl | cpu | 493 | 51.2 | 7.2 |
| Qwen2.5-VL-7B-Instruct-Q4_K_M | 4.36 | qwen2vl | vulkan | 493 | 62.5 | 7.2 |
| nemotron-nano-12b-v2-vl-Q2_K | 4.39 | nemotron_h | cpu | 508 | 11.7 | 6.7 |
| Ministral-8B-Instruct-2410-Q4_K_M | 4.57 | llama | vulkan | 498 | 61.3 | 7.2 |
| llava-llama-3-8b-v1_1-int4 | 4.58 | llama | cpu | 503 | 45.7 | 6.6 |
| llava-llama-3-8b-v1_1-int4 | 4.58 | llama | vulkan | 503 | 62.0 | 7.1 |
| tiiuae-falcon-7b-instruct-Q4_K_M | 4.63 | falcon | cpu | 536 | 7.1 | 4.3 |
| tiiuae-falcon-7b-instruct-Q4_K_M | 4.63 | falcon | vulkan | 536 | 2.3 | 2.3 |
| gemma-4-E4B-it-Q4_K_M | 4.64 | gemma4 | vulkan | 495 | 51.8 | 8.5 |
| Qwen3-8B-Q4_K_M | 4.68 | qwen3 | cpu | 493 | 50.5 | 6.6 |
| Qwen3-8B-Q4_K_M | 4.68 | qwen3 | vulkan | 493 | 60.3 | 6.9 |
| LFM2-8B-A1B-Q4_K_M | 4.7 | lfm2moe | cpu | 516 | 118.3 | 24.0 |
| swiss-ai.Apertus-8B-Instruct-2509.Q4_K_M | 4.71 | apertus | cpu | 546 | 43.5 | 7.3 |
| swiss-ai.Apertus-8B-Instruct-2509.Q4_K_M | 4.71 | apertus | vulkan | 546 | 52.2 | 6.8 |
| c4ai-command-r7b-12-2024-Q4_K_M | 4.71 | cohere2 | cpu | 500 | 44.3 | 6.1 |
| c4ai-command-r7b-12-2024-Q4_K_M | 4.71 | cohere2 | vulkan | 500 | 55.3 | 6.5 |
| swiss-ai_Apertus-8B-Instruct-2509-Q4_K_M | 4.71 | apertus | cpu | 546 | 36.4 | 6.2 |
| swiss-ai_Apertus-8B-Instruct-2509-Q4_K_M | 4.71 | apertus | vulkan | 546 | 53.1 | 6.8 |
| llava-v1.6-mistral-7b.Q5_K_M | 4.78 | llama | cpu | 580 | 5.7 | 5.6 |
| llava-v1.6-mistral-7b.Q5_K_M | 4.78 | llama | vulkan | 580 | 6.3 | 6.2 |
| youtu-vl-4b-Q8_0 | 4.85 | deepseek2 | cpu | 497 | 25.7 | 5.3 |
| Qwen3.5-9B-Q4_K_M | 5.29 | qwen35 | cpu | 480 | 31.9 | 6.3 |
| Qwen3.5-9B-Q4_K_M | 5.29 | qwen35 | vulkan | 480 | 42.6 | 6.3 |
| deepreinforce-ai_Ornith-1.0-9B-Q4_K_M | 5.5 | qwen35 | cpu | 480 | 30.5 | 6.0 |
| deepreinforce-ai_Ornith-1.0-9B-Q4_K_M | 5.5 | qwen35 | vulkan | 480 | 33.2 | 6.1 |
| Ternary-Bonsai-2-27B-PTQ1_0 | 5.54 | qwen35 | cpu | 522 | 7.7 | 0.5 |
| THUDM_GLM-4-9B-0414-Q4_K_M | 5.74 | glm4 | cpu | 470 | 36.9 | 6.0 |
| THUDM_GLM-4-9B-0414-Q4_K_M | 5.74 | glm4 | vulkan | 470 | 3.9 | 3.7 |
| DeepSeek-V2-Lite-Chat.Q2_K | 5.99 | deepseek2 | cpu | 508 | 45.5 | 19.3 |
| DeepSeek-V2-Lite-Chat.Q2_K | 5.99 | deepseek2 | vulkan | 508 | 10.2 | 8.5 |
| kimi-vl-a3b-thinking-Q2_K | 6.13 | deepseek2 | cpu | 506 | 49.9 | 17.1 |
| gemma-4-12b-it-Q4_K_M | 6.63 | gemma4 | vulkan | 499 | 4.4 | 4.1 |
| Qwen1.5-MoE-A2.7B-Chat.Q4_K_M | 8.84 | qwen2moe | cpu | 493 | 102.2 | 15.6 |
| Qwen1.5-MoE-A2.7B-Chat.Q4_K_M | 8.84 | qwen2moe | vulkan | 493 | 90.7 | 12.6 |
| GLM-4.7-Flash-Q2_K | 10.57 | deepseek2 | cpu | 470 | 31.7 | 9.3 |
| granite-4.0-h-small-Q2_K | 10.97 | granitehybrid | cpu | 493 | 13.9 | 8.2 |
| gpt-oss-20b-MXFP4 | 11.28 | gpt-oss | cpu | 514 | 7.0 | 7.2 |
| gpt-oss-20b-MXFP4 | 11.28 | gpt-oss | vulkan | 514 | 8.7 | 8.7 |
| Mistral-Small-3.2-24B-Instruct-2506-Q4_K_S | 12.62 | llama | cpu | 1032 | 13.6 | 2.7 |
| Mistral-Small-3.2-24B-Instruct-2506-Q4_K_S | 12.62 | llama | vulkan | 1032 | 2.7 | 2.7 |
| Muse-Glimmer-30B-UD-Q4_K_XL | 14.79 | muse-glimmer | cpu | 503 | 2.3 | 2.3 |
| Muse-Glimmer-30B-UD-Q4_K_XL | 14.79 | muse-glimmer | vulkan | 503 | 2.4 | 2.2 |
| Trinity-Mini-Q4_K_M | 14.84 | afmoe | cpu | 443 | 15.0 | 14.7 |
| Trinity-Mini-Q4_K_M | 14.84 | afmoe | vulkan | 443 | 10.4 | 10.8 |
| DeepSeek-V2-Lite-Chat.Q8_0 | 15.56 | deepseek2 | cpu | 508 | 45.3 | 11.7 |
| Qwen3-Coder-30B-A3B-Instruct-Q4_K_M | 17.28 | qwen3moe | vulkan | 482 | 11.0 | 10.8 |
| Phi-3.5-MoE-instruct-Q3_K_M | 18.66 | phimoe | cpu | 587 | 18.2 | 9.4 |
| Phi-3.5-MoE-instruct-Q3_K_M | 18.66 | phimoe | vulkan | 587 | 19.2 | 5.7 |
| EXAONE-4.5-33B-Q4_K_M | 18.67 | exaone4 | cpu | 481 | 11.6 | 1.9 |
| Nous-Hermes-2-Mixtral-8x7B-DPO.i1-Q4_K_S | 24.91 | llama | cpu | 582 | 11.9 | 4.8 |
| Nous-Hermes-2-Mixtral-8x7B-DPO.i1-Q4_K_S | 24.91 | llama | vulkan | 582 | 4.2 | 4.1 |
| Qwen3.6-35B-A3B-UD-Q6_K | 27.3 | qwen35moe | cpu | 480 | 61.3 | 3.0 |
| Qwen3.6-35B-A3B-UD-Q6_K | 27.3 | qwen35moe | vulkan | 480 | 3.1 | 3.0 |
| tencent.Hunyuan-A13B-Instruct.Q3_K_S | 32.4 | hunyuan-moe | cpu | 476 | 10.2 | 6.5 |
| tencent.Hunyuan-A13B-Instruct.Q3_K_S | 32.4 | hunyuan-moe | vulkan | 476 | 2.8 | 2.8 |
| cerebras_GLM-4.5-Air-REAP-82B-A12B-Q2_K | 33.28 | glm4moe | cpu | 470 | 7.4 | 4.5 |
| GLM-4.5-Air-Q2_K | 42.21 | glm4moe | cpu | 470 | 4.1 | 4.9 |

> The Vulkan row for Qwen3-8B, the decode and prefill rows for Qwen2.5-0.5B and the other family-level Vulkan rows that the 2026-10-11 recheck covers are kept above. Vulkan has no llama.cpp reference in this file, so no Vulkan ratio exists.

## Text generation, Vulkan iGPU, SmolLM2-1.7B-Instruct Q4_K_M (2026-08 / 2026-09-28)

Vulkan has no llama.cpp reference in this file, so these rows are C#-only.

| Scenario | Prompt | Vulkan t/s | Performance Check |
|---|---|---:|---|
| prefill | 43 tok | 75.4 | 2026-08 |
| prefill | 267 tok | 53.5 | 2026-08 |
| prefill | 3.2k tok (default, SnapKV on) | 45.9 | 2026-08 |
| decode | short ctx | 24.0 | 2026-08 |
| decode | @3.2k ctx | 6.4 | 2026-08 |

Context scaling (2026-08, flash attention and SnapKV fix): Vulkan prefill ~83 t/s at 43 tok, ~84 at 267, ~78 at 773, ~66 at 1,621, 45.9 at 3,218 (default). CPU prefill over the same range: ~50, ~49, ~49, ~42.

Vulkan KV dtype (3,239 tok, SnapKV off, 2026-08): fp32 prefill 53.2 t/s, decode 6.0 t/s; bf16 prefill 52.0 (noise), decode 9.4 t/s (+57%, perplexity +0.023%); q8_0 decode ~7.6 t/s (perplexity +0.143%). A bf16 default is blocked pending `--tq` and SnapKV compatibility.

Vulkan Q4_K matvec bandwidth (`VulkanBackend.MatMul`, 50 matvecs per submit, best of 5, 2026-09-28). The DDR4 peak is about 51.2 GB/s (dual-channel). Q4_K is at ~83–86% of that peak, shape-independent:

| Shape | Q4_K (GB/s) | Q6_K (GB/s) |
|---|---:|---:|
| 2048×2048 | 42.6 | 36.0 |
| 8192×2048 | 44.2 | 37.3 |
| 2048×8192 | 42.5 | 37.9 |
| 4096×4096 | 43.2 | 37.9 |

Q6_K (~72% of peak) is the remaining matvec gap. iGPU numbers (rule 13).

## Newly-supported architectures (2026-09-13, first persistent timings)

First real timings for these architectures. C#-only; no llama.cpp ratio was taken unless noted. The 2026-10-08 sweep row is the newest where present.

| Model | Scenario | Backend | C# t/s | Performance Check | Notes |
|---|---|---|---:|---|---|
| Phi-3-mini-4k Q4_K_M (`phi3`) | prefill, 569 tok (matched) | CPU | 35.4 | 2026-10-08 | Newest. Was 40.1 on 2026-09-13 (−12%, not bisected). llama.cpp 68.5 (0.52x) |
| Phi-3-mini-4k Q4_K_M | decode, 24 tok gen | CPU | 11.0 | 2026-10-08 | Was 9.8 on 2026-09-13 |
| Phi-3-mini-4k Q4_K_M | prefill / decode | Vulkan iGPU | 10.3 / 8.3 | 2026-09-10 | - |
| phi-2 Q4_K_M (`phi2`) | prefill, 572 tok | CPU | 36.0 | 2026-09-13 | - |
| phi-2 Q4_K_M | decode, 24 tok gen | CPU | 9.4 | 2026-09-13 | - |
| Falcon3-3B-Instruct Q4_K_M (`llama`) | prefill, 500 tok | CPU | 102.4 | 2026-09-13 | Superseded by 2026-10-08 sweep: prefill 153.3 (llama.cpp 123.5, 1.24x) |
| Falcon3-3B-Instruct Q4_K_M | decode | CPU | 14.8 | 2026-09-13 | Superseded by 2026-10-08: 17.1 (llama.cpp 23.3, 0.73x) |
| StarCoder2-3B Q4_K_M (`starcoder2`) | code completion, 6 tok prompt, 40 tok gen (raw completion, identity chat template) | CPU | 12.2 | 2026-09-13 | Verified against llama.cpp greedy text 2026-10-29 |
| GPT-2 124M Q8_0 (`gpt2`) | raw completion, 4 tok prompt, 24 tok gen | CPU | 44.5 prefill / 62.9 decode | 2026-09-13 | Verified 2026-10-29. Newer 532-tok recheck (2026-10-11): prefill 896.2, decode 116.3 (in the C#-only table) |
| StableLM-Zephyr-3B Q4_K_M (`stablelm`) | prefill, 516 tok | CPU | 86.9 | 2026-09-13 | Superseded by ratio row: 108.4 prefill (2026-09-13) |
| StableLM-Zephyr-3B Q4_K_M | decode | CPU | 13.2 | 2026-09-13 | Ratio row: 15.8 decode (2026-09-13) |
| ERNIE-4.5-0.3B Q8_0 (`ernie4_5`) | prefill, 525 tok | CPU | 390.1 | 2026-09-13 | Ratio row 400.8 (2026-09-13); 2026-10-08 sweep decode 57.0 |
| ERNIE-4.5-0.3B Q8_0 | decode | CPU | 63.7 | 2026-09-13 | - |
| CodeShell-7B Q4_K_M (`codeshell`) | code completion, 8 tok prompt, 24 tok gen | CPU | 6.4 prefill / 4.2 decode | 2026-09-13 | Raw completion |
| Pythia-160M Q8_0 (`gptneox`, reputable conversion) | prefill, 503 tok | CPU | 1103.7 | 2026-09-13 | Decode 107.0 (24 tok). The earlier gibberish row was a bad third-party checkpoint, not a code defect (see LOG) |
| SmolLM3-3B Q4_K_M (`smollm3`) | prefill, 705 tok | CPU | 77.3 | 2026-09-13 | Decode 10.6 (24 tok, all thinking tokens). Sweep 2026-10-11 gives 110.7 prefill / 15.5 decode for the 1.78 GB file |
| Apertus-8B-Instruct Q4_K_M | prefill, 536 tok | CPU | 32.4 | 2026-09-13 | Ratio row 43.5 (2026-09-13, after the 2026-10-11 recheck) |
| Apertus-8B-Instruct Q4_K_M | decode | CPU | 4.7 | 2026-09-13 | Ratio row 7.3 |
| XVERSE-7B-Chat Q4_K_M (`xverse`) | prefill, 836 tok | CPU | 44.9 | 2026-09-13 | Verified 2026-10-29 (greedy text) |
| XVERSE-7B-Chat Q4_K_M | decode | CPU | 5.6 | 2026-09-13 | - |
| Orion-14B-Chat Q4_K_M | prefill, 577 tok | CPU | 23.6 | 2026-09-13 | - |
| Orion-14B-Chat Q4_K_M | decode | CPU | 3.1 | 2026-09-13 | - |
| c4ai-command-r7b-12-2024 Q4_K_M (`cohere2`) | prefill, 500 tok | CPU | 44.3 | 2026-10-11 | Newest (was 5.9 on 2026-09-13). Decode 6.1 (2026-10-11). No llama.cpp reference in this file for this model |
| Hunyuan-0.5B-Instruct Q8_0 (`hunyuan-dense`) | prefill, 474 tok | CPU | 229.2 | 2026-09-13 | Decode 46.5 (2026-09-13); sweep 2026-10-08: decode 53.4 |
| Maincoder-1B Q4_K_M (`maincoder`) | code completion, 4 tok prompt, 40 tok gen | CPU | 33.4 | 2026-09-13 | Prefill 30.1 (4 tok) |
| Phi-3.5-MoE Q3_K_M (`phimoe`) | prefill / decode | CPU | 18.2 / 9.4 | 2026-10-11 | Newest measurement. Decode was 5.7 right after the fix (2026-09-26). Gibberish was from missing RMSNorm bias, output bias, top-k renormalization and LongRoPE (dba59ad). Greedy parity with llama-server (`PhiMoeGreedyParityTests`) |
| Jais (`jais`, v1) | - | - | Rejected by design: only `jais2` is claimed. The `jais2` checkpoint is gated (401) | 2026-09-13 | Not measured |

## Large alpha MoE ports, correctness-only re-check (2026-11-07)

Not admitted (CLAUDE.md rule 14) — no STATUS.md/README row. All six archs touched during the 2026-11-07 porting session, re-tested fresh this pass. **No throughput numbers**: the real-weight tests use a 5-6 token smoke prompt (`"The capital of France is"`), far too short for a trustworthy t/s figure per the CLAUDE.md perf-pass rule (realistic input, several samples) — reporting one from this prompt would be noise dressed as data. Wall-clock below is whole-process time (model open + construct + forward), not isolated prefill time.

| Arch (GGUF id) | Checkpoint | Test | Result | Performance Check | Notes |
|---|---|---|---|---|---|
| `qwen4exp` (Qwen 3.8 Flash Next) | Qwen3.8-Flash-Next UD-IQ1_S, 72.5 GB (paged) | `Qwen4ExpRealWeightsSmokeTests` (2 tests) | **PASS** | 2026-11-07 | Prefill + 1 decode step, finite logits. Whole process 25.7s |
| `glm5next` / alias `glm5-next` (GLM-5.3-Flash) | GLM-5.3-Flash UD-IQ1_S, 87 GB (paged) | `Glm5NextRealWeightsSmokeTests` (2 tests) | **PASS** | 2026-11-07 | Arch-string alias resolves; prefill produces finite logits. Whole process 29.9s |
| `deepseek4` (DeepSeek-V4 Flash) | DeepSeek-V4-Flash UD-IQ1_S, 86 GB (paged) | `DeepSeek4RealWeightsSmokeTests` (1 test) | **PASS (refusal by design)** | 2026-11-07 | Checkpoint declares YaRN rope scaling (factor 16), which this alpha port correctly refuses at construction (`NotSupportedException`) instead of running wrong attention. Prefill/decode never reached — this is the intended, safe outcome, not a gap |
| `deepseek41` (DeepSeek-V4.1 Flash) | shard 1 only (~335 GB full; only the ~5.6 MB metadata/tensor-info shard fits), renamed off the `-00001-of-00005` pattern | `DeepSeek41RealWeightsSmokeTests` (1 test, new) | **PASS (refusal by design)** | 2026-11-07 | Real checkpoint declares `rope.scaling.factor=16` and `compress_ratios` starting `[0,0,2,2,...]`; constructor correctly refuses on the compress-ratio check (checked before YaRN) before touching any tensor data — metadata-only check, same shape as deepseek4's finding |
| `deepseek32` (DeepSeek-V3.2) | none on this machine | — | **untestable** | 2026-11-07 | No cheap shard exists for this one — even shard 1 alone is ~45 GB regardless of quant (metadata isn't split out into its own tiny shard the way deepseek41/glm-dsa's GGUFs are); not downloaded |
| `glmdsa` (GLM-5.2) | shard 1 only (~9 MB of a much larger checkpoint), renamed off the `-00001-of-00006` pattern | `GlmDsaRealWeightsSmokeTests` (1 test, new) | **PASS (metadata-only)** | 2026-11-07 | Construction needs full tensor data (no metadata-only guard exists in this arch's ctor, unlike deepseek4/41), so only hyperparameter parsing was checked, not construction. Every value in `GlmDsaAlpha.cs`'s own spec comment matched the real checkpoint exactly (n_head=64, q_lora_rank=2048, kv_lora_rank=512, indexer_n_head=32, indexer_head_size=128, indexer_top_k=2048) |

Found and fixed in this pass: `ModelLocator.FindDeep` only searched one folder level under each root, but these three checkpoints live two levels down (`_models/<name>/<quant>/*.gguf`); all three RealWeights tests were silently skipping before the fix (commit `5d472a3d`).

## DeepSeek family (`deepseek2`)

| Model | Scenario | Backend | C# t/s | C++ t/s | Ratio | Performance Check | Notes |
|---|---|---|---:|---:|---:|---|---|
| DeepSeek-V2-Lite-Chat Q8_0 | prefill, 502 tok | CPU | 45.3 (2026-10-11) | 56.90 (2026-09-11) | 0.80x (cross-date) | 2026-10-11 | Same-pass pair: 28.0 vs 56.90 = 0.49x (2026-09-11) |
| DeepSeek-V2-Lite-Chat Q8_0 | decode, 502 tok | CPU | 11.7 (2026-10-11) | 14.83 (2026-09-11) | 0.79x (cross-date) | 2026-10-11 | Same-pass pair: 13.6 vs 14.83 = 0.92x (2026-09-11) |
| DeepSeek-V2-Lite-Chat Q8_0 | any | Vulkan iGPU | no number | - | - | 2026-09-12 | Crash: `Missing tensor: blk.0.attn_k.weight`. MLA uses split projection tensors the Vulkan upload path does not read |
| DeepSeek-V2-Lite-Chat Q2_K | prefill 12-195 tok / decode | CPU | 8.7-10.9 / 11.6-12.0 | - | - | 2026-09-26 | Output correct (16/16 greedy parity with llama-server, `DeepSeek2GreedyParityTests`) |
| DeepSeek-V2-Lite-Chat Q8_0 | prefill / decode | CPU | 10.2 / 4.6 | - | - | 2026-09-26 | Q8_0 read from the slow drive; superseded for speed by the rows above |
| DeepSeek-V2-Lite-Chat Q2_K | decode, 502 tok (sweep) | CPU | 25.9 | - | - | 2026-10-08 | Coherent. Diverges from llama-server at token 9 (bugstofix #24) |

> Correctness: earlier DeepSeek rows measured speed of wrong output. The 2026-09-26 fixes made the output correct (see the LOG).

## gpt-oss-20B (`gpt-oss`, MXFP4)

| Model | Scenario | Backend | C# t/s | Performance Check | Notes |
|---|---|---|---:|---|---|
| gpt-oss-20B MXFP4 | prefill, 514 tok | CPU | 7.0 | 2026-10-11 | Recheck. Was 12.6 for a 72-tok prompt on 2026-09-11 |
| gpt-oss-20B MXFP4 | decode, 514 tok prompt | CPU | 7.2 | 2026-10-11 | Recheck |
| gpt-oss-20B MXFP4 | prefill / decode | Vulkan iGPU (full) | 8.7 / 8.7 | 2026-10-11 | Recheck. Hybrid `-g 8` (2026-09-13): 3.7 / 6.3 |

> Output matches llama.cpp's top-10 ranking; one 0.002-nat near-tie flips (2026-10-29).

## EXAONE-4.5-33B (`exaone4`, Q4_K_M)

| Model | Scenario | Backend | C# t/s | Performance Check | Notes |
|---|---|---|---:|---|---|
| EXAONE-4.5-33B | prefill, 196 tok, batched | CPU (`-g 0`) | 10.3 | 2026-09-26 | Batched prefill works (post-norm-only layers). 16/16 greedy tokens identical to llama-server |
| EXAONE-4.5-33B | decode, 16 tok gen | CPU | 1.5-1.9 | 2026-09-26 | Both short and 196-token prompts |
| EXAONE-4.5-33B | prefill, 196 tok | Vulkan hybrid (`-g 16`) | 1.8 | 2026-09-26 | Hybrid has no batched prefill |
| EXAONE-4.5-33B | decode, 12-16 tok gen | Vulkan hybrid | 1.8-1.9 | 2026-09-26 | Level with CPU on this shared-memory iGPU (rule 13) |
| EXAONE-4.5-33B + mmproj q8_0 | image (324 soft tok) + 27 tok gen | CPU | 1.9 / 1.8 | 2026-09-26 | `--image` works after the marker fix |
| EXAONE-4.5-33B | prefill / decode | CPU (`-g 0`, forced, earliest) | 1.6 / 1.7 | 2026-09-12 | Superseded by the 2026-09-26 rows |
| EXAONE-4.5-33B | sweep decode | CPU | 2.0 | 2026-10-08 | Coherent |

## Speculative decoding (CPU)

Speculation is a loss on this hardware (verifying k tokens costs about k× compute; Q4_K dot is ~87% compute-bound). VNNI would be the prerequisite.

| Target | Draft | Scenario | C# t/s | Plain baseline | Acceptance | Performance Check |
|---|---|---|---:|---:|---:|---|
| Qwen3-8B Q4_K_M | Qwen3-0.6B Q8_0 | decode, draft-n 4 | 4.3 (−37%) | - | 62% | 2026-08 |
| Qwen3-4B Q4_K_M | DSpark block-7 | decode, DSpark | 2.5 (−75%) | 10.0 | 23% (14/60) | 2026-09-10 |
| Qwen3.8-27B UD-Q3_K_XL | self MTP, depth 1, after shared-decode IQ kernels | decode, 96 tok | 2.7 (+12% vs plain 2.4) | 2.4 | 76% | 2026-10-08 |
| Qwen3.8-27B UD-Q3_K_XL | MTP, depth 1 (earlier) | decode, 64 tok | 1.9 (−21%) | 2.4 | 50% | 2026-10-08 |
| Qwen3.6-27B UD-Q3_K_XL (superseded 2026-11-03) | self MTP, forced | decode, 96 tok | 2.0-2.1 (0.77-0.81x) | 2.6 | 45-76% | 2026-10-08 |
| Qwen3.6-27B and Qwen3.8-27B UD-Q3_K_XL | self MTP after the Q3_K shared-decode kernels (PRs 217, 218) | decode, 96-token request, 3 runs each | Qwen3.6 2.7-2.8 (1.04-1.08x); Qwen3.8 2.8 (1.17x) | 2.6 / 2.4 | 83% / 88% | 2026-11-03 |
| Qwen3.8-27B-UD-Q3_K_XL | self MTP (`--spec-type mtp`), 3 rounds on the same build | decode, 96 tok, `--temp 0 --no-thinking`, `-g 0` | 2.9 / 3.0 / 2.8 (median 2.9) | 2.3 / 2.3 / 2.2 (median 2.3) | not recorded | 2026-11-06 |

> The MTP rows are the only speculative wins measured. Since 2026-11-03 (Q3_K kernels) Qwen3.8 is +17% and Qwen3.6 is +4-8% at depth 1; re-measured 2026-11-06 on the same build, Qwen3.8 is +26% (median 2.9 vs 2.3 t/s, three rounds each); the Qwen3.6 2026-10-08 row above is superseded. Qwen3.6-27B, the same architecture with a different quant mix, is 0.77–0.81x of plain. `--spec-type auto` does not engage MTP on CPU; users who pass `--spec-type mtp` should compare with plain on their own file.

## Historical CUDA (†) and other prior-hardware rows

These come from the prior Zen 4 + RTX 4070 Ti 12 GB machine (commit `0c171ed`, 2026-06-16 or 2026-06-22). No newer measurement exists for these keys. Treat them as indicative; the codebase has changed since.

| Model | Scenario | Backend | Value (t/s) | llama.cpp (t/s) | Ratio | Performance Check |
|---|---|---|---:|---:|---:|---|
| Qwen3-Coder 30B-A3B Q4_K_M | prefill | CUDA (†) | 102.6 | - | - | 2026-06-16 |
| Qwen3-Coder 30B-A3B Q4_K_M | decode | CUDA (†) | 28.0 | - | - | 2026-06-16 |
| Qwen3-Coder 30B-A3B Q4_K_M (original) | decode | CPU | 22.4 | - | - | 2026-06-16 |
| Qwen3-Coder 30B-A3B Q4_K_M (`--tq`) | decode | CPU | 22.6 | - | - | 2026-06-16 |
| Qwen3.6-35B-A3B Q4_K_M | prefill | CUDA (†) | 475.4 | - | - | 2026-06-16 |
| Qwen3.6-35B-A3B Q4_K_M | decode | CUDA (†) | 24.5 | - | - | 2026-06-16 |
| Qwen3.6-35B-A3B Q4_K_M | decode | Vulkan (†) | 22.8 | - | - | 2026-06-16 |
| Qwen3.6-35B-A3B Q4_K_M (original) | decode | CPU | 9.3 | - | - | 2026-06-16 |
| Qwen3.6-35B-A3B-MTP Q4_K_M | prefill | CUDA (†) | 480.2 | - | - | 2026-06-16 |
| Qwen3.6-35B-A3B-MTP Q4_K_M | decode (`--no-thinking`) | CUDA (†) | 33.3 | ~41 (est.) | ~0.81x | 2026-06-16 |
| Qwen3.6-27B-MTP Q4_K_M | prefill | CUDA (†) | 22.0 | - | - | 2026-06-16 |
| Qwen3.6-27B-MTP Q4_K_M | decode (`--no-thinking`) | CUDA (†) | 12.3 | - | - | 2026-06-16 |
| Qwen3.6-27B-MTP Q4_K_M (original) | decode | CPU | 3.6 | - | - | 2026-06-16 |
| Carnice 35B-A3B-MTP (APEX) | prefill (`--no-thinking`) | CUDA (†) | 522.0 | - | - | 2026-06-16 |
| Carnice 35B-A3B-MTP (APEX) | decode | CUDA (†) | 26.5 | - | - | 2026-06-16 |
| Gemma4 E4B QAT Q4_0 | prefill | CUDA (†) | 3666 | - | - | 2026-06-16 |
| Gemma4 E4B QAT Q4_0 | decode | CUDA (†) | 100.4 | - | - | 2026-06-16 |
| Gemma4 E4B QAT Q4_0 | prefill | Vulkan (†) | 35 | - | - | 2026-06-22 |
| Gemma4 E4B QAT Q4_0 | decode | Vulkan (†) | 39.5 | - | - | 2026-06-22 |
| Gemma4 12B QAT Q4_0 | prefill | CUDA (†) | 1714 | - | - | 2026-06-16 |
| Gemma4 12B QAT Q4_0 | decode | CUDA (†) | 54.1 | 57 | 0.95x | 2026-06-16 |
| Gemma4 12B QAT Q4_0 | prefill | Vulkan (†) | 17.0 | - | - | 2026-06-16 |
| Gemma4 12B QAT Q4_0 | decode | Vulkan (†) | 19.1 | - | - | 2026-06-22 |
| Gemma4 12B QAT Q4_0 | decode | CPU (†, Zen 4) | 5.1 | - | - | 2026-06-16 |
| Llama-4 Scout 17B-16E Q4_K_M | decode (smoke) | CPU (†) | 4.3 | - | - | 2026-06-16 |
| Llama-4 Scout 17B-16E Q4_K_M | decode (smoke) | CUDA (†) | 2.6 | - | - | 2026-06-16 |

> Gemma4 12B CUDA decode (†) is within ~6% of llama.cpp, the best verified parity in the file, on prior hardware.

## Text encoders (embeddings / rerank), vs ONNX Runtime

32 passages of ~86 tokens (2,752 tokens), CPU F32, median of 5 timed runs after warm-up (`EncoderBenchmarkTests`, `STINGRAY_RUN_HEAVY_TESTS=1`). Batch-32 values are "ours / ORT" as measured in two runs; ORT timings swing between runs, so both runs are shown.

| Model | Batch 32: ours | Batch 32: ORT | One-by-one: ours | One-by-one: ORT | Performance Check |
|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | 193 / 201 ms (166 / 160 emb/s) | 363 / 172 ms | 250 / 224 ms | 455 / 237 ms | 2026-09-25 |
| bge-small-en-v1.5 | 285 / 272 ms (112 / 118 emb/s) | 543 / 367 ms | 456 / 438 ms | 1127 / 448 ms | 2026-09-25 |
| bge-large-en-v1.5 | 3062 / 3150 ms (10.5 / 10.2 emb/s) | 3522 / 3373 ms | 4058 / 4022 ms | 4536 / 4859 ms | 2026-09-25 |
| nomic-embed-text-v1.5 | 1236 / 1236 ms (25.9 emb/s) | 1530 / 1491 ms | 1535 / 1532 ms | 2385 / 1964 ms | 2026-09-25 |

### Same encoders vs llama.cpp (C++/ggml), F32 and Q8_0

Same 32 passages (2,752 tokens). `llama-server --embedding -np 32`, whole batch in one ubatch, CPU, median of 5 after warm-up, `STINGRAY_BENCH_LLAMA_THREADS=16`. llama.cpp timings include localhost HTTP and JSON.

| Model | Batch 32: ours F32 | Batch 32: llama.cpp F32 | Batch 32: llama.cpp Q8_0 | One-by-one: ours F32 | One-by-one: llama.cpp F32 | One-by-one: llama.cpp Q8_0 | Performance Check |
|---|---|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | **170 ms** | 216 ms | 236 ms | **212 ms** | 259 ms | 235 ms | 2026-09-25 |
| bge-small-en-v1.5 | **263 ms** | 422 ms | 422 ms | **405 ms** | 443 ms | 408 ms | 2026-09-25 |
| bge-large-en-v1.5 | **2924 ms** | 3906 ms | 4670 ms | **3201 ms** | 4261 ms | 3597 ms | 2026-09-25 |
| nomic-embed-text-v1.5 | **1220 ms** | 1378 ms | 1356 ms | 1370 ms | 1393 ms | **1364 ms** | 2026-09-25 |

Ours (F32) is ahead of llama.cpp's best (F32 or Q8_0) batched on all four: 1.1x (nomic) to 1.6x (bge-small). Quantisation does not buy encoder speed on this CPU (Ryzen 5700G, AVX2, no VNNI): llama.cpp's Q8_0 is level with its own F32, so no int8 encoder path was added.

## Embeddings (CPU), GGUF path and Qwen3-Embedding

Comparative benchmark against `llama-bench -embd 1 -n 0 -t 6 -r 3` (single prompts) and `-t 16` where noted.

| Model | Scenario | C# t/s | llama.cpp t/s | Ratio | Performance Check |
|---|---|---:|---:|---:|---|
| Qwen3-Embedding-0.6B Q8_0 | pp64 (74 tok) | 47.0 | 299.78 | 0.16x | 2026-09-17 |
| Qwen3-Embedding-0.6B Q8_0 | pp128 (128 tok) | 46.1 | 308.89 | 0.15x | 2026-09-17 |
| Qwen3-Embedding-0.6B Q8_0 | pp512 (524 tok) | 43.4 | 304.09 | 0.14x | 2026-09-17 |
| Qwen3-Embedding-0.6B Q8_0 | batch4 (4×128 tok) | 45.5 | 304.09 (pp512) | 0.15x | 2026-09-17 |
| Qwen3-Embedding-0.6B Q8_0 | batch8 (8×64 tok) | 46.6 | 299.78 (pp64) | 0.16x | 2026-09-17 |
| Qwen3-Embedding-0.6B Q8_0 | embed, 605 tok, after the fused Q8_0 4-input dot (3 runs, 3.39-3.57 s) | **~175** | 349.64 (`-t 16`; 229.13 at 6 thr) | **~0.50x** | 2026-10-01 |
| Qwen3-0.6B Q8_0 | prefill, 623 tok, after the fused Q8_0 dot (158-184) | 174 | 443.15 (pp512, `-t 16`) | **0.39x** | 2026-10-01 |
| Qwen2.5-0.5B-Instruct Q4_K_M | prefill, 633 tok, int8 prefill tier off (default since 7791e3c9) | 142 (141-143) | 524.22 (pp512, `-t 16`) | 0.27x | 2026-10-01 |
| Qwen2.5-0.5B-Instruct Q4_K_M | prefill, 633 tok, int8 prefill tier on | 284 (260-308) | 524.22 (pp512, `-t 16`) | 0.54x | 2026-10-01 |

> The Qwen3-Embedding pp rows and the 2026-10-01 embed row are different prompts, so only the newest row per prompt is the current number. The int8 on/off choice is a numerics-policy decision (perf-sweep-plan 5.2), not a perf fix.

### ONNX encoders (`stingray embed -m <file>.onnx`), real WordPiece tokenizer

| Model | Scenario | Backend | C# result | Output dim | Performance Check |
|---|---|---|---|---|---|
| all-MiniLM-L6-v2 (quantized ONNX) | 1 text, 12 real tokens, mean pooling | CPU | 26 ms (best of 3) | 384 | 2026-09-11 |
| bge-small-en-v1.5 (quantized ONNX) | 1 text, 12 real tokens, mean pooling | CPU | 28 ms (best of 3) | 384 | 2026-09-11 |
| bge-base-en-v1.5 (quantized ONNX) | 1 text, 12 real tokens, mean pooling | CPU | 30 ms (best of 3) | 768 | 2026-09-11 |
| bge-large-en-v1.5 (quantized ONNX) | 1 text, 12 real tokens, mean pooling | CPU | 41 ms (best of 3) | 1024 | 2026-09-11 |

> No C++ reference was run for these four rows. Each reports its real native output dimension, which shows the forward pass is real.

## Vision-Language Model Text Backbones (CPU and Vulkan iGPU, text-only)

These run the LLM backbone under a VLM in plain text mode (no `--image`). They are LLM throughput rows, not vision-encode rows. Ratio pairs are same-pass, as in the LLM ratio table.

| Model | Scenario | Backend | C# t/s | llama.cpp t/s | Ratio | Performance Check |
|---|---|---|---:|---:|---:|---|
| InternVL3-2B Q4_K_M (Qwen2-1.5B backbone) | prefill, 503 tok | CPU | 131.8 | 168.90 | 0.78x | 2026-09-10 |
| InternVL3-2B Q4_K_M (Qwen2-1.5B backbone) | decode, 503 tok prompt, 24 tok gen | CPU | 18.3 | 37.94 | 0.48x | 2026-09-10 |
| InternVL3-2B Q4_K_M (Qwen2-1.5B backbone) | prefill, 503 tok | Vulkan iGPU | 23.2 | - | - | 2026-09-10 (newer: 281.8 on 2026-10-11, C#-only table) |
| InternVL3-2B Q4_K_M (Qwen2-1.5B backbone) | decode, 503 tok prompt | Vulkan iGPU | 21.8 | - | - | 2026-09-10 |
| Granite-4.0-3B-Vision Q4_K_M (Granite backbone) | prefill, 494 tok | CPU | 58.4 | 69.82 | 0.84x | 2026-09-10 |
| Granite-4.0-3B-Vision Q4_K_M (Granite backbone) | decode, 494 tok prompt, 24 tok gen | CPU | 8.7 | 18.09 | 0.48x | 2026-09-10 |
| Granite-4.0-3B-Vision Q4_K_M (Granite backbone), Vulkan fixed | prefill, 494 tok | Vulkan iGPU | 44.1 | - | - | 2026-09-11 |
| Granite-4.0-3B-Vision Q4_K_M (Granite backbone), Vulkan fixed | decode, 494 tok prompt | Vulkan iGPU | 10.8 | - | - | 2026-09-11 |
| Granite-Vision-3.2-2B Q3_K_S (Granite backbone) | prefill, 620 tok, after `DotQ3K_Q8K_Avx2` rewrite | CPU | 42.2 | 81.95 (pp512) | 0.51x | 2026-10-02 |
| Granite-Vision-3.2-2B Q3_K_S (Granite backbone) | decode, 620 tok prompt, after the same rewrite | CPU | 25.9 | 34.95 (tg64) | 0.74x | 2026-10-02 |
| Granite-Vision-3.2-2B Q3_K_S (Granite backbone), Vulkan fixed | prefill, 573 tok | Vulkan iGPU | 4.1 | - | - | 2026-09-11 |
| Granite-Vision-3.2-2B Q3_K_S (Granite backbone), Vulkan fixed | decode, 573 tok prompt | Vulkan iGPU | 4.1 | - | - | 2026-09-11 |
| dots.ocr Q8_0 (Qwen2-1.5B backbone) | prefill, 477 tok | CPU | 96.3 | 135.37 | 0.71x | 2026-09-10 |
| dots.ocr Q8_0 (Qwen2-1.5B backbone) | decode, 477 tok prompt, 24 tok gen | CPU | 20.5 | 25.97 | 0.79x | 2026-09-10 |
| dots.ocr Q8_0 (Qwen2-1.5B backbone) | prefill, 477 tok | Vulkan iGPU | 13.8 | - | - | 2026-09-10 |
| dots.ocr Q8_0 (Qwen2-1.5B backbone) | decode, 477 tok prompt | Vulkan iGPU | 12.9 | - | - | 2026-09-10 |

## Vision-Language Model Real Image Encoding (CPU)

These runs feed a real image through `--image`/`--mmproj` and generate an image-conditioned description. Image: `docs/diffusion-samples/ltx_test_apple_256.png` (256×256). Prompt: "Describe this image in one sentence.", `-n 30`, `--temp 0`, CPU. Timings are combined prefill+decode t/s over the image and text tokens. The C++ `llama-mtmd-cli` prints only the vision-encoder time (2488-3226 ms for InternVL3-2B, 2026-09-11), a different slice of the pipeline, so no ratio is given.

| Model | Scenario | Tokens | C# t/s | Output check | Performance Check |
|---|---|---|---:|---|---|
| InternVL3-2B Q4_K_M + mmproj-q8_0 | prefill | 294 (256 image + 38 text) | 28.1 (28.4/28.1/27.8) | image-grounded (correct colours and shapes) | 2026-09-11 |
| InternVL3-2B Q4_K_M + mmproj-q8_0 | decode | 23-30 tok gen | 27.4 (27.2/27.3/27.6) | as above | 2026-09-11 |
| Granite-4.0-3B-Vision Q4_K_M + mmproj-f16 | prefill | 614 (576 image + 38 text) | 13.5 (13.6/13.6/13.5) | **not image-grounded** (degenerate) | 2026-09-11 |
| Granite-4.0-3B-Vision Q4_K_M + mmproj-f16 | decode | 10 tok gen | 12.1 (10.0/13.2/13.1) | not image-grounded | 2026-09-11 |
| Granite-Vision-3.2-2B Q3_K_S + mmproj-f16 | prefill | 785 (729 image + 56 text) | 18.3 (19.2/17.6/18.1) | **not image-grounded** (degenerate) | 2026-09-11 |
| Granite-Vision-3.2-2B Q3_K_S + mmproj-f16 | decode | 26 tok gen | 16.1 (17.6/15.3/15.4) | not image-grounded | 2026-09-11 |
| Gemma-4-12B-it Q4_K_M + mmproj-qat-q4_0 (`gemma4uv`, Vulkan iGPU) | prefill | 70 (49 image + 21 text) | 4.5 | coherent, plausible; not conclusive for this test image | 2026-09-12 |
| Gemma-4-12B-it Q4_K_M + mmproj-qat-q4_0 | decode | 22 tok gen | 4.4 | as above | 2026-09-12 |
| Gemma-3-4B-it Q4_K_M + mmproj-f16 | prefill | 274 (256 image + 18 text) | 11.6 (11.2/12.4/11.2) | image-grounded | 2026-09-11 |
| Gemma-3-4B-it Q4_K_M + mmproj-f16 | decode | 27 tok gen | 11.1 (12.5/10.4/10.8) | image-grounded | 2026-09-11 |
| MiniCPM-V-2.6 Q4_K_M + mmproj-f16 (resampler, Vulkan iGPU) | prefill | 92 (64 image + 28 text) | 7.5 (7.5/7.5/7.4) | image-grounded, byte-identical across 3 runs | 2026-09-12 |
| MiniCPM-V-2.6 Q4_K_M + mmproj-f16 | decode | 14 tok gen | 7.4 (7.4/7.4/7.4) | as above | 2026-09-12 |
| dots.ocr Q8_0 + mmproj-Q8_0 | prefill | 93 (81 image + 12 text) | 21.2 | 1-token degenerate output | 2026-09-11 |

> **Granite-family rows are timing only.** All Granite VLM runs produced degenerate, non-image-grounded output, identical across runs. The timing is real; correctness is not verified. Root cause: the Granite projector is a simplified single-linear approximation and is not wired to the real GGUF tensor names (see the LOG). Gemma-4-12B's output was plausible but not conclusive against the test image, which is a known weak sample.
> Harness: `stingray` CLI `--image`/`--mmproj`.

### Unsupported-architecture vision attempts with timings (garbled output, timing only)

| Checkpoint | Architecture | Prefill t/s | Decode t/s | Output |
|---|---|---:|---:|---|
| DeepSeek-OCR-2 Q4_K_M | `deepseek2-ocr` | 41.9 | 30.9 | garbled, mixed-language (4096 image + 7 text tokens) |
| PaddleOCR-VL-1.6 | `paddleocr` | 51.5 | 39.2 | four repeated newline tokens |
| Step3-VL-10B Q2_K (after tensor-name fix) | `qwen3` (admitted) | 10.0 | 9.9 | garbled (expected for Q2_K on an unverified architecture) |

> Other vision attempts (Kimi-VL-A3B, YouTu-VL-4B, Nemotron-Nano-12B-v2-VL, LLaVA-1.5-7B, MiMo-VL-7B) produced crashes or no generation, so they have no usable timing. Their details are in the LOG.

## Vision Encoder (CPU)

| Component | Scenario | Backend | C# result | Performance Check |
|---|---|---|---|---|
| VisionOps.Attention / AttentionGqa | 1024-token, 16-head ViT-L | CPU | >1.2x over scalar | 2026-08-20 |

> The scalar reference stays in `VisionOpsBenchmarkTests.cs` as a regression baseline.

## TTS / Audio Synthesis

RTF = wall-clock ÷ audio duration (lower is better; <1.0x is faster than real time). "Warm" rows time generation after a warm-up call. "Cold CLI" rows include per-call model load (`stingray tts`), so they are not comparable with warm rows. C++ = `audiocpp_cli` (`examples/audio.cpp`) unless stated.

| Pipeline | Scenario | Backend | C# wall | C# RTF | C++ wall | C++ RTF | Ratio | Performance Check |
|---|---|---|---:|---:|---:|---:|---:|---|
| Piper lessac-medium (ONNX) | text → 2.45 s audio, warm | CPU | 0.47 s | **0.19x** | - | - | - | 2026-09-09 |
| Piper lessac-medium (ONNX) | text → ~2.2-2.4 s audio, cold CLI | CPU / Vulkan | 1.08 s / 1.05 s (RTF ~0.46-0.47x both) | - | - | - | - | 2026-09-10 |
| MMS-TTS eng (VITS) | text → 3.65 s audio | CPU | 1.38 s | **0.38x** | - | - | - | 2026-09-09 |
| Kokoro-82M (Q8_0 GGUF) | text → 2.93 s audio, warm | CPU | 2.60 s | **0.89x** | - | - | - | 2026-09-09 |
| Kokoro-82M (Q8_0 GGUF) | text → 2.93 s audio, cold CLI | CPU | ~3.9 s (mean of 3) | ~1.33x | - | - | - | 2026-09-10 |
| Kokoro-82M (Q8_0 GGUF) | text → 2.93 s audio, cold CLI | Vulkan iGPU | ~3.7 s (mean of 3) | ~1.27x | - | - | - | 2026-09-10 |
| MeloTTS zh_en (ONNX) | text → 2.74 s audio, warm | CPU | 3.11 s | **1.14x** | - | - | - | 2026-09-09 |
| MeloTTS zh_en (ONNX) | text → 2.73 s audio, cold CLI | CPU / Vulkan iGPU | 4.48 s (1.64x) / 4.24 s (1.56x) | - | - | - | - | 2026-09-10 |
| QwenTTS 0.6B (Q8_0 GGUF) | text → 2.16 s audio | CPU | 6.59 s | **3.05x** | 11.38 s (4.07x) | - | **1.33x** | 2026-09-09 |
| QwenTTS 0.6B, after Horizontal Pass A audit | text → 2.16 s audio, single-shot process (includes load; not comparable with the row above) | CPU | 12.107 / 9.447 / 9.634 s | - | - | - | - | 2026-09-12 |
| CosyVoice3 (DiT + HiFT) | text → 3.00 s audio | CPU | 17.20 s | **5.73x** | 31.66 s (9.05x) | - | **1.58x** | 2026-09-09 |
| CosyVoice3 DiT, GPU residency | one 22-layer forward, t=200 frames, 5-rep mean | Vulkan iGPU | **444.3 ms/call** | - | 567.7 ms/call (CPU, same test) | - | 1.28x vs CPU | 2026-09-14 |
| CosyVoice3, reference-conditioning cache | repeated calls, same reference audio | CPU | call 1 34.36 s, call 2 29.41 s (~14% faster on cache hit) | - | - | - | - | 2026-09-12 |
| Chatterbox Turbo (Q4_K) | text → 2.52 s audio, warm | CPU | 13.72 s | **5.45x** | 16.24 s (6.77x) | - | **1.24x** | 2026-09-09 |
| Chatterbox Turbo (Q4_K) | text → ~2.2-2.3 s audio, cold CLI | CPU / Vulkan iGPU | 6.00 s (2.68x) / 6.11 s (2.63x) | - | - | - | - | 2026-09-10 |
| Parler-TTS Mini v1 | text → 2.81 s audio | CPU | 17.36 s | **6.18x** | - | - | - | 2026-09-09 |
| Parler T5 text encoder, GPU residency | one 24-layer encode, t=64 tokens, 5-rep mean | Vulkan iGPU | **137.8 ms/call** | - | 389.7 ms/call (CPU, same test) | - | 2.83x vs CPU | 2026-09-14 |
| FishSpeech S2 Pro (Q4_K) | text → 3.44 s audio | CPU | 28.46 s | **8.28x** | 36.12 s (10.95x) | - | **1.32x** | 2026-09-09 |
| F5-TTS Base (DiT) | text → 2.77 s audio, warm | CPU | 27.25 s | **9.82x** | 43.2 s (15.6x; mean of 3) | - | **1.59x** | 2026-09-11 (C++ ref) / 2026-08-28 (C#) |
| F5-TTS Base (DiT) | text → 2.77 s audio, cold CLI | CPU | 31.09 s | 11.21x | - | - | - | 2026-09-10 |
| F5-TTS Base (DiT) | text → 2.77 s audio, cold CLI | Vulkan iGPU | 59.58 s | 21.48x | - | - | - | 2026-09-10 |
| F5-TTS Base (DiT), GPU residency v2 (GPU RoPE) | one 22-layer forward, t=200, 5-rep mean | Vulkan iGPU | **642.7 ms/call** | - | 887.9 ms/call (CPU, same test) | - | 1.38x vs CPU | 2026-09-14 |
| F5-TTS Base (Paragraph) | text → 14.5 s audio | CPU | 10.20 s | **0.70x** | - | - | - | 2026-09-09 |
| CosyVoice2-0.5B | text → 8.00 s audio | CPU | 23.20 s | **2.90x** | - (no `cosyvoice2` family in audio.cpp) | - | - | 2026-09-10 |
| XTTS-v2 | text → 3.16 s audio (voice-cloning reference) | CPU | 8.758 s (mean of 3; was 10.16 s before the ResNet encoder fix) | **2.774x** (was 3.22x) | - (`xtts_inference.cpp` is source only) | - | - | 2026-09-12 |
| MOSS-TTS-Nano (100M) | text → 3.04 s audio, 40 frames | CPU | 3.46 s | **1.14x** | - | - | - | 2026-09-10 |
| PersonaPlex (7B LLM + Mimi codec) | text → 4.00 s audio, 50 frames, after Horizontal Pass A | CPU | **135.140 s** (total, includes model load) | ~32x (est.) | - | - | - | 2026-09-12 |
| PersonaPlex, zero-copy Q8_0, decode step | one `ForwardPass.Prefill` decode step (1 token, warm), 8 samples × 3 runs | CPU | **~301 ms/step** (was ~660 ms) | - | - | - | ~2.2x (this machine) | 2026-09-28 |
| Higgs Audio TTS (4B LLM + codec) | text → 2.64 s audio | CPU | 36.01 s (mean of 3) | **13.64x** | - | - | - | 2026-09-10 |
| VoxCPM2 (LLM + residual LM + CFM/DiT + AudioVAE) | text → 3 patches, 4 CFM steps (smoke params, wall time only) | CPU | 15.8 s (single run) | - | - | - | - | 2026-09-10 |
| NeuTTS-2E (speech codes only, no codec decode) | text → 32 speech codes (wall time only) | CPU | 6.42 s | - | - | - | - | 2026-09-10 |
| VibeVoice-TTS 1.5B (LLM + speech diffusion) | text → 8.00 s audio, 60 tokens, after Horizontal Pass A (single run) | CPU | 91.589 s | **~11.4x** | - | - | - | 2026-09-12 |
| VibeVoice-1.5B TTS (q8_0), after perf pass | 1-sentence script → ~4.3-4.8 s audio, 10 diffusion steps, guidance 1.5 | CPU | **24.5 / 26.4 s** (test wall incl. load; was 65-72 s) | ~5.5x | `audiocpp_cli` CPU: 19.0-19.9 s | ~1.3x behind | - | 2026-09-25 |
| OmniVoice (MaskGIT + acoustic decoder), after batching | text → 3.20 s audio, 32 MaskGIT steps, guidance 2.0, 4 seeds | CPU | **~31 s** (30.8-31.6; was 118-129 s) | ~9.7x | `audiocpp_cli` CPU: 16.2-16.9 s | ~1.9x behind | - | 2026-09-25 |
| OmniVoice (MaskGIT + acoustic decoder) | text → 3.20 s audio, 12 MaskGIT steps, single run | CPU | 43.05 s | ~13.45x | - | - | - | 2026-09-10 |
| MusicGen-small (T5 + delayed pattern + EnCodec) | text → 3.00 s audio, after the matmul and batching fixes (mean of 3) | CPU | **6.98 s** | **2.33x** | - | - | - | 2026-09-28 |
| AudioGen-medium (T5-large + delayed pattern + EnCodec) | text → 3.00 s audio, after CFG batching (mean of 3) | CPU | **19.00 s** | **6.33x** | - | - | - | 2026-09-28 |
| Stable Audio 3 Small Music (DiT + taae_v2 VAE) | text → 6.00 s audio, 8-step CFG | CPU | **123.51 s** mean (best 115.73 s) | **20.59x** | - | - | - | 2026-09-17 |
| Stable Audio 3 Small Music, GPU-resident diffusion loop | text → 4.00 s audio, 16-step CFG | Vulkan iGPU | 27.12 s | 6.78x | - | - | 5.90x vs CPU (iGPU) | 2026-09-18 |
| Stable Audio 3 Small Music, GPU-resident diffusion loop | text → 6.00 s audio, 8-step CFG | Vulkan iGPU | 20.93 s | 3.49x | - | - | - | 2026-09-18 |
| Stable Audio 3 Medium (diff-attention DiT + SAME-L VAE) | text → 4.00 s audio, 12-step CFG, seed 1234, after P.V vectorization | CPU | **44.2-44.9 s** | **11.1-11.2x** | - | - | - | 2026-10-02 |
| Stable Audio 3 Medium, default (auto) backend | text → 4.00 s audio, 12-step CFG, seed 1234 (GPU DiT 33.0 s, VAE on CPU 6.0 s) | Vulkan iGPU | **40.5 s** | **10.12x** | - | - | - | 2026-10-02 |
| ACE-Step Turbo (Qwen3 text encoder + DiT + Oobleck VAE) | text → 2.50 s audio, 8-step CFG, unaccelerated (CPU silence) | CPU | 250.0 s | 100.0x | - | - | - | 2026-09-17 |
| ACE-Step Turbo, silence internalized + GPU DiT | text → 2.50 s audio, 8-step CFG | Vulkan iGPU | **28.92 s** | **11.57x** | - | - | 8.64x vs CPU (iGPU) | 2026-09-18 |
| ACE-Step 1.5 Turbo | text → 10 s music, 8 steps | CPU (+ DiT on Vulkan iGPU) | **17.3-17.8 s** (3 runs) | - | audio.cpp CPU, q8_0: 37 s | ~2.1x faster (mixed backends) | - | 2026-09-28 |
| MiniMax-Music3 vocoder | 200 latent frames, 204,800 samples | CPU | **4.57-4.64 s** | - | - | - | - | 2026-09-28 |
| MiniMax-Music3 (full pipeline: condition encoder, RVQ depth decoder, DiT, vocoder) | 200 frames, ~8 s audio | CPU | **489.5-528.6 s** (DiT 31.73 s/step) | - | C++ `minimaxmusic.cpp`: 477.8 s (DiT 34.3 s/step) | - | **~1.02-1.11x** | 2026-09-05 |

> RTF < 1.0x = faster than real time. Piper (0.19x), MMS-TTS (0.38x) and Kokoro (0.89x) are faster than real time on CPU. Autoregressive pipelines (QwenTTS, CosyVoice3, Chatterbox, Parler, FishSpeech) are compute-bound on CPU.
> Harness: `scripts/bench-audio.ps1`, `scripts/bench-cpp.ps1 -Suite Tts`.

## TTS Streaming Latency (TTFA)

Time-to-first-audio is the latency from prompt ingestion to the first playable chunk. No C++ streaming reference was available for any family in this table (`audiocpp_cli` refuses `--mode streaming` for these families).

| Pipeline | Scenario | Backend | TTFA | Total | Performance Check |
|---|---|---|---:|---:|---|
| FishSpeech S2 Pro (stream) | 1-frame | CPU | **0.634 s** | 63.55 s | 2026-09-09 |
| Piper lessac (stream) | 16-frame | CPU | **0.776 s** | 6.47 s | 2026-09-09 |
| MMS-TTS eng (stream) | 16-frame | CPU | **0.996 s** | 16.40 s | 2026-09-09 |
| Parler-TTS Mini (stream) | 16-frame | CPU | **1.712 s** | 17.40 s | 2026-09-09 |
| QwenTTS 0.6B (stream) | 1-frame | CPU | **2.728 s** | 72.53 s | 2026-09-09 |
| MeloTTS (stream) | - | CPU | **6.540 s** | 6.55 s | 2026-09-09 |
| Kokoro-82M (stream) | 1-chunk | CPU | **6.592 s** | 6.60 s | 2026-09-09 |
| F5-TTS (stream) | - | CPU | **27.034 s** | 27.03 s | 2026-09-09 |
| Chatterbox Turbo (stream) | - | CPU | **13.669 s** | 13.67 s | 2026-09-09 |

> Harness: `scripts/bench-audio.ps1 -Suite Streaming`.

## Forced Alignment / Audio Timing

| Model | Scenario | Backend | C# wall | C# RTF | C++ wall | C++ RTF | Ratio | Performance Check |
|---|---|---|---:|---:|---:|---:|---:|---|
| Qwen3 Forced Aligner 0.6B | 14.1 s audio | CPU | 1.68 s | 0.12x | 1.71 s | 0.12x | **1.01x** | 2026-09-09 |

## ASR / Speech Recognition

| Model | Scenario | Backend | C# wall | C# RTF | C++ wall | C++ RTF | Ratio | Performance Check |
|---|---|---|---:|---:|---:|---:|---:|---|
| Whisper Tiny (39M) | 14.1 s audio | CPU | 1.36 s (mean of 3) | 0.097x | 0.630 s (mean of 3) | 0.045x | **0.46x** | 2026-09-13 |
| Whisper Base (39M) | 14.1 s audio | CPU | 0.84 s | 0.070x | 0.82 s | 0.058x | **0.83x** | 2026-09-09 |
| Whisper Small (244M) | 14.1 s audio | CPU | 2.42 s | 0.202x | 2.36 s | 0.168x | **0.83x** | 2026-09-09 |
| Whisper Medium (769M) | 14.1 s audio | CPU | 6.71 s | 0.560x | 6.93 s | 0.492x | **0.88x** | 2026-09-09 |
| Whisper Large-v3 (1.5B) | 14.1 s audio | CPU | 11.32 s | 0.943x | 12.69 s | 0.900x | **0.95x** | 2026-09-09 |
| Voxtral-Mini-4B-Realtime, after the Q8_0 audio encoder (sweep 4) | 14.1 s audio (mean of 3) | CPU | **108.861 s** | 7.74x | 24.05 s (1.71x) | - | **0.221x** | 2026-09-12 |
| Qwen3-ASR 0.6B (safetensors), prompt template fixed | 14.1 s audio (b.wav) | CPU | **5.16 s** (mean of 3) | **0.367x** | - | - | - | 2026-09-12 |
| VibeVoice-ASR (7B-class LLM) | 14.1 s audio, after Horizontal Pass A | CPU | **124.841 s** (mean of 3) | 8.872x | - | - | - | 2026-09-12 |
| VibeVoice ASR (q8_0 GGUF), LibriSpeech 3.5 s | greedy | CPU | 47-56 s (test wall incl. load) | ~13-16x | `audiocpp_cli` CPU: 11.3-12.6 s | ~3.3-4.5x | ~4x behind | 2026-09-25 |
| VibeVoice ASR (q8_0 GGUF), LibriSpeech 15 s | greedy | CPU | 142 s (test wall incl. load) | ~9.5x | `audiocpp_cli` CPU: 31.1 s | - | - | 2026-09-25 |
| VibeVoice ASR, after perf pass | LibriSpeech 3.5 s clip, greedy | CPU | **~17.5 s** compute (was 40.8 s) | ~5x | `audiocpp_cli` CPU: 12.4 s | ~1.4x behind | - | 2026-09-25 |
| Citrinet-ASR (Jasper-style conv encoder), like-for-like | ~3.5 s LibriSpeech clip, warm | CPU | **~60-70 ms** (mel 10.8 + encoder 49-59 ms, median of 20) | - | 31-39 ms (`audiocpp_cli --metrics` wall, 8 thr, 3 runs) | - | ~0.6x | 2026-10-01 |
| Nemotron 3.5 ASR Streaming 0.6B, after Horizontal Pass A | real audio clip, single run | CPU | **7.440 s** | - | - (see LOG: C++ attempt blocked, GGUF tensor-name mismatch) | - | - | 2026-09-12 |
| NVIDIA Parakeet-CTC 0.6B (q4_k), NativeAOT, after memory pass | LibriSpeech 6930-75918-0001, 14.2 s | CPU | **1.04 s** (mean of 3), peak RAM 1.05 GB | **0.073x** | 1.14 s (CrispASR, 0.60 GB) | 0.080x | **1.10x** | 2026-09-27 |
| NVIDIA Parakeet-TDT 0.6B v2 (q4_k), NativeAOT, after memory pass | same clip | CPU | **1.43 s** (mean of 3), peak RAM 1.12 GB | **0.100x** | 1.66 s (CrispASR, 0.65 GB) | 0.117x | **1.16x** | 2026-09-27 |
| FunASR-Nano (Paraformer-based, `paraformer-q8.gguf`), after Horizontal Pass A | synthetic-tone input, 99 tokens (timing only; output is not meaningful on this input) | CPU | **11.953 s** (single run) | - | - | - | - | 2026-09-12 |
| SenseVoice (ONNX) | ~3.5 s LibriSpeech clip | CPU | 2229 ms (single run) | ~0.64x | - | - | - | 2026-09-11 |
| FunASR Paraformer (ONNX path) | real Mandarin speech clip | CPU | 2171 ms (single run) | - | - | - | - | 2026-09-11 |

> RTF < 1.0x = faster than real time. Whisper Base runs at 14.3x real time on CPU, Small at 5.0x, Medium at 1.79x and Large-v3 at 1.06x. Voxtral-Mini-4B is far slower than its C++ reference (see the ratio above); its transcripts are correct.
> Harness: `scripts/bench-audio.ps1`, `WhisperFullPipelinePerfBenchTests.cs`.

## Voice Conversion (RVC)

| Component | Scenario | Backend | C# wall | Ratio | Performance Check |
|---|---|---|---:|---:|---|
| RVC v2 (`default` voice, 40 kHz), after perf pass | `a.wav` 5.95 s → 5.94 s; convert (rmvpe 7.7 s, hubert 5.1 s, synthesizer 7.0 s) | CPU | **19.9 / 19.7 s** (was 482.6 s) | ~3.3x (C++ audio.cpp: 8.2 s, ~2.4x behind) | 2026-09-25 |
| RVC (RMVPE + Hubert + synthesizer), after Horizontal Pass A | 3 real tests combined (pitch + content + synthesis) | CPU | 105.355 s combined | - | 2026-09-12 |

## Image & Video Diffusion (CPU and Vulkan iGPU)

Out of this file's LLM/TTS/ASR focus (pipeline `OpenTail.Stingray.Diffusion`). Same dev machine throughout: Ryzen 7 5700G CPU and Radeon Vega 8 iGPU sharing system RAM, so rule 13 applies to every Vulkan number. "Total" = generate time. N/M = not measured. ✅ = correct image checked by eye. C++ = vendored `sd-cli` / `sd.cpp`, where the checkpoint is supported there.

### Status matrix (2026-09-24, and later rows where noted)

| Model | CPU working | GPU working | CPU speed | GPU speed | C++ CPU ref | C++ GPU ref |
|---|---|---|---|---|---|---|
| Wan 2.1 (1.3B) | ✅ | ✅ Vulkan | **74.0-76.9 s**, 256² 1 frame 20-step (2026-09-24) | **107.5 s**, 512² 2 frames 4-step distilled (2026-09-13, Phase 4) | 256.6 s (256² 1 frame 20-step, same files) | - |
| LTX-Video 2B | ✅ | ✅ Vulkan | **133.3 s**, 512² 1 frame 20-step (2026-09-24) | 144.0 s, same config (measured before the 13× VAE fix; stale) | N/M (vendored sd-cli is LTX-2 only) | N/M |
| FLUX.2-dev | ✅ | ✅ Vulkan | 258.8 s, 512² 2-step | **208.1 s** full GPU, quantized Q4_K, 512² 2-step (2026-09-25); 134.1 s hybrid, 128² 4-step (2026-09-24) | N/M | N/M |
| Qwen Image (Q3_K_S) | ✅ | ✅ Vulkan, quantized weights | **277.4 s**, 256² 8-step (2026-09-24) | **159.4 / 161.1 s**, same config (2026-09-24) | **231.4 s**, same config | N/M |
| FLUX.1-schnell | ✅ | ✅ Vulkan | **178.8 s**, 512² 4-step (2026-09-24) | **132.0 s**, same config (2026-09-25, register-tiled flash attention) | 191.7 s | 99.8 s (2026-09-16) |
| SDXL-Turbo | ✅ | ✅ Vulkan | **42.4 s**, 512² 4-step, CFG 0 (2026-09-24) | **35.8 s**, same config (2026-09-24) | N/M | 21.2 s (older config) |
| SD 3.5 medium | ✅ matches C++ | ✅ Vulkan, quantized Q4_K/Q5_K | **82.5 s**, 256² 20-step (2026-09-24) | **84.9 / 86.7 s**, same config (2026-09-24) | N/M (clip_g not on disk) | 48.0 s, same config (2026-09-15) |
| HunyuanVideo (720p cfgdistill fp8) | ✅ colours fixed 2026-09-25 | ⚪ not attempted | **185.7 s** generate, 256² 1 frame 8-step (~19 s per step, 2026-09-25); ~55 s per step at 512² | N/A | N/M (vendored sd-cli is HunyuanVideo 1.5 only) | N/M |
| Z-Image-Turbo | ✅ | ✅ Vulkan | **87.8-97.8 s**, 256² 4-step (2026-09-24) | **52.8-57.6 s**, same config (2026-09-24) | N/M | N/M |

### Current timings by configuration (newest measurement per key)

| Model | Config | Backend | Time | Date | Notes |
|---|---|---|---:|---|---|
| Z-Image-Turbo (S3-DiT + Qwen3-4B) | 256², 4 steps, seed 42, Q4_0 DiT | CPU | 87.8-97.8 s (2 runs) | 2026-09-24 | Q8 activations on the DiT matmul, im2col+GEMM Conv2D. Latent std 1.45, clean apple |
| Z-Image-Turbo | 256², 4 steps | Vulkan iGPU, GPU-resident | 52.8-57.6 s (53.5 / 52.8 / 57.6) | 2026-09-24 | GPU-resident block chain. Latent std 1.4475, clean apple |
| Z-Image-Turbo | 256², 4 steps, CPU, after the scheduler sign fix | CPU | 193.8-231.6 s (3 runs, pre-perf-pass) | 2026-09-12 | Output verified on the apple reference |
| Z-Image-Turbo | 256², 4 steps | Vulkan iGPU | 183.6 s | 2026-09-12 | Same code path as CPU, correct image |
| Z-Image-Turbo | 512², default steps | CPU / Vulkan | 871.8 s (2026-09-10, not visually verified at the time) / 472.7 s (2026-09-11, noise output) | 2026-09-10 / 2026-09-11 | Both timings predate the correctness fixes; not a current result |
| FLUX.1-schnell (Q4_K_S) | 512², 4 steps, seed 42 | CPU | 178.8 s (CLIP 0.7, T5 14.8, DiT 148.8, VAE 14.6) | 2026-09-24 | GEMM-backed Linear. C++ 191.7 s, ratio ~0.93 |
| FLUX.1-schnell (Q4_K_S) | 512², 4 steps | Vulkan iGPU, quantized weights, T5 on CPU | 173.6 s (DiT 35.0 s/step) | 2026-09-24 | Superseded the same day by 132.0 s below |
| FLUX.1-schnell (Q4_K_S) | 512², 4 steps | Vulkan iGPU, register-tiled flash attention (`FlashAttention128`) | **132.0 s** (CLIP 0.6, T5 15.8 on CPU, DiT 23.5 s/step, VAE 7.3) | 2026-09-25 | Attention isolated at FLUX.1 shape: 35-48 ms/call vs 184 ms before. C++ GPU ref 20.45 s/step, so the remaining gap is Q4_K GEMM decode (~905 vs 1,250 GFLOP/s) |
| FLUX.2-dev (Q4_K_S) | 512², 2 steps, guidance 3.5 | Vulkan iGPU, full GPU | **208.1 s** (step 2 78.2 s) | 2026-09-25 | Legacy attention kernel 223.1 s in the same session. One run each, so indicative. Cold first run 285.6 s. FP16 weights 298.4 s (2026-09-24) |
| FLUX.2-dev (Q4_K_S) | 512², 2 steps | CPU | 258.8 s | 2026-09-24 | - |
| FLUX.2-dev | 128², 4 steps, guidance 3.5 | Vulkan iGPU, default hybrid | 134.1 s | 2026-09-24 | Correct image. Checked visually (the GPU test asserts only finite output) |
| FLUX.2-dev | 128², 4 steps | Vulkan iGPU, full GPU (`STINGRAY_FLUX2_GPU_SINGLE_BLOCKS=1`) | 267.8 s | 2026-09-24 | Slower than hybrid on this shared-memory iGPU (rule 13) |
| FLUX.2-dev | 128², 4 steps | CPU | 97.8 s | 2026-09-18 | - |
| Qwen Image (Q3_K_S) | 256², 8 steps, CFG 4, seed 42 | CPU | 277.4 s (33.5 s/step) | 2026-09-24 | C++ 231.4 s (27.6 s/step), ~1.2x behind |
| Qwen Image (Q3_K_S) | 256², 8 steps | Vulkan iGPU, quantized weights on the GPU | 159.4 / 161.1 s (17.8 s/step after the first step) | 2026-09-24 | Kernel parity vs CPU dequant: relErr 2.6e-7 to 8.8e-7. Image visually identical to CPU. The pre-existing full-model GPU parity test (synthetic inputs, 60 blocks) is at cosine 0.989 against a 0.99 threshold; see the LOG |
| SD3.5-medium (Q4_K_M) | 256², 20 steps, CFG 4.5, C++ noise injected | CPU | 82.5 s | 2026-09-24 | Was 284.8 s before the CLIP-G and T5 mask fixes and the packed-GEMM change |
| SD3.5-medium (Q4_K_M) | 256², 20 steps | Vulkan iGPU, quantized weights | 84.9 / 86.7 s | 2026-09-24 | FP16 weights 89.2 s. Matches the C++ reference image |
| SDXL-Turbo | 512², 4 steps, guidance 0 | CPU | 42.4 s (text 2.2, steps 9.8/6.4/6.6/6.7, VAE 10.9) | 2026-09-24 | Was 615.7 s at CFG 7.5 |
| SDXL-Turbo | 512², 4 steps, guidance 0 | Vulkan iGPU | 35.8 s | 2026-09-24 | CPU and GPU images visually identical |
| Wan2.1-T2V-1.3B | 256², 1 frame, 20 steps, CFG 6, F32 DiT | CPU | 74.0 / 76.9 s (DiT ~3.2 s/step, VAE 6.5 s; text cached) | 2026-09-24 | Was 209.6 s before packing F32 weights. C++ 256.6 s, so ours is ~3.3x faster per DiT step |
| Wan2.1-T2V-1.3B | 512², 2 frames, 20 steps (40 DiT forwards) | CPU | ~816 s (~680 ms/block) | 2026-09-13 | Baseline 4238.7 s (2026-09-11), before the workspace, SIMD and K/V-cache work |
| Wan2.1-T2V-1.3B | 512², 2 frames, 4-step distilled | Vulkan iGPU, full 30-block batched DiT, AVX2 3D VAE, parallel frame decode | **107.5 s** (DiT 766.7 ms/step, VAE 8.5 s, UMT5 ~7 s) | 2026-09-13 | 39.4x faster than the 4238.7 s CPU baseline. Parity <1e-2 |
| Wan2.2-T2V-A14B | 128², 1 frame, 2 steps, zero conditioning, dual-model swap | CPU | 40.0 s (VAE decode 2.35 s) | 2026-09-19 | First real run of the high/low-noise swap. Smoke test only; not a quality measurement |
| LTX-Video-2B | 512², 1 frame, 20 steps, CFG 3, real T5 | CPU | **133.3 s** (T5 ~22, DiT ~5.3 s/step, VAE 3.9) | 2026-09-24 | Was 311.6 s. CPU and Vulkan are pixel-identical for the same seed. No C++ number (sd-cli treats it as LTX-2) |
| LTX-Video-2B | 512², 1 frame, 20 steps | Vulkan iGPU | 144.0 s | before 2026-09-24 | Stale (pre-VAE-fix) |
| HunyuanVideo (720p cfgdistill fp8) | 256², 1 frame, 8 steps, distilled guidance 6 | CPU | **185.7 s** generate (~19 s/step, VAE decode 4.3 s) | 2026-09-25 | Was 148.0 s VAE decode before the im2col conv (VAE 5.8 s at 256² on 2026-09-24). Colours fixed by the `post_quant_conv` step |
| HunyuanVideo | 512², 1 frame, 12 steps | CPU | 700.0 s | 2026-09-24 | ~55 s/step |

> Diffusion wall times are a different domain from the LLM rows above. C++ comparisons are only where the vendored `sd-cli` / `sd.cpp` supports the checkpoint.

### Further configurations (newest measurement per config)

| Model | Config | Backend | Time | Date | Notes |
|---|---|---|---:|---|---|
| Wan2.2-T2V-A14B | 128², 1 frame, 2 steps, real UMT5-XXL conditioning, dual-model swap | CPU | 50.7 s (UMT5 encode cond+uncond 13.2 s; VAE 2.2 s) | 2026-09-20 | Supersedes the zero-conditioning run (40.0 s, 2026-09-19), which is kept in the table above |
| Wan2.1-T2V-1.3B | 256², 1 frame, 20 steps, seed 42 | Vulkan iGPU | 132.8 s (DiT 79.4 s, 3.97 s/step) | 2026-09-15 | C++ Vulkan 60.3 s (DiT 45.2 s), C++ CPU 62.3 s. Ours ~2.2x slower on Vulkan at that date |
| SD3.5-medium (Q4_K_M) | 256², 20 steps, CFG 4.5 | CPU | 175.3 s (2026-09-19, before the correctness fix); 267.0 s after the missing-component fix (2026-09-19); 82.5 s (2026-09-24, current) | 2026-09-24 | C++ reference 54.4 s (CPU golden run, 2026-09-20) |
| SD3.5-medium (Q4_K_M) | 256², 20 steps, CFG 4.5 | Vulkan iGPU | 120.5 s (2026-09-20, before quantized weights); 84.9 / 86.7 s (2026-09-24, current) | 2026-09-24 | C++ Vulkan 48.01 s (2026-09-15) |
| Stable Diffusion 1.5 | 512², 20 steps, CFG 7.5, seed 42 | CPU | 2789.9 s | 2026-09-15 | No newer CPU run |
| Stable Diffusion 1.5 | 512², 20 steps, CFG 7.5 | Vulkan iGPU, batched recording + tiled attention | 170.8 s | 2026-09-19 (re-verification; 172.2 s on 2026-09-15) | C++ Vulkan 80.36 s (UNet 71.64 s, VAE 8.35 s, 2026-09-15) |
| Stable Diffusion 1.5 + RealESRGAN_x4plus | 512² → 2048², 20 steps | Vulkan iGPU | 781.3 s (base ~657 s, upscaler ~124.5 s) | 2026-09-11 | No newer run |
| SDXL-Turbo | 512², 4 steps, CFG 0, seed 42 | Vulkan iGPU | 40.7 s (text 3.03 s, denoise 14.46/5.15/5.16/5.15 s, VAE 7.48 s) | 2026-09-19 | Superseded by 35.8 s (2026-09-24, table above). C++ Vulkan 21.18 s at cfg≈1.0 (2026-09-11) |
| SDXL-Turbo | 512², 4 steps, CFG 0, seed 42 | CPU | 615.7 s (text 2.55 s, denoise 62.73/57.40/55.54/80.03 s, VAE 357.35 s) | 2026-09-19 | Superseded by 42.4 s (2026-09-24). Kept because it is the only CPU row from the same pass as the Vulkan 40.7 s |
| FLUX.2-dev (Q4_K_S, Mistral-24B) | 512², 20 steps, guidance 3.5, production close-out | CPU | 2341.5 s (~39 min) | 2026-09-19 | No newer 512² 20-step run |
| FLUX.2-dev (Q4_K_S) | single DiT forward, 512² (1088 tokens, 56 layers) | CPU | 94.99 s (matmul 72.05 s, ~830 GFLOP/s; dequant 0.00 s) | 2026-09-18 | Microbenchmark |
| FLUX.2-dev (Q4_K_S) | double-block loop only, 8 blocks, 1024 image tokens | CPU vs Vulkan | CPU 17,092 ms; Vulkan batched 21,480 ms (unbatched 21,604 ms); PyTorch CPU 22,232 ms | 2026-09-19 | Vulkan slower than CPU at this shape on the iGPU (rule 13) |
| Qwen-Image (Q3_K_S) | single DiT forward, real weights | Vulkan iGPU vs CPU | GPU 217.1 s vs CPU 69.6 s (cosine 0.990, maxDiff 0.218) | 2026-09-19 | Superseded by the 2026-09-24 GPU path (159.4 s total, table above) |
| LTX-Video-2B v0.9.1 | 256², 1 frame, 25 steps, CFG 3.0 | CPU | 100.5 s (3 runs ~100 s each) | 2026-09-11 | Different config from the 512² row above. Before the `TimestepEmbedder` sin/cos fix (2026-09-14); the fix is a correctness change with no timing change |
| Z-Image-Turbo | 256², 4 steps, seed 42, GPU residency | Vulkan iGPU | 63.6 s (2026-09-20, residency on, before the norm-weight fix) | 2026-09-20 | Superseded by 52.8-57.6 s (2026-09-24) |
| SDXL SpatialTransformer attention kernel (microbenchmark) | 4096×4096 self-attention, Level 1 | CPU AVX2 / FP32 GPU / FP16 GPU | 512.7 ms / 345.7 ms / 150.7 ms (FP16 2.29x faster than FP32; 3.40x vs CPU) | 2026-09-17 | Other shapes: 1024×1024 self 63.6 / 43.8 / 18.8 ms; 4096×77 cross 8.8 / 15.6 / 6.7 ms; 256×256 self 4.4 / 2.8 / 1.5 ms |

> Dropped to the LOG, because each was superseded by a newer run of the same config, or was a pre-fix run with noise output: the Z-Image 512² CPU and Vulkan timings from 2026-09-10 and 2026-09-11, the HunyuanVideo 4-step zero-conditioning and real-conditioning runs (434.0 s and 599.5 s, 2026-09-19, noise output, before the 2026-09-25 fixes), and the older Wan2.1 512² GPU runs.

## Sweeps and experiments (CPU, 2026-10)

### Smoke-test workload, CPU, 256×256 `stingray image` (2026-10-30)

Wall time per run including model load, `--backend cpu --seed 42`. Two runs each; output byte-identical between runs. No C++ reference timed.

| Pipeline | Steps | Wall, run 1 / run 2 |
|---|---:|---:|
| SD 1.5 | 4 | 15 s / 13 s |
| SDXL-Turbo | 1 | 14 s / 10 s |
| FLUX.1-schnell Q4_K_S | 2 | 75 s / 50 s |
| Z-Image-Turbo Q4_0 | 2 | 75 s / 63 s |
| SD 3.5 Medium Q4_K_M | 20 | 61 s (one run; 4 steps is not usable) |

### Q8_0 batched prefill scheduling, CPU (2026-10-30)

Q8_0 weights use the int8 batched-prefill dispatch (16-row blocks × token groups). Output is bit-identical to the old path (`MatMulBatchedQ8EquivalenceTests`). Interleaved runs; control `STINGRAY_CPU_PREFILL_Q8=0`; raw prompt, greedy, 8 threads; llama.cpp b10306 `llama-bench -p N -t 8 -r 2` on the same token count.

| Model (Q8_0) | Prompt tok | Old path t/s (2 runs) | New path t/s (2 runs) | Gain | llama.cpp pp t/s | Ratio (new) |
|---|---:|---:|---:|---:|---:|---:|
| granite-4.0-h-1b | 687 | 96.3 / 105.1 | 143.8 / 141.7 | +42% | 114.8 | 1.24x |
| granite-4.0-h-350m | 687 | 305.2 / 341.9 | 517.2 / 518.5 | +60% | 485.5 | 1.07x |
| Qwen3-0.6B | 708 | 229.6 / 235.3 | 374.7 / 364.3 | +59% | 385.5 | 0.96x |
| ERNIE-4.5-0.3B | 850 | 362.3 / 370.7 | 554.0 / 599.1 | +57% | 539.0 | 1.07x |
| LFM2-1.2B | 780 | 138.9 / 160.0 | 207.1 / 184.7 | +31% | 185.0 | 1.06x |
| gpt2 | 864 | 863.5 / 937.8 | 1501.3 / 1483.2 | +66% | 1654.7 | 0.90x |

Decode is unchanged (single-token path).

### Decode is compute-bound per thread (2026-10-30)

Greedy decode, 64 tokens, raw prompt, CPU, default threads (16) vs `llama-bench -t 8` (b10306).

| Model | Stingray t/s | llama.cpp t/s | Ratio |
|---|---:|---:|---:|
| Qwen3-0.6B Q8_0 | 47.0 | 54.9 | 0.86x |
| granite-4.0-h-1b Q8_0 | 18.9 | 21.0 | 0.90x |
| SmolLM2-1.7B Q4_K_M | 27.1 | 35.0 | 0.77x |
| Qwen2.5-3B Q4_K_M | 15.8 | 19.6 | 0.80x |
| Qwen3-4B Q4_K_M | 12.5 | 14.9 | 0.84x |
| Mistral-7B Q4_K_M | 7.5 | 9.2 | 0.81x |
| Qwen3-8B Q4_K_M | 6.9 | 8.3 | 0.84x |

Mistral-7B thread scaling (decode t/s, Stingray / llama.cpp): 1 thread 1.2 / 4.0; 2 threads 2.2 / 7.2; 4 threads 3.8 / 8.6; 6 threads 5.1 / 8.7; 8 threads 6.2 / 8.6; 16 threads 7.7 / 8.3. Fixed (bit-identical): the decode two-row Q4_K kernel unpacks the scales and mins once per super-block. Mistral-7B after the fix: 1 thread 1.65 t/s, 4 threads 5.0, 16 threads 7.8. The remaining gap is the activation format (Q8_KS), an owner decision; see the next section.

### Decode activation format: Q8_KS (default) vs Q8_K (2026-10-30)

Wikitext-2 test, one 1024-token chunk, CPU, scored token by token (decode path). llama.cpp b10306 `llama-perplexity -c 1024 --chunks 1`, same second half.

| Model (Q4_K_M) | llama.cpp PPL | Q8_KS PPL | Q8_K PPL | Q8_K − Q8_KS, paired mean dNLL |
|---|---:|---:|---:|---|
| SmolLM2-1.7B | 5.151 | 5.172 | 5.168 | −0.0036 ± 0.0043 (no difference) |
| Qwen3-4B | 7.688 | 7.624 | 7.715 | +0.0089 ± 0.0039 (Q8_K +0.9% PPL) |
| Mistral-7B | 3.035 | 3.012 | 3.019 | +0.0035 ± 0.0021 (+0.35%, marginal) |

Decode t/s, 24 tokens, interleaved, two runs each:

| Model | Threads | Q8_KS | Q8_K (GEMM at N = 1) |
|---|---:|---|---|
| Mistral-7B Q4_K_M | 1 | 1.7, 1.7 | **5.2, 5.0** (llama.cpp 4.0) |
| Mistral-7B Q4_K_M | 4 | 5.1, 6.5 | **8.0, 8.1** (llama.cpp 8.6) |
| Mistral-7B Q4_K_M | 16 | 8.6, 8.6 | 8.2, 8.0 (llama.cpp 8.3) |
| Qwen3-4B Q4_K_M | 1 | 3.5, 3.6 | **8.1, 8.2** |
| Qwen3-4B Q4_K_M | 4 | 12.4, 12.5 | 13.0, 13.3 |
| Qwen3-4B Q4_K_M | 16 | 13.9, 14.2 | 13.2, 12.8 |

**Decision (2026-10-30): Q8_K decode is used automatically with one CPU thread** (`-t 1` or `STINGRAY_CPU_THREADS=1`). With more threads the default stays Q8_KS. Mistral-7B Q4_K_M after the change: 1 thread 5.1 t/s (was 1.7; llama.cpp 4.0–4.7), 8 threads 8.8 t/s (llama.cpp 8.6–9.45). `STINGRAY_CPU_DECODE_VIA_GEMM=1` forces it at any thread count.

### MoE lot, CPU and Vulkan iGPU (2026-10-04)

Method: `docs/reference/benchmark-prompt.txt` (468-587 tokens), 24 generated tokens, `--temp 0 -t 6`, three runs per cell, median. llama.cpp has only the CPU and RPC backends vendored, so llama.cpp GPU rows do not exist here; GPU columns are ours alone.

| Model | Scenario | Backend / path | C# (OT, t/s) | C++ (llama.cpp, t/s) | Ratio |
|---|---|---|---:|---:|---:|
| Qwen1.5-MoE-A2.7B Q4_K_M | prefill | CPU | 81.9 | 97.0 | 0.84x |
| Qwen1.5-MoE-A2.7B Q4_K_M | decode | CPU | 12.0 | 24.2 | 0.50x |
| Qwen1.5-MoE-A2.7B Q4_K_M | prefill / decode | Vulkan hybrid (`-g 4`) | 63.8 / 8.4 | n/a | - |
| Qwen1.5-MoE-A2.7B Q4_K_M | prefill / decode | Vulkan full GPU (`-g -1`) | 71.9 / 10.8 | n/a | - |
| Mixtral-8x7B (Nous-Hermes-2-DPO i1-Q4_K_S) | prefill | CPU | 8.8 | 11.0 | 0.80x |
| Mixtral-8x7B (Nous-Hermes-2-DPO i1-Q4_K_S) | decode | CPU | 2.4 | 5.7 | 0.42x |
| Mixtral-8x7B (Nous-Hermes-2-DPO i1-Q4_K_S) | prefill / decode | Vulkan hybrid (`-g 4`) | 8.6 / 2.2 | n/a | - |
| Mixtral-8x7B (Nous-Hermes-2-DPO i1-Q4_K_S) | prefill / decode | Vulkan full GPU (`-g -1 -c 1024`) | 3.9 / 3.8 | n/a | - |
| Phi-3.5-MoE-instruct Q3_K_M | prefill | CPU | 16.8 | 31.8 | 0.53x |
| Phi-3.5-MoE-instruct Q3_K_M | decode | CPU | 6.2 | 12.5 | 0.50x |
| Phi-3.5-MoE-instruct Q3_K_M | prefill / decode | Vulkan hybrid (`-g 4`) | 13.8 / 3.8 | n/a | - |
| Phi-3.5-MoE-instruct Q3_K_M | prefill / decode | Vulkan full GPU (`-g -1 -c 1024`) | 16.8 / 5.3 | n/a | - |
| GLM-4.5-Air Q2_K | prefill | CPU | 2.9 | 10.9 | 0.27x |
| GLM-4.5-Air Q2_K | decode | CPU | 1.3 | 6.1 | 0.21x |
| GLM-4.5-Air Q2_K | prefill / decode | Vulkan layer split (`-g 4`) | 2.5 / 0.5 | n/a | - |
| GLM-4.5-Air Q2_K | prefill / decode | `-g -1` | 2.6 / 0.5 (one run) | n/a | - |
| Llama-4-Scout-17B-16E Q3_K_M | prefill | CPU | 6.6 | 11.4 | 0.58x |
| Llama-4-Scout-17B-16E Q3_K_M | decode | CPU | 2.7 | 4.6 | 0.59x |
| Llama-4-Scout-17B-16E Q3_K_M | prefill / decode | Vulkan hybrid (`-g 4`) | 3.3 / 0.5 | n/a | - |
| Llama-4-Scout-17B-16E Q3_K_M | prefill / decode | `-g -1` (hybrid, 30 GPU + 18 CPU layers) | 2.8 / 0.3 | n/a | - |

Reading: on CPU we are below llama.cpp on every MoE model here: prefill 0.27–0.84x, decode 0.21–0.59x. The worst gap is GLM-4.5-Air (a 128-expert, 47-layer MoE at Q2_K).

### Full re-sweep of available text checkpoints, CPU (2026-10-08)

Release `stingray` CLI, `-g 0`, ~800–1100-token prompt, 32 generated tokens, best of 3, machine otherwise idle. Reference `llama-bench -p <same N> -n 32 -t 6`.

| Model | OT prefill (t/s) | llama.cpp prefill | Ratio | OT decode @long ctx | llama.cpp tg32 | Ratio |
|---|---:|---:|---:|---:|---:|---:|
| SmolLM2-135M Q4_K_M (791 tok) | 965.4 | 839.5 | 1.15x | 102.7 | 249.8 | 0.41x |
| SmolLM2-360M Q4_K_M (791) | 457.6 | 351.1 | 1.30x | 66.1 | 131.1 | 0.50x |
| SmolLM2-1.7B Q4_K_M (791) | 218.6 | 172.9 | 1.26x | 23.2 | 37.8 | 0.61x |
| Qwen3-0.6B Q8_0 (900) | 222.5 | 280.6 | 0.79x | 40.0 | 59.6 | 0.67x |
| Qwen3-4B Q4_K_M (900) | 96.4 | 77.8 | 1.24x | 12.0 | 16.6 | 0.72x |
| Falcon3-3B Q4_K_M (773) | 153.3 | 123.5 | 1.24x | 17.1 | 23.3 | 0.73x |
| LFM2-1.2B Q8_0 (850) | 146.7 | 187.8 | 0.78x | 28.8 | 34.1 | 0.85x |
| Phi-3-mini-4k Q4_K_M (569, matched) | 35.4 | 68.5 | 0.52x | 11.0 | — | — |
| SmolLM3 Q4_K_M (1123) | 129.2 | 99.0 | 1.31x | 15.5 | 21.3 | 0.73x |
| Mistral-7B-Instruct-v0.3 Q4_K_M (1005) | 53.8 | 44.4 | 1.21x | 7.3 | 9.1 | 0.80x |
| Qwen3-8B Q4_K_M (900) | 55.2 | 44.0 | 1.26x | 6.5 | 8.5 | 0.77x |

Short-context decode re-check (128 generated, 3 runs; llama.cpp `-p 0 -n 128 -t 6`):

| Model | OT decode (t/s) | llama.cpp tg128 | Ratio | Verdict |
|---|---:|---:|---:|---|
| SmolLM2-1.7B Q4_K_M | 28.2–30.8 | 35.5 | 0.79–0.87x | better than the 2026-07 row |
| Mistral-7B Q4_K_M | 7.8–8.0 | 9.43 | 0.83–0.85x | −4%, within run noise |
| Qwen3-8B Q4_K_M | 7.4 | 8.54 | 0.87x | −4%, within run noise |

**Regression found by this sweep:** Phi-3-mini prefill, 35.4 t/s (0.52x of llama.cpp), down 12% from 40.1 (2026-09-13). Closed 2026-10-08 without bisecting (low model usage). Details in the LOG.

### Load-and-generate smoke sweep, other checkpoints (2026-10-08)

One run each, `stingray -g 0 -p "The capital of France is" -n 16 --temp 0 --allow-unverified-arch`, raw prompt, idle machine. Speeds are single-run and are not benchmarks. Result: 46 models tried, 41 loaded and generated, 4 crashed at load (fixed the same day), 1 timed out.

| Model | Decode t/s (single run) | Output |
|---|---:|---|
| granite-4.0-h-350m Q8_0 | 52.2 | correct |
| ERNIE-4.5-0.3B Q8_0 | 57.0 | correct |
| jais-590m-chat Q4_K_M | 13.8 | correct |
| qwen2.5-0.5b / coder-0.5b Q4_K_M | 48.5 / 48.7 | correct |
| Hunyuan-0.5B Q8_0 | 53.4 | thinking-style, coherent |
| Maincoder-1B Q4_K_M | 10.7 | correct |
| qwen2.5-1.5b / coder-1.5b Q4_K_M | 27.4 / 26.3 | correct |
| SmolLM2-1.7B Q5_K_M | 20.5 | correct |
| granite-4.0-h-1b Q8_0 | 19.2 | correct |
| stablelm-zephyr-3b Q4_K_M | 19.0 | coherent |
| TinyLlama-4x1.1B-MoE Q4_K_M | 28.7 | correct |
| qwen2.5-3b / coder-3b Q4_K_M | 15.9 / 15.7 | correct |
| phi-2 Q4_K_M | 12.7 | correct |
| gemma-3-4b-it Q4_K_M | 13.6 | coherent (raw prompt, instruct model) |
| OLMoE-1B-7B Q4_K_M | 38.4 | correct |
| Ministral-8B Q4_K_M | 7.9 | correct |
| gemma-4-E4B-it Q4_K_M | 11.2 | correct |
| gemma-4-12b-it Q4_K_M | 5.2 | correct |
| LFM2-8B-A1B Q4_K_M | 29.3 | correct |
| c4ai-command-r7b Q4_K_M | 7.2 | correct |
| Apertus-8B Q4_K_M | 6.4 | correct |
| DeepSeek-V2-Lite Q2_K | 25.9 | coherent (diverges from llama-server at token 9) |
| DeepSeek-V2-Lite Q8_0 | 12.5 | coherent |
| Qwen1.5-MoE-A2.7B Q4_K_M | 13.6 | correct |
| GLM-4.7-Flash Q2_K | 13.0 | coherent (thinking) |
| Mistral-Small-3.2-24B Q4_K_S | 2.8 | correct |
| Trinity-Mini Q4_K_M | 15.8 | coherent |
| Qwen3-Coder-30B-A3B Q4_K_M | 12.0 | correct |
| Phi-3.5-MoE Q3_K_M | 11.1 | correct |
| EXAONE-4.5-33B Q4_K_M | 2.0 | coherent |
| Nous-Hermes-2-Mixtral-8x7B Q4_K_S | 4.9 | correct |
| Hunyuan-A13B Q3_K_S | 5.7 | coherent |
| GLM-4.5-Air Q2_K (42 GB) | 0.7 | coherent (thinking); memory-bound |

Models whose output looked odd on this run (gpt2, pythia-160m, starcoder2, xverse, gpt-oss-20b) were base or code models fed a raw prompt. A 2026-10-29 follow-up verified gpt2, starcoder2 and xverse against llama.cpp b10306's greedy text; gpt-oss matches its top-10 ranking. Details in the LOG.

Crashed at load and fixed the same day: Ornith-1.0-9B, Qwen3.6-27B, Qwen3.8-27B, Qwen3.6-35B-A3B (`qwen35` hybrid arch). Timed out: cerebras GLM-4.5-Air-REAP-82B (33 GB, inconclusive).

### GC / managed-allocation cost during decode (2026-10-08)

Measured with `STINGRAY_GC_STATS=1`, differencing a short and a long run.

| Workload (CPU, greedy) | Extra tokens | Extra managed allocation | Extra GC pause | Decode time | GC share of decode |
|---|---:|---:|---:|---:|---:|
| SmolLM2-135M Q4_K_M (169 t/s) | 280 | 270 MiB (~1 MiB/token) | 15 ms | 1.65 s | 0.9% |
| Qwen3.8-27B UD-Q3_K_XL, MTP depth 1 | 80 | 213 MiB (~2.7 MiB/token) | 111 ms | ~31 s | 0.4% |

Verdict: small. Scratch memory is already in pre-allocated `NativeMemory`, so an arena would gain under 1%. Not worth building.

### Scratch arena experiment, phase P0 (2026-10-10): no-go

Question: does reusing scratch memory across prefill calls make CPU prefill measurably faster (go threshold about 2%)? Decision: stop at P0 and close the item. The per-call matmul scratch sites are not on the hot path. The allocations that remain are about ten large zeroed buffers per prefill call. Measured upper bound on the saving: 95 MiB in 8.3 ms, 200 MiB in 17.9 ms, which is below the threshold. Scope: CPU prefill on dense Q4_K_M/Q8_0 models, one CPU, Windows allocator. Plan: `docs/done/2026-10-09-scratch-arena-experiment-plan.md`.

### Opt-in numerics gates (C# A/B on one machine, not vs llama.cpp)

Switches that trade speed for closer agreement with llama.cpp's arithmetic. Gate-off vs gate-on medians, CPU, Release, 96-token greedy decode. The gate stays off by default.

| Gate | Model | Gate off (t/s) | Gate on (t/s) | Change | Check (date) |
|---|---|---:|---:|---:|---|
| `STINGRAY_Q5K_DECODE_Q8K=1` (Q8_K activations for every Q5_K decode matvec) | SmolLM2-1.7B-Instruct Q5_K_M | 21.0 | 13.3 | −37% | 2026-10-01 |
| same | phi-2 Q4_K_M (32 Q5_K tensors) | 11.7 | 9.8 | −16% | 2026-10-01 |

Fidelity side (paired per-token NLL vs llama.cpp, rms gate off → on): GLM-4.5 0.328 → 0.204; phi-2 0.121 → 0.091; Phi-3-mini 0.127 → 0.112; jais-590M 0.205 → 0.169; Ornith-9B 0.066 → 0.063; SmolLM2 0.142 → 0.148. See `docs/00-current-work.md` item 19.

---

## Known Measurement Gaps & C++ Reference Head-to-Head

Tracks parity with authoritative C++ references (`sd-cli.exe`, `audiocpp_cli.exe`, `llama.cpp`). "Out by" is the ratio of ours to the reference; values under 1.0x are faster than the reference.

### Diffusion vs C++ reference (newest head-to-head per config)

| Model | Scenario / stage | Backend | C# (ours) | C++ ref | Ratio | Date |
|---|---|---|---:|---:|---:|---|
| Stable Diffusion 3.5 Medium | End-to-end, 256², 20 steps, CFG 4.5 | Vulkan iGPU | 116.8 s | 48.01 s | 2.43x | 2026-09-15 |
| Stable Diffusion 3.5 Medium | MMDiT denoise loop (20 steps) | Vulkan iGPU | ~105.0 s | 36.81 s | 2.85x | 2026-09-15 |
| Stable Diffusion 3.5 Medium | Text conditioning (CLIP-L + CLIP-G + T5-XXL) | Vulkan iGPU | ~10.5 s | 9.50 s | 1.11x | 2026-09-15 |
| Stable Diffusion 3.5 Medium | VAE decode | Vulkan iGPU | ~1.30 s | 1.70 s | 0.76x (ours faster) | 2026-09-15 |
| Stable Diffusion 3.5 Medium | End-to-end, CPU baseline | CPU | 536.4 s | 48.01 s (Vulkan ref) | 11.17x | 2026-09-15 |
| Stable Diffusion 1.5 | End-to-end, 512², 20 steps, CFG 7.5 | Vulkan iGPU | 171.7 s | 80.36 s | 2.14x | 2026-09-16 |
| Stable Diffusion 1.5 | UNet denoise loop (20 steps) | Vulkan iGPU | 171.7 s | 71.64 s | 2.40x | 2026-09-16 |
| Stable Diffusion 1.5 | Text conditioning (CLIP-L) | Vulkan iGPU | ~0.50 s | 0.35 s | 1.43x | 2026-09-15 |
| Stable Diffusion 1.5 | VAE decode | Vulkan iGPU | ~12.0 s | 8.35 s | 1.44x | 2026-09-15 |
| Stable Diffusion 1.5 | End-to-end, CPU baseline | CPU | 2789.9 s | 80.36 s (Vulkan ref) | 34.72x | 2026-09-15 |
| Stable Diffusion 1.5 + ControlNet Canny | End-to-end, 512², 20 steps | Vulkan iGPU | 204.4 s | 104.2 s (est.) | 1.96x | 2026-09-16 |
| SDXL-Turbo | End-to-end, 512², 4 steps, guidance 0 | Vulkan iGPU | 33.8 s | 21.18 s | see LOG | 2026-09-17 |
| SDXL-Turbo | UNet denoise loop (4 steps) | Vulkan iGPU | ~24.2 s | 10.45 s | 2.32x | 2026-09-17 |
| SDXL-Turbo | Text conditioning (CLIP-L + CLIP-G) | Vulkan iGPU | 3.07 s | 1.86 s | 1.65x | 2026-09-17 |
| SDXL-Turbo | VAE decode | Vulkan iGPU | 6.35 s | 8.85 s | 0.72x (ours faster) | 2026-09-17 |
| SDXL-Turbo (CPU baseline) | End-to-end, 512², 4 steps | CPU | 832.3 s | 21.18 s (Vulkan ref) | 39.29x | 2026-09-11 |
| LTX-Video-2B | End-to-end, 512², 1 frame, 20 steps | Vulkan iGPU | 156.2 s warm / 181.3 s cold | n/a (sd-cli blocked on audio cross-attn) | - | 2026-09-19 |
| LTX-Video-2B | End-to-end, 512², 1 frame, 20 steps | CPU | 399.5 s | n/a | - | 2026-09-18 |
| Stable Diffusion 3.5 Medium (quantized) | End-to-end, 256², 20 steps | Vulkan iGPU | 91.3 s warm / 101.2 s cold | 48.01 s | 1.90x | 2026-09-19 |
| FLUX.1-schnell (Q4_K_S) | End-to-end, 512², 4 steps | Vulkan iGPU | 198.1 s warm / 285.6 s cold | 99.8 s | 1.98x | 2026-09-16 |
| FLUX.1-schnell | DiT denoise loop (4 steps, 57 blocks/step) | Vulkan iGPU | ~160.0 s | 81.81 s | 1.95x | 2026-09-16 |
| FLUX.1-schnell | Text conditioning (CLIP-L + T5-XXL) | Vulkan iGPU | 28.06 s | ~11.29 s | 2.49x | 2026-09-13 |
| FLUX.1-schnell | VAE decode | Vulkan iGPU | 13.38 s | 8.78 s | 1.52x | 2026-09-13 |
| FLUX.1-schnell (CPU baseline) | End-to-end, 512², 4 steps | CPU | 853.0 s | 251.1 s (C++ CPU) | 3.40x | 2026-09-13 |
| Wan2.1-T2V-1.3B | End-to-end, 256², 1 frame, 20 steps | Vulkan iGPU | 132.8 s (2026-09-15); 122.1 s after the grid-artifact fix (2026-09-18) | 60.25 s | 2.20x | 2026-09-15 |
| Wan2.1-T2V-1.3B | DiT denoise loop (20 steps) | Vulkan iGPU | 79.4 s | 45.20 s | 1.76x | 2026-09-15 |
| Wan2.1-T2V-1.3B | Text conditioning (UMT5-XXL) | Vulkan iGPU | 38.7 s | 13.02 s | 2.98x | 2026-09-15 |
| Wan2.1-T2V-1.3B | VAE decode (3D causal VAE) | Vulkan iGPU | 6.05 s | 2.02 s | 3.00x | 2026-09-15 |
| Wan2.1-T2V-1.3B (CPU) | End-to-end, 512², 1 frame, 20 steps | CPU | 425.7 s | 196.25 s (C++ CPU) | 2.17x | 2026-09-14 |
| Wan2.1-T2V-1.3B (CPU) | DiT denoise loop (20 steps) | CPU | ~340.0 s | 173.76 s | 1.96x | 2026-09-14 |
| Wan2.1-T2V-1.3B (CPU) | Text conditioning (UMT5-XXL) | CPU | ~65.0 s | 12.98 s | 5.01x | 2026-09-14 |
| Wan2.1-T2V-1.3B (CPU) | VAE decode | CPU | ~18.0 s | 9.48 s | 1.90x | 2026-09-14 |
| F5-TTS Base (DiT) | End-to-end, text → 2.77 s audio | CPU | 27.25 s | 43.20 s | 0.63x (ours faster) | 2026-09-11 |
| FishSpeech S2 Pro (Q4_K) | End-to-end, text → 3.44 s audio | CPU | 28.46 s | 36.12 s | 0.79x (ours faster) | 2026-09-09 |

> Diffusion and TTS wall times above are at different configs from the per-config tables in the diffusion and audio sections; the ratio is only meaningful within a row. Several "ours" values here are older runs than the current table entries.

### Remaining unmeasured gaps (blocked on build, SDK or hardware)

| Model / area | Scenario | Backend | Why blank / blocked | What is needed |
|---|---|---|---|---|
| Piper, Kokoro, MeloTTS, MMS-TTS | text → audio | CPU | No C++ `.exe` in the example build trees (checked 2026-09-10) | Build the reference |
| Parler-TTS | text → audio | CPU | `parler` is not a registered family in `audiocpp_cli` | Family support upstream |
| All models | any | CUDA | No CUDA device on the dev machine | Nvidia hardware. Q6_K/Q5_K prefill currently dequantizes to FP16 and calls cuBLAS; direct MMQ is the open opportunity |
| All models | any | Vulkan iGPU | No llama.cpp Vulkan reference | Discrete GPU |
| audio.cpp TTS/ASR pipelines | any | Vulkan iGPU | `audiocpp_cli --backend vulkan` runs, but pipelines are not GPU-wired on our side | Expected to be the largest TTS win |
| TTS pipelines (all) | full synthesis | GPU | Not wired to GPU yet | GPU residency work |
| CPU KV cache | bf16 / q8 dtype | CPU | `PagedKvCache` is fixed to fp32 | Vulkan showed +57% decode at no quality cost |
| Llama-4-Scout 17B-16E Q4_K_M | prefill + decode | CPU | Cancelled by explicit user instruction (~93 GB vs 64 GB RAM) | Larger host |
| EXAONE-4.5-33B Q4_K_M | prefill + decode | Vulkan hybrid | `HybridForwardPass` post-norm gap is fixed (2026-09-26); the hybrid path is measured in the LLM table | - |
| Carnice 35B-A3B-MTP (APEX) | prefill / decode | CPU | No public repo located | Original checkpoint |
| Qwen3.8-27B | chat-template correctness | CPU | Three Jinja chat-template rendering gaps logged as warnings | Template work (not a perf gap) |
| QwenTTS, CosyVoice3 (streaming) | TTFA vs C++ | CPU | Both refuse `--mode streaming` ("only supports offline sessions") | Upstream support |
| FishSpeech S2 Pro (streaming) | TTFA vs C++ | CPU | `s2.exe --stream-file` works but prints no first-chunk timestamp | Printed first-chunk timestamp |
| FunASR Paraformer (GGUF path) | any | CPU | `paraformer-q8.gguf` lacks the `pf.vocab` metadata; throws on load (2026-09-11) | Re-convert the GGUF |
| SenseVoice | any | any | Implemented 2026-09-11 (see the ASR table, `~0.64x`); no C++ run | - |
| Granite-4.0-3B-Vision, Granite-Vision-3.2-2B | vision-encode correctness | CPU | Degenerate, not image-grounded output; projector is a simplified approximation | Projector rewrite (see LOG) |
| `deepseek2`-tagged VLMs (MLA) | any | CPU | Only "Lite" MLA (`q_lora_rank==0`) is implemented | Full MLA (`q_lora_rank>0`) |

---

## Strengths & weaknesses (current)

This section reads across the ratio tables above; it does not replace them.

**Where we are genuinely strong**
- **Dense 7–12B models reach prefill parity** on CPU: Gemma-4-12B (1.02x), Qwen3-8B (1.02x), Mistral-7B-Instruct-v0.3 (1.00x), Ministral-8B (0.99x).
- **Whisper family (0.83–0.95x across all four sizes)** and the **Qwen3 Forced Aligner (1.01x)** are the most consistently near-parity subsystem.
- **Short-context decode is close to parity** on most dense LLMs (SmolLM2-1.7B 0.88–0.89x), so the bandwidth-bound decode path is sound.
- **Audio diffusion GPU residency (2026-09-18):** Stable Audio 3 Small in 27.12 s (4.0 s audio, 16 steps) and ACE-Step Turbo in 28.92 s on the Vulkan iGPU, down from several minutes on CPU.
- **FLUX.2 double-block loop** on CPU (17.09 s) beats the PyTorch CPU reference (22.23 s) on identical hardware (1.30x).

**Where we clearly trail**
- **Decode falls behind as context grows:** SmolLM2 decode ratio 0.88x at short context, 0.67x at 3.2k (2026-09-10). Re-measured at ~3.7k on 2026-11-01 it is 0.93–0.98x, so the long-context gap may not reproduce; re-open only with a matched-depth measurement.
- **OLMoE decode** is about 0.81x on the 2026-10-30 re-measure (was 0.50x on 2026-09-10).
- **Hybrid-GDN prefill** was the worst architecture (0.05–0.16x on 2026-09-10); batched prefill brought the 35B to 0.63–0.96x.
- **MoE models on CPU** are below llama.cpp everywhere measured: prefill 0.27–0.84x, decode 0.21–0.59x (2026-10-04).
- **Small models decode poorly relative to llama.cpp:** SmolLM2-135M 0.125x, 360M 0.19x (2026-09-11), largely on the older path.

**Vulkan iGPU** (shared memory; rule 13 applies)
- Vulkan prefill usually loses to CPU prefill on dense and MoE models at these sizes, while Vulkan decode is often close to CPU.
- Hybrid-GDN on Vulkan depends on whether the model fits in VRAM. Full offload of Ornith-9B gives 4x over CPU; partial offload of Qwen3.6-27B is slower than CPU.

---

## MoE expert residency under memory pressure, CPU (2026-11-09)

Question (SharpMind review, item R2): how much does CPU MoE decode depend on the OS keeping the memory-mapped expert pages resident? Protocol and fixed thresholds: [2026-11-09-moe-expert-residency-measurement-plan.md](docs/4-performance/2026-11-09-moe-expert-residency-measurement-plan.md). Binary: `stingray` NativeAOT from commit `5d6e2044`. Model: `Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf` (17.28 GB, `qwen3moe`, 48 layers, 128 experts, 8 used). Command: `stingray -f <~181-token prompt> -n 128 --temp 0 --repeat-penalty 1.0 -t 8 -g 0 --single-turn`. Hardware: Ryzen 7 5700G, 64 GB RAM, model on a Samsung 870 QVO SATA SSD (F:). Session 2026-11-09, one run at a time, order A B A B A B.

| Condition | Runs | Free RAM during run | Decode t/s: median (runs) | Prefill t/s (median) | Wall s (median) | Hard-fault reads per run | Output |
|---|---|---:|---|---:|---:|---|---|
| A, control (no ballast) | 3 | 40-48 GB | **14.1** (15.3, 14.1, 13.4) | 56.5 | 24.9 | 0.0 (warm), 4.6, 5.2 GiB | `69195351a9d6b52a` |
| B, pressure (ballast to about 8 GB free) | 3 | 8-9 GB | **5.6** (5.6, 4.8, 7.1) | 6.4 | 54.8 | 17.3, 20.2, 18.5 GiB | `69195351a9d6b52a` |

Ratio B/A: decode **0.40**, prefill **0.11**. All six generated texts are identical (the paging does not change the numerics). Attempt 1, invalid for condition B (constant-byte ballast compressed away; sampler and hash defects), is recorded in the plan, section 7.1.

Verdict against the rules fixed before the runs: H0 (B at least 0.90 of A) is rejected. H1 needs the hard-fault reads to be the dominant cost. The read volume is consistent with that (about 18-20 GiB in 50-60 s, roughly 350-400 MB/s on average, against a drive rated at about 560 MB/s sequential), but the drive's random-fault throughput was not measured, so dominance is not shown. Not enough for a residency prototype yet.

What this does show: when the mapped expert working set does not fit in available RAM, CPU MoE decode on this machine runs at about 0.4x, and prefill at about 0.1x. What it does not show: whether a process-managed residency policy would read fewer bytes. That needs routing traces under the same budget (hit rate of a pinned set versus the OS's LRU), which were not collected.

## Opt-in and no-go items

- **CUDA:** no numbers. All CUDA rows are from prior hardware (†) and are historical.
- **Scratch arena:** no-go (2026-10-10). See the sweeps section.
- **Q8_K decode:** used automatically with one CPU thread (2026-10-30). See the sweeps section.
