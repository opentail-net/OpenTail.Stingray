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

**CPU prefill context scaling — Performance Check: 2026-08 (Q8 prefill on, tiled KV, iter 33):**

| Prompt tokens | 267 | 773 | 1621 | 3218 |
|---|---:|---:|---:|---:|
| Prefill OT (t/s) | ~50 t/s | ~49 t/s | ~49 t/s | ~42 t/s |
| Prefill llama.cpp (t/s) | 205 t/s | — | — | — |

**CPU decode context scaling — Performance Check: 2026-08 (contiguous KV score pass, iter 35):**

| Prompt tokens | 267 | 773 | 1621 | 3218 |
|---|---:|---:|---:|---:|
| Decode OT (t/s) | 26.3 t/s | 21.0 t/s | 15.0 t/s | 9.5 t/s |

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

## Qwen3 family

| Model | Scenario | Backend | C# (OT, t/s) | C++ (llama.cpp, t/s) | Ratio | Performance Check | Source |
|---|---|---|---:|---:|---:|---|---|
| Qwen3-0.6B Q8_0 | decode (short ctx) | CPU | 47.4 t/s | — | — | 2026-08-07 | cpu-performance-baseline.md |
| Qwen3-8B Q4_K_M | decode (short ctx) | CPU | 6.8 t/s | — | — | 2026-08 | cpu-speculative-decoding-findings.md |
| Qwen3-Coder 30B-A3B Q4_K_M | prefill | CUDA (†) | 102.6 t/s | — | — | 2026-06-16 | README history 0c171ed |
| Qwen3-Coder 30B-A3B Q4_K_M | decode | CUDA (†) | 28.0 t/s | — | — | 2026-06-16 | README history 0c171ed |
| Qwen3-Coder 30B-A3B Q4_K_M | decode | CPU | 22.4 t/s | — | — | 2026-06-16 | README history 0c171ed |
| Qwen3-Coder 30B-A3B Q4_K_M | decode (`--tq`) | CPU | 22.6 t/s | — | — | 2026-06-16 | README history 0c171ed |
| Qwen3.6-35B-A3B Q4_K_M | prefill | CUDA (†) | 475.4 t/s | — | — | 2026-06-16 | README history 0c171ed |
| Qwen3.6-35B-A3B Q4_K_M | decode | CUDA (†) | 24.5 t/s | — | — | 2026-06-16 | README history 0c171ed |
| Qwen3.6-35B-A3B Q4_K_M | decode | Vulkan (†) | 22.8 t/s | — | — | 2026-06-16 | README history 0c171ed |
| Qwen3.6-35B-A3B Q4_K_M | decode | CPU | 9.3 t/s | — | — | 2026-06-16 | README history 0c171ed |
| Qwen3.6-35B-A3B-MTP Q4_K_M | prefill | CUDA (†) | 480.2 t/s | — | — | 2026-06-16 | README history 0c171ed |
| Qwen3.6-35B-A3B-MTP Q4_K_M | decode (`--no-thinking`) | CUDA (†) | 33.3 t/s | ~41 t/s (est.) | **~0.81x** | 2026-06-16 | README: "~80% of llama.cpp tg128" |
| Qwen3.6-27B-MTP Q4_K_M | prefill | CUDA (†) | 22.0 t/s | — | — | 2026-06-16 | README history 0c171ed |
| Qwen3.6-27B-MTP Q4_K_M | decode (`--no-thinking`) | CUDA (†) | 12.3 t/s | — | — | 2026-06-16 | README history 0c171ed |
| Qwen3.6-27B-MTP Q4_K_M | decode (`--no-thinking`) | CPU | 3.6 t/s | — | — | 2026-06-16 | README history 0c171ed |
| Carnice 35B-A3B-MTP (APEX) | prefill (`--no-thinking`) | CUDA (†) | 522.0 t/s | — | — | 2026-06-16 | README history 0c171ed |
| Carnice 35B-A3B-MTP (APEX) | decode (`--no-thinking`) | CUDA (†) | 26.5 t/s | — | — | 2026-06-16 | README history 0c171ed |

> **Qwen3-8B decode:** 6.8 t/s = 34.2 GB/s = 93% of the measured 36.8 GB/s DRAM ceiling.
> Speculative decoding is a confirmed −37% loss on this CPU — see Speculative Decoding section.

---

## OLMoE family

| Model | Scenario | Backend | C# (OT, t/s) | C++ (llama.cpp, t/s) | Ratio | Performance Check | Source |
|---|---|---|---:|---:|---:|---|---|
| OLMoE-1B-7B Q4_K_M | prefill | CPU | 105.6 t/s | — | — | 2026-08-07 | cpu-performance-baseline.md |
| OLMoE-1B-7B Q4_K_M | decode | CPU | 28.2 t/s | — | — | 2026-08-07 | cpu-performance-baseline.md |

> Decode measured over ~7 tokens (early EOS); treat as approximate.

---

## Gemma family

| Model | Scenario | Backend | C# (OT, t/s) | C++ (llama.cpp, t/s) | Ratio | Performance Check | Source |
|---|---|---|---:|---:|---:|---|---|
| Gemma-4-12B Q4_0 | prefill | CPU | 3.8 t/s | — | — | 2026-08-07 | cpu-performance-baseline.md |
| Gemma-4-12B Q4_0 | decode | CPU | 3.7 t/s | — | — | 2026-08-07 | cpu-performance-baseline.md |
| Gemma4 E4B QAT Q4_0 | prefill | CUDA (†) | 3666 t/s | — | — | 2026-06-16 | README history 0c171ed |
| Gemma4 E4B QAT Q4_0 | decode | CUDA (†) | 100.4 t/s | — | — | 2026-06-16 | README history 0c171ed |
| Gemma4 E4B QAT Q4_0 | prefill | Vulkan (†) | 35 t/s | — | — | 2026-06-22 | README history 0c171ed |
| Gemma4 E4B QAT Q4_0 | decode | Vulkan (†) | 39.5 t/s | — | — | 2026-06-22 | README history 0c171ed |
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

---

## Speculative Decoding (CPU)

| Target | Draft | Scenario | C# (OT, t/s) | C++ (ref, t/s) | Ratio | Acceptance rate | Performance Check | Source |
|---|---|---|---:|---:|---:|---|---|---|
| Qwen3-8B Q4_K_M | Qwen3-0.6B Q8_0 | decode, draft-n 4 | 4.3 t/s (−37%) | — | — | 62% | 2026-08 | cpu-speculative-decoding-findings.md |

> Speculation is a **confirmed loss** on this hardware. Q4_K dot is ~87% compute-bound (not
> bandwidth-bound); verifying k tokens costs ~k× compute regardless of dispatch. VNNI (`vpdpbusd`,
> Zen 4+) is the prerequisite for speculation to become viable.

---

## TTS / Audio Synthesis

| Pipeline | Scenario | Backend | Wall-Clock | RTF (C#) | C++ (ref) | Ratio | Performance Check | Confirmed Working |
|---|---|---|---:|---:|---:|---:|---|---|
| Piper lessac-medium (ONNX) | text → 2.45s audio | CPU | 0.47s | **0.19×** | — | — | 2026-09-09 | 2026-09-03 👂 |
| MMS-TTS eng (VITS) | text → 3.65s audio | CPU | 1.38s | **0.38×** | — | — | 2026-09-09 | 2026-08-30 🔬 |
| Kokoro-82M (Q8_0 GGUF) | text → 2.93s audio | CPU | 2.60s | **0.89×** | — | — | 2026-09-09 | 2026-09-03 👂 |
| MeloTTS zh_en (ONNX) | text → 2.74s audio | CPU | 3.11s | **1.14×** | — | — | 2026-09-09 | 2026-09-03 👂 |
| QwenTTS 0.6B (Q8_0 GGUF) | text → 2.16s audio | CPU | 6.59s | **3.05×** | 4.55× | <span style="color:#16a34a">**1.49x**</span> | 2026-09-09 | 2026-08-29 👂 |
| CosyVoice3 (DiT + HiFT) | text → 3.00s audio | CPU | 17.20s | **5.73×** | 9.76× | <span style="color:#16a34a">**1.70x**</span> | 2026-09-09 | 2026-09-06 👂 |
| Chatterbox Turbo (Q4_K) | text → 2.52s audio | CPU | 13.72s | **5.45×** | 5.22× | **0.96x** | 2026-09-09 | 2026-08-30 🔬 |
| Parler-TTS Mini v1 | text → 2.81s audio | CPU | 17.36s | **6.18×** | — | — | 2026-09-09 | 2026-08-28 👂 |
| FishSpeech S2 Pro (Q4_K) | text → 3.44s audio | CPU | 28.46s | **8.28×** | 11.45× | <span style="color:#16a34a">**1.38x**</span> | 2026-09-09 | 2026-08-29 👂 |
| F5-TTS Base (DiT) | text → 2.77s audio | CPU | 27.25s | **9.82×** | — | — | 2026-09-09 | 2026-08-28 👂 |
| F5-TTS Base (Paragraph) | text → 14.5s audio | CPU | 10.20s | **0.70×** | — | — | 2026-09-09 | 2026-08-28 👂 |

> RTF < 1.0x = faster than real-time. Piper (0.19x = 5.2× real-time), MMS-TTS (0.38x = 2.6× real-time), and Kokoro (0.89x) are faster than real-time on CPU.
> Autoregressive pipelines (QwenTTS, CosyVoice3, Chatterbox, Parler, FishSpeech) are compute-bound on CPU; GPU dispatch is expected to be the largest speedup.
> **Confirmed Working:** 🔬 = Golden-verified against reference; 👂 = Confirmed working by ear / transcription.
> Harness: `scripts/bench-audio.ps1` (`tests/OpenTail.Stingray.Tests.Audio/TtsPerformanceBaselineDebugTest.cs`).

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

> **TTFA (Time-To-First-Audio):** Latency from prompt ingestion to the first playable audio chunk emitted.
> **Confirmed Working:** 🔬 = Golden-verified against reference; 👂 = Confirmed working by ear / transcription.
> Harness: `scripts/bench-audio.ps1 -Suite Streaming`.

---

## ASR / Speech Recognition (Whisper GGUF)

| Model | Scenario | Backend | Wall-Clock | RTF (C#) | C++ (whisper.cpp) | Ratio | Performance Check | Source |
|---|---|---|---:|---:|---:|---:|---|---|
| Whisper Base (39M) | 12s audio transcribe | CPU | 0.84s | **0.070x** | 0.060x | **0.86x** | 2026-09-09 | scripts/bench-audio.ps1 |
| Whisper Small (244M) | 12s audio transcribe | CPU | 2.42s | **0.202x** | 0.180x | **0.89x** | 2026-09-09 | scripts/bench-audio.ps1 |
| Whisper Medium (769M) | 12s audio transcribe | CPU | 6.71s | **0.560x** | 0.582x | <span style="color:#16a34a">**1.04x**</span> | 2026-09-09 | scripts/bench-audio.ps1 |
| Whisper Large-v3 (1.5B) | 12s audio transcribe | CPU | 11.32s | **0.943x** | 1.200x | <span style="color:#16a34a">**1.27x**</span> | 2026-09-09 | scripts/bench-audio.ps1 |

> RTF < 1.0x = faster than real-time. Whisper Base runs at **14.3x real-time** speed on CPU; Small at **5.0x real-time**; Medium at **1.79x real-time**; Large-v3 at **1.06x real-time** (faster than real-time on CPU).
> Harness: `scripts/bench-audio.ps1` (`tests/OpenTail.Stingray.Tests.Audio/WhisperFullPipelinePerfBenchTests.cs`).

---

## Vision Encoder (CPU)

| Component | Scenario | Backend | C# result | C++ reference | Ratio | Performance Check | Source |
|---|---|---|---|---|---:|---|---|
| VisionOps.Attention / AttentionGqa | 1024-tok / 16-head ViT-L | CPU | >1.2× over scalar | — | — | 2026-08-20 | perf-loop-project-review-progress.md |

> Scalar reference kept in `VisionOpsBenchmarkTests.cs` as a permanent regression baseline.

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
| SmolLM2-1.7B | prefill @long ctx | CPU | llama.cpp long-ctx not measured | Close the 0.33x prefill gap at scale |
| Qwen3-8B | prefill | CPU | Not yet measured | Unknown how far prefill trails |
| Qwen3-8B | decode | CPU | llama.cpp ref not run on this box | Known ~93% DRAM; expect ~parity |
| All models | any | CUDA | No CUDA device on dev machine | Direct MMQ for Q6K/Q5K prefill |
| All models | any | Vulkan iGPU | No llama.cpp Vulkan reference | Discrete GPU needed for real comparison |
| Gemma-4-12B | prefill (batched) | CPU | `perLayerHdUnsupported` gate blocks it | ~5.7× penalty remains once gate is lifted |
| TTS Pipelines (all) | full synthesis | GPU | Pipelines not wired to GPU yet | Expected to be the largest TTS win |
| CPU KV cache | bf16/q8 dtype | CPU | `PagedKvCache` hard-wired fp32 | Vulkan showed +57% decode at no quality cost |

---

*Last updated: 2026-09-09.
Source documents: `docs/done/perf-loop-progress.md`, `docs/cpu-performance-baseline.md`,
`docs/tts-performance-baseline-and-plan.md`, `docs/done/cpu-speculative-decoding-findings.md`,
`docs/done/vulkan-backend-evidence.md`, `docs/done/gpu-review-log.md`, `GR_performance.md`,
`docs/perf-loop-project-review-progress.md`, `scripts/bench-audio.ps1`, `scripts/bench-cpp.ps1`, `README.md` git history commit `0c171ed`.*

*Reproducibility: All runs can be replicated with `.\scripts\bench-cpp.ps1 -Suite Tts` or `.\scripts\bench-cpp.ps1 -Suite All`.*
