# README open items — fixable on this PC (plan, 2026-09-26)

The README status-matrix rows that are not fully green, minus what this machine cannot fix (no
checkpoint, too large for 64 GB RAM, upstream GGUF bug): DeepSeek-V3.2, DeepSeek-V4, Llama 4
vision, MobileNetV5 and MiMo-VL are out of scope.

Order: quickest expected fix first. The estimates are guesses; the log below records what
actually happened. Every fix follows CLAUDE.md: real weights, an independent reference (llama.cpp
/ vendored C++ / recorded reference outputs), timed test runs, and the README row updated with the
dated evidence in the same pass.

| # | Item | Why it is where it is | Expected size |
|---|---|---|---|
| 1 | SD3/3.5 (GPU/Vulkan) row: Status/Confidence columns swapped | table edit | minutes |
| 2 | Stable Audio 3 Small Music + Medium still 🟡 though the rows say they match the reference | re-grade from the existing evidence, or name the open gap | minutes |
| 3 | Nemotron-Nano-12B-v2-VL: `nemotron_h` text backbone crashes (per-layer `feed_forward_length` array; 0 = pure Mamba layer) | root cause already known | small |
| 4 | dots.ocr: decode stops after one token | suspected prompt-format mismatch | small–medium |
| 5 | Kimi-VL + YoutuVL: split `attn_k_b` / `attn_v_b` MLA layout not read | known math (llama.cpp deepseek2 absorption path); unblocks two models | medium |
| 6 | PaddleOCR-VL: degenerate output | text architecture (`paddleocr`) has no validated forward pass | medium |
| 7 | DeepSeek-OCR / OCR2: garbled | text architecture (`deepseek2-ocr`) has no validated forward pass | medium |
| 8 | Step3-VL: garbled | unvalidated architecture on a Q2_K checkpoint | medium–large |
| 9 | IBM Granite Vision 3.2 / 4.0: output not image-grounded | investigation; 3.2 via LlavaAdapter, 4.0 via QFormer projector | large |
| 10 | CosyVoice 2: audio only partly right | investigation | large |
| 11 | Stable Audio 3 Small SFX: darker than the reference | investigation | large |
| 12 | Chronos-Bolt / Chronos-2: no numeric reference | needs an independent oracle without new Python reference scripts | large |
| 13 | 🟢-but-⚪ diffusion rows (HunyuanVideo, FLUX.2, Qwen Image, SD3 CPU): not independently verified | needs reference outputs (vendored C++ / recorded) | large |

## Log
- 2026-09-26 #1 DONE: SD3/3.5 (GPU/Vulkan) row had Status and Confidence swapped; now
  🟢 (closed 2026-09-24) / ⚪, matching the CPU row.
- 2026-09-26 #2 RE-SCOPED, not a re-grade: Stable Audio 3 Small Music and Medium are 🟡 for a real,
  named gap. At CFG 1 (conditional-only) ours is much darker than the reference; at the official
  CFG 7 they match. Item 11 (Small SFX "darker than the reference") is the same symptom, so #2 and
  #11 become one investigation, placed with the large items after #10.
