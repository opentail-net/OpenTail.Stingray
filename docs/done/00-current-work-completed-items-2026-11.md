# Current work: outstanding items, quickest first - completed items

Ticked items moved from [docs/00-current-work.md](../00-current-work.md) on 2026-11-03. Each keeps its section heading and nested lines.

## 13. MoE variants of the recurrent families

  - [x] Admit Granite 4.0-H tiny/small (MoE) with PPL parity. Small DONE 2026-09-28 (top-k renormalisation fix; +1.2% residual at -c 2048 logged).

## 4. Performance

- [x] 2. **Stale: re-measured 2026-11-01 at 0.97x** (61.45 against 63.45 t/s, medians of 4 interleaved rounds, Q6_K, 550 tokens, CPU; detail in [plan 6 D2](../00-current-work.md#open-work-as-of-2026-11-03)). Qwen3.6-35B-A3B prefill was 0.63x of llama.cpp (Phase 8 of
   [done/2026-09-25-hf-top-downloads-coverage-plan.md](2026-09-25-hf-top-downloads-coverage-plan.md)).

## 1. Transcription `model` field (completed 2026-11-05)

  - [x] Transcription `model` selects an installed ASR pipeline, and an unknown name returns 400 that names the field (validation: PR 143; selection through `IWhisperPipelineProvider.Create(model)`: PR 165). Tests in `TranscriptionModelFieldTests` (13 pass).
