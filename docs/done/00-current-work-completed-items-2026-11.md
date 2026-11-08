# Current work: outstanding items, quickest first - completed items

Ticked items moved from [docs/00-current-work.md](../00-current-work.md) on 2026-11-03. Each keeps its section heading and nested lines.

## 13. MoE variants of the recurrent families

  - [x] Admit Granite 4.0-H tiny/small (MoE) with PPL parity. Small DONE 2026-09-28 (top-k renormalisation fix; +1.2% residual at -c 2048 logged).

## 4. Performance

- [x] 2. **Stale: re-measured 2026-11-01 at 0.97x** (61.45 against 63.45 t/s, medians of 4 interleaved rounds, Q6_K, 550 tokens, CPU; detail in [plan 6 D2](../00-current-work.md#open-work-as-of-2026-11-03)). Qwen3.6-35B-A3B prefill was 0.63x of llama.cpp (Phase 8 of
   [done/2026-09-25-hf-top-downloads-coverage-plan.md](2026-09-25-hf-top-downloads-coverage-plan.md)).

## 1. Transcription `model` field (completed 2026-11-05)

  - [x] Transcription `model` selects an installed ASR pipeline, and an unknown name returns 400 that names the field (validation: PR 143; selection through `IWhisperPipelineProvider.Create(model)`: PR 165). Tests in `TranscriptionModelFieldTests` (13 pass).

## Evidence for the engine (added 2026-11-05, from the external review)

- [x] **E1. Long-running inference stability.** Done 2026-11-05: `LongRunSoakTests` (opt-in, `STINGRAY_SOAK=1`) passed on real Qwen3-0.6B Q8_0 on CPU, 30 greedy repeats in one process, identical output, working set flat after warm-up. Repeated prompts, long contexts and several sessions, looking for memory growth, stale state, and KV-cache corruption over time. Done when a soak run at temperature 0 gives identical output on every repeat, and resident memory is flat after warm-up.
  First step: write a soak test on the fake engine in `tests/OpenTail.Stingray.Tests.Core` that loops N prompts and asserts identical output and flat allocations. Then the same loop against a real small model on CPU.
- [x] **E2. CPU performance regression gate.** Done 2026-11-05, as a local gate: `scripts/perf-gate.ps1` measures prefill and decode for qwen2.5-0.5b and 1.5b (median of 3) and fails on a drop of more than 15% against `scripts/perf-baseline.json`, per machine. Hosted CI has no model files, so it cannot run this gate; run it before a release or after a kernel change. Checked: unchanged code passes (exit 0); a baseline inflated 2x fails all four metrics (exit 1).

## Owner action (does not block the list above)

- [x] **Run once:** `pwsh scripts/qualify-tasks.ps1`. Run 2026-11-06: chat, speak and transcribe all PASS with no licence flag (the default speak model is Kokoro, Apache-2.0; the Piper licence applies only if Piper is chosen with `-m piper-lessac`).

## Moved 2026-11-08 from 00-current-work.md

  - [x] *(stale, closed 2026-11-03: `lfm2moe` was admitted 2026-10-01; the batched drift was flash-64 attention's online softmax, not a batched-path bug; see STATUS and [bugstofix item 12](done/bugstofix-resolved-2026-10.md))* Admit LFM2-MoE (`lfm2moe`) with PPL parity. BLOCKED 2026-09-28 (timeboxed): PPL +6.3% and per-token vs batched disagree by 9%; see bugstofix.
