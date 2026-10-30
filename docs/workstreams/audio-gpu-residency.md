# Workstream: audio GPU residency

> This file summarises its owner plans. If the two disagree, the owner plan is authoritative; correct this file.

**Owners:** [../4-performance/audio/080-tts-asr-gpu-residency-checklist.md](../4-performance/audio/080-tts-asr-gpu-residency-checklist.md) (TTS and ASR transformer stack, 28 open) and [../4-performance/audio/079-minimax-music3-gpu-residency-plan.md](../4-performance/audio/079-minimax-music3-gpu-residency-plan.md) (MiniMax-Music3, 7 open). **Last verified:** 2026-10-30 (file-level only; the items were not re-run).

## Correct state

- **The goal:** keep the transformer stacks resident on the GPU, one engine at a time.
- **Hardware limit:** this machine has an integrated Radeon only. CLAUDE.md rule 13 applies: an iGPU result cannot show whether a discrete GPU would be faster. Residency results here describe this APU.
- **Measurement:** both plans carry the "one thing at a time" rule, which now lives in [../reference/performance-measurement-rules.md](../reference/performance-measurement-rules.md).
- **079's own status (2026-09-27):** the goal is still open. The 2026-10-30 check did not re-run its items.

## Open items

- [ ] 080: 28 open items, phase by phase (see the file)
- [ ] 079: 7 open items (see the file)
- [ ] Any residency speed claim needs a discrete-GPU run or an explicit iGPU-only caveat

## Cadence

- Review when an audio engine's residency path changes, or when a discrete GPU is available.
- Re-read at least every two weeks while any item is open.
- Close when each engine in 080 is either resident and measured, or written off with the reason.
