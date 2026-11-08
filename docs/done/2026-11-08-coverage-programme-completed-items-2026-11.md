# Coverage programme (2026-11-07) - completed items

Ticked items moved from [2026-11-07-coverage-completion-programme-plan.md](../2-coverage/2026-11-07-coverage-completion-programme-plan.md) on 2026-11-08. Only fully closed items move; each keeps its text. Open items stay in the programme plan.

## Checklist (state as of 2026-11-08)

- [x] Phase 0: coverage baseline current (`ported-families-todo.md` + the table above)
- [x] Phase 1, step 1: reproduce the Qwen4Exp garbled-text claim with a fixed prompt (2026-11-07; claim was real)
- [x] Phase 1 root cause and fix: conv1d weight transpose; regression test `Layer0GdnMixerOutput_MatchesLlamaCppEvalCallback_AfterConv1dTransposeFix`
- [x] Phase 1 side fix: `TruncateTo` contract violation (`TruncateTo_RejectsIntermediateLength_OnlyAcceptsZeroOrCurrent`)
- [x] Phase 2A: `glm-dsa` metadata-only real-weight check (`GlmDsaRealWeightsSmokeTests`, 2026-11-07)
- [x] Phase 2A: indexer top_k-shortcut optimisation (output unchanged, 2026-11-07)
- [x] Phase 2A: synthetic-correctness checklist complete 2026-11-08 (MLA absorption, DSA scoring, Hadamard rotation, sigmoid MoE weighting, shared-expert contribution); two bugs fixed, three already correct
- [x] Phase 3 side fix: `deepseek4` MoE hash-routing weight and selection-bias ordering (2026-11-07/08)

## Recommended starting point

- [x] 2. Phase 2A's remaining synthetic-level correctness checklist (MLA absorption, indexer scoring,
   Hadamard rotation, sigmoid MoE routing, shared-expert contribution against the llama.cpp source) --
   done 2026-11-08 (see Checklist above).
