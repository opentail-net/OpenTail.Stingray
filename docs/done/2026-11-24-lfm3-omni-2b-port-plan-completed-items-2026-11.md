# Completed items split from 2026-11-24-lfm3-omni-2b-port-plan.md (2026-11-25)

Finished items moved verbatim out of [../2-coverage/2026-11-24-lfm3-omni-2b-port-plan.md](../2-coverage/2026-11-24-lfm3-omni-2b-port-plan.md). Open items stay there. Each block notes the section it came from.

> from: ## Not verified (blocking admission)

- [x] **Independent reference (resolved 2026-11-25).** The llama.cpp master build at `33aeb17` registers `lfm3`. Golden: `lfm3-omni-2b.golden.json`, teacher-forced, 16 tokens requested, chat prompt with the `<|startoftext|>` BOS (scratchpad, not the repo). `stingray admit-arch --golden`: **PASS (Exact)**, 11/11 matched, 11 confident. Stepwise and prefill agree (max |diff| 0). Our tokenizer agrees with llama-tokenize on the prompt. Not yet: timed runs, more prompts, a prompt past the 256-token window, a longer continuation, and the STATUS row (owner decision).

> from: ## Not verified (blocking admission)

- [x] **Licence** (`other`, LFM 1.0): not a gate (owner, 2026-11-24: inference only, nothing shipped).


## Sub-items completed under open parents (2026-11-25)

> from: - [ ] **Performance and DRY pass** (CLAUDE.md rule 7): not yet done. Three timed runs are recorded above; a me
  - [x] DRY check, 2026-11-25 (commit c714ec54): the three-field override LFM3 and Spark-X both applied on `GatedSwa` (RoPE on every layer, no embedding scale, per-head attention gate) is now `FamilyModelSemantics.GatedSwaPerHeadRopeOnAllLayers`. Field values are unchanged. Verified: Tests.ForwardPass.Fast `Spark25SyntheticTests` 2/2, Tests.Core `ArchitectureDescriptorContractTests` 11/11 (Debug). Real-weight LFM3 golden re-run after this change: done 2026-11-25 on the Debug CLI at HEAD. Default golden 11/11 Exact (11 confident), long golden 10/10 Exact (9 confident). Same counts as before the refactor, so the numerics are unchanged.


## Items completed at the top level, leftover from the first pass (2026-11-25)

> from: ## Not verified (blocking admission)

- [x] **STATUS.md row**: added 2026-11-25 (`docs/STATUS.md`, text model families group).

