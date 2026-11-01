# Gemma family coverage

> Completed items moved to [../done/2026-10-31-gemma-family-plan-completed-items-2026-10.md](../done/2026-10-31-gemma-family-plan-completed-items-2026-10.md) on 2026-11-01. This file keeps only the open items; a finished item that still has an open child stays here with it.

> This file summarises its owner plan. If the two disagree, the owner plan is authoritative; correct this file.

**Owner:** this file (moved from docs/workstreams on 2026-10-31). **Last verified:** 2026-10-30.
**Evidence:** [../done/2026-10-11-llm-reply-sanity-check.md](../done/2026-10-11-llm-reply-sanity-check.md) (the Gemma 2 fix record). **Card:** [../models/gemma.md](../models/gemma.md). **Status row:** [../STATUS.md](../STATUS.md).

## Correct state

- **Gemma 2 (`gemma2`):** on the admitted list since the first release. Before the 2026-10-11 fix its output was incoherent, and the sanity check (`76def766`) called that "not a recent regression". The descriptor had no model semantics. The fix (`4a6027a6`) added the embedding scale, GeGLU, alternating sliding-window layers, and attention and final logit softcapping. Checked against llama-server on gemma-2-2b-it Q4_K_M: two cases exact (48/48), one near-tie (46/48, both misses inside the reference's own margin). **CPU only.** Vulkan and CUDA attention have no softcap, so the descriptor records a `BackendLimitation` for them (checked 2026-10-31; enforcement at load was not confirmed).
- **Gemma 1 (`gemma`):** has an architecture descriptor (`GemmaArchitectures.cs`) and was on the admitted list from the registry migration (`71619b9d`). It never worked before 2026-11-01: every position predicted BOS until `FamilyModelSemantics.Gemma1` was added (see the done item below). It now has a golden (22/24 near-tie, gemma-1.1-2b-it Q8_0) and a STATUS row; CPU only.
- **Gemma 3 and Gemma 4 E4B:** STATUS rows exist with dated evidence.

## Open items

- [ ] Gemma 2 beyond the 2B file: the record has no evidence for other sizes

## Decisions for the owner

- None open. (Gemma 1 was downloaded and checked 2026-11-01; Gemma 2 GPU soft-capping was decided against by the owner 2026-11-01.)

## Cadence

- Review when Gemma 1 is attempted, and whenever a softcap kernel lands on Vulkan or CUDA.
- Re-read at least every two weeks while any item is open.
- Close when Gemma 1 has a recorded verdict and Gemma 2's backend gaps are either built or stated as permanent in STATUS.
