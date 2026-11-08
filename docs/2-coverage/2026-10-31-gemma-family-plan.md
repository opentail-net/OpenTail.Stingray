# Gemma family coverage

> Completed items moved to [../done/2026-10-31-gemma-family-plan-completed-items-2026-10.md](../done/2026-10-31-gemma-family-plan-completed-items-2026-10.md) on 2026-11-01. This file keeps only the open items; a finished item that still has an open child stays here with it.

> The state of each Gemma checkpoint lives only in its STATUS row ([../STATUS.md](../STATUS.md)) and in the model card ([../models/gemma.md](../models/gemma.md)). This file does not repeat them.

**Owner:** this file (moved from docs/workstreams on 2026-10-31). **Last verified:** 2026-11-08.

## Current state (pointers only)

- **Gemma 2 (`gemma2`), Gemma 1 (`gemma`), Gemma 3, Gemma 4 E4B, Gemma 4 26B-A4B:** see their STATUS rows.
- **Fix records:** Gemma 2 softcapping and semantics fix (`4a6027a6`, sanity check `76def766`, [../done/2026-10-11-llm-reply-sanity-check.md](../done/2026-10-11-llm-reply-sanity-check.md)). Gemma 1 `FamilyModelSemantics.Gemma1` (2026-11-01, see the done file above).
- **Backend limitation (Gemma 2):** the descriptor's `BackendLimitation` (`GemmaArchitectures.cs`) is read by the planner, which reports Vulkan/CUDA as refused for this architecture (`StaticPlanCommand.cs`), and by the GPU perplexity command, which refuses (`PerplexityCommand.cs`). Not confirmed: that the engine's own load path refuses it. Checked 2026-11-08.

## Open items

- [ ] Gemma 2 beyond the 2B file: the record has no evidence for other sizes

## Decisions for the owner

- None open. (Gemma 1 was downloaded and checked 2026-11-01; Gemma 2 GPU soft-capping was decided against by the owner 2026-11-01.)

## Cadence

- Review when Gemma 1 is attempted, and whenever a softcap kernel lands on Vulkan or CUDA.
- Re-read at least every two weeks while any item is open.
- Close when Gemma 2's backend gaps are either built or stated as permanent in STATUS.
