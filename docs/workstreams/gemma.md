# Workstream: Gemma

**Owner:** this file. **Last verified:** 2026-10-30.
**Evidence:** [../done/2026-10-11-llm-reply-sanity-check.md](../done/2026-10-11-llm-reply-sanity-check.md) (the Gemma 2 fix record). **Card:** [../models/gemma.md](../models/gemma.md). **Status row:** [../STATUS.md](../STATUS.md).

## Correct state

- **Gemma 2 (`gemma2`):** on the admitted list since the first release. Before the 2026-10-11 fix its output was incoherent, and the sanity check (`76def766`) called that "not a recent regression". The descriptor had no model semantics. The fix (`4a6027a6`) added the embedding scale, GeGLU, alternating sliding-window layers, and attention and final logit softcapping. Checked against llama-server on gemma-2-2b-it Q4_K_M: two cases exact (48/48), one near-tie (46/48, both misses inside the reference's own margin). **CPU only.** Vulkan and CUDA attention have no softcap, so the descriptor refuses them.
- **Gemma 1 (`gemma`):** has an architecture descriptor (`GemmaArchitectures.cs`). No golden, no local checkpoint, no STATUS row. It was on the admitted list from the registry migration (`71619b9d`). Whether it ever worked is not recorded.
- **Gemma 3 and Gemma 4 E4B:** STATUS rows exist with dated evidence.

## Open items

- [ ] Gemma 1: obtain a checkpoint (download is authorised for `models/_models/`), capture a golden against llama-server, and add a STATUS row or a dated "not verified" note
- [ ] Gemma 2 on Vulkan and CUDA: needs an attention-logit softcap kernel there (currently refused, not supported)
- [ ] Gemma 2 beyond the 2B file: the record has no evidence for other sizes

## Decisions for the owner

- Whether Gemma 1 is worth a checkpoint download and a golden, given it is an older family.

## Cadence

- Review when Gemma 1 is attempted, and whenever a softcap kernel lands on Vulkan or CUDA.
- Re-read at least every two weeks while any item is open.
- Close when Gemma 1 has a recorded verdict and Gemma 2's backend gaps are either built or stated as permanent in STATUS.
