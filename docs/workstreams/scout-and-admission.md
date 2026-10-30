# Workstream: checkpoint scout and architecture admission

**Owner:** [../3-product-and-runtime/2026-10-09-checkpoint-scout-and-ai-admission-plan.md](../3-product-and-runtime/2026-10-09-checkpoint-scout-and-ai-admission-plan.md). **Playbook:** [../reference/architecture-admission-agent-playbook.md](../reference/architecture-admission-agent-playbook.md). **Last verified:** 2026-10-30.

## Correct state

- **Phases 0 to 2 are done** per the plan's own status: the read-only local GGUF scout (`src/OpenTail.Stingray.Cli/Scout/ScoutCommand.cs`), the feature rules (`ScoutFeatures`, 29 synthetic tests) and the relatives ranking (`Engine/Scout/ArchSignature.cs`).
- **Phase 3 (memory-gated pretests)** is implemented as a script, `scripts/scout-pretest.ps1`, not in C#. The plan lists its behaviour; the script is advisory and makes no admission decision.
- **Phase 5, the agent playbook,** is written (the playbook above). Its calibration half is deferred.
- **Phase 4, Hugging Face repository discovery,** is deferred by decision (plan section 8.1).
- Scout is advisory evidence, not an admission authority (CLAUDE.md rule 15). Admission still needs an independent reference and a golden.

## Open items

- [ ] Phase 4: Hugging Face repository discovery (deferred, plan section 8.1)
- [ ] Phase 5b: calibration of the ranking and feature rules, with misses recorded in the plan's calibration notes (deferred)

## Cadence

- Review when an unfamiliar architecture is triaged, and whenever a scout ranking turns out wrong (record the miss).
- Re-read at least every two weeks while the deferred phases stay open.
- Close when Phase 4 and 5b are either built or written off in the plan.
