# Workstream: multi-model runtime, residency and scheduling

**Owner:** [../3-product-and-runtime/032-multi-model-inference-runtime-plan.md](../3-product-and-runtime/032-multi-model-inference-runtime-plan.md). **Completed slices:** [../done/032-multi-model-inference-runtime-plan-completed-items-2026-10.md](../done/032-multi-model-inference-runtime-plan-completed-items-2026-10.md). **Last verified:** 2026-10-30.

## Correct state (from the plan's status section)

- **Implemented:** Phases 1, 2 and 4. Phase 7 for the stateless request surface (chat completions, messages, responses, model listing).
- **Phase 3, residency:** 9 slices, covering host and accelerator admission, eviction and observability. Accelerator accounting is in progress.
- **Phase 5:** the no-lock half is proven with real models.
- **Phase 6, fair scheduling:** 4 slices (the queue step of the eviction hierarchy, bounded, observable, fair among queued waiters, and immune to a fresh caller jumping the queue). **Execution-level service quantum and starvation protection are still open.**
- **Phase 7, durable restore:** deferred. A session does not survive a restart in multi-model mode.

## Open items

- [ ] Phase 3: accelerator accounting (in progress)
- [ ] Phase 6: service quantum and starvation protection at execution level
- [ ] Phase 7: durable session restore across a restart in multi-model mode (deferred)
- [ ] Phase 7: expose model identity through `IInferenceService` (plan line 598)
- [ ] Phase 6: per-model pending queues, request age and service quantum as a full design (plan line 483)
- [ ] Phase 3: resident tracking, memory estimates and `IResourceBudget` (plan line 349), with the open accounting above

## Cadence

- Review after each slice lands, and before any change to scheduling or residency defaults.
- Re-read at least every two weeks while any item is open.
- Close when Phase 6 has the service quantum, Phase 7 restore is built or written off, and the plan's status matches the code.
