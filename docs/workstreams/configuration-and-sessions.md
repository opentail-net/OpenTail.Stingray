# Workstream: configuration and session forking

**Owners:** [../3-product-and-runtime/04-quality-of-life-improvements-plan.md](../3-product-and-runtime/04-quality-of-life-improvements-plan.md) (configuration, observability gaps, compatibility) and [../3-product-and-runtime/010-forward-pass-context-isolation-for-session-forking-plan.md](../3-product-and-runtime/010-forward-pass-context-isolation-for-session-forking-plan.md) (session forking). **Last verified:** 2026-10-30.

## Correct state

- **Configuration (plan 04):** item 1 is done (the inventories are name-complete and classified row by row, dated 2026-08-15). Items 2 to 6 are open. The plan's status line says the planning and diagnostics baseline shipped.
- **Session forking (plan 010):** planning only, as its status section says. Checked 2026-10-30: `IForwardPass.CreateContext()` still returns `this` (`src/OpenTail.Stingray.Core/IForwardPass.cs:14`), so forked sessions still share one forward pass. The guard the plan cites, `ThrowIfForwardPassNotIsolated` in `InferenceSession.cs`, is not in the repository, so that reference is stale.
- **Overlap:** plan 04's item 4 (server observability) is covered by the [observability workstream](observability.md). Its item 6 (sessions out of the product configuration) depends on plan 010's outcome.

## Open items

- [ ] 04 item 2: extend effective configuration from static plan inputs to real server and loader startup values
- [ ] 04 item 3: extend the golden capability fixtures only where a loader actually executes the route
- [ ] 04 item 4: server observability (VRAM and RAM breakdown, streaming timing, per-request cache signal). Covered by the observability workstream
- [ ] 04 item 5: model and API compatibility corpus, with protocol smoke coverage
- [ ] 04 item 6: keep sessions out of the product configuration surface until the proven CPU-dense restart lane is proven
- [ ] 010: `CreateContext` per backend (CPU dense, Vulkan and CUDA, hybrid, hybrid GDN)
- [ ] 010: a concurrency test for a fork bound to a real model, with parent and child driven at once
- [ ] 010: remove the `NotSupportedException` path from the fork guard once isolation exists (the guard itself is not in the repository; confirm before acting)

## Cadence

- Review when a configuration or session change lands, and when the observability workstream closes.
- Re-read at least every two weeks while any item is open.
- Close when plan 04 items 2 to 6 are each done or dropped, and plan 010 is either implemented or written off.
