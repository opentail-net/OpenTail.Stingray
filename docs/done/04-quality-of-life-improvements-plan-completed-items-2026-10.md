# Completed items split from 04-quality-of-life-improvements-plan.md (2026-10-30)

Finished items moved verbatim out of [../3-product-and-runtime/04-quality-of-life-improvements-plan.md](../3-product-and-runtime/04-quality-of-life-improvements-plan.md). Open items stay there. Each block notes the section it came from.

> from: ## Remaining work
- [x] 1. **DONE, 2026-08-15 / 2026-09-02.** Env-var/CLI inventory reconciliation, alias dedup, and a
   dead-switch sampling check (0 of 20 sampled `bench`/`experimental` rows were dead). Split out to
   [done/quality-of-life-improvements-plan-item1-env-var-inventories.md](quality-of-life-improvements-plan-item1-env-var-inventories.md)
   2026-09-27.

---

## More completed items moved from 04-quality-of-life-improvements-plan.md on 2026-11-01

Finished items moved verbatim out of [../3-product-and-runtime/04-quality-of-life-improvements-plan.md](../3-product-and-runtime/04-quality-of-life-improvements-plan.md). Open items stay there. Each block is preceded by the heading it sat under. Relative links inside the blocks were rebased for this folder.

> from: ### Open items
- [x] 010: `CreateContext` per backend: **not needed for session forking** (no caller in `src/`; see the correction at the top of plan 010). Whether to remove the method is an owner decision

> from: ### Open items
- [x] 010: a concurrency test for a fork bound to a real model, with parent and child driven at once: done 2026-10-31, `HotSessionForkRealModelTests.Fork_TwoBranchesGeneratingConcurrently_DoNotDisturbEachOtherOrTheParent` (CPU dense, SmolLM2-1.7B)

> from: ### Open items
- [x] 010: the fork guard (`ThrowIfForwardPassNotIsolated`) is not in the repository; nothing to remove

