# 2026-10-08-golden-parity-and-admission-tooling-plan - completed items 2026-11

Ticked items moved from [2026-10-08-golden-parity-and-admission-tooling-plan.md](../2-coverage/2026-10-08-golden-parity-and-admission-tooling-plan.md) on 2026-11-08. Each keeps its text and nested lines. Open items stay in the source doc.

- [x] **0. Decisions** (this document). D1: golden location (`tests/.../Goldens/`, recommended, so reviewers see them in the diff). D2: near-tie margin tolerance (start 0.02 logit, configurable, recorded in the result; revisit with data from the 25 migrated receipts). D3: SHA-256 mandatory for new goldens (recommended yes), optional for migrated ones until each file is re-hashed. D4: migrate all 25 legacy classes over time, keeping exceptions (DeepSeek2 until `engineSettings` covers it). **Confirmed shipped 2026-11-06:** D1 and D3 match exactly (all 24 goldens on disk, under that path, all hashed); D2's default is live and has not yet been revisited with receipt data (that revisit is still open, tracked separately, not a blocker on this decision being "taken").
- [x] **1. `ModelLocator` and loud skips** (done 2026-10-09)
  - [ ] 1.b (optional) Point `Tests.Cuda` / other `FindModelPath` copies at the same locator.
