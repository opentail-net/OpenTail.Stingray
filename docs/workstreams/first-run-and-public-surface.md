# Workstream: first run and public surface

**Owner:** this file. **Last verified:** 2026-10-30.
**Plans:** [../3-product-and-runtime/2026-10-09-known-good-checkpoints-and-first-run-plan.md](../3-product-and-runtime/2026-10-09-known-good-checkpoints-and-first-run-plan.md) (catalogue, first-run seam, favourites), [../3-product-and-runtime/103-front-door-design.md](../3-product-and-runtime/103-front-door-design.md) (front door), [../3-product-and-runtime/2026-10-09-public-developer-surface-and-front-door-plan.md](../3-product-and-runtime/2026-10-09-public-developer-surface-and-front-door-plan.md) (public C# surface and task commands).

## Correct state

- **Known-good plan (status 2026-10-09, checked against code 2026-10-30):** P0 first-run seam done (`SetupFlow.cs`). P1 shared preflight done for `chat` and `setup`, not yet for `run` and `serve`. P2 qualification done (Qwen2.5 0.5B, 1.5B and 7B goldens). P3 favourites code present, end-to-end `models use` not re-checked. P4 `--verify` present, `--local` not found. P5 docs-as-tests open.
- **Front door:** steps 1 and 2 done. Steps 3 (starter manifest, recipes) and 4 (task commands) are open in [../3-product-and-runtime/103-front-door-design.md](../3-product-and-runtime/103-front-door-design.md).
- **Recipes (2026-10-11):** README recipes landed. `stingray setup chat` then `stingray chat` ran on a clean model home in 5 to 6 s. **Not run:** the CLI `speak` and `transcribe` recipes.
- **Task command files exist:** `ChatCommand.cs`, `SpeakCommand.cs`, `TranscribeCommand.cs`, `CatalogTaskResolver.cs`.
- **Public C# surface:** the samples exist under `samples/`. No compile test for an external consumer was found, so the plan's A1 and D1 are open.

## Open items

- [ ] Run the `speak` and `transcribe` recipes and record the result (backlog item 1, section 3)
- [ ] Front door step 3: starter manifest and README recipes around the task commands
- [ ] Front door step 4: task commands that need no paths once set up, with a one-line fix for a missing model
- [ ] Known-good P0 leftover: `pull`'s download and listing parser still use their own client
- [ ] Known-good P1 leftover: `run` and `serve` not wired to the preflight
- [ ] Known-good P3: re-check `models use` end to end
- [ ] Known-good P4: `--local` inventory, or record that it is not planned
- [ ] Known-good P5: docs-as-tests
- [ ] Public surface: A1 (compiling external consumer), A2 to A4, B5, B6, C2, D1 to D3, E

## Decisions for the owner

- Whether `--local` (known-good P4) is wanted at all.

## Cadence

- Review after each recipe run, and at each release-gate review.
- Re-read at least every two weeks while any item is open.
- Close when the recipes are recorded as run, each P0 to P5 item is done or dropped in writing, and the external-consumer compile test exists.
