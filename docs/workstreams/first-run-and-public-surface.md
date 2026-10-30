# Workstream: first run and public surface

> This file summarises its owner plan. If the two disagree, the owner plan is authoritative; correct this file.

**Owner:** this file. **Last verified:** 2026-10-30.
**Plans:** [../3-product-and-runtime/2026-10-09-known-good-checkpoints-and-first-run-plan.md](../3-product-and-runtime/2026-10-09-known-good-checkpoints-and-first-run-plan.md) (catalogue, first-run seam, favourites), [front-door design](../3-product-and-runtime/2026-10-09-public-developer-surface-and-front-door-plan.md#front-door-design-merged-2026-10-30) (front door), [../3-product-and-runtime/2026-10-09-public-developer-surface-and-front-door-plan.md](../3-product-and-runtime/2026-10-09-public-developer-surface-and-front-door-plan.md) (public C# surface and task commands).

## Correct state

- **Known-good plan (checked against code 2026-10-30):** P0 seam done. P1 preflight wired to `chat`, `setup`, `run` and `serve`. P2 qualification done (Qwen2.5 0.5B, 1.5B, 7B goldens). P3 favourites implemented (`models use`, `--clear`), end-to-end run not recorded. P4 `models --local` and `--verify` implemented. P5 partly done.
- **Front door:** steps 1 and 2 done. Steps 3 (starter manifest, recipes) and 4 (task commands) are open in [front-door design](../3-product-and-runtime/2026-10-09-public-developer-surface-and-front-door-plan.md#front-door-design-merged-2026-10-30).
- **Recipes (2026-10-11):** README recipes landed. `stingray setup chat` then `stingray chat` ran on a clean model home in 5 to 6 s. **Not run:** the CLI `speak` and `transcribe` recipes.
- **Task command files exist:** `ChatCommand.cs`, `SpeakCommand.cs`, `TranscribeCommand.cs`, `CatalogTaskResolver.cs`.
- **Public C# surface:** external consumer compile check exists: `scripts/verify-package-consumers.ps1`, 17 checks, 0 failed on 1.0.9 (2026-10-30). It proves compilation against packed packages, not inference. A1 and D1 ticked; D2 and D3 open.

## Open items

- [ ] Run the `speak` and `transcribe` recipes and record the result (backlog item 1, section 3)
- [ ] Front door step 3: starter manifest and README recipes around the task commands
- [ ] Front door step 4: task commands that need no paths once set up (registered; the one-line missing-model fix not re-checked)
- [ ] Known-good P0 leftover: `pull`'s download and listing parser still use their own client
- [x] Known-good P1 leftover: `run` and `serve` call `LoadPreflight` (checked 2026-10-30)
- [ ] Known-good P3: re-check `models use` end to end
- [x] Known-good P4: `models --local` and `--verify` implemented (checked 2026-10-30)
- [ ] Known-good P5: docs-as-tests
- [ ] Public surface: A2 to A4, B5 (application-name item), C2, D2, D3, E

## Decisions for the owner

- No open decision: `--local` exists.

## Cadence

- Review after each recipe run, and at each release-gate review.
- Re-read at least every two weeks while any item is open.
- Close when the recipes are recorded as run, each P0 to P5 item is done or dropped in writing, and D2 and D3 are done or dropped.
