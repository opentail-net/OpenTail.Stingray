# Completed items split from 2026-10-09-known-good-checkpoints-and-first-run-plan.md (2026-10-31)

Finished items moved verbatim out of [../3-product-and-runtime/2026-10-09-known-good-checkpoints-and-first-run-plan.md](../3-product-and-runtime/2026-10-09-known-good-checkpoints-and-first-run-plan.md). Open items stay there. Each block is preceded by the heading it sat under. Relative links inside the blocks were rebased for this folder.

> from: ## Checklist (2026-10-30; code checked)
- [x] P0 first-run seam (`src/OpenTail.Stingray.Cli/SetupFlow.cs`)

> from: ## Checklist (2026-10-30; code checked)
- [x] P1 shared preflight for `chat` and `setup`

> from: ## Checklist (2026-10-30; code checked)
- [x] P1 leftover: `run` and `serve` call `LoadPreflight` (`RunCommand.cs:228`, `InferenceEngineLoader.cs:35`; checked 2026-10-30). Estimator location not re-checked. No test run recorded for this item.

> from: ## Checklist (2026-10-30; code checked)
- [x] P2 variants and qualification scope (Qwen2.5 0.5B, 1.5B, 7B goldens under `tests/OpenTail.Stingray.Tests.ForwardPass/Goldens/`)

> from: ## Checklist (2026-10-30; code checked)
- [x] P4 inventory and verify: `-l|--local` and `--verify` both exist (`ModelsCommand.cs`; checked 2026-10-30, the earlier "not found" was a search miss on the short alias). No test run recorded for this item.

> from: ### Open items
- [x] Front door step 4 is superseded in the plan: `chat`, `speak` and `transcribe` are implemented; `describe` and `image` are out of scope. The one-line missing-model fix was not re-checked.

> from: ### Open items
- [x] Known-good P1 leftover: `run` and `serve` call `LoadPreflight` (checked 2026-10-30)

> from: ### Open items
- [x] Known-good P4: `models --local` and `--verify` implemented (checked 2026-10-30)

