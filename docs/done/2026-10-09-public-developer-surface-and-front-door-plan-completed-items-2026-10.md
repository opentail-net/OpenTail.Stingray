# Completed items split from 2026-10-09-public-developer-surface-and-front-door-plan.md (2026-10-31)

Finished items moved verbatim out of [../3-product-and-runtime/2026-10-09-public-developer-surface-and-front-door-plan.md](../3-product-and-runtime/2026-10-09-public-developer-surface-and-front-door-plan.md). Open items stay there. Each block is preceded by the heading it sat under. Relative links inside the blocks were rebased for this folder.

> from: ## Checklist (2026-10-30; code checked)
- [x] A1. Prove the public API with a compiling consumer: `scripts/verify-package-consumers.ps1`, 1.0.9 re-run on 2026-10-30: 17 checks, 0 failed (exit 0). Earlier 1.0.8 run: 2026-10-11.

> from: ## Checklist (2026-10-30; code checked)
- [x] B1. Shared catalogue resolution (`src/OpenTail.Stingray.Cli/CatalogTaskResolver.cs`)

> from: ## Checklist (2026-10-30; code checked)
- [x] B2. `stingray chat` (`ChatCommand.cs`)

> from: ## Checklist (2026-10-30; code checked)
- [x] B3. `stingray speak` (`SpeakCommand.cs`)

> from: ## Checklist (2026-10-30; code checked)
- [x] B4. `stingray transcribe` (`TranscribeCommand.cs`)

> from: ## Checklist (2026-10-30; code checked)
- [x] B5. Register commands (`Program.cs:115-119`) and update catalogue run templates (`ModelCatalog.cs` now uses `stingray chat`, `speak`, `transcribe`; checked 2026-10-30)

> from: ## Checklist (2026-10-30; code checked)
- [x] B5 item 3, application name: `SetApplicationName("stingray")` and every user-facing `opentail-llm-cli` mention in `src/` renamed, 2026-10-31 (`CliBrandingTests`: root `--help` prints `stingray <OPTIONS> [COMMAND]`; a source scan keeps the old name out)

> from: ## Checklist (2026-10-30; code checked)
- [x] B6. CLI tests for the task commands (`tests/OpenTail.Stingray.Tests.Cli/FrontDoorTaskCommandsTests.cs` exists; tests not run on 2026-10-30)

> from: ## Checklist (2026-10-30; code checked)
- [x] C1. Repository-root README (2026-10-11 evidence in `00-current-work.md` item 3.1)

> from: ## Checklist (2026-10-30; code checked)
- [x] D1. Compile the public C# samples and README snippets against the package: 13 of 13 on 2026-10-11 (1.0.8); 1.0.9 re-run on 2026-10-30 passed (17 checks, 0 failed).

> from: ### Order of work
- [x] 3. (Historical.) Task commands: done (registered; see B2 to B4). The library facade is superseded by §3.2. Rewriting the README recipes around them is C2.

