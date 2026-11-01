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


---

## More completed items moved from 2026-10-09-public-developer-surface-and-front-door-plan.md on 2026-11-01

Finished items moved verbatim out of [../3-product-and-runtime/2026-10-09-public-developer-surface-and-front-door-plan.md](../3-product-and-runtime/2026-10-09-public-developer-surface-and-front-door-plan.md). Open items stay there. Each block is preceded by the heading it sat under. Relative links inside the blocks were rebased for this folder.

> from: ## Checklist (2026-10-30; code checked)
- [x] Recipes: run `stingray speak` and `stingray transcribe` as README recipes and record the result. 2026-11-01, the published CLI 1.0.9 installed from nuget.org into a temporary tool path, with a temporary model home (`STINGRAY_MODEL_HOME`) holding the Piper lessac voice and `ggml-base.bin` copied from disk, so no download and no voice-licence prompt: `stingray models` showed speak and transcribe as installed; `stingray speak "Hello from Stingray." -o hello.wav` generated 1.47 s of audio in 0.85 s (0.58x RTF, 65 KB WAV); `stingray transcribe hello.wav` answered "Hello from TINGOT." (1.42 s). Both recipes work end to end; the transcript mishears the last word, so the round trip is not exact. The `setup speak` licence prompt itself was not exercised

> from: ## Checklist (2026-10-30; code checked)
- [x] D2. Verify the actual NuGet package *(2026-11-01: `scripts/verify-nuget-package.ps1` packs, checks the required package entries (README, assemblies, notices) and consumes the package from a clean project; `scripts/verify-package-consumers.ps1` compiles every marked README block, 17 checks and 0 failed; and the package served by nuget.org, version 1.0.9, was installed into a clean project outside the repo, the README sample built with 0 errors and 0 warnings, and the published CLI answered a prompt on a real model (Qwen3-0.6B Q8_0, CPU, 43 tok/s decode). CI does not run the real-weight step, by design)*

> from: ### The front door (tooling)
- [x] 4. **SUPERSEDED (2026-10-30; describe and image are not part of the current plan; chat, speak and transcribe are implemented).** Task commands that need no paths once set up: `stingray chat`, `speak`, `transcribe`,
   `describe`, `image`. A missing model gives a one-line fix (`run: stingray setup speak (63 MB)`),
   never a stack trace.

> from: ### The front door (tooling)
- [x] 6. **SUPERSEDED by §3.2 (no separate facade).** A small library facade so the README's C# is three lines per task, and the snippets are
   compiled by a test so they cannot rot:
   `await using var chat = await StingrayChat.OpenAsync("qwen2.5-0.5b");` /
   `Speech.SynthesizeAsync(...)` / `Transcriber.TranscribeAsync(...)`. The facade is built on the
   existing engines, not new math.
