# Completed items split from 103-front-door-design.md (2026-10-30)

Finished items moved verbatim out of [../3-product-and-runtime/103-front-door-design.md](../3-product-and-runtime/103-front-door-design.md). Open items stay there. Each block notes the section it came from.

> from: ## The front door (tooling)
- [x] 2. **A model home outside the repo**: `%LOCALAPPDATA%\stingray\models` / `~/.cache/stingray/models`,
   overridable, searched after explicit paths. The repo-relative `models/` stays a developer
   convenience.

> from: ## The front door (tooling)
- [x] 3. **`stingray setup <task>`**: shows size, licence and expected speed, asks, then downloads the
   bundle through the existing `pull` machinery (resumable, sha-checked).

> from: ## The front door (tooling)
- [x] 5. **`stingray models`** (the per-task view; RAM-based hiding of options is not done): installed bundles, which tasks are ready, what to run to fix the rest.

> from: ## Order of work
- [x] 1. Move `README.md` to `docs/STATUS.md`; write the new README around what works today: explicit
   model downloads, commands and C# that have actually been run. Fix the NuGet README's
   non-compiling snippets in the same pass. (DONE 2026-09-27)

> from: ## Order of work
- [x] 2. Starter manifest + model home + `setup` + `models`. (DONE 2026-09-28; embed/rerank, describe-image and image entries not yet: each needs a public file whose hash matches a tested one)

