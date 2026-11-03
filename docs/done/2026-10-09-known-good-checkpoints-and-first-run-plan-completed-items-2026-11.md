# Known-good checkpoints, favourites and first-run: plan - completed items

Ticked items moved from [docs/3-product-and-runtime/2026-10-09-known-good-checkpoints-and-first-run-plan.md](../3-product-and-runtime/2026-10-09-known-good-checkpoints-and-first-run-plan.md) on 2026-11-03. Each keeps its section heading and nested lines.

## Checklist (2026-10-30; code checked)

- [x] P0 leftover: `pull`'s download and listing parser still use their own client and parser *(done, checked 2026-11-01 against the code: `PullCommand` downloads through `ModelDownloader.DownloadAsync`, the same code `setup` uses, and lists through `ExternalHttpClient`, `PullCommand.cs` lines 61 and 171)*

- [x] P5 docs as tests: a test that fails when a cited file, hash or evidence disappears *(done for the catalogue, checked 2026-11-01: `CatalogGuardTests` pins every entry's hash, revision, size and evidence and that each qualification cites a golden that exists and pins the same file; `StatusMatrixCitationTests` covers STATUS. README commands being run, not just compiled, is not covered: `verify-package-consumers.ps1` compiles the marked snippets and runs in CI since plan 6 B1.)*

## Open items

- [x] Known-good P0 leftover: `pull`'s download and listing parser still use their own client *(done, checked 2026-11-01 against the code: `PullCommand` downloads through `ModelDownloader.DownloadAsync`, the same code `setup` uses, and lists through `ExternalHttpClient`, `PullCommand.cs` lines 61 and 171)* (the item is closed: only a note remains, in 061)

- [x] Known-good P5: docs-as-tests *(see the checklist at the top: done for the catalogue and STATUS; README commands are compiled in CI, not run)*
