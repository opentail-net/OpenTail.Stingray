# Handing work to a smaller AI model

For work that is bounded, pattern-following and checked by an automated test. Paste the **preamble**, then **one** task card. Review the diff yourself (or have a stronger model review it) before committing: a smaller model is fast but is the likelier one to make a plausible-looking mistake.

## Do NOT delegate

- `GgufModel` / the loaders, forward passes, kernels, anything numeric (parity bugs look fine and are wrong).
- The memory estimator and its calibration (needs real measurements and judgment), `ExternalAccess` (safety policy), admission or qualification claims (truthfulness matters more than speed).
- Anything that needs a heavy real-weight run, a download, or a decision about what a result *means*.

## Preamble (paste first)

```
You are working in the C#/.NET 10 repo OpenTail.Stingray (C:\Git-Public\OpenTail.Stingray). Read CLAUDE.md first and obey it. Key rules:
- Warnings are errors. No Python. No reflection/dynamic code (NativeAOT). Source-generated JSON only.
- NEVER pass --nologo to `dotnet test`. Build the test project, then run its .exe directly:
  dotnet build tests/OpenTail.Stingray.Tests.Cli -c Release
  tests/OpenTail.Stingray.Tests.Cli/bin/Release/net10.0/OpenTail.Stingray.Tests.Cli.exe -class <Full.Namespace.Class>
  (the class name must be namespace-qualified, or zero tests run). Run the whole project's .exe before you finish.
- "N passed" is not evidence unless real tests ran: check the Total count and that it went UP by about the number you added.
- A test that returns early when a model file is missing and reports "passed" proves nothing. Tests you write must always really run (synthetic data, fake HTTP). Do not add tests that need downloads or real weights.
- Do not touch files outside the list in your task. Do not edit CLAUDE.md. Do not commit; leave changes in the working tree.
- Do not use subagents. Do not put scratch files in the repo; use the OS temp directory.
- Keep output small: pipe long command output through `Select-String`/`-First N`.
Existing helpers to reuse, not copy: tests/OpenTail.Stingray.Tests.Cli/HubFixtures.cs (FakeHub with Range support + a tiny GGUF builder), SyntheticGguf (Core tests), ExternalHttpClient (all network goes through it), HubClient (Hub API parsing), ScoutAnalyzer.
When finished, report: files changed; test counts before/after (whole project); anything you were unsure of; anything you did NOT do. If an instruction conflicts with the code, stop and say so instead of guessing.
```

## Task card A: catalogue guard tests (plan P5, mechanical)

**Goal:** tests that fail when the model catalogue rots.
**Files you may change:** add `tests/OpenTail.Stingray.Tests.Cli/CatalogGuardTests.cs` only.
**Do:** for every entry in `ModelCatalog.Entries` assert: `Id` is unique and lower-case; `Task` is one of `ModelCatalog.Tasks`; exactly one default per task (first entry) and at most 3 alternatives per task; each file has a 64-hex `Sha256`, a 40-hex `Revision`, `Size > 0`, a non-empty `RepoPath`; `RunTemplate` is non-empty; `LicenceNeedsConsent` is true whenever `Licence` does not start with `MIT`, `Apache-2.0` or `BSD`; no two entries share a (repo, path, revision) with different sha256; the `Evidence` text is non-empty. Make failure messages name the entry id.
**Done when:** the new tests pass, the whole Cli project passes, and deliberately breaking one catalogue field makes exactly the matching test fail (try it, then revert).
**Expected effort:** small (1-2 hours for a smaller model).

## Task card B: demand-ranked backlog command (list-only, spec exists)

**Goal:** `stingray scout --backlog [--limit N]` prints which Hugging Face GGUF architectures are not registered/admitted here, ranked by 30-day downloads.
**Read first:** `docs/reference/061-coverage-tooling.md` (external access, remote scout) and `todo.md` (the "Demand-ranked backlog" item). The Hub list endpoint is verified to return `gguf.architecture`, `downloads` (30-day, per repository), `gated`, `gguf.totalFileSize` in ONE call: `https://huggingface.co/api/models?filter=gguf&sort=downloads&direction=-1&limit=N&expand[]=gguf&expand[]=downloads&expand[]=gated`.
**Files you may change:** `src/OpenTail.Stingray.Cli/Scout/Backlog.cs` (new), `ScoutCommand.cs` (add the flag), a new test file, `docs/reference/061-coverage-tooling.md` (one short section), regenerate `docs/reference/cli-option-inventory.md` with `scripts/gen-cli-option-inventory.ps1` and fill the new rows' Class with `diagnostic`.
**Rules:** all network through `ExternalHttpClient` (the gateway enforces the policy and host allow-list); `--limit` default 50, hard cap 200; list-only, never download; downloads are per repository, not per file: say so in the output; group by architecture; mark each architecture `admitted` / `not admitted` / `unregistered` using `ArchitectureRegistry.Find`; do NOT print families that are `NotAdmitted` ported-but-unverified (CLAUDE.md rule 14: keep them out of public output: show them as `unregistered`). Test with a fake handler like `HubFixtures.FakeHub` (add the list route); no live network in tests.
**Done when:** tests cover grouping, ranking order, limit cap, gated flag, malformed JSON entries skipped, policy off throws `ExternalAccessDeniedException`, and `--limit` validation; whole Cli project passes.
**Expected effort:** medium (half a day for a smaller model); review closely, especially the registry marking.

## Task card C: `pull` should reuse `HubClient`'s listing parser

**Goal:** remove the duplicated Hub listing parser in `PullCommand`.
**Files you may change:** `src/OpenTail.Stingray.Cli/PullCommand.cs`, `tests/OpenTail.Stingray.Tests.Cli/PullCommandListingTests.cs` only.
**Do:** make `PullCommand.ParseGgufListing` (keep its signature, tests call it) delegate to `HubClient` parsing for names, sizes and published sha256 without changing results. Keep the existing behaviour for a response with no `sha` (the Hub client requires one; `pull` must keep working on such JSON, so parse the file list separately from the commit). Do NOT change the download code or move it to the gateway in this task.
**Done when:** every existing `PullCommandListingTests` test passes unchanged, you add tests for the no-`sha` case, and the whole Cli project passes.
**Expected effort:** small-medium (2-3 hours).

## Task card D: signature harvest (data only, no code)

**Goal:** add reference signatures for admitted architectures that have none, from hosted models, without downloading any weights.
**Files you may add:** `src/OpenTail.Stingray.Cli/Scout/Signatures/*.signature.json` only. Do not edit code, tests, or any doc (report results back instead).
**Find the gaps:** a throwaway local test or script (do not commit it) printing `ArchitectureRegistry.All` ids with `Status == Admitted` that have no file in `Scout/Signatures/` (the file name starts with the architecture id).
**For each gap:** use the Hub list API (`https://huggingface.co/api/models?filter=gguf&search=<name>&sort=downloads&direction=-1&expand[]=gguf&expand[]=downloads&expand[]=gated`) to find a repo whose `gguf.architecture` equals the id and that is public (not gated), licence apache-2.0 or mit, and has a SINGLE-file (not split) Q4/Q5/Q8 quantisation under about 6 GB. Then emit to a temp path:
`stingray scout -r <owner/repo> -f <file.gguf> --emit-signature <temp>.json`, read `structure_id` from the JSON, and move it to `Scout/Signatures/<id>--<first 12 chars of structure_id>.signature.json`. If the file for that name already exists, `--emit-signature` merges the origin: that is correct, keep the merge.
**Rules:** never hand-edit a signature (it carries a content hash and the loader rejects edits); never relax a guard test; if an architecture has no qualifying small public single-file repo, SKIP it and list it in your report; the command refuses non-admitted architectures, so a refusal means the id is not admitted (do not work around it).
**Done when:** `dotnet build tests/OpenTail.Stingray.Tests.Cli -c Release` and the whole Cli project pass (the guard `Embedded_signatures_are_valid_and_belong_to_admitted_architectures` checks every file), and the report lists: architecture, repo, file, commit, and the skipped ones with the reason.
**Expected effort:** 2-3 hours, mostly browsing the Hub.

## Task card E: local inventory (`models --local [--verify]`; plan P4)  **start only after other work on `ModelsCommand.cs` and the option inventory has finished**

**Read first:** `docs/3-product-and-runtime/2026-10-09-known-good-checkpoints-and-first-run-plan.md` sections 3 (G5, G7), 5.2 and P4.
**Files you may change:** `src/OpenTail.Stingray.Cli/ModelsCommand.cs`, a new `src/OpenTail.Stingray.Cli/LocalInventory.cs`, a new test file, and (last step) `docs/reference/cli-option-inventory.md` regenerated with `scripts/gen-cli-option-inventory.ps1` (Class `stable`).
**Do:** `stingray models --local` lists (1) each catalogue entry with its state (Missing / Partial / Installed) and (2) every GGUF found in the model home and in `STINGRAY_MODEL_DIRS`, reusing the scanning that `ListModelsCommand` already does (do not write a second scanner). For each GGUF read only its index (`GgufModel.Open`, no weights) and show: architecture, whether it is admitted (`ArchitectureRegistry`; show a ported-not-verified family as plain "not supported", CLAUDE.md rule 14), the dominant quantisation, size, and whether it fits this machine via `LoadPreflight.EvaluateFile`. `--verify` additionally re-hashes catalogue files with `ModelFingerprinter.Compute` (it caches in a `.sha256` sidecar) and reports Intact / Damaged against the catalogue sha256; it is opt-in because hashing is slow. Never modify or delete a model file.
**Tests:** a temp model home with synthetic GGUFs (`HubFixtures.Gguf`) covering: catalogue states, an admitted and an unregistered architecture, an unreadable file reported as unreadable without stopping the scan, fits / does not fit (inject RAM), `--verify` intact and damaged, no file modified.
**Done when:** the new tests and the whole Cli project pass, and the option inventory guard passes.
**Expected effort:** medium (3-4 hours).
## Follow-up to card B and D (do these first)

1. Backlog: treat `AdmissionStatus.Experimental` like `NotAdmitted` in `ResolveStatus` (rule 14: both are ported-but-unverified; show them as unregistered) and add a test.
2. Backlog: replace the bare `catch { }` in `Parse` with `catch (Exception ex) when (ex is JsonException or InvalidOperationException or FormatException)`.
3. Backlog: remove `TotalFileSizeBytes` (it sums every quant of every repo, which means nothing) or rename and document it honestly.
4. Backlog: add `&pipeline_tag=text-generation` to the Hub query (verify the Hub honours it together with `filter=gguf` and say so in your report); also label `(none)` as "no architecture declared" and `clip` as "projector". The live list currently shows audio / diffusion / embedding families as "unregistered", which is misleading for this registry.
5. Signatures: remove `qwen2moe--3fdf99ea8c33` (its source repo `RichardErkhov/katuni4ka_-_tiny-random-qwen1.5-moe-gguf` has no licence tag and is a random-weights test model; card D requires apache-2.0 or mit). Replace it only with a real, licensed checkpoint if one under about 6 GB exists; otherwise list `qwen2moe` as skipped. In your report state each repo's licence and whether it is an official release or a community build.

## Task card F: favourites (plan P3)

**Read first:** `docs/3-product-and-runtime/2026-10-09-known-good-checkpoints-and-first-run-plan.md` section 5.2 and P3.
**Files you may change:** new `src/OpenTail.Stingray.Cli/Favourites.cs`, `src/OpenTail.Stingray.Cli/CatalogTaskResolver.cs`, `src/OpenTail.Stingray.Cli/ModelsCommand.cs`, a new test file, and (last step) the regenerated `docs/reference/cli-option-inventory.md` (Class `stable`) plus one short section in `docs/reference/061-coverage-tooling.md`.
**Decisions already made (change nothing without asking):** the file is `favourites.json` in a per-user config directory: `%APPDATA%\stingray` on Windows, `$XDG_CONFIG_HOME/stingray` (default `~/.config/stingray`) elsewhere, overridable by the environment variable `STINGRAY_CONFIG_DIR` (register it in `KnownEnvironmentVariables`, add its row to `docs/reference/env-var-inventory.md` and bump the count sentence there: the Core tests enforce both). Format: `{ "chat": "qwen2.5-1.5b" }` mapping an alias to a CATALOGUE id only (no user-chosen repo files in this task). Commands: `stingray models use <task> <id>` (validates the id exists AND serves that task; refuses otherwise) and `stingray models use <task> --clear`; `stingray models` marks the favourite.
**Resolution order** in `CatalogTaskResolver`: explicit `--model-file` > explicit `--model <id>` > favourite > catalogue default. A favourite that names a removed or wrong-task id must produce an error that says what to run to fix it (`stingray models use chat <id>` or `--clear`); never silently fall back to the default. A corrupt or unreadable `favourites.json` is an error naming the file, never silently ignored and never overwritten.
**Tests:** temp config dir; each step of the resolution order; set / clear; wrong task; unknown id; stale favourite; corrupt file; a write is atomic (write a temp file then rename) and never leaves a half-written file; no test touches the real user config directory.
**Done when:** the new tests and the whole Cli and Core projects pass, and the option and environment-variable inventory guards pass.
**Expected effort:** medium (3-4 hours).
## After the work comes back

Check: the diff touches only the listed files; test totals went up by about the count added; run the Core guards too (`Tests.Core` classes `ArchitectureCapabilitiesBaselineTests` and `KnownEnvironmentVariablesTests`) because they scan all of `src/`; try one deliberate break to confirm a new test can fail.
