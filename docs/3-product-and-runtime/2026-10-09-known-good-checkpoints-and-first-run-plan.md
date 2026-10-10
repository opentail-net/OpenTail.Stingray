# Known-good checkpoints, favourites and first-run: plan

**Status:** In progress (2026-10-09). **P0 first-run seam DONE** (`SetupFlow`; `chat`/`speak`/`transcribe` offer to install a missing model, interactive only; `pull --revision` and default commit pin). P0 leftovers: `pull`'s download and listing parser still use their own client/parser instead of the gateway/`HubClient`. **P1 shared preflight DONE for `chat` and `setup`** (see 061; Unknown warns rather than blocks; `run`/`serve` not wired, estimator still in the Cli project). **P2 DONE (2026-10-10):** `FamilyId` + `Qualifications` on `CatalogEntry`; three Qwen2.5 chat entries (0.5b default Exact 32/32, 1.5b and 7b NearTie 31/32, each pinned to a llama-server 10306 golden under `tests/OpenTail.Stingray.Tests.ForwardPass/Goldens/`); `setup` offers the largest smaller qualified same-family model that fits when the chosen one does not; `models <task>` and the setup summary print a scoped "Qualified" row (unqualified entries say so). **P3 favourites, P4 `models --local [--verify]` DONE** (cards F, E). **P5 partly done:** catalogue guards plus a test that every cited golden exists and pins the entry's file name, size and SHA-256; README/STATUS rows citing catalogue ids not yet covered. Speeds on the 1.5b/7b rows are single golden runs (33.3 and 8.0 tok/s decode), not rule-7 measurements. Reviews an outside proposal (a ChatGPT conversation shared by the user) against the code as it is today.
**Related:** [front-door design](103-front-door-design.md) (catalog rules), [scout plan](2026-10-09-checkpoint-scout-and-ai-admission-plan.md), `docs/reference/061-coverage-tooling.md` (scout, external access), `todo.md` (top section).

## 1. The proposal, and my verdict

Make the path from "I want to chat" to "a model is loaded" a single guided flow, built from parts that stay separate:

1. **Known-good catalogue**: particular checkpoint files that proved useful, with exact identity and a stated scope of what was qualified.
2. **User favourites**: personal aliases (`chat`, `coding`, `vision`, `speech`) that select from the catalogue and can be overridden.
3. **Local inventory**: which exact files are present, intact and usable on this machine.
4. Joined by one workflow that reuses **scout** (remote feasibility), the existing **installer** (acquisition) and a **preflight** (guard before loading). No new download mechanism, no new package format.

**Verdict: the direction is right and I would build it.** Two corrections from reading the code:

- **Most of components 1 and 4 already exist** (section 2). The work is extending them, not creating a second catalogue.
- **"Bullet-proof" is the wrong target.** The right claim is the scoped one: *structure understood, this exact file intact, budget feasible, configuration X passed tests Y*. Never "this model works". Section 8 turns the proposal's five requirements into acceptance criteria.

## 2. What exists today (verified in the code, 2026-10-09)

| Piece | Where | What it does |
|---|---|---|
| Catalogue | `Core/Catalog/ModelCatalog.cs` | Static C# table (NativeAOT safe, a typo is a compile error). `CatalogEntry`: id, task, why, files, licence + consent flag, hardware, speed, evidence, run template. `CatalogFile`: repo, **pinned revision**, path, **sha256**, size. 3 entries: one per task (`qwen2.5-0.5b` chat, `piper-lessac`, `whisper-base`). |
| Catalogue rules | `103-front-door-design.md` | One default per task, at most three alternatives each justified by a user-level reason, green status row **and** an automated test on those exact files. Anything else works through `pull -r` / `-m`. The catalogue is a recommendation list, not the coverage list. |
| Model home | `Core/Catalog/ModelHome.cs` | Flat directory outside the repo (`STINGRAY_MODEL_HOME`). `StateOf` = Missing / Partial / Installed by **existence and size only**. |
| Installer | `Core/Catalog/ModelDownloader.cs` | Resumable `.part`, SHA-256 verified **before** the rename, so a file at its final name was verified at install time. Honours the shared external-access policy. |
| `stingray setup <task\|id>` | `Cli/SetupCommand.cs` | Summary (size, licence, needs, speed), licence consent, **asks before downloading** (`--yes` to skip, refuses when non-interactive), then installs. |
| `stingray models` | `Cli/ModelsCommand.cs` | Installed bundles, which tasks are ready, the command that fixes each gap. |
| Task commands | `Cli/CatalogTaskResolver.cs`, `ChatCommand.cs` | Resolve flag > catalogue default; if Missing or Partial print "Run: stingray setup chat" and stop. **No preflight before load.** |
| Scout, estimator, remote scout | `Cli/Scout/*` | Dossier, memory gate (CPU, plain-attention models), `scout -r` reads only a hosted file's index at a pinned commit. Lives in the Cli project. |
| External access | `Core/Net/*` | One policy, default allowed, deny wins. |

## 3. Gaps, honestly

| # | Gap | Consequence |
|---|---|---|
| G1 | A task command that finds nothing only *tells* the user to run `setup`. | The first-run experience has a seam. |
| G2 | One chat entry (0.5B). No quant variants, no alternative. | Nothing to fall back to when the default does not fit; "fits in 8 GB" alternatives do not exist. |
| G3 | Evidence, hardware and speed are free text. No scope (backend, quant, context, capability, engine version). | "Known-good" cannot be stated precisely, so it can be over-read. |
| G4 | No preflight before a task command loads. "about 1 GB RAM" is prose, not a check. | An unknown or over-budget load is attempted. |
| G5 | `StateOf` trusts size. Corruption or a swapped file after install goes unnoticed until it misbehaves. | "Installed" means "was verified once". |
| G6 | No favourites layer. | A user wanting a different chat model must pass `-m` every time. |
| G7 | `models` shows the catalogue only. A user's own GGUFs (pulled, or in `models/`) are invisible to it. | No single "what do I have and can it run" view. |
| G8 | `pull` resolves `main`, not a commit. | "The file scout inspected" and "the file pulled" can differ. |
| G9 | The estimator and gate live in `Cli/Scout`, so a loader cannot call them without referencing the CLI. | The preflight cannot be shared. |

## 4. Principles and non-goals

- **Fail closed.** Unknown architecture semantics, unrecognised layout, unsupported dtype, or an **Unknown** memory estimate means *blocked*, with the reason. An override exists only as an explicit, loud flag.
- **Never substitute a look-alike.** A fallback may only choose another *qualified* entry of the **same model family**. Otherwise explain the constraint and stop.
- **Never download without asking** in an interactive session (existing `setup` behaviour). Non-interactive: print the command, do nothing.
- **Never accept a licence on the user's behalf.** Show it; the existing consent flag stays.
- **Claims are scoped.** Every recommendation carries what was qualified. "Qualified for CPU chat on engine X" does not imply Vulkan, vision, tools or speech.
- **Non-goals:** a second catalogue format; a model package format; a new downloader; automatic catalogue updates from the Hub; automatic admission; silent substitution; telemetry.

## 5. Design

### 5.1 Components and the workflow

```
 task command (chat, speak, ...)
   1. resolve   explicit flag > favourite > catalogue default          [favourites + catalogue]
   2. locate    is that exact file installed, intact?                   [local inventory]
   3. if absent remote-scout the pinned file; run the preflight         [scout -r + shared preflight]
                (size, memory at this context, backend, licence)
   4. show      checkpoint, download size, memory need, licence; ASK     [existing setup summary]
   5. acquire   pinned revision, resumable, sha256 verified             [existing installer]
   6. preflight again on the local file (cheap, index only)             [shared preflight]
   7. load only if allowed                                              [task command]
 if blocked at 3 or 6: offer the next QUALIFIED entry of the same family that fits, else explain.
```
Each box stays an independent, separately testable unit. The workflow is the only new code that knows all of them.

### 5.2 Data model (additive, no migration)

- **`CatalogEntry.FamilyId`**: links entries that are the same model at different quantisations, so a fallback never leaves the family.
- **`CatalogEntry.Qualifications`**: a list of `(Backend, Capabilities[chat|tools|vision|thinking...], MaxContextTested, EngineRevision, Evidence, Date, Result)`. The existing free-text `Evidence` stays for humans; the structured list is what code reads. An entry with no qualification for the current backend is shown as *unqualified here*, never as known-good.
- **Quant alternatives are catalogue entries**, not a new concept. This respects the "at most three alternatives, each with a user-level reason" rule ("fits in 8 GB", "better quality").
- **Favourites file** (`favourites.json`, in a per-user config directory, not the model home): `alias -> catalogue id`, or `alias -> {repo, revision, file, sha256}` for a model the user chose themselves. The second form is shown as **unqualified** and gets scout's dossier, never the "known-good" badge. Resolution never silently ignores a bad favourite: it errors with the fix.
- **Inventory** is computed, not stored: catalogue state plus a scan of the model home and `STINGRAY_MODEL_DIRS`, each file read by index only (scout, local) for architecture, admitted?, quant, size. `--verify` re-hashes (slow, opt-in) and reuses the existing `.sha256` sidecar convention (`ModelFingerprinter`).

### 5.3 The shared preflight (closes G4, G9)

Move the estimator and gate out of `Cli/Scout` into a library project (Engine, beside the planner, since it needs `ModelHyperparams` and the architecture registry). Contract: input = a GGUF index (local or remote), budget, reserve, backend, context; output = `Allowed | Blocked | Unknown` plus the reasons and the components. Scout, task commands, `run`, `serve` and `setup` all call it. One evaluator means scout can never say "fits" while the loader says otherwise. Backend scope is explicit: today CPU only; GPU stays unestimated and reported as such (CLAUDE.md rule 13).

### 5.4 Qualification workflow

An entry earns a qualification for scope S when: the architecture is admitted (static scout), the file is pinned and hash-verified, an automated test ran on those exact files for S (the existing rule), and a pretest receipt (`scripts/scout-pretest.ps1`) is recorded. Qualifications record the engine revision and **age**: a catalogue test fails when the evidence it cites no longer exists, and re-qualification is expected when a kernel the scope depends on changes. Near-tie numerical disagreement stays distinct from real divergence, and "it loaded" never promotes anything.

## 6. Phases

Order is by value per effort; each phase ships on its own and is useful alone.

| Phase | Work | Exit criteria |
|---|---|---|
| **P0 first-run seam** (small) | Task commands, when the entry is Missing/Partial and the terminal is interactive, run the same summary + consent + confirm + install path as `setup`, in-process, then continue. Non-interactive: print the one command. `pull --revision`; `pull` shares `HubClient`'s listing parser and the gateway (G8). | `stingray chat` on a clean machine reaches a reply after one confirmation; a test with a fake Hub proves no download happens without a yes. |
| **P1 shared preflight** | Section 5.3. Task commands call it before load; `setup` calls it on the remote file before downloading. | A model whose estimate is Unknown or over budget is refused before any load or download, with the reason; scout and the loader agree (a test pins that). |
| **P2 variants and qualification scope** | `FamilyId`, `Qualifications`; add two chat alternatives (a larger quality pick, a small-RAM pick) with real receipts; fallback within a family. Needs the **quant picker** (below) to choose them with evidence. | Every chat entry states what was qualified; a default that does not fit offers the qualified alternative, never an unqualified look-alike. |
| **P3 favourites** | `favourites.json`, `stingray models use <alias> <id>`, resolution order, unqualified badge for user-chosen models. | A user can set a different chat model once and it is used; a stale favourite fails with a clear fix. |
| **P4 inventory and verify** | `stingray models --local`, `--verify` (opt-in re-hash), integrity state shown (G5, G7). | One command answers "what do I have, is it intact, can this machine run it". |
| **P5 docs as tests** | README and `docs/STATUS.md` rows cite catalogue ids; a test fails when a cited file, hash or evidence disappears (front-door item 7). | The recommendation list cannot rot silently. |

**Depends on, in progress today:** remote scout (done), quant picker (next), signature harvest for admitted architectures without local files.

## 7. Risks and open questions

- **Where do favourites live?** Proposal: per-user config directory (`%APPDATA%\stingray`, `$XDG_CONFIG_HOME/stingray`), `STINGRAY_CONFIG` override. Needs the user's decision.
- **Keep "ask before downloading" the default for task commands?** Recommended yes. `--yes` already exists for scripts.
- **First qualification scope.** CPU chat only is the honest start. Vulkan and tool-calling get their own columns when someone runs them, not before.
- **Non-catalogue favourites and licences.** The Hub's licence field is a hint, not a guarantee; show it and require one explicit yes for anything not plainly permissive, as the catalogue does.
- **Catalogue growth.** Favourites are what lets users go beyond the catalogue without the catalogue growing past its "keep it small" rule.
- **Gated and private repos.** Report and stop; never accept terms for the user.
- **Hub API shape.** Remote feasibility depends on it; failure is loud and falls back to "download then preflight locally" only with the user's yes.
- **Re-qualification cost.** Who re-runs qualifications when kernels change, and how often, is a process decision; at minimum `verify-goldens` plus the catalogue's own tests in CI.
- **Estimator coverage.** Hybrid, MLA and RWKV models are Unknown (blocked) until modelled; a catalogue entry for such a family cannot pass the preflight yet, and should not be offered as a default.

## 8. The five requirements as acceptance criteria

| # | Requirement | Today | Done when |
|---|---|---|---|
| 1 | Exact checkpoint identity: repo, immutable revision, file, size, checksum, verified after download | **Exists** in the catalogue and installer | also true for user-chosen favourites (P3) and for `pull` (G8, P0) |
| 2 | Fail-closed preflight, the **same** evaluator in scout and loader | Scout side exists; loader does not call it | P1 |
| 3 | Resource safety: weights + KV at the requested context + scratch + headroom | Exists for plain-attention models on CPU | calibration extended to other families and backends, or reported Unknown |
| 4 | Qualification tied to a configuration | **Missing** | P2 |
| 5 | Layered validation: static, integrity, resources, then execution and goldens; near-tie kept distinct; loading is not qualification | Pieces exist; not wired to the catalogue | P2 and P5 |

**Residual risk, to be stated in the user docs rather than hidden:** a tensor index cannot prove the weights mean what they should; a checksum proves identity relative to a digest, not that the publisher's model is correct; a golden covers its inputs and configuration, not every prompt or path.

## 9. Open issues found in review (2026-10-10), to resolve in this order

| # | Issue | Fix | Status |
|---|---|---|---|
| R1 | `LoadPreflight.EvaluateCatalogEntryAsync` reads only `entry.MainFile`'s index, so a sharded model (qwen2.5-7b) is estimated from shard 1 alone, and the smaller-model offer inherits that. | Pass every GGUF shard of the entry to `RemoteGgufReader.ReadAsync` (it accepts a shard list). Regression test: both shards contribute to the combined index and the estimate. | done 2026-10-10: all GGUF shards of the bundle are read; test `Catalog_preflight_reads_every_shard_of_a_split_bundle` (fails with the fix reverted) |
| R2 | The golden format (`GoldenFile`) records one file name, size and SHA-256, and the catalogue guard checks only `MainFile`, so the 7B receipt does not pin both shards. | Extend the golden evidence to list all files of a split model; the guard validates the whole list against the catalogue. Needs a re-capture or migration of `qwen2.5-7b.golden.json`. | done 2026-10-10: `GoldenModel.Files` lists every shard, `CheckPin` verifies all, `capture-golden` records them, `qwen2.5-7b` golden re-captured (same result, NearTie 31/32), guard checks the list against the catalogue; `GoldenSplitPinTests` |
| R3 | `LocalInventory.Scan` opens every `*.gguf`; `GgufModel.Open` resolves sibling shards, so a complete split model shows twice (each row with a single shard's name and size). | Report a logical split model once: first shard's name and path, summed size; an incomplete set is one unreadable row naming the missing shards; other models unaffected. Reuse `GgufModel.ResolveShardPaths`. Tests with synthetic shards from `HubFixtures.Gguf`. | done (already in `LocalInventory.Scan`, with five split-model tests in `LocalInventoryTests`) |
| R4 | **Documentation sweep after the above.** Update every doc that now misstates the code: this plan's status line and section 5.3 (estimator now lives in `Engine/Scout`, namespace `OpenTail.Stingray.Engine.Scout`; `run`/`serve` wiring state), `docs/reference/061-coverage-tooling.md` (preflight, `models`, qualifications, `--local` split models), `docs/reference/delegating-to-a-smaller-model.md` (cards G done, new follow-up cards), `docs/STATUS.md` and `docs/RUNNING.md` (new Qwen2.5 rows with dated, measured speed and RAM), `docs/reference/cli-option-inventory.md` if options change, and the CLAUDE.md architecture topology if it mentions the Scout location. Do it last, once the code is settled, in one commit. | open |
