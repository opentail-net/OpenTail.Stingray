# Completed items split from 2026-10-08-golden-parity-and-admission-tooling-plan.md (2026-10-30)

Finished items moved verbatim out of [../2-coverage/2026-10-08-golden-parity-and-admission-tooling-plan.md](../2-coverage/2026-10-08-golden-parity-and-admission-tooling-plan.md). Open items stay there. Each block notes the section it came from.

> from: ## Phases (each ends: build clean with warnings as errors, Fast suites pass, one commit)
  - [x] 1.1 Implement `ModelLocator` with roots and the diagnostic text. Unit-test with temp directories (Fast suite).

> from: ## Phases (each ends: build clean with warnings as errors, Fast suites pass, one commit)
  - [x] 1.2 Replace the 25 `FindModel()` copies in `Tests.ForwardPass` (mechanical; the shape is identical). Net effect: about 35 hard-coded drive literals leave the repo.

> from: ## Phases (each ends: build clean with warnings as errors, Fast suites pass, one commit)
  - [x] 1.3 Measure (results below): re-run the heavy classes one process at a time and count classes that moved from **skipped to running** (today ~10 skip). Record the list in the progress doc; any newly running class that fails is a real finding, not a regression of this change.

> from: ## Phases (each ends: build clean with warnings as errors, Fast suites pass, one commit)
- [x] **2. Golden model and runner (no model needed to test it)** (done 2026-10-09)
  - [x] 2.1 `GoldenFile`, `GoldenJsonContext`, load/save; round-trip and schema-version tests; **the no-machine-paths guard test over `Goldens/`**.
  - [x] 2.2 `GoldenParityRunner` with a scripted stub `IForwardPass`: tests for `Exact`, `NearTie` (our top-2 equals the reference, margin under tolerance), `Diverged`, teacher-forced mode, first-divergence index, and the stepwise-vs-prefill check detecting an injected inconsistency.

> from: ## Phases (each ends: build clean with warnings as errors, Fast suites pass, one commit)
- [x] **3. Consumers and validation of the harness itself**
  - [x] 3.1 `GoldenParityTests` (heavy Theory) with the loud skip.
  - [x] 3.2 `admit-arch --golden`.
  - [x] 3.3 (done 2026-10-09: SmolLM3, Olmo2, Gpt2, GptNeoX, Qwen2Moe and Mixtral are all converted and agree with the legacy verdicts) **Validate before trusting:** (a) convert 6 existing receipts of different kinds (SmolLM3, Olmo2, Gpt2, GptNeoX, Qwen2Moe, Mixtral teacher-forced) into goldens by hand-copying their arrays; the harness must reproduce each legacy class's verdict; (b) negative tests: change one token (must report `Diverged`), swap to our top-2 at a small margin (must report `NearTie`); (c) confirm known real cases: DeepSeek2 reports its token-9 divergence (bugstofix #24), and the four classes that print a near-tie report (Afmoe, GlmMoe, HunyuanMoe, LlamaFour) get the same counts from the shared runner.

> from: ## Phases (each ends: build clean with warnings as errors, Fast suites pass, one commit)
- [x] **4. `capture-golden`** (done 2026-10-09)
  - [x] 4.1 Implementation per Design D (process management, free-port selection, kill-tree, timeouts).
  - [x] 4.2 (done 2026-10-09, see results) **Validate the capture itself:** (a) capture twice on SmolLM2-135M, tokens must be byte-identical (determinism at `-t 4`; if not, record and document the thread-count sensitivity); (b) capture for a model that has a pasted legacy array (SmolLM3 or Olmo2) and compare token for token with the legacy array; (c) capture with `--prompt-ids` and with `--prompt` and check `llama-tokenize` ids equal the ids our own tokenizer produces for the same text (feeds the tokenizer-oracle fact).
  - [x] 4.3 (done 2026-10-09: missing server tool, a server that exits immediately, and a 5 s load timeout on the 12 GB model each exit 1, write no file and leave no orphan llama-server; a busy port is an unreproducible race and Ctrl-C is not automatable here, so both are untested) Failure paths tested: server fails to start, port busy, tool missing, Ctrl-C leaves no orphan `llama-server`.

> from: ## Phases (each ends: build clean with warnings as errors, Fast suites pass, one commit)
- [x] **5. Hash-pinned receipts** (done 2026-10-09)
  - [x] 5.1 `lfs.oid` capture and post-download verification in `pull` (+ `stingray hash`).
  - [x] 5.2 `ModelFingerprint` with the sidecar cache; harness `UnpinnedFile` behaviour.
  - [x] 5.3 Tests: hash of a small file against a known SHA-256; mismatch reporting; cache invalidated by size or mtime change.

> from: ## Phases (each ends: build clean with warnings as errors, Fast suites pass, one commit)
- [x] **6. Migrate the legacy receipts**
  - [x] 6.1 (done 2026-10-09, see the progress table; classes deliberately kept are listed after it) Convert in batches of ~5. Each batch: golden created by `capture-golden` where the checkpoint is on disk (and compared with the pasted array first), otherwise copied from the array and marked `reference.engine = "legacy-receipt"`; old class deleted only once the golden version passes with the same verdict.
  - [x] 6.2 Move each class's provenance prose into `notes`.
  - [x] 6.3 Update `docs/reference/adding-an-architecture.md` step 4 to the real sequence: `pull` -> `capture-golden` -> `admit-arch --golden` -> commit the golden with the descriptor.

> from: ## Phases (each ends: build clean with warnings as errors, Fast suites pass, one commit)
- [x] **7. Parity verification & baseline diffing** (done 2026-10-09)
  - [x] 7.1 `stingray verify-goldens [--dir] [--golden] [--baseline] [--diff] [--strict]`: run every golden whose model is present, verify pins and hyperparameter guards, print a Spectre.Console summary table, write/diff per-host baselines (model, arch, verdict, matched tokens, speed t/s, elapsed time, host environment).
  - [x] 7.2 Batched-versus-serial and retained-prefix checks: completed inventory of existing coverage across `ContinuousBatchingTests`, `ContinuousBatchingRetainedStateTests`, `BatchInvariantGemmTests`, `DecodeViaGemmInvarianceTests`, and `HotSession*Tests`.

