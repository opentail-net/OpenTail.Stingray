# Golden parity harness, automatic reference capture and hash-pinned receipts (plan)

**Date:** 2026-10-08. **Status:** proposed, not started.
**Origin:** a review of `examples/TensorSharp/TensorSharp` (its `ParityHarness`, `.parity` goldens, release catalog and `TestMatrix`) for anything that helps *admit and verify* models. This plan adopts the three ideas that are worth it and adapts them to what Stingray already has. It does not touch the descriptor work (see [2026-10-08-architecture-capabilities-plan.md](2026-10-08-architecture-capabilities-plan.md)); the two meet only in `docs/reference/adding-an-architecture.md` step 4.

**What this does not change:** the expensive part of admission is still a real checkpoint plus an independent reference plus a STATUS row (CLAUDE.md rules 10, 12, 14). This plan removes the *mechanical* cost around it (hand-pasted token arrays, 25 copies of a model finder, silent skips, a manual capture step, unpinned files) and makes the evidence stronger. Nothing here replaces judgement about a divergence (see [numerics-investigation-method.md](../reference/numerics-investigation-method.md): reproducible is not accurate).

## What TensorSharp does, and what we take

| TensorSharp | Verdict |
|---|---|
| `ParityHarness parity <model> --ref golden.json`: raw token ids in, greedy output compared with a recorded `llama.cpp` golden; one generic harness, a JSON file per model (`.parity/ref_text.json`) | **Adopt** (Phases 2-3, 6) |
| `gen_ref.py` captures the golden from a running `llama-server` | **Adopt the idea, in C#** (Phase 4). Python is excluded by project rule. |
| `release-model-catalog.json` pins SHA-256 per file | **Adopt** for receipts (Phase 5) |
| `TestMatrix` (discover GGUFs, filter by backend, per-host baselines); `ParityHarness --batched / --retained-continuation / --arena-zero-reuse` | **Maybe later** (Phase 7) after Phases 1-6 prove useful |
| `engine_comparison` (OpenAI-HTTP harness vs llama.cpp and vLLM), ~100 Python scripts for release/agent/browser/image-mask workflows | **Not adopted** |

## Facts this plan rests on (each checked against the repo or by running the tool, 2026-10-08)

1. **25 `*GreedyParityTests` classes** in `tests/OpenTail.Stingray.Tests.ForwardPass` (counted by script). **All 25 carry their own copy of `FindModel()`.** 23 build the dense `Engine.ForwardPass`, 2 build the RWKV passes, and one (DeepSeek2) needs special setup: `SimdKernels.Q8PrefillEnabled = true` for its receipt. At least 13 have an inline expected-token array. The facts they contain, by name: 12 classes pair the llama.cpp receipt (`GreedyContinuation_MatchesLlamaCpp`) with **`DecodeStepwise_AgreesWithSinglePassPrefill`, a model-independent self-consistency check**; 7 use a **teacher-forced** short/long-prompt pair against `llama-server` (the Mixtral family); the rest are one-offs (`TopCandidates_AtDivergence`, ...). The prose in each file's doc comment is the provenance (checkpoint, licence, date, caveats).
2. **Each class hard-codes its own list of model folders, and the lists disagree.** Of the 25: 12 search the repo's `models/` and `models/_models/` plus `E:\models` only, 7 also search `H:\_models` and `K:\_other_models`, 3 add `K:\_other_models` only, and 3 hard-code no drive. (`models\_models` is a symlink to `F:\_models`, which is why `F:` works everywhere.) So whether a checkpoint is found depends on which class asks, and a miss skips silently (rule 12). Filename drift also skips: GptNeox wanted `pythia-160m-Q8_0.gguf`, the disk has `pythia-160m.Q8_0.gguf`. Those drive literals are **already checked in**: 491 lines in 242 test files repo-wide contain a drive-letter path, and 20 tracked files contain a Windows user-profile path.
3. **`admit-arch` tokenizes the prompt with our tokenizer** and inserts BOS itself, so its verdict includes any tokenizer difference. The reference tokens must be captured **by hand** (`--reference-tokens`; the command tells the user to run `llama-server`).
4. **The vendored `tools/llama.cpp/llama-server.exe` (build 10306, `6b5c2efb4`) does what the plan needs.** Verified by running it on SmolLM2-135M: `POST /completion` with `"prompt":[ids]`, `temperature:0, top_k:1, seed:0, cache_prompt:false, return_tokens:true` returns `"tokens":[...]`, `tokens_evaluated` equal to the id count (so **no BOS is injected for an id array**), and echoes `generation_settings`.
5. **`tools/llama.cpp/llama-tokenize.exe --ids [--no-bos] -p "..."` prints the independent prompt ids** (verified: `[504, 3575, 282, 4649, 314]` for "The capital of France is" on SmolLM2).
6. **`pull` records no integrity data.** It reads each file's `lfs.size` from the Hugging Face tree API (`PullCommand.cs:162`) but not `lfs.oid` (the SHA-256) and does not verify after download.
7. The "near-tie vs real divergence" reasoning is **re-implemented per test** (e.g. `CudaForwardPassKvDtypeTests`; the "N confident positions matched, M near-tie differences of K" report is printed by the Afmoe, GlmMoe, HunyuanMoe and LlamaFour classes, each with its own copy of the logic), and `Tests.Cuda` has its own `FindModelPath` copies too.
8. Conventions to follow: one source-generated `JsonSerializerContext` per feature (`DoctorJsonContext`, `StaticPlanJsonContext`, ...), and `ImageCommand` already spawns an external tool (`sd-cli`) with `ProcessStartInfo`.

## Design

### A. The golden file
One JSON file per (model file, set of cases), checked in under `tests/OpenTail.Stingray.Tests.ForwardPass/Goldens/<architecture>.golden.json` (small text; generated text is a few dozen tokens). Schema v1:
```
{ "schema": 1, "architecture": "olmo2", "notes": "<the provenance prose that today lives in the test's doc comment>",
  "model":     { "fileName": "OLMo-2-0425-1B-Q8_0.gguf", "sizeBytes": 1580000000, "sha256": "...", "source": "allenai/OLMo-2-0425-1B-GGUF" },
  "reference": { "engine": "llama-server", "build": "10306 (6b5c2efb4)", "args": "-ngl 0 -c 512 -t 4",
                 "sampling": { "temperature": 0, "top_k": 1, "seed": 0, "repeat_penalty": 1.0, "cache_prompt": false },
                 "capturedUtc": "2026-10-08T..." },
  "cases": [ { "name": "capital", "promptText": "The capital of France is", "promptTokens": [..], "tokenizerOracle": "llama-tokenize",
               "nPredict": 24, "tokens": [..], "text": "...", "mode": "free" | "teacherForced" } ],
  "engineSettings": { } }
```
`promptTokens` are authoritative: our tokenizer is not in the loop for the receipt. `promptText` + `tokenizerOracle` let a *separate* check compare our tokenizer with the oracle (one fact, one reason to fail). `engineSettings` carries named overrides for the rare model that needs them (DeepSeek2: `q8Prefill: true`); anything not expressible stays a legacy class, documented as an exception.

### B. Verification library (`OpenTail.Stingray.Engine.Verification`, no test dependencies)
- `GoldenFile` + `GoldenJsonContext` (source-generated).
- `ModelLocator.Find(params string[] fileNames)`: searches, in order, `STINGRAY_MODEL_DIRS` (path list, new), the repo `models/` and `models/_models/`, then every fixed drive's `_models`, `_other_models`, `models` folders, then `E:\models`. Returns the path and the roots searched. Replaces the 25 copies, and the Cuda copies in Phase 1b.
- `GoldenParityRunner`: for each case, feeds `promptTokens`, then
  - **free mode:** greedy continuation; **teacher-forced mode:** feed the reference tokens and compare our argmax at each position;
  - at every mismatch it records our top-1/top-2 token and **logit margin**, and classifies the case as `Exact`, `NearTie` (our top-2 equals the reference token and margin below a configured tolerance) or `Diverged`;
  - always also runs the **decode-stepwise vs single-pass prefill** self-check (fact 1) on the same prompt;
  - returns a structured result (first divergence index, counts, margins). No asserting inside the library.
- Hash handling (Phase 5): `ModelFingerprint` (size + SHA-256 with a size/mtime-keyed sidecar cache so a 12 GB file is hashed once).

### C. Consumers
1. **`GoldenParityTests`** (heavy; one Theory case per golden file; inherits `HeavyTestBase`). A missing model produces a **loud, specific skip**: expected file name, the roots searched, and the golden's `source`.
2. **`stingray admit-arch --golden <file>`**: uses the golden's token ids (no tokenizer in the loop), prints the existing ADMIT/REJECT verdict and the pasteable receipt block, now with near-tie classification and the golden's provenance.
3. **`stingray capture-golden`** (Phase 4), below.

### D. `capture-golden` (C#, no Python)
`stingray capture-golden -m <gguf> --prompt "text" [--prompt-ids 1,2,3] [--n 24] [--mode free|teacher] [--out path] [--server <exe>] [--threads 4]`:
1. Fingerprint the model (size, SHA-256).
2. Tokenize with the **vendored `llama-tokenize --ids`** (default BOS behaviour of the model, as `llama-server` would for text; record `addBos`). If `--prompt-ids` is given, use them as-is.
3. Start `llama-server` on a free port (`-ngl 0 -c <ctx> -t <threads>`), wait for `/health`, `POST /completion` with the id array and the greedy sampling above, read `tokens`, `generation_settings`, then **kill the process tree** (also on failure and on Ctrl-C).
4. Write the golden JSON with the build string (`--version`), args and sampling echoed back by the server, the date, and the model fingerprint.
5. Print what it captured and a reminder: a golden is evidence only for *this* file and *this* `llama.cpp` build.
Refuses clearly if `tools/llama.cpp` binaries are absent; `--server` overrides the path.

### E. Hash-pinned receipts
- `pull` records `lfs.oid` from the tree API, verifies the downloaded file's SHA-256 against it, and reports a mismatch loudly. (Resumed or pre-existing files are verified on demand with a new `stingray hash <file>` helper, or by `capture-golden`.)
- The harness compares the located file's fingerprint with the golden's `sha256`. On mismatch it **does not skip**: it runs, labels the verdict `UnpinnedFile (not the verified file)`, and prints both hashes. This turns the pythia dot-versus-dash ambiguity into an explicit statement.

## What we do better than TensorSharp (deliberately)
1. The prompt ids come from the **independent oracle tokenizer**, and our tokenizer is then checked against it as a separate fact, so a tokenizer bug and a forward-pass bug cannot hide behind each other.
2. Mismatches are **classified by logit margin** (exact / near-tie / diverged), not a binary pass/fail, which matches how we already reason about drift.
3. **Provenance is structured** (build, args, sampling, hash, date) instead of prose in a doc comment.
4. Model discovery spans **all drives** and is configurable; skips are **loud**.
5. The classification logic is **unit-testable without a checkpoint** (scripted stub forward pass) and so can live in the Fast suite.
6. The stepwise-decode-vs-prefill self-check runs for **every** golden for free.

## Phases (each ends: build clean with warnings as errors, Fast suites pass, one commit)

- [ ] **0. Decisions** (this document). D1: golden location (`tests/.../Goldens/`, recommended, so reviewers see them in the diff). D2: near-tie margin tolerance (start 0.02 logit, configurable, recorded in the result; revisit with data from the 25 migrated receipts). D3: SHA-256 mandatory for new goldens (recommended yes), optional for migrated ones until each file is re-hashed. D4: migrate all 25 legacy classes over time, keeping exceptions (DeepSeek2 until `engineSettings` covers it).
- [x] **1. `ModelLocator` and loud skips** (done 2026-10-09)
  - [x] 1.1 Implement `ModelLocator` with roots and the diagnostic text. Unit-test with temp directories (Fast suite).
  - [x] 1.2 Replace the 25 `FindModel()` copies in `Tests.ForwardPass` (mechanical; the shape is identical). Net effect: about 35 hard-coded drive literals leave the repo.
  - [x] 1.3 Measure (results below): re-run the heavy classes one process at a time and count classes that moved from **skipped to running** (today ~10 skip). Record the list in the progress doc; any newly running class that fails is a real finding, not a regression of this change.
  - [ ] 1.b (optional) Point `Tests.Cuda` / other `FindModelPath` copies at the same locator.
- [x] **2. Golden model and runner (no model needed to test it)** (done 2026-10-09)
  - [x] 2.1 `GoldenFile`, `GoldenJsonContext`, load/save; round-trip and schema-version tests; **the no-machine-paths guard test over `Goldens/`**.
  - [x] 2.2 `GoldenParityRunner` with a scripted stub `IForwardPass`: tests for `Exact`, `NearTie` (our top-2 equals the reference, margin under tolerance), `Diverged`, teacher-forced mode, first-divergence index, and the stepwise-vs-prefill check detecting an injected inconsistency.
- [ ] **3. Consumers and validation of the harness itself**
  - [x] 3.1 `GoldenParityTests` (heavy Theory) with the loud skip.
  - [x] 3.2 `admit-arch --golden`.
  - [ ] 3.3 **Validate before trusting:** (a) convert 6 existing receipts of different kinds (SmolLM3, Olmo2, Gpt2, GptNeoX, Qwen2Moe, Mixtral teacher-forced) into goldens by hand-copying their arrays; the harness must reproduce each legacy class's verdict; (b) negative tests: change one token (must report `Diverged`), swap to our top-2 at a small margin (must report `NearTie`); (c) confirm known real cases: DeepSeek2 reports its token-9 divergence (bugstofix #24), and the four classes that print a near-tie report (Afmoe, GlmMoe, HunyuanMoe, LlamaFour) get the same counts from the shared runner.
- [x] **4. `capture-golden`** (done 2026-10-09)
  - [x] 4.1 Implementation per Design D (process management, free-port selection, kill-tree, timeouts).
  - [x] 4.2 (done 2026-10-09, see results) **Validate the capture itself:** (a) capture twice on SmolLM2-135M, tokens must be byte-identical (determinism at `-t 4`; if not, record and document the thread-count sensitivity); (b) capture for a model that has a pasted legacy array (SmolLM3 or Olmo2) and compare token for token with the legacy array; (c) capture with `--prompt-ids` and with `--prompt` and check `llama-tokenize` ids equal the ids our own tokenizer produces for the same text (feeds the tokenizer-oracle fact).
  - [x] 4.3 (done 2026-10-09: missing server tool, a server that exits immediately, and a 5 s load timeout on the 12 GB model each exit 1, write no file and leave no orphan llama-server; a busy port is an unreproducible race and Ctrl-C is not automatable here, so both are untested) Failure paths tested: server fails to start, port busy, tool missing, Ctrl-C leaves no orphan `llama-server`.
- [x] **5. Hash-pinned receipts** (done 2026-10-09)
  - [x] 5.1 `lfs.oid` capture and post-download verification in `pull` (+ `stingray hash`).
  - [x] 5.2 `ModelFingerprint` with the sidecar cache; harness `UnpinnedFile` behaviour.
  - [x] 5.3 Tests: hash of a small file against a known SHA-256; mismatch reporting; cache invalidated by size or mtime change.
- [ ] **6. Migrate the legacy receipts**
  - [ ] 6.1 Convert in batches of ~5. Each batch: golden created by `capture-golden` where the checkpoint is on disk (and compared with the pasted array first), otherwise copied from the array and marked `reference.engine = "legacy-receipt"`; old class deleted only once the golden version passes with the same verdict.
  - [ ] 6.2 Move each class's provenance prose into `notes`.
  - [ ] 6.3 Update `docs/reference/adding-an-architecture.md` step 4 to the real sequence: `pull` -> `capture-golden` -> `admit-arch --golden` -> commit the golden with the descriptor.
- [ ] **7. Optional, only if Phases 1-6 prove useful**
  - [ ] 7.1 `stingray verify-goldens [--dir]`: run every golden whose model is present, print a table, write a **per-host baseline** (pass / near-tie / diverged / skipped and decode t/s) so regressions like today's Qwen3.5 crash and the Phi-3 prefill drop appear as a diff. First compare with `TestMatrix`'s baselines and with the ad hoc sweep scripts used on 2026-10-08.
  - [ ] 7.2 Batched-versus-serial and retained-prefix checks (TensorSharp's `--batched`, `--retained-continuation`): first inventory what `ContinuousBatchingTests` and the session tests already assert, then add only the missing invariants.

## Risks and how they are handled
| Risk | Handling |
|---|---|
| Harness bug makes every receipt pass or fail | Phase 3.3 validation: legacy verdicts reproduced, negative tests, known real cases (DeepSeek2 and the four near-tie classes) |
| `llama.cpp` greedy is not bit-stable across threads or builds | Golden pins build, args and thread count; capture determinism test (4.2a); near-tie class absorbs legitimate flips; regenerate on purpose, never silently |
| A golden proves nothing about a different file | SHA-256 pin and the explicit `UnpinnedFile` verdict |
| Hashing 12 GB per run is slow | Sidecar cache keyed by size and mtime; hash only on first use |
| Orphaned `llama-server` (RAM held) | Kill the process tree in `finally`, on Ctrl-C and on timeout; test it (4.3) |
| Capture and tests together exhaust RAM | Documented: run one heavy process at a time (the same rule as the heavy suites); `capture-golden` takes the same machine-wide mutex as `HeavyTestBase` |
| `tools/llama.cpp` absent in another clone | Clear error naming the missing file and the `--server` override |
| Scope creep into a model store / manifest system | Phase 5 only records and checks hashes; `pull`'s documented non-goal (no model-store/alias system) stays |
| New test code replaces proven receipts too early | Phase 6 deletes a legacy class only after the golden version matches its verdict |
| Python crept in | None planned; capture is C# (project rule) |

## Out of scope
Automating STATUS rows; replacing the need for a real checkpoint and reference; GPU (CUDA/Vulkan) golden runs (CPU first; the runner takes a forward-pass factory so GPU can follow); audio, diffusion and vision verification (separate pipelines with their own oracles); the TensorSharp Python tooling.

## Order of value
Phase 1 pays immediately (fewer silent skips, 25 duplicates gone). Phases 2-3 give the harness and its proof. Phase 4 removes the manual capture step in `admit-arch`. Phase 5 makes receipts unambiguous. Phase 6 is bulk migration. Phase 7 is optional.

## What travels with the repo, and what does not
- **Goldens travel; checkpoints never did.** A golden is a small text file (one or a few prompts, a few dozen token ids, provenance): roughly 1-3 KB each, so even all ~70 admitted families come to well under 200 KB. They are committed beside the tests and move with `git` exactly as the pasted `int[]` arrays in the 25 classes do today. Nothing new has to be dragged around.
- **What a machine needs to *run* a golden:** the checkpoint (as now), and nothing else. No `llama.cpp`, no reference run: the reference output is in the golden. That is the point of recording it.
- **What a machine needs to *create* a golden:** the vendored `tools/llama.cpp` binaries (`llama-server`, `llama-tokenize`) and the checkpoint. Only the person admitting a new family does this, once per file.
- **A golden is tied to one file.** It pins the SHA-256 (Phase 5). On another machine with a different quantization or conversion of the "same" model, the harness runs but labels the verdict `UnpinnedFile` instead of silently skipping or silently trusting, so a golden never claims more than it verified.
- **If a clone lacks a checkpoint** the golden's test skips loudly (file name, roots searched, source repo to `pull` from), exactly the situation the current silent skips hide.

## No machine paths in checked-in files (answering "would we check your drive paths into git?")
**No: this plan removes drive literals; it must not add any.**
- **Goldens hold identity, not location:** `model.fileName` (a bare name), `sizeBytes`, `sha256`, `source` (a Hugging Face repo id), the `llama.cpp` build string, a *whitelisted* set of server settings (`-ngl`, `-c`, `-t`), sampling parameters, token ids and text. No directory, no drive letter, no host or user name.
- **Where the leaks would come from, and how each is closed:**
  1. `llama-server`'s response contains `"model": "<the path it was started with>"` (verified 2026-10-08: it echoes `models/_models/...`; with an absolute path it would echo `F:\...`). `capture-golden` **never copies that field**; it records `fileName` only.
  2. The server command line contains `-m <absolute path>`. The golden stores **named settings**, not a raw command line.
  3. Human-written `notes`. Covered by the guard below.
  4. `ModelLocator` roots: the checked-in code names only **folder names** (`_models`, `_other_models`, `models`) and enumerates fixed drives **at run time**; per-machine locations come from the `STINGRAY_MODEL_DIRS` environment variable, which never enters git.
- **Guard test (Fast suite, added with Phase 2):** scans every file under `Goldens/` and fails on a drive-letter path (`X:\` or `X:/`), a UNC path, `/Users/`, `/home/`, or the current user/machine name. It runs on every `dotnet test`, so a leak is caught before it is committed.
- **Phase 1.2 is a net clean-up:** replacing the 25 `FindModel()` copies deletes about 35 hard-coded `E:\models` / `H:\_models` / `K:\_other_models` literals from the parity classes.
- **Not part of this plan, worth a separate item:** the other ~240 test files with drive literals and the 20 tracked files that contain `C:\Users\Dmitri`. A ratchet like the architecture-literal one (count may only go down) would stop new ones; the existing ones need an owner decision.

## Phase 1 results (2026-10-09)
- `Engine/Verification/ModelLocator` + 4 Fast tests (priority of names, miss text, `STINGRAY_MODEL_DIRS`, ancestor `models/_models`). The 25 `FindModel()` copies in `Tests.ForwardPass` are now one-line calls (24 files, -422/+24 lines, plus GraniteHybrid's variant); no drive-letter literal remains in the parity classes. `STINGRAY_MODEL_DIRS` registered (inventory 271).
- **Measured effect on skips: none.** Run per class with `STINGRAY_RUN_HEAVY_TESTS=1`: 10 classes find their checkpoint and start real work (Afmoe, DeepSeek2, Exaone45, GlmMoe, GraniteHybrid, HunyuanMoe, Mixtral, MixtralStyle, PhiMoe, Qwen2Moe) plus GptNeox and SmolLm3 and part of Olmoe; **13 still skip, because the checkpoint is genuinely not on any drive under the wanted name.** The old hard-coded lists were not hiding files that the new search finds. The gain is that a miss is now loud: `[ModelLocator] NOT FOUND: <names>. Searched N folder(s): ...`.
- What the missing ones want, and what exists: Apertus `Apertus-8B-Instruct-2509-Q4_K_M.gguf` (disk has a *different conversion*, `swiss-ai.Apertus-8B-Instruct-2509.Q4_K_M.gguf`, 4.7 GB vs the receipt's 5.06 GB); Gpt2 `gpt2-f16.gguf` (disk: Q8_0); Maincoder `maincoder-1b-Q8_0.gguf` (disk: Q4_K_M); Falcon `falcon-7b-instruct-Q4_K_M.gguf` (disk: Falcon3-3B); Olmo2 `OLMo-2-0425-1B-Q8_0.gguf`, Olmo `olmo-1b-Q8_0.gguf`, Glm4 `THUDM_GLM-4-9B-0414-Q4_K_M.gguf`, LlamaFour (93 GB): absent.
- **Tried the Apertus alias; it fails** (" Paris.\n France is a country..." vs the receipt's " Paris, which is also the country's largest city."). That is a different file, not an engine regression, and exactly the case Phase 5's SHA-256 pin exists for. The alias was reverted: a receipt must not silently run on another conversion.

## Phase 2-4 results and design corrections (2026-10-09)
Built: `Engine/Verification/{GoldenFile, GoldenParityRunner, GoldenForwardPassSource, GoldenCaptureParsing}`, `GoldenParityTests` (heavy, data-driven), `admit-arch --golden`, `capture-golden`; 31 Fast tests (runner with a scripted forward pass, capture parsers, machine-path guard, locator); 4 checked-in goldens (`SmolLM2-135M`, `SmolLM3`, `OLMoE`, `pythia-160m`). CLI option inventory regenerated (241).

**Two corrections that real models forced (both are in the code and tested):**
1. **The stepwise-vs-prefill argmax equality is not a useful gate.** On SmolLM2-135M decode and batched prefill differ by at most 0.96 logits (inside the existing bound) yet flip the argmax at a near-tie (gap 0.08). If two paths differ by at most D everywhere and their argmax differs, the two tokens were within 2D of each other, so a flip is *always* explained by the measured difference and adds nothing beyond the bound on D. The self-check therefore gates on the difference bound; argmax agreement is reported ("informational") and enforceable via `ParityOptions.RequireStepwiseArgmaxAgreement` for anyone who wants the legacy strictness.
2. **Near-tie classification uses the reference's own confidence, not only ours.** The legacy teacher-forced classes already did this by hand (`ConfidentMargin = 1.5` nats against llama-server's recorded per-position margins). Goldens now carry `margins` (top-1 minus top-2 log-probability from `n_probs`, verified to be real log-probs in build 10306); a mismatch where the reference margin is below `ConfidentMargin` is a near-tie, as is one within our own logit-gap tolerance.

**Validation against real checkpoints:**
- Capture is deterministic: two captures of SmolLM2-135M gave identical tokens (`-t 4`). No orphan `llama-server` after a capture. Our tokenizer agreed with `llama-tokenize` on the prompt.
- **SmolLM3: the independent capture reproduced the legacy receipt text exactly** (`" Paris. The Eiffel Tower is a famous landmark in Paris. The Eiffel Tower was built for the "`), 24/24 by the harness.
- **pythia-160m (GptNeox): near-tie at position 22 where llama.cpp's own margin was 0.002 nats** (a coin flip); this is exactly the late divergence the legacy receipt documents and why it asserts only a prefix. The harness now states it precisely.
- **OLMoE: near-tie at position 18 (reference margin 0.25) after 18 exact tokens** against build 10306. The legacy receipt's reference text differs from this fresh capture from token 2 (and that class carries an `Olmoe_TopCandidates_AtDivergence` fact), so the older reference diverged from our engine where the newer build agrees for 18 tokens; not investigated further.
- **Negative test on a real model:** corrupting one recorded token makes `admit-arch --golden` exit 1 with `Diverged, 10/11 matched; @10 expected 12345 got 22387 gap 23.2`. A malformed golden is rejected with a clear message.
- Heavy `GoldenParityTests`: 4 goldens, 35.6 s, all pass on real weights.

Still open in Phases 3-4: the teacher-forced cross-check on a Mixtral-style model and the known real divergences (DeepSeek2 token 9), and 4.3 (failure paths: server fails to start, port busy, tool missing, Ctrl-C leaves no orphan).

## What the new structure could lose, and the corrections (asked 2026-10-09: "are we losing anything?")
Nothing has been deleted yet: the 25 legacy classes still exist and keep running. These are the places a naive migration would be **worse**, and what the plan now requires:
1. **Extra assertions in the legacy classes are not covered by a generic harness.** Examples: `Assert.Equal(16, hp.RopeDim)` (GptNeox partial RoPE), `SharedExpertIntermediateDim == 2816` (DeepSeek2), the GraniteHybrid cache tests (5 facts), `*_TopCandidates_AtDivergence`, Mixtral's long-prompt batched-prefill case. **Rule for Phase 6: a legacy class is deleted only after every assertion in it is either reproduced by the golden harness or moved to a small dedicated test.** Receipt-only classes can go; classes with architecture-specific guards keep those guards.
2. **The tokenizer is deliberately out of the receipt loop, but the legacy classes also asserted `tokenizer.Encode(prompt) == ids`.** That coverage must not vanish: a `GoldenTokenizerParity` Theory (our `Encode(promptText)` + the model's BOS rule against the golden's oracle ids, one reason to fail) is added before the first legacy class is deleted.
3. **Near-tie is a pass, which is looser than the legacy exact-text asserts.** To keep it strictly *more* informative than before: in free mode, after a near-tie mismatch the runner continues the comparison **teacher-forced** from the reference tokens for the rest of the continuation, so a near-tie at position 22 no longer hides what happens at 23. Implemented as part of Phase 6 prerequisites.
4. **References move to a newer llama.cpp build** (10306), so a migrated model may report a near-tie where the old receipt passed exactly (OLMoE did). Each migration records old versus new outcome in the plan; where the old reference was captured on another build it is kept as a second case (`reference.engine = "legacy-receipt"`), not discarded.
5. **Provenance prose** (checkpoint, licence, why a caveat exists) is long in the doc comments and short in `notes`. Migration copies it in full; a golden's `notes` is allowed to be long.
6. **New moving parts:** goldens are tied to one file by SHA-256 (a different conversion reports `UnpinnedFile`, as intended), creating one needs `tools/llama.cpp`, and the harness is code that can itself be wrong. Mitigated by the negative tests and by cross-checking against legacy receipts before trusting it.
7. **Not lost:** the model-independent stepwise-vs-prefill check (now run for every golden), the heavy-suite machine-wide gate, the skip-when-missing behavior (now loud), and every legacy test until its replacement is proven.
Where the new structure is clearly **better**: one finder instead of 25, structured provenance and hash pinning, near-tie reasoning that uses the reference's own confidence, goldens that can be reviewed in a diff and regenerated by one command, and the harness logic being unit-tested without a checkpoint.

## Phase 5 results (2026-10-09)
- **Correction to this plan, found by calling the real API:** the default Hugging Face model response carries neither file sizes nor hashes; `?blobs=true` returns `size` and **`lfs.sha256`** (the field is `sha256`, not `oid`). `pull` now requests `?blobs=true`, records each file's published SHA-256, and verifies the downloaded file (also when the file was already complete). Mismatch -> exit 1 with a clear message; no published hash -> "not verified" note. Real check: `pull -r bartowski/Qwen2.5-0.5B-Instruct-GGUF -q Q4_K_M` downloaded 379 MiB and printed `SHA-256 verified (6eb923e7d26e...)`.
- `ModelFingerprinter` (Engine/Verification): SHA-256 + size, cached in `<file>.sha256` keyed by size and modification time (a change to either, a corrupt sidecar, or an unwritable folder all fall back to hashing); `CheckPin(golden, path)` -> `Verified | Mismatch | NotRecorded`, shown by `admit-arch --golden` and by `GoldenParityTests` (a mismatch labels the result `UNPINNED` instead of skipping or silently trusting). New `stingray hash -m <file>` prints the fingerprint (second call: `(cached)`). `capture-golden` uses the same cache.
- Tests: 4 fingerprint tests (reuse, invalidation, unwritable sidecar, corrupt sidecar, three pin statuses) and 2 `pull` listing tests using the real response shape. CLI inventory regenerated (242 options). Suites: Tests.Cli 443, ForwardPass.Fast 1091, focused Core classes: 0 failed.

## Phase 6 progress (migration of the legacy parity classes)
**Audit of what the legacy classes assert beyond the receipt (2026-10-09)** found only a few recurring kinds, so they became generic golden features instead of ~12 near-duplicate classes:
`architecture` string (checked for every golden), `expectedHyperparameters` (ModelHyperparams property -> invariant value, e.g. `ropeDim=16`, `numExperts=8`, `hasFfnBias=true`; an unknown name fails, so a typo cannot pass), and `minConfident` per case (the "too few confident positions to be evidence" rule of the teacher-forced classes). `capture-golden --expect ... --min-confident N` records them. Not migrated by design: classes with architecture-specific facts the harness does not model (GraniteHybrid's Mamba cache tests, DeepSeek2's special setup, Olmoe's `TopCandidates_AtDivergence`).

| Batch | Classes | Outcome |
|---|---|---|
| A (2026-10-09) | SmolLm3, GptNeox | **Migrated and deleted.** SmolLM3: exact 24/24 (identical to the legacy text). GptNeox: near-tie at position 22 (reference margin 0.002), **the remaining positions are still compared** (23/24 matched) which the legacy prefix assertion could not do; its five hyperparameter guards (`ropeDim=16`, `xieluAlphaN=null`, `hasFfnBias`, `hasNormBias`, `useParallelResidual`) moved into the golden; provenance prose copied into `notes`; descriptor comments now cite the golden path. |
