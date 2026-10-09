# Checkpoint Scout and AI-Assisted Architecture Admission

**Status:** Proposed implementation plan (2026-10-09). Drafted by another AI; reviewed against HEAD (all named files/commands exist). **Committed scope: Phases 1–2 plus a short playbook and the optional pretest script (§8.1). Phases 4–5 are deferred, see §8.1.**  
**Target:** .NET 10 / C# 14, NativeAOT-safe, local-first  
**Primary constraint:** Make useful progress on the existing 64 GB RAM development machine; do not make oversized real-weight runs a prerequisite for the scout itself.  
**Scope:** GGUF-first checkpoint inspection and admission assistance. SafeTensors/package inspection remains a related, separate profile system unless an explicit later phase justifies unifying them.

## 1. Objective

Build `stingray scout` so a developer or coding model can point Stingray at a GGUF checkpoint and receive an evidence-backed answer to these questions before writing or changing engine code:

1. What is this exact artifact? What architecture does it declare, what is actually in its tensor index, and which files are required alongside it?
2. Which existing Stingray architecture is the closest structural relative, and exactly what differs?
3. What model features are directly observed, and what features are only hypotheses suggested by metadata or tensor names?
4. Which of Stingray's current loader, dtype, tokenizer, kernel, backend, and admission constraints appear relevant?
5. Is it safe to run the next test on this machine under an explicit memory budget?
6. Which existing automated checks can be run now, and what is the smallest useful next experiment if the evidence is inconclusive?
7. What evidence is still missing before the family can be admitted?

The goal is to reduce repeated manual reconnaissance and avoid writing code when the actual problem is a tokenizer, quantization, tensor-layout, companion-file, or resource issue.

**The scout is decision support, not an automatic architecture porter or an admission authority.** Static similarity, finite logits, successful model construction, and synthetic tests must never silently promote a model to `Admitted`.

## 2. Repository facts and existing building blocks

Audit these at the current HEAD before implementation; reuse them rather than creating parallel systems.

- `src/OpenTail.Stingray.Cli/Program.cs` registers the current CLI surfaces, including `inspect`, `list-models` (with `--deep`), `list-metadata`, `list-tensors`, `plan`, `pull`, `admit-arch`, `capture-golden`, `verify-goldens`, `hash`, and the existing task commands.
- `src/OpenTail.Stingray.Cli/AdmitArchCommand.cs` already performs tokenizer triage, prints the `blk.0.*` tensor inventory, tries a real CPU forward pass behind the diagnostic admission bypass, and accepts reference tokens or a golden. It currently acts as one admission-oriented diagnostic path; do not duplicate it wholesale.
- `src/OpenTail.Stingray.Engine/Architectures/ArchitectureProbe.cs`, `ArchitectureDescriptor.cs`, `ArchitectureRegistry.cs`, and `BuiltInArchitectures.cs` provide the static registry, relabelled-file probes, admission state, backend/capability declarations, and NativeAOT-compatible architecture manifest.
- `src/OpenTail.Stingray.Engine/ModelCompatibility.cs` contains the existing admission gate and supported weight-dtype checks.
- `src/OpenTail.Stingray.Engine/Verification/` already contains the golden-parity runner, model locator, model fingerprinter, and related evidence types. The current workflow includes `capture-golden`, `admit-arch --golden`, and `verify-goldens`; inspect their actual current semantics before wiring them in.
- `src/OpenTail.Stingray.Cli/PullCommand.cs` already discovers GGUF files in a Hugging Face repository, selects quantizations, handles shards, and checks the published SHA-256 when available.
- `src/OpenTail.Stingray.Core/ModelPackageInspector.cs` and `ModelPackageCapability.cs` inspect a narrow set of SafeTensors profiles from headers without loading weights. This is useful precedent, but it is not the GGUF admission registry and must not be presented as comprehensive SafeTensors support.
- `docs/reference/adding-an-architecture.md` documents the current descriptor/manifest/admission flow. `docs/reference/061-coverage-tooling.md` describes existing coverage tooling. `docs/2-coverage/2026-10-08-golden-parity-and-admission-tooling-plan.md` contains the rationale and validation history for the golden workflow. Check the plan's remaining unchecked/failure-path items against code and current results before extending it; its title/status may lag its completed phases.
- Existing `stingray inspect`, `list-tensors`, `list-metadata`, and `list-models --deep` must remain useful standalone commands. Scout should compose their underlying capabilities through reusable services, not shell out to these CLI commands or create competing versions of their reports.

Current architectural invariants remain in force: no Python, no reflection/dynamic plugin loading, no `Reflection.Emit`, source-generated JSON where needed, warnings-as-errors, static architecture registration, and fail-closed admission. Follow `CLAUDE.md`, especially the real-reference, heavy-test, truthful-status, and not-admitted rules.

## 3. Deliverables

### A. `stingray scout`

Initial local usage:

```text
stingray scout -m <model.gguf>
stingray scout -m <model.gguf> --format json -o scout-report.json
stingray scout -m <model.gguf> --budget 64G
stingray scout -m <model.gguf> --pretest --budget 64G
```

Use the CLI's established option conventions; validate the final syntax against the existing command router. Static inspection must be the default. Running model inference must be an explicit opt-in.

### B. Stable, AI-consumable report

Define versioned C# report records and a source-generated JSON serialization context. A report must be stable enough for a coding model to compare against a report from a different checkpoint, commit, or machine. Use schema versioning and explicit `Known`, `Estimated`, `Hypothesis`, `Unknown`, `Blocked`, `Passed`, `Failed`, and `NotRun` distinctions where appropriate; do not encode unknown values as zero or false.

Include at least:

- **Artifact identity:** bare filename, size, GGUF version, declared architecture, resolved descriptor id/aliases if any, source repo/revision if known, hash state (`verified`, `not computed`, or `failed`), shard inventory/completeness if discoverable. Do not write absolute machine paths, usernames, or drive letters into checked-in receipts or portable report examples.
- **Metadata summary:** key architecture/hyperparameter values, tokenizer metadata and chat-template presence, context-related fields, quantization/version values where present. Preserve original key/value evidence for conclusions that depend on it.
- **Complete tensor-index summary:** tensor count, bytes grouped by dtype, shape signatures, normalized per-layer pattern counts, missing/duplicate/unexpected patterns when there is a known contract, layer-to-layer irregularities, and representative names/shapes. Never load full tensor values for the default scout pass.
- **Architecture resolution:** declared id, resolution result from the existing registry/probe path, admission status and refusal reason, and candidate relatives with an explainable list of matching and differing structural facts.
- **Feature findings:** observed features and hypotheses separately. Every finding must include a stable finding id, short interpretation, confidence category, evidence items (metadata keys and/or tensor names, dtypes and shapes), and a caveat. No claim that an architecture supports a feature follows merely from detecting a feature-shaped tensor.
- **Compatibility blockers:** dtypes not covered by the actual compatibility path, missing required assets, tokenizer warning patterns, metadata/config anomalies, unsupported backend capabilities and structural mismatches. Distinguish a confirmed blocker from a suspicion.
- **Resource preflight:** known checkpoint/file sizes; an estimated or unknown execution working set; configured budget and reserved headroom; estimated KV/cache/workspace components where the data exists; the source and confidence of each estimate; whether an execution stage is allowed, blocked, or not assessed. Mapping a file does not imply the whole file is resident; file size alone must not be reported as peak RAM.
- **Suggested next actions:** ordered tests/commands and why each is the cheapest useful next experiment. Include required reference, asset, hardware, and context-size caveats. Suggestions are never automatic code changes.
- **Pretest receipts:** one result per stage with start/end time, build identity, checkpoint identity/pin, test parameters, outcome, and failure detail. Do not collapse `NotRun` or `Blocked` into `Passed`.

Provide human-readable terminal output and JSON output (`stdout` and/or `-o`), with deterministic ordering. The JSON report and any saved log should be sufficient for an AI coding model to continue without having to infer missing facts from a screenshot of terminal output.

### C. AI architecture-admission playbook

Add an agent-oriented, step-by-step document, for example `docs/reference/architecture-admission-agent-playbook.md`, and link it from `docs/reference/adding-an-architecture.md` and the coverage-tooling index. This is an operational contract for coding models, not another general tutorial.

The playbook must state:

1. Source-of-truth order: repository code/tests and a recorded reference implementation (the revision the contributor actually used) outrank names, model cards, scout hypotheses, and previous conversational claims.
2. Run the scout first and treat its output as evidence and a hypothesis list, never as proof of numerical semantics.
3. Check the current active backlog, memory budget, installed checkpoints, matching reference binaries, licenses, and existing architecture/quantization support before porting.
4. Inspect the actual upstream implementation for each semantic difference before coding. Record the exact file, commit/revision and the operation being implemented.
5. Distinguish tokenizer, weight dtype/dequantization, tensor layout, architecture semantics, runtime/cache, and backend problems. Do not “fix” all of them by adding broad aliases or weakening a parity test.
6. Use existing descriptor registration and tests; make a Core or central-dispatch change only when the evidence shows that the abstraction cannot express the model.
7. Keep resource-infeasible real-weight tests parked. Synthetic/specification evidence must be labelled at its actual level; a model that cannot be tested on this host remains not admitted and not advertised.
8. Capture a hash-pinned independent reference using the existing golden tooling when a compatible oracle exists. Compare prompt tokenization separately from forward-pass parity. Record exact/near-tie/diverged outcomes and reference margins; never label an arbitrary mismatch “near-tie” without the relevant evidence.
9. Run appropriate targeted tests, then required broader tests one heavy process at a time. Obey `CLAUDE.md` test-runner cautions: a zero-test/no-op run is not evidence.
10. Update descriptor status, goldens, test evidence, and `docs/STATUS.md` together only after the admission criteria pass. If an oracle, checkpoint, license decision, or hardware is missing, report the blocker and move to another feasible backlog item instead of claiming completion.
11. Produce a concise conclusion: what was changed, what was verified, what was not tested, actual commands/outputs, and the exact remaining admission gate.

The playbook should reference existing commands and canonical docs instead of copying their content. Keep each rule imperative and unambiguous so an agent can follow it without needing a human interpretation of marketing language.

## 4. Feature hypothesis system

Implement an explainable, conservative probe system. Do not use an LLM or an opaque embedding similarity model as the source of truth. Prefer static C# rules over GGUF metadata and tensor-name/shape evidence.

Initial candidate findings (only report a finding when the relevant evidence is actually observed):

- Dense versus expert-routed FFN, expert count/top-k/router and possible shared-expert patterns.
- Attention projection layouts: fused versus separate Q/K/V, Q/K norms, head dimensions, GQA/MQA, MLA-related low-rank projection metadata, compressed-attention/indexer tensor families.
- Recurrent/hybrid signals: GDN, Mamba-2, short convolution, RWKV-related state/tensor patterns and per-layer heterogeneity.
- MTP/NextN or other auxiliary prediction heads, identified as hypotheses until the tensor contract and runtime path are understood.
- RoPE/position semantics: type, scaling, partial/interleaved/multi-axis indicators, and layer-specific differences where metadata supports them.
- Tokenizer shapes: model, merges, scores, token types, BOS/EOS, byte fallback and known problematic combinations already documented in `AdmitArchCommand` and tokenizer code.
- Multimodal companion evidence: projector/vision/audio tensor groups, known external sidecar naming patterns, and metadata hints for image/audio input. Detecting a projector does not prove a complete user-facing multimodal path.
- Unusual/experimental weight formats, quant mixes, per-layer dtype changes, and storage types for which the actual execution path has gaps.

For each rule, unit-test its positive signal, negative signal and ambiguous case with synthetic metadata/tensor indexes. Prefer qualitative confidence levels whose meanings are documented over arbitrary numeric confidence scores. Findings must name the exact evidence that triggered them.

### Architecture similarity

Rank existing descriptors using a deterministic, explainable signature comparison: normalized tensor suffix patterns, relative shape patterns, metadata invariants and known layer topologies. Output the matched/different facts that led to a candidate. Avoid raw string similarity as the main signal. Do not let similarity rewrite `general.architecture`, resolve unknown semantics automatically, or bypass admission.

Initially report candidate similarities only. Calibrate any numeric ranking on known admitted and known refused/ported-not-admitted checkpoints before representing it as a useful likelihood. A new architecture with superficially similar tensors must not inherit its candidate's admission or semantics.

## 5. Resource policy: 64 GB host

The scout must always allow static inspection of a model that is too large to execute. Do not download weights or start inference as a side effect of ordinary `scout`.

- Accept an explicit budget such as `--budget 64G`, and a configurable safety reserve/headroom. The user's working policy is not to schedule real-weight tests whose credible peak working set exceeds this budget after reserve.
- Separate file size, mapped bytes, expected resident weight pages, context/KV cache, scratch/temporary buffers, and other known allocations. Label unknown terms as unknown; do not generate false precision.
- Use existing planner/memory-estimation code where its inputs and scope actually apply. Do not silently reuse a GPU-VRAM estimate as a host-RAM estimate.
- If the estimate is unknown, incomplete, or too close to the limit, permit static inspection but mark execution pretests `Blocked` or require an explicit override with a clear warning. Never auto-run a 72.5 GB, 98.6 GB, 236 GiB or larger checkpoint on this host just because it can be memory-mapped or stored on disk.
- The budget gate is a safety policy, not a proof that a model will fit. Record actual peak working set for tests that do run where it can be measured.
- For automatic Hugging Face discovery, rank GGUF candidates by quantization and published byte size before downloading. No checkpoint downloads, gated-license acceptance, or high-cost reference launches without explicit user action.

## 6. Gated pre-admission pipeline

Compose existing code through shared services where necessary. Do not implement an inference runner inside the reporting DTOs or duplicate the golden logic.

| Stage | Work | Meaning of success | Failure handling |
|---|---|---|---|
| 0. Artifact | Open GGUF header/index, check readability and shard presence, compute hash only when requested/required or use cached fingerprint | Artifact is readable and identified | `Failed` for corruption; retain partial report if possible |
| 1. Static contract | Metadata invariants, architecture resolution, tensor-index signature, dtype coverage, tokenizer warning checks | Reported structure agrees with known contracts, where such a contract exists | Separate confirmed mismatch from unsupported/unknown profile |
| 2. Feasibility | Resource-budget gate, reference availability, checkpoint pin, backend capability | A specific next stage is safe and meaningful to run | `Blocked`/`Unknown`, never mislabelled as pass |
| 3. Construction/smoke | Only when explicitly requested and the budget gate permits: reuse the selected existing loader/forward-pass path; test prefill and a small bounded decode; reject exceptions, empty logits and non-finite values | The selected implementation executes on this artifact | `Passed` means smoke only, never admission |
| 4. Internal consistency | Run applicable existing stepwise/prefill, cache, batch or model-specific invariants | The tested internal invariant stays within its declared bound | Report exact failing invariant and parameters |
| 5. Independent reference | Run existing hash-pinned golden verification; create/capture a reference only by explicit request and when a compatible oracle/resource budget exists | The behaviour agrees with that reference at the documented level | Exact, near-tie, divergence, oracle unavailable, or not run must remain distinct |
| 6. Admission readiness | Check complete evidence set, timed run, license/release status, required tests and status documentation | A human/coding-agent can make an evidence-backed admission decision | No automatic production admission or status promotion |

For an unknown architecture, a real forward pass through the diagnostic bypass must be explicit (for example, an opt-in `--try-unverified` flag), bounded by the resource policy, and visibly described as unverified. Scout's default operation should be read-only and static.

Keep each stage independently callable/testable. Stage 3+ should reuse the current heavy-test mutex or an equivalent single-heavy-process guard, with reliable cancellation and process-tree cleanup for external reference tools.

## 7. Hugging Face discovery (later phase, not first implementation)

Add an optional repository-discovery mode only after local GGUF scout works:

```text
stingray scout -r <owner/repo> --budget 64G
```

It should query repository file metadata, enumerate GGUF choices/shards, identify quantizations and reported sizes, estimate whether each candidate is worth downloading, and print ranked candidates with the reasons. Reuse or factor the existing `PullCommand` listing logic; do not maintain a second independent Hugging Face client with subtly different hash and shard rules.

Default to list-only. A download must be a separate explicit action through existing `pull` behaviour. Respect `HF_TOKEN`/gated status, published hashes and licenses. Do not scrape every model on Hugging Face or create an unbounded background discovery service.

## 8. Suggested phases

### 8.1 Scope decision and rationale (2026-10-09)

**Committed:** Phase 1 (static GGUF scout + JSON), Phase 2 (feature findings + relatives), a short version of the playbook (§3C), and the optional `scripts/scout-pretest.ps1` wrapper (replaces the C# Phase 3).

**Deferred, and why:**

- **C# pretest runner (old Phase 3):** `admit-arch`, `capture-golden` and `verify-goldens` already do the smoke/reference/parity work, so a C# runner only adds orchestration, plus in-process cancellation, process-tree cleanup and a heavy-run guard. A script gets the same convenience for far less cost and is disposable. Its estimation inputs come from Phase 1 output, so the interface should not be guessed before that exists.
- **HF discovery (Phase 4):** only pays off if scout proves useful on local files first, and it forces a refactor of working `PullCommand` hash/shard logic for speculative value.
- **Calibration audit (Phase 5):** needs Phase 1–2 output to audit. Similarity ranking is not called useful until audited.
- **Stage 5/6 inside scout:** scout only *reports the state* of existing golden/admission evidence (present, pinned, absent). It does not drive golden tooling or decide readiness; the playbook checklist covers that.

**Prerequisites pulled into Phase 1:** extract the shared tokenizer-triage and `blk.0.*` inventory service from `AdmitArchCommand` (used by both commands); give `admit-arch` and `verify-goldens` clean, documented exit codes if they lack them; define the stage-receipt JSON schema (`Passed`/`Failed`/`Blocked`/`NotRun`) even though only the script writes it.

### 8.2 `scripts/scout-pretest.ps1` (optional follow-up after Phase 2)

An opt-in, disposable wrapper, improved iteratively:

- Reads the scout JSON (never scrapes terminal text); compares the scout's estimated working set plus reserve to `--budget`; unknown estimate means `Blocked`, not run.
- Runs existing commands (`admit-arch`, optionally `verify-goldens`) as child processes with timeout; kills the process tree on timeout/Ctrl+C; records peak working set; takes the same single-heavy-process guard as the heavy test suites.
- Writes a receipt JSON (checkpoint identity, build identity, parameters, per-stage outcome) to the scratchpad/temp by default.
- **Non-destructive by contract:** never writes to or deletes checkpoints, never downloads, never edits repo files, never changes admission/status. Writes only to its output path.
- Fast-suite coverage does not apply; verification is a run on a small model already in `models/`, with timing checked per `CLAUDE.md` rule 12.
- **Documentation required with the script:** a section in `docs/reference/061-coverage-tooling.md` (usage, flags, receipt schema, exit codes, what it will not do, worked example on a small model), linked from the playbook. The script's `-?` help must match it.

### Phase 0 — repository audit and contracts

- Re-read the exact current implementations and tests named in §2; do not rely only on this plan's descriptions.
- Inventory the report surfaces already emitted by `inspect`, `plan`, `admit-arch`, `list-tensors --summary`, `list-metadata`, `list-models --deep`, `pull` and golden verification.
- Identify reusable services, gaps, and code paths that currently duplicate logic.
- Confirm that the current golden plan's proposed/not-started header is reconciled with the phases already landed. List the actual unfinished items separately; do not restart completed work.
- Write a short, reviewed report schema and explicit stage state model before implementation.

### Phase 1 — read-only local GGUF scout

- Implement the report records, source-generated JSON context and `stingray scout -m <file>` static mode.
- Reuse GGUF metadata/tensor index, `ArchitectureProbe`/`ArchitectureRegistry`, dtype allowlists, existing hash cache and shared reporting primitives where appropriate.
- Produce deterministic full-inventory and irregularity summaries plus a JSON output file/standard output mode.
- Add focused Fast tests using synthetic indexes/fixtures; static scout tests must not need a large real checkpoint.
- Confirm from code/tests that this mode does not construct a forward pass or read tensor values.

### Phase 2 — feature evidence and architecture relatives

**Status 2026-10-09:** feature rules landed (`Scout/ScoutFeatures.cs`, 29 synthetic positive/negative/ambiguous tests). Relatives ranking landed the same day (below).
Real-checkpoint check (index-only scout, 13 archived GGUFs + 17 local, 11 s total, no weights read): positives were as expected on
GLM-4.5-Air (`ffn.expert_routed`, `ffn.shared_expert`, `mtp.nextn_head` Known), EXAONE-4.5 (`mtp.nextn_head` Known), DeepSeek-V2-Lite and GLM-4.7-Flash
(`attn.mla_low_rank`), Phi-3.5-MoE (routed, correctly no shared expert), Qwen3.6-35B-A3B (`recurrent.hybrid_ssm`, `attn.qkv_layout_mixed`, `rope.multi_axis`),
Qwen3-VL (`rope.multi_axis`), Jais (`attn.fused_qkv`); no spurious findings on dense llama/qwen3 or the audio/diffusion files. Not covered by any real file:
`attn.sparse_indexer`, `recurrent.rwkv_time_mix`, `multimodal.projector_file` (synthetic only). This is a smoke audit, not the calibration audit.

**Relatives ranking (decided 2026-10-09, built):** signatures taken from admitted GGUFs with scout itself, checked in as data, structure kept apart from origin (see
`docs/reference/061-coverage-tooling.md`). Chosen over a table generated from llama.cpp's `llama-arch.cpp` because the question that matters for a community variant is
"which architecture we can already RUN is nearest, and where exactly does it differ?", which needs rank, layer coverage and feature facts that the upstream table does not
carry (it is names only; required/optional tensors and layer topology live in code), and because it has no dependence on anyone's llama.cpp version or local files. The
upstream table is not used: llama.cpp is not part of this project (xamples/ and 	ools/llama.cpp are git-ignored, so other contributors do not have it), and a basis for ranking must be something every contributor has. Scout therefore depends on nothing but its own checked-in signatures.

**Leave-one-architecture-out check (2026-10-09, `scripts/scout-leave-one-out.ps1`, 12 structures of 9 families, 12 shipped signatures).** Each real file was scouted with its own architecture's
signatures removed. Nearest remaining parent and structural difference count: qwen3 <-> qwen3vl 1 (tensors identical, only `rope.multi_axis` differs); llama (3 structures) -> qwen3 3/4/5
(exactly the Q/K-norm tensors and feature); exaone4 -> qwen3 11; jais -> llama 14 (fused QKV, LayerNorm biases); phimoe -> llama 18; deepseek2 -> llama 19 and glm4moe 24;
glm4moe -> deepseek2 20; qwen35moe -> qwen3vl 34. Observations, not thresholds (n=12): the parents named are sensible in every case; counts split cleanly into <=11 (a real near-relative)
and >=14 (nothing close; the list says what is missing); no file matched a *different* architecture with zero differences, so no false relabel signal. Limits: qwen3 vs qwen3vl show that
identical tensors can hide a semantic difference only visible in metadata, which is exactly what the caveat on every finding says; and the check cannot show what the ranking does for a
family far from everything shipped, beyond "large count".
Open risks: the shipped set covers only what had a checkpoint on the maintainer's machine; structural identity says nothing about metadata values, so every relabel
candidate still needs a golden; the ranking is uncalibrated (counts, not probabilities).

- Add a small, statically registered or otherwise explicitly composed set of feature-finding rules. (done)
- Compare tensor signatures against registered descriptors; explain all matches and differences. (done, against checked-in admitted-architecture signatures)
- Unit-test positive, negative, ambiguous and unknown signals. (done)
- Keep hypotheses and confidence meanings in the report schema/documentation. Do not automatically alter registry behaviour. (done)

**Working-set estimator (2026-10-09, built; unblocks the memory gate for plain-attention dense and MoE text models).** `HostMemoryEstimator` gives an upper-bound CPU-run estimate
(weights + Q4_K repack copy + base + fp32 KV + prefill scratch), calibrated on real runs and checked blind on one MoE; see the table in `docs/reference/061-coverage-tooling.md`.
Result: at or above measured in all 7 runs, 1.05-1.07x where the run filled the context. It also found that a Q4_K_M file costs about 1.8x its size on this CPU path because of the
repack copy. What it does not do: MLA (DeepSeek2), hybrid/GDN and RWKV families stay Unknown and blocked; other MoE families and GPU placement are unmeasured. Costs named per
CLAUDE.md rule 11: this took a measurement pass (about 10 real runs, one of 245 s because the 18.7 GB MoE loads from the archive disk) before the formula could be trusted, plus one
discarded run that had silently failed (prompt over the context).
**`scripts/scout-pretest.ps1` (2026-10-09, built).** Behaviour and the checks it passed are in `docs/reference/061-coverage-tooling.md`. Not yet covered: Ctrl+C mid-run and a hard kill of PowerShell; stage 4 has no
command to call. It is only as strong as `admit-arch`'s exit codes, which cannot tell a failed check from a tool error, so the script classifies by the printed verdict line.
### Phase 3 — memory-gated pretests (superseded by §8.2: implemented as a script, not C#; the points below define its behaviour)

- Implement `--budget`, reserve/headroom handling, estimate provenance and the decision to allow/block each stage.
- Add opt-in construction/smoke testing, reuse the existing admission and golden services, and standardize `Passed`/`Failed`/`Blocked`/`NotRun` stage receipts.
- Ensure cancellation, one-heavy-run-at-a-time behaviour and external-process cleanup.
- Exercise this phase first with small models already on disk and a clearly below-budget model. Do not use oversized checkpoints to test the resource gate.

### Phase 4 — HF repository discovery (DEFERRED, see §8.1)

- Factor candidate listing from `PullCommand` if needed, preserving its hash and shard handling.
- Add list-only `scout -r`; show all viable quant options and memory-budget findings without downloading anything.
- Test malformed repositories, gated responses, shard groups, missing hashes and multiple quant variants via mocked HTTP/API responses.

### Phase 5 — agent playbook, integration and calibration (playbook is written with Phase 1; calibration DEFERRED, see §8.1)

Playbook must also restate `CLAUDE.md` rules 6 (no subagents), 12 (check timing, not just pass/fail) and 14 (ported-not-verified stays internal). Scout output for such families is internal and must not feed STATUS, README or catalogs. Saved reports/receipts go to scratchpad/temp by default; only synthetic, path-scrubbed examples are committed.

- Add the AI admission playbook and report schema reference. Link them from `adding-an-architecture.md`, `061-coverage-tooling.md`, and the appropriate coverage index.
- Update the CLI option inventory and docs through the existing project process.
- Run on a small, varied set of available GGUFs: a supported dense model, an admitted MoE/hybrid model, a known ported-but-not-admitted architecture, and a malformed or unusual fixture. Keep every real checkpoint within the configured RAM policy.
- Compare its recommendations against known cases and record false positives/missed findings. Do not call similarity or feature discovery useful until that audit has been performed.

## 9. Test and verification requirements

- Unit tests for deterministic report serialization/schema version; path privacy in portable examples; partial reports after errors; stable ordering; known/unknown architecture distinction; dtype counts/bytes; shard completeness; tokenizer findings; feature rules; architecture similarity explanations; and memory-gate thresholds.
- Negative tests: familiar tensor names with conflicting dimensions must not be labelled a match; one isolated MTP-like tensor must not prove working MTP; a vision projector alone must not imply full VLM support; missing reference/checkpoint must not pass; unknown memory estimate must not silently allow a costly run.
- Integration tests on a small real model: static scout output, explicit smoke test, hash pin, existing golden path, and truthful outcomes. Record exact build and artifact identity.
- Ensure `scout` does not regress `inspect`, `plan`, `admit-arch`, `pull`, `capture-golden`, or `verify-goldens`.
- Follow repo test-runner rules. Any command that reports zero tests or completes suspiciously quickly is not evidence; build the test executable first and use the project's qualified direct-run convention if filter-based invocation is unreliable.
- No full-repository real-weight sweep as a default test. Run heavy cases one process at a time and use only models within the memory budget.

## 10. Definition of done

- `stingray scout -m <GGUF>` creates a deterministic, versioned report without loading tensor data or constructing a forward pass.
- The report differentiates observed facts, hypotheses, estimates, blockers and missing evidence; each hypothesis points to exact supporting metadata/tensors.
- It lists architecture-relative matches with explainable differences and does not silently reuse or admit their semantics.
- All execution pretests are opt-in and pass a conservative budget gate; oversized checkpoints remain inspectable but do not get executed automatically.
- If requested and safe, pretests reuse the existing engine/golden/reference paths and persist test receipts with artifact identity.
- AI playbook provides an unambiguous repo-grounded admission loop and links to authoritative source files rather than duplicating long documentation.
- Optional HF discovery reuses `pull` metadata rules, ranks candidate quants by size and never downloads implicitly.
- Fast tests cover all static logic, with bounded real-checkpoint evidence for each delivered stage.
- The existing admission gate remains fail-closed. No architecture's status, catalog presence or public support claims change solely because Scout says it looks similar.

## 11. Explicit non-goals

- Automatically writing or merging a new forward pass, kernel, descriptor, or admission decision.
- Treating a tensor signature as proof of architectural equivalence.
- Loading all checkpoint tensor values during the normal scout run.
- Automatically downloading models, accepting model licenses, or starting expensive `llama-server` jobs.
- Trying to run enormous disk-paged checkpoints on the 64 GB host by default.
- Replacing the existing golden harness, architecture registry, `inspect`, `plan`, `pull` or `ModelPackageInspector`.
- Building a global Hugging Face crawler, cloud service, vector database or opaque ML-based model classifier.
- Claiming audio, vision, diffusion or SafeTensors verification is covered by the GGUF text-admission pipeline. Those paths need modality-specific probes and independent references in later, separately scoped work.

## 12. Expected result

The intended workflow becomes:

```text
checkpoint / HF repository
          |
          v
      stingray scout
          |
          +--> artifact + complete tensor signature
          +--> architecture identity and nearest relatives
          +--> observed features vs. hypotheses
          +--> compatibility gaps + resource decision
          +--> safe, ordered next tests
                    |
                    v
       existing tests / admit-arch / goldens
                    |
                    v
        evidence-backed admission decision
```

The value is not “AI guesses the architecture.” The value is that an AI coding model receives a structured, evidence-citing dossier and a repeatable next action, allowing it to spend less time rediscovering file facts and more time implementing and proving the actual semantic delta.
