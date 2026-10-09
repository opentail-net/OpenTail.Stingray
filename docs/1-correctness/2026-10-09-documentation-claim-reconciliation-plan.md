# Documentation Claim Reconciliation Plan

> **Status:** Proposed — evidence-gathering pass, not an implementation verdict  
> **Date:** October 9, 2026  
> **Scope:** Resolve five specific disagreements found while reviewing the documentation architecture plan.  
> **Rule:** Do not treat a status document, a source-code inspection, a historical test result, or a reviewer assertion as conclusive on its own. Reconcile them against the same current commit and the strongest practical evidence available.

## 1. Purpose and boundary

The [Documentation Architecture Plan](2026-10-09-documentation-architecture-plan.md) should not be expanded into a general model-documentation project to settle these questions. This is a separate, bounded evidence pass whose output is a short reconciliation report and, only after conclusions are supported, a list of targeted documentation corrections or follow-up engineering tasks.

Some earlier evidence may be stale. A historical failure may have been fixed since its entry was written; a newer success may cover only one component or a weaker test than the public claim; and a source file or test name may have changed. The audit must establish which is true for each item.

**Do not pre-label an item as broken or unsupported.** The starting state for every item is *disputed / needs reconciliation*.

### In scope

- [ ] **Item 1**: The proposed public C# example's model-home API call ([§R1](#r1-public-c-example-model-home-resolution)).
- [ ] **Item 2**: HunyuanVideo and Wan 2.2 end-to-end support classifications ([§R2](#r2-diffusion-claims-hunyuanvideo-and-wan-22)).
- [ ] **Item 3**: The claim that AVX-512 execution is verified ([§R3](#r3-avx-512-implemented-path-versus-verified-execution)).
- [ ] **Item 4**: The environment-variable registry's documented source filename ([§R4](#r4-environment-registry-source-of-truth-filename-and-drift-checks)).
- [ ] **Item 5**: The claimed TurboQuant KV-cache quantization modes ([§R5](#r5-turboquant-kv-cache-modes-and-the-guides-scope)).

### Out of scope

- Creating all model architecture cards or benchmarking documentation against TensorSharp.
- New model-family ports, optimizations, or broad backend redesign.
- Rewriting the status matrix before checking it.
- Treating lack of access to suitable hardware or checkpoints as proof of a failing implementation.

If an audit uncovers an implementation defect, capture a minimal reproduction and create a separately scoped follow-up. Do not quietly mix feature implementation into this evidence pass.

## 2. Evidence rules

### 2.1 Establish one reproducible baseline

At the start, record:

- [ ] Repository URL, branch, exact HEAD SHA, audit date, and whether the working tree is clean.
- [ ] The exact blob/commit for each plan, source file, test and status row cited in the report.
- [ ] SDK/runtime and OS for builds; CPU ISA and GPU/device/driver for hardware tests.
- [ ] Exact checkpoint revision or file hash, quantization, command, parameters, and outcome for model runs.

Read the repository's current contributor/test instructions, including CLAUDE.md, before selecting commands. Do not infer current code state from a search snippet or mix a main-branch source file with a result from a different commit. Repeat key checks against the final commit if code changes during the audit.

### 2.2 Use evidence appropriate to the claim

- **A source declaration** establishes that an API or code path exists, not that it compiles for a package consumer or produces correct output.
- **A unit test** establishes only what its assertions cover. A test that loads weights or produces finite numbers is not by itself proof of coherent end-user output.
- **A recorded success or failure** applies to the commit, checkpoint, backend, parameters and test scope recorded with it. Check for a later fix and rerun the smallest valid reproducer.
- **A status table** is a lead to investigate, not the final arbiter when its own evidence notes disagree with its status label.
- **No available test, hardware or trustworthy reference** means *unverified in this audit*, not *failed*.
- If independent reference parity is unavailable, say so. Do not present visual plausibility or successful execution as numerical parity.

Each conclusion must link to the inspected source and test/evidence, and say what was actually demonstrated.

## 3. Work items and acceptance criteria

### R1. Public C# example: model-home resolution

**Disagreement to check:** the canonical sample in the documentation architecture plan calls `ModelHome.ResolveModelPath("qwen2.5-0.5b")`, while the model-home source inspected during review appeared to expose a different namespace and API. Verify this against the current checkout and published package before labelling the sample invalid.

- [ ] **R1.1 Inspect API declarations:** Inspect the current `ModelHome`, `ModelCatalog`, public API and package project files. Search the repository for `ResolveModelPath` and any overload or compatibility facade; inspect recent history for API moves or renames.
- [ ] **R1.2 Determine intended public contract:** Establish which model-resolution operation is intended for external NuGet consumers, not just internal CLI code. Confirm that required types are public and exported from the package.
- [ ] **R1.3 Test compilation in clean consumer project:** Create a clean .NET 10 consumer project using the exact package version the docs intend to document. Compile the sample exactly as written, without project-reference shortcuts or extra internal references.
- [ ] **R1.4 Diagnose compilation failures (if any):** If it fails, classify the cause: stale sample, stale source snapshot, a missing public API, wrong import, package/content mismatch, or a missing model/catalog prerequisite. Do not repair the sample until the intended public contract is decided.
- [ ] **R1.5 Validate corrected sample:** Test the corrected candidate in that same external-project setup. If model weights are needed to test runtime behaviour, separate API compilation from model installation and inference.
- [ ] **R1.6 Complete acceptance criteria:** Record the exact package version, API and namespace found, compile result, and proven corrected sample.

**Evidence to inspect:** [ModelHome.cs](../../src/OpenTail.Stingray.Core/Catalog/ModelHome.cs), [ModelCatalog.cs](../../src/OpenTail.Stingray.Core/Catalog/ModelCatalog.cs), [public package README](../../src/OpenTail.Stingray/README.md), public API contract tests, and packed NuGet artifact used by the test.

---

### R2. Diffusion claims: HunyuanVideo and Wan 2.2

**Disagreement to check:** the architecture plan lists HunyuanVideo and Wan 2.1/2.2 as end-to-end supported, while the current STATUS.md evidence notes include an unresolved noise output for HunyuanVideo and a limited, zero-conditioning Wan 2.2 A14B smoke run. These notes might be stale; neither the positive plan listing nor the negative historical note is sufficient without a current check.

- [ ] **R2.1 Inspect pipeline source and history:** Inspect current pipeline wiring, real conditioning, VAE/decode path, model-family detection, CLI/API availability and latest commits affecting both models. Determine whether the status notes describe code that has since changed.
- [ ] **R2.2 Delineate variants:** Split the assessment by exact model/variant. Do not let Wan 2.1 evidence stand in for Wan 2.2 A14B; record the low-noise and high-noise checkpoint pair if the variant requires it.
- [ ] **R2.3 Inventory latest test and saved artifacts:** Locate the latest applicable tests, saved outputs, intermediate tensors, reference commands and reports. Determine whether they exercise real model weights and real prompt conditioning, and whether a later success supersedes an older failure.
- [ ] **R2.4 Run targeted reproducers:** Run the smallest test that resolves each disputed claim, then escalate to an end-to-end run only where needed. Use a representative prompt, fixed seed and explicit dimensions, steps, scheduler and guidance. Record backend, checkpoint hashes and elapsed time.
- [ ] **R2.5 Assess HunyuanVideo:** Distinguish (a) component/numerical checks, (b) complete pipeline execution and (c) expected user-visible output. Check whether any alleged reference is the matching HunyuanVideo v1 variant; do not use a HunyuanVideo 1.5 result as proof of v1 parity unless equivalence is demonstrated.
- [ ] **R2.6 Assess Wan 2.2 A14B:** Distinguish existing Wan 2.1 end-to-end evidence from Wan 2.2 A14B's dual-model low/high-noise path. A finite image from zero conditioning may demonstrate wiring, but cannot alone establish prompt-conditioned support.
- [ ] **R2.7 Record visual/numerical evidence:** Where current output is judged visually, retain the actual output (or a stable evidence link) and the prompt/configuration. Where feasible, check intermediate parity against an applicable independent implementation.
- [ ] **R2.8 Complete acceptance criteria:** Assign an evidence-led finding for each variant (end-to-end verified, quality/parity unresolved, components verified only, unverified/blocked).

**Evidence to inspect:** [STATUS.md](../STATUS.md), [RUNNING.md](../RUNNING.md), diffusion pipeline implementations and tests, recent related commits, [PerformanceLeague.md](../../PerformanceLeague.md), and saved diffusion evidence relevant to the exact variants.

---

### R3. AVX-512: implemented path versus verified execution

**Disagreement to check:** a broad CPU verification statement lists AVX-512, while the external-hardware queue describes the local machine as AVX2/FMA and reserves some hardware validation for suitable runners. There may be separate AVX-512 runner or CI evidence; find it before changing the claim.

- [ ] **R3.1 Inventory source and CI receipts:** Search current source, tests, CI workflows, benchmark receipts and engineering logs for AVX-512-specific execution. Inspect relevant commit history and hardware qualification notes.
- [ ] **R3.2 Separate verification dimensions:** Distinguish whether the implementation is present, dispatch is guarded correctly, tests run on non-AVX-512 hardware without illegal instructions, and the AVX-512 kernel has actually executed on hardware reporting the ISA as supported.
- [ ] **R3.3 Audit hardware receipt details:** Ensure any strong receipt identifies exact machine/CPU, OS/runtime, commit, AVX-512 flags, kernel/test invoked, model/fixture, and parity/benchmark numbers.
- [ ] **R3.4 Run local dispatch and fallback tests:** Check local dispatch guards and non-AVX-512 fallback behavior. If an external runner is available, rerun focused tests; if not, record hardware execution as unresolved.
- [ ] **R3.5 Delineate claims in documentation:** Ensure public claims distinguish between *implemented*, *unit/dispatch tested*, *hardware-executed*, and *benchmark-measured*.
- [ ] **R3.6 Complete acceptance criteria:** Provide separate outcomes for each dimension with concrete evidence.

**Evidence to inspect:** SIMD dispatch and kernel code, ISA-specific tests, CI workflows/artifacts, [external-hardware work queue](../9-external-hardware/90-external-hardware-work.md), and relevant performance records.

---

### R4. Environment registry: source-of-truth filename and drift checks

**Disagreement to check:** the architecture plan names `StingrayEnv.cs`, while the source found during review was `KnownEnvironmentVariables.cs`. Establish whether the former has since been added, renamed, or is a mistaken reference—and validate what the tests and generator actually enforce.

- [ ] **R4.1 Search source tree and history:** Search the full current tree and git history for both `StingrayEnv.cs` and `KnownEnvironmentVariables.cs`; inspect the registry, every consumer, and its tests.
- [ ] **R4.2 Verify authority and generator boundaries:** Read the [environment-variable inventory](../reference/env-var-inventory.md) and generator/test rules. Verify whether the registry is the authoritative name list, a scanner output, or both.
- [ ] **R4.3 Run drift-check test suite:** Run the existing registry/inventory drift tests using the repository's prescribed test command (`dotnet test`). Inspect assertion scope and check for ghost entries.
- [ ] **R4.4 Compare representative entries:** Compare representative and potentially problematic entries from source to inventory: spelling, default, effect, classification and diagnostic usage. Regenerate inventory if established workflow requires it.
- [ ] **R4.5 Complete acceptance criteria:** Name the authoritative source, link the relevant test/generator, record clean drift-check result, and update the filename in documentation.

---

### R5. TurboQuant KV-cache modes and the guide's scope

**Disagreement to check:** the proposed guide describes “TurboQuant FP8/INT4 KV caching”, while the source inspected during review described Lloyd-Max/FastScan and KVarN modes, including 4-bit keys and 2-bit values for KVarN. Other FP8 or KV-dtype functionality may exist elsewhere; first trace the actual supported data paths.

- [ ] **R5.1 Trace KV-cache production paths:** Trace all production KV-cache implementations from public parameters/CLI flags/environment settings into storage, encode/decode and attention consumption. Search for FP8 cache types and TurboQuant-specific classes.
- [ ] **R5.2 Audit enums, layouts, and precision:** Read quantizer enums, defaults, accepted configuration values, serialization/layout code and fallback behaviour. Establish the effective precision/layout for keys and values for every advertised mode.
- [ ] **R5.3 Audit test and benchmark coverage:** Link each mode to tests and benchmarks. Confirm whether tests check round-trip/error/quality behaviour or only packing mechanics, and capture model-specific limitations.
- [ ] **R5.4 Verify feature boundaries:** Verify that paged allocation and prefix caching are described as separate cache mechanisms where appropriate, rather than as quantization modes.
- [ ] **R5.5 Clarify FP8 relationship:** If FP8 KV-cache storage exists, identify its exact implementation and configuration and clarify its relationship to TurboQuant (or lack thereof).
- [ ] **R5.6 Complete acceptance criteria:** Produce a source-to-configuration matrix listing each KV-cache mode, effective key/value representation, activation mechanism, coverage evidence and limitations.

**Evidence to inspect:** [TurboQuantKvCache.cs](../../src/OpenTail.Stingray.Engine/TurboQuantKvCache.cs), `OpenTail.Stingray.TurboQuant` implementation, cache parameters/enums, CLI and environment configuration, cache tests, and benchmarks.

---

## 4. Reporting format

Produce a compact reconciliation table with one row per disputed claim:

| Field | Required content |
|---|---|
| Claim as written | Quote or link to the precise wording being checked |
| Snapshot | Commit SHA and date of source, docs and test evidence |
| Current implementation | Relevant source path(s) and actual dispatch/configuration |
| Current verification | Test/command, fixture or checkpoint hash, environment, and observed result |
| Evidence limits | What the test does not establish; unavailable reference/hardware/weights |
| Finding | Confirmed current; claim stale; implementation changed; partially verified; unverified/blocked; or ambiguous claim |
| Action | No change, correct wording, update status evidence, fix a sample, or create a separate engineering task |

Use the narrowest defensible conclusion. For example, a compilation failure establishes that the exact sample does not compile in the tested setup; it does not prove the library lacks all equivalent model-resolution capability. An end-to-end run with noise establishes a problem for that exact checkpoint/configuration/run; it does not prove every configuration fails. Conversely, a component test passing does not establish end-to-end quality.

---

## 5. Execution order

- [ ] **Phase 1: Snapshot and source inventory (all five items)**
  - [ ] Pin one commit SHA and record branch/environment baseline.
  - [ ] Gather source paths, tests, docs, and git history for R1–R5.
- [ ] **Phase 2: Fast, deterministic checks**
  - [ ] R4: Run environment inventory drift tests (`KnownEnvironmentVariables.cs`).
  - [ ] R1: Test external C# consumer compilation for `ModelHome`.
  - [ ] R5: Audit TurboQuant and KV-cache enums, layouts, and test suite.
  - [ ] R3: Check SIMD dispatch guards and AVX2 fallback correctness.
- [ ] **Phase 3: Model-specific reproducers (R2)**
  - [ ] R2: Check HunyuanVideo component/numerical tests and latest pipeline status.
  - [ ] R2: Check Wan 2.1 vs Wan 2.2 conditioning tests.
  - [ ] Escalate to full generation only if needed to resolve disputed public claim.
- [ ] **Phase 4: Independent review of findings**
  - [ ] Challenge both positive and negative interpretations against concrete data.
  - [ ] Complete the reconciliation table in §4.
- [ ] **Phase 5: Targeted disposition and documentation updates**
  - [ ] Apply evidence-backed corrections to the Documentation Architecture Plan.
  - [ ] File separate implementation/hardware tasks for unresolved engineering issues.

---

## 6. Definition of done

- [ ] All five disagreements (R1–R5) have a completed evidence row, even if the result is blocked or uncertain.
- [ ] Findings reference a consistent commit snapshot and reproducible commands.
- [ ] Historical failures have been checked for later fixes; recent success claims have been checked for scope and quality.
- [ ] No conclusion is based solely on a documentation label, a class name, or an unsupported assumption.
- [ ] Every changed public claim points to its canonical evidence source. Unavailable evidence is labelled as such, not converted into a failure.
- [ ] The audit report recommends a small set of plan/document edits or separately scoped engineering tasks; it does not silently implement new model or backend features.
