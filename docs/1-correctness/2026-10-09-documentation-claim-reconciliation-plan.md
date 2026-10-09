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

1. The proposed public C# example's model-home API call.
2. HunyuanVideo and Wan 2.2 end-to-end support classifications.
3. The claim that AVX-512 execution is verified.
4. The environment-variable registry's documented source filename.
5. The claimed TurboQuant KV-cache quantization modes.

### Out of scope

- Creating all model architecture cards or benchmarking documentation against TensorSharp.
- New model-family ports, optimizations, or broad backend redesign.
- Rewriting the status matrix before checking it.
- Treating lack of access to suitable hardware or checkpoints as proof of a failing implementation.

If an audit uncovers an implementation defect, capture a minimal reproduction and create a separately scoped follow-up. Do not quietly mix feature implementation into this evidence pass.

## 2. Evidence rules

### 2.1 Establish one reproducible baseline

At the start, record:

- Repository URL, branch, exact HEAD SHA, audit date, and whether the working tree is clean.
- The exact blob/commit for each plan, source file, test and status row cited in the report.
- SDK/runtime and OS for builds; CPU ISA and GPU/device/driver for hardware tests.
- Exact checkpoint revision or file hash, quantization, command, parameters, and outcome for model runs.

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

**Procedure**

1. Inspect the current ModelHome, ModelCatalog, public API and package project files. Search the whole repository for ResolveModelPath and any overload or compatibility facade; inspect recent history for API moves or renames.
2. Establish which model-resolution operation is intended for external NuGet consumers, not just internal CLI code. Confirm that required types are public and exported from the package.
3. Create a clean .NET 10 consumer project using the exact package version the docs intend to document. Compile the sample exactly as written, without project-reference shortcuts or extra internal references.
4. If it fails, classify the cause: stale sample, stale source snapshot, a missing public API, wrong import, package/content mismatch, or a missing model/catalog prerequisite. Do not repair the sample until the intended public contract is decided.
5. Test the corrected candidate in that same external-project setup. If model weights are needed to test runtime behaviour, separate API compilation from model installation and inference.

**Accept when:** the report includes the exact package version, API and namespace found, the compile result, and—if changed—a proven corrected example. The documentation plan must not claim a sample is validated just because its types exist in the repository.

**Evidence to inspect:** [ModelHome.cs](../../src/OpenTail.Stingray.Core/Catalog/ModelHome.cs), [ModelCatalog.cs](../../src/OpenTail.Stingray.Core/Catalog/ModelCatalog.cs), [public package README](../../src/OpenTail.Stingray/README.md), the public API contract tests, and the packed NuGet artifact used by the test.

### R2. Diffusion claims: HunyuanVideo and Wan 2.2

**Disagreement to check:** the architecture plan lists HunyuanVideo and Wan 2.1/2.2 as end-to-end supported, while the current STATUS.md evidence notes include an unresolved noise output for HunyuanVideo and a limited, zero-conditioning Wan 2.2 A14B smoke run. These notes might be stale; neither the positive plan listing nor the negative historical note is sufficient without a current check.

**Procedure**

1. Inspect current pipeline wiring, real conditioning, VAE/decode path, model-family detection, CLI/API availability and latest commits affecting both models. Determine whether the status notes describe code that has since changed.
2. Split the assessment by exact model/variant. Do not let Wan 2.1 evidence stand in for Wan 2.2 A14B; record the low-noise and high-noise checkpoint pair if the variant requires it.
3. Locate the latest applicable tests, saved outputs, intermediate tensors, reference commands and reports. Determine whether they exercise real model weights and real prompt conditioning, and whether a later success supersedes an older failure.
4. Run the smallest test that resolves each disputed claim, then escalate to an end-to-end run only where needed. Use a representative prompt, fixed seed and explicit dimensions, steps, scheduler and guidance. Record backend, checkpoint hashes and elapsed time.
5. For HunyuanVideo, distinguish (a) component/numerical checks, (b) complete pipeline execution and (c) expected user-visible output. Check whether any alleged reference is the matching HunyuanVideo v1 variant; do not use a HunyuanVideo 1.5 result as proof of v1 parity unless equivalence is demonstrated.
6. For Wan, distinguish existing Wan 2.1 end-to-end evidence from Wan 2.2 A14B's dual-model low/high-noise path. A finite image from zero conditioning may demonstrate wiring, but cannot alone establish prompt-conditioned support.
7. Where current output is judged visually, retain the actual output (or a stable evidence link) and the prompt/configuration. Where feasible, check intermediate parity against an applicable independent implementation; do not substitute visual judgement for parity.

**Accept when:** each variant has an evidence-led finding at the strongest level actually tested—e.g. end-to-end verified, end-to-end run but quality/parity unresolved, components verified only, not re-tested, or blocked by unavailable prerequisites. If the prior failure has been fixed, point to the fix and passing current evidence. If it has not, explain the reproducible current limitation. Update plan/status wording only after this conclusion is recorded.

**Evidence to inspect:** [STATUS.md](../STATUS.md), [RUNNING.md](../RUNNING.md), diffusion pipeline implementations and tests, recent related commits, [PerformanceLeague.md](../../PerformanceLeague.md), and saved diffusion evidence relevant to the exact variants.

### R3. AVX-512: implemented path versus verified execution

**Disagreement to check:** a broad CPU verification statement lists AVX-512, while the external-hardware queue describes the local machine as AVX2/FMA and reserves some hardware validation for suitable runners. There may be separate AVX-512 runner or CI evidence; find it before changing the claim.

**Procedure**

1. Search current source, tests, CI workflows, benchmark receipts and engineering logs for AVX-512-specific execution. Inspect relevant commit history and hardware qualification notes.
2. Separate the questions: is the implementation present; is dispatch guarded correctly; can tests run on non-AVX-512 hardware without illegal instructions; and has the AVX-512 kernel actually executed on a machine that reports the relevant ISA as supported?
3. A strong hardware receipt must identify the exact machine/CPU, OS/runtime, commit, AVX-512 support flags, kernel/test invoked, model or fixture, correctness/parity result and benchmark methodology where performance is claimed.
4. If existing CI or external-runner evidence satisfies those requirements, rerun the focused tests at that commit or a compatible current commit. If no such runner is available, check dispatch and fallback correctness locally but record hardware execution as unresolved—not as a failure.
5. Ensure claims do not conflate AVX-512 implementation, successful AVX2 fallback, and measured performance on AVX-512 hardware.

**Accept when:** the report gives separate outcomes for *implemented*, *dispatch/unit-tested*, *executed on AVX-512 hardware*, and *performance measured*, with concrete evidence for each positive claim. Do not broaden or downgrade the public statement before searching existing receipts and checking later changes.

**Evidence to inspect:** SIMD dispatch and kernel code, ISA-specific tests, CI workflows/artifacts, the [external-hardware work queue](../9-external-hardware/90-external-hardware-work.md), and relevant performance records.

### R4. Environment registry: source-of-truth filename and drift checks

**Disagreement to check:** the architecture plan names StingrayEnv.cs, while the source found during review was KnownEnvironmentVariables.cs. Establish whether the former has since been added, renamed, or is a mistaken reference—and validate what the tests and generator actually enforce.

**Procedure**

1. Search the full current tree and relevant history for both names; inspect the registry, every consumer and its tests.
2. Read the [environment-variable inventory](../reference/env-var-inventory.md) and generator/test rules. Verify whether the registry is the authoritative name list, a scanner output, or both; do not assume the word “generated” means every description, default and effect is generated.
3. Run the existing registry/inventory drift checks using the repository's prescribed test command. Inspect the assertion scope and any unclassified rows or ghost entries that could pass by matching comments instead of actual reads.
4. Compare representative and potentially problematic entries from source to inventory: spelling, default, effect, classification and diagnostic usage. Regenerate the inventory if that is the established workflow and review the diff.

**Accept when:** the report names the current authoritative source and generated/manual boundaries, links the relevant test/generator, and records a clean or failing drift-check result. Correct the filename in the documentation plan only once the repository snapshot proves which name is current.

### R5. TurboQuant KV-cache modes and the guide's scope

**Disagreement to check:** the proposed guide describes “TurboQuant FP8/INT4 KV caching”, while the source inspected during review described Lloyd-Max/FastScan and KVarN modes, including 4-bit keys and 2-bit values for KVarN. Other FP8 or KV-dtype functionality may exist elsewhere; first trace the actual supported data paths.

**Procedure**

1. Trace all production KV-cache implementations from public parameters/CLI flags/environment settings into storage, encode/decode and attention consumption. Search for FP8 cache types as well as TurboQuant-specific classes; do not infer that every FP8 code path is TurboQuant.
2. Read quantizer enums, defaults, accepted configuration values, serialization/layout code and fallback behaviour. Establish the effective precision/layout for keys and values for every advertised mode.
3. Link each mode to tests and benchmarks. Confirm whether tests check round-trip/error/quality behaviour or only packing mechanics, and capture model-specific limitations if existing results demonstrate them.
4. Verify that paged allocation and prefix caching are described as separate cache mechanisms where appropriate, rather than as quantization modes. Confirm whether each is current and exposed before documenting it as a user-selectable feature.
5. If FP8 KV-cache storage does exist, identify the exact implementation and configuration that activates it and explain its relationship—or lack of relationship—to TurboQuant. If it exists only for weights or a different backend/component, do not call it FP8 TurboQuant KV caching.

**Accept when:** a source-to-configuration matrix lists each actual KV-cache mode, effective key/value representation, activation mechanism, coverage evidence and known limitations. The guide title and claims follow that matrix, not assumptions about the research paper or class names.

**Evidence to inspect:** [TurboQuantKvCache.cs](../../src/OpenTail.Stingray.Engine/TurboQuantKvCache.cs), the OpenTail.Stingray.TurboQuant implementation, cache parameters/enums, CLI and environment configuration, cache tests, and benchmarks.

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

## 5. Execution order

1. **Snapshot and source inventory (all five items):** pin one commit and gather the source, tests, docs and history. This identifies cheap checks first and prevents a stale citation from driving the investigation.
2. **Fast, deterministic checks:** external sample compilation; environment inventory drift tests; SIMD dispatch/unit tests; cache mode/config/test inspection.
3. **Model-specific reproducers:** rerun only the minimum HunyuanVideo/Wan tests needed to resolve the latest evidence. Escalate to long generation tests only if smaller checks cannot answer the disputed user-visible claim.
4. **Independent review of findings:** ensure each result distinguishes what was inspected, what ran, and what the result proves. A second pass should challenge both the earlier positive and negative interpretations.
5. **Targeted disposition:** make only evidence-backed edits to the architecture plan and affected canonical docs. File separate implementation/hardware work for unresolved product issues; keep this pass focused on finding the truth, not expanding scope.

## 6. Definition of done

- All five disagreements have a completed evidence row, even if the result is blocked or uncertain.
- Findings reference a consistent commit snapshot and reproducible commands, or explain precisely why reproduction was not possible.
- Historical failures have been checked for later fixes; recent success claims have been checked for scope and quality.
- No conclusion is based solely on a documentation label, a class name, or an unsupported assumption.
- Every changed public claim points to its canonical evidence source. Unavailable evidence is labelled as such, not converted into a failure.
- The audit report recommends a small set of plan/document edits or separately scoped engineering tasks; it does not silently implement new model or backend features.
