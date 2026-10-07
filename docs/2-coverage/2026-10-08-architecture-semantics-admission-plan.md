# Architecture Semantics Admission Plan (2026-10-08)

**Baseline:** `main` @ `5c591e6a`. Supersedes `2026-10-06-architecture-plugin-admission-plan.md`
(that plan's descriptor/registry/factory/`ExecutionPlan` work has landed; only the remainder below is open).

**Goal:** adding an ordinary model architecture is a local component change (one `*Architecture.cs` + one
`BuiltInArchitectures` line + evidence), with no edit to `ModelGraph.cs`, `ExecutionPlanner`, `ForwardPassSelection`,
`RuntimeInstance`, CLI or Server.

Origin: an external design review (ChatGPT), evaluated against the code. Verdicts are in section 6.

## 1. Already done (do not redo)
`ArchitectureDescriptor` / `ArchitectureRegistry` / `BuiltInArchitectures` (identity, aliases, admission, evidence,
`DetectFromProbe`, `RecognizeRelabelledFile`, `ForwardPassFamily`, `ConstructForwardPass`, `ArchitectureLoadContext`
fed from `ExecutionPlan`). Explicit manifest, no reflection (NativeAOT-safe). The planner stays authoritative for
backend/placement/KV/batching/speculation.

## 2. Still wrong (verified against the tree)
1. `ModelHyperparams.FromGgufMetadata` (`Core/ModelGraph.cs:601`; file is 1633 lines) is still the architecture
   catalogue: ~29 named-arch branches (Gemma*, Granite/MiniCPM, DeepSeek2, JAIS, Cohere2, Llama4, SmolLM3, OLMo*, Qwen-VL,
   LFM2, Nemotron-H, AFMoE, Hunyuan, ...). It runs **before** the registry resolves identity.
2. `ArchitectureProbe.Hyperparams` is `required`, so detection depends on already-interpreted semantics; consumers read
   `ctx.Probe.Hyperparams` (`CommonForwardPassFactory.cs:19,115`, `DeepSeek2Architectures.cs:32`).
3. `Model.cs:116`, `ModelDescription.cs:67` (FromPackage) and `:99` (FromModel) each parse and resolve independently.
4. `DefaultForwardPassFactories.cs` (12 lines) looks like dead scaffolding (only a self-reference found; re-confirm before deleting).

## 3. Target design (kept from the review)
Resolve identity first, then apply descriptor-owned semantics:

```
raw ArchitectureProbe -> ArchitectureRegistry.TryResolve -> descriptor
  -> Core generic baseline ModelHyperparams (metadata namespace chosen explicitly)
  -> descriptor.ApplyModelSemantics(ctx)   (optional, load-time, pure: baseline in, final out)
  -> final ModelHyperparams -> Model / ModelDescription -> ExecutionPlanner -> RuntimeInstance
  -> ArchitectureLoadContext{ Probe, Hyperparams, Plan, backends } -> ConstructForwardPass
```

- **One hook** on the existing descriptor: `Func<ModelArchitectureSemanticsContext, ModelHyperparams>? ApplyModelSemantics`.
  Null means the baseline is already correct, so ordinary models cost nothing. No second registry, no profile/factory registries.
- **Core-owned `ModelArchitectureSemanticsContext`**: metadata, tensor source, `DeclaredArchitecture`,
  `CanonicalArchitecture`, `MetadataArchitecture`, `Baseline`. No Engine/plan/backend types (Core must not reference Engine).
- **Three names kept distinct**: declared (what the file says), canonical (registry result), metadata namespace
  (where keys live: declared for relabelled files, canonical for metadata-free ones). Replacing `arch` with `descriptor.Id`
  blindly would break relabelled files.
- **One Engine resolver** `ArchitectureModelResolver.Resolve(source, path) -> ResolvedModelArchitecture`
  (descriptor?, raw probe, final hyperparams, the three names). It owns the ordering and is used by `Model.Load` and
  `ModelDescription.FromPackage/FromModel`. Unknown arch: `Descriptor=null`, baseline hyperparams, still fail-closed at admission
  (no new `Model.Load` throw). `NotAdmitted` descriptors still run semantics, for diagnostics.
- `ArchitectureProbe` becomes raw (drop `Hyperparams`); `ArchitectureLoadContext` gains `required ModelHyperparams Hyperparams`.
- Complex families keep semantics in a companion file (e.g. `Gemma4ModelSemantics.cs`) beside the descriptor.

## 4. Stingray-specific additions (not in the review)
- **Non-LLM callers.** ~25 `FromGgufMetadata` callers sit in Audio (CosyVoice, FishSpeech, Orpheus, QwenASR/TTS, FunASR),
  Diffusion (Flux2, HunyuanVideo, QwenImage, AceStep text encoders), Cli (Run draft model, Perplexity, StaticPlan, AdmitArch),
  `EmbeddingEngine` and benches. Several feed synthetic tensor sources that rely on `general.architecture` tricks (see comments in
  `FishSpeechTensorSource`, `QwenTtsTalkerTensorSource`, `CosyVoiceLlmTensorSource`). Phase 0 classifies each as
  A generic facts (Core baseline), B needs arch semantics (resolver; first check whether Audio/Diffusion may reference Engine,
  otherwise add a small Core-level semantics delegate), or C intentional raw-parser test.
- **Behaviour-preserving shim.** `FromGgufMetadata` stays during migration as baseline plus the legacy branches not yet moved.
  A branch is deleted from Core in the same commit its semantics move: never two live copies of a rule.
- **Equivalence gate per migrated arch**: golden test that old-path and new-path `ModelHyperparams` are field-for-field equal on
  synthetic metadata and, where a GGUF is in `models/`, on the real header (disk budget: one model at a time).
  Real-weight receipts (CLAUDE.md rule 14) are re-run, never replaced by synthetic tests.
- **Planning facts** (`CanBatch`, `LayerHeadDim`, ...) in `ModelDescription` must come from the resolved hyperparams; add a
  `static-plan` output diff test over the admitted set.
- **Descriptor contract tests**: hook optional; Admitted descriptors keep evidence + factory; hook signature takes only Core types.
- `PrefillHandoffFamilies`, chat formatting, multimodal/vision dispatch, SafeTensors: out of scope, unchanged.

## 5. Phases
0. Audit callers; classify each `ModelGraph.cs` branch: A generic / B semantics / C execution / D frontend / E handoff (table in an appendix to this doc).
1. Raw `ArchitectureProbe`; `ArchitectureLoadContext.Hyperparams`; move the 3 factory sites off `Probe.Hyperparams`.
2. Core `CreateBaseline` + `ModelArchitectureSemanticsContext`; legacy shim keeps old behaviour.
3. `ApplyModelSemantics` + `ArchitectureModelResolver`, with resolution-order tests (declared / alias / relabelled / metadata-free / unknown).
4. Route `Model.Load` and `ModelDescription.FromPackage/FromModel` through the resolver; convergence test.
5. Pilots in order: **SmolLM3** (NoPE step 4), **Cohere2** (LayerNorm, parallel residual, SWA, logit scale), **Gemma4**
   (per-layer dims/RoPE/KV heads/KV source, embed scale, softcap). If Gemma4 needs more than the one hook, stop and re-assess.
6. Remaining branches by group (Llama4/NoPE, OLMo*, JAIS*, Granite/MiniCPM, Qwen-VL, LFM2, Nemotron-H, AFMoE, Hunyuan, Maincoder);
   DeepSeek2 last (MLA semantics separate from its Vulkan execution).
7. Delete emptied legacy branches and `DefaultForwardPassFactories.cs`; update docs still describing pre-`5c591e6` switches.
8. New-arch exercises: (A) plain dense, (B) semantic variant: zero central edits; (C) new compute family: execution edits allowed.

**Done when:** an ordinary new arch is a new file + one registration + evidence; no factory reads `Probe.Hyperparams`; one
resolution path; admission stays fail-closed; real-weight receipts intact; no reflection.

## 6. Review verdicts
- **Adopted:** raw probe, identity-before-semantics, single descriptor hook, three-name namespace model, Core context type,
  single resolver, `LoadContext.Hyperparams`, the three pilots, DeepSeek2 last, success metric "central files touched".
- **Adjusted:** unknown archs stay fail-closed at admission; legacy shim with per-branch deletion; non-LLM callers handled
  explicitly; per-arch equivalence gates.
- **Dropped as redundant:** long non-goal prose, duplicate definition-of-done lists, recommendations to keep things already kept
  (`ForwardPassFamily`, manifest, planner ownership).
- **Unverified:** whether Audio/Diffusion reference Engine; whether Cohere2 has a descriptor (no `Cohere*` file, probably in `OtherAdmittedArchitectures.cs`).
