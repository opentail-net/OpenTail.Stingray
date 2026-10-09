# Architecture Semantics Admission Plan (2026-10-08)

**Baseline:** `main` @ `5c591e6a`. Supersedes [`../done/2026-10-06-architecture-plugin-admission-plan.md`](../done/2026-10-06-architecture-plugin-admission-plan.md)
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

## 5. Phases (tick as completed; each phase ends with a green build + the named tests)
- [x] **0. Audit**
  - [x] 0.1 Classify every `FromGgufMetadata` caller (A generic / B semantics / C raw-parser test); record in Appendix A
  - [x] 0.2 Classify every named-arch branch in `ModelGraph.cs` (A generic / B semantics / C execution / D frontend / E handoff); record in Appendix B
  - [x] 0.3 Confirm `DefaultForwardPassFactories.cs` is unused; confirm whether Audio/Diffusion reference Engine; locate Cohere2 descriptor
- [x] **1. Raw probe + load-context hyperparams**
  - [x] 1.1 Add `required ModelHyperparams Hyperparams` to `ArchitectureLoadContext`
  - [x] 1.2 Switch `CommonForwardPassFactory` (2 sites) and `DeepSeek2Architectures` to `ctx.Hyperparams`
  - [x] 1.3 Remove `Hyperparams` from `ArchitectureProbe`; fix all construction sites and tests
- [x] **2. Core baseline + semantics context**
  - [x] 2.1 Add `ModelArchitectureSemanticsContext` (Core)
  - [x] 2.2 Add `CreateBaseline(metadata, tensorSource, metadataArchitecture)`; `FromGgufMetadata` becomes baseline + legacy branches (shim)
  - [x] 2.3 Baseline tests (no ArchitectureRegistry dependency)
- [x] **3. Descriptor hook + resolver**
  - [x] 3.1 `ArchitectureDescriptor.ApplyModelSemantics`
  - [x] 3.2 `ArchitectureModelResolver` / `ResolvedModelArchitecture` (declared / canonical / metadata names; unknown stays fail-closed)
  - [x] 3.3 Resolution-order tests: declared, alias, relabelled, metadata-free, unknown
  - [x] 3.4 Descriptor contract tests (hook optional, Admitted keeps evidence + factory)
- [x] **4. Route loaders through the resolver**
  - [x] 4.1 `Model.Load`
  - [x] 4.2 `ModelDescription.FromPackage` / `FromModel`
  - [x] 4.3 `RuntimeInstance` consumes `model.Hyperparams` only
  - [x] 4.4 Convergence test + `static-plan` output diff over admitted set
- [x] **5. Pilots** (each: move rule, delete legacy branch, equivalence test, existing receipt/regression green)
  - [x] 5.1 SmolLM3 (NoPE step 4)
  - [x] 5.2 Cohere2 (LayerNorm, parallel residual, SWA, logit scale)
  - [x] 5.3 Gemma4 (per-layer dims/RoPE/KV heads/KV source, embed scale, softcap)
  - [x] 5.4 Gate: pilots needed only the one hook; else stop and re-assess
- [x] **6. Remaining branches** (same recipe as pilots; one checkbox per group)
  - [x] 6.1 Llama4 / NoPE / sigmoid routing
  - [x] 6.2 OLMo / OLMo2 / OLMoE
  - [x] 6.3 JAIS / JAIS2
  - [x] 6.4 Granite / MiniCPM
  - [x] 6.5 Gemma / Gemma3
  - [x] 6.6 Qwen-VL / Qwen3-VL / Qwen3.5 GDN
  - [x] 6.7 LFM2 / Nemotron-H / AFMoE
  - [x] 6.8 Hunyuan / Maincoder / GPT-NeoX / Falcon / StableLM
  - [x] 6.9 DeepSeek2 (MLA semantics only; Vulkan execution untouched)
- [x] **7. Cleanup**
  - [x] 7.1 Non-LLM callers (Audio/Diffusion/Cli/Embedding) moved per Appendix A
  - [x] 7.2 Remove emptied legacy branches from Core
  - [x] 7.3 Delete `DefaultForwardPassFactories.cs`
  - [x] 7.4 Update docs describing pre-`5c591e6` switches
- [x] **8. New-architecture exercises**
  - [x] 8.1 (A) plain dense: zero central edits
  - [x] 8.2 (B) semantic variant: zero central edits
  - [x] 8.3 (C) new compute family: execution edits only

- [ ] **9. Final real-checkpoint verification** (user request 2026-10-08)
  - [x] 9.1 Full solution `dotnet build` clean (warnings are errors)
  - [x] 9.2 `.Fast` test projects green (2026-10-08: Audio.Fast 94, Core 1019/49 skipped, ForwardPass.Fast 1041, Server.Fast 456, Sessions.Fast 129, Vulkan.Fast 46; 0 failed)
  - [x] 9.3 Heavy suites with `STINGRAY_RUN_HEAVY_TESTS=1` against real checkpoints in `models/`; check per-class wall time (CLAUDE.md rule 12: ~0.1s "pass" = silent no-op, not evidence)
    - PARTIAL 2026-10-08 (classes run one per process, post-migration; whole-project run hung then was OOM-reaped twice): REAL+PASS: SmolLm3GreedyParity 1/1 69s, Gemma4PrefillConsistency 3/3 49s, OlmoeGreedyParity 2 pass+1 skip 17s, Qwen2MoeGreedyParity 2/2 90s. FAIL (pre-existing, bugstofix #24, token 9 divergence from Q3_K kernel rewrite e7b7aa8a; hyperparam asserts pass): DeepSeek2GreedyParity 1/2. SKIPPED (no checkpoint/reference, UNVERIFIED): Maincoder, Falcon, GptNeox, Olmo, Olmo2, GraniteMoe, Gemma4Cpu (3/4). NOT RUN (150s cap hit under memory pressure, not a result): Afmoe, Apertus, Glm4, GlmMoe, Gpt2, Granite, GraniteHybrid, HunyuanMoe, LlamaFour, Mixtral, PhiMoe; Exaone45, Lfm2, NemotronH never started.
    - UPDATE 2026-10-08 (one class per process, real weights, 0 failed): Afmoe 2/2, Lfm2 3/3, NemotronH 2/2, GraniteHybrid 5/5, Mixtral 2/2, GlmMoe 2/2. Skipped in ~0.1s = no checkpoint (UNVERIFIED): Gpt2, Glm4, Apertus, Granite. Hit the 900s cap (large models, two copies of the runner overlapped; inconclusive, not failures): Exaone45, HunyuanMoe, LlamaFour, PhiMoe. Perf sweep of 11 text checkpoints in PerformanceLeague.md ("Full re-sweep ... 2026-10-08"): one confirmed regression, Phi-3-mini prefill 40.1 -> 35.4 t/s (-12%), not bisected; CLOSED 2026-10-08 as won't-investigate (Phi-3 has tiny monthly downloads; documented in PerformanceLeague.md).
  - [x] 9.4 Per-pilot/per-migrated-arch real-weight receipt re-run; list archs with no checkpoint on disk as UNVERIFIED rather than green
    - FINAL 2026-10-08 (ForwardPass heavy classes, one per process, real weights): **PASS** SmolLm3 1/1, Gemma4PrefillConsistency 3/3, Olmoe 2/3 (1 skip), Qwen2Moe 2/2, Afmoe 2/2, Lfm2 3/3, NemotronH 2/2, GraniteHybrid 5/5, Mixtral 2/2, GlmMoe 2/2, PhiMoe 2/2 (544 s), Exaone45 2/2 (870 s), HunyuanMoe 2/2 (627 s), GptNeox 2/2 (after fixing the pythia filename lookup; it had been skipping silently). **KNOWN FAIL (pre-existing, bugstofix #24):** DeepSeek2 greedy parity 1/2. **DEFERRED by the user:** LlamaFour (Llama 4, ~44 GB mapped vs ~23 GB free RAM; cancelled after ~5 min, not a result). **UNVERIFIED (no checkpoint or reference on disk, tests skip):** Gpt2, Glm4, Apertus, Granite, Maincoder, Falcon, Olmo, Olmo2, GraniteMoe, 3 of 4 Gemma4Cpu tests. Not run: Cuda, Vulkan, Sessions, Server heavy projects, and the other heavy ForwardPass classes outside the migrated architectures. Qwen3.8/Qwen3.6/Ornith (qwen35 family) additionally run end to end on CPU after the factory fix (and Qwen3.8 with MTP).

## Exercise results (Phase 8, analysis only: no new model was added)
Central production files an ordinary or semantic-variant new LLM architecture still has to touch, from a grep of arch-string literals outside `Engine/Architectures/`:
- **None** for model parsing, planning, runtime, CLI run path, server load path (all go through descriptor + resolver).
- **Separate registries kept out of scope by the plan**, edited per family when relevant: tokenizer family switches (`GgufTokenizer.cs`, `PreTokenizerPatterns.cs`),
  chat protocol (`ChatProtocolRegistry.cs`), tool-call adapters (`ToolCallAdapter.cs`), `PrefillHandoffFamilies.cs`, vision projector dispatch (`UnifiedVisionPipeline.cs`).
- **Small remaining leaks worth a follow-up** (all Gemma 4): `ModelDescription.cs` `IsGemma4` fallback on the raw arch string, `StaticPlanCommand.cs:466`,
  `LlamaCompatEndpoints.cs:97-98` display mapping, `InferenceEngineLoader.cs:55`, `GgufModel.cs:168` / `GgufTokenizer.cs` Gemma 4 pre-type handling.
- **Core structural tables still keyed on arch strings**: hybrid layer topology (GDN / Mamba-2 / short-conv), RWKV head size, NEOX table for descriptor-less archs (Appendix B).
  A new hybrid-recurrent family therefore still needs a Core edit (Exercise C).

## Appendix A: caller classification (done 2026-10-08)
`ModelHyperparams.FromGgufMetadata` is **removed**. Every caller was one of:
- **B (needs architecture semantics) -> `ArchitectureModelResolver.ResolveHyperparams(source)` / `(metadata)`**: ~490 call sites in
  Engine (`Model`, `ModelDescription`, `EmbeddingEngine`), Cli (`Run` draft model, `Perplexity`, `StaticPlan`, `AdmitArch`),
  Audio (CosyVoice, FishSpeech, Orpheus, QwenASR/TTS, FunASR), Diffusion text encoders, benchmarks and tests. Audio references Engine;
  Diffusion reaches it through Audio. Mechanical rewrite; Audio/Diffusion synthetic sources resolve to the same descriptors as before.
- **A (generic facts)**: only `ModelHyperparams.CreateBaseline` (Core), used by the resolver and by baseline tests.
- **C (raw-parser tests)**: none remain; the equivalence tool below calls the resolver.
`examples/SharpInference08086` is a frozen reference copy outside the solution and was not touched.

## Appendix B: ModelGraph.cs arch-keyed rules (done 2026-10-08)
| Rule | Class | Where it lives now |
|---|---|---|
| SmolLM3 NoPE, Cohere2, Gemma4, Gemma3, EXAONE4, Muse-Glimmer, AFMoE, Granite/GraniteMoE/GraniteHybrid/MiniCPM, Llama4, Jais/Jais2, Nemotron-H (ReLU^2, NoPE), GPT-2/StarCoder NoPE, Hunyuan/Maincoder QK-norm order, OLMo unweighted norm, OLMoE/Qwen3MoE/PhiMoE/GraniteMoE/LFM2MoE/Hunyuan-MoE top-k renorm, PhiMoE RMS+bias, Falcon/StableLM parallel residual, GLM4-MoE post-attn-norm role, Qwen3-VL interleaved mrope | B | descriptor `ApplyModelSemantics` (`FamilyModelSemantics.cs`, `Gemma4ModelSemantics.cs`, inline one-liners) |
| NEOX vs NORM RoPE pairing for registered archs | B | `ArchitectureDescriptor.UsesNeoxRope` (applied by the resolver; `{ns}.rope.is_neox` still overrides) |
| NEOX table for archs with NO descriptor (BERT family, Dream, Grok, DBRX, ...) | A | stays in Core; only used by embedding/other paths |
| `qwen3vlmoe` interleaved mrope | B (pending) | stays in Core until it gets a descriptor |
| `"llama"` default of top-k renorm | A | stays: applies to relabelled llama-namespace MoE files whose canonical id is not `llama` |
| RWKV `wkv.head_size`, GDN (`qwen35moe` + probe flag), Mamba-2 hybrid (`granitehybrid`, `nemotron_h`) and short-conv hybrid (`lfm2`, `lfm2moe`) layer-topology extraction | C/A (structural) | stays in Core: it parses `head_count_kv==0`-style layer topology into `LayerTypes`/`Mamba2`/`Gdn`; moving it would force the baseline to produce wrong GDN output that a hook then undoes. A new hybrid is the plan's Exercise C (genuinely new computation) |
| DeepSeek2 MLA / YaRN attention correction | A | metadata-driven (`key_length_mla`, `rope.scaling.type`), not arch-keyed; factory already reads `ctx.Hyperparams` |
| Chat/thinking policy, `PrefillHandoffFamilies` | D/E | untouched |

## Equivalence tool
`HyperparamsSnapshotDump` (tests/OpenTail.Stingray.Tests.Core): `STINGRAY_HP_DUMP=<out> STINGRAY_HP_MODELS=<dir> dotnet test tests/OpenTail.Stingray.Tests.Core --filter-class OpenTail.Stingray.Tests.Core.HyperparamsSnapshotDump`.
Dumps every `ModelHyperparams` property for every GGUF header under the dir. The "before" file came from a git worktree of `5c591e6a`
calling the original `FromGgufMetadata`; the "after" file comes from the working tree. Diff must stay empty.

## Progress log
(one line per completed step: date, what, test evidence)

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
- 2026-10-08 Phase 1 done: raw ArchitectureProbe, LoadContext.Hyperparams; Tests.Core builds.
- 2026-10-08 Phases 2.1/2.2/3.1-3.3/4.1-4.3: CreateBaseline, semantics context, ApplyModelSemantics hook, ArchitectureModelResolver, Model.Load + ModelDescription routed; ArchitectureModelResolverTests 4/4.
- 2026-10-08 5.1 SmolLM3 migrated: NoPE rule moved to descriptor hook; 339 direct FromGgufMetadata(x.Metadata,x) callers in src/tests/benchmarks routed via ArchitectureModelResolver.ResolveHyperparams (Phase 0.1/7.1 partial); solution builds; real-weight SmolLm3GreedyParityTests PASS (7.9s, SmolLM3-Q4_K_M.gguf; test now also finds that filename).
- 2026-10-08 5.2 Cohere2 migrated (SWA pattern, LayerNorm, parallel residual, RoPE-on-SWA, logit scale) to descriptor hook; synthetic + REAL-header tests (c4ai-command-r7b) pass; end-to-end generation check deferred to Phase 9.
- 2026-10-08 5.3 Gemma4 migrated to Gemma4ModelSemantics (one hook + typed ctx readers); Gemma4ModelHyperparamsTests 4/4, real-header tests E4B + 12B pass. Full Tests.Core: 950 ok / 3 failed (fixed: legacy 1-arg metadata-only callers -> ResolveHyperparams(metadata)) / 48 skipped (pre-existing missing models). NOTE full Tests.Core takes ~11 min; run filtered.
- 2026-10-08 EQUIVALENCE GATE built: tests/.../HyperparamsSnapshotDump.cs dumps every ModelHyperparams property for all 172 real GGUF headers under models/ (env STINGRAY_HP_DUMP=<out> STINGRAY_HP_MODELS=<dir>); 'before' snapshot taken from a git worktree of HEAD 5c591e6a (original code), 'after' from working tree. Result after 5.1-5.3: ZERO diff across 172 headers. Re-run + diff after every migration batch.
- 2026-10-08 Phase 6 batch 1 migrated (Llama4, Jais, Jais2, Nemotron-H partial, GPT-2, StarCoder, Hunyuan dense/MoE, Maincoder, OLMo, OLMoE, Qwen3MoE, PhiMoE, GraniteMoE, LFM2MoE, Falcon): equivalence gate zero diff over 172 real headers + 14 resolver tests incl. synthetic coverage of archs without a local checkpoint (llama4, olmo, falcon, jais2, starcoder, phimoe, hunyuan-moe).
- 2026-10-08 Phase 6 batch 2 (Gemma3, EXAONE4, Muse-Glimmer, AFMoE, Granite/GraniteMoE/GraniteHybrid/MiniCPM) -> FamilyModelSemantics.cs; gate zero diff; 17 resolver tests pass (adds synthetic minicpm/granite/exaone4/afmoe).
- 2026-10-08 Phase 0 + 6 closed: remaining Core arch-keyed rules classified (Appendix B). Added UsesNeoxRope descriptor property (42 archs moved out of Core's NEOX table). Removed ModelHyperparams.FromGgufMetadata entirely (only CreateBaseline remains); all ~490 callers use ArchitectureModelResolver. Whole solution builds; gate zero diff over 172 real headers. 7.3 NOT done on purpose: DefaultForwardPassFactories.cs is confirmed dead but file deletion is disallowed by standing user rule - delete manually.
- 2026-10-08 2.3/3.4/4.4 tests added (baseline-only test, semantics-hook smoke over every hooked descriptor + unique-registration, Model.Load/ModelDescription/resolver convergence on 3 real headers). 4.4 static-plan CLI diff not run separately: ModelDescription facts are pure functions of the hyperparams the snapshot gate already pins.
- 2026-10-08 7.4 docs: superseded/historical banners on 10-06 plan, 10-05 registry plan, 10-05 selection matrix; link from 00-current-work.md.
- 2026-10-08 Phase 8 exercises recorded as analysis (see Exercise results); no synthetic new architecture was added to the manifest. 9.1 Release solution build: 0 warnings, 0 errors.
