# Architecture descriptors as statically registered model plugins: finish the job (plan)

> Completed items moved to [done/2026-10-08-architecture-capabilities-plan-completed-items-2026-10.md](../done/2026-10-08-architecture-capabilities-plan-completed-items-2026-10.md) on 2026-10-30. This file keeps only open work.

**Date:** 2026-10-08. **Status:** All phases (0-5) done 2026-10-08. **Follows:** [2026-10-08-architecture-semantics-admission-plan.md](2026-10-08-architecture-semantics-admission-plan.md) (all phases 1-8 done).
**Origin:** an outside design suggestion (ChatGPT): make each architecture a self-contained "model plugin" that owns what is true of the model, while the planner owns what is true of the machine and the request. This plan checks that suggestion against the code at commit `6a768250` and scopes what is actually worth doing.

## Verdict

**Reasonable, and mostly already true. Do a bounded finish, not a redesign.** What the suggestion describes as the destination is largely where `ArchitectureDescriptor` already is: identity, aliases, admission, evidence, model semantics (`ApplyModelSemantics`), RoPE flag, forward-pass family, factory (`CreateForwardPass`), load setup, and capability fields (`SupportedBackends`, `SupportsContinuousBatching`, `CanBatchPredicate`, `SupportsImageInput`, `SupportsAudioInput`, `ProjectorFileHints`). The planner already reads `SupportedBackends` from the descriptor and does not hard-code "Qwen -> CUDA" (`ExecutionPlanner.cs:411-473`). The Phase 8 exercises showed an ordinary new architecture needs no central edits for parsing, planning, the CLI run path or the server load path.

What remains is a **short, countable list of places that still branch on an architecture string** (inventory below), plus two real duplications. The boundary the suggestion calls "the most important shape Stingray has discovered" (model knowledge vs machine/request knowledge) is right, and the cheapest way to keep it true is a **guard test**, not more structure. That test is Phase 0.

### Corrections to the suggestion
1. **No dynamic plugin loading, agreed, and it is not on the table.** Static registration keeps NativeAOT / no-reflection intact.
2. **Layering limits "move it all into the descriptor".** `OpenTail.Stingray.Core` is below `Engine` and cannot see `ArchitectureDescriptor`. Facts that Core's parser needs must be *passed in* (as `ModelArchitectureSemanticsContext` already is) or stay in Core until the non-LLM callers are moved. See Phase 3 and decision D1.
3. **Vision is already keyed correctly.** `UnifiedVisionPipeline` selects an encoder by the mmproj file's `projector_type`, not by the LLM architecture. Those ~15 string hits are projector types, not architecture branches. Out of scope.
4. **`PrefillHandoffFamilies` is planner-owned evidence, not model knowledge.** It is a table of measured GPU hand-off decisions keyed by architecture plus a metadata fingerprint. It stays with the planner; it only needs a test that its keys are registered architectures.
5. **One file per architecture is a layout choice, not a design change.** Do it where an architecture has real semantics or a factory (Qwen35, Gemma4, Granite, NemotronH, DeepSeek2, gpt-oss...). The 37 plain-dense descriptors in `OtherAdmittedArchitectures.cs` (1,095 lines) can group by family. Mechanical, reviewable, last.

## Inventory: where architecture knowledge still lives outside `Engine/Architectures/`

Found by grepping architecture-id literals in `src/` (excluding Audio/Diffusion and the architecture folder).

| # | Location | What it knows | Owner after this plan | Phase |
|---|---|---|---|---|
| 1 | `Core/ModelGraph.cs:735-780` | `qwen35moe` => hybrid GDN; `granitehybrid`/`nemotron_h` => Mamba-2 hybrid; `lfm2`/`lfm2moe` => short-conv hybrid; `nemotron_h` single-sublayer blocks; `rwkv6/7` head dim; line 697 list of refused arches | descriptor **traits**, passed into Core via the semantics context | 3 |
| 2 | `Engine/Planning/ModelDescription.cs:170,241` | `IsGemma4` derived from `hp.LayerHeadDim is not null \|\| arch == "gemma4"` | descriptor **capability** flag | 1 |
| 3 | `Engine/Chat/ChatProtocolRegistry.cs` (per-protocol `Architectures` arrays) **and** descriptor `FallbackChat` | which chat format an architecture uses, **in two places** (`RunCommand.cs:2937-2998` reads `FallbackChat`) | one: descriptor `ChatProtocolId` | 2 |
| 4 | `Core/ToolCallAdapter.cs:166+` | arch-keyed dictionary of adapters (`qwen2`, `qwen3`, `qwen35`, ...) | descriptor `ToolCallAdapterId` (Core keeps the adapters) | 2 |
| 5 | `Server/Endpoints/LlamaCompatEndpoints.cs:97-98` | arch -> llama.cpp compat family name | descriptor `LlamaCompatName` | 2 |
| 6 | `Server/InferenceEngineLoader.cs:55` | `targetArch != "gemma4"` special case | capability flag | 1 |
| 7 | `Cli/PerplexityCommand.cs:402-412` | picks `rwkv6/7` and `gpt-oss` forward passes by arch | `descriptor.CreateForwardPass` / `ForwardPassFamily` | 1 |
| 8 | `Cli/StaticPlanCommand.cs:466` | `?? (arch == "gemma4")` fallback for image input | drop the fallback; descriptor is authoritative | 1 |
| 9 | `Engine/PrefillHandoffFamilies.cs` | measured hand-off table keyed by arch + fingerprint | **planner** (stays); add key-registered test | 0 |
| 10 | `Vision/*` | projector types | out of scope | n/a |
| 11 | `Core/PreTokenizerPatterns.cs`, `GgufTokenizer.cs` | tokenizer `pre` names, not architectures | out of scope | n/a |

## Target shape

Split the descriptor's growing field list into three small immutable records so it does not become a 40-field bag, and so the planner can be shown to read only one of them:

```
ArchitectureDescriptor
 ├─ identity / admission / evidence        (as today)
 ├─ ApplyModelSemantics, UsesNeoxRope      (as today; model facts that change hyperparameters)
 ├─ Capabilities  : what the model CAN do   -> read by the PLANNER
 │     backends, continuous batching (+ predicate), image/audio input,
 │     speculative/MTP head, recurrent state, forward-pass family
 ├─ Traits        : structural facts about the model -> read by PARSERS / loaders
 │     hybrid kind (none | gated-delta-net | mamba2 | short-conv), single-sublayer blocks,
 │     per-layer shapes, head-dim source
 └─ Integrations  : names of the things that adapt the model to the outside world
       chat protocol id, tool-call adapter id, llama-compat name, projector file hints
```

Rule that keeps the boundary honest: **a descriptor says what the model can do; it never says what to do on this machine.** "Supports CUDA" is a capability. "Use CUDA" is the planner's decision from machine + request.

## Phases

Each phase ends green: `dotnet build` 0 warnings (warnings are errors), the `.Fast` suites pass, and the phase-specific check below. One commit per phase.

- [x] **0. Baseline and guard (do first; makes the rest safe)** (0.3 deferred)
  - [ ] 0.3 (deferred: low value, table is private) Test that every key in `PrefillHandoffFamilies` names a registered architecture.
  - **Exit:** baseline + guard committed, passing on the unchanged tree.



- [x] **3. Structural traits; shrink Core's parser (inventory #1)** (done 2026-10-08)
  - [ ] 3.1 Decide D1 (below).
  - **Exit:** `ModelHyperparams` equality test across all admitted architectures (the old-vs-new equality harness from the semantics plan) still passes; real-weight spot checks on Granite-H, NemotronH, LFM2 and Qwen3.8 (these are the hybrids that exercise the moved rules); static-plan output diff clean.



## Out of scope
- Dynamic DLL / assembly discovery (NativeAOT, trimming, no reflection).
- Moving execution policy (which backend, how many layers on GPU, MTP depth, tiering) into descriptors. It stays in the planner.
- Changing admission policy (CLAUDE.md rule 14: "ported, not verified" stays internal).
- Vision encoder selection (already projector-keyed) and tokenizer `pre` patterns.
- Audio / Diffusion pipelines (separate registries).

## Decisions needed
- **D1. May Core accept a traits input from Engine?** Recommended **yes**, via the existing `ModelArchitectureSemanticsContext`. It adds no new dependency direction (Engine already calls into Core) and lets the arch literals leave `ModelGraph.cs`. The alternative (leave structural rules in Core permanently) keeps ~9 literals in the lowest layer forever.
- **D2. File layout:** per-architecture files only where there is real semantics/factory (recommended) vs strictly one file per architecture (~60 tiny files).
- **D3. `FallbackChat`:** retire entirely (recommended once `ChatProtocolId` exists) vs keep as a derived convenience.

## Risks
| Risk | Mitigation |
|---|---|
| Behavior drift in chat prompts or tool-call parsing | rendered-prompt + adapter-id snapshot in the Phase 0 baseline; Phase 2 does not merge without it |
| Descriptor becomes a 40-field bag | three records; planner reads `Capabilities` only (enforced by test) |
| Core/Engine layering tangles | traits passed *in* through the semantics context; no Core reference to Engine |
| Hybrid-model regressions invisible to Fast suites (the `qwen35` factory crash on 2026-10-08 was exactly this) | Phase 3 requires real-weight runs of the four hybrids; Phase 1.3 adds the two-way family/factory validation |
| Review fatigue | one commit per phase; Phase 4 is moves only |

## Size (relative, no calendar promises)
P0 small. P1 small-medium. P2 medium (prompt-rendering care). P3 medium (layering + real-weight checks). P4 small but noisy. P5 small. The value is front-loaded: Phase 0 plus Phase 1 remove most of the central branching and give the guard that keeps it removed.

## Findings from writing this plan (worth acting on regardless)
- Chat format is selected by **two** mechanisms that can disagree (`ChatProtocolRegistry.Architectures` vs `FallbackChat`).
- The `qwen35` / `qwen35moe` descriptors pointed at the wrong factory (`CreateDense` for a Gated-DeltaNet hybrid) until fixed on 2026-10-08, and **no check could notice**: they never declared a `ForwardPassFamily`, so it defaulted to `Dense`, and the only validation in `ArchitectureDescriptor` is one-directional (a *non-dense* family may not use `CreateDense`). `granitehybrid` has the same gap (uses `CreateHybridGdn`, declares no family). Concrete fix, added to Phase 1 below: declare `ForwardPassFamily.HybridGdn` on `qwen35`, `qwen35moe` and `granitehybrid`, and add the reverse check (`CreateHybridGdn` requires `HybridGdn`; likewise for the other families).

## Phase 2 findings (2026-10-08)
- **`FallbackChat` and `ChatProtocolId` answer different questions and disagree for six architectures.** `FallbackChat` picks the prompt layout the CLI (`RunCommand`) and server (`ChatTemplate`) fall back to when a GGUF has no usable Jinja template; the chat protocol drives the engine API (`ChatSession`: rendering, media placeholders, thinking markers, output parser). For gemma, gemma2, gemma3, gemma4 and granitehybrid/granitemoe the fallback says ChatML while the protocol says gemma/granite. Merging them would change no-template behavior for those families, so they stay separate and **D3 is answered "keep, document"** unless you decide those families' no-template fallback should follow the protocol (then it is a one-line change per descriptor plus a behavior note).
- Four llama family names (`llama3`, `llama31`, `llama32`, `llama33`) that callers pass as an "architecture" keep selecting the llama3 protocol through a small table inside `ChatProtocolRegistry`; they are not descriptors and must not become admitted architectures.
- Literals removed from the ratchet: `ChatProtocolRegistry.cs` 22 -> 3, `LlamaCompatEndpoints.cs` 9 -> 2.

## Phase 3 evidence (2026-10-08)
- **Resolved hyperparameters are identical before and after for every real header on disk**: `HyperparamsSnapshotDump` over `models\_models`, `F:\_models`, `K:\_other_models` and `H:\_models` (about 175 distinct models, 13,588 property lines for the largest folder) differs in zero lines. The files cover glm4moe, granitehybrid, lfm2, lfm2moe, nemotron_h, paddleocr, qwen2vl, qwen35, qwen35moe, qwen3vl and llama. RWKV and qwen3vlmoe have no real checkpoint here; synthetic resolver tests cover them.
- Static plan JSON for five models (Qwen3.8, Qwen3.6, Granite-H, LFM2, Gemma4): 0 differing lines apart from the timestamp.
- Real-weight generation (`The capital of France is`, 16 tokens): granite-4.0-h-1b, granite-4.0-h-350m and LFM2-8B-A1B produce text identical to the 2026-10-08 sweep.
- Tests: ArchitectureCapabilitiesBaselineTests 14, ArchitectureModelResolverTests 18, RealHeaderSemanticsTests 7 (real headers), descriptor/registry contract tests pass; ForwardPass.Fast 1056, Server.Fast 456, Sessions.Fast 129 pass.
- **Net literal count in Core is only -2 (ModelGraph 15 -> 1, plus 12 in the shim table)**. The gain is structural: on the Engine path a new hybrid or multi-axis-rope architecture declares `Traits` in its own descriptor and needs no Core edit. The shim table is the remaining debt.
- Behavior note: a relabelled file (declared name differs from the resolved descriptor) now uses the resolved descriptor's traits; before it used the declared name's. No real file on disk exercises the difference.

## Phase 5 exercise results (2026-10-08)
Added two hypothetical families to the working tree, ran the guards, then reverted (nothing committed):
- `exercise-dense` (plain descriptor with `CreateDense`) and `exercise-hybrid` (`ForwardPassFamily.HybridGdn`, `Traits = Mamba2`, an `ApplyModelSemantics` tweak, `CreateHybridGdn`).
- **Code edits: one new file with the two descriptors plus one `yield return` per family in `BuiltInArchitectures.cs`.** Both resolved correctly through `ArchitectureModelResolver` with **no Core or planner edit**: the hybrid got its Mamba-2 layer pattern from `Traits` (`IsMamba2Layer` populated), its tweak from the semantics hook (`NoRopeLayerStep = 4`), and `Capabilities.Family = HybridGdn`; the dense one resolved with the default chat protocol.
- **Gates that fired, as designed:** the facts baseline (regenerate: one command), and the admitted-set snapshot, which exists in **two** tests (`ArchitectureDescriptorContractTests` and `ArchitectureRegistryTests`, a duplication worth removing). Not fired: the literal ratchet, the planner-boundary test, descriptor validation, forward-pass selection tests.
- **One real defect found in my own Phase 3 work and fixed:** `DescriptorTraits_MatchTheByNameTable` demanded that every descriptor with `Traits` also appear in Core's by-name table, so a new hybrid *did* force a Core edit. The test now only constrains architectures the table already knows. Without this exercise that trap would have shipped.
- Not measured: the verification cost (checkpoint, independent reference, STATUS row), which is unchanged by this work and dominated historically (see the 2026-10-08 discussion of pre-October admissions).
