# Architecture descriptors as statically registered model plugins: finish the job (plan)

**Date:** 2026-10-08. **Status:** proposed, not started. **Follows:** [2026-10-08-architecture-semantics-admission-plan.md](2026-10-08-architecture-semantics-admission-plan.md) (all phases 1-8 done).
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

- [ ] **0. Baseline and guard (do first; makes the rest safe)**
  - [ ] 0.1 Snapshot test: for every registered architecture, serialize today's `ModelDescription` facts, chat-protocol selection, tool-call adapter id, llama-compat name, `SupportedBackends` and batching capability into a checked-in baseline (`tests/.../ArchitectureFactsBaseline.json`). Phases 1-3 must leave it unchanged except where a phase says otherwise.
  - [ ] 0.2 **Architecture-literal guard test:** scan `src/**/*.cs` for any registered architecture id used as a string literal outside `Engine/Architectures/`, with an explicit allowlist initialised from the inventory above. It fails on any *new* literal and shrinks as phases land. This is what stops regression.
  - [ ] 0.3 Test that every key in `PrefillHandoffFamilies` names a registered architecture.
  - [ ] 0.4 Re-run the smoke matrix (`scripts` copy of the 2026-10-08 smoke run, 8-10 representative models incl. Qwen3.8 with MTP) and save outputs as the pre-change reference. Greedy text must match after every phase.
  - **Exit:** baseline + guard committed, passing on the unchanged tree.

- [ ] **1. Capabilities record; remove the capability special cases (inventory #2, 6, 7, 8)**
  - [ ] 1.1 Introduce `ArchitectureCapabilities` (move `SupportedBackends`, `SupportsContinuousBatching`, `CanBatchPredicate`, `SupportsImageInput`, `SupportsAudioInput` into it; add `IsGemma4`-style needs as named capabilities, not "is gemma4").
  - [ ] 1.2 Rewrite `ModelDescription.IsGemma4`, `InferenceEngineLoader`, `PerplexityCommand`, `StaticPlanCommand` to read capabilities / `CreateForwardPass`.
  - [ ] 1.3 Declare `ForwardPassFamily.HybridGdn` on `qwen35`, `qwen35moe`, `granitehybrid`; add the reverse validation in `ArchitectureDescriptor` (each factory requires its family and vice versa). Check first what `ForwardPassSelection` does with the family so this does not change plan selection (baseline test 0.1 covers it).
  - [ ] 1.4 Keep the old property names as forwarding properties for one phase to keep the diff reviewable; remove them at the end of the phase.
  - **Exit:** baseline unchanged; guard allowlist loses entries 2, 6, 7, 8; planner still touches only `Capabilities.SupportedBackends` (assert in 0.2's test that `Planning/` reads no other descriptor member).

- [ ] **2. Integrations record; fold the chat and tool-call duplication (inventory #3, 4, 5)**
  - [ ] 2.1 Add `ChatProtocolId`; make `ChatProtocolRegistry` look up by id from the descriptor and delete its arch arrays. Retire `FallbackChat` or reduce it to a derived value. Resolve the two-mechanism overlap so there is exactly one source of truth.
  - [ ] 2.2 Add `ToolCallAdapterId`; keep adapter implementations in Core, resolve through the id.
  - [ ] 2.3 Add `LlamaCompatName`.
  - **Exit:** baseline unchanged (including rendered chat prompt per architecture for a fixed message set, which is added to the baseline in 0.1 for this phase); allowlist loses 3, 4, 5.
  - **Risk:** prompt rendering is user-visible. The per-architecture rendered-prompt snapshot is the safeguard; do not merge without it.

- [ ] **3. Structural traits; shrink Core's parser (inventory #1)**
  - [ ] 3.1 Decide D1 (below).
  - [ ] 3.2 Add `ArchitectureTraits` (hybrid kind, single-sublayer, head-dim source) and pass them to Core through `ModelArchitectureSemanticsContext`, replacing the `arch is "granitehybrid" or ...` tests in `ModelGraph.cs`.
  - [ ] 3.3 The legacy parser stays as the shim for the non-LLM callers (earlier plan Appendix A) until they are moved; mark each remaining literal with the caller that still needs it.
  - **Exit:** `ModelHyperparams` equality test across all admitted architectures (the old-vs-new equality harness from the semantics plan) still passes; real-weight spot checks on Granite-H, NemotronH, LFM2 and Qwen3.8 (these are the hybrids that exercise the moved rules); static-plan output diff clean.

- [ ] **4. Layout (mechanical, last)**
  - [ ] 4.1 Split `OtherAdmittedArchitectures.cs` by family into files of sensible size; give architectures with real semantics or a factory their own file.
  - [ ] 4.2 Move `FamilyModelSemantics.cs` content next to the architectures that use it.
  - **Exit:** no behavior change; diff is moves only (verify with `git diff --stat -M`).

- [ ] **5. Prove it and document it**
  - [ ] 5.1 **New-architecture exercise:** add a hypothetical dense architecture and a hypothetical hybrid in a test project; assert each needs edits in **one** file only and the guard test stays green.
  - [ ] 5.2 NativeAOT publish (`dotnet publish src/OpenTail.Stingray.Cli -c Release -r win-x64`) builds and runs a smoke model; confirms no reflection crept in.
  - [ ] 5.3 Update `docs/reference/OpenTail.Stingray-Design.md` (descriptor = model plugin; planner = machine/request policy) and the "add a new architecture" recipe.
  - **Exit:** smoke matrix from 0.4 reproduces identically; AOT binary works.

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
