# Completed items split from 2026-10-08-architecture-capabilities-plan.md (2026-10-30)

Finished items moved verbatim out of [2026-10-08-architecture-capabilities-plan.md](2026-10-08-architecture-capabilities-plan.md). Open items stay there. Each block notes the section it came from.

> from: ## Phases
  - [x] 0.1 (done 2026-10-08: `ArchitectureFacts_MatchBaseline`, 71 architectures, `Baselines/ArchitectureFacts.txt`) Snapshot test: for every registered architecture, serialize today's `ModelDescription` facts, chat-protocol selection, tool-call adapter id, llama-compat name, `SupportedBackends` and batching capability into a checked-in baseline (`tests/.../ArchitectureFactsBaseline.json`). Phases 1-3 must leave it unchanged except where a phase says otherwise.

> from: ## Phases
  - [x] 0.2 (done 2026-10-08: `ArchitectureLiterals_OutsideTheArchitectureFolder_OnlyShrink`, a ratchet over 37 files, `Baselines/ArchitectureLiterals.txt`; may only shrink) **Architecture-literal guard test:** scan `src/**/*.cs` for any registered architecture id used as a string literal outside `Engine/Architectures/`, with an explicit allowlist initialised from the inventory above. It fails on any *new* literal and shrinks as phases land. This is what stops regression.

> from: ## Phases
  - [x] 0.4 (the 2026-10-08 smoke sweep in PerformanceLeague.md is the pre-change reference; re-run it at the end of Phase 3 and 5) Re-run the smoke matrix (`scripts` copy of the 2026-10-08 smoke run, 8-10 representative models incl. Qwen3.8 with MTP) and save outputs as the pre-change reference. Greedy text must match after every phase.

> from: ## Phases
- [x] **1. Capabilities record; remove the capability special cases (inventory #2, 6, 7, 8)**
  - [x] 1.1 (done 2026-10-08; `ArchitectureDescriptor.Capabilities` + `ArchitectureCapabilities` record; `ExecutionPlanner`, `ModelDescription` and `ForwardPassSelection` now read only it; decision: provide it as a read-only projection `descriptor.Capabilities` over the existing flat init fields instead of moving 71 descriptors' syntax; same planner-facing narrowing, far less churn) Introduce `ArchitectureCapabilities` (move `SupportedBackends`, `SupportsContinuousBatching`, `CanBatchPredicate`, `SupportsImageInput`, `SupportsAudioInput` into it; add `IsGemma4`-style needs as named capabilities, not "is gemma4").
  - [x] 1.2 (done 2026-10-08, 7 literals removed: ModelDescription IsGemma4 now structural only; server image-input check reads `SupportsImageInput`; Perplexity reads `ForwardPassFamily`; static-plan fallback dropped. Behavior identical: only gemma4 declares image input) Rewrite `ModelDescription.IsGemma4`, `InferenceEngineLoader`, `PerplexityCommand`, `StaticPlanCommand` to read capabilities / `CreateForwardPass`.
  - [x] 1.3 (done 2026-10-08; plan JSON for 5 hybrid/dense models identical before and after except timestamps; facts baseline changed in exactly the 3 family lines; reverse validation + test added) Declare `ForwardPassFamily.HybridGdn` on `qwen35`, `qwen35moe`, `granitehybrid`; add the reverse validation in `ArchitectureDescriptor` (each factory requires its family and vice versa). Check first what `ForwardPassSelection` does with the family so this does not change plan selection (baseline test 0.1 covers it).
  - [x] 1.4 (not needed: the flat init fields stay, so there is nothing to forward or remove)
  - **Exit:** baseline unchanged; guard allowlist loses entries 2, 6, 7, 8; planner still touches only `Capabilities.SupportedBackends` (assert in 0.2's test that `Planning/` reads no other descriptor member).

> from: ## Phases
- [x] **2. Integrations record; fold the chat and tool-call duplication (inventory #3, 4, 5)** (done 2026-10-08, with two deliberate deviations noted below)
  - [x] 2.1 (done: `ArchitectureDescriptor.ChatProtocolId` set on the 9 non-default descriptors (llama, llama4, granite, granitehybrid, granitemoe, gemma, gemma2, gemma3, gemma4); `ChatProtocolRegistry.For` resolves through the descriptor; the per-protocol `Architectures` arrays and `ChatProtocol.Architectures` are gone; facts baseline `chatProtocol` column unchanged for all 71 architectures. **Deviation: `FallbackChat` is NOT retired**, see D3 below.) Add `ChatProtocolId`; make `ChatProtocolRegistry` look up by id from the descriptor and delete its arch arrays. Retire `FallbackChat` or reduce it to a derived value. Resolve the two-mechanism overlap so there is exactly one source of truth.
  - [x] 2.2 (decided NOT to move: `ToolCallAdapterRegistry` is already pluggable via `Register`, its table is keyed by adapter implementation, and routing it through descriptors would need Engine-to-Core registration at startup with static-initialization-order risk for little gain; left as is) Add `ToolCallAdapterId`; keep adapter implementations in Core, resolve through the id.
  - [x] 2.3 (done by reuse: the llama-server `/props` template name is now `ChatProtocolRegistry.For(arch).Id`, no separate field. Side effect: gemma3 now reports `gemma` instead of `chatml`, granite family reports `granite`) Add `LlamaCompatName`.
  - **Exit:** baseline unchanged (including rendered chat prompt per architecture for a fixed message set, which is added to the baseline in 0.1 for this phase); allowlist loses 3, 4, 5.
  - **Risk:** prompt rendering is user-visible. The per-architecture rendered-prompt snapshot is the safeguard; do not merge without it.

> from: ## Phases
  - [x] 3.2 (`Core/ModelArchitectureTraits.cs`: Hybrid kind, single-sublayer, wkv head size, default gating func, default top-k normalize, multi-axis rope keys, deepstack; `ArchitectureDescriptor.Traits` declared on 12 descriptors; `CreateBaseline(..., traits)`; ModelGraph.cs went from 15 architecture-name literals to 1, the NeoX list for architectures without descriptors) Add `ArchitectureTraits` (hybrid kind, single-sublayer, head-dim source) and pass them to Core through `ModelArchitectureSemanticsContext`, replacing the `arch is "granitehybrid" or ...` tests in `ModelGraph.cs`.

> from: ## Phases
  - [x] 3.3 (the by-name table `ModelArchitectureTraits.Legacy` is the shim: 12 literals in one place, pinned to the descriptors by `DescriptorTraits_MatchTheByNameTable`; delete it when the direct `CreateBaseline`/`FromGgufMetadata` callers are gone; `qwen3vlmoe` has no descriptor and lives only there) The legacy parser stays as the shim for the non-LLM callers (earlier plan Appendix A) until they are moved; mark each remaining literal with the caller that still needs it.

> from: ## Phases
- [x] **4. Layout (mechanical, last)** (done 2026-10-08)
  - [x] 4.1 (done: `OtherAdmittedArchitectures` is now a partial class across 10 files by family: Olmo, GptFamily, Exaone, Mistral, HybridRecurrent, ErnieGlm, Hunyuan, Jais, CnVendor, plus the remainder (7 descriptors) in the original file; all 37 descriptors preserved line for line, verified by comparing sorted non-blank lines; the descriptors are independent so partial-class static initialization order cannot matter; all references to `OtherAdmittedArchitectures.X` are unchanged) Split `OtherAdmittedArchitectures.cs` by family into files of sensible size; give architectures with real semantics or a factory their own file.
  - [x] 4.2 (decided NOT to move: its helpers are shared by several families (Granite by three descriptors, GatedSwa by two files, Gemma3, Exaone4), so there is no single "next to" location; left as the shared-helpers file) Move `FamilyModelSemantics.cs` content next to the architectures that use it.
  - **Exit:** no behavior change; diff is moves only (verify with `git diff --stat -M`).

> from: ## Phases
- [x] **5. Prove it and document it**
  - [x] 5.1 (done 2026-10-08, performed on the working tree and reverted; results below) **New-architecture exercise:** add a hypothetical dense architecture and a hypothetical hybrid in a test project; assert each needs edits in **one** file only and the guard test stays green.
  - [x] 5.2 (done 2026-10-08: publish succeeded once `vswhere` was on PATH, 22.8 MB `stingray.exe`, no IL warnings in the output; it loaded and generated on SmolLM2-135M, granite-4.0-h-350m and LFM2-8B-A1B, with text identical to the JIT run on the two hybrids) NativeAOT publish (`dotnet publish src/OpenTail.Stingray.Cli -c Release -r win-x64`) builds and runs a smoke model; confirms no reflection crept in.
  - [x] 5.3 (done: `docs/reference/adding-an-architecture.md`, linked from CLAUDE.md) Update `docs/reference/OpenTail.Stingray-Design.md` (descriptor = model plugin; planner = machine/request policy) and the "add a new architecture" recipe.
  - **Exit:** smoke matrix from 0.4 reproduces identically; AOT binary works.

