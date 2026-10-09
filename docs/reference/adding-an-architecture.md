# Adding a model family (architecture)

**Principle.** An `ArchitectureDescriptor` says what is true of the *model*. The planner (`ExecutionPlanner`, `ForwardPassSelection`) decides what to do on *this machine for this request*. A descriptor never says "use CUDA"; it says "I can run on CUDA" (`Capabilities`), and the planner reads only `descriptor.Capabilities` (a test enforces it). Registration is static (`BuiltInArchitectures`, no reflection), so NativeAOT and trimming stay clean. There is no dynamic plugin loading, deliberately.

Measured 2026-10-08 by actually adding a hypothetical dense family and a hypothetical Mamba-2 hybrid to the working tree and running the guards
([plan](../2-coverage/2026-10-08-architecture-capabilities-plan.md), Phase 5): the code edit is one descriptor plus one manifest line. The rest below are
deliberate gates, not code.

**Admitting a checkpoint as an agent?** Start with [architecture-admission-agent-playbook.md](architecture-admission-agent-playbook.md) (run `stingray scout` first).

## Steps for an ordinary family
1. **Descriptor.** Add a `static readonly ArchitectureDescriptor` in `src/OpenTail.Stingray.Engine/Architectures/` (a family file, e.g. `GptFamilyArchitectures.cs`, or its own `<Name>Architecture.cs` when it has real semantics or a factory).
   Required: `Id`, `Status`, `EvidenceDoc`, exactly one of `StatusAnchor` / `StatusExemption`, and `CreateForwardPass` (use `CommonForwardPassFactory.CreateDense`, or `.CreateHybridGdn` with `ForwardPassFamily.HybridGdn`). `Validate()` runs when the registry is built and refuses inconsistent combinations.
2. **Manifest.** One `yield return` line in `BuiltInArchitectures.cs`.
3. **Admission gates (intentional, so admission is a conscious edit):** add the id to the admitted-set snapshot in **both** `ArchitectureDescriptorContractTests.AdmittedSet_IsExactlyTheSnapshot` and `ArchitectureRegistryTests.AdmittedSet_IsExactlyTheSnapshot` (there are two copies today; a candidate for de-duplication), then regenerate the facts baseline:
   `STINGRAY_UPDATE_BASELINE=1 tests/OpenTail.Stingray.Tests.Core/bin/Debug/net10.0/OpenTail.Stingray.Tests.Core.exe -class OpenTail.Stingray.Tests.Core.ArchitectureCapabilitiesBaselineTests`
   and review the one new line in `Baselines/ArchitectureFacts.txt`.
4. **Evidence and docs (CLAUDE.md rules 10 and 14):** a real checkpoint, an independent reference (llama-server greedy match or perplexity), timed runs, then the `docs/STATUS.md` row. Until that exists the family stays "NotAdmitted"/internal. This step, not the code, is where the time goes.
   The sequence: `stingray pull -r <owner/repo>` (verifies the published SHA-256) -> `stingray capture-golden -m <gguf> --prompt "..." --mode teacher --min-confident N --expect <hyperparam>=<value>,...` (records the llama-server reference as token ids with the file's hash, no paths) -> `stingray admit-arch -m <gguf> --golden <file>` (exit 0 = exact or near-tie only, 1 = diverged) -> commit the golden under `tests/OpenTail.Stingray.Tests.ForwardPass/Goldens/` together with the descriptor. `GoldenParityTests` then runs it automatically; no new test class. Provenance prose goes in the golden's `notes`. See [the golden-parity plan](../2-coverage/2026-10-08-golden-parity-and-admission-tooling-plan.md).

## When the family has a twist, declare it on the descriptor (no Core edit)
| Need | Where |
|---|---|
| Hyperparameter interpretation (NoPE every N layers, parallel residual, embedding scale, per-layer shapes) | `ApplyModelSemantics = ctx => ctx.Baseline with { ... }` (see `Gemma4ModelSemantics.cs`, `FamilyModelSemantics.cs`) |
| Layout the parser needs while reading the file (Mamba-2 / short-conv / gated-delta-net hybrid, single-sublayer blocks, wkv head size, multi-axis rope keys) | `Traits = new() { Hybrid = HybridKind.Mamba2, ... }` |
| Rope variant | `UsesNeoxRope = true` |
| Chat format beyond ChatML | `ChatProtocolId = "llama3" \| "llama4" \| "granite" \| "gemma"` (a new format needs a new `ChatProtocol` registered in `ChatProtocolRegistry`) |
| No-template fallback prompt layout (CLI/server) | `FallbackChat` (distinct from `ChatProtocolId`; see the plan's Phase 2 findings) |
| Backend / batching / image / audio support | `SupportedBackends`, `SupportsContinuousBatching` (+ `CanBatchPredicate`), `SupportsImageInput`, `SupportsAudioInput` |
| A new compute path | a forward pass plus `CreateForwardPass = ctx => ...` |

## What still lives outside the descriptor (separate registries, by design)
Tokenizer family switches (`GgufTokenizer.cs`, `PreTokenizerPatterns.cs`), tool-call adapters (`ToolCallAdapterRegistry.Register`), the measured prefill hand-off table (`PrefillHandoffFamilies.cs`, planner-owned evidence), and vision projector dispatch (`UnifiedVisionPipeline.cs`, keyed by the mmproj projector type, not the architecture). Edit these only when the family needs them.

## Guards you will meet
- `ArchitectureLiterals_OutsideTheArchitectureFolder_OnlyShrink`: a quoted architecture id may not appear in a file outside `Engine/Architectures/` more often than the checked-in baseline. Put the knowledge on the descriptor instead. Regenerate only after *removing* literals.
- `PlannerAndSelection_ReadOnlyCapabilities_FromTheDescriptor`: planning code may read `descriptor.Capabilities`, `Id`, `BackendLimitation` and nothing else.
- `ArchitectureFacts_MatchBaseline`: 71 lines of facts (admission, family, backends, batching, chat protocol, tool adapter); a change must be intended and explained in the commit.
- `DescriptorTraits_MatchTheByNameTable`: for architectures that appear in `ModelArchitectureTraits.Legacy` (a shim for direct callers of the baseline parser), the descriptor must agree. A new architecture need not appear there.
