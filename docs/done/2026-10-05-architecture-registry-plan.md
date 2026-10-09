# Architecture descriptor registry: plan and migration (2026-10-05)

> **Historical.** The descriptor/registry/factory migration here is complete. Open follow-up work (model semantics owned by the descriptor): [../2-coverage/2026-10-08-architecture-semantics-admission-plan.md](../2-coverage/2026-10-08-architecture-semantics-admission-plan.md).

Source of the idea: TensorSharp's `Architecture/ModelArchitectureDescriptor.cs` + `BuiltInArchitectures.cs`
(BSD-3; see THIRD_PARTY_NOTICES.md). Code: `src/OpenTail.Stingray.Engine/Architectures/`.

## Problem

`ModelCompatibility` held a ~800-line `HashSet<string>` with admission evidence in comments, and about
25 `arch == "..."` checks in `Cli`, `Server` and `Engine` decided behaviour (default thinking, fallback
chat template, forward-pass choice). Adding a family touched all of them, and the CLI and server could drift.

## Design

One `ArchitectureDescriptor` per family (`Id`, `Aliases`, `Status` = Admitted / NotAdmitted / Experimental,
`EvidenceDoc`, `RefusalReason`, `ExperimentalEnvVar`, `ThinkingDefaultOff`, `FallbackChat`), listed in the
explicit `BuiltInArchitectures` manifest (no reflection, NativeAOT-safe). `Validate()` runs at registry build.
**A descriptor wins over the legacy allowlist**; unmigrated architectures keep working unchanged.

## Migration does not need checkpoints

Migration is behaviour-preserving: it moves where an existing decision is recorded. An admitted family stays
admitted on its existing receipt; a refused family stays refused. It is verified by unit tests asserting the
status and behaviour are unchanged (`ArchitectureRegistryTests`), not by running models. Checkpoints matter
only for *promotion* (NotAdmitted -> Admitted), which stays on the normal path of CLAUDE.md rule 14 and is
not part of this work.

## Steps

- [x] 1. Registry + `gemma4`, `granite`, `llama`, `llama4`; CLI and server read thinking default and fallback chat format from it.
- [x] 2. The seven "NOT admitted" blocks (`glm-dsa`, `glm5next`, `diffusion-gemma`, `qwen4exp`, `deepseek41`, `deepseek4`, `deepseek32`) become `NotAdmitted` descriptors; `muse-glimmer` (+ `muse_glimmer` alias) becomes `Admitted`. The refusal message now quotes the descriptor's reason and record.
- [x] 3. Remove the now-duplicate legacy entries for `gemma4`, `granite`, `llama`, `llama4` (their evidence comments move into descriptor files).
- [x] 4. Move the remaining admitted allowlist entries into descriptors, grouped by trunk family (llama-like, qwen, gemma, phi, rwkv, deepseek2, ...), carrying each evidence comment into the file. Mechanical; no behaviour change. Reconcile the stale `minicpm` "NOT admitted" note that was removed in step 2: the allowlist entry says admitted 2026-09-01.
- [x] 5. Add the `Experimental` refusal path and quant-gate registry. PRISM remains a quant-type gate, not an architecture, declared through `ExperimentalQuantGate`; no family was promoted to `Experimental`.
- [x] 6. Forward-pass factory field + `SupportedBackends`, replacing the hand-written selection in `InferenceEngineLoader` / `RunCommand` (`gpt-oss`, `rwkv*`, hybrid GDN, ...).
  - **What stays in the frontends**:
    - Hardware placement: layer counts, `TierPlanner.Plan` (pricing VRAM and layer splits), CUDA/Vulkan backend resolution, device selection.
    - Object construction: instantiating the chosen forward-pass classes (`ForwardPass`, `CudaForwardPass`, `CudaHybridForwardPass`, `GpuForwardPass`, `DeepSeek2GpuForwardPass`, `GptOssForwardPass`, `GptOssGpuForwardPass`, `VulkanHybridGdnForwardPass`, `RwkvForwardPassBase`), wiring delegates (`Forward`, `Prefill`, `ResetCache`), and handling backend/pass disposal.
    - AnsiConsole output: terminal markup, progress displays, warnings (e.g. greedy reasoning warnings, fallback notes).
  - **Audited CLI checks and status**:
    - SafeTensors package gates (`--ngl`, `--tq`, `--draft-model`, `--draft-lookup`, `--dspark-model`): Delegated to `ForwardPassSelection.Select(IsSafeTensors: true)`, with CLI formatting the returned refusal into rich terminal markup.
    - SafeTensors image input refusal: Left in CLI frontend because SafeTensors packages do not integrate multimodal vision projector dispatch.
    - Mutual exclusivity of `--draft-model` and `--draft-lookup`: Left in CLI frontend speculative decoder argument parsing before model loading (also mirrored in selector).
    - Missing draft model file path: Left in CLI frontend filesystem validation prior to tensor loading.
    - DSpark CLI options validation (`--spec-type dspark`, `--dspark-model`, `--spec-type mtp`, confidence range): Left in CLI frontend option parsing as early CLI failure points.
    - GGUF architecture admission check (`ModelCompatibility.ValidateForTextGeneration`): Uses registry descriptors with `--allow-unverified-arch` override handled in frontend.
    - Hybrid GDN + TurboQuant / Speculative decoding: Delegated to `ForwardPassSelection.Select` (returns canonical refusal string).
    - DeepSeek2 MLA Vulkan selection: Delegated to `ForwardPassSelection.Select` (returns `DeepSeek2Vulkan`); object instantiation and backend setup stay in frontend.
    - Generic unsupported GPU feature fallback: Delegated to `ForwardPassSelection.Select` (`UnsupportedGpuPath`, `UnsupportedPartialCudaPath`); note and layer clamp stay in frontend.
    - RWKV recurrent refusal (TQ / drafts): Delegated to `ForwardPassSelection.Select`; CPU pass creation and GPU notice stay in frontend.
    - gpt-oss refusals and Vulkan/CPU selection: Delegated to `ForwardPassSelection.Select` (`GptOssVulkan` / `GptOssCpu`); object instantiation and GPU notice stay in frontend.
    - TurboQuant mode string parsing: Delegated to `ForwardPassSelection.Select(ValidateTurboQuantModeOnly: true)`.
    - TurboQuant KVarN preconditions (SnapKV, CUDA, MoE, partial offload): Delegated to `ForwardPassSelection.Select(ValidateKVarNOnly: true)`.
    - TurboQuant head dimension validation: Delegated to `ForwardPassSelection.Select(ValidateTurboQuantHeadDimOnly: true)`.
    - CUDA hybrid post-TierPlanner KVarN check: Left in CLI frontend because `cudaGpuLayers` is calculated dynamically by `TierPlanner` during backend configuration.
    - Dead checks: None (all dead string checks were removed during S5 migration).
- [x] 7. Contract test enforces explicit STATUS anchors or exemptions for every `Admitted` descriptor, every `NotAdmitted` descriptor's internal-table entry and public-doc exclusion, and evidence-doc existence. Removed two stale public DeepSeek alpha rows; no architecture status changed.
- [ ] 8. Optional `DetectFromTensors` for GGUFs with missing or relabelled architecture metadata; `admit-arch` emits a descriptor stub.

## Out of scope

Qwen3's default-system-prompt rule (Jinja path injects only for `qwen3`, fallback also for `qwen3moe`) needs a decision on which behaviour is intended before it can become a flag.
