# Architecture descriptor registry: plan and migration (2026-10-05)

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
- [ ] 4. Move the remaining admitted allowlist entries into descriptors, grouped by trunk family (llama-like, qwen, gemma, phi, rwkv, deepseek2, ...), carrying each evidence comment into the file. Mechanical; no behaviour change. Reconcile the stale `minicpm` "NOT admitted" note that was removed in step 2: the allowlist entry says admitted 2026-09-01.
- [ ] 5. `Experimental` for gated families (PRISM is a quant-type gate in `ValidateForTextGeneration`, not an architecture, so it stays there unless it gets its own flag).
- [ ] 6. Forward-pass factory field + `SupportedBackends`, replacing the hand-written selection in `InferenceEngineLoader` / `RunCommand` (`gpt-oss`, `rwkv*`, hybrid GDN, ...).
- [ ] 7. Contract test: every `Admitted` descriptor has a `docs/STATUS.md` row; every `NotAdmitted` one appears in the "Ported, not verified" table.
- [ ] 8. Optional `DetectFromTensors` for GGUFs with missing or relabelled architecture metadata; `admit-arch` emits a descriptor stub.

## Out of scope

Qwen3's default-system-prompt rule (Jinja path injects only for `qwen3`, fallback also for `qwen3moe`) needs a decision on which behaviour is intended before it can become a flag.
