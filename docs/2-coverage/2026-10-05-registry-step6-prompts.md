# Prompts: registry step 6 (forward-pass selection), split into 5 small sections

Plan: `docs/2-coverage/2026-10-05-architecture-registry-plan.md` (step 6). Give an agent the **Shared preamble** plus
**one** section at a time, in order; each section ends in its own commit and leaves the tree green. Do not give an agent
several sections at once.

## Why it is split

Forward-pass selection is duplicated and partly diverging between `Cli/RunCommand.cs` (~lines 1400-2075) and
`Server/InferenceEngineLoader.cs` (~lines 560-820). It mixes three kinds of decision: **architecture** (rwkv, gpt-oss,
DeepSeek2 MLA), **model shape** (`hp.IsHybridSsm`, `hp.IsMoE`, `hp.KvLoraRank`) and **hardware/flags** (backend, `-g N`,
TurboQuant, draft models, CUDA/Vulkan placement). Only the first kind belongs in descriptors. The order is: record what
exists (S1), put the decision in a pure, testable function (S2), declare per-family facts in descriptors (S3), then switch the
two frontends over one at a time (S4, S5). Sections S1-S3 change no runtime behaviour at all.

Known divergence to preserve, not fix: for DeepSeek2 MLA the CLI additionally refuses the GPU path when a draft model or
`--draft-lookup` is set; the server does not have those flags. Record any other divergence in S1.

---

## Shared preamble (give with every section)

You are working in `C:\Git-Public\OpenTail.Stingray` (C# 14 / .NET 10, Windows, PowerShell/Git Bash). Read `CLAUDE.md` first and
obey it: `TreatWarningsAsErrors`, never pass `--nologo` to `dotnet test`, no scratch files in the repo root (use the OS temp
scratchpad), no subagents. The registry is in `src/OpenTail.Stingray.Engine/Architectures/`
(`ArchitectureDescriptor`, `ArchitectureRegistry`, `BuiltInArchitectures`, ~60 admitted + 7 NotAdmitted descriptors).

Rules for every section:
- **Behaviour-preserving.** Same forward pass chosen for the same inputs, same refusals, same messages. Do not "fix"
  anything you notice; list it in your report.
- Use the Edit tool for source edits. Do not generate C# through Python strings with `\n` escapes. Files are LF in the
  working copy.
- Run only the targeted test classes below (the full `Tests.Core` takes ~3 minutes and has **3 known unrelated failures** in
  `KnownEnvironmentVariablesTests`; do not fix or count them). Invoke built test `.exe`s with fully namespace-qualified class
  names and confirm `Total` is not 0.
- Commit only the files you changed (the tree has unrelated modified files: `.gitignore`, two docs under
  `docs/1-correctness`, `docs/2-coverage/ported-families-todo.md`, `stage_diagnostics_report.txt`, `tools/engine-compare/`).
  End the commit message with the attribution line the session requires, if any.
- Finish with a short report: what you did, anything surprising, anything you chose not to do.

---

## Section 1 of 5: Characterize (docs and baselines only; no production code changes)

Goal: a written, line-referenced map of today's selection logic, plus frozen output baselines.

Tasks:
1. Create `docs/2-coverage/2026-10-05-forward-pass-selection-matrix.md`. For **both** `RunCommand.cs` and
   `InferenceEngineLoader.cs`, list every point that chooses a forward-pass class or refuses a combination (file:line,
   condition, class chosen or refusal text). Include the draft-model and TurboQuant guards.
2. Classify each condition as **arch-driven**, **shape-driven** (`hp.*` or tensor presence) or **hardware/flag-driven**.
3. Add a "CLI vs server divergences" section: every place the two disagree for the same inputs (known one above; find others).
4. Capture **baselines** for a model already in `models/` (find a small GGUF, e.g. SmolLM2-135M-Instruct; run
   `ls models` and look inside its folder), greedy (`--temp 0`), a fixed prompt, 24 tokens, on CPU (`-g 0`) and, if the iGPU
   Vulkan path works here, `-g -1`. Record the exact command, the generated text and token ids, in a "Baselines" section of the
   doc. If a gpt-oss or rwkv GGUF is present locally, add it too; if not, say so. (This dev machine has no discrete GPU, so
   timing is not part of this; only identity of output.)
5. Do **not** edit any `.cs` file.

Verify: the doc exists, every row has a file:line that you re-read, and the baselines were produced by a build of the
current HEAD (`dotnet build src/OpenTail.Stingray.Cli -c Release` first). Commit the doc only.

---

## Section 2 of 5: Pure selector (new code and tests only; frontends untouched)

Goal: the decision as a pure function, tested against the S1 matrix. Nothing calls it yet.

Tasks:
1. In `src/OpenTail.Stingray.Engine/Architectures/`, add `ForwardPassSelection.cs` with:
   - `enum ForwardPassKind` (one value per distinct forward-pass class in the S1 matrix, e.g. `CpuDense`, `CpuHybridGdn`,
     `Rwkv`, `GptOssCpu`, `GptOssVulkan`, `DeepSeek2Vulkan`, `CudaDense`, `CudaHybridGdn`, `CudaHybrid`, `VulkanDense`,
     `VulkanHybridGdn`, `VulkanHybrid`, `VulkanLayerSplit`, ... exactly what the matrix shows);
   - `record ForwardPassRequest` with only plain data (architecture id, `IsHybridSsm`, `IsMoE`, `KvLoraRank`, "has MLA
     tensors", `NumLayers`, requested backend, `nGpuLayers`, TurboQuant, has-draft, ...; no `GgufModel`, no backends);
   - `static ForwardPassSelection.Select(ForwardPassRequest) -> ForwardPassDecision` (a kind, or a refusal with the exact
     text the frontends use today).
2. Encode the S1 matrix as a `[Theory]` in `tests/OpenTail.Stingray.Tests.Core/ForwardPassSelectionTests.cs`: one case per
   matrix row, including the documented CLI-vs-server divergences as an explicit input flag (e.g. `Frontend`), so both
   behaviours stay representable.
3. Do **not** touch `RunCommand.cs` or `InferenceEngineLoader.cs`. Hardware-placement details (which layers go to which GPU,
   MoE auto-hybrid sizing) stay in the frontends; the selector only returns the *kind* and may take a "needs hybrid
   placement" boolean as input.

Verify: build `Tests.Core`; run `OpenTail.Stingray.Tests.Core.ForwardPassSelectionTests` and `ArchitectureRegistryTests`
(both pass, nonzero totals). Commit.

---

## Section 3 of 5: Declare per-family facts in descriptors

Goal: arch-driven selection and refusals come from descriptors, not name checks.

Tasks:
1. Add to `ArchitectureDescriptor`: `ForwardPassFamily` (enum: `Dense`, `HybridGdn`, `Rwkv`, `GptOss`, `DeepSeek2Mla`,
   plus any other the S1 matrix needs; default `Dense`) and `SupportedBackends` (flags: Cpu, Vulkan, Cuda; default all) with
   a `BackendLimitation` string (required when not all backends are supported, like TensorSharp's `MultiGpuLimitation`).
   Extend `Validate()` accordingly.
2. Populate them **only from constraints the code already enforces** (cite the file:line in a comment on each descriptor
   you change): rwkv6/rwkv7 (CPU only), gpt-oss (CPU, Vulkan full-offload only), deepseek2 and deepseek2-ocr (MLA: Vulkan
   full offload, no CUDA), muse-glimmer (CPU only; GPU passes refuse it). Leave every other descriptor at defaults. If a
   constraint is unclear, leave the default and list it in your report; never guess.
3. Make `ForwardPassSelection.Select` consult the descriptor (via `ArchitectureRegistry.Find`) for the arch-driven rows
   instead of string comparisons, keeping every S2 test green with unchanged expectations.
4. Add registry tests: each limited descriptor has a non-empty `BackendLimitation`; `Select` refuses an unsupported backend
   with the descriptor's limitation text; `AdmittedSet_IsExactlyTheSnapshot` still passes.

Verify: run `ForwardPassSelectionTests`, `ArchitectureRegistryTests`, `ArchitectureDocsContractTests`; build the CLI and
`Tests.Server.Fast` with 0 warnings. Commit.

---

## Section 4 of 5: Switch the server loader over

Goal: `InferenceEngineLoader.cs` asks the selector for arch-driven and shape-driven decisions.

Tasks:
1. In `src/OpenTail.Stingray.Server/InferenceEngineLoader.cs`, replace the `RwkvForwardPassBase.IsRwkv(arch)`,
   `arch == "gpt-oss"` and DeepSeek2-MLA conditions with a `ForwardPassSelection.Select` call (build the
   `ForwardPassRequest` from the loaded model, `hp` and the loader's parameters; `Frontend = Server`).
2. Keep object construction, disposal, backend creation and layer placement exactly where they are; only the "which kind?"
   test changes. Keep the exact refusal exceptions and messages.
3. Do not touch `RunCommand.cs`.
4. Re-run the S1 baseline through the **server path** if it can be exercised with a small model (the server host:
   `STINGRAY_MODEL=<gguf> dotnet run --project src/OpenTail.Stingray.Server.Host -c Release`, one deterministic
   completion request, `temperature 0`); output must match the S1 baseline text. If the server can't be run here, say so and
   rely on tests.

Verify: build `Server` and `Tests.Server.Fast`; run `OpenTail.Stingray.Tests.Server.Fast.exe` (all pass), plus
`ForwardPassSelectionTests`. Commit.

---

## Section 5 of 5: Switch the CLI over, then clean up

Goal: `RunCommand.cs` uses the same selector; the duplicate name checks are gone; docs and plan are updated.

Tasks:
1. In `src/OpenTail.Stingray.Cli/RunCommand.cs`, replace the rwkv / gpt-oss / DeepSeek2-MLA and hybrid-vs-dense kind
   decisions with `ForwardPassSelection.Select` (`Frontend = Cli`), keeping the CLI-only draft-model/`--draft-lookup` guard
   expressed as a request flag so it behaves exactly as today. Construction, placement and messages stay put.
2. Remove any now-unused arch-name checks left in either frontend, and delete nothing else.
3. Re-run the **S1 baselines** with the CLI (CPU, and Vulkan if recorded); texts and token ids must be identical.
   Record "identical, run on <date>" in the matrix doc's Baselines section.
4. Update `docs/2-coverage/2026-10-05-architecture-registry-plan.md`: tick step 6 and note what stayed in the frontends
   (hardware placement). Update the matrix doc's divergence list with anything now unified.

Verify: build the CLI with 0 warnings; run `OpenTail.Stingray.Tests.Cli` (all classes), `ForwardPassSelectionTests`,
`ArchitectureRegistryTests`, `Tests.Server.Fast`; plus `Tests.ForwardPass.Fast` if it completes in a few minutes (report its
duration; `STINGRAY_RUN_HEAVY_TESTS` stays unset). Commit.
