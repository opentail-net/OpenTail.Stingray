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
   you change): rwkv6/rwkv7 (CPU only), gpt-oss (CPU, Vulkan full-offload only), deepseek2 only (MLA: Vulkan
   full offload, no CUDA; `deepseek2-ocr` stays `Dense` / `SupportedBackends.All`, asserted in `ArchitectureRegistryTests`,
   and the MLA pass is chosen by shape, i.e. `KvLoraRank` + tensor presence, not by architecture name), muse-glimmer (CPU only; unsupported GPU requests fall back to CPU, as the existing loader does). Leave every other descriptor at defaults. If a
   constraint is unclear, leave the default and list it in your report; never guess.
3. Make `ForwardPassSelection.Select` consult the descriptor (via `ArchitectureRegistry.Find`) for the arch-driven rows
   instead of string comparisons, keeping every S2 test green with unchanged expectations.
4. Add registry tests: each limited descriptor has a non-empty `BackendLimitation`; `Select` preserves the existing CPU fallback for unsupported GPU requests; `AdmittedSet_IsExactlyTheSnapshot` still passes.

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

Goal: `RunCommand.cs` uses the same selector as the server, with the selector's rules no longer duplicated in the CLI;
docs and plan are updated. This is the riskiest section: behaviour must not change, and "identical output" is only
proven where a real baseline exists, so build the baselines first.

Note on size: `ForwardPassSelection.Select` is ~470 lines and already covers more than rwkv / gpt-oss / DeepSeek2 / hybrid:
SafeTensors refusals, `--tq-mode` parsing, TurboQuant head-dim and KVarN rules, and the hybrid-GDN guards, all with a
`ForwardPassFrontend.Cli` branch. The CLI still has its own copies of these (e.g. TQ mode parsing ~2087-2101, head-dim
checks ~2160-2185, SafeTensors gates ~815-840, hybrid guards ~1974-1983, rwkv/gpt-oss ~2016-2055). Use line numbers from
the S1 matrix doc, but re-check them: they drift.

### Part A: baselines for the paths the S1 baselines did not cover (before touching production code)

1. In `models/_models/` there are `gpt-oss-20b-MXFP4.gguf` and `DeepSeek-V2-Lite-Chat.Q2_K.gguf` (confirm with `ls`; check
   RAM first and run **one heavy job at a time**: do not run these in parallel with a build or test run). If an RWKV
   GGUF is present use it too; if not, say so, and do **not** download one in this section (the selector tests are the only
   cover for rwkv; record that explicitly).
2. For each available model, capture a CPU baseline with the **unchanged** CLI: greedy, fixed prompt, small `-n` (e.g. 16),
   `--seed 1 --verbose-prompt --no-display-prompt`, `-g 0`. Where the iGPU can run it, also capture a full-Vulkan baseline
   (`-g -1 --backend vulkan`). Record exact commands, prompt ids, generated text and token ids in the matrix doc's Baselines
   section, in the same format as the existing SmolLM2 entries. Note which selection path each one exercises (gpt-oss CPU /
   gpt-oss Vulkan, DeepSeek2 MLA Vulkan vs generic).
3. Also capture the **refusal texts** of the CLI today for: `--tq` on a hybrid-GDN model if one is in `models/` (else skip),
   `--tq-mode bogus`, `--tq-mode kvarn` without `--tq`, `--tq` with `--draft-lookup` on gpt-oss. These are exit-code-1
   runs and cheap. Record the exact stderr line for each.
4. Server baseline for S4 (it was never recorded): start the server host on SmolLM2-135M, send one chat completion at
   temperature 0 with the same prompt, record the text and compare with the CLI baseline. If the server cannot be run here,
   say so.
Commit the doc changes (docs only) before Part B.

### Part B: switch the CLI over

5. In `RunCommand.cs`, build a `ForwardPassRequest` (`Frontend = Cli`) and call `ForwardPassSelection.Select` for the
   decisions the selector already owns. Express the CLI-only draft-model/`--draft-lookup` guard as request flags
   (`HasDraftModel`, `DraftLookup`) so it behaves exactly as today. Construction, hardware placement (layer counts,
   TierPlanner, CUDA/Vulkan resolution) and `AnsiConsole` output stay in the CLI.
6. **Move, don't duplicate:** for every rule the selector now owns (SafeTensors gates, TQ mode parsing, TQ head-dim /
   KVarN rules, hybrid-GDN guards, rwkv / gpt-oss refusals), the CLI must consume the selector's result and delete its own
   copy. The selector's refusal text must be byte-identical to what the CLI printed (use the strings recorded in step 3;
   the CLI wraps them in `[red]Error:[/]` and `Markup.Escape`, so check markup characters such as `[[8, 1024]]`).
7. For any CLI check that the selector does **not** cover and that you therefore leave in place, list it in your report
   with a one-line reason. Do not silently keep both versions of a rule, and do not add rules to the selector that the CLI
   did not have.
8. If a selector rule disagrees with what the CLI did (different order of refusals, different text, different condition),
   **stop and fix the selector test + selector to match the CLI's current behaviour** (the baseline is today's CLI),
   never the other way round. Add a selector test row for each such case. Also note the CLI-vs-selector ordering: the CLI
   validates some things (TQ mode) after model load; do not change which message wins when two conditions hold.
9. Remove arch-name checks left in either frontend that are now dead, and delete nothing else.

### Part C: verify and document

10. Rebuild the CLI; re-run **every** baseline from S1 and Part A (CPU and Vulkan) and the refusal-text runs. Token ids,
    texts and stderr lines must be identical. Record "identical, run on <date>" next to each in the matrix doc. Any
    difference: stop, do not "explain it away", report it.
11. Update `docs/2-coverage/2026-10-05-architecture-registry-plan.md`: tick step 6, and note what stayed in the frontends
    (hardware placement, plus any item from step 7). Update the matrix doc's divergence list with anything now unified, and
    say which paths are covered only by selector tests (e.g. rwkv if no checkpoint).

Verify: CLI build with 0 warnings; run `OpenTail.Stingray.Tests.Cli` (all classes), `ForwardPassSelectionTests`,
`ArchitectureRegistryTests`, `ArchitectureDocsContractTests`, `Tests.Server.Fast`, using fully namespace-qualified class
names and checking `Total` is not 0; plus `Tests.ForwardPass.Fast` if it completes in a few minutes (report its duration;
`STINGRAY_RUN_HEAVY_TESTS` stays unset). Timings from this iGPU-only machine are not evidence about GPU speed (rule 13);
only output identity is being checked. Commit Part B and C separately from Part A.

Report must include: the baselines table (before/after identical yes/no), the list of CLI checks deliberately left in place,
the refusal strings compared, and which families have no real-weight baseline.
