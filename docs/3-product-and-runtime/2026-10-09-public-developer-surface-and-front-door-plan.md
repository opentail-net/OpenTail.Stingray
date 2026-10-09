# OpenTail.Stingray: Public Developer Surface and Front Door — Implementation Plan

## 1. Objective

Make Stingray substantially easier for a .NET developer to discover, install, embed, and use without learning the internal engine architecture or manually assembling execution pipelines.

The intended outcomes are:

1. A small, stable, documented public C# API for loading a model and running multi-turn chat.
2. Convenient path-free CLI entry points for chat, text-to-speech, and transcription, backed by the existing `ModelCatalog` and `ModelHome`.
3. One coherent first-run workflow: install Stingray, install a recommended model, run a task, and then use the same runtime from C#.
4. A NuGet README containing examples that demonstrably compile against the package being published.
5. Automated protection against documentation drift, broken package contents, misleading task availability, and lifecycle/API regressions.

This is a developer-experience and public-contract task. It is not a new inference-engine implementation.

## 2. Repository findings: existing state

Start from the current checkout and treat the source as authoritative. The code below already exists and should be reused rather than recreated.

### 2.1 Public model/context/executor/session API

Existing files:

* `src/OpenTail.Stingray.Engine/Api/Model.cs`
* `src/OpenTail.Stingray.Engine/Api/ModelContext.cs`
* `src/OpenTail.Stingray.Engine/Api/IModelParams.cs`
* `src/OpenTail.Stingray.Engine/Api/IContextParams.cs`
* `src/OpenTail.Stingray.Engine/Api/IInferenceParams.cs`
* `src/OpenTail.Stingray.Engine/Api/IModel.cs`
* `src/OpenTail.Stingray.Engine/Api/IExecutor.cs`
* `src/OpenTail.Stingray.Engine/Api/InteractiveExecutor.cs`
* `src/OpenTail.Stingray.Engine/Api/StatelessExecutor.cs`
* `src/OpenTail.Stingray.Engine/Api/BatchedExecutor.cs`
* `src/OpenTail.Stingray.Engine/Api/ChatHistory.cs`
* `src/OpenTail.Stingray.Engine/Api/ChatSession.cs`
* `src/OpenTail.Stingray.Engine/GenerateChunk.cs`

The intended public usage pattern is already supported structurally:

`Model.Load(...)` → `Model.CreateContext(...)` → `InteractiveExecutor` → `ChatSession`.

The relevant namespaces are `OpenTail.Stingray`, `OpenTail.Stingray.Executors`, and `OpenTail.Stingray.Engine`.

`Model.CreateContext` returns `IModelContext`, and `InteractiveExecutor` has an overload accepting that interface. A new convenience context factory is not needed merely to make the proposed usage compile.

`InferenceParams` already exposes sampling, stop conditions, reasoning controls, speculative-decoding settings, and structured-output constraints. `GenerateChunk` already represents `Text`, `Thinking`, `Usage`, `Stop`, and `ToolCall` output.

There is an existing test suite at `tests/OpenTail.Stingray.Tests.Core/PublicApiContractsTests.cs`. Extend it where necessary. Do not create a second, competing facade over the same engine.

### 2.2 Model catalogue and installation

Existing files:

* `src/OpenTail.Stingray.Core/Catalog/ModelCatalog.cs`
* `src/OpenTail.Stingray.Core/Catalog/ModelHome.cs`
* `src/OpenTail.Stingray.Core/Catalog/ModelDownloader.cs`
* `src/OpenTail.Stingray.Cli/SetupCommand.cs`
* `src/OpenTail.Stingray.Cli/ModelsCommand.cs`
* `tests/OpenTail.Stingray.Tests.Core/ModelCatalogTests.cs`

The catalogue currently has three entries: `qwen2.5-0.5b` for chat, `piper-lessac` for speech synthesis, and `whisper-base` for transcription.

The existing design deliberately uses pinned Hugging Face revisions, expected SHA-256 values, file sizes, licensing notes, hardware notes, and evidence references. Preserve those integrity properties.

`stingray setup` and `stingray models` already exist.

### 2.3 Existing task commands

The existing command implementations are:

* `src/OpenTail.Stingray.Cli/RunCommand.cs`
* `src/OpenTail.Stingray.Cli/TtsCommand.cs`
* `src/OpenTail.Stingray.Cli/SttCommand.cs`
* `src/OpenTail.Stingray.Cli/Program.cs`

The command app currently registers the existing commands as the default/path-oriented run command, `tts`, and `stt`. It also registers `setup` and `models`.

It does not yet provide the intended simple `chat`, `speak`, and `transcribe` user-facing command set.

The correct approach is to reuse the existing inference and speech implementations, not create alternate implementations of token generation, TTS, or ASR.

### 2.4 NuGet packaging and documentation

The packable library project is `src/OpenTail.Stingray/OpenTail.Stingray.csproj`. It bundles the Stingray assemblies into the `OpenTail.Stingray` package and explicitly includes `src/OpenTail.Stingray/README.md` as its NuGet README.

**That package README is currently stale.** Its chat example constructs an `InferenceEngine` with a constructor that does not exist and calls `StreamChatAsync`, which is not the current API. It also pins `--version 1.0.6` while the source is at 1.0.7 (see `103-front-door-design.md`). This is a concrete defect, not a cosmetic documentation issue.

The repository-root `README.md`, the package README, the CLI help and the front-door design document must agree about how a new user gets started.

Do not take the checkboxes in the historical LLamaSharp realignment document as proof that every example or lifecycle contract works end to end.

## 3. Design decisions to preserve

### 3.1 Stingray remains a general AI runtime

Use neutral public names:

* `Model`
* `ModelContext`
* `ModelParams`
* `ContextParams`
* `InferenceParams`
* `IExecutor`
* `InteractiveExecutor`
* `StatelessExecutor`
* `BatchedExecutor`
* `ChatHistory`
* `ChatSession`

Do not add an LLamaSharp binary dependency, rename generic public concepts to include `LLama`, or make the public API a facade over llama.cpp.

The goal is application-level familiarity for LLamaSharp users, not binary compatibility with LLamaSharp.

### 3.2 One canonical C# chat surface

Use the existing model/context/executor/session chain as the canonical example. Do not add a separate `StingrayChat` abstraction merely to shorten the README, unless a later, independently justified design requires one.

### 3.3 Model catalog and inference API are different layers

`ModelCatalog` answers which ready-to-use bundle is recommended and where its files belong.

`Model` and `ModelContext` load and execute a chosen model.

A catalog ID is not a file path; a model file path is not a catalogue entry. Keep these concepts separate in the implementation and public examples.

### 3.4 Do not force every AI domain through the LLM facade

The generic executor/session API is primarily the text-generation surface. Speech, vision, image generation and other domains keep their existing domain-specific public pipeline APIs.

The CLI may offer one consistent front door for these tasks, but it must not pretend that TTS or diffusion is just an LLM `ChatSession`.

### 3.5 Preserve path-oriented and advanced CLI use

The new task commands must supplement, not break, the existing root command, `tts`, `stt`, `pull`, `setup`, `models`, `image`, `embed`, `rerank`, and the specialist commands.

Do not remove advanced model-path and engine-selection options from the existing commands just to make the new commands simpler.

## 4. Phase A — Stabilise the public C# contract

### A1. Prove the public API with a compiling consumer

Create a small sample project that consumes the public API using normal `using` directives and no internal engine scaffolding.

The existing `samples/OpenTail.Stingray.Sample.Chat` is **not** a suitable starting point as-is: it hand-assembles `GgufModel`, `GgufTokenizer`, `CpuBackend`, `ForwardPass` and `InferenceEngine`, which is exactly the scaffolding this plan removes. Either rewrite it onto `Model.Load` → `ChatSession` (preferred, so there is one chat sample) or add a new sample and retire/relabel the old one as an advanced engine-level example. Do not leave two samples both presented as "the chat sample".

The essential usage should be equivalent to:

```csharp
using OpenTail.Stingray;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Executors;

using var model = Model.Load(
    new ModelParams(modelPath)
    {
        Backend = "cpu",
        GpuLayerCount = 0
    });

using var context = model.CreateContext(
    new ContextParams
    {
        ContextSize = 2048
    });

var executor = new InteractiveExecutor(context);
var session = new ChatSession(executor);

session.AddSystemMessage(
    "You are a helpful, concise assistant.");

await foreach (var chunk in session.ChatChunksAsync(
    "Explain quantisation in one paragraph.",
    new InferenceParams
    {
        MaxTokens = 128,
        Temperature = 0.7f,
        EnableThinking = false
    }))
{
    switch (chunk.Kind)
    {
        case GenerateChunkKind.Text:
            Console.Write(chunk.Text);
            break;

        case GenerateChunkKind.Thinking:
            // Optional: handle reasoning separately.
            break;
    }
}

Console.WriteLine();
```

This is a target shape rather than permission to guess at signatures. Compile it against the actual current types and adjust only where the source proves necessary.

The sample must demonstrate the real public contract. It must not use internal constructors, test helpers, manually assembled `ForwardPass` instances, CLI-private helpers, or repository-specific global usings.

### A2. Audit ownership, disposal, and lock-order protocol

Document and test the ownership rules before presenting the API as stable.

Verify the following:

* Disposing a `ModelContext` releases the context-owned engine, cache and allocations.
* Disposing a `Model` disposes its live child contexts and then releases its model resources, according to the current implementation.
* Disposing a context and then its parent model is safe.
* Creating and using an executor with a disposed context fails predictably.
* Exceptions during context construction do not leak partially created resources.
* Multiple contexts can be created from one loaded model and disposed independently.
* The model/context lifecycle behaves correctly if creation and disposal are attempted concurrently.

**Deadlock and race prevention:**
1. In `Model.CreateContext`, `_disposed` must be checked and registration confirmed under `_lock` to prevent creating a context over an already-disposing model.
2. In `Model.Dispose()` vs `ModelContext.Dispose()`, avoid lock-order inversion (`Model._lock` vs `ModelContext._lock`). `Model.Dispose()` must snapshot active contexts under `Model._lock`, mark `_disposed = true`, clear the context set, and **release `Model._lock` before disposing child contexts**. Child contexts can then safely call `_model.UnregisterContext(this)` without deadlocking against `Model.Dispose()`.
3. Test concurrent context creation, parent disposal, and child disposal with bounded completion checks (avoid infinite hangs).

The primary simple example should use ordinary `using` declarations in an order that disposes the context before its model. Do not make the public lifecycle depend on users knowing internal disposal details.

### A3. Audit misleading parameter options

`ModelParams` currently exposes several settings that the loader explicitly rejects when they request unsupported behaviour, including memory locking, disabling memory mapping, nonzero device selection and multi-device tensor split.

Audit each public option:

* If it is supported, test and document it.
* If it is intentionally unsupported, fail early with a clear message and document the limitation.
* If the option has not been released and serves no genuine public purpose, consider removing it before treating the API as stable.

Do not implement unrelated GPU features just to mimic LLamaSharp's parameter list.

For any change affecting public API compatibility, check the existing NuGet version and release policy first. Do not silently break already published consumers.

### A4. Interface conformance, incomplete-turn semantics, and behaviour tests

Extend `PublicApiContractsTests.cs` with missing lifecycle, interface conformance, and API-contract tests:

1. **Public Interface Conformance**:
   * In `InteractiveExecutor`, `StatelessExecutor`, and `BatchedExecutor`, ensure `IInferenceParams` implementations are properly honored (extracting sampling parameters via a general adapter) rather than using `(inferenceParams as InferenceParams)?.ToSamplingParams() ?? new SamplingParams()` which silently ignores custom implementations.
   * `InteractiveExecutor(IModelContext)` must accept any valid `IModelContext` implementation rather than casting to `ModelContext` and throwing.
2. **Incomplete-Turn Semantics in `ChatSession`**:
   * Explicitly choose, document, and test turn failure/cancellation semantics: if prompt formatting, executor invocation, or generation throws an exception or is cancelled before the assistant response completes, roll back the pending user message from `ChatHistory` so history remains turn-paired and cleanly retryable.
3. **Contract and Behaviour Tests**:
   * Constructor/factory validity and parameter defaults.
   * Model/context lifecycle, disposal order, multiple contexts, and exceptions.
   * Null and disposed-object behaviour.
   * `ChatHistory` operations and turn rollback on cancellation/exception.
   * Reasoning, text and usage chunk handling where relevant.
   * Custom `IInferenceParams` and `IModelContext` conformance test.
   * Both string streaming and typed chunk streaming.

Do not expand this phase into new inference math or new architectures.

**Phase A exit gate:** A clean consumer project compiles using the documented API, lock-order and interface conformance are verified, and focused public-contract tests pass.

## 5. Phase B — Implement path-free CLI task commands

Implement three commands that use the existing catalogue and model home:

```text
stingray chat
stingray speak "Hello from Stingray." --output hello.wav
stingray transcribe recording.wav
```

These examples illustrate the intended command shape. The implementation and generated help text must define the final option names consistently.

### B1. Shared catalogue resolution

Add a small CLI-level helper if needed, for example `CatalogTaskResolver`, to centralise task-model lookup and installation-state handling.

Its responsibilities are:

1. Resolve the requested task or catalogue ID using `ModelCatalog`.
2. Use `ModelHome.Default()` and the installed file location.
3. Distinguish a fully installed entry from a missing or partial installation, using `ModelHome.StateOf` (`InstallState.Missing | Partial | Installed`).

   > Note: `StateOf` checks file existence and **size only**; it does not re-hash. Integrity is guaranteed at install time (`ModelInstaller` renames from `.part` only after the SHA-256 check). There is therefore no "integrity-failure" state at task-invocation time. Decide explicitly: (a) accept size-only checks at run time and document that `stingray setup <task>` re-verifies (it already re-checks present files), or (b) add an opt-in verify (e.g. reuse the `hash` command's cached hash beside the file). Do not hash multi-hundred-MB files on every invocation. Adjust B6 and the definition of done to match the choice.
4. Return the resolved catalogue entry and local main-file path.
5. Produce an actionable failure message when the entry is unavailable.
6. Keep explicit model-file overrides separate from catalogue-ID resolution.

Reuse the existing installation rules. Do not independently invent a second model-home convention or bypass the catalogue's pinned identity.

**Do not download anything automatically during a task invocation.** Installing models is the explicit job of `stingray setup`. When a required model is missing, tell the user which command fixes it, for example:

```text
Model for chat is not installed.
Run: stingray setup chat
```

If the model is partially installed (including an interrupted `.part` download or a wrong-sized file), report that state clearly and point to `stingray setup <task>`, which resumes and re-verifies. Do not continue with an incomplete file.

### B2. `stingray chat`

The chat command should use the newly established public API directly:

`Model.Load` → `CreateContext` → `InteractiveExecutor` → `ChatSession`.

Avoid building a second bespoke generation loop on top of `RunCommand` just for this user-facing route.

Recommended behaviour:

* `stingray chat` starts a multi-turn interactive chat session.
* `stingray chat "Question"` may provide a convenient single-turn entry point; agree and document the exact behaviour.
* The default model is the `chat` entry in `ModelCatalog`, currently `qwen2.5-0.5b`.
* A `--model` option, if provided, means a catalogue ID. A separate `--model-file` override can preserve support for a user-supplied path.
* Support the basic options users genuinely need: backend, context size, generation budget, temperature and reasoning display/control.
* Keep the initial option set small; advanced inference switches continue to be available via the existing path-oriented command.
* Stream output rather than waiting for the entire response.
* Handle Ctrl+C cooperatively, release resources, and return meaningful exit codes.
* Ensure `Thinking` chunks are either intentionally displayed separately or excluded by a documented default. Never accidentally mix internal reasoning into the normal answer because a chunk kind was ignored.

Use the model's real chat template and existing `ChatSession` behaviour. Do not hardcode a ChatML prompt formatter in the CLI.

### B3. `stingray speak`

The default is the `speak` entry in `ModelCatalog`, currently `piper-lessac`.

Keep the entry simple:

```text
stingray speak "Hello from Stingray."
stingray speak "Hello from Stingray." --output hello.wav
```

Responsibilities:

* Resolve the installed `piper-lessac` catalogue bundle.
* Pass its ONNX model and companion configuration to the existing Piper/TTS implementation.
* Preserve the output-path option and clear success/failure reporting.
* **Licence consent policy:** Consent is explicitly gated at install time by `SetupCommand` (`LicenceNeedsConsent` / `--accept-licence` or interactive prompt). Once verified and installed in `ModelHome`, `stingray speak` operates on the installed voice assets without re-prompting on every invocation. Document this policy truthfully without claiming non-existent runtime token enforcement.
* Do not copy the TTS engine-selection logic or the inference implementation out of `TtsCommand`.

Reuse the existing implementation through an intentional shared execution method/service or a small catalogue-aware entry point. Do not simulate command composition with fragile console-output interception.

The existing `stingray tts` command and its advanced engines/options must continue working unchanged.

### B4. `stingray transcribe`

The default is the `transcribe` entry in `ModelCatalog`, currently `whisper-base`.

Expected form:

```text
stingray transcribe recording.wav
stingray transcribe recording.wav --output transcript.txt
```

The audio input path is, naturally, still required. “Path-free” means that users should not need to specify the model checkpoint path.

Responsibilities:

* Resolve the installed Whisper checkpoint from `ModelCatalog`.
* Reuse the existing STT pipeline and current transcription implementation.
* Preserve supported options such as language and output-file selection if they can be exposed without making the basic command cumbersome.
* Fail clearly when the input file or model is missing or invalid.
* Make audio sample-rate/channel requirements truthful. If the implementation only accepts 16 kHz WAV, state that in the help and README or reuse/add the existing resampling path if the intended first-run experience requires broader WAV support.
* Preserve the existing `stingray stt` options and all non-Whisper models supported through that command.

Do not implement another transcription pipeline.

### B5. Register commands and update catalogue run templates

1. **Update `ModelCatalog.RunTemplate`**:
   * Update the recommended run templates in `src/OpenTail.Stingray.Core/Catalog/ModelCatalog.cs`:
     * `qwen2.5-0.5b`: `stingray chat`
     * `piper-lessac`: `stingray speak "Hello from Stingray."`
     * `whisper-base`: `stingray transcribe <audio.wav>`
   * This ensures `stingray models` advertises the new front-door commands instead of the older path-oriented ones. Update `ModelCatalogTests.cs` accordingly.
2. **Register Commands**:
   * Register `chat`, `speak`, and `transcribe` in `src/OpenTail.Stingray.Cli/Program.cs`.
   * Do not replace or rename the default command or existing `tts` and `stt` commands.
3. **Application Name Consistency**:
   * Check if `config.SetApplicationName("stingray")` can be safely aligned and verify tests.
4. **Option Inventory**:
   * Regenerate the CLI option inventory through `scripts/gen-cli-option-inventory.ps1`.

### B6. CLI tests

Add focused tests for:

* Correct default catalogue entry for each task.
* Case-insensitive catalogue-ID resolution if supported.
* Installed, missing and partial (including wrong-size and leftover `.part`) model states; add a hash-mismatch case only if option (b) in B1 is adopted.
* Correct model path passed to the underlying task implementation.
* Missing model message includes the right `stingray setup <task>` suggestion.
* Missing input and invalid output paths fail cleanly.
* Unknown catalogue IDs fail clearly.
* No network requests occur simply because a task command is launched.
* Cancellation and error handling.
* Existing `tts`, `stt`, `setup`, `models` and default command behaviour remain intact.

Use local stubs or fakes for the fast tests. Real inference and real audio verification belong in explicit integration/smoke tests, not in fake tests that claim to prove numerical correctness.

**Phase B exit gate:** All three commands work through the catalogue and existing runtime implementations; missing or damaged assets produce actionable errors; legacy commands still work.

## 6. Phase C — Make the NuGet README correct and useful

Rewrite `src/OpenTail.Stingray/README.md` as the documentation a developer actually sees on the NuGet package page.

The current broken `InferenceEngine`/`StreamChatAsync` sample must disappear. Do not simply replace those two names while leaving the user to assemble the internal stack.

Recommended structure:

1. One-sentence explanation of what Stingray is and why a .NET developer might use it.
2. Installation command: `dotnet add package OpenTail.Stingray`.
3. A verified end-to-end C# chat example using the public API from Phase A.
4. How to obtain the example model with `stingray setup chat` and where the CLI stores it.
5. A compact example or link showing typed output chunks, including reasoning metadata where appropriate.
6. Links to the domain-specific speech, vision and diffusion APIs.
7. A concise section describing what is supported and where to find the detailed verification matrix.
8. Accurate limitations: hardware/backend requirements, model downloads, sample paths and any unsupported settings.

Do not put a giant capability matrix or a list of unverified model families on the package front page. Link to `docs/STATUS.md` for model-by-model confidence and evidence.

### C1. Align the repository-root README

Update `README.md` so that the first-run story matches the actual CLI and package README:

```text
install CLI
→ stingray setup chat
→ stingray chat
→ optional C# integration
```

Show `speak` and `transcribe` as working routes only after Phase B proves them.

The root README may be more tutorial-oriented than the NuGet README, but it must not imply that the catalogue contains models it does not contain or that a path-free task is available before the command exists.

### C2. Update the front-door and API documents

Update:

* `docs/3-product-and-runtime/103-front-door-design.md`
* `docs/3-product-and-runtime/2026-10-06-llamasharp-api-realignment-plan.md`

Record the real status of the facade, task commands, package README and executable documentation tests. Resolve their current differences in status language and milestone checkboxes.

`103-front-door-design.md` step 6 still specifies a **new** facade (`StingrayChat.OpenAsync`, `Speech.SynthesizeAsync`, `Transcriber.TranscribeAsync`, `Models.EnsureAsync`) and step 4 lists `describe` and `image` task commands. This plan (section 3.2) deliberately supersedes the chat facade and defers `describe`/`image`. Edit that document to say so explicitly (mark step 6 superseded, step 4 partially done with `describe`/`image` deferred) rather than leaving contradictory open checkboxes.

These documents should say what is implemented and what has been verified—not treat a completed design document as proof of a tested user workflow.

**Phase C exit gate:** The public documentation demonstrates the actual API and commands, with no stale examples or unsupported capability claims.

## 7. Phase D — Make examples executable evidence

Documentation must be tested, not merely reviewed.

### D1. Compile the public C# sample

Keep the example used in the NuGet README in a buildable sample project or an equivalent compile-test arrangement.

The sample must use the actual public package surface. Do not copy the code into a test project with special internal access or extra references that package consumers do not have.

Add it to the ordinary build/test workflow so API changes break the build when they invalidate the documented example.

### D2. Verify the actual NuGet package

A successful project-reference build alone is not sufficient because the project uses a custom packaging arrangement that bundles multiple assemblies and explicitly packages its README.

Add a reproducible package qualification step:

1. Build and pack `OpenTail.Stingray`.
2. Inspect the resulting `.nupkg` to confirm the public facade assemblies and XML documentation are present.
3. Confirm the package contains the current README at the expected location.
4. Consume the produced package from a clean temporary project using only the package reference and documented public dependencies.
5. Compile the public API example against that produced package.
6. Run a small real-weight smoke test where a suitable checkpoint is available.

The ordinary CI path should not require a multi-gigabyte model download. Separate compile/package correctness from model-dependent integration testing.

### D3. Add model-backed task qualification

The existing catalogue and installer tests prove useful parts of the installation machinery, but they do not alone prove that each public task actually works end to end.

Define a reproducible qualification command or integration-test set that:

* Uses the exact pinned catalogue files.
* Runs one small real chat continuation.
* Generates a real speech file with the bundled Piper voice.
* Transcribes a known input recording and checks a known expected result.
* Checks output files, exit codes and useful error messages.
* Records the model identity and verification evidence.

If a required checkpoint or reference recording is missing, report the integration test as skipped/blocked with a clear reason. Do not quietly count a no-op as a successful real-weight test.

Treat these as smoke/contract checks, not a substitute for existing numerical golden verification.

**Phase D exit gate:** The package and README examples compile independently of the repository's internal build graph, and the three catalogue-backed task routes have reproducible end-to-end evidence.

## 8. Phase E — Close-out and release-quality review

Before calling the work complete:

* Run the normal Debug and Release builds under the project's warning-as-error policy.
* Run the targeted Core, CLI and public-contract test suites.
* Run existing session and forward-pass fast tests affected by the facade.
* Run the package qualification step.
* Run the model-backed chat/speak/transcribe smoke tests where their exact assets are available.
* Confirm no existing command was removed, renamed or given a new incompatible default.
* Confirm unsupported model options are not described as working.
* Confirm no new reflection/dynamic-code path or Python-based tooling was introduced.
* Confirm NativeAOT and trimming constraints remain satisfied.
* Update the CLI option inventory, public documentation and front-door plan with the actual results.
* Record exact commands, pass/fail counts and any skipped/blocked integration tests.

Follow the repository's `CLAUDE.md` instructions. In particular, warnings are errors; do not use `--nologo` with `dotnet test`; build test projects before invoking their executable directly; use fully qualified test filters if needed; do not treat a sub-second no-op as real-weight verification.

## 9. Explicitly out of scope

Do not turn this work into a general rewrite.

* No new model architectures or inference kernels.
* No changes to inference arithmetic unless a concrete public-API integration bug exposes one.
* No new speculative-decoding implementation; expose and document only what the existing engine truly supports.
* No new general-purpose package manager or model registry service.
* No automatic downloads during chat, speech or transcription.
* No replacement of `ModelCatalog` with another source of truth.
* No reimplementation of TTS/ASR or duplication of the existing CLI pipeline logic.
* No attempt at binary compatibility with LLamaSharp.
* No broad CLI rewrite or removal of the advanced commands.
* No new convenience facade that duplicates `Model`, `ModelContext`, `IExecutor` or `ChatSession` without a separate demonstrated need.
* No claim that every advertised vision, audio or diffusion capability has been verified because the developer front door is improved.

## 10. Definition of done

The task is complete only when all of the following are true:

1. **Public C# API:** The documented `Model.Load` → `CreateContext` → `InteractiveExecutor` → `ChatSession` pattern compiles in a real consumer project using only the public API.
2. **Lifecycle:** Model/context ownership, disposal, cancellation and history behaviour are covered by regression tests.
3. **Chat front door:** `stingray chat` runs against the installed catalogue model without requiring a model path.
4. **Speech front door:** `stingray speak` runs the installed Piper bundle without a checkpoint path.
5. **Transcription front door:** `stingray transcribe <input>` uses the installed Whisper bundle without a checkpoint path and accurately documents accepted input formats.
6. **Catalog integrity:** Every command resolves only through the catalogue's installed state, with clear failure behaviour for missing/partial bundles; hashes are verified at install time by `setup`; the licence-consent decision from B3 is implemented and tested. No implicit network download occurs.
7. **Compatibility:** Existing root, `tts`, `stt`, `setup`, `models` and specialist command behaviour remains intact.
8. **NuGet package:** The produced package includes the right assemblies and current README, and the documented code compiles when consuming the packed artifact.
9. **Documentation tests:** The public C# example compiles in the standard build/test process; model-backed checks have explicit evidence and honest skip/block reporting.
10. **Documentation consistency:** Root README, package README, command help and design docs describe the same real workflow.
11. **Engineering constraints:** Warnings-as-errors and NativeAOT requirements remain satisfied; no unnecessary dependencies or internal access leaks have been added.
12. **Evidence:** The final report lists files changed, exact commands run, test results, package validation and any remaining limitations.

## 11. Required implementation report

At the end, provide:

* A short summary of what changed.
* A file-by-file change list.
* The final C# API sample and exact CLI usage.
* Exact build, test and pack commands with results.
* Which tests used real model weights and which were fast stub-based tests.
* Any model-dependent tests skipped and why.
* Any remaining compatibility or unsupported-parameter caveats.

The guiding principle is: **make Stingray's easiest path genuinely easy, and ensure every public example is backed by code that compiles and behaviour that has actually been tested.**
