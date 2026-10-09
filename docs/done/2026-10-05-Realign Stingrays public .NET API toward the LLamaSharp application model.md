# Task: Realign Stingray's public .NET API toward the LLamaSharp application model

Repository:

https://github.com/opentail-net/OpenTail.Stingray

Working directory:

`C:\Git-Public\OpenTail.Stingray`

## Objective

Redesign Stingray's **public application-facing API** so that it follows the useful architectural and ergonomic conventions of SciSharp LLamaSharp, while remaining a **general-purpose AI runtime** rather than becoming a LLaMA/llama.cpp clone.

This is an API architecture task, not a superficial rename exercise.

The desired outcome is that a higher-level .NET application can be written against a small, stable executor/session abstraction and have either a LLamaSharp-backed implementation or a Stingray-backed implementation underneath it with minimal wrapper-specific code.

The target conceptual model is:

```text
Model
  ↓
Context
  ↓
Executor
  ↓
Session
```

with explicit model/context parameters and generation parameters.

LLamaSharp reference repository:

`https://github.com/SciSharp/LLamaSharp`

Study the current `master` branch and particularly:

- [x] `LLama/Abstractions/ILLamaExecutor.cs`
- [x] `LLama/Abstractions/IContextParams.cs`
- [x] `LLama/Abstractions/IModelParams.cs`
- [x] `LLama/Common/ModelParams.cs`
- [x] `LLama/Common/InferenceParams.cs`
- [x] `LLama/ChatSession.cs`
- [x] `LLama/InteractiveExecutor.cs`
- [x] `LLama/StatelessExecutor.cs`
- [x] `LLama/Batched/BatchedExecutor.cs`
- [x] `LLama/ChatHistory.cs`
- [x] `docs/Architecture.md`
- [x] `docs/Tutorials/Executors.md`
- [x] `docs/Tutorials/ChatSession.md`

Do not assume the LLamaSharp API from memory. Re-read the current source.

---

## Critical design principle

Do **not** mechanically rename Stingray classes to LLamaSharp names.

In particular:

- [x] Do not introduce `LLamaWeights` merely because LLamaSharp has it;
- [x] Do not introduce LLaMA-specific naming into a runtime that supports Qwen, Gemma, RWKV, gpt-oss, DeepSeek, vision, diffusion, speech, etc.;
- [x] Do not make the Stingray core depend on LLamaSharp simply to obtain API compatibility;
- [x] Do not duplicate Stingray's engine underneath a second parallel API indefinitely.

Instead:

> Adopt the useful *separation of concerns and usage pattern* of LLamaSharp, while using Stingray-appropriate names and preserving Stingray's richer capabilities.

The API should communicate "general .NET AI runtime", not "llama.cpp wrapper".

---

## Current Stingray starting point

Inspect the current source before making changes.

Important current concepts include:

- [x] `IInferenceEngine`
- [x] `InferenceEngine`
- [x] `ContinuousBatchingEngine`
- [x] `SamplingParams`
- [x] `GenerateChunk`
- [x] `GenerateChunkKind`
- [x] `ITokenizer`
- [x] Existing session/runtime classes
- [x] Current model loading/runtime ownership
- [x] Current server/CLI APIs
- [x] Architecture registry and forward-pass selection

The current engine already has useful semantics that should **not** be discarded merely to resemble LLamaSharp.

In particular, Stingray already has:

```text
IAsyncEnumerable<string> GenerateAsync(...)
IAsyncEnumerable<GenerateChunk> GenerateChunksAsync(...)
```

and typed output including concepts such as:

```text
Text
Thinking
Usage
Stop
```

- [x] Preserve this richer native capability.

The LLamaSharp-shaped API should be a natural public layer over this, not a replacement for useful typed Stingray functionality.

---

## Required first step: API archaeology and design report

Before editing production code, inspect both repositories and produce a short design report in:

`docs/3-product-and-runtime/<date>-llamasharp-api-realignment-plan.md`

Use the current date in the filename.

The report must contain:

- [x] **1. LLamaSharp public API map**: List the important public concepts and their responsibilities:
  - `ModelParams`
  - `IModelParams`
  - `IContextParams`
  - `LLamaWeights`
  - `LLamaContext`
  - `ILLamaExecutor`
  - `InferenceParams`
  - `InteractiveExecutor`
  - `InstructExecutor`
  - `StatelessExecutor`
  - `BatchedExecutor`
  - `ChatHistory`
  - `ChatSession`
  *(Do not attempt to reproduce every LLamaSharp type. Identify what a normal higher-level application actually needs.)*

- [x] **2. Stingray mapping**: For each important concept, identify:
  ```text
  LLamaSharp concept
  → current Stingray equivalent
  → desired Stingray public concept
  → keep / rename / split / introduce / retire
  → reason
  ```

- [x] **3. Explicit naming decisions**: Make and document decisions for at least:
  - model object naming
  - context object naming
  - executor naming
  - generation-parameter naming
  - chat/session naming
  - tokenizer exposure
  - batching exposure
  *(The report must explicitly explain why any LLamaSharp names are deliberately NOT copied, e.g. `LLamaWeights` → not adopted → reason: Stingray is not LLaMA-specific.)*

- [x] **4. Compatibility target**: Define exactly what "LLamaSharp-like" means for this task.
  - The goal is **application-level familiarity and wrapper interchangeability**, not CLR binary compatibility with `LLama.Abstractions`.
  - Do not pretend that two identically shaped interfaces are binary-compatible.
  - If a very small optional compatibility package is useful later, mention it, but do not make it part of this core migration unless the design requires it.

---

## Proposed target API shape

The exact names may be adjusted during the design phase, but the resulting API should resemble this structure:

```csharp
Model
    CreateContext(...)

Context
    Tokenizer
    model/runtime context state

IExecutor
    InferAsync(...)

InferenceParams

InteractiveExecutor
StatelessExecutor
BatchedExecutor

ChatHistory
ChatSession
```

- [x] Prefer a neutral `Model` / `Context` concept rather than LLaMA-specific names.
- [x] Do not blindly use these exact names if they create ambiguity with existing .NET APIs; make the final choice deliberately and document it.

---

## API requirements

- [x] **1. Model lifecycle**
  - The public API should permit:
    ```csharp
    var parameters = new ModelParams(modelPath)
    {
        ContextSize = 4096,
        ...
    };

    using var model = Model.Load(parameters);

    using var context = model.CreateContext(parameters);

    var executor = new InteractiveExecutor(context);
    ```
  - The exact concrete types and factories can differ, but the lifecycle must be clear:
    ```text
    load model
    → create context
    → create executor
    → optionally create session
    ```
  - The model/weights object must be reusable for multiple contexts where Stingray's implementation permits it.
  - Do not duplicate large model state unnecessarily.

- [x] **2. Context parameters**
  - Create a coherent context-configuration abstraction inspired by LLamaSharp's `IContextParams`.
  - Only expose settings that Stingray actually supports or can sensibly represent.
  - Do not create fake properties purely for API similarity.
  - Where Stingray has concepts that LLamaSharp doesn't expose, keep them Stingray-specific (e.g. context length, batch/physical batch size, sequence count, embedding mode, thread controls, K/V cache types, RoPE/YaRN overrides, backend/device configuration).
  - Use evidence from the current Stingray implementation.

- [x] **3. Model parameters**
  - Create a coherent model-load configuration type inspired by LLamaSharp `ModelParams`.
  - It should express things such as: model path, backend, GPU/device preferences, GPU layer count, context settings, relevant loading options.
  - Do not put generation/sampling options into model parameters.

- [x] **4. Inference parameters**
  - Realign `SamplingParams` toward an LLamaSharp-like `InferenceParams` concept if the archaeology shows that this improves the public API.
  - Do not merely rename the existing record; separate concerns where useful.
  - The preferred conceptual split is:
    ```text
    InferenceParams
        generation budget
        stopping
        sampling
        thinking
        constraints / generation options
    ```
  - Potentially `SamplingParams` as a nested or reusable component if that gives a cleaner architecture.
  - Preserve existing Stingray features such as: temperature, top-k, top-p, seed, max-new-token budget, stop tokens/sequences, logit bias, thinking controls, allowed choices / constraints, and other currently supported controls.
  - Do not remove capabilities just because LLamaSharp does not expose them under the same name.

- [x] **5. Executor model**
  - Provide the LLamaSharp-style conceptual executor separation:
    ```text
    InteractiveExecutor
    StatelessExecutor
    BatchedExecutor
    ```
    only where the semantics genuinely differ.
  - Do not create empty wrapper classes whose only purpose is naming.
  - Map them to the existing Stingray runtime appropriately:
    ```text
    StatelessExecutor
        → one-shot/no retained conversational state

    InteractiveExecutor
        → stateful single-context generation

    BatchedExecutor
        → ContinuousBatchingEngine / equivalent
    ```
  - Document exactly what each executor owns and what it does not.

- [x] **6. Common executor interface**
  - Create a small executor abstraction with the same broad usage pattern as LLamaSharp:
    ```csharp
    IAsyncEnumerable<string> InferAsync(
        string text,
        IInferenceParams? inferenceParams = null,
        CancellationToken cancellationToken = default);
    ```
  - The exact name/types should be decided after reviewing the current code.
  - The interface should be intentionally small.
  - Do not put model registry, backend diagnostics, batching metrics, placement policy, or server-specific concepts into the basic executor interface (those belong elsewhere).

- [x] **7. Preserve Stingray typed generation**
  - The public API must continue to provide a richer typed stream in addition to the simple text stream:
    ```csharp
    IAsyncEnumerable<GenerateChunk> InferChunksAsync(...)
    ```
    or another clearly named equivalent.
  - The relationship should be:
    ```text
    simple text API
        → convenient compatibility surface

    typed chunk API
        → canonical Stingray-native surface
    ```
  - Do not make higher-level applications lose access to thinking/usage/stop metadata.

- [x] **8. ChatSession / ChatHistory**
  - Design a Stingray equivalent of LLamaSharp's `ChatSession` and `ChatHistory`.
  - The public API should support the normal flow:
    ```csharp
    var session = new ChatSession(executor);

    session.AddSystemMessage(...);

    await foreach (var text in session.ChatAsync(...))
    {
        ...
    }
    ```
  - Support: system messages, user messages, assistant messages, history access, conversation continuation, cancellation, generation parameters.
  - Where appropriate, preserve Stingray's existing session/KV-cache mechanisms rather than inventing an independent state system.
  - Do not copy LLamaSharp's transform architecture unless Stingray actually needs it.

- [x] **9. Batching**
  - `BatchedExecutor` should map cleanly to Stingray's existing continuous batching implementation.
  - Do not promise batching merely because the type exists. Capability must reflect actual runtime:
    ```text
    supported model/pass
    → batching allowed

    unsupported combination
    → clear refusal/fallback
    ```
  - Use existing capability checks rather than duplicating a second batching compatibility matrix in the new API.

---

## Backward compatibility

Treat the current Stingray public API as existing API.

Before removing or renaming public types:

- [x] 1. Identify every production/test/sample use;
- [x] 2. Determine whether the old API is already published/advertised;
- [x] 3. Decide whether compatibility shims are appropriate;
- [x] 4. Avoid keeping two complete APIs forever.

If compatibility aliases are required, they must be extremely thin and documented as migration helpers.

Do not build a permanent duplicate abstraction stack. The preferred end state is:

```text
new coherent public API
        ↓
existing runtime implementation
```

not:

```text
old API
   ↓
adapter
   ↓
new API
   ↓
another adapter
   ↓
engine
```

---

## NativeAOT and architecture constraints

Everything must continue to work with Stingray's existing constraints:

- [x] .NET 10
- [x] NativeAOT
- [x] `TreatWarningsAsErrors`
- [x] No reflection-based discovery
- [x] No dynamic code generation
- [x] No Python dependency
- [x] No llama.cpp/ggml runtime dependency

Do not introduce an abstraction that requires runtime reflection or dynamic proxy generation.

The public API must remain compatible with Stingray's current NativeAOT design.

---

## Do not break the engine architecture

The following should remain engine concerns:

- [x] Architecture registry
- [x] Forward-pass selection
- [x] CPU/Vulkan/CUDA implementations
- [x] TierPlanner
- [x] MoE routing
- [x] KV/cache implementation
- [x] Tensor loading
- [x] Backend creation
- [x] Kernel selection
- [x] Placement
- [x] Quantisation implementation

Executors should **coordinate** these components, not absorb their implementation.

The public API should not start knowing that a particular model is:

```text
Qwen3.6
DeepSeek MLA
RWKV
gpt-oss
```

Those decisions remain inside the runtime.

---

## Required implementation sequence

Work in small commits.

- [x] **Commit 1 — design**
  - Add the API archaeology/design document only (`docs/3-product-and-runtime/<date>-llamasharp-api-realignment-plan.md`).
  - No runtime changes.

- [x] **Commit 2 — foundational API types**
  - Introduce the new public concepts:
    - model parameters
    - context parameters
    - inference parameters
    - executor abstraction
    - model/context abstractions
  - Do not migrate the whole engine yet.
  - Build and test.

- [x] **Commit 3 — model/context lifecycle**
  - Implement:
    ```text
    Model
    → Context
    ```
    against the existing Stingray runtime.
  - Ensure ownership/disposal is correct (a model must not be disposed before dependent contexts/executors).
  - Test multiple contexts where supported.

- [x] **Commit 4 — executor migration**
  - Implement:
    - interactive executor
    - stateless executor
    - batched executor where appropriate
    using the existing engine implementations.
  - Avoid duplicating inference algorithms.

- [x] **Commit 5 — ChatSession / ChatHistory**
  - Build the high-level conversational API.
  - Reuse existing session/runtime functionality.

- [x] **Commit 6 — compatibility cleanup**
  - Update samples/tests/documentation to use the new public API.
  - Only retain old API aliases where there is a concrete reason.
  - Delete duplicate public paths once all consumers migrate.

---

## Testing requirements

Add tests for the public API at the level a higher-level application would use it.

At minimum:

- [x] **API shape tests**: Verify the expected public types and core methods exist.

- [x] **Lifecycle tests**:
  ```text
  Model
  → Context
  → Executor
  → Generate
  → Dispose
  ```
  Verify disposal order and failure behaviour.

- [x] **Text generation**:
  - Use at least one existing small real GGUF.
  - Verify:
    - deterministic greedy generation
    - expected token IDs
    - expected text
    - streaming completion

- [x] **Stateful session**:
  - Verify:
    ```text
    session turn 1
    → session turn 2
    ```
    retains conversation state correctly.

- [x] **Stateless executor**: Verify independent calls do not accidentally inherit previous state.

- [x] **Batched executor**: Where a currently supported model/path exists, verify multiple requests generate independently and correctly.

- [x] **Typed stream**:
  - Verify the simple text API and typed chunk API agree on user-facing text.
  - Verify thinking/usage/stop metadata remains available through the richer API.

- [x] **Tokenizer**: Expose tokenizer functionality only where the design calls for it, and test encode/decode behaviour against the existing `ITokenizer`.

---

## LLamaSharp comparison tests

- [x] Create a small documentation/sample project or test fixture demonstrating the same conceptual task implemented twice:
  ```text
  LLamaSharp version
  Stingray version
  ```
- [x] Measure application-level similarity.
- [x] Do NOT require binary compatibility with LLamaSharp assemblies.
- [x] A wrapper should be able to abstract over the two implementations without having to understand Stingray internals.

---

## Documentation

- [x] Update the README and relevant reference documentation so a new .NET developer can understand:
  ```text
  Model
  Context
  Executor
  Session
  InferenceParams
  ```
  within one page.

- [x] Include a "Coming from LLamaSharp" document ([docs/3-product-and-runtime/coming-from-llamasharp.md](../3-product-and-runtime/coming-from-llamasharp.md)):
  ```text
  LLamaSharp concept
  → Stingray concept
  → notes / differences
  ```
  Be explicit about deliberate differences.

- [x] Do not claim "LLamaSharp compatible" unless the implementation actually supports the claimed level of compatibility. Use more precise language such as:
  > "LLamaSharp-shaped application API"
  or
  > "designed to make higher-level executor wrappers portable between LLamaSharp and Stingray"
  unless stronger compatibility has genuinely been proven.

---

## Quality bar

Before declaring completion, answer these questions in the final report:

- [x] 1. What parts of LLamaSharp were intentionally mirrored?
- [x] 2. What parts were intentionally NOT mirrored?
- [x] 3. Which current Stingray types were renamed?
- [x] 4. Which current Stingray types were split?
- [x] 5. Which current Stingray types were retained?
- [x] 6. Did any engine functionality get lost? The answer should be **no**.
- [x] 7. Can one higher-level wrapper target both implementations with only a thin backend-specific adapter?
- [x] 8. Is the public API still clearly a general AI runtime rather than a LLaMA wrapper?
- [x] 9. Is NativeAOT still clean?
- [x] 10. What migration burden remains for existing Stingray callers?

### Important

Do not optimize for "looks like LLamaSharp" at the expense of good API design.

The target is:

> **LLamaSharp familiarity + Stingray architecture + Stingray capabilities.**

Do not stop after renaming classes.

Do not create a facade and declare success.

The desired result is for the **actual Stingray public API** to have a coherent model/context/executor/session design that a .NET developer familiar with LLamaSharp immediately understands, while still making sense for non-LLaMA model families and Stingray's broader AI scope.

---

## Closure notes (2026-10-06)

- Real-weights verification: the earlier `PublicApiContractsTests` resolved `models/...` relative to the working directory and silently no-opped (0.14 s for 17 tests; a deliberately wrong assertion still passed). Paths now anchor on the repo root. Re-run: 27 tests, ~9.5 s, weights loaded; `ChatSession_RealSmolLM2_GreedyMatchesCliBaseline_AndStatefulTurns` matches the CPU greedy baseline in `2026-10-05-forward-pass-selection-matrix.md` and checks a second stateful turn with `PrefillTokensReused > 0`.
- Runtime integration: Both `ModelContext.CreateDefaultEngine()` and `CreateContinuousBatchingEngine()` directly consult `ForwardPassSelection` and `ArchitectureRegistry` through unified `BuildForwardPass` as the single source of truth. Validates admission status, unsupported backend strings, TurboQuant head dim and mode constraints, and SafeTensors offload restrictions. Maps across all forward pass kinds: `CpuDense`, `SafeTensorsCpu`, `CpuHybridGdn`, `RwkvCpu`, `GptOssCpu`, `GptOssVulkan`, `DeepSeek2Vulkan`, `CudaDense`, `CudaHybridGdn`, `CudaHybrid`, `VulkanDense`, `VulkanHybridGdn`, `VulkanHybrid`, and `VulkanLayerSplit`. `CreateContinuousBatchingEngine` verifies `IBatchedForwardPass` capability and fails closed with `NotSupportedException` for non-batchable architectures.
- Cleaned public parameters: Inert speculative decoding fields (`DraftModelPath`, `DraftLookup`, `DSparkModelPath`) and `Embeddings` were pruned from `IModelParams`/`IContextParams`. `FlashAttention` is wired to `GpuForwardPass.DisableFlashAttention` and `TurboQuantHeadDim` is wired into request `HeadDim`.
- Multi-sequence batching verification: `BatchedExecutor_ConcurrentDualInference_ExecutesConcurrentlyOnContinuousBatchingEngine` inspects `IContinuousBatchingObservability` (`BatchedArgmaxSteps`, `BatchedArgmaxSequences`, `BatchedFullLogitsSteps`, `BatchedFullLogitsSequences`) and asserts `totalSteps > 0`, `totalSeqs >= 2`, and `totalSeqs > totalSteps`, proving multi-sequence lock-step batched decode steps on `ContinuousBatchingEngine` with real model weights.
- Old engine types (`InferenceEngine`, `ContinuousBatchingEngine`, `SamplingParams`) are intentionally retained as the runtime layer under the new API; see `coming-from-llamasharp.md`.
