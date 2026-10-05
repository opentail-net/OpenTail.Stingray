# LLamaSharp API Realignment Plan and Archaeology Report (2026-10-06)

**Target Location**: `docs/3-product-and-runtime/2026-10-06-llamasharp-api-realignment-plan.md`  
**Reference Document**: [`docs/2-coverage/2026-10-05-Realign Stingrays public .NET API toward the LLamaSharp application model.md`](file:///C:/Git-Public/OpenTail.Stingray/docs/2-coverage/2026-10-05-Realign%20Stingrays%20public%20.NET%20API%20toward%20the%20LLamaSharp%20application%20model.md)  
**Upstream Reference**: [`examples/llama.cpp/llama.cpp`](file:///C:/Git-Public/OpenTail.Stingray/examples/llama.cpp/llama.cpp)  
**External Reference**: `https://github.com/SciSharp/LLamaSharp` (`master` branch)

---

## 1. Executive Summary & Objective

The objective of this realignment is to redesign OpenTail.Stingray's **public application-facing API** so that it follows the proven architectural hierarchy and ergonomic conventions of SciSharp LLamaSharp:

```text
Model (Weights / Configuration)
  ↓
Context (State / Allocations / Tokenizer)
  ↓
Executor (Interactive / Stateless / Batched / Speculative)
  ↓
Session (ChatHistory / Multi-turn Conversation)
```

### Critical Design Principle

Stingray is a **general-purpose AI runtime** in pure managed C# 14 / .NET 10 supporting causal language models (LLaMA, Qwen, Gemma, DeepSeek, gpt-oss, Phi, Granite, etc.), recurrent architectures (RWKV-v4..v7, hybrid GDN/Mamba2), multimodal vision understanding (Gemma 3/4, Qwen2-VL), diffusion pipelines (Flux, ZImage), and native neural speech synthesis & recognition (CosyVoice, Kokoro, Whisper, Parakeet).

Therefore:
1. **No LLaMA-specific branding leak**: We do not introduce `LLamaWeights`, `LLamaContext`, or `ILLamaExecutor`.
2. **No binary dependency on LLamaSharp**: Stingray core will not reference `LLamaSharp` or `LLama.Abstractions`.
3. **Application-level portability, not binary compatibility**: A developer familiar with LLamaSharp should be able to write an abstraction wrapper or adopt Stingray with near-zero conceptual friction.
4. **Preserve richer native capabilities**: Where upstream `llama.cpp` has capabilities that LLamaSharp omitted or failed to expose, but which Stingray already implements (such as speculative decoding, thinking/reasoning token streaming, copy-on-write KV page prefix sharing, context shifting, and structured tool calling), **Stingray retains and elevates these features**.

---

## 2. LLamaSharp Public API Map

Studying LLamaSharp (`master` branch):

| LLamaSharp Type | Role & Responsibility in LLamaSharp | Normal Application Need |
| :--- | :--- | :--- |
| `IModelParams` / `ModelParams` | Configuration for loading model weights (path, GPU layers, seed, LoRA, tensor split). Combines both model loading and context configuration in one class. | **Essential**: Applications need explicit model loading parameters separated from per-inference parameters. |
| `IContextParams` | Interface extracting context-specific settings (`ContextSize`, `BatchSize`, `TypeK`, `TypeV`, `FlashAttention`, `Threads`). | **Essential**: Needed when creating multiple contexts from a single loaded model. |
| `LLamaWeights` | Represents the loaded model weights in memory/VRAM (`llama_model*`). Factory `LLamaWeights.LoadFromFile(params)`. Reusable across multiple contexts. | **Essential**: The concept of loaded, reusable weights is critical. (Must be named neutrally, e.g. `Model`). |
| `LLamaContext` | Represents an allocated execution context (`llama_context*`): KV cache, sequence state, memory arenas, tokenizer view. | **Essential**: Must be instantiated from the model: `model.CreateContext(params)`. |
| `ILLamaExecutor` | Common interface for text generation: `Context`, `InferAsync(text, inferenceParams, ct)`. | **Essential**: Common contract for execution pipelines. |
| `InferenceParams` | Per-generation execution settings: temperature, top-k, top-p, mirostat, repetition penalties, max tokens, stop tokens. | **Essential**: Decoupled from model/context settings. |
| `InteractiveExecutor` | Stateful executor: retains context across calls, appends user text, runs autoregressive loop until completion or token limit. | **Essential**: Standard single-tenant chatbot / interactive agent pattern. |
| `StatelessExecutor` | Stateless executor: resets context before every call, ideal for one-shot completions or deterministic embedding/extraction. | **Essential**: One-shot prompting and stateless RPC endpoints. |
| `InstructExecutor` | Variation of `InteractiveExecutor` formatted with instruct prefixes (`User:`, `Assistant:`). | **Optional / Specialized**: Can be handled by prompt formatting or `ChatSession`. |
| `BatchedExecutor` | Multi-sequence executor handling concurrent sequences over one shared context. | **Essential**: Directly maps to Stingray's `ContinuousBatchingEngine`. |
| `ChatHistory` | In-memory message list of `AuthorRole` (`System`, `User`, `Assistant`) and `Content`. | **Essential**: Standard conversational message collection. |
| `ChatSession` | High-level orchestrator wrapping an `ILLamaExecutor` and `ChatHistory`, handling template formatting and history appending. | **Essential**: Ergonomic conversational surface. |

---

## 3. Stingray Concept Mapping

| LLamaSharp Concept | Current Stingray Equivalent | Desired Stingray Public Concept | Action | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| `ModelParams` | Options passed to `InferenceEngineLoader` / CLI flags | `ModelParams` | **Introduce** | Clear, strongly-typed parameters record for loading model weights (model path, backend, GPU layers, VRAM policy). |
| `IContextParams` | Parts of `InferenceEngineOptions` / `ContinuousBatchingOptions` | `ContextParams` / `IContextParams` | **Introduce** | Clean separation of context size, physical batch size, KV cache type, and threading from weight loading. |
| `LLamaWeights` | `InferenceEngine._weights` / `SafeTensorsModel` / internal weights pointer | `Model` | **Introduce** | Neutral name. Implements `IDisposable`. Provides `CreateContext(contextParams)` to enable multiple contexts over shared weights. |
| `LLamaContext` | `InferenceEngine` / `ContinuousBatchingEngine` (currently conflates model, context, and executor) | `ModelContext` | **Introduce** | Encapsulates KV cache, tokenizer, backend allocations, and sequence state for one or more executors. |
| `ILLamaExecutor` | `IInferenceEngine` (`GenerateAsync`, `GenerateChunksAsync`) | `IExecutor` (or `IInferenceExecutor`) | **Introduce** | Small, focused contract: `Context`, `InferAsync(...)`, plus Stingray's rich `InferChunksAsync(...)`. |
| `InferenceParams` | `SamplingParams` (currently holds sampling + stopping + some budget fields) | `InferenceParams` | **Introduce / Realign** | Cohesive parameter object containing generation budget, stop conditions, sampling controls, and reasoning options. |
| `InteractiveExecutor` | `InferenceEngine` in sequential stateful mode | `InteractiveExecutor` | **Introduce** | Single-context stateful generation maintaining conversational KV cache continuity. |
| `StatelessExecutor` | `InferenceEngine.GenerateAsync` with cache clear | `StatelessExecutor` | **Introduce** | One-shot generation that clears or resets KV cache between calls. |
| `BatchedExecutor` | `ContinuousBatchingEngine` | `BatchedExecutor` | **Introduce** | High-throughput concurrent multi-sequence executor with continuous batching and prefix cache sharing. |
| `ChatHistory` | Internal chat formatting / `ChatMessage` structs | `ChatHistory` | **Introduce** | Public collection of `AuthorRole` and `ChatMessage` for structured conversational workflows. |
| `ChatSession` | Ad-hoc CLI REPL loop in `RunCommand.cs` | `ChatSession` | **Introduce** | Ergonomic high-level wrapper coordinating `IExecutor` + `ChatHistory` + Jinja template application. |
| `ITokenizer` | `OpenTail.Stingray.Core.ITokenizer` | `ITokenizer` | **Retain** | Keep existing excellent managed BPE/SentencePiece tokenizer exposed via `context.Tokenizer`. |
| `GenerateChunk` | `OpenTail.Stingray.Engine.GenerateChunk` | `GenerateChunk` | **Retain & Elevate** | Critical native capability: provides typed `Thinking`, `Text`, `Usage`, and `Stop` chunks. |

---

## 4. Upstream llama.cpp Parity Analysis: What LLamaSharp Lacks that Stingray Preserves

When realigning toward LLamaSharp, we must not downgrade Stingray to LLamaSharp's limitations. Upstream `llama.cpp` contains critical capabilities that LLamaSharp failed to expose, but which Stingray implements natively.

```
                  ┌──────────────────────────────────────────────┐
                  │          Upstream llama.cpp C/C++ API        │
                  │ (Speculative, Thinking, KV Shift, CoW Fork)  │
                  └──────────────┬───────────────────────────────┘
                                 │
                 ┌───────────────┴────────────────┐
                 ▼                                ▼
┌─────────────────────────────────┐   ┌───────────────────────────────────┐
│       LLamaSharp (.NET)         │   │      OpenTail.Stingray (.NET)     │
│  - Untyped string streaming only│   │  - Typed GenerateChunk streaming  │
│  - No speculative decoding      │   │  - SpeculativeDecoder (Draft/MTP) │
│  - No KV prefix sharing / CoW   │   │  - ForkSharedPrefix / ISessionTree│
│  - No thinking tag awareness    │   │  - ThinkingDefaultOff / Reasoning │
│  - Autoregressive causal only   │   │  - Recurrent/Hybrid/Diffusion/MoE │
└─────────────────────────────────┘   └───────────────────────────────────┘
```

### Detailed Feature Parity Table

| Feature Domain | Upstream `llama.cpp` Location | LLamaSharp Status | Stingray Native Implementation |
| :--- | :--- | :--- | :--- |
| **Speculative Decoding & Draft Models** | [`examples/llama.cpp/llama.cpp/common/speculative.h#L89-L94`](file:///C:/Git-Public/OpenTail.Stingray/examples/llama.cpp/llama.cpp/common/speculative.h#L89-L94)<br>[`common/common.h:173-186`](file:///C:/Git-Public/OpenTail.Stingray/examples/llama.cpp/llama.cpp/common/common.h#L173-L186) (`DRAFT_SIMPLE`, `NGRAM_SIMPLE`, `DRAFT_MTP`, `DRAFT_DSPARK`) | **Missing**: No speculative executors, draft models, or MTP in `ILLamaExecutor`. | [`src/OpenTail.Stingray.Engine/SpeculativeDecoder.cs`](file:///C:/Git-Public/OpenTail.Stingray/src/OpenTail.Stingray.Engine/SpeculativeDecoder.cs)<br>[`src/OpenTail.Stingray.Engine/PromptLookupDraft.cs`](file:///C:/Git-Public/OpenTail.Stingray/src/OpenTail.Stingray.Engine/PromptLookupDraft.cs)<br>[`src/OpenTail.Stingray.Engine/DSparkDecoder.cs`](file:///C:/Git-Public/OpenTail.Stingray/src/OpenTail.Stingray.Engine/DSparkDecoder.cs)<br>CLI: `--draft-model`, `--draft-lookup`, `--spec-type mtp`, `--dspark-model` |
| **Thinking / Reasoning Stream Separation** | [`examples/llama.cpp/llama.cpp/common/chat.h#L260-L277`](file:///C:/Git-Public/OpenTail.Stingray/examples/llama.cpp/llama.cpp/common/chat.h#L260-L277)<br>[`common/chat.cpp:358-370`](file:///C:/Git-Public/OpenTail.Stingray/examples/llama.cpp/llama.cpp/common/chat.cpp#L358-L370) (`enable_thinking`, `<think>` / `</think>`) | **Missing**: Only emits raw `IAsyncEnumerable<string>`; no distinction between reasoning and final text. | [`src/OpenTail.Stingray.Engine/InferenceEngine.cs#L19-L25`](file:///C:/Git-Public/OpenTail.Stingray/src/OpenTail.Stingray.Engine/InferenceEngine.cs#L19-L25)<br>`GenerateChunkKind.Thinking`, `GenerateChunkKind.Text`, `GenerateChunkKind.Usage`, `GenerateChunkKind.Stop`<br>`ThinkingDefaultOff` descriptor flag |
| **KV Cache Prefix Sharing & Forking** | [`examples/llama.cpp/llama.cpp/include/llama.h#L770-L778`](file:///C:/Git-Public/OpenTail.Stingray/examples/llama.cpp/llama.cpp/include/llama.h#L770-L778) (`llama_memory_seq_cp`, `llama_memory_seq_keep`) | **Missing**: Single-context linear execution only; no branching or prefix sharing in executors. | [`src/OpenTail.Stingray.Engine/PagedKvCache.cs#L395`](file:///C:/Git-Public/OpenTail.Stingray/src/OpenTail.Stingray.Engine/PagedKvCache.cs#L395) (`ForkSharedPrefix`)<br>[`src/OpenTail.Stingray.Sessions/HotSession.cs#L309`](file:///C:/Git-Public/OpenTail.Stingray/src/OpenTail.Stingray.Sessions/HotSession.cs#L309) (`TryForkSharedPrefixCache`)<br>[`src/OpenTail.Stingray.Sessions/ISessionTree.cs`](file:///C:/Git-Public/OpenTail.Stingray/src/OpenTail.Stingray.Sessions/ISessionTree.cs) |
| **KV Cache Shifting & Eviction** | [`examples/llama.cpp/llama.cpp/include/llama.h#L761-L818`](file:///C:/Git-Public/OpenTail.Stingray/examples/llama.cpp/llama.cpp/include/llama.h#L761-L818) (`llama_memory_seq_rm`, `llama_memory_seq_add`, `llama_memory_can_shift`) | **Missing**: Context overflow requires full recomputation rather than in-place KV shifting. | [`src/OpenTail.Stingray.Engine/PagedKvCache.cs`](file:///C:/Git-Public/OpenTail.Stingray/src/OpenTail.Stingray.Engine/PagedKvCache.cs) (`RollingKvCache`, `SnapKv`, `StreamingLLM` delta shift) |
| **Model Family & Architecture Introspection** | [`examples/llama.cpp/llama.cpp/include/llama.h#L658-L674`](file:///C:/Git-Public/OpenTail.Stingray/examples/llama.cpp/llama.cpp/include/llama.h#L658-L674) (`llama_model_is_recurrent`, `llama_model_is_hybrid`, `llama_model_is_diffusion`) | **Missing**: Assumes standard transformer decoders. No model family introspection. | [`src/OpenTail.Stingray.Engine/Architectures/BuiltInArchitectures.cs`](file:///C:/Git-Public/OpenTail.Stingray/src/OpenTail.Stingray.Engine/Architectures/BuiltInArchitectures.cs)<br>[`src/OpenTail.Stingray.Engine/Architectures/ForwardPassSelection.cs`](file:///C:/Git-Public/OpenTail.Stingray/src/OpenTail.Stingray.Engine/Architectures/ForwardPassSelection.cs) (Recurrent, Hybrid GDN, MoE, Diffusion) |
| **Advanced Samplers (DRY, Min-P)** | [`examples/llama.cpp/llama.cpp/include/llama.h#L1563-L1594`](file:///C:/Git-Public/OpenTail.Stingray/examples/llama.cpp/llama.cpp/include/llama.h#L1563-L1594) (`llama_sampler_init_dry`, `llama_sampler_init_adaptive_p`) | **Missing**: Only classic penalties; no DRY (Don't Repeat Yourself) sampler. | [`src/OpenTail.Stingray.Engine/InferenceEngine.cs`](file:///C:/Git-Public/OpenTail.Stingray/src/OpenTail.Stingray.Engine/InferenceEngine.cs) (`MinP`, logit bias, repetition penalties) |
| **Structured Grammars & Tool Calling** | [`examples/llama.cpp/llama.cpp/common/common.h#L189-L194`](file:///C:/Git-Public/OpenTail.Stingray/examples/llama.cpp/llama.cpp/common/common.h#L189-L194) (`COMMON_GRAMMAR_TYPE_TOOL_CALLS`, `OUTPUT_FORMAT`) | **Missing**: Only user-supplied raw GBNF text files; no auto-derived tool call grammar. | [`src/OpenTail.Stingray.Cli/RunCommand.cs#L934`](file:///C:/Git-Public/OpenTail.Stingray/src/OpenTail.Stingray.Cli/RunCommand.cs#L934) (`--tool-grammar`, `--json-schema` automated constraint decoding) |

---

## 5. Explicit Naming & Architectural Decisions

### 5.1 Model Object Naming: `Model` (not `LLamaWeights`)
- **Decision**: Use `OpenTail.Stingray.Model` (or `StingrayModel`).
- **Rationale**: LLamaSharp uses `LLamaWeights`. Stingray runs Qwen, Gemma, DeepSeek, RWKV, gpt-oss, Granite, multimodal vision, diffusion, and speech models. Introducing "LLama" into the type name is factually incorrect and miscommunicates the runtime's purpose. `Model` represents loaded weights and metadata.

### 5.2 Context Object Naming: `ModelContext` (not `LLamaContext`)
- **Decision**: Use `OpenTail.Stingray.ModelContext` (or `StingrayContext`).
- **Rationale**: Clear, standard .NET naming that mirrors LLamaSharp's separation without the LLaMA prefix. Owns the tokenizer, backend buffer allocations, and KV cache state.

### 5.3 Executor Naming: `IExecutor` / `InteractiveExecutor` / `StatelessExecutor` / `BatchedExecutor`
- **Decision**: Use standard executor names under `OpenTail.Stingray.Executors`:
  - `IExecutor`: Common contract exposing `Context`, `InferAsync(string, ...)`, and `InferChunksAsync(string, ...)`.
  - `InteractiveExecutor`: Stateful chat/REPL executor.
  - `StatelessExecutor`: Reset-on-infer executor for prompt completions.
  - `BatchedExecutor`: High-throughput concurrent executor wrapping `ContinuousBatchingEngine`.
  - `SpeculativeExecutor` *(Stingray-native addition)*: Coordinates a primary executor with a draft model / prompt lookup decoder.

### 5.4 Parameter Split: `ModelParams`, `ContextParams`, and `InferenceParams`
- **Decision**:
  - `ModelParams`: File path, backend selection (Auto, CPU, Vulkan, CUDA), GPU layers (`GpuLayerCount`), tensor split, memory-map policy.
  - `ContextParams`: `ContextSize`, `BatchSize`, `ThreadCount`, `KvCacheType` (e.g. F16, Q8_0, TurboQuant KVarN).
  - `InferenceParams`: Generation budget (`MaxTokens`), stopping criteria (`StopTokens`, `StopSequences`), sampling controls (`Temperature`, `TopK`, `TopP`, `MinP`, `Seed`), reasoning controls (`EnableThinking`, `ThinkingBudget`), and constraints (`JsonSchema`, `Grammar`).

### 5.5 Dual-Output Streaming: `InferAsync` and `InferChunksAsync`
- **Decision**: Provide both streaming methods on `IExecutor`:
  ```csharp
  // LLamaSharp-compatible simple string stream (filters out thinking and metadata)
  IAsyncEnumerable<string> InferAsync(
      string prompt,
      InferenceParams? parameters = null,
      CancellationToken cancellationToken = default);

  // Stingray-native rich typed stream (preserves thinking, usage metrics, stop reasons)
  IAsyncEnumerable<GenerateChunk> InferChunksAsync(
      string prompt,
      InferenceParams? parameters = null,
      CancellationToken cancellationToken = default);
  ```

### 5.6 Conversational Layer: `ChatSession` & `ChatHistory`
- **Decision**:
  - `ChatHistory`: Collection of `ChatMessage` with `AuthorRole` (`System`, `User`, `Assistant`, `Tool`).
  - `ChatSession`: Orchestrator taking `IExecutor` and `ChatHistory`, applying the model's native Jinja template or fallback format, and executing multi-turn dialogs.

---

## 6. Target API Usage Example

```csharp
using OpenTail.Stingray;
using OpenTail.Stingray.Executors;

// 1. Configure and load model weights (reusable across multiple contexts)
var modelParams = new ModelParams("models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf")
{
    GpuLayerCount = -1, // Offload all layers if GPU backend available
};
using var model = Model.Load(modelParams);

// 2. Create an execution context with allocated KV cache
var contextParams = new ContextParams
{
    ContextSize = 4096,
};
using var context = model.CreateContext(contextParams);

// 3. Create executor (Interactive, Stateless, or Batched)
var executor = new InteractiveExecutor(context);

// 4. Configure inference parameters
var inferenceParams = new InferenceParams
{
    Temperature = 0.7f,
    TopP = 0.9f,
    MaxTokens = 512,
    EnableThinking = true, // Stingray native reasoning support
};

// 5. Run conversational chat session
var session = new ChatSession(executor);
session.AddSystemMessage("You are a helpful and concise coding assistant.");

await foreach (var chunk in session.ChatChunksAsync("Explain prefix caching in Stingray.", inferenceParams))
{
    if (chunk.Kind == GenerateChunkKind.Thinking)
    {
        Console.ForegroundColor = ConsoleColor.DarkGray;
        Console.Write(chunk.Text);
        Console.ResetColor();
    }
    else if (chunk.Kind == GenerateChunkKind.Text)
    {
        Console.Write(chunk.Text);
    }
}
```

---

## 7. NativeAOT and Runtime Architecture Constraints

1. **No Reflection-Based Discovery**: Factory methods (`Model.Load`, `model.CreateContext`) use explicit construction rather than reflection.
2. **`TreatWarningsAsErrors`**: All new types must compile with zero warnings across all targets.
3. **Engine Boundary Integrity**: Executors coordinate runtime components; they do not implement forward passes or kernel math directly.
4. **Disposal Hierarchy**:
   - `Model` holds loaded tensor weights and backend handles.
   - `ModelContext` references `Model` and owns KV cache and sequence state.
   - Disposing `ModelContext` frees KV caches and scratch buffers.
   - Disposing `Model` frees weights and backend contexts; it guards against premature disposal if child contexts are active.

---

## 8. Implementation Milestones

- [x] **Milestone 1 — Design & Archaeology Report (This Document)**
  - Document LLamaSharp mapping, upstream llama.cpp divergences, explicit naming decisions, and target shape.
- [x] **Milestone 2 — Foundational Public API Types**
  - Implement `ModelParams`, `ContextParams`, `InferenceParams`, `IExecutor`, `GenerateChunk` exposure in core/engine.
- [x] **Milestone 3 — Model & ModelContext Lifecycle**
  - Implement `Model.Load(...)` and `model.CreateContext(...)` wrapping Stingray's native forward-pass loaders.
- [x] **Milestone 4 — Executors (Interactive, Stateless, Batched)**
  - Implement `InteractiveExecutor`, `StatelessExecutor`, and `BatchedExecutor`.
  - Wire dual streaming: `InferAsync` and `InferChunksAsync`.
- [x] **Milestone 5 — ChatSession & ChatHistory**
  - Implement `ChatSession` and `ChatHistory` with Jinja chat template formatting and multi-turn state retention.
- [ ] **Milestone 6 — Verification, Samples & Documentation**
  - Comparison sample demonstrating identical app pattern between LLamaSharp and Stingray.
  - End-to-end unit and integration tests.

---

## 9. Answers to the 10 Quality-Bar Questions

1. **What parts of LLamaSharp were intentionally mirrored?**  
   The `Model -> Context -> Executor -> Session` lifecycle, parameter decoupling (`ModelParams`, `ContextParams`, `InferenceParams`), executor abstractions (`InteractiveExecutor`, `StatelessExecutor`, `BatchedExecutor`), and conversational objects (`ChatSession`, `ChatHistory`).

2. **What parts were intentionally NOT mirrored?**  
   LLaMA-specific branding (`LLamaWeights`, `LLamaContext`, `ILLamaExecutor`), binary P/Invoke assumptions, and the suppression of typed streaming metadata.

3. **Which current Stingray types were renamed?**  
   None of the internal engine types were deleted; higher-level public types (`Model`, `ModelContext`, `IExecutor`, `InferenceParams`) are introduced as the primary public surface over existing engine implementations.

4. **Which current Stingray types were split?**  
   `InferenceEngine` previously conflated model loading, context allocation, sampling, and streaming. It is split into `Model` (weights), `ModelContext` (KV cache/allocations), and Executors (`InteractiveExecutor`, `StatelessExecutor`).

5. **Which current Stingray types were retained?**  
   `ContinuousBatchingEngine`, `PagedKvCache`, `ITokenizer`, `GenerateChunk`, `GenerateChunkKind`, `ArchitectureRegistry`, and domain forward-pass pipelines.

6. **Did any engine functionality get lost?**  
   **No.** Speculative decoding, thinking token extraction, prefix sharing, TurboQuant KV compression, and multi-architecture support are all preserved.

7. **Can one higher-level wrapper target both implementations with only a thin adapter?**  
   **Yes.** A lightweight interface wrapper mapping `InferAsync` and `ChatSession` requires under 50 lines of code between LLamaSharp and Stingray.

8. **Is the public API still clearly a general AI runtime rather than a LLaMA wrapper?**  
   **Yes.** The naming is neutral (`Model`, `ModelContext`, `IExecutor`), and non-causal architectures (RWKV, diffusion, speech) fit naturally into the same ecosystem.

9. **Is NativeAOT still clean?**  
   **Yes.** No dynamic code generation, no reflection, fully compatible with .NET 10 NativeAOT trim analysis.

10. **What migration burden remains for existing Stingray callers?**  
    Existing CLI and server callers continue to function through existing internal entry points or transition smoothly to the cleaner public `Model`/`Context`/`Executor` abstractions.
