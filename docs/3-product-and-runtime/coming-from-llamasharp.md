# Coming from LLamaSharp

Stingray exposes a **LLamaSharp-shaped application API** (`Model` -> `ModelContext` -> executor -> `ChatSession`) so
higher-level wrappers can move between the two with a thin adapter. It is **not** binary- or source-compatible with
LLamaSharp and does not claim to be. Namespace: `OpenTail.Stingray.Engine` (types under `src/OpenTail.Stingray.Engine/Api`).
Design rationale and upstream feature comparison: [2026-10-06-llamasharp-api-realignment-plan.md](2026-10-06-llamasharp-api-realignment-plan.md).

| LLamaSharp | Stingray | Notes / deliberate differences |
| :--- | :--- | :--- |
| `LLamaWeights.LoadFromFile(ModelParams)` | `Model.Load(path)` / `Model.Load(ModelParams)` | Neutral name: the runtime is not LLaMA-only (Qwen, Gemma, gpt-oss, DeepSeek, RWKV, ...). Architecture is picked from the GGUF via the architecture registry, not by the caller. |
| `ModelParams` (path + context fields) | `ModelParams` (weights) + `ContextParams` (context) | Split: model path/backend/GPU layers vs context size/batch/KV type/threads. |
| `weights.CreateContext(params)` -> `LLamaContext` | `model.CreateContext(contextParams)` -> `ModelContext` | Several contexts over one `Model`; disposing the model disposes its live contexts. Context exposes `Tokenizer` (`ITokenizer`), `ContextSize`, `Reset()`. |
| `ILLamaExecutor` | `IExecutor` | `InferAsync(string, ...)` mirrors LLamaSharp (text only). `InferChunksAsync(...)` is Stingray-only and yields typed `GenerateChunk`s (Text, Thinking, Usage, Stop). |
| `InteractiveExecutor` | `InteractiveExecutor` | Stateful; KV continuity across calls. |
| `StatelessExecutor` | `StatelessExecutor` | Resets the context before each call; no state leaks between calls. |
| `BatchedExecutor` | `BatchedExecutor` | Backed by `ContinuousBatchingEngine`; not a low-level per-token batch API. |
| `InferenceParams` | `InferenceParams` | Maps onto the engine's `SamplingParams` (`ToSamplingParams()`); adds reasoning/thinking options. Sampling pipeline objects (`ISamplingPipeline`) are not mirrored. |
| `ChatHistory` / `ChatSession` | `ChatHistory` / `ChatSession` | Prompt rendered through the GGUF's Jinja chat template (override via `PromptFormatter`). `ChatAsync` returns user-facing text only; `ChatChunksAsync` keeps thinking/usage/stop. |
| `LLamaTokenizer`-style helpers | `context.Tokenizer` | Existing managed BPE/SentencePiece tokenizers. |

## Intentionally not mirrored

- Native handles/`SafeLLamaHandle`s, `NativeApi`, llama.cpp backend selection: Stingray has no llama.cpp/ggml dependency.
- LLaMA-specific type names and LoRA/state-save APIs not yet needed by Stingray applications.
- `Stingray` extras with no LLamaSharp equivalent: typed chunk stream, speculative decoding, KV prefix sharing/forking,
  diffusion/speech/vision pipelines (these stay on their own typed APIs).

## Old public paths

`InferenceEngine`, `ContinuousBatchingEngine`, `IInferenceEngine` and `SamplingParams` remain public. They are the engine
the new layer is built on (`ModelContext.Engine`), used by the server, CLI and sessions, and are the extension point
for custom engines (`ModelContext.SetEngine`). They are not a second parallel implementation, so they were kept rather than
removed.
