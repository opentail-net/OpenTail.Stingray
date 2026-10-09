# Guides: how to do each thing

Short, task-by-task how-tos. Each one says what you get, what you need, the command that works,
the options that matter, and the limits. For the overview of everything Stingray can do, start with
[WHAT-YOU-CAN-DO.md](../WHAT-YOU-CAN-DO.md). When something goes wrong,
[TROUBLESHOOTING.md](../TROUBLESHOOTING.md).

| I want to… | Guide |
|---|---|
| Chat with a language model | [chat.md](chat.md) |
| Ask about a picture or read a document | [vision.md](vision.md) |
| Get JSON out of a model, or let it call my functions | [structured-output-and-tools.md](structured-output-and-tools.md) |
| Turn text into speech, or transcribe audio | [speech.md](speech.md) |
| Search by meaning (embeddings and reranking) | [search-embeddings-rerank.md](search-embeddings-rerank.md) |
| Generate images or short video | [images-and-video.md](images-and-video.md) |
| Serve all of this to OpenAI, Anthropic or Cohere clients | [serve-an-api.md](serve-an-api.md) |

## Technical Deep Dives

In-depth technical guides for engine internals, memory optimization, and deployment:

| System / Subsystem | Guide |
|---|---|
| KV cache compression, TurboQuant & prefix reuse | [turboquant-kv-cache.md](turboquant-kv-cache.md) |
| Pure C# vectorization, SIMD intrinsics & thread pools | [managed-simd-kernels.md](managed-simd-kernels.md) |
| Compute backends (CPU, Vulkan shaders, CUDA) | [hardware-backends.md](hardware-backends.md) |
| Publishing single-file zero-dependency NativeAOT binaries | [native-aot-deployment.md](native-aot-deployment.md) |

---

Picking a model: [MODELS.md](../MODELS.md). The exact command, memory and speed for a particular
model: [RUNNING.md](../RUNNING.md). How well it was checked: [STATUS.md](../STATUS.md). Architecture
deep dives: [models/](../models/README.md).

