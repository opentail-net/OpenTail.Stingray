# TensorSharp parity matrix (Phase 0 of the selective-port plan)

Written 2026-10-03 for [examples/tensorsharp-parity-selective-port-plan.md](../../examples/tensorsharp-parity-selective-port-plan.md).
This is the anti-scope-creep fence: every TensorSharp capability gets one row, a Stingray status, and a
decision. It is internal; do not describe Stingray externally as "TensorSharp parity".

**Important correction to the plan.** The plan was written assuming most of these capabilities were absent.
A code and STATUS check on 2026-10-03 found several already exist. Status column legend:

- **verified**: STATUS.md carries a sourced, checked row for it.
- **exists, unchecked**: found in source by search; behaviour not run or reviewed in this pass.
- **absent**: nothing found in source (searched `src/**/*.cs`, excluding `bin`/`obj`).

| Capability | TensorSharp | Stingray today | Port? | Reason / remaining gap |
| :--- | :--- | :--- | :--- | :--- |
| Text embeddings, encoder models | GGUF BERT / XLM-R (Arctic Embed L v2, MiniLM) | **verified** (STATUS, 2026-09-25): native `TransformerEncoder` from HF safetensors + `tokenizer.json`. **GGUF `bert` encoders added and checked 2026-10-03** (MiniLM Q8_0, Arctic Embed L v2 Q8_0, vs llama.cpp; see [validation/embeddings.md](../validation/embeddings.md)) | DONE for `bert`-arch GGUF | Open: GGUF `nomic-bert`, GPU, load-time F32 dequantisation (4x RAM), HTTP-level tests against a real encoder |
| Cross-encoder rerank | Yes | **verified** (STATUS): ms-marco-MiniLM, bge-reranker-v2-m3; `/v1/rerank` | NO | Already done |
| `/v1/embeddings` | Yes | **fixed + tested 2026-10-03**: `input` now accepts a string or a string array (the array form used to fail as a 400 "Invalid JSON"); empty/non-string input is rejected, not dropped. `EmbeddingEndpointTests` (no model needed). Numerical parity: `Tests.Embeddings` | DONE (wire format) | Real-encoder HTTP run (batch, base64, dimensions) done 2026-10-03 on MiniLM; Arctic over HTTP not yet |
| `/api/embed` (Ollama) | Yes | **added 2026-10-03**, same handler as `/v1/embeddings`, Ollama response shape `{model, embeddings, total_duration, prompt_eval_count}` and `{"error": "..."}` errors; tested for validation and 404 | DONE (wire format) | `truncate`, `options`, `keep_alive` are ignored |
| Resident embedding encoder | Yes | **exists, unchecked**: `EncoderPipelineFactory.GetSharedEmbedding` keeps a shared engine | VERIFY | Concurrency, bounded batch, cancellation, unload not evidenced |
| Sparse expert streaming / larger-than-VRAM MoE | Yes | **exists, unchecked**: `ExpertSlotManager` (Vulkan) and `CudaExpertSlotManager`, SLRU caches; `TierPlanner`; CPU `--moe-warmpin` | VERIFY first | Slot managers are VRAM-limited with mmap'd host weights. Not shown: larger-than-*RAM* paging, hot-set adaptation over time (your "experts get warmer" idea), or any real-model measurement |
| Larger-than-RAM models | Yes | **absent as a feature**; models are mmap'd, so the OS pages, but nothing plans or measures it | YES, via the MoE work | Needed for GLM-5.3 (236 GiB) / DeepSeek verification on this 64 GB host |
| Multi-GPU layer split | Yes | **exists, unchecked**: `SmartOffloadPlanner` (multi-GPU plan), `MultiDevicePipeline`, `VulkanLayerSplitForwardPass` (GPU + CPU split) | VERIFY | No CUDA multi-device code found (`CudaMulti`/NCCL/peer-access searches empty); no multi-GPU hardware here to test |
| MoE tensor parallelism | Experimental | absent | LATER | Complex, not core |
| IQ2/IQ3 kernels | CUDA | CPU + Vulkan: IQ1_M/S, IQ2_S/XS/XXS, IQ3_S/XXS, IQ4_NL/XS appear in `Tensor.cs`/`Dequantize`/`SimdKernels`/Vulkan shaders. **CUDA: none found** | YES (CUDA) | Coverage claim in plan is wrong for CPU/Vulkan; CUDA IQ kernels are the real gap and cannot be tested without an NVIDIA GPU |
| Model auto-download | Yes | **exists**: `pull`, `ModelCatalog`, `ModelDownloader` | FINISH | |
| SHA-pinned model files | Yes | **done 2026-10-03**: every `CatalogFile` carries a 40-hex commit `Revision` (URL is `resolve/<commit>/`); the SHA-256 and size of all 4 catalog files were re-checked against what HF serves at that commit. Tests: stale revision, offline mode (`STINGRAY_OFFLINE`/`HF_HUB_OFFLINE`), explicit local file, hash mismatch, resume, multi-file | DONE | `stingray pull -r` still follows `main` (it is for arbitrary repos); the embedding models are not catalog entries yet (needs an `embed` task) |
| Multi-file model dependencies (mmproj, tokenizer) | Yes | **unchecked** | VERIFY | |
| Hardware eligibility in `stingray models` | Catalog columns | **unchecked** (`SmartOffloadPlanner` has the inputs) | LATER | Steam-style greyed-out idea |
| Structured capability metadata | Yes | **exists, unchecked**: `GET /capabilities` in `CompatibilityEndpoints` | VERIFY | Compare schema with plan §6.1 |
| LoRA | Yes | **exists, unchecked**: `Core/Lora` (`LoraAdapter`, `LoraLayer`, `LoraRegistry`), used by `ForwardPass` | FINISH/VERIFY | Coverage-driven |
| Sub-agents | Yes | no | **NO** | Product/orchestration scope |
| Browser / Playwright skill | Yes | no | **NO** | Application scope |
| TensorAgent | Yes | no | **NO** | Application scope |
| iOS / macOS / Windows app | Yes | no | **NO** | Different product |
| Image editor | Yes | no | **NO** | Different product |
| MP4 audio muxing | Yes | no | **NO** | Application feature |
| Desktop web UI | Yes | no | LATER / OPTIONAL | After runtime coverage |

## What this does to the plan's order

1. **Phase 1 (Embeddings) shrinks** from "build a primitive" to: GGUF BERT/XLM-R encoder loading, an `/api/embed`
   adapter, the missing runtime tests (concurrency, bounded batch, cancellation, unload), and
   `docs/validation/embeddings.md`.
2. **Phase 2 (manifests)** is a finish job: add the revision pin and dependency lists to the existing catalog.
3. **Phase 3 (larger-than-memory MoE)** starts with measurement of the existing slot managers on a real MoE model, not
   a new design. It is also the same machinery needed to verify GLM-5.3 on this host.
4. **Phase 4 (multi-GPU)** cannot be verified on this machine (integrated GPU only); planner-level unit tests only.
5. **Phase 5 (quantisation)**: the gap is CUDA IQ kernels, which cannot be run here either.

Everything marked "exists, unchecked" needs a first look before any port work is justified.
