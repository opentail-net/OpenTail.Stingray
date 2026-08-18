# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenTail.Stingray is a high-performance LLM inference engine and image generation pipeline in C# 14 / .NET 10. It reads GGUF model files and runs transformer inference on CPU (AVX2/AVX-512 SIMD), Vulkan compute shaders, and CUDA/cuBLAS. Architectures supported include `llama`/`llama4`, `qwen2`, `qwen3`, `qwen3moe`, `qwen35moe` (hybrid Gated-DeltaNet + attention + MoE), `gemma`/`gemma2`/`gemma3`/`gemma4`, `phi2`/`phi3`, and OLMoE. `deepseek2` (DeepSeek-V2/V3/R1's MLA attention) has a CPU-only implementation that loads and runs but produces numerically wrong output — not yet admitted as supported; see `docs/bugstofix.md`. It also supports text-to-image and text-to-video generation (Stable Diffusion 1.5, SDXL, SD 3/3.5, FLUX.1, Z-Image-Turbo, Qwen Image & Edit, Wan 2.1/2.2 Video, HunyuanVideo, and LTX-Video), 4× image upscaling via RRDBNet (Real-ESRGAN), Multimodal Vision understanding (Gemma 4 `gemma4uv`/`gemma4v`, Gemma 3 SigLIP, Llama 4), Native Text-to-Speech (TTS) with Voice Cloning (Kokoro-82M, Piper VITS, F5-TTS Flow-Matching DiT, Chatterbox-Turbo, and MeloTTS Multilingual VITS), and Native Dynamic Multi-LoRA Serving across concurrent sessions. Targets NativeAOT for single-binary deployment.

## Build & Test Commands

```bash
dotnet build                # Debug build
dotnet build -c Release     # Release (NativeAOT opts: IlcOptimizationPreference=Speed, IlcInstructionSet=native)
dotnet test                 # Run all tests (~3,200 tests across 14 test projects). Never add --nologo:
                            # Microsoft.Testing.Platform rejects it and reports "Zero tests ran".
                            # Tests.Sessions/Tests.ForwardPass/Tests.Vulkan (real model / real GPU
                            # device, serial) skip by default — set STINGRAY_RUN_HEAVY_TESTS=1 to
                            # actually run them. See "Test Projects" below.
dotnet test --filter "FullyQualifiedName~SomeTest"  # Run a single test
dotnet test tests/OpenTail.Stingray.Tests.ForwardPass  # Run one test project

# Run CLI inference (RunCommand is the implicit default command)
dotnet run --project src/OpenTail.Stingray.Cli -c Release -- \
  -m models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf -p "prompt" --temp 0

# GPU backend (all layers offloaded; -g/-1 = all layers, --backend cuda|vulkan|auto)
dotnet run --project src/OpenTail.Stingray.Cli -c Release -- \
  -m models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf -p "prompt" --temp 0 -g -1

# Inspect a GGUF file
dotnet run --project src/OpenTail.Stingray.Cli -c Release -- list-metadata -m model.gguf
dotnet run --project src/OpenTail.Stingray.Cli -c Release -- list-tensors  -m model.gguf

# Perplexity over a corpus (accuracy gate for KV compression, issue #180). Supports
# --tq/--tq-mode exactly like the run command (auto = KVarN where supported, else Lloyd-Max).
dotnet run --project src/OpenTail.Stingray.Cli -c Release -- \
  perplexity -m model.gguf -f corpus.txt -c 2048 --tq

# Whole-turn structured output (grammar-constrained decoding, issues #423/#425). Mirrors
# llama.cpp's -j/--json-schema; the entire response is constrained to the schema. The root
# must be an object schema with at least one property; --json-schema-ordered emits keys in
# declared order. Server exposes the same via OpenAI/Anthropic response_format:json_schema.
dotnet run --project src/OpenTail.Stingray.Cli -c Release -- \
  -m model.gguf --temp 0 -p "Extract name and age from: Alice is 30." \
  -j '{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"]}'

# VibeThinker-1.5B (Qwen2-based math/reasoning, issue #282). Loads as a standard
# qwen2 GGUF (QKV bias but no output-projection bias, no QK-norm, 28 layers / 2 KV
# heads, ChatML, tied embeddings). `download-model.ps1 -Model vibethinker` fetches the
# default Q8_0 (near-lossless); `-Model vibethinker-q4` is the smaller quant. Recommended
# sampling: temp 0.6, top_p 0.95, top_k 0, and no system prompt (the chat template supplies
# the math one). Emits a long <think> chain-of-thought then a \boxed{} answer.
dotnet run --project src/OpenTail.Stingray.Cli -c Release -- \
  -m models/VibeThinker-1.5B.Q8_0.gguf -g -1 \
  --temp 0.6 --top-p 0.95 --top-k 0 \
  -p "If 5x + 3 = 2x + 18, what is x? Show your reasoning."

# Multimodal vision (Gemma 4, Gemma 3, Llama 4): pass one or more images with --image
# and --mmproj. UnifiedVisionPipeline auto-detects the projector type.
dotnet run --project src/OpenTail.Stingray.Cli -c Release -- \
  -m models/gemma-4-E4B-it.gguf --mmproj models/gemma-4-E4B-it-mmproj.gguf -g 0 \
  --image photo.png -p "Describe <image>"

# Start API server (OpenAI + Anthropic compatible). OpenTail.Stingray.Server is the
# ASP.NET Core library that ships AddOpenTailStingray() / MapOpenTailStingray();
# OpenTail.Stingray.Server.Host is the runnable demo host you'd publish.
STINGRAY_MODEL=models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf \
  dotnet run --project src/OpenTail.Stingray.Server.Host -c Release

# NativeAOT publish (the three packable frontends + libraries)
dotnet publish src/OpenTail.Stingray.Cli -c Release -r win-x64
dotnet publish src/OpenTail.Stingray.Server.Host -c Release -r win-x64

# Benchmarks
dotnet run --project benchmarks/OpenTail.Stingray.Bench -c Release -- --filter '*'

# Models: scripts/download-model.ps1 fetches known presets (smollm2, vibethinker,
# qwen3-8b, olmoe-1b-7b, qwen3-coder-30b-a3b, qwen36-35b-a3b[-mtp], ornith-9b/-35b,
# gemma4-12b-qat, gemma4-e4b-qat, llama4-scout, z-image-turbo[-q8], realesrgan-x4, ...).
# Run with `-Model <name>` (PowerShell). See the script header for the full ValidateSet.

# Ornith-1.0 (DeepReinforce, MIT) — agentic-coding "self-scaffolding" RL finetunes of
# Qwen3.5 / Gemma 4 bases, NOT a new architecture. Self-scaffolding is a training-time
# technique; at inference they're ordinary transformers. GGUF arches reduce to ones
# already dispatched: 9B = `qwen35`, 35B/397B = `qwen35moe`. Validated end-to-end
# (issue #411): the bartowski 9B Q4_K_M GGUF actually ships GDN tensors, so
# `_opentailllm.is_hybrid_ssm` auto-activates and it takes the SAME hybrid Gated-DeltaNet +
# attention path as the 35B/397B MoE variants (24 GDN + 8 full-attention layers,
# full_attention_interval=4) — not a plain dense transformer as the arch name alone
# suggests. Full CUDA offload (-g -1) fits comfortably in 8 GB VRAM (~3 GB weights
# uploaded; GDN state + dense FFN run on CPU by design of CudaHybridGdnForwardPass).
# Chat template loads via JinjaChatTemplate and tool calls parse via the qwen35moe-style
# QwenToolCallAdapter (Qwen3.6 XML `<function=..><parameter=..>` inside `<tool_call>`).
# They're tagged image-text-to-text, but the Qwen3.5 vision projector is unimplemented —
# the text GGUF path is text-only, which is what the coding use case needs.
# `download-model.ps1 -Model ornith-9b` (Q4_K_M, 5.5 GB).
dotnet run --project src/OpenTail.Stingray.Cli -c Release -- \
  -m models/deepreinforce-ai_Ornith-1.0-9B-Q4_K_M.gguf -g -1 \
  --temp 0.6 --top-p 0.95 --top-k 20 -p "Write a Python LRU cache."

# Image generation with upscaling (Z-Image-Turbo + RRDBNet). ImageCommand auto-detects
# Z-Image vs FLUX from the model. Z-Image uses a Qwen3-4B text encoder; FLUX uses CLIP-L + T5.
dotnet run --project src/OpenTail.Stingray.Cli -c Release -- image \
  -m models/z_image_turbo-Q5_K_M.gguf \
  --vae models/z-image-turbo/vae \
  --qwen-encoder models/Z-Image-AbliteratedV1.Q5_K_M.gguf \
  --qwen-tokenizer models/z-image-turbo/tokenizer/tokenizer.json \
  --upscaler models/RealESRGAN_x4plus.safetensors \
  --upscale-blend 0.8 \
  -p "a serene mountain lake at sunrise" -W 512 -H 512 --steps 4 -o out.png

# Image generation micro-benchmarks
dotnet run --project benchmarks/OpenTail.Stingray.ImageBench -c Release -- --bench --filter '*'
```

## Architecture

The solution (`OpenTail.Stingray.slnx`) is a four-layer stack, bottom-up:

1. **Core** (`OpenTail.Stingray.Core`) — GGUF parser (memory-mapped), BPE/SPM tokenizer (`Microsoft.ML.Tokenizers`), Jinja chat templates (`JinjaChatTemplate`), tool-call adapter, UTF-8 stream decoder, tensor types, model graph, and grammar-constrained decoding (`Grammar/`: `ITokenConstraint` + `JsonSchemaOutputConstraint` for whole-turn JSON-schema structured output, `JsonToolArgumentConstraint` and the per-family tool-argument constraints — Qwen/Gemma — plus `ToolSchemaCompiler`, issues #423/#425). Everything depends on this. Defines the central interfaces (`IComputeBackend`, `IImageOpsBackend`, `IForwardPass`, `ITokenizer`, `ITokenConstraint`).
2. **Compute Backends** — Three implementations of `IComputeBackend`; `CudaBackend` and `VulkanBackend` also implement `IImageOpsBackend` for convolutional image ops:
   - `OpenTail.Stingray.Cpu` — AVX2/AVX-512 SIMD kernels (`SimdKernels`), Q4_K_M/Q6_K/Q8 dequantization (`Dequantize`), Gated-DeltaNet kernels (`GdnKernels`), optional OpenBLAS GEMM (`BlasInterop`)
   - `OpenTail.Stingray.Vulkan` — Vulkan compute via `Vortice.Vulkan`, SPIR-V shaders, GPU buffer pool
   - `OpenTail.Stingray.Cuda` — cuBLAS GEMM + NVRTC runtime-compiled kernels. `CudaTextKernels` (RMSNorm/RoPE/softmax/GQA attention/Q4_K-Q6_K-F32 matvecs/KV-append), `CudaKernels` (im2col + conv for DiT/RRDBNet), `CudaWsKernels` (weight-stationary batched-decode matvecs, issue #194), `CudaRaggedKernels` (ragged batched decode for SnapKV-evicted caches), plus `GpuBufferPool` to eliminate per-GEMM `cudaMalloc`/`cudaFree` overhead
3. **Engine** (`OpenTail.Stingray.Engine`) — Forward-pass orchestration, KV cache, sampling, speculative decoding, MoE expert offloading, continuous batching. Depends on Core + backends.
4. **Frontends** —
   - **CLI** (`OpenTail.Stingray.Cli`, `Spectre.Console.Cli`, llama.cpp-compatible flags): `RunCommand` (default text/vision inference; also whole-turn JSON-schema structured output via `-j`/`--json-schema`/`--json-schema-file`), `ImageCommand` (`image` subcommand), `PerplexityCommand` (`perplexity` — accuracy gate over a corpus, honours `--tq`/`--tq-mode`), `ListMetadataCommand` (`list-metadata`), `ListTensorsCommand` (`list-tensors`).
   - **API Server**: `OpenTail.Stingray.Server` is an ASP.NET Core class library exposing `AddOpenTailStingray()` / `MapOpenTailStingray()` with the `OpenTailStingrayServerOptions` options pattern (OpenAI `/v1/chat/completions` + `/v1/models`, Anthropic `/v1/messages`, OpenAI Responses, `/health`, `/metrics`). `OpenTail.Stingray.Server.Host` is the runnable demo host (one `Program.cs`, AOT-published) that consumes it.

Supporting libraries:
- **OpenTail.Stingray.Diffusion** — Native image-generation pipelines. `ZImagePipeline` (Z-Image-Turbo: `ZImageDiT` single-stream S3-DiT + Qwen3-4B encoder + FLUX VAE) and `ImagePipeline` (`FluxDiT` multi-stream MMDiT + CLIP-L/T5 encoders). Includes `VaeDecoder`, `RRDBNet` (Real-ESRGAN 4× upscaler), `EulerFlowScheduler`, 2D RoPE, FP8 conversion, and Safetensors/GGUF weight loaders. Text encoders live in `TextEncoders/`.
- **OpenTail.Stingray.Vision** — Unified Multimodal Vision Pipeline (`UnifiedVisionPipeline` / `IVisionEmbedder`). Auto-detects and runs Gemma 4 encoder-free (`gemma4uv`), Gemma 4 ViT (`gemma4v`), Gemma 3 SigLIP (`gemma3`), and Llama 4 (`llama4`) projectors, with model-matched image preprocessing and soft-token embedding.
- **OpenTail.Stingray.TurboQuant** — KV cache compression. Two codecs: KVarN (Hadamard + dual-axis Sinkhorn variance normalization + asymmetric RTN, 4-bit K / 2-bit V, 128-token tiles — issue #180) and Lloyd-Max codebooks (3-4 bit; severely degrades quality on QK-norm models such as Qwen3, issue #432). `--tq-mode` defaults to `auto`: KVarN where supported, else Lloyd-Max fallback with a quality warning (#436). Lloyd-Max remains the fallback for Vulkan / partial-offload / MoE-on-GPU / SnapKV. KVarN runs on CPU (AVX2 fused read kernels) and the CUDA decode path (CUDA-graph decode + chunked prefill). Codebook data lives in `codebooks/`.
- **OpenTail.Stingray.Pipeline** — 3-tier memory hierarchy (VRAM → pinned RAM → NVMe), SLRU expert cache, async prefetcher.
- **OpenTail.Stingray.Sessions** — Transactional, revisioned hot-session orchestration over inference state. Design notes in `docs/reference/adr-0001-session-cache-lifecycle.md` and `docs/session-native-inference-runtime-plan.md`; restart-continuation is still experimental internals rather than a supported product feature (see `CHANGELOG.md`).

## Key Interfaces & Patterns

- `IComputeBackend` (in Core) is the central abstraction — defines MatMul, RmsNorm, RoPE, Softmax, SiLU, Attention, and memory management. CPU, Vulkan, and CUDA backends implement it.
- `IImageOpsBackend` (in Core) — extends `IComputeBackend` with convolutional image ops (Conv2d, LeakyRelu, CatChannels, PixelShuffle, Upsample2x). Implemented by `CudaBackend` and `VulkanBackend` for the RRDBNet upscaler and VAE.
- `IForwardPass` (in Core) — per-token forward pass. Implementations in Engine: `ForwardPass` (CPU dense), `GpuForwardPass` (Vulkan), `CudaForwardPass` (CUDA dense), `HybridForwardPass`/`CudaHybridForwardPass` (dense + MoE expert offload), `HybridGdnForwardPass`/`CudaHybridGdnForwardPass` (qwen35moe hybrid Gated-DeltaNet + MoE). Has `Forward`, `Prefill`, `TruncateTo`, `ResetCache`, `VocabSize`, `MaxSeqLen`.
- `IBatchedForwardPass` (in Engine) — multi-token batched prefill/decode used by continuous batching.
- `PagedKvCache` (in Engine) — lazily allocated paged KV cache used by `ForwardPass`. Pages (16 positions) allocated on first write; `TruncateTo` is a soft operation (enables prefix reuse); `Reset` returns pages to a warm pool. Other cache types: `KvCache` (simple), `CudaSequenceKvCache` (per-sequence GPU), `TurboQuantKvCache` (KVarN 4/2-bit or Lloyd-Max 3-4 bit compressed). `IMultiSlotKvCache` abstracts per-sequence/multi-slot caches. `SnapKvSelector` does prefill-time SnapKV eviction; `GdnStateCache` snapshots Gated-DeltaNet state for MTP rollback.
- `IInferenceEngine` (in Engine) — top-level generation interface used by the server: `GenerateAsync(prompt, sp, ct) → IAsyncEnumerable<string>`. Implemented by `InferenceEngine` (single-user, prefix caching) and `ContinuousBatchingEngine` (multi-user batching, activated via `STINGRAY_MAX_BATCH`).
- `ForwardPass.BatchForwardMulti(tokens[], positions[], caches[])` — batched multi-sequence decode; amortizes weight reads N× across concurrent users. Each sequence has its own `PagedKvCache`. Not supported for MoE or TurboQuant.
- `ForwardPass.PrefillWithCache(tokens, cache, startPos)` — prefills a per-sequence cache (used by `ContinuousBatchingEngine` during request admission). Admission is chunked (`STINGRAY_PREFILL_CHUNK`, default 256 tokens) and interleaved with decode steps; multiple in-flight prompts prefill as one packed pass via `ForwardPass.PrefillPackedMulti` and admission is gated by a KV token budget (`STINGRAY_KV_BUDGET_MB`) — issue #183.
- **Speculative decoding** — `SpeculativeDecoder` (general draft-model speculation), `MtpDecoder` + `MtpBatchTail` (self-speculative Multi-Token Prediction / NEXTN heads, e.g. Qwen3.6-27B-MTP, with folded k-token batched verify, issue #207), `PromptLookupDraft` (prompt-lookup draft), and `DSparkDecoder` + `DSparkDraftModel`/`CudaDSparkDraftModel` (DeepSeek DSpark block-parallel safetensors draft heads, docs/dspark-plan.md / PR #413: EAGLE-3-style backbone conditioned on target hidden-state taps via `IForwardPass.EnableHiddenTaps` — CPU and dense-CUDA targets both capture; rank-256 Markov re-bias + confidence-trimmed verify on the host (`DSparkHostHeads`); greedy only — `--dspark-model <safetensors-or-dir> --temp 0` with `-g 0` or `-g -1`, placement via `DSparkPlacementPlanner` / `--dspark-place` / `STINGRAY_DSPARK_*` (GPU draft needs a CUDA target and free VRAM — pass `-c` to bound the target's KV solve); fetch heads with `download-model.ps1 -Model dspark-qwen3-4b`; server: `STINGRAY_DSPARK_MODEL` on the single-user engine (`MaxBatchSize` 1), engaging on greedy `enable_thinking:false` requests. On a 4B target the un-graphed verify pass caps DSpark below plain graph-replayed decode — see the plan doc's Phase-4 numbers before benchmarking). Toggle from the CLI (`--mtp`, `--draft-model`) or server (`SpecType`).
- `Sampler` (in Engine) — temperature, top-k, top-p (nucleus), min-p, repetition penalty, logit bias, and grammar-constrained decoding (applies an `ITokenConstraint` token mask per step — used for tool-argument grammars and whole-turn JSON-schema structured output).
- MoE expert offload: `ExpertSlotManager`/`CudaExpertSlotManager` (SLRU VRAM expert cache), `MoEPrefetcher` (async SSD→RAM→VRAM), `TierPlanner` + `HardwareProfile` (three-tier placement), `MmapPrefault`, `WarmPinConfig`. `--cpu-moe` / `STINGRAY_CPU_MOE` keeps routed experts on the CPU (issues #80/#93).
- Hot paths use `NativeMemory`, `Span<T>`, and GPU buffers — no managed heap allocations.
- Unsafe code is used throughout for performance. `AllowUnsafeBlocks` is enabled globally.

## Build Constraints

Shared settings live in `Directory.Build.props` (net10.0, LangVersion 14, Nullable enable, ImplicitUsings):

- **TreatWarningsAsErrors** is enabled globally — all warnings must be resolved.
- **Trim and AOT analyzers** are enabled (`IsTrimmable`, `EnableTrimAnalyzer`, `EnableAotAnalyzer`, warnings not suppressed) — code must be NativeAOT-compatible (no reflection-heavy patterns, no dynamic code generation). Server JSON uses a source-generated `OpenTailStingrayJsonContext`.
- **InvariantGlobalization** is on — no culture-specific string operations.
- Vulkan shaders are GLSL `const string`s in `src/OpenTail.Stingray.Vulkan/Shaders.cs`, precompiled to SPIR-V committed in `Shaders.Precompiled.g.cs` (keyed by an FNV-1a `ShaderCompiler.StableHash`) so the NativeAOT binary needs no glslc at runtime; `ShaderCompiler.Compile` falls back to glslc only on a table miss. After adding/editing/removing a shader const, regenerate the table with `scripts/gen-spirv.ps1` (runs `tools/SpirvGen`, needs the Vulkan SDK) — `VulkanPrecompiledShaderTests` fails on drift. Shaders needing extensions the bundled glslc lacks (`SgemmBf16`, `SgemmFp8`) are recorded in `SkippedShaders` and fall back at runtime by design.
- Versioning is a plain `<Version>` in `Directory.Build.props` — no derived and no pre-release versions. To release: bump it, commit, tag `stingray-v<Version>` (that prefix, not a bare `v` — the pre-rename `v1.0.0`/`v1.0.1` tags belong to the old OpenTail.LLM series). Publication is tag-triggered only, and CI fails the run if the tag and `<Version>` disagree. Only the `OpenTail.Stingray` meta-package, `OpenTail.Stingray.Server`, and `OpenTail.Stingray.Cli` are packable.

## Test Projects

~3,200 tests across 14 projects (xUnit v3, `[Fact]`/`[Theory]`):

| Test Project | Covers |
|---|---|
| Tests.Core | GGUF parsing, tokenizer (SPM/BPE), Jinja chat templates, model graph, tool-call adapter, grammar constraints / JSON-schema structured output, UTF-8 stream decode |
| Tests.ForwardPass.Fast | Forward pass, KV cache, sampler, dequant/SIMD kernels, quantization parity — everything in the largest suite that needs no real model or GPU device. Runs in parallel. |
| Tests.ForwardPass | The real-model subset of the above (bit-exactness/parity acceptance tests) plus the perf-gauge benchmarks (slow by design — JIT warmup + timed loops, would corrupt their own measurements if parallelized). Real model, serial. |
| Tests.Pipeline | Memory hierarchy, image pipeline integration |
| Tests.TurboQuant | KV cache compression (codebooks, encode/decode parity) |
| Tests.Server.Fast | API endpoints (OpenAI/Anthropic compatibility) against a fake inference engine — no real model. Runs in parallel. |
| Tests.Server | The one real-model test in the Server surface (session restart/persistence across two host instances). Real model, serial. |
| Tests.Sessions.Fast | Hot-session orchestration (`OpenTail.Stingray.Sessions`) against fakes — no real model. Runs in parallel. |
| Tests.Sessions | The real-model subset of session orchestration (golden-replay/greedy-parity acceptance tests). Real model, serial. |
| Tests.Cli | GPU device queries, CLI flags (e.g. `--cpu-moe`) |
| Tests.Vision | Unified vision pipeline, Gemma 3/4 and Llama 4 ViT encoders, image I/O, mmproj loading |
| Tests.Diffusion | Diffusion pipeline acceptance and conformance tests (SD 1.5, SDXL, SD 3, Qwen Image, Wan, Hunyuan, LTX-Video) |
| Tests.Audio | Kokoro-82M Text-to-Speech unit tests (G2P, PLBERT, AdaIN, iSTFT, WavWriter) |
| Tests.Cuda | CUDA backend correctness. Silent-skips fast on a machine with no CUDA card (see below). |
| Tests.Vulkan.Fast | The handful of Vulkan-adjacent tests needing no real GPU device (config/predicate logic, shader-constant consistency). Runs in parallel. |
| Tests.Vulkan | Real Vulkan device tests (parity, E2E, shader dispatch). Real GPU, serial. |

**Fast/heavy split.** `Tests.Sessions`, `Tests.ForwardPass`, `Tests.Vulkan`, and `Tests.Server`
each split into a `.Fast` sibling (no real model or GPU device, default xUnit parallelism — safe
to run on every save) and the plain-named project (real model and/or real GPU device, forced
serial via `xunit.runner.json`, since concurrent real-model/real-device work is what caused a
59.5 GB test-suite memory blowup this split exists to prevent). The plain-named heavy projects
additionally **skip every test by default** — every test class derives from a per-project
`HeavyTestBase` whose constructor does `Assert.SkipUnless(...)` on `STINGRAY_RUN_HEAVY_TESTS`.
This is deliberate, not a hidden gap: a full `dotnet test` run stays fast for everyday iteration,
and the skip shows up honestly in the results (not silently absent) so it's visible when it
matters. Set `STINGRAY_RUN_HEAVY_TESTS=1` to actually run the heavy suites — before a commit that
touches model-loading/GPU-dispatch code, or in CI.

`Tests.Cuda`/`Tests.Vulkan` (the GPU-hardware projects) additionally gate on hardware
availability regardless of `STINGRAY_RUN_HEAVY_TESTS`: `Assert.SkipUnless(CudaTestGpu.IsAvailable, ...)`
/ the `VulkanBackend` try/skip pattern means a run on a machine without that card still completes
in well under a second instead of waiting out ~400 individual timeouts.

Shared test data lives in `tests/fixtures/`.

**Test runner.** xunit v3 runs on Microsoft.Testing.Platform, selected by the repo-root
`global.json`. Two consequences that cost real time if forgotten:

- Never pass `--nologo` to `dotnet test`. MTP rejects it and reports "Zero tests ran" with exit 5,
  which reads exactly like a discovery failure.
- If `global.json` goes missing, `dotnet test` falls back to VSTest, finds no adapter (there is no
  `xunit.runner.visualstudio`), and **exits 0 having run nothing** — a silent green.

Filtering uses `--filter-class` / `--filter-method`, not `--filter`. TRX receipts come from the
`Microsoft.Testing.Extensions.TrxReport` extension (pinned at 1.9.1 to match the MTP version
xunit.v3 3.2.2 binds to): `dotnet test <proj> -- --report-trx --report-trx-filename x.trx`.

**The flag spellings differ by entry point, and mixing them up looks like a discovery failure.**
Through `dotnet test <proj> -- …` the double-dash forms above are correct. Running the built test
executable directly — `tests/<Proj>/bin/Release/net10.0/<Proj>.exe`, which is much faster when
iterating because it skips build/restore evaluation — you get xunit's own parser, which wants
**single-dash** `-class` / `-method` (and `-class-` / `-method-` to exclude), and rejects
`--filter-class` with `error: unknown option`. Same for `--minimum-expected-tests` (used by CI to
make thin discovery fail): accepted by `dotnet test`, rejected by the exe.

## Samples & Scripts

- `samples/OpenTail.Stingray.Sample.Chat` — minimal streaming chat using the library directly.
- `samples/OpenTail.Stingray.Sample.ToolCall` — tool/function-calling flow.
- `samples/OpenTail.Stingray.Sample.HotRouting` — hot-session routing over `OpenTail.Stingray.Sessions`.
- `benchmarks/` — `OpenTail.Stingray.Bench` (text-inference BenchmarkDotNet suite), `OpenTail.Stingray.ImageBench` (image-generation micro-benchmarks), and `SnapKvEval` (SnapKV eviction quality/accuracy evaluation harness).
- `scripts/` — PowerShell benchmark drivers (`bench-*.ps1`), `download-model.ps1` (model fetcher), `setup-openblas.ps1` / `setup-llamacpp.ps1`, and Python reference-generation helpers for llama.cpp cross-checking (`gemma4uv_ref.py`, `extract_reference.py`, `compare_tokens.py`).

## Design Documentation

Detailed architecture doc at `docs/reference/OpenTail.Stingray-Design.md` covering all subsystems, algorithms, data layouts, and the (mostly completed) implementation phases. `docs/research/` and the various `docs/*-plan.md` files hold per-feature design notes (Gemma 4, qwen35moe, MoE offloading, KV-compression feasibility).





