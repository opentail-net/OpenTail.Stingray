# CLAUDE.md

Guidance for AI coding agents working on OpenTail.Stingray.

## Project Overview

OpenTail.Stingray is a high-performance LLM inference engine, image/video diffusion pipeline, multimodal vision understanding runtime, and native neural speech synthesis & recognition (TTS/ASR) engine in C# 14 / .NET 10. Reads GGUF model files and executes on CPU (AVX2/AVX-512 SIMD), Vulkan compute shaders, and CUDA/cuBLAS. Targets NativeAOT single-binary deployment.

---

## Critical Rules & Traps (Do Not Break)

1. **Test Runner `--nologo` Trap**:
   * **Never pass `--nologo` to `dotnet test`**. Microsoft.Testing.Platform (MTP) rejects it with exit 5 and reports *"Zero tests ran"*.
2. **Fast vs. Heavy Test Suites**:
   * Heavy tests (`Tests.ForwardPass`, `Tests.Sessions`, `Tests.Vulkan`, `Tests.Server`) skip by default to prevent memory blowups.
   * Set `STINGRAY_RUN_HEAVY_TESTS=1` to run full real-model/real-GPU test suites before committing major engine/kernel changes.
   * Everyday iteration should target `.Fast` test projects (e.g. `Tests.ForwardPass.Fast`, `Tests.Server.Fast`, `Tests.Audio`).
3. **Test Filter Syntax**:
   * Through `dotnet test`: use `--filter-class <Name>` or `--filter-method <Name>` (e.g., `dotnet test tests/OpenTail.Stingray.Tests.Audio -- --filter-class QwenAsrTests`).
   * When invoking built test `.exe` directly: use single-dash `-class <Name>` / `-method <Name>`.
4. **Hard Compiler & Build Constraints**:
   * `TreatWarningsAsErrors` is enabled globally â all compiler warnings must be fixed.
   * `InvariantGlobalization` is enabled â no culture-specific string operations.
   * NativeAOT / trim analyzers are active â no reflection-heavy patterns or dynamic code generation; use source-generated JSON (`OpenTailStingrayJsonContext`).
5. **Vulkan SPIR-V Shaders**:
   * Shaders are defined in `src/OpenTail.Stingray.Vulkan/Shaders.cs` and precompiled into `Shaders.Precompiled.g.cs`. If GLSL shader constants change, run `scripts/gen-spirv.ps1` (requires Vulkan SDK) or tests will fail on table drift.

---

## Standard Build & Test Commands

```bash
dotnet build                                                 # Debug build
dotnet build -c Release                                      # Release build (NativeAOT optimizations)
dotnet test tests/OpenTail.Stingray.Tests.ForwardPass.Fast  # Run fast forward pass tests
dotnet test tests/OpenTail.Stingray.Tests.Audio              # Run audio (TTS/ASR) tests

# Run CLI Text Inference
dotnet run --project src/OpenTail.Stingray.Cli -c Release -- -m models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf -p "prompt" --temp 0 -g -1

# Run API Server Host (OpenAI / Anthropic compatible)
STINGRAY_MODEL=models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf dotnet run --project src/OpenTail.Stingray.Server.Host -c Release

# NativeAOT Publish
dotnet publish src/OpenTail.Stingray.Cli -c Release -r win-x64
dotnet publish src/OpenTail.Stingray.Server.Host -c Release -r win-x64
```

---

## Architecture Topology

The solution (`OpenTail.Stingray.slnx`) is organized into four core layers:

1. **Core** (`OpenTail.Stingray.Core`) â Central types & abstractions (`IComputeBackend`, `IForwardPass`, `ITokenizer`, `ITokenConstraint`), GGUF memory-mapped parser, BPE/SPM tokenizers, Jinja chat templates, and grammar-constrained JSON schema decoding.
2. **Compute Backends**:
   * `OpenTail.Stingray.Cpu` â AVX2/AVX-512 SIMD kernels (`SimdKernels`), Q4_K/Q6_K/Q8 dequantization (`Dequantize`), Gated-DeltaNet kernels (`GdnKernels`).
   * `OpenTail.Stingray.Cuda` â cuBLAS GEMM, NVRTC runtime kernels (`CudaTextKernels`, `CudaWsKernels`), GPU buffer pools.
   * `OpenTail.Stingray.Vulkan` â Vortice.Vulkan compute shaders, SPIR-V pipeline cache.
3. **Engine** (`OpenTail.Stingray.Engine`) â Forward pass dispatch (`ForwardPass`, `CudaForwardPass`, `HybridGdnForwardPass`), paged KV cache (`PagedKvCache`), token sampling (`Sampler`), continuous batching (`ContinuousBatchingEngine`), and speculative decoding (`MtpDecoder`, `DSparkDecoder`).
4. **Domain Pipelines**:
   * `OpenTail.Stingray.Audio` â Native TTS/ASR (`CosyVoice 3`, `Qwen3-TTS 12Hz`, `Kokoro-82M`, `F5-TTS`, `MeloTTS`, `Piper`, `OpenAI Whisper`, `NVIDIA NeMo Parakeet ASR`, `Alibaba Qwen3-ASR & ForcedAligner`, `Silero VAD`).
   * `OpenTail.Stingray.Diffusion` â Text-to-image/video (`ZImageDiT`, `FluxDiT`, `RRDBNet` Real-ESRGAN upscaler, `VaeDecoder`).
   * `OpenTail.Stingray.Vision` â Multimodal vision (`UnifiedVisionPipeline` for Gemma 4 `gemma4uv`/`gemma4v`, Gemma 3, Llama 4).
   * `OpenTail.Stingray.TurboQuant` â KV cache compression (KVarN Hadamard + Sinkhorn variance normalization).
   * `OpenTail.Stingray.Sessions` â Hot-session lifecycle, prefix sharing, and revisioned state management.
5. **Frontends**:
   * `OpenTail.Stingray.Cli` â CLI entry point (`run`, `image`, `perplexity`, `list-metadata`, `list-tensors`).
   * `OpenTail.Stingray.Server` / `Server.Host` â High-performance ASP.NET Core endpoint library and runnable AOT host.

---

## Documentation References

* **Subsystem Architecture & Layouts**: [docs/reference/OpenTail.Stingray-Design.md](docs/reference/OpenTail.Stingray-Design.md)
* **Model Command Examples & Archive**: [docs/reference/claude-reference-archive.md](docs/reference/claude-reference-archive.md)
* **Active Engineering Backlog**: [docs/00-current-work.md](docs/00-current-work.md)
