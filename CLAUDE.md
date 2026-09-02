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
   * Through `dotnet test`: use `--filter-class <Name>` or `--filter-method <Name>` (e.g., `dotnet test tests/OpenTail.Stingray.Tests.Audio -- --filter-class QwenAsrTests`). This has been unreliable in practice (silently reports "Zero tests ran" even for a real, discoverable test) — if it does, don't keep retrying flag variants; fall back to invoking the built `.exe` directly.
   * When invoking built test `.exe` directly (`tests/<Project>/bin/<Config>/net10.0/<Project>.exe -class <Name>`): the class name must be **fully namespace-qualified** (e.g. `-class OpenTail.Stingray.Tests.Audio.CosyVoice3ClipGenDebugTests`, not just `CosyVoice3ClipGenDebugTests`) — the bare class name silently matches zero tests. Build the test project first (`dotnet build tests/<Project> -c Release`) since the `.exe` doesn't rebuild itself.
4. **Hard Compiler & Build Constraints**:
   * `TreatWarningsAsErrors` is enabled globally â all compiler warnings must be fixed.
   * `InvariantGlobalization` is enabled â no culture-specific string operations.
   * NativeAOT / trim analyzers are active â no reflection-heavy patterns or dynamic code generation; use source-generated JSON (`OpenTailStingrayJsonContext`).
5. **Vulkan SPIR-V Shaders**:
   * Shaders are defined in `src/OpenTail.Stingray.Vulkan/Shaders.cs` and precompiled into `Shaders.Precompiled.g.cs`. If GLSL shader constants change, run `scripts/gen-spirv.ps1` (requires Vulkan SDK) or tests will fail on table drift.
6. **Do not use subagents in this project**:
   * Do all work directly in the main session — no `Agent`/subagent delegation. The user has explicitly opted out of subagent use here (they add overhead and drain tokens without the coordinating session retaining useful context). This applies project-wide, not just to the Audio rebuild work.
7. **Performance pass + DRY pass once a model's port is complete**:
   * When all porting/wiring work on a given model/pipeline is done (every stage golden-verified, wired end-to-end, tests passing), do a performance pass and a DRY pass on it before considering it finished.
   * Performance pass: measure, don't assume. Use real weights and a realistic (not trivially short) input, take enough samples to trust the result (a handful of runs each side, not one), and only keep a change if it's measurably better — a plausible-sounding optimization that isn't actually faster gets reverted, even if the reasoning behind it seemed sound. Write the measured numbers down (e.g. in the relevant progress doc), not just "should be faster."
   * DRY pass: check for logic duplicated across files for the same model/pipeline (e.g. an encoder and decoder that copy-pasted the same `Linear`/`LayerNorm`/attention helpers) and extract shared code, following the existing `Primitives/*Kernels.cs` convention. Re-run the affected golden/structural tests after extracting to confirm no numerical regression.
8. **Check the real reference before "fixing" code that looks wrong**:
   * The `examples/*.cpp` folder holds real, working C++/GGML reference ports checked into this repo specifically so this codebase's math can be diff'd against them. If a piece of ported math looks suspicious, grep the reference's actual implementation for that stage *before* rewriting it — a structure that looks like a bug is sometimes the real, intentional algorithm. Reverting a correct port because it "looked wrong" wastes a full round-trip and produces a confident-sounding wrong diagnosis.
9. **Don't commit scratch/debug output to the repo root**:
   * Investigation artifacts (one-off debug scripts, raw tensor/error dumps, tuning-iteration audio/image outputs) belong in the OS temp scratchpad while you work, not the repo root — a 2026-08-31 review found ~25 such files had been committed there over time (`chatterbox_lunch_darling_*.wav`, `*_tensors.txt`, `scratch_*.cs`, etc.), which the root-level `.gitignore` patterns now block going forward. Real, weight-driven samples still belong in `docs/audio-samples/`/`docs/diffusion-samples/` with a descriptive filename (not the repo root) — **but as of 2026-09-03 these two directories are `.gitignore`d and local-only**: generate real samples into them as before for your own/the user's reference, but do not `git add`/commit anything inside them (large binary media accumulated there and bloated the repo). The one exception is `docs/diffusion-samples/README.md` itself, which stays tracked normally. A doc that needs to cite a specific sample links to it by filename/path in prose.
10. **Keep the README's "What actually works today" matrix honest**:
    * When a pipeline's real, verified status changes (newly working, a bug found that downgrades it, a gap closed), update the status matrix near the top of `README.md` in the same pass — cite the dated finding in `docs/audio-review-progress.md` / `docs/diffusion-samples/README.md` (or wherever the evidence lives), don't just assert a color. The matrix's whole value is that every row is sourced, not aspirational — an unsourced or stale row is worse than no row.

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
   * `OpenTail.Stingray.Audio` â Native TTS/ASR (`CosyVoice 2/3`, `Qwen3-TTS 12Hz`, `Fish Speech S2 Pro`, `Kokoro-82M`, `F5-TTS`, `MeloTTS`, `Piper`, `Chatterbox-Turbo`, `Orpheus-TTS`, `Parler-TTS`, `OpenAI Whisper`, `NVIDIA NeMo Parakeet ASR`, `Alibaba Qwen3-ASR & ForcedAligner`, `Silero VAD`).
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
* **Coverage tooling** (`pull`, `admit-arch`, `gen-vision-scaffold`): [docs/061-coverage-tooling.md](docs/061-coverage-tooling.md)

---

## Coverage tooling: `pull` / `admit-arch` / `gen-vision-scaffold`

Three CLI commands exist specifically to speed up this project's "run any GGUF from Hugging Face"
goal — see [docs/061-coverage-tooling.md](docs/061-coverage-tooling.md) for full usage and how
each works internally. Short version:

* `stingray pull -r <owner/repo>` — download a GGUF straight from a Hugging Face repo id (quant
  selection, sharded-checkpoint support, resumable download). Use this instead of fetching
  checkpoints by hand before running/admitting a new architecture.
* `stingray admit-arch -m <path> [--reference-tokens ids] [-p prompt]` — triages a GGUF whose
  `general.architecture` isn't yet in `ModelCompatibility`'s allowlist: tokenizer/tensor
  inventory, a real bypassed forward-pass run, and (given a reference token sequence captured
  from `llama-server`/`llama-tokenize`) a pasteable `ADMIT`/`REJECT` verdict block. Use this as
  the first step whenever admitting a new text-generation architecture — it does not replace
  capturing a real independent reference, only the mechanical comparison against it.
* `stingray gen-vision-scaffold -m <mmproj> -a <arch>` — real tensor/metadata inventory plus a
  starter `<Arch>VisionEmbedderParityTests.cs` for a new vision architecture. The independent
  oracle is the real vendored `tools/llama.cpp/llama-mtmd-cli.exe`/`llama-mtmd-debug.exe`
  binaries, not a hand-written reimplementation — **do not add new Python reference scripts**
  (`scripts/*_ref.py`); that pattern predates this project's no-Python rule and is not being
  extended.
