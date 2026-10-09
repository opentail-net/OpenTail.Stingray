# Development & Contributing Guide

> **Scope:** Architecture overview, build instructions, test harness tiers, and coding guidelines for contributors to OpenTail.Stingray.  
> **Prerequisites:** [.NET 10 SDK](https://dotnet.microsoft.com/download), x64 CPU with AVX2 (or ARM64). Optional: Vulkan-capable GPU or NVIDIA CUDA 12.x.

---

## 1. Solution Architecture & Project Map

OpenTail.Stingray is built as a pure managed C# solution with zero Python and zero native C++ P/Invoke dependencies. Projects are organized with clear boundaries:

```
src/
├── OpenTail.Stingray.Core/       <- Tensor structures, GGUF/Safetensors loaders, ModelHome, ModelCatalog
├── OpenTail.Stingray.Cpu/        <- Managed SIMD kernels (AVX-512, AVX2, ARM Neon) & thread pool
├── OpenTail.Stingray.Vulkan/     <- SPIR-V compute shaders, memory allocator, GPU dispatch
├── OpenTail.Stingray.Cuda/       <- Managed CUDA driver interop & custom CUDA kernels
├── OpenTail.Stingray.Engine/     <- Public API facade, Model, ModelContext, Executors, ChatSession
├── OpenTail.Stingray.Audio/      <- TTS (Piper, Kokoro, Qwen3-TTS), STT (Whisper), DSP resamplers
├── OpenTail.Stingray.Vision/     <- Unified vision embedders (Qwen-VL, Pixtral, DeepSeek-OCR, etc.)
├── OpenTail.Stingray.Diffusion/  <- DiT pipelines (FLUX.1, SD 3.5, Wan Video, Z-Image-Turbo)
├── OpenTail.Stingray.Server/     <- ASP.NET Core OpenAI & Anthropic compatible API server
├── OpenTail.Stingray.Cli/        <- Command-line tool (stingray)
└── OpenTail.Stingray/            <- Primary NuGet metapackage entry point
```

---

## 2. Building From Source

Clone the repository and build using the .NET 10 SDK:

```bash
git clone https://github.com/opentail-net/OpenTail.Stingray.git
cd OpenTail.Stingray

# Build all projects in Release configuration
dotnet build -c Release
```

To run the CLI directly from source:

```bash
dotnet run --project src/OpenTail.Stingray.Cli -c Release -- --help
```

---

## 3. Test Suites & Verification Tiers

To ensure both rapid inner-loop development and rigorous mathematical parity against reference implementations, the test suite is divided into tiers:

### Tier 1: Core Contracts & CLI Tests (Fast, seconds)
Validates API contracts, options parsing, model metadata, environment variable registries, and packaging:
```bash
dotnet test tests/OpenTail.Stingray.Tests.Core
dotnet test tests/OpenTail.Stingray.Tests.Cli
```

### Tier 2: Fast Forward-Pass Verification (~1–2 minutes)
Performs forward-pass validation across hundreds of model architectures and quantization layouts using synthetic or lightweight tensor fixtures:
```bash
dotnet test tests/OpenTail.Stingray.Tests.ForwardPass.Fast
```

### Tier 3: Golden Parity & Real Weights Verification (Minutes, requires models)
Validates token-by-token logits against upstream `llama.cpp` golden captures or real GGUF weights:
```bash
# Capture or verify golden parity
stingray capture-golden -m <model.gguf>
stingray verify-goldens
```

---

## 4. Managed SIMD & Memory Guidelines

1. **Zero-Allocation Execution:** The inner inference loop must never allocate heap objects per token. Use reusable context buffers, `Span<T>`, `ReadOnlySpan<T>`, and pinned memory.
2. **SIMD Vectorization:** Kernel dispatches should leverage `System.Runtime.Intrinsics` (`Vector256<float>`, `Vector512<float>`) with appropriate fallback paths when vector extensions are absent.
3. **Quantized Math:** Quantized formats (Q4_K, Q6_K, Q8_0) use fused dequantize-and-multiply blocks to minimize memory traffic.
4. **Thread Safety & Context Isolation:** `Model` holds immutable read-only weights. All mutable inference state (KV cache, token scratch buffers) belongs exclusively to `ModelContext`. Multiple contexts can evaluate concurrently against one loaded `Model`.

---

## 5. NativeAOT & Trimming Conventions

Stingray is designed to publish as a self-contained, single-file NativeAOT executable:
- **No Dynamic Reflection:** Do not use `Type.GetType()`, unannotated reflection, or dynamic code generation.
- **Serialization:** Use source-generated `System.Text.Json` contexts for all JSON serialization in server and CLI endpoints.
- **Attribute Annotations:** Where reflection is unavoidable, annotate types with `[DynamicallyAccessedMembers]`.

---

## 6. Contributing Checklist

Before submitting a pull request or committing changes:
1. Ensure all Core tests pass: `dotnet test tests/OpenTail.Stingray.Tests.Core`
2. If changing CLI options, regenerate the inventory: `pwsh ./scripts/gen-cli-option-inventory.ps1`
3. If changing environment variables, verify the registry: `dotnet test tests/OpenTail.Stingray.Tests.Core -- --filter-class *KnownEnvironmentVariablesTests*`
4. Follow the project's governing standard: **A capability may be advertised only when its implementation status and verification evidence support the exact claim being made.**

For in-depth architectural notes, consult [docs/reference/OpenTail.Stingray-Design.md](docs/reference/OpenTail.Stingray-Design.md) and [CLAUDE.md](CLAUDE.md).
