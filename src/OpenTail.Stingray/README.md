# OpenTail.Stingray

Local LLM inference and image generation for .NET 10 — no Python, no P/Invoke to llama.cpp, no server sidecar. Reads GGUF models and runs them in-process on CPU (AVX2/AVX-512), Vulkan, or CUDA. Publishes as a single NativeAOT binary.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![.NET 10](https://img.shields.io/badge/.NET-10-blue)]()
[![NativeAOT](https://img.shields.io/badge/NativeAOT-ready-green)]()

> **Built by [opentail.net](https://opentail.net)**

This is the **library** package. For the command-line tool install [`OpenTail.Stingray.Cli`](https://www.nuget.org/packages/OpenTail.Stingray.Cli); for OpenAI/Anthropic-compatible HTTP endpoints install [`OpenTail.Stingray.Server`](https://www.nuget.org/packages/OpenTail.Stingray.Server).

---

## Why this exists

Running a local model from .NET usually means shelling out to a Python server, or binding to a native
library and marshalling across the boundary on every token. Both put your inference loop in another
process, another runtime, and another dependency tree.

Stingray is the engine itself, written in C#. The forward pass, the SIMD kernels, the KV cache and the
samplers are managed code you can step into, profile and extend — and it AOT-publishes into your app
as one binary with no runtime dependency on Python or llama.cpp.

---

## Install

```
dotnet add package OpenTail.Stingray
```

## Quick start

```csharp
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Engine;

var model     = GgufModel.Open("models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf");
var hp        = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
var tokenizer = GgufTokenizer.FromGgufModel(model);
var backend   = new CpuBackend();
var forward   = new ForwardPass(model, backend, hp);

// The engine takes ownership of the forward pass plus anything in `owned`, so one
// Dispose releases the memory-mapped GGUF, the backend and all native scratch.
using var engine = new InferenceEngine(
    forward, tokenizer, modelId: "smollm2", owned: [backend, model]);

var sampling = new SamplingParams
{
    Temperature  = 0.7f,
    TopP         = 0.95f,
    MaxNewTokens = 512,
    StopTokenIds = [.. tokenizer.EogTokenIds],   // model's end-of-generation tokens
};

await foreach (var chunk in engine.GenerateChunksAsync("The capital of France is", sampling))
{
    if (chunk.Kind == GenerateChunkKind.Text)
        Console.Write(chunk.Text);
}
```

For GPU inference swap `CpuBackend` for `VulkanBackend` or `CudaBackend`, or use `HybridForwardPass`
to keep some layers on the CPU. A complete multi-turn chat example, including prefix-cache reuse
across turns, lives in `samples/OpenTail.Stingray.Sample.Chat`.

---

## What's in the package

One `dotnet add package` brings the whole stack — all nine assemblies ship inside this package rather
than as separate NuGet dependencies:

| Assembly | Purpose |
|---|---|
| `OpenTail.Stingray.Core` | GGUF parsing, BPE/SentencePiece tokenizers, chat templates, tensor types, grammar-constrained decoding |
| `OpenTail.Stingray.Cpu` | CPU SIMD backend (AVX2/AVX-512), dequantization kernels, optional OpenBLAS GEMM |
| `OpenTail.Stingray.Vulkan` | Vulkan compute backend with precompiled SPIR-V shaders |
| `OpenTail.Stingray.Cuda` | CUDA backend — cuBLAS GEMM plus NVRTC runtime-compiled kernels |
| `OpenTail.Stingray.Engine` | Forward pass, paged KV cache, samplers, speculative decoding, continuous batching |
| `OpenTail.Stingray.Vision` | Gemma 4 encoder-free vision projector |
| `OpenTail.Stingray.Diffusion` | Z-Image-Turbo and FLUX.1 text-to-image, Real-ESRGAN 4× upscaling |
| `OpenTail.Stingray.Pipeline` | Three-tier VRAM → RAM → NVMe weight streaming for MoE expert offload |
| `OpenTail.Stingray.TurboQuant` | KV-cache compression (KVarN 4/2-bit, Lloyd-Max 3/4-bit) |

---

## Model formats

**GGUF is the supported deployment format**, and the only route to the block-quantized fast paths.

SafeTensors support is deliberately narrow, and says so rather than guessing: a Hugging Face model
*directory* holding dense **Llama or Mistral** weights in F32/F16/BF16 runs on **CPU only**. It does
not cover quantized weights, other architectures, GPU backends, batching or persisted sessions.
Anything outside that profile is refused with a named reason instead of running and being subtly
wrong. Use `stingray capabilities` to see the published profile, or `stingray inspect -m <dir>` for a
verdict on a specific package before you try to run it.

## Model architectures

Explicitly handled: **Llama 3.x / Llama 4**, **Qwen2 / Qwen3 / Qwen3-MoE**, **Qwen3.5 / 3.6 hybrid**
(Gated-DeltaNet + attention + MoE), **Gemma / Gemma 2 / 3 / 4**, **Phi-2 / Phi-3**, and **OLMoE**.

Gemma 4 covers sliding-window attention, per-layer head dims, fused GELU-tanh and logit soft-clipping.
Speculative decoding is available through draft models, prompt-lookup, MTP/NEXTN heads and DSpark
EAGLE-3 draft heads.

## Optional native dependencies

None are required — each is probed at runtime and skipped cleanly when absent.

- **OpenBLAS** — CPU GEMM acceleration. Auto-detected on `PATH`.
- **Vulkan drivers** — any recent AMD / Intel / NVIDIA driver. No extra install on Windows; macOS needs MoltenVK.
- **CUDA Toolkit 12.x** — NVIDIA only; needs `nvcuda.dll` and `cublas64_*.dll` on `PATH`.

## NativeAOT

Every assembly is trim-safe and AOT-compatible:

```
dotnet publish -c Release -r win-x64      # or linux-x64, osx-arm64, ...
```

---

## Links

- [Repository and documentation](https://github.com/opentail-net/OpenTail.Stingray)
- [Issues](https://github.com/opentail-net/OpenTail.Stingray/issues)

---

## Acknowledgements

Forked from **[SharpInference](https://github.com/pekkah/SharpInference)** by Pekka Heikura (MIT), which remains actively developed upstream; copyright is retained in `LICENSE` alongside ours.

Interoperates with **[llama.cpp](https://github.com/ggml-org/llama.cpp)**'s GGUF format and quantization block layouts, and follows `llama-cli` flag names where the meaning matches — **no llama.cpp code is used**. **[LLamaSharp](https://github.com/SciSharp/LLamaSharp)** was studied as the reference for .NET inference API design; **no LLamaSharp code is used**, and unlike it this engine is managed C# end to end rather than P/Invoke bindings to native llama.cpp.

## License

MIT. Copyright © 2026 Pekka Heikura · Copyright © 2026 OpenTail.
