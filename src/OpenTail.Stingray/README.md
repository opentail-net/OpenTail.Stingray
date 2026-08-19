# OpenTail.Stingray

Unified native Multimodal AI runtime for .NET 10 — 100% managed C#, running in-process with zero Python, zero P/Invoke, and zero server sidecars. Reads GGUF and Safetensors models and runs them natively on CPU (AVX2/AVX-512), Vulkan, or CUDA. Publishes as a single NativeAOT binary.

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

Stingray is the engine itself, written in pure C#. The forward pass, the SIMD kernels, the KV cache, the audio DSP, the diffusion schedulers, and the samplers are managed code you can step into, profile, and extend — and it AOT-publishes into your app as one binary with no runtime dependency on Python, llama.cpp, or external DLLs.

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

One `dotnet add package` brings the whole stack — all assemblies ship inside this package rather
than as separate fragmented NuGet dependencies:

| Assembly | Purpose & Capabilities |
|---|---|
| `OpenTail.Stingray.Core` | GGUF/Safetensors parsing, BPE/SentencePiece tokenizers, chat templates, tensor types, grammar-constrained decoding, `HardwareCapabilities` zero-latency profiler, `SmartOffloadPlanner` |
| `OpenTail.Stingray.Cpu` | CPU SIMD backend (AVX2/AVX-512/NEON), fast dequantization kernels (`Q4_K`, `Q8_0`), optional OpenBLAS GEMM |
| `OpenTail.Stingray.Vulkan` | Vulkan compute backend with precompiled SPIR-V shaders |
| `OpenTail.Stingray.Cuda` | CUDA backend — cuBLAS GEMM plus NVRTC runtime-compiled kernels |
| `OpenTail.Stingray.Engine` | Forward pass, paged KV cache, samplers, speculative decoding, continuous batching |
| `OpenTail.Stingray.Vision` | Unified Multimodal Vision (Gemma 4 `gemma4uv`/`gemma4v`, Gemma 3 SigLIP `gemma3`, Llama 4 ViT, GLM-4V) |
| `OpenTail.Stingray.Audio` | Studio Audio: Whisper STT ($O(N)$ KV cache + Silero VAD), 5 Neural TTS engines (Kokoro, Piper, Melo, Chatterbox, F5-TTS), rational sinc resampler, ATSC A/85 downmix, TPDF dithered WAV export |
| `OpenTail.Stingray.Diffusion` | Diffusion & Video: SD 1.5, SDXL, SD 3/3.5, FLUX.1, FLUX.2 (Klein & Kontext), FLUX 3 Multimodal Video+Audio, Stable Audio 3 (Variable-Length 44.1kHz Stereo), Z-Image-Turbo, Wan 2.1/2.2, HunyuanVideo, LTX-Video, LCM 1–4 step, StreamDiffusion pipelining, Animated GIF and PNG sequence exporters, Real-ESRGAN upscaling |
| `OpenTail.Stingray.Pipeline` | Three-tier VRAM → RAM → NVMe weight streaming for MoE expert offload |
| `OpenTail.Stingray.TurboQuant` | KV-cache compression (KVarN 4/2-bit, Lloyd-Max 3/4-bit) |

---

## Model formats

* **GGUF Models:** Supported natively across LLMs, Vision, and Quantized Diffusion (`Q4_0`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`).
* **Safetensors Checkpoints:** Multi-precision support (`F32`, `F16`, `BF16`, `F8_E4M3`, `F8_E5M2`) across UNet, DiT, MMDiT, Video DiT, and LoRA files with automatic tensor header inspection and key alias normalizers.
