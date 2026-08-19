# OpenTail.Stingray

> **The Unified Native Multimodal AI Engine for .NET 10**  
> 100% pure managed C# — Zero Python, Zero P/Invoke to llama.cpp, Zero external server sidecars.  
> Runs in-process on CPU (AVX-512/AVX2/NEON), Vulkan, or CUDA. Publishes as a single NativeAOT binary.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![.NET 10](https://img.shields.io/badge/.NET-10-blue)]()
[![NativeAOT](https://img.shields.io/badge/NativeAOT-ready-green)]()
[![NuGet](https://img.shields.io/nuget/v/OpenTail.Stingray.svg)](https://www.nuget.org/packages/OpenTail.Stingray)

Built by [opentail.net](https://opentail.net)

---

## Why Stingray?

In Python and C++, running modern local AI requires juggling 4–5 fragmented, heavy tools (`llama.cpp` for text, `whisper.cpp` for speech, `kokoro` for voice synthesis, and `ComfyUI` or `diffusers` for image/video generation).

**Stingray unifies the entire local AI ecosystem into one clean, lightweight .NET library.** All tensor math, SIMD kernels, KV caches, diffusion flow-schedulers, audio DSP resamplers, and tokenizers are written in high-performance managed C# that you can step into, profile, and deploy anywhere.

---

## Superpowers at a Glance

* 💬 **Text LLMs & Sparse MoE:** Llama 3 / 3.1 / 3.2 / 3.3 / 4, Qwen 2.5 / Qwen 3.5 MoE, DeepSeek-V3 / R1, Gemma 3 / 4, SmolLM2. Full tool calling, JSON schema constrained decoding, and grammar masks.
* 👁️ **Multimodal Vision:** Native Gemma 4 unified (`gemma4uv`) & ViT (`gemma4v`), Gemma 3 SigLIP (`gemma3`), Llama 4 ViT, and GLM-4V patch grids.
* 🎙️ **Studio Audio Stack:**
  * **Speech-to-Text (STT):** Whisper with $O(N)$ KV Cache, cross-attention projection, and Silero VAD.
  * **5 Neural Voice Engines (TTS):** Kokoro, Piper (VITS), MeloTTS, Chatterbox, and F5-TTS Flow-Matching DiT with real-time clause streaming.
  * **Broadcast DSP:** Rational windowed-sinc resampler, ATSC A/85 downmixing, and TPDF dithered 16-bit/24-bit WAV exporter.
* 🎨 **State-of-the-Art Diffusion & Video:**
  * **Image Architectures:** SD 1.5, SDXL, SD 3 / 3.5 (MMDiT), FLUX.1 (schnell/dev), FLUX.2 (Klein & Kontext multi-reference), FLUX 3 (3D/4D RoPE Multimodal), and Z-Image-Turbo.
  * **Acoustic Diffusion:** Stable Audio 3 (Variable-Length 44.1kHz Stereo DiT).
  * **Video Diffusion:** Wan 2.1/2.2 Video, HunyuanVideo, and LTX-Video.
  * **Real-Time Streaming:** Latent Consistency Models (LCM 1–4 step) and `StreamBatchPipeline` (30–60 FPS real-time webcam/video streaming).
  * **Exporters:** Animated GIF and animated PNG sequence writers with Floyd-Steinberg dithering.
* ⚡ **Zero-Latency Hardware Engine:** Nanosecond SIMD evaluation (`AVX-512`, `AVX2`, `ARM Neon`), sub-millisecond cached GPU topology (`~/.stingray/hardware_cache.json`), and `SmartOffloadPlanner` for discrete GPUs, APUs (AMD Vega/Intel Iris), and CPUs.

---

## Installation

```bash
dotnet add package OpenTail.Stingray
```

---

## Quick Starts

### 1. LLM Chat with Streaming

```csharp
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Engine;

// 1. Open GGUF model and initialize hardware backend
var model     = GgufModel.Open("models/Llama-3.2-3B-Instruct-Q4_K_M.gguf");
var hp        = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
var tokenizer = GgufTokenizer.FromGgufModel(model);
var backend   = new CpuBackend();
var forward   = new ForwardPass(model, backend, hp);

using var engine = new InferenceEngine(forward, tokenizer, modelId: "llama3", owned: [backend, model]);

var sampling = new SamplingParams
{
    Temperature  = 0.7f,
    TopP         = 0.9f,
    MaxNewTokens = 512,
    StopTokenIds = [.. tokenizer.EogTokenIds]
};

// 2. Stream generation in real-time
await foreach (var chunk in engine.GenerateChunksAsync("Explain quantum computing in simple terms:", sampling))
{
    if (chunk.Kind == GenerateChunkKind.Text)
        Console.Write(chunk.Text);
}
```

---

### 2. Studio Voice Synthesis (Text-to-Speech)

```csharp
using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Audio.Kokoro;

ITextToSpeechPipeline tts = new KokoroPipeline();
await tts.LoadAsync("models/kokoro-v1.0.safetensors");

// Synthesize high-fidelity 24kHz audio
AudioBuffer audio = await tts.SynthesizeAsync("Welcome to OpenTail Stingray.", voice: "af_heart", speed: 1.0f);

// Save with broadcast-quality TPDF dithering
WavWriter.Write("output.wav", audio);
```

---

### 3. Image Generation with FLUX.2 or SDXL

```csharp
using OpenTail.Stingray.Diffusion;
using OpenTail.Stingray.Diffusion.Flux2;

var pipeline = new Flux2Pipeline();
await pipeline.LoadAsync("models/flux-2-klein-4step.safetensors");

var prompt = new DiffusionPrompt
{
    Positive = "A cinematic shot of an astronaut riding a horse on Mars, 8k, photorealistic",
    Width = 1024,
    Height = 1024,
    Steps = 4,
    GuidanceScale = 3.5f
};

var rgb = await pipeline.GenerateAsync(prompt);
PngWriter.Write("astronaut.png", rgb, prompt.Width, prompt.Height);
```

---

## What's in the Package

One `dotnet add package OpenTail.Stingray` brings the complete, unified engine:

| Assembly | Modality & Functionality |
|---|---|
| `OpenTail.Stingray.Core` | GGUF/Safetensors parser, BPE/SentencePiece tokenizers, chat templates, grammar masks, `HardwareCapabilities` zero-latency profiler, and `SmartOffloadPlanner`. |
| `OpenTail.Stingray.Engine` | Autoregressive forward pass, paged KV cache, speculative decoding, continuous batching, and sampling. |
| `OpenTail.Stingray.Cpu` | CPU SIMD kernels (AVX-512/AVX2/NEON), fast dequantization (`Q4_K`, `Q8_0`), and multithreaded GEMM. |
| `OpenTail.Stingray.Vulkan` | High-performance Vulkan compute backend with precompiled SPIR-V shaders. |
| `OpenTail.Stingray.Cuda` | CUDA backend with cuBLAS GEMM and runtime NVRTC shader compilation. |
| `OpenTail.Stingray.Vision` | Unified vision projection for Gemma 4 (`gemma4uv`/`gemma4v`), Gemma 3 SigLIP, Llama 4, and GLM-4V. |
| `OpenTail.Stingray.Audio` | Studio Audio: Whisper STT ($O(N)$ KV Cache + Silero VAD), 5 Neural TTS engines (Kokoro, Piper, Melo, Chatterbox, F5-TTS), rational sinc resampler, ATSC A/85 downmixing, and TPDF dithered WAV export. |
| `OpenTail.Stingray.Diffusion` | Diffusion & Video: SD 1.5, SDXL, SD 3/3.5, FLUX.1, FLUX.2, FLUX 3 Multimodal, Stable Audio 3, Z-Image-Turbo, Wan 2.1/2.2, HunyuanVideo, LTX-Video, LCM 1–4 step distillation, StreamDiffusion pipelining, and GIF/PNG sequence exporters. |
| `OpenTail.Stingray.Pipeline` | Three-tier VRAM → RAM → NVMe weight streaming for MoE expert offload. |
| `OpenTail.Stingray.TurboQuant` | KV-cache compression (KVarN 4/2-bit, Lloyd-Max 3/4-bit). |

---

## Related Packages

* **[`OpenTail.Stingray.Server`](https://www.nuget.org/packages/OpenTail.Stingray.Server):** Drop-in ASP.NET Core endpoints exposing full OpenAI Chat, Audio (`/v1/audio/*`), and Image Generation (`/v1/images/*`) API parity.
* **[`OpenTail.Stingray.Cli`](https://www.nuget.org/packages/OpenTail.Stingray.Cli):** Standalone command-line tool (`stingray run`, `stingray image`, `stingray tts`, `stingray serve`).

---

## License

OpenTail.Stingray is licensed under the [MIT License](https://opensource.org/licenses/MIT).
