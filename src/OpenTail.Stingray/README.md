# OpenTail.Stingray

> **Ever wanted to try local AI beyond LLMs?**
>
> OpenTail.Stingray brings text, vision, speech, image generation, audio and video diffusion into one .NET 10 library. Run models locally from C# on CPU, Vulkan or CUDA — without a Python runtime or an external inference server.

**One library. Multiple AI modalities. Native .NET deployment.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![.NET 10](https://img.shields.io/badge/.NET-10-blue)]()
[![NativeAOT](https://img.shields.io/badge/NativeAOT-ready-green)]()
[![NuGet](https://img.shields.io/nuget/v/OpenTail.Stingray.svg)](https://www.nuget.org/packages/OpenTail.Stingray)

Built by [opentail.net](https://opentail.net)


## Install

```bash
dotnet add package OpenTail.Stingray
```

Requires .NET 10 and a 64-bit x86 CPU with AVX2 (most PCs from 2015 on). ARM64 has SIMD kernels but is not yet measured by the project. A GPU is optional: any Vulkan-capable card, or NVIDIA with CUDA 12.

**What you do and do not need.** Stingray's own inference code (tensor kernels, tokenizers, KV caches, samplers, diffusion schedulers) is managed C#. You do not need Python, llama.cpp, ggml or a separate inference server. Some execution paths still load native code that is not Stingray's: the GPU driver (Vulkan loader, or the CUDA 12 runtime for the CUDA backend), and ONNX Runtime, which this package depends on for the ONNX-based speech engines such as Piper. The CPU backend for GGUF language models needs none of these.

## Your first model

Models are not bundled. The easiest way to get a small, checksum-verified one is the companion CLI:

```bash
dotnet tool install -g OpenTail.Stingray.Cli
stingray setup chat        # Qwen2.5 0.5B Instruct, 469 MB, Apache-2.0
```

Then, in your app (`Program.cs`):

<!-- verify:consumer -->
```csharp
using OpenTail.Stingray;
using OpenTail.Stingray.Core.Catalog;
using OpenTail.Stingray.Executors;

// An explicit path works too; this resolves the model that `stingray setup chat` installed.
string modelPath = args.Length > 0
    ? args[0]
    : ModelHome.ResolveModelPath("qwen2.5-0.5b");

using var model = Model.Load(modelPath);
using var context = model.CreateContext(new ContextParams { ContextSize = 4096 });
var executor = new InteractiveExecutor(context);
var session = new ChatSession(executor);

await foreach (string piece in session.ChatAsync("In one sentence, why do developers write unit tests?"))
    Console.Write(piece);
```

Any GGUF works in place of that model: `stingray pull -r <owner/repo>` downloads one, and `stingray scout -r <owner/repo>` tells you first whether Stingray supports it and whether it fits your machine.

### Streaming with options and a GPU

<!-- verify:consumer -->
```csharp
using OpenTail.Stingray;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Executors;

// Backend "auto" picks a GPU when one is available; GpuLayerCount = -1 offloads every layer.
using var model = Model.Load(new ModelParams("models/qwen2.5-0.5b-instruct-q4_k_m.gguf")
{
    Backend = "auto",
    GpuLayerCount = -1
});
using var context = model.CreateContext(new ContextParams { ContextSize = 2048 });
var session = new ChatSession(new InteractiveExecutor(context));
session.AddSystemMessage("You are a helpful, concise assistant.");

await foreach (var chunk in session.ChatChunksAsync(
    "Explain quantisation in one paragraph.",
    new InferenceParams { MaxTokens = 128, Temperature = 0.7f, EnableThinking = false }))
{
    if (chunk.Kind == GenerateChunkKind.Text)
        Console.Write(chunk.Text);
}
Console.WriteLine();
```

The chat API (`Model`, `ChatSession`, executors) is for language models. Speech, vision and image generation have their own pipeline types, shown next, because their inputs and outputs are not chat messages.

## Speech

**Text to speech** with a [Piper](https://huggingface.co/rhasspy/piper-voices) voice. Download both files of a voice, for example `en_US-lessac-medium.onnx` (63 MB) and `en_US-lessac-medium.onnx.json`; `stingray setup speak` does this for you.

<!-- verify:consumer -->
```csharp
using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Audio.Piper;

using var tts = PiperPipeline.FromConfigFile("en_US-lessac-medium.onnx.json");
tts.Generate(new AudioGenerationRequest { Text = "Hello from a voice that never left this computer.", OutputPath = "hello.wav" });
```

**Speech to text** with Whisper. Download [`ggml-base.bin`](https://huggingface.co/ggerganov/whisper.cpp/blob/main/ggml-base.bin) (148 MB), or run `stingray setup transcribe`:

<!-- verify:consumer -->
```csharp
using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Audio.Whisper;

using var stt = WhisperPipeline.Load("ggml-base.bin");
var (samples, sampleRate, _) = WavReader.ReadWav("hello.wav");
Console.WriteLine(stt.Transcribe(new SpeechToTextRequest { AudioSamples = samples, SampleRate = sampleRate }).Text);
```

Other speech engines (Kokoro, Qwen3-TTS, XTTS-v2, Fish Speech, Chatterbox; Parakeet and Qwen3-ASR for recognition) have different setup needs and different levels of verification: see [docs/STATUS.md](https://github.com/opentail-net/OpenTail.Stingray/blob/main/docs/STATUS.md) before relying on one, and the [Speech sample](https://github.com/opentail-net/OpenTail.Stingray/tree/main/samples/OpenTail.Stingray.Sample.Speech) for a complete program.

## Images in, embeddings out

This opens a vision projector (`mmproj` GGUF) and turns an image into visual-token embeddings. It is the building block for image understanding, not a finished "describe this picture" call; for end-to-end image questions use the CLI (`stingray -m <model.gguf> --image photo.png -p "..."`) and the per-model status in [docs/STATUS.md](https://github.com/opentail-net/OpenTail.Stingray/blob/main/docs/STATUS.md).

<!-- verify:consumer -->
```csharp
using OpenTail.Stingray.Vision;

using var embedder = UnifiedVisionPipeline.Open("models/mmproj-deepseek-ocr-2-q8_0.gguf");
float[] visualTokens = embedder.EmbedImageFile("document.png", out int tokenCount);
Console.WriteLine($"Generated {tokenCount} visual tokens with embedding dim {embedder.EmbeddingDim}");
```

## Image, music and video generation

Diffusion models need several files each (a diffusion model, a text encoder and a VAE), and the setup differs per model, so there is no one-liner. Start from the [Diffusion sample](https://github.com/opentail-net/OpenTail.Stingray/tree/main/samples/OpenTail.Stingray.Sample.Diffusion) (Stable Diffusion 1.5) and see the real output collections: [images and video](https://github.com/opentail-net/OpenTail.Stingray/blob/main/docs/diffusion-samples/README.md) and [music and sound effects](https://github.com/opentail-net/OpenTail.Stingray/blob/main/docs/audio-samples/README.md). Speeds are on integrated graphics unless a row says otherwise; several video models are experimental.

### Backend selection

| Hardware | `Backend` setting | Execution details |
|---|---|---|
| **CPU only** (x86-64) | `Backend = "cpu"` | Managed C# with AVX2 and AVX-512 SIMD kernels. |
| **AMD, Intel or NVIDIA GPU** | `Backend = "vulkan"` | Vulkan compute shaders (needs a Vulkan driver). |
| **NVIDIA GPU (CUDA 12.x)** | `Backend = "cuda"` | cuBLAS and CUDA kernels (needs the CUDA 12 runtime). |
| **Auto-detect** | `Backend = "auto"` | Selects an available GPU; combine with `GpuLayerCount = -1` to offload every layer. |

Not every model is supported on every backend. Per-model, per-backend evidence is in [docs/STATUS.md](https://github.com/opentail-net/OpenTail.Stingray/blob/main/docs/STATUS.md); measured speeds and memory are in [docs/RUNNING.md](https://github.com/opentail-net/OpenTail.Stingray/blob/main/docs/RUNNING.md).

---

## Beyond the starter models

`stingray models` lists a short curated catalogue (chat, speak, transcribe). The engine supports many more architectures, at different levels of verification. To check a model before you download it:

```bash
stingray scout -r <owner/repo> --quants --budget 32G   # architecture, blockers, which quantisations fit, a pinned pull command
stingray pull -r <owner/repo> --revision <commit>      # fetch exactly what scout inspected
stingray inspect -m <model directory>                # verdict on a downloaded SafeTensors package
```

If an architecture is not yet admitted, [the admission guide](https://github.com/opentail-net/OpenTail.Stingray/blob/main/docs/reference/adding-an-architecture.md) explains how it gets verified and added.

---

## What is inside the package

The package bundles every `OpenTail.Stingray.*` library (Core, Cpu, Vulkan, Cuda, Engine, Sessions, Vision, Audio, Pipeline, TurboQuant, Diffusion). The lists below show what the code contains. **They are not a statement that every item has the same maturity, speed, backend coverage or setup effort**: that is what [docs/STATUS.md](https://github.com/opentail-net/OpenTail.Stingray/blob/main/docs/STATUS.md) records, model by model, with dates.

### Why one library

In Python and C++, running modern local AI requires juggling 4–5 fragmented, heavy tools (`llama.cpp` for text, `whisper.cpp` for speech, `kokoro` for voice synthesis, and `ComfyUI` or `diffusers` for image/video generation).

**Stingray unifies the entire local AI ecosystem into one clean, lightweight .NET library.** All tensor math, SIMD kernels, KV caches, diffusion flow-schedulers, audio DSP resamplers, and tokenizers are written in high-performance managed C# that you can step into, profile, and deploy anywhere.


### Capabilities in the code


* 💬 **Text LLMs & Sparse MoE:** Llama 3 / 3.1 / 3.2 / 3.3 / 4, Mistral 7B / Mixtral MoE / Ministral, Qwen 2.5 / Qwen 3.5 MoE, DeepSeek-V3 / R1, Gemma 3 / 4, SmolLM2. Full tool calling, JSON schema constrained decoding, and grammar masks.
* 👁️ **Multimodal Vision (11+ Architectures):**
  * **Alibaba:** Qwen2.5-VL, Qwen3-VL (3D Conv stem, M-RoPE, 2x2 spatial merge).
  * **DeepSeek:** DeepSeek-OCR & DeepSeek-OCR2 (Dual SAM + CLIP ViT fusion, 1024x1024 grid).
  * **Mistral AI:** Pixtral 12B (2D Continuous RoPE, SwiGLU, dynamic aspect ratios).
  * **LLaVA Team:** LLaVA-1.5, LLaVA-NeXT, LLaVA-OneVision (CLIP/SigLIP ViT + GELU MLP).
  * **OpenGVLab:** InternVL 2.5, InternVL 3, InternVL 4 (PixelShuffle 2x2 downsampling).
  * **OpenBMB:** MiniCPM-V 2.6 (Dynamic HD 9-slice grid + 2D sinusoidal cross-attention Resampler).
  * **Zhipu AI:** GLM-4V, GLM-4.5V, GLM-OCR (Dual Conv2D stem, 2D M-RoPE, Conv2D patch merger).
  * **NVIDIA:** Nemotron-V2-VL / Nemotron-4-Nano (Learned register tokens + 2x2 merge + Squared ReLU MLP).
  * **Baidu / Dots:** PaddleOCR-VL, Dots-OCR (2D M-RoPE + patch merger + GELU MLP).
  * **Moonshot AI:** Kimi K2.5 / Kimi-VL (3D learned position embeddings + 2D interleaved RoPE).
  * **Google & Meta:** Gemma 4 UV (`gemma4uv`), Gemma 4 ViT (`gemma4v`), Gemma 3 SigLIP (`gemma3`), and Llama 4 (`llama4`).
* 🎙️ **Studio Audio & Voice Stack:**
  * **Speech-to-Text (ASR):** Whisper (Large-v3 / Turbo), NVIDIA NeMo Parakeet FastConformer CTC, Alibaba Qwen3-ASR (0.6B/1.7B), Qwen3-ForcedAligner (word timestamps), FunASR Paraformer, SenseVoice, and Silero VAD.
  * **Neural Voice (TTS) and voice cloning:** Qwen3-TTS 12Hz (includes a speaker encoder), Coqui XTTS-v2 (GPT2 autoregressive codec + FiLM-conditioned HiFi-GAN, zero-shot voice cloning; **voice cloning is verified end to end for XTTS-v2 only**, see STATUS.md), Kokoro-82M, Chatterbox-Turbo, F5-TTS Flow-Matching DiT, CosyVoice 2.0 / 300M, Piper VITS, Meta MMS-TTS (Massively Multilingual VITS), and MeloTTS Multilingual VITS.
  * **Broadcast DSP:** Rational windowed-sinc resamplers, ATSC A/85 downmixing, and TPDF dithered 16-bit/24-bit WAV exporter.
* 🎨 **State-of-the-Art Diffusion & Video:**
  * **Image Architectures:** SD 1.5, SDXL, SD 3 / 3.5 (MMDiT), FLUX.1 (schnell/dev), FLUX.2 (Klein & Kontext multi-reference), and Z-Image-Turbo.
  * **Acoustic Diffusion:** Stable Audio 3 (Variable-Length 44.1kHz Stereo DiT).
  * **Video Diffusion:** Wan 2.1/2.2 Video, HunyuanVideo, and LTX-Video.
  * **Streaming:** Latent Consistency Models (LCM, 1-4 step) and `StreamBatchPipeline` for continuous image streams. No measured frame rate is recorded in STATUS.md, so none is claimed here.
  * **Exporters:** Animated GIF and animated PNG sequence writers with Floyd-Steinberg dithering.
* ⚡ **Hardware selection:** SIMD kernels (AVX-512, AVX2, ARM NEON), a cached GPU topology probe, and `SmartOffloadPlanner`, which decides how many layers go on a discrete GPU, an APU or the CPU.


---

## Samples

The [sample catalogue](https://github.com/opentail-net/OpenTail.Stingray/blob/main/samples/README.md) says, for every sample, who it is for, which model files it needs, how to run it and how far it has been verified. In short:

* **Chat, Speech, Diffusion and ChatServer** are written against the public package API. They build from source against project references by default, and `scripts/verify-package-consumers.ps1` rebuilds them against the packed NuGet package to prove the public API is enough.
* **QuickStart and ToolCall** use engine internals and are for contributors.
* **HotRouting** is a simulation of scheduling behaviour, not an inference benchmark.

## Verification and provenance

Language models are compared with llama.cpp token by token wherever a reference run exists, and speech and diffusion models are checked against their reference implementations or recorded outputs. The per-model evidence, with dates and the machine it was measured on, is in [docs/STATUS.md](https://github.com/opentail-net/OpenTail.Stingray/blob/main/docs/STATUS.md). Where nothing is recorded there, treat the model as unverified.

---

## License

MIT License. Copyright (c) 2026 OpenTail. Model files have their own licences; check each model's page.
