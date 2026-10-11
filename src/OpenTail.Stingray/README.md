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

**What you do and do not need.** Stingray's own inference code (tensor kernels, tokenizers, KV caches, samplers, diffusion schedulers) is managed C#. You do not need Python, llama.cpp, ggml or a separate inference server. Some execution paths still load native code that is not Stingray's: the GPU driver (Vulkan loader, or the CUDA 12 runtime for the CUDA backend), and ONNX Runtime, which this package depends on for the ONNX-based speech engines such as Piper. A CPU run may still initialise the Vulkan backend at start-up and print a few diagnostic lines to stderr (for example the Vulkan device limits); they are not errors.

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

### Watching what the engine does

The engine reports what it did to an in-process event bus, so you can forward timings to your own logging or metrics without touching the hot paths. With no subscriber, emitting costs one volatile read.

<!-- verify:consumer -->
```csharp
using OpenTail.Stingray;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Core.Catalog;
using OpenTail.Stingray.Executors;

using var events = EngineEvents.Subscribe(new ConsoleSink());

using var model = Model.Load(args.Length > 0 ? args[0] : ModelHome.ResolveModelPath("qwen2.5-0.5b"));
using var context = model.CreateContext(new ContextParams { ContextSize = 2048 });
var executor = new InteractiveExecutor(context);
await foreach (var piece in executor.InferAsync("Name three colours.", new InferenceParams { MaxTokens = 24 }))
    Console.Write(piece);
Console.WriteLine();

// Keep sinks quick and thread-safe; an exception thrown here is swallowed so telemetry cannot break inference.
sealed class ConsoleSink : IEngineEventSink
{
    public void OnEvent(in EngineEvent e) => Console.Error.WriteLine(EngineEvents.ToJsonLine(e));
}
```

The event kinds are `ModelLoaded` (architecture, build time), `PrefillCompleted` and `DecodeCompleted` (token counts and times), `MtpStep` (speculative-decoding drafts proposed and accepted) and `GcStats`. **What the library emits today:** a package consumer calling `Model.Load` and an executor, as above, receives `ModelLoaded` (the only one verified from a library run: the snippet printed `{"event":"ModelLoaded","name":"qwen2","ms":2956,...}`). The prefill, decode and GC events are emitted by the command line (`STINGRAY_EVENTS=1` prints them as JSON lines), not by the executors, so time a library call yourself with `Stopwatch` and the token count. **What it does not report:** GPU memory use or per-token timings or traces, and it is not an OpenTelemetry or `ILogger` integration (forward events from your sink yourself). The server host keeps its own request metrics.

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

Running modern local AI usually means several separate tools: one for text, one for speech, one for voice synthesis, another for image and video generation. Stingray puts them behind one .NET package so a single application can use all of them in-process. The tensor kernels, KV caches, samplers, tokenizers, diffusion schedulers and audio processing are written in C# that you can step into and profile.

### What the code contains

- **Language models:** Llama, Mistral and Mixtral, Qwen, Gemma, DeepSeek, Phi, gpt-oss, GLM, Granite, LFM2 and others, including mixture-of-experts models; tool calling, JSON-schema constrained output and grammar masks.
- **Vision:** projector pipelines for Qwen-VL, DeepSeek-OCR, Pixtral, LLaVA, InternVL, MiniCPM-V, GLM-V, Nemotron-VL, PaddleOCR-VL, dots.ocr, Kimi-VL, Gemma 3 and 4, and Llama 4.
- **Speech:** recognition (Whisper, Parakeet, Qwen3-ASR, FunASR Paraformer, SenseVoice, Silero VAD) and synthesis (Piper, Kokoro, Chatterbox, F5-TTS, CosyVoice, XTTS-v2, Qwen3-TTS, Fish Speech, MeloTTS, MMS-TTS), plus resampling, downmixing and WAV export. Voice cloning is verified end to end for XTTS-v2 only.
- **Generation:** images (Stable Diffusion 1.5, XL and 3.5, FLUX.1 and FLUX.2, Z-Image-Turbo, Qwen Image), video (Wan, LTX-Video, HunyuanVideo), music and sound effects (ACE-Step, Stable Audio 3, MiniMax-Music3, MusicGen, AudioGen), latent-consistency streaming, and GIF or animated-PNG export. No streaming frame rate is recorded in STATUS.md, so none is claimed.
- **Hardware selection:** SIMD kernels (AVX-512, AVX2, ARM NEON), a cached GPU topology probe, and `SmartOffloadPlanner`, which decides how many layers go on a discrete GPU, an APU or the CPU.

---

## Samples

The [sample catalogue](https://github.com/opentail-net/OpenTail.Stingray/blob/main/samples/README.md) says, for every sample, who it is for, which model files it needs, how to run it and how far it has been verified. In short:

* **Chat, Speech, Diffusion and ChatServer** are written against the public package API. They build from source against project references by default, and `scripts/verify-package-consumers.ps1` rebuilds them against the packed NuGet package to prove the public API is enough.
* **QuickStart and ToolCall** use engine internals and are for contributors.
* **HotRouting** is a demonstration of session-state retention across roles that uses a real model if one is present and otherwise falls back to a simulation; it is not an inference benchmark.

## Verification and provenance

Language models are compared with llama.cpp token by token wherever a reference run exists, and speech and diffusion models are checked against their reference implementations or recorded outputs. The per-model evidence, with dates and the machine it was measured on, is in [docs/STATUS.md](https://github.com/opentail-net/OpenTail.Stingray/blob/main/docs/STATUS.md). Where nothing is recorded there, treat the model as unverified.

---

## License

MIT License. Copyright (c) 2026 OpenTail. Model files have their own licences; check each model's page.
