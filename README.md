# OpenTail.Stingray

**Local AI for .NET.** Language models, vision, speech, music, sound effects and image generation,
all running inside your own .NET process.

**No Python. No llama.cpp. No sidecar server.** The engine is managed C# that runs on your CPU, or
your GPU through Vulkan or CUDA. (The native pieces that can still be involved are your GPU driver,
and ONNX Runtime for the ONNX-based speech engines such as Piper.)

[![NuGet](https://img.shields.io/nuget/v/OpenTail.Stingray.svg)](https://www.nuget.org/packages/OpenTail.Stingray)
[![.NET 10](https://img.shields.io/badge/.NET-10-blue)](https://dotnet.microsoft.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

```text
> stingray setup chat
> stingray chat "In one sentence, why do developers write unit tests?"
Developers write unit tests to ensure individual components work as intended, facilitating code
quality, maintainability, and testing of software components.
```

That is a 469 MB model, downloaded and checksum-verified by `stingray setup chat`. On an AMD Ryzen 7
5700G desktop (integrated graphics, which `stingray chat` picked automatically through Vulkan) the
second command took 5 to 6 seconds in total, including loading the model (measured 2026-10-11).
Sampling is random, so the wording differs from run to run.

## Quick start: chat from C#

You need the [.NET 10 SDK](https://dotnet.microsoft.com/download) and a 64-bit x86 CPU with AVX2
(most PCs from 2015 on). A GPU is optional.

**1. Create a project and add the package**

```bash
dotnet new console -n HelloStingray && cd HelloStingray
dotnet add package OpenTail.Stingray
```

**2. Download a small model** (Qwen2.5 0.5B Instruct, 469 MB). The `stingray` CLI
fetches verified bundles straight into your local model store:

```bash
dotnet tool install -g OpenTail.Stingray.Cli
stingray setup chat
```

This verifies SHA-256 integrity and stores the model in `ModelHome`. You can also pull any GGUF with `stingray pull -r <repo>` or download it manually.

**3. Replace `Program.cs`**

<!-- verify:consumer -->
```csharp
using OpenTail.Stingray;
using OpenTail.Stingray.Core.Catalog;
using OpenTail.Stingray.Executors;

// Resolve model from CLI argument or default ModelHome bundle:
string modelPath = args.Length > 0 
    ? args[0] 
    : ModelHome.ResolveModelPath("qwen2.5-0.5b");

// Load the model, allocate execution context, and drive interactive chat:
using var model = Model.Load(modelPath);
using var context = model.CreateContext(new ContextParams { ContextSize = 4096 });
var executor = new InteractiveExecutor(context);
var session = new ChatSession(executor);

await foreach (string piece in session.ChatAsync("In one sentence, why do developers write unit tests?"))
    Console.Write(piece);
```

For advanced applications requiring internal thinking tokens or detailed usage metrics, use `session.ChatChunksAsync(...)` which yields typed `GenerateChunk` streams.

*(Contributors who want to see the engine internals wired by hand (`GgufModel`, `CpuBackend`, `ForwardPass`, `InferenceEngine`) can read [samples/QuickStart](samples/QuickStart/README.md). It is not a package example: it uses types that are not part of the public API.)*

**4. Run it**

```bash
dotnet run
```

## Three Ways to Run

Stingray offers three integrated interfaces depending on your workflow:

| Interface | Package / Command | Best For |
|---|---|---|
| **In-Process Library** | `dotnet add package OpenTail.Stingray` | Language models, vision, speech, music, sound effects and image generation, embedded directly in your .NET application, with no Python, no llama.cpp and no separate server (the GPU driver, and ONNX Runtime for ONNX-based speech engines, are the native pieces involved). |
| **Command-Line CLI** | `dotnet tool install -g OpenTail.Stingray.Cli` (`stingray`) | Immediate terminal chat, path-free model setup (`stingray setup chat`), local inventory, and batch scripting. |
| **Local API Server** | `dotnet add package OpenTail.Stingray.Server` | Hosting OpenAI-compatible (`/v1/chat/completions`) endpoints for existing tools and web UIs with zero sidecars. |

**New here? [What can I do with Stingray?](docs/WHAT-YOU-CAN-DO.md)** is a one-page tour of every task it
handles, what you need and where the limits are, and the [task guides](docs/guides/README.md) show how to do
each one. The [samples](samples/README.md) are runnable programs, and say which of them use only the
public package API and how far each was verified. If something goes wrong, see
[troubleshooting](docs/TROUBLESHOOTING.md).

## Why Stingray

- **It's just a NuGet package.** `dotnet add package OpenTail.Stingray` and your app can run
  models. For CPU use there is nothing else to install on the user's machine or keep in sync, and it
  publishes with NativeAOT into a single executable.
- **One library for every modality.** Language models, vision, speech, music, sound effects and
  image generation come from the same package, so you don't glue five tools together.
- **It reads the models you already have, in the format you have them.** GGUF files (the
  llama.cpp format), Hugging Face SafeTensors folders, the weight files in a local Ollama store
  (they are GGUF), ONNX voices and whisper.cpp `.bin` models, with no conversion step and no
  re-download. What each format can do, and where it stops, is in the
  [format table](#formats-it-reads) and the guide [Use the models you already have](docs/guides/using-models-you-already-have.md).
- **Checked against the reference implementations.** Where a llama.cpp reference run exists,
  language models are compared with it token by token; speech and diffusion models are checked
  against their reference implementations or recorded outputs. What is verified, and how, is
  recorded model by model in [docs/STATUS.md](docs/STATUS.md). Where nothing is recorded there,
  treat a model as unverified.

## Speak and listen

**Text to speech** with a [Piper](https://huggingface.co/rhasspy/piper-voices) voice. Download
both files of a voice, for example `en_US-lessac-medium.onnx` (63 MB) and
`en_US-lessac-medium.onnx.json`
([here](https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/lessac/medium)):

<!-- verify:consumer -->
```csharp
using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Audio.Piper;

using var tts = PiperPipeline.FromConfigFile("en_US-lessac-medium.onnx.json");
tts.Generate(new AudioGenerationRequest { Text = "Hello from a voice that never left this computer.", OutputPath = "hello.wav" });
```

**Speech to text** with Whisper. Download
[`ggml-base.bin`](https://huggingface.co/ggerganov/whisper.cpp/blob/main/ggml-base.bin) (148 MB):

<!-- verify:consumer -->
```csharp
using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Audio.Whisper;

using var stt = WhisperPipeline.Load("ggml-base.bin");
var (samples, sampleRate, _) = WavReader.ReadWav("hello.wav");
Console.WriteLine(stt.Transcribe(new SpeechToTextRequest { AudioSamples = samples, SampleRate = sampleRate }).Text);
// Hello from a voice that never left this computer.
```

On the test machine below (measured 2026-10-11), the voice generated 2.7 seconds of audio in 1.8
seconds, and Whisper transcribed it back word for word. Short sentences with ordinary words come back
exactly; proper nouns and brand names often do not (it heard "OpenTail Stingray" as "Benail Tina").

## Serve an OpenAI-compatible API

Add `OpenTail.Stingray.Server` to an ASP.NET project, and existing OpenAI clients can talk to a
local model:

<!-- verify:consumer -->
```csharp
using OpenTail.Stingray.Server;

var builder = WebApplication.CreateBuilder(args);
builder.Services.AddOpenTailStingray(builder.Configuration, o => o.ModelPath = "models/qwen2.5-0.5b-instruct-q4_k_m.gguf");

var app = builder.Build();
app.MapOpenTailStingray();   // /v1/chat/completions, /v1/models, Anthropic and Responses APIs
app.Run("http://localhost:5080");
```

```bash
curl localhost:5080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"local","messages":[{"role":"user","content":"Name three planets, comma-separated."}]}'
# ... "content":"Mercury, Venus, and Earth." ...
```

See [samples/ChatServer](samples/ChatServer/Program.cs).

## From the command line

The easiest way to get started is the task-based front door:

```bash
dotnet tool install -g OpenTail.Stingray.Cli

# 1. Download recommended models with verified checksums
stingray setup chat          # Qwen2.5 0.5B Instruct, 469 MB
stingray setup speak         # Piper en_US-lessac-medium, 60 MB (asks you to accept the voice data licence)
stingray setup transcribe    # Whisper base, 141 MB
stingray models              # check status of installed tasks

# 2. Run them path-free
stingray chat                # interactive multi-turn chat
stingray chat "What is a unit test?"
stingray speak "Hello from Stingray." -o hello.wav
stingray transcribe hello.wav
```

Advanced and path-oriented workflows remain available:

```bash
stingray pull -r Qwen/Qwen2.5-0.5B-Instruct-GGUF
stingray -m models/qwen2.5-0.5b-instruct-q4_k_m.gguf                        # interactive chat
stingray -m models/qwen2.5-0.5b-instruct-q4_k_m.gguf -p "What is a unit test?"
stingray tts -e piper -m en_US-lessac-medium.onnx.json -t "Hello!" -o hello.wav
stingray stt -m base --model-file ggml-base.bin -i hello.wav
```

Add `-g -1` to run a language model on the GPU. Flag names follow llama.cpp's `llama-cli` where
they mean the same thing. Model assets live in a per-user folder (`%LOCALAPPDATA%\stingray\models` or
`~/.cache/stingray/models`; set `STINGRAY_MODEL_HOME` to move it).

## Finding models

There are two lists, and they are different on purpose.

- **The starter catalogue** is what `stingray models` shows: a short, curated list for three tasks
  (chat, speak, transcribe), each with a recommended model and a few alternatives. Every entry is
  pinned to an exact file revision with a SHA-256, shows its licence, RAM need and measured speed,
  and says what was checked, or that nothing was. `stingray setup <task>` installs one, the path-free commands run it,
  and `stingray models use <task> <id>` keeps your favourite.
- **Everything else the engine can run** is tracked in [docs/STATUS.md](docs/STATUS.md), model by
  model. An architecture the engine has admitted is not the same as a model we recommend or have
  verified on your backend: read the row before you rely on it.

**[docs/MODELS.md](docs/MODELS.md)** is the curated list for humans, one table per task (chat,
images, speech, transcription, image generation, search), linking straight to the files on Hugging
Face. **[docs/RUNNING.md](docs/RUNNING.md)** says how to run each one well: the command, the RAM it
really needs and its measured speed. For how a model family works inside Stingray (the GGUF
architecture keys it uses, which classes run it, checkpoints and measured results), see the
**[architecture cards](docs/models/README.md)**.

### Formats it reads

You do not have to convert or re-download what you already have. Point `-m` (CLI) or `Model.Load` (C#) at it:

| Format | What it is for | What was checked (2026-10-11), and where it stops |
|---|---|---|
| **GGUF** | Language models and vision models (with their `mmproj`), and many speech, music and image checkpoints. Quantised weights; CPU, Vulkan and CUDA. | The main path, and most of [docs/STATUS.md](docs/STATUS.md). 103 text-generation GGUFs from many families were loaded and run in an all-checkpoint sweep (CPU 101, Vulkan 85). |
| **SafeTensors** (Hugging Face folders) | Image and video generation, speech and vision models, and language models of a few dense families (Llama and Mistral, Qwen2 and 3, GPT-2). | A SmolLM2-135M folder ran on the CPU. For language models only unquantised weights (F32, F16, BF16) on the CPU are supported; use GGUF for quantised or GPU runs. `stingray capabilities` lists the exact profiles. |
| **Ollama store** | Models you already pulled with Ollama, with no Ollama daemon needed. | An Ollama weight file is a GGUF: `-m ~/.ollama/models/blobs/sha256-<digest>` (the digest of the model layer in the manifest) ran a Qwen3-VL 2B model. **Loading through the manifest file itself does not work in the CLI yet** (it fails with an invalid-GGUF error), and the library's manifest reader has been tested only on synthetic fixtures. |
| **ONNX** | Speech engines such as Piper. | Piper ran. It needs ONNX Runtime, a native component that ships as a dependency of the package. |
| **whisper.cpp `.bin`** | Whisper speech recognition. | `ggml-base.bin` ran (the speech sample). |
Stingray uses a model file where it is: read-only and memory-mapped, never copied into its own model folder. Only `stingray setup` and `stingray pull` download into Stingray's model folder.
**How to point Stingray at models already on your disk (any folder of GGUF files, an Ollama store, Hugging Face downloads), and what was and was not verified: [Use the models you already have](docs/guides/using-models-you-already-have.md).**

### Trying a model that is not in the catalogue

Check it before you download it, then fetch exactly what you checked:

```bash
stingray scout -r owner/repo --quants --budget 32G      # architecture, blockers, which quantisations fit this machine
stingray pull  -r owner/repo --revision <commit> -q Q4_K_M   # the commit scout printed, so you get the file it inspected
stingray -m models/<file>.gguf -p "Hello"               # run it; add -g -1 for the GPU
```

`scout` reads only the file index (no weights) and never decides anything for you: it reports what
it found, and `--budget` is what makes it judge whether the model fits your RAM. If the
architecture is not admitted yet, [docs/reference/adding-an-architecture.md](docs/reference/adding-an-architecture.md)
and [CONTRIBUTING.md](CONTRIBUTING.md) describe the evidence a new architecture needs (a real
checkpoint, an independent reference, timed runs) before a support claim is made.

## What else it can do

The recipes above are the verified starting points. The engine covers much more, at varying
levels of polish:

- **Language models**: Llama, Qwen, Gemma, Mistral, Phi, DeepSeek, gpt-oss, GLM-4, Granite 4.0-H, Nemotron-H, LFM2 and many more GGUF
  architectures, on CPU, Vulkan or CUDA. Tool calling, JSON-schema constrained output, speculative
  decoding. How each family works: [model architecture cards](docs/models/README.md).
- **Vision**: images in, text out. Gemma, Qwen-VL, LLaVA, Pixtral, InternVL, OCR models such as dots.ocr. How it works: [vision card](docs/models/multimodal-vision.md).
- **Speech**: text-to-speech engines including Kokoro, XTTS-v2 (voice cloning), Qwen3-TTS, Fish
  Speech and Chatterbox; speech recognition with Whisper, Parakeet and Qwen3-ASR. How it works: [audio card](docs/models/audio-pipelines.md).
- **Music and sound effects**: ACE-Step 1.5, Stable Audio 3 (music and SFX), MiniMax-Music3 (with vocals), MusicGen and AudioGen. From C# today; no command-line entry yet. **Listen to real clips in [docs/audio-samples](docs/audio-samples/README.md).**
- **Image and video generation**: FLUX.1 / FLUX.2, Stable Diffusion 1.5 / XL / 3.5, Z-Image-Turbo, Qwen Image, Wan, LTX-Video,
  HunyuanVideo. These need several model files each; a guided setup is on the way. See the [image and video samples](docs/diffusion-samples/README.md); how it works: [diffusion card](docs/models/diffusion-pipelines.md).

What is verified, what is partial and what is experimental is tracked per model, with dated
evidence, in **[docs/STATUS.md](docs/STATUS.md)**. Speed comparisons between the speech engines
are in the same file.

## Hardware and speed

The timings in this README come from one machine: an AMD Ryzen 7 5700G (8 cores, integrated
graphics only) with 64 GB of RAM, on Windows 11. The table below is the CPU; the chat example at the
top used the same machine's integrated graphics. Rough guide:

| You want to | Download | RAM | On that CPU |
|---|---|---|---|
| Chat with a small model | 0.5 GB | 2 GB | ~60-70 tokens/s (Qwen2.5 0.5B) |
| Chat with a 7–8B model | 4–5 GB | 8 GB | about 6-8 tokens/s. On this integrated GPU, generating is no faster than the CPU (memory bandwidth is shared), though reading a long prompt is about 20% faster; a discrete GPU is untested ([measurements](PerformanceLeague.md)) |
| Text to speech (Piper) | 63 MB | < 1 GB | faster than real time |
| Speech to text (Whisper base) | 148 MB | < 1 GB | ~2x real time |
| Image generation | 5–12 GB | 16 GB+ | minutes per image; a GPU is recommended |

A GPU is optional: any Vulkan-capable card, or NVIDIA with CUDA 12.

### Pick a backend

| Your Hardware | Backend (`--backend`) | Execution Details |
|---|---|---|
| **CPU only** (x86-64) | the default for `stingray -m ...` (`-g 0`); no `--backend` value needed | Managed C# with AVX2 and AVX-512 SIMD kernels. Needs no GPU. |
| **AMD, Intel, or NVIDIA GPU** | `--backend vulkan` | SPIR-V compute shaders via Vulkan (needs a Vulkan driver). Tested on an AMD integrated GPU only. |
| **NVIDIA GPU** | `--backend cuda` | cuBLAS and CUDA 12.x kernels (needs the CUDA 12 runtime). **Unverified here: there is no NVIDIA card on the development machine.** |
| **Automatic selection** | `--backend auto` (the default when `-g` is set) | Picks CUDA when `-g` is set and CUDA is available, otherwise Vulkan; `-g -1` offloads every layer. |

## Documentation & Architecture

| Document | What's inside |
|---|---|
| **[docs/README.md](docs/README.md)** | Central documentation index, task guides, and technical references. |
| **[docs/STATUS.md](docs/STATUS.md)** | Model support matrix with dated verification evidence and confidence tiers. |
| **[docs/RUNNING.md](docs/RUNNING.md)** | Recommended execution commands, memory requirements, and measured speeds. |
| **[docs/WHAT-YOU-CAN-DO.md](docs/WHAT-YOU-CAN-DO.md)** | Plain-language tour of what it can do: language models, vision, speech, music, sound effects and image generation. |
| **[docs/MODELS.md](docs/MODELS.md)** | Curated, tested starter models with download links, licenses, and file sizes. |
| **[docs/models/](docs/models/README.md)** | Architecture cards: one page per model family with checkpoints, classes, memory and measured results. |
| **[CONTRIBUTING.md](CONTRIBUTING.md)** | Ways to help, by what you have, and what happens to your work. |
| **[DEVELOPMENT.md](DEVELOPMENT.md)** | Contributor guide for solution layout, SIMD internals, and test harnesses. |

## Get involved

Stingray is MIT-licensed and there are several ways to help that need no C#: run a model and tell us
what happened, measure on hardware we do not have (NVIDIA, a discrete GPU, ARM64), fix a number in the
docs, or triage a GGUF we do not yet support. See **[CONTRIBUTING.md](CONTRIBUTING.md)** for the paths
and **[docs/WANTED.md](docs/WANTED.md)** for what is open right now.

## Building from source

```bash
git clone https://github.com/opentail-net/OpenTail.Stingray && cd OpenTail.Stingray
dotnet build -c Release
dotnet run --project samples/QuickStart -c Release -- chat models/qwen2.5-0.5b-instruct-q4_k_m.gguf "Hello"
```

The [documentation index](docs/README.md) sorts `docs/` into user guides, design notes and active
engineering work; contributors should read
[CLAUDE.md](CLAUDE.md) for the build and test conventions.

## License

MIT — Copyright (c) 2026 OpenTail. Model files have their own licenses; check each model's page.
