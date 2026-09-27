# OpenTail.Stingray

**Local AI for .NET.** Chat with language models, turn text into speech, transcribe audio, read
images and generate pictures, inside your own .NET process. There is no Python, no sidecar
server and no native binaries to ship: the engine is managed C# that runs on your CPU, or your
GPU through Vulkan or CUDA.

[![NuGet](https://img.shields.io/nuget/v/OpenTail.Stingray.svg)](https://www.nuget.org/packages/OpenTail.Stingray)
[![.NET 10](https://img.shields.io/badge/.NET-10-blue)](https://dotnet.microsoft.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

```text
> dotnet run -- chat qwen2.5-0.5b-instruct-q4_k_m.gguf "In one sentence, why do developers write unit tests?"
Developers write unit tests to ensure individual components of a program work as intended,
enhancing code quality, maintaining stability, and facilitating testing and debugging.
```

That answer came from a 469 MB model running on an ordinary desktop CPU, in about three seconds.

## Why Stingray

- **It's just a NuGet package.** `dotnet add package OpenTail.Stingray` and your app can run
  models. Nothing to install on the user's machine, nothing to keep in sync, and it publishes with
  NativeAOT into a single executable.
- **One library for text, speech and images.** The same package does chat, text-to-speech,
  speech-to-text, image understanding and image generation, so you don't glue five tools together.
- **It reads the models people already use.** GGUF files from Hugging Face (the llama.cpp
  format), plus the common ONNX/safetensors releases for speech and diffusion.
- **Checked against the reference implementations.** Language models are compared token by token
  with llama.cpp. What is verified, and how, is recorded model by model in
  [docs/STATUS.md](docs/STATUS.md).

## Quick start: chat from C#

You need the [.NET 10 SDK](https://dotnet.microsoft.com/download) and a 64-bit x86 CPU with AVX2
(most PCs from 2015 on). A GPU is optional.

**1. Create a project and add the package**

```bash
dotnet new console -n HelloStingray && cd HelloStingray
dotnet add package OpenTail.Stingray
```

**2. Download a small model** (Qwen2.5 0.5B Instruct, 469 MB). The `stingray` command-line tool
fetches GGUF files from Hugging Face:

```bash
dotnet tool install -g OpenTail.Stingray.Cli
stingray pull -r Qwen/Qwen2.5-0.5B-Instruct-GGUF
```

This saves `models/qwen2.5-0.5b-instruct-q4_k_m.gguf`. You can also download it from the
[model page](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF) by hand.

**3. Replace `Program.cs`**

```csharp
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Engine;

// Load the model and its tokenizer, and run it on the CPU.
using var model = GgufModel.Open("models/qwen2.5-0.5b-instruct-q4_k_m.gguf");
var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
var tokenizer = GgufTokenizer.FromGgufModel(model);
using var cpu = new CpuBackend();
var forward = new ForwardPass(model, cpu, hp, maxContextLength: 4096);
await using var engine = new InferenceEngine(forward, tokenizer, "qwen", forward);

// Format the question with the model's own chat template, then stream the answer.
string prompt = tokenizer.ChatTemplate!.Render(new Dictionary<string, object?>
{
    ["messages"] = JinjaChatTemplate.BuildMessages("In one sentence, why do developers write unit tests?"),
    ["add_generation_prompt"] = true,
});
await foreach (string piece in engine.GenerateAsync(prompt, new SamplingParams { Temperature = 0.7f, MaxNewTokens = 200 }))
    Console.Write(piece);
```

**4. Run it**

```bash
dotnet run
```

Yes, that is more wiring than it should be. A one-line `Open("model")` API is being designed
([docs/103](docs/3-product-and-runtime/103-front-door-design.md)). The code above is what works today, and it is
compiled and run as [samples/QuickStart](samples/QuickStart/Program.cs).

## Speak and listen

**Text to speech** with a [Piper](https://huggingface.co/rhasspy/piper-voices) voice. Download
both files of a voice, for example `en_US-lessac-medium.onnx` (63 MB) and
`en_US-lessac-medium.onnx.json`
([here](https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/lessac/medium)):

```csharp
using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Audio.Piper;

using var tts = PiperPipeline.FromConfigFile("en_US-lessac-medium.onnx.json");
tts.Generate(new AudioGenerationRequest { Text = "Hello from a voice that never left this computer.", OutputPath = "hello.wav" });
```

**Speech to text** with Whisper. Download
[`ggml-base.bin`](https://huggingface.co/ggerganov/whisper.cpp/blob/main/ggml-base.bin) (148 MB):

```csharp
using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Audio.Whisper;

using var stt = WhisperPipeline.Load("ggml-base.bin");
var (samples, sampleRate, _) = WavReader.ReadWav("hello.wav");
Console.WriteLine(stt.Transcribe(new SpeechToTextRequest { AudioSamples = samples, SampleRate = sampleRate }).Text);
// Hello from a voice that never left this computer.
```

On the test machine below, the voice generates 2.7 seconds of audio in 1.2 seconds, and Whisper
transcribes it back word for word.

## Serve an OpenAI-compatible API

Add `OpenTail.Stingray.Server` to an ASP.NET project, and existing OpenAI clients can talk to a
local model:

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

The same engine as a command-line tool, which is handy for trying models before writing code:

```bash
dotnet tool install -g OpenTail.Stingray.Cli

stingray pull -r Qwen/Qwen2.5-0.5B-Instruct-GGUF
stingray -m models/qwen2.5-0.5b-instruct-q4_k_m.gguf                        # interactive chat
stingray -m models/qwen2.5-0.5b-instruct-q4_k_m.gguf -p "What is a unit test?"
stingray tts -e piper -m en_US-lessac-medium.onnx.json -t "Hello!" -o hello.wav
stingray stt -m base --model-file ggml-base.bin -i hello.wav
```

Add `-g -1` to run a language model on the GPU. Flag names follow llama.cpp's `llama-cli` where
they mean the same thing.

## Finding models

**[docs/MODELS.md](docs/MODELS.md)** is a short, curated list of models to start with, one table
per task (chat, images, speech, transcription, image generation, search). Each entry links straight
to the file on Hugging Face and shows its size, licence and whether we have tested that exact file.
**[docs/RUNNING.md](docs/RUNNING.md)** then says how to run each one well: the command, the RAM it
really needs and its measured speed.

## What else it can do

The recipes above are the verified starting points. The engine covers much more, at varying
levels of polish:

- **Language models**: Llama, Qwen, Gemma, Mistral, Phi, DeepSeek, gpt-oss and many more GGUF
  architectures, on CPU, Vulkan or CUDA. Tool calling, JSON-schema constrained output, speculative
  decoding.
- **Images in, text out**: Gemma, Qwen-VL, LLaVA, Pixtral, InternVL, OCR models such as dots.ocr.
- **Speech**: text-to-speech engines including Kokoro, XTTS-v2 (voice cloning), Qwen3-TTS, Fish
  Speech and Chatterbox; speech recognition with Whisper, Parakeet and Qwen3-ASR.
- **Image and video generation**: FLUX, Stable Diffusion 1.5 / XL / 3.5, Z-Image-Turbo, Wan,
  HunyuanVideo. These need several model files each; a guided setup is on the way.

What is verified, what is partial and what is experimental is tracked per model, with dated
evidence, in **[docs/STATUS.md](docs/STATUS.md)**. Speed comparisons between the speech engines
are in the same file.

## Hardware and speed

The timings in this README come from an AMD Ryzen 7 5700G (8 cores, integrated graphics only)
with 64 GB of RAM, CPU only, on Windows 11. Rough guide:

| You want to | Download | RAM | On that CPU |
|---|---|---|---|
| Chat with a small model | 0.5 GB | 2 GB | ~24 tokens/s |
| Chat with a 7–8B model | 4–5 GB | 8 GB | a few tokens/s; a GPU helps a lot |
| Text to speech (Piper) | 63 MB | < 1 GB | faster than real time |
| Speech to text (Whisper base) | 148 MB | < 1 GB | ~2x real time |
| Image generation | 5–12 GB | 16 GB+ | minutes per image; a GPU is recommended |

A GPU is optional: any Vulkan-capable card, or NVIDIA with CUDA 12.

## What's next

A guided "front door" is being designed in [docs/103](docs/3-product-and-runtime/103-front-door-design.md): a small
catalog of verified models per task, `stingray setup <task>` to fetch them, task commands that need
no file paths, and a one-line C# API.

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
