# What can I do with Stingray?

A plain-language tour for someone who has just found the project. It says what you can do, how you
would do it, what you need, and where the limits are. It is a map, not the evidence: every claim
here is backed by a dated row in [STATUS.md](STATUS.md) (how well it was checked) and
[RUNNING.md](RUNNING.md) (the exact command, the memory it needs and the speed we measured).

Stingray is a library, a command-line tool (`stingray`) and an OpenAI/Anthropic-compatible server,
all running the same engine in plain .NET with no Python and no native binaries. You pick the
interface that fits: a one-off try is the CLI, an app is the NuGet package, an existing OpenAI
client is the server.

## At a glance

| I want to… | What you get | How you use it | Typical model | Honest caveat |
|---|---|---|---|---|
| **Chat with a language model** | Streamed answers, tool calling, JSON-schema output | CLI, C#, server | Qwen2.5 0.5B to 8B, Qwen3, Gemma 3/4, Mistral, Phi, gpt-oss, Granite, LFM2 and more | Speed depends on the CPU. A small model is 60+ tokens/s here, a 7-8B model a few to ~9 tokens/s |
| **Get code help** | Same as chat, with code-tuned models | CLI, C#, server | Qwen2.5-Coder, Qwen3-Coder, Maincoder, StarCoder2 | Same speed rules as chat |
| **Ask about a picture or read a document** | Describe photos, answer questions, OCR | CLI (`--image` + `--mmproj`), C# | Gemma 3, Qwen2.5-VL / Qwen3-VL, InternVL3, LLaVA-1.5, Granite Vision, dots.ocr, PaddleOCR-VL | Needs two files (model + `mmproj`). Several vision models run on the CPU only; the server's image path is narrower than the CLI's |
| **Turn text into speech** | A `.wav` file; some engines clone a voice from a short clip | CLI (`stingray tts`), C#, server | Piper (fastest), Kokoro, Chatterbox, F5-TTS, XTTS-v2, Fish Speech, Qwen3-TTS, CosyVoice, Parler | Piper is faster than real time on a CPU; Kokoro and the large engines are slower than real time here (up to ~11x slower for Fish Speech) |
| **Transcribe speech** | Text with timestamps, translation to English | CLI (`stingray stt`), C#, server | Whisper (tiny to large-v3), Parakeet, Qwen3-ASR, SenseVoice, Paraformer, Voxtral | Whisper large-v3 runs slower than real time on this CPU; Parakeet and SenseVoice are much faster |
| **Search by meaning** | Text embeddings, reranking, CLIP image/text embeddings | CLI (`embed`, `rerank`), C#, server | Qwen3-Embedding, BGE, MiniLM, BGE-M3, ms-marco rerankers | |
| **Generate images** | PNG from a prompt, image-to-image, inpainting | CLI (`stingray image`: SD 1.5, SD 3.5, FLUX.1, Z-Image-Turbo), C#, server | Stable Diffusion 1.5/XL/3.5, FLUX.1/FLUX.2, Z-Image-Turbo, Qwen Image | Each model needs several files. On a CPU it is seconds for a 256×256 preview and minutes for a full image. A GPU helps a lot |
| **Generate short video** | Short clips from text | CLI for Wan (`stingray image --video-frames`), C# for the rest | Wan 2.1/2.2, LTX-Video, HunyuanVideo | Slow without a GPU: one 256×256 frame is 23 s on Wan 1.3B and over six minutes on Wan 14B on the reference CPU |
| **Make music and sound effects** | Audio from a text prompt | C# library (no CLI command yet) | ACE-Step 1.5, Stable Audio 3, MiniMax-Music3, MusicGen, AudioGen | Library-only today; ACE-Step 2 s of audio takes 31 s on the CPU |
| **Convert a voice** | Re-voice audio with a trained model | C# library | RVC | Library-only |
| **Forecast time series, classify images** | Numeric forecasts, labels | C# library | Chronos, timm MobileNetV3/EfficientNet, CLIP | Library-only |
| **Serve all of this over HTTP** | `/v1/chat/completions`, `/v1/audio/speech`, `/v1/images/generations`, `/v1/embeddings`, `/v1/messages`, `/v1/responses` and more | `OpenTail.Stingray.Server` | Any of the above | Endpoint list: [Server README](../src/OpenTail.Stingray.Server/README.md) |

## What do I need?

| Resource | What to expect |
|---|---|
| **CPU** | 64-bit x86 with AVX2 (most PCs since about 2015). ARM64 and Apple Silicon are not tested |
| **RAM** | A 0.5B chat model needs about 1 GB, a 7-8B model about 5-8 GB, a 30B mixture-of-experts model about 18 GB. Image and video models need 5-12 GB of files and 16 GB+ of RAM. [RUNNING.md](RUNNING.md) lists the measured peak for each model |
| **GPU** | Optional. Vulkan works on any vendor and is tested on an AMD integrated GPU. CUDA paths exist, but we have not been able to test them on an NVIDIA card, so treat CUDA as unverified. An integrated GPU is often slower than the CPU for small models |
| **Disk** | Models are the cost. A small chat model is under 1 GB, a good everyday one 2-5 GB, image and video models 5-30 GB |
| **.NET** | The .NET 10 SDK |

## How much can I trust it?

Stingray's rule is that a model is not called working until it has been checked against an
independent reference. The words in [STATUS.md](STATUS.md) mean:

- **Verified against a reference** (🔬): an automated test compares output with llama.cpp or another
  trusted implementation, down to the tensors or the token sequence.
- **Checked by ear or by reading** (👂): the whole pipeline ran on real weights and a person or a
  transcription confirmed the output is right.
- **Implemented, not yet checked** (⚪, 🔵): the code exists. Do not rely on it.

Two things to know. Verification was done on one machine (a Ryzen 7 5700G with integrated graphics),
so GPU and speed claims come from there. And a model that is not in STATUS.md has not been checked,
even if it happens to load.

## What does not work yet

- **Some architectures are not supported:** Gemma 1/2, Jamba, pure Mamba, Llama 4 vision,
  DeepSeek-V3.2/V4 (alpha code, never run) and GLM-4.5-class `glm4moe` models. See the open list in
  [00-current-work.md](00-current-work.md).
- **Known partial results:** DeepSeek-V2-Lite matched llama.cpp's greedy output with int8 prefill
  (the CPU default again since 2026-10-02, [ADR-0003](reference/adr-0003-cpu-int8-prefill-default.md)),
  but since the Q3_K kernel rewrite `e7b7aa8a` it departs at token 9 on the test prompt
  (decision pending, [bugstofix](1-correctness/bugstofix.md) item 24); CosyVoice 2 can garble the end of a sentence;
  MiMo-VL is partial.
- **Music, voice conversion and time-series models are library-only,** with no command-line entry.
- **Speculative decoding does not speed up this CPU** (it lacks the instructions it needs).
- **Prefix caching is off for hybrid and recurrent models** (Granite 4.0-H, Nemotron-H, LFM2),
  because their state cannot be rewound.

## Where next

1. **First run:** the [README](../README.md) quick start, or `stingray setup chat`. Then the [task guides](guides/README.md): chat, vision, structured output and tools, speech, search, images and video, serving.
2. **Pick a model:** [MODELS.md](MODELS.md). It is short and curated.
3. **Run it properly:** [RUNNING.md](RUNNING.md) has the command, memory and speed for each model.
4. **Check it is verified:** [STATUS.md](STATUS.md).
5. **Command-line options:** `stingray --help`, `stingray <command> --help`, and
   [reference/cli-option-inventory.md](reference/cli-option-inventory.md) for every option.
6. **Use it from C#:** the [package README](../src/OpenTail.Stingray/README.md) and the samples in
   [samples/](../samples).
7. **Serve it:** the [Server README](../src/OpenTail.Stingray.Server/README.md).
8. **Tune for your hardware:** [reference/recommended-configurations.md](reference/recommended-configurations.md).
9. **Something broke:** run `stingray doctor`, then see the [troubleshooting notes](TROUBLESHOOTING.md).
