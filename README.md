# OpenTail.Stingray

**Local LLM inference, multimodal vision, speech, and image generation, written in C#.** No Python, no P/Invoke to llama.cpp, no
sidecar process — the engine itself is managed .NET 10 code that runs in your process and
NativeAOT-publishes to a single binary.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![.NET 10](https://img.shields.io/badge/.NET-10-blue)]()
[![NativeAOT](https://img.shields.io/badge/NativeAOT-ready-green)]()
[![NuGet](https://img.shields.io/nuget/v/OpenTail.Stingray.svg)](https://www.nuget.org/packages/OpenTail.Stingray)

> **Built by [opentail.net](https://opentail.net)**

Runs GGUF and SafeTensors models on CPU (AVX2/AVX-512 SIMD) and GPU (Vulkan compute shaders or CUDA cuBLAS), with:
- **OpenAI- and Anthropic-compatible API server** (/v1/chat/completions, /v1/audio/speech, /v1/audio/transcriptions), native dynamic Multi-LoRA serving,
- **Native Multimodal Vision Understanding** across 17+ architectures:
  - Alibaba Qwen2.5-VL / Qwen3-VL
  - DeepSeek-OCR & DeepSeek-OCR2
  - Mistral Pixtral 12B
  - LLaVA-1.5, LLaVA-NeXT, LLaVA-OneVision
  - OpenGVLab InternVL 2.5 / 3 / 4
  - OpenBMB MiniCPM-V 2.6
  - Zhipu AI GLM-4V, GLM-4.5V, GLM-OCR
  - NVIDIA Nemotron-V2-VL / Nemotron-4-Nano
  - Baidu / Dots PaddleOCR-VL & Dots-OCR
  - Moonshot AI Kimi K2.5 / Kimi-VL
  - LG Exaone 4.5-VL
  - IBM Granite Vision 3.2 / Granite 4.0 Vision
  - Tencent Hunyuan-VL & Tencent Youtu-VL
  - Xiaomi MiMo-VL
  - StepFun Step3-VL
  - Google Gemma 4 unified (`gemma4uv`), Gemma 4 ViT (`gemma4v`), Gemma 3 SigLIP (`gemma3`), and Meta Llama 4 (`llama4`),
- **Native Speech-to-Text (ASR) & Forced Alignment**: OpenAI Whisper Large-v3 & Turbo with 100 languages, NVIDIA NeMo Parakeet FastConformer CTC/TDT, Alibaba Qwen3-ASR 0.6B/1.7B, Qwen3-ForcedAligner, FunASR Paraformer, SenseVoice, and Silero VAD,
- **Native Text-to-Speech (TTS) Synthesis & Voice Cloning**: Alibaba Qwen3-TTS 12Hz (with ERes2NetV2 192-dim voice cloning speaker encoder), Fish Speech S2 Pro, Kokoro-82M, Piper VITS, F5-TTS Flow-Matching DiT, CosyVoice 2.0 / 300M, Chatterbox-Turbo, Orpheus-TTS (SNAC codec), Parler-TTS Mini, and MeloTTS Multilingual VITS with clause-level real-time streaming,
- **Native Studio-Grade DSP**: Rational windowed-sinc resamplers, ATSC A/85 broadcast downmixing, and TPDF noise-shaped dithered WAV export,
- **Native Diffusion & Video Synthesis Pipelines**: Stable Diffusion 1.5 (with ControlNet conditioning), SDXL, SD 3/3.5, FLUX.1, FLUX.2 (Klein & Kontext Multi-Reference DiT), FLUX 3 Multimodal Video+Audio DiT, Stable Audio 3 Continuous MMDiT (Variable-Length 1s-6min 44.1kHz Stereo), Z-Image-Turbo, Qwen Image & Edit, Wan 2.1/2.2 Video, HunyuanVideo, and LTX-Video with zero-dependency Animated GIF and sequence exporters.

---

## Try it

```bash
dotnet tool install -g OpenTail.Stingray.Cli

stingray -m models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf -p "Once upon a time"
stingray -m models/Qwen3-8B-Q4_K_M.gguf -p "Explain mmap" -g -1     # all layers on GPU
stingray -m models/Qwen3-8B-Q4_K_M.gguf                             # interactive chat
stingray tts -t "Hello from OpenTail Stingray!" -v af_heart -o speech.wav # native TTS
```

Or use it as a library:

```bash
dotnet add package OpenTail.Stingray --version 1.0.6
```

| Package | What it gives you |
|---|---|
| [`OpenTail.Stingray`](https://www.nuget.org/packages/OpenTail.Stingray) | The engine — all core, vision, audio, diffusion, and hardware assemblies in one package |
| [`OpenTail.Stingray.Cli`](https://www.nuget.org/packages/OpenTail.Stingray.Cli) | The `stingray` command-line tool |
| [`OpenTail.Stingray.Server`](https://www.nuget.org/packages/OpenTail.Stingray.Server) | ASP.NET Core endpoints — OpenAI, Anthropic and Responses APIs |

**Coming from llama.cpp?** Flag names follow `llama-cli` where the meaning matches, single-dash
spellings included (`-ngl`, `-ctk`, `-md`, `-fa`). Flags that cannot be honoured are refused with a
named reason rather than silently ignored.

**Requirements:** .NET 10, x86-64 with AVX2. GPU is optional — any Vulkan-capable card, or NVIDIA
with CUDA 12.x. Building from source needs the .NET 10 SDK and `dotnet build -c Release`.

---

## License

MIT License — Copyright (c) 2026 OpenTail.
