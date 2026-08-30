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

* 💬 **Text LLMs & Sparse MoE:** Llama 3 / 3.1 / 3.2 / 3.3 / 4, Mistral 7B / Mixtral MoE / Ministral, Qwen 2.5 / Qwen 3.5 MoE, DeepSeek-V3 / R1, Gemma 3 / 4, SmolLM2. Full tool calling, JSON schema constrained decoding, and grammar masks.
* 👁️ **Multimodal Vision (11+ Architectures):**
  * **Alibaba:** Qwen2.5-VL, Qwen3-VL (3D Conv stem, M-RoPE, $2\times 2$ spatial merge).
  * **DeepSeek:** DeepSeek-OCR & DeepSeek-OCR2 (Dual SAM + CLIP ViT fusion, $1024\times 1024$ grid).
  * **Mistral AI:** Pixtral 12B (2D Continuous RoPE, SwiGLU, dynamic aspect ratios).
  * **LLaVA Team:** LLaVA-1.5, LLaVA-NeXT, LLaVA-OneVision (CLIP/SigLIP ViT + GELU MLP).
  * **OpenGVLab:** InternVL 2.5, InternVL 3, InternVL 4 (PixelShuffle $2\times 2$ downsampling).
  * **OpenBMB:** MiniCPM-V 2.6 (Dynamic HD 9-slice grid + 2D sinusoidal cross-attention Resampler).
  * **Zhipu AI:** GLM-4V, GLM-4.5V, GLM-OCR (Dual Conv2D stem, 2D M-RoPE, Conv2D patch merger).
  * **NVIDIA:** Nemotron-V2-VL / Nemotron-4-Nano (Learned register tokens + $2\times 2$ merge + Squared ReLU MLP).
  * **Baidu / Dots:** PaddleOCR-VL, Dots-OCR (2D M-RoPE + patch merger + GELU MLP).
  * **Moonshot AI:** Kimi K2.5 / Kimi-VL (3D learned position embeddings + 2D interleaved RoPE).
  * **Google & Meta:** Gemma 4 UV (`gemma4uv`), Gemma 4 ViT (`gemma4v`), Gemma 3 SigLIP (`gemma3`), and Llama 4 (`llama4`).
* 🎙️ **Studio Audio & Voice Stack:**
  * **Speech-to-Text (ASR):** Whisper (Large-v3 / Turbo), NVIDIA NeMo Parakeet FastConformer CTC, Alibaba Qwen3-ASR (0.6B/1.7B), Qwen3-ForcedAligner (word timestamps), FunASR Paraformer, SenseVoice, and Silero VAD.
  * **Neural Voice & Voice Cloning (TTS):** Qwen3-TTS 12Hz (with ERes2NetV2 192-dim voice cloning speaker encoder), Coqui XTTS-v2 (GPT2 autoregressive codec + FiLM-conditioned HiFi-GAN, zero-shot voice cloning), Kokoro-82M, Chatterbox-Turbo, F5-TTS Flow-Matching DiT, CosyVoice 2.0 / 300M, Piper VITS, Meta MMS-TTS (Massively Multilingual VITS), and MeloTTS Multilingual VITS.
  * **Broadcast DSP:** Rational windowed-sinc resamplers, ATSC A/85 downmixing, and TPDF dithered 16-bit/24-bit WAV exporter.
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
dotnet add package OpenTail.Stingray --version 1.0.6
```

---

## Quick Starts

### 1. Unified Multimodal Vision (Qwen-VL, DeepSeek-OCR, Pixtral, LLaVA, InternVL, MiniCPM, GLM-4V)

```csharp
using OpenTail.Stingray.Vision;

// 1. Open any multimodal vision GGUF projector (auto-detects architecture)
using var embedder = UnifiedVisionPipeline.Open("models/mmproj-deepseek-ocr-2-q8_0.gguf");

// 2. Load and embed an image into visual token embeddings
float[] visualTokens = embedder.EmbedImageFile("document.png", out int tokenCount);
Console.WriteLine($"Generated {tokenCount} visual tokens with embedding dim {embedder.EmbeddingDim}");
```

### 2. LLM Chat with Streaming

```csharp
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Engine;

// 1. Open GGUF model and initialize hardware backend
var model = GgufModel.Open("models/Llama-3.2-3B-Instruct-Q4_K_M.gguf");
var cpu   = new CpuBackend();
using var engine = new InferenceEngine(model, cpu);

// 2. Stream tokens in real time
await foreach (var token in engine.StreamChatAsync("Explain quantisation in simple terms."))
{
    Console.Write(token);
}
```

### 3. Native TTS Synthesis & Voice Cloning

```csharp
using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Audio.QwenTTS;

// 1. Initialize Qwen3-TTS pipeline
using var tts = new QwenTtsPipeline();

// 2. Generate 24kHz speech with reference audio voice cloning
var result = tts.Generate(new AudioGenerationRequest
{
    Text = "OpenTail Stingray is the unified, high-performance native .NET 10 multimodal AI engine.",
    ReferenceAudioPath = "samples/speaker_reference.wav",
    OutputPath = "output.wav"
});

Console.WriteLine($"Synthesized {result.Duration.TotalSeconds:F2}s of audio to output.wav");
```

---

## Verification & Provenance

Stingray rigorously validates models against real weight binary checkpoints on disk. See [`docs/048-model-provenance-and-real-weights-verification-plan.md`](docs/048-model-provenance-and-real-weights-verification-plan.md) for the complete provenance matrix and benchmark test runs.

---

## License

MIT License — Copyright (c) 2026 OpenTail.
