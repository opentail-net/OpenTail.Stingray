# Model Architecture Cards

> **Purpose:** Detailed architectural reference for supported model families in OpenTail.Stingray.  
> **Audience:** Contributors, performance engineers, and developers integrating specific model families.  
> **Governing Rule:** Every card documents how Stingray's managed C# engine *actually executes* the architecture, including tensor layout, custom kernels, RoPE parameters, and verified status.

---

## Supported Architecture Cards

| Model Family | Key Architectural Focus | Status & Implementation Highlights | Architecture Card |
|---|---|---|---|
| **Qwen Series** (2.5, 3.5 MoE) | SwiGLU, sliding-window attention, M-RoPE, Jinja chat templates | Full CPU SIMD & Vulkan support; tool calling and structured BNF outputs | [qwen-series.md](qwen-series.md) |
| **DeepSeek Series** (V2, V3, R1) | Multi-head Latent Attention (MLA), DeepSeekMoE routing, thinking mode | 3.5× KV compression via latent vector caching; top-k expert dispatch | [deepseek.md](deepseek.md) |
| **Multimodal Vision** (11+ architectures) | Unified Vision Tower, patch embeddings, spatial merging | Qwen-VL, Pixtral, DeepSeek-OCR, LLaVA, InternVL token projection into text sequence | [multimodal-vision.md](multimodal-vision.md) |
| **Audio & Speech Pipelines** (TTS & STT) | VITS phonemizer, Whisper encoder-decoder, real-time streaming DSP | Piper (0.129× RTF), Kokoro-82M, Whisper (Tiny–Large-v3), rational windowed-sinc resamplers | [audio-pipelines.md](audio-pipelines.md) |
| **Diffusion & Video Pipelines** | Flow-matching DiT, Euler schedulers, VAE tiling, text conditioning | End-to-end: SD 1.5/XL/3.5, FLUX.1, Wan, Z-Image-Turbo; Component verified: Qwen Image; Active: FLUX.2 | [diffusion-pipelines.md](diffusion-pipelines.md) |

---

## Verification Criteria

Each architecture card cross-references:
1. Active entry and test evidence in [docs/STATUS.md](../STATUS.md).
2. Measured memory and speed in [docs/RUNNING.md](../RUNNING.md).
3. Associated forward-pass test fixtures in `tests/OpenTail.Stingray.Tests.ForwardPass.Fast`.
