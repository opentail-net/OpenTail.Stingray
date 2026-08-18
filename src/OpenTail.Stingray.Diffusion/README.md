# OpenTail.Stingray.Diffusion

Native, high-performance diffusion inference runtime for .NET 10 — 100% managed C#, running in-process with zero Python, zero subprocesses, zero P/Invoke, and full NativeAOT compatibility.

Supports the entire modern image and video diffusion landscape: **Stable Diffusion 1.5, SDXL, Stable Diffusion 3 / 3.5, FLUX.1, Z-Image-Turbo, Qwen Image & Edit, Wan 2.1 / 2.2 Video, and HunyuanVideo / 1.5**, accelerated natively on CPU (AVX2/AVX-512) and GPU (Vulkan / CUDA).

[![.NET 10](https://img.shields.io/badge/.NET-10-blue)]()
[![NativeAOT](https://img.shields.io/badge/NativeAOT-ready-green)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

> **Part of the [OpenTail.Stingray](https://github.com/opentail-net/OpenTail.Stingray) ecosystem by [opentail.net](https://opentail.net)**

---

## Supported Architectures & Models

| Model Family | Core Backbone | Text Conditioning | Schedulers | Latent Channels | VAE | Capabilities |
|---|---|---|---|:---:|:---:|---|
| **Stable Diffusion 1.5** | 4-Stage UNet + Spatial Cross-Attn | CLIP-L (768d, ViT-L/14) | Euler, Euler-A, DDIM, DPM++ 2M, DPM++ 2M Karras | 4 | 4-ch AutoencoderKL | Text-to-Image, Img2Img, ControlNet, LoRA |
| **Stable Diffusion XL (SDXL)** | 3-Stage UNet + 2048d Cross-Attn | Dual: CLIP-L (768d) + OpenCLIP-bigG (1280d) + Pooled (1280d) + Micro-Coords | Euler, Euler-A, DDIM, DPM++ 2M, DPM++ 2M Karras | 4 | 4-ch AutoencoderKL | Text-to-Image, Img2Img, LoRA |
| **Stable Diffusion 3 / 3.5** | MMDiT Dual-Stream + Single-Stream DiT | Triple: CLIP-L (768d) + OpenCLIP-bigG (1280d) + T5-XXL + Pooled (2048d) | Rectified Flow-Matching | 16 | 16-ch AutoencoderKL | Text-to-Image |
| **FLUX.1 (schnell / dev)** | MM-DiT Dual-Stream + Single-Stream DiT | Dual: CLIP-L + T5-XXL | Rectified Flow-Matching | 16 | 16-ch AutoencoderKL | Text-to-Image |
| **Z-Image-Turbo** | S3-DiT Scaled Transformer | Qwen3-4B LLM Text Encoder | Rectified Flow-Matching (4 steps) | 16 | 16-ch AutoencoderKL | Distilled Text-to-Image |
| **Qwen Image & Edit** | 60-layer MM-DiT + 3D-RoPE {16,56,56} | Qwen2.5-VL (3584d) | Rectified Flow-Matching (s=3.0) | 16 | 16-ch Qwen/Wan VAE | Text-to-Image, Image-to-Image Edit |
| **Wan 2.1 / 2.2** | Video DiT (1.3B / 14B) + 3D-RoPE {44,42,42} | UMT5-XXL (4096d) | Rectified Flow-Matching (s=3.0/5.0) | 16 | 16-ch Causal Wan VAE | Text-to-Video, Image-to-Video, Dual-Model A14B, Video Sequences |
| **HunyuanVideo / 1.5** | 54-layer Dual/Single DiT + 3D-RoPE {16,56,56} | Dual: LLaMA-3/Qwen-VL + CLIP-L/ByT5 | Rectified Flow-Matching (s=7.0) | 16 | 16-ch Causal VAE | Text-to-Video, Image-to-Video, Video Sequences |

---

## Key Capabilities

* **100% Native C# Engine:** Every tensor operation, cross-attention layer, normalization kernel, scheduler step, 3D RoPE calculation, and VAE decode runs directly inside the .NET 10 CLR / NativeAOT runtime.
* **Unified Abstractions (`IDiffusionPipeline`):** Consistent API across UNet, DiT, MMDiT, and Video DiT model families with standardized generation requests and progress reporting.
* **Video Generation & Frame Sequence Exporters:** Multi-frame video latents are automatically decoded through 16-channel 3D causal VAEs and exported both as primary anchor frames and complete numbered PNG sequences (`_frame_000.png`).
* **Universal Multi-Scheduler:**
  * **Euler & Euler Ancestral:** Discrete 1000-step linear beta schedule.
  * **DPM++ 2M & DPM++ 2M Karras:** 2nd-order Adams-Bashforth ODE solver with Karras $\sigma$-distribution ($\rho = 7.0$).
  * **DDIM:** Deterministic inversion and sampling trajectory.
  * **Rectified Flow-Matching:** Euler flow trajectory with configurable flow shift ($s = 3.0, 5.0, 7.0$).
* **LoRA Runtime Engine (`DiffusionLoraApplier`):** Load `.safetensors` LoRA weights and apply low-rank parameter deltas $\Delta W = \alpha \cdot \frac{1}{r}(A \times B)$ in-memory to base weights.
* **Universal VAE Subsystem:**
  * `VaeDecoder`: Decodes 4-channel and 16-channel latents to RGB pixels $[3, H, W]$ and video frames $[3, T, H, W]$.
  * `VaeEncoder`: Encodes input images to Gaussian latent distributions for **img2img**, **video initialization (I2V)**, and inpainting workflows.
* **Integrated Super-Resolution (`RRDBNet`):** Built-in ×2 and ×4 Real-ESRGAN upscaling with bicubic blending support.
