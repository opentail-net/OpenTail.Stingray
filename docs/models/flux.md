# FLUX Family (FLUX.1 schnell / dev & FLUX.2)

[← Back to Architecture Cards](README.md)

Related: [stable-diffusion](stable-diffusion.md), [z-image-turbo](z-image-turbo.md), [wan-video](wan-video.md). Verification: [STATUS.md](../STATUS.md); measured commands: [RUNNING.md](../RUNNING.md).

| Property | Value |
|---|---|
| **Provider** | Black Forest Labs (BFL) |
| **Model Formats** | GGUF (`flux1-schnell-Q4_K_S.gguf`) & SafeTensors |
| **Engine Implementations** | `OpenTail.Stingray.Diffusion.ImagePipeline`, `FluxDiT` |
| **Architecture Type** | 12B Rectified Flow Matching DiT with Joint Attention (MMDiT) |
| **Text Encoders** | Dual CLIP ViT-L/14 + T5-XXL |
| **Scheduler** | `EulerFlowScheduler` with dynamic resolution shifts |
| **Status & Confidence** | 🟢 **Admitted & Golden-Verified** (`FLUX.1 schnell` and `dev` end-to-end verified) |

> [!NOTE]
> **FLUX.2:** Actively supported in source with real weight loading across DiT, Mistral text conditioning taps, and VAE unshuffle (Tier 3 in `docs/STATUS.md`).
> **FLUX.3:** Strictly excluded (closed as unreleased speculative scaffolding per Rule 14).

---

## 1. Overview & Architectural Highlights

**Backends:** CPU and Vulkan GPU (see [STATUS](../STATUS.md)). Measured 2026-10-10 on a Ryzen 5700G iGPU (AMD Radeon, shared memory): FLUX.1-schnell Q4_K_S, 512×512, 4 steps, Vulkan: 158 s.

FLUX.1, created by Black Forest Labs (the original architects of Stable Diffusion), is the preeminent open-weights image generation model. It produces photorealistic imagery, accurate text rendering, and complex prompt adherence through a 12-billion-parameter rectified flow matching transformer.

### Key Architectural Characteristics
* **Joint Multimodal Transformer Blocks (MMDiT):**
  * Processes image latents and text conditioning tokens in a shared sequence with joint attention matrices, allowing text to attend directly to visual features and vice versa.
* **Dual Text Encoders:**
  * **OpenAI CLIP ViT-L/14:** Extracts pooled semantic embeddings for global image composition.
  * **Google T5-XXL:** Evaluates rich, long-form descriptive prompt representations up to 512 tokens.
* **RoPE Position Embeddings on Image Patches:**
  * 2D Rotary Position Embeddings are applied across spatial image patches, preserving geometric consistency across resolutions from 512×512 up to 2048×2048.
* **Rectified Flow Matching:**
  * Latents follow linear velocity trajectories, requiring only 4 steps in `FLUX.1-schnell` for crisp 1024×1024 generation.

---

## 2. Checkpoints & Recommended GGUFs

| Variant | Parameters | Steps | Recommended Quantization | Typical Size | Hugging Face Repository |
|---|---|---|---|---|---|
| **FLUX.1-schnell** | 12B | 4 steps | `Q4_K_S` / `Q5_K_M` | ~7.2 GB / ~8.9 GB | [black-forest-labs/FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) *(GGUF: [city96/FLUX.1-schnell-gguf](https://huggingface.co/city96/FLUX.1-schnell-gguf))* |
| **FLUX.1-dev** | 12B | 20–50 steps | `Q4_K_S` | ~7.2 GB | [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) *(GGUF: [city96/FLUX.1-dev-gguf](https://huggingface.co/city96/FLUX.1-dev-gguf))* |
| **VAE (AE)** | 83M | — | `ae.safetensors` | ~335 MB | [black-forest-labs/FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) |

---

## 3. Usage & Code Examples

### C# High-Level Generation

```csharp
using OpenTail.Stingray.Diffusion;

// 1. Load the FLUX.1 components: DiT (GGUF), VAE, CLIP-L and T5-XXL encoders, plus both tokenizers.
//    Replace the <...> placeholders with your local files. Optional last argument: an IComputeBackend (GPU).
using var pipeline = ImagePipeline.Load(
    "models/flux1-schnell-Q4_K_S.gguf",
    "models/flux-vae/ae.safetensors",
    "<clip-l encoder file>",  "<clip-l tokenizer file>",
    "<t5-xxl encoder file>",  "<t5-xxl tokenizer file>");

// 2. Generate a 1024x1024 image in 4 steps (width/height must be divisible by 16)
pipeline.Generate(
    "A majestic lion crowned in starlight, cinematic lighting, 8k resolution",
    width: 1024,
    height: 1024,
    steps: 4,
    guidance: 1.0f, // schnell uses guidance 1.0; dev uses ~3.5
    outputPath: "lion.png",
    progress: (step, total) => Console.Write($"\rStep {step}/{total}"));
```

### CLI Command

```bash
# Generate image with FLUX.1 schnell
stingray image -m models/flux1-schnell-Q4_K_S.gguf \
               --vae models/flux-vae/ae.safetensors \
               --steps 4 \
               -W 1024 -H 1024 \
               -p "A cyberpunk city street in the rain with glowing neon signs" \
               -o cyberpunk.png
```
