# FLUX Family (FLUX.1 schnell / dev & FLUX.2)

[← Back to Architecture Cards](README.md)

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

// 1. Initialize FLUX.1 pipeline
using var pipeline = ImagePipeline.Load(
    ditPath: "models/flux1-schnell-Q4_K_S.gguf",
    vaePath: "models/ae.safetensors",
    clipPath: "models/clip_l.safetensors",
    clipTokenizerPath: "models/clip_tokenizer.json",
    t5Path: "models/t5xxl_fp16.safetensors",
    t5TokenizerPath: "models/t5_tokenizer.json");

// 2. Generate 1024x1024 image in 4 steps
pipeline.Generate(new ImageGenerationRequest
{
    Prompt = "A majestic lion crowned in starlight, cinematic lighting, 8k resolution",
    Width = 1024,
    Height = 1024,
    Steps = 4,
    Guidance = 1.0f, // schnell uses guidance 1.0
    OutputPath = "lion.png",
    Progress = (step, total) => Console.Write($"\rStep {step}/{total}")
});
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
