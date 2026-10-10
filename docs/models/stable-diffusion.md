# Stable Diffusion Family (SD 1.5, SDXL, SD 3 / 3.5 MMDiT)

[← Back to Architecture Cards](README.md)

| Property | Value |
|---|---|
| **Provider** | Stability AI |
| **Model Formats** | SafeTensors (`.safetensors`) and GGUF |
| **Engine Implementations** | `OpenTail.Stingray.Diffusion.StableDiffusion.StableDiffusionPipeline`, `SDXLPipeline`, `SD3Pipeline` |
| **Backbones** | 2D UNet (SD 1.5 / SDXL) and Multimodal Diffusion Transformer (SD 3 / 3.5 MMDiT) |
| **Text Encoders** | CLIP ViT-L/14, OpenCLIP ViT-bigG, and T5-XXL |
| **Status & Confidence** | 🟢 **Admitted & Golden-Verified** (All components Level 2 Proven, end-to-end image generation verified) |

---

## 1. Overview & Architectural Highlights

The Stable Diffusion lineage spans three major architectural generations, all natively executable within OpenTail.Stingray without Python or external AUTOMATIC1111 / ComfyUI sidecars:

### Generation 1: Stable Diffusion 1.5 (Classic Latent UNet)
* **Resolution:** Native $512\times 512$ synthesis.
* **UNet Backbone:** 860M parameter cross-attention UNet operating in an $8\times$ downsampled latent space ($64\times 64\times 4$).
* **Text Encoder:** Single OpenAI CLIP ViT-L/14.
* **Performance:** Generates in ~18 seconds on modern AVX2 CPUs, sub-second on Vulkan / CUDA GPUs.

### Generation 2: SDXL (Stable Diffusion XL)
* **Resolution:** Native $1024\times 1024$ synthesis.
* **Dual Text Encoders:** Combines CLIP ViT-L with OpenCLIP ViT-bigG, concatenating pooled text embeddings and cross-attention tokens.
* **Micro-Conditioning:** Injects image dimensions, crop coordinates, and target aspect ratio as conditioning vectors into the cross-attention layers.

### Generation 3: Stable Diffusion 3 & 3.5 (MMDiT)
* **Architecture:** Replaces the classic UNet with a Multimodal Diffusion Transformer (MMDiT).
* **Separate Text & Image Streams:** Blocks maintain distinct attention weights for text representations and image latents before joining them in a shared self-attention layer.
* **Triple Text Encoders:** Uses CLIP ViT-L, OpenCLIP ViT-bigG, and Google T5-XXL.

---

## 2. Checkpoints & Recommended Models

| Model | Architecture | Native Resolution | File Format | Typical Size | Hugging Face Repository |
|---|---|---|---|---|---|
| **SD 1.5 Pruned EMA** | UNet 2D | 512×512 | `v1-5-pruned-emaonly.safetensors` | ~4.27 GB | [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) |
| **SDXL 1.0 Base** | UNet 2D Dual-CLIP | 1024×1024 | `sd_xl_base_1.0.safetensors` | ~6.94 GB | [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) |
| **SD 3.5 Medium** | MMDiT | 1024×1024 | `sd3.5_medium.safetensors` | ~5.80 GB | [stabilityai/stable-diffusion-3.5-medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) |
| **SD 3.5 Large** | MMDiT | 1024×1024 | `sd3.5_large.safetensors` | ~16.2 GB | [stabilityai/stable-diffusion-3.5-large](https://huggingface.co/stabilityai/stable-diffusion-3.5-large) |

---

## 3. Usage & Code Examples

### C# Image Generation (SD 1.5)

```csharp
using OpenTail.Stingray.Diffusion;
using OpenTail.Stingray.Diffusion.StableDiffusion;

// 1. Load pipeline from standalone SafeTensors checkpoint
using var pipeline = StableDiffusionPipeline.Load("models/v1-5-pruned-emaonly.safetensors");

// 2. Generate image
pipeline.Generate(new ImageGenerationRequest
{
    Prompt = "A vintage steam locomotive crossing a stone bridge in autumn, oil painting style",
    NegativePrompt = "blurry, low quality, distorted",
    Width = 512,
    Height = 512,
    Steps = 20,
    Guidance = 7.5f,
    OutputPath = "locomotive.png",
    Progress = (step, total) => Console.Write($"\rStep {step}/{total}")
});
```

### CLI Command

```bash
# Generate image from prompt
stingray image -m models/v1-5-pruned-emaonly.safetensors \
               -W 512 -H 512 --steps 20 \
               -p "A cozy cabin in snowy woods at twilight, warm chimney smoke" \
               -o cabin.png
```
