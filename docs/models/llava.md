# LLaVA Family (LLaVA-1.5, LLaVA-NeXT, LLaVA-OneVision)

[← Back to Architecture Cards](README.md)

| Property | Value |
|---|---|
| **Provider** | LLaVA Team / Haotian Liu et al. |
| **GGUF Projector Architecture** | `clip.projector_type = "mlp"` |
| **Engine Implementations** | `UnifiedVisionPipeline`, `LlavaVisionEncoder` |
| **Vision Backbone** | OpenAI CLIP ViT-L/14 or SigLIP SO400M |
| **Projector Type** | 2-layer GELU Multi-Layer Perceptron (MLP) |
| **Tiling Strategy** | AnyRes multi-crop dynamic patch slicing |
| **Status & Confidence** | 🟢 **Admitted & Golden-Verified** (Verified against `llama-llava-cli` reference) |

---

## 1. Overview & Architectural Highlights

The LLaVA (Large Language and Vision Assistant) series is the pioneer of open visual instruction tuning, connecting pre-trained vision backbones to language decoders via lightweight multi-layer projection networks.

### Key Architectural Characteristics
* **Vision Backbone:**
  * **LLaVA-1.5:** Utilizes OpenAI's `clip-vit-large-patch14-336` running at fixed $336\times 336$ resolution.
  * **LLaVA-NeXT & OneVision:** Adopts Google's `SigLIP-SO400M-14-384`, supporting arbitrary resolution through AnyRes dynamic grid slicing.
* **Two-Layer GELU Projector:**
  Features a clean, high-speed 2-layer projection matrix with GeLU non-linear activation:
  $$\text{Visual Tokens} = W_2 \cdot \text{GELU}(W_1 \cdot Z_{\text{vision}})$$
* **AnyRes Dynamic Multi-Crop:**
  High-resolution images are partitioned into high-detail tiles ($336\times 336$ or $384\times 384$) alongside a downsampled thumbnail overview image. Visual tokens from all crops are injected with spatial newline separator tokens into the language stream.

---

## 2. Checkpoints & Recommended GGUFs

| Model | Parameters | Projector GGUF | Language Model GGUF | Size | Hugging Face Repository |
|---|---|---|---|---|---|
| **LLaVA-1.5-7B** | 7.0B | `mmproj-model-f16.gguf` | `ggml-model-q4_k.gguf` | ~4.5 GB | [liuhaotian/llava-v1.5-7b](https://huggingface.co/liuhaotian/llava-v1.5-7b) *(GGUF: [mys/ggml_llava-v1.5-7b](https://huggingface.co/mys/ggml_llava-v1.5-7b))* |
| **LLaVA-1.5-13B** | 13.0B | `mmproj-model-f16.gguf` | `ggml-model-q4_k.gguf` | ~7.8 GB | [liuhaotian/llava-v1.5-13b](https://huggingface.co/liuhaotian/llava-v1.5-13b) *(GGUF: [mys/ggml_llava-v1.5-13b](https://huggingface.co/mys/ggml_llava-v1.5-13b))* |
| **LLaVA-NeXT-8B** | 8.0B | `mmproj-llava-next-8b-f16.gguf` | `llava-next-8b-q4_k_m.gguf` | ~5.1 GB | [lmms-lab/llama3-llava-next-8b](https://huggingface.co/lmms-lab/llama3-llava-next-8b) |

---

## 3. Usage & Code Examples

### C# High-Level Vision Embedding

```csharp
using OpenTail.Stingray.Vision;

// 1. Open LLaVA projector
using var vision = UnifiedVisionPipeline.Open("models/mmproj-llava-v1.5-7b-f16.gguf");

// 2. Embed image
float[] visualTokens = vision.EmbedImageFile("sample.jpg", out int tokenCount);

Console.WriteLine($"Generated {tokenCount} LLaVA tokens.");
```

### CLI Command

```bash
# Query an image with LLaVA 1.5
stingray -m models/llava-v1.5-7b-q4_k.gguf \
         --mmproj models/mmproj-model-f16.gguf \
         --image sample.jpg \
         -p "Describe what you see in this image."
```
