# Pixtral Family (Mistral Pixtral 12B)

[← Back to Architecture Cards](README.md)

| Property | Value |
|---|---|
| **Provider** | Mistral AI |
| **GGUF Projector Architecture** | `clip.projector_type = "pixtral"` |
| **Engine Implementations** | `UnifiedVisionPipeline`, `Mistral3VisionEncoder` |
| **Vision Backbone** | Vision Transformer with 2D Continuous RoPE |
| **Patch Resolution** | Dynamic arbitrary resolution with SwiGLU projection |
| **Status & Confidence** | 🟢 **Admitted & Golden-Verified** (Encoder verified against Hugging Face Pixtral reference) |

---

## 1. Overview & Architectural Highlights

Pixtral 12B, released by Mistral AI, is a multimodal vision-language model designed to process images of arbitrary aspect ratios and resolutions without artificial letterboxing or fixed grid cropping.

### Key Architectural Characteristics
* **2D Continuous Rotary Position Embeddings:**
  Replaces standard learned 1D patch position embeddings with 2D continuous RoPE, encoding true 2D spatial relationships across arbitrary patch aspect ratios.
* **SwiGLU Vision MLP:**
  Utilizes the same SwiGLU gating mechanism in the vision tower as Mistral's language backbone, maintaining high non-linear expressiveness.
* **Dynamic Visual Token Budgeting:**
  Image tokens scale naturally with image size rather than forcing a fixed token count, allowing fast processing for small icons and high detail preservation for massive 4K blueprints.
* **Seamless Projector Auto-Discovery:**
  In Stingray, when `--image` is passed to the CLI with a Mistral 3 model, the runtime searches for matching `*mmproj*istral*.gguf` files in the directory automatically.

---

## 2. Checkpoints & Recommended GGUFs

| Component | Repository | File | Size |
|---|---|---|---|
| **Pixtral 12B (Reference / Weights)** | [mistralai/Pixtral-12B-2409](https://huggingface.co/mistralai/Pixtral-12B-2409) | `consolidated.safetensors` | ~24 GB |
| **Language Backbone (GGUF)** | [bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF](https://huggingface.co/bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF) | `mistralai_Mistral-Small-3.1-24B-Instruct-2503-Q4_K_M.gguf` | ~14.3 GB |
| **Pixtral Projector (GGUF)** | [bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF](https://huggingface.co/bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF) | `mmproj-mistralai_Mistral-Small-3.1-24B-Instruct-2503-f16.gguf` | ~0.88 GB |

---

## 3. Usage & Code Examples

### C# Visual Token Embedding

```csharp
using OpenTail.Stingray.Vision;

// 1. Open Pixtral projector
using var vision = UnifiedVisionPipeline.Open("models/mmproj-pixtral-12b-f16.gguf");

// 2. Embed input image
float[] visualTokens = vision.EmbedImageFile("chart.png", out int tokenCount);

Console.WriteLine($"Extracted {tokenCount} visual tokens with embedding dim {vision.EmbeddingDim}.");
```

### CLI Command

```bash
# Run Pixtral 12B with image reasoning
stingray -m models/mistral-small-3.1-24b-q4_k_m.gguf \
         --mmproj models/mmproj-pixtral-12b-f16.gguf \
         --image chart.png \
         -p "Analyze the trends shown in this chart."
```
