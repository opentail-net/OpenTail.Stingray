# Qwen-VL Family (Qwen2.5-VL & Qwen3-VL)

[← Back to Architecture Cards](README.md)

| Property | Value |
|---|---|
| **Provider** | Alibaba Cloud (Qwen Team) |
| **GGUF Projector Architecture** | `qwen2vl`, `qwen3vl` (`clip.projector_type = "qwen2vl"`) |
| **Engine Implementations** | `UnifiedVisionPipeline`, `OpenTail.Stingray.Vision.QwenVlEmbedder` |
| **Vision Backbone** | 3D Convolutional stem + Vision Transformer (ViT) |
| **Spatial Merge Mechanism** | $2\times 2$ spatial patch merge with learned projections |
| **Positional Encoding** | 3D Rotational Positional Embedding (M-RoPE across time, height, and width) |
| **Status & Confidence** | 🟢 **Admitted & Golden-Verified** (`Qwen3-VL` wikitext PPL 9.835 vs 9.851 llama-perplexity; vision encoder parity verified) |

---

## 1. Overview & Architectural Highlights

Qwen2.5-VL and Qwen3-VL represent the state of the art in open multimodal vision-language models, supporting native dynamic resolution, arbitrary aspect ratios, document parsing, and video sequence understanding.

### Key Architectural Characteristics
* **3D Convolution Stem:**
  Processes video clips as 3D tubes and images as single-frame 2D slices without aspect-ratio distortion or fixed square resizing.
* **$2\times 2$ Spatial Patch Merging:**
  Downsamples spatial feature maps by a factor of 4 prior to language model injection, reducing a $1024\times 1024$ image from 4,096 raw patches down to 1,024 language tokens.
* **3D Multimodal RoPE (M-RoPE):**
  Divides head rotary dimensions into four distinct sections: temporal ($T$), vertical ($H$), horizontal ($W$), and non-rotated channels.
  * For text: time, height, and width share the same position ID ($p$).
  * For visual tokens: position IDs reflect 2D pixel coordinates and video frame timestamps.
* **DeepStack Vision Slices (Qwen3-VL):**
  Unlike models that concatenate visual tokens solely at the input layer, Qwen3-VL injects multi-level vision representations into intermediate transformer layers (`deepstack` slices after layers $0 \dots N_{\text{deepstack}}-1$).

---

## 2. Checkpoints & Recommended GGUFs

| Model | Parameters | Projector GGUF | Recommended Language GGUF | Hugging Face Repository |
|---|---|---|---|---|
| **Qwen2.5-VL-3B-Instruct** | 3.8B | `mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf` | `Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf` (~2.4 GB) | [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) *(GGUF: [Qwen/Qwen2.5-VL-3B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct-GGUF))* |
| **Qwen2.5-VL-7B-Instruct** | 7.6B | `mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf` | `Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf` (~4.8 GB) | [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) *(GGUF: [Qwen/Qwen2.5-VL-7B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct-GGUF))* |
| **Qwen3-VL-2B-Instruct** | 2.5B | `mmproj-Qwen3-VL-2B-Instruct-f16.gguf` | `Qwen3-VL-2B-Instruct-Q8_0.gguf` (~2.6 GB) | [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) *(or [unsloth/Qwen3-VL-2B-Instruct-GGUF](https://huggingface.co/unsloth/Qwen3-VL-2B-Instruct-GGUF))* |

---

## 3. Usage & Code Examples

### C# Multimodal Vision Pipeline

```csharp
using OpenTail.Stingray.Vision;

// 1. Open the Qwen-VL projector (auto-detects 3D conv stem and 2x2 merger)
using var visionPipeline = UnifiedVisionPipeline.Open("models/mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf");

// 2. Embed an image into visual token vectors
float[] visualTokens = visionPipeline.EmbedImageFile("invoice.png", out int tokenCount);

Console.WriteLine($"Embedded {tokenCount} visual tokens (Dimension: {visionPipeline.EmbeddingDim}).");
```

### CLI Command

```bash
# Query an image or document using Qwen2.5-VL
stingray -m models/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf \
         --mmproj models/mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf \
         --image invoice.png \
         -p "Extract all line items and the total amount from this invoice."
```
