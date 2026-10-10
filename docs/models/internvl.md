# InternVL Family (InternVL 2.5, 3, 4)

[← Back to Architecture Cards](README.md)

| Property | Value |
|---|---|
| **Provider** | OpenGVLab / Shanghai AI Laboratory |
| **GGUF Projector Architecture** | `clip.projector_type = "internvl"` |
| **Engine Implementations** | `UnifiedVisionPipeline`, `InternVlEmbedder` |
| **Vision Backbone** | InternViT-300M / InternViT-6B |
| **Spatial Downsampling** | PixelShuffle $2\times 2$ downsampling + MLP |
| **Dynamic Tiling** | Native 448-pixel dynamic grid partitioning ($N \times 448\times 448$) |
| **Status & Confidence** | 🟢 **Admitted & Golden-Verified** (Level 2 Proven, tested with real weight tensors) |

---

## 1. Overview & Architectural Highlights

InternVL is OpenGVLab's flagship multimodal foundation model, widely recognized for leading multimodal benchmark leaderboards across OCR, document analysis, mathematical diagram reasoning, and general vision-language dialogue.

### Key Architectural Characteristics
* **InternViT Vision Backbone:**
  Employs a custom Vision Transformer trained natively from scratch (up to 6B parameters in InternViT-6B) rather than off-the-shelf contrastive CLIP models, capturing finer structural details.
* **PixelShuffle $2\times 2$ Downsampling:**
  Instead of simple pooling or stride-2 convolutions, InternVL applies a PixelShuffle spatial rearrangement operation. A $2\times 2$ block of visual patch tokens ($4 \times D$) is reorganized along the channel dimension before a single linear projection downsamples the token sequence by $4\times$.
* **Dynamic Patch Slicing:**
  Images are dynamically divided into $N$ tiles of $448\times 448$ pixels (from 1 up to 12 tiles depending on resolution and aspect ratio), preserving fine details without geometric distortion.

---

## 2. Checkpoints & Recommended GGUFs

| Model | Total Parameters | Vision Backbone | Recommended Quantization | Typical Size |
|---|---|---|---|---|
| **InternVL2_5-1B** | 1.2B | InternViT-300M | `Q4_K_M` | ~0.9 GB |
| **InternVL2_5-2B** | 2.2B | InternViT-300M | `Q4_K_M` | ~1.6 GB |
| **InternVL2_5-8B** | 8.1B | InternViT-300M | `Q4_K_M` | ~5.2 GB |
| **InternVL2_5-26B** | 25.5B | InternViT-6B | `Q4_K_M` | ~16.5 GB |

---

## 3. Usage & Code Examples

### C# Vision Token Extraction

```csharp
using OpenTail.Stingray.Vision;

// 1. Initialize InternVL projector with PixelShuffle merger
using var internvl = UnifiedVisionPipeline.Open("models/mmproj-internvl2.5-8b-f16.gguf");

// 2. Extract visual representations
float[] visualTokens = internvl.EmbedImageFile("document.png", out int tokenCount);

Console.WriteLine($"Embedded {tokenCount} visual tokens with embedding dim {internvl.EmbeddingDim}.");
```

### CLI Command

```bash
# Query with InternVL 2.5
stingray -m models/internvl2.5-8b-q4_k_m.gguf \
         --mmproj models/mmproj-internvl2.5-8b-f16.gguf \
         --image document.png \
         -p "Extract all text and tabular information."
```
