# Gemma 4 Vision & UV Family (Gemma 4 UV / Gemma 4 ViT)

[← Back to Architecture Cards](README.md)

| Property | Value |
|---|---|
| **Provider** | Google DeepMind |
| **GGUF Projector Architecture** | `gemma4uv`, `gemma4v`, `gemma3` |
| **Engine Implementations** | `UnifiedVisionPipeline`, `Gemma4VisionEmbedder` |
| **Vision Backbone** | Google SigLIP SO400M / Gemma 4 Unified Vision (UV) |
| **Projection Type** | Direct Concat + Linear Projection |
| **Status & Confidence** | 🟢 **Admitted & Golden-Verified** (`Gemma 4 E4B vision`, Level 2 Proven) |

---

## 1. Overview & Architectural Highlights

Google's Gemma 4 and Gemma 3 multimodal vision architectures natively embed visual understanding into Google's compact foundation models, supporting both single-crop visual question answering and multi-image reasoning.

### Key Architectural Characteristics
* **Gemma 4 UV (Unified Vision):**
  * Employs the `gemma4uv` architecture, which unifies patch tokenization and positional encoding directly within Google's shared tensor graph.
  * Eliminates separate vision-language projector drift by sharing embedding scales.
* **SigLIP SO400M Feature Alignment:**
  * Uses sigmoid loss pre-trained vision representations rather than softmax cross-entropy, yielding better zero-shot classification and dense visual alignment.
* **Direct Linear Dimension Mapping:**
  * Maps vision patch embeddings directly into language hidden dimensions without intermediate downsampling, preserving fine spatial layout details.

---

## 2. Checkpoints & Recommended GGUFs

| Model | Modality | Projector GGUF | Language Model GGUF | Hugging Face Repository |
|---|---|---|---|---|
| **Gemma 4 E4B-IT** | Vision + Text | `mmproj-gemma-4-E4B-it-f16.gguf` | `gemma-4-E4B-it-Q4_K_M.gguf` (~2.9 GB) | [ggml-org/gemma-4-E4B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF) |
| **Gemma 3 4B-IT** | Vision + Text | `mmproj-gemma-3-4b-it-f16.gguf` | `gemma-3-4b-it-Q4_K_M.gguf` (~2.7 GB) | [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it) *(GGUF: [bartowski/gemma-3-4b-it-GGUF](https://huggingface.co/bartowski/gemma-3-4b-it-GGUF))* |

---

## 3. Usage & Code Examples

### C# High-Level Vision Embedding

```csharp
using OpenTail.Stingray.Vision;

// 1. Open Gemma 4 UV projector
using var vision = UnifiedVisionPipeline.Open("models/mmproj-gemma-4-E4B-it-f16.gguf");

// 2. Embed image
float[] visualTokens = vision.EmbedImageFile("photo.jpg", out int tokenCount);

Console.WriteLine($"Embedded {tokenCount} visual tokens (Embedding dim: {vision.EmbeddingDim}).");
```

### CLI Command

```bash
# Query an image with Gemma 4 E4B
stingray -m models/gemma-4-E4B-it-Q4_K_M.gguf \
         --mmproj models/mmproj-gemma-4-E4B-it-f16.gguf \
         --image photo.jpg \
         -p "What objects are present in this image?"
```
