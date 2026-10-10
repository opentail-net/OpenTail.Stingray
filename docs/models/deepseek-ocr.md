# DeepSeek-OCR Family (DeepSeek-OCR & DeepSeek-OCR2)

[← Back to Architecture Cards](README.md)

| Property | Value |
|---|---|
| **Provider** | DeepSeek AI |
| **GGUF Projector Key** | `deepseek2_ocr` |
| **Engine Implementations** | `DeepSeek2Architectures.DeepSeek2Ocr`, `UnifiedVisionPipeline` |
| **Vision Backbone** | Dual SAM (Segment Anything) ViT + CLIP ViT feature fusion |
| **Target Grid** | Fixed multi-scale high-density grid ($1024\times 1024$) |
| **Projection Mechanism** | Dense Conv2D patch merger + 2-layer Linear projector |
| **Status & Confidence** | 🟢 **Admitted & Golden-Verified** (`DeepSeek-OCR-2`, Level 2 Proven) |

---

## 1. Overview & Architectural Highlights

DeepSeek-OCR and DeepSeek-OCR2 are specialized vision-language architectures optimized for document layout analysis, optical character recognition, complex mathematical table parsing, and PDF extraction.

### Key Architectural Characteristics
* **Dual SAM + CLIP ViT Fusion:**
  * **SAM (Segment Anything Model) Tower:** Captures pixel-precise mask boundaries, stroke details, and character edge features.
  * **CLIP ViT Tower:** Encodes global semantic and linguistic context.
  * The outputs of both towers are concatenated and fused through cross-attention before projecting into the language decoder.
* **$1024\times 1024$ Multi-Scale Processing:**
  High-resolution document pages are partitioned into $1024\times 1024$ dense grids with sub-patch feature extraction, resolving fine 6pt fonts and multi-column tabular data that standard $448\times 448$ vision towers blur.
* **Low-Latency Conv2D Patch Compression:**
  A 2D convolutional merger compresses adjacent token grids into the DeepSeek-V2 latent space.

---

## 2. Checkpoints & Recommended GGUFs

| Model | Projector GGUF | Size | Primary Function | Hugging Face Repository |
|---|---|---|---|---|
| **DeepSeek-OCR-2-Q8** | `mmproj-deepseek-ocr-2-q8_0.gguf` | ~850 MB | Document & invoice parsing | [deepseek-ai/deepseek-vl2](https://huggingface.co/deepseek-ai/deepseek-vl2) |
| **DeepSeek-OCR-2-F16** | `mmproj-deepseek-ocr-2-f16.gguf` | ~1.6 GB | Full precision dense text OCR | [deepseek-ai/deepseek-vl2](https://huggingface.co/deepseek-ai/deepseek-vl2) |

---

## 3. Usage & Code Examples

### C# High-Density Document Parsing

```csharp
using OpenTail.Stingray.Vision;

// 1. Load DeepSeek-OCR dual SAM+CLIP projector
using var ocrPipeline = UnifiedVisionPipeline.Open("models/mmproj-deepseek-ocr-2-q8_0.gguf");

// 2. Extract visual token representation from a PDF scan or document image
float[] visualTokens = ocrPipeline.EmbedImageFile("scanned_contract.png", out int tokenCount);

Console.WriteLine($"Extracted {tokenCount} high-density OCR tokens (Dim: {ocrPipeline.EmbeddingDim}).");
```

### CLI Command

```bash
# Extract full markdown from an image of a complex multi-column document
stingray -m models/deepseek-v2-lite-chat-q4_k_m.gguf \
         --mmproj models/mmproj-deepseek-ocr-2-q8_0.gguf \
         --image complex_table.png \
         -p "Transcribe this document into valid GitHub-flavored markdown, maintaining all table structures."
```
