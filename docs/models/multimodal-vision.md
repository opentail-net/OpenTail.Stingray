# Architecture Card: Multimodal Vision Towers

> **Namespace:** `OpenTail.Stingray.Vision`  
> **Verification Status:** 🔬 Verified across 11+ projector architectures (see [docs/STATUS.md](../STATUS.md))  
> **Supported Formats:** GGUF (`mmproj-*.gguf`) and HF SafeTensors vision models  
> **Key Implementations:** `UnifiedVisionPipeline.cs`, `VisionProjector.cs`, `PatchMerger.cs`

---

## 1. Architectural Highlights

Stingray provides a unified multimodal vision subsystem capable of processing images, OCR documents, and video frames without external Python or C++ libraries. It abstracts the vision encoder pipeline across 11+ distinct architectures:

| Architecture | Vision Backbone | Projector / Spatial Merger Type | Dynamic Tiling / Resolution | Hugging Face Repository |
|---|---|---|---|---|
| **Qwen2.5-VL / Qwen3-VL** | 3D Conv stem + ViT | $2\times 2$ spatial merge with learned weights | Arbitrary aspect ratio native 2D windowing | [ggml-org/Qwen2.5-VL-7B-Instruct-GGUF](https://huggingface.co/ggml-org/Qwen2.5-VL-7B-Instruct-GGUF) |
| **DeepSeek-OCR / OCR2** | Dual SAM + CLIP ViT fusion | Dense Conv2D + Linear Projection | Fixed $1024\times 1024$ multi-scale grid | [deepseek-ai/DeepSeek-OCR-2](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2) |
| **Pixtral (Mistral AI)** | 2D Continuous RoPE ViT | SwiGLU Projection MLP | Dynamic image token budgeting | [mistralai/Pixtral-12B-2409](https://huggingface.co/mistralai/Pixtral-12B-2409) |
| **LLaVA-1.5 / NeXT / OneVision** | CLIP / SigLIP ViT | 2-layer GELU MLP | AnyRes multi-crop tiling | [liuhaotian/llava-v1.5-7b](https://huggingface.co/liuhaotian/llava-v1.5-7b) |
| **InternVL 2.5 / 3 / 4** | InternViT-6B | PixelShuffle $2\times 2$ downsampling + MLP | Dynamic patch slicing ($N \times 448\times 448$) | [OpenGVLab/InternVL2_5-8B](https://huggingface.co/OpenGVLab/InternVL2_5-8B) |
| **MiniCPM-V 2.6** | SigLIP ViT | 2D sinusoidal cross-attention Resampler | HD 9-slice grid | [openbmb/MiniCPM-V-2_6](https://huggingface.co/openbmb/MiniCPM-V-2_6) |
| **GLM-4V / GLM-OCR** | Dual Conv2D stem + ViT | Conv2D patch merger + 2D M-RoPE | Multi-resolution aspect ratio bucketing | [THUDM/glm-4v-9b](https://huggingface.co/THUDM/glm-4v-9b) |
| **Nemotron-VL** | Learned register token ViT | Squared ReLU MLP | Multi-crop image grids | [nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1](https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1) |
| **Dots-OCR / PaddleOCR-VL** | Lightweight ViT | Patch merger + GELU MLP | High-density text layout grids | [PaddlePaddle/PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL) |
| **Gemma 3 / 4** | SigLIP / Gemma 4 UV | Concat + Linear Projection | Fixed resolution single/multi-crop | [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it) |

---

## 2. Stingray Engine Execution Model

The end-to-end vision inference pipeline operates in three discrete stages:

```
[ Input Image (.png/.jpg) ]
           │
           ▼
[ Image Preprocessor ] ──► Normalization, dynamic slicing & bilinear resizing
           │
           ▼
[ Vision Tower (ViT) ] ──► Patch embeddings + 2D spatial position encoding
           │
           ▼
[ Projector / Merger ] ──► Spatial downsampling & projection to LLM hidden dimension
           │
           ▼
[ Visual Tokens (float[]) ] ──► Injected into LLM context at <image> placeholder slots
```

1. **Pre-processing:** Fast managed bilinear resizing and channel normalization convert raw image buffers into input tensor spans without allocating auxiliary unmanaged memory.
2. **Vision Forward Pass:** Evaluates convolutional stems, self-attention blocks, and feed-forward layers using SIMD-accelerated CPU kernels or GPU compute shaders.
3. **Context Ingestion:** The generated visual embeddings ($N_{\text{tokens}} \times D_{\text{model}}$) are directly written into the text context's KV cache at the positions allocated by `<image>` or `<|vision_start|>` tokens. Subsequent generation is handled identically to text decoding.

---

## 3. Practical Usage & Commands

### CLI Usage
To analyze an image from the command line, provide both the language model and its companion `mmproj` projector:

```bash
stingray -m models/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf \
         --mmproj models/mmproj-Qwen2.5-VL-7B-Instruct-q8_0.gguf \
         --image document.png \
         -p "Transcribe the table in this image to markdown."
```

### Public C# API
```csharp
using OpenTail.Stingray.Vision;

// 1. Open any multimodal vision GGUF projector (auto-detects architecture)
using var embedder = UnifiedVisionPipeline.Open("models/mmproj-deepseek-ocr-2-q8_0.gguf");

// 2. Load and embed an image into visual token embeddings
float[] visualTokens = embedder.EmbedImageFile("receipt.jpg", out int tokenCount);

Console.WriteLine($"Extracted {tokenCount} visual tokens (Embedding dim: {embedder.EmbeddingDim}).");
```

---

## 4. Verification Evidence & Constraints

- **Verified Status:** Projector tensor matching and embedding equivalence verified against llama.cpp's `llama-minicpmv-cli` and `llama-llava-cli` (see [docs/STATUS.md](../STATUS.md)).
- **Hardware Boundary:** Several vision encoders execute exclusively on CPU SIMD paths; GPU offload of large vision towers (e.g. InternViT 6B) is gated on device VRAM availability.
