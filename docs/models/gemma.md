# Gemma Family (Gemma 1, 2, 3, Gemma 4 / E4B)

[← Back to Architecture Cards](README.md)

| Property | Value |
|---|---|
| **Provider** | Google DeepMind |
| **GGUF Architecture Keys** | `gemma`, `gemma2`, `gemma3`, `gemma3n`, `gemma4` |
| **Engine Implementations** | `GemmaArchitectures.Gemma`, `Gemma2`, `Gemma3`, `Gemma4Architecture` |
| **Forward Pass Factory** | `CommonForwardPassFactory.CreateDense` |
| **Modalities** | Text, Multimodal Vision (Gemma 4 E4B, Gemma 3 SigLIP) |
| **Thinking Mode** | Supported (`ThinkingDefaultOff = true` by default) |
| **Tool Calling & JSON BNF** | Supported via native Gemma chat templates |
| **Status & Confidence** | 🟢 **Admitted & Golden-Verified** (`Gemma 4 E4B text (gemma4)`, Level 2 Proven) |

---

## 1. Overview & Architectural Highlights

Gemma models, built by Google DeepMind, introduce several non-standard transformer design decisions aimed at high mathematical stability and low perplexity across compact parameter sizes (2B to 27B, plus Gemma 4 E4B).

### Key Architectural Characteristics
* **RMSNorm with Unit Offset ($1 + \text{weight}$):** Unlike standard Llama RMSNorm, Gemma weights multiply by $(1 + w_i)$ or scale inputs by $\sqrt{d_{\text{model}}}$. Stingray applies exact Gemma normalization semantics to prevent numeric divergence.
* **Dual Residual Norms (Gemma 2 / 3):** Pre-attention and post-attention RMSNorms are applied per block, ensuring high-gradient stability.
* **Logit Soft-Capping:** In Gemma 2 and 3, attention logits and final output logits are soft-capped via hyperbolic tangent scaling:
  $$\text{logits} = C \cdot \tanh\left(\frac{\text{logits}}{C}\right)$$
  where $C_{\text{attn}} = 50.0$ and $C_{\text{final}} = 30.0$.
* **NeoX-Style RoPE:** Uses NeoX RoPE rotation ordering (`UsesNeoxRope = true`).
* **Sliding Window Attention (SWA):** Alternates between full global attention and local sliding window attention (typically 4096 tokens).
* **Gemma 4 E4B Multimodal Integration:** Seamlessly supports direct image projection into the text sequence using SigLIP/ViT projectors (`SupportsImageInput = true`).

---

## 2. Checkpoints & Recommended GGUFs

| Model | Hugging Face Repository | Parameters | Quantization | Size | Purpose |
|---|---|---|---|---|---|
| **Gemma-2-2B-IT** | [google/gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it) *(GGUF: [bartowski/gemma-2-2b-it-GGUF](https://huggingface.co/bartowski/gemma-2-2b-it-GGUF))* | 2.6B | `Q4_K_M` | ~1.7 GB | Fast on-device chat and CPU edge inference |
| **Gemma-2-9B-IT** | [google/gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it) *(GGUF: [bartowski/gemma-2-9b-it-GGUF](https://huggingface.co/bartowski/gemma-2-9b-it-GGUF))* | 9.2B | `Q4_K_M` | ~5.8 GB | Strong general-purpose instruction following |
| **Gemma-3-1B-IT** | [google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it) | 1.1B | `Q4_K_M` | ~0.8 GB | Ultra-compact edge model |
| **Gemma-4-E4B-IT** | [ggml-org/gemma-4-E4B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF) | 4.4B | `Q8_0` / `Q4_K_M` | ~7.5 GB / ~2.9 GB | Google's multimodal reasoning checkpoint |

---

## 3. Hardware Requirements & Memory Estimation

| Model | Quant | Context | Host RAM (CPU) | Recommended Placement |
|---|---|---|---|---|
| **Gemma 2 2B** | Q4_K_M | 4,096 | ~2.5 GB | CPU SIMD (AVX2/AVX-512) |
| **Gemma 4 E4B** | Q4_K_M | 4,096 | ~4.2 GB | CPU SIMD or Vulkan / CUDA GPU |
| **Gemma 4 E4B** | Q8_0 | 4,096 | ~8.9 GB | Discrete GPU (Vulkan / CUDA) or 16 GB+ RAM |
| **Gemma 2 9B** | Q4_K_M | 8,192 | ~8.5 GB | Dedicated GPU (Vulkan / CUDA) |

---

## 4. Usage & Code Examples

### C# Streaming Chat

```csharp
using OpenTail.Stingray;
using OpenTail.Stingray.Executors;

// 1. Initialize Gemma 4 E4B with auto GPU offload
using var model = Model.Load(new ModelParams("models/gemma-4-E4B-it-Q4_K_M.gguf")
{
    Backend = "auto",
    GpuLayerCount = -1
});

// 2. Allocate context
using var context = model.CreateContext(new ContextParams { ContextSize = 4096 });
var executor = new InteractiveExecutor(context);
var session = new ChatSession(executor);

// 3. Stream text
await foreach (var chunk in session.ChatChunksAsync("Why do modern transformers use grouped-query attention?"))
{
    if (chunk.Kind == GenerateChunkKind.Text)
    {
        Console.Write(chunk.Text);
    }
}
Console.WriteLine();
```

### CLI Command

```bash
# Run Gemma 4 E4B interactively
stingray -m models/gemma-4-E4B-it-Q4_K_M.gguf -g -1

# Multimodal image query with companion projector
stingray -m models/gemma-4-E4B-it-Q4_K_M.gguf --mmproj models/mmproj-gemma-4-e4b.gguf --image photo.jpg -p "Describe this photo."
```
