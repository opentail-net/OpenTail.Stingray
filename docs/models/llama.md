# Llama Family (Llama 3 / 3.1 / 3.2 / 3.3 / Llama 4)

[← Back to Architecture Cards](README.md)

| Property | Value |
|---|---|
| **Provider** | Meta AI |
| **GGUF Architecture Key** | `llama`, `llama4` |
| **Engine Implementations** | `LlamaArchitecture` (`Id = "llama"`), `Llama4Architecture` (`Id = "llama4"`) |
| **Forward Pass Factory** | `CommonForwardPassFactory.CreateDense` |
| **Modalities** | Text (Multimodal vision supported via separate Mllama / Vision Tower projectors) |
| **Thinking Mode** | Optional / Model-dependent |
| **Tool Calling & JSON BNF** | Fully supported via Llama 3 / Llama 4 structured formatters |
| **Status & Confidence** | 🟢 **Admitted & Golden-Verified** (Level 2: Proven real prefill and greedy decode loop) |

---

## 1. Overview & Architectural Highlights

The Llama architecture family is the foundation of modern open-weights generative AI. OpenTail.Stingray provides 100% managed C# execution across all generations, from edge models (Llama 3.2 1B/3B) to large reasoning models (Llama 3.3 70B and Llama 4).

### Key Architectural Traits
* **RMSNorm:** Root Mean Square Layer Normalization applied before self-attention and feed-forward layers.
* **SwiGLU Activation:** `Swish(Gate) * Up` projection in the feed-forward network.
* **Grouped-Query Attention (GQA):** Reduced key-value head counts (e.g., 8 KV heads vs. 32 query heads) for accelerated generation and smaller KV cache footprints.
* **RoPE Frequency Scaling (Llama 3.1 / 3.3):** High-frequency baseline (`rope_theta = 500000`) paired with YaRN / linear scaling to support context windows up to 128k tokens.
* **Llama 4 Innovations:**
  * **L2 QK-Norm:** Unweighted $L_2$ normalization on query and key vectors for training stability at scale (`UseL2QkNorm = true`).
  * **NoPE Interleaving:** No-Position-Embedding (`NoRopeLayerStep = 4`), skipping RoPE rotation every 4th layer.
  * **Sigmoid Gating:** Feed-forward gating with sigmoid scaling prior to projection.
  * **Chunked Attention:** Chunked attention blocks (8192 tokens) with temperature scaling.

---

## 2. Checkpoints & Recommended GGUFs

| Model | Hugging Face Repository | Parameters | Context Window | Recommended Quantization | Typical Size |
|---|---|---|---|---|---|
| **Llama-3.2-1B-Instruct** | [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) *(GGUF: [bartowski/Llama-3.2-1B-Instruct-GGUF](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF))* | 1.23B | 128k tokens | `Q4_K_M` / `Q8_0` | ~0.8 GB / ~1.3 GB |
| **Llama-3.2-3B-Instruct** | [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) *(GGUF: [bartowski/Llama-3.2-3B-Instruct-GGUF](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF))* | 3.21B | 128k tokens | `Q4_K_M` | ~2.0 GB |
| **Llama-3.1-8B-Instruct** | [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) *(GGUF: [bartowski/Meta-Llama-3.1-8B-Instruct-GGUF](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF))* | 8.03B | 128k tokens | `Q4_K_M` / `Q5_K_M` | ~4.9 GB / ~5.7 GB |
| **Llama-3.3-70B-Instruct** | [meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) *(GGUF: [bartowski/Llama-3.3-70B-Instruct-GGUF](https://huggingface.co/bartowski/Llama-3.3-70B-Instruct-GGUF))* | 70.6B | 128k tokens | `Q4_K_M` (Multi-shard) | ~42.5 GB |

*Multi-shard files (e.g. `Llama-3.3-70B-Instruct-Q4_K_M-00001-of-00009.gguf`) are automatically discovered and assembled into a unified tensor map by `GgufModel.Open`.*

---

## 3. Hardware Requirements & Memory Estimation

Host memory for Llama models is estimated ahead of loading via `HostMemoryEstimator`:
$$\text{Peak RAM} \approx \text{Weight Bytes} + \text{Q4\_K Repack Copy} + \text{fp32 KV Cache}(N_{\text{ctx}}) + \text{Scratch Scratchpad}$$

| Model Size | Quantization | Context | RAM (CPU Run) | Recommended Backend |
|---|---|---|---|---|
| **1B** | Q4_K_M | 4,096 | ~1.4 GB | CPU (AVX2/AVX-512) |
| **3B** | Q4_K_M | 4,096 | ~3.1 GB | CPU or Vulkan APU |
| **8B** | Q4_K_M | 8,192 | ~6.5 GB | Vulkan GPU / CUDA (or 16 GB+ CPU) |
| **70B** | Q4_K_M | 4,096 | ~48.0 GB | Multi-GPU CUDA or High-RAM CPU host |

---

## 4. Usage & Code Examples

### C# High-Level API

```csharp
using OpenTail.Stingray;
using OpenTail.Stingray.Executors;

// 1. Load model with automatic hardware acceleration
using var model = Model.Load(new ModelParams("models/Llama-3.2-3B-Instruct-Q4_K_M.gguf")
{
    Backend = "auto",
    GpuLayerCount = -1 // Offload all layers if GPU is present
});

// 2. Allocate context and interactive chat session
using var context = model.CreateContext(new ContextParams { ContextSize = 4096 });
var executor = new InteractiveExecutor(context);
var session = new ChatSession(executor);
session.AddSystemMessage("You are a helpful, concise AI assistant.");

// 3. Stream generated response tokens
await foreach (var chunk in session.ChatChunksAsync("Explain AVX-512 in one sentence."))
{
    if (chunk.Kind == GenerateChunkKind.Text)
    {
        Console.Write(chunk.Text);
    }
}
Console.WriteLine();
```

### CLI Execution

```bash
# Single-prompt completion on CPU
stingray -m models/Llama-3.2-3B-Instruct-Q4_K_M.gguf -p "What are the advantages of managed SIMD?"

# Multi-turn interactive chat on GPU
stingray -m models/Llama-3.1-8B-Instruct-Q4_K_M.gguf -g -1 --backend vulkan
```
