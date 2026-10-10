# IBM Granite Family (Granite 3.x, Granite MoE, Granite 4.0-H Hybrid Mamba-2)

[← Back to Architecture Cards](README.md)

| Property | Value |
|---|---|
| **Provider** | IBM Research |
| **GGUF Architecture Keys** | `granite`, `granitemoe`, `granitehybrid` |
| **Engine Implementations** | `GraniteArchitecture.Descriptor`, `GraniteArchitectures.Granitemoe`, `Granitehybrid` |
| **Forward Pass Factory** | `CommonForwardPassFactory.CreateDense` / `CreateHybridGdn` |
| **Modalities** | Text, Code |
| **Thinking Mode** | Optional |
| **Tool Calling & JSON BNF** | Native Granite function-calling format |
| **Status & Confidence** | 🟢 **Admitted & Golden-Verified** (`24/24 exact greedy match on dense & MoE; Mamba-2 SSM parity on 4.0-H`) |

---

## 1. Overview & Architectural Highlights

IBM's Granite series delivers enterprise-grade models with verified Apache-2.0 governance and mathematical modifications specifically targeting deep enterprise reasoning, coding, and ultra-fast hybrid recurrence.

### Key Architectural Characteristics
* **The "Scale Trio":**
  Granite applies explicit multipliers that differ from conventional Llama scaling:
  * $\text{ResidualScale}$: Multiplies residual streams at block boundaries.
  * $\text{AttentionScaleOverride}$: Custom attention scaling factor overriding $1 / \sqrt{d_k}$.
  * $\text{LogitScale}$: Reciprocal scaling applied directly to pre-softmax output logits.
  * $\text{EmbeddingScale}$: Root-embedding scaling ($\sqrt{d_{\text{model}}}$).
* **Granite 4.0-H Mamba-2 Hybrid Recurrence (`granitehybrid`):**
  * Interleaves Mamba-2 state space model (SSM) selective scan layers with standard multi-head attention.
  * Executed in managed C# via `ForwardPass.Mamba2.cs` with an in-memory recurrent state matrix, reducing the KV cache footprint by up to 75% compared to pure attention models.
  * Evaluated on CPU with Wikitext perplexity within 0.3% of `llama.cpp` (17.95 vs 17.92 on 350M; 8.78 vs 8.75 on 1B).
* **Granite MoE (`granitemoe`):**
  * Sparse Mixture-of-Experts routing with normalized top-$k$ gating and dense base residual paths.
  * Golden verified with 24-of-24 token exact match against `llama.cpp`.

---

## 2. Checkpoints & Recommended GGUFs

| Model | Hugging Face Repository | Parameters | Architecture | Recommended Quantization | Size | Purpose |
|---|---|---|---|---|---|---|
| **Granite-3.1-2B-Instruct** | [lmstudio-community/granite-3.1-2b-instruct-GGUF](https://huggingface.co/lmstudio-community/granite-3.1-2b-instruct-GGUF) | 2.5B | Dense | `Q4_K_M` / `Q8_0` | ~1.6 GB / ~2.7 GB | Edge enterprise instruction |
| **Granite-3.1-8B-Instruct** | [lmstudio-community/granite-3.1-8b-instruct-GGUF](https://huggingface.co/lmstudio-community/granite-3.1-8b-instruct-GGUF) | 8.2B | Dense | `Q4_K_M` | ~4.9 GB | General language & reasoning |
| **Granite-3.0-1B-A400M** | [ibm-granite/granite-3.0-1b-a400m-instruct](https://huggingface.co/ibm-granite/granite-3.0-1b-a400m-instruct) | 1.4B | MoE | `Q4_K_M` | ~1.1 GB | Lightweight high-speed MoE |
| **Granite-4.0-H-1B** | [ibm-granite/granite-4.0-h-1b](https://huggingface.co/ibm-granite/granite-4.0-h-1b) | 1.5B | Hybrid Mamba-2 | `Q8_0` / `Q4_K_M` | ~1.6 GB / ~0.9 GB | Next-gen hybrid state space |

---

## 3. Hardware Requirements & Performance Profile

On the reference AMD Ryzen 7 5700G (CPU AVX2 SIMD):
* **IBM Granite 4.0-H 1B (`granite-4.0-h-1b-Q8_0.gguf`):** Prompt prefill ~63.4 tok/s (over 511 tokens); greedy decode ~21.0 tok/s (2026-09-28).

| Model | Quant | Context | Host RAM (CPU) | Recommended Placement |
|---|---|---|---|---|
| **Granite 3.1 2B** | Q4_K_M | 4,096 | ~2.4 GB | CPU SIMD |
| **Granite 4.0-H 1B** | Q8_0 | 2,048 | ~1.6 GB | CPU SIMD |
| **Granite 3.1 8B** | Q4_K_M | 8,192 | ~6.0 GB | Vulkan / CUDA GPU or 16 GB+ RAM |

---

## 4. Usage & Code Examples

### C# Streaming Chat

```csharp
using OpenTail.Stingray;
using OpenTail.Stingray.Executors;

// 1. Load IBM Granite 4.0-H Hybrid Mamba-2
using var model = Model.Load(new ModelParams("models/granite-4.0-h-1b-Q8_0.gguf")
{
    Backend = "cpu"
});

// 2. Allocate context
using var context = model.CreateContext(new ContextParams { ContextSize = 4096 });
var executor = new InteractiveExecutor(context);
var session = new ChatSession(executor);

// 3. Query code or enterprise instruction
await foreach (var chunk in session.ChatChunksAsync("Write a C# method that computes the SHA-256 hash of a byte array using IncrementalHash."))
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
# Run IBM Granite 4.0-H Hybrid on CPU
stingray -m models/granite-4.0-h-1b-Q8_0.gguf -p "Explain Mamba-2 state space recurrence in simple terms."
```
