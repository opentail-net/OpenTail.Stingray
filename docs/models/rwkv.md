# RWKV Family (RWKV-6 Finch & RWKV-7 Goose)

[← Back to Architecture Cards](README.md)

| Property | Value |
|---|---|
| **Provider** | Bo Peng / RWKV Foundation |
| **GGUF Architecture Keys** | `rwkv6`, `rwkv7` |
| **Engine Implementations** | `RwkvArchitectures.Rwkv6`, `Rwkv7`, `RwkvForwardPassBase`, `Rwkv7ForwardPass` |
| **Forward Pass Family** | `ForwardPassFamily.Rwkv` |
| **Modalities** | Text |
| **Supported Backends** | CPU only (AVX2 / AVX-512 SIMD; GPU requests fall back to CPU) |
| **KV Cache Footprint** | **0 bytes** (Pure linear recurrent state, constant $O(1)$ RAM regardless of sequence length) |
| **Status & Confidence** | 🟢 **Admitted & Golden-Verified** (`RWKV-7 Goose` and `RWKV-6 Finch`, Wikitext PPL within 0.3% of llama.cpp) |

---

## 1. Overview & Architectural Highlights

RWKV (Receptance Weighted Key Value) combines the parallelizable training advantages of Transformers with the constant-memory, linear-time $O(1)$ inference of Recurrent Neural Networks (RNNs).

### Key Architectural Characteristics
* **Zero KV Cache Overhead:**
  Unlike standard transformers which require gigabytes of expanding KV cache memory as context grows, RWKV maintains a fixed-size hidden recurrent state per layer ($S_t \in \mathbb{R}^{d \times d}$). Memory usage never increases, whether processing 10 tokens or 100,000 tokens.
* **RWKV-6 (Finch) Time-Mixing:**
  * Uses data-dependent decay vectors ($W_t$) and time-mixing interpolation to gate linear attention states.
  * LayerNorm with explicit learned bias.
* **RWKV-7 (Goose) State-Evolution Kernel:**
  * Replaces static linear attention with a continuous-time matrix decay evolution mechanism, improving in-context learning and multi-step reasoning.
  * In Stingray, vector-matrix operations are evaluated with optimized SIMD inner products (`Rwkv7ForwardPass.cs`).
* **World Trie Tokenizer:**
  Stingray includes a native C# trie-based tokenizer matching `llama-tokenize` on complex CJK, unicode emoji, and byte sequences.

---

## 2. Checkpoints & Recommended GGUFs

| Model | Parameters | Version | Recommended Quantization | Typical Size |
|---|---|---|---|---|
| **RWKV-6-World-1.6B** | 1.6B | Finch v6 | `Q8_0` / `Q4_K_M` | ~1.7 GB / ~1.0 GB |
| **RWKV-7-Goose-World3-1.5B** | 1.5B | Goose v7 | `Q8_0` / `Q4_K_M` | ~1.6 GB / ~0.95 GB |
| **RWKV-6-World-3B** | 3.0B | Finch v6 | `Q4_K_M` | ~1.9 GB |
| **RWKV-6-World-7B** | 7.0B | Finch v6 | `Q4_K_M` | ~4.3 GB |

---

## 3. Hardware Requirements & Memory Profile

Because RWKV requires no growing KV cache, host memory requirements are strictly bounded by model weights and a tiny fixed recurrent state buffer:

| Model | Quant | Context Length | Host RAM | Recommended Backend |
|---|---|---|---|---|
| **RWKV-7 1.5B** | Q4_K_M | 1k to 100k+ tokens | **~1.2 GB (Fixed)** | CPU SIMD |
| **RWKV-6 1.6B** | Q8_0 | 1k to 100k+ tokens | **~1.9 GB (Fixed)** | CPU SIMD |
| **RWKV-6 7B** | Q4_K_M | 1k to 100k+ tokens | **~4.9 GB (Fixed)** | CPU SIMD |

---

## 4. Usage & Code Examples

### C# High-Level API

```csharp
using OpenTail.Stingray;
using OpenTail.Stingray.Executors;

// 1. Load RWKV-7 checkpoint (runs on CPU SIMD)
using var model = Model.Load(new ModelParams("models/rwkv7-goose-world3-1.5b-q8_0.gguf")
{
    Backend = "cpu"
});

// 2. Allocate context
using var context = model.CreateContext(new ContextParams { ContextSize = 4096 });
var executor = new InteractiveExecutor(context);
var session = new ChatSession(executor);

// 3. Stream generated response with O(1) memory
await foreach (var chunk in session.ChatChunksAsync("Explain how linear RNNs achieve constant-memory decoding."))
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
# Run RWKV-7 Goose on CPU
stingray -m models/rwkv7-goose-world3-1.5b-q8_0.gguf -p "In a quiet forest at dawn," --backend cpu
```
