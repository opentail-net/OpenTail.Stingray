# Liquid LFM2 Family (LFM2-1.2B & LFM2-MoE 8B-A1B)

[← Back to Architecture Cards](README.md)

| Property | Value |
|---|---|
| **Provider** | Liquid AI |
| **GGUF Architecture Keys** | `lfm2`, `lfm2moe` |
| **Engine Implementations** | `HybridRecurrentArchitectures.Lfm2`, `Lfm2moe`, `ForwardPass.ShortConv.cs` |
| **Forward Pass Factory** | `CommonForwardPassFactory.CreateDense` with gated short-convolution dispatch |
| **Modalities** | Text |
| **Hybrid Kind** | `HybridKind.ShortConv` (Interleaved Causal Depthwise Conv & Attention) |
| **Status & Confidence** | 🟢 **Admitted & Golden-Verified** (`LFM2-1.2B PPL 10.919 vs 10.916; LFM2-MoE 8B bit-for-bit golden parity`) |

---

## 1. Overview & Architectural Highlights

Liquid AI's LFM2 (Liquid Foundation Model) blends non-transformer recurrent signal processing with standard multi-head attention. Instead of treating every layer as an attention block, LFM2 delegates lower layers to low-latency **gated short-convolution mixers**, dramatically reducing memory traffic and KV cache pressure.

### Key Architectural Characteristics
* **Gated Short-Convolution Layers:**
  * Identified in GGUF metadata where `attention.head_count_kv == 0`.
  * Evaluated via `ForwardPass.ShortConv.cs`: input projections map to $(b, c, x)$ vectors, followed by causal depthwise 1D convolution over a small temporal window ($k = 3$ or $4$), SiLU gating, and residual FFN addition.
* **Interleaved Attention Layers:**
  * Selected higher layers run standard GQA attention blocks with RoPE.
* **LFM2-MoE Routing:**
  * MoE feed-forward networks employ sigmoid gating combined with an exponentiated selection bias (`exp_probs_b`), followed by top-$k$ weight renormalization.
  * In Stingray, batched prefill matches token-by-token bit-for-bit with llama.cpp, controllable via `STINGRAY_LFM2_MOE_BATCHED_PREFILL`.
* **Automatic BOS Insertion:**
  * LFM2 depends strictly on a leading Begin-Of-Sequence (`<s>`) token to initialize internal convolution states; Stingray ensures BOS prepending even when raw prompt strings omit it.

---

## 2. Checkpoints & Recommended GGUFs

| Model | Hugging Face Repository | Parameters | Active Parameters | Context | Recommended Quantization | Typical Size |
|---|---|---|---|---|---|---|
| **LFM2-1.2B** | [LiquidAI/LFM-1B](https://huggingface.co/LiquidAI/LFM-1B) | 1.2B | 1.2B | 32k | `Q8_0` / `Q4_K_M` | ~1.3 GB / ~0.75 GB |
| **LFM2-8B-A1B-MoE** | [LiquidAI/LFM-7B](https://huggingface.co/LiquidAI/LFM-7B) | 8.0B | 1.1B | 32k | `Q4_K_M` | ~4.8 GB |

---

## 3. Hardware Requirements & Performance Profile

On the reference AMD Ryzen 7 5700G (CPU AVX2 SIMD):
* **LFM2 1.2B (`LFM2-1.2B-Q8_0.gguf`):** Prompt prefill ~101 tok/s (over 538 tokens); greedy decode 28–30 tok/s (2026-09-28).

| Model | Quant | Context | Host RAM (CPU) | Recommended Placement |
|---|---|---|---|---|
| **LFM2 1.2B** | Q8_0 | 4,096 | ~1.3 GB | CPU SIMD |
| **LFM2 8B MoE** | Q4_K_M | 8,192 | ~5.5 GB | CPU SIMD or 6 GB+ GPU |

---

## 4. Usage & Code Examples

### C# High-Level API

```csharp
using OpenTail.Stingray;
using OpenTail.Stingray.Executors;

// 1. Initialize Liquid LFM2
using var model = Model.Load(new ModelParams("models/LFM2-1.2B-Q8_0.gguf")
{
    Backend = "cpu"
});

// 2. Allocate context
using var context = model.CreateContext(new ContextParams { ContextSize = 4096 });
var executor = new InteractiveExecutor(context);
var session = new ChatSession(executor);

// 3. Fast generation
await foreach (var chunk in session.ChatChunksAsync("Summarize the benefits of hybrid short-convolution models."))
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
# Run Liquid LFM2 on CPU
stingray -m models/LFM2-1.2B-Q8_0.gguf -p "The future of edge foundation models is"
```
