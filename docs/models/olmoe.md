# OLMoE Family (OLMoE-1B-7B 64-Expert MoE)

[← Back to Architecture Cards](README.md)

| Property | Value |
|---|---|
| **Provider** | Allen Institute for AI (Ai2) |
| **GGUF Architecture Key** | `olmoe` |
| **Engine Implementations** | `OlmoArchitectures.Olmoe`, `ForwardPass.Moe.cs`, `PerChannelRmsNorm` |
| **Forward Pass Factory** | `CommonForwardPassFactory.CreateDense` / `CreateMoe` |
| **Modalities** | Text |
| **MoE Configuration** | 64 routed experts, 8 active per token (~1.3B active parameters out of 6.9B total) |
| **Supported Backends** | CPU (AVX2/AVX-512), Vulkan GPU, CUDA GPU |
| **Status & Confidence** | 🟢 **Admitted & Golden-Verified** (`OLMoE-1B-7B Instruct`, Perplexity parity & Vulkan/CPU prefill handoff verified) |

---

## 1. Overview & Architectural Highlights

OLMoE (Open Language Model of Experts), developed by the Allen Institute for AI (Ai2), is a fully open Mixture-of-Experts architecture. It pairs 64 fine-grained experts with high-throughput routing, achieving Llama 3 8B performance while consuming only ~1B active parameters per token.

### Key Architectural Characteristics
* **Whole-Vector Per-Channel QK Normalization:**
  * Unlike standard per-head RMSNorm where normalization is bounded within each head dimension ($d_h$), OLMoE computes an RMS norm across the **entire projection vector** spanning all heads ($d_{\text{heads}} \times d_h = 2048$ elements) before applying per-channel weights.
  * In Stingray, `ForwardPass.Helpers.cs` implements exact whole-vector `PerChannelRmsNorm` prior to RoPE rotation.
* **Raw Post-Softmax Expert Gating:**
  * OLMoE skips top-$k$ weight renormalization, using raw post-softmax probabilities directly for expert weighting.
* **No Shared Expert:**
  * All active computation is routed through the 8 selected experts, maximizing parameter specialization.
* **Heterogeneous GPU & CPU Prefill Handoff:**
  * Fully verified for GPU prefill handoff (`PrefillHandoffFamilies.cs`), achieving byte-exact K/V cache replication across both Vulkan and CUDA backends.

---

## 2. Checkpoints & Recommended GGUFs

| Model | Parameters | Active Parameters | Context | Recommended Quantization | Typical Size |
|---|---|---|---|---|---|
| **OLMoE-1B-7B-0924-Instruct** | 6.9B | 1.3B | 4,096 | `Q4_K_M` / `Q8_0` | ~4.3 GB / ~7.2 GB |
| **OLMoE-1B-7B-0924 (Base)** | 6.9B | 1.3B | 4,096 | `Q4_K_M` | ~4.3 GB |

---

## 3. Hardware Requirements & Performance Profile

On the reference AMD Ryzen 7 5700G (CPU AVX2 SIMD):
* **OLMoE 1B-7B Instruct (`OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf`):**
  * Prompt prefill: ~36.2 tok/s.
  * Greedy decode: ~23.3 tok/s (2026-09-28).

| Model | Quant | Context | Host RAM (CPU) | Recommended Placement |
|---|---|---|---|---|
| **OLMoE-1B-7B** | Q4_K_M | 4,096 | ~4.8 GB | CPU SIMD or 6 GB+ VRAM GPU |
| **OLMoE-1B-7B** | Q8_0 | 4,096 | ~7.8 GB | 16 GB+ RAM or 8 GB+ VRAM GPU |

---

## 4. Usage & Code Examples

### C# High-Level API

```csharp
using OpenTail.Stingray;
using OpenTail.Stingray.Executors;

// 1. Load OLMoE with auto GPU layer offload
using var model = Model.Load(new ModelParams("models/OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf")
{
    Backend = "auto",
    GpuLayerCount = -1
});

// 2. Allocate context
using var context = model.CreateContext(new ContextParams { ContextSize = 4096 });
var executor = new InteractiveExecutor(context);
var session = new ChatSession(executor);

// 3. Fast generation
await foreach (var chunk in session.ChatChunksAsync("Explain the difference between dense and sparse mixture-of-experts."))
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
# Run OLMoE 1B-7B with CPU SIMD execution
stingray -m models/OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf -p "Explain why sparse MoE models are fast during inference."
```
