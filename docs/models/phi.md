# Microsoft Phi Family (Phi-2, Phi-3, Phi-3.5, PhiMoE)

[← Back to Architecture Cards](README.md)

Related: [llama](llama.md), [mistral](mistral.md), [smollm](smollm.md), [olmoe](olmoe.md). Verification: [STATUS.md](../STATUS.md); measured commands: [RUNNING.md](../RUNNING.md).

| Property | Value |
|---|---|
| **Provider** | Microsoft Research |
| **GGUF Architecture Keys** | `phi2`, `phi3`, `phimoe` |
| **Engine Implementations** | `PhiArchitectures.Phi2`, `Phi3`, `Phimoe` |
| **Forward Pass Factory** | `CommonForwardPassFactory.CreateDense` / `CreateMoe` |
| **Modalities** | Text |
| **Thinking Mode** | Optional / Model dependent |
| **Tool Calling & JSON BNF** | Fully supported |
| **Status & Confidence** | 🟢 **Admitted & Golden-Verified** (`Phi-3.5-MoE-instruct Q3_K_M vs llama-server: 24/24 exact greedy tokens`) |

---

## 1. Overview & Architectural Highlights

Microsoft's Phi series prioritizes "textbook quality" synthetic and curated pre-training data, achieving reasoning and mathematical performance matching models 3–5× their parameter size.

### Key Architectural Characteristics
* **Phi-2:**
  * Uses LayerNorm with learned bias.
  * Parallel residual connections (attention and MLP compute concurrently on the same normalized input).
  * NeoX-style partial Rotary Position Embedding (RoPE applied to a fraction of head dimensions).
* **Phi-3 & Phi-3.5 (Dense):**
  * RMSNorm with bias: Unlike standard Llama RMSNorm, Phi-3 layers incorporate an explicit additive bias term (`UsesLayerNorm = false`).
  * LongRoPE (Su scaled RoPE): Switches dynamically between short factors ($c \le 4096$) and long factors ($c > 4096$) with attention factor scaling (`rope.scaling.attn_factor`) to support context lengths up to 128k tokens without fine-tuning degradation.
* **Phi-3.5-MoE (`phimoe`):**
  * 16 routed experts, with 2 active experts per token (6.6B active parameters out of 41.9B total).
  * Top-$k$ weight renormalization: Expert router logits are normalized across chosen top-$k$ experts (`NormalizeMoeTopKWeights = true`).
  * Explicit output bias support.

---

## 2. Checkpoints & Recommended GGUFs

| Model | Hugging Face Repository | Parameters | Active Parameters | Context | Recommended Quantization | Typical Size |
|---|---|---|---|---|---|---|
| **Phi-2** | [microsoft/phi-2](https://huggingface.co/microsoft/phi-2) *(GGUF: [TheBloke/phi-2-GGUF](https://huggingface.co/TheBloke/phi-2-GGUF))* | 2.7B | 2.7B | 2k | `Q4_K_M` | ~1.7 GB |
| **Phi-3-mini-4k-instruct** | [microsoft/Phi-3-mini-4k-instruct-gguf](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf) | 3.8B | 3.8B | 4k | `Q4_K_M` | ~2.3 GB |
| **Phi-3.5-mini-instruct** | [microsoft/Phi-3.5-mini-instruct](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) *(GGUF: [bartowski/Phi-3.5-mini-instruct-GGUF](https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF))* | 3.8B | 3.8B | 128k | `Q4_K_M` | ~2.3 GB |
| **Phi-3.5-MoE-instruct** | [microsoft/Phi-3.5-MoE-instruct](https://huggingface.co/microsoft/Phi-3.5-MoE-instruct) *(GGUF: [bartowski/Phi-3.5-MoE-instruct-GGUF](https://huggingface.co/bartowski/Phi-3.5-MoE-instruct-GGUF))* | 41.9B | 6.6B | 128k | `Q3_K_M` / `Q4_K_M` | ~18.5 GB / ~24.0 GB |

---

## 3. Hardware Requirements & Performance Profile

On the reference AMD Ryzen 7 5700G (CPU AVX2 SIMD):
* **Microsoft Phi-2:** Prompt prefill ~5.4 tok/s; greedy decode ~11.4 tok/s (2026-09-28).
* **Microsoft Phi-3 Mini 4K:** Prompt prefill ~5.6 tok/s; greedy decode ~10.8 tok/s (2026-09-28).

| Model | Quant | Context | Host RAM (CPU) | Recommended Placement |
|---|---|---|---|---|
| **Phi-2** | Q4_K_M | 2,048 | ~2.8 GB | CPU SIMD |
| **Phi-3 Mini** | Q4_K_M | 4,096 | ~3.0 GB | CPU SIMD or 4 GB+ VRAM GPU |
| **Phi-3.5 MoE** | Q4_K_M | 4,096 | ~26.5 GB | 32 GB+ RAM or Multi-GPU |

---

## 4. Usage & Code Examples

### C# High-Level API

```csharp
using OpenTail.Stingray;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Executors;

// 1. Initialize Phi-3.5 Mini
using var model = Model.Load(new ModelParams("models/Phi-3.5-mini-instruct-Q4_K_M.gguf")
{
    Backend = "auto",
    GpuLayerCount = -1
});

// 2. Allocate context
using var context = model.CreateContext(new ContextParams { ContextSize = 4096 });
var executor = new InteractiveExecutor(context);
var session = new ChatSession(executor);

// 3. Query reasoning capabilities
await foreach (var chunk in session.ChatChunksAsync("Solve step-by-step: If 5 machines make 5 widgets in 5 minutes, how long do 100 machines take to make 100 widgets?"))
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
# Run Phi-3 Mini on CPU
stingray -m models/Phi-3-mini-4k-instruct-Q4_K_M.gguf -p "Instruct: Explain gravity in one sentence.\nOutput:"

# Run Phi-3.5-MoE with GPU layer offload
stingray -m models/Phi-3.5-MoE-instruct-Q4_K_M.gguf -g 24 --backend vulkan
```
