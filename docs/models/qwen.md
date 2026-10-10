# Qwen Family (Qwen 2.5, Qwen 3, Qwen 3.5 MoE / Gated DeltaNet, Qwen-Coder)

[← Back to Architecture Cards](README.md)

Related: [Qwen3.5 / 3.6 / 3.8 hybrid card](qwen35.md), [Qwen-VL](qwen-vl.md), [Qwen Audio](qwen-audio.md), [Qwen Series (older overview)](qwen-series.md). Measured commands: [RUNNING.md](../RUNNING.md); verification: [STATUS.md](../STATUS.md).

| Property | Value |
|---|---|
| **Provider** | Alibaba Cloud (Qwen Team) |
| **GGUF Architecture Keys** | `qwen`, `qwen2`, `qwen2moe`, `qwen3`, `qwen3moe`, `qwen35`, `qwen35moe`, `qwen2vl`, `qwen3vl` |
| **Engine Implementations** | `QwenArchitectures` (`Qwen2`, `Qwen3`, `Qwen3moe`, `Qwen35`, `Qwen35moe`) |
| **Forward Pass Factory** | `CommonForwardPassFactory.CreateDense` / `CreateHybridGdn` |
| **Modalities** | Text, Code, Multimodal Vision (`Qwen2.5-VL`, `Qwen3-VL`) |
| **Thinking Mode** | Supported (`<think>...</think>` traces in Qwen 3 and reasoning fine-tunes) |
| **Tool Calling & JSON BNF** | Native ChatML function calling & JSON schema constrained generation |
| **Status & Confidence** | 🟢 **Admitted & Golden-Verified** (`Qwen2.5 0.5B default catalogue model; Qwen3.8-27B 24/24 exact greedy receipt`) |

---

## 1. Overview & Architectural Highlights

The Qwen series represents Alibaba's state-of-the-art open model lineage. In OpenTail.Stingray, Qwen is a tier-1 family:
* **`qwen2.5-0.5b`** serves as Stingray's default out-of-the-box chat catalogue model (`stingray setup chat`).
* Full hybrid recurrent architectures (**Qwen 3.5 / 3.6 / 3.8 Gated DeltaNet + MoE**) are natively implemented in managed C# (`ForwardPassFamily.HybridGdn`).

### Key Architectural Characteristics
* **SwiGLU & RMSNorm:** Standard bias-free SiLU-gated feed-forward networks with RMSNorm.
* **Weighted QK-Norm:** Query and key vectors undergo RMS normalization per-head to prevent logit explosion on long sequences. In Qwen, QK-norm occurs prior to RoPE.
* **Interleaved M-RoPE (Qwen2.5-VL / Qwen3-VL):** Multimodal Rotary Position Embedding with sections $[24, 20, 20, 0]$, enabling 3D spatio-temporal positioning.
* **Gated DeltaNet Hybrid Recurrence (Qwen 3.5 MoE):**
  * Replaces standard full attention on select layers with linear-time Gated DeltaNet recurrent memory states.
  * Every 4th block retains full attention (`full_attention_interval = 4`), combining infinite-receptive-field attention with fast $O(1)$ recurrent step updates.
  * Multi-Token Prediction (MTP) layer (block 64) supported and safely handled.
* **Unsloth Dynamic Quantization Support:** Full support for `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, and `IQ4_XS` dequantization kernels.

---

## 2. Checkpoints & Recommended GGUFs

| Model | Hugging Face Repository | Parameters | Architecture | Recommended Quant | Size | Purpose |
|---|---|---|---|---|---|---|
| **Qwen2.5-0.5B-Instruct** | [Qwen/Qwen2.5-0.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF) | 0.49B | Dense | `Q4_K_M` | 469 MB | **Stingray Default Chat** (`stingray setup chat`) |
| **Qwen2.5-1.5B-Instruct** | [Qwen/Qwen2.5-1.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF) | 1.54B | Dense | `Q4_K_M` | 986 MB | High-quality compact edge instruction |
| **Qwen2.5-7B-Instruct** | [Qwen/Qwen2.5-7B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF) | 7.61B | Dense | `Q4_K_M` (2 shards) | 4.68 GB | Robust reasoning, coding, and tool calling |
| **Qwen2.5-Coder-7B** | [Qwen/Qwen2.5-Coder-7B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF) | 7.61B | Dense | `Q4_K_M` | 4.68 GB | Premier open code-generation model |
| **Qwen3-0.6B-Base** | [Qwen/Qwen3-0.6B-GGUF](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF) | 0.6B | Dense | `Q8_0` | ~1.2 GB | Lightweight reasoning model with thinking |
| **Qwen3.8-27B** | [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) | 27B | Hybrid Gated DeltaNet | `UD-Q3_K_XL` | ~14.8 GB | Hybrid recurrent reasoning model; see the [hybrid card](qwen35.md) |

### Qwen3-Coder 30B-A3B

Sparse MoE coding model (`qwen3moe`, 128 experts, ~3B active per token; top-k weights normalised). It runs on the dense forward-pass path, not the hybrid one.

| Repository | File | Measured |
|---|---|---|
| [unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF](https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF) | `Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf` | About 17.6 GB peak RAM; decode 8.1-15.4 tok/s over 108 tokens (2026-10-01, large spread across passes); load about 35 s |

It is one of the two models verified on the Vulkan partial-offload MoE path (`-g N`, experts through a slot cache; logits within cosine 0.996-0.999 of CPU at 16 slots, 2026-10-03) and on CPU-prefill + KV handoff (2026-10-04). See [STATUS](../STATUS.md).

---

## 3. Hardware Requirements & Performance Profile

On reference AMD Ryzen 7 5700G (CPU AVX2 SIMD):
* **Qwen2.5 0.5B Instruct:** Prompt prefill ~318 tok/s (661 tokens); greedy decode ~62.9–75.0 tok/s.
* **Qwen2.5 1.5B Instruct:** Prompt prefill ~113 tok/s; greedy decode ~24.1 tok/s.
* **Qwen2.5-Coder 0.5B:** Prompt prefill ~90.8 tok/s; greedy decode ~57.0 tok/s.

| Model | Quant | Context | Host RAM (CPU) | Recommended Placement |
|---|---|---|---|---|
| **Qwen2.5 0.5B** | Q4_K_M | 4,096 | ~1.0 GB | Any CPU |
| **Qwen2.5 1.5B** | Q4_K_M | 4,096 | ~2.5 GB | CPU SIMD |
| **Qwen2.5 7B** | Q4_K_M | 8,192 | ~9.0 GB | 12 GB+ RAM CPU or 6 GB+ VRAM GPU |
| **Qwen3.8 27B** | UD-Q3_K_XL | 8,192 | ~18.5 GB | 32 GB+ RAM or Multi-GPU |

---

## 4. Usage & Code Examples

### Fast Chat with ModelHome

```csharp
using OpenTail.Stingray;
using OpenTail.Stingray.Executors;

// 1. Resolve default catalogue model or path
string modelPath = ModelHome.ResolveModelPath("qwen2.5-0.5b");

// 2. Load model and start session
using var model = Model.Load(modelPath);
using var context = model.CreateContext(new ContextParams { ContextSize = 4096 });
var executor = new InteractiveExecutor(context);
var session = new ChatSession(executor);

// 3. Stream tokens
await foreach (var chunk in session.ChatChunksAsync("Explain the difference between GQA and MHA."))
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
# First run setup (downloads verified bundle)
stingray setup chat

# Launch interactive chat
stingray chat

# Or run with explicit model and GPU offloading
stingray -m models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf -g -1 --backend vulkan
```
