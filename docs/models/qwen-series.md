# Architecture Card: Qwen Series (Qwen 2.5 & Qwen 3)

> **Architectures:** `qwen2`, `qwen2moe`, `qwen3`  
> **Verification Status:** 🔬 Verified token-by-token against llama.cpp (see [docs/STATUS.md](../STATUS.md))  
> **Supported Sizes:** 0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B, and MoE variants  
> **Source References:** `OpenTail.Stingray.Core/Loaders/GgufModelLoader.cs`, `OpenTail.Stingray.Engine/ForwardPass.cs`

---

## 1. Architectural Highlights

The Qwen model family uses a decoder-only Transformer with several key characteristics:
- **Grouped Query Attention (GQA):** Standard across larger models to reduce KV cache memory footprints.
- **SwiGLU Activations:** Gated Feed-Forward Networks utilizing elementwise Swish and Linear projections.
- **Rotary Position Embeddings (RoPE):** High base frequency (typically $\theta = 1,000,000$) enabling context extensions up to 32k–128k tokens without degradation.
- **Sliding Window Attention (SWA):** Alternating full and sliding window attention layers in specific configurations.
- **RMSNorm with Learnable Scales:** Applied before attention and FFN blocks.

---

## 2. Stingray Engine Implementation

### Managed SIMD Execution
- In CPU execution (`OpenTail.Stingray.Cpu`), Qwen's projection matrices (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) are evaluated using AVX-512 / AVX2 fused dequantize-and-multiply kernels.
- Quantized formats supported: `Q4_K`, `Q6_K`, `Q8_0`, `Q4_0`, `Q5_K`, and `FP16`.

### Attention & KV Caching
- GQA heads are managed via `PagedKvCache` or standard continuous ring buffers.
- For memory-constrained environments, Qwen models support **TurboQuant** (`--tq` mode), compressing FP16 KV states down to 4-bit / 2-bit or Lloyd-Max codebooks. Note that QK-norm configurations require `--tq-mode kvarn`.

### Chat Template & Tool Calling
- Qwen models utilize the ChatML prompt structure:
  ```text
  <|im_start|>system
  You are a helpful assistant.<|im_end|>
  <|im_start|>user
  Hello!<|im_end|>
  <|im_start|>assistant
  ```
- Stingray includes a built-in Jinja chat template engine that parses the embedded GGUF template metadata `tokenizer.chat_template`, automatically formatting multi-turn messages, system prompts, and structured tool calls into the model's native format.

---

## 3. Practical Usage & Commands

### CLI Quick Start
```bash
# Path-free default (Qwen2.5 0.5B Instruct)
stingray setup chat
stingray chat "Summarize the advantages of GQA."

# Explicit path with GPU acceleration
stingray -m models/qwen2.5-7b-instruct-q4_k_m.gguf -p "Explain SIMD" --backend vulkan --gpu-layers -1
```

### Public C# API
```csharp
using OpenTail.Stingray;
using OpenTail.Stingray.Executors;

string modelPath = ModelHome.ResolveModelPath("qwen2.5-0.5b");
using var model = Model.Load(modelPath);
using var context = model.CreateContext(new ContextParams { ContextSize = 4096 });
var session = new ChatSession(new InteractiveExecutor(context));

await foreach (var token in session.ChatAsync("What is SwiGLU?"))
{
    Console.Write(token);
}
```

---

## 4. Verification Evidence

- **Golden Parity:** Verified token-by-token against `llama-cli` on `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` and `Qwen2.5-7B-Instruct-Q4_K_M.gguf`.
- **Fast Forward Pass Tests:** `QwenForwardPassTests` in `tests/OpenTail.Stingray.Tests.ForwardPass.Fast`.
- **Measured CPU Speed:** 60–70 tokens/sec on Ryzen 7 5700G (0.5B model, AVX2).
