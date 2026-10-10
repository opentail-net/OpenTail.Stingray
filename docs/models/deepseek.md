# DeepSeek Family (DeepSeek-V2, V2.5, Coder-V2, DeepSeek-V3, R1)

[← Back to Architecture Cards](README.md)

Related: [deepseek-ocr](deepseek-ocr.md), [qwen](qwen.md), [gpt-oss](gpt-oss.md). Verification: [STATUS.md](../STATUS.md); measured commands: [RUNNING.md](../RUNNING.md).

| Property | Value |
|---|---|
| **Provider** | DeepSeek AI |
| **GGUF Architecture Keys** | `deepseek`, `deepseek2`, `deepseek2_ocr`, `deepseek3` |
| **Engine Implementations** | `DeepSeek2Architectures.DeepSeek2`, `DeepSeek2Ocr`, `MlaAttention` |
| **Forward Pass Factory** | `CommonForwardPassFactory.CreateDense` / `CreateMoe` with native MLA cache |
| **Modalities** | Text, Code, Reasoning (`<think>` tokens), Vision OCR (`DeepSeek-OCR` & `OCR2`) |
| **Thinking Mode** | Native first-class support (`GenerateChunkKind.Thinking`) |
| **Tool Calling & JSON BNF** | Fully supported |
| **Status & Confidence** | 🟢 **Admitted & Verified** (`DeepSeek-V2-Lite` Level 2 Proven; V3/R1 MLA and MoE forward passes admitted) |

> [!NOTE]
> Per CLAUDE.md rule 14, experimental or unverified lineage branches (`deepseek4`, `deepseek41`, `deepseek32`) remain internal (`NotAdmitted` in `ModelCompatibility.cs`) and are not offered in public catalogues until verified with real checkpoints.

---

## 1. Architectural Highlights & Math

DeepSeek architectures introduce two fundamental breakthroughs in large-scale transformer efficiency:

### Multi-Head Latent Attention (MLA)
Standard Multi-Head Attention (MHA) and Grouped Query Attention (GQA) store full-rank Key and Value matrices per token, creating severe memory bottlenecks at long context:
* **Low-Rank Compression:** Keys and Values are jointly compressed into a low-dimensional latent vector $c_t^{KV} \in \mathbb{R}^{d_c}$ (e.g. 512 dimensions):
  $$c_t^{KV} = W^{DKV} x_t$$
* **Decoupled RoPE:** Rotary position embeddings cannot commute with low-rank compression. DeepSeek solves this by separating position encoding into an independent small vector:
  $$k_t^R = \text{RoPE}(W^{KR} x_t)$$
  The attention score is computed by combining the projected latent dot-product with the RoPE dot-product.
* **3.5× KV Memory Reduction:** OpenTail.Stingray's `MlaAttention` caches the compressed latent vector $c_t^{KV}$ directly in the KV buffer rather than uncompressing it, cutting KV RAM by ~70–75%.

### DeepSeekMoE (Fine-Grained Expert Routing)
* Replaces coarse MoE experts with many fine-grained experts (e.g. 64–256 experts), activating top-$k$ experts per token plus dedicated shared experts that are always evaluated.
* In Stingray, shared experts and routed experts are parallelized across SIMD worker threads.

---

## 2. Checkpoints & Recommended GGUFs

| Model | Hugging Face Repository | Parameters | Active Parameters | Context | Recommended Quantization | Typical Size |
|---|---|---|---|---|---|---|
| **DeepSeek-V2-Lite-Chat** | [deepseek-ai/DeepSeek-V2-Lite-Chat](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat) *(GGUF: [bartowski/DeepSeek-V2-Lite-Chat-GGUF](https://huggingface.co/bartowski/DeepSeek-V2-Lite-Chat-GGUF))* | 15.7B | 2.4B | 32k | `Q4_K_M` | ~8.9 GB |
| **DeepSeek-Coder-V2-Lite** | [deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct) *(GGUF: [bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF](https://huggingface.co/bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF))* | 15.7B | 2.4B | 32k | `Q4_K_M` | ~8.9 GB |
| **DeepSeek-R1-Distill-Qwen-1.5B** | [deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) *(GGUF: [bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF](https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF))* | 1.78B | 1.78B | 128k | `Q4_K_M` | ~1.1 GB |
| **DeepSeek-R1-Distill-Qwen-7B** | [deepseek-ai/DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) *(GGUF: [bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF](https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF))* | 7.61B | 7.61B | 128k | `Q4_K_M` | ~4.7 GB |
| **DeepSeek-R1-Distill-Llama-8B** | [deepseek-ai/DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B) *(GGUF: [bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF](https://huggingface.co/bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF))* | 8.03B | 8.03B | 128k | `Q4_K_M` | ~4.9 GB |
| **DeepSeek-V3 / R1 (Full)** | [deepseek-ai/DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3) · [R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) *(GGUF: [unsloth/DeepSeek-V3-GGUF](https://huggingface.co/unsloth/DeepSeek-V3-GGUF))* | 671B | 37B | 128k | Multi-shard Q4/Q8 | ~404 GB |

---

## 3. Hardware Requirements & Memory Estimation

| Model | Quant | Context | Host RAM (CPU) | Recommended Placement |
|---|---|---|---|---|
| **DeepSeek-V2-Lite** | Q4_K_M | 4,096 | ~10.5 GB | 16 GB+ CPU or 12 GB VRAM GPU |
| **R1-Distill-Qwen-1.5B** | Q4_K_M | 8,192 | ~2.5 GB | Any modern CPU |
| **R1-Distill-Qwen-7B** | Q4_K_M | 8,192 | ~9.0 GB | Vulkan GPU / CUDA or 16 GB+ CPU |
| **DeepSeek-V3 Full (671B)** | Q4_K_M | 8,192 | ~430 GB | Enterprise Multi-GPU cluster |

---

## 4. Usage & Code Examples

### Handling Reasoning Streams in C#

DeepSeek-R1 emits internal reasoning chains wrapped in `<think>...</think>`. Stingray parses this into typed chunks:

```csharp
using OpenTail.Stingray;
using OpenTail.Stingray.Executors;

// 1. Load DeepSeek reasoning checkpoint
using var model = Model.Load("models/deepseek-r1-distill-qwen-7b-q4_k_m.gguf");
using var context = model.CreateContext(new ContextParams { ContextSize = 4096 });
var executor = new InteractiveExecutor(context);
var session = new ChatSession(executor);

// 2. Stream and visually separate thinking traces from final output
await foreach (var chunk in session.ChatChunksAsync("Solve: How many 'r's are in strawberry?"))
{
    if (chunk.Kind == GenerateChunkKind.Thinking)
    {
        Console.ForegroundColor = ConsoleColor.DarkGray;
        Console.Write(chunk.Text);
        Console.ResetColor();
    }
    else if (chunk.Kind == GenerateChunkKind.Text)
    {
        Console.ForegroundColor = ConsoleColor.White;
        Console.Write(chunk.Text);
        Console.ResetColor();
    }
}
Console.WriteLine();
```

### CLI Command

```bash
# Run DeepSeek reasoning interactively
stingray -m models/deepseek-r1-distill-qwen-7b-q4_k_m.gguf -g -1 --backend vulkan
```
