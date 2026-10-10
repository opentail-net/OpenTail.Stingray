# Mistral & Mixtral Family (Mistral 7B, Mistral 3, Mixtral 8x7B/8x22B, Ministral)

[← Back to Architecture Cards](README.md)

| Property | Value |
|---|---|
| **Provider** | Mistral AI |
| **GGUF Architecture Keys** | `mistral`, `mistral3`, `ministral`, and relabelled `llama` |
| **Engine Implementations** | `OtherAdmittedArchitectures.Mistral3`, `Ministral`, `LlamaArchitecture` |
| **Forward Pass Factory** | `CommonForwardPassFactory.CreateDense` / `CreateMoe` |
| **Modalities** | Text, Vision (Pixtral 12B via `UnifiedVisionPipeline`) |
| **Thinking Mode** | Optional / Instruction fine-tune dependent |
| **Tool Calling & JSON BNF** | Native function calling format with Tekken tokenizer support |
| **Status & Confidence** | 🟢 **Admitted & Golden-Verified** (`Mistral 7B Instruct v0.3`, `Mixtral 8x7B`, `Ministral 8B`) |

---

## 1. Overview & Architectural Highlights

Mistral AI revolutionized open LLMs by combining high-efficiency Sliding-Window Attention (SWA), dense GQA transformers, and Sparse Mixture of Experts (MoE) routing.

### Key Architectural Characteristics
* **Sliding Window Attention (SWA):** Earlier layers attend to the previous $W = 4096$ tokens, reducing KV cache computation while maintaining long-context coherence via multi-layer receptive fields.
* **Tekken Tokenizer Support:** Modern Mistral releases (Mistral 3, Ministral, Pixtral) employ the high-compression Tekken BPE tokenizer (~131k vocabulary), replacing the older SentencePiece Llama format.
* **Sparse Mixture-of-Experts (Mixtral 8x7B / 8x22B):**
  * Softmax top-2 gating routing tokens dynamically across 8 expert networks per feed-forward block.
  * In Stingray, MoE expert GEMMs run parallelized across AVX-512/AVX2 threads on CPU, or batch-fused on GPU.
* **Automatic Recognition of Relabelled GGUFs:**
  * Many community conversions (e.g. `bartowski/Mistral-Small-3.1-24B-Instruct-GGUF`) declare `general.architecture = "llama"`.
  * Stingray's `RecognizeRelabelledFile` inspector automatically identifies Mistral 3 metadata and routes the graph to the correct Mistral execution pipeline.

---

## 2. Checkpoints & Recommended GGUFs

| Model | Hugging Face Repository | Parameters | Active Parameters | Context | Recommended Quantization | Typical Size |
|---|---|---|---|---|---|---|
| **Mistral-7B-Instruct-v0.3** | [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) *(GGUF: [bartowski/Mistral-7B-Instruct-v0.3-GGUF](https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF))* | 7.2B | 7.2B | 32k | `Q4_K_M` | ~4.4 GB |
| **Ministral-8B-Instruct-2410** | [mistralai/Ministral-8B-Instruct-2410](https://huggingface.co/mistralai/Ministral-8B-Instruct-2410) *(GGUF: [bartowski/Ministral-8B-Instruct-2410-GGUF](https://huggingface.co/bartowski/Ministral-8B-Instruct-2410-GGUF))* | 8.0B | 8.0B | 32k | `Q4_K_M` | ~5.0 GB |
| **Mixtral-8x7B-Instruct-v0.1** | [mistralai/Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) *(GGUF: [TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF](https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF))* | 46.7B | 12.9B | 32k | `Q4_K_M` | ~26.4 GB |
| **Mistral-Small-3.1-24B-Instruct** | [mistralai/Mistral-Small-3.1-24B-Instruct-2503](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503) *(GGUF: [bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF](https://huggingface.co/bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF))* | 24.0B | 24.0B | 128k | `Q4_K_M` | ~14.3 GB |

---

## 3. Hardware Requirements & Performance Profile

On the reference AMD Ryzen 7 5700G (CPU AVX2 SIMD):
* **Mistral 7B Instruct v0.3:** Prompt prefill ~7.6 tok/s; greedy decode ~8.3 tok/s (128 tokens, CPU `-g 0`).
* **Ministral 8B Instruct:** Prompt prefill ~5.7 tok/s; greedy decode ~7.5 tok/s (CPU `-g 0`).

| Model | Quant | Context | RAM (CPU Run) | Recommended Backend |
|---|---|---|---|---|
| **Mistral 7B** | Q4_K_M | 4,096 | ~5.2 GB | CPU SIMD or 6 GB+ VRAM GPU |
| **Ministral 8B** | Q4_K_M | 4,096 | ~5.8 GB | CPU SIMD or 8 GB+ VRAM GPU |
| **Mixtral 8x7B** | Q4_K_M | 4,096 | ~29.0 GB | 32 GB+ System RAM or Multi-GPU |
| **Mistral Small 24B** | Q4_K_M | 8,192 | ~17.5 GB | Discrete GPU (Vulkan/CUDA) or High-RAM CPU |

---

## 4. Usage & Code Examples

### C# Streaming Chat

```csharp
using OpenTail.Stingray;
using OpenTail.Stingray.Executors;

// 1. Load Mistral 7B with CPU SIMD or auto GPU offload
using var model = Model.Load(new ModelParams("models/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf")
{
    Backend = "auto",
    GpuLayerCount = -1
});

// 2. Allocate context
using var context = model.CreateContext(new ContextParams { ContextSize = 4096 });
var executor = new InteractiveExecutor(context);
var session = new ChatSession(executor);

// 3. Stream output
await foreach (var chunk in session.ChatChunksAsync("Compare sliding-window attention with standard multi-head attention."))
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
# Run Mistral 7B on CPU with 8 worker threads
stingray -m models/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf -t 8 -p "Explain sparse MoE routing."

# Run Ministral 8B on Vulkan GPU
stingray -m models/Ministral-8B-Instruct-2410-Q4_K_M.gguf -g -1 --backend vulkan
```
