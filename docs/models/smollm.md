# SmolLM Family (SmolLM2 135M / 360M / 1.7B, SmolLM3)

[← Back to Architecture Cards](README.md)

| Property | Value |
|---|---|
| **Provider** | Hugging Face (Loubna Ben Allal et al.) |
| **GGUF Architecture Keys** | `llama` (SmolLM2), `smollm3` (SmolLM3) |
| **Engine Implementations** | `LlamaArchitecture.Descriptor`, `OtherAdmittedArchitectures.Smollm3` |
| **Forward Pass Factory** | `CommonForwardPassFactory.CreateDense` |
| **Modalities** | Text, Code, Reasoning |
| **Thinking Mode** | Supported in SmolLM3 (reasoning traces) |
| **Tool Calling & JSON BNF** | ChatML format with structured grammar support |
| **Status & Confidence** | 🟢 **Admitted & Golden-Verified** (`SmolLM2 135M/360M/1.7B; SmolLM3 24/24 exact greedy receipt`) |

---

## 1. Overview & Architectural Highlights

SmolLM2 and SmolLM3 are purpose-built compact language models designed by Hugging Face for on-device reasoning, edge computing, and ultra-high-throughput local processing. In OpenTail.Stingray, the SmolLM family serves as the primary benchmark suite for CPU SIMD kernels and worker-thread synchronization.

### Key Architectural Characteristics
* **Specialized 2-Stage Pre-Tokenizer:**
  SmolLM utilizes the `LLAMA_VOCAB_PRE_TYPE_SMOLLM` tokenizer cascade (`PreTokenizerPatterns.cs`), which splits numeric digits individually before applying GPT-2 style byte-level BPE regex matching.
* **Non-GQA Small Shapes vs High GQA Scaling:**
  * SmolLM2-135M and 360M have tight hidden dimensions ($d = 576$ and $960$), requiring low-latency thread wakeups.
  * SmolLM2-1.7B uses full multi-head attention with 384 KiB/token KV footprint, heavily benefiting from Stingray's int8 quantized prefill kernels.
* **SmolLM3 NoPE Interleaving:**
  SmolLM3 incorporates reasoning traces with periodic position embeddings: RoPE is applied on 3 out of 4 layers, skipping rotation every 4th layer (`NoRopeLayerStep = 4`).
* **Optimized Spin-Park Worker Pool:**
  Because small models spend only a few microseconds per matrix-vector product, Stingray's `SpinParkWorkerPool` prevents OS thread sleep penalties, delivering up to +27% decode throughput over `Parallel.For`.

---

## 2. Checkpoints & Recommended GGUFs

| Model | Hugging Face Repository | Parameters | Context | Recommended Quantization | Typical Size |
|---|---|---|---|---|---|
| **SmolLM2-135M-Instruct** | [HuggingFaceTB/SmolLM2-135M-Instruct-GGUF](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct-GGUF) | 135M | 8k | `Q4_K_M` / `Q8_0` | ~95 MB / ~145 MB |
| **SmolLM2-360M-Instruct** | [HuggingFaceTB/SmolLM2-360M-Instruct-GGUF](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct-GGUF) | 360M | 8k | `Q4_K_M` / `Q8_0` | ~230 MB / ~385 MB |
| **SmolLM2-1.7B-Instruct** | [HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF) | 1.71B | 8k | `Q4_K_M` | ~1.05 GB |
| **SmolLM3-Reasoning** | [HuggingFaceTB/SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) | 3.0B | 32k | `Q4_K_M` | ~1.95 GB |

---

## 3. Hardware Requirements & Performance Profile

On the reference AMD Ryzen 7 5700G (CPU AVX2 SIMD):
* **SmolLM2 135M:** Prompt prefill ~48.5 tok/s; greedy decode ~21.2 tok/s.
* **SmolLM2 360M:** Prompt prefill ~89.2 tok/s; greedy decode ~51.4 tok/s.
* **SmolLM2 1.7B:** Prompt prefill ~86.2 tok/s; greedy decode ~29.8 tok/s (`-g 0`, 128 tokens).
* **SmolLM3 Q4_K_M:** Prompt prefill ~119.5 tok/s; greedy decode ~11.9 tok/s.

| Model | Quant | Context | Host RAM (CPU) | Recommended Placement |
|---|---|---|---|---|
| **135M** | Q4_K_M | 2,048 | < 0.5 GB | Embedded / Any CPU |
| **360M** | Q4_K_M | 4,096 | ~0.7 GB | Embedded / Any CPU |
| **1.7B** | Q4_K_M | 4,096 | ~1.5 GB | CPU SIMD or entry GPU |
| **SmolLM3** | Q4_K_M | 8,192 | ~2.5 GB | CPU SIMD or 4 GB GPU |

---

## 4. Usage & Code Examples

### C# High-Level API

```csharp
using OpenTail.Stingray;
using OpenTail.Stingray.Executors;

// 1. Initialize SmolLM2 1.7B for low-memory edge chat
using var model = Model.Load(new ModelParams("models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf")
{
    Backend = "cpu"
});

// 2. Allocate context
using var context = model.CreateContext(new ContextParams { ContextSize = 4096 });
var executor = new InteractiveExecutor(context);
var session = new ChatSession(executor);

// 3. Fast generation
await foreach (var chunk in session.ChatChunksAsync("List 3 practical uses of small language models on edge devices."))
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
# Instant edge prompt completion
stingray -m models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf -p "Once upon a time in a byte-sized universe" --temp 0.7
```
