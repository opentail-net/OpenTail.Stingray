# Technical Guide: TurboQuant & KV Cache Management

> **Topic:** Memory-efficient Key-Value caching, TurboQuant quantization, and prefix cache reuse  
> **Audience:** Developers and DevOps operators configuring memory-constrained deployments  
> **Source References:** `OpenTail.Stingray.Engine/TurboQuant/`, `PagedKvCache.cs`

---

## 1. The KV Cache Bottleneck

During autoregressive language generation, previous Key and Value states must be retained to compute self-attention for subsequent tokens. In standard FP16 precision, KV cache memory scales linearly with sequence length:

$$\text{Memory (bytes)} = 2 \times \text{layers} \times \text{kv\_heads} \times \text{head\_dim} \times \text{seq\_len} \times 2$$

For an 8B model with 32 layers and 32k context, uncompressed FP16 KV cache consumes over **8 GB of RAM**, often exceeding the memory footprint of the quantized model weights themselves.

---

## 2. TurboQuant Quantization Modes

Stingray features **TurboQuant**, an adaptive KV-cache quantization engine that compresses cached states on the fly with minimal perplexity degradation:

| Mode | Bit Budget | Technique | Ideal Workloads |
|---|---|---|---|
| **`kvarn`** | 4-bit K / 2-bit V | Variable-bit asymmetric quantization with per-head outlier channels | General LLM decoding; models with QK-norm (e.g. Qwen 2.5/3, Gemma) |
| **`lloydmax`** | 3-bit / 3-bit | Non-uniform codebook clustering optimized for Gaussian tensor distributions | Standard LLaMA / Mistral models |
| **`fp8`** | 8-bit (E4M3 / E5M2) | Standard FP8 floating point quantization | High-fidelity generation with 50% memory reduction |

---

## 3. Configuring TurboQuant

### From the Command Line
Enable TurboQuant using the `--tq` and `--tq-mode` flags:

```bash
# Automatic optimal mode selection
stingray -m models/qwen2.5-7b-instruct-q4_k_m.gguf --ctx-size 16384 --tq

# Explicit kvarn mode for QK-norm architectures
stingray -m models/qwen2.5-7b-instruct-q4_k_m.gguf --ctx-size 32768 --tq --tq-mode kvarn
```

### From the C# API
```csharp
using OpenTail.Stingray;
using OpenTail.Stingray.Executors;

var contextParams = new ContextParams
{
    ContextSize = 16384,
    EnableTurboQuant = true,
    TurboQuantMode = "kvarn"
};

using var model = Model.Load("models/qwen2.5-7b-instruct-q4_k_m.gguf");
using var context = model.CreateContext(contextParams);
var session = new ChatSession(new InteractiveExecutor(context));
```

---

## 4. Paged Attention & Prefix Reuse

- **Paged Allocation:** Rather than allocating a contiguous 16k buffer upfront, `PagedKvCache` allocates memory in fixed 16-token pages on demand, preventing fragmentation.
- **Prefix Caching:** When multiple requests share common system prompts or documents, Stingray hashes prefix token sequences and reuses cached KV pages without re-evaluating the prompt prefill.
- *Caveat:* Prefix caching is automatically disabled for recurrent or hybrid models (such as Granite 4.0-H or Nemotron-H) whose internal states cannot be non-linearly rewound.
