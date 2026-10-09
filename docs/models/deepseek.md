# Architecture Card: DeepSeek Series (MLA & DeepSeekMoE)

> **Architectures:** `deepseek`, `deepseek2`, `deepseek3`  
> **Verification Status:** 🔬 Verified on DeepSeek-V2-Lite; MLA and MoE routing implemented (see [docs/STATUS.md](../STATUS.md))  
> **Supported Models:** DeepSeek-V2-Lite, DeepSeek-V2.5, DeepSeek-Coder-V2, DeepSeek-V3, DeepSeek-R1  
> **Source References:** `OpenTail.Stingray.Core/Loaders/GgufModelLoader.cs`, `OpenTail.Stingray.Engine/Attention/MlaAttention.cs`

---

## 1. Architectural Highlights

DeepSeek architectures deviate substantially from standard LLaMA-style designs through two fundamental innovations:

### Multi-Head Latent Attention (MLA)
Standard Multi-Head Attention (MHA) and Grouped Query Attention (GQA) store full-rank Key and Value tensors for every token, dominating memory consumption in long-context decoding.

MLA compresses Keys and Values into a single low-dimensional latent vector $c_t^{KV}$ (e.g. 512 or 576 dimensions):
1. **Down-projection:** $c_t^{KV} = W^{DKV} x_t$ (cached in the KV store).
2. **Up-projection during attention:** $K_t = W^{UK} c_t^{KV}$ and $V_t = W^{UV} c_t^{KV}$.
3. **Decoupled RoPE:** Because Rotary Position Embeddings cannot easily be applied to compressed latent representations without losing associativity, MLA decouples position information into a separate small Key vector $K_t^R = \text{RoPE}(W^{KR} x_t)$, which is concatenated with the up-projected keys.

This achieves **~3.5× to 4× KV cache compression**, allowing much larger context sizes in local RAM.

### DeepSeekMoE (Fine-Grained Experts)
- Replaces standard coarse experts with many smaller routed experts (e.g., 64–256 experts) alongside dedicated shared experts.
- Tokens route to top-$k$ experts using normalized affinity scores, maintaining computational cost while vastly increasing parameter capacity.

---

## 2. Stingray Engine Implementation

### Native Compressed KV Cache
- Rather than decompressing keys and values into full tensor shapes prior to cache storage, Stingray's `MlaAttention` caches the compressed latent vector $c_t^{KV}$ directly in memory.
- During decoding, matrix-matrix operations are reorganized: query projections are multiplied by $W^{UK}$ and $W^{UV}$ before computing dot-products against the cached latents, minimizing VRAM / RAM bandwidth requirements.

### Int8 Prefill Optimization
- Per [ADR-0003](../reference/adr-0003-cpu-int8-prefill-default.md), DeepSeek-V2 CPU prefill defaults to int8 quantization to maximize SIMD throughput across high-context prompts.

### Reasoning & Thinking Mode Support
- DeepSeek-R1 introduces internal reasoning tokens wrapped within `<think>` ... `</think>` tags.
- In the public API, `ChatSession.ChatChunksAsync` categorizes thinking tokens under `GenerateChunkKind.Thinking`, allowing applications to display collapsible reasoning sections separate from final responses.

---

## 3. Practical Usage & Commands

### CLI Invocation
```bash
# Run DeepSeek-V2-Lite with automatic hardware offload
stingray -m models/deepseek-v2-lite-chat-q4_k_m.gguf -p "Compare MLA against standard GQA."
```

### Public C# API with Thinking Stream Inspection
```csharp
using OpenTail.Stingray;
using OpenTail.Stingray.Executors;

using var model = Model.Load("models/deepseek-r1-distill-qwen-7b-q4_k_m.gguf");
using var context = model.CreateContext(new ContextParams { ContextSize = 4096 });
var session = new ChatSession(new InteractiveExecutor(context));

await foreach (var chunk in session.ChatChunksAsync("Solve this logic puzzle: ..."))
{
    if (chunk.Kind == GenerateChunkKind.Thinking)
    {
        Console.ForegroundColor = ConsoleColor.DarkGray;
        Console.Write(chunk.Text);
        Console.ResetColor();
    }
    else if (chunk.Kind == GenerateChunkKind.Text)
    {
        Console.Write(chunk.Text);
    }
}
```

---

## 4. Verification Evidence & Known Status

- **DeepSeek-V2-Lite:** Verified token-by-token greedy output against llama.cpp. Note: departure at token 9 observed under experimental Q3_K kernels; Q4_K_M and int8 prefill remain the verified standard (see [docs/STATUS.md](../STATUS.md) and [docs/1-correctness/bugstofix.md](../1-correctness/bugstofix.md)).
- **DeepSeek-V3 / R1:** MLA projection matrices, MoE routing, and parsing are validated in synthetic forward-pass test suites. Full multi-GPU sharding is tracked in `docs/9-external-hardware/`.
