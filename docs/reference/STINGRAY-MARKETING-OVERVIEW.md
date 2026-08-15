# OpenTail.Stingray — The In-Process AI Inference Engine for .NET & Local Autonomous Agents

> **High-Performance Local AI Inference. Zero Sidecars. Zero Network Latency. Native .NET 10 & NativeAOT.**

---

## Executive Summary

**OpenTail.Stingray** is the world’s most advanced, in-process AI inference engine built natively in managed **.NET 10** and **NativeAOT**. 

Unlike traditional AI wrappers (such as Ollama or vLLM) that require heavy Python sidecars, external C++ processes, or HTTP socket serialization, **Stingray runs directly inside your application process**. It delivers hardware-accelerated local LLM inference, zero-copy session multiverse branching, native skill integration, and cutting-edge memory compression with zero external dependencies.

---

## Key Value Propositions

### ⚡ 1. In-Process Zero-Latency Architecture
- **Zero Network Overhead**: Eliminates HTTP socket serialization and IPC process management.
- **NativeAOT Ready**: Compiles directly into a single, self-contained native binary with lightning-fast startup.
- **Pure .NET 10 Managed Code**: No P/Invoke bindings to external C++ DLLs or Python environments.

### 🧠 2. First-Class Native Skills, Instructions & Tools
Stingray is the first inference engine to elevate **Skills, Instructions, and Tools into native C# inference concepts** (`ISkill`, `IInstruction`, `ITool`):
- **0 ms Skill Re-Prefill**: Skill instructions participate in Paged KV Prefix Caching, reusing prompt context instantly across turns.
- **Guaranteed Tool Sampling**: Tools arm Stingray’s token-level `ToolGrammar` constraint sampler, guaranteeing 100% valid JSON/XML tool calls without output syntax corruption.
- **Clean Application Boundaries**: Runtimes map directly onto Stingray without hacking opaque prompt strings.

### 🌲 3. Zero-Copy Session Multiverse (`session.Fork()`)
- **Instant Branching**: Spawn child inference sessions in microseconds using copy-on-write page tables.
- **Parallel Exploration**: Explore multiple reasoning paths or agent workflows in parallel without duplicating physical KV memory.
- **Automatic KV Memory Governor**: Auto-suspends idle sessions to disk and rehydrates on demand to guarantee memory safety.

### 🚀 4. Cutting-Edge Attention & Compression (DeepSeek MLA)
- **3.5× KV Memory Compression**: Native support for **Multi-head Latent Attention (MLA)** in DeepSeek-V2 (`deepseek2`).
- **Memory-Efficient Local AI**: Retains compressed 576-dim latent vectors in memory rather than expanding full 2,048-dim head tensors, saving 70%+ RAM and unlocking massive L3 cache bandwidth.

### 🏎️ 5. Extreme Hardware Acceleration
- **CPU**: Hand-tuned AVX2/AVX-512 SIMD kernels with fused micro-GEMM achieving **15× to 17× micro-kernel speedups**.
- **GPU**: Native Vulkan compute shaders with SPIR-V Path 2 tiling, cooperative matrix acceleration, and batched prefill/decode.

---

## Competitive Matrix

| Feature / Metric | **OpenTail.Stingray** | **Ollama / llama.cpp** | **vLLM** |
| :--- | :--- | :--- | :--- |
| **Execution Environment** | **In-Process .NET 10 / NativeAOT** | Separate Process / C++ Server | Python Process / PyTorch |
| **Native Skills & Tools API** | **First-Class (`ISkill`, `ITool`)** | ❌ None (Prompt String Injection) | ❌ None (API Server Wrapper) |
| **Grammar Sampling** | **Token-Level `ToolGrammar`** | GBNF Grammars | Outlines / Guided Decoding |
| **Session Multiverse Branching** | **Zero-Copy (`Fork()`)** | ❌ None | ❌ None |
| **State Management** | **Paged KV Cache + Cold Tiering** | Ring Buffer / Basic Paged KV | PagedAttention (Python Server) |
| **DeepSeek MLA Support** | **Native Compressed Cache** | Supported | Supported |

---

## Supported Architectures

Stingray natively supports leading open-weights models across GGUF quantization formats (Q4_K, Q8_0, Q2_K, FP16):
- **DeepSeek**: DeepSeek-V2-Lite (`deepseek2` with MLA + DeepSeekMoE)
- **Qwen**: Qwen 2.5, Qwen 3, Qwen 3.5 MoE
- **Mistral / Llama**: Mistral 3, Ministral, Llama 3, SmolLM2
- **Google Gemma**: Gemma 2, Gemma 4
- **Enterprise**: Command-R, Granite, OLMoE

---

## Who Is Stingray Built For?

1. **Enterprise .NET Developers**: Embed local AI directly into desktop, server, or cloud applications with enterprise security and zero external process dependencies.
2. **Autonomous Agent Developers**: Build high-throughput, multi-agent systems with zero-copy session branching, prefix-cached skill prompts, and guaranteed grammatical tool calling.
3. **Edge & On-Premise Engineers**: Deploy ultra-fast local LLM inference on consumer CPUs and integrated Vulkan GPUs without requiring massive cloud infrastructure.

---

*Built with pride by the **OpenTail team** — [opentail.net](https://opentail.net)*
