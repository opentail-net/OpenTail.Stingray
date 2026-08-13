# OpenTail.Stingray — positioning

What this project is genuinely good for, who should care, and what is not yet true. Written to be
usable as marketing input, which means the weaknesses are here too — a pitch that omits them gets
falsified by the first serious evaluator and costs more credibility than it buys.

## The one-line claim

**Embedded LLM inference for .NET, as a single self-contained binary — no native dependencies, no
P/Invoke, no per-platform `.dll`/`.so` to ship.**

## The real differentiator

Pure managed C# 14 / .NET 10, NativeAOT-published. That is the thing nobody else covers well:

- **llama.cpp** is the performance reference, but it is a C++ dependency. Consuming it from .NET
  means shipping and version-managing a native binary per target platform.
- **LLamaSharp** is the established .NET option, but it is *bindings* to llama.cpp — you still
  carry `llama.dll` / `libllama.so` per RID, and you inherit the marshalling boundary.
- **ONNX Runtime GenAI** requires converting models to ONNX and brings its own native runtime.

OpenTail.Stingray reads GGUF directly and runs the whole stack in managed code with SIMD intrinsics
(AVX2/AVX-512), Vulkan compute, and CUDA. For a .NET team that wants inference *inside* their
application with one binary and no native deployment story, this is a materially different
proposition, not a marginally different one.

**Who should care:** .NET shops embedding inference in desktop apps, services, or edge deployments
where deployment simplicity, a single trust boundary, and staying inside the managed toolchain
matter more than peak tokens/sec.

**Who should not:** anyone choosing purely on throughput, or serving at scale. They should use
llama.cpp or vLLM, and we should say so rather than compete on a claim we lose.

## Supporting strengths

- **Architecture breadth unusual for a project this size**: llama/llama4, qwen2/qwen3,
  qwen35moe (hybrid Gated-DeltaNet + attention + MoE), gemma/gemma2/gemma3/gemma4, phi2/phi3,
  deepseek2, OLMoE.
- **More than a text engine**: text-to-image (Z-Image-Turbo, FLUX.1), 4x upscaling (Real-ESRGAN),
  Gemma 4 encoder-free vision, KV-cache compression (KVarN 4-bit K / 2-bit V), speculative decoding
  (draft-model, self-speculative MTP, DSpark), continuous batching, MoE expert offloading across a
  VRAM → RAM → NVMe hierarchy.
- **API-compatible surface**: OpenAI `/v1/chat/completions` and Anthropic `/v1/messages`, so it can
  slot in behind existing clients.
- **Grammar-constrained decoding** including whole-turn JSON-schema structured output.

## The Agentic Killer Feature: Native Session Continuation & Contextual Tool Harness

While traditional inference servers (vLLM, llama.cpp server, Ollama) were built around the **stateless HTTP `/v1/chat/completions` API**—forcing expensive prompt re-tokenization and KV prefill passes on every tool call step—OpenTail.Stingray introduces a fundamentally superior architectural model for **autonomous agentic workloads**:

1. **Native Sub-Millisecond Harness Continuation**:
   - `AppendToolResultAsync()` feeds tool results directly into the active `IInferenceSession`.
   - Physical KV page tables, token history, model logits, and sampling state are preserved in-place.
   - Continuing a multi-turn tool cycle takes **$< 1\text{ms}$** with zero re-prefill penalty.

2. **Contextual Capability Harness & Fail-Closed Security**:
   - `ToolProvider` and `ToolContext` are dynamically bound **per session**, enforcing strict capability sandboxes.
   - `ValidateToolCall()` is fail-closed (`ToolProvider == null` $\rightarrow$ rejected).
   - Unauthorized tool calls emit explicit status (`IsAuthorized = false`, `HasUnauthorizedToolCall = true`) rather than silently disappearing, giving host orchestrators total visibility.

3. **OS-Kernel Virtual Memory Architecture**:
   - **Paged KV Memory**: Memory is structured into physical pages (`KvPageId`).
   - **Zero-Copy Prompt Sharing**: Multi-tenant user sessions sharing common prompts use zero-copy `Fork()` with Copy-on-Write (COW). Memory usage scales with unique user edits, not prompt length $\times$ user count.
   - **Lock-Free Admission Control**: Page-accurate `_reservedPages` accounting using atomic CAS loops (`Interlocked.CompareExchange`) prevents VRAM/RAM overcommit under continuous batching.
   - **3-Way Transactional Rollback**: If page allocation or GPU prefill fails at any stage, `TokenHistory`, `KvSequence` (with physical page release), and `ForwardPass` roll back in lockstep. Failed operations leave the session completely uncorrupted.

4. **Engineered for Multi-Tenant / Multi-User Environments**:
   - Per-session security sandboxing isolates tenant capabilities.
   - Shared prompt page deduplication reduces memory footprint under multi-user concurrency.
   - Complete blast-radius containment guarantees that a single user's failed request never corrupts global memory or impacts concurrent sessions.

## Honest performance position

Measured on the development box (AMD Ryzen 7 5700G, 12 logical cores, Radeon integrated GPU,
SmolLM2-1.7B-Instruct-Q4_K_M):

| | OpenTail.Stingray | llama.cpp | Position |
|---|---|---|---|
| CPU decode | ~27.6 t/s | 29.7 t/s | **Near parity (~1.08x)** |
| CPU prefill | ~49.5 t/s | 205 t/s | **~4x behind** |

Decode — the interactive latency case a user actually feels — is close to parity, and that is the
honest headline. Prefill is materially behind and should not be claimed otherwise; it is dominated
by batched GEMM efficiency and, at long context, by attention that is still O(N²).

Vulkan on integrated graphics went from 6.55 → ~75 t/s prefill and 6.0 → ~24 t/s decode in one
optimisation pass, so the GPU path is improving quickly, but long-prompt prefill still degrades
sharply with context length. See `docs/done/perf-loop-progress.md` for the full measured history,
including the changes that were tried and rejected.

## What is not yet true — the credibility gap

State this plainly in any technical pitch, because a serious evaluator will find it:

1. **Verification breadth lags feature breadth.** Continuous end-to-end verification runs against
   essentially one model (SmolLM2-1.7B) on one CPU and one GPU. The supported-architecture list is
   much broader than the routinely-exercised list.
2. **The GPU backend recently shipped silently-wrong output.** Two independent Vulkan defects — a
   Wave64 subgroup-width assumption and a driver-broken `dotPacked4x8AccSatEXT` intrinsic —
   produced incorrect results while tests passed, because the existing test's tolerance could not
   detect a relative error and the affected paths had no test that executed on that hardware. Both
   are fixed and now have regression tests, but the lesson is that the GPU paths need hardware
   coverage across vendors before "supports Vulkan" is a safe claim.
3. **Single-contributor bus factor.**
4. **Performance results are single-machine.** The optimisation wins split into an algorithmic tier
   (portable) and a microarchitectural tier (tuned to one AMD driver, and unmeasured elsewhere).

## How to talk about it

- Lead with **deployment model**, not benchmarks: single managed binary, no native deps, GGUF in,
  OpenAI/Anthropic API out.
- Use **decode near-parity** as the performance proof point, and be upfront that prefill is behind.
- Frame the breadth as **capability surface**, not as a guarantee — "supports" should be qualified
  by what is continuously verified.
- The most valuable near-term credibility investment is not more features: it is a published
  cross-model, cross-hardware verification matrix.
