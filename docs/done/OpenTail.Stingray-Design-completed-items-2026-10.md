# Completed items split from OpenTail.Stingray-Design.md (2026-11-01)

Finished items moved verbatim out of [../reference/OpenTail.Stingray-Design.md](../reference/OpenTail.Stingray-Design.md). Open items stay there. Each block is preceded by the heading it sat under. Relative links inside the blocks were rebased for this folder.

> from: ### Phase 1: CPU Reference Implementation ✅
- [x] GGUF parser with memory-mapped tensor access (zero-copy via `MemoryMappedFile`)

> from: ### Phase 1: CPU Reference Implementation ✅
- [x] Tokenizer integration via `Microsoft.ML.Tokenizers` (BPE with special token handling)

> from: ### Phase 1: CPU Reference Implementation ✅
- [x] Q4_K and Q6_K scalar dequantization matching ggml-quants.c

> from: ### Phase 1: CPU Reference Implementation ✅
- [x] Full LLaMA-family forward pass: GQA attention, SwiGLU FFN, interleaved RoPE

> from: ### Phase 1: CPU Reference Implementation ✅
- [x] FP32 KV cache with per-layer buffers

> from: ### Phase 1: CPU Reference Implementation ✅
- [x] Temperature / top-k / top-p / min-p sampling

> from: ### Phase 1: CPU Reference Implementation ✅
- [x] AVX2 SIMD: fused dequant-matvec with multi-accumulator FMA chains

> from: ### Phase 1: CPU Reference Implementation ✅
- [x] Multi-threaded MatVec via `Parallel.For` (24 threads)

> from: ### Phase 1: CPU Reference Implementation ✅
- [x] Batched prefill with layer-by-layer cache-hot weight reuse

> from: ### Phase 1: CPU Reference Implementation ✅
- [x] OpenBLAS GEMM integration for large-batch prefill

> from: ### Phase 1: CPU Reference Implementation ✅
- [x] CLI chat REPL (`stingray`) with llama.cpp-compatible flags

> from: ### Phase 1: CPU Reference Implementation ✅
- [x] Reference test: output matches llama.cpp greedy decode token-for-token

> from: ### Phase 1: CPU Reference Implementation ✅
- [x] BenchmarkDotNet harness measuring decode and prefill throughput

> from: ### Phase 2: Vulkan GPU Acceleration ✅
- [x] Vortice.Vulkan device initialization and dedicated compute queue

> from: ### Phase 2: Vulkan GPU Acceleration ✅
- [x] GPU buffer management: device-local VRAM, staging transfers, cached download buffers

> from: ### Phase 2: Vulkan GPU Acceleration ✅
- [x] Compute shader pipeline: GLSL→SPIR-V via glslc, descriptor sets, push constants

> from: ### Phase 2: Vulkan GPU Acceleration ✅
- [x] Compute shaders: MatVecQ4K, MatVecQ6K, MatVecF32, RMSNorm, RoPE, softmax, SiLU, attention, embedding lookup, KV append

> from: ### Phase 2: Vulkan GPU Acceleration ✅
- [x] Batched command buffer: all ~240 dispatches per token in one submission with memory barriers

> from: ### Phase 2: Vulkan GPU Acceleration ✅
- [x] FP32 KV cache in VRAM (per-layer, device-local)

> from: ### Phase 2: Vulkan GPU Acceleration ✅
- [x] GPU attention with shared-memory parallel reduction (no atomics, no PCIe round-trips)

> from: ### Phase 2: Vulkan GPU Acceleration ✅
- [x] All weights resident in VRAM (Q4_K raw, Q6_K raw, F32 norms)

> from: ### Phase 2: Vulkan GPU Acceleration ✅
- [x] Zero managed allocation per decode token

> from: ### Phase 2: Vulkan GPU Acceleration ✅
- [x] GPU forward pass validated against CPU token-for-token

> from: ### Phase 2: Vulkan GPU Acceleration ✅
- [x] NativeAOT-ready CLI with IlcOptimizationPreference=Speed, IlcInstructionSet=native

> from: ### Phase 2b: Qwen3 8B Scaling
- [x] Architecture-agnostic hyperparameter extraction (arch-prefixed GGUF metadata keys)

> from: ### Phase 2b: Qwen3 8B Scaling
- [x] Attention bias support: optional Q/K/V/O bias tensors (detected via `_opentailllm.has_attn_bias` synthetic metadata)

> from: ### Phase 2b: Qwen3 8B Scaling
- [x] CPU forward pass: bias addition after Q/K/V/O MatVec projections (both single-token and batched prefill)

> from: ### Phase 2b: Qwen3 8B Scaling
- [x] GPU forward pass: bias tensors uploaded to VRAM, applied via AddInPlace shader dispatches

> from: ### Phase 2b: Qwen3 8B Scaling
- [x] Dynamic RoPE theta from GGUF metadata (supports Qwen3's 1M+ base frequency)

> from: ### Phase 2b: Qwen3 8B Scaling
- [x] All shaders dimension-agnostic via push constants (no hardcoded sizes)

> from: ### Phase 2b: Qwen3 8B Scaling
- [x] KV cache scales to any head count/dim configuration

> from: ### Phase 2b: Qwen3 8B Scaling
- [x] Per-head QK-norm: RMSNorm on Q/K per head (CPU + HeadNorm GPU shader), detected via `_opentailllm.has_qk_norm`

> from: ### Phase 2b: Qwen3 8B Scaling
- [x] Vocab size inference from `tokenizer.ggml.tokens` array when `{arch}.vocab_size` metadata missing

> from: ### Phase 2b: Qwen3 8B Scaling
- [x] Quantized embedding table in VRAM: EmbedLookupQ4K shader dequantizes per-token (saves ~1.7 GB vs F32)

> from: ### Phase 2b: Qwen3 8B Scaling
- [x] Auto VRAM context sizing: estimates weight/scratch footprint, gives remaining to KV cache

> from: ### Phase 2b: Qwen3 8B Scaling
- [x] CLI `-c`/`--ctx-size` flag (matches llama.cpp) with model info in output

> from: ### Phase 2b: Qwen3 8B Scaling
- [x] End-to-end validation: Qwen3 8B Q4_K_M produces coherent output on CPU and GPU

> from: ### Phase 2b: Qwen3 8B Scaling
- [x] Performance benchmark: decode t/s on RTX 4070 Ti

> from: ### Phase 3: TurboQuant KV Cache Compression ✅
- [x] Lloyd-Max codebook generation (offline tool `tools/CodebookGen`, 3-bit and 4-bit codebooks for d=128 and d=256)

> from: ### Phase 3: TurboQuant KV Cache Compression ✅
- [x] Walsh-Hadamard transform (scalar reference + AVX2 SIMD butterfly, `WalshHadamard.cs`)

> from: ### Phase 3: TurboQuant KV Cache Compression ✅
- [x] 3-bit quantize / dequant with bit packing (`BitPacking.cs`, `TurboQuantOps.cs`)

> from: ### Phase 3: TurboQuant KV Cache Compression ✅
- [x] Fused dequant-dot-product (CPU scalar + AVX2 + Vulkan `TqKvAppend` / `TqAttention` shaders)

> from: ### Phase 3: TurboQuant KV Cache Compression ✅
- [x] Adaptive precision: FP32 recent window (256 tokens) + TQ compressed history (`TurboQuantKvCache.cs`)

> from: ### Phase 3: TurboQuant KV Cache Compression ✅
- [x] K/V magnitude profiler for per-model bit budget selection (`MagnitudeProfiler.cs`)

> from: ### Phase 3: TurboQuant KV Cache Compression ✅
- [x] MSE validation tests (25 tests covering round-trip, WHT, bit packing, dequant-dot accuracy)

> from: ### Phase 3: TurboQuant KV Cache Compression ✅
- [x] GPU TQ end-to-end: `GpuForwardPass` TQ mode with compressed VRAM KV cache, `TqRotateQuery` + `TqAttention` shaders

> from: ### Phase 3: TurboQuant KV Cache Compression ✅
- [x] Hybrid TQ end-to-end: `HybridForwardPass` TQ mode for GPU-resident layers plus CPU-side TQ cache for offloaded layers

> from: ### Phase 3: TurboQuant KV Cache Compression ✅
- [x] Needle-in-a-haystack test at 1K / 2K / 4K / 8K: TurboQuantNeedleTests verifies the compressed needle key has the highest raw attention score (top 1%) vs random background keys

> from: ### Phase 4a: Hybrid GPU/CPU Offloading ✅
- [x] HardwareProfile: auto-detect VRAM, RAM, CPU cores, PCIe bandwidth, AVX-512

> from: ### Phase 4a: Hybrid GPU/CPU Offloading ✅
- [x] TierPlanner: greedy layer placement (embedding+output always GPU, layers packed until VRAM full)

> from: ### Phase 4a: Hybrid GPU/CPU Offloading ✅
- [x] Pinned host memory (`VkMemoryPropertyFlags.HostVisible | HostCoherent`, BAR fallback)

> from: ### Phase 4a: Hybrid GPU/CPU Offloading ✅
- [x] HybridForwardPass: GPU layers + CPU layers + hidden state transfer via pinned buffer

> from: ### Phase 4a: Hybrid GPU/CPU Offloading ✅
- [x] CLI `-g N` three-way dispatch: CPU-only / all-GPU / hybrid, `-g -1` auto-detect

> from: ### Phase 4a: Hybrid GPU/CPU Offloading ✅
- [x] Q5_K dequantization support (scalar fallback for Llama 3.1 70B mixed quantization)

> from: ### Phase 4a: Hybrid GPU/CPU Offloading ✅
- [x] BpeTokenizer fallback for Llama 3.1 tokenizer compatibility

> from: ### Performance Optimization Pass ✅
- [x] Q5_K fused AVX2 dequant-matvec (2.7x on 70B, matching llama.cpp)

> from: ### Performance Optimization Pass ✅
- [x] AVX-512 DotQ4K and DotQ5K kernels (+5% on 8B CPU via reduced loop overhead)

> from: ### Performance Optimization Pass ✅
- [x] Physical-core-only Parallel.For (no SMT for SIMD workloads)

> from: ### Performance Optimization Pass ✅
- [x] Async GPU submit API (`EndRecordAndSubmitAsync` + `WaitForGpu`)

> from: ### Performance Optimization Pass ✅
- [x] Mmap weight prefaulting in HybridForwardPass

> from: ### Performance Optimization Pass ✅
- [x] Multi-row workgroups: 8 rows per workgroup, 32 threads per row

> from: ### Performance Optimization Pass ✅
- [x] subgroupAdd for zero-barrier reduction (eliminates shared memory reduction tree)

> from: ### Performance Optimization Pass ✅
- [x] Register-based scale/min precomputation (3 uint32 reads → 8 scale pairs)

> from: ### Performance Optimization Pass ✅
- [x] unpackHalf2x16 for d/dmin (single-instruction FP16 decode)

> from: ### Performance Optimization Pass ✅
- [x] Q6_K shader modernization (8-row + subgroupAdd, currently uses old 256-thread reduction)

> from: ### Performance Optimization Pass ✅
- [x] Reduce barrier count: use buffer-specific VkBufferMemoryBarrier instead of global barriers (~540 per token)

> from: ### Performance Optimization Pass ✅
- [x] Fold logits download into main command buffer (eliminate second submit-wait)

> from: ### Performance Optimization Pass ✅
- [x] Compute-shader buffer copy (replace vkCmdCopyBuffer transfer to stay in compute pipeline stage)

> from: ### Performance Optimization Pass ✅
- [x] Fix attention shader for seq_len > 256: replaced fast/tiled split with stored-scores path (shared float scores[4096]) matching TqAttention; triple-pass retained only for seq_len > 4096

> from: ### Performance Optimization Pass ✅
- [x] Fused gate+up MatVec (single Parallel.For for both FFN projections, halves thread dispatch count)

> from: ### Performance Optimization Pass ✅
- [x] Parallel attention heads (per-thread score buffers, Parallel.For over heads)

> from: ### Performance Optimization Pass ✅
- [x] KV cache head-major transpose (contiguous access per KV head across positions — token-major is already optimal)

> from: ### Performance Optimization Pass ✅
- [x] Precomputed RoPE cos/sin tables (avoid MathF.Pow/Cos/Sin per head per layer)

> from: ### Phase 5: MoE Inference — Llama 4 Scout ✅
- [x] Detect MoE architecture from GGUF metadata (`{arch}.expert_count`, `{arch}.expert_used_count`)

> from: ### Phase 5: MoE Inference — Llama 4 Scout ✅
- [x] Parse expert tensor naming: `blk.{i}.ffn_gate_exps.weight`, `blk.{i}.ffn_up_exps.weight`, `blk.{i}.ffn_down_exps.weight`, `blk.{i}.ffn_gate_inp.weight` (router)

> from: ### Phase 5: MoE Inference — Llama 4 Scout ✅
- [x] Extend `ModelHyperparams` with `NumExperts`, `NumActiveExperts`, `IsMoE`

> from: ### Phase 5: MoE Inference — Llama 4 Scout ✅
- [x] MoE router: compute `ffn_gate_inp` MatVec → top-k softmax → expert indices + weights

> from: ### Phase 5: MoE Inference — Llama 4 Scout ✅
- [x] Sparse expert FFN: only execute selected experts, weighted-sum outputs

> from: ### Phase 5: MoE Inference — Llama 4 Scout ✅
- [x] CPU MoE reference path for correctness and fallback

> from: ### Phase 5: MoE Inference — Llama 4 Scout ✅
- [x] True GPU MoE FFN execution for GPU-resident layers

> from: ### Phase 5: MoE Inference — Llama 4 Scout ✅
- [x] True hybrid MoE execution for GPU-resident layers with CPU execution for offloaded layers

> from: ### Phase 5: MoE Inference — Llama 4 Scout ✅
- [x] MoE-aware `TierPlanner` placement using actual uploaded weight size rather than raw GGUF bytes

> from: ### Phase 5: MoE Inference — Llama 4 Scout ✅
- [x] Smart defaults for Scout-class models: cap auto context to practical VRAM defaults and keep giant fixed tensors on CPU when that improves placement

> from: ### Phase 5: MoE Inference — Llama 4 Scout ✅
- [x] Scout decode benchmarks: CPU, CPU+TQ3, auto hybrid, auto hybrid+TQ3

> from: ### Phase 5: MoE Inference — Llama 4 Scout ✅
- [x] Scout microbenchmarks: router/top-k and MoE FFN layer

> from: ### Phase 5: MoE Inference — Llama 4 Scout ✅
- [x] Expert slot cache in VRAM: N slots (sized to fit available VRAM after attention weights + KV cache)

> from: ### Phase 5: MoE Inference — Llama 4 Scout ✅
- [x] SLRU eviction policy: probationary → protected segments, exploiting routing skew (~20% experts handle ~80% tokens)

> from: ### Phase 5: MoE Inference — Llama 4 Scout ✅
- [x] `Channel<T>` async prefetch pipeline: router reveals needed experts → enqueue DMA from pinned RAM → VRAM

> from: ### Phase 5: MoE Inference — Llama 4 Scout ✅
- [x] Router-driven predictive prefetching (1-token lookahead from router logits)

> from: ### Phase 5: MoE Inference — Llama 4 Scout ✅
- [x] CPU fallback for expert cache misses: compute on CPU via mmap while GPU handles cached experts

> from: ### Phase 5: MoE Inference — Llama 4 Scout ✅
- [x] Expert access frequency profiler: per-layer hit rate metrics, hot expert identification

> from: #### Phase 7a: Working API Server (✅ Complete)
- [x] ASP.NET Core Minimal API with NativeAOT compatibility

> from: #### Phase 7a: Working API Server (✅ Complete)
- [x] Anthropic Messages API: `POST /v1/messages` (streaming SSE + non-streaming)

> from: #### Phase 7a: Working API Server (✅ Complete)
- [x] OpenAI Chat Completions API: `POST /v1/chat/completions` (streaming SSE + non-streaming)

> from: #### Phase 7a: Working API Server (✅ Complete)
- [x] Model listing: `GET /v1/models`

> from: #### Phase 7a: Working API Server (✅ Complete)
- [x] Health endpoint: `GET /health` with model ID and uptime

> from: #### Phase 7a: Working API Server (✅ Complete)
- [x] Prometheus-format metrics: `GET /metrics` (request counts, tokens generated, uptime)

> from: #### Phase 7a: Working API Server (✅ Complete)
- [x] Source-generated JSON serialization throughout (NativeAOT-compatible)

> from: #### Phase 7a: Working API Server (✅ Complete)
- [x] Configuration via `STINGRAY_MODEL` / `STINGRAY_N_GPU_LAYERS` env vars and `appsettings.json`

> from: #### Phase 7a: Working API Server (✅ Complete)
- [x] `IInferenceEngine` interface for testability; `WebApplicationFactory` integration tests (8 tests)

> from: #### Phase 7a: Working API Server (✅ Complete)
- [x] Chat templates: ChatML (Qwen2/SmolLM2), Llama 3.x, Llama 4

> from: #### Phase 7a: Working API Server (✅ Complete)
- [x] Serialized single-request via `SemaphoreSlim`; concurrent callers block in arrival order

> from: #### Phase 7b: PagedKvCache + Batched Prefill + Prefix Caching (✅ Complete)
- [x] Lazy page allocation: pages (16 positions each) allocated on first write, not upfront

> from: #### Phase 7b: PagedKvCache + Batched Prefill + Prefix Caching (✅ Complete)
- [x] Per-layer pool: `_pool[layer][slot]` — each slot is `PageSize × kvDim × 2 floats` (keys + values)

> from: #### Phase 7b: PagedKvCache + Batched Prefill + Prefix Caching (✅ Complete)
- [x] Free-list warm pool: returned pages are reused across requests without `NativeMemory.Free`

> from: #### Phase 7b: PagedKvCache + Batched Prefill + Prefix Caching (✅ Complete)
- [x] Soft `TruncateTo(n)`: moves length pointer without freeing pages — enables zero-copy prefix reuse

> from: #### Phase 7b: PagedKvCache + Batched Prefill + Prefix Caching (✅ Complete)
- [x] Full `Reset()`: returns all slots to warm pool — use at start of unrelated request

> from: #### Phase 7b: PagedKvCache + Batched Prefill + Prefix Caching (✅ Complete)
- [x] Eliminates GBs-upfront pre-allocation (Llama 4 Scout's 10M context would have been 40TB+)

> from: #### Phase 7b: PagedKvCache + Batched Prefill + Prefix Caching (✅ Complete)
- [x] Replaces `KvCache` in `ForwardPass`; `KvCache` retained for `HybridForwardPass` compatibility

> from: #### Phase 7b: PagedKvCache + Batched Prefill + Prefix Caching (✅ Complete)
- [x] Added to `IForwardPass` interface — all implementations provide it

> from: #### Phase 7b: PagedKvCache + Batched Prefill + Prefix Caching (✅ Complete)
- [x] `ForwardPass`: existing per-layer batched GEMM path; updated to accept `startPos` for prefix reuse

> from: #### Phase 7b: PagedKvCache + Batched Prefill + Prefix Caching (✅ Complete)
- [x] `GpuForwardPass`, `HybridForwardPass`: sequential `Forward()` fallback

> from: #### Phase 7b: PagedKvCache + Batched Prefill + Prefix Caching (✅ Complete)
- [x] `InferenceEngine` now uses `Prefill()` instead of a loop of `Forward()` calls

> from: #### Phase 7b: PagedKvCache + Batched Prefill + Prefix Caching (✅ Complete)
- [x] `FindCacheablePrefix()`: compares new prompt tokens against previous request, finds longest page-aligned common prefix

> from: #### Phase 7b: PagedKvCache + Batched Prefill + Prefix Caching (✅ Complete)
- [x] On hit: `TruncateTo(prefixLen)` keeps cached K/V, then `Prefill(suffix, prefixLen)` fills only the new portion

> from: #### Phase 7b: PagedKvCache + Batched Prefill + Prefix Caching (✅ Complete)
- [x] On miss: `ResetCache()` (full reset) + `Prefill(allTokens)`

> from: #### Phase 7b: PagedKvCache + Batched Prefill + Prefix Caching (✅ Complete)
- [x] Eliminates repeated system-prompt prefill cost for multi-turn chat API workloads

> from: #### Phase 7c: Continuous Batching (✅ Complete)
- [x] Batched decode step for N sequences simultaneously — one token per sequence

> from: #### Phase 7c: Continuous Batching (✅ Complete)
- [x] Shared weight reads (Q/K/V/FFN GEMMs) amortized across N sequences per decode step

> from: #### Phase 7c: Continuous Batching (✅ Complete)
- [x] Per-sequence `PagedKvCache` — each sequence has its own independent KV cache

> from: #### Phase 7c: Continuous Batching (✅ Complete)
- [x] Per-sequence: individual RoPE, cache append at `positions[n]`, and causal attention against `caches[n]`

> from: #### Phase 7c: Continuous Batching (✅ Complete)
- [x] Not supported for MoE models or with TurboQuant (throws `NotSupportedException`)

> from: #### Phase 7c: Continuous Batching (✅ Complete)
- [x] `PrefillWithCache(tokens, cache, startPos)` — prefills a given sequence's cache (used during admission)

> from: #### Phase 7c: Continuous Batching (✅ Complete)
- [x] `ForwardCore(token, pos, cache)` — single-token forward into explicit cache (supports MoE)

> from: #### Phase 7c: Continuous Batching (✅ Complete)
- [x] `CreateCache()` — factory method returns a compatible `PagedKvCache` for the model's dimensions

> from: #### Phase 7c: Continuous Batching (✅ Complete)
- [x] Implements `IInferenceEngine` — drop-in replacement for `InferenceEngine`

> from: #### Phase 7c: Continuous Batching (✅ Complete)
- [x] Unbounded request channel — callers enqueue via `GenerateAsync`, stream results back via `IAsyncEnumerable<string>`

> from: #### Phase 7c: Continuous Batching (✅ Complete)
- [x] Background batcher loop:
  1. Admits pending requests up to `_maxBatchSize` (prefills each individually with own `PagedKvCache`)
  2. Batched decode: `BatchForwardMulti` processes all active sequences in one forward pass per step
  3. Samples next token per sequence; writes to that sequence's output channel
  4. Retires sequences that hit EOS, `MaxNewTokens`, or cancellation; disposes cache

> from: #### Phase 7c: Continuous Batching (✅ Complete)
- [x] Enabled via `STINGRAY_MAX_BATCH` environment variable (server `Program.cs`); `> 1` activates continuous batching

> from: #### Phase 8: API Completeness — Metrics, Observability, LogitBias (✅ Complete)
- [x] `int QueueDepth` — pending requests waiting to start (0 for serialized engine, real queue depth for batching engine)

> from: #### Phase 8: API Completeness — Metrics, Observability, LogitBias (✅ Complete)
- [x] `int ActiveRequests` — requests currently generating tokens (0 or 1 for serialized, batch count for batching engine)

> from: #### Phase 8: API Completeness — Metrics, Observability, LogitBias (✅ Complete)
- [x] Interlocked counters in both engines; zero-overhead for callers

> from: #### Phase 8: API Completeness — Metrics, Observability, LogitBias (✅ Complete)
- [x] **Fixed bug**: `RecordRequest()` / `RecordTokens()` were never called — now wired into both OpenAI and Anthropic handlers

> from: #### Phase 8: API Completeness — Metrics, Observability, LogitBias (✅ Complete)
- [x] Token count uses actual decoded token chunks (not character count)

> from: #### Phase 8: API Completeness — Metrics, Observability, LogitBias (✅ Complete)
- [x] `opentail-llm_tokens_per_second` gauge — lifetime-average tokens/second since server start

> from: #### Phase 8: API Completeness — Metrics, Observability, LogitBias (✅ Complete)
- [x] `opentail-llm_queue_depth` gauge — `engine.QueueDepth` (live pending requests)

> from: #### Phase 8: API Completeness — Metrics, Observability, LogitBias (✅ Complete)
- [x] `opentail-llm_active_requests` gauge — `engine.ActiveRequests` (live active sequences)

> from: #### Phase 8: API Completeness — Metrics, Observability, LogitBias (✅ Complete)
- [x] `/metrics` handler now injects `IInferenceEngine` for live observable state

> from: #### Phase 8: API Completeness — Metrics, Observability, LogitBias (✅ Complete)
- [x] `IReadOnlyDictionary<int, float>? LogitBias` added to `SamplingParams`

> from: #### Phase 8: API Completeness — Metrics, Observability, LogitBias (✅ Complete)
- [x] Applied before temperature scaling in `Sampler.Sample()` (additive in logit space, range [-100, 100])

> from: #### Phase 8: API Completeness — Metrics, Observability, LogitBias (✅ Complete)
- [x] Out-of-range token IDs silently skipped

> from: #### Phase 8: API Completeness — Metrics, Observability, LogitBias (✅ Complete)
- [x] `logit_bias` field added to `ChatCompletionRequest` (OpenAI wire format: `{"tokenId": bias}` with string keys)

> from: #### Phase 8: API Completeness — Metrics, Observability, LogitBias (✅ Complete)
- [x] String → int key conversion in endpoint handler; `Dictionary<string, float>` registered in `AppJsonContext`

> from: ### Phase 8: CUDA Backend ✅
- [x] `CuBlasInterop.cs` — P/Invoke bindings for `cublasSgemm`, `cublasGemmEx`, `cudaMalloc`, `cudaFree`, `cudaMemcpy`, `cudaDeviceSynchronize`

> from: ### Phase 8: CUDA Backend ✅
- [x] `CudaBackend.cs` — `IComputeBackend` implementation: `Upload` (fp32 + bf16), `UploadBf16`, `Allocate`, `Free`, `Download`, `DownloadBf16`, `Sgemm`, `Synchronize`

> from: ### Phase 8: CUDA Backend ✅
- [x] `Sgemm(C, A, B, m, k, n)` — `C[m,n] = A[m,k] @ B[n,k]ᵀ` via cuBLAS COLUMN_MAJOR row-swap trick

> from: ### Phase 8: CUDA Backend ✅
- [x] bf16 SGEMM: `cublasGemmEx` with `CUDA_R_16BF` compute for Tensor Core acceleration on sm_80+

> from: ### Phase 8: CUDA Backend ✅
- [x] fp32 SGEMM: `cublasSgemm` with TF32 compute on sm_80+ (automatic, no code change)

> from: ### Phase 8: CUDA Backend ✅
- [x] `CudaBackend.IsAvailable()` — runtime probe via `cudaGetDeviceCount`

> from: ### Phase 8: CUDA Backend ✅
- [x] Auto-select in `ImageCommand.cs`: CUDA → Vulkan → CPU

