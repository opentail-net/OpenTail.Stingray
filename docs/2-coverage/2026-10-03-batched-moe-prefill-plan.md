# Plan: batched MoE prefill for the Vulkan hybrid / GPU paths

Written 2026-10-03. Companion to [2026-10-03-streamed-residency-plan.md](2026-10-03-streamed-residency-plan.md), which lists this as its largest remaining item. **Status: plan only, nothing implemented.**

**Revision 2026-10-03 (after an external review; each point was checked against the code before adopting).** Corrected: the hybrid's CPU KV is `KvCache` (FP32, `[maxSeqLen, kvDim]` per layer), not the `PagedKvCache` that `ForwardPass` uses; the hybrid's GPU KV is FP32 (`gpu.Allocate` default dtype), the fp16 KV I cited belongs to `GpuForwardPass`, so Phase 1 has no precision conversion; Phase 1 is fresh sequences only (`startPos == 0`); `PagedKvCache` can narrow itself to BF16 (explicitly via `STINGRAY_KV_DTYPE` / `STINGRAY_KV_STORE`, or automatically at 1,024 tokens in `auto` mode), so the CPU prefill pass must be pinned to F32 KV or the handoff would silently put BF16-rounded K/V into the hybrid.

## Context

`HybridForwardPass.Prefill` (`src/OpenTail.Stingray.Engine/HybridForwardPass.cs:691`) is a loop of single-token `Forward` calls, and `GpuForwardPass` excludes every MoE model from its batched trunk (`ComputeCanBatchedTrunk`, `GpuForwardPass.cs:1270`: "MoE (mid-trunk router submit)"). Measured on this machine, OLMoE-1B-7B Q4_K_M, 631-token prompt, Release build:

| Path | Prefill tok/s |
|---|---:|
| CPU only (`-g 0`, batched MoE) | **141.6** |
| Vulkan hybrid (`-g 8`) | 25.1 |
| Vulkan full GPU (`-g -1`) | 26.6 |

Dense control (SmolLM2-1.7B, 35-token prompt): CPU 104.7, full GPU 78.7. So on this integrated GPU, even the existing dense batched GPU prefill loses to the CPU. Any plan that assumes "GPU prefill is faster" is untested here and, for this machine, probably wrong. Prefill is the largest remaining cost of the hybrid path (decode was fixed in commit `4556fee1`: 2.1-2.7 -> 18.7 tok/s on a 4-slot cache).

Goal: hybrid and full-GPU prefill on MoE models is never slower than the CPU-only path on the same machine, is correct (logits equal the sequential path), and is faster than the CPU where the hardware allows (discrete GPU). Keep the project's rules: no subagents, measure before keeping a change, exact-parity tests, state iGPU results as iGPU results (CLAUDE.md rules 6, 7, 12, 13).

## What the references do (examined 2026-10-03)

### Our own code (reuse first)
- **CPU batched MoE prefill** (`ForwardPass.Moe.cs:488 MoeFfnBatched`): (1) route every token on the exact per-token F32 path, (2) bucket (token, slot) pairs by expert in CSR order, (3) per used expert gather its tokens and run gate/up, SiLU*mul, down as batched matmuls via `MoeBatchedExperts.Run` (`MoeBatchedExperts.cs`, experts in parallel with per-worker buffers), (4) **reduce the unweighted down partials per token in top-k slot order** (reducing in expert order moved OLMoE logits by up to 0.20; this ordering is the parity contract), (5) shared expert as an ordinary batched FFN. `MoeBatchedExperts.Run` is already shared by `ForwardPass` and `HybridGdnForwardPass`; it is plain C# over host pointers and can be called from any pass.
- **CUDA hybrid batched trunk** (`CudaHybridForwardPass.PrefillBatchedTrunk`, line 1863; MoE stage "#410", line ~1500): GPU attention trunk batched over N tokens; for MoE: one D2H of every token's post-FFN-norm row, router + top-k on the host, CSR-grouped expert dots on the CPU, one H2D into the residual stream, one batched add. FFN math kept bit-identical to the per-token path. This is the pattern to port to Vulkan.
- **CUDA GDN "op-offload"** (`CudaHybridGdnForwardPass.cs:739-800`, measured in its comments): transiently uploads each layer's host-resident expert tensor to the GPU and runs gather -> GEMM -> SiLU*mul -> GEMM on the GPU. +28-67% prefill from about 120 tokens; below that it loses (it uploads the whole layer regardless of N), hence a gate `STINGRAY_MOE_GPU_PREFILL_MIN_TOKENS` (default 64). Lessons recorded there: double-buffer the weight upload against compute; register-in-place pinning instead of a pinned copy (a copy evicted the page cache that decode relies on); a GPU router GEMM with host softmax/top-k; not bit-exact (argmax-stable).
- **Vulkan today**: batched trunk exists for dense models only (`PrefillBatchedChunk`, kernels `RmsNormBatched`, `MatMulBatched`, `RoPEBatched`, `KvAppendBatched`, `AttentionBatched`, ...). `MatMulBatched` is "Path 1": weight-stationary matvec with an N-accumulator register limit, **at most 16 tokens per dispatch** (`VulkanMatMulPathConfig.MaxTokensPerDispatch`); "Path 2" (shared-memory tiled quantised GEMM) is a reserved seam that nothing implements. Q4_K and Q6_K only. There is **no** top-k, gather/scatter, expert-count, or expert-offset batched kernel (the only expert-offset kernel is the MXFP4 matvec `row_offset` for gpt-oss).
- **Hybrid structure**: `HybridForwardPass` keeps per-token GPU layers (`GpuLayer`) and per-token CPU layers (`CpuLayer`, its own scratch); KV for GPU layers is on the GPU, for CPU layers in `_cpuKvCache`.

### llama.cpp (`examples/llama.cpp/llama.cpp`)
- **When to offload**: scheduler op-offload sends an op whose weights are in host memory to the GPU if the batch is large enough: `n_tokens >= 32` for `MUL_MAT_ID` (`GGML_OP_OFFLOAD_MIN_BATCH`, default 32; `ggml-backend.cpp:1110`, `ggml-vulkan.cpp:16129`, `ggml-cuda.cu:5734`). Vulkan tags devices as integrated (`is_integrated_gpu`) but does not change the threshold for it.
- **Vulkan MoE prefill is fully on the GPU, no host round trip**: top-k comes from graph ops; `count_experts.comp` builds per-expert counts, start offsets and a packed row-id list grouped by expert (`(i01<<16)|i00`, "hoisted" up to 1024 experts); then one tiled GEMM dispatch (`mul_mm.comp` compiled with `MUL_MAT_ID`) runs with the expert index as the grid's third dimension: each workgroup loads its tile's row ids into shared memory, gathers activation rows through them, multiplies against the expert's weight slice, and scatters to the destination rows. Workgroups for an expert with no tokens exit immediately.
- **Tile choice from tokens per expert**: `n_per_expert = ceil(n_ids * n_tokens / n_experts)` selects the s/m/l tile and the "aligned" variant; quantised weights have `mmq` variants (Q8_1 activations, integer dot) and cooperative-matrix variants. Decode (few tokens) uses a different kernel, `mul_mat_vec_id`.
- **CPU side**: `mul_mat_id` groups rows by expert the same way (counts, then per-expert mat-mat).

### vLLM (`examples/vllm`)
- `moe_align_block_size`: sorts token indices by expert and **pads each expert's list to a multiple of the tile size**, returning `sorted_token_ids`, per-block `expert_ids`, `num_tokens_post_padded`; one fused GEMM kernel walks the blocks (each block belongs to one expert). Padding trades wasted lanes for a trivial kernel; llama.cpp's dynamic row ids trade a counting pass for no padding.
- Tile size follows the token count (`get_default_config`: `BLOCK_SIZE_M` 16 for M<=32, 32 for M<=96, 64 for M<=512, else 128; group-M only helps when tokens per expert > 128). Top-k weighting and the per-token reduction are a separate step.

### TensorSharp (`examples/TensorSharp/TensorSharp`, docs only; source not read)
- Decode runs each host layer's routed experts on the CPU from the memory map (thread team woken once per layer). **Prefill of 128+ tokens (`TS_HOST_MOE_DEVICE_MIN_BATCH`) streams each host layer's used experts to the GPU per chunk**, after faulting their pages in on 16 threads first (their 1,818-token prefill 37.2-38 s -> 15.7-16.6 s). The engine plans the split from the accelerator working set and the RAM left as page cache; wiring more layers than fit makes both decode and prefill worse. Published: 4.5-10.9x prefill over llama.cpp's offloaded configurations (on a 2-GPU server, not comparable to this machine).

### SharpMind
Training-oriented; its `MoEFfnLayer` was not found to contain a batched-prefill design. Nothing to take for this plan beyond the testing lessons already used in the residency plan.

### Not examined
TensorSharp's source (only its docs), `ggml-cpu` `mul_mat_id` source, `onnxruntime` MoE op, `XNNPACK`. None is needed for the first two phases.

## Key insight: three hardware regimes need different answers

| Regime | Example | Best prefill for MoE layers |
|---|---|---|
| A. Shared-memory GPU | this machine (Ryzen 5700G + integrated Radeon) | CPU batched MoE (141 tok/s). GPU compute does not beat it; measured dense GPU 78.7 vs CPU 104.7 |
| B. Discrete GPU, experts resident in VRAM | desktop GPU with enough VRAM | GPU grouped GEMM (llama.cpp style); needs new Vulkan kernels |
| C. Discrete GPU, experts in host RAM (oversized) | the larger-than-VRAM goal | stream used experts per chunk and run on the GPU if PCIe makes it faster than the CPU (TensorSharp, our CUDA op-offload, llama.cpp op-offload), else CPU |

A single dispatcher chooses per chunk (Phase 5). Regime A is the only one I can measure here; regimes B and C can be tested for correctness here but not for speed (rule 13).

## Design (phases, in order; each ends with a measurement and a test before the next starts)

### Phase 0: instrument and baseline (small, do first)
- Add `STINGRAY_PREFILL_TIMING=1` to `HybridForwardPass` (and `GpuForwardPass`) printing, per prefill: time in GPU attention, router wait, CPU MoE, GPU MoE, CPU-layer attention, KV traffic. Same style as `STINGRAY_MOE_TIMING`.
- Record the baseline matrix in the doc: OLMoE, Qwen3-Coder-30B; prompt lengths 20 / 128 / 631 / 2048; `-g 0`, `-g 8`, `-g -1`; 3 runs each. This fixes the crossover numbers the dispatcher needs (rule 7: measure, do not assume).
- Baseline matrix, staged: OLMoE first (prompts 20 / 128 / 631 / 2048; `-g 0`, `-g 8`, `-g -1`; 3 runs each); Qwen3-Coder-30B only once Phase 1 mechanics work. Do not block Phase 1 on the full matrix.
- Read-only checks (answers recorded in this doc before coding): (a) the layout and conversion boundary between `ForwardPass.Cache` (`PagedKvCache`: pages, F32 / BF16 store, auto-narrowing at 1,024 tokens, layer indexing, physical vs logical positions) and `HybridForwardPass._cpuKvCache` (`KvCache`: FP32, `[maxSeqLen, kvDim]` per CPU layer, CPU layer index = layer - nGpuLayers) and the hybrid GPU KV tensors (FP32, `[maxSeqLen * kvDim]` per GPU layer); (b) whether a CPU `ForwardPass` can be built over the same `GgufModel` without duplicating weights (they are memory-mapped; construction pre-faults them) and what its construction costs.

### Phase 1: CPU-prefill fast path with exact KV handoff (recommended first implementation; no new kernels)
Name it for what it is: it does **not** make Vulkan MoE prefill batched; it makes a fresh prompt's prefill run on the already-proven CPU batched MoE path and hands the result to the hybrid. "Phase 1 done" must not be read as "Vulkan has batched MoE".

**Scope and hard rules**
- Fresh sequences only: `startPos == 0`. For `startPos > 0` (a later turn) fall back to the existing sequential hybrid prefill. A newly built CPU pass does not hold the hybrid's earlier K/V, so it cannot compute attention for new tokens correctly; supporting it needs a separate "Phase 1b" (import or shadow the prefix KV into the CPU cache), not done here.
- **No semantic change**: no new KV precision, no new attention/RoPE/router/FFN arithmetic, no new GPU kernels. It is a scheduling and cache-transfer optimisation only.
- The CPU prefill pass **must be constructed with F32 KV, explicitly**: BF16 store off, BF16 rounding off, auto-narrowing off. `PagedKvCache` can otherwise narrow itself to BF16 (`STINGRAY_KV_DTYPE=bf16`, `STINGRAY_KV_STORE=bf16`, or `auto` at 1,024 tokens). If F32 KV cannot be guaranteed for the pass (for example the environment forces BF16 and the constructor cannot override it), refuse the fast path and use the sequential hybrid prefill. Otherwise long prompts would put BF16-rounded K/V into an F32 hybrid and change decode numerics versus the sequential path.
- Disabled when: TurboQuant or SnapKV is active, the model/backend is already refused by the hybrid, the KV layout or per-layer head dimensions cannot be copied exactly, or any precision conversion would be needed.
- **Same semantics, not merely the same file.** Both passes loading a GGUF does not make the handoff safe; there have been bugs where CPU and GPU supported a model but interpreted one architectural feature differently. Admit the fast path per model family only where the CPU pass and the hybrid are known to share token and KV semantics: per-layer head dimensions, sliding-window attention, RoPE variants and scaling, shared KV, post-attention / post-FFN transforms, residual and embedding/output scaling, multimodal and deepstack inputs. Reuse the existing admission and capability gates (`ModelCompatibility`, `GpuForwardPass.PartialOffloadUnsupportedReason`, the hybrid constructor's refusals) rather than re-deriving them, and require evidence: a family is admitted only if a hybrid-versus-CPU logit parity test exists for it (today `VulkanHybridOlmoeParityTests` covers OLMoE, Qwen3-Coder-30B and Qwen3-0.6B). Everything else stays on the sequential path.

**Steps**
1. Build one CPU `ForwardPass` over the same `GgufModel` on first use and keep it for the hybrid's lifetime (construction resolves and pre-faults weights and allocates cache and scratch; paying that per prompt would defeat the benchmark). Reset its cache between requests.
   - *Memory model*: weights are shared by mapping, state is not. `GgufModel` owns the memory-mapped weights; the `HybridForwardPass` owns its GPU/CPU hybrid state; the retained CPU pass owns only its own scratch and KV. Construction must not copy or re-upload weight tensors (verify with peak private memory in Phase 0 check (b)).
   - *Single owner*: the retained CPU prefill pass is stateful and not re-entrant. Concurrent requests must either serialize their use of the fast path or obtain separate CPU prefill state; one request's `Reset()` must never run while another's prefill or handoff is reading the cache. Concurrency is not solved in this plan (the hybrid pass is already single-sequence), only stated as an invariant so nobody removes the serialisation.
2. For a fresh prompt with `N >= gate` (start at 32, tune from Phase 0): `ForwardPass.Prefill(tokens)`; chunk long prompts to bound activation scratch.
3. Read the completed `ForwardPass.Cache` (`PagedKvCache`) and copy K/V rows `[0, N)`: CPU-resident layers into the hybrid `KvCache` (F32 rows), GPU-resident layers into the existing hybrid GPU K/V tensors (F32 slices via `VulkanBackend.Upload`/`UploadRaw`, one transfer per layer). Set the hybrid's position counters as the sequential path would.
4. Return the CPU prefill logits as the prompt's final logits; keep the hybrid decode path unchanged.
5. **Memory budget**: the CPU pass holds a full-model temporary KV cache that grows with the prompt. Report peak private memory, and fall back to the sequential path when the temporary cache would exceed the configured budget. Chunking bounds scratch, not this cache.

**Files**: `HybridForwardPass.cs` (Prefill dispatch, KV accessors), new `HybridPrefillHandoff.cs` in `Engine`, a KV upload helper in `VulkanBackend` if missing, reuse `ForwardPass.Prefill`.

**Tests (three separate properties, replacing "40 decode tokens match token-for-token")**
- A. Handoff exactness: CPU `PagedKvCache` rows vs the hybrid's CPU-layer `KvCache` rows and a download of the GPU-layer tensors: byte-identical. This proves the transfer.
- B. First-decode parity: decode after handoff vs decode after sequential hybrid prefill, under the existing cross-backend contract (cosine > 0.99, argmax flips only on a near-tie of the reference logits). The K/V values differ in low bits because CPU kernels produced them instead of GPU kernels, so bitwise equality is not the right bar.
- C. Generation: same prompt, greedy, near-ties documented; plus the boundary prompt lengths below.
- Short prompts (below the gate) must not regress; first-use construction, warmed CPU prefill, handoff, and end-to-end are benchmarked separately (see Acceptance).

**Acceptance** (record these figures separately; the user-facing number is the last one): fresh OLMoE 631-token prompt through `-g 8`: (i) warmed CPU-prefill component >= 130 tok/s (CPU-only is 141.6); (ii) KV handoff time; (iii) first-use construction time; (iv) end-to-end prefill throughput including handoff and hybrid setup. 130 tok/s is a regression target for the CPU component on this machine, not a general Phase 1 guarantee. Tests A-C pass; peak memory recorded; `startPos > 0` demonstrably still on the old path.

### Phase 2: batched hybrid trunk with host MoE (Vulkan port of CUDA #410), for when the GPU attention trunk is worth keeping
Idea: process the chunk layer by layer instead of token by token. GPU layers: batched attention trunk (`RmsNormBatched` -> `MatMulBatched` q/k/v -> `RoPEBatched` -> `KvAppendBatched` -> `AttentionBatched` -> o-proj -> residual) in chunks of at most 16 tokens (Path 1 limit); FFN for MoE layers via the CUDA #410 pattern: one D2H of the chunk's post-FFN-norm rows, host router (exact per-token F32 path), `MoeBatchedExperts.Run` over CSR buckets, reduce in top-k slot order, one H2D, batched add. CPU layers: batched CPU trunk (reuse the `ForwardPass.PrefillCore` layer logic over the hybrid's CPU weights/KV).
- Value: keeps GPU attention and KV native (no handoff), lets GPU layers and CPU layers both batch, and is the structure Phases 3-4 plug into (swap the host MoE step for a GPU one).
- Extend `ComputeCanBatchedTrunk` so MoE no longer disqualifies when the host-MoE stage is available; add the missing batched pieces only if absent (batched residual add, row-broadcast bias exists).
- Cost: moderate; the per-layer D2H/H2D replaces per-token round trips, but the chunk cap of 16 tokens means ~40 round trips for 631 tokens per layer unless the cap is lifted (Phase 3 Path 2) or the host stages run on the whole prompt per layer (activations for N tokens are only N*embDim*4 bytes: 631*2048*4 = 5 MB, so the D2H/H2D can cover the whole prompt per layer with the attention trunk chunked internally; do that).
- **Exit criterion (hard gate, not a note)**: proceed only if the Phase 0/1 measurements show GPU attention for the chunk beating CPU batched attention on the target hardware; on this machine the dense numbers suggest it will not, in which case Phase 2 is skipped here. Acceptance when run: equal to Phase 1's logits on the same prompts; on this machine it should be at least as fast as Phase 1 only if GPU attention beats CPU attention, which dense measurements say it may not; keep it only if the Phase 0 numbers justify it (rule 7).

### Phase 3: GPU grouped-expert prefill on Vulkan (regime B; informed by llama.cpp, not a port of it)
Reduce how much new numerical machinery arrives at once. Split into 3a and 3b.

**Invariants (name them in code and tests)**
- *MoE reduction order is part of the numerical parity contract*: partials are reduced per token in top-k slot order, never expert order (it moved OLMoE logits by up to 0.20).
- *Bucket correctness, not bucket order*: every (token, slot) pair appears exactly once in its selected expert's bucket and keeps its destination identity. Order within a bucket is not load-bearing (vLLM's own tests only check regions), so atomic-counter bucketing is acceptable; our CPU CSR happens to be stable but the GPU need not be.

**Phase 3a (first GPU version)**
1. GPU router GEMM over the chunk's normed rows; download only the router logits (`N * numExperts` floats, 160 KB for 631 tokens x 64 experts).
2. **Exact selection stays on the host**, reusing `RouteExperts` (`ForwardPass.Moe.cs:643`): it already encodes softmax vs sigmoid, DeepSeek-style gating with the selection bias, selected-vs-weighted semantics, renormalisation and expert scaling, with deterministic tie-breaks. Upload `selected[N*k]` and `weights[N*k]` (tiny).
3. `MoeCountExperts` on the GPU: per-expert counts, prefix offsets, grouped row ids (llama.cpp `count_experts.comp` is the reference). The hoisted path is limited by the packed row-id format `(i01 << 16) | i00`, which requires both indices <= 65,535 (llama.cpp checks `nei0 <= 0xffff && nei1 <= 0xffff`), and by a maximum expert count (1,024 today; treat it as the current limit, not a constant). Stingray must either use a wider row-id format or fall back to a non-hoisted path for longer sequences; test with synthetic indices past 65,535 even though the first real tests use 631 and 2,048 tokens.
4. `MatMulBatchedExpert`: a **new grouped, indirect kernel variant**, not a small extension of `MatMulBatched`: grid (row tile, tokens-of-expert tile, expert); weight offset = expert * rowsPerExpert * bytesPerRow (the missing expert offset); gather activations by row id, scatter outputs by row id. Reuse Path 1's accumulator strategy where practical; Q4_K and Q6_K first. Real speed for large prompts needs the unimplemented Path 2 tiled GEMM (a separate project; tile size chosen from tokens per expert, as llama.cpp and vLLM do).
5. `MoeReduceSlotOrder`: per token, sum the k unweighted down partials in slot order, add the shared expert.

**Phase 3b (only if profiling shows the small router download and host selection matter)**: fully on-GPU router + top-k, matching `SelectTopK` tie-breaking exactly (adversarial near-tie tests). vLLM treats fused grouped top-k as a backend-specific optimisation, which supports deferring it.

Validation here is correctness (cosine vs CPU, argmax near-tie rule, as `VulkanHybridOlmoeParityTests`). Local micro-performance evidence is still useful (dispatch count, bytes moved, grouping and routing overhead, kernel time) to reject a poor design before it reaches a discrete GPU, but it is not evidence about other hardware. Ship off by default behind `STINGRAY_VULKAN_MOE_PREFILL=1` until measured on a discrete GPU.

### Phase 4: oversized experts, stream per chunk (regime C)
For layers whose experts are not resident: upload only the experts the chunk uses (count them from the CSR buckets; the CUDA op-offload uploads the whole layer, which wastes bandwidth when few experts are used, e.g. 128-256 expert models and short chunks), double-buffered against compute, after parallel prefault of those pages (TensorSharp, 16 threads). Reuse the lease/in-flight machinery from `ExpertSlotManager` (a staging slot is a transient lease). Gate by tokens (llama.cpp 32, our CUDA 64, TensorSharp 128; measure ours).
- Depends on Phase 3 kernels (or the CPU path as the fallback below the gate).

### Phase 5: dispatcher (a measured strategy table, not a device threshold)
Choose among: CPU full prefill (Phase 1), host-routed hybrid (Phase 2), GPU-resident grouped MoE (Phase 3), GPU-streamed experts (Phase 4). The crossover depends on far more than the device: llama.cpp users report different `GGML_OP_OFFLOAD_MIN_BATCH` optima for Q4_K_M and Q8_0 on the same laptop (reported upstream; not verifiable from this checkout), and our own CUDA op-offload comments show a default of 64 against a measured benefit from about 120 tokens, TensorSharp uses 128, llama.cpp 32. So the calibration key is: model fingerprint, quantisation, backend/device, CPU ISA, layer placement, expert residency, KV configuration, and a prompt-length bucket. Store measured thresholds per key (model home), with conservative defaults when no entry exists, and an override `STINGRAY_PREFILL_MOE=cpu|gpu|auto`. Use a small in-process calibration of a fixed MoE chunk on both devices only to seed a missing entry.

## Tests and acceptance (all real-weight tests skip visibly without the checkpoint; check wall time per rule 12)
- Parity: hybrid/GPU prefill logits vs the sequential path and vs `-g 0`, cosine > 0.99 and no argmax flip beyond the 2% near-tie rule, OLMoE (64 experts, top-8), Qwen3-Coder-30B (128 experts), plus a model with a shared expert and sigmoid gating where a checkpoint exists.
- Chunk invariance: one prompt prefilled in one chunk vs several chunks gives the same logits; multi-turn (`startPos > 0`); prompts of 1, 2, 15, 16, 17, 32, 33, 631 tokens (the 16-token tile edge and the dispatcher gate edge).
- Expert-occupancy adversarial tests with synthetic routing (not only real prompts): all tokens to one expert; uniform; one token per expert; all top-k slots distinct; some experts empty; one expert receiving almost everything; token indices past 65,535 for the packed row-id path.
- Slot-order reduction test (the bug class that moved logits by 0.20): reduce in expert order must fail it.
- KV handoff (Phase 1): tests A-C above (byte-exact copy, tolerance-based first-decode parity, greedy generation with near-ties documented); the prefill pass asserted to use F32 KV when `STINGRAY_KV_STORE=auto` and the prompt exceeds 1,024 tokens.
- Memory: scratch bounded for long prompts; `StreamedExpertMemoryBoundTests`-style leak check after repeated prefills; live GPU buffer count flat.
- Performance gates recorded in the doc (3 runs each): OLMoE 631 tokens through `-g 8` >= 130 tok/s after Phase 1 (CPU-only 141.6). For every tested hardware/model/prompt configuration, `auto` must select a strategy whose measured prefill throughput is no worse than the CPU-only baseline within an agreed tolerance; explicit `cpu`, `gpu` and `auto` modes stay available for diagnostics.
- Regression: existing `VulkanHybridOlmoeParityTests`, `ExpertSlotManagerConcurrencyTests`, dense `VulkanArchLogitParityTests`, `ForwardPass.Fast`, `Server.Fast`.

## Risks
- The iGPU may never beat the CPU for prefill; then Phases 2-4 only matter on other hardware. Mitigation: Phase 1 first, calibrate, ship Phase 3/4 off by default until measured on a discrete GPU (a decision for you: where can that be measured?).
- KV handoff mismatches (BF16 narrowing in `PagedKvCache`, TurboQuant, SWA, per-layer head dims) or the temporary CPU KV cache and duplicated scratch exceeding the memory budget: Phase 0 questions (a) and (b) are the gate; if they fail, Phase 2 becomes the first implementation.
- Parity: reduction order and top-k tie-breaking are exact-match requirements; both have named tests.
- Path 1's 16-token limit makes Vulkan batched matmul amortise little for experts with many tokens; the real fix is the unimplemented Path 2 tiled GEMM (large, separate).
- CUDA has the same structure and several of the same defects (see the residency plan); not changed or testable here.

## Order and rough size
Phase 0 (hours) -> Phase 1 (about a day or two, biggest win here, lowest risk) -> measure -> Phase 2 only if justified -> Phase 3 kernels (days, needs a discrete GPU to judge) -> Phase 4 (days, after 3) -> Phase 5 (small, last, driven by the measurements).

## Decisions for you
1. Approve Phase 0 + Phase 1 as the first implementation (CPU prefill with KV handoff), with Phases 2-5 gated on its measurements.
2. Is there a machine with a discrete GPU where Phase 3/4 speed can be measured? Without one they stay correctness-only and off by default.
3. Whether the plan should also cover CUDA (same structure, untestable here) or stay Vulkan-only.

## Verification of the plan itself
Read-only checks to run first: the two Phase 0 questions; confirm `SimdKernels` batched kernels are callable from the hybrid's CPU layers; confirm no other AI is editing `HybridForwardPass.cs` or `ForwardPass.*.cs` before starting (git status).
