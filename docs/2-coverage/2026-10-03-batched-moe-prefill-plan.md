# Plan: batched MoE prefill for the Vulkan hybrid / GPU paths

Written 2026-10-03. Companion to [2026-10-03-streamed-residency-plan.md](2026-10-03-streamed-residency-plan.md), which lists this as its largest remaining item. **Status: plan only, nothing implemented.**

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
- Read-only checks that decide Phase 1's shape (answer them in the doc): (a) is `HybridForwardPass._cpuKvCache` the same `CpuKvCache` type `ForwardPass` uses, and what is the GPU KV layout (Vulkan stores fp16; dtype/TurboQuant variants); (b) can a CPU `ForwardPass` be constructed over the same `GgufModel` for all layers without duplicating weights (they are memory-mapped; the CPU pass pre-faults them).

### Phase 1: "prefill where it is fastest, hand off the KV" (recommended first implementation; no new kernels)
Idea: for a prompt of N >= gate tokens (default 32, llama.cpp's number, tuned in Phase 0), run the whole prefill on the CPU with the existing batched path (`ForwardPass.Prefill`, which already includes `MoeFfnBatched`), then copy each layer's K/V rows [startPos, startPos+N) into the hybrid's caches: GPU layers' KV buffers (fp16 conversion), CPU layers' `_cpuKvCache`. Decode then continues on the hybrid unchanged.
- Why first: it reuses the whole verified CPU path, so the result is the CPU prefill speed (141 tok/s on OLMoE here, 5.6x today) with logits identical to `-g 0`, and correctness reduces to "is the KV handoff exact", which is easy to test. It is also the right answer for regime A permanently.
- New code: a `KvHandoff` helper (CPU KV -> GPU KV buffers via `VulkanBackend.Upload`/`UploadRaw`, per layer, one transfer per layer per prompt); hybrid `Prefill` dispatch (`n >= gate && cpuPrefillAvailable`); a lazily built CPU `ForwardPass` sharing the model (constructed only when a long prompt arrives; released or kept per memory budget).
- Files: `HybridForwardPass.cs` (Prefill, KV accessors), new `HybridPrefillHandoff.cs` in `Engine`, `VulkanBackend` (KV upload helper if missing), reuse `ForwardPass.Prefill`.
- Limits to handle explicitly: TurboQuant KV (skip handoff, fall back to per-token), SWA layers, `startPos > 0` (multi-turn: handoff appends), models whose hybrid path already refuses (Gemma 4 etc.), memory (CPU pass scratch for N tokens: chunk the prompt to bound it), thinking about snap-KV (skip).
- Acceptance: OLMoE 631-token prefill >= 130 tok/s through `-g 8` (CPU-only is 141.6); logits after prefill within the existing hybrid parity contract vs the sequential hybrid path; 40 decode tokens after the handoff match the sequential-prefill run token-for-token (same device kernels for decode); Qwen3-Coder-30B same; multi-turn (second prompt with `startPos > 0`) matches.

### Phase 2: batched hybrid trunk with host MoE (Vulkan port of CUDA #410), for when the GPU attention trunk is worth keeping
Idea: process the chunk layer by layer instead of token by token. GPU layers: batched attention trunk (`RmsNormBatched` -> `MatMulBatched` q/k/v -> `RoPEBatched` -> `KvAppendBatched` -> `AttentionBatched` -> o-proj -> residual) in chunks of at most 16 tokens (Path 1 limit); FFN for MoE layers via the CUDA #410 pattern: one D2H of the chunk's post-FFN-norm rows, host router (exact per-token F32 path), `MoeBatchedExperts.Run` over CSR buckets, reduce in top-k slot order, one H2D, batched add. CPU layers: batched CPU trunk (reuse the `ForwardPass.PrefillCore` layer logic over the hybrid's CPU weights/KV).
- Value: keeps GPU attention and KV native (no handoff), lets GPU layers and CPU layers both batch, and is the structure Phases 3-4 plug into (swap the host MoE step for a GPU one).
- Extend `ComputeCanBatchedTrunk` so MoE no longer disqualifies when the host-MoE stage is available; add the missing batched pieces only if absent (batched residual add, row-broadcast bias exists).
- Cost: moderate; the per-layer D2H/H2D replaces per-token round trips, but the chunk cap of 16 tokens means ~40 round trips for 631 tokens per layer unless the cap is lifted (Phase 3 Path 2) or the host stages run on the whole prompt per layer (activations for N tokens are only N*embDim*4 bytes: 631*2048*4 = 5 MB, so the D2H/H2D can cover the whole prompt per layer with the attention trunk chunked internally; do that).
- Acceptance: equal to Phase 1's logits on the same prompts; on this machine it should be at least as fast as Phase 1 only if GPU attention beats CPU attention, which dense measurements say it may not; keep it only if the Phase 0 numbers justify it (rule 7).

### Phase 3: GPU grouped-expert prefill on Vulkan (regime B; llama.cpp design)
Kernels, in dependency order, each with a golden test against the CPU `MoeBatchedExperts` result:
1. `MoeRouterTopK`: batched router GEMM + softmax (or sigmoid) + top-k + optional renormalisation, on the GPU, output `selected[N*k]`, `weights[N*k]`. Must tie-break exactly like `SelectTopK` (host) so selected experts are identical; test with adversarial near-ties.
2. `MoeCountExperts` (llama.cpp `count_experts.comp`): per-expert counts, prefix offsets, packed row ids grouped by expert in a stable order. Counts up to 1024 experts. Output stays on the GPU.
3. `MatMulBatchedExpert` (a grouped, id-driven variant of `MatMulBatched`): grid (rows tile, tokens-of-expert tile, expert); weight base offset = expert * rowsPerExpert * bytesPerRow (the missing "expert offset"); gather activations by row id, scatter results by row id. Q4_K and Q6_K first (the dtypes `MatMulBatched` already handles), then the rest. Initially on Path 1's register-accumulator design (16 tokens per tile) so no new GEMM is needed; real speed on large prompts needs the tiled Path 2 GEMM, which is its own project (shared-memory tiles, tile size chosen from tokens per expert as llama.cpp and vLLM do).
4. `MoeReduceSlotOrder`: per token, sum the k unweighted down partials **in slot order** (parity), add shared-expert output.
- Decision to record: dynamic row ids (llama.cpp, no padding) rather than vLLM-style padding; simpler to match our CSR semantics.
- Validation here: correctness only (cosine vs CPU, argmax near-tie rule, as `VulkanHybridOlmoeParityTests`); speed cannot be judged on the iGPU. Needs a discrete GPU to decide the crossover; until then ship it off by default behind `STINGRAY_VULKAN_MOE_PREFILL=1`.

### Phase 4: oversized experts, stream per chunk (regime C)
For layers whose experts are not resident: upload only the experts the chunk uses (count them from the CSR buckets; the CUDA op-offload uploads the whole layer, which wastes bandwidth when few experts are used, e.g. 128-256 expert models and short chunks), double-buffered against compute, after parallel prefault of those pages (TensorSharp, 16 threads). Reuse the lease/in-flight machinery from `ExpertSlotManager` (a staging slot is a transient lease). Gate by tokens (llama.cpp 32, our CUDA 64, TensorSharp 128; measure ours).
- Depends on Phase 3 kernels (or the CPU path as the fallback below the gate).

### Phase 5: dispatcher
`PrefillMoeStrategy.Choose(n, device, expertResidency)` returning `CpuBatched` (Phase 1/2), `GpuResident` (Phase 3), `GpuStreamed` (Phase 4). Inputs: chunk tokens, integrated vs discrete GPU (Vulkan device type; llama.cpp tracks it too), expert residency, and a one-time calibration (time a small fixed MoE chunk on both devices at startup or cache per device in the model home) rather than hard-coded thresholds. Env override `STINGRAY_PREFILL_MOE=cpu|gpu|auto`. Defaults must never pick a path that Phase 0 showed to be slower on the detected hardware class.

## Tests and acceptance (all real-weight tests skip visibly without the checkpoint; check wall time per rule 12)
- Parity: hybrid/GPU prefill logits vs the sequential path and vs `-g 0`, cosine > 0.99 and no argmax flip beyond the 2% near-tie rule, OLMoE (64 experts, top-8), Qwen3-Coder-30B (128 experts), plus a model with a shared expert and sigmoid gating where a checkpoint exists.
- Chunk invariance: one prompt prefilled in one chunk vs several chunks gives the same logits; multi-turn (`startPos > 0`); prompts of 1, 2, 15, 16, 17, 32, 33, 631 tokens (the 16-token tile edge and the dispatcher gate edge).
- Slot-order reduction test (the bug class that moved logits by 0.20): reduce in expert order must fail it.
- KV handoff exactness (Phase 1): continue decoding 40 tokens after handoff and compare with the sequential prefill; fp16 round trip documented.
- Memory: scratch bounded for long prompts; `StreamedExpertMemoryBoundTests`-style leak check after repeated prefills; live GPU buffer count flat.
- Performance gates recorded in the doc (3 runs each): OLMoE 631 tokens through `-g 8` >= 130 tok/s after Phase 1 (CPU-only 141.6); never below CPU-only on the same machine for prompts >= the gate.
- Regression: existing `VulkanHybridOlmoeParityTests`, `ExpertSlotManagerConcurrencyTests`, dense `VulkanArchLogitParityTests`, `ForwardPass.Fast`, `Server.Fast`.

## Risks
- The iGPU may never beat the CPU for prefill; then Phases 2-4 only matter on other hardware. Mitigation: Phase 1 first, calibrate, ship Phase 3/4 off by default until measured on a discrete GPU (a decision for you: where can that be measured?).
- KV handoff format mismatches (fp16, TurboQuant, SWA, per-layer head dims) or CPU-pass memory duplication: Phase 0 questions (a) and (b) are the gate; if they fail, Phase 2 becomes the first implementation.
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
