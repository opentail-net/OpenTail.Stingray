# Plan: batched MoE prefill for the Vulkan hybrid / GPU paths

> Completed items moved to [done/2026-10-03-batched-moe-prefill-plan-completed-items-2026-10.md](../done/2026-10-03-batched-moe-prefill-plan-completed-items-2026-10.md) on 2026-10-30. This file keeps only open work.

## Checklist (2026-10-30; code checked)

- [ ] Phase 1 CUDA variant: written, compiles, never run (no NVIDIA GPU here; code: `CudaHybridPrefillHandoff.cs`)
- [ ] Phase 2: batched hybrid trunk with host MoE (gated on the Phase 0/1 measurements)
- [ ] Phase 3: GPU grouped-expert prefill on Vulkan (3a and, if justified, 3b)
- [ ] Phase 4: oversized experts, streamed per chunk (regime C)
- [ ] Phase 6: discrete-GPU decode miss split and device-side cache decisions
- [ ] Decision for the user: approve Phase 0 + 1 as the first implementation (Phase 1 is done; Phases 2-5 gated on its measurements)
- [ ] Decision for the user: a discrete-GPU machine for Phase 3/4 speed measurement

Written 2026-10-03. Companion to [2026-10-03-streamed-residency-plan.md](2026-10-03-streamed-residency-plan.md), which lists this as its largest remaining item. **Status: plan only, nothing implemented.**

**Revision 2026-10-03 (after an external review; each point was checked against the code before adopting).** Corrected: the hybrid's CPU KV is `KvCache` (FP32, `[maxSeqLen, kvDim]` per layer), not the `PagedKvCache` that `ForwardPass` uses; the hybrid's GPU KV is FP32 (`gpu.Allocate` default dtype), the fp16 KV I cited belongs to `GpuForwardPass`, so Phase 1 has no precision conversion; Phase 1 is fresh sequences only (`startPos == 0`); `PagedKvCache` can narrow itself to BF16 (explicitly via `STINGRAY_KV_DTYPE` / `STINGRAY_KV_STORE`, or automatically at 1,024 tokens in `auto` mode), so the CPU prefill pass must be pinned to F32 KV or the handoff would silently put BF16-rounded K/V into the hybrid.

## Implementation status (2026-10-04)

Phase 0 and Phase 1 are implemented for the Vulkan hybrid; Phase 1b (full-offload `GpuForwardPass`) is implemented; the CUDA variant is written, compiles, and has **never run** (no NVIDIA GPU here; opt-in, default off). Phases 2-5 are not started. What exists:

- `HybridPrefillHandoff.cs` (Vulkan hybrid), `GpuPrefillHandoff.cs` (full-offload Vulkan, MoE only, F32 or packed-fp16 KV), `CudaHybridPrefillHandoff.cs` (opt-in, F32 KV only), shared `KvHandoff.cs`; `PagedKvCache.Bf16RoundingRequested`, `ForwardPass.MoeRoutingObserver`, `ExpertSlotManager.PromoteOnPreload/Capacity`.
- Tests: `HybridCpuPrefillHandoffTests` (byte-exact K/V and logits, decode parity, boundary lengths, `startPos > 0`, budget refusal, expert warm-up), `GpuCpuPrefillHandoffTests` (same for full GPU, both KV dtypes, fp16 packing), `CudaHybridCpuPrefillHandoffTests` (not run).
- Environment variables: `STINGRAY_HYBRID_CPU_PREFILL` (`0` off, `all` lifts the family list), `..._MIN_TOKENS`, `..._KV_BUDGET_MB`, `..._WARM`, `STINGRAY_GPU_CPU_PREFILL`, `STINGRAY_CUDA_HYBRID_CPU_PREFILL`, `STINGRAY_PREFILL_TIMING`.

**Phase 0 baseline** (OLMoE-1B-7B Q4_K_M, Ryzen 5700G + integrated Radeon, Release, 16 generated tokens, 3 runs each; prompts are varied prose from the docs, so token counts are whatever they tokenised to):

| Prompt (tokens) | Configuration | Prefill tok/s (mean of 3) | Decode tok/s (mean of 3) |
|---|---|---:|---:|
| 41 | CPU only (`-g 0`) | 73.3 | 21.5 |
| 41 | hybrid `-g 8`, sequential prefill | 17.2 | 23.1 |
| 41 | hybrid `-g 8`, CPU prefill + handoff | 45.0 | 15.8 (cold cache) |
| 41 | full GPU `-g -1` | 31.0 | 28.5 |
| 168 | CPU only | 99.2 | 35.9 |
| 168 | hybrid sequential | 21.5 | 22.5 |
| 168 | hybrid handoff | 87.8 | 18.6 (cold cache) |
| 168 | full GPU | 29.5 | 26.6 |
| 740 | CPU only | 141.7 | 31.9 |
| 740 | hybrid sequential | 21.5 | 17.0 |
| 740 | hybrid handoff | 133.3 | 14.0 (cold cache) |
| 740 | full GPU | 26.0 | 22.5 |
| 2,597 | CPU only | 140.9 | 21.1 |
| 2,597 | hybrid sequential | 15.7 | 10.6 |
| 2,597 | hybrid handoff | 142.1 | 9.2 (cold cache) |
| 2,597 | full GPU | 19.6 | 14.3 |

Reading it (integrated GPU, rule 13: nothing here says anything about a discrete GPU):
- Pure CPU is the best configuration on this machine for MoE at every prompt length: it prefills at 99-142 tok/s and decodes at 21-36.
- The hybrid's sequential prefill is 17-22 tok/s; the CPU-prefill handoff lifts it to 88-142 (5-9x), reaching the CPU-only prefill speed at 2,597 tokens.
- Full-GPU decode is the fastest GPU decode (28.5 at short prompts) and its MoE prefill the slowest (20-31); Phase 1b hands the two together.
- Decode after a handoff was 3-5 tok/s slower than after a sequential prefill until the expert warm-up below.

**Phase 1 results**: byte-identical K/V rows (GPU tensors and CPU cache, every layer) and logits identical to the CPU pass; decode after the handoff has cosine 0.996-0.9999 against sequential hybrid prefill on OLMoE, Qwen3-0.6B and Qwen3-Coder-30B; the handoff itself costs about 70-80 ms for 740 tokens, first-use construction of the retained CPU pass about 200 ms.

**Expert warm-up (found by measurement, 2026-10-04).** After a CPU prefill the GPU expert cache is empty (the sequential prefill filled it as a side effect), so the first decode steps missed 62-74% of lookups (decode 14-15 tok/s against 18-20). Three attempts:
1. Preload the hottest experts after the handoff, synchronously: 510 experts took about 2 s (end-to-end prefill 133 -> 98 tok/s) and did not help, because new entries land in the small probationary segment and a bulk preload evicts its own earlier entries (miss rate stayed 31% at 48 tokens).
2. The same in the background with promotion: misses halved but decode speed did not improve (the uploads compete with decode).
3. **Queue each GPU layer's hot experts the moment the CPU prefill reports that layer's routing (`MoeRoutingObserver`), upload them on a background thread while the CPU is still busy with later layers, promote each to the protected segment.** Misses during decode: 0.0% (128 tokens: 1 of 8,192); decode 19.2-20.4 tok/s, equal to the sequential baseline (18.4-20.3); cost: about 8% of CPU prefill throughput (uploads share DRAM bandwidth with the CPU), no added latency after the prefill.

**Phase 1b results (full-offload `GpuForwardPass`, `-g -1`, OLMoE, 3 runs each, 64 decoded tokens):** 740-token prompt: prefill 126-130 tok/s with the handoff against 17-27 sequential, decode 22.0-22.3 against 23.0 (no cold-cache effect: every expert is already resident); 2,597 tokens: 131-140 against 19.8-20.1, decode 14.2-14.7 against 14.6-14.7. The KV handoff costs about 255 ms at 740 tokens and 690-766 ms at 2,597 (packing fp16 on the CPU). Tests: F32 copy byte-exact, packed fp16 equal to an independent `Half` conversion, decode cosine 0.9993-0.9999 against sequential GPU prefill.

**CPU path profile and experiments (2026-10-04; OLMoE, 740 tokens, CPU only, `STINGRAY_PROFILE_PREFILL=1`; new `[MoeExperts]` breakdown in `MoeBatchedExperts`):**
- The FFN is 79% of the prefill trunk (QKV 10%, attention 5%, output projection 3%). Inside the routed-expert stage: 93 rows per expert run on average; worker time is gate matmul 37%, up 35%, down 27%, everything else under 1%; Q4_K matmuls account for 24.6 s of summed worker time and Q6_K for 3.4 s; workers are 6.8 of 8 busy on average.
- Per worker the Q4_K path runs at about 40 GFLOP/s and the Q6_K path at about 59, against a Zen 3 fp32 peak of about 130 per core. The kernel comments describe the exact path (int8-activation dots, four tokens per weight read) as dot-compute bound; micro-GEMM is off by default and limited to 16 rows, so it never applies at about 93 rows.
- **`STINGRAY_MOE_PREFILL_Q8=1`: +11% (153 to 171 tok/s), identical text. Not adopted**: exactness is the policy for MoE prefill (Granite NLL), and 11% does not justify revisiting it.
- **Longest-expert-first scheduling of the expert loop: 10% slower (122-137 against 148-154 tok/s), reverted.** The expert workers also run kernels that parallelise internally, so the idle time is not a simple tail problem.
- Conclusion: no cheap exact win exists in the CPU expert stage. Real headroom needs a new Q4_K GEMM that reuses each dequantised weight tile across the roughly 90 rows of an expert (ExLlamaV3's "reconstruct hot experts, dense GEMM" idea); that is a kernel project of its own, to be started from a micro-benchmark, not from this plan.

**Parallel prefault already exists** (`MmapPrefault`, issue #221: OS read-ahead hint plus a parallel stride read, gated on RAM fit). TensorSharp's gain came from faulting only the experts a layer uses, which only matters when the model does not fit in RAM; the experts already fault from many threads inside the expert loop. Deferred until GLM-5.3 is verified and a larger-than-RAM run is possible.

**Measured chooser (Phase 5), design only:** on first use time the real kernels (CPU expert matmul at decode and at prefill row counts, host-to-device copy, GPU batched matmul, and the CPU/copy pair overlapped), store a JSON profile in the model home keyed by device and expert dtype (model detail optional, mismatched profiles ignored as FreeToken does), and have the dispatcher and `stingray plan` read it. On this machine it would pick the CPU. Not built yet.

## Context

`HybridForwardPass.Prefill` (`src/OpenTail.Stingray.Engine/HybridForwardPass.cs:691`) is a loop of single-token `Forward` calls, and `GpuForwardPass` excludes every MoE model from its batched trunk (`ComputeCanBatchedTrunk`, `GpuForwardPass.cs:1270`: "MoE (mid-trunk router submit)"). Measured on this machine, OLMoE-1B-7B Q4_K_M, 631-token prompt, Release build:

| Path | Prefill tok/s |
|---|---:|
| CPU only (`-g 0`, batched MoE) | **141.6** |
| Vulkan hybrid (`-g 8`) | 25.1 |
| Vulkan full GPU (`-g -1`) | 26.6 |

Dense control (SmolLM2-1.7B, 35-token prompt): CPU 104.7, full GPU 78.7. So on this integrated GPU, even the existing dense batched GPU prefill loses to the CPU. Any plan that assumes "GPU prefill is faster" is untested here and, for this machine, probably wrong. Prefill is the largest remaining cost of the hybrid path (decode was fixed in commit `4556fee1`: 2.1-2.7 -> 18.7 tok/s on a 4-slot cache).

Goal: hybrid and full-GPU prefill on MoE models is never slower than the CPU-only path on the same machine, is correct (logits equal the sequential path), and is faster than the CPU where the hardware allows (discrete GPU). Keep the project's rules: no subagents, measure before keeping a change, exact-parity tests, state iGPU results as iGPU results (CLAUDE.md rules 6, 7, 12, 13).

## What the references do

Reference survey (llama.cpp, vLLM, TensorSharp, FreeToken, ExLlamaV3, our own code): [../reference/2026-10-03-batched-moe-prefill-reference-survey.md](../reference/2026-10-03-batched-moe-prefill-reference-survey.md).


What this changes in the plan: Phase 4 is validated and gains a concrete shape (double-buffered whole-layer stream, position == expert id, optional hit copy, batched asynchronous copies without tiny entries); Phase 3 gains a reconstruct tier for hot experts and rows-based kernel selection; Phase 5's calibration should be keyed first by device and expert format (FreeToken's key) with the model as secondary detail, and measure the *overlapped* CPU/transfer pair, not each alone; a new Phase 6 (decode miss split, device-side cache bookkeeping) is added.
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

### Which families get the handoff (2026-10-04)

The handoff gate is no longer an architecture-name list: a structural check (MLA, recurrent state, own forward pass, LongRoPE are incompatible) then a parity receipt. Coverage matrix, how other engines decide, and the admission recipe: [family coverage table](2026-10-04-moe-handoff-closure-plan.md#family-coverage-table-merged-2026-10-30-from-2026-10-04-moe-handoff-coveragemd).

### Phase 5 status: measured chooser built (2026-10-04)

`stingray calibrate` (`Engine/Calibration/ExpertCalibration.cs`, `Cli/CalibrateCommand.cs`) times the real kernels on this machine for Q4_K and Q6_K expert matrices (1024x2048): CPU matmul at 1 row (decode) and 93 rows (OLMoE's average per expert), host-to-device copy, GPU batched matmul on resident weights, and the CPU matmul and copy **at the same time**. It writes JSON to `<model home>/calibration/<gpu>_<threads>t.json` (a profile from another schema version is ignored) and applies explicit rules: cold experts at decode go to the CPU when its weight consumption exceeds 2x the overlapped copy rate (FreeToken's rule); resident-weight prefill goes to the GPU only at 1.25x the CPU's rate; streamed prefill compares copy + GPU time with CPU time. Rules are unit-tested with synthetic numbers (`ExpertCalibrationTests`).

Result on this machine (Ryzen 5700G + integrated Radeon, 8 workers; iGPU numbers only, rule 13):

| | Q4_K | Q6_K |
|---|---:|---:|
| CPU decode, weights consumed | 31-32 GB/s | 37-40 GB/s |
| CPU prefill (93 rows) | 324-338 GFLOP/s | 503-508 GFLOP/s |
| Host-to-device copy, alone / with CPU running | 4.7 / 2.0-2.2 GB/s | 4.3 / 2.1-2.2 GB/s |
| GPU matmul, resident weights (16 rows per dispatch) | 308-315 GFLOP/s | 221-224 GFLOP/s |
| CPU rate kept while copying | 91-92% | 92-93% |

Verdict: CPU for cold decode experts (about 15-19x the copy rate), CPU for resident prefill (GPU 0.97x / 0.44x of the CPU), CPU for streamed prefill. This matches the end-to-end matrix above (CPU-only is the fastest configuration here) from a 20-second probe instead of a full model run. The copy path is slow in absolute terms (4.7 GB/s alone, halved when the CPU runs), so copying experts is a poor trade on shared-memory hardware.

Not done: no engine code reads the profile yet, because the engine has no batched GPU MoE prefill (Phase 3) or streamed prefill (Phase 4) to choose; the decode-side consumer is Phase 6. The calibration is infrastructure for those. A discrete GPU is expected to flip the decode and streamed-prefill verdicts; that is unmeasured here.

### Phase 6 (new, discrete-GPU only): decode miss split and device-side cache decisions
From FreeToken: split a decode step's expert misses between PCIe fetch and CPU compute by the overlapped bandwidth ratio, and move the slot-cache lookup and copy planning onto the device so decode needs no per-layer host round trip (and can be captured in a graph). Not measurable on the integrated GPU (fetching and CPU compute share the same DRAM); needs a discrete GPU, and CUDA code is deferred like the rest of the CUDA work.

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

## Open items from the batched-prefill workstream (ported 2026-10-31)

- [ ] Gemma 4 on Vulkan with a prompt under the CPU-prefill handoff threshold runs at 9.3 tok/s (PerformanceLeague, 2026-10-11 recheck). The minimum is `STINGRAY_HYBRID_CPU_PREFILL_MIN_TOKENS`, default 32 (`src/OpenTail.Stingray.Engine/Planning/EngineEnvironment.cs`), so the 19-token prompt skips the handoff. Gemma 4 has its own batched trunk (`ForwardGemma4`, `GpuForwardPass.cs`): check which path a short Gemma 4 prompt takes before changing the threshold.

## Decisions for you
1. Approve Phase 0 + Phase 1 as the first implementation (CPU prefill with KV handoff), with Phases 2-5 gated on its measurements.
2. Is there a machine with a discrete GPU where Phase 3/4 speed can be measured? Without one they stay correctness-only and off by default.
3. Whether the plan should also cover CUDA (same structure, untestable here) or stay Vulkan-only.

## Verification of the plan itself
Read-only checks to run first: the two Phase 0 questions; confirm `SimdKernels` batched kernels are callable from the hybrid's CPU layers; confirm no other AI is editing `HybridForwardPass.cs` or `ForwardPass.*.cs` before starting (git status).

---

## Research findings (2026-10-31)

Facts are in [research/2026-10-31-coverage-facts.md](research/2026-10-31-coverage-facts.md).

- **llama.cpp CPU MoE path:** the plan's comparison relies on the reference survey, not on the CPU source. The CPU `mul_mat_id` function was not reached by web fetch (the file is too long for the tool), and the local checkout has no `ggml` source. The "how the reference groups tokens per expert" claim is therefore still unchecked, as the 2026-10-31 4-performance pass also recorded.
- **Migration:** the streamed-residency plan's item 1 duplicated this plan and now points here (2026-10-31).
- **Licence of the models used:** the Qwen1.5-MoE and Hunyuan-A13B licences affect which checkpoints can be named as permanent fixtures (see the moe-handoff closure plan).

**Next check:** read `ggml_compute_forward_mul_mat_id` from a llama.cpp checkout at a recorded revision, and write the grouping rule into the reference survey.

---

## Thread state (from the retired workstreams folder, 2026-10-31: batched-moe-prefill.md)

### Correct state

- **Done (code checked 2026-10-30):** Phase 0 (baseline), Phase 1 (CPU-prefill fast path with exact KV handoff: `HybridPrefillHandoff.cs`, `GpuPrefillHandoff.cs`), Phase 1b (full-offload `GpuForwardPass` handoff), and the Phase 5 measured chooser (`Engine/Calibration/ExpertCalibration.cs`).
- **Not built:** Phase 2 (batched hybrid trunk with host MoE, gated on the Phase 0/1 measurements), Phase 3 (GPU grouped-expert prefill), Phase 4 (streamed experts for oversized models), Phase 6 (discrete-GPU decode miss split).
- **Phase 1 CUDA variant:** written and compiles, never run (no NVIDIA GPU here).
- **Chooser not wired:** the engine does not read the calibration profile yet, because the Phase 3/4 paths it would choose between do not exist.
- **Correction to the backlog:** [../00-current-work.md](../00-current-work.md) item 5 says "`VulkanLayerSplitForwardPass` prefills per token". That is accurate for the layer-split and MoE paths only. `GpuForwardPass` already batches (see [vulkan-gemm-llm-prefill.md](../4-performance/gpu/vulkan-gemm/vulkan-gemm.md)).

### Open items

- [ ] Phase 2 decision: proceed only if the Phase 0/1 measurements show GPU attention beating CPU batched attention on the target hardware (hard gate)
- [ ] Phase 3 (3a, and 3b only if profiling justifies it), tested on a discrete GPU before any default change
- [ ] Phase 4 (streamed experts), after the Phase 3 kernels exist, or with the CPU fallback below the gate
- [ ] Phase 5 wiring: the engine reads the profile, once there is more than one path to choose between
- [ ] CUDA variant: run on an NVIDIA machine
- [ ] Phase 6 (discrete GPU only)

- [ ] Gemma 4 on Vulkan with a prompt below the CPU-prefill handoff threshold runs at 9.3 tok/s (PerformanceLeague, 2026-10-11 recheck, finding 5). The minimum is `STINGRAY_HYBRID_CPU_PREFILL_MIN_TOKENS`, default 32 (`Planning/EngineEnvironment.cs`), the same gate as llama.cpp's op offload (`HybridPrefillHandoff.cs`). The 19-token prompt skips the handoff. Gemma 4 has its own batched trunk (`ForwardGemma4`, `GpuForwardPass.cs`): check which path a short Gemma 4 prompt takes before changing the threshold.

### Decisions for the owner

- Sign-off on Phase 0 and Phase 1 as the first implementation. Phase 1 is built; this is recorded for the owner's approval.
- A machine with a discrete GPU to measure Phase 3/4 speed. Without one they stay correctness-only and off by default.

### Cadence

- Review when a discrete-GPU machine is available, or when a Phase 2 gate measurement is taken.
- Re-read at least every two weeks while any phase is open.
- Close when each phase is built and gated, or written off with the reason in the plan.
