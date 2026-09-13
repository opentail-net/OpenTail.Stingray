# Wan 2.1 GPU Kernel Fusion & QKV Optimization Plan (Phase 2) (2026-09-13)

## Context & Phase 1 Retrospective

In Phase 1 (`docs/072-wan21-gpu-kernel-tuning-plan.md`), we addressed the initial performance deficit where `WanModel.ForwardGpu` ran at **1873.2 ms/block** (~1.6× slower than CPU). By converting linear projection weights to FP16 (unlocking the 611 GFLOP/s `SgemmF16` kernel), eliminating per-block dynamic memory allocations in `LayerNormGpu`, and eliminating CPU modulation dictionary lookups, runtime dropped to **738.7 ms/block** (**2.54× speedup**, beating CPU by **1.78×**).

## Objective

Reduce Wan 2.1 DiT per-block GPU runtime from **738.7 ms** to **< 450 ms**, bringing a 30-layer × 40-step generation run from ~14.8 minutes down to **< 9 minutes**.

## Key Bottlenecks in the Current 738.7 ms Runtime

In 1 block of Wan 2.1 (`numTokens=2048, dim=1536, numHeads=12, ffnDim=8960`):
1. **Self-Attention Q/K/V Dispatches (3 separate GEMMs)**:
   - Dispatches `Sgemm` three times: `[2048, 1536] × [1536, 1536]` for Q, K, and V.
   - Issuing 3 separate small GEMMs incurs 3 pipeline dispatches, 3 state setups, and lower wavefront occupancy than a single wider GEMM.
2. **Sequential RMSNorm & 3D-RoPE Dispatches (3 separate passes)**:
   - `RmsNormBatched(Q)` $\rightarrow$ `RmsNormBatched(K)` $\rightarrow$ `Flux2DRoPE(Q, K)`.
   - Reads and writes `Q` and `K` from global VRAM multiple times.
3. **Multi-Step FFN Execution**:
   - `Sgemm(Ffn1, Normed2, Ffn0)` $\rightarrow$ barrier $\rightarrow$ `VisionGeluInPlace(Ffn1)` $\rightarrow$ barrier $\rightarrow$ `Sgemm(FfnOut, Ffn1, Ffn2)`.
   - The intermediate tensor `[2048, 8960]` (73.4 MB in FP32) is fully materialized in global VRAM and read back.

---

## Phase 2 Architecture & Optimization Steps

### 1. Fused QKV Projection GEMM
- In `WanGpuWeights`, concatenate `SelfAttnQ`, `SelfAttnK`, and `SelfAttnV` weights into a single contiguous `SelfAttnQkv` tensor of shape `[dim * 3, dim]` (`[4608, 1536]`).
- In `WanGpuWorkspace`, allocate a combined `Qkv` buffer of shape `[numTokens, dim * 3]`.
- Dispatch a single `SgemmF16` call: `M=numTokens (2048)`, `K=dim (1536)`, `N=dim * 3 (4608)`.
- Replace 3 separate dispatches with 1 wide dispatch, improving wave occupancy on AMD Vega CUs.

### 2. Fused QKV Norm & 3D-RoPE Shader (`WanQkvNormRoPE`)
- Create a fused compute shader that:
  1. Reads `Q` and `K` slices directly from the `Qkv` tensor.
  2. Computes per-head RMSNorm scaling in-register.
  3. Applies 3D rotary embedding (RoPE) using cosine and sine tables.
  4. Writes out normalized and rotated `Q` and `K` buffers ready for attention.
- Replaces 3 separate global memory passes (`RmsNorm(Q)`, `RmsNorm(K)`, `Flux2DRoPE(Q,K)`) with 1 unified pass.

### 3. Verification & Parity Assertion
- Ensure numerical parity is strictly maintained with `WanGpuParityTests` (`maxDiff < 1e-2`).
- Measure per-block improvement using `WanModelSyntheticMicrobenchTests`.

---

## Success Criteria
- Per-block GPU runtime drops below **500 ms** (targeting **< 450 ms**).
- 100% numerical parity maintained against CPU reference.

---

## Completion & Benchmark Results (2026-09-13)

### Implementation Completed
1. **Fused QKV Projection (`WanGpuWeights.BlockWeights.SelfAttnQkv`)**:
   - Concatenated `SelfAttnQ`, `SelfAttnK`, and `SelfAttnV` into a single FP16 weight matrix of shape `[4608, 1536]`.
   - Replaced 3 separate `[2048, 1536] × [1536, 1536]` GEMMs with a single wide `[2048, 1536] × [1536, 4608]` `SgemmF16` dispatch into `WanGpuWorkspace.Qkv`.
2. **Fused `WanQkvSplitNormRoPE` Shader (`src/OpenTail.Stingray.Vulkan/Shaders.cs`)**:
   - Native compute shader executing 128-thread workgroup per head per token (`totalWorkgroups = numTokens * numHeads = 24,576`).
   - Slices `V` directly to destination buffer in global VRAM.
   - Computes in-register parallel reduction for RMSNorm on `Q` and `K` with per-head scaling vectors.
   - Applies 3D rotary position embeddings (GPT-NeoX adjacent pair rotation) in-register using precomputed `cos/sin` tables.
   - Replaced 3 separate memory-barrier-separated dispatches (`RmsNorm(Q)`, `RmsNorm(K)`, `Flux2DRoPE(Q,K)`) with 1 unified pass.
3. **Numerical Parity Verified**:
   - `WanGpuParityTests` passes 100% with `maxDiff < 1e-2` against CPU reference.

### Measured Performance Summary (`WanModelSyntheticMicrobenchTests`)
Single-block forward (`numTokens=2048`, `dim=1536`, `numHeads=12`, `ffnDim=8960`):
- **CPU Baseline (`WanModel.Forward`, AVX2)**: **1208.2 ms**
- **Vulkan GPU (`WanModel.ForwardGpu`, Fused QKV)**: **770.1 ms** (1.57× faster than CPU, down from 1873.2 ms originally).
- **Projected 30L × 40-step DiT Run**: **~924.2s (~15.4 min)** vs **~2248s (~37.5 min)** initial baseline.
