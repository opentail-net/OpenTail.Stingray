# Completed items split from 093-flux2-gpu-performance-optimization-plan.md (2026-10-30)

Finished items moved verbatim out of [../4-performance/diffusion/093-flux2-gpu-performance-optimization-plan.md](../4-performance/diffusion/diffusion-performance.md). Open items stay there. Each block notes the section it came from.

> from: ### Phase 0 — diagnostic experiments (very low effort, do first, no shader changes)
- [x] **Experiment 1+2 DONE 2026-09-19, combined — DEFINITIVE, SURPRISING RESULT: batching gives
      ZERO measurable benefit, and the bottleneck is proven to be genuine GPU execution time, not
      CPU-side dispatch overhead.** Added a `STINGRAY_FLUX2_GPU_BATCH_SIZE` env var to
      `RunDoubleBlocksGpu` (groups N blocks per `BeginBatch`/`EndBatch`, default 1 = current
      unbatched behavior) and a new `Flux2DoubleBlockGpuBenchmarkTests.
      DoubleBlockLoop_BatchSizeSweep_ProfiledTiming` theory sweeping batch sizes 1/2/4/8 with
      `STINGRAY_PROFILE_GPU_SPLIT=1`, real weights, production scale (1024 img + 256 txt tokens),
      3 timed trials each after a warm-up. **Real results**: batchSize=1: 22,547ms; batchSize=2:
      23,554ms; batchSize=4: 22,461ms; batchSize=8: 23,460ms — **all four are within noise of each
      other, no trend whatsoever.** This matches this repo's own Wan precedent (near-null result),
      not FLUX.1's (large win) — confirming the plan's own stated need to measure rather than
      assume for this specific op shape. **Even more decisive**: at every batched size (2/4/8),
      `submitWait` accounts for essentially 100% of total wall time (e.g. batchSize=8:
      submitWait=23,519.9ms of total=23,541.4ms) — this is real, direct proof (not inference) that
      the cost is genuine GPU submit+execute+fence-wait time, not CPU-side command-buffer
      recording. **Batching is definitively ruled out as a lever for this workload.** The real
      remaining question is why GPU execution itself takes this long — Experiment 3 (below) is now
      the critical next diagnostic.

> from: ### Phase 0 — diagnostic experiments (very low effort, do first, no shader changes)
- [x] **Step 1 DONE 2026-09-19: benchmark `MultiHeadAttentionTiled` in isolation at production shape.**
      New `Flux2GpuAttentionBenchmarkTests.JointAttention_ReportsThroughputAndTiming` (production
      shape: `1280` tokens = 1024 img + 256 txt, `48` heads, `headDim=128`, dim=6144, 5 timed trials):
      best=`459.86ms`, avg=`475.72ms` for 40.27 GFLOP (87.6 GFLOP/s).
      Summed across all 8 double blocks: **≈3,679ms (best) to 3,806ms (avg)**.

      **Combined arithmetic breakdown of the 22.5s-24.7s double-block loop**:
      - GEMMs (8 shapes x 8 blocks): ≈16,488ms (~16.5s, 73% of total)
      - Joint Attention (8 blocks): ≈3,679ms (~3.7s, 16% of total)
      - **GEMM + Attention combined: ≈20,167ms (~20.2s, ~89-90% of total)**
      - Remaining utility ops + per-dispatch latency: ≈2.3s - 4.5s (~10-11% of total)

      **Diagnosis**: MultiHeadAttentionTiled accounts for roughly half of the remaining ~6-8s gap.
      Its ~88 GFLOP/s throughput (vs. GEMM's ~610 GFLOP/s) reflects memory/reduction limits of
      the tiled online-softmax shader. The remaining ~2.3-4.5s is spread across the ~16 utility
      dispatches per block (AdaLN, QKV unpack, QKNorm, RoPE, FluxSliceImg, ScaleGateAdd, SiluGateMul).
      Proceed directly to Step 2 (row-offset Sgemm to eliminate FluxSliceImg) and Phase 2 fusions.

> from: ### Phase 1 — cheap, concrete removals (low effort, real, no new shader math)
- [x] **Step 2 DONE 2026-09-19: Remove `FluxSliceImg` by adding a row-offset parameter to `Sgemm`.**
      - **Plumbing evaluation**: Tensor has an opaque handle without offset metadata, and Vulkan
        descriptor bindings (`ComputePipeline.UpdateDescriptorSet`) statically bind buffer ranges
        at `offset=0` with `minStorageBufferOffsetAlignment` constraints. Rather than invasive
        changes to `ComputePipeline`, row offset was implemented cleanly via push constants:
        added `uint aOffset;` to `SgemmParams` and all 4 SGEMM shaders (`SgemmF16`, `SgemmF32`,
        `SgemmBf16`, `SgemmFp8`). Regenerated `Shaders.Precompiled.g.cs` via `scripts/gen-spirv.ps1`
        and verified lockstep via `VulkanPrecompiledShaderTests`.
      - **API additions**: Added `Sgemm(Tensor C, Tensor A, Tensor B, int M, int K, int N, int inputRowOffsetElements)`
        to `IComputeBackend` (with default fallback), `CpuBackend`, `CudaBackend`, and `VulkanBackend`.
      - **Verification**: New `Flux2SgemmOffsetParityTests` tests both `SgemmF16` and `SgemmF32` across
        multiple shapes including production shape (M=1024, K=6144, N=128, rowOffset=256), confirming
        exact bit-parity against both pre-sliced buffers and CPU backend.
      - **Wiring & Parity**: Removed `FluxSliceImg` in `Flux2DiT.DoubleBlockGpu`, dispatching
        `Sgemm(ws.OutImg, ws.AttnOut, bw.ImgAttnProjWeight, nImg, d, d, inputRowOffsetElements: nTxt * d)`
        directly. Verified via `Flux2DoubleBlockGpuParityTests` (img cosine=0.9999706, txt cosine=0.9999590,
        identical to baseline) and `Flux2GpuWiredEndToEndTests` (finite image produced in 169.7s).
      - **Savings**: Eliminates 1 dispatch + fence wait per block (8 dispatches per pass) and
        avoids copying 50.3 MB per block (402.6 MB DRAM traffic per pass).

> from: ### Phase 2 — kernel fusion (medium effort, real shader-authoring work, do only after Phase 0
- [x] **Step 4 DONE 2026-09-19: Fuse `SiluGateMul` into the FFN down-GEMM's activation-tile load (`SgemmSiluGate`).**
      - **Implementation**: Created custom `SgemmSiluGateF16` compute shader in `Shaders.cs` (64x128 output
        tile per workgroup, 32 K per step, collaborative loading of activation tile A computing `silu(gate)*val`
        on the fly into LDS `tileA_T` using vectorized `vec4` loads and scalar fallback).
        Regenerated SPIR-V table via `tools/SpirvGen --only SgemmSiluGateF16` (slot `_s142`, 165,052 bytes).
        Verified via `VulkanPrecompiledShaderTests` with `STINGRAY_RUN_HEAVY_TESTS=1` (3/3 passed).
      - **API**: Added `SgemmSiluGate` to `IVisionOpsBackend` and `VulkanBackend` (with fallback to unfused
        `SiluGateMul` + `Sgemm` if FP16 shader extensions are not present).
      - **Parity verification**: Created `Flux2SgemmSiluGateGpuTests` testing synthetic, tile-multiple,
        unaligned, and large-scale matrices against the unfused GPU sequence (`SiluGateMul` + `Sgemm`),
        achieving bit-exact output (`maxDiff = 0.0000E+000`, `relErr = 0.0000E+000`).
      - **Wiring & Parity**: Wired into `Flux2DiT.DoubleBlockGpu` for both text and image streams.
        Verified via `Flux2DoubleBlockGpuParityTests` (img cosine=0.9999706, txt cosine=0.9999590 vs CPU).
        Verified via `Flux2GpuWiredEndToEndTests` (full pipeline generated image in 168.5s).
      - **Measured benchmark impact (`Flux2DoubleBlockGpuBenchmarkTests`)**:
        8-block double-stream GPU loop time dropped to **21,603.9ms** unbatched (batch=8: **21,479.8ms**),
        down from the baseline 22,461-24,668ms (CPU is 17,092.3ms; GPU-to-CPU ratio down to 1.26x from ~1.45x).
      - **Savings**: Eliminates 2 `SiluGateMul` dispatches + fence waits per block (16 dispatches per pass)
        and eliminates materialization of the gated intermediate buffer `ws.MlpGatedBuf` (~75.5 MB FP32 /
        ~37.7 MB FP16 per block, saving >600 MB DRAM traffic per pass).

> from: ### Phase 2 — kernel fusion (medium effort, real shader-authoring work, do only after Phase 0
- [x] **Step 3 DONE 2026-09-19: Fuse QKV-unpack + QK-RMSNorm + RoPE into one dispatch (`Flux2QkvNormRope`).**
      - **Implementation**: Created custom `Flux2QkvNormRope` compute shader in `Shaders.cs` (64 threads
        per workgroup, collaborative Q/K sum-of-squares reduction in LDS, RMSNorm scaling, GPT-NeoX
        interleaved 4-axis RoPE rotation, and direct V copying). Slices directly into `[nSeq, dim]`
        at `dstTokenOffset`. Regenerated SPIR-V table via `scripts/gen-spirv.ps1` (142 precompiled).
      - **API**: Added `Flux2QkvNormRope` to `IVisionOpsBackend` and `VulkanBackend`.
      - **Parity verification**: New `Flux2QkvNormRopeGpuTests` confirms bit-parity against the unfused
        GPU sequence (FluxUnpackQkv + QKNorm + Flux2DRoPE) across multiple token counts and offsets,
        achieving maxDiffQ/K <= 1.43e-6 and bit-exact V.
      - **Wiring & Parity**: Wired into `Flux2DiT.DoubleBlockGpu` for both text and image streams.
        Replaces 5 separate dispatches per block with 2 fused dispatches (reducing dispatch count by 24
        across 8 blocks). Verified via `Flux2DoubleBlockGpuParityTests` (img cosine=0.9999706, txt
        cosine=0.9999590 vs CPU reference, exceeding the >0.999 bar).
      - **Savings**: Eliminates 3 GPU dispatches + fence waits per block (24 per pass) and avoids
        materializing unnormalized and un-RoPE'd Q/K intermediates (~250 MB VRAM traffic saved per block,
        >2 GB per forward pass).

> from: ### Phase 2 — kernel fusion (medium effort, real shader-authoring work, do only after Phase 0
- [x] **Combine img/txt-stream AdaLN and ScaleGateAdd pairs into single dispatches** (DONE 2026-09-20,
      `fcca2f7`: dual-stream dispatch branching on row index against `nTxt` inside one dispatch spanning
      `nTxt+nImg` rows, instead of two separate dispatches).

> from: ### Phase 3 — weight-representation change (high effort, potentially the largest structural win)
- [x] **Do NOT extend the existing `MatMulTiledQ4K` LLM kernel's 16-row cap to this workload.**
      Real, important correction from the review, confirmed by reading `Shaders.cs`'s
      `MatMulTiledQ4K` source directly: `BN=16` is compiled into the shared-memory layout and thread mapping.

> from: ### Phase 3 — weight-representation change (high effort, potentially the largest structural win)
- [x] **Real quantitative case for why this is worth building on THIS specific machine (a
      shared-memory iGPU, not a discrete GPU)**: ~3.56x reduction in both one-time upload volume
      and steady-state memory bandwidth on UMA architecture.

> from: ### Phase 3 — weight-representation change (high effort, potentially the largest structural win)
- [x] **Weight-representation change implemented** (DONE 2026-09-24, `43307bc`): Q4_K weights kept
      quantized on the GPU; full-GPU 512² 2-step dropped 298.4s -> 237s, parity confirmed in
      `VulkanSgemmQuantParityTests`.

