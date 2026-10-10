# Vulkan quantized GEMM: closing the gap to ggml-vulkan (plan, 2026-10-10)

Status: **proposed, not started** (for second opinion before any kernel work). Owner: the Qwen Image performance
thread; see [the diffusion samples and Qwen plan](2026-10-10-diffusion-samples-and-qwen-image-performance-plan.md)
for the measurements this builds on.

## 1. Goal and success criteria

Make Stingray's Vulkan dequant-in-shader GEMM family (`SgemmQ3K/Q4K/Q5K`, in `src/OpenTail.Stingray.Vulkan/Shaders.cs`)
as fast as the reference matmul in ggml-vulkan on the same GPU, then go further where the data allows.

| Gate | Metric (Qwen Image Q3_K_S, 512x512, one DiT forward, Vega 8 iGPU, Vulkan) | Today | Target |
|---|---|---|---|
| G1 | Forward time, CFG 2.5 workload | 24.4 s | at most 20.6 s (reference parity) |
| G2 | Same | 24.4 s | at most 17 s (stretch) |
| G3 | No other model regresses (FLUX.1/2, SD3.5, Wan, Z-Image, LLM Vulkan prefill) | n/a | speed within noise or better, outputs within tolerance |

A phase is kept only if it is measurably faster and numerically acceptable (CLAUDE.md rules 7, 11, 13). iGPU-only
timings describe this APU and driver; they say nothing definitive about a discrete GPU.

## 2. Evidence so far (all measured 2026-10-10, 5700G / Vega 8, wave64, 63 GB shared RAM, Vulkan)

- The DiT forward is GEMM-bound: of about 24 s per forward, MLP + norm 57%, QKV 19%, out-proj/gate 8%,
  QK-norm/concat/RoPE 8%, attention 6%, modulation 1%. Everything outside the 60 blocks is under 100 ms.
- Reference (`sd-cli --backend vulkan0`, same files and settings): 20.6 s per forward vs ours 24.4 s (about 18% slower);
  text encode and VAE are at parity.
- Ruled out by measurement: workspace/RoPE cache churn (about 40 ms per forward), per-block submit count, raster-order
  change, push-constant split-K inside the existing shader.
- Microbenchmark (Q3_K weights, M=1024): the existing kernel reaches about 970-1040 GFLOP/s at K=3072 but 670 at
  K=12288, and a K=12288 product run as four contiguous K=3072 chunks is 31% faster (112 ms to 77.6 ms). So cost
  depends on the activation row stride/footprint, not only on FLOPs. Splitting K inside the kernel did not reproduce
  the gain (the stride stays large), so a memory-layout or tile change is needed.
- **Small-M waste.** The text stream has 12 (cond) or 5 (uncond) tokens but goes through the 128-row tile: M=12 costs
  3.1 ms (QKV-shaped), 9.4 ms (MLP-up), 15.5 ms (MLP-down) against 0.3, 1.2 and 1.1 ms for one-row matvec. About 37 ms
  per block (about 2.2 s per forward) is spent on a stream that needs about 5 ms. 58 of 60 blocks are Q3_K, so the
  shapes are representative.

## 3. What the reference does differently (ggml-vulkan, `examples/stable-diffusion.cpp/ggml/src/ggml-vulkan/`)

Verified in the vendored source, not assumed:

1. **Tile selection by shape.** `ggml_vk_guess_matmul_pipeline` picks small (32x32 tile) when m or n is 32 or less,
   medium (64x64) when 64 or less, otherwise large (128x128). Non-coopmat large config:
   `{128 threads, BM 128, BN 128, BK 16 (32 for quant), WM 64, WN 64, WMITER 2, TM 4, TN 4}`. Ours is one fixed config:
   128x256 tile, 512 threads, 8x8 accumulators, about 48 KB LDS, so one workgroup per CU.
2. **FP16 shared-memory tiles and a dot-product inner loop** (`mul_mm.comp`: `FLOAT_TYPEV2 buf_a/buf_b`,
   `dot_product()` over K-pairs, FP32 accumulate), with spec constants for all tile parameters and an `ALIGNED` variant.
   Ours is FP32 LDS with an outer-product loop.
3. **Split-K with a real reduce shader** and a heuristic (`ggml_vk_guess_split_k`: K at least 2048 and tile count at most
   half the core count). Only helps when tiles under-fill the GPU; not our MLP-down case, but relevant elsewhere.
4. **Fusions:** RMS-norm + mul (+ RoPE) in one dispatch, add + RMS-norm (`multi_add`), bias/scale epilogues fused into
   mat-vec, GLU/GEGLU fused, top-k MoE routing fused, and a graph reorder pass (`GGML_VK_DISABLE_GRAPH_OPTIMIZE`).
5. **Attention:** `flash_attn` scalar path plus mask pre-reduction (`flash_attn_mask_opt`) and a split-KV reduce.
6. **Submission policy:** up to 100 nodes per submit (`GGML_VK_MAX_NODES_PER_SUBMIT`), async transfers, preallocated
   scratch, suballocation, and UMA-aware memory choices.
7. **Integer-dot (mmq) path:** quantises activations to Q8_1 and uses integer dot products. Needs hardware support
   (Vega 8 / gfx90c does not expose it; many RDNA/NVIDIA GPUs do), so it is a discrete-GPU lever, not for this machine.
8. **A per-op timing/FLOP logger** (`GGML_VK_PERF_LOGGER`), which is the model for our profiling.
9. **Implicit-GEMM conv (`conv2d_mm`)** with no im2col, relevant to VAE decode (about 9 s of the 2-step baseline).

## 4. Phases (each is a separate, reversible change with its own A/B)

### Phase 0: harness and safety net (no behaviour change)
- Move the throwaway GEMM microbenchmark into the repo (`tools/kernel-bench-cs` exists) with the Qwen, FLUX.2, Wan and
  Z-Image shapes (M, K, N) at the M values they actually use, for Q3_K, Q4_K, Q5_K and the F16 path.
- Per-op GPU timing via Vulkan timestamp queries (`timestampValidBits`, `timestampPeriod`), modelled on the ggml logger,
  so the per-group Qwen split stops depending on host synchronisation.
- Correctness gate: existing `VulkanSgemmQuantParityTests` plus a new shape-sweep parity test against the CPU
  dequantise-then-GEMM reference (relative error bound recorded per dtype).
- Enumerate every caller of the quantized GEMM (Qwen, FLUX.1/2, SD3.5/MMDiT, Wan, Z-Image, Vulkan LLM prefill) and give
  each a before-snapshot (speed and output).

### Phase 1: small-M kernel (expected about 1.9 s per forward, about 8%; lowest risk)
- A dedicated quantized kernel for M up to about 16 (single tile of rows per workgroup, weights decoded once, memory
  bound), selected by shape the way ggml selects its small tile. Candidates: extend the existing matvec to several rows,
  or a 32-row tile variant.
- Keep M=1 on the existing matvec. Gate on parity and on the benchmark table; the target is within 1.5x of weight-read
  bandwidth for M of 5 to 16.

### Phase 2: tile and layout fix for long K (expected 6-8%)
- Try, in order: (a) a 128x128 / 128-thread config with higher occupancy (2+ workgroups per CU), (b) the same tile with
  the K loop reordered, (c) the chunked-MLP layout (hidden activation kept as four contiguous 3072-wide chunks; weights
  re-laid out at upload; partial sums added). Measure each against the sweep table; keep the best.
- Decide split-K only if a config leaves tiles under-filling the 8 CUs.

### Phase 3: new tiled kernel family, FP16 shared tiles (the big-upside phase; uncertain)
- Parameterise the kernel with specialisation constants (BM, BN, BK, WM, WN, WMITER, TM, TN) like `mul_mm.comp`, with a
  dot-product inner loop and FP16 LDS (FP32 accumulate), plus an FP32-LDS variant selectable by environment switch as
  the rollback.
- Sweep configs on this machine for each Qwen shape and pick per shape; record the table.
- Success: GEMM throughput above 1.2 TFLOP/s on the large shapes, or phase abandoned (see stop rules).
- **Numerics decision (needs sign-off before merge):** FP16 activation tiles can overflow (range 65504) if the
  residual stream has outliers, and weights lose 5e-4 relative precision (small next to Q3_K's own quantisation noise).
  Validation: same-noise latent cosine and PSNR versus the current FP32 path and versus `sd-cli`; per CLAUDE.md the
  decision is recorded in code, tests, a decision note and the performance docs, with an environment-switch rollback.

### Phase 4: fusions and epilogues (each small, each measured)
- Bias add and GELU into the GEMM epilogue (the MLP group has two extra passes over a 1024x12288 buffer).
- QK-norm + concat + RoPE as one or two dispatches (that group is 8% of the forward).
- Attention: profile after the GEMM work; only act if it grows past about 10%.

### Phase 5: propagate and document
- Re-run FLUX.1, FLUX.2, SD3.5, Wan, Z-Image and Vulkan LLM prefill: speed and output checks versus the Phase 0 snapshot.
- Update `PerformanceLeague.md`, `STATUS.md` (only if a verified status changes), `RUNNING.md` timings, and the sample
  gallery README with the new Qwen numbers.

## 5. Expected outcome (honest range, to be replaced by measurements)

| After | Qwen forward | Notes |
|---|---|---|
| Today | 24.4 s | |
| Phase 1 | about 22.5 s | text-stream waste removed |
| Phase 2 | about 20.5-21 s | roughly reference parity (G1) |
| Phase 3 | 17-20 s if FP16 tiles pay off, otherwise unchanged | most uncertain |
| Phase 4 | a further 1-2 s | fusion savings |

A full 512x512, 20-step generation is 40 forwards: about 16.5 minutes today, about 11-13 minutes if G2 is reached. This
iGPU shares system memory with the CPU, so a large multiple is unlikely; the same kernels may do much better on a
discrete GPU, which this machine cannot show.

## 6. Risks and mitigations

- **Cross-model regression:** the kernel family is shared. Mitigation: Phase 0 snapshots, per-phase environment-switch
  rollback (`STINGRAY_VK_QGEMM=legacy`), and no merge without Phase 5-style checks.
- **Driver behaviour:** AMD Windows drivers and FP16 shared memory / packed math can differ from expectations.
  Mitigation: prototype in the standalone benchmark first; never trust an iGPU result as a general claim.
- **Precision:** see the numerics decision in Phase 3.
- **Shader table drift:** shaders are precompiled; every change needs `scripts/gen-spirv.ps1` and the drift test.
- **Effort sink:** Phase 3 is open-ended. Mitigation below.

## 7. Stop rules

- Each phase has a measured gate; a phase that does not beat the previous best by a margin larger than run-to-run noise
  (about 2-3% on the forward) is reverted.
- Abandon Phase 3 if a prototype FP16-tile kernel is not at least 15% faster than the best Phase 2 kernel on the
  large Qwen shapes in the standalone benchmark.
- Stop the whole effort at reference parity (G1) unless the user asks for the stretch target.

## 8. Questions for the second opinion

1. Is the Phase order right (small-M, then stride/tile, then FP16 tiles), or is there a cheaper route to the long-K
   throughput loss than chunked layouts? What would explain a stride-dependent loss on a UMA iGPU (TLB reach, memory
   channel interleaving, L2 line conflicts)? Which experiment separates them?
2. Does Vega 8 (gfx90c) give FP16 shared tiles a real benefit when accumulating in FP32 (no dot2/packed-FP16 FMA
   with FP32 accumulate)? ggml has a `GGML_VK_DISABLE_DOT2` switch: what does it key on?
3. Is a K-contiguous dot-product inner loop (ggml) better than the outer-product register tile for LDS bandwidth on GCN?
4. Anything in the list of ggml techniques in section 3 that should be higher priority, or something missing?
5. Is FP16 activation overflow a realistic risk for MM-DiT residual streams, and what is the cheapest guard?
6. Reasonable stop criteria for a kernel project on a non-representative GPU.
