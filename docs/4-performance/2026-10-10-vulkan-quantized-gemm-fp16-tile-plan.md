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
- **Establish the baseline properly.** The 24.4 s and 20.6 s figures come from single runs (an earlier uninstrumented
  run took 138.2 s against 115.4 s instrumented), so first record the median and spread of at least five
  uninstrumented runs of both Stingray and the reference, keeping one-time weight upload separate from steady-state
  forward time, and keep instrumented numbers out of the comparison.
- **Record the environment with every benchmark row:** GPU name, driver and ICD, subgroup size, the real
  `maxComputeSharedMemorySize` (the reference's device line reports 32768 while our kernel declares about 48 KiB of
  shared arrays, so query it directly rather than assume), FP16 and storage-16 support, whether
  `VK_VALVE_shader_mixed_float_dot_product` (the dot2 path) and integer-dot are exposed, model file hashes, dimensions,
  seed, CFG and the reference build.
- **Find out which shader variant the reference really runs on this GPU.** Its device line (2026-10-10) says
  `uma 1, fp16 1, bf16 0, warp size 64, shared memory 32768, int dot 0, matrix cores none`. Use ggml's pipeline logging
  (`GGML_VK_PERF_LOGGER`, `VK_KHR_pipeline_executable_properties`, or its debug log) to see whether it picks the FP16
  matmul, the dot2 variant or the FP32 one, and which tile (s/m/l) each Qwen shape uses. The "FP16 tiles explain the
  gap" idea stays a hypothesis until that is known.
- Move the throwaway GEMM microbenchmark into the repo (`tools/kernel-bench-cs` exists) with the Qwen, FLUX.2, Wan and
  Z-Image shapes (M, K, N) at the M values they actually use, for Q3_K, Q4_K, Q5_K and the F16 path.
- Per-op GPU timing via Vulkan timestamp queries (`timestampValidBits`, `timestampPeriod`), modelled on the ggml logger,
  so the per-group Qwen split stops depending on host synchronisation.
- Correctness gate: existing `VulkanSgemmQuantParityTests` plus a new shape-sweep parity test against the CPU
  dequantise-then-GEMM reference (relative error bound recorded per dtype).
- Enumerate every caller of the quantized GEMM (Qwen, FLUX.1/2, SD3.5/MMDiT, Wan, Z-Image, Vulkan LLM prefill) and give
  each a before-snapshot (speed and output).

### Phase 1: small-M kernel (provisional estimate: about 1.9 s per forward, about 8%; lowest risk)
The saving is an estimate to test, not an expected result: it compares today's M=12 GEMM time with one-row matvec
time, and a one-row matvec is not a prediction for a kernel that handles several rows while reusing decoded weights
(twelve separate matvecs would re-read the weights twelve times).
- A dedicated quantized kernel for small M (a single tile of rows per workgroup, weights decoded once, memory bound),
  selected by shape the way ggml selects its small tile. Candidates: extend the existing matvec to several rows, or a
  32-row tile variant.
- Benchmark current GEMM, a one-row-matvec loop and each prototype across **M = 1, 2, 4, 5, 8, 12, 16, 32, 64, 128** on
  the real Qwen shapes (K=3072 and K=12288 included). The comparison that matters is the whole M-row product.
- Keep M=1 on the existing matvec. Success is defined by effective bandwidth: quantised-weight bytes read divided by
  measured kernel time, compared with a separately measured sustained-memory-bandwidth benchmark on the same device
  (target: a stated fraction of that ceiling, fixed after the ceiling is measured). Gate on parity and on end-to-end
  forward time.

### Phase 2: tile and layout fix for long K (expected 6-8%)
- Try, in order, isolating the simple reversible changes before the invasive one: (a) a 128x128 / 128-thread config
  with higher occupancy (2+ workgroups per CU); (b) a K-loop-order change on its own; (c) the chunked-MLP layout.
- (c) must pay its whole cost. The 112 ms to 77.6 ms result is an isolated GEMM measurement. The production path has to
  produce the contiguous activation chunks (ideally by making the MLP-up projection write that layout directly, or by
  fusing the rearrangement into it) and has to add the partial outputs. If a copy or a separate add is needed, benchmark
  it and include it; do not treat packing as free. Weights are re-laid out once at upload, so that part is not a
  per-forward cost.
- Measure each against the sweep table and the end-to-end forward; keep the best.
- Decide split-K only if a config leaves tiles under-filling the 8 CUs.

### Phase 3: new tiled kernel family (the big-upside phase; uncertain)
Two different ideas were bundled in the first draft and are now separate experiments, because FP16 shared storage and
FP16-pair dot products are different optimisations and the benefit of either depends on the generated shader, the
device features and compiler behaviour (vectorisation, conversion overhead and occupancy all matter).
- Parameterise the kernel with specialisation constants (BM, BN, BK, WM, WN, WMITER, TM, TN), like `mul_mm.comp`, so the
  sweep is data, not code.

| Experiment | Implementation | What it tests |
|---|---|---|
| A | FP32 shared tiles, FP32 accumulate | baseline (today's arithmetic, new tile parameters) |
| B | FP16 shared tiles, convert to FP32 for the multiply and accumulate | whether narrower shared tiles improve occupancy or data movement |
| C | FP16 shared tiles plus FP16-pair dot products with FP32 accumulate, only where the device exposes it | whether the arithmetic path adds a further gain |

- FP16 accumulation is out of scope: it is a separate and riskier numerical change.
- For each variant record resource usage and, where tooling permits, register pressure and occupancy, and use the same
  operation-count convention and methodology as the baseline so GFLOP/s figures are comparable.
- Keep an FP32-LDS variant selectable by an environment switch as the rollback.
- Sweep configs on this machine for each Qwen shape, pick per shape, and record the table.

**Numerical gate for any FP16 variant (needs sign-off before merge).** The FP32 parity tests use synthetic activations in
[-1, 1], which does not show how FP16 behaves on real Qwen Image activations, and an FP16 path should not be expected
to meet the FP32 tolerance automatically. So:
- Measure first: actual activation ranges, any non-finite values, and error growth at several layers and timesteps, on
  real weights with captured real activations. Do not add a clamp pre-emptively (it would itself change the maths);
  overflow past 65,504 is an unproven risk, and rounding drift across 60 blocks matters more to investigate.
- GEMM output relative-L2 and maximum absolute error against the existing FP32 quantised path, on real data.
- A one-block comparison with identical inputs, then a full-model same-noise generation.
- A documented, evidence-based acceptance threshold (latent cosine and PSNR) against both the current Stingray path and
  the reference, recorded in code, tests, a decision note and the performance docs, with the switch above as rollback.

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

- Each phase has a measured gate; a phase that does not beat the previous best by more than run-to-run noise
  (about 2-3% on the forward) is reverted.
- **One gate for Phase 3** (replaces the earlier two that could disagree): keep it only if it is (1) at least 15% faster
  than the best Phase 2 kernel on the target large shapes in the benchmark, **and** (2) measurably faster on the
  end-to-end Qwen forward, not just the microbenchmark, **and** (3) numerically accepted on the real model workload, **and**
  (4) shows no material regression in the other representative models. Otherwise abandon it. Throughput above
  1.2 TFLOP/s is reported, not gated.
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

## 9. Second opinion received (ChatGPT, 2026-10-10: 8/10, keep the plan and revise in place)

Incorporated above: (1) Phase 0 now establishes a repeatable median baseline, records the environment and finds the
reference's actual shader variant on this GPU before the FP16 explanation is trusted; (2) the Phase 1 saving is marked
provisional and benchmarked across M = 1 to 128 with a defined bandwidth success criterion; (3) the chunked-layout
experiment must include activation-layout production and partial-sum costs and comes after the simpler tile and loop
changes; (4) FP16 shared storage and FP16-pair dot products are separate experiments, FP16 accumulation stays out of
scope, and the numerical gate uses real activations and measures ranges before any guard is added; (5) Phase 3 has a
single stop gate. Its answers to section 8: the phase order is right; FP16 benefit on Vega 8 is plausible but not
established by FP16 support alone (dot2 is a separate extension from integer-dot); a K-contiguous dot loop is not
assumed better; shape-specific tiling and the small-M path are the immediate priorities; integer-dot is not the first
target for this device; FP16 overflow is unproven and must be measured, not guarded pre-emptively; the stop criteria
are reasonable once unified. Not expanding into attention or unrelated fusions while the GEMM work is unmeasured.

## 10. Progress log

### Phase 0 (partial) and reference data, 2026-10-10
`GGML_VK_PERF_LOGGER=1` on `sd-cli` (same GPU, one forward, 512x512). Reference device line: `uma 1, fp16 1, bf16 0, warp size
64, shared memory 32768, int dot 0, matrix cores none`. Reference GPU time per call (its m = our N, its n = our M):

| Shape (our M x K x N) | Reference | Stingray before Phase 1 | Stingray after Phase 1 |
|---|---|---|---|
| 1024 x 3072 x 12288 (MLP-up) | 65.0 ms (1189 GFLOP/s) | about 79 ms | unchanged (large kernel) |
| 1024 x 12288 x 3072 (MLP-down) | 110.5 ms (699 GFLOP/s) | 96-116 ms | unchanged |
| 1024 x 3072 x 3072 (QKV, out) | 17.7 ms (1094 GFLOP/s) | 18.6-19.4 ms | unchanged |
| 12 x 3072 x 12288 | 2.70 ms | 9.4 ms | 3.09 ms |
| 12 x 12288 x 3072 | 3.29 ms | 15.5 ms | 7.77 ms |
| 12 x 3072 x 3072 | 0.74 ms | 3.1 ms | 0.84 ms |

The reference shows the same long-K weakness (699 GFLOP/s at K=12288), so that behaviour is hardware or memory related,
not specific to our kernel, and the chunked layout could beat the reference there. Still open for Phase 0: five-run median
baselines, the real `maxComputeSharedMemorySize`, and which pipeline variant ggml selected (the logger names the op and
shape but not the shader variant).

### Phase 1 (small-M kernel), 2026-10-10: kept
- New kernels `SgemmQ3KSmall / Q4KSmall / Q5KSmall` (`Shaders.cs`): same decode and K loop as the large kernels, workgroup
  tile 16 activation rows x 64 weight rows, 128 threads (8x16), 2x4 register tile per thread. Used when M is at most
  `STINGRAY_VK_QGEMM_SMALLM` (default 32; 0 disables). M=1 stays on the matvec. First attempt used an 8x8 per-thread tile in a
  2x8 workgroup, which left only 16 threads for a loader written for 128: NaN output, caught by the parity check, rewritten.
- Correctness: output is **bit-identical** to the large kernel for M = 2..32 on all three Qwen shapes (maxAbsDiff 0), the
  full-model image is byte-identical to the old path, and the new `VulkanSgemmQuantSmallMTests` (Q3_K/Q4_K/Q5_K, M = 2..33,
  long K, N not a tile multiple) plus the existing quant parity and shader-drift tests pass.
- Speed: kernel at M=12 is 2.3-3.7x faster than before (table above), not yet at reference parity for MLP-down (7.8 vs
  3.3 ms). In the real forward (512x512, GPU, alternating A/B, block time per forward): old 24.95 / 24.91 s, new 23.58 /
  22.43 s, so about 1.4-2.5 s (6-10%), consistent with the 1.3-1.7 s predicted. Run-to-run drift between rounds is about 5%,
  so the exact figure still needs the plan's five-run median.
- Follow-up candidates for M-small MLP-down (K=12288): a smaller BN for more workgroups, a larger BK to halve barrier
  overhead, or split-K with a reduce; worth about 0.2 s per forward, so low priority.

### Phase 2(a) 128x128 tile with 256 threads, 2026-10-10: rejected, reverted
Derived mechanically from the large kernel (workgroup 16x16, 128 activation rows x 128 weight rows, 32 KiB shared tiles, two
workgroups per CU). Output checksums bit-identical to the existing kernel. Standalone benchmark, Q3_K, M=1024: MLP-up
78.3 -> 93.9 ms (worse), MLP-down 111.6 -> 105.4 ms (-6%), QKV/out 18.5 -> 19.6 ms (worse). Used only for K=12288 it would save
about 0.36 s per forward (1.5%), below the plan's 2-3% noise floor, so it was not kept. Occupancy is therefore not what
limits MLP-up; the reference's 65 ms there likely comes from its inner-loop formulation, which stays a Phase 3 question.

### Phase 2(c) chunked image-stream MLP, 2026-10-10: kept
- Image-stream MLP weights are stored as 4 up-projection row slices and 4 down-projection K slices (3072 wide), uploaded as
  raw quantized tensors (down slices re-gathered per row at upload), and the hidden activation lives in four 3072-wide buffers
  (`QwenImageGpuWeights.TryUploadMlpChunks`, `QwenImageGpuWorkspace.ImgMlpChunks`, `QwenImageModel.RunBlockGpu`). Up
  slices each get their bias slice and GELU; the down projection accumulates four partial products with the existing
  `AddScaledInPlace`. `STINGRAY_QWEN_MLP_CHUNKS=0` restores the single-GEMM MLP; blocks whose MLP weights are not raw K-quants
  of the expected shape fall back automatically.
- Standalone benchmark (Q3_K, M=1024), whole MLP per block: 188.6 ms single GEMMs -> 156.3 ms chunked including the three
  adds (3.0 ms), saving 32.3 ms per block (1.87 s per forward).
- Real forward (512x512, GPU, alternating A/B, block time per forward): single 23.15 / 23.20 s, chunked 21.28 / 21.92 s
  (-1.3 to -1.9 s, 5.5-8%). One-time cost: weight upload about 0.7 s longer (6.2 s vs 5.4 s) for the re-gather.
- Correctness: the output is not bit-identical (the K sum is accumulated in four partial sums), so the gate is numerical:
  same-noise 512x512 image PSNR 79.1 dB with a maximum channel difference of 1 of 255 (631 of 262,144 pixels differ);
  `QwenImageSdCppParityTests` velocity cosine 0.999676 / relative L2 0.0255 and `QwenImageGpuParityTests` cosine 0.996550 /
  max diff 0.3071 are unchanged to the printed precision with the switch on or off.
