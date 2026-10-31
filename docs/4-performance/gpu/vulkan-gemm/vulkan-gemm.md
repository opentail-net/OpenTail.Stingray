# Vulkan quantized GEMM: owner file

**Owner:** this file (2026-10-30). It replaces two separate docs, kept below verbatim as parts:

1. **Part 1, Path 1 / Path 2 scaffold** (was `gpu/vulkan-gemm/README.md`). **This part describes an older state.** Path 3, the tiled quantised GEMM, has been the default since 2026-10-11 (see [../../../workstreams/vulkan-gemm-llm-prefill.md](vulkan-gemm.md), which owns the LLM prefill thread). Read Part 1 for its measurements and reasoning, not for the current default.
2. **Part 2, FP16-tile project** (was `2026-10-10-vulkan-quantized-gemm-fp16-tile-plan.md`). It is the diffusion-side GEMM project, and it carries the numerics decision.

Not included: the CPU expert-GEMM plan ([../../2026-10-04-q4k-expert-gemm-plan.md](../../2026-10-04-q4k-expert-gemm-plan.md)). It is about CPU MoE prefill, not Vulkan, so it keeps its own file.

The measurement rules are in [../../../reference/performance-measurement-rules.md](../../../reference/performance-measurement-rules.md).

---

## Part 1: Path 1 / Path 2 scaffold (older state)

## Vulkan quantized matmul — Path 1 / Path 2 scaffold

> Completed items moved to [done/vulkan-gemm-README-completed-items-2026-10.md](../../../done/vulkan-gemm-README-completed-items-2026-10.md) on 2026-10-30. This file keeps only open work.

The same shape as `docs/repack-gemm/` on the CPU side: an incumbent (Path 1), a reserved slot for a
faster future kernel (Path 2), a runtime switch between them, and counters so the choice is decided
by measurement rather than argument.

**Path 2 does not exist.** This document exists so the decision to write it — or to delete the
scaffold and walk away — can be made from numbers that are available today.

### Why

Every quantized matmul shader in `Shaders.cs` is a `MatVec*`. The five `Sgemm*` shaders are dense
float, serving the diffusion path. So the quantized matmul is matrix-**vector** with register
blocking over tokens: one workgroup per 8 output rows, `acc[MAX_NTOK]` accumulators in VGPRs, a
weight word reused across up to `MaxBatchVerifyK` = 16 tokens. Above 16 tokens `GpuForwardPass`
chunks, and every chunk re-streams the whole model.

This is structurally what the CPU prefill was before Path 2 landed there. The CPU fix was not a
wider register tile — it was another level of the memory hierarchy, a repacked tile that stayed
resident while activation columns streamed past. The GPU equivalent is a weight tile in shared
memory: what llama.cpp calls `mul_mm` as distinct from `mul_mv`.

### Actionable items checklist

- [x] **Tiled Q6_K shader**: Implement tiled Q6_K kernel for `ffn_down` (1418 of 9914 dispatches) to remove fallback to Path 1. (Done: `SgemmQ6K` exists and `TryMatMulBatchedPath2` accepts Q6_K; this checkbox was stale, checked 2026-10-31.)
- [ ] **Widen tile & prefill chunk**: Raise BN > 16 (chunk 32 / chunk 64) once Q6_K is supported, chasing the projected 1.36x ceiling.
- [ ] **Perplexity gate**: Run Wikitext-2 (8191 scored tokens) validation on Path 2 before considering making it default.

### Using it

| variable | effect |
|---|---|
| `STINGRAY_VULKAN_MM_PATH` | `3` (default since 2026-10-11), `1` (old multi-token mat-vec) or `2`. Path 2 serves Q4_K and Q6_K up to `TiledQ4KBn` tokens and declines other shapes, which fall through to Path 1 (corrected 2026-10-31, `VulkanBackend.cs` 3177-3183). |
| `STINGRAY_VULKAN_MM_STATS` | `1` to count dispatches and weight bytes; off by default so an unmeasured run is byte-identical to before. |

```
STINGRAY_VULKAN_MM_STATS=1 opentail-llm-cli -m model.gguf --backend vulkan -g -1 -f prompt.txt
```

Code: `src/OpenTail.Stingray.Vulkan/VulkanMatMulPath.cs` (switch + counters), the seam and
`TryMatMulBatchedPath2` stub in `VulkanBackend.MatMulBatched`, tests in
`tests/.../VulkanMatMulPathTests.cs` (27, no GPU required).

### Baseline, measured

AMD Radeon integrated, Vulkan 1.3, SmolLM2-1.7B-Instruct-Q4_K_M (1006.7 MiB), 931-token prompt:

```
[GpuForwardPass] prefill N=931 chunk=16
[VulkanMatMul] path=Path1
  dispatches : path1=9914 path2=0 declines=0 fallback=0
  amortization: 15.78 tokens/dispatch (156416 token-dispatches over 9914 dispatches)
  weight I/O : 54626.6 MiB total, 58.7 MiB/token over 931 tokens
Prefill: 931 tokens, 74.5 t/s
```

Three cross-checks that the counters measure what they claim:

- 9914 dispatches / 59 chunks ≈ 168 = 24 layers × 7 trunk matmuls. Structure matches the model.
- 54626.6 MiB / 59 chunks = **925.9 MiB per chunk** against a **1006.7 MiB** model file. The
  difference is embeddings and output, which are not trunk matmuls. Weight I/O is real.
- `fallback=0` — no dtype fell to the per-token loop on this model.

Setting `STINGRAY_VULKAN_MM_PATH=2` reports `declines=9914`, identical weight I/O, and 74.7 t/s
against 74.5. The seam costs nothing and cannot change an answer.

### The prize is 1.54x, not 59x — measured

53.3 GiB streamed to prefill 931 tokens of a 1 GiB model looks like a 59x redundancy, and that was
the number this scaffold was built to chase. It is the wrong number, and the chunk-size split makes
the right one measurable without writing a single line of GLSL.

Chunk size is inversely proportional to how many times prefill streams the weights, so sweeping it
is a direct experiment on how much of prefill time is weight bandwidth. Same prompt, same model,
`STINGRAY_VULKAN_PREFILL_CHUNK`:

| chunk | prefill t/s | weight GiB | time (s) |
|---:|---:|---:|---:|
| 1 | 12.2 | 841.8 | 76.31 |
| 2 | 22.7 | 421.3 | 41.01 |
| 4 | 37.9 | 210.7 | 24.56 |
| 8 | 58.4 | 105.8 | 15.94 |
| 16 | 75.3 | 53.3 | 12.36 |

Two-point fit on the extremes, checked against the three interior points (all within 4.3%):

```
time(s) ≈ weight_GiB / 12.33 + 8.04
```

Weight streaming costs **12.33 GiB/s marginal**, and there is a **fixed 8.04 s** that no amount of
weight amortization touches. At the default chunk of 16 that splits as **4.3 s of weights against
8.0 s of everything else** — so weights are 35% of prefill, not the whole of it.

Therefore, with weight traffic driven to zero:

| chunk | projected t/s | vs chunk 16 |
|---:|---:|---:|
| 32 | 91 | 1.21x |
| 64 | 102 | 1.36x |
| 128 | 108 | 1.44x |
| ∞ (perfect GEMM) | **116** | **1.54x** |

**1.54x is the ceiling**, assuming a tiled kernel adds no cost of its own and leaves the fixed 8.04 s
untouched. Most of it arrives by chunk 32–64; the tail is worth little. A real kernel lands below
this.

#### What the fixed 8.04 s is, and why it matters more

Activation traffic is `(rows / N_ROWS) × nTok × cols × 4` bytes per dispatch — each workgroup covers
8 rows and re-reads the whole activation block. The dispatch count falls as `1/chunk` while per-
dispatch activation traffic rises linearly with it, so **total activation traffic is chunk-
independent**, which is exactly why it sits in the fixed term and not the slope. That it fits the
constant so cleanly is corroboration, not coincidence.

So a Path 2 that only widens the tile chases the smaller half. A Path 2 that stages activations in
shared memory attacks the larger one. Any tiled kernel written here should be designed against the
8.04 s, not the 53.3 GiB.

### Path 2 exists: tiled Q4_K GEMM, +4.6%

`Shaders.MatMulTiledQ4K` — BM=64 rows x BN=16 tokens, BK=64, TM=TN=2, both operands staged in LDS
with llama.cpp's `+1` bank-conflict padding. BK=64 is one Q4_K "c-chunk" (8 dwords whose low nibbles
are sub-block 2c and high nibbles 2c+1), so the dequant is structurally identical to the incumbent's
and the two differ in one variable only.

| | Path 1 | Path 2 |
|---|---|---|
| prefill, 931 tok | 74.5 t/s | **77.9 t/s** (1.046x) |
| dispatches served | 9914 | 8496 (+1418 Q6_K declines → Path 1) |
| weight I/O | 54626.6 MiB | 54626.6 MiB — *unchanged* |

Weight traffic is identical, so the gain is purely structural: no per-token shared-memory tree
reduction (Path 1 runs 16 of them, 7 barriers each, per workgroup), and each staged value is reused
across the tile instead of re-read per lane. Full suite passes on **both** paths, 1261/1261.

#### Why it is only 4.6% of a possible 54%

BN is still 16, so the tile amortizes exactly as many tokens as Path 1 did and streams the weights
the same 59 times. **The prize needs BN > 16, and that is now a `#define` rather than a register
array** — the `acc[MAX_NTOK]`-per-VGPR ceiling that capped Path 1 does not exist here.

The blocker is Q6_K. `ffn_down` is Q6_K (1418 of 9914 dispatches, 1 per layer), it declines to Path 1,
and Path 1 throws above nTok=16. So raising the chunk requires a **tiled Q6_K shader first** — that
is the next commit, and it unlocks chunk 64 for a projected 1.36x.

#### Numerics: Path 2 is not a drop-in

Path 2 carries one running sum per output across K; Path 1 forms per-lane partials then a tree
reduction. Same additions, different association. Measured **9.4e-3 max abs logit delta against the
per-token path, versus 5e-3 for Path 1** — it broke `BatchedPrefill_ShortPromptBelowOneChunk_
MatchesPerToken` on first run.

The tolerance is now path-aware rather than loosened, so Path 1's tighter guarantee is preserved and
Path 2's looser one is stated with its reason. Greedy argmax matches on both.

**Path 2 must not become the default on a 4.6% speed win alone.** A delta at this scale is inside the
noise that flipped 2 of 6 short-prompt argmaxes when int8 activations were tried. It needs the same
perplexity gate the CPU's Path 2 cleared (wikitext-2, 8191 scored tokens), which has not been run for
this kernel.

### Toolchain: unblocked

`glslc` built from source at `examples/glslc/shaderc` — **5,568,512 bytes**, shaderc v2026.4-dev /
glslang 11.1.0-1516 / spirv-tools v2026.3.

**Regenerating the whole SPIR-V table drifts ~33 of 89 shaders** against the committed bytes, because
that toolchain is much newer than whatever produced them. The regenerated table passes all 1261 tests
and is performance-neutral (74.9/75.5 vs 74.5/75.3 t/s) — but that is one AMD iGPU, so it is not
evidence for other drivers, and it was restored rather than adopted.

New shaders therefore splice in instead: bump `Count`, add one hash-keyed `case` line, append one
`_sNN` array. The switch does not care about ordering. `VulkanPrecompiledShaderTests` confirms the
other 89 entries stay byte-identical.

### Previously: blocked on a toolchain, not on effort

**This machine has no Vulkan SDK and no `glslc`.** `ShaderCompiler` serves a committed SPIR-V table
and shells out to `glslc` only on a table miss, and `tools/SpirvGen` needs the SDK to regenerate it —
so a new or edited shader cannot be compiled here at all, and `VulkanPrecompiledShaderTests` fails on
drift by design. No Path 2 kernel was written, because an unverifiable one is worse than none.

This is the same shape as the CUDA finding in `../done/gpu-review-log.md`: a real defect, a cheap fix,
blocked on a toolchain rather than hardware. **Installing the Vulkan SDK unblocks it** — no different
GPU required.

The cheapest first increment, when unblocked: `MatVecBatchedQ4K` and `MatVecBatchedQ6K` with
`MAX_NTOK` at 32, gated behind Path 2, `Path2MaxTokensPerDispatch` raised to match, and the `nTok`
range check in `MatMulBatched` relaxed. Projected 1.21x, and it tests the register-pressure question
the existing shader comment leaves open ("32 was not tried") for a few lines of change. Note
`Path2Cap_CannotExceed_WhatPath1CanFallBackTo` will fail until the Path 1 shaders are widened too —
that is deliberate: Path 2 may decline any shape, and a decline falls through to a Path 1 that would
otherwise throw on the wider chunk.

**To drop the whole thing:** delete `VulkanMatMulPath.cs`, the seam and stub in `VulkanBackend.cs`,
the report block in `GpuForwardPass.Prefill`, the two test files, and three entries in
`KnownEnvironmentVariables`. The `MaxBatchVerifyK` / `MaxPrefillChunk` split is worth keeping
regardless — it is what made the sweep above possible.

### Negative result: the metric was wrong first

The first version divided weight bytes by the summed `nTok` and reported **357.6 KiB/token**. The
true figure is **58.7 MiB/token** — off by 168x, the number of matmuls in the model, because summed
`nTok` counts token-*dispatches* and not tokens.

It was caught by running the counters on hardware and checking the total against an independently
computed model size, before any conclusion was drawn from them. `WeightBytesPerToken` now requires
the caller to pass the token count, `Report(0)` omits the line rather than fabricating it, and
`BytesPerToken_IsNotDerivableFromTokenDispatches` pins the distinction.

The headline metric was changed to **tokens/dispatch** for the same reason: it needs no external
input, so it cannot be mis-attributed this way at all.

---

## Part 2: FP16-tile project (diffusion)

## Vulkan quantized GEMM: closing the gap to ggml-vulkan (plan, 2026-10-10)

### Checklist (2026-10-30; checked against the progress log)

Note: the status line above ("proposed, not started") is stale; the progress log below records work done on 2026-10-10.

- [ ] Phase 0: harness and safety net (partial; the reference data is recorded)
- [x] Phase 1: small-M kernel (kept, 2026-10-10)
- [x] Phase 2(c): chunked image-stream MLP (kept, 2026-10-10; about 2.6 s, 10.7%, faster)
- [x] Phase 2(a): 128x128 tile with 256 threads (rejected and reverted, 2026-10-10)
- [x] Phase 2 MLP-up follow-ups: stopped, nothing cheap left (2026-10-10)
- [ ] Five-run median of the Phase 1 + 2(c) checkpoint (outstanding per the log)
- [ ] Phase 3: new tiled kernel family (the big-upside phase; needs its numerics decision)
- [ ] Phase 4: fusions and epilogues
- [ ] Phase 5: propagate and document

Status (corrected 2026-10-30): Phase 1 (small-M kernel) and Phase 2(c) (chunked MLP) kept on 2026-10-10; Phase 2(a) rejected; Phase 3 and later not started. The progress log below is the record. Originally: proposed, not started (for second opinion before any kernel work). Owner: the Qwen Image performance
thread; see [the diffusion samples and Qwen plan](../../diffusion/diffusion-performance.md#part-4-qwen-image-and-the-image-samples)
for the measurements this builds on.

### 1. Goal and success criteria

Make Stingray's Vulkan dequant-in-shader GEMM family (`SgemmQ3K/Q4K/Q5K`, in `src/OpenTail.Stingray.Vulkan/Shaders.cs`)
as fast as the reference matmul in ggml-vulkan on the same GPU, then go further where the data allows.

| Gate | Metric (Qwen Image Q3_K_S, 512x512, one DiT forward, Vega 8 iGPU, Vulkan) | Today | Target |
|---|---|---|---|
| G1 | Forward time, CFG 2.5 workload | 24.4 s | at most 20.6 s (reference parity) |
| G2 | Same | 24.4 s | at most 17 s (stretch) |
| G3 | No other model regresses (FLUX.1/2, SD3.5, Wan, Z-Image, LLM Vulkan prefill) | n/a | speed within noise or better, outputs within tolerance |

A phase is kept only if it is measurably faster and numerically acceptable (CLAUDE.md rules 7, 11, 13). iGPU-only
timings describe this APU and driver; they say nothing definitive about a discrete GPU.

### 2. Evidence so far (all measured 2026-10-10, 5700G / Vega 8, wave64, 63 GB shared RAM, Vulkan)

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

### 3. What the reference does differently (ggml-vulkan, `examples/stable-diffusion.cpp/ggml/src/ggml-vulkan/`)

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

### 4. Phases (each is a separate, reversible change with its own A/B)

#### Phase 0: harness and safety net (no behaviour change)
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

#### Phase 1: small-M kernel (provisional estimate: about 1.9 s per forward, about 8%; lowest risk)
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

#### Phase 2: tile and layout fix for long K (expected 6-8%)
- Try, in order, isolating the simple reversible changes before the invasive one: (a) a 128x128 / 128-thread config
  with higher occupancy (2+ workgroups per CU); (b) a K-loop-order change on its own; (c) the chunked-MLP layout.
- (c) must pay its whole cost. The 112 ms to 77.6 ms result is an isolated GEMM measurement. The production path has to
  produce the contiguous activation chunks (ideally by making the MLP-up projection write that layout directly, or by
  fusing the rearrangement into it) and has to add the partial outputs. If a copy or a separate add is needed, benchmark
  it and include it; do not treat packing as free. Weights are re-laid out once at upload, so that part is not a
  per-forward cost.
- Measure each against the sweep table and the end-to-end forward; keep the best.
- Decide split-K only if a config leaves tiles under-filling the 8 CUs.

#### Phase 3: new tiled kernel family (the big-upside phase; uncertain)
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

#### Phase 4: fusions and epilogues (each small, each measured)
- Bias add and GELU into the GEMM epilogue (the MLP group has two extra passes over a 1024x12288 buffer).
- QK-norm + concat + RoPE as one or two dispatches (that group is 8% of the forward).
- Attention: profile after the GEMM work; only act if it grows past about 10%.

#### Phase 5: propagate and document
- Re-run FLUX.1, FLUX.2, SD3.5, Wan, Z-Image and Vulkan LLM prefill: speed and output checks versus the Phase 0 snapshot.
- Update `PerformanceLeague.md`, `STATUS.md` (only if a verified status changes), `RUNNING.md` timings, and the sample
  gallery README with the new Qwen numbers.

### 5. Expected outcome (honest range, to be replaced by measurements)

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

### 6. Risks and mitigations

- **Cross-model regression:** the kernel family is shared. Mitigation: Phase 0 snapshots, per-phase environment-switch
  rollback (`STINGRAY_VK_QGEMM=legacy`), and no merge without Phase 5-style checks.
- **Driver behaviour:** AMD Windows drivers and FP16 shared memory / packed math can differ from expectations.
  Mitigation: prototype in the standalone benchmark first; never trust an iGPU result as a general claim.
- **Precision:** see the numerics decision in Phase 3.
- **Shader table drift:** shaders are precompiled; every change needs `scripts/gen-spirv.ps1` and the drift test.
- **Effort sink:** Phase 3 is open-ended. Mitigation below.

### 7. Stop rules

- Each phase has a measured gate; a phase that does not beat the previous best by more than run-to-run noise
  (about 2-3% on the forward) is reverted.
- **One gate for Phase 3** (replaces the earlier two that could disagree): keep it only if it is (1) at least 15% faster
  than the best Phase 2 kernel on the target large shapes in the benchmark, **and** (2) measurably faster on the
  end-to-end Qwen forward, not just the microbenchmark, **and** (3) numerically accepted on the real model workload, **and**
  (4) shows no material regression in the other representative models. Otherwise abandon it. Throughput above
  1.2 TFLOP/s is reported, not gated.
- Stop the whole effort at reference parity (G1) unless the user asks for the stretch target.

### 8. Questions for the second opinion

1. Is the Phase order right (small-M, then stride/tile, then FP16 tiles), or is there a cheaper route to the long-K
   throughput loss than chunked layouts? What would explain a stride-dependent loss on a UMA iGPU (TLB reach, memory
   channel interleaving, L2 line conflicts)? Which experiment separates them?
2. Does Vega 8 (gfx90c) give FP16 shared tiles a real benefit when accumulating in FP32 (no dot2/packed-FP16 FMA
   with FP32 accumulate)? ggml has a `GGML_VK_DISABLE_DOT2` switch: what does it key on?
3. Is a K-contiguous dot-product inner loop (ggml) better than the outer-product register tile for LDS bandwidth on GCN?
4. Anything in the list of ggml techniques in section 3 that should be higher priority, or something missing?
5. Is FP16 activation overflow a realistic risk for MM-DiT residual streams, and what is the cheapest guard?
6. Reasonable stop criteria for a kernel project on a non-representative GPU.

### 9. Second opinion received (ChatGPT, 2026-10-10: 8/10, keep the plan and revise in place)

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

### 10. Progress log

#### Phase 0 (partial) and reference data, 2026-10-10
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

#### Phase 1 (small-M kernel), 2026-10-10: kept
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

#### Phase 2(a) 128x128 tile with 256 threads, 2026-10-10: rejected, reverted
Derived mechanically from the large kernel (workgroup 16x16, 128 activation rows x 128 weight rows, 32 KiB shared tiles, two
workgroups per CU). Output checksums bit-identical to the existing kernel. Standalone benchmark, Q3_K, M=1024: MLP-up
78.3 -> 93.9 ms (worse), MLP-down 111.6 -> 105.4 ms (-6%), QKV/out 18.5 -> 19.6 ms (worse). Used only for K=12288 it would save
about 0.36 s per forward (1.5%), below the plan's 2-3% noise floor, so it was not kept. Occupancy is therefore not what
limits MLP-up; the reference's 65 ms there likely comes from its inner-loop formulation, which stays a Phase 3 question.

#### Phase 2(c) chunked image-stream MLP, 2026-10-10: kept
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

#### Checkpoint after Phases 1 and 2(c), 2026-10-10 (same session, alternating A/B, 512x512, GPU, 3 steps, CFG 1)
Steady forward block time (the first forward is 3-4 s slower in both builds because of cold weights, and is excluded):

| Build | Round 1 | Round 2 | Typical |
|---|---|---|---|
| Original (small-M kernel and chunked MLP both off) | 24.45 s | 24.55 s | about 24.5 s |
| Current (both on) | 21.41 s | 22.37 s | about 21.9 s |
| `sd-cli --backend vulkan0`, same workload | n/a | n/a | 20.9 s average over 3 forwards including its lazy weight load; about 20.6 s measured earlier |

That is about 2.6 s (10.7%) faster, and the gap to the reference falls from about 18% to about 6% (G1 target: 20.6 s). Run-to-run
drift is about 5%, so treat the gain as 2-3 s and the remaining gap as 1-1.5 s. The plan's five-run median is still outstanding.
Per-shape reference deficits left: MLP-up at M=1024 (65 ms vs about 78 ms: roughly 0.8 s per forward) and the QKV/out shapes
(about 1 ms each: 0.3 s); the reference has no advantage on MLP-down, which our chunking now beats.

#### Phase 2 MLP-up follow-ups, 2026-10-10: nothing cheap left, stopped
- Fully unrolling the 32-step compute loop (the kernel keeps an 8-step inner unroll with `dont_unroll` on the outer): checksums
  bit-identical, no speed change (3072x3072 chunk 18.7 ms vs 18.7-19.3; MLP-up 78 ms vs 78-79 ms). Rejected, reverted.
- Together with the raster-order swap, split-K, and the 128x128 / 256-thread tile (all rejected above), the cheap structural
  changes to the existing outer-product kernel are exhausted. MLP-up chunked (4 x 18.5 ms = 74 ms) is within about 12% of the
  reference (65 ms) and the same-shape chunk is within 4% (18.5 vs 17.7 ms). The reference's 1189 GFLOP/s on that shape
  (60% of FP32 peak) is higher than the plain-FP32 loop has reached here, which points at packed FP16 arithmetic or a
  different inner-loop structure: Phase 3 territory, with its numerics decision.
- Observation: the unchanged Q3_K kernel at K=12288 (MLP-down) varied between 96 and 117 ms across repeated runs, while the
  K=3072 shapes were stable to about 1 ms. Long-K behaviour is cache/TLB sensitive, one more reason the chunked layout helps.

---

## Research findings (2026-10-30)

Checked in code and vendor docs. Session date 2026-10-30 (the run crossed midnight; the checks were all done in that session).

- **"Path 2 declines every shape today" is stale.** `VulkanBackend.cs` 3177-3183 (`TryMatMulBatchedPath2`) serves Q4_K and Q6_K when `nTok <= TiledQ4KBn` and `cols % 256 == 0`, and declines the rest. The "Using it" table and the scaffold text describe the state before the tiled Q4_K kernel was wired. Path 3 is still the default, so routing is unchanged. **Fix the doc text, not the code.**
- **Path 2's accuracy caveat still stands.** The doc says Path 2 is not a drop-in (per-output association differs from Path 1). That is a property of the kernel, and the code's Path 2 comment (around lines 3185-3189) is consistent with it. The perplexity gate it lists is not run.
- **The "tiled Q6_K shader" checkbox is done in code.** `SgemmQ6K` and `VulkanBackend.SgemmQuant` exist (leads plan, 2026-10-11), and `TryMatMulBatchedPath2` already accepts Q6_K. The checkbox should be closed or re-scoped.
- **Phase 3, cooperative matrix, is not available on this iGPU.** RADV exposes `VK_KHR_cooperative_matrix` only on RDNA3 and newer, and this is a Vega-class (gfx90c) part. The FP16 shared-tile experiment (B) and the FP16-pair dot experiment (C) need a separate check: `VK_VALVE_shader_mixed_float_dot_product` and the dot2 path are not in the device-extension list read in `VulkanBackend.cs` (~384-386). The doc says the dot2 extension is an open question; the code does not enable it.
- **Extension-name capability detection** (see [GR_performance.md](../../GR_performance.md), finding 2) applies to every Vulkan path in this doc, including the FP16 variants. A feature query is needed before any FP16 shader is enabled.

Not checked: the Phase 1 and 2(c) numbers, the Qwen Image forward times, and the reference numbers from `sd-cli`.

**Next checks:** update the "Using it" table; close or re-scope the Q6_K checkbox; before any Phase 3 work, run the feature query from the snippets file to see whether the FP16 features are available on this iGPU.

---

## Research findings (2026-10-31, deep pass)

Excerpts are in [research/2026-10-31-performance-snippets.md](../../research/2026-10-31-performance-snippets.md).

- **Phase 3 experiment C (FP16-pair dot products, "only where the device exposes it"):** the relevant extension is `VK_VALVE_shader_mixed_float_dot_product`. It is new (spec dated 2026-02-04), unratified, and requires `VK_KHR_shader_float16_int8` or Vulkan 1.2. The device check is not in `VulkanBackend.cs`, so the experiment cannot be scheduled until the feature query exists ([refpage](https://docs.vulkan.org/refpages/latest/refpages/source/VK_VALVE_shader_mixed_float_dot_product.html)).
- **The open question "does Vega 8 expose dot2?" stays open.** The mixed-dot spec is not the same as `VK_KHR_shader_integer_dot_product`, which OpenTail does enable (for DP4A). Log both on this device before deciding.
- **Subgroup size:** the Phase 3 kernel design must respect the `requiredSubgroupSize` bound, which must be a power of two between the device min and max ([Khronos refpage](https://docs.vulkan.org/refpages/latest/refpages/source/VkPipelineShaderStageRequiredSubgroupSizeCreateInfo.html)).
- **Cooperative matrix** is RDNA3 and newer on RADV ([Phoronix](https://www.phoronix.com/news/RADV-VK_KHR_cooperative_matrix)). Phase 3's "FP16 shared tiles" route is the only Phase 3 route on this iGPU.
- **Not verified:** the Mesa RADV range on GCN (the source fetch was blocked), and the ggml shader variant the reference uses on this GPU (Phase 0, still open).

---

## Thread state (from the retired workstreams folder, 2026-10-31: vulkan-gemm-llm-prefill.md)

### Correct state

- **Diagnosis (2026-10-10):** the batched prefill already exists. `GpuForwardPass` has a weight-amortising batched trunk for dense Q4_K/Q6_K models. The backlog wording "prefills per token" describes `VulkanLayerSplitForwardPass` and the MoE path, not `GpuForwardPass`. The slow part is the matmul kernel: about 0.3 TFLOP/s on a 493-token prompt.
- **Path 3 (2026-10-11), default:** `STINGRAY_VULKAN_MM_PATH=3` is the default since 2026-10-11 (`VulkanMatMulPath.cs`; set `=1` to roll back).

| Qwen3-4B-Q4_K_M, 493-token prefill, Vulkan iGPU | Prefill |
|---|---|
| Path 1 (default) | 38.4 t/s (12.8 s) |
| Path 3, 128-token chunks, tiled quantised GEMM, sliced flash attention | 71.3 t/s (6.9 s), +86% |
| CPU, same prompt (2026-10-10) | 102.8 t/s |

- Greedy generation identical to Path 1. Weight traffic 64.2 GB to 8.3 GB.
- Sliced flash attention (16-query slices) fixed a fallback that had made Path 3 slower before the fix.

### Open items

- [ ] Larger prefill chunks (256); check scratch buffers and batched attention/KV-append accept them
- [ ] A second dense model under Path 3 (the first measurement is Qwen3-4B only)
- [ ] Decode regression check on Vulkan
- [x] Path 3 default decided on 2026-10-11 (gate: at least 2x today's Vulkan prefill, and identical greedy output; recorded in the leads plan, Lead 2)
- [ ] Confirm the per-kernel CPU-parity tests (`VulkanSgemmQuantEntryTests`, 6 cases) run in the fast suite

- [x] K-quants other than Q4_K and Q6_K stay off the batched Vulkan trunk. **Closed 2026-10-31:** `GpuForwardPass.ComputeCanBatchedTrunk` accepts Q3_K and Q5_K under Path 3 (`GpuForwardPass.cs` 1352-1357; the earlier "only Q4_K and Q6_K" note was stale). SmolLM2-1.7B Q5_K_M now prefills 66.8-85.5 t/s against 84.7-86.0 for its Q4_K_M file (503 tokens, `-g -1`, 3 runs each; the Q4_K_M range is from idle re-runs, the lowest Q5_K_M run has no known cause). The 2026-10-11 figure (22.3 vs 235 tok/s, PerformanceLeague finding 5) predates Path 3. Remaining K-quants (Q2_K, IQ types) were not checked.

### Decisions for the owner

- None open: the Path 3 default was decided on 2026-10-11.

### Cadence

- Review after each Path 3 measurement, and whenever a Vulkan kernel commit lands.
- Re-read at least every two weeks while any item is open.
- Close when the default decision is recorded and the STATUS and RUNNING rows match it.
