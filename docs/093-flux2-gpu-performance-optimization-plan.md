# 093 — FLUX.2 GPU double-block performance optimization plan

**Status, 2026-09-19: plan drafted from an external second-opinion review (ChatGPT), cross-checked
against this repo's own code and history before accepting any of it, then prioritized. Execution
starting with the cheapest, most diagnostic experiments first.**

## Why this doc exists

`docs/091` closed FLUX.2's double-block GPU work as "real, correct, wired into the pipeline, but
1.49x slower than CPU on this dev iGPU" — a real measured result, not a guess. Per the operator's
own instruction, GPU stays wired regardless of today's speed, and the next real task is making it
faster. This doc is that optimization plan.

## External review corrections and confirmations (verified against this repo, not taken on faith)

An external review (asked to read this actual codebase, not just reason abstractly) made one
**important arithmetic correction** to the analysis that produced `docs/091`/`PerformanceLeague.md`'s
numbers, and several claims worth stating precisely:

- **GFLOP correction**: the largest single matmul in a double block (`1024×6144×36864`, the image
  FFN up-projection) is **≈463.9 GFLOP** (`2×M×K×N`), NOT the ≈2.8 GFLOP figure used loosely in an
  earlier informal ChatGPT-prompt draft (that number conflated "output element count" with FLOPs).
  Confirmed by direct calculation: `2 × 1024 × 6144 × 36864 = 463,856,467,968` ≈ 463.9 GFLOP.
- **Aggregate per-block GEMM cost**: summing all 4 GEMM types (QKV, O-proj, FFN-up, FFN-down) for
  both streams gives **≈1.256 TFLOP/block**, **≈10.05 TFLOP** across all 8 double blocks per
  forward pass — confirmed by re-deriving the same arithmetic independently.
- **This changes the diagnosis materially**: at the isolated `SgemmF16` benchmark's own measured
  ~611 GFLOP/s (a real number from this project's FLUX.1 optimization history, not this reviewer's
  guess), the arithmetic-only floor for 10.05 TFLOP is **≈16.4s** — close to the CPU's own 16.6s,
  and meaningfully below the GPU's measured 24.7s. **This means the GEMMs are NOT intrinsically too
  small/inefficient a shape for this GPU to handle well** (the original framing in `docs/091`
  implicitly suggested "this iGPU just loses on small ops," which undersold how large these
  matmuls actually are) — there is a real, closable ~8.3s/pass gap between "what the GEMM
  throughput alone should cost" and "what the whole double-block pass actually costs," and that gap
  is where the optimization opportunity lives (utility-dispatch overhead, unnecessary materialized
  intermediates, dispatch/fence-wait count, FP16 weight-traffic volume).
- **Real self-correction the review made, using this repo's OWN history**: it initially assumed
  "24 dispatches → batch them → obvious big win," then found and cited THIS repo's own
  `docs/083`/PerformanceLeague Wan entry showing batching an 8x28-Wan-block pass only bought ~2.5%
  (1920.5ms → 1873.2ms/block) — a real, documented counterexample already in this project's own
  history. **Conclusion adopted here: measure the batch-size sweep for THIS workload specifically,
  don't assume the FLUX.1 batching win (real, ~significant, documented in `PerformanceLeague.md`)
  generalizes to every block-shaped workload on this iGPU.**

## Existing tooling already in this codebase (do not rebuild)

Confirmed by reading `VulkanBackend.cs` directly:

- `STINGRAY_PROFILE_GPU_SPLIT=1` env var (`s_profGpuSplit`) + `VulkanBackend.ResetGpuProfile()` /
  `PrintGpuProfile(string label)` already exist and are ALREADY the exact record-vs-submit-wait
  split instrumentation the review recommended building — `SubmitAndWait()` already accumulates
  real `submitWait` time (`s_profSubmitWaitMs`, genuine `vkQueueSubmit`+`vkWaitForFences` wall time,
  not an estimate) whenever this flag is set. This is the same mechanism that produced the
  `record=...ms submitWait=...ms` lines seen in this session's own SD1.5 verification run. No new
  instrumentation code is needed for the "record vs. submit+wait" diagnostic — just wrap the
  existing double-block loop with `ResetGpuProfile()`/`PrintGpuProfile()` and read `record` as
  `(wall-clock total) - submitWait`.
- `BeginBatch()`/`EndBatch()` already exist and do exactly what the review describes (record N
  dispatches into one command buffer, one submit, one fence-wait, deferred frees). **Confirmed:
  `Flux2DiT.RunDoubleBlocksGpu`/`DoubleBlockGpu` currently call NEITHER — every single dispatch in
  the 24-op-per-block sequence is its own individual submit+wait right now.** This is the real,
  concrete, already-identified gap the batch-sweep experiment targets.

## Prioritized plan (adopted from the external review, re-ordered/adjusted only where this repo's
own evidence disagreed or added nuance)

### Phase 0 — diagnostic experiments (very low effort, do first, no shader changes)

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
- [ ] **Experiment 3: production-shape GEMM ladder.** NOW THE HIGHEST-PRIORITY REMAINING
      DIAGNOSTIC given Experiment 1+2's result (real GPU execution time, not CPU overhead, is the
      confirmed bottleneck) — this determines whether that execution time is dominated by the
      GEMMs themselves running below their isolated-benchmark throughput, or by the ~16 small
      utility dispatches per block (AdaLN/unpack/QKNorm/RoPE/attention/ScaleGateAdd/SiluGateMul)
      that aren't GEMMs at all. Benchmark the actual 8 GEMM shapes this workload uses
      (`1024×6144×18432`, `256×6144×18432`, `1024×6144×6144`, `256×6144×6144`,
      `1024×6144×36864`, `256×6144×36864`, `1024×18432×6144`, `256×18432×6144`) with the exact
      `SgemmF16` shader and buffer layout already in production use — not a generic microbenchmark.
      Real decision rule: if these land around 500-600 GFLOP/s, GEMM throughput itself is fine and
      the ~8.3s gap lives in the non-GEMM utility dispatches (Phase 1/2's fusion targets become the
      priority); if they collapse to 250-350 GFLOP/s at these specific shapes, the GEMM shader
      itself is the real problem, ahead of any fusion work on the smaller ops.
- [ ] **Experiment 3: production-shape GEMM ladder.** Benchmark the actual 8 GEMM shapes this
      workload uses (`1024×6144×18432`, `256×6144×18432`, `1024×6144×6144`, `256×6144×6144`,
      `1024×6144×36864`, `256×6144×36864`, `1024×18432×6144`, `256×18432×6144`) with the exact
      `SgemmF16` shader and buffer layout already in production use — not a generic microbenchmark.
      Real decision rule: if these land around 500-600 GFLOP/s, GEMM throughput itself is fine and
      the gap lives elsewhere (utility dispatches, materialization, dispatch count); if they
      collapse to 250-350 GFLOP/s at these specific shapes, the GEMM shader/dispatch parameters
      for THIS data size are the real problem, ahead of any fusion work.

### Phase 1 — cheap, concrete removals (low effort, real, no new shader math)

- [ ] **Remove `FluxSliceImg` by adding a row-offset parameter to `Sgemm`.** Real, concrete
      finding from the review: `ws.AttnOut`'s img-stream rows are already contiguous starting at
      row `nTxt` — `FluxSliceImg` exists only because `Sgemm`'s current signature has no way to
      read a matmul's input starting at a row offset into an existing buffer. **Needs real
      verification before assuming this is a one-line change**: `Tensor` (`Core/Tensor.cs`) wraps
      an opaque backend-owned `Handle` with no offset/stride field, and `VulkanBackend.GetBuffer`
      needs to be checked for whether Vulkan buffer bindings in this codebase already support a
      byte-offset bind (`vkCmdBindDescriptorSets` dynamic offsets, or a manually-offset
      `VkDescriptorBufferInfo`) or whether this needs new plumbing. Real next step: read
      `GetBuffer`/`DispatchOrRecord`'s descriptor-set binding code before assuming this is cheap —
      if row-offset binding doesn't exist anywhere in this codebase yet, this becomes a real (if
      still valuable) new capability, not a one-line change, and should be re-costed honestly.
- [ ] **Apply the same row-offset idea anywhere else a materialize-then-immediately-consume
      pattern exists** in the double-block sequence, once the offset capability exists (audit
      `DoubleBlockGpu` for other candidates once Sgemm supports it).

### Phase 2 — kernel fusion (medium effort, real shader-authoring work, do only after Phase 0
data justifies it)

- [ ] **Fuse `SiluGateMul` into the FFN down-GEMM's activation-tile load** (the review's own
      top-priority fusion pick, with real reasoning: this intermediate is `n×18432` — the LARGEST
      materialized intermediate in the block, ~36MiB at FP16 for the image stream alone — versus
      `ScaleGateAdd`'s `n×6144`, a much smaller, lower-payoff fusion target). Real implementation
      note from the review, worth preserving: don't attempt to fuse the UP-gemm and DOWN-gemm into
      one giant kernel (the up-projection's full `1024×36864` output can't cheaply live in
      registers/LDS) — instead, modify the down-GEMM's own tile-loading code to compute
      `silu(gate)*value` from the up-projection's raw output AS it loads each activation tile, never
      materializing the gated result as its own buffer.
- [ ] **Fuse QKV-unpack + QK-RMSNorm + RoPE into one dispatch** (`Flux2QkvNormRope`, mirroring the
      real, already-shipped FLUX.1 precedent for the same class of fusion). Confirmed real
      candidate: Q and K share identical normalization structure, RoPE immediately follows
      normalization, V needs neither transform, and there's no correctness reason to materialize
      the unpacked-but-not-yet-normalized Q/K/V tensors as their own round-trip through GPU memory
      first.
- [ ] **Combine img/txt-stream AdaLN and ScaleGateAdd pairs into single dispatches** (branch on
      row index against `nTxt` inside one dispatch spanning `nTxt+nImg` rows, instead of two
      separate dispatches). Real, correctly deprioritized by the review below the two fusions
      above — smaller payloads (`n×6144`, not `n×18432`), test only after the higher-value fusions
      are measured.

### Phase 3 — weight-representation change (high effort, potentially the largest structural win)

- [ ] **Do NOT extend the existing `MatMulTiledQ4K` LLM kernel's 16-row cap to this workload.**
      Real, important correction from the review, confirmed by reading `Shaders.cs`'s
      `MatMulTiledQ4K` source directly (already done in `docs/088`'s own memory-budget
      investigation): `BN=16` is compiled into the shared-memory layout and thread mapping,
      genuinely not a runtime parameter. Chunking a 1024-row matmul into 64 sequential 16-row
      dispatches against that kernel would reintroduce exactly the dispatch-count problem Phase 0/1
      are trying to eliminate. **A real, new, large-tile (`BM=64-128`, `BN=64-128`) Q4_K GEMM
      kernel sized for diffusion-scale M is a different, larger task than "reuse the LLM kernel."**
- [ ] **Real quantitative case for why this is worth building on THIS specific machine (a
      shared-memory iGPU, not a discrete GPU)**: the review's own arithmetic, re-verified —
      `6144×36864` FP16 weight matrix is `6144×36864×2 ≈ 432MiB`; the same matrix at real Q4_K
      density (~4.5 bits/param ≈ 0.5625 bytes/param) is `≈121MiB`, a real ≈3.56x reduction in BOTH
      the one-time upload volume (currently 34.4s for ~15.7GB total across the 8 double blocks) AND
      every steady-state read of that weight during compute — relevant on a UMA system where GPU
      "VRAM" bandwidth is the same physical DRAM bus the CPU also uses.
- [ ] **Real, cheap-first experiment before committing to the full kernel**: build ONE large-tile
      Q4_K GEMM sized for the single largest shape (`1024×6144×36864`) only, and measure cold-start
      upload cost + single-GEMM compute time against the current FP16 path for that exact shape —
      do not build all 8 GEMM shapes' Q4_K variants before this single real data point justifies
      the larger investment.

## Explicitly deferred / rejected by the review, with reasoning preserved

- **Fusing AdaLN directly into the QKV GEMM** — correctly rejected: risks degrading the
  already-tuned high-throughput tiled GEMM shader's occupancy/register/LDS behavior by contaminating
  it with reduction logic, for a comparatively small elementwise op's worth of savings.
- **Rewriting joint attention before the QKV→norm→RoPE chain is GPU-resident and profiled** —
  correctly deferred: this project's own history (cited by the review) shows attention rewrites can
  regress further when the surrounding pipeline still has other transfer/dispatch costs dominating;
  benchmark the existing tiled attention shape in isolation only after Phase 0/1/2's other work
  lands, not before.

## Execution order for this session

Given Phase 0's three experiments are genuinely cheap (reuse existing profiling infra, no new
shaders, no new kernels) and directly gate every later phase's priority, they are being executed
now, in order, before any Phase 1+ code changes.
