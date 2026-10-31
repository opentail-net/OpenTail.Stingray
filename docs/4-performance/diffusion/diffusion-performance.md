# Diffusion performance: owner file

**Owner:** this file (2026-10-30). It replaces four separate plans, which are kept below verbatim, one per part:

1. **Part 1, umbrella plan** (was `094-diffusion-performance-plan.md`): ground rules, phases 6 and 9, working log.
2. **Part 2, FLUX.1 Vulkan GEMM handoff** (was `069-flux-vulkan-gemm-perf-handoff.md`).
3. **Part 3, FLUX.2 double-block plan** (was `093-flux2-gpu-performance-optimization-plan.md`).
4. **Part 4, Qwen Image and the image samples** (was `2026-10-10-diffusion-samples-and-qwen-image-performance-plan.md`).

Completed items from the earlier split remain in [done/094-diffusion-performance-plan-closed-phases.md](../../done/094-diffusion-performance-plan-closed-phases.md), [done/069-flux-vulkan-gemm-perf-handoff-completed-items-2026-10.md](../../done/069-flux-vulkan-gemm-perf-handoff-completed-items-2026-10.md) and [done/093-flux2-gpu-performance-optimization-plan-completed-items-2026-10.md](../../done/093-flux2-gpu-performance-optimization-plan-completed-items-2026-10.md).

The FLUX.1 DiT gap (Part 2) is also tracked in the [performance pickup](../2026-10-30-performance-pickup.md) (rows 2-4). The measurement rules are in [../../reference/performance-measurement-rules.md](../../reference/performance-measurement-rules.md).

---

## Part 1: umbrella plan

## 094 — Diffusion Stack Performance Plan (2026-09-19)

> **SPLIT 2026-09-27.** Phases 0, 1, 2, 3, 4, 5, 7 and 8 are closed and moved verbatim to
> [done/094-diffusion-performance-plan-closed-phases.md](../../done/094-diffusion-performance-plan-closed-phases.md); each is cited there. Open here: **Phase 6** (FLUX.1: the per-kernel
> comparison with sd.cpp and on-GPU quantized matmul are not settled by the later work: FLUX.1 GPU went
> 173.6s -> 132.0s with register-tiled flash attention in a5daf08, and 017765c gave a measured gap
> diagnosis) and **Phase 9** (the cross-model DRY and perf-doc consistency pass). The findings below
> were true on 2026-09-19 and several are now outdated (SD3.5, Qwen Image and HunyuanVideo all have
> GPU paths recorded in `PerformanceLeague.md`).

Cross-examined `PerformanceLeague.md`'s diffusion section against the actual source (not just the
doc's own prose) before writing this plan. Real findings from that cross-examination that change
the starting picture:

- **SD3.5 already has a real GPU-resident `MMDiTModel.ForwardGpu`** (`src/OpenTail.Stingray.Diffusion/SD3/MMDiTGpuWeights.cs`,
  `MMDiTGpuWorkspace.cs`, wired via `Forward()`'s automatic `IVisionOpsBackend`/`IImageOpsBackend`
  dispatch, committed 2026-09-15, `a991037`) — but `PerformanceLeague.md` never recorded a single
  GPU timing row for it. A real parity test already exists (`Sd3BaselineTests.TestSd35GpuVsCpuParity`)
  but its timing output was never captured into the doc. This is a documentation gap, not a missing
  feature.
- **LTX-Video also already has a real `LtxVideoModel.ForwardGpu`** (`LtxVideoGpuWeights.cs`/
  `LtxVideoGpuWorkspace.cs`) — the doc's "not attempted" framing was about the DiT's correctness bug,
  not the GPU port, but reads ambiguously. Needs disambiguating once correctness is fixed.
- **HunyuanVideo's "never attempted, no wired CLI/test path" line (PerformanceLeague.md ~962-964) is
  flatly wrong and stale.** `ImageCommand.IsHunyuanVideo`/`RunHunyuanVideo` wires it into `stingray image`,
  real checkpoints are present in `models/hunyuanvideo/`, and `HunyuanVideoRealWeightsTests` (a real,
  non-no-op test — verified by actually running it) executes a genuine forward pass + VAE decode in
  64.2s wall for a 32×32/1-frame/1-step smoke case. No GPU path exists for it at all (`grep` for
  `ForwardGpu`/`IVisionOpsBackend` in `HunyuanVideoModel.cs` returns nothing) and no production-scale
  timing has ever been recorded — that part of the "opportunity" framing was right, just for the
  wrong reason.
- **Qwen Image genuinely has zero GPU code** (confirmed by grep — no hit at all) — the one gap the
  doc had exactly right.

### Ground rules for this whole plan (per CLAUDE.md, restated so the loop doesn't drift)

- Every "done" checkbox needs: a real weights run, a real measured number (not an estimate), and
  either a fixed correctness bug verified against `examples/stable-diffusion.cpp` / `examples/diffusers`
  / `examples/flux` / `examples/flux2` where applicable, or an explicit note that no reference exists
  for that case.
- Update `PerformanceLeague.md` in the same pass a number is measured — don't batch documentation to
  the end.
- If an item stalls (missing checkpoint, crash, correctness blocker upstream of perf), write the exact
  blocker into this doc under that checkbox and move to the next one. Do not stop the loop.
- No subagents for this work (project-wide rule) — all done in the main/looped session directly.
- A performance win must be measured, not assumed (a handful of runs, keep the number even if it's a
  negative result — see PerformanceLeague.md's own many recorded reverts for the expected format).
- **"Useful" is not only wall-clock speed — memory footprint counts as a real, first-class win too**
  (2026-09-19, user directive). This project has repeated, real evidence that memory pressure is
  often the actual bottleneck, not raw compute: Qwen Image's CPU path was OOM-killed at 53GB+ heading
  toward ~75GB on a 64GB machine before `QuantizedWeightCache` fixed it (docs/086); FLUX.2's own GPU
  residency plan is explicitly capped at 8 double-blocks specifically because the 48 single-blocks
  would need ~63GB, this machine's entire RAM (`Flux2GpuWeights.cs`'s own doc comment); several
  models in this doc measure "7.4× less memory" or "fits in 17.37GB where PyTorch needs ~120GB" as a
  headline result alongside (sometimes instead of) a wall-clock number. When scoping or measuring any
  item below: **record peak memory (CPU RSS and/or GPU VRAM, e.g. via `STINGRAY_PROFILE_GPU_SPLIT=1`'s
  live/peak device-local byte tracking, already wired in `VulkanBackend`) alongside wall-clock time**,
  and treat a real, measured memory reduction as worth documenting and keeping even when wall-clock is
  flat or slightly worse — the same standard already applied to Stingray-vs-PyTorch comparisons
  elsewhere in `PerformanceLeague.md`. A change that trades a little speed for a lot of headroom (e.g.
  making a model fit on this iGPU's shared-memory budget at all, or avoiding an OOM kill outright) is
  a real win, not a wash, and should be reported as such rather than only through a speed lens.

---

### Checklist

#### Phase 6 — FLUX.1: attack the remaining DiT-loop gap (currently ~3.1x behind C++ per the last profiled row)

- [ ] Real per-kernel GFLOP/s comparison: Stingray's `SgemmF16` (611 GFLOP/s measured) vs. what
      `stable-diffusion.cpp`'s GGML backend achieves on the identical iGPU for the identical matmul
      shapes — get a real number, not an assumption, by adding equivalent GFLOP/s instrumentation to
      the sd.cpp side if it doesn't already print one.
- [ ] Investigate on-GPU quantized matmul (skip the FP16-dequant-then-GEMM step) as the next lever,
      since GGML's advantage is believed to come from direct quantized cooperative-matrix ops. Scope
      real feasibility against Vortice.Vulkan's cooperative-matrix extension support before committing
      to an implementation.
- [ ] If implemented, real parity test first, then real end-to-end timing, documented.

#### Phase 9 — Cross-model DRY + perf-doc consistency pass (per CLAUDE.md's own performance+DRY pass rule)

- [ ] Once every phase above lands a real change, check for logic duplicated across the newly-touched
      GPU paths (e.g. if Qwen Image's `ForwardGpu` and FLUX.2's single-block port both end up
      hand-rolling the same QKV-norm-RoPE fusion pattern) and extract shared code under
      `Primitives/*Kernels.cs`, matching the existing convention already used for `WanAttention`.
- [ ] Re-run affected golden/structural parity tests after any extraction to confirm zero numerical
      drift.
- [ ] Final full re-read of `PerformanceLeague.md`'s diffusion section for internal consistency (no
      stale "not attempted" claims left, every GPU path that exists in code has at least one recorded
      timing row) and update `README.md`'s status matrix per CLAUDE.md rule 10 if any pipeline's
      real status changed (e.g. HunyuanVideo moving from unlisted/unknown to a real, timed, CPU-only
      🟡/🔴 row depending on what the coherence check in Phase 4 finds).

---

### Working log

(Append real findings/blockers here as the loop executes, dated, so this doc stays the source of
truth for what's actually been tried — same discipline as `PerformanceLeague.md` itself.)

---

## Part 2: FLUX.1 Vulkan GEMM handoff

> **STATUS 2026-09-27.** Still open, target not met, but the gap has closed substantially since this

> Completed items moved to [done/069-flux-vulkan-gemm-perf-handoff-completed-items-2026-10.md](../../done/069-flux-vulkan-gemm-perf-handoff-completed-items-2026-10.md) on 2026-10-30. This file keeps only open work.
> was written: `PerformanceLeague.md`'s 2026-09-25 "register-tiled flash attention (headDim=128)" row
> measured **132.0s** total (DiT 23.5s/step, down from the 374.7s this doc's own body cites), vs this
> doc's **99.8s** target — now **~1.32x** slower, not the ~3.75x this doc's numbers implied. Re-check
> against `PerformanceLeague.md` before assuming either the 374.7s or the 99.8s gap size below is
> still current.

## FLUX.1-schnell Vulkan GEMM/Attention Performance Handoff (2026-09-13)

### Where things stand

FLUX.1-schnell's Vulkan GPU-resident path (`FluxDiT.ForwardGpu`/`FluxGpuWeights`/`FluxGpuWorkspace`,
`src/OpenTail.Stingray.Diffusion/`) is **numerically correct** as of this handoff — both the
long-standing "repeating tiled background" artifact (see `docs/done/056-flux-tiling-artifact-handoff.md`
Round 9) and an earlier NaN/black-image regression in the GPU-resident path itself are fixed and
verified with real end-to-end runs producing clean, coherent, on-prompt images matching the real
C++ reference's quality. **This is now a pure performance problem, not a correctness one.**

Real, measured numbers (512×512, 4 steps, seed 42, same checkpoint/prompt, same AMD Cezanne iGPU
throughout — see hardware correction below):

| | Ours (Vulkan, GPU-resident) | `stable-diffusion.cpp` (real GGML reference, Vulkan) | Ratio |
|---|---|---|---|
| CLIP-L encode | 0.71s | ~part of 11.29s combined clip+t5 | — |
| T5-XXL encode | 77.6s | ~part of the same 11.29s | slower |
| **DiT denoising loop (4 steps)** | **756.1s (~189s/step)** | **81.81s (~20.45s/step)** | **~9.2× slower** |
| VAE decode | 12.6s | 8.78s | ~1.4× slower |
| **Total** | **858s** | **99.8s** | **~8.6× slower** |

Within our DiT loop, real Vulkan-fence-level instrumentation (`STINGRAY_PROFILE_GPU_SPLIT=1`) shows
**686.4s of 756.1s (91%) is genuine `vkQueueSubmit`+GPU-execute+`vkWaitForFences` time**, across 1027
dispatches. **This rules out excessive CPU-side command-buffer recording or memory-barrier overhead
as the dominant cause** — the GPU itself is taking a very long time to execute the actual compute.
The entire ~8.6× gap is concentrated in the DiT loop; VAE decode and CLIP are already close to the
reference's speed.

Full narrative and every number's provenance: `PerformanceLeague.md`'s FLUX section (search
"granular stage profiling"), and `docs/done/056-flux-tiling-artifact-handoff.md`'s Round 9.

### Hardware correction (important — read before optimizing)

This machine's iGPU is an **AMD Ryzen 7 5700G "Cezanne" integrated GPU** — 8 compute units,
**GCN/Vega-derived architecture**, not RDNA2. Its native wavefront width is **64**, not RDNA2's
32/64-configurable-but-commonly-32 convention. Do not apply RDNA2-specific tuning assumptions
(e.g. wave32-first heuristics) — this is older-generation GCN.

One relevant fact already true in this codebase, easy to miss: the current SGEMM shader
(`Shaders.SgemmF32`/`SgemmF16`, `src/OpenTail.Stingray.Vulkan/Shaders.cs` around line 5860) declares
`layout(local_size_x = 8, local_size_y = 8)` — **64 threads per workgroup**, matching Cezanne's
64-wide wavefront exactly. The "32×32" in prior notes refers to the *output tile* one workgroup
computes (32×32, via 4×4 register blocking per thread across 8×8=64 threads), not a 32-thread
workgroup — so the wavefront-width mismatch a first read of "32×32 tile" might suggest is **not**
actually present. Confirm this is still true before assuming it's an issue (a diagnostic session
raised subgroup width as a candidate before this correction; it may still be worth measuring
`VkPhysicalDeviceSubgroupProperties.subgroupSize` for real — see Task 1 below — this codebase
already has a `HasSubgroupSizeControl`/`MinSubgroupSize`/`MaxSubgroupSize` mechanism at
`VulkanBackend.cs` around line 372-392 and `ComputePipeline.ShouldPinSubgroupSize32`
(`ComputePipeline.cs` line 67) that pins `requiredSubgroupSize=32` under some conditions — check
whether that pinning is actually firing for this device/shader and whether it helps or hurts here).

### External analysis (ChatGPT, 2026-09-13, given the table above)

A structured second opinion was sought given the size of the gap. Full response is preserved in
this session's transcript; summary of its ranked hypothesis and recommended methodology:

**Ranked likelihood** (its own numbers): GEMM kernel underutilizing the GPU (50%), attention
implementation (25%), memory/register/occupancy effects touching both (15%), barriers/sync (5%),
other (5%). It independently arrived at the same "barriers are not the cause" conclusion the real
686.4s/756.1s measurement above already supports — with the math spelled out: even a *pathological*
1ms-per-dispatch barrier cost across ~1027 dispatches is only ~1s, nowhere near explaining hundreds
of seconds.

**Its core recommendation, in priority order — do NOT skip straight to changing tile sizes**:






### Practical notes for whoever picks this up

- **No subagents** — do all work directly in the main session for this project (this project's
  `CLAUDE.md` rule 6, applies here as everywhere else in this repo).
- **Never remove or revert the other AI's GPU-residency work** — it is real, correct, and a genuine
  architectural improvement over the pre-residency per-matmul CPU/GPU ping-pong. Only fix/add
  forward, the same discipline this session and the artifact-hunting rounds before it followed.
- **Do not commit without checking with the user first** — this is still someone else's in-flight
  work with fixes layered on top by this session; nothing has been committed to git as of this
  handoff.
- **Measurement rules:** see [../../reference/performance-measurement-rules.md](../../reference/performance-measurement-rules.md) (one thing at a time, memory check before benchmarks).
- **The existing profiling flags are real and already wired**: `STINGRAY_PROFILE_GPU_SPLIT=1`
  (submit+GPU-exec+fence-wait totals, `VulkanBackend.cs`) and `STINGRAY_PROFILE_DECODE=1`
  (CLIP/T5/setup/DiT-loop/VAE stage split, `ImagePipeline.cs`) are both wired into the FLUX CLI path
  now (`RunFlux` in `src/OpenTail.Stingray.Cli/ImageCommand.cs`, added this session) — use them
  rather than re-deriving timing from scratch, and extend them (real Vulkan timestamp queries,
  per-shader-type breakdown) rather than replacing them.
- **A real full 512×512/4-step run costs ~850-900s right now** — expensive to iterate on. Prefer the
  small-nImg/nTxt synthetic-scale technique already used to bisect the earlier NaN bug
  (`tests/OpenTail.Stingray.Tests.Diffusion/FluxGpuVsCpuForwardBisectDebugTest.cs`, ~70s per run,
  real weights, tiny nImg=16/nTxt=8) for the GEMM microbenchmark and per-shader-type profiling work
  — much faster iteration, and the GFLOP/s-per-shape measurement doesn't need a full image's worth
  of tokens to be meaningful.
- **Repro command** (full real run, once ready to re-verify at scale):
  ```
  dotnet run --project src/OpenTail.Stingray.Cli -c Release -- image \
    -m models/flux1-schnell/flux1-schnell-Q4_K_S.gguf \
    --vae models/flux1-schnell/ae.safetensors \
    --clip-l models/flux1-schnell/clip_l.safetensors --clip-tokenizer models/flux1-schnell/tokenizer_2/tokenizer.json \
    --t5xxl models/flux1-schnell/t5xxl_fp8_e4m3fn.safetensors --t5-tokenizer models/flux1-schnell/tokenizer_t5/tokenizer.json \
    -p "a red apple on a wooden table" \
    --steps 4 --seed 42 --verbose --backend vulkan \
    --output <path>.png
  ```
  (set `STINGRAY_PROFILE_GPU_SPLIT=1`/`STINGRAY_PROFILE_DECODE=1` in the environment first for the
  stage/dispatch breakdown.)
- **Real C++ reference for comparison** is already built and confirmed working on this machine:
  `examples/stable-diffusion.cpp/build/bin/sd-cli.exe`. Use `--diffusion-model` (not `-m`) to load
  our GGUF checkpoint — its tensor names lack the `model.diffusion_model.` prefix sd.cpp's
  version-detector expects, and `--diffusion-model` auto-prepends it at load time. Vulkan backend:
  `--backend vulkan0`.

### Success criterion

- [ ] Close the remaining DiT loop gap to reach the **99.8s** target (the real C++ reference's Vulkan time on
  this exact hardware/checkpoint/prompt/config; currently at 132.0s with register-tiled flash attention).
  Update `PerformanceLeague.md` honestly with whatever is found, following its existing style
  in the FLUX section (measured numbers, single-run caveats, no fabrication) — do not round favorably
  or claim a win before a real end-to-end run confirms it.

---

### Post-Optimization Results (2026-09-13 Update)

#### 1. Kernel Optimization Gains
- **`SgemmF16` (64×128 tile, 128 threads, transposed LDS, FP32 accumulator)**:
  - Single Lin1 (`[1280, 21504, 3072]`): 1590.4 ms (106 GFLOP/s) → **276.8 ms (611 GFLOP/s)** (**5.75× speedup**).
  - Single Lin2 (`[1280, 3072, 15360]`): 1046.5 ms (115 GFLOP/s) → **204.4 ms (591 GFLOP/s)** (**5.12× speedup**).
  - Double Img MLP0 (`[1280, 12288, 3072]`): 477.2 ms → **133.3 ms** (**3.58× speedup**).
  - Double Img MLP2 (`[1280, 3072, 12288]`): 603.7 ms → **129.6 ms** (**4.66× speedup**).
  - Double Img QKV (`[1280, 9216, 3072]`): 352.6 ms → **100.3 ms** (**3.52× speedup**).
- **`MultiHeadAttentionTiled128` (32×16 tile, 128 threads / 2 wavefronts, vectorized `vec4`, in-register dot products)**:
  - MultiHeadAttention: 371.3 ms (54.2 GFLOP/s) → **204.7 ms (98.3 GFLOP/s)** (**1.81× speedup**, maxDiff: `7.45e-8`).

#### 2. End-to-End Real FLUX Generation Benchmark (512×512, 4 steps, seed 42)
- **Baseline Total Time**: 858.4s (DiT loop: 756.1s)
- **Optimized Total Time**: **374.7s** (**2.29× overall speedup**, DiT loop ~272s, **2.78× DiT speedup**)
- **Numerical Parity**: 100% verified against CPU reference across all 57 blocks (`FluxGpuVsCpuForwardBisectDebugTest` passing in 72s).


---

## Part 3: FLUX.2 double-block plan

> **STATUS 2026-09-27 (checkboxes below not updated at the time).** Done since this was written:

> Completed items moved to [done/093-flux2-gpu-performance-optimization-plan-completed-items-2026-10.md](../../done/093-flux2-gpu-performance-optimization-plan-completed-items-2026-10.md) on 2026-10-30. This file keeps only open work.
> Phase 3's weight-representation change (Q4_K weights kept quantized on the GPU, full-GPU 512²
> 2-step 298.4s -> 237s, parity in `VulkanSgemmQuantParityTests`; 43307bc, 2026-09-24) and the
> img/txt AdaLN/ScaleGateAdd dual-stream dispatch (fcca2f7, 2026-09-20). Still open: Experiment 3
> (production-shape GEMM ladder) and the row-offset audit of `DoubleBlockGpu`.

## 093 — FLUX.2 GPU double-block performance optimization plan

**Status, 2026-09-19: plan drafted from an external second-opinion review (ChatGPT), cross-checked
against this repo's own code and history before accepting any of it, then prioritized. Execution
starting with the cheapest, most diagnostic experiments first.**

### Why this doc exists

`docs/091` closed FLUX.2's double-block GPU work as "real, correct, wired into the pipeline, but
1.49x slower than CPU on this dev iGPU" — a real measured result, not a guess. Per the operator's
own instruction, GPU stays wired regardless of today's speed, and the next real task is making it
faster. This doc is that optimization plan.

### External review corrections and confirmations (verified against this repo, not taken on faith)

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

### Existing tooling already in this codebase (do not rebuild)

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

### Prioritized plan (adopted from the external review, re-ordered/adjusted only where this repo's
own evidence disagreed or added nuance)

#### Phase 0 — diagnostic experiments (very low effort, do first, no shader changes)

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

      **[x] DONE 2026-09-19 — DEFINITIVE, CLEAN RESULT: GEMM throughput is NOT the problem.**
      New `Flux2GpuGemmLadderBenchmarkTests.ProductionShapeGemmLadder_ReportsThroughput` (real
      `Sgemm` dispatch, real FP16 weight upload, exact production shapes, 3 timed trials each,
      no real weights needed since this only exercises the GEMM shader/buffer path):

      | Op | Shape | Time | GFLOP/s |
      |---|---|---:|---:|
      | QKV proj (img) | 1024×6144×18432 | 382.78ms | 605.9 |
      | QKV proj (txt) | 256×6144×18432 | 90.56ms | 640.2 |
      | O proj (img) | 1024×6144×6144 | 128.32ms | 602.5 |
      | O proj (txt) | 256×6144×6144 | 30.52ms | 633.2 |
      | FFN up (img) | 1024×6144×36864 | 767.14ms | 604.7 |
      | FFN up (txt) | 256×6144×36864 | 180.37ms | 642.9 |
      | FFN down (img) | 1024×18432×6144 | 386.08ms | 600.7 |
      | FFN down (txt) | 256×18432×6144 | 95.18ms | 609.2 |

      **All 8 production shapes land at 600-643 GFLOP/s — matching or exceeding the isolated ~611
      GFLOP/s microbenchmark, zero degradation at these specific sizes.** Summing all 8 GEMM times
      gives ≈2061ms per block; ×8 blocks ≈**16,488ms (16.5s) for GEMM compute alone across the
      whole double-block pass — almost exactly matching both the CPU baseline (16.3-16.6s) and the
      arithmetic-only theoretical floor (16.4s) independently derived earlier in this doc.**

      **Real conclusion: the GEMMs are essentially optimal already. The ~6-8s gap between this
      16.5s GEMM-only floor and the measured 22.5-24.7s double-block-pass total is NOT explained by
      GEMM inefficiency — it must live in the ~16 non-GEMM utility dispatches per block (AdaLN,
      QKV-unpack, QK-RMSNorm, RoPE, ScaleGateAdd, SiluGateMul) and/or the joint attention dispatch
      itself (NOT covered by this GEMM ladder — `MultiHeadAttentionTiled`, a real, separate,
      substantial compute op over the full 1280-token sequence, not yet benchmarked in isolation).**
      This makes Phase 1 (remove `FluxSliceImg`) and Phase 2 (fuse the utility-dispatch chains) the
      correctly-prioritized next real work — exactly where the plan's own decision rule said to
      look if GEMM throughput came back clean.



#### Phase 1 — cheap, concrete removals (low effort, real, no new shader math)

- [ ] **Apply the same row-offset idea anywhere else a materialize-then-immediately-consume
      pattern exists** in the double-block sequence, once the offset capability exists (audit
      `DoubleBlockGpu` for other candidates once Sgemm supports it).

#### Phase 2 — kernel fusion (medium effort, real shader-authoring work, do only after Phase 0
data justifies it)


#### Phase 3 — weight-representation change (high effort, potentially the largest structural win)


### Explicitly deferred / rejected by the review, with reasoning preserved

- **Fusing AdaLN directly into the QKV GEMM** — correctly rejected: risks degrading the
  already-tuned high-throughput tiled GEMM shader's occupancy/register/LDS behavior by contaminating
  it with reduction logic, for a comparatively small elementwise op's worth of savings.
- **Rewriting joint attention before the QKV→norm→RoPE chain is GPU-resident and profiled** —
  correctly deferred: this project's own history (cited by the review) shows attention rewrites can
  regress further when the surrounding pipeline still has other transfer/dispatch costs dominating;
  benchmark the existing tiled attention shape in isolation only after Phase 0/1/2's other work
  lands, not before.

### Execution order for this session

Given Phase 0's three experiments are genuinely cheap (reuse existing profiling infra, no new
shaders, no new kernels) and directly gate every later phase's priority, they are being executed
now, in order, before any Phase 1+ code changes.

---

## Part 4: Qwen Image and the image samples

## Diffusion samples to finish, and the Qwen Image performance plan (2026-10-10)

### Checklist (2026-10-30; checked against the log)

- [x] SD1.5, SDXL-Turbo, SDXL base (1024), FLUX.1-schnell, Z-Image-Turbo, SD3.5 Medium (section A "already done")
- [x] Wan 2.1 one frame (CPU); LTX-Video one frame (abstract output)
- [ ] Qwen Image Q3_K_S at 512x512 on the GPU (section A order 1; first attempt hit the 20-minute cap on CPU)
- [ ] Wan 2.2 14B, one frame, 256x256
- [ ] HunyuanVideo 720 fp8, one frame, 256x256
- [ ] FLUX.2-dev + Mistral-Small-3.2-24B, 512x512 (never run before)
- [ ] B0 workload: built and under test (status 2026-10-10)
- [x] B1 baseline: measured results section (Qwen Image Q3_K_S, 60/60 blocks resident)
- [ ] B2 instrument, B3 candidate optimisations, B4 A/B protocol on each change
- [ ] B5 iGPU limits recorded (not a discrete-GPU result)

Two goals: finish the apple gallery in `docs/diffusion-samples/`, and find out why Qwen Image is so slow. This
machine has an integrated Radeon only, so every timing here is an iGPU timing (CLAUDE.md rule 13). Rules 7, 11
and 12 apply: measure before and after, flag any performance-pass cost, and a timing is only evidence if the log
shows what really ran. One model run at a time, always.

### A. Remaining images (serial, lightest first, single frame, GPU)

Prompt `a red apple on a wooden table`, seed 42, `--device 0`. Each result records the backend actually used (read
from the log), the time, and what the image shows. Already done: SD1.5, SDXL-Turbo, SDXL base (1024), FLUX.1-schnell,
Z-Image-Turbo, SD3.5 Medium, Wan 2.1 (one frame, CPU), LTX-Video (one frame, abstract output).

| Order | Item | Notes |
|---|---|---|
| 1 | Qwen Image Q3_K_S, 512x512 | Section B. First attempt hit a 20-minute cap on the CPU path (no `--device 0`). |
| 2 | Wan 2.2 14B, one frame, 256x256 | Heavy. |
| 3 | HunyuanVideo 720 fp8, one frame, 256x256 | Heavy. |
| 4 | FLUX.2-dev + Mistral-Small-3.2-24B, 512x512 | Heaviest. Never run before, and its CLI route is new. |

Dropped, with the reason stated in the gallery README: video clips (a 33-frame Wan 2.1 attempt ran 2h37m with no
output). SDXL base Q4_0 GGUF is on hold: it uses sd.cpp-style tensor names (`input_blocks.…`) that our loader does
not read.

### B. Qwen Image performance

#### B0. Make the workload real (done in this change)
- The CLI never dispatched Qwen Image (`IsQwenImage` was unused) and never passed a text encoder to
  `QwenImagePipeline.Load`. Fixed: dispatch added, `--qwen-encoder` now selects the overload that builds the real
  Qwen2.5-VL encoder. Without it the pipeline silently falls back to non-text conditioning, which is not a valid
  baseline. Open item: make that fallback loud (a warning) instead of silent.
- Confirm from the log that the run says "on Vulkan GPU", not "on CPU".

#### B1. Baseline (before any optimisation)
1. 512x512, **2 steps**, CFG 2.5, seed 42, `--device 0`: split load / text encode / first forward / later forward /
   VAE decode.
2. Same at 256x256 for resolution scaling.
3. Repeat the 512x512 2-step case three or more times without extra instrumentation for a stable end-to-end figure.
4. The full 512x512, 20-step run, once, with a long cap. If it times out, record a timeout, not an invented time.
   Known context: `PerformanceLeague.md` has Qwen Image at 256x256, 8 steps, CFG 4: Vulkan about 160s, CPU about
   277s. That is not a baseline for 512x512/20 steps.

#### B2. Instrument (only what the baseline cannot answer)
Stage timers in the Qwen Image CLI route: model load, text encode, per-step conditional and unconditional forward,
VAE, PNG write; Vulkan submit/wait counts and time; peak working set. Treat `submitWait` as submission plus GPU
plus fence wait, not pure GPU time. Vulkan timestamp queries are the right tool for a per-kernel GPU split if the
stage timers say the GPU is the problem.

#### B3. Candidate optimisations (one at a time, each A/B'd against B1)
Ranked by what to investigate first. Each is a hypothesis to verify in the code before it is attempted.
1. **Workspace and RoPE cache churn.** `QwenImageModel` keeps a single resident GPU workspace and RoPE cache keyed
   by image and text token counts; conditional and unconditional contexts can differ in length, so they may evict
   each other every forward. Verify by logging workspace creations and cache hits.
2. **Submission count.** If the GPU forward begins and ends a batch per transformer block, that is about 61
   submit-and-wait operations per forward and about 2,400 per 20-step CFG run. Test a single batch across the
   resident block chain, keeping the barriers.
3. **Work shared between the CFG branches.** Timestep modulation outputs and the image input projection are
   identical for both branches at a given step; cache them.
4. **True batched CFG.** Only after the above, prototyped on one block against two separate forwards. High risk
   (joint attention, differing text lengths).
5. **Kernel tuning (Q3_K GEMM, fused attention).** Only if profiling names them as the bottleneck. Quantised GPU
   weights and a tiled attention path already exist; do not regress them.

Out of scope for engine work, but possible operating-point changes: fewer steps or a distilled LoRA (needs
compatible LoRA support), text-embedding caching across runs, lower-resolution presets.

#### B4. A/B protocol (every change)
Fixed model files, prompt, seed, steps, CFG and resolution; verify the device and that all blocks are resident.
At least five alternating A/B/A/B runs on the short workload; compare medians and spread, not the best run; then
the full target for finalists. Re-run the existing Qwen Image numerical check against `sd-cli` (same noise and
settings): a speedup that changes the result beyond tolerance is not a win. Keep a change only if it is measurably
faster; revert otherwise. Write measured numbers into `PerformanceLeague.md`-style records, dated.

#### B5. What iGPU timings cannot prove
A GPU path losing to or beating the CPU here describes this APU and driver, not a discrete GPU. A roughly 16 GB
placement budget is shared system memory, not VRAM.

### Status
- 2026-10-10: plan written. B0 code change built and under test.
- 2026-10-31 (ported from the retired workstream): Qwen Image has no backlog item in `00-current-work.md`, so this section is its only tracker. A Qwen Image row exists in `docs/STATUS.md` (🟢, 2026-09-27, CPU and Vulkan) and records real Qwen2.5-VL text conditioning as wired. Open here: the Q3_K_S 512x512 GPU sample, the five-run median, and a backlog pointer.

### Measured results (2026-10-10, Ryzen 5700G iGPU, Vulkan, Qwen Image Q3_K_S, 60/60 blocks resident)

Workload: 512x512, real Qwen2.5-VL encoder, seed 42. Reference: stable-diffusion.cpp `sd-cli --backend vulkan0`,
same files and settings. `STINGRAY_QWEN_PROFILE=1` (stage log) and `=ops` (per-group split) are new, opt-in.

| Stage (512x512, 2 steps, CFG 2.5 = 4 forwards) | Stingray | sd-cli (Vulkan) |
|---|---|---|
| Text encode (cond + uncond) | 2.0 s | 2.9 s |
| One DiT forward | about 24.4 s (407 ms/block) | about 20.6 s |
| First-forward weight upload | +5.4 s | lazy, 2.1 s |
| VAE decode | 9.0 s | 9.5 s |
| Total | 115.4 s (an earlier uninstrumented run: 138.2 s) | 94.7 s |

Also: 256x256, 8 steps, CFG 4: 151.3 s (recorded 159.4 / 161.1 s on 2026-09-24: no regression); 512x512 20 steps would
be 40 forwards, about 16 minutes of DiT time at the current rate.

Where the forward time goes (sum over 60 blocks, 1 forward, `=ops` split, perturbed by extra syncs): MLP up/down +
norm 13.7 s (57%), norm + QKV GEMMs 4.5 s (19%), out-proj + gate 2.0 s (8%), QK-norm + concat + RoPE 1.9 s (8%),
attention 1.5 s (6%), modulation 0.2 s (1%). The forward is GEMM-bound; effective GEMM throughput is about 700-1000
GFLOP/s, roughly a third of the iGPU's FP32 peak, and the C++ reference sits at about the same level.

Hypotheses checked against the data:
- **Workspace / RoPE cache churn: rejected.** The workspace is rebuilt every forward (cond and uncond text lengths
  differ: 12 vs 5 tokens) but costs about 40 ms of a 24,000 ms forward.
- **Per-block submit batching: not the cost.** Everything outside the 60 blocks is under 100 ms per forward.
- **Raster order of the quantized GEMM (so concurrent workgroups share the A slab): no effect.** Reverted.
- **Push-constant split-K inside the quantized GEMM: rejected.** It made MLP-down slower (125 ms vs 96-116 ms) and
  perturbed the other shapes. Reverted.
- **Real, repeatable finding.** Microbenchmark (M=1024, Q3_K weights): throughput falls with the activation's row
  stride, not with FLOPs: 971 GFLOP/s at K=3072 (N=3072), 927 at K=6144, 668 at K=12288. The same K=12288 product run as
  4 chunks of K=3072 in separate contiguous buffers took 77.6 ms instead of 112 ms (-31%). MLP-down (K=12288) is the
  affected shape; 58 of 60 blocks are Q3_K. Estimated win if the MLP hidden activation were kept as four contiguous
  3072-wide chunks: about 27-35 ms/block, 1.6-2.0 s per forward (6-8%), before the cost of summing the four partial
  outputs. Not implemented: it needs the MLP weights re-laid out into chunks at upload and a partial-sum step, for a
  gain that still leaves the forward above the reference.
- **Remaining gap to the reference (about 18% per forward)** is probably the GEMM kernel itself: ggml-vulkan's matmul
  uses FP16 shared-memory tiles, ours uses FP32. A redesign is the only lever with a large upside, and it is a kernel
  project with its own numerics decision (rule: approved numerics changes are documented everywhere).

---

## Research findings (2026-10-30)

Checked in code and vendor docs:

- **Subgroup pin on this iGPU.** The FLUX handoff asks whether `ComputePipeline.ShouldPinSubgroupSize32` fires on this device. The condition is at `ComputePipeline.cs` 67-70. The device comment in `Shaders.cs` (~4622) says the subgroup is fixed at 64, so the pin does **not** fire here. That is inferred from the comment, not from a runtime query. **Confirm by logging `MinSubgroupSize` and `MaxSubgroupSize` at startup.**
- **Cooperative-matrix GEMM (Phase 3 feasibility) does not apply to this iGPU.** RADV exposes `VK_KHR_cooperative_matrix` only on RDNA3 and newer; this machine is Vega-class (gfx90c). `VulkanBackend.cs` ~384 reads the extension but never enables it. Phase 3's FP16-tile route is the only route on this box.
- **Attention already uses subgroup shuffles.** `FlashAttention128` (`Shaders.cs` ~10580) relies on `subgroupShuffleXor` and needs a subgroup of at least 16. Any change to the subgroup pin must keep that working.

Not checked: the Qwen Image MLP-chunking, FLUX GEMM and Wan numbers. They are the doc's own measurements, and I did not re-run them.

**Next checks:** log the subgroup range on the iGPU; run the Phase 1 small-M kernel against a one-row matvec loop (the plan's own benchmark) before any Phase 3 work.

---

## Research findings (2026-10-31, deep pass)

Excerpts are in [research/2026-10-31-performance-snippets.md](../research/2026-10-31-performance-snippets.md).

- **Qwen Image chunked MLP is in code.** `QwenImageModel.cs`, `QwenImageGpuWeights.cs` and `QwenImageGpuWorkspace.cs` contain the chunk path (13 matches, 2026-10-31). The Part 4 claim is code-backed.
- **Subgroup and cooperative-matrix limits** (see the Vulkan GEMM doc): the FP16 tile project cannot use cooperative matrix on this iGPU, and the subgroup pin must follow the `requiredSubgroupSize` power-of-two bound ([Khronos](https://docs.vulkan.org/refpages/latest/refpages/source/VkPipelineShaderStageRequiredSubgroupSizeCreateInfo.html)).
- **Long-K throughput drop (Part 4, "stride" finding):** no external source confirmed a TLB or memory-channel cause. The doc's own experiment (contiguous chunks give -31%) is the evidence. It stays a measurement, not a cited mechanism.
- **Not checked:** the FLUX and Wan timings and the reference numbers. They are the doc's own runs.

---

## Thread state (from the retired workstreams folder, 2026-10-31: qwen-image-and-diffusion-performance.md)

### Correct state

- **Backlog:** Qwen Image has no backlog item in [../00-current-work.md](../../00-current-work.md). Items 8 and 9 there are FLUX, not Qwen Image. This file is the only place the thread is tracked.
- **STATUS:** a Qwen Image row exists in [../STATUS.md](../../STATUS.md) (🟢, 2026-09-27, CPU and Vulkan) and records real Qwen2.5-VL text conditioning as wired. Corrected 2026-10-31: the documentation-architecture plan's 'text conditioning remains open' line was stale and is fixed.
- **Baseline (2026-10-10, Ryzen 5700G iGPU, Vulkan, Q3_K_S, 60/60 blocks resident):** the measured-results table in the samples plan compares Stingray with `sd-cli --backend vulkan0`.
- **FP16-tile project, phases (see that plan's checklist):** Phase 1 (small-M kernel) kept. Phase 2(c) (chunked image-stream MLP) kept: about 2.6 s (10.7%) faster, gap to the reference about 6% (was about 18%). Phase 2(a) rejected. Phase 2 follow-ups stopped as cheap changes ran out.
- The five-run median for the Phase 1 plus 2(c) checkpoint is not recorded yet. The iGPU timings describe this APU, not a discrete GPU (CLAUDE.md rule 13).

### Open items

- [ ] Remaining image samples, in order: Qwen Image Q3_K_S at 512x512 on the GPU (the first attempt hit the 20-minute CPU cap); Wan 2.2 14B one frame; HunyuanVideo 720 fp8 one frame; FLUX.2-dev with Mistral-Small-3.2-24B at 512x512
- [x] Qwen Image text conditioning integration: wired per the STATUS row (2026-09-27)
- [ ] Five-run median for the Phase 1 plus 2(c) checkpoint
- [ ] FP16-tile phases 3 (new tiled kernel family; carries the numerics decision), 4 and 5
- [x] STATUS row for Qwen Image exists (2026-09-27); re-check it when the Q3_K_S 512x512 GPU sample lands

### Decisions for the owner

- The numerics decision that FP16-tile Phase 3 carries.

### Cadence

- Review after each dated measurement; record the numbers in `PerformanceLeague.md`, not only here.
- Re-read at least every two weeks while any item is open.
- Close when the STATUS row exists, the median is recorded, and each FP16-tile phase is kept or rejected in writing.
