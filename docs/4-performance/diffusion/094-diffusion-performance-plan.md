# 094 — Diffusion Stack Performance Plan (2026-09-19)

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

## Ground rules for this whole plan (per CLAUDE.md, restated so the loop doesn't drift)

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

## Checklist

### Phase 6 — FLUX.1: attack the remaining DiT-loop gap (currently ~3.1x behind C++ per the last profiled row)

- [ ] Real per-kernel GFLOP/s comparison: Stingray's `SgemmF16` (611 GFLOP/s measured) vs. what
      `stable-diffusion.cpp`'s GGML backend achieves on the identical iGPU for the identical matmul
      shapes — get a real number, not an assumption, by adding equivalent GFLOP/s instrumentation to
      the sd.cpp side if it doesn't already print one.
- [ ] Investigate on-GPU quantized matmul (skip the FP16-dequant-then-GEMM step) as the next lever,
      since GGML's advantage is believed to come from direct quantized cooperative-matrix ops. Scope
      real feasibility against Vortice.Vulkan's cooperative-matrix extension support before committing
      to an implementation.
- [ ] If implemented, real parity test first, then real end-to-end timing, documented.

### Phase 9 — Cross-model DRY + perf-doc consistency pass (per CLAUDE.md's own performance+DRY pass rule)

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

## Working log

(Append real findings/blockers here as the loop executes, dated, so this doc stays the source of
truth for what's actually been tried — same discipline as `PerformanceLeague.md` itself.)
