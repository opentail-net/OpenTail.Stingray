# 094 — Diffusion Stack Performance Plan (2026-09-19)

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

---

## Checklist

### Phase 0 — Documentation truth pass (fast, unblocks accurate planning)

- [ ] Run `Sd3BaselineTests.TestSd35GpuVsCpuParity` to completion, capture the real GPU-vs-CPU
      forward-pass timing it already prints, and add a real SD3.5 Vulkan row to `PerformanceLeague.md`
      (single-forward timing first; full 20-step end-to-end GPU timing is a separate item below).
- [ ] Correct the HunyuanVideo footnote in `PerformanceLeague.md` (~line 962-964): it IS wired into
      the CLI and has a real passing weights test; add a real row citing the 64.2s/32×32/1-step smoke
      timing measured during this plan's own audit, and mark full-scale timing as the open item (see
      Phase 4).
- [ ] Add one line to `PerformanceLeague.md`'s LTX-Video section clarifying that a real `ForwardGpu`
      GPU-resident path already exists in code (`LtxVideoGpuWeights.cs`) but has never been benchmarked
      because the DiT's own correctness bug (unbounded latent-std growth) makes any GPU timing not
      worth taking yet — so readers don't conclude "GPU work not started" when it's "GPU work started,
      blocked on upstream correctness."

### Phase 1 — SD3.5: **CRITICAL REGRESSION FOUND** — fix correctness before any more perf work counts

- [x] Real end-to-end SD3.5 Vulkan run, same config as the CPU baseline (256×256, 20 steps, seed 42,
      "a red apple on a wooden table") using the already-built `ForwardGpu` path — ran via the
      already-existing `Sd3BaselineTests.GenerateApple_20Steps_Sd35_Cpu`/`..._Vulkan`/`TestSd35GpuVsCpuParity`,
      2026-09-19. **Real numbers**: CPU 175.3s (down from the doc's stale 536.4s — 3.06× faster,
      consistent with the shared cross-model infra wins e.g. SDXL's own 2026-09-19 3.16× jump);
      Vulkan 78.1s warm / 87.1s cold (first-ever GPU timing for this model); GPU-vs-CPU forward parity
      cosine 0.999724, maxDiff 0.118420 (**notably higher maxDiff than every other model's parity
      check in this doc**, e.g. FLUX's ~5.5e-3 — flagged, not yet explained). Vulkan is only ~1.63×
      slower than the C++ reference (78.1s vs 48.01s) — much closer than any prior SD3.5 number on
      record, IF the output were correct.
  - **But visual inspection of both output PNGs (`sd35_medium_apple_cpu_256_20steps.png`,
    `sd35_medium_apple_vulkan_256_20steps.png`) shows garbled, incoherent color-block noise on BOTH
    backends — not a red apple, not "coherent geometric structure" as `PerformanceLeague.md` line 877
    currently (wrongly) claims.** The real C++ reference PNG
    (`sd35_medium_apple_cpp_vulkan_256_20steps.png`) is a genuine, correct, photorealistic apple —
    confirming this is a real regression in our own port, not a reference/prompt mismatch. This
    contradicts the doc's post-2026-09-05-fix claim and must be re-diagnosed: either a real regression
    landed after 2026-09-05's dual-attention-norm/VAE-scale/unpatchify/pos-embed fixes, or that
    correctness claim was itself never re-verified visually (the doc's own words hedge: "output ...
    matching the verified post-fix appearance in docs/057" — a citation, not a fresh look). The high
    parity maxDiff (0.118) between CPU and GPU, both producing garbage, suggests the bug is upstream
    of the backend split (shared `MMDiTModel.Forward` CPU math, or the scheduler/VAE common to both).
  - **Root-causing this is now the actual top priority for SD3.5** — every item below in this phase is
    blocked on it. This is exactly the "check the real reference before fixing code that looks wrong"
    situation CLAUDE.md warns about: bisect against `examples/stable-diffusion.cpp`'s real MMDiT
    forward stage-by-stage (same technique already used successfully for Z-Image's sign-convention bug
    and FLUX's T5-padding bug) rather than guessing.
- [ ] **Bisect the SD3.5 regression**: dump intermediate latents at each of the 20 steps (same
      `STINGRAY_ZIMAGE_DUMP_LATENT`-style env-var-gated pattern used for Z-Image) and compare divergence
      point against a step-by-step dump from `examples/stable-diffusion.cpp`'s MMDiT (`--diffusion-model`
      flag pattern already established for FLUX/SD1.5 C++ reference runs in this doc). Find the first
      step where output diverges meaningfully, then check that step's the exact ops against
      `examples/diffusers`' real `SD3Transformer2DModel` for the same stage.
- [ ] Once root-caused and fixed: re-verify with a fresh visual check (not just golden/numeric parity —
      this bug proves numeric-only checks can miss real breakage) and re-run both CPU and Vulkan timings.
- [ ] Re-verify output is visually coherent (not just non-crashing) before trusting any further timing.
- [ ] Compare against the already-captured C++ reference (`stable-diffusion.cpp` Vulkan: 48.01s total,
      MMDiT sampling 36.81s/1.75s-per-step) — compute the real ratio, same format as every other model
      in the doc.
- [ ] If the gap is large (expected, since this GPU path has apparently never been measured or tuned),
      apply the proven residency/fusion playbook from SDXL/SD1.5/FLUX (staging-copy reduction, fused
      QKV/RoPE/norm shaders, tiled attention now that Q/K/V will already be resident) — but only after
      the baseline number above justifies it. Re-measure after each change, same discipline as the
      SDXL Stage 0-5 arc.
- [ ] Document every stage's real number in `PerformanceLeague.md`, including any reverted regressions.

### Phase 2 — Qwen Image: first GPU port (biggest true gap — zero GPU code exists)

- [ ] Scope `QwenImageModel`'s architecture (attention head dims, channel counts, block structure) and
      identify which existing Vulkan primitives (from FLUX.1/FLUX.2/SD3.5's residency work) directly
      apply vs. need new shapes (cf. SD1.5 needing new `MultiHeadAttentionTiled40/80/160` variants for
      its non-64/128 head dims).
- [ ] Build `QwenImageGpuWeights`/`QwenImageGpuWorkspace` + `ForwardGpu`, following the
      Upload-once/resident-chain pattern (not per-op dispatch — that mistake has already been made and
      unmade three times in this codebase, don't repeat it).
- [ ] Real numerical parity test (GPU vs CPU forward, real weights) before any timing claim.
- [ ] Real end-to-end Vulkan timing vs the existing 348.4s CPU baseline. Document in
      `PerformanceLeague.md`. No C++ reference exists for Qwen Image in `examples/` — note that
      explicitly rather than fabricating a comparison.

### Phase 3 — FLUX.2: extend GPU residency past the double-blocks, re-test at real resolution

- [ ] The current GPU path only covers 8 double-blocks; 48 single-blocks + Mistral-24B text encoder +
      VAE are still CPU. Port `Flux2DiT`'s single-block loop to GPU residency using the same
      `Flux2GpuWeights`/`Flux2GpuWorkspace` infra, reusing the fused `Flux2QkvNormRope`/`SgemmSiluGate`
      shaders already built for double-blocks where the math matches.
  - Blocker note (fill in if hit): FLUX.2's single-block structure may differ enough from double-block
    (concatenated stream vs. separate img/txt streams) that shaders need real variants, not reuse —
    check `examples/flux2/src/flux2/model.py`'s `SingleStreamBlock` before assuming reuse works.
- [ ] Re-run the GPU-wired end-to-end test at production resolution (512×512, not 128×128) — the doc's
      own 2026-09-19 finding was that GPU lost to CPU specifically in the *small-token* 128px regime;
      confirm whether that reverses at real scale, per the doc's own hypothesis.
- [ ] Compare against `examples/flux2`'s real PyTorch reference (already used for the double-block
      microbenchmark — 22.23s/8-block-CPU) and, if buildable, `examples/stable-diffusion.cpp` if it
      gains FLUX.2 support; otherwise state plainly that no C++ yardstick exists yet for FLUX.2 at
      full-pipeline scale.
- [ ] Document every measured stage in `PerformanceLeague.md`.

### Phase 4 — HunyuanVideo: first real production-scale measurement

- [ ] Real end-to-end run via `stingray image` (not just the 32×32 unit-test smoke case) at a
      realistic resolution/frame count for this checkpoint, CPU first.
- [ ] Visually verify output coherence (this model has never been visually judged in this project per
      the doc's own history) before trusting any timing as a "working" row.
- [ ] Document real CPU timing in `PerformanceLeague.md`, replacing the incorrect "never attempted"
      line from Phase 0.
- [ ] No GPU path exists yet (confirmed by grep) — scope whether a `ForwardGpu` port is worth building
      given the CPU timing found, using the same residency playbook as everywhere else. If the CPU
      timing is large enough to justify it, build it; if blocked (e.g. checkpoint too large for
      am iGPU's VRAM budget), document the real memory numbers that make it impractical rather than
      guessing.
- [ ] No C++ reference for HunyuanVideo exists in `examples/stable-diffusion.cpp` per the doc's own
      2026-09-13 check — re-verify that's still true (a newer sd.cpp vendor drop may have added it)
      before repeating the "no reference" conclusion.

### Phase 5 — LTX-Video: fix the correctness blocker, then benchmark the existing GPU path

- [ ] Pick up the open diagnostic from `docs/077` (2026-09-14 entry): latent std grows ~3.5%/step
      unbounded across the denoising loop, suspected in `head.head`'s final projection scale. Verify
      against `examples/stable-diffusion.cpp`'s or the real LTX-Video HF reference's exact output-head
      scaling constant.
  - Blocker note (fill in if hit): if this needs a Python-side reference dump and no vendored
    reference produces one, state that precisely — do not add a new Python reference script
    (explicitly disallowed by `docs/061`/CLAUDE.md's coverage-tooling rule).
- [ ] Once fixed, re-run the existing golden/structural test suite (12 `[Fact]`s per the doc's
      2026-09-14 entry) to confirm no regression, then a real end-to-end visual re-run.
- [ ] Only once correctness is real: benchmark the already-built `LtxVideoModel.ForwardGpu` path
      end-to-end for the first time ever, and document real numbers (CPU already has one:
      100.5s/256×256/25-steps, but was never verified correct — re-verify at the same time).

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

### Phase 7 — Z-Image-Turbo: apply the proven GPU-residency playbook (currently barely faster than CPU)

- [ ] Port `ZImageDiT`'s per-op GPU calls (`ZImageGpuWeights.cs`/`ZImageGpuWorkspace.cs` already exist
      — check whether they already do residency or are per-op like FLUX.1 was before its 2026-09-13
      residency work) to the Upload-once/resident-chain pattern.
- [ ] Real end-to-end Vulkan timing vs. the existing 183.6s baseline (256×256, 4 steps).
- [ ] No C++ reference exists for Z-Image in `examples/` — document that plainly rather than
      fabricating one.

### Phase 8 — Wan2.1/Wan2.2: close remaining gap, check Wan2.2 coverage exists at all

- [ ] Wan2.1-T2V-1.3B is at 2.2x behind C++ at production scale (132.8s vs 60.3s) — the doc's own
      profiling already attributes most of the gap to UMT5-XXL text encoding (2.98x slower) and VAE
      decode (3x slower) rather than the DiT loop itself. Attack the two already-identified real
      bottlenecks (T5 streaming-from-disk vs. resident GGUF-quantized loading; VAE decode kernel
      efficiency) rather than re-touching the DiT loop, which is already close to parity.
- [ ] Check whether `WanModel`/`WanPipeline` supports Wan2.2 checkpoints at all (different
      architecture/config vs 2.1, or just a different checkpoint under the same class?) — grep the
      real HF repo's config before assuming compatibility. If genuinely unsupported, that's a coverage
      gap, not a perf gap — scope it separately, don't conflate.

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
