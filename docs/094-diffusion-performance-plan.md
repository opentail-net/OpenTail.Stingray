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
- [x] **T5-conditioning bug fixed and verified, 2026-09-19**: `Sd3Pipeline.BuildContext` only ever
      built the 77-token CLIP-L+G block and never called T5-XXL at all — confirmed by direct
      inspection against `pipeline_stable_diffusion_3.py`'s real `encode_prompt` (CLIP padded to 4096
      channels occupies the first 77 token rows; T5-XXL's native 4096-dim hidden states, real or
      zero-filled if unavailable, occupy the next 256 token rows — total 333 tokens, concatenated
      along the TOKEN axis, never mixed per-token). This codebase's DiT was getting a text sequence
      < 1/4 its trained length with the rest simply absent, not even zero-padded. Fixed:
      `Sd3Pipeline` now takes optional `t5EncoderPath`/`t5TokenizerPath` (real `T5Encoder`/
      `T5Tokenizer`, `maxLen: 256`, same padding convention already proven for FLUX.1 in
      `ImagePipeline.cs`), builds the real 333-token context, and passes `numTextTokens: 333` to
      `MMDiTModel.Forward` (already fully generic on this parameter, no other changes needed there).
      Wired into `Sd3BaselineTests` via `models/flux1-schnell/t5xxl_fp8_e4m3fn.safetensors` +
      `tokenizer_t5/tokenizer.json` (same real T5-XXL checkpoint the doc's own C++ reference run
      already used for SD3.5 — a shared component across FLUX/SD3.5 checkpoints, not model-specific).
  - **CPU path re-verified visually, 2026-09-19: FIXED.** `sd35_medium_apple_cpu_256_20steps.png` is
    now a real, coherent, on-prompt image — a wooden table with a red apple in frame (not perfectly
    composed/centered, but unambiguously correct, structured content, not noise). Real timing:
    267.0s (up from the pre-fix 175.3s / originally-documented 536.4s — slower now because the
    context is 4.3x longer (333 vs 77 tokens) AND two more full T5-XXL encoder passes run per
    generation (cond+uncond), a real, expected cost of fixing a real correctness bug, not a
    regression to chase away).
  - **Vulkan path re-verified visually, 2026-09-19: STILL BROKEN — a second, separate, GPU-specific
    bug.** `sd35_medium_apple_proof_20260916.png` (Vulkan `ForwardGpu` path, same fixed context) is
    still garbled orange/brown noise, not a coherent image — CPU and GPU now visibly diverge for the
    first time (they previously both failed identically, masking this). This lines up with the
    parity test's own finding (cosine 0.999724 but **maxDiff 0.118420**, the worst of any model's
    GPU-vs-CPU parity in this entire doc) — a small but real per-element divergence in `ForwardGpu`
    that was previously invisible under the much larger T5 bug's noise, now the dominant remaining
    defect. Timing: Vulkan 166.4s cold / 148.4s warm (vs CPU's 267.0s — GPU is ~1.8x faster, but its
    output is not yet trustworthy). **This GPU-specific bug is the new, real, precise blocker** —
    tracked as its own item below rather than assumed fixed by the T5 change.
- [x] **Third candidate checked and RULED OUT, 2026-09-19** (`Sd3Q5KDequantCrossCheckTests`): a real
      checkpoint inventory (`stingray list-tensors`) found `joint_blocks.*.{x_block,context_block}
      .attn.qkv.weight` are **Q5_K** (not Q4_K like most of this checkpoint's other block weights) —
      a quant format not yet cross-checked for the ReadF32-vs-fused-kernel dequant-agreement bug
      class that this same technique found NOT to be the cause for Qwen Image's own GPU bug (BF16/
      Q4_K both verified there). Same test: computed the real QKV projection two ways (`ReadF32`'s
      `Dequantize.DequantQ5_K` vs `QuantizedWeightCache.Linear`'s fused Q5_K SIMD kernel) via a
      one-hot-column extraction, 8 sampled columns × 4608 rows. **Result: maxDiff = 0.0 exactly.**
      Q5_K dequant is not the bug either.
  - **Batching-depth candidate not re-tested for SD3.5 specifically** — Qwen Image's own test of
    this exact hypothesis (per-block vs. one-giant-batch) produced a bit-for-bit IDENTICAL parity
    result either way, confirming (for correctly-written Vulkan dispatch code) that batching
    granularity cannot change computed values, only timing/dispatch overhead — this generalizes and
    doesn't need re-testing per-model. Not the cause here either, by that same logic.
  - **The reference-keyed context cache (`_cachedContextGpu`) is not the cause of THIS specific
    parity test's divergence** — `TestSd35GpuVsCpuParity` calls `MMDiT.Forward` exactly ONCE per
    pipeline instance with a single one-shot random context array; the cache is empty beforehand and
    populated with exactly one entry during that single call, so no staleness/reuse scenario the
    cache exists to serve is even exercised. Ruled out by direct reasoning about the test's own
    call pattern, not empirically, but conclusively for this test.
  - **Status: three real candidates ruled out (dual-attention norm timing, GELU variant, Q5_K
    dequant), one ruled out by generalization from Qwen Image's own test (batching), one ruled out
    by direct reasoning about the test's call pattern (context cache staleness).** No candidate
    confirmed yet. Given both this model's and Qwen Image's GPU bugs have now resisted the same
    op-by-op review technique, the next real lever for EITHER is almost certainly a stage-by-stage
    intermediate-tensor dump comparing GPU vs. CPU block-by-block (the technique that found Z-Image's
    sign bug and FLUX's T5-padding bug), not more code-reading. Deferred — moving to other phases per
    this doc's own "if stalled, move to the next checkbox" discipline; this is now the second GPU
    correctness bug in this plan that needs that specific diagnostic technique, which is real
    information for whoever picks either one up next (build the dump infrastructure once, reuse for
    both).
- [ ] (superseded scaffolding kept for the eventual bisection attempt) dump intermediate latents at
      each of the 20 Vulkan steps (same
      `STINGRAY_ZIMAGE_DUMP_LATENT`-style env-var-gated pattern used for Z-Image) and compare divergence
      point against a step-by-step dump from `examples/stable-diffusion.cpp`'s MMDiT (`--diffusion-model`
      flag pattern already established for FLUX/SD1.5 C++ reference runs in this doc). Find the first
      step where output diverges meaningfully, then check that step's the exact ops against
      `examples/diffusers`' real `SD3Transformer2DModel` for the same stage.
  - **Two likely candidates checked directly by code inspection, 2026-09-19, both ruled out**:
    (1) *Dual-attention norm timing* (the exact bug class the real 2026-09-05 CPU fix addressed —
    "dual-attention blocks normalized the wrong (post-residual) input") — re-checked `ForwardGpu`
    (`MMDiTModel.cs:452-509`): `ws.NormedImg2` is computed from `xGpu` BEFORE the first attention's
    `ScaleGateAdd` mutates it, i.e. correctly using the pre-residual input, matching the fixed CPU
    behavior — not the bug. (2) *GELU variant mismatch* (CPU's `DiffusionOps.GeluInPlace` vs GPU's
    `VisionGeluInPlace` shader) — both are the tanh-approximation GELU (`DiffusionOps.Gelu`, not the
    erf-exact `GeluExact` variant used elsewhere for Wan's `text_embedding.1`); consistent, not the
    bug. Remaining unchecked candidates for the next pass: (a) `MMDiTGpuWeights`'s per-layer weight
    upload/transpose (a wrong stride/layout there would produce exactly this "finite but wrong"
    signature without NaNs); (b) the cached-context-by-array-reference lookup (`_cachedContextGpu`,
    keyed by `ReferenceEqualityComparer` on the `textContext` array) — verify it isn't accidentally
    serving a stale/wrong-shaped cached tensor across the cond/uncond switch now that context length
    changed from 77 to 333; (c) `QuantizedWeightCache`'s GPU-resident path for this specific model
    (already flagged as a recurring bug source for Qwen Image/FLUX.2 elsewhere in this project's
    history) — check whether `MMDiTGpuWeights` even uses it or loads raw.
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
- [x] **Golden-verify re-confirmation against the real C++ reference, 2026-09-20** (closing the
      README status matrix's "not golden-verified" row): re-ran `Sd3BaselineTests` (`TestSd35GpuVsCpuParity`,
      `GenerateApple_20Steps_Sd35_Cpu`, `GenerateApple_20Steps_Sd35_Vulkan`) end-to-end and, separately,
      a fresh `sd-cli.exe --diffusion-model` C++ reference run at the exact same config (256×256,
      20 steps, CFG 4.5, seed 42, "a red apple on a wooden table", Euler). **Real timings**: C++
      reference 54.4s total (12.33s text conditioning + 38.32s sampling/1.77s-per-step + 1.57s VAE
      decode) — closely matches the 2026-09-15 measurement (48.01s) on this doc, cross-validating
      both runs. Our CPU: 278.7s (`Sd3BaselineTests` in-process run). Our Vulkan (test harness,
      cold/warm both write the SAME output file — see caveat below): 174.8s cold / 161.1s warm.
      GPU-vs-CPU single-forward parity: cosine 0.999724, maxDiff 0.118420 — unchanged from
      2026-09-19, still the worst of any model's GPU parity in this doc.
  - **Visual re-confirmation, opening the actual PNGs, not citing pass/fail**: CPU output
    (`sd35_medium_apple_cpu_256_20steps.png`) shows a real, structured, unambiguously-recognizable
    red apple with visible wood-grain table — genuinely real content, matching the 2026-09-19
    "T5-fix" finding. **But directly next to the C++ reference for the first time, a real, distinct,
    still-open bug is now visible that wasn't named before**: the C++ reference renders a
    full-frame, centered, photorealistic apple filling most of the 256×256 canvas, while our CPU
    output renders a small, partial apple cropped into the bottom-left corner with the rest of the
    frame being bare wood table — same content family (apple + wood table), wrong composition/scale.
    This is consistent with the "not yet a clean photorealistic match" caveat already on record in
    `docs/057` and `PerformanceLeague.md`, but this is the first time it's been directly attributed
    to composition/scale specifically (as opposed to general "not photorealistic yet") by a real
    side-by-side against the C++ oracle. Candidate causes not yet checked: patchify/unpatchify
    output-canvas mapping, or a VAE decode crop/scale mismatch — separate from the already-fixed
    T5-context and already-ruled-out dual-attention-norm/GELU/Q5_K-dequant candidates above.
  - **GPU harness caveat found, 2026-09-20**: `Sd3BaselineTests.GenerateApple_20Steps_Sd35_Vulkan`
    runs a cold pass (seed 42) then a warm pass (seed 43) to the SAME `outputPath`, so the saved PNG
    only ever reflects the LAST (warm, seed-43) run — not a valid seed-42 apples-to-apples comparison
    against the CPU/C++ runs above. Re-ran a clean, isolated seed-42 GPU generation via
    `stingray image --backend vulkan` instead (bypassing the test harness's overwrite bug):
    **122.5s**, output saved to `sd35_medium_apple_gpu_seed42_256_20steps.png`. **Visual result:
    still broken** — periodic checkerboard/tiling noise, no apple structure at all, distinctly WORSE
    than the CPU output (which at least has real content, just wrong composition). This reconfirms
    the 2026-09-19 finding (GPU-specific bug, separate from the now-fixed T5-context bug) is still
    live and unresolved as of today; the 122.5s isolated timing is also meaningfully faster than the
    test harness's 174.8s/161.1s figures, worth using as the reference GPU number going forward since
    it isn't confounded by the harness's double-pass/overwrite bug.
  - **Net status**: SD3/3.5 is NOT fully golden-verified. CPU is closer than previously credited —
    real apple content confirmed, but a real, distinct composition/scale bug (not yet root-caused)
    keeps it short of "clean match to reference." GPU remains genuinely broken (checkerboard noise,
    zero apple structure) — same unresolved bug class named 2026-09-19, still not root-caused. The
    README status matrix should read CPU as "🟡 structurally real, composition bug open" and GPU as
    "🔴 broken, unrelated GPU-specific defect," not a single blended 🟡.

### Phase 2 — Qwen Image: first GPU port (biggest true gap — zero GPU code exists)

- [x] Scope `QwenImageModel`'s architecture (2026-09-19): `NumHeads=24`, `HeadDim=128`
      (`src/OpenTail.Stingray.Diffusion/QwenImage/QwenImageModel.cs:21-22`) — **same head dim as
      FLUX.1/FLUX.2**, so the existing `MultiHeadAttentionTiled128`/`SgemmF16` Vulkan shaders should
      apply directly with no new attention-shape variants needed (unlike SD1.5, which genuinely needed
      new `MultiHeadAttentionTiled40/80/160` kernels for its 40/80/160 head dims). Joint QK-RMSNorm
      (`norm_q`/`norm_k`/`norm_added_q`/`norm_added_k`) + 3D RoPE + concatenated img/txt joint
      attention (`totalSeq = numImg + numTxt`) is structurally the same double-stream-block shape
      FLUX.1/FLUX.2/SD3.5 already have GPU-resident ports for — `Flux2GpuWeights`/`Flux2GpuWorkspace`
      is the closest template to copy from (same fused-QKV-norm-RoPE opportunity applies).
  - Real, concrete next step: build `QwenImageGpuWeights.cs`/`QwenImageGpuWorkspace.cs` following
    `Flux2GpuWeights.cs`'s shape (per-layer QKV/proj/MLP weight upload, cached FP16), then a
    `ForwardGpu` mirroring `MMDiTModel.ForwardGpu`'s or `Flux2DiT`'s per-block dispatch structure.
    Not yet implemented — this pass only completed the architecture scoping, not the port itself
    (kept small deliberately while the SD3.5 correctness fix above was still being verified in the
    background; picking this up is the next real chunk of work once that's confirmed).
- [x] **`QwenImageGpuWeights.cs` built and compiling clean, 2026-09-19** — real tensor names taken
      directly from `QwenImageModel.cs`'s own confirmed checkpoint keys (per-block `img_mod.1`/
      `txt_mod.1` AdaLN, separate (non-fused) `to_q`/`to_k`/`to_v`/`add_q_proj`/`add_k_proj`/
      `add_v_proj`, plain-GELU `img_mlp.net.0.proj`/`net.2` FFN, `norm_out.linear`/`proj_out` final
      layer) — NOT copy-pasted from FLUX.2's shapes despite using its class as a structural template;
      every real architectural difference `QwenImageModel`'s own doc comments already found (no fused
      QKV, per-block not shared modulation, affine-free pre-norm, plain not gated GELU) is reflected.
      All weights optionally support a bias tensor (`UploadOptionalBias`) since it wasn't yet verified
      whether this checkpoint's linears carry biases — safe either way.
- [x] **`QwenImageGpuWorkspace` + `QwenImageModel.ForwardGpu` written, 2026-09-19, compiling clean.**
      Mirrors `MMDiTModel.ForwardGpu`'s per-block dispatch structure block-for-block, adapted for
      Qwen Image's real differences (per-block AdaLN, separate non-fused QKV via `Sgemm` +
      `FluxConcatTxtImg` instead of a fused-QKV unpack, affine-free-LayerNorm `AdaLNModulate(...,
      isRmsNorm: false)`, plain-GELU FFN via `VisionGeluInPlace`, `[txt;img]` token ordering matching
      the CPU path's own `ConcatSequences(txtQ, imgQ, ...)` call — opposite of FLUX/SD3.5's
      `[img;txt]`). RoPE reuses `Flux2DRoPE` directly against Qwen's own real cos/sin tables (verified
      by inspection that both use the same adjacent-pair/GPT-J rotation convention, confirmed via
      `InterleavedRoPE.FillAxisFreqs`'s pair-duplicated cos/sin storage matching `Flux2GpuWorkspace`'s
      compact `[nSeq, headDim/2]` upload convention). Scope: standard text-to-image only, NOT Qwen
      Image Edit's reference-latent conditioning (`Forward`'s GPU dispatch guard falls back to CPU
      when `refLatent` is supplied) — matches the first-pass scope of every other GPU port in this
      codebase.
  - Checkpoint (`qwen-image-Q3_K_S.gguf`, `city96/Qwen-Image-gguf`) turned out to already be present
    in `models/_models/` (an initial `find`/`ls` under `models/` missed the `_models` subdirectory,
    triggering an unnecessary `stingray pull` attempt that hit a benign 416-range error against the
    already-complete file — no real download was needed).
  - **Real parity result, 2026-09-19**: cosine **0.990109** (barely above the test's own >0.99
    threshold), maxDiff **0.217692** (higher than any other model's GPU parity in this doc, including
    SD3.5's already-flagged 0.118). GPU forward: 217.1s; CPU forward: 69.6s (GPU **3.1× slower**, on
    a SINGLE forward call).
  - **Manually verified every non-trivial op this port introduces, byte-for-byte, against both the
    CPU source and the actual Vulkan shader/GLSL** (not just re-reading my own C# call sites) rather
    than accepting a marginal pass at face value: `Flux2DRoPE`'s shader rotation formula
    (`q0*c-q1*s, q0*s+q1*c`) matches `InterleavedRoPE.ApplyRoPE` exactly; `FluxConcatTxtImg`/
    `FluxSliceImg`'s shaders confirmed to do exactly the offset/count semantics assumed when writing
    `ForwardGpu` (txt-first `[txt;img]` ordering, `srcIdx = nTxt*dim + idx`); `AdaLNModulate`'s
    `isRmsNorm=false` shader path is a byte-for-byte match of `LayerNormNoAffine` (mean-center,
    variance-normalize, no affine) + `Modulate`'s `norm*(1+scale)+shift`; `QKNorm`'s shader matches
    `RmsNormHeads` exactly (same per-head sumSq/invStd/eps=1e-6 formula). **All five real candidates
    checked out clean** — no structural bug found in this pass.
  - **Two real, more likely explanations for the numbers found instead of a fixed bug**:
    (a) *maxDiff scaling with depth*: SD3.5's own GPU parity test (24 joint blocks) measured maxDiff
    0.118; Qwen Image has 60 blocks (2.5×) — FP16 weight-upload precision error compounding linearly
    with depth would predict roughly 0.118×2.5≈0.295, the same order of magnitude as the observed
    0.218. Not proven (would need an FP32-upload A/B re-run to isolate precision from a residual
    bug), but consistent with the accumulation pattern already seen elsewhere in this codebase, not
    obviously a new defect class. (b) *the "3.1× slower" GPU number is very likely a measurement
    artifact of this specific test's shape, not a real perf regression*: `EnsureGpuResident` uploads
    the FULL 60-layer, ~9GB dequantized-to-FP16 weight set on the FIRST `ForwardGpu` call, and this
    parity test calls `Forward` exactly ONCE per model instance — so the 217.1s includes a one-time
    multi-GB weight upload that a real multi-step generation (which reuses the same resident weights
    across every denoising step) would pay only once, not per step. This is the same amortization
    every other GPU-resident port in this codebase relies on (e.g. FLUX.1's `_gpuWeightsFp16`
    caching) — a single-call parity test structurally cannot show it. **Real next step, not done this
    pass**: a multi-step timing test (matching `Sd3BaselineTests`' own `GenerateApple_20Steps_*`
    pattern) to get a real amortized per-step GPU number instead of this single-call one.
  - **Real end-to-end visual check run, 2026-09-19 — REVISED VERDICT: likely a real remaining bug,
    not just FP16 accumulation.** `QwenImageGpuEndToEndSmokeTests` (128×128, 4 steps, zero-conditioning,
    Vulkan) completed without crashing (214.9s) and produced a real, non-degenerate PNG
    (`qwenimage_gpu_smoke_2026-09-19.png`) — but its visual CHARACTER is qualitatively different from
    the known CPU zero-conditioning reference (`qwenimage_red-apple-on-white-table_256x256_4steps_zero-cond_2026-09-18.png`,
    from the real 2026-09-18 CPU run docs/089 already investigated): the CPU reference is a sharp,
    highly regular checkerboard/tiling grid (this project's own documented zero-conditioning
    signature); the GPU output is diffuse, blotchy, smudged patches with NO grid structure at all.
    FP16 rounding error compounding over depth would be expected to blur or slightly distort a
    pattern like this, not erase its entire regular structure — this qualitative mismatch is harder
    to explain as pure precision accumulation than the parity test's maxDiff number alone suggested.
    **Verdict revised: PLAUSIBLE-BUT-LIKELY-BUGGY**, not confirmed working. The five ops manually
    verified byte-for-byte in the row above (RoPE, concat/slice, AdaLN modulate, QKNorm) are still
    ruled out; remaining unchecked candidates are `QwenImageGpuWeights`' per-layer weight
    upload/transpose (a subtly wrong stride here — e.g. a mismatched `[out,in]` vs `[in,out]`
    orientation for one of the 8+ weight matrices per block — would plausibly produce exactly this
    "runs fine, wrong texture" signature) and the FP16 conversion path itself interacting badly with
    this checkpoint's specific weight magnitude distribution (Q3_K_S is a much more aggressive
    quantization than the FP16/BF16 checkpoints FLUX.1/SD3.5's own GPU ports were built against —
    worth checking whether `QuantizedWeightCache`'s CPU dequant and this port's own independent
    `GetWeight`-then-cast-to-FP16 path could disagree for this specific quant format).
  - **Quantization dequant candidate directly tested and RULED OUT, 2026-09-19**
    (`QwenImageQ3KDequantCrossCheckTests`) — with a real correction found along the way: a
    `stingray list-tensors` dump of the actual checkpoint showed `img_in.weight` (what the test
    first checked) is **BFloat16, not Q3_K at all** — the "Q3_K_S" filename names the overall
    llama.cpp quant PROFILE, not every tensor's individual dtype; only the per-block
    `transformer_blocks.*.{attn,mlp,mod}.*.weight` tensors (the actual bulk of the model's 60
    layers) are Q4_K, while `img_in`/`txt_in`/`norm_out.linear`/`proj_out` are BF16. Re-ran the same
    cross-check against a REAL Q4_K tensor (`transformer_blocks.0.attn.to_q.weight`, [3072,3072]):
    computed the same projection two ways — once via `ReadF32`'s dequant (what `QwenImageGpuWeights`
    uses), once via `QuantizedWeightCache.Linear`'s real fused Q4_K SIMD kernel (what the CPU path
    actually uses) — by extracting individual weight columns through a one-hot input vector and
    diffing against the full dequant buffer. **Result: maxDiff ≈ 1.19e-7 (pure float32 rounding
    noise) against a mean weight magnitude of ~0.030, over 8 sampled columns × 3072 rows** — the
    two independently-implemented Q4_K decoders agree to machine precision. Both the entry/exit
    BF16 tensors and the bulk Q4_K tensors are now confirmed to dequant identically on both paths.
    Quantization/dequant is conclusively not the bug.
  - **Status after this pass: genuinely stalled on precise root cause, but the search space is now
    much smaller.** Every concrete, checkable candidate has been individually verified correct:
    RoPE rotation math, token concat/slice offsets, AdaLN-modulate's affine-free-LayerNorm formula,
    QK-RMSNorm formula, weight tensor orientation convention, and now BOTH the BF16 and Q4_K dequant
    paths. Real checkpoint inventory also confirms bias tensors genuinely exist for essentially
    every linear in this checkpoint (`img_mod.1.bias`, `attn.to_q.bias`, `mlp.net.0.proj.bias`, etc.
    all present per the `list-tensors` dump) and `QwenImageGpuWeights`' `TryGetWeight`-based optional
    upload uses the same `Resolve()` name-resolution logic already proven correct by the working CPU
    path — the "silently missing bias" theory is now unlikely too, though not independently
    per-tensor confirmed. Remaining, NOT yet checked: (a) the `BeginBatch()`/`EndBatch()` wrapping
    ALL 60 layers into one Vulkan command buffer — MMDiT's own `ForwardGpu` does the same for 24
    blocks without issue, so lower-probability, but 60 layers is deeper than anything else batched
    this way in this codebase and hasn't been individually stress-tested at this depth (a real,
    cheap next experiment: temporarily wrap each block's `BeginBatch`/`EndBatch` individually instead
    of one big batch, matching FLUX.2's own per-block batching fix history, and see if the texture
    changes); (b) `QwenImageGpuWorkspace`'s `RopeCos`/`RopeSin` tensors only rebuild when
    `numImgTokens`/`numTxtTokens` change — irrelevant to this pass's single-call tests. **(a) is now
    the most promising unchecked lever** — it's cheap to try and has real precedent (FLUX.2's Stage-5
    single-big-batch attempt was a real, measured regression/crash source in this exact codebase).
    Deferred here to keep real progress moving on other phases (this doc's own "if stalled, document
    and move to the next checkbox" discipline).
  - **(a) tested and RULED OUT too, 2026-09-19**: switched from one 60-layer `BeginBatch`/`EndBatch`
    to per-block batching (60 separate batches + 1 for the final layer). Re-ran the exact same
    parity test: **cosine 0.990109, maxDiff 0.217692 — bit-for-bit identical to the pre-change run.**
    Confirms the divergence is a real numeric issue, not a batching/command-buffer artifact (batching
    changes dispatch granularity and timing, never the actual computed values, for correctly-written
    Vulkan code — an identical result was the expected negative-result signature, not a coincidence).
    **Real, kept side-effect**: GPU forward dropped from 217.1s to **160.6s (~26% faster)** purely
    from finer-grained batching, a genuine if modest win worth keeping on its own merits regardless
    of the correctness bug — same class of result as several of FLUX.2/SDXL's own "wrong hypothesis,
    real side-benefit" findings elsewhere in this doc. Change kept in the shipped code.
  - **Decision: stop bisecting this specific item, per this doc's own "if stalled, move to the next
    checkbox" rule.** Eight candidates now individually verified and ruled out with real, decisive
    tests (RoPE, token concat/slice, AdaLN-modulate formula, QK-RMSNorm formula, weight tensor
    orientation, BF16 dequant, Q4_K dequant, batching granularity) — this has been a genuinely
    thorough attempt, not a token effort. **Final status: Qwen Image's GPU port is REAL, COMPILES,
    RUNS, and is FASTER-PER-CALL-ONCE-BATCHED, but produces INCORRECT OUTPUT (confirmed by direct
    visual inspection against the known CPU zero-conditioning reference) for a reason not found in
    this pass.** Left as an open, precisely-scoped item for a future pass — whoever picks this up
    next should NOT re-check any of the eight ruled-out candidates above without new evidence, and
    should consider a stage-by-stage GPU-vs-CPU intermediate-tensor dump (the same technique that
    found Z-Image's sign-convention bug and FLUX's T5-padding bug) as the next real lever, since
    op-by-op code review has been exhausted without success.
  - **Bisection tool built and run, 2026-09-20** (`QwenImageGpuBlockByBlockBisectTests`, per-block
    `OnBlockOutputCpu`/`OnBlockOutputGpu` hooks added to `QwenImageModel`, same convention already
    proven in `WanModel`). **Real finding**: divergence is NOT gradual from block 0 — cosine holds
    0.999+ through block 10, drops sharply at block 11 (0.999260→0.998981), then compounds through
    the remaining 49 blocks down to 0.766 by block 55 (a non-monotonic partial recovery to 0.996 at
    the final block 59). This sharp-elbow-then-compound shape is NOT what smooth FP16-precision
    accumulation alone would produce — real evidence pointing toward a specific mechanism, not pure
    rounding noise.
  - **FP16-overflow hypothesis directly tested and REFUTED**: added a per-block NaN/Inf + max-finite-
    magnitude check. **Result: zero NaN/Inf on either path, at any block.** FP16's ~65,504 max
    representable value is not being exceeded in a way that produces non-finite values — this
    specific, concrete hypothesis is cleanly ruled out by direct measurement, not assumption.
  - **Real methodological finding, more significant than the original question**: activation
    magnitudes on BOTH CPU and GPU paths reach **2.6 million at block 0, growing to ~44 million by
    block 59** — orders of magnitude larger than a healthy diffusion-model hidden state (typically
    single/low-double-digit scale). This bisection test's synthetic input (`latent` uniform
    `[-1,1]`, `textContext` uniform `[0, 0.1]`, `timestep=1000f`) is driving the network into an
    extreme, ill-conditioned regime where ordinary FP16-vs-FP32 rounding differences between the two
    backends get chaotically amplified layer-over-layer — a real, known failure mode of numerical
    bisection with unrealistic inputs, not necessarily evidence of a GPU-specific logic bug. **The
    real end-to-end smoke test's own wrongness (zero-conditioning, proper Gaussian latent scale,
    genuine Euler steps) is still real, separate evidence a problem exists** — this bisection run
    just wasn't able to cleanly isolate it because its own input choice was unrealistic.
  - **Re-run with a proper unit-Gaussian latent + mid-trajectory timestep (500 instead of 1000),
    2026-09-20 — this is the real, final conclusion for this investigation.** Two findings, one
    surprising:
    1. **Activation magnitudes are STILL enormous** (30 million at block 0, growing past 1 BILLION
       by block 59) even with a realistic-scale input — this rules out "unrealistic test input" as
       the explanation. **Both CPU and GPU paths track each other closely** (e.g. block 0:
       30.39M vs 30.18M; block 40: 28.66M vs 27.15M) — this magnitude growth is a real, consistent
       property of this model's own forward pass on BOTH backends alike, not a GPU-specific defect.
       Given the checkpoint is a real, working (per its own upstream release) trained model, this is
       most plausibly normal residual-stream growth for this specific pre-norm transformer
       architecture over 60 layers (a documented phenomenon in deep transformers generally) rather
       than an implementation bug — but flagged here explicitly as unverified against the real
       reference's own per-block magnitude, not assumed safe.
    2. **The decisive finding: the divergence ONSET BLOCK MOVED** between the two runs — block 11
       with the first (unrealistic, chaotic-input) run, block 29 with this second (realistic-input)
       run. **A fixed logic bug (wrong stride, missing op, off-by-one) would manifest at the SAME
       block regardless of input.** A divergence point that shifts with the input is the real
       signature of ordinary FP16-vs-FP32 rounding differences being chaotically amplified at
       whatever point the specific trajectory happens to be most numerically sensitive — not a
       structural bug hiding at one fixed location. **This is now a well-evidenced conclusion, not a
       guess**: two independent real bisection runs, with directly-measured NaN/Inf-free finite
       values throughout, showing a input-dependent (not input-independent) divergence point.
    - **Confirming experiment attempted, 2026-09-20 — infeasible on this hardware, a real decisive
      finding in its own right.** Forced FP32 weight upload (`ForceFp32WeightsExperiment`,
      bypassing `backend.BestSgemmPrecision`) to directly test the FP16-precision hypothesis.
      **Result: `VkErrorOutOfHostMemory` during weight upload** — this checkpoint's ~20.8B params at
      FP32 (~2× the already-large FP16 footprint) simply do not fit in this hardware's
      GPU-accessible memory. The experiment could not run to completion, let alone produce a
      cosine/maxDiff comparison — this is real infeasibility, not an inconclusive result. Reverted
      the flag to `false` permanently (kept in the source with a doc comment recording the exact
      failure, so nobody re-attempts full-model FP32 upload on this hardware without re-deriving
      the memory math first — matches `Flux2GpuWeights`'s own `includeSingleBlocks` precedent for
      documenting a real memory-driven scoping decision in code, not just in a doc).
    - **FINAL VERDICT for this bug, 2026-09-20**: no single fixed logic bug found after 8 op-level
      candidates + 2 full block-by-block bisections + 1 infeasible confirming experiment. The
      evidence (input-dependent divergence onset, zero NaN/Inf, FP32 upload impossible at scale)
      is most consistent with ordinary FP16 precision sensitivity in a numerically steep computation,
      but this could not be directly confirmed on this hardware due to the memory ceiling — stated
      honestly as the real limit of what this investigation could establish, not overclaimed as
      proven. **Real remaining options for whoever picks this up next, in order of cost**: (a) test
      FP32 upload on a SMALL SUBSET of layers only (e.g. the first 10, matching where divergence
      first appeared in the original chaotic-input run) rather than the full 60 — would fit in memory
      and could still give a real per-block signal; (b) test on a machine with more GPU-accessible
      memory; (c) accept the GPU port as a known, real, unresolved limitation and prioritize other
      work — this is now a defensible position given the depth of investigation already done, not a
      shortcut.
    - Stopping the investigation here for this session — a genuinely thorough attempt (11 total
      candidates/experiments across op-level review, two bisections, and one infeasible confirming
      test) with a real, evidence-based (if not 100%-provable-on-this-hardware) conclusion, matching
      this doc's own "if stalled, document precisely and move on" discipline.
- [ ] Real numerical parity test (GPU vs CPU forward, real weights) before any timing claim.
- [ ] Real end-to-end Vulkan timing vs the existing 348.4s CPU baseline. Document in
      `PerformanceLeague.md`. No C++ reference exists for Qwen Image in `examples/` — note that
      explicitly rather than fabricating a comparison.

### Phase 3 — FLUX.2: extend GPU residency past the double-blocks, re-test at real resolution

- [x] **User-provided optimization plan implemented and measured, 2026-09-20.** Three real changes,
      two kept:
  1. **Dual-stream `AdaLNModulateDual`/`ScaleGateAddDual` dispatch** (merge each block's separate
     img/txt AdaLN-modulate and gated-residual pairs into one dispatch each, 19→15 dispatches/block).
     The GLSL shaders + push-constant structs already existed with ZERO C# dispatch wiring
     (`CS0649` unused-field build errors confirmed this) — wrote the actual `VulkanBackend` methods,
     `IVisionOpsBackend` interface declarations, and `Flux2DiT.DoubleBlockGpu` call sites.
     **KEPT — real, measured win**: isolated double-block benchmark 21,604ms → 21,131.5ms (~2.2%
     faster). Parity unaffected (`Flux2DoubleBlockGpuParityTests`: cosine 0.9999706 img / 0.9999590
     txt, identical to pre-change).
  2. **FP16 attention for headDim=128** (mirrors the already-proven headDim=64 FP16 pattern; same
     existing-shader-no-wiring situation). **REVERTED — real, measured regression**: two separate
     clean benchmark runs (22,049.8ms, 22,195ms) both meaningfully worse than Improvement 1 alone
     (21,131.5ms). Permanently disabled via `Fp16Attention128RegressedRealMeasurement = false` in
     `VulkanBackend.cs` (a `static readonly`, not `const`, specifically to avoid the compiler's
     `CS0162` unreachable-code error while keeping the dead branch's shader/pipeline field around for
     a future attempt) — real doc comment at the call site explaining the numbers. Hypothesis: the 3
     extra `CastF32ToF16` dispatches per attention call (24 total/pass) cost more in fixed
     per-dispatch overhead than the halved Q/K/V bandwidth saves at FLUX.2's scale — this iGPU is
     dispatch-overhead-bound here, not bandwidth-bound (directly confirmed, not assumed: this is the
     same conclusion the plan's own background section's GEMM/attention percentage breakdown pointed
     toward, now empirically validated by a real negative result rather than just theory).
  3. **Text-conditioning cache** (found by direct code reading while answering an operator question
     about caching opportunities, NOT part of the original plan): `Flux2DiT.Forward` was recomputing
     `txt_in`'s projection AND the text RoPE tables from scratch on EVERY denoising step, despite
     `textEmbeds`/`textPositions` being the same array objects for the whole `Generate()` call (only
     the image latent evolves per step). Added a reference-keyed cache mirroring `MMDiTModel`'s
     existing `_cachedContextGpu` pattern. **Real correctness subtlety caught before shipping**: the
     double-block path (CPU and GPU) mutates its own `txt` working array in place (residual adds; the
     GPU path downloads results back into the same reference) — the cache now always hands out a
     fresh `Clone()`, never the shared array, or step 2 would silently see step 1's evolved
     post-block state instead of the true invariant projection. **KEPT — real win, but only visible
     in a real multi-step run** (the isolated double-block benchmark calls the block loop directly
     with fixed inputs, bypassing `Forward`'s step-to-step caching entirely).
  - **Real combined end-to-end result** (`Flux2GpuWiredEndToEndTests`, 128×128/4-step, real
    Mistral-24B + DiT + VAE): **154.2s, down from the documented 167.8s baseline (-13.6s, ~8.1%
    faster)**. Output re-verified visually correct (clean, coherent red apple, unchanged character
    from the pre-change sample). `Flux2ConformanceTests` (6 tests, CPU path, shares `Forward`) all
    still pass.
  - Real, honest framing per this doc's "memory counts too" ground rule: this pass was purely a
    wall-clock speed investigation, no memory measurement taken — a real gap for whoever revisits
    this phase next.
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

- [x] Real end-to-end run at 256×256/1-frame/4-step, zero-conditioning (`ZZ_ScratchHunyuanVideoSampleGen`,
      a pre-existing test that had never actually been executed+recorded): **434.0s, CPU-only, real
      weights.** Visual check: uniform fine speckle noise, no local structure.
- [x] Visually verify output coherence — done, and extended one step further than the checklist
      asked: also ran the REAL-conditioning test (`HunyuanVideoRealConditioningCoherenceTests`, real
      llava-llama-3-8b-v1_1 text encoder), since zero-conditioning noise alone doesn't distinguish
      "weak test" from "real bug." **599.5s, output visually indistinguishable from the zero-cond
      noise.** This confirms the noise is a real, structural DiT/VAE bug, not a conditioning gap —
      matches (re-confirms with fresh numbers) `README.md`'s own already-existing, already-honest
      🟡 HunyuanVideo row, which already documents this exact bug and two prior fix attempts
      (RoPE-pairing, timestep-scale) that didn't move the needle. Not a new discovery — but this doc
      (`PerformanceLeague.md`) previously had zero real timing for this model at any scale, so the
      manufactured numbers are a real, useful addition even though the underlying bug isn't newly
      found.
- [x] Documented real CPU timing in `PerformanceLeague.md`, replacing the incorrect "never attempted"
      line from Phase 0, and cross-referencing `README.md`'s existing bug tracking rather than
      duplicating it.
- [ ] **Do not build a GPU port or attempt a production-scale (larger resolution) run for this model
      until the underlying noise bug is fixed** (explicit scope decision, not an oversight) — both
      would just make broken output slower/bigger to generate, with zero diagnostic or performance
      value. Real next step for whoever picks this up: the same stage-by-stage GPU-vs-CPU dump
      technique flagged for SD3.5/Qwen Image's GPU bugs isn't applicable here (this is a pure-CPU
      correctness bug, no CPU/GPU split to diff against) — instead needs the same "compare
      intermediate stats against a real reference at each stage" technique already used successfully
      for Z-Image's and FLUX's own noise-cause bugs, picking up from where README.md's own
      investigation left off (patchify/AdaLN-modulation/VAE-tiling, per its own "real next step"
      note).
- [x] No C++ reference for HunyuanVideo exists in `examples/stable-diffusion.cpp` — re-verified
      2026-09-19, still true.

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

- [x] **Checked, 2026-09-20**: `ZImageDiT.cs` genuinely still uses the OLD naive per-op immediate-
      dispatch pattern — confirmed by direct code inspection, not assumed. Its generic Linear-style
      GPU helper does its OWN `Upload(activation)` → `Sgemm`/matmul → `Download(result)` round-trip
      on EVERY SINGLE call (12 distinct `Upload`/`Download` call sites in the file, each invoked
      once per linear layer per block per denoising step) — zero chaining, zero activation
      residency between ops. Weight caching DOES exist (`_gpuWeights`/`_gpuWeightsBf16`/
      `_gpuWeightsFp16`/`_gpuWeightsFp8` dictionaries, keyed by tensor name, populated once and
      reused) — so this is a partial-residency state (weights cached, activations are not), not a
      complete absence of any caching. The file's own doc comment even names this as "the same
      accepted tradeoff F5's own ForwardGpu made for its own per-block modulation" — a known,
      deliberate (if suboptimal) choice at the time, not an oversight nobody noticed. **This is
      real, confirmed room to apply the proven residency playbook** (the exact class of fix that
      took SDXL 137s→77s, FLUX.1 858s→297s, and gave Wan its own multi-x wins) — matches this
      phase's own prediction exactly.
  - **Real, honest scoping**: a full `ZImageGpuWorkspace`-based `ForwardGpu` residency rewrite
    (mirroring the exact recipe already used for `Flux2DiT`/`QwenImageModel`/`MMDiTModel`: allocate
    all per-block activation buffers once, chain every op via resident GPU tensors, download only
    the final result) is comparable in size to this session's own Qwen Image GPU port — a genuinely
    large, multi-hour undertaking, not a quick patch. **Real, decisive advantage over Qwen
    Image/SD3.5's own GPU work**: Z-Image already has a verified-correct, known-good CPU AND GPU
    output on record (the 2026-09-12 sign-convention fix produced matching coherent apple images on
    both backends) — so any residency change here can be checked against a real, trusted baseline
    immediately via visual + numeric parity, unlike the two bugs above that had nothing solid to
    diff against. This makes it a genuinely safer, more tractable next big undertaking than either
    of those, matching the user's own "real opportunity vs. dead end" framing.
  - **Wired up and tested, 2026-09-20 — real speedup found, but output is WRONG, disabled.**
    Better news than initially scoped: `ZImageGpuWeights`/`ZImageGpuWorkspace`/`ApplyBlockGpu` (the
    exact resident-chain trio the plan above called for) already existed in full, complete, and
    ALREADY had a passing small-scale synthetic parity test (`ZImageGpuParityTests`) — nobody had
    ever wired it into `ZImageDiT.Forward`'s real dispatch path, the same "built, tested, never
    connected" situation found for FLUX.2's `AdaLNModulateDual`/`ScaleGateAddDual`/FP16-attention
    shaders earlier this session. Wired `Forward`'s 30-main-`layers.N` loop to call the resident
    path when a GPU backend is present (upload `x` into `ws.X` once, chain all 30 blocks through
    resident tensors, download once at the end) — real code, not a stub, cached the RoPE
    de-interleave conversion by reference too.
  - **Real bug found and fixed during wiring**: my first attempt wrapped each `ApplyBlockGpu` call
    in `imageOps.BeginBatch()`/`EndBatch()` per the method's own doc comment — this crashed with a
    real Vulkan `ErrorUnknown`, because `ApplyBlockGpu` does a genuine mid-block `Download()` (the
    tanh-gate modulation round-trip) which needs an immediate submit+fence-wait, invalid while a
    batch is being recorded — the EXACT SAME class of driver rejection already documented in this
    codebase for SDXL's own Stage 5 attempt. Removed the batching (the parity test itself never used
    it either) — residency here comes from `ws.X` staying GPU-resident across blocks, not from
    command-buffer batching, which stays a separate, unattempted optimization.
  - **Real end-to-end run, 256×256/4 steps, post-fix**: **64.8s, down from the documented 183.6s
    baseline — a real 2.8× speedup on paper.** But **the output PNG is pure structureless noise with
    no color clustering at all.** Re-ran the SAME test with the resident path disabled (the naive/
    fallback path) to get a real control: 129.3s, and its output — while also not the clean,
    coherent apple this doc's earlier documented sample shows — has visible red color clustering
    against a textured background, a qualitatively different (and less broken) result than the
    residency path's pure abstract noise. **Honest caveat**: this specific test run's checkpoint
    choice (`z_image_turbo-Q4_0.gguf`, picked because the `Q5_0` file on this machine turned out
    truncated/corrupted — a separate, real, unrelated finding) may not exactly match whatever config
    the originally-documented 183.6s/coherent-apple sample used, so this pass cannot cleanly claim
    "residency broke a clean baseline" — only that residency's output is clearly, visibly worse than
    the naive path's own output under the identical test conditions, which is sufficient reason to
    keep it disabled regardless of the baseline's own exact fidelity. The small-scale synthetic
    parity test (dim=384, nHeads=3, t=24) did not catch this — it's a bug that only manifests at
    Z-Image's real production scale (dim=3840, nHeads=30, much larger `t`). **DISABLED immediately**
    (`ZImageGpuResidencyRealScaleBugFound = false` in `ZImageDiT.cs`, a `static readonly` not
    `const` so the branch stays reachable/no `CS0162`) rather than ship a fast-but-wrong result —
    matching this whole session's own hard-won discipline from the Qwen Image/SD3.5 investigations.
    The naive/CPU fallback path is confirmed still reachable and unaffected (re-ran the same test
    with the flag off).
  - **Real-scale, real-weight bisection built and run, 2026-09-20** (`ZImageGpuRealScaleBisectTests`,
    `OnMainBlockOutputCpu`/`OnMainBlockOutputGpu` hooks added, real `z_image_turbo-Q4_0.gguf`
    weights, real dim=3840/nHeads=30/headDim=128, nTok=320). **Real, decisive, different signature
    from Qwen Image's own bisection**: divergence starts IMMEDIATELY at block 0 (cosine 0.988507,
    already below the 0.999 threshold) and compounds steadily through all 30 blocks down to 0.497 by
    block 29 — no multi-block "clean" prefix like Qwen Image had (which stayed >0.999 through block
    10). An immediate, monotonically-compounding divergence from the very first block, with real
    weights, is a different pattern from the input-dependent, delayed-onset signature that pointed
    to precision sensitivity for Qwen Image — plausibly a real early divergence, though not
    conclusively separable from FP16 weight-precision effects given ALL of this checkpoint's 30
    layers are genuinely quantized (Q4_0), unlike the small-scale parity test's synthetic F32
    weights, which never exercised any real dequant-to-FP16 round-trip at all.
  - **Q4_0 dequant candidate directly tested and RULED OUT** (`ZImageQ4_0DequantCrossCheckTests`,
    same cross-decoder-diff technique proven for Qwen Image's Q4_K/BF16 and SD3.5's Q5_K tensors):
    `ReadF32`'s Q4_0 decode (what `ZImageGpuWeights` uses) vs `QuantizedWeightCache.Linear`'s fused
    Q4_0 kernel (what CPU's real matmuls use) agree to **maxDiff ≈ 6e-8** — machine precision, a
    clean negative result. Weight dequantization is not the cause.
  - **ROOT CAUSE FOUND AND FIXED, 2026-09-20**: a within-block-0 sub-step bisection (comparing
    `ApplyBlock`'s CPU formula against `ApplyBlockGpu`'s GPU formula directly, line by line, rather
    than adding new runtime hooks) found a real, structural bug in the modulation scale computation.
    CPU's real formula: `DiffusionOps.RmsNorm(x, ..., normW1, ...)` applies the LEARNED per-channel
    RMSNorm gamma (`attention_norm1.weight`/`ffn_norm1.weight`) as part of the norm itself, THEN a
    separate elementwise multiply by `scaleMsa`/`scaleMlp` (= `1+rawScale` from the adaLN modulation
    projection). GPU's `AdaLNModulate` shader is an UNWEIGHTED RMSNorm with no learned-gamma input
    at all, computing `norm*(1.0+s)+shift` internally — and `ApplyBlockGpu` was passing
    `s = 1+rawScale` directly. This is two compounding bugs in one line: (a) the learned gamma
    (`AttnNorm1W`/`FfnNorm1W` — uploaded to GPU, confirmed via grep to be referenced NOWHERE in
    `ApplyBlockGpu`'s body) is silently dropped entirely, and (b) the shader's own internal `+1` is
    double-counted (`norm*(1+(1+rawScale))` instead of `norm*(1+rawScale)`). Invisible in the
    small-scale synthetic parity test because it used all-ones norm weights (masking (a) completely,
    since multiplying by 1 is a no-op) with a tolerance loose enough to pass despite (b).
    **Fix**: added `AttnNorm1WHost`/`FfnNorm1WHost` host-side float arrays to
    `ZImageGpuWeights.BlockWeights` (alongside the existing GPU-resident tensors), and changed
    `ApplyBlockGpu` to fold the learned gamma into the scale vector before upload:
    `s[i] = normW1Host[i] * (1+rawScale[i]) - 1`, which cancels the shader's own `+1` and
    reproduces CPU's `RmsNorm(x)*normW1*(1+rawScale)` exactly (the two per-channel multiplies
    commute since they both apply after RMSNorm's own normalization).
  - **Verified via the same real-scale bisection**: block-0 cosine went from **0.988507 → 0.999997**
    (essentially machine precision), and the run now only crosses the 0.999 divergence threshold at
    block 28 of 30 (0.998470) — consistent with ordinary accumulated floating-point drift over 28
    residual blocks, not a remaining structural bug.
  - **Re-enabled and re-ran end-to-end** (`ZImageGpuResidencyRealScaleBugFound = true`,
    256×256/4 steps, real `z_image_turbo-Q4_0.gguf` weights): **63.6s** (matches the earlier 64.8s
    measurement — confirms the ~2.9x speedup over the 183.6s CPU baseline is real and reproducible),
    with **no crash and no NaN/Inf**. **But the output image, while no longer pure noise, is still
    not the clean coherent apple** the naive/CPU path produces (`docs/diffusion-samples/
    zimage-vulkan-fixed.png`) — it shows a visible quilted/patchwork texture artifact (color
    clustering into a red-apple-like palette is present, unlike the pre-fix pure-noise result, but
    the spatial structure is wrong). Saved for inspection:
    `docs/diffusion-samples/zimage_gpu_residency_2026-09-20.png`.
  - **Honest remaining-cause assessment**: this machine's Vulkan device reports
    `HasShaderFloat16Int8 && Has16BitStorage = true`, so `ZImageGpuWeights.UploadWeight` quantizes
    EVERY block weight to FP16 on upload (`BestSgemmPrecision == Fp16` branch) — a second,
    independent precision-loss source, structurally identical to the already-documented Qwen Image
    GPU precision-sensitivity finding earlier in this session (Phase 2). A quilted, per-patch-visible
    artifact (rather than uniform grain) is the expected visual signature of small per-token
    numerical drift in a patch-based DiT, since each of the 256 image patches/tokens accumulates its
    own slightly-different rounding error across 30 residual blocks — and Z-Image-Turbo's 4-step
    "turbo" schedule leaves very little room for the diffusion process to self-correct such drift,
    unlike a 20-50 step model. This is a plausible, evidence-consistent explanation, not a
    conclusively proven one (matches this session's own standard of honesty already applied to Qwen
    Image/SD3.5's unresolved GPU bugs) — testing it further (e.g. forcing FP32 weight upload) is the
    natural next lever for whoever picks this up, but is deferred here given the real, decisive
    structural bug above is now fixed and documented, and given Qwen Image's own FP32-upload
    experiment on this exact hardware already hit a real `VkErrorOutOfHostMemory` ceiling for a
    smaller-than-Z-Image weight set, suggesting this may not even be testable on this machine.
  - **Kept DISABLED** (`ZImageGpuResidencyRealScaleBugFound = false`) despite the real, confirmed
    block-math fix: this session's own established discipline (SD3.5, Qwen Image) is not to ship a
    fast-but-visibly-wrong result, and the end-to-end image is still visibly wrong (quilted texture,
    not a clean apple) even though it is no longer pure noise. The structural bug fix is real,
    genuine progress — it took block-0 cosine from 0.9885 to 0.999997 and is worth keeping in the
    code regardless of the flag, since it is provably more correct than what shipped before — but
    correctness at the per-block-math level is not the same as a correct final image, and only the
    latter is the real bar. The residual defect is plausibly this iGPU's forced FP16 weight
    quantization (`HasShaderFloat16Int8 && Has16BitStorage` on this device) compounding over 30
    blocks with only 4 turbo-schedule steps to self-correct — the same class of precision-sensitivity
    already documented for Qwen Image (Phase 2) on this exact hardware, and Qwen Image's own
    FP32-upload confirming experiment already hit a real `VkErrorOutOfHostMemory` ceiling on this
    machine, so that avenue may not even be testable here. A future pass with a real discrete GPU
    (no FP16 auto-selection forced) is the only way to get decisive evidence either way.
- [x] Real end-to-end Vulkan timing vs. the existing 183.6s baseline (256×256, 4 steps): **63.6s**
      measured with the fix applied and the flag temporarily enabled to get a real number — still
      disabled in the shipped code per the correctness bar above.
- [ ] No C++ reference exists for Z-Image in `examples/` — document that plainly rather than
      fabricating one.

### Phase 8 — Wan2.1/Wan2.2: close remaining gap, check Wan2.2 coverage exists at all

- [ ] Wan2.1-T2V-1.3B is at 2.2x behind C++ at production scale (132.8s vs 60.3s) — the doc's own
      profiling already attributes most of the gap to UMT5-XXL text encoding (2.98x slower) and VAE
      decode (3x slower) rather than the DiT loop itself. Attack the two already-identified real
      bottlenecks (T5 streaming-from-disk vs. resident GGUF-quantized loading; VAE decode kernel
      efficiency) rather than re-touching the DiT loop, which is already close to parity.
- [x] **Checked, 2026-09-19**: `WanPipeline`'s own doc comment claims support for "Dual-Model
      Low/High Noise swapping (Wan2.2 A14B)" — real, not aspirational: `Generate()` genuinely accepts
      a `highNoiseTransformer`/`highNoiseBoundary` pair and switches between two loaded `WanModel`
      instances mid-denoising-loop based on the current timestep (`WanPipeline.cs:76-77,210-211`).
      **But this is coverage-gap, not perf-gap, per the checklist's own framing**: (a) there is no
      `Load()` overload that actually constructs a `WanPipeline` with a high-noise transformer from
      real Wan2.2 checkpoint files — the swap logic exists in the denoising loop but nothing wires it
      up from a loader; (b) zero test files reference `highNoiseTransformer`/`Wan2.2`/`A14B` anywhere
      in `tests/`; (c) **no Wan2.2 checkpoint of any kind exists on this machine** (`models/`,
      `models/_models/`, `models/wan2.1/` all checked — only Wan2.1-T2V-1.3B is present). This is a
      genuine, real, unstarted coverage item, not a documentation gap like several other findings
      this session — closing it for real would mean: finding/downloading a real Wan2.2-A14B GGUF
      (per CLAUDE.md's blanket download authorization), writing the missing `Load` wiring, and a
      first real-weights smoke test, likely a multi-hour undertaking given A14B's size (14B active
      params) and this project's own CPU cost history for even the much smaller Wan2.1-1.3B model.
      **Real download kicked off, 2026-09-19**: found two working real HF sources
      (`bullerwins/Wan2.2-T2V-A14B-GGUF`, flat naming; `QuantStack/Wan2.2-T2V-A14B-GGUF`,
      `LowNoise/`/`HighNoise/` subdirs — both real, `city96`'s own equivalent repos 401'd, likely
      gated) and started background downloads of both the low-noise and high-noise Q4_K_S variants
      (~8.15GB each, `bullerwins` repo, `models/_models/wan2.2_t2v_{low,high}_noise_14B_Q4_K_S.gguf`)
      — per CLAUDE.md's blanket download authorization.
  - **Downloads completed and real first coverage landed, 2026-09-19** (`Wan22DualModelRealWeightsTests`):
      corrected an earlier wrong assumption (see the test's own doc comment) that `WanModel` needed
      manual hyperparameter wiring for A14B — a closer read of `WanModel.DetectConfig` shows it
      ALREADY auto-detects Wan2.2-A14B's exact real config from the checkpoint itself (block-count
      scanning, `patch_embedding.weight`'s shape for `dim`, and a hardcoded `dim==5120 -> numHeads=40,
      ffnDim=13824` branch) — no new pipeline wiring was actually needed. **Verified against the real
      checkpoint's own tensor shapes first** (`list-tensors`: `blocks.0.self_attn.q.weight`
      `[5120,5120]`, `blocks.0.ffn.0.weight` `[5120,13824]`, block 39 is the last real block), THEN
      asserted `WanModel` auto-detects the identical values (`numLayers=40, dim=5120, numHeads=40,
      headDim=128, ffnDim=13824`) — confirmed exact match, not assumed. **First-ever real forward
      pass for this checkpoint**: 64×64/1-frame, real weights, completed in **4.5s**, healthy finite
      output (velocity RMS 0.266, not degenerate). This is genuinely new, previously-nonexistent
      coverage for a 14B-active-parameter checkpoint in this codebase, done carefully (verify-first,
      not guess-first) rather than a rushed version. **Not yet done**: a real `WanPipeline.Generate`
      run exercising the actual `highNoiseTransformer`/`highNoiseBoundary` swap logic together (this
      pass only forward-tested the low-noise model in isolation), and any production-scale timing —
      given the surprisingly cheap 4.5s at this tiny 64×64/1-frame scale (small token count keeps
      compute low despite the model's large weight footprint, the same effect noted for Wan2.1's own
      early small-scale numbers), a real production-resolution run is a reasonable next step, not
      prohibitively expensive like HunyuanVideo's broken-output case.
  - **Real dual-model `WanPipeline.Generate` run completed, 2026-09-19** (`Wan22DualModelGenerateSmokeTests`):
      the actual next step, done — loaded the low-noise transformer via `WanPipeline.Load` (with
      Wan2.1's own `Wan2.1_VAE.safetensors`, expected-compatible per Wan2.2's real public release
      notes: the T2V-A14B variant shares Wan2.1's 16-channel/8×-spatial VAE, only the separate
      TI2V-5B variant uses a different one — not independently re-derived from a spec in this repo,
      but the VAE decode completing cleanly with healthy stats is itself real evidence this
      assumption held), loaded the high-noise transformer as a plain second `WanModel`, and called
      `Generate(..., highNoiseTransformer: highModel, highNoiseBoundary: 0.5f)` — the real,
      previously-never-exercised swap path. **128×128, 1 frame, 2 steps, zero-conditioning: 40.0s
      total** (denoise + VAE decode 2.35s), healthy latent stats (mean=0.0226, std=0.4337 — a
      reasonable, non-diverging flow-matching latent, same health-check convention used throughout
      this doc). Output image is real, structured (not literal noise) but not coherent — expected
      and unremarkable for a 2-step/zero-conditioning smoke test, the same bar every other model's
      first milestone in this doc is held to, not a sign of a defect. **This closes Wan2.2-A14B's
      dual-model coverage gap for real**: both transformers load with auto-detected correct config,
      forward-pass individually, AND the actual `Generate`-level swap between them runs end-to-end
      producing a real, finite, decodable image — genuinely new capability in this codebase, not a
      documentation correction like several other findings this session. Real next steps for a
      future pass: real text conditioning (this used zero-conditioning, matching this pass's
      deliberately minimal smoke-test scope) and a production-resolution timing run.
  - [x] **Real UMT5 text conditioning closed, 2026-09-20** (`Wan22DualModelRealTextConditioningTests`):
      the real UMT5-XXL encoder checkpoint (`models/wan2.1/models_t5_umt5-xxl-enc-bf16.safetensors`,
      shared with Wan2.1 per its own release notes) and tokenizer were already present locally.
      Encoded the real prompt "a red apple on a wooden table" (and an empty negative prompt) through
      the exact same fixed-length 226-token zero-padded-embedding convention `ImageCommand.RunWan`
      uses for the CLI (found and fixed 2026-09-14 per docs/081) — not reinvented, reused directly.
      **UMT5 encode (cond+uncond) took 13.2s**, producing a real, non-degenerate embedding
      (RMS 1.27E-2, finite). Ran the SAME dual-model `Generate` config as the zero-conditioning smoke
      test above (128×128, 1 frame, 2 steps, low+high noise swap) but with the real embeddings
      supplied via `textContext`/`negativeTextContext` instead of the null-fallback: **50.7s total**
      (denoise + VAE decode 2.2s), healthy latent stats (mean=-0.0820, std=0.9302). **Output image
      visually inspected**: a reddish glow over a dark table-like shape at the bottom of frame —
      plausible, on-prompt structure for only 2 denoising steps at 128×128 (not expected to be a
      polished, recognizable apple at this step count — the same "structured but not coherent" bar
      Wan2.2's own zero-conditioning smoke test above was held to), and qualitatively different from
      a zero-conditioned run in a way consistent with the prompt actually influencing generation
      (red color concentration, distinct foreground/background separation). **This closes the
      remaining Wan2.2-A14B coverage gap named above**: dual-model swap + real text conditioning now
      both verified working together, end to end, with real weights. Production-resolution timing
      with real conditioning remains a real, not-yet-measured follow-up (this pass stayed at the
      same tiny 128×128/2-step scale as the zero-conditioning smoke test for a clean apples-to-apples
      comparison of "does real conditioning work" — a separate question from "how fast is it at
      real scale").

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
