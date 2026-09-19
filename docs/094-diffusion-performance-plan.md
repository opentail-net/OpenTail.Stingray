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
- [ ] **Bisect the remaining SD3.5 GPU-specific bug** (maxDiff 0.118, incoherent output despite a
      correct CPU reference now existing to diff against): dump intermediate latents at each of the 20
      Vulkan steps (same
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
