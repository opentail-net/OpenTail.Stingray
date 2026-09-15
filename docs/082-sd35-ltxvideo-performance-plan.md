# SD3/SD3.5, LTX-Video-2B, SD1.5+ControlNet Performance Improvement Plan (2026-09-15)

## Scope

Four image/video diffusion pipelines, covered in three parts below:

- **Part A — SD3/SD3.5-medium** (`src/OpenTail.Stingray.Diffusion/SD3/`, one shared
  `MMDiTModel`/`Sd3Pipeline` class pair, checkpoint-config-detected — SD3 and SD3.5 are the same
  architecture family in this codebase, not two separate ports).
- **Part B — LTX-Video-2B** (`src/OpenTail.Stingray.Diffusion/LTXVideo/`).
- **Part C — Stable Diffusion 1.5 (+ControlNet)** (`src/OpenTail.Stingray.Diffusion/
  StableDiffusion/` + `ControlNet/`) — added 2026-09-15, see Part C below; starts from a more
  advanced baseline than A/B (already correctness-verified, already has per-op GPU dispatch), so
  its phases are shorter.

## Why these models, together, now

SD3.5-medium and LTX-Video are the two remaining image/video diffusion pipelines in this codebase
with **zero GPU code** (`MMDiTModel.cs` and `LtxVideoModel.cs` both confirmed by grep to have no
`IComputeBackend`/`Gpu`/`Vulkan` references at all) — a full step earlier than FLUX, Wan2.1, and
Z-Image-Turbo, all of which are now GPU-resident and (per `docs/081`, this session) verified
correct end-to-end. SD1.5+ControlNet is a step ahead of both (real, verified output; the UNet
already has per-op GPU dispatch) but has never received the full-residency treatment those three
models did, and ControlNet's conditioning path has no GPU code at all — a real, scoped, likely
cheap win given how much of the underlying UNet machinery ControlNet reuses. This doc supersedes
the "recommended approach" sections of `docs/076` (SD3.5) and `docs/077` (LTX-Video) with a single,
phased execution plan that reflects:

1. The now-proven cross-reference bisection methodology (build/run the real
   `stable-diffusion.cpp` reference, instrument both sides with matching debug-tensor dumps,
   compare stage-by-stage) that found and fixed Wan2.1's CPU and GPU bugs this session — this is
   now the default tool for any remaining correctness question on these two models, not a novel
   technique to invent from scratch.
2. The full inventory of GPU-residency patterns already landed and battle-tested in this codebase
   (`FluxGpuWeights`/`FluxGpuWorkspace`, `WanGpuWeights`/`WanGpuWorkspace`, the single-command-
   buffer-per-forward-pass pattern just landed for Wan) — both models should reuse this
   architecture directly, not reinvent it.
3. Real current state: SD3.5-medium's checkpoint is **not currently present** on disk (`models/`
   has no `sd3.5`/`sd3_5` file); LTX-Video's checkpoint **is** present
   (`models/ltx-video-2b-v0.9.1.safetensors`, plus `models/ltx-t5/`). Disk currently has **69GB
   free** (up from the ~453MB that blocked this work when `docs/076` was written) — the SD3.5
   checkpoint can be re-downloaded now.

## Cross-cutting ground rules (apply to every phase below)

- **No subagents** — `CLAUDE.md` rule 6, project-wide.
- **Never remove/revert existing code** — CPU paths stay; GPU work is additive.
- **Ask before committing.**
- **Never run the full heavy test suite** — targeted single-class `.exe -class <FullyQualifiedName>` runs only.
- **Check `Get-CimInstance Win32_OperatingSystem | Select FreePhysicalMemory` before every
  benchmark/test run** — run one thing at a time, alone.
- **Monitor real CPU vs GPU utilization (Task Manager) during every "GPU" test run** — this
  project has now found silent CPU-fallback bugs twice (Wan's original `ForwardGpu`, and is the
  single most valuable 30-second check available before trusting any new GPU timing number).
- **Update `PerformanceLeague.md` honestly** — real measured numbers, explicit caveats, never a
  fabricated or assumed win. A "correctness fix that costs speed" (like FLUX's T5-padding fix,
  894s vs 649-683s pre-fix) is a normal, expected outcome to report plainly, not a failure to hide.
- **Correctness before speed, always, for both models** — see Phase 0 for each below. A fast GPU
  path built on top of an unverified or known-broken CPU baseline just doubles the debugging
  surface (this is `docs/077`'s own standing warning, still true).

---

# Part A — SD3.5-medium

## Current state

- Real CPU port exists and is **correctness-verified** (per `PerformanceLeague.md`: 5 real bugs
  found and fixed 2026-09-05 — dual-attention norm input, VAE scale/shift, unpatchify channel
  order, missing positional embedding). Last measured timing: **656.9s (~11 min)**, 256×256, 20
  steps, seed 42, 2026-09-02 (pre-dates the fixes; compute cost believed unchanged since the fixes
  were correctness-only, but this has **not been re-measured post-fix** — that gap is Phase 0
  below, not assumed).
- Zero GPU code. `MMDiTModel.cs` (622 lines) still on the "upload/Sgemm/download/free per call"
  starting line every other model began from.
- Architecture specifics already confirmed by direct source reading (`docs/076`, do not re-derive):
  `headDim=64` (use `MultiHeadAttentionTiled`, NOT the `Tiled128` kernel FLUX/Wan/Z-Image use),
  absolute 2D sincos position embedding (not RoPE — simpler, lower-risk port), AdaLN shift+scale+
  gate convention identical to FLUX's, uniform dual-stream blocks with a `contextPreOnly` flag on
  the last block only, and a real, non-trivial `dualAttn` second config bit (checkpoint-detected
  via `{blk}.x_block.attn2.qkv.weight` presence) that doubles the branch combinations to four
  (`contextPreOnly` × `dualAttn`) and adds a second image-only self-attention pass with its own
  gate/norm. **This is the real complexity floor for this port — budget for four branch
  combinations and a combined (not per-stream) joint-attention Q/K/V buffer, not the "simplest of
  the four models" framing `docs/076`'s own first draft used before its 2026-09-14 correction.**

## Phase A0 — re-establish a measurable CPU baseline (prerequisite)

1. Re-download the SD3.5-medium checkpoint (69GB free is now enough headroom; check `stingray pull`
   coverage-tooling first per `CLAUDE.md`'s `pull`/`admit-arch` section rather than a manual fetch).
2. Re-run the **exact same** config as the last recorded row (256×256, 20 steps, seed 42) on CPU,
   post-2026-09-05-fixes, and record the real number in `PerformanceLeague.md` next to (not
   replacing) the pre-fix 656.9s row, explicitly labeled as the correctness-verified baseline this
   whole plan measures against. If it differs meaningfully from 656.9s, say so plainly — don't
   assume "correctness-only fixes don't change speed" without checking.
3. Visually re-confirm the output is real and coherent (not just that the run completes) — save to
   `docs/diffusion-samples/` with a descriptive name.

**Exit criterion**: one real, dated, visually-confirmed-coherent CPU timing number for SD3.5-medium
at a fixed config, superseding the 2026-09-02 pre-fix number as this plan's reference point.

## Phase A1 — CPU performance pass (cheapest wins first, no GPU risk yet)

Before touching GPU residency, apply the same class of CPU optimization that already won 1.5-5.2×
on Z-Image-Turbo/Wan/FLUX's own CPU paths (context caching where inputs are step-invariant,
FlashAttention-style fused attention, AVX2/FMA vectorization in the hot per-block loop) to
`MMDiTModel.cs`'s CPU forward pass:

1. **Profile first** — do not guess. Add or reuse a per-stage stopwatch breakdown (same pattern
   as Wan's `[WanPipeline Profile]`/`[WanProfile]` logs added this session) covering: text
   conditioning, per-block attention, per-block FFN, AdaLN modulation, VAE decode. Identify the
   actual dominant cost before optimizing it.
2. **Check for step-invariant recomputation** — the position embedding crop (`AddCroppedPosEmbed`)
   and any per-step-constant projection (e.g. a modulation table that doesn't depend on the
   evolving latent) are candidates for once-per-generation caching instead of once-per-step,
   mirroring the "Invariant Caching" win already logged for FLUX's CPU pass.
3. **Vectorize the hot per-block loop** with `TensorPrimitives`/AVX2-FMA if not already using them
   — check whether `MMDiTModel.cs`'s current linear/attention helpers already route through the
   shared `Primitives/*Kernels.cs` convention (`CLAUDE.md` rule 7's DRY-pass expectation) or still
   use a naive per-element loop.
4. **Measure, keep only real wins** — per `CLAUDE.md` rule 7: a handful of runs each side, write
   down the actual numbers, revert anything that doesn't measurably help even if the reasoning
   seemed sound.

**Exit criterion**: a real, measured CPU speedup (or an honest "no win found" if that's what
happens) recorded in `PerformanceLeague.md`, using the Phase A0 baseline for comparison.

## Phase A2 — GPU residency (the main event)

Follow `docs/076`'s "Recommended approach" section directly (it is already detailed and doesn't
need restating in full here) — in short:

1. `MMDiTGpuWeights` (new) — mirror `FluxGpuWeights.cs`/`WanGpuWeights.cs`: upload all 24 blocks'
   img+txt stream Q/K/V/O, both attn and attn2 (when `dualAttn`) weights, norms, AdaLN-modulation
   weights, FFN weights, and the real `pos_embed` tensor once, resident in VRAM, same
   FP16-if-available `UploadWeight` pattern.
2. `MMDiTGpuWorkspace` (new) — preallocated buffers sized for real image+text token counts,
   including **one shared, `totalTokens`-sized Q/K/V buffer** (not two per-stream buffers — the
   real architecture concatenates img+txt tokens into one joint-attention call, confirmed in
   `docs/076`).
3. `JointBlockGpu` (new method) — AdaLN modulate (reuse `AdaLNModulate` as-is) → Q/K/V via `Sgemm`
   for both streams → joint attention via `MultiHeadAttentionTiled` (headDim=64) → O projection →
   gated residual (reuse `ScaleGateAdd` as-is) → FFN, with `contextPreOnly` and `dualAttn` branches
   both handled — **write one real parity test per one of the four branch combinations actually
   present in the real checkpoint** (check which combinations the real SD3.5-medium checkpoint
   actually uses via its own tensor inventory before assuming all four need independent coverage —
   likely only 1-2 combinations are real for this specific checkpoint, per `dualAttn`'s
   checkpoint-detected nature).
4. **Single-command-buffer-per-forward-pass from the start** — Wan's GPU work this session proved
   this pattern out (`ForwardGpuCore`, one `BeginBatch()`/`EndBatch()` wrapping the whole 30-block
   forward) and it's strictly better than per-block batching once correctness is established;
   build SD3.5's GPU path with this shape directly rather than per-block-then-refactor-later, but
   **keep a debug/instrumented per-block-batched fallback path** (same `debugCrossRefGpu`-gated
   pattern Wan uses) since mid-forward `Download()` calls for debugging are incompatible with a
   single fully-batched command buffer.
5. **Correctness-check with the cross-reference technique BEFORE trusting any GPU timing number** —
   this is the one real methodology update since `docs/076` was written: don't just rely on a
   CPU-vs-GPU internal parity test (Wan's own history shows internal parity can pass while both
   sides are independently wrong — though for SD3.5 specifically the CPU side is already externally
   verified per Phase A0, so this risk is lower here than it was for Wan). Still worth a cheap
   spot-check: instrument `stable-diffusion.cpp`'s own SD3.5 support (if present — check
   `examples/stable-diffusion.cpp`'s model support table first) the same way Wan's `wan.hpp` was
   instrumented, and compare at least the patch-embedding-equivalent first stage and one
   mid-network block, the same two checkpoints that caught Wan's GPU bug at its very first stage.

**Exit criterion**: a real, GPU-resident SD3.5-medium forward pass with a real numeric parity
assertion against the Phase A0/A1 CPU baseline, AND a real visually-confirmed-coherent end-to-end
GPU generation at the same config — not just a passing parity test (this is the exact lesson
Wan's GPU history teaches: parity passing is not sufficient evidence of correctness).

## Phase A3 — GPU kernel tuning

Only after A2's correctness is solid. Options, roughly in expected value order based on what
worked for FLUX/Wan:

1. Profile GPU stage breakdown the same way FLUX's `STINGRAY_PROFILE_GPU_SPLIT`/Wan's
   `STINGRAY_WAN_PROFILE` did — find the real dominant cost (very likely the DiT loop itself,
   per every other model's pattern) before optimizing blindly.
2. Check whether `MultiHeadAttentionTiled` (headDim=64) is itself a bottleneck — this kernel did
   NOT receive this session's `Tiled128` rewrite (32×16 vec4 tiles, 1.81× win on FLUX/Wan). If
   profiling shows attention dominating, a headDim=64-specific tiled/vectorized rewrite (mirroring
   the same technique, different tile shape) is a real, scoped, high-value follow-up — but only if
   the profile actually shows this as the bottleneck, not assumed.
3. Compare against a real `stable-diffusion.cpp` SD3.5 reference run (if supported) the same way
   Wan's final gap analysis worked (60.3s C++ Vulkan vs Wan's 132.8s) — gives a concrete target
   and identifies which stage (text encode / DiT loop / VAE decode) carries the remaining gap.

**Exit criterion**: real, measured numbers reported per stage, honestly compared against the C++
reference where available, following the exact reporting discipline established in Wan's/FLUX's
`PerformanceLeague.md` rows this session.

---

# Part B — LTX-Video-2B

## Current state — correctness is NOT yet solid, this changes the phase order

Unlike SD3.5-medium, LTX-Video's CPU path has a **known, precisely-characterized but not yet
root-caused** correctness bug (`docs/077`'s extensive 2026-09-14 investigation log — read it in
full before starting, it is not a starting point but a substantial investigation already done):

- The DiT transformer itself is verified correct to machine precision on a small golden fixture
  (timestep-embedding sin/cos flip bug found and fixed; RoPE, patchify, caption projection, block0,
  AdaLN modulation, self-attention, cross-attention, FFN, gating, and the final output layer all
  individually checked clean against the real C++ reference structurally).
- Despite that, a real end-to-end run **diverges**: the per-step latent standard deviation grows
  **unbounded** across a 20-step denoising loop (0.976 → 2.42, a ~2.5× growth), confirmed
  **scale-independent** (same ~2× growth pattern at a tiny 4-token control scale) — ruling out a
  real-scale-specific DiT bug and narrowing the cause to either (i) a subtle DiT magnitude-bias too
  small to fail a cosine-similarity-only golden check but which compounds multiplicatively over
  repeated self-feeding calls, or (ii) a genuinely missing renormalization/scale step in the
  32-step-loop integration itself.
- CFG, the Euler/scheduler formula, and the VAE decoder have each been independently ruled out via
  direct real-run experiments (CFG-disabled run still diverges; schedule dump shows no spike/NaN;
  VAE-alone noise-isolation test decodes cleanly).
- **The single most promising concrete next step already identified but not yet executed**: a
  tighter, magnitude-aware golden check (the existing `LtxVideoTrajectoryGoldenTests` uses
  cosine-similarity only, which is magnitude-blind by construction — exactly why it never caught
  this bug) at the REAL 256-token scale over the REAL 20-step loop, comparing latent std/max-abs
  growth against a real reference trajectory, not just angle.

## Phase B0 — finish the correctness investigation (hard prerequisite, do not skip to GPU work)

This is not optional per `docs/077`'s own standing warning: GPU residency on top of unresolved
correctness doubles the debugging surface. Apply the now-proven Wan methodology directly — this is
the strongest new tool available since `docs/077` was last updated:

1. **Check whether `stable-diffusion.cpp` supports LTX-Video** (grep `examples/stable-diffusion.cpp`
   model-family tables/source for `ltxv`/`ltx_video` — `docs/077` already cites
   `examples/stable-diffusion.cpp/src/model/diffusion/ltxv.hpp` as an existing structural
   reference, so a real, runnable C++ implementation likely already exists in the vendored tree).
2. If it does: instrument it with the SAME `wandbg_*`-style named debug-tensor dump pattern used
   for Wan (`capture_tensor`, raw float32 `.bin` dumps, `STINGRAY_*_DEBUG_CROSSREF`-gated), forcing
   identical latent/timestep/caption inputs into both implementations, and do a real byte-level,
   multi-STEP comparison (not just single-forward) — specifically capturing per-step latent
   mean/std/max-abs on BOTH sides across a real 20-step loop, directly testing the "DiT magnitude
   bias vs missing loop renormalization" open question from `docs/077`'s handoff. This is exactly
   the class of bug (right direction, wrong magnitude, invisible to angle-only checks, compounds
   over repeated self-feeding calls) the Wan investigation's replay/cross-reference technique was
   built to catch, and this bug's shape (scale-independent, ~3.5%/step consistent proportional
   overshoot) is unusually well-suited to a magnitude-tracking bisection.
3. If a magnitude-aware multi-step golden comparison isn't feasible against real diffusers output
   (no Python in this environment, per `docs/077`'s own noted blocker) — the C++ reference is the
   practical way around that blocker, exactly as it was for Wan (this environment couldn't get a
   real Python-side comparison for Wan either; the C++ reference is what actually unblocked it).
4. **Concrete first move given `docs/077`'s narrowed hypothesis**: add the C++ reference's own
   per-step latent-statistics logging (mirror `STINGRAY_LTX_DEBUG_STEPSTATS`) and directly compare
   growth curves — if the C++ reference's own latent std also grows over 20 steps (even by a
   smaller/different amount), that's a strong signal toward (ii) real, expected per-step growth
   that this port's VAE/output scaling just isn't compensating for correctly; if the C++
   reference's latent std stays flat/converges while this port's diverges, that's strong,
   actionable confirmation of (i), and the per-step delta between the two at each step directly
   localizes which block/stage introduces the compounding bias.

**Exit criterion**: the root cause is found and fixed, OR a definitive, evidence-backed answer to
whether the bug lives in the DiT (needs a model-level fix) vs the loop/output-scaling convention
(needs a scheduler-level fix) — either way, a real end-to-end run at 20 steps (the previously-
broken config) produces a visually coherent output, matching the same bar every other model in
this codebase now meets. Do not proceed to Phase B1 without this.

## Phase B1 — Phase 1 GPU dispatch (basic, matches `docs/077`'s own staged plan)

Once B0 is closed:

1. Add `_backend` field + constructor parameter to `LtxVideoModel`, and a `MatQ`-style per-matmul
   upload/`Sgemm`/download/free path — the same starting shape FLUX/Wan/Z-Image/SD3.5 all began
   from. This is intentionally NOT full residency yet — it's a minimal-surface-area correctness
   checkpoint (confirm GPU dispatch itself works and matches CPU) before the bigger residency
   rewrite, exactly as `docs/077` already recommends.
2. Real parity test using the model's own existing captured-intermediate fields (`LastProjInOut`,
   `LastCaptionProjOut`, `LastEmbeddedTimestep`, `LastTimestepProj`, `LastRopeCos`, `LastRopeSin`,
   `LastBlock0Out`) — genuinely useful pre-built infrastructure, no new golden data needed for this
   step.

**Exit criterion**: GPU dispatch produces numerically matching output to the (now-fixed) CPU path
at the per-stage granularity these existing capture fields cover.

## Phase B2 — full GPU residency

1. `LtxVideoGpuWeights`/`LtxVideoGpuWorkspace` (new files) — mirror `FluxGpuWeights.cs`/
   `WanGpuWeights.cs` directly. Real architecture specifics already confirmed in `docs/077` (do not
   re-derive): `headDim=64` (use `MultiHeadAttentionTiled`, not `Tiled128`), continuous 3D RoPE
   with **interleaved** pairing (same convention as `Flux2DRoPE`'s existing GPU kernel — likely
   directly reusable with LTX-Video's own frame/height/width axis-dim split, get the real split
   from `ComputeContinuous3DRoPE` directly, don't assume Wan's 44/42/42-style numbers transfer),
   `BasicTransformerBlock` with self-attention + cross-attention (against precomputed, GPU-resident
   T5-XXL caption K/V — same invariant-caching pattern as Wan's `PrecomputeCrossKvCacheGpu`) +
   AdaLN-single timestep branch, with `CrossAttentionAdaln`/`SelfAttentionGated`/
   `CrossAttentionGated` checkpoint-detected booleans (confirmed all `false` for the real
   0.9.1 checkpoint per `LtxVideoRealWeightsTests` — so the simpler code paths are the ones that
   actually need porting for this specific checkpoint, not a gap to work around).
2. `BasicTransformerBlockGpu` (new method) — self-attn (RMSNorm pre-norm, non-affine, eps=1e-6;
   QK-norm affine RMSNorm eps=1e-5, applied to the full `inner_dim` BEFORE head split, not
   per-head) → RoPE (applied as one "head" of width `d=2048` pre-split, per `docs/077`'s confirmed
   finding) → gated residual → cross-attn (raw `x` as query, no pre-norm, K/V from the
   once-per-generation `caption_projection` output, no RoPE, ungated residual add for this
   checkpoint's real config) → FFN (2048→8192→2048, tanh-GELU) → gated residual.
3. Single-command-buffer-per-forward-pass, same pattern as Wan/planned-SD3.5, with a debug/
   instrumented per-block-batched fallback for `Download()`-based debugging.
4. **Given B0's finding will likely implicate a specific stage** (the DiT loop itself, or the
   final output scaling) — whatever B0 found needs to be preserved correctly under the GPU
   dispatch's different operation ordering; re-verify the SAME fix/finding holds on GPU, don't
   assume a CPU-side fix automatically transfers bit-for-bit to a differently-batched GPU
   execution graph (this is precisely the class of subtlety that caused Wan's GPU bug to be a
   *different* bug from its CPU one, not the same one twice).

**Exit criterion**: real GPU-resident LTX-Video forward pass, numeric parity against the
B0-fixed CPU baseline, AND a real visually-confirmed-coherent GPU end-to-end generation at the
same 20-step config that was broken before B0.

## Phase B3 — GPU kernel tuning

Same shape as Phase A3: profile first, target the real dominant stage (very likely the DiT loop,
given video has a larger token count than image models at the same resolution), check whether
`MultiHeadAttentionTiled` (headDim=64, shared with SD3.5) needs its own tiled/vectorized rewrite
if profiling shows it as the bottleneck, compare against the real C++ reference if
`stable-diffusion.cpp` supports LTX-Video (checked in Phase B0 already).

---

# Part C — Stable Diffusion 1.5 (+ControlNet)

## Current state — already correctness-verified, already has basic GPU dispatch

Unlike Parts A and B, SD1.5 does not need a correctness-first phase: `PerformanceLeague.md`/
`README.md` both confirm a real, coherent, photorealistic 512×512/20-step generation
(2026-09-11), and `UNet2DConditionModel.cs`'s own doc comment already states it "supports both CPU
(SIMD AVX2/AVX-512) and GPU (Vulkan/CUDA SGEMM via `IComputeBackend`)". Confirmed by grep: it
already has a `GetGpuWeight`/upload-cache pattern (FP16-if-available, cached per-weight-name
across calls — a real, if partial, residency step already taken, further along than FLUX/Wan/
Z-Image/SD3.5/LTX-Video all started from) and an implicit-GEMM `Conv2dImplicitGemm` GPU path (the
same kernel already used and tuned for SDXL-Turbo's VAE decoder per `PerformanceLeague.md`'s
SDXL-Turbo row: 30.41s → ~25.2s). **`ControlNetModel.cs` has zero GPU code at all** — pure CPU,
confirmed by grep.

No current `PerformanceLeague.md` timing row exists for SD1.5 (checked — the model has a
correctness row but no measured-time row anywhere in the doc). **No SD1.5 checkpoint is currently
present under `models/`** either (checked by filename search) — like SD3.5, this needs a
re-download before any measurement can happen.

## Phase C0 — re-establish measurable baselines

1. Re-download an SD1.5 checkpoint (`stingray pull`, check `docs/diffusion-samples/README.md` or
   `docs/00-current-work.md` for whichever specific repo/quant was used for the 2026-09-11
   verification run, to keep the comparison apples-to-apples) and a ControlNet checkpoint
   compatible with it (SD1.5 ControlNet checkpoints are small, ~1.4GB each — cheap relative to the
   other downloads in this plan).
2. Run and time the SAME config as the 2026-09-11 verification (512×512, 20 steps) on CPU, and
   again with the existing partial-GPU-dispatch path (`Vulkan` backend, per the README's own
   "CPU / Vulkan" column) — record BOTH real numbers in `PerformanceLeague.md` for the first time,
   since neither currently exists there. Also run one real ControlNet-conditioned generation (CPU
   only, since ControlNet has no GPU path yet) and time it separately.
3. Visually re-confirm both the plain SD1.5 and the ControlNet-conditioned outputs are coherent —
   save to `docs/diffusion-samples/`.

**Exit criterion**: real, dated, visually-confirmed CPU and partial-GPU timing numbers for SD1.5
(both existed nowhere before this phase), plus one real ControlNet timing number, all newly
recorded in `PerformanceLeague.md`.

## Phase C1 — finish UNet GPU residency (upgrade the existing partial path)

The existing `GetGpuWeight`/`Conv2dImplicitGemm` pattern is real progress but is the same
"per-op dispatch, not true residency" starting shape FLUX/Wan/Z-Image all began from and then
outgrew. Apply the same transformation:

1. `Sd15GpuWeights`/`Sd15GpuWorkspace` (new, or extend the existing `_gpuWeights`/
   `_gpuWeightsNative` caching dictionaries into a real preallocated-workspace class) — mirror
   `FluxGpuWeights.cs`/`WanGpuWeights.cs`'s structure: upload every ResBlock/attention-block/
   cross-attention weight once, resident in VRAM, instead of the current lazy-cache-on-first-use
   pattern (which is a real win over re-uploading every call, but still leaves the per-block
   compute graph itself CPU-orchestrated with individual GPU round-trips per op, not a fused
   per-block or per-forward-pass GPU dispatch).
2. Single-command-buffer batching per UNet block (or per down/mid/up stage) — same
   `BeginBatch()`/`EndBatch()` pattern proven on Wan/planned for SD3.5/LTX-Video.
3. Real parity test (GPU vs CPU, real weights, real UNet forward at a fixed timestep/latent) before
   trusting any new GPU timing number.
4. **This is a smaller, lower-risk port than Parts A/B** — the architecture (convolutional UNet
   with cross-attention, no exotic dual-stream/joint-attention/RoPE complexity) is simpler than
   MMDiT or LTX-Video's block structure, and a real partial GPU path already exists and is
   correctness-verified to build on. Budget accordingly — this should be the cheapest phase in the
   whole plan.

**Exit criterion**: real, measured full-residency GPU timing for SD1.5, honestly compared against
both the Phase C0 CPU number and the Phase C0 partial-GPU number (a full-residency rewrite that
doesn't beat the existing partial-GPU path is a real, reportable finding, not an assumed win).

## Phase C2 — ControlNet GPU path

1. `ControlNetModel.cs` reuses much of `UNet2DConditionModel`'s own block structure (ControlNet is
   architecturally a trainable copy of the UNet's encoder with zero-convolution injection into the
   main UNet's decoder) — check directly how much of Phase C1's new `Sd15GpuWeights`/workspace
   machinery is literally reusable before writing a parallel `ControlNetGpuWeights` from scratch;
   a real DRY opportunity per `CLAUDE.md` rule 7's DRY-pass expectation.
2. Wire ControlNet's conditioning-image encoder + its per-block residual/zero-conv outputs through
   the same GPU-resident path, injected into the main UNet's GPU-resident forward pass at the
   matching block indices.
3. Real parity test + a real end-to-end ControlNet-conditioned generation, visually re-confirmed.

**Exit criterion**: ControlNet-conditioned generation has a real GPU path, measured honestly
against its Phase C0 CPU-only baseline.

## Practical constraints for Part C (same as Parts A/B)

Same cross-cutting ground rules at the top of this doc apply — no subagents, never revert
existing code, ask before committing, monitor real GPU utilization, honest `PerformanceLeague.md`
updates.

---

## Suggested execution order across all three parts

Given SD3.5 (Part A, once A0 downloads) and SD1.5 (Part C) both have solid, verifiable-or-already-
verified CPU baselines, and LTX-Video (Part B) does not:

1. **A0** (SD3.5 re-download + re-measurement) and **C0** (SD1.5 re-download + first-ever timing
   rows) — do these first, both cheap, both unblock the rest of their own parts. Can run in
   parallel with each other (different checkpoints, no shared resource contention beyond disk).
2. **B0** (LTX-Video correctness) — start next, since it's the longest pole and blocks all of
   Part B; the Wan cross-reference infrastructure built this session is directly reusable, so the
   marginal cost of extending it to LTX-Video is much lower than building it was the first time.
3. **C1 → C2** (SD1.5 UNet residency, then ControlNet) — likely the cheapest full part end-to-end
   given the more advanced starting point; a good candidate to finish first and bank a real win
   while B0's longer investigation continues.
4. **A1 → A2 → A3** (SD3.5 CPU pass, then GPU residency, then tuning) — independent of B0/C0's
   outcomes.
5. **B1 → B2 → B3** only after B0 closes.

This lets real, committable progress happen on SD3.5 and SD1.5 immediately while the (likely
longer) LTX-Video correctness investigation continues, rather than serializing everything behind
the hardest problem.

## Success criterion for this whole plan

SD3/SD3.5-medium, LTX-Video-2B, and SD1.5(+ControlNet) all reach the same bar every other model in
this codebase (FLUX, Wan2.1, Z-Image-Turbo) already meets: a real, visually-confirmed-coherent
image/video from a real prompt on **both CPU and GPU** (ControlNet: CPU baseline plus a real GPU
path), with `PerformanceLeague.md` updated honestly at each phase — a faster wrong answer is not
progress, per this project's own repeated, hard-won lesson from the Wan investigation.
