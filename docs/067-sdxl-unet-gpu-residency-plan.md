# SDXL UNet whole-graph GPU residency — implementation plan

**Status**: approved for implementation 2026-09-12, after external review. Revision 2 (folds in
review corrections + a code-verified finding about `FullSeqAttention`).

## Why

A real, measured profiling pass found that in a 137.4s SDXL-Turbo Vulkan run (512×512, 4 steps):
GPU submit+execution+fence-wait is only **2.7s (~2%)**; CPU-side staging-buffer Map/memcpy is
**43.6s (~32%)**, across **6511** separate dispatch/upload/download round-trips; the rest is
CPU-side work outside Vulkan.

Precise framing (softened per review — the stronger claim isn't fully supported by the
measurement): **GPU execution/submit time is currently a small fraction of total runtime;
CPU-side staging and orchestration dominate the measured Vulkan path.** The GPU could still be
doing inefficient work internally; it simply isn't where the measured wall time goes today. The
architecture treats the GPU as a synchronous RPC coprocessor: `Lin()`/`Conv()` each do
`Upload(x) → Sgemm/Conv-shader → Download(result) → Free()`, individually, per call.
`SpatialTransformer()` chains ~100 of these per call at the deepest blocks. Two attempts to fix
this by improving the GPU attention kernel both regressed — the kernel was fighting a cost one
level above it.

## Existing precedent, and two things verified against the actual current tree before starting

- `VaeDecoder.ResBlockGpu`: one ResBlock already runs as a GPU-resident Tensor-in/Tensor-out
  chain (one Upload, everything stays a GPU Tensor, one Download). Real ~22.5% win. Note: `VAE`'s
  *other* conv path (`ConvGpu`) still uploads/downloads per chunk — the residency pattern exists
  in one place, it is not yet the VAE's universal behavior.
- `VulkanBackend` has `BeginRecord()`/`EndRecordAndSubmit()`, `BeginBatch()`/`EndBatch()`,
  `RecordBarrier()` — used elsewhere (RRDBNet's batched conv path).
- `IComputeBackend.AddInPlace(Tensor dst, Tensor src)` is a real, existing elementwise GPU op —
  usable directly for residual/bias adds without writing a new shader.
- **Verified 2026-09-12, corrects an earlier assumption**: `IComputeBackend.FullSeqAttention`
  exists and is implemented on Vulkan/CPU/CUDA, but `VulkanBackend.FullSeqAttention`'s actual body
  (`VulkanBackend.cs:3529`) downloads Q/K/V to host, computes the attention math in plain CPU
  code, and only its signature looks GPU-resident — it is not a real GPU kernel today. It also
  takes a single `nTok` (assumes qLen == kvLen), so it doesn't cover SDXL's cross-attention shape
  (qLen = hw tokens, kvLen = 77 context tokens) without changes either. **Do not treat this as an
  existing usable GPU attention primitive** — `MultiHeadAttentionTiled` (confirmed still present,
  `IImageOpsBackend.cs:57`, numerically verified but currently unused after regressing in a
  non-resident context) remains the real candidate to re-test in Stage 4.

## Target end state

`Forward()` uploads real inputs once, keeps every intermediate as a GPU Tensor for the whole
down→mid→up pass (including skip connections held across the up-path), downloads only the final
output.

**Success metric, corrected per review — NOT simply "1 submit"**: minimize the necessary
transfer/submit boundaries, and *measure*, don't target a single arbitrary number:
- number of queue submits
- number of command buffers
- host↔device transfer bytes
- staging copies
- fence waits
- dispatch count

A design with 2-3 submits per step for a real, legitimate reason (e.g. a temporary CPU attention
island in Stage 3a) is a valid intermediate state, not a failure — as long as each stage's numbers
move in the right direction versus the previous stage, measured for real.

## CPU-island discipline (new rule, per review)

After residency work begins, any CPU-side operation remaining in `Forward()` must be explicitly
labeled as one of: **unavoidable model logic**, **temporary diagnostic fallback** (e.g. Stage 3a's
CPU attention), **known implementation gap**, or **performance bug**. Undocumented CPU work
creeping back in between GPU ops is exactly the failure mode this whole effort is trying to remove
— don't let it happen silently.

## Stage 0 — recording/residency proof-of-concept — DONE 2026-09-12

**Result: option 1 confirmed viable, no fallback needed.** Wrote
`tests/OpenTail.Stingray.Tests.Diffusion/GpuResidencyStage0PocTests.cs`: two GPU tensors uploaded
once, then two dependent `Sgemm` dispatches (second consumes the first's output tensor directly)
recorded into one `BeginRecord()`/`EndRecordAndSubmit()` session with only a `RecordBarrier()`
between them — no `Upload`/`Download` at any point for the intermediate tensor. Result matched a
real CPU reference computation to <1e-3 max abs diff. Passed on the first real run.

**Why this worked with no new Vulkan-backend code**: `Sgemm` already routes through
`DispatchOrRecord`, which already checks `_recording` and records into `_transferCmd` instead of
dispatching+waiting immediately — this mechanism already existed at the compute-dispatch level
(used elsewhere for batched recording, e.g. RRDBNet). The actual, narrower gap the original plan
draft was uncertain about is confirmed to be exactly what it looked like: `Upload`/`Download`
themselves always do an immediate `SubmitAndWait`, never checking `_recording` — but as long as no
stage needs to Upload/Download a tensor *mid-graph* (the whole point of residency), this is a
non-issue. **Decision: proceed with option 1** (no mid-graph CPU transfers at all) for every
subsequent stage; option 2 (making Upload/Download themselves recordable) is not needed.

## Stage 1 — SDXL Tensor primitive layer — DONE 2026-09-12 (broadened per review)

Not just `Lin`/`Conv`. Build the minimum Tensor-in/Tensor-out primitive set SDXL's `ResBlock`/
`SpatialTransformer` actually need:
- `Lin` / `Conv` (Tensor-in/Tensor-out, reusing existing `Sgemm`/`Conv2dImplicitGemm` — no new
  shader math)
- bias-add and residual-add: use the existing `AddInPlace` directly. **Do not write a new fused
  bias shader yet** (per review point #4) — keep bias as a real GPU tensor, add it with the
  existing elementwise op, benchmark, and only consider fusing into the GEMM epilogue later if
  profiling actually shows it matters.
- `SiLU`, `GroupNorm`/`GroupNormSilu` (already exists), `LayerNorm` — Tensor-in/Tensor-out wrappers
  around what's already real and shipped.
- **GPU layout/reshape operations** (new, per review point #7): SDXL's `SpatialTransformer`
  currently does `[C,H,W] → [HW,C]` and back on the CPU (`xSeq`/`xSpatial` arrays). Once resident,
  this must become either a real GPU transpose kernel or a documented view/stride — not a silent
  CPU round-trip reintroduced under a new name. Decide and implement this explicitly as part of
  Stage 1, not as an afterthought inside Stage 3.

No behavior change in `Forward()` yet — these are new entry points only, verified against the
existing CPU-reference math for each primitive individually before anything is wired in.

## Stage 2 — ResBlock residency in the UNet — DONE 2026-09-12

**Real result**: `ResBlockGpu` wired into `SdxlUNet2DConditionModel.ResBlock` (probe-once/fallback,
same as `VaeDecoder`'s). 512×512/4-step/seed=42: total wall time 137.4s→118.3s (~14% faster),
dispatch count 6511→6304, `stagingCopy` 43.6s→41.9s. Real, verified: output stays visually correct
(coherent, on-prompt apple-orchard content) at the fixed seed. Smaller win than the eventual target
since ResBlocks are a modest share of this UNet's total dispatch count — `SpatialTransformer`
(Stage 3) has far more Linear projections per call and should be the bigger lever. See
`PerformanceLeague.md`'s "GPU residency Stage 0+1+2" row for full detail, including a real
negative-then-fixed sub-finding (`GroupNormSiluGpuTensor`'s uncached weight/bias uploads initially
offset part of the win; caching them, matching the existing `_gpuWeightsNative` convention,
recovered it).

Reuse the *design pattern* already established by `VaeDecoder.ResBlockGpu` (per review point #5 —
don't assume its exact internals port verbatim, since `VaeDecoder`'s other conv path is still
non-resident). Implement `SdxlUNet2DConditionModel.ResBlock` against Stage 1's new primitives.
Verify: real timing + pixel-identical output vs. the current baseline.

## Stage 3a — SpatialTransformer residency, CPU attention fallback (renamed/split per review)

Chain GroupNorm→proj_in→norm→Q/K/V-projection→(**CPU attention, explicitly labeled a temporary
diagnostic fallback**: download Q/K/V → existing CPU `DiffusionOps.MultiHeadAttention` →
re-upload)→output-proj→+residual→(cross-attn, same CPU-attention-island approach)→(FFN: norm→
GEGLU→proj)→proj_out→+residual, with everything except the attention math itself staying
GPU-resident. This isolates a real, clean measurement: how much of the current cost is projection/
layout traffic versus attention math itself. Measure and record before moving to 3b.

## Stage 3b — SpatialTransformer GPU attention (renamed per review)

Only after 3a is measured: re-attempt `MultiHeadAttentionTiled` (the existing, numerically-verified,
currently-unused tiled shader) inside this now-resident context. Both prior regressions were
measured with the surrounding per-op Upload/Download tax still present — this is a genuinely new
experiment, not a repeat. Re-measure; do not assume it wins. `FullSeqAttention` is NOT a shortcut
here (see the verified finding above) — it would need a real GPU rewrite plus qLen≠kvLen support
to be usable, which is out of scope for this stage unless `MultiHeadAttentionTiled` itself proves
inadequate in the new context.

## Stage 4 — Cross-block residency (renumbered)

Remove the CPU `float[]` crossing between successive `ResBlock`/`SpatialTransformer`/`Downsample`/
`Upsample` calls in `Forward()`'s down→mid→up sequence. Skip-connection tensors must stay as GPU
`Tensor`s held across the whole down pass.

**Real measurement gate, not just "check headroom" (per review point #8)**: instrument and record,
before making the whole UNet resident:
- allocated device-local bytes
- live `Tensor` bytes at each point in the forward pass
- peak live bytes across one full forward call
- number of simultaneously-live skip tensors

1280-channel tensors at the deepest resolutions are a real, non-trivial size on this iGPU's shared
UMA memory. Prefer explicitly freeing each skip tensor immediately after its corresponding
concatenation consumes it, rather than holding all of them for the whole pass.

## Stage 5 — One recording session per denoising step (renumbered)

Wrap as much of the resident forward pass as Stages 1-4 achieved into one `BeginRecord()`/
`EndRecordAndSubmit()` session, per the corrected success metric above (minimize necessary
boundaries, don't force exactly 1).

**New risk, first-class per review point #9**: a whole SDXL forward pass means potentially
hundreds to thousands of dispatches referencing many distinct tensors in one recording session —
qualitatively different scale from RRDB-style batching this mechanism was built for. Before
attempting the full-scale version, run a dedicated stress test: can `ComputePipeline`'s
per-recording descriptor-set epoch mechanism safely recycle descriptor resources only after the
referencing submission has actually completed, at this scale? Verify this doesn't corrupt or
alias descriptors under real UNet-scale dispatch counts before trusting a full run's output.

## Stage 6 — Profiling + regression report (renumbered)

Re-run the exact `STINGRAY_PROFILE_GPU_SPLIT=1` SDXL-Turbo benchmark and report the new dispatch
count, submits, transfer bytes, `submitWait`, `stagingCopy`, and total wall time directly against
the original baseline (6511 dispatches / 2.7s submitWait / 43.6s stagingCopy / 137.4s wall) in
`PerformanceLeague.md`.

## Explicit non-goals for this pass

GEGLU/norm fusion beyond `GroupNormSilu`, reusable/persistent command buffers replayed across
denoising steps, INT8/quantized GPU GEMM, any attention-kernel algorithm work beyond re-measuring
`MultiHeadAttentionTiled` in Stage 3b.

## Open risks

- Stage 0's outcome may force a different Upload/Download design than assumed (recordable
  transfers rather than "no mid-graph transfers at all") — plan accordingly once that result is in.
- Stage 4's memory-pressure numbers are unmeasured until that stage's gate runs.
- Stage 5's descriptor-recycling behavior at full UNet scale is unverified until its stress test
  runs.
- `MultiHeadAttentionTiled`'s Stage 3b re-test could still regress even in a resident context —
  treat as a real open question, not a foregone conclusion.

## Verification discipline (unchanged)

Every stage ends with: (a) a real end-to-end SDXL-Turbo run at a fixed seed, (b) pixel-diff against
the current known-good baseline image, (c) real before/after timing, (d) the
`STINGRAY_PROFILE_GPU_SPLIT=1` profile re-run. A stage that doesn't measurably help or regresses
gets reverted and documented as a negative result, matching this session's existing discipline.
