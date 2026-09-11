# SDXL UNet whole-graph GPU residency — implementation plan

**Status**: plan only, not yet implemented. Written 2026-09-12 for external review before starting.

## Why

A real, measured profiling pass (`STINGRAY_PROFILE_GPU_SPLIT=1`, see `PerformanceLeague.md`'s
"Real GPU-split profiling" row and `docs/00-current-work.md`'s 2026-09-12 entry) found that in a
137.4s SDXL-Turbo Vulkan run (512×512, 4 steps):

- GPU submit + execution + fence-wait: **2.7s (~2%)**
- CPU-side staging-buffer `Map()`/memcpy (upload/download): **43.6s (~32%)**, across **6511**
  separate dispatch/upload/download round-trips
- Everything else (CPU-side im2col, dequant, orchestration): ~65%

The GPU itself is not the bottleneck. The current architecture treats the GPU as a synchronous
RPC coprocessor: `SdxlUNet2DConditionModel`'s `Lin()`/`Conv()` each do
`Upload(x) → Sgemm/Conv-shader → Download(result) → Free(x, result)`, individually, per call.
`SpatialTransformer()` at the deepest blocks (depth=10) chains on the order of 100 such calls.
Two independent attempts to fix this by improving the GPU attention kernel (a naive shader, then
a properly tiled flash-attention shader) both **regressed** real timing — because the kernel was
being asked to overcome a cost that exists one level above it. Neither attempt changed the
surrounding Upload/Download pattern.

## Existing precedent already in this codebase

This is not a new architecture from scratch — a smaller-scoped version of it already exists and
is verified working:

- `VaeDecoder.ResBlockGpu` (`src/OpenTail.Stingray.Diffusion/VaeDecoder.cs`): one `ResBlock`
  (`norm1→silu→conv1→norm2→silu→conv2→+skip`) runs as a single GPU-resident `Tensor`-in/
  `Tensor`-out chain — one `Upload` of the block's input, every intermediate stays a GPU `Tensor`,
  one `Download` of the result. Real, measured win: ~22.5% faster VAE decode.
- `VulkanBackend` already has the batching primitives this needs: `BeginRecord()`/
  `EndRecordAndSubmit()` (record N dispatches, submit once), `BeginBatch()`/`EndBatch()` (same,
  plus deferred `Free()`s), `RecordBarrier()` (compute→compute dependency barrier within a
  recording session). These are already used elsewhere (e.g. RRDBNet's batched conv path).
- `GroupNormSilu` (fused GPU shader), `Conv2dImplicitGemm` (tiled-GEMM native conv shader),
  `LinMulti` (dedup shared-input uploads across sibling projections) are all real, already-shipped
  GPU-resident or upload-reducing primitives that the new work should reuse, not replace.

**What's missing**: this pattern has only been applied to one block type (`ResBlock`, and only in
`VaeDecoder`, not yet in `SdxlUNet2DConditionModel`), not to `SpatialTransformer` (attention +
FFN), and not chained *across* blocks — every `ResBlock`/`SpatialTransformer` call in the UNet's
`Forward()` still crosses back to a CPU `float[]` between blocks today.

## Target end state

`SdxlUNet2DConditionModel.Forward()` uploads its real inputs (latent, timestep embedding, cross-
attention context) ONCE, keeps every intermediate activation as a GPU `Tensor` for the entire
down→mid→up pass (including skip-connection tensors held across the up-path), and downloads only
the final output. Ideally the whole forward pass for one denoising step is recorded into a single
`BeginRecord()`/`EndRecordAndSubmit()` session (one `vkQueueSubmit`, one fence wait) instead of
~6511 of them.

## Constraint that must be resolved first

`VulkanBackend.Upload()`/`Download()` currently do their own **immediate** `CopyBuffer` +
`SubmitAndWait()` — they do not check `_recording` and are not currently appendable into an
in-progress `BeginRecord()` session. `BeginRecord`/`BeginBatch` today only batch *compute
dispatches* (`Sgemm`, shader calls), not upload/download staging copies.

Two ways to resolve this, either is viable and should be decided as part of implementation, not
guessed here:

1. **Preferred / simpler**: restructure so mid-graph `Upload`/`Download` calls simply don't
   happen at all — every op between the single initial upload and the single final download
   consumes and produces GPU `Tensor` handles directly (this is what `ResBlockGpu` already does
   for one block; extend the same idea to the whole graph). Then a single `BeginRecord()` at the
   top of `Forward()` and `EndRecordAndSubmit()` at the bottom naturally covers every dispatch
   in between, with no upload/download in the middle to worry about.
2. **Fallback if (1) hits a real blocker**: extend `Upload`/`Download` (or add new
   `RecordUpload`/`RecordDownloadToStaging`-style variants — `RecordDownloadToStaging` already
   exists for a different caller) to be recordable into `_transferCmd` when `_recording` is true,
   deferring the actual host-visible copy until after the session's `EndRecordAndSubmit()`.

## Staged implementation plan (each stage independently shippable and verifiable)

Every stage ends with: (a) a real end-to-end SDXL-Turbo run at a fixed seed, (b) pixel-diff
against the current known-good baseline image (must be identical or explain any real, expected
numerical drift), (c) real before/after timing, (d) the `STINGRAY_PROFILE_GPU_SPLIT=1` profile
re-run to confirm dispatch count is actually dropping. A stage that doesn't measurably help or
regresses gets reverted and documented as a negative result, same discipline as the rest of this
session's perf work.

**Stage 1 — GPU-resident `Lin`/`Conv` primitives in `SdxlUNet2DConditionModel`.**
Add `Tensor`-in/`Tensor`-out variants of `Lin()`/`Conv()` (analogous to `VaeDecoder`'s
`ConvNativeTensor`/`GroupNormSiluTensor`) that take and return `CoreTensor` handles instead of
`float[]`, reusing the exact same `Sgemm`/`Conv2dImplicitGemm` dispatches already in use — no new
shader math. Bias-add (currently a CPU `TensorPrimitives.Add` loop after `Download`) needs a
GPU-side equivalent (either a small fused bias-add shader, or folding bias into the existing
GEMM/conv shader's epilogue if that's a smaller change). No behavior change yet — these are just
new entry points, not wired into `Forward()`.

**Stage 2 — `ResBlock` residency in the UNet (not just `VaeDecoder`).**
Port the exact `VaeDecoder.ResBlockGpu` pattern into `SdxlUNet2DConditionModel.ResBlock`, using
Stage 1's primitives plus the existing `GroupNormSilu` shader. Verify real timing + pixel-identical
output before moving on — this block type's residency is already proven safe in `VaeDecoder`, so
this stage is mostly plumbing, not new risk.

**Stage 3 — `SpatialTransformer` residency.**
Chain `GroupNorm→proj_in→(self-attn: norm→Q/K/V proj→attention→out proj→+residual)→
(cross-attn: norm→Q proj→K/V proj→attention→out proj→+residual)→(FFN: norm→GEGLU→proj→+residual)→
proj_out→+residual` as GPU `Tensor` ops throughout, using `LinMulti`'s existing shared-upload-dedup
idea generalized to the Tensor-in/Tensor-out form. Attention itself is the one open question:
- Initially, the safest option is one Download/re-Upload just around the attention math itself
  (real Q/K/V download → existing CPU `DiffusionOps.MultiHeadAttention` → re-upload the result),
  keeping everything else in this block GPU-resident. This alone should remove the majority of
  this block's ~100 Upload/Download round-trips (every Linear projection was one; now only the
  attention math crosses).
- Once that's shipped and measured, it's worth **re-attempting the existing tiled GPU attention
  shader** (`MultiHeadAttentionTiled`, already implemented and numerically verified, currently
  unused after regressing in isolation) inside this now-resident context — both prior regressions
  were measured with the *surrounding* per-op Upload/Download tax still present, which this
  profiling data suggests may have been the dominant cost the attention kernel was fighting, not
  its own math. Re-measure; do not assume it will win without a real run.

**Stage 4 — chain across blocks.**
Remove the CPU `float[]` crossing between successive `ResBlock`/`SpatialTransformer`/
`Downsample`/`Upsample` calls in `Forward()`'s down→mid→up sequence. Skip-connection tensors
(concatenated back in on the up-path) need to stay as GPU `Tensor`s held across the whole down
pass, not `float[]` as today — check real GPU memory headroom for holding several of these
simultaneously against this iGPU's ~16GB placement budget (see `PrintDeviceInfo`'s reported
values), since this is a real new constraint the current per-block-round-trip design never had to
consider.

**Stage 5 — one recording session per denoising step.**
Wrap the whole `Forward()` call (or as much of it as Stages 1-4 made resident) in a single
`BeginRecord()`/`EndRecordAndSubmit()`, replacing whatever per-block `EndBatch()` granularity
Stages 2-4 may have used as an intermediate stepping stone. This is the stage that should collapse
the measured 6511-dispatch/6511-round-trip count down to close to 1 real submit per step (a
handful more for the label-embedding/time-embedding setup and the final download).

**Stage 6 — re-measure and report.**
Re-run the exact `STINGRAY_PROFILE_GPU_SPLIT=1` SDXL-Turbo benchmark from the original
measurement and report the new dispatch count, `submitWait`, `stagingCopy`, and total wall time
next to the original 6511/2.7s/43.6s/137.4s baseline in `PerformanceLeague.md`.

## Explicit non-goals for this pass

Deferred, per the already-agreed phase order (measure → residency → fusion → reusable command
graphs → INT8 → attention-kernel-sophistication, attention last):
- Fusing GEGLU's two GEMMs into one kernel, or fusing norm+activation+projection into one shader
  beyond what `GroupNormSilu` already does — worth doing, but after residency proves out, not
  before.
- Reusable/persistent command buffers replayed across denoising steps with only push-constants
  changed (Stage 5 gets one submit *per step*; making that same recorded graph replayable across
  *all* steps without re-recording is a separate, later piece of work).
- INT8/quantized GPU GEMM.
- Any further attention-kernel algorithm changes beyond re-measuring the existing tiled shader in
  its new (residency) context per Stage 3.

## Open risks to flag for review

- GPU memory pressure from holding multiple skip-connection tensors resident simultaneously
  (Stage 4) — not yet measured against this iGPU's real placement budget.
- Whether `ComputePipeline`'s per-recording descriptor-set epoch mechanism (built for smaller
  batches like RRDBNet's conv batching) holds up correctly at UNet-forward-pass scale (potentially
  hundreds of dispatches in one recording session) — needs verification, not assumed.
- The Upload/Download-recordability constraint above (resolved via option 1 or 2) is a real open
  design decision, not a settled detail.
