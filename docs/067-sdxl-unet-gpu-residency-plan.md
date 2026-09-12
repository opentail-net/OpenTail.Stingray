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

## Stage 3a — SpatialTransformer residency, CPU attention fallback — DONE 2026-09-12 (renamed/split per review)

**Real result, the largest single win so far**: total wall time 118.3s→90.5s (~23.5% faster than
Stage 2, ~34% faster than the original 137.4s baseline). Dispatches 6304→4484 (1820 fewer),
`stagingCopy` 41.9s→17.5s (~58% drop), steady-state denoise step ~19s→~12.7s. Required 5 new GPU
primitives (`LayerNormGpu`, `GeGlu`, `PermuteChwToHwc`/`PermuteHwcToChw`, `GroupNormGpu`), all
verified against real CPU references before wiring in. Caught a real correctness bug mid-
implementation: initially used `GroupNormSilu` for `SpatialTransformer`'s pre-`proj_in` norm, which
fuses an activation the real model doesn't have there (unlike ResBlock's norms) — added a plain
`GroupNormGpu` instead before it ever ran end-to-end. Output re-verified pixel-identical to Stage
2's baseline at the same seed. Confirms the CPU attention island is exactly where nearly all of
this block's remaining round-trips now concentrate — see `PerformanceLeague.md`'s Stage 3a row for
full detail.

Chain GroupNorm→proj_in→norm→Q/K/V-projection→(**CPU attention, explicitly labeled a temporary
diagnostic fallback**: download Q/K/V → existing CPU `DiffusionOps.MultiHeadAttention` →
re-upload)→output-proj→+residual→(cross-attn, same CPU-attention-island approach)→(FFN: norm→
GEGLU→proj)→proj_out→+residual, with everything except the attention math itself staying
GPU-resident. This isolates a real, clean measurement: how much of the current cost is projection/
layout traffic versus attention math itself. Measure and record before moving to 3b.

## Stage 3b — SpatialTransformer GPU attention — DONE 2026-09-12, REAL REVERSAL (renamed per review)

**Real result: the tiled attention shader wins.** The exact same `MultiHeadAttentionTiled` shader
that regressed twice before (once as a naive shader, once as this tiled shader — both measured in
a non-resident context) is now a real, measured improvement: total wall time 90.5s→78.6s (~13%
faster than Stage 3a, ~43% faster than the original 137.4s baseline), dispatches 4484→2244 (nearly
halved), `stagingCopy` 17.5s→8.3s. The only thing that changed is that Q/K/V are now already
GPU-resident tensors (from `LinGpuTensor`, never downloaded) — the shader itself is unmodified.
This is real, direct vindication of the whole residency hypothesis: neither prior attention-kernel
attempt was actually testing kernel quality, they were both measuring the architectural tax around
the kernel. Output re-verified pixel-identical to Stage 3a's baseline at the same seed. Implemented
as `AttentionIsland` with its own independent probe/fallback (separate from the whole-block
residency flag), so a future regression here falls back to Stage 3a's CPU-island chain specifically.

Only after 3a is measured: re-attempt `MultiHeadAttentionTiled` (the existing, numerically-verified,
currently-unused tiled shader) inside this now-resident context. Both prior regressions were
measured with the surrounding per-op Upload/Download tax still present — this is a genuinely new
experiment, not a repeat. Re-measure; do not assume it wins. `FullSeqAttention` is NOT a shortcut
here (see the verified finding above) — it would need a real GPU rewrite plus qLen≠kvLen support
to be usable, which is out of scope for this stage unless `MultiHeadAttentionTiled` itself proves
inadequate in the new context.

## Stage 4 — Cross-block residency — DONE 2026-09-12 (renumbered)

**Real result**: total wall time 78.6s→77.1s (~2% further faster, ~44% faster than the original
137.4s baseline), dispatches 2244→1974. Memory gate answered with real numbers instead of
"checked headroom": peak device-local usage 6.67GB / 1831 live tensors at peak, well within this
iGPU's ~16GB placement budget at 512×512 -- no pressure found, so the "free skips immediately"
discipline below was precautionary rather than load-bearing at this resolution (worth re-checking
at 1024×1024, not yet done). `ForwardGpu` chains the entire down→mid→up sequence with only one
Upload (initial latent) and one Download (final output) for the whole forward pass; skip tensors
are freed immediately after their concat consumes them. Two bounded CPU islands remain (the 2
stride=2 downsample convs -- neither Tensor conv primitive supports stride>1, a real scoped gap).
Smaller win than Stages 2/3a/3b since most round-trips were already gone by then. Output
re-verified pixel-identical to Stage 3b's baseline at the same seed.

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

## Stage 5 — One recording session per denoising step — ATTEMPTED 2026-09-12, REVERTED (real negative result)

**Real failure, not yet solved.** First attempt: wrapped each CPU-island-bounded segment of
`ForwardGpu` in `imageOps.BeginBatch()`/`EndBatch()` (3 segments, split at the 2 stride=2
downsample-conv CPU islands) — `DispatchOrRecord` already auto-inserts a barrier after every
dispatch when batching (`_deferringFrees`), so no manual `BatchBarrier()` calls were needed for the
compute-dispatch chaining itself.

**Real crash on the very first real run**: `Error: [-13] ErrorUnknown - Vulkan error occured`,
right at the start of the first denoise step. Root cause (not yet fixed): `GetGpuWeight`/
`GetGpuBias`/`GetNativeConvWeights` cache GPU weight/bias tensors, but **upload them lazily on
first use** — a real, foreseen risk flagged before implementing (see the code comment removed with
this revert): on a cold cache (every real generation's first denoising step, since a fresh process
has never uploaded any of this checkpoint's weights yet), the FIRST call to essentially every
`Lin`/`Conv`/norm helper inside the batched region hits a cache miss and calls `Upload()` — which
internally does its own immediate `vkQueueSubmit`+fence-wait via `SubmitAndWait`. Calling that
while `_transferCmd` is still mid-recording (between `BeginBatch()`'s `vkBeginCommandBuffer` and
the matching `EndBatch()`'s `vkEndCommandBuffer`) is invalid Vulkan usage — submitting a command
buffer that hasn't been ended — and the driver correctly errors out.

**Reverted immediately, per this plan's own discipline** (do not silently absorb a failure into
the next stage): `SdxlUNet2DConditionModel.cs` restored via `git checkout` to Stage 4's committed
state, rebuilt and re-confirmed working before moving on.

**Follow-up attempt, same day — fixed the crash, found a DIFFERENT real regression.** The
"warm cache" theory above was incomplete: the actual unconditional-per-call `Upload()` causing the
crash was `ResBlockGpu`'s timestep-embedding upload (`tEmb` is real per-step activation data, not
a cacheable weight — it was being re-uploaded fresh on every one of the ~17 `ResBlockGpu` calls,
every single time, forever, not just on a cold cache). Fixed properly: `UploadSiluTEmb` now
uploads and SiLU-activates the shared `tEmb` ONCE per `ForwardGpu` call (outside any batch), and
`ResBlockGpu` takes the already-resident `Tensor` instead of re-uploading raw `float[]` each call.
This genuinely fixed the crash — batching then ran to completion.

**But real measurement then showed batching itself is a regression, not a win, on this iGPU**:
`submitWait` ballooned from ~1s to **23.9s** (an ~25x increase), peak GPU memory rose from 6.67GB
to 8.35GB (deferred frees hold far more simultaneously-live tensors per large batched segment than
the per-op-immediate path ever did), and total wall time was slightly *worse* than Stage 4
(79.1s vs 77.1s) despite dispatch count dropping (1974→1919). **Reverted the batching entirely**
(kept the real, correct `tEmbGpu`-once-per-call fix, which is a genuine simplification worth
keeping on its own — verified pixel-identical, dispatch count 1974→1910 from removing 16 redundant
uploads — but measured within run-to-run noise for wall time, not a clear win by itself either).

**Conclusion**: Stage 5, as designed (segment-level `BeginBatch`/`EndBatch`), is a confirmed real
negative result on this hardware, not just an implementation bug — large batched command-buffer
submissions appear to cost more in submit/wait latency and memory pressure than they save in
per-dispatch overhead, on this specific iGPU. This does not necessarily generalize to a discrete
GPU. Not pursued further this pass; `_unetForwardResidencySupported`'s Stage 4 (per-op-immediate,
fully cross-block-resident) path remains the shipped state.

## Stage 5 — One recording session per denoising step (original stage description, not yet re-attempted)

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
