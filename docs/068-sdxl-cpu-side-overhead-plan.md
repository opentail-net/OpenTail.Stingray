# SDXL CPU-side (non-Vulkan) overhead — implementation plan

**Status**: approved for implementation 2026-09-12, after external review. Revision 2 (folds in
review corrections). **Reframed per review**: this isn't really a "CPU optimization" plan — it's
"finish eliminating CPU orchestration from the VAE" (a GPU-residency track, same family as
docs/067) plus a separate "investigate first-use weight materialization" track (a CPU/I/O overlap
question). Kept as two explicitly separate tracks so measurements stay attributable.

## Why

The GPU residency plan (docs/067) took SDXL-Turbo's real 512×512/4-step Vulkan run from 137.4s to
~77-79s (~42-44% faster) by removing per-op CPU↔GPU round-trips, zero shader math changes. Once
done, `STINGRAY_PROFILE_GPU_SPLIT=1` showed `stagingCopy`+`submitWait` are now only ~10% of the
~77s total. **Over 85% of the remaining wall time never touches Vulkan at all.**

Real, measured stage breakdown (existing `-v` logging, real, not estimated):

```
Text encode (CLIP-L + CLIP-G, cond+uncond):  ~7.5s
Denoise step 1/4 (cold weight cache):        ~23-25s
Denoise step 2/4:                            ~9s
Denoise step 3/4:                            ~9s
Denoise step 4/4:                            ~9s
VAE decode:                                  ~19-20s
                                              -------
Total:                                       ~77-79s
```

Two real, concrete findings, not guesses:
1. Step 1 costs ~14-16s more than steady-state — `CachedWeightReader.Get()`'s lazy first-access
   read+cache (confirmed by reading the actual implementation).
2. VAE decode (~19-20s, ~25% of the run) has NOT had the residency treatment
   `SdxlUNet2DConditionModel` just got. `VaeDecoder` already has a per-block `ResBlockGpu` (the
   actual precedent the whole UNet plan was modeled on) but no cross-block residency,
   `DiffusionOps.Upsample2x` between blocks is still pure CPU (the UNet's analogous
   `Upsample2xGpu` already exists and is proven), and the VAE's one mid-block attention call
   (`MidAttnCompVis`) is CPU-only with no GPU path at all.

**Per review**: don't assume the composition of either number. "14-16s cold-start" could be disk
I/O, dequant compute, allocation, or locking in very different proportions, and each implies a
different fix. "VAE residency will help" is high-confidence for the conv/norm path (proven
machinery) but NOT for the attention block, which has a real prior warning sign: the SDXL UNet's
own attention-kernel work regressed twice (once by ~5x) before a *different* architectural context
made it a win. VAE's attention shape is different (single call, 64×64 resolution) — don't assume
either outcome; measure the convolution/residency path first, since it may make attention moot.

## Revised roadmap (per review, gated by real measurement at every step)

### A0 — Establish a clean baseline
Fixed seed, real end-to-end run, record: full CPU/GPU stage times, and an output image hash (or
saved PNG) to pixel-diff every later stage against. This is the reference point for the whole plan.

### A0/A1 — DONE 2026-09-12, fixed same-day

A0 baseline recorded: SDXL-Turbo 512×512/4-steps/seed=42, 77.0-77.2s total, output PNG SHA-256
`5a738c6f8ccc2551be0783923ed40ed566340b4a95b2ee8742d4b4d31383952f`.

A1 real result (`STINGRAY_PROFILE_WEIGHT_READ=1` added to `SafetensorsLoader.ReadF32`, splitting
metadata lookup / disk I/O / dtype conversion): of a 13.4s total weight-read cost (6.4GB, ~3.4B
floats), disk I/O was only **1.8s** — the dominant cost was `convert` (dequant) at **11.4s**, a
plain scalar F16→F32 loop achieving only ~256M floats/s. **This answers the plan's own open
question decisively**: dequant compute, not disk I/O, dominates — so weight prefetch (Stage
C) would not help (moving *when* a fixed CPU cost happens doesn't reduce it), exactly the risk the
plan's C1 gate was written to catch.

**Fixed same-day** (real, vectorized, verified byte-identical output): replaced the scalar loop
with `TensorPrimitives.ConvertToSingle` (required adding the `System.Numerics.Tensors` package
reference to `OpenTail.Stingray.Core`, which didn't have it — other projects already did). Real
measurement: convert 11.4s→2.4s (4.7x faster), total wall time 77.0s→67.6s (~12% faster), SHA-256
of the output PNG unchanged (byte-identical, zero numerical drift). This fix is backend-independent
(pure CPU dequant) and applies to every safetensors-backed pipeline in this codebase, not just
SDXL. See `PerformanceLeague.md`'s Stage A1 row for full detail.

**Remaining from A1 for a future pass, not blocking**: A1's per-weight instrumentation (first-
access order, per-tensor timing) described in the original plan wasn't built — the aggregate
numbers above already answered the "disk vs. dequant vs. allocation" question decisively enough
to act on, so the more granular per-weight breakdown wasn't needed to make this call. A2 (VAE
instrumentation) and A3 (text encode split) also remain for whoever picks up Stage B.

### A1 (original wording, superseded by the above) — Instrument `CachedWeightReader` (4-way split, not 2)
Not just "disk read + dequant" — measure separately:
- safetensors metadata/lookup time
- raw tensor bytes read (disk I/O)
- F16/quantized → F32 conversion (dequant compute)
- cache insertion / array allocation

Also record, per cold miss: weight name, first-access order, byte count, output float count. This
gives real throughput numbers (time/GB, time/million floats) that reveal whether the bottleneck is
storage bandwidth, dequant CPU cost, allocation overhead, or many-tiny-reads — each implies a
different fix, and guessing wrong wastes the rest of the plan on the wrong track.

### A2 — Instrument VAE's real breakdown
Time `VaeDecoder.Decode()`'s existing pieces separately: per-`ResBlock` (GPU-resident vs CPU
fallback), `Upsample2x`, `MidAttnCompVis`, any remaining CPU `GroupNorm`/`SiluInPlace` calls not
already covered by `ResBlockGpu`, and every CPU↔GPU boundary crossing (this is new, per review:
`VaeDecoder` is architected around `float[]`, so also count/time `float[]` allocations, `.Clone()`
calls, and Tensor↔float[] upload/download conversions — even if not dominant, this is the real
baseline to compare the residency rewrite against).

### A3 — Instrument text encode
CLIP-L vs CLIP-G split, tokenization vs. forward pass. Lower priority than A1/A2, real and
currently unmeasured.

**Gate**: do not proceed past A-stages until real numbers are in. This is the same discipline the
GPU residency plan used (measure before attempting a fix) — reapplied here rather than assumed
already satisfied just because it worked once.

### B0 — Isolated parity test: `Upsample2x` vs `Upsample2xGpu`
Per review: the primitive exists, but do not call swapping it "pure wiring" without verifying
numerical semantics first (coordinate mapping, channel layout, edge behavior, output ordering) via
a real tensor-level parity test — the same rigor every other GPU primitive in docs/067 got before
being wired into the real pipeline.

### B1 — VAE mid-block + one up-block residency (small, isolated, measured)
Per review: do NOT rewrite the whole VAE in one shot. Make the mid-block and the first up-block
GPU-resident (chaining `ResBlockGpu` calls without a CPU round-trip between them, using B0's
verified `Upsample2xGpu`), leave the rest of `Decode()` as-is, measure. If this alone cuts VAE
decode meaningfully (e.g. 20s→11s), that's a strong, clean signal before committing to the full
rewrite.

### B2 — VAE remaining ResBlocks resident
Extend B1's pattern to the rest of the down/up-block sequence, measure incrementally.

### B3 — GPU upsample wired throughout
Apply B0's verified `Upsample2xGpu` at every remaining upsample call site, measure.

### B4 — Remove remaining VAE CPU↔GPU boundaries
Final cleanup pass: only one Upload (input latent) and one Download (final RGB) for the whole
`Decode()` call, matching `ForwardGpu`'s shape. Measure.

### B5 — VAE mid-block attention: real experiment, not an assumption
Per review, explicitly reframed as an experiment with a real go/no-go gate, not "wire in the proven
tiled shader":
1. Measure the CPU `MidAttnCompVis` call's real cost in isolation (from A2).
2. Implement a GPU version (norm/QKV via existing `LayerNormGpu`/`LinGpuTensor`, attention via
   `MultiHeadAttentionTiled`, following the UNet's `AttentionIsland` pattern).
3. Verify via an isolated numerical parity test first.
4. Measure real end-to-end VAE timing with it wired in.
5. **Keep only if faster.** If B1-B4 already reduced VAE decode enough that this single 64×64
   attention call is a small fraction of what's left, this stage may not be worth pursuing at all
   — decide from B1-B4's real numbers, not in advance.

### C0 — Determine the real first-step weight working set
Using A1's per-weight instrumentation: which weights are actually read on step 1, in what order,
how many bytes. Do not blindly prefetch "the whole UNet."

### C1 — Prototype background weight prefetch (only if A1 shows I/O, not dequant, dominates)
Per review, this is now explicitly gated behind A1's finding, not attempted regardless. Also
resolve, before writing any concurrent code: is `CachedWeightReader`'s `lock (_cache)` (which wraps
both the dictionary lookup AND the `ReadF32` call) going to serialize a background prefetch against
the main thread's own reads? Is the underlying `IWeightLoader`/file reader safe for concurrent
reads at all? Does CLIP-L/CLIP-G's own text encoding already saturate available CPU cores, such
that background dequant work would slow text encoding down more than it saves — a real risk on
this Ryzen 7 5700G, not hypothetical.

### C2 — Only retain if TOTAL pipeline wall time improves
Explicit acceptance criterion per review: "prefetch itself got faster" is not sufficient; the whole
`Generate()` call's wall time must decrease. If contention with text encoding erases the gain,
revert.

### D — Final report
Same as docs/067's Stage 6: re-run the full stage breakdown against this plan's own A0 baseline,
report real before/after numbers in `PerformanceLeague.md`.

## Explicit non-goals for this pass

- **Do not pursue further denoise-step optimization** (steps 2-4 are already ~9s steady-state,
  proof the residency work succeeded) unless A-stage profiling shows real, material CPU-side cost
  remains there specifically — otherwise this plan drifts back into re-litigating docs/067's
  already-closed work.
- Rewriting `CachedWeightReader`'s dequantization to be faster (SIMD dequant paths etc.) — only if
  A1 shows dequant compute, not disk I/O, dominates the cold-cache cost.
- PNG/image encoding optimization — only if A-stage profiling shows it's non-negligible (not
  assumed either way yet).
- Any further Vulkan/GPU-residency work on the UNet itself — docs/067 is complete.
- Multi-request/server-side weight-cache sharing across separate generations — a different,
  larger architectural question for `SdxlPipeline`'s lifetime management, out of scope here.
- VAE attention GPU port is explicitly NOT assumed to be a win (see B5) — treat as a real
  experiment with a keep/revert gate, following the UNet attention work's own history of two
  regressions before one context made it a genuine win.

## Verification discipline (unchanged from docs/067)

Every stage: real end-to-end run at a fixed seed, pixel-diff against A0's baseline image, real
before/after timing, commit separately with the required attribution footer. A stage that doesn't
measurably help or regresses gets reverted and documented as a negative result, not silently
absorbed into the next stage.
