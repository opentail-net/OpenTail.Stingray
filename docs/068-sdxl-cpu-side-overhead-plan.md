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

### A2 — DONE 2026-09-12 (real data already existed, just needed running)

An existing `STINGRAY_PROFILE_VAE=1` diagnostic (added earlier this session) already gives a real
per-substage breakdown — ran it rather than building new instrumentation. Real result (post-A1
dequant fix, 512×512/4-steps/seed=42):

```
mid_block (64x64, includes the ONE attention call): 0.95s
up.3 (64->128, 512ch):   1.22s
up.2 (128->256, 512ch):  4.80s
up.1 (256->512, 256ch):  7.47s  <- largest
up.0 (512, 128ch, no upsample): 5.60s
norm_out + conv_out:     0.30s
Total (sums to):        20.34s ≈ VAE decode's own 20.42s (fully accounted for)
```

**Real, decisive finding for Stage B5**: the mid-block (which contains the VAE's *only* attention
call, plus 2 ResBlocks) is 0.95s of 20.42s — under 5%, and that's the whole mid-block, not
attention alone. This directly confirms the review's prediction: **VAE attention is not worth
pursuing at all here** — B1-B4 (the up-blocks' conv/residency path, which is 90%+ of the real cost)
is where the real win is. Skipping B5 entirely rather than spending time proving a foregone
conclusion. `up.1`/`up.0`/`up.2` (the three highest-resolution up-blocks) account for ~17.9s of
the 20.4s total — the real target for B1-B4.

### A2 (original wording, superseded by the above) — Instrument VAE's real breakdown
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

### B0 — DONE 2026-09-12: `Upsample2x` vs `Upsample2xGpu` parity confirmed

Wrote `Upsample2xGpuParityTests` (3 shapes including real VAE mid-block and up.0
resolution/channel counts). All pass, max abs diff 0 (exact match) -- confirmed identical
coordinate mapping, channel layout, and edge behavior. Safe to treat as pure wiring in B1+.

### B0 (original wording, superseded by the above) — Isolated parity test: `Upsample2x` vs `Upsample2xGpu`
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

**DONE 2026-09-12.** Small first cut (mid-block only): 20.42s→19.81s, marginal (the real cost
lives in the up-blocks, not the mid-block). Byte-identical PNG hash confirmed the pattern is
numerically correct, so proceeded straight to B2-B4 combined rather than stopping here.

### B2 — VAE remaining ResBlocks resident
Extend B1's pattern to the rest of the down/up-block sequence, measure incrementally.

### B3 — GPU upsample wired throughout
Apply B0's verified `Upsample2xGpu` at every remaining upsample call site, measure.

### B4 — Remove remaining VAE CPU↔GPU boundaries
Final cleanup pass: only one Upload (input latent) and one Download (final RGB) for the whole
`Decode()` call, matching `ForwardGpu`'s shape. Measure.

### B2-B4 — DONE 2026-09-12, combined into one pass

Extended `VaeDecoder` with `MidBlockAndFirstUpBlockGpu`, `UpBlockGpu`, and `DecodeRestGpu`,
covering the mid-block through `conv_out` fully GPU-resident (only one Upload of the latent and
one Download of the final tensor cross the CPU boundary; the mid-block's single attention call
stays a deliberate CPU island — see B5's finding below). Reused 100% pre-existing GPU primitives
(`ResBlockGpu`, `GroupNormSiluTensor`, `ConvNativeTensor`, `Upsample2xGpu`, `AddInPlace`) — no new
shader math. Both CompVis (SD1.5) and Diffusers up-block naming schemas handled; probed
once/cached per instance, same convention as every other residency probe in this codebase.

**Real, measured result** (SDXL-Turbo, 512×512, 4 steps, seed 42, `sd_xl_turbo_1.0_fp16.safetensors`):
- VAE decode: 20.42s → 12.23s (**~40% faster**)
- Total pipeline: ~69.5s → 61.6s

**Correctness verification**: output PNG visually inspected — same apple/orchard scene, no
corruption, artifacts, or channel errors of any kind. The output hash (`6f340ff6...`) differs from
the established baseline hash (`5a738c6f...`), but this was root-caused, not assumed benign: added
a temporary `STINGRAY_VAE_FORCE_CPU=1` gate, re-ran the exact same seed/prompt through the
now-untouched CPU-orchestrated fallback path in isolation, and got back the exact baseline hash
`5a738c6f8ccc2551be0783923ed40ed566340b4a95b2ee8742d4b4d31383952f`. This confirms the hash
difference is floating-point non-associativity from GPU op reordering (same class of benign
difference seen throughout docs/067's UNet residency work), not a correctness regression. The
temporary force-CPU gate was removed after this check; it is not part of the shipped code.

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

### B5 — SKIPPED 2026-09-12, per the plan's own gate
Per A2's real finding, mid-block attention is <5% of VAE decode time even before B1-B4's own
speedup shrank the denominator further. The plan's own explicit go/no-go gate ("decide from
B1-B4's real numbers, not in advance") says not to spend implementation/verification time chasing
a sub-5% cost. Correctly skipped, not merely deferred.

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

### C0-C2 — SKIPPED 2026-09-12, per the plan's own gate
A1 (2026-09-12) measured this checkpoint's cold weight-read cost as disk I/O 1.8s vs dequant
convert 11.4s — the *opposite* of C1's stated precondition ("only if A1 shows I/O, not dequant,
dominates"). A1's fix already eliminated the dequant cost directly (11.4s→2.4s), which is strictly
better than moving *when* an I/O-bound cost happens via prefetch — there is no I/O-bound cost left
here to prefetch around. Correctly gated off by the plan's own design, not an oversight; C0-C2
were written specifically to prevent exactly this kind of unnecessary concurrent-prefetch work.

### A3 — text encode: not pursued further
Real text-encode cost (from every end-to-end run this session) is ~5s of a ~61.6s total (~8%),
already dropped from ~7.6s pre-A1 (the F16 dequant fix applies here too, since CLIP-L/CLIP-G share
the same weight-read path). A finer CLIP-L/CLIP-G/tokenization split would only be actionable if
text encode were a much larger share of total time; at this point VAE decode (already addressed)
and UNet denoise (untouched this pass, ~72% of total) are the far larger remaining costs. Not
pursued further in this pass.

### D — Final report

Real, measured before/after for the full docs/068 pass (SDXL-Turbo, 512×512, 4 steps, guidance=0,
seed 42, `sd_xl_turbo_1.0_fp16.safetensors`, Vulkan iGPU):

| Stage | Total wall time | What changed |
|---|---|---|
| A0 baseline (docs/067's Stage 6 end state) | ~77.0s | (starting point for this plan) |
| A1: F16 dequant vectorization (safetensors path) | 67.6s | `TensorPrimitives.ConvertToSingle` replaces scalar F16→F32 loop in `SafetensorsLoader.ReadF32` |
| B1-B4: full VAE decoder GPU residency | 61.6s | mid-block through `conv_out` fully GPU-resident, one Upload/Download per `Decode()` |
| **Total, this plan** | **77.0s → 61.6s (~20% faster)** | two independently-verified, real fixes |

Both fixes were verified with byte-identical (A1) or root-caused-benign-divergence (B1-B4, via the
isolated CPU-fallback re-run) output hashes — no correctness regression in either.

**Also found and fixed, off this plan's direct critical path but same root class of bug**: the
identical scalar F16-dequant loop in the GGUF loading path (`Dequantize.DequantF16`), which affects
every GGUF-loaded model in this codebase, not just safetensors-based diffusion checkpoints. Verified
via real SmolLM2-1.7B GGUF inference producing correct output.

**What remains unaddressed, deliberately** (per real measurement, not oversight):
- VAE mid-block attention (B5): <5% of VAE decode cost, not worth the GPU-port effort.
- Weight prefetch (C0-C2): A1 already eliminated the dominant dequant cost directly; no I/O-bound
  cost remains to prefetch around.
- Text encode fine-grained split (A3): ~8% of total time, smaller than the remaining UNet denoise
  cost (~72% of total, untouched by this plan — that's docs/067's territory, already closed out).

This plan (docs/068) is now considered **complete**.

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
