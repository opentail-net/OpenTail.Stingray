# SDXL CPU-side (non-Vulkan) overhead — implementation plan

**Status**: plan only, not yet implemented. Written 2026-09-12, follow-on to
`docs/067-sdxl-unet-gpu-residency-plan.md` (now complete — see its Stage 6 final report).

## Why

The GPU residency plan took SDXL-Turbo's real 512×512/4-step Vulkan run from 137.4s to ~77-79s
(~42-44% faster) by removing per-op CPU↔GPU round-trips, with zero shader math changes. Once that
was done, re-running the same `STINGRAY_PROFILE_GPU_SPLIT=1` breakdown showed something new:
`stagingCopy` (~6.8s) + `submitWait` (~1s) is now only **~10%** of the ~77s total. **Over 85% of
the remaining wall time is CPU-side work that never touches Vulkan at all.** That's a real,
different bottleneck than anything the residency plan addressed, and it needs its own investigation
rather than assuming the same fixes apply.

## What the ~77s actually breaks down into (from existing stage-level `-v` logging, real measured)

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

Two things stand out immediately, both real and concrete, not guesses:

1. **Step 1 costs ~14-16s more than steady-state (steps 2-4).** This gap is not attention or conv
   compute — Stage 3b/4 already made those GPU-resident and fast. It's `CachedWeightReader.Get()`
   (`src/OpenTail.Stingray.Diffusion/CachedWeightReader.cs`): every weight is read from the
   safetensors file and cached **lazily, on first access** — the first denoising step is the first
   time almost every UNet weight is ever touched in the process, so it eats a real, one-time
   disk-read + dequant cost that every subsequent step skips.
2. **VAE decode is ~19-20s, roughly a quarter of the whole run — and it has NOT been through the
   residency treatment `SdxlUNet2DConditionModel` just got.** `VaeDecoder.Decode()`
   (`src/OpenTail.Stingray.Diffusion/VaeDecoder.cs`) already has a per-block `ResBlockGpu` (built
   earlier this session, the direct precedent the whole UNet residency plan was modeled on), but:
   - `Decode()` still calls the public `ResBlock()` wrapper once per block, each one doing its own
     Upload-at-entry/Download-at-exit — there is no cross-block residency (VaeDecoder is exactly
     where the UNet was before Stage 4).
   - `DiffusionOps.Upsample2x` between blocks is **pure CPU** (unlike the UNet's `Upsample2xGpu`,
     added during Stage 4) — a real, silent CPU round-trip between every up-block.
   - The VAE's mid-block attention (`MidAttnCompVis`, `VaeDecoder.cs`) is **CPU-only** — no GPU
     path exists for it at all, unlike `SpatialTransformer`'s now-resident GPU attention.

Both are concrete, bounded, already-precedented fixes — not new architecture.

## Staged plan

### Stage A — Real profiling of the non-Vulkan CPU time (measure first, same discipline as docs/067)

Before assuming the two items above account for all 85%, get a real breakdown. Add lightweight,
gated timing (same `STINGRAY_PROFILE_*` convention used throughout this codebase) around:
- `CachedWeightReader.Get()`'s cold-miss path specifically (disk read + dequant time, separated
  from cache-hit time) — confirm it explains most of step 1's ~14-16s excess.
- `VaeDecoder.Decode()`'s CPU-only pieces (`Upsample2x`, `MidAttnCompVis`, any remaining
  `DiffusionOps.GroupNorm`/`SiluInPlace` calls not already covered by `ResBlockGpu`).
- The final PNG encode step in the CLI (`ImageCommand.cs`) — cheap to check, not yet measured at
  all; rule it in or out rather than assuming it's negligible.
- Text encode's own internal split (CLIP-L vs CLIP-G, tokenization vs. forward pass) — lower
  priority than the two big items above, but real and unmeasured.

Do not proceed to Stage B/C's specific fixes until this confirms where the time actually is —
matching the exact lesson the GPU residency plan itself just relearned (the "GPU is slow" framing
was wrong; don't repeat that mistake here in a new "CPU is just slow" framing without a real
breakdown).

### Stage B — VAE decoder residency (apply the proven Stage 2-4 methodology)

This is the highest-confidence lever: `VaeDecoder` needs exactly what `SdxlUNet2DConditionModel`
already got, using primitives that already exist from the UNet work (no new shader math required
except the mid-block attention item below):

1. **Cross-block residency**: rewrite `VaeDecoder.Decode()`'s block sequence to keep `z` as a GPU
   `Tensor` throughout (mirroring `SdxlUNet2DConditionModel.ForwardGpu`), uploading the input latent
   once and downloading the final RGB output once, instead of round-tripping at every `ResBlock()`
   call.
2. **GPU upsample**: swap `DiffusionOps.Upsample2x` for the UNet's existing `Upsample2xGpu`
   (`IImageOpsBackend`, already implemented and used) — this primitive already exists, this is
   pure wiring, not new work.
3. **Mid-block attention residency**: `MidAttnCompVis` is a single self-attention call (VAE has
   exactly one, at the bottleneck resolution, unlike the UNet's many `SpatialTransformer` calls) —
   port it to use the existing `MultiHeadAttentionTiled` GPU primitive (proven to win in a resident
   context per docs/067 Stage 3b) plus `LayerNormGpu`/`LinGpuTensor` for its surrounding norm/QKV
   projections, following the exact `AttentionIsland` pattern already built for the UNet.

Verification: same discipline as docs/067 — real end-to-end run at a fixed seed, pixel-diff against
the current baseline image, real before/after timing via the existing `-v` stage logging, commit
each real sub-step separately.

### Stage C — Overlap weight loading with text encoding

Text encode (~7.5s) and the UNet/VAE's first weight touch are currently strictly sequential
(`SdxlPipeline.Generate`'s stage order: encode text, *then* start denoising, which is when
`CachedWeightReader` first reads UNet weights). These are independent — text encoding never touches
UNet/VAE weights. If Stage A's profiling confirms the cold-cache cost is real disk I/O + dequant
(not, say, GPU pipeline compilation), a background prefetch (`Task.Run` reading+caching the UNet's
weight tensors while CLIP-L/CLIP-G encode the prompt) could hide some or all of that ~14-16s behind
work that's happening anyway. Real risk to check: whether `CachedWeightReader`/`IWeightLoader` are
thread-safe for concurrent reads from a background prefetch thread while the main thread is still
using the object for text encoding (they're different weight prefixes/objects today, likely safe,
but verify rather than assume before wiring up background reads).

### Stage D — Re-profile and report

Same as docs/067's Stage 6: re-run `-v` stage timing (and Stage A's new CPU breakdown) against the
~77-79s baseline this plan starts from, report real before/after numbers in `PerformanceLeague.md`.

## Explicit non-goals for this pass

- Rewriting `CachedWeightReader`'s dequantization itself to be faster (e.g. SIMD dequant paths) —
  only pursue if Stage A's profiling shows dequant compute (not disk I/O) dominates the cold-cache
  cost; don't assume which one it is.
- PNG/image encoding optimization — only pursue if Stage A's profiling shows it's non-negligible.
- Any further Vulkan/GPU-residency work — that plan (docs/067) is complete; this plan is
  specifically about the newly-exposed non-Vulkan CPU time.
- Multi-request/server-side weight-cache sharing across generations — out of scope; this plan is
  about one `Generate()` call's own internal overlap opportunities, not cross-request caching
  (which is a different, larger architectural question for `SdxlPipeline`'s lifetime management).

## Verification discipline (same as docs/067)

Every stage: real end-to-end run at a fixed seed, pixel-diff against the current baseline image,
real before/after timing, commit separately with the required attribution footer. A stage that
doesn't measurably help or regresses gets reverted and documented as a negative result, not
silently absorbed into the next stage.
