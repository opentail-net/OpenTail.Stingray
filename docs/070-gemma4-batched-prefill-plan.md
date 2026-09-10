# Gemma 4 batched-prefill plan

**Status:** not started. Scoped 2026-09-10 after the PerformanceLeague backfill measured Gemma 4
CPU prefill at 0.12-0.13x of llama.cpp on two model sizes (E4B and 12B), both showing the
prefill≈decode signature that is the known signature of this gap.

## The problem

`ForwardPass.cs:1280` (`perLayerHdUnsupported = _layerHeadDim is not null`) routes every Gemma 4
prefill to the slow token-by-token `Forward()` path instead of the fast batched `PrefillCore` path
that every other supported architecture uses. This is not a missed optimization — it is a
deliberate, correctness-motivated block: `docs/done/gemma4-12b-evidence.md` records that forcing
the batched path via `STINGRAY_PER_LAYER_HD_PREFILL=1` does not merely produce wrong numbers, it
throws `AccessViolationException` — the batched path indexes KV using the model-wide head dim (512)
on layers that actually carry 256, reading/writing past the buffer. That flag now fails fast on
purpose so nobody re-triggers the crash by accident.

Measured cost (2026-09-10 backfill, `PerformanceLeague.md`):

| Model | Prefill ratio vs llama.cpp | Signature |
|---|---:|---|
| Gemma-4-12B-it Q4_K_M | 0.13x | prefill ≈ decode (3.5 vs 4.2 t/s) |
| Gemma4-E4B-it Q4_K_M | 0.12x | prefill ≈ decode (9.8 vs 9.7 t/s) |

Both confirm the original single-model finding (`cpu-performance-baseline.md`'s ~5.7x
size-adjusted prefill penalty) generalizes across the Gemma 4 family, not just one checkpoint.

## What `PrefillCoreAttention` is missing, concretely

Five real features, none currently implemented, all required together (a partial fix that skips
any one of these will silently produce wrong output on some Gemma-4-shaped layer):

1. **Per-layer head dim indexing.** Gemma 4 mixes head dims across layers (e.g. 256 on some layers,
   512 on others via `_layerHeadDim`). `PrefillCoreAttention` currently assumes one model-wide head
   dim baked into its KV indexing arithmetic.
2. **Per-layer KV head count / KV-layer sharing (`_layerKvSrc`).** Some layers share KV storage
   with an earlier layer (MQA/GQA mix); the batched path has no concept of a layer reading another
   layer's KV.
3. **`attention_k_eq_v`.** A Gemma-4-specific attention variant where K and V collapse to the same
   tensor; unimplemented in the batched path.
4. **Per-head V-norm before the cache write.** Applied in the sequential `RunTrunk`/`Attention()`
   path today; `PrefillCore`'s batched loop has no equivalent step.
5. **Sliding-window attention.** `PrefillCoreAttention` has no `windowSize` parameter at all —
   Gemma 4 is the only architecture that has ever needed it in the batched path, and it's always
   routed away before reaching this code today. (Note: `cohere2`'s SWA-without-per-layer-head-dims
   case already reuses this same routing check — `swaUnsupported` — so whatever sliding-window
   support gets added here should be written so cohere2 can eventually use it too, not Gemma-4-only.)

Also blocking, but architecturally separate (already tracked, don't conflate): `postNormUnsupported`
(post-attention/post-FFW norm, shared with OLMo2) and `unweightedNormUnsupported` (OLMo v1) route
through the *same* sequential fallback check but are unrelated bugs with their own fixes — do not
try to fix Gemma 4 by touching those flags.

## Plan

- [ ] **Re-read the full routing block** (`ForwardPass.cs` ~1240-1312) end to end before writing
      any code — the comments there already document several false starts (the superseded framings
      linked at `docs/reference/forwardpass-investigation-log.md
      #gemma-4-per-layer-head-dim-batched-prefill--superseded-framings`). Do not re-attempt any
      approach already marked superseded there without first understanding why it was abandoned.
- [ ] **Add per-layer head-dim awareness to `PrefillCoreAttention`'s KV indexing.** This is the
      actual crash-causing gap — the buffer walk that reads/writes past the 256-vs-512 boundary.
      Get this correct and bounds-safe first, with a unit test that specifically exercises a
      mixed-head-dim layer pair, before touching anything else.
- [ ] **Add per-layer KV-source indirection (`_layerKvSrc`)** so a layer can read another layer's
      KV cache instead of assuming every layer owns its own.
- [ ] **Add `attention_k_eq_v` support** to the batched path.
- [ ] **Add the per-head V-norm step** to the batched loop, matching `RunTrunk`'s sequential
      version exactly (same normalization, same placement relative to the cache write).
- [ ] **Add sliding-window masking** (`windowSize` parameter) to `PrefillCoreAttention`, written
      generally enough that `swaUnsupported`'s cohere2 case could adopt it later without a rewrite.
- [ ] **Remove `perLayerHdUnsupported` from the routing condition** only once all five items above
      are implemented and tested — a partial removal reopens the AccessViolationException risk.
- [ ] **Correctness gate before any performance claim**: golden/numerical-parity tests comparing
      batched-path output against the existing sequential path's output, layer-by-layer, on a real
      Gemma-4 checkpoint (both E4B and 12B, since they may exercise different head-dim/window
      configurations) — not just "it doesn't crash."
- [ ] **Performance pass** (per `CLAUDE.md` rule 7): measure real prefill throughput before/after on
      both E4B and 12B with `docs/benchmark-prompt.txt`, best-of-3, against the same llama.cpp
      reference numbers already recorded in `PerformanceLeague.md` (2026-09-10 backfill: 12B
      27.81 t/s, E4B 80.50 t/s). Only keep the change if it's measurably faster — write the
      before/after numbers down, don't estimate.
- [ ] **Update `PerformanceLeague.md`** with the new ratios once measured, and remove the
      `perLayerHdUnsupported`/Gemma-4-prefill-penalty framing from `cpu-performance-baseline.md`'s
      "reading it" section if it's no longer accurate.
- [ ] **DRY pass** (per `CLAUDE.md` rule 7): check whether the new per-layer-head-dim / sliding-
      window logic duplicates anything already in `RunTrunk`'s sequential `Attention()` — extract
      shared helpers if so, matching the existing `Primitives/*Kernels.cs` convention.

## Risks / open questions

- The AccessViolationException history means this needs careful, incremental, test-gated work —
  not a single large patch. Land the head-dim-indexing fix alone first, with its own crash-repro
  test turned into a regression test, before adding the other four features.
- Unclear whether all five gaps affect both E4B and 12B identically, or whether the two checkpoints
  exercise different subsets (e.g. different sliding-window layer patterns) — the archaeology step
  should confirm this against both real GGUFs before assuming one fix covers both.
- No CUDA/Vulkan equivalent of this gap has been checked — this plan is CPU (`PrefillCore`) only.
