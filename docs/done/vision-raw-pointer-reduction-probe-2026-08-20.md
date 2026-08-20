# Raw-pointer-surface reduction probe — InternVL — 2026-08-20

Third pass over `OpenTail.Stingray.Vision`, same day as the dtype-safety migration and the FFN
scratch-buffer fix. User asked to consider a redesign that reduces raw-pointer surface further,
conditional on speed staying acceptable — and specifically to probe on one already-well-serviced
model first, measure it, and only then decide whether to roll out further.

## What "raw-pointer surface" means here, and what's in/out of scope

Two different kinds of `unsafe` pointer live in every encoder:

1. **Matmul-bound weight tensors** (attention Q/K/V/out, FFN up/gate/down, projector) — accessed via
   `VisionTensorRef`'s `byte* Data`, zero-copy into the GGUF's memory-mapped file. These can be
   GB-scale. Materializing them into managed arrays would mean copying gigabytes into the managed
   heap per model load and giving up the zero-copy mmap design the whole engine relies on (matches
   how `OpenTail.Stingray.Cpu`/`OpenTail.Stingray.Engine` read the main LLM's weights). **Out of
   scope** — not touched, not a credible target for this kind of redesign.
2. **Norm/bias tensors** (LayerNorm/RmsNorm weight+bias, MatVec bias vectors) — O(dim) elements each,
   a few KB at most even for a large model, currently stored as raw `float*` fields resolved once at
   construction via `VisionOps.GetTensorPtr<float>` and held for the lifetime of the encoder. **This
   is the actual redesign target.**

The risk with (2) specifically: a `float*` field has no compiler-enforced relationship to the
lifetime or size of what it points to. It's currently safe because nothing in this codebase disposes
a `GgufModel` out from under a live encoder and the dtype guard (added in the earlier migration)
catches a dtype mismatch — but it's still a pointer with no bounds tracked by the runtime, into
memory whose validity isn't statically visible at the point of use. That's the exact shape of hazard
the whole migration effort exists to reduce.

## The redesign

Added `VisionOps.GetTensorArray(gguf, ...)` — resolves + dequantizes a norm/bias tensor to a managed
`float[]?` (null, not empty, when missing — see below for why that distinction matters), instead of
`GetTensorPtr<float>`'s raw `T*`. Converted `InternVlVisionEncoder` (the template encoder every other
migrated encoder was copied from) to use it: every norm-weight and bias field is now `float[]?`
instead of `float*`, resolved once at construction exactly like before, but as a GC-owned, bounds-
checked array with a lifetime tied to the encoder instance rather than a bare pointer into the GGUF's
mapping. At each use site (inside the per-layer loop, and around the pre/post-norm and projector
calls), the arrays are pinned locally with a single `fixed` statement scoped to that call --
mirroring the exact multi-variable `fixed` idiom `Gemma4VVisionEncoder` already uses for its own
per-layer scratch buffers, not a new pattern.

**A real correctness subtlety, caught before it shipped**: the natural first draft made
`GetTensorArray` return `[]` (empty array) for a missing tensor, matching `DequantizeToFloat32`'s
existing "empty means absent" convention, and changed the encoder's `!= null` checks to
`.Length > 0`. This is wrong and would have been a new, subtle bug: C#'s `fixed` statement is only
*guaranteed* to yield a null pointer when the array reference itself is null — pinning a genuinely
zero-length (non-null) array is not guaranteed to yield null (the address is unspecified/
non-dereferenceable, not necessarily zero). `VisionOps.MatVecAny`/`LayerNorm`/`RmsNorm` all branch on
`bias != null`/`weights != null` at the pointer level, so a non-null-but-meaningless pointer for an
absent tensor would have made those branches read from unspecified memory instead of skipping
correctly. Fixed by making `GetTensorArray` return `null` (not `[]`) for a missing tensor and keeping
the encoder's checks as `!= null`, so `fixed` on the null case is unambiguous. Caught during design,
before ever running it — not found by the benchmark or by luck.

## Probe measurement (the actual ask)

Built a throwaway benchmark (`tests/OpenTail.Stingray.Tests.Vision/InternVlRawPointerProbeBenchmark.cs`)
that loads the real local `InternVL3-2B.Q4_K_M.gguf` + `mmproj-internvl3-2b-q8_0.gguf`, preprocesses
the same test image used throughout this session's verification, and times 8 repeated `Forward()`
calls after 2 warmup iterations (CPU backend, single process, best-of-N / median reported per this
repo's established benchmark methodology). Used `git stash` to get a clean A/B: built and ran the
identical test against the original raw-`float*` code, then against the redesigned managed-array
code, same machine, same model, same image, back to back.

| | Before (raw `float*` fields) | After (`float[]?` + per-call `fixed`) |
|---|---|---|
| Median (of 8) | 9836.87ms | 9627.64ms |
| Best (of 8) | 9359.42ms | 9496.45ms |
| Raw samples | 9359, 9603, 9654, 9787, **9837**, 9926, 10122, 10185 | 9496, 9525, 9569, 9589, **9628**, 9715, 9801, 10941 |

Both sample sets show ~5-8% run-to-run spread on their own (this machine's InternVL3-2B Forward() is
inherently ~9.3-10.9s on CPU regardless of this change — a 2B-parameter ViT doing full-precision
matvecs, unrelated to what this probe touches). The initial before/after delta (median actually
*slightly* faster after, best *slightly* slower after) sat well inside that noise band on its own.

**Follow-up (same day)**: re-ran the "after" state 5 more times (48 samples total pooled with the
original run) to firm this up rather than rely on a single 8-sample run either direction:

| | Before (1 run, n=8) | After (6 runs pooled, n=48) |
|---|---|---|
| Min | 9359.42ms | 9299.88ms |
| Median | 9812.15ms | 9688.99ms |
| Mean | 9809.18ms | 9882.74ms |
| p25-p75 | — | 9569-10037ms |

With 6x more "after" samples, min and median both come out very slightly *faster* than the single
"before" run; mean is very slightly higher, pulled up by a handful of outlier-heavy tail runs (one
hit 12168ms, another 11177ms — a single run in particular ran hot across all 8 of its own samples).
That's consistent with ordinary system noise (thermal/OS-scheduling variance across ~9 minutes of
back-to-back runs) rather than a structural regression: a real slowdown from `fixed`-pinning would
shift every sample up together, not just drag the mean via a few high outliers while min/median hold
or improve. **Verdict unchanged, now on firmer footing: perf-neutral.** Pinning a handful of small
(few-KB) arrays once per layer via `fixed` costs nothing measurable next to the matmul-dominated cost
of a real forward pass.

Correctness re-verified after every build (same bar as the rest of this session): identical
`256 soft tokens (1536-dim)` output against the real Q8_0 mmproj, both before and after.

## Decision

Speed is fine — proceeding to redesign is justified per the "if speed is OK" condition. **Not rolled
out to the other 21 encoders in this pass** — this was explicitly scoped as "probe one model, decide
on the rest after," and the decision is: yes, worth doing, but as its own follow-up pass (mechanical,
same pattern, ~20 files, each needs its own build+correctness verification the way every prior pass
in this session did file-by-file). Flagging as the next concrete task rather than rushing it into
this same turn.

## Files touched

- `src/OpenTail.Stingray.Vision/VisionOps.cs` — added `GetTensorArray`.
- `src/OpenTail.Stingray.Vision/InternVlVisionEncoder.cs` — probe target, fully converted.
- `tests/OpenTail.Stingray.Tests.Vision/InternVlRawPointerProbeBenchmark.cs` — new throwaway timing
  probe; kept in the tree (skips cleanly if local model fixtures are absent, same convention as
  other real-model tests) since it's cheap insurance for re-measuring if the rollout proceeds.
