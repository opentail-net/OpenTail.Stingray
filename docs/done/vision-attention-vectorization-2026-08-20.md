# Vision attention vectorization — VisionOps.Attention / AttentionGqa

**Context:** part of the 2026-08-20 project-by-project performance/quality review
(`docs/perf-loop-project-review-progress.md`). Found while comparing `PixtralVisionEncoder`
(delegates its transformer block to `VisionOps.Attention`) against `Gemma3VisionEncoder` (hand-rolls
the same block inline) as a DRY inconsistency.

## The near-miss

The obvious "fix" for that inconsistency is to converge `Gemma3VisionEncoder` onto the shared
`VisionOps.Attention` helper. That would have been a **regression, not a cleanup**.
`Gemma3VisionEncoder`'s own doc comment explains why it never used `VisionOps.Attention` in the
first place: its inner score/weighted-sum loops use `TensorPrimitives.Dot`/`Multiply`/`Add`
specifically because naive scalar loops (i.e. what `VisionOps.Attention` had) were measured as "far
too slow" at Gemma3's 4096-patch grid — that's an explicit historical finding recorded in the class,
not a stylistic choice. Reading the target of a refactor before doing it caught this.

## The actual fix

Rather than downgrade Gemma3 to the slow shared path, upgraded the shared path to Gemma3's already-
proven-faster technique instead — same `TensorPrimitives.Dot`/`Multiply`/`Add` approach, same
`Parallel.For`-per-head structure, same numerically-stable softmax. Gemma3 itself was not touched
(zero regression risk to it, since it never called this method to begin with).

This benefits every encoder that already calls `VisionOps.Attention`/`AttentionGqa` — 14 confirmed
by grep: `CogVlm`, `DeepSeekOcr`, `DotsOcr`, `Exaone4`, `Granite4`, `HunyuanVl`, `InternVl`, `Llava`,
`MimoVl`, `Nemotron`, `PaddleOcr`, `Pixtral`, `Step3Vl`, `YoutuVl` — for free, in one shared file,
instead of each needing its own hand-rolled fast path.

## Measurement

Added `Benchmark_Attention_ScalarVsVectorized` and `Benchmark_AttentionGqa_ScalarVsVectorized` to
`tests/OpenTail.Stingray.Tests.Vision/VisionOpsBenchmarkTests.cs`, following that file's existing
pattern (keep the pre-change scalar implementation verbatim as a private comparison baseline,
benchmark old vs new, assert a real speedup — same shape as its existing `Benchmark_MatVecF16_
ScalarVsSimd`). Both are permanent regression benchmarks now, not one-off measurements.

At 1024 tokens / 16 heads / 64 head-dim (`Attention`) and 1024 tokens / 16 query heads / 8 kv heads /
64 head-dim (`AttentionGqa`, 2:1 GQA ratio) — representative of a real ViT-L-scale grid:

| | Scalar (old) | Vectorized (new) | Speedup |
|---|---:|---:|---:|
| `Attention` | measured per-run, see test | measured per-run, see test | **>1.2x** (asserted) |
| `AttentionGqa` | measured per-run, see test | measured per-run, see test | **>1.2x** (asserted) |

Both benchmarks also assert numerical agreement between old and new (`maxDiff < 1e-3`) — confirmed
passing, the two implementations compute the same result up to float-reassociation noise, not a
different algorithm.

## Correctness verification

- `tests/OpenTail.Stingray.Tests.Vision/VisionOpsTests.cs` (pre-existing, hand-computed reference
  values for `Attention` and `AttentionGqa`'s sink-attenuation behavior): 6/6 pass, unmodified.
- The two new benchmark tests (numerical-agreement + speedup assertions): pass.
- Full `Tests.Vision` suite (all 22 architectures) was attempted as an additional check but hung
  after 2 tests over 30 minutes and was killed. Traced: this repo's only three Forward()-level
  encoder tests with local model fixtures (`Gemma3VisionEncoderTests`, `Gemma4VVisionEncoderTests`,
  `Llama4VisionEncoderTests`) call encoders that **do not use** `VisionOps.Attention`/`AttentionGqa`
  at all, so whatever the suite hung on, it wasn't exercising this change — and none of the 14 real
  callers of the changed methods have a local Forward()-level fixture to test against anyway. The
  math-level verification above is the correctness evidence that actually covers this change.

## Files changed

- `src/OpenTail.Stingray.Vision/VisionOps.cs` — `Attention`, `AttentionGqa` vectorized.
- `tests/OpenTail.Stingray.Tests.Vision/VisionOpsBenchmarkTests.cs` — two new permanent benchmarks.
