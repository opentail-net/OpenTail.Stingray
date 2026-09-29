# Plan: Fish Speech S2 Pro correctness-test reconciliation

**Context:** `docs/1-correctness/bugstofix.md`'s Fish Speech S2 Pro entry and the 2026-09-27
`docs/103-quickest-first-plan.md` Audio sweep. This is first a test/oracle reconciliation task, not a
presumed product regression in `d377049`.

## Goal and scope

Determine which reported Fish Speech S2 Pro failures represent invalid or mis-scoped acceptance
tests, establish permanent regressions that match the implementation and quantization being tested,
and investigate `d377049` only if a valid control still demonstrates a regression.

The two reported failures are independent:

1. `FishSpeechCodecTests.Decode_RealWeights_MatchesGoldenPcmOutput` compares the current codec,
   including the `quantizer.post_module` transformer, against an obsolete golden that predates that
   transformer.
2. `FishSpeechFastArTests.Forward_RealWeights_MatchesGoldenOracle` compares a Q4_K_M-derived path
   against logits from the original full-precision safetensors reference, so its cosine is not by
   itself proof of a math defect.

Do not combine these into one `d377049` regression hypothesis.

## What history already establishes

| Evidence | Established fact |
| --- | --- |
| `7a68185` | It is an ancestor of `d377049`; its commit message says the Fast-AR Q4 golden cosine of about 0.44 was already present on clean HEAD before the performance commit. |
| `bd2a612` | Adds the real eight-layer `quantizer.post_module` transformer to codec decoding. Its message says the old codec golden skipped that transformer and became stale. Using the same reference-generated codes, the corrected C# codec reached 0.9999999 cosine against the reference. |
| `a7e720` | Records the remaining suite failures as the pre-existing Q4_K_M Fast-AR precision limitation and stale codec oracle, and reports successful real end-to-end listening verification. |
| Current Fast-AR tests | Include the Q4 external golden, a Q8_0 external golden, and `ForwardStep_MatchesForward_ForSamePrefix`. These controls test different claims; do not treat them as interchangeable. |
| Current `FishSpeechWeights` | Dequantizes loaded weights and normalizes the Fast-AR matrices to Q8_0 at load time. For a Q4_K_M checkpoint, the tested path is therefore approximately original weights → Q4_K_M → dequantized values → Q8_0 → inference. |

This history weighs against attributing either original mismatch to `d377049`; it does not prove every
optimization in that commit is correct. Preserve the cited commits and test outputs as evidence, not
as substitutes for current, valid regression tests.

## Phase 0 — Classify current failures against the pre-`d377049` revision

Run the relevant tests at the current revision and in a clean worktree at `7a68185`, using the same
checkpoint files, reference fixtures, runtime settings, and test invocation where possible. Record
commit hashes, test outcomes, checkpoint quantization, and numerical metrics. Treat this as a
classification experiment, not a test of whether `d377049` caused the failures.

| Test/control | Expected interpretation |
| --- | --- |
| Codec golden at `7a68185` | A similar ~0.05 cosine is consistent with the stale golden already predating `d377049`; it is not evidence that `d377049` introduced a codec regression. |
| Fast-AR Q4_K_M golden at `7a68185` | A similar ~0.44 cosine confirms the failure predates `d377049`, as the commit history already states. |
| Fast-AR Q8_0 external golden at both revisions | This is the numerical control against the full-precision oracle. Record its result; it helps separate broad Fast-AR math issues from Q4 quantization sensitivity. |
| Current `ForwardStep_MatchesForward_ForSamePrefix` | Checks cached/optimized `ForwardStep` against the non-cached `Forward` path; it is not an external numerical oracle. |

If any test cannot run at the historical revision because its fixture or method was introduced later,
record that limitation and run the closest equivalent without presenting it as the same test. Do not
infer causation merely because the current result differs from a historical one; compare the exact
paths and inputs.

## Phase 1 — Repair the codec oracle before changing codec math

The existing `FishSpeechCodecTests` golden is not a trustworthy correctness oracle for the current
codec: it was generated before `quantizer.post_module` was understood and does not exercise the full
current decode path.

1. Confirm the old golden-generation script omits the full `quantizer.post_module` transformer.
2. Use the same deterministic semantic/residual code sequence for the reference and C# codec.
3. Regenerate the permanent reference PCM using the complete reference decode path, including the
eight-layer `quantizer.post_module` transformer before upsampling/DAC decoding.
4. Preserve the exact code sequence and output PCM as deterministic test fixtures. Update the existing
golden generator/test rather than adding a second independent oracle implementation.
5. Run the corrected C# codec against the regenerated PCM and report length, cosine, max absolute
error, RMS error, and any other established PCM metric.

The already-recorded same-code experiment achieved approximately 0.9999999 cosine after the
`post_module` fix. Reproduce that path as the first validation. If it passes, retire or regenerate the
stale fixture; do not use its ~0.05 cosine as evidence that current codec math is broken. Investigate
codec implementation changes only if the corrected reference comparison fails.

## Phase 2 — Validate Fast-AR at the precision actually under test

### 2.1 Q8_0 is the numerical baseline

Run `Forward_Q8_0Weights_MatchesGoldenOracle` against the external Fast-AR logits generated from the
real safetensors reference. Report raw logits metrics in addition to cosine, including max/mean
absolute error, RMS, relative L2, cosine, argmax agreement, and top-k agreement. This is the existing
near-lossless numerical control for the implementation's Q8_0 Fast-AR matrices.

If Q8_0 fails, investigate the ordinary `Forward`/`Layer` math with the external oracle before
blaming Q4 quantization or `d377049`.

### 2.2 Q4_K_M needs quantization-aware acceptance

The existing Q4 test compares quantized-weight logits directly with the original full-precision
safetensors logits and asserts cosine greater than 0.99. That is not a valid universal acceptance
criterion for a Q4_K_M-derived path that is dequantized and requantized to Q8_0 internally. The
observed ~0.44 cosine may reflect compounded quantization rather than a mathematical implementation
error.

Replace the Q4 test's role with one of these, preferring the first when a suitable reference is
available:

1. Compare with a reference implementation using the same Q4_K_M checkpoint and the same conversion
   path, on identical inputs.
2. Otherwise characterize the Q4 path using fixed deterministic inputs and quantization-aware
   measures: top-1 argmax agreement, top-k overlap/rank agreement, logit correlation, and the raw
   error metrics. Establish acceptance from an empirical baseline before judging a code change; do
   not choose a threshold after seeing the result.

Keep the Q8_0 full-precision-logit comparison as the high-precision numerical test. Do not silently
loosen the Q4 cosine threshold while leaving its full-precision comparison semantics unchanged.

## Phase 3 — Inspect `d377049` only if a valid test shows a regression

Proceed here only if the regenerated codec oracle fails, the Q8_0 Fast-AR external golden fails, or a
properly designed same-quant/quantization-aware test demonstrates a new degradation at the current
revision relative to `7a68185`.

Trace only the failing execution path. The failing external
`FishSpeechFastAr.Forward_RealWeights_MatchesGoldenOracle` test exercises `Forward`/`Layer`, not the
KV-cached `ForwardStep` path. Changes to `MatVecDual`, `SiLuMul`, or `SumOfSquares` that are used only
by `ForwardStep` are not explanations for that external `Forward` golden failure; evaluate them with
the self-consistency test instead.

If evidence points to `d377049`, isolate individual changes and test them independently. Do not
revert or bisect the whole commit before narrowing the responsible path.

| Area/change in `d377049` | Investigation scope if relevant |
| --- | --- |
| Codec `TensorPrimitives.Sin` replacing scalar `MathF.Sin` | High-priority numerical comparison on corrected same-code codec reference. |
| Codec `TensorPrimitives.MultiplyAdd` in transposed convolution | High-priority comparison on corrected same-code codec reference. |
| Codec in-place `TensorPrimitives.Add` | Check only if the corrected codec comparison points to residual accumulation. |
| FullConv1d allocation/parallel threshold; code-loop ordering; ArrayPool changes | Lower numerical risk; inspect only when evidence traces to those paths. |
| Fast-AR `TensorPrimitives.Dot` / `MultiplyAdd` / `Add` in `Layer` | Relevant to the ordinary external `Forward` oracle; isolate only if Q8_0 or a valid Q4 reference fails. |
| Fast-AR `MatVecDual`, `SiLuMul`, `SumOfSquares` in `LayerStep` | Relevant to cached `ForwardStep`; validate with `ForwardStep_MatchesForward_ForSamePrefix`, not the ordinary `Forward` golden. |

Use one numerical change per experiment, record raw metrics, and preserve the unmodified comparison
control.

## Phase 4 — Permanent regression coverage and close-out

The final coverage should prove each claim with an appropriate test:

- **Codec:** regenerated permanent golden from the complete post-module reference path, using the
  exact deterministic code sequence; the C# decoder must pass strict raw PCM parity.
- **Fast-AR Q8_0:** external numerical golden against the full-precision reference, with documented
  raw metrics.
- **Fast-AR Q4_K_M:** same-quant reference if available; otherwise a documented
  quantization-aware test with deterministic inputs and pre-established criteria.
- **Cached Fast-AR:** retain `ForwardStep_MatchesForward_ForSamePrefix` as optimized-path
  self-consistency coverage; do not call it external parity.
- **End-to-end:** retain a real S2 Pro listening/reference check after any implementation fix; the
  earlier listening verification is positive evidence but does not replace numeric stage tests.

Update this plan, `docs/1-correctness/bugstofix.md`, and the Fish Speech row in `docs/STATUS.md` with
the actual corrected test results and any genuine remaining defects. Preserve the 2026-09-27
`docs/103-quickest-first-plan.md` entry as a dated record of what the sweep observed and suspected at
the time; do not rewrite it to imply that the historical suspicion was already disproved then.

## What not to do

- Do not start by bisecting `d377049`; the two reported failures were documented before it or against
  an oracle made stale by an earlier fix.
- Do not use the old codec golden as evidence against the implementation after `post_module` was
  added.
- Do not compare Q4_K_M logits with full-precision logits using an unexplained 0.99 cosine threshold
  and call that a correctness proof.
- Do not treat the Q8_0 external golden, Q4 quantization-aware validation, and ForwardStep
  self-consistency as interchangeable tests.
- Do not attribute a failure in `Forward` to an optimization used only by `ForwardStep`.
- Do not change test thresholds without correcting the reference semantics and recording the basis.
- Do not weaken or delete a valid test merely because it fails; replace only invalid or mis-scoped
  expectations with a test that proves the intended claim.

## Success criteria

This correctness-test reconciliation is complete when:

1. Current-versus-`7a68185` outcomes clearly classify which reported failures predate `d377049`.
2. The codec test uses a reference generated with the complete `quantizer.post_module` path and passes
   on identical codes with documented raw PCM metrics.
3. Q8_0 Fast-AR passes its external numerical oracle or any genuine failure is isolated and fixed.
4. Q4_K_M Fast-AR has a valid same-quant reference or a documented quantization-aware acceptance
   test; it no longer presents a full-precision cosine threshold as universal correctness proof.
5. Cached `ForwardStep` retains a separate self-consistency test, and end-to-end listening is
   recorded independently.
6. `bugstofix.md` and `STATUS.md` no longer label `d377049` the leading suspect without new evidence;
   remaining test debt and product status are accurately described.
