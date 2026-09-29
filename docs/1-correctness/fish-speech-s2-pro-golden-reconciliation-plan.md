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
| Codec golden at `7a68185` | Record the exact result and verify the codec path includes `post_module`; a failure against the same obsolete oracle is not evidence that `d377049` introduced a regression. |
| Fast-AR Q4_K_M golden at `7a68185` | Reproduce the exact metric with its input/checkpoint provenance. The commit history establishes that the mismatch predates `d377049`; do not assume the rounded historical value is the current reproduction. |
| Fast-AR Q8_0 external golden at both revisions | This is the higher-precision control against the full-precision oracle. Record its exact result and conversion path; it helps separate broad Fast-AR issues from Q4 quantization sensitivity. |
| Cached-path self-consistency at current revision | Run `Forward` ↔ `ForwardStep` with both Q8_0 and Q4_K_M weights. This checks the cached path at each quantization; it is not an external numerical oracle. At `7a68185`, run the Q4_K_M variant only if the same test exists there; otherwise record it as current-only coverage. |

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

### 2.1 Q8_0 is the higher-precision numerical control

Run `Forward_Q8_0Weights_MatchesGoldenOracle` against the external Fast-AR logits generated from the
real safetensors reference. Report raw logits metrics in addition to cosine, including max/mean
absolute error, RMS, relative L2, cosine, argmax agreement, and top-k agreement. This is a useful
higher-precision control, not an exact full-precision oracle: `FishSpeechWeights` loads GGUF tensors
as F32 and normalizes Fast-AR matrices to Q8_0, so even a Q8_0 checkpoint follows a
GGUF-Q8_0 → F32 → Q8_0 → inference path.

If this control fails, investigate the source checkpoint, dequantization/requantization path, loader,
and ordinary `Forward`/`Layer` math before attributing the mismatch to Q4_K_M or `d377049`.

### 2.2 Q4_K_M needs quantization-aware acceptance

The existing Q4 test compares quantized-weight logits directly with the original full-precision
safetensors logits and asserts cosine greater than 0.99. That is not a valid universal acceptance
criterion for a Q4_K_M-derived path that is dequantized and requantized to Q8_0 internally. A low
cosine may reflect compounded quantization rather than a mathematical implementation error.

Prefer a reference implementation using the exact effective Q4_K → F32 → Q8_0 path, but first
determine whether such a reference actually exists. Do not invent or build a questionable reference
merely to satisfy the test plan. If no exact-path reference is available, treat Q4 as a
characterization test and use deterministic inputs to record raw logit error/correlation, top-1
agreement, top-k overlap/rank agreement, and deterministic self-consistency (`Forward` vs
`ForwardStep`). Do not impose an absolute cosine threshold against the full-precision golden.

Add Q4_K_M coverage to the cached-path self-consistency check. Keep both cases:

- Q8_0 `Forward` ↔ `ForwardStep`;
- Q4_K_M `Forward` ↔ `ForwardStep`.

The existing `ForwardStep_MatchesForward_ForSamePrefix` uses only `s2-pro-q8_0.gguf`; add an
equivalent deterministic case using `s2-pro-q4_k_m.gguf`, so the production checkpoint's effective
weight path is also covered. Establish characterization metrics before judging code changes.

Keep the Q8_0 external-logit comparison as the higher-precision numerical control. Do not silently
loosen the Q4 cosine threshold while leaving its full-precision comparison semantics unchanged.

### 2.3 Reconcile the historical Q4 cosine values

The Fast-AR test/history reports a cosine around 0.44, while the explanatory comment in
`FishSpeechWeights.cs` reports Q4_K_M around 0.489. These may come from different revisions, inputs,
or comparison paths. Do not select one as the canonical result from memory. Re-run and record the
exact cosine alongside the commit, checkpoint file/hash and quantization, deterministic input, and
comparison path. After that reproduction, use the exact sourced value in current results; retain the
two approximate numbers only as historical context for why reconciliation is needed.

## Phase 3 — Inspect `d377049` only if a valid test shows a regression

Proceed here only if the regenerated codec oracle fails, the Q8_0 higher-precision control still
fails after its checkpoint/conversion path is checked, or the Q4 quantization-aware comparison or
either Q8_0/Q4 `Forward`-versus-`ForwardStep` self-consistency test demonstrates a new degradation at
the current revision relative to `7a68185`.

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
| Fast-AR `TensorPrimitives.Dot` / `MultiplyAdd` / `Add` in `Layer` | Relevant to the ordinary external `Forward` oracle; isolate only if a valid higher-precision or quantization-aware comparison fails. |
| Fast-AR `MatVecDual`, `SiLuMul`, `SumOfSquares` in `LayerStep` | Relevant to cached `ForwardStep`; validate against `Forward` with both Q8_0 and Q4_K_M weights, not the ordinary external `Forward` golden. |

Use one numerical change per experiment, record raw metrics, and preserve the unmodified comparison
control.

## Phase 4 — Permanent regression coverage and close-out

The final coverage should prove each claim with an appropriate test:

- **Codec:** regenerated permanent golden from the complete post-module reference path, using the
  exact deterministic code sequence; the C# decoder must pass strict raw PCM parity.
- **Fast-AR Q8_0:** higher-precision external numerical control against the full-precision
  reference, with documented raw metrics and the GGUF-to-Q8_0 conversion path recorded.
- **Fast-AR Q4_K_M:** same-effective-path reference if available; otherwise a documented
  quantization-aware characterization with deterministic inputs; do not use an absolute cosine
  threshold against the full-precision oracle when no exact effective-path reference exists.
- **Cached Fast-AR:** retain Q8_0 `ForwardStep_MatchesForward_ForSamePrefix` and add the corresponding
  Q4_K_M case as optimized-path self-consistency coverage; do not call either external parity.
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
- Do not describe the Q8_0 normalization path as lossless or use it as an exact oracle without
  accounting for GGUF dequantization and requantization.
- Do not compare Q4_K_M logits with full-precision logits using an unexplained 0.99 cosine threshold
  and call that a correctness proof.
- Do not manufacture an exact-effective-Q4 reference if none exists; use characterization and
  self-consistency instead.
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
3. Q8_0 Fast-AR's higher-precision control and checkpoint conversion path are measured; any remaining
   failure is isolated to checkpoint/conversion, loading, or model math.
4. Q4_K_M Fast-AR has a same-effective-path reference if one exists; otherwise it has documented
   quantization-aware characterization and does not present a full-precision cosine threshold as
   universal correctness proof.
5. Cached `ForwardStep` passes self-consistency with both Q8_0 and Q4_K_M weights, and end-to-end
   listening is recorded independently.
6. `bugstofix.md` and `STATUS.md` no longer label `d377049` the leading suspect without new evidence;
   remaining test debt and product status are accurately described.
