# Plan: Fish Speech S2 Pro correctness-test reconciliation

**Entry in:** `docs/1-correctness/bugstofix.md`, item **08** (Fish Speech S2 Pro correctness-test
reconciliation). Also informed by the 2026-09-27 `docs/103-quickest-first-plan.md` Audio sweep. This
is first a test/oracle reconciliation task, not a presumed product regression in `d377049`.

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
| `ce6a914` → `d377049` | `ce6a914` is the direct parent of `d377049` (verify the relationship before running the A/B). The preceding broad performance commit contains numerical optimizations already present at `ce6a914`; do not attribute those changes to `d377049`. The Fish Speech-specific delta in `d377049` includes codec SIMD substitutions and the Fast-AR `computeLogits:false` fast path. |
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

### Explicit causal A/B: `ce6a914` → `d377049`

The `7a68185` comparison establishes that the reported Fast-AR Q4 mismatch predates `d377049`; it is
not the direct A/B for changes introduced by `d377049`. For causal attribution, run matching tests
from clean worktrees at `ce6a914` and `d377049`, after verifying the exact parent/commit relationship.
Use identical checkpoint files, deterministic inputs, runtime settings, and reference artifacts; log
both full commit hashes and all metric provenance.

Attribute only deltas actually introduced by `d377049` to this comparison. The `TensorPrimitives`
Fast-AR `Layer`/`LayerStep` numerical changes predate this commit and must not be listed as
`d377049` regressions. The relevant Fast-AR delta here is the new optional `computeLogits` path that
skips final RMSNorm/output projection at position 0. The codec SIMD replacements are also in the
commit delta and should be assessed only against the corrected full-path codec oracle from Phase 1.

Run the actual production cache sequence for both Q8_0 and Q4_K_M checkpoints at each revision:

```text
Reset()
ForwardStep(hidden, computeLogits: false)           // position 0: populate KV cache only
ForwardStep(semanticEmbedding)                       // logits for codebook 1
ForwardStep(codebook1Embedding)                      // logits for codebook 2
...
ForwardStep(codebook8Embedding)                      // logits for codebook 9
```

Use the actual codebook count from the checkpoint and compare logits at each prediction position
where both revisions produce them. Also compare the resulting deterministic code sequence when the
same sampling/selection procedure applies. The initial hidden-state call must explicitly use
`computeLogits: false`; calling `ForwardStep(hidden)` with the default `true` does not exercise the
new skip-logits optimization and is insufficient as acceptance for this A/B. Keep this production-
sequence test distinct from the original full-prefix `Forward` golden and from generic
`ForwardStep` self-consistency. Since `ce6a914` predates the `computeLogits` parameter, its equivalent
baseline call necessarily computes and discards position-0 logits; keep the cache reset, inputs, and
all subsequent prediction positions identical, and compare the logits produced after the semantic
and codebook embeddings.

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

The existing `ForwardStep_MatchesForward_ForSamePrefix` uses only `s2-pro-q8_0.gguf` and calls
`ForwardStep(hidden)` with the default `computeLogits: true`. That does not exercise the production
position-0 optimization. Add Q8_0 and Q4_K_M checks that both reproduce the production sequence above,
including the initial `computeLogits: false` call, then compare the predicted-position outputs with
the corresponding `Forward` result. This covers the actual S2 Pro checkpoint's effective weight path
and the optimization introduced by `d377049`. Establish characterization metrics before judging
code changes.

Keep the Q8_0 external-logit comparison as the higher-precision numerical control. Do not silently
loosen the Q4 cosine threshold while leaving its full-precision comparison semantics unchanged.

### 2.3 Reconcile the historical Q4 cosine values

The Fast-AR test/history reports a cosine around 0.44, while the explanatory comment in
`FishSpeechWeights.cs` reports Q4_K_M around 0.489. These may come from different revisions, inputs,
or comparison paths. Do not select one as the canonical result from memory. Re-run and record the
exact cosine alongside the commit, checkpoint file/hash and quantization, deterministic input, and
comparison path. After that reproduction, use the exact sourced value in current results; retain the
two approximate numbers only as historical context for why reconciliation is needed.

## Phase 3 — Inspect the `ce6a914` → `d377049` delta only if a valid test shows a regression

Proceed here only if the regenerated codec oracle fails, the Q8_0 higher-precision control still
fails after its checkpoint/conversion path is checked, or the Q4 quantization-aware comparison or
either Q8_0/Q4 production-sequence self-consistency test demonstrates a new degradation. For a
regression attributed to `d377049`, require the direct `ce6a914` vs `d377049` A/B to reproduce the
difference on the affected execution path; the `7a68185` comparison is historical classification,
not the commit's causal control.

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
| Fast-AR `computeLogits:false` at initial hidden position | Introduced in `d377049`; validate only with the exact production sequence and both Q8_0/Q4_K_M checkpoint inputs, comparing at `ce6a914` vs `d377049`. |
| Fast-AR `TensorPrimitives.Dot` / `MultiplyAdd` / `Add` in `Layer`, `MatVecDual`, `SiLuMul`, `SumOfSquares` in `LayerStep` | Present before `d377049`; do not attribute these changes to it. They may be examined only under a separate causal comparison if new evidence points to them. |

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
- **Cached Fast-AR:** test Q8_0 and Q4_K_M `ForwardStep` against `Forward` using the production
  position sequence, including `Reset()` and `ForwardStep(hidden, computeLogits: false)` before the
  semantic embedding. This must exercise the actual `d377049` skip-logits optimization; generic calls
  with the default `computeLogits: true` are insufficient. These are optimized-path self-consistency
  checks, not external parity.
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
- Do not claim coverage of `d377049`'s skip-logits optimization if the test calls
  `ForwardStep(hidden)` with `computeLogits` left at its default `true`.
- Do not attribute the pre-existing `TensorPrimitives` Fast-AR layer optimizations to `d377049`; use
  the explicit `ce6a914` → `d377049` A/B for causality.
- Do not change test thresholds without correcting the reference semantics and recording the basis.
- Do not weaken or delete a valid test merely because it fails; replace only invalid or mis-scoped
  expectations with a test that proves the intended claim.

## Success criteria

This correctness-test reconciliation is complete when:

1. Current-versus-`7a68185` outcomes clearly classify which reported failures predate `d377049`, and
   the direct `ce6a914` → `d377049` A/B is recorded for causal assessment.
2. The codec test uses a reference generated with the complete `quantizer.post_module` path and passes
   on identical codes with documented raw PCM metrics.
3. Q8_0 Fast-AR's higher-precision control and checkpoint conversion path are measured; any remaining
   failure is isolated to checkpoint/conversion, loading, or model math.
4. Q4_K_M Fast-AR has a same-effective-path reference if one exists; otherwise it has documented
   quantization-aware characterization and does not present a full-precision cosine threshold as
   universal correctness proof.
5. Cached `ForwardStep` passes self-consistency with both Q8_0 and Q4_K_M weights using the actual
   production sequence and initial `computeLogits: false` call; end-to-end listening is recorded
   independently.
6. `bugstofix.md` and `STATUS.md` no longer label `d377049` the leading suspect without new evidence;
   remaining test debt and product status are accurately described.

## Resolution — 2026-10-01 (oracles repaired; no engine defect found)

Downloaded the original `fishaudio/s2-pro` (11 GB: `codec.pth`, two BF16 safetensors shards; not gated,
licence "other") to `E:\_models\s2-pro-original`. Both references are now generated from the real full-precision
weights; the scripts (`scratch-llamacpp-ref/fish_speech_codec_golden.py`, `fish_speech_fastar_golden.py`) are
gitignored local oracles, as before, with the older versions kept beside them as `*_OLD*`.

**Phase 1 — codec.** The old golden was a numpy transcription over the dequantized Q4_K_M GGUF that applied
`quantizer.post_module` as a bare RMSNorm (skipping the eight-layer `WindowLimitedTransformer`). The new golden
runs the real fish-speech `DAC` (upstream `modded_dac.py`/`rvq.py`, byte-identical to the vendored copies, config
from `modded_dac_vq.yaml`) with `codec.pth` through `DAC.from_indices` in PyTorch (0 missing keys; the only
"unexpected" keys are the non-persistent `freqs_cis`/`causal_mask` buffers). Same deterministic codes. C# codec vs
this golden (`FishSpeechCodecTests.Decode_RealWeights_MatchesGoldenPcmOutput`): length 4096 matches, **cosine
0.9999969, max abs error 1.154e-3, RMS error 1.814e-4 (0.26% of golden RMS 7.08e-2)**; the test now asserts
cosine > 0.9999, maxAbs < 4e-3, rmsErr < 6e-4. The old 0.052 cosine was the stale oracle.

**Phase 2 — Fast-AR.** `fish_speech_fastar_golden.py` now reads the local safetensors shard (default; reproduces
the old golden to relative L2 1.6e-8) or, with `FASTAR_GGUF=<gguf> FASTAR_TAG=<tag>`, the same float math on a
GGUF's tensors dequantized to float32 (the exact effective path for a quantized checkpoint). Pure-Python results,
no C# involved: original BF16 vs Q8_0-dequantized cosine 0.99955 (argmax agrees); original vs **Q4_K_M-dequantized
cosine 0.489**, relative L2 0.888, argmax flips (324 vs 497), top-10 overlap 8/10. So the Q4_K_M weights are far
from the original through exact float math, independent of this engine. C# vs the matching dequantized-path
reference: **Q4_K_M cosine 0.99836, relL2 5.7%, top-1 agrees, top-10 10/10; Q8_0 cosine 0.99892, relL2 5.6%,
top-1 agrees, top-10 9/10.** New tests `Forward_Q4KMWeights_MatchesDequantizedPathOracle` and
`Forward_Q8_0Weights_MatchesDequantizedPathOracle` assert these (cosine > 0.995 plus top-1 agreement). The old
Q4-vs-original test is now `Forward_Q4KM_vs_FullPrecision_CharacterizesQuantizationLoss` (C# measured 0.4406;
asserts only 0.30-0.70, no correctness claim). `ForwardStep_MatchesForward_ForSamePrefix` and the Q8_0-vs-original
golden still pass.

**Phase 3 (`ce6a914` → `d377049` A/B):** not run. Its trigger was a valid test showing a regression; both
repaired oracles pass at the current revision, so there is no regression evidence to attribute.

Practical note: Q4_K_M Fast-AR logits are about 0.49 cosine from the original model's (top-1 differs on this
input). That is a property of this checkpoint's quantization (Q4_K on `fast_embeddings`/`wo`/`w1`/`w3`), not of the
engine; if Fast-AR quality matters, prefer the Q8_0 checkpoint.

### Follow-up: which Fast-AR tensors carry the Q4_K_M loss (2026-10-01)

Ablation with the same float reference math (`scratch-llamacpp-ref/fish_fastar_q4_ablation.py`, gitignored; swaps
tensor groups between the original BF16 weights and the Q4_K_M/Q8_0 GGUFs dequantized), one deterministic Fast-AR
position, KL(original || variant) at T=1:

| Variant | KL (nats) | cosine | argmax | top-10 overlap |
| --- | ---: | ---: | --- | ---: |
| Pure Q4_K_M | 0.978 | 0.489 | 324 -> 497 | 8 |
| Original, only `wo` taken from Q4_K_M | 0.471 | 0.937 | 324 -> 497 | — |
| Original, only `w3` from Q4_K_M | 0.193 | 0.969 | agrees | — |
| Original, only one of `emb`/`wqkv`/`w1`/`w2`/`out`/norms from Q4_K_M | 0.0007-0.042 | >= 0.9965 | agrees | — |
| Q4_K_M with `wo` from Q8_0 | 0.385 | 0.832 | agrees | 8 |
| **Q4_K_M with `wo` + `w3` from Q8_0** | **0.079** | 0.966 | agrees | 9 |
| Q4_K_M with `wo`+`w3`+`wqkv` from Q8_0 | 0.029 | 0.979 | agrees | 9 |
| Q4_K_M with `wo`+`w3`+`wqkv`+`emb`+`w1` from Q8_0 | 0.021 | 0.9992 | agrees | 10 |
| Pure Q8_0 | 0.007 | 0.9995 | agrees | 10 |

The loss is concentrated in the Fast-AR attention output projection `wo` (Q4_K) and the FFN up-projection `w3`
(Q4_K); errors compound across layers (restoring only layer 0 cuts KL to 0.42, only layer 3 barely helps).
**Recipe:** keep everything Q4_K_M but store `fast_layers.*.attention.wo.weight` and
`fast_layers.*.feed_forward.w3.weight` as Q8_0 (about +71 MB over the 3.57 GB file; adding `wqkv` Q6_K -> Q8_0 is
+15 MB more) to recover about 92% of the KL gap (0.98 -> 0.08). Not built: `llama-quantize` rejects the
`fish-speech` architecture ("unknown model architecture"), and this repo has no GGUF quantizer, so producing the file
needs a small tensor-transplant tool (copy the Q4_K_M GGUF, substituting those tensors from `s2-pro-q8_0.gguf`; the
two files share tensor names). Caveats: one input at one position; the slow-AR stage's Q4_K_M sensitivity was not
measured; the C# loader already renormalizes Fast-AR matrices to Q8_0 at load, so a transplanted file would load
unchanged.

**Built and validated (2026-10-01).** New CLI command `stingray gguf-transplant --base <gguf> --donor <gguf> --tensors <regex> -o <out>`
(`src/OpenTail.Stingray.Cli/GgufTransplantCommand.cs`) copies the base file's header and KV section verbatim, re-emits the
tensor infos with the donor's dtype for matching tensors, and verifies the output byte-for-byte against its sources. Run with
`--tensors '^fast_layers\.\d+\.(attention\.wo|feed_forward\.w3)\.weight$'`, base `s2-pro-q4_k_m.gguf`, donor `s2-pro-q8_0.gguf`
it produced `E:\_models\s2-pro-mixed\s2-pro-q4_k_m-fastar-wo-w3-q8_0.gguf`: 8 tensors Q4_K -> Q8_0, **+67.5 MiB (3.57 -> 3.64 GB)**,
813 tensors verified. C# Fast-AR on it (`Forward_Q4KMWithQ8WoW3_MatchesDequantizedPathOracle_AndRecoversOriginal`, opt-in via
`STINGRAY_S2_MIXED_GGUF`): vs the float reference on its own dequantized tensors cosine 0.99988 (top-1 agrees, top-10 10/10);
vs the ORIGINAL weights **cosine 0.9670, top-1 agrees, top-10 9/10** (pure Q4_K_M: 0.4406; pure Q8_0 vs original 0.9995 in float
math). End-to-end `stingray tts -e fish` on the mixed file generated 1.95 s of audio in 19.6 s (not listened to). Still unmeasured:
slow-AR Q4_K_M sensitivity, and any listening comparison. The command has no unit test of its own; its built-in byte-for-byte verify
is the check.
