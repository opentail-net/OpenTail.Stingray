# Investigating a numerical discrepancy

How to decide whether a numerical difference between two execution paths is a bug, a precision
problem, or harmless FP noise, without defaulting to "keep the slow path". Distilled from
[the 2026-10-02 prefill numerics investigation](../1-correctness/2026-10-02-prefill-numerics-investigation.md),
which used every step below. Reusable harness code:
[patches/2026-10-02-prefill-numerics](../4-performance/patches/2026-10-02-prefill-numerics/README.md).

## Principles

- **Reproducible is not accurate.** A path that is bit-identical to decode is a deterministic
  control, not ground truth. On 2026-10-02 the "exact" per-token path turned out to be the
  noisiest component measured: its decode matvec quantizes activations to int8, about 1e-2 per
  projection, against about 1e-6 for the batched F32 GEMM. Don't call a path "exact"; name it by
  what it does (StrictSequential, Optimized, ...).
- **Reference per operation, not per model.** The numerical reference for one GEMM/matvec/
  recurrence is FP64 accumulation over the dequantized stored weights and the actual FP32
  inputs. No all-FP64 model exists to call "truth"; say "higher-precision reference".
- **Deep models amplify.** Differences of 1e-7 inside a layer reach 1e-2..1e-1 in the hidden
  state by the last layer and move logits by 0.5-3 within tens of tokens. A logit difference of
  that size, alone, says nothing about which path is wrong.
- **One receipt is not a policy.** A single greedy-parity flip shows sensitivity, not harm.
  Check the margin at the flipped token before acting on it.

## Steps

1. **Read the history first.** What test or receipt motivated the current behaviour? Is its
   checkpoint present? Does it already disagree with the reference elsewhere (e.g. at token 1)?
   Do code comments still describe what the code does?
2. **Check the external reference's actual algorithm.** Read the vendored llama.cpp source for
   the operation (`tools/llama.cpp/VERSION`), not documentation or a newer tree.
3. **Build the attribution matrix.** Put temporary switches inside the fast path so each
   change can be toggled alone, plus one config that turns all of them back. That config **must
   reproduce the slow path bit for bit**: it validates the harness. Capture logits at every
   position (teacher-forced), not only the last.
4. **Add a higher-precision reference** for the suspect operation (e.g. the recurrence in FP64)
   and measure **both** paths against it. "Different from the slow path" and "worse" are
   different claims.
5. **Per-position metrics.**
   - Margin m = top1 - top2 of the reference, δ = max |Δ logit|.
   - **Certificate: m > 2δ means the argmax provably cannot change.** Count flips at certified
     positions; it must be 0 (a non-zero count is a harness bug).
   - Tight certificate m > |Δ_top| + max_{j≠top} |Δ_j|, logged as a diagnostic.
   - Also argmax flip, cosine, KL(ref || x), top-5 overlap.
   - Don't use a single cosine threshold as the pass rule; it has no model-independent meaning.
6. **Prompt is the statistical unit.** Positions within a prompt are correlated. Report
   per-prompt rates (vulnerable %, flip %, conditional flip % = flips / vulnerable), with means or
   medians and bootstrap CIs over prompts. Use a corpus of tens of prompts across lengths (short
   chat, ~500, ~1200+).
7. **Paired comparison against the reference.** Per prompt, log(error_X / error_A), using KL
   or (1 - cos), not max |Δ| (tail-dominated). Report the median with a CI. Also tie-break the
   disagreements: when A and X disagree on top-1, which one matches the higher-precision
   reference?
8. **Locate where the error is born.** Inside the high-precision run, also run each candidate
   implementation of the suspect operation on **identical inputs and state**. Compare that
   local error with the per-layer hidden-state divergence. Small local error with large
   end-to-end divergence means amplification, not an operation defect.
9. **Test the handovers.** If the fast path hands state to a different path (prefill -> decode),
   prefill with each, then decode teacher-forced steps with the normal kernel. Drift must not
   jump at the first decode step relative to the reference-vs-slow-path drift.
10. **Chase outliers to the operation.** For an isolated position with huge KL, compare each
    projection at that row against the FP64 dot of the dequantized weights. That tells a kernel
    bug from an unstable point.
11. **Check product behaviour last.** Generated-continuation divergence, MTP draft acceptance,
    and agreement with llama.cpp (`llama-server` pre-sampling `n_probs`, `post_sampling_probs=false`;
    use its logprob gaps as a diagnostic, not as certificate input).

## Decision rule

Keep a slower or special-cased path only for **demonstrated behavioural harm**:
- a systematic disadvantage against the higher-precision reference over a prompt corpus;
- flips at certified positions;
- a pathological layer or regime;
- a handover discontinuity;
- or measurably worse product behaviour.

Otherwise, ship the faster path as the default and keep the slow one reachable as a control
(an env switch, documented in an ADR with the rollback criteria).

## Practical notes

- Run heavy harnesses one at a time. Numbers don't change under contention, but timings do, and
  memory adds up (a 9B model plus all-position logit buffers is about 10 GB).
- `RealWeights` tests that silently no-op when a checkpoint is missing look like passes; check
  the timing (CLAUDE.md rule 12).
- Keep experiment switches out of commits: save them as a patch next to the harness sources.
