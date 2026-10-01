# Plan: LFM2-MoE batched/per-token parity and architecture admission

**Entry in:** `docs/1-correctness/bugstofix.md`, item **12** (LFM2-MoE `lfm2moe` batched/per-token parity and admission).

## Context

Initially logged 2026-09-28 as `docs/103-quickest-first-plan.md` item 13.

Model: `LFM2-8B-A1B-Q4_K_M`.

Observed on WikiText `[256,1024)`:

| Configuration | PPL |
| --- | ---: |
| Stingray per-token, `-c 512` | 8.1860 |
| Stingray batched, `-c 512` | 8.9595 |
| llama.cpp `llama-perplexity --chunks 1`, `-c 512` | 8.7030 |
| Stingray batched, `-c 2048` | 15.8076 |
| llama.cpp, corresponding `-c 2048` | 14.8639 |

The batched/per-token discrepancy is too large to treat as ordinary floating-point variation. The
per-token result is not presumed correct merely because it is the existing control: neither Stingray
execution mode is the oracle. llama.cpp is the external numerical reference, and the investigation
must explain how each Stingray path compares against it.

Current implementation already contains LFM2-style graph wiring:

- NeoX-style backbone;
- LFM2 short-convolution layers;
- full-attention layers;
- sigmoid expert gating;
- `exp_probs_b` expert-bias behavior;
- top-k expert selection;
- top-k weight renormalization corresponding to `norm_w = true`.

The architecture is nevertheless not admitted by the generic model compatibility allowlist.

## Investigation log — attention divergence and safe default (2026-10-01)

Downloaded the official `LiquidAI/LFM2-8B-A1B-GGUF` Q4_K_M checkpoint to
`models/_models/LFM2-8B-A1B-Q4_K_M.gguf` (5,044,779,712 bytes; the Hugging Face file page reports
SHA-256 `d2185b22630fc68043dac7182f12e86e5ad14990229a90b6c9ad3f4421ddaf82`). With CPU Q8 off
and `STINGRAY_MIN_BATCH_BLAS=999999`, the first 1,024 WikiText tokens reproduced a smaller but
position-wise batched/per-token gap: `[256,1024)` PPL 7.5275 batched versus 7.6135 per-token; all
768 target NLLs differed (mean absolute delta 0.079244, maximum 1.601762). At `-c 2048`,
`[1024,+)` PPL was 14.3531 batched versus 14.2242 per-token. Chunk sizes 16, 64, and 256 also
changed NLLs at every target in `[256,1024)`, confirming that the experimental path's result
depends on batch shape.

Captured every layer output for the actual 256-token WikiText prefix with both Q8 gates off and
OpenBLAS excluded (`MinBatchForBlas = int.MaxValue`). Short-convolution layers 0 and 1 matched
bit-for-bit. The first difference was attention layer 2 at token position 1 (maximum absolute
layer-output delta 0.0002494); by layer 23 the maximum delta was 1.357519. This locates the
divergence in the batched attention path, not short-convolution state or expert kernels.

Until that arithmetic difference is explained against llama.cpp, LFM2-MoE no longer uses the
experimental recurrent batched trunk by default. `STINGRAY_LFM2_MOE_BATCHED_PREFILL=1` opts back
in. The separate `STINGRAY_RECURRENT_BATCHED_PREFILL=0` switch still disables recurrent batching
for all applicable models. With the LFM2-MoE safe default and both Q8 overrides unset, batched CLI
perplexity at `-c 1024` and `-c 2048` produced NLL dumps exactly equal to token-by-token execution
(1,023/2,047 targets; zero changed values; maximum absolute NLL delta 0). The corresponding
`[256,1024)` PPL is 7.6135 and `[1024,+)` PPL is 14.2242. At `-c 2048`, throughput was 27.30 tok/s
versus 73.62 tok/s in the experimental batch-256 run; the explicit opt-in retains that performance
while the correctness work continues.

Added `Lfm2MoeBatchedPrefillParityTests` against the real checkpoint and WikiText prefix. It forces
the experimental gate off while the general recurrent-batching switch stays on, then asserts
bit-identical full-vocabulary logits for every position against independent token-by-token passes.
Build and the real-weight test passed on 2026-10-01. This closes only batched/per-token parity under
the safe default: same-semantics llama.cpp comparison, explanation of the experimental attention
drift, PPL/reference reconciliation, and architecture admission remain open.

### Important distinction

Do not solve this by simply adding `lfm2moe` to `ModelCompatibility`. The architecture must first be demonstrated numerically correct. Admission is the final step, not the first.

## Goals

1. Identify why LFM2-MoE batched evaluation disagrees with the existing per-token implementation.
2. Establish exact correspondence between Stingray per-token, Stingray batched, and llama.cpp at identical token positions.
3. Localize the first numerical divergence to a specific layer/component.
4. Fix the smallest concrete correctness issue found.
5. Re-run PPL and hidden-state comparisons.
6. Only after parity is established, admit `lfm2moe` through the normal generic text-generation path.
7. Add regression coverage so the batched/per-token discrepancy cannot silently return.

## Non-goals

Do not redesign MoE execution, introduce parallel expert execution as part of the correctness fix, optimize expert dispatch yet, rewrite the working dense LFM2 implementation, change quantization formats, loosen numerical tolerances, or add `lfm2moe` to the compatibility allowlist before the numerical issue is understood.

## Phase 0 — Freeze the current baseline

Record the existing results exactly before changing inference code. For the fixed WikiText token slice `[256,1024)`, record:

- per-token PPL at `-c 512`;
- per-token PPL at `-c 2048`;
- batched PPL at `-c 512`;
- llama.cpp PPL/reference;
- batched PPL at `-c 2048`;
- llama.cpp corresponding result;
- exact token IDs used;
- chunk boundaries;
- batch size;
- position IDs;
- number of evaluated target tokens.

Do not rely solely on displayed PPL values. The debugging harness must evaluate exactly the same target token positions through all three paths: Stingray per-token, Stingray batched, and llama.cpp evaluation callback. Use the same input/target tokens and chunk boundaries.

Neither Stingray execution mode is assumed correct. Per-token is a control path, not a reference
implementation; llama.cpp is the external numerical oracle. The investigation must explain why each
Stingray path differs from that reference rather than using agreement with the per-token path alone as
proof of correctness.

## Phase 1 — Make the PPL comparison apples-to-apples

Before debugging model arithmetic, verify that all three evaluators score the same thing. Confirm identical tokenization, `[256,1024)` token range, context length, target-token count, initial recurrent/short-conv state, position numbering, BOS/prefix handling, chunk reset behavior, logits-to-NLL calculation, and absence of sampling/generation logic. For every comparison, capture the initial per-layer short-convolution state and prove that per-token and batched evaluation start from identical state; do not attribute downstream divergence to MoE until the shared recurrent state is established as identical.

For the first diagnostic comparison, calculate PPL directly from captured logits rather than relying on three independent PPL implementations. For each scored token, record:

- `token_index`;
- `target_token`;
- `per_token_logit[target_token]`;
- `batched_logit[target_token]`;
- `llama_reference_logit[target_token]`.

Derive PPL from those identical observations. Establish evaluation equivalence before broad model changes.

### 1.1 Small deterministic reproducer

Before using the full WikiText slice as the primary debugging instrument, find the smallest
deterministic token sequence and batch width for which batched and per-token logits diverge. Use that
same sequence, initial model state, and positions for all internal tracing and the llama.cpp comparison
where practical. Retain the fixed WikiText evaluation as the baseline and final PPL regression, not
the first-line debugging loop.

## Phase 2 — Add per-layer hidden-state instrumentation

For one short deterministic token sequence, run Stingray per-token, Stingray batched, and llama.cpp callback/reference where practical. Capture hidden state at every LFM2-MoE layer, including:

1. embedding output;
2. layer input;
3. short-conv output where applicable;
4. attention output where applicable;
5. pre-MoE hidden state;
6. MoE/router input;
7. router scores;
8. selected expert IDs;
9. selected expert weights after renormalization;
10. each selected expert output;
11. combined MoE output;
12. post-residual layer output;
13. final hidden state;
14. final logits.

For each stage record max absolute difference, maximum relative difference, cosine similarity, L2 error, and first token position showing divergence. Retain enough precision to distinguish numerical drift from indexing/state errors; do not rely on rounded decimal output.

## Phase 3 — Determine whether routing itself diverges

Existing work has ruled out serial versus parallel expert execution, serial versus parallel routing, and the isolated expert MatVec kernels on real Q4_K/Q6_K weights. Preserve those findings as established controls rather than repeating them. Compare the actual routed values in the real model for every MoE token:

- Compare all 32 unbiased sigmoid probabilities and all 32 selection scores after `exp_probs_b`, then compare the selected expert IDs and final mixture weights.
- Verify `exp_probs_b` is applied at the same stage as the reference. Distinguish its effect on expert selection from the original sigmoid probabilities used as mixture weights; do not apply bias to final mixture weights if the reference only uses it for selection.
- Compare selected expert IDs, selection order, score values, and near-ties.
- Verify selected sigmoid probabilities are renormalized identically; record `sum(selected_weights)` per token.

If router IDs and weights are identical but the MoE output differs, compare expert inputs/outputs next rather than continuing to investigate routing.

## Phase 4 — Compare the actual expert path

The isolated MatVec comparison has already shown that the underlying expert matmul primitive agrees on fixed real-weight inputs. Do not treat this as proof that `MoeBatchedExperts.Run` is correct. Compare the complete batched expert path using actual LFM2-MoE activations, expert IDs, bucket positions, and outputs. For each selected expert and identical token/expert ID, compare:

`expert_input → gate projection → activation → up/down projection → expert_output`

This should distinguish different MoE input activations, incorrect expert/token indexing, wrong expert weight lookup, batched gather/scatter errors, accumulation order, incorrect output placement, and incorrect mixture weighting. Pay particular attention to `batch row → token → selected expert → expert output slot → original token row`; matching expert IDs do not rule out a batched indexing/scatter error.

## Phase 5 — Check LFM2 hybrid state handling

If the first divergence occurs before or around a short-convolution layer, inspect recurrent state per sequence and token: state read/write index, position, sequence ID, convolution window, and state contents before and after the layer. Batched execution must match repeated one-token causal state transitions. Verify sequence isolation, position-based rather than batch-row-based state indexing, exactly-once state updates, token `t` to `t+1` state handoff, and chunk reset/reuse behavior. Do not assume state handling is the cause; use dense LFM2 as a control for shared hybrid architecture behavior.

## Phase 6 — Compare batched and per-token at progressively larger widths

Once the first divergence is known, reduce it to the smallest reproducible case. Test widths 1, 2, 4, 8, 16, 32, and 64; at each width compare the first divergent layer/token. Failure only for `n > 1` isolates batched execution; width-specific behavior can reveal vector-width/tail bugs; shifting divergence positions can expose state/indexing errors. Retain the expert-matmul result as a control so this tests the complete routed execution, not the MatVec primitive again.

## Phase 7 — Compare against llama.cpp at the same internal boundary

Use the llama.cpp evaluation callback as external reference, not only final PPL. Use llama.cpp internal tensors where accessible; otherwise use the evaluation callback and final logits as the external oracle. Do not make access to every intermediate llama.cpp hidden state a prerequisite for progress. For a short deterministic sequence, compare hidden state after layers where accessible, router outputs, selected experts, expert weights, and final logits. Classify the first divergence:

- Stingray per-token equals llama.cpp, batched differs: focus on Stingray batched execution.
- Stingray batched equals per-token, both differ from llama.cpp: focus on common Stingray graph/math.
- All three progressively differ: inspect shared assumptions such as tensor layout, normalization, activation, short-conv semantics, RoPE, or weight interpretation.
- Only some layers differ: fix the first divergent layer rather than later computations.

## Phase 8 — Fix the first proven divergence

Make the smallest correctness fix. Do not combine it with performance optimization, expert parallelism, new quantization support, or broad `ModelGraph` refactoring. Preserve the per-token path as control while modifying batched behavior. Immediately rerun the same deterministic hidden-state comparison; the original first divergence must disappear, then verify downstream layers through final logits within the established numerical envelope.

## Phase 9 — Reconcile PPL

After internal parity is established, rerun WikiText `[256,1024)` at `-c 512` and `-c 2048` for Stingray per-token, Stingray batched, and llama.cpp. Record both per-token and batched results at each context length so context-length sensitivity in each execution path can be distinguished. Per-token and batched execution must converge to the same numerical result on the same token sequence and initial model state, independent of batch width, before either mode is considered validated. Agreement between the two Stingray paths alone is insufficient to establish correctness; compare both against llama.cpp using identical logits/token sets. Do not demand byte-identical logits where operation ordering or quantization paths legitimately differ; require no large systematic drift, target-token logits within the established numerical envelope, expected top-ranked-token agreement, and PPL consistent with the reference evaluation. Use dense LFM2's working batched/per-token agreement as a practical numerical control, not as a replacement for llama.cpp reference comparison.

## Phase 10 — Add regression tests

Add deterministic tests for the discovered failure mode:

- fixed small sequence: per-token and batched logits and hidden states agree within established tolerance;
- routing parity: expert IDs, selected weights, and MoE output agree;
- batch-width matrix prevents width-specific regressions;
- if short-conv state is implicated, sequential state evolution agrees with batched execution;
- fixed WikiText slice records a known-good PPL range, rather than merely asserting evaluation completes.

Real-model tests may remain opt-in/manual if CI does not carry the multi-GB checkpoint.

## Phase 11 — Admit `lfm2moe`

Only after numerical checks pass:

1. Add `lfm2moe` to `ModelCompatibility`'s generic text-generation allowlist.
2. Confirm `RunCommand` admits the architecture normally.
3. Confirm Doctor/static model planning recognizes it consistently.
4. Confirm server/model loading paths do not have a separate architecture restriction.
5. Add a regression proving architecture admission.

Do not create a diagnostic-only bypass. The normal path should recognize `lfm2moe` because its implementation has been demonstrated correct.

## Phase 12 — End-to-end validation

Validate the real `LFM2-8B-A1B-Q4_K_M` checkpoint with normal `stingray -m ...` loading, normal prompt generation, greedy/deterministic generation, batched evaluation, per-token generation, WikiText PPL, and llama.cpp comparison. Also verify model metadata, tensor dimensions, expert count, top-k count, layer count, short-conv configuration, attention layer placement, tied/untied embedding assumptions, and EOS behavior.

## Success criteria

The issue is complete when:

- `lfm2moe` is correctly recognized by the model graph;
- per-token and batched execution converge to the same numerical result on deterministic inputs independent of batch width;
- both execution modes are separately validated against llama.cpp; agreement between the two Stingray modes alone is insufficient;
- the first known divergence is identified and fixed rather than hidden by tolerance changes;
- router expert IDs and mixture weights agree with the reference;
- MoE outputs agree within the established numerical envelope;
- hybrid short-conv state, where involved, is proven correct;
- final logits are consistent across execution modes;
- WikiText PPL no longer shows the large batched/per-token split;
- Stingray PPL is reproducibly comparable with llama.cpp using identical scored tokens;
- real `LFM2-8B-A1B-Q4_K_M` generation works;
- `lfm2moe` is admitted through the normal compatibility gate;
- regression tests protect architecture admission and batched/per-token parity.

## Key rule

**Do not add `lfm2moe` to the allowlist until the 9% batched/per-token discrepancy has been explained and fixed.**
