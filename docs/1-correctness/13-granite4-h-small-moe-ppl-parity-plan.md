# Plan: Granite 4.0-H small MoE PPL parity

**Entry in:** `docs/1-correctness/bugstofix.md`, item **13**.

## Context

Model: `granite-4.0-h-small-Q2_K`.

The architecture is already implemented and admitted as `granitehybrid`.

The large initial failure was already identified and fixed:

- missing MoE top-k weight renormalization;
- llama.cpp's Granite paths use `norm_w = true`;
- without renormalization, WikiText PPL was approximately 157 and generation was visibly broken;
- after the fix, the model generates normally and the remaining discrepancy is much smaller.

Current measured results:

| Configuration | Stingray | llama.cpp | Difference |
| --- | ---: | ---: | --- |
| WikiText `-c 512` | 9.3505 | 9.4103 | Stingray lower by ~0.6% |
| WikiText `-c 2048`, `[1024,+)` | 26.4155 batched | 26.1080 | Stingray higher by ~1.2% |
| WikiText `-c 2048`, `[1024,+)` | 26.5483 per-token | 26.1080 | Stingray higher by ~1.7% |

The fact that the sign and magnitude of the difference change with context length is important. It
must not be treated as evidence of a simple global scaling error.

Existing Granite-H dense controls also show that Mamba-2 summation order alone can move PPL by roughly
0.3%, so the remaining gap must be separated into recurrence/numerical effects, MoE effects,
quantization effects, and any actual model-math bug rather than assuming the entire 1.2% is a single
defect.

## Important distinction

This is **not** an architecture-admission task. `granitehybrid` is already admitted and the model
already runs correctly enough to produce coherent output. Do not reopen the completed Mamba-2 or
top-k-renormalization work unless the evidence localizes the remaining discrepancy there.

The goal is to explain the remaining PPL difference, not to force the number to match by loosening
tolerances or changing numerics blindly.

## Goals

1. Establish an apples-to-apples PPL comparison.
2. Determine whether the remaining discrepancy is primarily quantization-related, Mamba-2
   recurrence/numerical accumulation, MoE routing/mixture, attention, metadata/scale semantics, or
   another shared forward-pass issue.
3. Localize the first meaningful divergence rather than comparing only final PPL.
4. Quantify how much of the remaining gap is explained by known numerical effects such as summation
   order.
5. Fix any proven model/math bug.
6. Leave the model admitted only with a documented, reproducible correctness receipt.

## Non-goals

Do not:

- remove or change the already-fixed top-k renormalization;
- redesign the Mamba-2 implementation without evidence;
- redesign MoE execution;
- introduce new batching optimizations;
- change quantization formats merely to improve PPL;
- loosen numerical tolerances to make the result appear closer;
- treat agreement between Stingray's batched and per-token paths as proof of correctness;
- re-open unrelated Granite Vision work.

## Phase 0 — Freeze and verify the baseline

Reproduce the current measurements before changing inference code. Record:

- exact GGUF filename and hash if practical;
- GGUF architecture;
- WikiText token range;
- token count;
- exact token IDs;
- BOS/prefix handling;
- context length;
- chunk boundaries;
- batched PPL;
- per-token PPL;
- llama.cpp PPL;
- execution settings.

Run both `-c 512` and `-c 2048`. Where possible, calculate PPL from the same captured target-token logits rather
than trusting three independent PPL implementations. The comparison must score identical target
tokens.

### Baseline interpretation

Do not assume the lower `-c 512` Stingray PPL means Stingray is better, and do not assume the higher
`-c 2048` result means an architectural bug. The first question is whether the divergence grows with
sequence length because small recurrent/numerical differences accumulate.

## Phase 1 — Validate the Granite metadata contract

Before changing kernels, dump the model metadata and resolved `ModelHyperparams` for the real
checkpoint. Verify against the reference model:

- embedding scale;
- residual scale;
- attention scale;
- logit scale;
- NoPE/position handling;
- number of layers;
- attention-layer positions;
- Mamba-2 layer positions;
- Mamba-2 state size;
- Mamba-2 head count;
- convolution width;
- expert count;
- active expert count;
- expert intermediate dimension;
- shared-expert dimensions;
- tied embeddings.

Verify the GGUF representation against the source model's configuration rather than assuming names
and numerical conventions are identical. The current Granite configuration uses an embedding
multiplier, attention multiplier, logits scaling, residual multiplier, NoPE attention, and a hybrid
Mamba-2/MoE structure; confirm each for the actual GGUF rather than inferring from another Granite
checkpoint. If metadata and resolved hyperparameters disagree, fix that before numerical tracing.

## Phase 2 — Add a small deterministic logit comparison

Use a short deterministic token sequence rather than the full WikiText corpus as the primary
 debugging loop. Run:

1. Stingray per-token;
2. Stingray batched;
3. llama.cpp reference.

For every target token record token ID, target-token logit, top-5 logits and IDs, maximum logit
difference, top-1 agreement, and the logit margin between top-1 and top-2. Determine whether the PPL
discrepancy is diffuse small numerical drift across many tokens, a small number of badly wrong
tokens, or systematic drift appearing after longer contexts. Identify near-tie positions explicitly.

## Phase 3 — Locate the first divergent Granite-H layer

Instrument the Stingray model to capture per-layer output for the same deterministic sequence. At
minimum capture:

- embedding;
- every Mamba-2 output;
- every attention output;
- every MoE output;
- residual output after each layer;
- final normalized hidden state;
- final logits.

For each layer record max absolute difference between batched and per-token, cosine similarity, L2
error, and first token position where the difference becomes material. Distinguish `batched vs
per-token` from `Stingray vs llama.cpp`; neither Stingray path is the oracle.

## Phase 4 — Separate Mamba-2 drift from MoE drift

This is the central isolation step. Granite-H already has a useful dense/smaller-model control for the
shared Mamba-2 implementation. Determine whether the first significant divergence in the small MoE
checkpoint occurs:

### Before an MoE layer

Investigate Mamba-2 recurrence, convolution, selective scan, recurrent state, summation order, and
projection precision.

### Inside an MoE layer

Investigate router probabilities, expert selection, expert weight normalization, expert input, expert
outputs, shared expert, and weighted combine.

### Only after an MoE layer

Investigate the MoE path first rather than modifying Mamba-2.

Do not assume the MoE is responsible merely because this checkpoint is the MoE variant.

## Phase 5 — Quantization discrimination

The current local checkpoint is Q2_K only. Obtain a second quantization of the **same model**,
preferably `granite-4.0-h-small-Q4_K_M.gguf`, and run both Stingray and llama.cpp against that same
file. The purpose is not to claim Q4 is intrinsically "more correct"; determine whether the
Stingray-versus-reference discrepancy changes materially with weight quantization.

- If the gap remains similar across Q2_K and Q4_K_M, quantization is less likely to explain the
  remaining engine/reference difference. Concentrate on model math, recurrence, routing, and numerical
  ordering.
- If the gap changes materially with quantization, investigate affected quantized operations,
  activation range, dequantization, and accumulation before touching higher-level model logic.
- If both engines shift similarly when moving Q2_K to Q4_K_M, treat that as expected model
  quantization behavior, not an engine bug.

The important measurement is the **Stingray-minus-reference gap at each quantization**, not the
absolute PPL difference between Q2 and Q4.

## Phase 6 — Measure Mamba-2 summation-order contribution

Existing Granite-H work has demonstrated that summation order alone can move PPL by roughly 0.3%.
Use the small MoE model to quantify this rather than assuming the same amount applies. Run the Mamba-2
recurrence with the same mathematical inputs using:

1. current optimized accumulation order;
2. reference/ggml-equivalent scalar accumulation order, where practical.

Compare per-layer hidden states, final logits, and PPL. This gives an explicit bound for how much of
the remaining error can plausibly come from Mamba-2 floating-point ordering. Do not change the
production accumulation order merely to make PPL closer unless reference comparison demonstrates that
the current order is materially responsible for the discrepancy.

## Phase 7 — Audit MoE routing and combine

If the first divergence enters an MoE block, compare the complete real routing path. For every tested
token capture all expert routing probabilities, selected expert IDs, selected weights, sum of selected
weights, expert outputs, shared-expert output, and final combined MoE output.

For Granite's current softmax/top-k path verify explicitly:

`softmax(all experts) → top-k → renormalize selected weights → optional scaling`

matches the reference. The already-fixed normalization bug must have a dedicated regression test and
must not be "fixed" a second time in another place. Also verify top-k selection order, expert tensor
indexing, batched token-to-expert bucket mapping, final token scatter, and shared-expert accumulation
order.

If routing IDs and weights match but the MoE output does not, stop investigating the router and
compare expert inputs/outputs.

## Phase 8 — Check batched versus per-token state behaviour

The current measurements show a smaller difference between batched and per-token execution at
`-c 2048`: `26.4155 batched` versus `26.5483 per-token`. Determine whether this difference comes from
Mamba-2 state evolution, different accumulation order, batched MoE, batched projections, or PPL
evaluation mechanics.

Run the same deterministic sequence at increasing lengths: 1, 2, 4, 8, 16, 32, 64, 128, and 256
tokens. For each length record the first divergent layer and first divergent token. If divergence
grows smoothly with sequence length, quantify the accumulation rate rather than treating it as a
discrete bug. If it suddenly appears at a particular width or boundary, inspect indexing, state reuse,
or vector-tail handling.

## Phase 9 — Compare with llama.cpp at the first proven divergence

Where llama.cpp internal callbacks are practical, capture the corresponding reference tensor.
Otherwise use the evaluation callback and final logits. Do not make complete intermediate-state
extraction from llama.cpp a prerequisite.

Classify the result:

- Stingray per-token approximately equals llama.cpp; batched differs: primarily a Stingray batched
  execution issue.
- Stingray batched approximately equals per-token; both differ from llama.cpp: primarily a shared
  Stingray model/math issue.
- Difference begins in Mamba-2 and grows: quantify recurrence/numerical-order contribution before
  changing model equations.
- Difference begins in MoE: investigate router, expert inputs, expert outputs, and combine.
- Difference appears only at final logits: inspect final normalization/output projection/logit
  scaling before changing earlier layers.

## Phase 10 — Fix only a proven bug

Once the first real divergence is identified, make the smallest correctness change, preserve the other
execution mode as a control, do not combine the fix with optimization, immediately rerun the
 deterministic comparison, confirm that the original divergence disappears, and verify downstream
layers and final logits.

If evidence shows that the remaining PPL gap is entirely explained by legitimate floating-point
ordering or quantization behavior, do not manufacture a code change simply to reduce the percentage.
Document the measured cause instead.

## Phase 11 — Re-run the full PPL matrix

After the internal comparison is understood, rerun:

| Model | Context | Stingray batched | Stingray per-token | llama.cpp |
| --- | --- | ---: | ---: | ---: |
| Q2_K | 512 | record | record | record |
| Q2_K | 2048 | record | record | record |
| Q4_K_M | 512 | record | record | record |
| Q4_K_M | 2048 | record | record | record |

Use identical scored tokens and evaluation semantics. The key metric is `Stingray - llama.cpp` for
each quantization/context combination. This should show whether the remaining discrepancy is
quantization-sensitive, context-sensitive, batched-only, or a true model-math difference.

## Phase 12 — Regression tests and final receipt

Add deterministic tests for the actual discovered failure. At minimum:

- metadata/hyperparameter contract test;
- per-token versus batched parity on a short deterministic sequence;
- Granite MoE router/top-k normalization test;
- shared-expert regression if implicated;
- Mamba-2 state regression if implicated;
- first-divergent-layer diagnostic test or fixture where practical.

For a real-weight test, prefer a small deterministic prompt and recorded reference logits rather than
making a multi-GB model mandatory for every CI run. Record the final PPL receipt for Q2_K, Q4_K_M if
investigated, `-c 512`, and `-c 2048`.

## Success criteria

The issue is complete when:

- the current Q2_K PPL discrepancy has been reproduced from identical scored tokens;
- the context-dependent behavior is explained;
- the first meaningful internal divergence has been identified;
- any actual model/math bug has been fixed;
- top-k renormalization remains correct;
- Mamba-2 numerical-order effects have been measured rather than guessed;
- quantization effects have been separated from engine/reference differences where the Q4 control is
  available;
- batched and per-token execution are consistent within the measured numerical envelope;
- Stingray's remaining difference from llama.cpp is either materially reduced or quantitatively
  explained;
- no tolerance change is being used to conceal an unexplained divergence;
- Granite 4.0-H small remains correctly admitted and regression-tested.

## Key rule

**Do not chase the 1.2% PPL number directly. First determine how much comes from Mamba-2 accumulation,
MoE execution, quantization, and evaluation semantics. Only the unexplained remainder is a correctness
bug.**
