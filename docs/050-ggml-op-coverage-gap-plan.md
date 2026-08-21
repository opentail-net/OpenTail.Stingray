# 050 — GGML Op Coverage Gap Implementation Plan

Source: `docs/bugstofix.md`'s 2026-08-21 GGML op-list diff (`enum ggml_op` in
`examples/ggml/include/ggml.h` vs. what this engine implements). Five ops/op-groups are
genuinely missing kernel implementations, none currently blocking any admitted architecture.
This plan adds the *kernels* (with unit tests validated against ggml's own reference semantics)
without touching `ModelCompatibility.cs`'s allowlist — architecture admission stays gated behind
real-GGUF greedy-parity verification, per this repo's standing rule (see `docs/bugstofix.md`'s
IqCodebooks.cs and deepseek2 entries for precedent). Reference C implementations for all five
live in `examples/ggml/src/ggml-cpu/ops.cpp`.

Priority order below is by (a) how many real model families the op unlocks and (b) how self
contained the math is. Each phase is independently shippable and independently tested; do not
block phase N+1 on phase N being "wired in" anywhere beyond kernel + unit test.

---

## Phase 1 — `GGML_OP_SSM_SCAN` (Mamba / Mamba2 selective scan)

**Value**: highest. This is the one missing piece for the entire Mamba/Mamba2/Jamba/Zamba/
FalconMamba/Codestral-Mamba family. We already have `GGML_OP_SSM_CONV` (`GdnKernels.cs`) — the
causal conv1d prelude — so this closes the loop for real SSM/hybrid-SSM support.

- **ggml reference**: `ggml_compute_forward_ssm_scan_f32`, `ops.cpp:9627`. Signature:
  `ggml_ssm_scan(ctx, s, x, dt, A, B, C, ids, K)` — `s` is the recurrent state
  `[d_state, d_inner, n_seqs]` (or per-head for Mamba2), `x` the conv-output input, `dt` the
  per-channel time-step (softplus'd), `A`/`B`/`C` the SSM parameter tensors, `ids` selects which
  state slot each sequence in the batch reads/writes (continuous-batching support), `K` mirrors
  gated-delta-net's "how many trailing state snapshots to keep" parameter.
- **Where it goes**: `src/OpenTail.Stingray.Cpu/GdnKernels.cs` (same file as `SsmConv`, same
  per-sequence-state-slot conventions already established there) — new `SsmScan(...)` kernel.
  New graph node in `src/OpenTail.Stingray.Engine/` alongside how `GatedDeltaNet`/hybrid-GDN
  graphs are wired (see `HybridGdnForwardPass`), but do NOT add a new architecture string to
  `ModelCompatibility.cs` yet.
- **Algorithm** (from the ggml doc comment + reference impl): per-timestep recurrence
  `state = state * exp(dt * A) + dt * B * x`, output `y = C^T * state (+ D * x if present)`,
  parallel over `(d_inner, n_seqs)`, sequential over `n_tokens` — same shape as the
  `GatedDeltaNet` chunked-recurrent kernel already in this codebase, so port that kernel's
  parallelization pattern rather than inventing a new one.
- **Test strategy**: hand-construct small tensors (e.g. `d_state=4, d_inner=8, n_tokens=6,
  n_seqs=2`), compute the reference recurrence in a scalar C# loop directly from the formula
  above (not from ggml — no ggml binary dependency in tests), and assert the SIMD/parallel kernel
  matches it exactly (integer/exact recurrence, no quantization involved, so this can be an exact
  equality test unlike the quant kernels' tolerance-based ones).
- **Effort**: substantial — new kernel + new graph shape + state-slot bookkeeping analogous to
  `PagedKvCache`'s but for SSM state, not KV. This is the biggest single piece of this plan.

## Phase 2 — `GGML_OP_RWKV_WKV6` / `GGML_OP_RWKV_WKV7`

**Value**: high — unlocks RWKV6/RWKV7 architectures, a distinct recurrent-attention family from
both standard transformer and Mamba.

- **ggml reference**: `ggml_compute_forward_rwkv_wkv6_f32` (`ops.cpp:10263`),
  `ggml_compute_forward_rwkv_wkv7_f32` (`ops.cpp:11244`). Signatures:
  `ggml_rwkv_wkv6(ctx, k, v, r, tf, td, state)` (time-first/time-decay recurrence),
  `ggml_rwkv_wkv7(ctx, r, w, k, v, a, b, state)` (WKV7's generalized delta-rule form — closer in
  shape to `GatedDeltaNet` than WKV6 is).
- **Where it goes**: new `RwkvKernels.cs` in `src/OpenTail.Stingray.Cpu/` (separate file — WKV6/7
  are a different recurrence family from GDN/SSM despite superficial similarity, and mixing them
  into `GdnKernels.cs` would blur that). New graph path in `Engine/`, again without an
  allowlist entry.
- **Algorithm**: both are per-timestep linear recurrences over `(head, d_head)` state, WKV6 using
  a scalar time-decay `td`/time-first `tf` pair per head, WKV7 using vector-valued `a`/`b`
  (in-context learning rate terms, structurally the same shape as `GatedDeltaNet`'s
  `beta`/rank-1 update). Implement WKV7 first if effort must be split — its math is closer to
  code we already have (`GatedDeltaNet`), reducing the chance of a fresh algorithmic bug.
- **Test strategy**: same approach as Phase 1 — scalar reference loop from the recurrence
  formula, exact-match assertion against the SIMD kernel on small hand-built tensors.
- **Effort**: moderate per variant; do WKV7 then WKV6 (or vice versa) as two separate,
  independently-landable sub-phases.

## Phase 3 — `GGML_OP_LIGHTNING_INDEXER` + `GGML_OP_DSV4_HC_COMB` / `_HC_PRE` / `_HC_POST`

**Value**: moderate, but these four ops are a *coupled set* for one architecture generation
(DeepSeek-V4's sparse "lightning indexer" attention gate plus its hyper-connections residual
replacement, per the ggml doc comment's arXiv link) — implement together or not at all, since a
partial set unlocks nothing.

- **ggml reference**: `ggml_compute_forward_dsv4_hc_comb_f32` (`ops.cpp:10992`), `..._hc_pre_f32`
  (`ops.cpp:11098`), `..._hc_post_f32` (`ops.cpp:11163`). `lightning_indexer`'s CPU forward is in
  `ggml-cpu.c` per the earlier grep — locate it precisely before starting (not yet pinned down in
  this survey).
- **Math** (from `ggml.h`'s own doc comments, already fairly complete):
  - `lightning_indexer(q, k, weights, mask)`: a prescaled, masked query-key indexer score,
    `res[n_kv, n_batch, 1, ne3]` — essentially a cheap gating attention-score precursor used to
    decide which KV positions the real attention pays attention to (DeepSeek's sparse attention
    mechanism). Broadcast rule `ne3 % ne33 == 0` on the mask.
  - `dsv4_hc_comb(mixes, scale, base, eps, n_iter)`: builds a doubly-stochastic-ish mixing matrix
    over `hc` streams via iterated softmax-then-normalize (`n_iter` rounds, alternating
    normalization axis) — this is the most algorithmically novel piece, worth writing a
    standalone scalar reference implementation from the doc comment's formula before touching
    SIMD.
  - `dsv4_hc_pre(x, weights)`: a weighted sum over the `hc` stream axis — cheap, straightforward.
  - `dsv4_hc_post(x, residual, post, comb)`: broadcasts `x` back out to per-stream residuals via
    `post` gating plus the `comb` mixing matrix from `hc_comb` — also straightforward once
    `hc_comb`'s output shape is validated.
- **Where it goes**: new `Dsv4Kernels.cs` (hyper-connections ops) + extend the existing MLA/
  attention kernels for `lightning_indexer` (lives conceptually next to `MlaAttention.cs` given
  this is a DeepSeek-family op). New graph work in `Engine/`.
- **Test strategy**: `dsv4_hc_comb`'s iterative-normalization step is the one piece here worth
  extra scrutiny — write the scalar reference directly from the doc comment
  ("softmax over dst, add eps, normalize over src, repeat n_iter-1 more times alternating axes")
  and cross-check row/column sums equal 1 (or 1+eps) after normalization, not just numeric
  equality, since a mis-ordered axis swap would still produce *some* plausible-looking numbers.
- **Effort**: moderate-to-substantial, mostly in getting `hc_comb`'s iterated normalization loop
  exactly right — the other three ops are comparatively mechanical once shapes are pinned down.

## Phase 4 — `GGML_OP_SOLVE_TRI`

**Value**: low on its own (no specific architecture in this survey needs it directly), but it's
a generic O(n³) triangular solve `Ax=B` primitive ggml added — worth having as reusable
infrastructure since a future architecture (or a future DSV4-adjacent op) may assume it exists.
ggml itself only implements the lower-triangular, right-hand-side, non-unitriangular variant
today (per its own `TODO` comment) — match that scope exactly, do not over-build.

- **ggml reference**: forward-substitution solve, `ggml-cpu.c`/`ops.cpp` (exact line not yet
  pinned in this survey — locate at implementation time).
- **Where it goes**: a small standalone static method in `SimdKernels.cs` (it's a generic linear
  algebra primitive, not tied to any specific architecture's kernel file).
- **Test strategy**: construct a small triangular `A` and dense `B` by hand, verify
  `A @ solve_tri(A, B) == B` to floating-point tolerance — this self-checks without needing a
  ggml-derived reference value at all.
- **Effort**: small. Good candidate to do *first* if someone wants a quick, low-risk warm-up
  before tackling Phase 1's much larger SSM_SCAN piece.

## Phase 5 — `GGML_OP_WIN_PART` / `GGML_OP_WIN_UNPART` (+ `GET_REL_POS`/`ADD_REL_POS`)

**Value**: lowest — per ggml's own comments these are "used in sam" (Segment Anything Model),
i.e. Swin-style windowed vision attention. `OpenTail.Stingray.Vision` currently covers Gemma3/
Gemma4/Llama4, none of which use windowed attention — this unlocks a vision architecture family
we have zero other infrastructure for yet (no SAM tokenizer/preprocessing path either), so this
is the one phase where "implement the kernel anyway" has the weakest near-term payoff.

- **ggml reference**: `ggml_compute_forward_win_part_f32` (`ops.cpp:9867`); `win_unpart` and the
  rel-pos ops are nearby in the same file (not yet individually pinned down).
- **Math**: `win_part` is a pure reshape/pad operation (partition `[C,H,W,1]` into
  `[C,w,w,n_windows]` non-overlapping tiles with zero-padding at the edges if `H`/`W` aren't
  multiples of `w`); `win_unpart` is its exact inverse. `get_rel_pos`/`add_rel_pos` implement
  SAM's relative-position attention bias. None of these involve floating-point reduction-order
  subtlety — they're index/copy operations, so correctness is about getting the padding/tiling
  arithmetic exactly right, not numerical tolerance.
- **Where it goes**: `src/OpenTail.Stingray.Vision/` if/when a SAM-family model is actually
  planned; until then, lowest priority of the five and reasonable to defer entirely unless a
  concrete SAM/Swin request appears.
- **Test strategy**: partition then unpart a hand-built tensor with non-multiple-of-`w`
  dimensions, assert exact round-trip (padding regions excluded) — a pure shape/copy test, no
  tolerance needed.
- **Effort**: small per-op, but lowest value — do last, and only on explicit request.

---

## Cross-cutting rules for all five phases

1. **No allowlist changes.** None of these land in `ModelCompatibility.cs`'s
   `IsTextGenerationArchitectureSupported`/`IsSupportedWeightDType` until a real GGUF using the
   architecture has been greedy-parity-verified against llama.cpp, per this repo's standing rule.
2. **Exact-match tests, not tolerance-based, where the op is non-quantized.** All five op
   families here operate on F32/F16 tensors with no int/quant reduction — unlike the IQ-format
   dequant work, correctness tests should assert exact (or near-machine-epsilon) equality against
   a scalar reference loop written directly from each op's math, not a loosened RMSE budget.
3. **One phase, one PR-sized chunk.** Each phase above is independently buildable/testable;
   don't let Phase 1's size block starting Phase 4 (SOLVE_TRI) as a quick win first if that's
   preferred ordering.
4. **Pin down exact `ops.cpp` line numbers for `lightning_indexer` and `win_unpart`/rel-pos
   before starting** those phases — this survey found their *file* (`ggml-cpu.c` for
   lightning_indexer's dispatch, `ops.cpp` for the rest) but not their exact line ranges.
