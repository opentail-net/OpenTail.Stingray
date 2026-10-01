# GGML op coverage verification — SSM_SCAN, RWKV6/7, DeepSeek-V4, SOLVE_TRI, and WIN_PART/UNPART

**Status: CLOSED 2026-10-02 as a source-pinned coverage re-audit.** This report classifies current
implementations and records which capabilities remain future work; it does not claim that every
listed GGML op or architecture is implemented. The remaining RWKV6, RWKV7, and generic `SOLVE_TRI`
gaps are tracked separately as items 19, 20, and 21 in `bugstofix.md`.

**Original entry:** `docs/1-correctness/bugstofix.md`, item **17**.

## 0. Current state — re-audit before implementing

The original op-list investigation identified five genuine coverage groups:

1. `GGML_OP_SSM_SCAN`;
2. `GGML_OP_RWKV_WKV6` / `GGML_OP_RWKV_WKV7`;
3. `GGML_OP_LIGHTNING_INDEXER` / `GGML_OP_DSV4_HC_COMB` / `DSV4_HC_PRE` / `DSV4_HC_POST`;
4. `GGML_OP_SOLVE_TRI`;
5. `GGML_OP_WIN_PART` / `GGML_OP_WIN_UNPART`.

The repository has moved since that investigation.

### Already present (verify against current source; do not duplicate)

Mamba-2 selective scan is implemented on CPU in
`src/OpenTail.Stingray.Engine/ForwardPass.Mamba2.cs`. The current path performs:

```text
in_proj → causal depthwise SSM_CONV → selective SSM scan → D skip
		→ SiLU gate → grouped RMSNorm → out_proj
```

and is used by Granite 4.0-H / Nemotron-H.

DeepSeek-4 also contains engine-level implementations including
`DeepSeek4Graph.LightningIndexerScore`, `DeepSeek4Graph.SelectTopKIndices`,
`DeepSeek4Graph.HyperConnectionSinkhorn`, and corresponding calls from `DeepSeek4ForwardPass`.

Do not blindly create duplicate implementations merely because corresponding GGML enum names remain
absent. The goal is to close genuine kernel/architecture coverage gaps while preserving existing
working code.

## 1. Objective

Establish reusable CPU implementations for every genuinely missing operation required by targeted
model families, while keeping architecture admission fail-closed until real-weight verification
exists. Establish:

- Mamba selective-scan coverage beyond the current Granite/Nemotron-specific Mamba-2 path;
- RWKV6 and RWKV7 recurrent-attention kernels;
- any genuinely missing DeepSeek-V4 kernel semantics;
- a reusable triangular solve;
- window partition/unpartition primitives only when justified by an actual target model;
- deterministic scalar-reference tests for every new primitive;
- no accidental duplication of working DeepSeek/Mamba logic;
- no premature `ModelCompatibility` architecture-admission changes.

## 2. Scope

### In scope

- `GGML_OP_SSM_SCAN`;
- Mamba-1 / additional Mamba-family CPU reuse of selective scan;
- Mamba-2 kernel extraction/generalization where useful;
- `GGML_OP_RWKV_WKV6` and `GGML_OP_RWKV_WKV7`;
- genuinely missing DeepSeek-V4 `LIGHTNING_INDEXER` semantics;
- genuinely missing `DSV4_HC_*` semantics;
- `GGML_OP_SOLVE_TRI`;
- `GGML_OP_WIN_PART` and `GGML_OP_WIN_UNPART`;
- scalar/reference tests;
- architecture-specific graph wiring where a real target requires it;
- documentation/status reconciliation.

### Explicitly not in scope

- blindly rewriting the existing Mamba-2 mixer;
- replacing tested `DeepSeek4Graph` helpers without evidence;
- CUDA/Vulkan implementations;
- performance tuning before correctness;
- admitting new architectures before real-weight parity;
- full DeepSeek-V4 completion as a side effect of implementing these primitives;
- unrelated ggml ops.

## 3. Phase 0 — Refresh the op-gap inventory

Before changing code, compare current vendored ggml/llama.cpp implementation against current Stingray
code. Use:

```text
examples/ggml/include/ggml.h
examples/ggml/src/ggml-cpu/ops.cpp
examples/ggml/src/ggml-cpu/ggml-cpu.c
examples/llama.cpp/llama.cpp/src/models/
```

Classify each requested operation exactly once:

- **A.** Completely absent.
- **B.** Partially implemented under a different engine-level name.
- **C.** Implemented and tested already.
- **D.** Implemented but unverified / known risky.

Do not use the historical August op-list as source of truth when current code disagrees. Produce an
internal matrix and verify each proposed equivalent; these statuses are investigation targets, not
preconfirmed conclusions:

| GGML op | Candidate Stingray equivalent | Initial status to verify | Possible remaining work |
| --- | --- | --- | --- |
| `SSM_SCAN` | Inline scan in `ForwardPass.Mamba2.cs` | Partial/generalization | Extract/reuse/verify |
| `RWKV_WKV6` | None identified yet | Missing | Kernel + graph |
| `RWKV_WKV7` | None identified yet | Missing | Kernel + graph |
| `LIGHTNING_INDEXER` | `DeepSeek4Graph.LightningIndexerScore` | Partial | Validate op contract |
| `DSV4_HC_COMB` | `HyperConnectionSinkhorn` / related helpers | Partial | Validate exact semantics |
| `DSV4_HC_PRE` | DeepSeek4 graph logic | Partial | Validate exact semantics |
| `DSV4_HC_POST` | DeepSeek4 graph logic | Partial | Validate exact semantics |
| `SOLVE_TRI` | None identified yet | Missing | Generic primitive |
| `WIN_PART` | None identified yet | Missing | Primitive when justified |
| `WIN_UNPART` | None identified yet | Missing | Primitive when justified |

The absence of a GGML op enum or a same-named Stingray method does not prove the operation itself is
absent from Stingray.

### Re-audit result (2026-10-02; authoritative over the historical initial statuses above)

References were checked at llama.cpp `63c5ef5ad47490d7dcd86d9f83c3d4304d75dc84` and vendored ggml
`8c63e70982c95ceb862e3a1073a2c1beef75d60a` (v0.20.2). Categories below describe the current
Stingray implementation against the requested GGML operator contract: **A** absent, **B** partial
or specialized equivalent, **C** implemented and tested for the requested scope, **D** present but
unverified or known risky. A passing primitive test does not verify an architecture graph or real
model behavior.

| GGML op | Class | Current Stingray evidence and remaining gap |
| --- | --- | --- |
| `SSM_SCAN` | **B** | Scalar-A Mamba-2 scan is inline in `ForwardPass.Mamba2.cs`, used by admitted Granite-H/Nemotron-H, with parity fixtures in `GraniteHybridGreedyParityTests` and `NemotronHParityTests`. This is not a reusable implementation of the full GGML contract (including `ids`, `K`, and general A layouts); do not duplicate or rewrite the working path. |
| `RWKV_WKV6` | **A** | No Stingray WKV6 kernel or architecture graph found. Reference CPU implementation is `examples/ggml/src/ggml-cpu/ops.cpp` (`ggml_compute_forward_rwkv_wkv6_f32`); llama.cpp call sites include `src/models/rwkv6-base.cpp`. No RWKV real-weight parity evidence; architecture remains unadmitted. |
| `RWKV_WKV7` | **A** | No Stingray WKV7 kernel or architecture graph found. Reference CPU implementation is `ops.cpp` (`ggml_compute_forward_rwkv_wkv7_f32`); llama.cpp call site is `src/models/rwkv7-base.cpp`. No RWKV real-weight parity evidence; architecture remains unadmitted. |
| `LIGHTNING_INDEXER` | **B** | `DeepSeek4Graph.LightningIndexerScore` implements per-key/head dot → ReLU → prescaled weight sum → additive mask for a single query. The graph supplies scale/mask and selects top-k. It is not a general port of ggml's tensor/broadcast contract, including F16 mask and K dtype conversion; DeepSeek-V4 graph remains unverified. |
| `DSV4_HC_COMB` | **D** | `HyperConnectionGate`/`HyperConnectionSinkhorn` now match ggml flat `[dst,src]` storage, softmax/normalization axes and order, and configured `n_iter`; scalar oracle tests cover iteration counts 1 and 3. Gate/graph remain alpha and unverified against real weights. |
| `DSV4_HC_PRE` | **B** | `HyperConnectionMixDown` matches the stream-weighted reduction for the eager single-token path. Gate construction is in `HyperConnectionGate`; tensor batching/broadcast and full graph behavior have no real-weight verification. |
| `DSV4_HC_POST` | **B** | `HyperConnectionMixUp` matches the eager single-token formula and now indexes the ggml flat `[dst,src]` matrix as `dst + hc*src`; asymmetric tests cover the axis convention. Full graph remains alpha/unverified. |
| `SOLVE_TRI` | **A** | No reusable generic triangular solve found. `GdnKernels` contains a specialized chunked GDN forward-substitution for its derived intra-chunk system, not a generic GGML `A X = B` primitive. Reference is `ops.cpp` (`ggml_compute_forward_solve_tri_f32`). |
| `WIN_PART` | **B** | `DeepSeekOcr2VisionEncoder.SamWindowedAttention` manually zero-pads and gathers windows; DeepSeek OCR2 has Rainbow-pattern reference fixtures. No standalone reusable GGML operation/test. Fixtures use synthetic Rainbow input, not real-image preprocessing. |
| `WIN_UNPART` | **B** | The same vision path scatters window outputs back over the non-padding region. It has output parity fixtures as part of the SAM encoder, but no isolated operation test; Rainbow fixtures do not validate production preprocessing on real images. |

Focused `DeepSeek4AlphaTests` ran on 2026-10-02: 32 passed, including Sinkhorn scalar-oracle cases,
configured-iteration wiring, and asymmetric HC mix-up coverage. The two DeepSeek OCR2 SAM reference tests were attempted
but skipped by the test harness because `STINGRAY_RUN_HEAVY_TESTS=1` was not set; they are not counted
as verified in this audit. No GGML enum-equivalence, synthetic unit test, or existing parity fixture
changes the `ModelCompatibility` admission gate. No new RWKV, generic solve, or window primitive is
authorized solely by this matrix; pursue them only with a scoped target and independent scalar tests.

## 4. Phase 1 — Generalize and verify `SSM_SCAN`

### 4.1 Current implementation control

Treat current Mamba-2 implementation in `src/OpenTail.Stingray.Engine/ForwardPass.Mamba2.cs` as the
primary working control; do not rewrite it first. Compare line by line with llama.cpp's Mamba-2
`ssm_scan` path. The implementation handles scalar-A Mamba-2 recurrence over state dimension, head,
token, grouped B/C projections, per-token `dt`, per-head `A`, and D skip.

### 4.2 Extract reusable kernel only after parity is understood

Move recurrence into a reusable CPU kernel only if doing so preserves current Granite/Nemotron
behavior exactly. A sensible destination is `src/OpenTail.Stingray.Cpu/SsmKernels.cs`, rather than
expanding `GdnKernels.cs`: GDN and classical Mamba SSM are different recurrences despite both carrying
state. Expose the minimum generic primitive required by the model graph, not the entire Mamba-2 mixer.

### 4.3 Preserve state semantics

Explicitly account for state persistence across tokens, reset at sequence start, batch/sequence
separation, token-order dependence, state writes after every timestep, and no accidental sharing
between heads or sessions. Recurrence stays sequential along token/time even where head/state axes can
be parallelized.

### 4.4 Mamba-1 compatibility

Determine whether the reusable kernel covers Mamba-1 selective scan directly or whether tensor
shapes/parameters require a separate wrapper. Do not assume Mamba-1 equals the Mamba-2 scalar-A branch.
If mathematically reusable, add a thin architecture-specific adapter rather than a duplicate
recurrence.

### 4.5 Tests

Create a deterministic scalar reference with tiny dimensions, for example state size 4, inner size 8,
six tokens, and two sequences. Verify zero/non-zero initial state, multiple timesteps, state carry,
sequence independence, reset, positive/negative `A`, varying `dt`, and batched versus one-token-at-a-
time execution. Compare the production kernel to an independently written scalar reference that does
not call production code.

## 5. Phase 2 — RWKV WKV6

Create a dedicated recurrent-attention kernel rather than putting it in `GdnKernels.cs`. Suggested
location: `src/OpenTail.Stingray.Cpu/RwkvKernels.cs`. Reference
`ggml_compute_forward_rwkv_wkv6_f32` in current vendored ggml CPU source.

Before implementation, document exact reference tensor shapes and axis ordering: recurrent state,
key/value, receptance, time-first tensor, time-decay tensor, update ordering, output calculation, and
batch/sequence behavior. Do not rely on a verbal description such as "time decay".

### Test

Use a small hand-constructed scalar recurrence and independently calculate `state(t)`, `output(t)`,
and `state(t+1)` for several timesteps. Test zero initial state and non-zero carried state. Kernel and
scalar oracle must agree within appropriate tolerance. Do not use AVX2 as the only correctness oracle.

## 6. Phase 3 — RWKV WKV7

Implement WKV7 separately after WKV6 layout conventions are understood. Reference
`ggml_compute_forward_rwkv_wkv7_f32`. Treat WKV7 as its own recurrence; do not merely parameterize
WKV6.

Verify `r`, `w`, `k`, `v`, `a`, and `b` axis ordering; recurrent state shape; update ordering; output
timing; sequence reset; and batching.

### Test

Use an independently written scalar WKV7 reference with small dimensions and at least two timesteps.
Add edge cases distinguishing update-before-read from read-before-update, element-wise decay, and
rank/update ordering. Wrong state-update order must fail.

## 7. Phase 4 — Validate existing DeepSeek-V4 indexer semantics

Do not create `Dsv4Kernels.cs` automatically. Compare `DeepSeek4Graph.LightningIndexerScore` directly
against the exact current ggml `LIGHTNING_INDEXER` operation. Verify query/key dimensions, head and
weight broadcasting, masking, causal behavior, activation, output shape, batch/head ordering, and
numerical scale.

Existing helper tests are controls, not sufficient proof merely because synthetic invariants pass. If
the helper exactly matches the op, keep it and document it as the Stingray implementation. Add a new
CPU kernel only if an existing helper demonstrably lacks required semantics or needs generalization
for another graph.

## 8. Phase 5 — Validate `DSV4_HC_COMB`

Compare existing DeepSeek4 hyper-connection combination logic, including Sinkhorn normalization,
against `ggml_compute_forward_dsv4_hc_comb_f32` and the current DeepSeek4 reference graph. Check
source/destination stream-axis convention, scale/base transforms, epsilon placement, number of Sinkhorn
iterations, alternating normalization axes, output normalization, and matrix shape across
hyper-connection counts.

Resolve the documented potential mismatch: one path has historically hard-coded one Sinkhorn
iteration despite configurable iteration-count metadata. Establish the exact reference contract before
calling this complete.

### Tests

Construct matrices where row and column normalization differ visibly. Verify exact iteration count,
row/column normalization, epsilon handling, and deterministic output against an independent scalar
implementation. Test `hc = 1`, `hc = 2`, `hc = 4`, and at least one larger synthetic case.

## 9. Phase 6 — Validate `DSV4_HC_PRE` and `DSV4_HC_POST`

Determine whether current `DeepSeek4Graph` / `DeepSeek4ForwardPass` code implements these operations
exactly. For `HC_PRE`, verify stream-axis reduction, weights, broadcast dimensions, and output layout.
For `HC_POST`, verify residual path, post gating, combination matrix, broadcast order, and output
stream ordering.

Do not build new kernels if existing graph helpers are semantically identical. If a helper is only
approximately equivalent, replace it with the smallest exact implementation.

### Tests

Use tiny tensors with deliberately distinct values per stream and asymmetric dimensions, so axis swaps
and transpositions are immediately visible.

## 10. Phase 7 — `SOLVE_TRI`

Implement a small generic lower-triangular solve primitive, in
`src/OpenTail.Stingray.Cpu/SimdKernels.cs` or a dedicated linear-algebra helper if cleaner. Use current
ggml as the contract; do not implement a general LAPACK replacement.

Match supported scope exactly: lower-triangular matrix, RHS matrix/vector, non-unit diagonal,
supported dtype combinations, and output layout.

### Tests

Construct a lower-triangular 3×3 system with one RHS and verify `A × X ≈ B`. Also test multiple RHS
columns, larger matrices, negative values, non-unit diagonals, and near-zero but non-zero diagonals.
Where practical, compare against independently calculated forward substitution rather than only the
matrix product.

## 11. Phase 8 — `WIN_PART` / `WIN_UNPART`

This is the lowest-priority group. Do not implement merely because ggml contains the enum. First
identify a concrete target model that consumes these operations. The current repo has no admitted
SAM/Swin-style vision architecture requiring them, so this is future model coverage rather than a
current correctness blocker.

If justified by a real target, implement in `src/OpenTail.Stingray.Vision/` with a small dedicated
window-partition utility.

### `WIN_PART`

Verify channel layout, height/width layout, window dimensions, window count, row/column order, edge
padding, and output tensor ordering.

### `WIN_UNPART`

Verify it exactly inverts `WIN_PART` over the non-padding region.

### Tests

Use asymmetric tensor dimensions `C=3`, `H=5`, `W=7`, window size 4, and fill every element with a
unique value. Partition then unpartition must round-trip exactly; also test dimensions divisible by the
window size. Do not add `GET_REL_POS` / `ADD_REL_POS` unless a selected real target requires them.

## 12. Phase 9 — Wire kernels into architecture graphs only after primitive tests pass

Kernel correctness and graph correctness are separate gates:

```text
primitive kernel → architecture-specific graph → small synthetic graph test
				 → real GGUF → reference parity → only then admission
```

Do not use a failing real model to establish a new primitive's mathematics when a tiny synthetic
reference isolates it. Add Mamba/RWKV graph plumbing only after recurrence tests pass. Integrate
validated DeepSeek-V4 primitives into `DeepSeek4ForwardPass` one operation at a time.

## 13. Phase 10 — Real-weight verification strategy

Evidence requirements differ by family.

### Mamba

Use the smallest practical real Mamba/Mamba-2 GGUF to verify tokenization, state initialization, first
and subsequent recurrent tokens, prefill versus decode, and greedy sequence against llama.cpp. Existing
Granite/Nemotron Mamba-2 is the control and must retain behavior.

### RWKV6 / RWKV7

Use tractable real checkpoints. Establish prompt tokenization equality, capture reference greedy
continuation, compare token by token, and verify state reset between independent prompts. Do not admit
on kernel unit tests alone.

### DeepSeek-V4

The repository documents DeepSeek-V4 as blocked by model size: smallest cited quant is roughly 99 GB,
exceeding the current machine's practical 64 GB RAM envelope. Do not require a real-weight receipt to
validate individual kernels. Use exact ggml/reference-derived primitive tests and synthetic graph
tests. Keep `deepseek4` unadmitted until a tractable real checkpoint can be executed and compared.

### Window attention

Do not admit an architecture solely because `WIN_PART` / `WIN_UNPART` utility tests pass. A real
target model must exist first.

## 14. Phase 11 — Performance only after correctness

Begin with scalar correctness paths, deterministic tests, simple layouts, and no speculative
vectorization. Only after parity consider AVX2, `Vector<T>`, FMA, parallelism, and cache-aware tiling.
Do not parallelize recurrent kernels across timesteps; parallelize only independent heads/channels/
sequences where reference semantics permit it. Correct state ordering takes priority over throughput.

## 15. Phase 12 — Documentation and admission cleanup

Once evidence is complete, update `docs/1-correctness/bugstofix.md` to describe actual remaining gaps;
do not claim `SSM_SCAN` is entirely absent when Mamba-2 has a working CPU scan, or that DeepSeek-V4
indexer/hyper-connections are entirely absent when helpers exist.

Update `docs/2-coverage/050-ggml-op-coverage-gap-plan.md` or make this item-17 plan authoritative;
do not leave conflicting current documents. Change `docs/STATUS.md` architecture status only after
real-weight verification. Primitive completion is not model-family support.

## Audit closure scope

This item is complete when the requested ops have been re-audited against pinned current ggml and
llama.cpp sources, existing Stingray equivalents and their limits have been classified, confirmed
DeepSeek HC contract defects from the audit are fixed and tested, conflicting historical coverage
claims are reconciled, and architecture admission remains fail-closed. This audit is now closed on
that basis. Closure does **not** mean every op is implemented or every target architecture is
supported.

Remaining concrete capabilities are separate future items: RWKV6 (item 19), RWKV7 (item 20), and a
generic `SOLVE_TRI` primitive (item 21). Mamba-1/general SSM_SCAN reuse and isolated reusable window
primitives remain deferred until a concrete target justifies them. DeepSeek-V4 helper/graph code stays
alpha and unadmitted without real-weight verification. `ModelCompatibility` remains fail-closed.

## Key rules

1. **Re-audit before coding.** The repository already implements some operations from the historical gap.
2. **Do not duplicate working code.** Mamba-2 scan and DeepSeek4 helpers are controls, not disposable code.
3. **Primitive correctness comes before architecture admission.**
4. **No real-weight evidence means no allowlist entry.**
5. **Do not require a 99 GB+ DeepSeek-V4 checkpoint merely to prove a CPU primitive.**
6. **Do not turn `WIN_PART` / `WIN_UNPART` into a vision architecture project without a concrete target.**
7. **Keep recurrent state semantics explicit:** correct one-token recurrence can still have faulty lifecycle, sequence isolation, or reset behavior.
8. **Use current vendored ggml/llama.cpp source as the numerical contract, not remembered formulas or older reference versions.**

Preferred debugging order:

```text
current source audit → exact ggml contract → standalone scalar reference → kernel → kernel tests
					 → architecture graph → small graph test → real GGUF → llama.cpp parity → admission
```
