# Plan: IQ1_S / IQ1_M / IQ2_XS / IQ2_S coverage verification and backlog closure

**Entry in:** `docs/1-correctness/bugstofix.md`, item **16**.

## 0. Current state — this is already implemented

The original backlog entry is stale relative to the current repository. The implementation already
contains:

- `IqCodebooks.Iq1sGrid` — the 2048-entry `iq1s_grid` required by `IQ1_S` and `IQ1_M`;
- `IqCodebooks.Iq2XsGrid` — the 512-entry `iq2xs_grid` required by `IQ2_XS`;
- `IqCodebooks.Iq2SGrid` — the 1024-entry `iq2s_grid` required by `IQ2_S`;
- `IqCodebooks.Iq1sDelta = 0.125f`;
- `Dequantize.DequantIq1S`, `Dequantize.DequantIq1M`, `Dequantize.DequantIq2Xs`, and
  `Dequantize.DequantIq2S`;
- all four cases in `Dequantize.ToFloat32`;
- all four dtypes admitted by `ModelCompatibility.IsSupportedWeightDType`;
- CPU execution through existing scalar/dequant fallback and kernel paths.

The implementation history records this work as completed. `IQ2_XS` and `IQ2_S` were originally
missing formats and were later added after a real Qwen3.8-27B UD-Q3_K_XL checkpoint required them.
`IQ1_S` and `IQ1_M` were subsequently implemented with independently cross-checked dequantization
and dot-product paths.

Do not reimplement these formats unless verification below finds a concrete defect.

## 1. Objective

Convert item 16 from an obsolete implementation request into a verified coverage receipt. Establish:

1. all four formats are implemented;
2. codebooks are the real llama.cpp/ggml tables, not reconstructed approximations;
3. block layouts, scale extraction, sign handling, and shift logic are correct;
4. `Dequantize.ToFloat32` dispatches all four correctly;
5. admitted formats execute through the real CPU matmul path;
6. existing real-weight evidence for `IQ2_XS` / `IQ2_S` remains valid;
7. weaker `IQ1_S` / `IQ1_M` evidence is documented honestly;
8. stale documentation claiming these formats remain unimplemented is corrected.

This is a verification/cleanup task, not a new implementation task.

## 2. Scope

### In scope

- `src/OpenTail.Stingray.Cpu/IqCodebooks.cs`;
- `src/OpenTail.Stingray.Cpu/Dequantize.cs`;
- CPU kernel dispatch involving the four dtypes;
- `src/OpenTail.Stingray.Engine/ModelCompatibility.cs`;
- existing IQ1/IQ2 correctness tests;
- existing real-weight Qwen3.8-27B evidence;
- stale documentation references;
- additional targeted regression tests where they materially strengthen coverage.

### Not in scope

- inventing a new IQ quantization algorithm;
- rewriting already-working dequantizers;
- AVX2 optimization work for `IQ1_M`;
- CUDA/Vulkan native IQ kernels;
- downloading a huge checkpoint solely to manufacture a correctness receipt;
- changing unrelated IQ formats;
- performance work unless verification exposes an actual regression.

## 3. Phase 1 — Direct source audit

Read current implementations before making changes.

### 3.1 `IqCodebooks.cs`

Confirm the file contains `Iq1sGrid` with exactly 2048 entries, `Iq2XsGrid` with exactly 512,
`Iq2SGrid` with exactly 1024, and `Iq1sDelta == 0.125f`. Confirm comments identify the real ggml/
llama.cpp table provenance rather than generated or guessed values.

### 3.2 `Dequantize.cs`

Confirm `ToFloat32(...)` dispatches `IQ1_S → DequantIq1S`, `IQ1_M → DequantIq1M`,
`IQ2_XS → DequantIq2Xs`, and `IQ2_S → DequantIq2S`. Confirm each decoder uses the correct block
size and byte regions.

| Format | Elements/block | Bytes/block |
| --- | ---: | ---: |
| `IQ1_S` | 256 | 50 |
| `IQ1_M` | 256 | 56 |
| `IQ2_XS` | 256 | Verify exact GGML layout |
| `IQ2_S` | 256 | Verify exact GGML layout |

Do not replace the format-specific decoders with a generic IQ decoder merely for symmetry; their
layouts differ.

## 4. Phase 2 — Verify the real codebooks

Compare current literals against the vendored ggml checkout under `examples/ggml` (especially
`src/ggml-common.h` and `src/ggml-quants.c`), rather than an unrelated or moving upstream revision.
Record the checked-out reference revision with the verification result, especially if it differs from
the revision used by any compiled llama.cpp artifact. Verify entry counts, complete contents,
signedness interpretation, byte packing order, `IQ1S_DELTA = 0.125f`, `IQ2_XS` sign-mask tables
used alongside its grid, and `IQ2_S`'s distinct `qh`/`qs` index construction. Do not reconstruct
tables from a mathematical formula. A one-off comparison script is acceptable; do not add permanent
tooling unless it has ongoing value.

## 5. Phase 3 — Verify `IQ1_S` semantics

Check specifically because its sign/shift semantics differ from the IQ2/IQ3 family. Confirm the
decoder matches ggml's `dequantize_row_iq1_s` for the 16-bit `qh` field, scale bits, global sign bit,
3-bit high index components, 11-bit grid index construction, use of already-signed `iq1s_grid` values,
`±0.125f` delta, and output ordering across all eight 32-element groups. The sign bit is not an
ordinary per-element sign mask.

Keep `SimdKernelsIq1STests` as correctness evidence. Preserve the independent comparison:

`dequantize → F32 dot` versus independently-derived IQ1_S Q8_K dot kernel.

This guards against both paths sharing the same wrong formula.

## 6. Phase 4 — Verify `IQ1_M` semantics

Confirm its 56-byte/256-element block layout; `qs`, `qh`, and `scales` offsets; shared block FP16
scale extraction from the top bits of four 16-bit scale words; two 3-bit sub-scales per relevant
scale word; two scale groups within each 32-element region; four independent sign/delta bits; four
grid indices per group; and final output ordering.

Do not simplify `IQ1_M` into the `IQ1_S` decoder. Keep `SimdKernelsIq1MTests` as the primary
automated cross-check. Its independent dequantizer-versus-dot-kernel comparison is especially
important because `IQ1_M` has no equivalent AVX2 implementation to provide a second implementation.

## 7. Phase 5 — Verify `IQ2_XS`

Confirm the implementation matches ggml's `dequantize_row_iq2_xs`: `Iq2XsGrid` lookup, scale
extraction, 7-bit sign-field handling, `KSignsIq2Xs`, `KMaskIq2Xs`, `qs` index construction, output
ordering, and 256-element block boundaries. Preserve AVX2/scalar equivalence tests and Q8_K-paired
independent dot checks.

## 8. Phase 6 — Verify `IQ2_S`

Do not treat `IQ2_S` as a trivial `IQ2_XS` variant. Confirm the 1024-entry grid; 10-bit grid index
constructed from `qs` and additional `qh` bits; ggml-matching sign extraction and scale handling;
output ordering; and absence of accidental reuse of IQ2_XS-specific indexing where IQ2_S differs.
Preserve code comments explaining that `IQ2_S`'s `qh` contribution is part of the format definition.

## 9. Phase 7 — Verify CPU execution and admission

Check the complete route:

```text
GGUF tensor dtype → ModelCompatibility.IsSupportedWeightDType → tensor loading
				  → MatVec dispatch → IQ-specific kernel or MatVecDequantFallback → output
```

Confirm none of the four formats is admitted but fails later because its matmul path is absent.
Distinguish the existing routes: `IQ1_S` / `IQ1_M` use `MatVecDequantFallback`; their Q8_K-paired
dot routines are correctness oracles, not dispatched matvec paths. `IQ2_XS` / `IQ2_S` have
dedicated scalar/AVX2 matvec kernels. Do not expand item 16 into performance work.

## 10. Phase 8 — Existing real-weight evidence

Use existing Qwen3.8-27B UD-Q3_K_XL evidence as the real-weight receipt for `IQ2_XS` and `IQ2_S`.
That checkpoint mixes `IQ2_S`, `IQ2_XS`, `IQ3_XXS`, and `IQ4_XS`; its existing receipt reports a full
24-of-24-token exact greedy match against llama.cpp. This demonstrates these formats participate in a
real model, not merely that they load in isolation. Do not rerun it solely to create another copy of
the receipt unless current regression is suspected.

## 11. Phase 9 — `IQ1_S` / `IQ1_M` real-weight limitation

Do not manufacture a "real model verified" claim. Current documented evidence is:

- `IQ1_S`: independently cross-checked dequantizer and dot-kernel tests;
- `IQ1_M`: independently cross-checked dequantizer and scalar dot-kernel tests;
- no tractable real GGUF checkpoint yet supplies a llama.cpp IQ1_S/IQ1_M greedy receipt.

A DeepSeek-V3.2 IQ1_M checkpoint exists, but its published size is roughly 149 GB, making it
inappropriate to download/load merely to close this documentation gap. Do not require a huge-model
download to close item 16. Record the limitation accurately: IQ1_S/IQ1_M are implemented, admitted,
and independently formula-cross-checked, but currently lack a real-weight end-to-end receipt. A future
tractable checkpoint can strengthen evidence without reopening this item.

## 12. Phase 10 — Add targeted dequant regression tests only where useful

If current tests do not directly exercise `Dequantize.ToFloat32`, add a small focused test class, for
example under `tests/OpenTail.Stingray.Tests.ForwardPass.Fast/`. Cover one deterministic block per
dtype; multiple blocks for block-offset progression; non-zero values in important control fields;
exact expected outputs for hand-constructed blocks; alignment/output-length assertions; and dispatch
through `Dequantize.ToFloat32`.

Keep tests small and deterministic. Expected outputs must come from independently derived
calculations or known reference bytes, not a helper shared with production code.

## 13. Phase 11 — Search for stale documentation

Search the repository for claims that these formats are still missing, especially `IQ1_S still
unimplemented`, `IQ1_M still unimplemented`, `IQ2_XS still unimplemented`, `IQ2_S still
unimplemented`, `iq1s_grid missing`, and `IqCodebooks coverage gap`.

Do not rewrite historical records whose purpose is to preserve what was true at the time. Distinguish
historical record of the original gap (retain), current status document (correct), and current open
backlog (remove/close). Reconcile stale present-tense statements in
`docs/done/05-cpu-architecture-kernel-opportunities.md` that still describe IQ1_S/IQ1_M as
unimplemented.

## 14. Phase 12 — Close item 16 cleanly

Once the audit passes, remove item 16 from the open section of `docs/1-correctness/bugstofix.md`.
Link the existing implementation evidence in `docs/done/01-gguf-model-coverage-plan.md` and
`docs/done/05-cpu-architecture-kernel-opportunities.md` from the closure entry rather than recreating
the historical implementation record. Add a new `docs/done/` record only if current verification
produces genuinely new evidence. The closure must state:

- `IQ1_S`, `IQ1_M`, `IQ2_XS`, and `IQ2_S` are implemented;
- codebooks are present and reference-derived;
- all four formats are admitted;
- automated correctness coverage is present;
- real-weight Qwen3.8 receipt covers `IQ2_XS` / `IQ2_S`;
- `IQ1_S` / `IQ1_M` have no real-weight receipt yet but have independent formula cross-checks;
- no known implementation gap remains; the only remaining evidence limitation is the lack of a
  tractable real-weight `IQ1_S` / `IQ1_M` end-to-end receipt.

## Success criteria

Item 16 is complete when the required real ggml tables are present; all four formats have working
`Dequantize.ToFloat32` dispatch; block layouts and special sign/scale/index semantics are independently
checked; CPU matvec dispatch reaches a valid implementation; IQ1_S/IQ1_M independent cross-checks and
IQ2_XS/IQ2_S kernel tests pass; the Qwen3.8-27B receipt remains valid for IQ2_XS/IQ2_S; the lack of
real-weight IQ1_S/IQ1_M coverage is stated honestly; stale current documentation is corrected; and
item 16 is removed from the open correctness backlog or explicitly marked closed. No known
implementation gap remains; the only remaining evidence limitation is the lack of a tractable
real-weight IQ1_S/IQ1_M end-to-end receipt.

## Key rule

**Do not reimplement code that is already present.** The original item describes a real historical
coverage gap, but the current repository has already crossed that gap. Verify the implementation,
preserve the strongest evidence available, document the IQ1_S/IQ1_M real-weight limitation honestly,
and remove the stale backlog entry only after the audit passes.
