# CPU prefill — current work

**Status:** existing int8 batched paths and runtime tuning are historical work. The repacked-GEMM
line is closed.

## Open decision

Decide whether the existing default-on CPU batched-prefill fast path is release-quality across its
eligible model/format matrix. `STINGRAY_CPU_PREFILL_Q8` is enabled unless explicitly set to `0`,
but only audited prefill callers pass the load-bearing `allowQ8` permission. The current int8
dispatch covers Q4_K, Q3_K, Q6_K and Q4_0; Q5_K, Q2_K, Q8_0 and F32 deliberately use the existing
per-token fallback.

## Required evidence

1. Expose the exact eligibility predicate in plan/startup diagnostics.
2. Retain fixture coverage including Q3_K and Q6_K routes, plus the unsupported-format fallback.
3. Run greedy-token and corpus/perplexity checks with the path on and off.
4. Measure prefill with interleaved arms, warm-up, multiple samples, and recorded hardware settings.
5. Verify continuous batching, packed prefill, cancellation, speculation, and fallback.

Historical evidence: [done/cpu-prefill-plan-2026-07.md](done/cpu-prefill-plan-2026-07.md),
[done/cpu-prefill-repack-gemm-plan.md](done/cpu-prefill-repack-gemm-plan.md), and
[done/repack-gemm](done/repack-gemm).

**Current focused receipt (2026-08-07):** `MatMulBatchedQ8EquivalenceTests` passes **24/24** in
Release. It includes Q3_K's 4/8-input dispatch/remainder cases and Q2_K's 600-token production
fallback, so the remaining decision is release-quality/performance evidence, not those two
correctness seams.
