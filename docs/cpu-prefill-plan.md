# CPU prefill — current work

**Status:** existing int8 batched paths and runtime tuning are historical work. The repacked-GEMM
line is closed.

## Open decision

Decide whether the existing CPU batched-prefill fast path is safe to enable by default across its
eligible model/format matrix.

## Required evidence

1. Expose the exact eligibility predicate in plan/startup diagnostics.
2. Build a fixture matrix including Q3_K and Q6_K routes.
3. Run greedy-token and corpus/perplexity checks with the path on and off.
4. Measure prefill with interleaved arms, warm-up, multiple samples, and recorded hardware settings.
5. Verify continuous batching, packed prefill, cancellation, speculation, and fallback.

Historical evidence: [done/cpu-prefill-plan-2026-07.md](done/cpu-prefill-plan-2026-07.md),
[done/cpu-prefill-repack-gemm-plan.md](done/cpu-prefill-repack-gemm-plan.md), and
[done/repack-gemm](done/repack-gemm).
