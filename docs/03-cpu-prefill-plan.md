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

1. Expose the exact eligibility predicate in plan/startup diagnostics. **Partially complete:**
   `GET /status` publishes the process-wide `configuration.cpu_q8_prefill_enabled` gate and,
   for a built-in CPU model load, `configuration.cpu_batched_prefill` with the supported
   model-level trunk decision and reason. It deliberately rejects TurboQuant, unsupported MoE,
   and per-layer-head-dimension models (including the experimental force switch). The remaining
   per-request predicate (one-token/control-only prompts and individual weight routes) needs a
   separate execution receipt rather than a misleading load-time claim.
2. Retain fixture coverage including Q3_K and Q6_K routes, plus the unsupported-format fallback.
3. Run greedy-token and corpus/perplexity checks with the path on and off.
   Include short/pathological all-special-token prompts: ordinary-prompt cosine is 0.988–0.999
   versus F32, but a two-token all-control input reached cosine **-0.45**.

   **Resolved (2026-08-07).** `ForwardPass.Prefill` now routes an all-control-token prompt through
   the sequential F32 path instead of the int8 one (`IsAllControlTokenPrompt`). Control-only
   sequences are structural probes rather than user prose, so the exemption costs nothing on
   normal prompts — a mixed prompt, including the usual BOS + text, stays eligible for Q8. Pinned
   by `PrefillDecodeSelfConsistencyTests`, which asserts the F32 invariant with the gate off, a
   loose ≳0.98 bound with it on, and exact agreement for the all-control case.

   Note what this does *not* claim: Q8 prefill is still an approximation on ordinary prompts, and
   its quality gate remains perplexity, not a unit test.

   **Initial corpus receipt (2026-08-07):** on the first 64 Wikitext-2 test tokens with
   `Qwen3-8B-Q4_K_M`, CPU batched prefill at a 64-token chunk scored **25.9500 PPL** with Q8
   enabled versus **25.8974 PPL** with `STINGRAY_CPU_PREFILL_Q8=0` (mean-NLL delta 0.002027),
   while throughput was **19.02** versus **6.26 tok/s**. This is a focused quality/performance
   smoke, not the required multi-length, interleaved release measurement.

    **Deferred / still untested (2026-08-07):** the planned full Wikitext-2 release-quality
    comparison remains open. The evaluator correctly requires `--batched` and a Q8-eligible chunk
   (the default loop would not exercise CPU batched prefill), but the 2,048-token F32 control arm
   exceeded the available work window. Its Q8-on companion completed in 97.7 s (7.3859 PPL), which
   is deliberately **not** recorded as a quality result because the matched F32 arm did not finish.
   Resume with paired Q8-on/Q8-off runs at the same corpus slice, context, chunk size, model hash,
   and environment; retain both raw outputs before drawing any conclusion.

   **Packed-admission regression (2026-08-07):** an all-control request in a packed admission
   batch now forces the *whole* batch through exact sequential F32 admission. Previously the
   stated whole-batch fallback called `PrefillWithCache` for each neighbour, allowing ordinary
   neighbours back onto Q8 and making their numerical path arrival-dependent. The new mixed
   control/ordinary regression pins both logits against token-by-token decode.
4. Measure prefill with interleaved arms, warm-up, multiple samples, and recorded hardware settings.
5. Verify continuous batching, packed prefill, cancellation, speculation, and fallback.

Historical evidence: [done/cpu-prefill-plan-2026-07.md](done/cpu-prefill-plan-2026-07.md),
[done/cpu-prefill-repack-gemm-plan.md](done/cpu-prefill-repack-gemm-plan.md), and
[done/repack-gemm](done/repack-gemm).

**Current focused receipt (2026-08-07):** `MatMulBatchedQ8EquivalenceTests` passes **24/24** in
Release. It includes Q3_K's 4/8-input dispatch/remainder cases and Q2_K's 600-token production
fallback, so the remaining decision is release-quality/performance evidence, not those two
correctness seams.
