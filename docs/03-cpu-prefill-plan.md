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

   **Paired corpus receipt — COMPLETE (2026-08-08).** The 2026-08-07 attempt left this open because
   the F32 control arm did not finish; only its Q8-on companion (97.7 s, 7.3859 PPL) completed, and
   that number was correctly withheld as a quality result. The obstruction was the *work window*, not
   the measurement: `-c` is tokens scored rather than context length, so each arm is a single
   2,048-token pass, and F32 takes ~5 min. Both arms have now been run back to back.

   Model `Qwen3-8B-Q4_K_M.gguf` (SHA-256 `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`),
   corpus `scripts/kvarn-gate/wiki.test.raw` (Wikitext-2 test), `-c 2048 --batched
   --batch-chunk-size 512`, CPU backend, no OpenBLAS, 2,047 tokens scored. Identical in every
   respect except `STINGRAY_CPU_PREFILL_Q8`:

   | Arm | mean NLL | PPL | elapsed | tok/s |
   |---|---|---|---|---|
   | `STINGRAY_CPU_PREFILL_Q8=0` (F32 control) | 2.002994 | **7.4112** | 316.5 s | 6.47 |
   | `STINGRAY_CPU_PREFILL_Q8=1` (Q8 on) | 1.999567 | **7.3859** | 97.9 s | 20.92 |

   **Δ mean NLL −0.003427 nats, ΔPPL −0.341%, at 3.23× the throughput.** Q8 scores marginally
   *lower* perplexity than the exact path. That is not evidence Q8 is more accurate — both runs are
   deterministic, so this is a real reproducible numerical difference rather than noise, but the
   direction is incidental. The defensible claim is only this: on this slice the int8 approximation
   costs nothing measurable in quality while tripling prefill throughput.

   Per position bucket (F32 → Q8): `[1,256)` 12.3890 → 12.4026 (+0.110%), `[256,1024)` 6.0789 →
   6.0579 (−0.345%), `[1024,+)` 7.5660 → 7.5318 (−0.452%). Q8 is slightly worse only in the first
   bucket and better beyond it — independently reproducing the same shape seen in the earlier
   unrecorded multi-length runs.

   The Q8 arm reproduced 7.3859 to four decimals against the 2026-08-07 partial run, confirming the
   configuration is pinned by model hash, corpus slice and chunk size.

   **Scope of this receipt:** one model, one 2,048-token slice, one chunk size, single sample, no
   warm-up and no interleaving. It closes the corpus/perplexity half of this item. It does **not**
   satisfy item 4, which still needs interleaved arms, warm-up and multiple samples before the
   throughput ratio above is quotable as a performance result.

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
