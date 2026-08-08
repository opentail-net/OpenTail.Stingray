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

   **COMPLETE (2026-08-08).** Method: one warm-up run per arm (discarded), then three
   **interleaved** rounds `F32, Q8, F32, Q8, F32, Q8` — not `AAA/BBB` — so thermal drift and any
   background load hit both arms alike. Same production harness as the quality receipt
   (`perplexity … --batched --batch-chunk-size 512`), `Qwen3-8B-Q4_K_M`, `-c 1024`, wiki.test.raw.
   Hardware: Ryzen 7 5700G, 6 cores / 12 threads, **AVX2 only (no AVX-512, no VNNI)**, OpenBLAS not
   present (sequential fallback), CPU backend, `-g 0`.

   | Arm | samples (tok/s) | median | mean | spread |
   |---|---|---|---|---|
   | `STINGRAY_CPU_PREFILL_Q8=0` (F32) | 6.91, 6.84, 6.90 | **6.90** | 6.883 | 1.01% |
   | `STINGRAY_CPU_PREFILL_Q8=1` (Q8)  | 24.17, 24.02, 23.31 | **24.02** | 23.833 | 3.58% |

   **Ratio 3.48× on medians (3.46× on means).** Run-to-run spread is 1.0% (F32) and 3.6% (Q8), so
   the ratio is comfortably outside the noise — this is the first time that has been demonstrated
   rather than assumed from a single sample per arm.

   **Warm-up mattered, asymmetrically:** the discarded first Q8 run scored 23.40 tok/s, 2.6% below
   its steady-state median, while F32's warm-up run was indistinguishable from its steady state.
   Charging that JIT/page-cache cost to whichever arm happened to run first is exactly the
   methodology error the warm-up exists to remove, and it biases *against* Q8.

   **Determinism confirmed as a by-product:** perplexity was bit-identical across all four runs of
   each arm (F32 7.2594, Q8 7.2426), so the throughput samples differ only in timing. ΔPPL −0.231%
   at this length, consistent in sign and magnitude with the −0.341% measured at `-c 2048`.

   **Relationship to the earlier 3.23× figure:** that was `-c 2048`, single sample, no warm-up,
   sequential. This is `-c 1024` with clean methodology. The two are **not** directly comparable —
   context length changes the prefill mix — so treat 3.48× as the quotable figure *at 1024 tokens on
   this hardware*, and do not describe it as a correction of 3.23×.

   **Scope:** one model, one context length, one chunk size, one machine. AVX2-only, so a VNNI or
   AVX-512 host would land elsewhere; the int8 path is precisely the one that benefits most from
   instruction sets this CPU does not have.
5. Verify continuous batching, packed prefill, cancellation, speculation, and fallback.

   **Chunked prefill does NOT match single-shot prefill under Q8 (2026-08-08, first evidence).**
   `ContinuousBatchingTests.PrefillWithCache_Chunked_MatchesFull` fails on the default configuration:
   prefilling `[1,2,3,5,7,11,13,17,19,23,29,31,37]` in one call vs three chunks of 5/5/3 gives a
   logit of **1.8636 vs 1.5929** — a 0.27 gap against the test's 2-decimal-place tolerance.

   Isolated to the int8 activation path by direct experiment: the same test **passes with
   `STINGRAY_CPU_PREFILL_Q8=0`** and fails with it on (the default). The mechanism is that Q8
   activation quantisation is computed per prefill call, so a 13-row batch and a 5-row batch derive
   different scales; the test's comment anticipated "different FP accumulation order" but the
   divergence is an order of magnitude larger than that would explain.

   **This is pre-existing, not a regression.** It surfaced only because `Tests.ForwardPass` had never
   been run to completion on this machine. Verified not to come from the 2026-08-08 session work:
   the only change to `ForwardPass.cs` is a read-only `MinRewindLength` property and
   `src/OpenTail.Stingray.Cpu/` is untouched.

   **Measured 2026-08-08 — the shipped default is CLEAN; only sub-256 chunks diverge.**
   SmolLM2-1.7B-Q4_K_M, 600-token prose prompt, full prefill vs chunked, comparing final logits
   (vocab 49,152). Each chunk size run with the int8 path on (default) and off, to separate the two
   mechanisms:

   | chunk | Q8 on: maxAbsDiff | Q8 off: maxAbsDiff | logits past 2dp (Q8 on) | argmax agrees |
   |---|---|---|---|---|
   | 5   | 0.9310 | 0.5037 | 97.93% | yes |
   | 64  | 0.9708 | 0.5440 | 97.71% | yes |
   | **256 (shipped default)** | **0.0000** | **0.0000** | **0.00%** | yes |

   **Two additive mechanisms, not one:**
   1. *Flash-64 threshold crossing* — present with Q8 OFF (~0.50), so it is not a quantisation
      effect. This is the **already-documented KNOWN RESIDUAL** in `ForwardPass.cs` (~line 2656):
      the flash-64-vs-incumbent decision is monotonic in sequence position, which is exact for any
      chunk ≥ 256, but a prompt whose total exceeds 256 while its chunks do not takes the incumbent
      early and flash-64 later. The comment's own example is "600 tokens at chunk 64" — which is
      literally what this measurement reproduces.
   2. *Q8 per-call activation scales* — the extra ~0.43 that appears when the int8 path is on. This
      is the mechanism behind the failing unit test, whose 13-token prompt at chunk 5 sits entirely
      below the flash-64 threshold and therefore cannot involve (1). Consistent with that test
      passing under `STINGRAY_CPU_PREFILL_Q8=0`.

   **Correction to the 2026-08-08 note initially filed here:** it claimed a prompt admitted under
   batching can produce different logits than the same prompt served single-shot "on the default
   configuration". That is **wrong**. At `STINGRAY_PREFILL_CHUNK=256`, the shipped default, chunked
   and single-shot prefill agree to 0.0000 across the whole vocabulary. The divergence requires a
   chunk size below 256, which no shipped configuration uses.

   **Argmax survived all six configurations on this prompt** — but that is one prompt, and with
   ~96–98% of logits past 2dp and maxAbs ~0.5–1.0, a temperature/top-p sampler would see a
   materially different distribution even where greedy does not move. Do not generalise "greedy
   agrees" into "sub-256 chunking is safe".

   **Standing status of the red test:** `PrefillWithCache_Chunked_MatchesFull` remains failing and
   is a legitimate red — it pins mechanism (2) at a chunk size no deployment uses. Fixing it means
   making Q8 activation scales chunk-invariant; suppressing it would discard the only automated
   detector for that mechanism.

Historical evidence: [done/cpu-prefill-plan-2026-07.md](done/cpu-prefill-plan-2026-07.md),
[done/cpu-prefill-repack-gemm-plan.md](done/cpu-prefill-repack-gemm-plan.md), and
[done/repack-gemm](done/repack-gemm).

**Current focused receipt (2026-08-07):** `MatMulBatchedQ8EquivalenceTests` passes **24/24** in
Release. It includes Q3_K's 4/8-input dispatch/remainder cases and Q2_K's 600-token production
fallback, so the remaining decision is release-quality/performance evidence, not those two
correctness seams.


### Mechanism (2026-08-08, follow-up) — the earlier "per-call activation scale" explanation is WRONG

That claim does not survive reading the code. Int8 activation quantisation is **per row**:
`DotQ6K` calls `QuantizeRowToQ8K(input, cols, scratch)`, with per-super-block scales inside a single
token's activation vector. Batch size cannot change the quantisation of a given row, so "a 13-row
batch and a 5-row batch derive different scales" is not the mechanism.

**Measured instead** (13 tokens, SmolLM2-1.7B-Q4_K_M, max abs logit difference):

| Comparison | maxDiff | What it isolates |
|---|---|---|
| `full[13]` vs `[5,5,3]` | 1.4335 | baseline divergence |
| `full[13]` vs `[12,1]` | 1.0651 | |
| `full[13]` vs `[6,6,1]` | 1.3838 | |
| `[12,1]` vs `[6,6,1]` | **1.0905** | **same last chunk (1 row), different history → still diverges** |
| `[5,5,3]` vs `[10,3]` | 1.4335 | `[10,3]` is numerically indistinguishable from `full[13]` |

The fourth row is the decisive one: two runs whose **final call is byte-identical in shape** still
differ, so this is not a property of the last batch. Earlier chunks compute different values, write
them to KV, and the difference propagates.

**Corrected two-mechanism model:**
1. **Flash-64 threshold** — the attention path is chosen on `startPos + N >= 256`. A 600-token prompt
   at chunk 256 crosses on its first chunk exactly as the single-shot call does, which is why those
   two agree to **0.0000**; at chunk 64 the early chunks take the incumbent and later ones flash-64,
   so they diverge. This mechanism **cannot apply to the 13-token test at all** — it never reaches 256.
2. **Batch-shape-dependent numerics in the incumbent int8 batched path** — the remaining, and for
   short prompts the only, mechanism. Q8-gated (the test passes with `STINGRAY_CPU_PREFILL_Q8=0`),
   and consistent with the 4/8-input dispatch and remainder handling that path documents. Different
   row counts select different microkernels with different accumulation order.

**Consequence for any fix:** making activation scales chunk-invariant would achieve nothing, because
they already are. A fix has to make the int8 batched path's *dispatch and accumulation order*
independent of batch shape — a substantially bigger change, and one that would cost throughput on
the very path measured at 3.48x. Not attempted here. The shipped default (chunk 256) is unaffected.
