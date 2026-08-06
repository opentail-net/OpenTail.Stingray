# Real AVX2 GEMM port — seam-by-seam

**Status: DONE — final verdict is a loss, honestly measured.** All 9 seams built, composed,
correctness-verified (<0.1% vs trusted), allocation-free. Single-threaded per-unit result was a
genuine ~1.65-1.7x win — the strongest per-unit number in this whole investigation — but it did
**not** survive scaling to real throughput. At the reference shape (2048×2048, batch=256), the
best-measured configuration (coarse-grained `Parallel.For` over row-groups, matching the shipped
path's own granularity) reached ~93% of the shipped `TryMatMulBatchedQ8`/`_8In` path's throughput
in the better of two runs — close, but not a win, and the first run of the same configuration was
noticeably worse (71%), showing real run-to-run variance at this scale. A naive flat-2D
`Parallel.For` (16,384 tiny tasks) was actively harmful — 0.13-0.28x of shipped throughput — from
thread-pool scheduling overhead, not the kernel itself; that finding on its own confirms §26's
older lesson (granularity matters more than kernel width) held again here, on a structurally
different kernel.

**This closes the "why is llama.cpp faster" investigation with an answer, not a workaround: the
gap is not (purely) a missing algorithmic technique.** This port used llama.cpp's actual
register-reuse trick, faithfully, seam-by-seam-verified, and still landed at best near parity
with a much simpler kernel this codebase already had — meaning the extra intricacy of the real
kernel's technique does not pay for itself in a managed-runtime (.NET JIT) implementation the way
it does in hand-tuned C++. Not shipping this — see "What would make this worth shipping" below,
which this port did not clear.
**Owner:** unassigned
**Last updated:** 2026-07-24

## Why this document exists

`docs/cpu-prefill-repack-gemm-plan.md` §13-§31 explored an extensive range of ideas on top of
this project's *own* AVX2 idiom for a repacked-weight GEMM/GEMV kernel — none of it closed the
gap to llama.cpp's actual performance, and a threading investigation (§25-28) ruled out
scheduling as the cause. Reading llama.cpp's real plain-AVX2 GEMM kernel
(`ggml_gemm_q4_K_8x8_q8_K`, `examples/cpp/llama.cpp/ggml/src/ggml-cpu/arch/x86/repack.cpp`,
lines ~2818-3157) directly confirmed why: it processes 4 tokens × 8 weight columns
simultaneously with weight-nibble and activation-byte reuse both happening in-register via
dense cross-lane shuffle/blend choreography — a materially different (and much larger) design
than anything attempted in this codebase's own idiom so far.

**This is a genuine, identified, real technique** — not a dead end. But the source is ~340
lines of interdependent shuffle/blend immediates where a single wrong constant produces
silently-wrong logits, not a crash. Given the stakes (real inference correctness), this is
being built and verified **one isolated seam at a time** against hand-computed scalar
references, rather than transcribed whole and trusted — same discipline that caught the
1717%-error indexing bug in `docs/cpu-prefill-repack-gemm-plan.md` §18, just applied
proactively instead of after the fact.

Each seam is a pure data transform with no dependency on the seams after it, so a mistake is
caught exactly where it was introduced, not somewhere downstream in a hard-to-localize way.

## File layout

- `src/OpenTail.Stingray.Cpu/RealAvx2Gemm.cs` — the port, one seam per method.
- `tests/OpenTail.Stingray.Tests.ForwardPass/RealAvx2GemmSeamTests.cs` — one test per seam, verified
  against an independently-computed scalar reference (not against other `RealAvx2Gemm` code).

**Rule: do not extend `RealAvx2Gemm.cs` without a matching seam test that exercises it against
a hand-computed reference, not another piece of this same port.**

## Seams — identified from reading the source, tracked here

| # | Seam | C source | Status |
|---|---|---|---|
| 1 | Column-pair rearrange: 8 columns → `[0,1,4,5]`/`[2,3,6,7]` groups | `rhs_raw_mat_0145_0`/`_2367_0` (`_mm256_blend_epi32` + `_mm256_permutevar8x32_epi32`) | **Done.** `RearrangeColumnPairs0145_2367`. Traced by hand (see code comment for the derivation), verified against a hand-built reference buffer with per-column recognizable byte values — passed on the first run. |
| 2 | 4-bit → 8-bit nibble unpack | `rhs_mat_0145_00`/`_10` etc. (`_mm256_and_si256` + `_mm256_srli_epi16`) | **Done.** `UnpackNibbles`. Same mask/shift pattern already used correctly elsewhere in this codebase; verified against a plain scalar per-byte reference — passed on the first run. |
| 3 | Weight-side "sp1/sp2" duplicate-shuffle (`_mm256_shuffle_epi32` imm 136/221) — extracts/duplicates 4-byte groups across lanes so a single load serves multiple `maddubs` lanes | `rhs_mat_0145_00_sp1`/`_sp2` etc. | **Done.** `DuplicateShuffleWeightPattern`. Traced imm 136→pattern[0,2,0,2], imm 221→pattern[1,3,1,3] (per 128-bit lane, via `vpshufd`/`Avx2.Shuffle`); verified against dwords read directly from the input buffer — passed on the first run. |
| 4 | Activation-side load + cross-128-lane duplicate (`_mm256_permute2f128_si256`) | `lhs_mat_01_00`/`_23_00` | **Done.** `BroadcastLowHigh128`. Traced `permute2f128(x,x,0)`→broadcast-low-128-to-both-halves, `permute2f128(x,x,17)`→broadcast-high; verified against the input buffer directly — passed on the first run. Confirms the seam captures register-level duplication only — still need to decide the activation source format (adapt to 4 separate per-token `QuantizeRowToQ8K` buffers vs. building `block_q8_Kx4`) before composition, not before this seam. |
| 5 | Activation-side sp1/sp2 duplicate-shuffle (imm 160/245) | `lhs_mat_01_00_sp1`/`_sp2` etc. | **Done.** `DuplicateShuffleActivationPattern`. Traced imm 160→pattern[0,0,2,2], imm 245→pattern[1,1,3,3] — different from seam 3's [0,2,0,2]/[1,3,1,3] despite an identical input dword layout, because the target duplication differs (weight alternates columns; activation duplicates one token's own bytes). Verified against dwords read directly from the input buffer — passed on the first run. |
| 6 | `maddubs_epi16` dot + `add_epi16` combine (weight × activation, both shuffle patterns) | `iacc_mat_00_0_sp1` etc. | **Done.** `MaddubsAccumulate4`. Layout-agnostic: 4 `maddubs_epi16` (u8×s8→i16 pairs) combined via the same nested-`add_epi16` order as the source (not reassociated). First seam with real arithmetic — verified against a scalar reference summing 8 individual byte products per int16 lane (modest byte magnitudes to stay overflow-safe), not just a byte-position check. Passed on the first run. |
| 7 | `madd_epi16` with per-column scales | `iacc_mat_00_0 = madd_epi16(iacc_mat_00_0, scale_0145_0)` etc. | **Done.** `ScaleAndReduce0145`/`ScaleAndReduce2367`. Traced `vpshufd` imm 68→per-lane dword pattern [0,1,0,1], imm 238→[2,3,2,3] (source lines 2983-2984), applied to the already-built-and-tested `utmp` scale vector (reused from `RepackedGemm`, not re-derived), then `_mm256_madd_epi16` → `Avx2.MultiplyAddAdjacent(Vector256<short>,Vector256<short>)` returning `Vector256<int>` directly. Verified against a fully independent scalar reference (hand-shuffled dword pattern + scalar adjacent-pair dot products) — passed on the first run. |
| 8 | "Straighten to 4 row vectors" blend/shuffle | `iacc_row_0`..`_3` (`_mm256_blend_epi32` + `_mm256_shuffle_epi32` imm 78/204) | **Done.** `StraightenToRowVectors`. Traced `vpshufd` imm 78→per-lane dword pattern [2,3,0,1] (swaps low/high dword pairs within each 128-bit lane), `blend_epi32` imm 204→dwords {2,3,6,7} from the second operand, {0,1,4,5} from the first; untangles seam 7's row-pair-interleaved accumulators into one vector per row. Verified against a fully independent scalar swap+blend on plain int arrays — passed on the first run. Also noted (not yet built): a stray `ContinuousBatchingConstraintTests` failure appeared in two of the last three full-suite runs and passes in isolation — confirmed flaky/pre-existing, unrelated to this port (touches grammar/batching masking, not GEMM). |
| 9 | Final FP scale + bsums-based min correction, output store | `acc_rows[rp*4+i]`, `acc_min_rows[...]` | **Done.** `BsumsMinCorrection` + `ScaleAndAccumulateRow`. Two functions: (a) the bsums-hsum broadcast (`vpshufd` imm 0/85/170/255, each broadcasting one token's dword-pair across the vector) + `madd_epi16` combine; (b) the row-scale broadcast (`vshufps`, same four immediates, float lanes) + two `fmadd` accumulations (scale term, min term) + final subtract. Shuffle immediates had to become a compile-time `switch` over a `tokenIndex` int rather than a runtime byte parameter — CA1857 flagged the original signature (JIT can't emit `vpshufd`/`vshufps` with a non-constant immediate). Verified via `[Theory]` tests over all 4 token indices: (a) against a scalar broadcast + adjacent-pair dot product; (b) using exactly-representable-in-float32 integers throughout so the scalar reference doesn't need FMA-vs-separate-rounding tolerance. Passed on the first run after the compile-time-constant fix. |

## Composition — in progress, one real subtlety found before writing code

Started composing seams 1-9 into the full kernel using this codebase's existing activation
convention (4 independent `SimdKernels.QuantizeRowToQ8K` buffers, per seam 4's noted decision —
confirmed: seams 4-6 are genuinely layout-agnostic, so a token pair can be built in-register from
two separate 8-byte loads instead of one interleaved `block_q8_Kx4` load, with no change to any
seam).

**Found before writing the composed loop, not after:** `RepackedGemm.RepackQ4K8Rows` stores
`d[]`/`dmin[]` in **natural column order** (0..7) — already validated bit-exact against the
scalar dequantizer (checkpoint 1) and consumed that way by every existing "old idiom" kernel
(`GemmQ4K8x8x4Q8K_Avx2` etc., which visit columns serially). The real kernel's register pipeline
is different: seam 1 rearranges qs bytes into two column-groups (`[0,1,4,5]` / `[2,3,6,7]`), and
seams 3/5/6/7/8 each fold/shuffle/`maddubs` those bytes — `maddubs_epi16` in particular merges
*adjacent byte pairs*, so a lane in `iacc_row_i` (seam 8's output) is not a simple pass-through of
"column N" the way it was before seam 6. `col_scale_f32`/`col_dmin_f32` (seam 9) and the final
output store must index whichever lane order that chain actually produces — assuming it's
`[0,1,4,5,2,3,6,7]` or matches on-disk natural order without checking would be exactly the kind
of silent-wrong-logits mistake this whole seam discipline exists to prevent.

**Resolved — empirically, not just by algebra.** Hand-tracing seams 1→2→3→6→7→8's shuffle/blend
immediates predicted natural column order at the output with no extra permutation needed. Rather
than trust that chain of reasoning alone, built a composition-level proof test
(`ComposedSeams1Through8_ProduceNaturalColumnOrderPerToken` in `RealAvx2GemmSeamTests.cs`):
column *c*'s weight nibble = *c*+1 uniformly, token *t*'s activation byte = *t*+1 uniformly, unit
scales, run through the actual seam functions end-to-end (not re-derived scalar code), checked
against a first-principles expected value (`32 × columnValue × tokenValue`, since 32
elements/column/sub-block each contribute uniformly) computed independently of any
`RealAvx2Gemm` code. **Passed on the first run.** Confirms: `col_scale_f32`/`col_dmin_f32` load
directly in natural order (matching `RepackedGemm.RepackQ4K8Rows`'s existing natural-order `d[]`/
`dmin[]` storage, already validated at checkpoint 1), and the final output store needs no
permutation — `row0..row3` from seam 8 are already each token's natural-order 8-column result.

## Composition complete — correctness verified, allocation-free, first perf number is a real win

Composed all 9 seams into `RealAvx2Gemm.GemmQ4K8x8x4Q8K_RealAvx2` (matches
`RepackedGemm.GemmQ4K8x8x4Q8K_Avx2`'s signature/activation-buffer layout exactly — 4 independent
`QuantizeRowToQ8K` scratch buffers, per seam 4's decision). One real bug found and fixed:
`BuildActLhsPair` was missing the `chunk*64` base offset into the activation buffer, so every
chunk after 0 silently read chunk 0's data — caught immediately by the correctness test
(121% relative error), not shipped.

**Correctness:** `RealAvx2Gemm_MatchesTrustedGemmQ4K8x8x4Q8K_WithinTolerance` in
`RepackedGemmQ4KRoundTripTests.cs` — <0.1% relative error vs the trusted kernel on random
Q4_K-shaped data. Passing.

**Allocation-free:** the initial composition allocated `new Vector256<T>[4]` arrays and local-
function closures per chunk per super-block (O(blocksPerRow × 4) heap garbage) — refactored to
unrolled named locals (kk0..kk3, no arrays) and a plain static `BuildActLhsPair` helper (no
closure). Re-verified correctness held after the refactor.

**Single-threaded per-unit result (real, strong, but not the final answer):**
`PerfGauge_RealAvx2Gemm_VsTrusted_SingleThreaded` (2048-element rows, 2000 iters, 20 warmup, run
twice): the composed real-AVX2 kernel is **~1.65-1.7x faster per-unit** than
`GemmQ4K8x8x4Q8K_Avx2` (23.5µs/call vs 39.5µs/call, run1 ratio 0.587x, run2 ratio 0.607x — stable
across runs). This is the first genuinely strong per-unit result across the entire investigation
(`cpu-prefill-repack-gemm-plan.md` §13-§31's attempts all showed weaker or no per-unit gains).

**Scaled measurement (the one that actually decides it):**
`PerfGauge_RealAvx2Gemm_VsShipped_ParallelForScaled` (2048×2048/batch=256 reference shape, 10
warmup, 2 timed runs), against `SimdKernels.TryMatMulBatchedQ8` (the shipped `_8In` path, same
underlying weight bytes and comparably-quantized activations — a like-for-like throughput
comparison, not the earlier correctness comparison which intentionally used matching Q8_K
conventions):

| Config | run1 | run2 |
|---|---|---|
| Shipped `TryMatMulBatchedQ8` | 14,653-16,743 tok/s | 15,414-15,617 tok/s |
| Real-AVX2, flat-2D `Parallel.For` (rowGroup×tokenGroup, 16,384 tasks) | 0.126-0.639x shipped | 0.275-0.734x shipped |
| Real-AVX2, coarse `Parallel.For` (rowGroups only, 256 tasks) | 0.710x shipped | **0.928x shipped** |

The flat-2D over-decomposition is actively harmful (thread-pool scheduling overhead dominates at
16k+ tiny tasks) — confirms §26's granularity lesson held again on this structurally different
kernel. The coarse variant, matching the shipped path's own row-outer granularity, gets close —
93% of shipped throughput in the better run — but never exceeds it, and shows real run-to-run
variance (71% → 93% across two runs of the identical configuration). **Verdict: loss.** Real
per-unit wins do not survive scaling, exactly the pattern of every prior attempt in this
investigation (§18/§23/§24) — this port is the closest any attempt has gotten, but "closest" is
not "beats."

## PersistentThreadPool experiment — modest, noisy improvement; doesn't change the verdict

After the loss above, investigated whether `Parallel.For`'s dispatch overhead specifically (not
the kernel) was the remaining gap, since OpenBLAS (already vendored for reference at
`examples/cpp/OpenBLAS`, BSD-3-Clause) and ggml both avoid `Parallel.For`/`ThreadPool`-style
dispatch entirely via a persistent-worker design: threads spawned once, blocking on a real OS
wait between calls, no per-call thread-pool queue. Read OpenBLAS's actual Windows implementation
directly (`driver/others/blas_server_win32.c`) rather than working from memory — confirmed it's
persistent `CreateThread` workers + a critical-section-protected queue + Win32 events, and
confirmed `exec_blas_async`/`blas_thread_init` are *not* `OPENBLAS_EXPORT`-marked, i.e. not part
of the public DLL surface, so there is no way to hand our kernel to OpenBLAS's own pool via
P/Invoke — any persistent-pool benefit has to come from a small hand-rolled equivalent
(`src/OpenTail.Stingray.Cpu/PersistentThreadPool.cs`).

Three designs tried, in order:

1. **Spin-wait between calls** (busy loop, no real block) — roughly tied with or worse than plain
   `Parallel.For`. Theory: idle spinning workers still burn a full core each, self-competing with
   the active ones on this shared box.
2. **Real blocking wait** (`AutoResetEvent`, matching OpenBLAS's actual `WaitForMultipleObjects`
   design) + static partition — the best-performing design measured. Across 4 valid runs (a 5th
   run's shipped baseline was itself anomalously slow — ~3,500 tok/s vs. the usual ~15-20k — and
   excluded as noise, not signal): mean ≈0.71x of shipped throughput vs. plain coarse
   `Parallel.For`'s ≈0.64x, beating it in 5 of 8 paired runs, and touching **100.7% of shipped
   throughput** in one run — the only time anything in this investigation crossed parity.
3. **Dynamic chunk-claiming + caller-thread participation** (per a second-opinion review that
   proposed this as more sophisticated/robust — worker threads atomically claim ranges from one
   shared `Interlocked.Add` cursor instead of a static up-front split, closer to how a
   general-purpose scheduler would handle uneven work) — tried at two grain sizes (2 and 16
   row-groups per claim). **Measured consistently worse** than design 2: mean ≈0.56-0.73x, never
   beating plain coarse `Parallel.For` in 6 of 6 paired comparisons at grain=16. Read: the 256
   row-groups here are near-identical in cost, so there's no load-balancing gain to offset the
   real cost of every worker thread hitting one shared, contended cache line. Design 2's static
   per-worker partition, where no field is shared across workers at dispatch time, wins for this
   specific near-uniform workload — dynamic claiming is solving a problem (load imbalance) this
   workload doesn't have, while paying its overhead anyway. Reverted to design 2.

**This machine is too noisy for fine-grained verdicts.** The shipped baseline alone swung from
~3,500 to ~19,900 tok/s across otherwise-identical repeated runs (up to ~6x) — background load on
a shared dev box, not a code effect. Differences smaller than roughly 2x between configurations
measured here should not be trusted as signal.

**Does this change the loss verdict above? No.** Design 2 is a real, if noisy and unreliable,
improvement over `Parallel.For` — but "sometimes touches parity, averages ~0.71x, never reliably
wins" is not a win. The persistent-pool avenue was worth ruling out explicitly (it's the actual
mechanism OpenBLAS/ggml use, so its absence was a legitimate open question), but it does not
change this port's overall conclusion: not shipping it.
2. Correctness: tolerance-based (<0.1%, matching the existing scalar-vs-AVX2 cross-check
   convention in `docs/cpu-prefill-repack-gemm-plan.md`) against `GemmQ4K8x8x4Q8K_Avx2` (the
   already-trusted from-scratch AVX2 kernel) — same 4-token×8-column shape, so a direct,
   apples-to-apples numerical check.
3. Performance: same discipline as everything in `cpu-prefill-repack-gemm-plan.md` — single-
   threaded per-unit first, then scaled with `Parallel.For` (flat 2D granularity, §26 found
   that mattered more than kernel width), then the real CLI benchmark with ≥10 warmup calls
   (§29's lesson) before trusting any number, run at least twice.
4. Full `Tests.ForwardPass` suite must stay green throughout.

## What would make this worth shipping

Per `cpu-prefill-repack-gemm-plan.md` §18/§23/§24, every prior attempt at a repacked-weight
kernel — including ones with real, measured per-unit wins — lost to the already-shipped
`_4In`/`_8In` path once scaled and parallelized. This port is only worth wiring into
`ForwardPass`/`MatMulBatched` if it *beats* that shipped path's steady-state throughput
(≈17,000-20,000 tok/s on the 2048×2048/batch=256 reference shape, per §25/§29) at real batch
scale — not just "beats our own earlier repacked-GEMM attempts," which is a much lower bar
this document's whole prior exploration already cleared without mattering in the end.
