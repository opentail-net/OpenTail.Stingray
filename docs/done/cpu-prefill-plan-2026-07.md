# CPU prefill performance — plan

**Status:** the `_4In` int8 batched-prefill path (§12) is implemented, gated off by default,
and measured at ~1.3× on top of runtime tuning — gap to llama.cpp was **~4.6×**. Extended to
`DotQ4K_Q8KS_8In` (8-token register reuse, docs/cpu-prefill-repack-gemm-plan.md §19) plus
parallel activation quantization — re-measured gap **~3.7-4.3×** (real, reproduced across two
runs after a noisy first run was ruled out), inside the 4-6× ceiling this phase targeted.
`_8In` subsequently extended to `DotQ6K_Q8K_8In` and `DotQ3K_Q8KS_8In` (§30-§31 of the repack
plan) — same proven widening pattern, same-family byte-exact correctness, gap stable at
~3.9-4.1×; SmolLM2 has no Q3_K tensors so that extension's win isn't visible on this specific
model, but it's real for any model/MoE routing that does hit Q3_K.
Quality verification done at both the greedy-token level
(§13) and now corpus-level perplexity (§14, prefill-routed tool built and run: gate on shows
a −0.4% perplexity delta, noise-level) — still only one model/one corpus, so default-on is not
yet decided. One earlier kernel-level attempt was made and
reverted (see [Attempt 1](#attempt-1--failed-measured-reverted)); its cause was never fully
identified despite direct profiling (§11), but is superseded — the `_4In` design that shipped
is structurally different and unaffected by whatever attempt 1's issue was. Runtime-config
tuning (§9) landed a real ~25-30% win with zero kernel risk, prior to the `_4In` work.
**Owner:** unassigned
**Last updated:** 2026-07-24

Prefill on the CPU backend runs at roughly decode speed when it should be several times
faster. This document records the measurements, one failed attempt, an external design
review, the revised plan, and a runtime-tuning pass that improved things without touching
kernel numerics.

---

## 1. The problem

`SimdKernels.MatMulBatched` (`src/OpenTail.Stingray.Cpu/SimdKernels.cs`) has no batched path when
OpenBLAS is absent. It degenerates to:

```csharp
for (int n = 0; n < batchSize; n++)
    MatVec(output + n * rows, weights, input + n * cols, rows, cols, dtype);
```

Each `MatVec` runs its own `Parallel.For` over all rows, so an N-token prefill streams the
entire weight matrix **N times**.

The OpenBLAS path is not the answer either: it dequantizes the whole weight matrix to F32
into a thread-static buffer (an 8× blow-up for Q4_K) and then SGEMMs. It also reintroduces a
native dependency the project deliberately avoids.

---

## 2. Measurements

All numbers: **Ryzen 7 5700G** (Zen 3, 8c/16t, 512 KB L2/core, 16 MB L3, dual-channel DDR4,
**no AVX-512**), CPU-only, no OpenBLAS, `SmolLM2-1.7B-Instruct-Q4_K_M` (1005 MiB).

### OpenTail.Stingray baseline

Via `opentail-llm-cli -f <prompt> --temp 0 -n 1`, two runs each:

| Prompt | Prefill |
|---|---|
| 87 tok | 24.4 / 25.2 t/s |
| 261 tok | 31.2 / 31.3 t/s |
| 903 tok | 32.5 / 31.8 t/s |

### llama.cpp b8585 reference — same machine, same model file

Via `tools/llama.cpp/llama-bench.exe` (fetched by `scripts/setup-llamacpp.ps1 -Variant cpu`):

| Config | Result |
|---|---|
| `-t 16 -p 87` | 193.36 ± 4.48 t/s |
| `-t 16 -p 261` | 197.55 ± 1.60 t/s |
| `-t 16 -p 903` | 185.07 ± 2.32 t/s |
| `-t 16 -n 64` (decode) | 28.70 ± 1.08 t/s |
| `-t 6 -p 512` (its default threads) | 154.10 ± 0.80 t/s |
| `-t 6 -n 128` (decode) | 34.17 ± 0.55 t/s |

### The gap

| Prompt | OpenTail | llama.cpp | Gap |
|---|---|---|---|
| 87 tok | 24.4 | 193.4 | **7.9×** |
| 261 tok | 31.2 | 197.6 | **6.3×** |
| 903 tok | 32.5 | 185.1 | **5.7×** |
| decode | ~24 | 28.7 | 1.2× |

**Conclusions from the data:**

- **Decode is competitive** (within ~20%). The SIMD dot kernels are fine; this is not a
  general "the engine is slow" problem. Prefill is the entire gap.
- The real gap is **~6×**, not the 10× previously quoted from README numbers measured on
  different (Zen 4 + OpenBLAS) hardware. Target accordingly.
- **Flat-with-length is not itself the pathology.** llama.cpp's prefill is also flat
  (193 → 198 → 185). It just plateaus ~6× higher. Both engines are bandwidth-limited; the
  difference is how much weight traffic each generates per token.
- llama.cpp reaches 154 t/s on **6 threads** vs 185 on 16 — ~80% of peak at ~⅜ the threads,
  i.e. genuinely compute-efficient, not merely parallel. OpenTail uses all 16 for 32 t/s.

**Fairness caveat:** `llama-bench` loads `ggml-cpu-haswell.dll`, i.e. an AVX2 path — the same
instruction-set tier as OpenTail's AVX2 kernels on this Zen 3 chip. Worth confirming OpenTail
isn't silently falling back to a scalar path for some dtype, which would change the diagnosis.

---

## 3. Attempt 1 — failed, measured, reverted

**What:** the obvious loop reorder. Rows outermost, tokens tiled 32-inner, so each weight row
is fetched once and reused across the tile. Identical `Dot*` calls, byte-exact, loop nest only.

**Result: ~4× slower.**

| Prompt | Baseline | Attempt 1 |
|---|---|---|
| 87 tok | 24.4 / 25.2 | 8.6 / 7.7 |
| 261 tok | 31.2 / 31.3 | 7.9 / 8.6 |
| 903 tok | 32.5 / 31.8 | 8.4 / 8.1 |

Reverted. A comment at the call site records the measurement.

**My post-hoc explanation** (input locality: the per-token loop keeps one F32 activation
vector pinned in L1 and walks weights as one prefetcher-friendly sequential stream; tiling
replaced that with a ~256 KB F32 activation working set re-read per row).

**External review disagreed**, and this is unresolved:

> "the proposed explanation for the failed loop reorder is too confident. F32 activation
> locality likely contributed, but it does not by itself explain a sustained 4× regression:
> the total activation reads are broadly similar, and a 32-token tile is L2-sized on this CPU."

Other candidates not yet ruled out — **output-store layout** (writes strided by `rows`, so
threads on adjacent rows contend for the same cache lines → false sharing), changed
**parallel scheduling**, and **intrinsic codegen** differences. Treat attempt 1 as evidence
the implementation was bad at machine level, *not* as proof of a single cause.

---

## 4. Key discovery — the batched primitives already exist

`SimdKernels` already contains an int8 activation-quantization path with **multi-input** dot
kernels, unused by `MatMulBatched`:

| Family | Kernels |
|---|---|
| Q4_K | `DotQ4K_Q8KS`, `DotQ4K_Q8KS_2In`, `DotQ4K_Q8KS_4In` |
| Q6_K | `DotQ6K_Q8K`, `DotQ6K_Q8K_2In`, `DotQ6K_Q8K_4In` |
| Q3_K | `DotQ3K_Q8KS`, `DotQ3K_Q8KS_2In`, `DotQ3K_Q8KS_4In` |
| Q8_0 | `DotQ8_0_Q8K`, `DotQ8_0_Q8KS` |
| helpers | `Q8KScratchBytes`, `QuantizeRowToQ8K`, `QuantizeRowToQ8KS` |

`_4In` reads **one weight row and dots it against four pre-quantized activations** — literally
the batch dimension. Already used by `CudaHybridForwardPass` fused-FFN paths.

Why this differs from attempt 1: weight reuse happens at **register level inside one kernel
call**, not across a cache-resident tile; and Q8 activations are ~4× smaller than F32, so a
4-token working set stays small.

> ⚠️ **`Q8K` and `Q8KS` are different scratch formats and are NOT interchangeable.** Q4_K uses
> the `Q8KS` family; Q6_K/Q8_0 use `Q8K`. Routing a dtype through the wrong quantizer produces
> silently wrong output.

---

## 5. External review findings (codex / gpt-5.6-terra)

| # | Severity | Finding |
|---|---|---|
| 1 | medium | `_4In` has a **hard 4× weight-reuse ceiling**; it cannot alone deliver 10–20×. Set a realistic **4–6×** target and measure residual bottlenecks. |
| 2 | high | **Do not** replace the byte-exact `MatMulBatched == N × MatVec` oracle with a tolerance test. Preserve it for the default path; *add* a separate bit-exact batched-vs-per-token Q8 contract. |
| 3 | high | Dispatch must explicitly map each dtype → its quantizer → its dot family. Q8K/Q8KS are not interchangeable. |
| 4 | medium | Use a **batch/work threshold**, not `batchSize > 1`. Batch 2–3 and small projections can regress from up-front quantization plus row-stationary traversal. |
| 5 | high | Treat Q8 prefill as **explicit opt-in, default off**, until CPU decode routing, prefix-cache and speculative assumptions, and end-to-end quality are verified. The existing routed-expert env gates do not authorize trunk-wide divergence. |

Two further corrections:

- **Do not add an outer 16/32-token tile** around `_4In`. Calling `_4In` four times for a
  16-token tile still reloads the weights four times — no extra reuse, and it risks repeating
  attempt 1's behaviour. A genuine 8/16-input kernel is a separate, later experiment needing a
  register-pressure/codegen benchmark; four may be the deliberate sweet spot.
- On OpenBLAS: the honest argument is "wrong dependency for this project and poor for this
  memory budget", **not** "cannot be faster". A tuned SGEMM gains far more reuse than four
  inputs at large batch.

On divergence, the review clarified that a cached prefix is internally consistent (it holds
the KV states prefill actually computed). The real risks are a **different greedy token near a
logit tie** and **altered speculative acceptance**, plus tests that require batched execution
to match sequential.

---

## 6. Revised plan

1. **Profile attempt 1 before building on its explanation.** Specifically test the
   false-sharing hypothesis on the strided output writes — now a stronger candidate than the
   input-locality story currently written into the source comment. Until this is understood,
   any new implementation risks the same unexamined failure.
2. **Verify whether CPU decode already uses the Q8 dots.** Cheap, and it gates everything
   downstream: if decode already runs int8, the prefill/decode divergence concern largely
   evaporates. (Evidence so far is from `CudaHybridForwardPass`, which does not answer it.)
3. **Implement the `_4In` path** with an explicit per-dtype quantizer→dot mapping, hard-fail
   or fall back on unmapped dtypes (Q5_K, Q2_K, F32), and a threshold of `batchSize >= 4`
   plus a matrix-work estimate.
4. **Keep the byte-exact oracle**; add the Q8-vs-Q8 exact contract alongside it.
5. **Ship behind a default-off gate**; target **4–6×**, not 10×.

---

## 7. Verification

- **`MatMulBatchedEquivalenceTests`** (`tests/OpenTail.Stingray.Tests.ForwardPass/`) — 25 tests,
  currently green. Asserts `MatMulBatched(N)` is **byte-identical** to N × `MatVec` across
  Q4_K/Q6_K/Q8_0/F32, batch sizes straddling `MinBatchForBlas`, and shapes crossing K-block
  and row-tile boundaries, plus a batch-independence check. Built *before* attempt 1; it is
  what any future attempt must satisfy. Keep it for the default path.
- **New Q8 contract:** batched Q8 output must be bit-identical to N × per-token Q8 dot.
- **Accuracy:** measure error vs the F32 reference with per-dtype bounds; use the
  `perplexity` CLI command (the existing accuracy gate for KV-compression work) before/after.
- **End-to-end:** greedy generation on a fixed prompt stays coherent; ideally identical tokens.
- **Performance:** re-run the three prompts above, two runs each, *and* `llama-bench` on the
  same model, so every attempt is scored against a real external reference rather than its own
  previous number. Consider scripting this as `scripts/bench-vs-llamacpp.ps1`.
- **Regression:** full `Tests.ForwardPass` suite green (note: ~16 pre-existing Vulkan/GPU
  failures on a device-less box, plus one timing-sensitive continuous-batching test that
  passes in isolation).

---

## 8. Open questions

1. ~~What actually caused attempt 1's 4× regression?~~ **Partially answered, see §11 — NOT the traversal pattern itself; likely per-element dispatch overhead, still to verify.**
2. ~~Does CPU decode already route through the Q8 dots?~~ **Answered, see §10 — no.**
3. Is OpenTail on the AVX2 path for every dtype involved, or silently scalar somewhere?
4. Is a genuine 8- or 16-input kernel worth prototyping after `_4In` lands, or does register
   pressure make four the ceiling on this microarchitecture?

---

## 9. Runtime-config tuning — landed

Before returning to kernel work, a round of small, reversible csproj/env changes was tried —
same "small change → run → compare" discipline as the kernel attempt, but against pure .NET
runtime knobs first, since they're zero risk to correctness (no kernel numerics touched, so
`MatMulBatchedEquivalenceTests` can't even see these changes) and cheap to try.

### Bug found in the bench script itself

`bench-vs-llamacpp.ps1`'s `Invoke-OpenTailCli` hard-coded `-n 1` for every run regardless of
`-DecodeTokens`, so "decode t/s" was dominated by process-startup/JIT-tiering noise, not
steady state. Measured directly: `-n 1` reads ~24-25 t/s and noisy; `-n 64`/`-n 128` back to
back on identical input reads ~28-30 t/s, consistently. **This alone likely explains** why the
pristine `_dome` snapshot appeared faster than current OpenTail.Stingray in the very first 3-way
run — a `-Runs 1` pass with a broken decode measurement, not a real regression. Fixed: the
script now threads `-DecodeN $DecodeTokens` through properly. Re-verify the dome comparison
with the fixed script (`-Runs 3`, no `-SkipDome`) before concluding anything about regression.

### Experiments (Ryzen 7 5700G, 8c/16t, same model/prompts as §2)

| # | Change | Prefill (87 / 261 / 903 tok) | Decode | Verdict |
|---|---|---|---|---|
| 0 | Baseline (post bench-script fix) | 21.8 / 24.1 / 27.1 | 21.6 | — |
| 1 | `ServerGarbageCollection=true`, `ConcurrentGarbageCollection=false` in `OpenTail.Stingray.Cli.csproj` | 21.9 / 25.7 / 28.3 | 21.6 | **Kept** — small, consistent win, no regression |
| 2 | + `TieredCompilationQuickJitForLoops=false` | 25.7 / 27.1 / 29.2 | 19.9 → confirmed 27-28 at steady state (150 tok) | **Kept** — the 64-tok decode dip was startup noise in a short window, not real |
| 3 | `STINGRAY_CPU_THREADS` sweep: 16/12/8/6 | all land in same ~27-32 t/s band | ~22-24 across all | **No change** — thread count is not a lever here; contention/scheduling was not the bottleneck. Left at default (`Environment.ProcessorCount`) |

**Net result, both changes kept:** prefill **29.7 / 31.8 / 29.2 t/s**, roughly **+25-30%**
over the pre-tuning baseline, decode ~22.4 t/s (flat to slightly up). Verified: full solution
builds 0 warnings/errors; `MatMulBatchedEquivalenceTests` still 25/25 byte-exact (expected —
these are runtime settings, not kernel changes).

Both changes are in `src/OpenTail.Stingray.Cli/OpenTail.Stingray.Cli.csproj` only — they don't touch the
library projects, so anything embedding `OpenTail.Stingray.Engine`/`Cpu` directly (the server host,
for instance) does not get them for free. Worth deciding whether to propagate
`ServerGarbageCollection`/`TieredCompilationQuickJitForLoops` to `OpenTail.Stingray.Server.Host.csproj`
too, since the server is presumably also throughput-sensitive and long-running (which cuts the
other way on `TieredCompilationQuickJitForLoops`: a long-lived server process tiers up
naturally over time, so the "skip OSR" argument that helps a short CLI invocation is weaker
there — measure separately before copying the setting over).

**This does not change the diagnosis or the plan.** ~30 t/s is still ~5.5-6x behind
llama.cpp's ~170 t/s reference (§2) — squarely in the 4-6x-gap territory the external review
predicted for the *kernel* fix, meaning runtime tuning alone will not close it. The `_4In`
work in §6 is still the path to the next real jump; this section is upstream low-hanging fruit
that makes the eventual kernel win start from a higher baseline, not a substitute for it.

---

## 10. Step 2 answered — dense CPU decode does NOT use the Q8 dots

Checked all three call sites of the `_Q8K`/`_Q8KS` family (`DotQ4K_Q8KS`, `DotQ6K_Q8K`,
`DotQ3K_Q8KS`, `DotQ8_0_Q8K`, and their `_2In`/`_4In` variants). Every single one lives in
`CudaHybridForwardPass.cs`, `CudaHybridGdnForwardPass.cs`, or `HybridGdnForwardPass.cs` — the
**MoE / hybrid-GatedDeltaNet** forward-pass classes used for Qwen3.5-style architectures'
expert-layer down-projections.

`src/OpenTail.Stingray.Engine/ForwardPass.cs` — the **plain dense forward pass**, which is what
`RunCommand.cs` instantiates for an ordinary llama-arch model (confirmed: `SmolLM2` is plain
`llama` arch, not MoE/hybrid-GDN, so it runs through `ForwardPass`, not `HybridGdnForwardPass`)
— has **zero** references to `Q8K`/`Q8_K` anywhere. Grep count: 0.

**This means the divergence risk from finding #5 does NOT evaporate — it's confirmed live.**
For any dense model (which includes the `SmolLM2` fixture this whole investigation has been
measuring against), decode runs pure F32 dot kernels today. Adding `_4In` Q8 batching to
`MatMulBatched`'s prefill path would introduce a **first-time** numerical difference between
prefill and decode for dense models specifically — today they agree exactly (both F32), so
there is no existing precedent to lean on for dense architectures the way there might be for
MoE ones. This raises rather than lowers the bar for finding #5's default-off gate:

- The gate must be default-off with no exceptions for dense models.
- The Q8-vs-Q8 exact contract (§7) needs a dense-model end-to-end test, not just the MoE path
  the existing `_Q8K`/`_Q8KS` call sites already exercise — those don't validate this new
  dense-prefill use case at all.
- Perplexity/greedy-token comparison (§7) should run on a plain dense arch (SmolLM2 or similar)
  specifically, since that's the case with zero prior int8 usage to fall back on.

Step 1 (profiling attempt 1's real failure cause) remains the next step — it still blocks any
new implementation that reuses row-stationary traversal, and `_4In`'s per-row-call structure
means it's less exposed to that risk than attempt 1 was, but "less exposed" isn't "verified."

---

## 11. Step 2 answered — attempt 1's traversal pattern is NOT the cause

Built an isolated microbenchmark (`scratchpad/prefill-profile/`, not part of the repo — a
throwaway console app referencing `OpenTail.Stingray.Cpu` directly) to test the row-outer/tiled
traversal against the exact same production `SimdKernels.DotQ4K` kernel, without any
CLI/model-loading/tokenization noise. Realistic shape (8192×2048, Q4_K), batch 32, tile 32,
30 timed iterations after 10 warmup, `Environment.ProcessorCount` threads.

Two configurations tested:

1. **Warm cache** — one 2.3 MB matrix reused across all timed iterations (fits comfortably
   in the 16 MB L3).
2. **Cold cache** — 16 separately-allocated ~2.3 MB matrices (>16 MB total, exceeding L3),
   cycled by iteration index, so each timed call reads memory the previous call couldn't have
   left resident. This matters: real prefill never re-reads one layer's weight matrix within
   a run (1 GB total weights, no cross-layer cache reuse), so the naive warm-cache version
   risked testing a regime the real workload doesn't have.

**Result — both configurations agree, and neither reproduces the regression:**

| Variant | Warm-cache ratio vs A | Cold-cache ratio vs A |
|---|---|---|
| B: row-outer tiled, direct write (attempt 1 exact) | 1.02× | 1.03× |
| C: row-outer tiled, thread-local scratch + bulk copy | 0.89× (faster) | 0.96× (faster) |
| B-serial (no threading at all) vs A-serial | 0.96× | 1.03× |

All three variants produce byte-identical output to the per-token baseline in both runs
(sanity-checked). **Row-outer/token-tiled traversal is statistically equivalent to — if
anything marginally faster than — the per-token loop on this hardware**, whether the cache is
warm or cold, threaded or not. This holds even for the exact direct-write pattern (variant B)
that was hypothesized to suffer from false sharing.

**Conclusion: the loop-order/access-pattern itself is ruled out as the cause of attempt 1's
measured ~4× regression.** Neither of the two hypotheses on the table (input locality from my
original explanation, output false-sharing from the external review) reproduces in isolation.
Something about the *specific reverted implementation*, not the traversal pattern in general,
caused the real-world slowdown — most likely one of:

- **Per-element dispatch overhead.** The actual reverted code evaluated a `dtype switch`
  once per (row, token) pair inside the hot loop (262144 evaluations per tile at this shape),
  rather than resolving the dtype once per call outside the loop. This microbenchmark has no
  switch at all (single dtype, direct `DotQ4K` call), so it cannot rule this in *or* out — it
  only rules out the traversal pattern itself. Worth testing explicitly before the `_4In` work:
  add a switch-per-element variant to this harness and compare.
- **Something specific to the full pipeline** (JIT codegen differences between an isolated
  exe and the real CLI assembly, GC behaviour under the full call graph, interaction with
  `ForwardPass`'s surrounding allocations) not reproducible in a standalone microbenchmark.

**Actionable takeaway for step 3:** this is good news, not a dead end. It removes the biggest
reason to be wary of row-outer traversal for the `_4In` implementation — the pattern itself is
fine on this CPU. The practical implication is to make sure the `_4In` implementation resolves
its per-dtype quantizer→dot mapping **once per `MatMulBatched` call, outside the hot loop**
(a `switch` on `dtype` immediately at the top of the function, producing a function pointer or
enum tag used inside the loop), not per-element as attempt 1 apparently did. That was already
the plan's intent (§6 step 3: "explicit per-dtype quantizer→dot mapping"), and this finding is
a concrete reason to hold to it strictly rather than take a shortcut.

Scratch harness kept at `scratchpad/prefill-profile/` (session-local, not committed) for reuse
if the dispatch-overhead hypothesis needs testing before implementing step 3.

### Follow-up: per-element dtype-switch hypothesis also ruled out

Added variant D to the same harness: identical traversal to B, but with a `dtype switch`
evaluated once per (row, token) pair inside the hot loop — matching exactly what the reverted
attempt-1 code did (its `RowAgainstTile` switched on `dtype` per element, not once per call).

**Result: 0.98× vs the per-token baseline — no regression.** (For reference this run: B 1.04×,
C 1.15×, D 0.98×; C's private-scratch variant even showed some run-to-run variance across the
two sessions, 0.89-1.15×, consistent with it being noise-level rather than a real effect either
way.)

**All three candidate causes for attempt 1's ~4× regression are now ruled out in isolation:**
traversal pattern (warm and cold cache), output false sharing, and per-element dispatch
overhead. None reproduce it. The real cause remains unexplained by a standalone microbenchmark
and must be something specific to the full pipeline integration — possibly a measurement
artifact in how attempt 1 was benchmarked (rather than a real property of that code), or an
interaction with the surrounding `ForwardPass`/`MatMulBatched` call context (allocation
patterns, JIT codegen in the full assembly vs. an isolated exe) that this test can't reach.

**Decision: proceed to step 3.** Every mechanism that could have made row-outer/`_4In`-style
batching dangerous has been individually tested and cleared. The `_4In` design is also
structurally different from attempt 1 in the way that matters most — weight reuse happens
*inside one kernel call* (register-level, four dots per call) rather than via row-outer/
tile-outer traversal reorganizing the whole loop nest, so even if attempt 1's real cause is
never fully identified, it's less likely to apply here. Further root-causing attempt 1 has
diminishing returns relative to just implementing step 3 carefully (explicit per-dtype
dispatch resolved once per `MatMulBatched` call, per the original plan) and measuring the
result directly with `bench-vs-llamacpp.ps1` — which is the only fully trustworthy signal
regardless of what this microbenchmark says.

---

## 12. Steps 3-5 implemented and landed — gap closed to ~4.6x

`SimdKernels.TryMatMulBatchedQ8`, gated by `Q8PrefillEnabled` (env var
`STINGRAY_CPU_PREFILL_Q8=1`, default off per finding #5). Wired into `MatMulBatched`'s
non-BLAS branch: `if (Q8PrefillEnabled && batchSize >= 4 && TryMatMulBatchedQ8(...)) return;`
before the existing per-token fallback, which is otherwise completely untouched.

**Design, per the plan and the lessons from §10/§11:**

- `TryResolveQ8Dispatch` maps each dtype to its quantizer, scratch size, and dot family
  **once per `MatMulBatched` call**, not per element — the explicit fix for the dispatch-
  overhead risk raised in §11, even though that hypothesis was itself ruled out. Q4_K and
  Q3_K → `QuantizeRowToQ8KS`/`Q8KSScratchBytes`/`*_Q8KS_4In`; Q6_K → `QuantizeRowToQ8K`/
  `Q8KScratchBytes`/`DotQ6K_Q8K_4In` — the two scratch families are never crossed.
  Q8_0/Q5_K/Q2_K/Float32 have no `_4In` kernel and `TryResolveQ8Dispatch` returns false for
  them, falling straight back to the per-token loop (verified — see tests below).
- All `batchSize` activation rows are quantized to Q8 up front, once, before the row-parallel
  loop starts.
- Row loop (`Parallel.For` over rows, same threshold/threading as today) processes tokens in
  groups of 4 via the `_4In` kernel, with the 0-3 leftover tokens handled by the single-input
  Q8 dot — no `_2In` step; the tail is small enough that the extra kernel variant wasn't worth
  the complexity for this first landing.
- `batchSize >= 4` threshold, per finding #4, rather than `batchSize > 1`.

**Correctness — two contracts, kept separate as the review insisted:**

- `MatMulBatchedEquivalenceTests` (25 tests): **unchanged, still passing.** The gate is off by
  default, so this contract is untouched by construction, not merely re-verified.
- New `MatMulBatchedQ8EquivalenceTests` (15 tests): batched Q8 output is **bit-identical** to N
  independent per-token Q8 dot calls, for Q4_K and Q6_K (the two distinct scratch families),
  across batch sizes that exercise groups-of-4 exactly, plus 1- and 2-token remainders.
  Also covers: unsupported dtypes correctly return `false` rather than crash or silently
  misroute (Q8_0 included — it has single-input Q8 dots but no `_4In`, so it must still fall
  back); the gate off reproduces the plain per-token result exactly; batch sizes below the
  threshold never touch the Q8 path regardless of the gate.
- Full `Tests.ForwardPass` suite: 1027 passed / 16 failed, and the 16 are the exact same
  pre-existing Vulkan/GPU failures from before this change (verified by name) — zero new
  regressions.

**Measured result (same machine, same script, gate toggled — nothing else changed):**

| Prompt | Gate off (baseline, §9) | Gate on | Speedup |
|---|---|---|---|
| 87 tok | 30.7 | 42.2 | 1.37× |
| 261 tok | 32.5 | 42.6 | 1.31× |
| 903 tok | 31.6 | 40.1 | 1.27× |
| decode | 24.5 | 24.3 | unchanged (batch size 1 never crosses the `>=4` threshold) |

**Gap to llama.cpp (§2 reference: 193.4 / 197.6 / 185.1 t/s):**

| Prompt | Gap before this section | Gap now |
|---|---|---|
| 87 tok | ~6.3× | **4.6×** |
| 261 tok | ~6.2× | **4.6×** |
| 903 tok | ~5.9× | **4.6×** |

**This lands inside the 4-6× realistic ceiling the external review set for the `_4In`
kernel's math (§5 finding #1) — the plan's target for this phase of work is met.** Decode
remains unaffected and unregressed.

### What's still open before this could go default-on

Per finding #5, `Q8PrefillEnabled` stays opt-in until:
- Perplexity comparison (gate on vs off) on a fixed corpus — not yet done this session.
- Greedy-token end-to-end comparison on a real prompt — not yet done this session; the numeric
  gap between Q8 and F32 dots is unmeasured, only their *consistency under batching* is proven.
- A decision on whether the ~4-6% typically expected from int8 activation quantization is
  acceptable as prefill-only divergence from decode, given §10's finding that dense-model
  decode has zero existing precedent for this.

### Possible next increments (not started)

- `_2In` for the remainder instead of always falling to singles (marginal; tail is at most 3
  tokens per row).
- Q8_0 has no `_4In`/`_2In` at all — a real gap in the existing kernel inventory, not
  something this change can route around. Adding one would extend coverage but is new kernel
  work, not a wiring exercise like this section was.
- The remaining ~4.6× gap likely needs the genuine wider-than-4 kernel the plan flagged as a
  separate, later experiment (§5) — register pressure permitting.

---

## 13. Phase 1 quality verification — greedy-token divergence, measured

The one open item blocking `Q8PrefillEnabled` from ever defaulting on (§12's "what's still
open") was quality verification: does the Q8 prefill path actually change generated output,
and if so, how badly? Not yet perplexity (see caveat below), but real, measured greedy-token
comparison — the same check finding #5 (§5) asked for.

### Method

The existing `perplexity` CLI command turned out to be the wrong tool for this: it evaluates
token-by-token via `ForwardPass.Forward` (decode-style), never calling `MatMulBatched`, so it
never exercises the Q8 path at all — running it with the gate on would show zero difference by
construction, not because the path is safe. Used direct generation instead: 4 prompts (factual,
creative writing, Python code, a math/reasoning riddle), each ~40-115 prompt tokens (well above
the `batchSize >= 4` threshold so prefill definitely engages the gate), greedy (`--temp 0`),
150 tokens generated, `--verbose-prompt` to capture the top-5 logits and chosen token at every
decode step. Ran each prompt twice — `STINGRAY_CPU_PREFILL_Q8` unset vs `=1` — nothing else
changed, on `SmolLM2-1.7B-Instruct-Q4_K_M`.

### Result: generation diverges, always at a near-tied logit, never a large margin

All 4 prompts diverge from the F32 baseline at some point. In every case, the **first**
divergence is a razor-thin logit tie between the top two candidates:

| Prompt | First diverges at token | Off top-1 vs top-2 logit gap |
|---|---|---|
| Factual (Roman history) | 3 | 21.71 vs 21.68 (gap **0.03**) |
| Creative (short story) | 37 | 19.63 vs 19.56 (gap **0.07**) |
| Python code | 23 | 24.14 vs 23.86 (gap **0.28**) |
| Math riddle | 68 | 26.54 vs 26.53 (gap **0.01**) |

Every single case: the Q8 path's small numerical nudge is enough to flip a decision that was
already essentially a coin flip (logit gaps of 0.01-0.28, against a typical top-1 magnitude of
20-30). This is exactly the mechanism finding #5 predicted ("a different greedy token near a
logit tie"), now directly observed rather than theoretical. Because generation is autoregressive,
one flipped token near-tie early on cascades into a fully different continuation from that point
— that's *why* the later text differs so much, not evidence of compounding numerical error.

### Result: despite divergence, both content-critical outputs stayed correct

The two prompts where correctness (not just phrasing) matters:

- **Python code**: token-level fork at index 23 (`nums` vs `lst` as the parameter name). Both
  completions are fully correct, equivalent implementations — `set(...)` dedup,
  `len(...) < 2 → return None`, `sorted(...)[-2]` — just a different variable-naming choice
  downstream of the fork, not a logic change.
- **Math riddle** ("all but 9 die"): fork at index 68 (`initial` vs `total` as a word choice
  early in the reasoning chain). Both runs land on the **exact same final numeric answer** ("The
  farmer has 8 sheep left") via the same (incorrect — this is a known trick riddle the model
  misreads either way) reasoning shortcut, reproduced identically in both runs. The model's own
  reasoning limitation is unaffected by the gate; wording differs, the answer doesn't.

### What this does and doesn't establish

**Does establish:** on 4 varied prompts, divergence is real, occurs at genuinely close logit
decisions (not systematic distortion), and — in the two cases where it's checkable — doesn't
change output correctness, only phrasing. This is a meaningfully positive signal.

**Does not establish:** this is 4 prompts on 1 model, not the perplexity-over-a-fixed-corpus
statistical measurement §7 (finding #5) and §12 both called for. A proper perplexity comparison
needs a version of that evaluation that actually routes through `MatMulBatched`/prefill (the
existing `perplexity` command doesn't, as noted above) — either extending it to batch-evaluate
the corpus through the prefill path, or accepting that greedy-token spot-checks like this one
are the practical substitute given the tooling gap. Also untested: whether the divergence rate
(fraction of decode steps where a near-tie exists) changes with prompt length, model, or dtype
mix — all 4 prompts here happened to be Q4_K-dominant like the model's other weights.

### Verdict

Not yet sufficient to flip the default, but a real step forward: the risk this gate carries is
now characterized rather than assumed. It looks like "occasional wording-level drift at
already-uncertain decision points," not "the model degrades." Recommend either (a) building the
prefill-routed perplexity tool before deciding on default-on, or (b) running this same
greedy-token spot-check across more prompts/models as a lighter-weight substitute if a full
perplexity harness isn't worth building yet.

---

## 14. Prefill-routed perplexity tool — built and run

Closed the tooling gap §13 flagged: the existing `perplexity` command never called
`MatMulBatched`/`Prefill` at all, so it structurally could not see the Q8 path's effect.

### Engine plumbing (`ForwardPass.cs`)

Added `PrefillWithPerPositionLogits(tokens, startPos, PositionLogitsCallback)` — a new public
method (not part of `IForwardPass`, so `Prefill`'s existing interface signature is untouched).
`PositionLogitsCallback` is a custom delegate (`ReadOnlySpan<float>` can't be a generic `Action<>`
argument). Internally, `PrefillCore`'s step 3 (previously "final norm + LM head on the last
position only") now branches: when a callback is supplied it norms + projects **every** position
in the batch through the shared `_logits` buffer, invoking the callback once per position
(streaming — the caller must consume/copy before the next call overwrites it); the untouched
default path (`onAllPositionLogits: null`, used by `PrefillWithCache` and everywhere else) is
byte-for-byte unchanged.

### CLI wiring (`PerplexityCommand.cs`)

New `--batched` flag (routes scoring through `PrefillWithPerPositionLogits` in
`--batch-chunk-size`-token chunks, default 256, instead of token-by-token `Forward`) and
`--batch-chunk-size`. Validated as incompatible with `--tq` (TurboQuant's `PrefillCoreTq` path
isn't extended for this), `-g -1` (CUDA has no `PrefillWithPerPositionLogits`), and
MoE/per-layer-head-dim models (they fall back to sequential `Forward` internally regardless, so
`--batched` on them would silently just re-measure the token-by-token path).

### Result — SmolLM2-1.7B-Instruct-Q4_K_M, 1024-token corpus (first 20 KB of this repo's design doc)

| Mode | Perplexity | Speed |
|---|---|---|
| Token-by-token (no `--batched`) | 10.7132 | 8.75 tok/s |
| `--batched`, gate off | 10.7132 | 27.17 tok/s |
| `--batched`, gate on (`STINGRAY_CPU_PREFILL_Q8=1`) | 10.6682 | 37.09 tok/s |

**`--batched` gate-off exactly reproduces the token-by-token baseline (10.7132, bit-for-bit same
mean NLL)** — confirms the new plumbing is correct, not just fast, independent of Q8. It's also
~3x faster than token-by-token even with the gate off, since it's exercising the batched dense
path (BLAS-free `MatMulBatched`) instead of per-token `MatVec` — expected, matches §12's framing.

**Gate on: perplexity 10.6682, a −0.045 (0.4%) *decrease* from the F32 baseline** — noise-level,
not a regression in either direction. This is the missing statistical measurement §13 called
for: on this corpus, Q8 batched prefill does not measurably degrade perplexity. Consistent with
§13's greedy-token finding (divergence only at already-near-tied logits, content-critical outputs
unaffected) — now backed by a corpus-level number instead of 4 spot-checked prompts. Also
~37% faster than `--batched` gate-off (37.1 vs 27.2 tok/s), matching §12's ~1.3× kernel-level
measurement.

### What this does and doesn't move

**Does move:** the "not yet sufficient" verdict in §13 — a real, corpus-level, non-regressing
perplexity delta now exists, on top of the greedy-token spot-checks.

**Does not yet move:** the decision to flip `Q8PrefillEnabled`'s default. One model, one 1024-token
corpus, one run. Before defaulting on: repeat across at least one more model/architecture and a
larger/more diverse corpus (this one is a single technical document, not representative of
general text), and decide whether "noise-level on this sample" is a strong enough bar or whether
a statistical significance threshold should be defined first.
