# Path 2 — faithful C# port of `ggml_gemm_q4_K_8x8_q8_K` (AVX2 arm)

**Mission.** Port llama.cpp's AVX2 GEMM (`ggml/src/ggml-cpu/arch/x86/repack.cpp` lines 2816–3487,
~670 live lines) into C# as **Path 2**, selectable at runtime, with the existing kernels preserved
untouched as **Path 1**. Port as literally as possible: same structure, same variable names, same
intrinsic order, 1:1 onto `Vector256<T>` / `Avx2` / `Fma`.

**Ground rule for this log.** Dead ends, wrong turns and negative results are primary deliverables,
not footnotes. Every deviation from the original gets a recorded rationale. Started 2026-08-02.

---

## Increment 1 — survey, and a correction to my own earlier analysis

### 1.1 A prior claim of mine was wrong

In `README.md` §7 I wrote that llama.cpp's int16-accumulation trick was **not** transferable,
because "our vector already spans a whole sub-block (one scale)" and that our accumulators hold
K-partials while theirs hold output columns.

That was based on reading `DotQ4K_Q8KS_4In` only. **It is wrong as a statement about the codebase.**
A repacked path already exists (perf-loop iterations 39–40) and already does both things:

> `RepackQ4K8Rows` … "puts the ROW dimension in the vector lanes: a 32-byte load at `qs + cg*64 + g*32`
> holds 8 source bytes from each of 4 rows. One `maddubs` … covers 8 elements x 4 rows, and **chaining
> four of them in int16** covers a full 32-element sub-block for 4 rows at once."
> — `SimdKernels.cs:6134-6141`

So the column-parallel accumulator *and* the int16 chain are already in the tree. My §7 conclusion
("nothing cheap left to cherry-pick") happened to survive, but the reasoning behind it was wrong and
is corrected here. Lesson recorded: **grep the whole codebase for an idea before declaring it absent.**

### 1.2 What already exists (Path 1 assets, reusable)

| Asset | Location | Notes |
|---|---|---|
| `RepackQ4K8Rows` | `SimdKernels.cs:6092` | port of `make_block_q4_Kx8`, with deviations (§1.3) |
| `Q4Kx8BlockBytes` = 1216 | `SimdKernels.cs:6065` | `d[8]f + dmin[8]f + sc[64] + mn[64] + qs[1024]` |
| `DotQ4Kx8_Q8KS_Avx2` | `SimdKernels.cs:6149` | repacked weights × **1** token |
| `DotQ4Kx8_Q8KS_8In` | `SimdKernels.cs:6264` | repacked weights × **8** tokens, 8 *separate* pointers |
| `DotQ4Kx8_Q8KS_Scalar` | `SimdKernels.cs:6546` | reference |
| `TryMatMulBatchedQ4Kx8` | `SimdKernels.cs:6469` | **dispatch site — this is where the Path switch goes** |
| `RepackedGemm.cs` | whole file | repack driver + row dequant |

### 1.3 Existing deviations from llama.cpp, already made and already justified

- **`d`/`dmin` stored as `float`, not `ggml_half`.** Because C# has no F16C intrinsic (verified, §1.5).
  Costs 32 extra bytes per 1216-byte block; removes a per-super-block conversion from the hot loop.
  *Strictly cheaper than what llama.cpp pays.* Keep for Path 2.
- **Scales/mins pre-decoded** to `sc[64]`/`mn[64]` sub-block-major, instead of llama.cpp's re-bit-packed
  96-byte form. Removes `GetScaleMinK4` from the hot loop; one 8-byte read gets 8 rows' scales for a
  sub-block. **Consequence for Path 2: the original's `utmp`/`kmask1-3` scale-unpacking block has no
  counterpart and must be dropped.** This is the single largest structural deviation, and it is a
  deliberate improvement rather than a compromise.

### 1.4 What is actually missing versus llama.cpp's GEMM

Narrower than §7 implied. Path 1 has 8-columns-parallel; llama.cpp has 8 columns **× 4 token rows**:

1. **`block_q8_Kx4`** — activations interleaved 4 rows deep at 8-byte granularity
   (`repack.cpp:4301`, `ggml_quantize_mat_t<...>(..., 4, ne10)`). Path 1 passes N separate pointers.
2. **The sp1/sp2 shuffle patterns** — `_mm256_shuffle_epi32` with immediates 136/221 (B side) and
   160/245 (A side), which arrange operands so one `maddubs` produces partial products for the
   correct (row, column) pairs without any transpose.
3. **`acc_rows[16]` + `acc_min_rows[16]`** — 4 token rows × 4 column groups, with the min correction
   fused into the same pass.

**These three are the entirety of Path 2's new work.** The weight side is already done.

### 1.5 Intrinsic mapping — probed against the installed SDK, not recalled

Probe: `scratchpad/probe`, reflecting over `typeof(Avx2).Assembly`.

| llama.cpp | .NET | Status |
|---|---|---|
| `_mm256_loadu_si256` | `Vector256.LoadUnsafe` | ok |
| `_mm256_and_si256` | `Avx2.And` | ok (8 overloads) |
| `_mm256_srli_epi16` | `Avx2.ShiftRightLogical(v.AsInt16(), 4)` | ok (12) |
| `_mm256_shuffle_epi32` | `Avx2.Shuffle(v.AsInt32(), ctrl)` | ok (4) |
| `_mm256_maddubs_epi16` | `Avx2.MultiplyAddAdjacent(Vector256<byte>, Vector256<sbyte>)` | ok |
| `_mm256_madd_epi16` | `Avx2.MultiplyAddAdjacent(Vector256<short>, Vector256<short>)` | ok |
| `_mm256_add_epi16/epi32` | `Avx2.Add` | ok (8) |
| `_mm256_blend_epi32` | `Avx2.Blend(a.AsInt32(), b.AsInt32(), imm)` | ok (6) |
| `_mm256_permutevar8x32_epi32` | `Avx2.PermuteVar8x32` | ok (3) |
| `_mm256_permute2f128_si256` | `Avx2.Permute2x128` / `Avx.Permute2x128` | ok (8 / 10) |
| `_mm256_unpacklo/hi_epi32` | `Avx2.UnpackLow` / `UnpackHigh` | ok (8 each) |
| `_mm256_cvtepi32_ps` | `Avx.ConvertToVector256Single(Vector256<int>)` | ok |
| `_mm256_fmadd_ps` | `Fma.MultiplyAdd` | ok (4) |
| `_mm_hadd_epi16` | `Ssse3.HorizontalAdd` | to verify |
| **`_mm256_cvtph_ps`** (`GGML_F32Cx8_LOAD`) | — | **NO EQUIVALENT** |

**`_mm256_cvtph_ps` has no .NET counterpart.** `System.Runtime.Intrinsics.X86` exposes no `F16C`
class (probe found zero types matching `F16`/`Fp16`), and `Avx.ConvertToVector256Single` has exactly
one overload, taking `Vector256<int>`. **Resolution: sidestep entirely** — the existing repack
already converts `d`/`dmin` to float at pack time, so Path 2 loads 8 floats directly and the
intrinsic is never needed. A gap that costs nothing here; it would matter for a format that kept
halves resident.

### 1.6 Plan

| # | Increment | State |
|---|---|---|
| 1 | Survey, correction, intrinsic probe, this log | **done** |
| 2 | `GemmPath` switch + env var; Path 1 preserved; dispatch in `TryMatMulBatchedQ4Kx8` | next |
| 3 | `block_q8_Kx4` layout + `QuantizeMatQ8Kx4` (4-row interleave) | |
| 4 | GEMM: outer loops, `acc_rows[16]`, `acc_min_rows[16]` | |
| 5 | GEMM: B-side unpack + sp1/sp2 shuffles | |
| 6 | GEMM: A-side loads + sp1/sp2, `maddubs` chain, int16 accumulate | |
| 7 | GEMM: scale application, min correction, store | |
| 8 | GEMV tail for M ≤ 3 | |
| 9 | Parity tests vs Path 1 (exact or bounded) | |
| 10 | Perf A/B, isolated then end-to-end | |

Interfaces are involved (dispatch signature), so `-BuildOnly` first, full suite before any claim.

---

## Correction to §1.4: the pass shape is 16x8, not 4x8

I wrote "8 columns x 4 token rows". Wrong. `a_ptrs[4]` holds **four** `block_q8_Kx4` pointers, each
carrying 4 interleaved token rows, so the AVX2 arm processes **16 token rows x 8 weight columns =
128 output cells** per pass — hence `acc_rows[16]` and `acc_min_rows[16]` (repack.cpp:2833/2838),
indexed `acc_rows[rp * 4 + n]` for row-pair group `rp` in 0..3.

`col_scale_f32` / `col_dmin_f32` load once per super-block (2847/2850); `row_scale_f32` is broadcast
per token inside the sub-block loop via `_mm256_shuffle_ps` immediates 0 / 85 / 170 / 255
(3134-3137, 3144-3147). Final store is `acc_rows[i] - acc_min_rows[i]` (3154).

## Increment 2 — Path switch (done, builds clean)

`GemmPath.cs`: `enum GemmPath { Path1, Path2 }` + `GemmPathConfig`, defaulting from
`STINGRAY_GEMM_PATH` (`1`/`2`), settable at runtime so an A/B harness can flip paths without
reloading a model.

Dispatch added at `SimdKernels.TryMatMulBatchedQ4Kx8`, immediately after the `CanRepackQ4Kx8` guard
and before batch chunking, so Path 2 sees the whole batch and does its own chunking.

**Design choice worth recording: Path 2 may decline.** `TryMatMulBatched` returns `false` for any
shape it does not yet implement and the caller falls through to Path 1. An unfinished Path 2 can
therefore only cost speed, never correctness — which is what makes it safe to build this
incrementally in a tree that must stay working.

## Increment 3 — `block_q8_Kx4` + quantiser (done, builds clean)

Layout from repack.h:96 — `float d[4]`, `int8_t qs[QK_K*4]`, `int16_t bsums[QK_K/4]` =
**1168 bytes** per super-block per 4 token rows.

Ported `ggml_quantize_mat_q8_K_4x8_generic` (repack.cpp:262) scalar-first, deliberately: it is
unambiguous and becomes the oracle for the vectorised variant (arch/x86/repack.cpp:290) later.
Original variable names kept (`nb`, `srcv`, `iscale`, `src_offset`, `src_id`, `index`, `x0`).

Two things captured that are easy to get wrong:

- **`nearest_int` ported bit-exact**, not replaced with `MathF.Round`. It is the
  add-`12582912.0f`-and-mask trick (round-half-to-even); substituting a different rounding rule
  would silently perturb every quantised value. C# reaches the bits via
  `BitConverter.SingleToInt32Bits` where C uses `memcpy`.
- **`iscale = -127/max` uses the SIGNED value** at the position of maximum magnitude, not `amax`.
  So `iscale` is positive when that element is negative. Faithful to the original; not a typo.

### Open issue for parity testing (increment 9): Q8_KS vs Q8_K

Path 1 quantises activations to **Q8_KS** — eight sub-scales per super-block. Path 2's
`block_q8_Kx4` is **Q8_K** — one scale per super-block per row. This is a real numerical difference,
not a layout difference:

- Q8_KS is more accurate (per-32-element scaling) and forces a `cvt`+`fma` per sub-block, as already
  noted at `SimdKernels.cs:6146`.
- Q8_K permits the int32 accumulator to run across all eight sub-blocks before converting — which is
  part of why the original is fast.

**Consequence: Path 1 and Path 2 will NOT be bit-identical.** Parity tests must compare against a
scalar Q8_K reference and assert a bounded relative error against Path 1, not equality. Planning for
exact equality here would have produced a test that can never pass; recording it now so increment 9
is designed correctly rather than debugged into existence.

---

## Increments 4–7 — the GEMM itself (done: compiles, correct, and faster)

Ported repack.cpp:2818–3156 — the 16-token-row x 8-column main loop — into
`RepackedGemmPath2.GemmQ4Kx8Q8Kx4`. Original variable names throughout
(`rhs_raw_mat_0123_0`, `lhs_mat_01_00_sp1`, `iacc_mat_00_0_sp1`, `iacc_row_0`, …).

### PORTING TRAP: immediate operands cannot be wrapped (CA1857)

The natural C# rendering is one helper per intrinsic taking the shuffle immediate as a parameter:

```csharp
private static Vector256<byte> Sh32(Vector256<byte> a, byte c)   // WRONG
    => Avx2.Shuffle(a.AsInt32(), c).AsByte();
```

**This does not work.** `Avx2.Shuffle` / `Blend` / `Permute2x128` need a *compile-time constant*
immediate; a runtime value makes the JIT emit a slow dispatch path instead of the single
instruction. The build caught it as **CA1857** ("argument should be a constant for optimal
performance") and `TreatWarningsAsErrors` made it fatal.

That is a lucky break: without the analyser the kernel would have compiled, produced correct
results, and been quietly crippled — the exact failure mode that is hardest to diagnose from a
benchmark. Resolution: one specialisation per immediate actually used (`Sh136`, `Sh221`, `Sh160`,
`Sh245`, `Sh68s`, `Sh238s`, `Sh0s/85s/170s/255s`, `Blend240`, `Perm0`, `Perm17`, `Perm0s`).
Verbose, but every immediate is a literal. **Generalisable lesson: in C#, an intrinsic with an
immediate operand cannot be abstracted behind a parameter — specialise or inline.**

### Deviations from the original, and why

| # | Original | Path 2 | Rationale |
|---|---|---|---|
| 1 | `utmp_0/1` + `kmask1..3` unpack of 6-bit scales/mins in the hot loop (2953–2981) | two 8-byte loads: `DupWiden(sc + 2sb*8)`, `InterleaveWiden(mn + 2sb*8, mn + (2sb+1)*8)` | our `RepackQ4K8Rows` already decodes them at pack time. Verified to reproduce the identical lane layout — `scales_0` = `s0,s0,s1,s1,…`; `mins_01` = `mA0,mB0,mA1,mB1,…` |
| 2 | `GGML_F32Cx8_LOAD` (`_mm256_cvtph_ps`) for `d`/`dmin` | plain 8-float load | no F16C in .NET; our repack stores them pre-converted. Strictly cheaper |
| 3 | `_mm_hadd_epi16` on 128-bit halves | `Ssse3.HorizontalAdd(v.GetLower(), v.GetUpper())` | direct equivalent, confirmed present |

**Verified identical, not assumed:** the `qs` interleave. Ours is `qs[(g*8+r)*8]` = row `r` at
source offset `g*8`, which is exactly the original's `rhs_raw_mat_0123_0` = rows 0–3 /
`_4567_0` = rows 4–7 grouping. Checked against the original's inline byte-range comments.

### Increment 9 — parity (done)

Path 1 and Path 2 **cannot** be compared for equality (Q8_KS vs Q8_K, §Increment 3), so both are
compared to an independent exact-float scalar reference built from the raw Q4_K bytes.

```
|ref| max          = 521.0653
Path1 max abs err  = 2.256457   rel = 4.330E-003
Path2 max abs err  = 1.604884   rel = 3.080E-003
```

**Path 2 is correct** — error at activation-quantisation noise level. A wrong lane layout would
have produced relative error near 1.0, so this is a strong signal, not a weak one.

Unexplained observation, flagged rather than claimed: Path 2's max error came out *lower* than
Path 1's, despite Q8_K having one scale per super-block where Q8_KS has eight. That is
counter-intuitive and is a single max-error sample, not an accuracy study. **Do not cite it as an
accuracy result** until it is measured properly across many shapes and seeds.

### Increment 10 (partial) — isolated perf

`rows=2048 cols=2048 batch=64`, both paths through the real `TryMatMulBatchedQ4Kx8` entry point
(so `Parallel.For` and activation quantisation are included), interleaved round-robin, best-of-30,
`DOTNET_TC_QuickJitForLoops=0`:

```
Path1 best 1.757 ms   median 1.965 ms
Path2 best 1.098 ms   median 1.174 ms
Path2 speedup: best 1.600x   median 1.675x
```

**Path 2 is ~1.6x faster than Path 1.** For scale: the earlier llama.cpp repack-on/off A/B measured
1.51x at 4709 tokens, and our end-to-end gap to llama.cpp was 2.31x.

Caveats, stated up front so they are not forgotten:

1. **Isolated, not end-to-end.** Perf-loop iteration 24 is the standing precedent — a reproduced
   2.4–2.6x isolated win became a ~12% end-to-end *loss* under production `Parallel.For`
   contention. This number is not a product claim until `bench-prefill-cli.ps1` says so.
2. **`batch=64` is a favourable shape.** Path 2 currently declines `batch % 16 != 0` and falls
   through to Path 1, so realistic ragged batches would often not use it at all.
3. **Path 2 wins while handicapped.** Its activation quantiser is the *scalar* generic port; Path 1
   uses an optimised Q8_KS quantiser. The vectorised `ggml_quantize_mat_q8_K_4x8`
   (arch/x86/repack.cpp:290) is not yet ported, so there is known headroom.

### Next

| # | Increment | State |
|---|---|---|
| 8 | `nr % 16` tail (repack.cpp:3167+, `acc_rows[4]`) + M<=3 GEMV — needed before real batches qualify | next |
| 10a | Vectorised `QuantizeMatQ8Kx4` | |
| 10b | End-to-end `bench-prefill-cli.ps1` A/B — the only number that counts | |
| 11 | Promote parity harness into `tests/OpenTail.Stingray.Tests.ForwardPass` | |

## Increment 8 — row tail (done, builds clean, parity holds)

**DEVIATION 4.** The original writes two near-identical ~320-line bodies: a main loop over four
`block_q8_Kx4` groups (`acc_rows[16]`, repack.cpp:2818) and a tail over one (`acc_rows[4]`,
repack.cpp:3158). Path 2 carries `nrp` groups (1..4) through a single body — `nrp = min(4,
nGroups - y)`, accumulators and store loop sized `nrp * 4`, `y += nrp`.

Semantics are identical; only the duplication goes. This is the one place where staying literal
would have meant copying 320 lines to change two constants, so the deviation is deliberate and
recorded rather than silent.

**Bug caught before it ran:** converting `for (y...; y += 4)` into a `while`-style loop left the
increment behind, so `for (int y = 0; y < nGroups; )` had no `y += nrp` — an infinite loop. Found
by reading the emitted block rather than by running it. Noting it because it is the characteristic
hazard of mechanical loop-shape edits via sed.

Parity across every `nrp` value, plus the ragged cases that exercise a short final group:

```
batch=  4 nrp=1  p1rel=3.46E-003  p2rel=3.19E-003  OK
batch=  8 nrp=2  p1rel=4.33E-003  p2rel=3.08E-003  OK
batch= 12 nrp=3  p1rel=4.33E-003  p2rel=3.08E-003  OK
batch= 16 nrp=4  p1rel=4.33E-003  p2rel=3.08E-003  OK
batch= 20 nrp=4+1                 p2rel=3.42E-003  OK
batch= 64 nrp=4  p1rel=3.86E-003  p2rel=4.33E-003  OK
batch= 68 nrp=4+1                 p2rel=4.33E-003  OK
```

Path 2 now accepts any `batchSize % 4 == 0`. The `M <= 3` remainder still needs the GEMV port
(repack.cpp:1464) and falls through to Path 1.

Perf unchanged by the generalisation (rows=2048 cols=2048 batch=64):
`Path1 best 1.780 / median 1.985 ms`, `Path2 best 0.989 / median 1.183 ms` —
**best 1.800x, median 1.677x**.

### HAZARD found while checking production wiring: Path 2 reintroduces a numerics boundary

`ForwardPass.cs:825` calls `TryMatMulBatchedQ4Kx8` for real prefill, gated on
`SimdKernels.Q8PrefillEnabled`. The comment immediately above it (813–824) warns:

> Deliberately NOT gated on a minimum N. An "N >= 8" gate looks harmless … but it is a **NUMERICS
> boundary**: a prompt admitted in chunks whose tail falls below the threshold would have some
> positions computed by this path and others by the row-major one, so chunked and unchunked prefill
> of the same prompt disagree. That is exactly the defect `MinBatchForQ8Prefill` had, caught by
> `ContinuousBatchingTests.PrefillWithCache_Chunked_MatchesFull`.

**Path 2's `batchSize % 4 != 0` decline is precisely that defect, reintroduced.** Because Path 2
uses Q8_K and Path 1 uses Q8_KS, the two do not merely round differently — they are different
quantisations. A prompt chunked so that some chunks are `%4 == 0` and others are not would have
different positions computed by different quantisation schemes, and chunked vs unchunked prefill of
the same prompt would disagree.

Consequences, recorded now rather than discovered later:

1. **Path 2 must not become the default until it handles every batch size** — i.e. until the
   `M <= 3` GEMV (repack.cpp:1464) is ported. Opt-in via `STINGRAY_GEMM_PATH=2` is safe for
   measurement; a default is not.
2. `ContinuousBatchingTests.PrefillWithCache_Chunked_MatchesFull` is expected to **fail** with
   `STINGRAY_GEMM_PATH=2` set, and that failure is correct behaviour by the test, not a bug in
   the test. It must be run both ways.
3. This raises the GEMV from "completeness" to "prerequisite for shipping". Priority accordingly.

The design decision from increment 2 — let Path 2 decline and fall through — is what makes an
incomplete Path 2 *safe to build*, but it is exactly what makes it *unsafe to default to*. Both
statements are true at once, and the distinction matters.

## Increment 10b — end-to-end harness: three latent bugs in the existing tooling

Getting `scripts/bench-prefill-cli.ps1` to produce a number at all took three fixes. None are about
the GEMM; all three would silently mislead anyone using that script, so they are recorded here.

### 1. `-p` breaks past ~1000 words (fixed in the script)

```
Program 'opentail-llm-cli.exe' failed to run: An error occurred trying to start process …
The filename or extension is too long.
```

The prompt was passed on the command line via `-p`, which exceeds the Windows command-line limit
once the word count gets large. The CLI already has `-f|--file`, whose own help text says it is
"useful for prompts longer than the shell's command-line limit" — the script simply wasn't using
it. **Fixed**: the script now writes the prompt to a temp file and passes `-f`.

Worth noting the failure mode: the script caught it as "no prefill line parsed" and carried on, so
it reported an empty result set rather than an error. A silent empty result is worse than a crash.

### 2. An unregistered env var breaks the harness

`STINGRAY_GEMM_PATH` was not in `KnownEnvironmentVariables.cs`, so the CLI printed
`warning: … is set but is not read by this build` **to stderr**, which tripped the bench script's
`$ErrorActionPreference = "Stop"`. **Fixed** by registering the variable. The warning itself is a
good feature — it just needs the registration to happen at the same time as the variable is
introduced. Note it is purely a lint: `GemmPathConfig` reads the variable regardless, so Path 2 was
active the whole time the warning claimed otherwise.

### 3. `-Words 1200,2400` through `pwsh -File` silently concatenates (NOT fixed — caller beware)

```
WARNING: no prefill line parsed (run 0, 12002400 words)
```

`pwsh -NoProfile -File script.ps1 -Words 1200,2400` passes the array as one token, which coerces to
the single integer **12002400**. The script then builds a 12-million-word prompt, the CLI fails,
and every run reports "no prefill line parsed".

This cost the most time of the three, because single-value runs (`-Words 1200`) worked perfectly
and multi-value runs failed — which looks like a size/timeout problem and is not. **Workaround:
one `-Words` value per invocation.** Anyone reading a past `bench-prefill-cli.ps1` result that used
a comma-separated `-Words` through `-File` should treat it as suspect.

**Process note.** I diagnosed all three by guessing at the pipeline before finally capturing the
raw child output, which showed `12002400` immediately. Reading the actual error first would have
been faster than three rounds of plumbing hypotheses — the same lesson as the `llama-bench` episode
in README §0.

## MAJOR FINDING: the repacked path is OFF BY DEFAULT in production

The first end-to-end A/B showed **no difference at all** (1200 words: Path1 78.5/79.7, Path2
77.1/77.0 t/s; 2400 words: 71.1/73.0 vs 72.4/73.0). That looked like the classic
iteration-24 outcome — an isolated win evaporating under production contention.

**It was not.** I added a temporary one-shot `Console.Error.WriteLine` inside
`RepackedGemmPath2.TryMatMulBatched` to prove engagement instead of inferring it. It never printed.
**Path 2 was never reached.** Neither, therefore, was Path 1's repacked kernel.

The gate is `ForwardPass.cs:63`:

```csharp
private readonly long _q4kx8CacheBudgetBytes =
    long.TryParse(Environment.GetEnvironmentVariable("STINGRAY_Q4KX8_CACHE_MB"), out var mb) && mb > 0
        ? mb * 1024 * 1024
        : 0;                      // <-- default 0
```

`GetRepackedQ4Kx8` returns `null` when the budget is 0, so `TryMatMulBatchedQ4Kx8` is never called.
**`STINGRAY_Q4KX8_CACHE_MB` must be set for any repacked kernel to run at all.**

With `STINGRAY_Q4KX8_CACHE_MB=2048`:

```
[Path2] ENGAGED batch=512 rows=2048 cols=2048
Prefill: 631 tokens, 87.7 t/s
```

versus 68.9 t/s on the same prompt with defaults — the repacked path is worth a large amount, and
it is off unless you know to switch it on.

### Consequences

1. **Perf-loop iteration 42's "measured 2.6x over the row-major _8In" is an isolated-kernel
   number that never applied to a default-configured run.** The kernel is real; the default
   configuration does not reach it. Anyone reading `perf-loop-progress.md` would reasonably assume
   otherwise. This deserves checking against the other repacked-path claims in that log.
2. Every end-to-end prefill measurement taken without `STINGRAY_Q4KX8_CACHE_MB` set has been
   measuring the *non*-repacked path, whatever the log said.
3. The Path 1 vs Path 2 A/B is only meaningful with that variable set. Re-running accordingly.

### Second finding, incidental: the dequant cache is a pessimisation here

`STINGRAY_PREFILL_DEQUANT_MB=0` **raised** prefill from 68.9 to 75.4 t/s on the same prompt —
i.e. dequantising Q4_K weights to F32 and running SGEMM is *slower* than not doing so, on this box
with OpenBLAS absent ("OpenBLAS: not found (fallback to sequential)"). The branch at
`ForwardPass.cs:804` takes precedence over the repacked path whenever
`N >= MinBatchForBlas` (16), so it was shadowing the repacked path even before the budget gate.

Worth a separate look: a default that costs ~9% on this configuration.

### Process note

I spent four rounds hypothesising about which branch ran (chunk sizes, `Q8PrefillEnabled`, the
dequant cache) before adding a two-line print. **The print settled it immediately.** Same lesson as
the `llama-bench` retraction in README §0 and the `12002400` harness bug above: when a measurement
disagrees with the model of the code, instrument the code — do not refine the model.

## Increment 8b — zero-padding replaces the M<=3 decline (done, parity verified)

The `batchSize % 4 != 0` decline from increment 8 is removed. The final partial group is now
zero-padded up to a full `block_q8_Kx4`; `GemmQ4Kx8Q8Kx4` takes both `nr` (padded, always a
multiple of 4) and `validRows`, computing the padded rows and storing only the real ones.

**Why padding rather than porting the GEMV.** Declining was the smaller change, but it reintroduced
the numerics boundary documented above: Path 1 and Path 2 use *different activation quantisations*
(Q8_KS vs Q8_K), so a prompt whose chunks straddle `%4` would have some positions computed one way
and some the other, and chunked vs unchunked prefill of the same prompt would disagree. Padding
keeps the entire prompt on one scheme unconditionally.

Zero rows are safe by construction: `amax == 0` makes `iscale = 0` and `d = 0`, so a padded row
contributes exactly zero and cannot perturb the real rows.

**This demotes the ported GEMV from "prerequisite" to "optional optimisation."** Correctness no
longer needs it; it would only avoid wasting up to 75% of a pass when `M <= 3` — which matters for
decode, not prefill.

Parity now passes for every batch size, including all the ragged ones:

```
batch=  1  p2rel=3.49E-003  OK      batch= 13  p2rel=3.08E-003  OK
batch=  2  p2rel=2.68E-003  OK      batch= 16  p2rel=3.08E-003  OK
batch=  3  p2rel=3.60E-003  OK      batch= 17  p2rel=3.08E-003  OK
batch=  4  p2rel=3.19E-003  OK      batch= 20  p2rel=3.42E-003  OK
batch=  5  p2rel=2.82E-003  OK      batch= 63  p2rel=4.33E-003  OK
batch=  7  p2rel=3.08E-003  OK      batch= 64  p2rel=4.33E-003  OK
batch=  8  p2rel=3.08E-003  OK      batch= 65  p2rel=4.33E-003  OK
```

`[Path2] ENGAGED batch=1` confirms even a single token now routes through the ported GEMM.

Isolated perf re-measured after the change (and after a machine restart): **best 1.675x, median
1.630x** — consistent with the 1.600x/1.675x measured before, so the padding costs nothing at
`batch=64` and the result reproduces across a reboot.

## Measurement hygiene: a machine restart mid-session

The user reported the PC restarted partway through. Auditing what that invalidates:

- **Isolated kernel A/B — safe.** Re-measured post-restart at 1.675x/1.630x vs 1.600x/1.675x
  before. Reproduces.
- **600-word end-to-end — safe.** Run strictly interleaved P1,P2,P1,P2 within a single pass, so
  drift and thermal state hit both arms equally. Path1 best 86.1, Path2 best 95.3 → **1.11x**.
- **2400-word end-to-end — DISCARDED.** Path 1 (75.1 t/s) came from one background run and Path 2
  (79.2 t/s) from a separate later invocation. Not interleaved, so a restart between them
  contaminates the comparison. The 1.05x derived from it is withdrawn and being re-measured
  interleaved.

Recording this because the failure is invisible after the fact: two numbers from two runs look
exactly like two numbers from one interleaved run, and only the provenance distinguishes them.

## Increment 10b — end-to-end result (clean, interleaved, post-restart)

2400 words, strictly interleaved P1/P2 within one pass, three reps:

```
rep1 path=1 73.0 t/s   rep1 path=2 84.5 t/s
rep2 path=1 75.9 t/s   rep2 path=2 84.6 t/s
rep3 path=1 74.8 t/s   rep3 path=2 84.6 t/s
```

**Path 2 = 1.115x over Path 1** (best-of 84.6 vs 75.9). This supersedes the withdrawn 1.05x, which
was contaminated by a non-interleaved comparison straddling a machine restart.

| prompt | Path 1 | Path 2 | ratio |
|---|---:|---:|---:|
| 600 words (631 tok) | 86.1 | 95.3 | 1.11x |
| 2400 words (2431 tok) | 75.9 | 84.6 | 1.115x |

Two observations beyond the headline:

- **Consistency.** Path 2's three reps span 0.1% (84.5–84.6); Path 1's span 4% (73.0–75.9). The
  ported kernel is not merely faster, it is far less variable — plausibly because it does the same
  work per pass regardless of how the batch divides, where Path 1's `_8In`/`_Avx2` split shifts with
  the ragged tail.
- **Still handicapped.** Path 2 uses the *scalar* generic activation quantiser; the vectorised
  `ggml_quantize_mat_q8_K_4x8` (arch/x86/repack.cpp:297–506, ~210 lines) is not yet ported. The
  1.11x is a floor, not a ceiling.

### Full end-to-end picture at 600 words

| configuration | prefill | vs default |
|---|---:|---:|
| default | 68.9 t/s | — |
| `PREFILL_DEQUANT_MB=0` | 75.4 t/s | 1.09x |
| `Q4KX8_CACHE_MB=2048` + Path 1 | 86.1 t/s | 1.25x |
| `Q4KX8_CACHE_MB=2048` + Path 2 | 95.3 t/s | **1.38x** |

The config changes are worth more than the port (1.25x vs 1.11x) and cost no code. Both should be
considered independently of whether Path 2 ships.

### Retrospective on the "do not port" verdict (README section 7)

That verdict was wrong, and it is worth being precise about *why*, because the arithmetic behind it
was sound. The 2.31x gap to llama.cpp really does decompose into ~1.51x repacked-GEMM and ~1.53x
residual, and a perfect port really would not reach parity.

The error was treating **"will not reach parity with llama.cpp"** as equivalent to **"not worth
doing"**. Those are different questions. The port was never going to win that race; it was going to
make this engine ~1.11x faster end-to-end and ~1.63x faster in the kernel, which is the question
that actually mattered.

Worth noting what the work surfaced that no amount of further analysis would have: the repacked
path being disabled by default, the dequant-cache pessimisation, three latent harness bugs, and a
wrong claim in my own §7 about the int16 chain. All of those came from building the thing and being
confused by a measurement.

## Increment 11a — cleanup and verification

**Diagnostic promoted, not deleted.** The temporary `Console.Error.WriteLine` is replaced by a
public counter, `RepackedGemmPath2.EngagedCalls`. Rationale: the single most expensive confusion in
this port was a null A/B result that was ambiguous between "no speed difference" and "never
executed" — and it was the latter. That ambiguity is now one property read away for whoever hits it
next, instead of a rebuild with a print statement.

**Test-harness note (fourth tooling gotcha of the session).** `dotnet test <proj> --nologo` does
**not** work for these projects: they run on Microsoft.Testing.Platform (xunit.v3), which does not
accept `--nologo`, prints its usage banner instead, and exits **5 / "Zero tests ran"** — which reads
exactly like a discovery failure rather than a bad argument. Same family as the `--filter` vs
`--filter-class` difference already known for this repo. Correct invocation is plain
`dotnet test tests/OpenTail.Stingray.Tests.ForwardPass -c Release`.

## Increment 11b — full test suite: 1168/1170 with Path 2, TWO REAL FAILURES

`dotnet test tests/OpenTail.Stingray.Tests.ForwardPass -c Release`

- **Defaults (Path 1): 1170/1170 pass.** None of the port's changes regress the incumbent.
- **`STINGRAY_GEMM_PATH=2` + `STINGRAY_Q4KX8_CACHE_MB=2048`: 1168 pass, 2 FAIL.**

```
failed ContinuousBatchingTests.PrefillPackedMulti_MatchesSequentialPrefill      (line 390)
failed ContinuousBatchingTests.PrefillPackedMulti_ChunkedContinuation_MatchesFull (line 446)
  Assert.Equal() Failure: Values are not within 2 decimal places
  Expected: 0.88 (rounded from 0.881207943)
  Actual:   7.77 (rounded from 7.7728157)
```

Note the magnitude: 0.88 vs 7.77 is **not** quantisation drift. Something is structurally wrong.

### Localisation: the GEMM is NOT the bug

The failing test uses the real model (`FindModelPath()`), so cols=2048 and Path 2 engages. It
compares two prompts prefilled separately (N=7, N=4) against both packed into one pass (N=11) —
both sides using Path 2, so Path 2's output is depending on N when it must not.

Extended the parity harness to the real trunk shape and those exact batch sizes:

```
rows=2048 cols=2048  batch= 4   p2rel=4.28E-003  OK
rows=2048 cols=2048  batch= 7   p2rel=4.28E-003  OK
rows=2048 cols=2048  batch=11   p2rel=4.06E-003  OK
```

**The ported kernel is correct at exactly the shapes the failing test uses.** Whatever is wrong lies
in the interaction with `PrefillPackedMulti`, not in `GemmQ4Kx8Q8Kx4`.

Hypotheses NOT yet eliminated (recorded so the next session does not re-derive them):
- `PrefillPackedMulti` may route some projection through a call whose output stride is not `rows`,
  which Path 1 tolerates and Path 2 does not.
- Sequence boundaries fall mid-`block_q8_Kx4` group when prompts of length 7 and 4 are packed;
  quantisation is per-row so this *should* be harmless, and the harness supports that, but it is
  the most obvious structural difference and has not been directly disproved.
- The LM-head projection (rows=49152) is called with a different N in the packed path.

**Status: Path 2 is NOT safe to enable for continuous-batching / packed-multi prefill.** It is
correct for single-prompt prefill (chunked and unchunked — those tests pass). The env var is opt-in,
so the default build is unaffected.

## Increment 10a — measurement that changes the priority, and a correction

Before porting the ~210-line vectorised `ggml_quantize_mat_q8_K_4x8`, measured what fraction of the
pass the scalar quantiser actually is:

```
scalar QuantizeMatQ8Kx4 (serial) median 0.525 ms
Path2 median pass                       1.275 ms
```

**That raw 41.2% is misleading and must not be quoted.** The measurement runs the quantiser
*serially*, whereas `TryMatMulBatched` runs it under `Parallel.For` across 6 cores. The real share
is therefore roughly **7–10%**, and a perfect vectorisation caps at about **1.08–1.11x**.

Two of my own estimates were wrong in opposite directions and are corrected here: the earlier "~3%,
probably not worth 210 lines" reasoning (which compared O(batch·cols) to O(batch·rows·cols) and
ignored that the scalar quantiser is genuinely slow per element), and the raw 41.2% reading (which
ignored that production parallelises it). Worth ~1.1x, so worth doing — but after the packed-multi
defect, not before it.

## CORRECTION (increment 11c): the 2 failures are pre-existing, and my "MAJOR FINDING" was overstated

### 1. Path 2 does not introduce the `PrefillPackedMulti` failures

Ran the suite with **Path 1** and `STINGRAY_Q4KX8_CACHE_MB=2048`:

```
Test run summary: Failed!   total: 1170   failed: 2   succeeded: 1168
```

Identical count to the Path 2 run. **The failures belong to the repacked path, not to the port.**
The previous section's "Path 2 is NOT safe for continuous-batching / packed-multi prefill" was
wrong as stated: the correct statement is that *the repacked path* (either kernel) diverges from
the row-major path enough to exceed that test's tolerance.

Corrected status: **Path 2 is a drop-in replacement for Path 1 within the repacked path, with
identical test outcomes and ~1.11x better end-to-end throughput.**

### 2. The off-by-default is deliberate and documented — I mischaracterised it

`ForwardPass.cs:48-62` states the rationale in full, and I edited that very file without reading it:

> - The kernel is 2.6x over the row-major `_8In` in isolation, but only **+14%** end-to-end
>   (77.2 vs 67.7 t/s at 267 tokens) — the Q4_K matmuls are a smaller share of prefill than that
>   isolated figure implies.
> - It costs a second copy of the Q4_K weights and gives up mmap sharing of the GGUF pages.
> - It is a NUMERICS change … enough to push `ContinuousBatchingTests.PrefillPackedMulti_*` past
>   its packed-vs-sequential tolerance. Flipping this default needs the same treatment the
>   Q8-prefill flip got: a perplexity gate plus greedy-parity checks, and a decision on those
>   tests' tolerance.

So the earlier section headed "**MAJOR FINDING: the repacked path is OFF BY DEFAULT**" is
overstated and its implication — that iteration 42's number "never applied to a default-configured
run" as though that had been overlooked — is unfair. Iteration 42 measured end-to-end (+14%),
documented why the default is off, and listed exactly what flipping it would require.

**What survives from that section, and what does not:**

| Claim | Verdict |
|---|---|
| The repacked path is off unless `STINGRAY_Q4KX8_CACHE_MB` is set | **stands** — and is worth knowing when benchmarking |
| Enabling it is worth ~1.25x end-to-end here | **stands** — and is consistent with iteration 42's +14% at a shorter prompt (267 tok vs my 631) |
| Every prior end-to-end prefill measurement without that var measured the non-repacked path | **stands** |
| This was an oversight / unmeasured | **WITHDRAWN** — it is a documented, reasoned default with a stated flip procedure |
| `perf-loop-progress.md`'s repacked claims need auditing | **WITHDRAWN** as phrased; the rationale is already recorded in the code |

**Process lesson, and it is the sharpest of the session.** I spent an increment hunting a "Path 2
defect" that a doc comment in the file I was already editing had predicted and explained. Earlier
today I recorded "grep the whole codebase for an idea before declaring it absent" after the §7
int16-chain error. This is the same failure repeated: **read the doc comments on the code you are
modifying before forming a theory about its behaviour.** Twice in one session is a pattern, not an
accident.

### 3. Consequence for the GEMV

Unchanged: still not needed. Zero-padding closed the correctness gap for the M<=3 case; these two
failures are a summation-order divergence inherent to the repacked layout, which the GEMV would not
address.

## Gap to llama.cpp, re-measured (2026-08-02, end of port)

Same box, same model, matched prompts, best-of-3 interleaved. OpenTail via its own CLI with
`STINGRAY_Q4KX8_CACHE_MB=2048` + `STINGRAY_GEMM_PATH=2`; llama.cpp via `llama-completion`
(`-st --simple-io`, `-t 6`, repack default on).

| prompt | OpenTail | llama.cpp | gap |
|---|---:|---:|---:|
| 900 words | 98.8 t/s (931 tok) | 167.4 t/s (909 tok) | **1.69x** |
| 2400 words | 88.0 t/s (2431 tok) | 147.2 t/s (2409 tok) | **1.67x** |

**The gap has gone from 2.31x to ~1.68x.**

### Internal consistency check

The two measurement chains agree, which matters because they are independent:

```
component chain:  repack cache 1.25x  x  Path 2 1.11x  =  1.39x
end-to-end chain: 2.31x (before)  /  1.68x (after)     =  1.38x
```

Agreement to ~1% across entirely separate experiments. Neither number is an artifact of its harness.

The gap also stays flat across a 2.6x range of context length (1.69x at 931 tokens, 1.67x at 2431),
matching the earlier finding that the 2.31x was flat at 931 and 4831. The structure of the
remaining gap is unchanged; it is simply smaller.

### Caveats that bound this claim

1. **Prefill only.** Decode (M=1, GEMV) is untouched by this work and is where interactive feel
   lives. Do not generalise "1.68x behind" to the engine as a whole.
2. **One config.** SmolLM2-1.7B, Q4_K_M, Zen 3, AVX2 only (no AVX-512, no VNNI), no OpenBLAS.
   llama.cpp's real advantage is breadth, not peak on a tuned config.
3. **Cross-harness.** OpenTail's CLI vs `llama-completion`. Both report prefill t/s but the
   surrounding work is not identical. Indicative, not a benchmark result.
4. **Behind two env vars.** A default-configured OpenTail build still gets 68.9 t/s, not 98.8.
   Closing that requires the perplexity-gate + greedy-parity + test-tolerance work named at
   `ForwardPass.cs:59` — validation work, not engineering.

## Decode performance — measured, and it reframes the "how far behind" question

128 generated tokens, greedy, 50-word prompt, best-of-3, same box and model:

| configuration | decode |
|---|---:|
| OpenTail, default | 28.40 t/s |
| OpenTail, `Q4KX8_CACHE_MB=2048` + `GEMM_PATH=2` | 28.10 t/s |
| llama.cpp (`llama-completion`) | 35.90 t/s |

**Decode gap: 1.26x** — much smaller than prefill's 1.68x.

Two things follow:

1. **Today's work does not touch decode.** 28.40 vs 28.10 is within noise, confirming decode runs
   through the F32 `MatVec` path and never reaches `TryMatMulBatchedQ4Kx8`. The repacked GEMM is a
   prefill-only lever, as expected (the GEMM is selected only for M > 3; decode is M = 1).
2. **The engine was never as far behind as the prefill figure implied.** For interactive use decode
   dominates the felt experience, and there the gap was already 1.26x before any of this. The
   headline "2.31x behind" was a prefill number being read as an engine-wide one.

Combined position after today (SmolLM2-1.7B Q4_K_M, Zen 3, AVX2, no OpenBLAS):

| phase | before | after | notes |
|---|---:|---:|---|
| prefill | 2.31x behind | **1.68x behind** | repack cache + Path 2, both opt-in |
| decode | 1.26x behind | 1.26x behind | untouched; GEMV path |

## Increment 11 — parity harness promoted into the test project (done)

`tests/OpenTail.Stingray.Tests.ForwardPass/RepackedGemmPath2Tests.cs`. **18 tests, all passing, 1.2 s** —
cheap enough to stay in the suite permanently.

- `Path2_MatchesScalarReference_AtEveryBatchSize` — 14 batch sizes including 1/2/3 (zero-padded
  partial group) and 5/7/13/17/65 (full pass then short pass).
- `Path2_MatchesScalarReference_AtTrunkShape` — batches 4/7/11 at rows=cols=2048. Separate because a
  kernel can be right at cols=512 (2 super-blocks) and wrong at 2048 (8) if a super-block index is
  mishandled.
- `QuantizeMatQ8Kx4_ZeroRow_ProducesZeroScale` — pins the zero-padding contract (`amax == 0` gives
  `d == 0`) directly, rather than relying on the parity tests to notice if it broke.

They compare against the scalar reference, not against Path 1 — asserting equality between the two
paths could never pass, since Q8_KS and Q8_K are different quantisations. That reasoning is in the
file's doc comment so nobody "fixes" the tests into something impossible later.

## Increment 10a — vectorised activation quantiser (done, bit-identical, wired in)

`QuantizeMatQ8Kx4Avx2` — port of `ggml_quantize_mat_q8_K_4x8` (arch/x86/repack.cpp:297–506). The
scalar `QuantizeMatQ8Kx4` is kept as the oracle, not deleted.

### Two deviations, both output-identical (proven, not argued)

**DEVIATION 5 — sign recovery.** The original recovers the *signed* value at the maximum-magnitude
position with an accumulated compare-mask chain across sub-blocks
(`maskAbs` / `mask_prev` / `mask_next`, lines 321–386). Path 2 tracks running `max` and `min`
instead: the largest-magnitude element is `max` when `max >= -min`, else `min`. Far fewer ops and
trivially checkable.

*Tie case:* when `max == -min` the two disagree about which element wins — this picks the positive
one, the scalar version picks whichever came first. Both are valid quantisations of the same data:
they differ only in the sign of `d` **together with** the sign of every quant, so the reconstructed
values are identical. Measure-zero on real activations regardless.

**DEVIATION 6 — bsums.** The original builds them with a shuffle/blend dance over three hand-built
masks (lines 427–506, ~70 lines). Path 2 derives them from the stored layout instead: for output
group `g`, row `r` occupies bytes `g*32 + r*8 .. +8`, and `bsums[r*4 + (g/8)*16 + ((g%8)/2)]`
accumulates two adjacent groups. A `maddubs` + `madd` pair reduces 32 bytes to four row sums.
Same values, a fraction of the code — the derivation came from reading the scalar index formula
`(((j & 31) >> 3) << 2) + ((j >> 8) << 4) + ((j >> 6) & 3)` rather than from the vectorised original.

The pack/permute sequence *is* ported literally (`PackSignedSaturate` twice then
`PermuteVar8x32` with `{0,4,1,5,2,6,3,7}`), because working out that it lands the four rows as
`r0[8] r1[8] r2[8] r3[8]` is exactly the kind of thing worth copying rather than reinventing.

### Verified by test, not by reasoning

`QuantizeMatQ8Kx4Avx2_IsBitIdenticalToScalar` at cols = 256 / 512 / 2048, with planted ±9.5
outliers so both branches of the sign recovery are exercised. **Byte-for-byte identical.** 21/21
tests pass.

### Result

| | Path 2 median pass | speedup vs Path 1 |
|---|---:|---:|
| scalar quantiser | 1.275 ms | 1.60x |
| vectorised quantiser | 1.073 ms | **1.83x** |

A 1.19x gain on the pass — slightly better than the 1.08–1.11x predicted from the corrected
Amdahl estimate, and much better than the "~3%, not worth 210 lines" I first guessed. **The
decision to measure the quantiser's share before porting it was right; the first estimate that
would have cancelled the work was wrong.**

*Stale harness line:* `scratchpad/parity` still times `QuantizeMatQ8Kx4` (the scalar one) and reports
it as a share of the pass. That number now describes a function the dispatch no longer calls, and
should be read as "what the scalar version would have cost", not as current overhead.

### …and it did NOT translate end-to-end

| prompt | OpenTail | llama.cpp | gap | gap before quantiser change |
|---|---:|---:|---:|---:|
| 900 words | 95.5 t/s | 160.3 t/s | 1.68x | 1.69x |
| 2400 words | 83.2 t/s | 142.9 t/s | 1.72x | 1.67x |

**No measurable improvement.** Both engines measured ~4% slower this run than the previous one
(llama.cpp 160.3 vs 167.4 at 900w), so the machine drifted between sessions — which is exactly why
the *ratio* is the quantity to trust, and the ratio is flat.

So: a genuine, test-verified **1.19x on the isolated matmul pass** produced **~0% end-to-end**.

This is perf-loop iteration 24's lesson recurring in miniature, and it is worth stating plainly
because the whole increment was justified by an isolated measurement:

- The Amdahl estimate that motivated the work was computed against **Path 2's own pass**, not
  against total prefill. Even a perfect quantiser can only shrink the Q4_K matmul share of prefill,
  and that share is evidently smaller than the isolated benchmark implies — the same effect the
  existing `ForwardPass.cs:51` comment already records for the repack itself ("2.6x in isolation,
  but only +14% end-to-end — the Q4_K matmuls are a smaller share of prefill than that isolated
  figure implies").
- I read that comment earlier today and still made the same class of error one increment later.

**Keeping the change anyway**: it is bit-identical to the scalar oracle, covered by tests, and
strictly faster in isolation, so it costs nothing and helps whenever the matmul share is higher
(larger batches, bigger `cols`). But it must not be quoted as an end-to-end win, and the honest
summary of this increment is: **correct, verified, and worth ~nothing at the shapes measured.**

### Standing gap after all work

**~1.68–1.72x behind llama.cpp on prefill** (from 2.31x), **1.26x on decode** (untouched).
The prefill improvement came from the repack cache being enabled (~1.25x) and Path 2 replacing
Path 1 (~1.11x). The quantiser vectorisation contributed nothing measurable at these shapes.

---

# SUMMARY / HANDOVER (2026-08-02)

## What exists now

| File | Role |
|---|---|
| `src/OpenTail.Stingray.Cpu/GemmPath.cs` | `GemmPath` enum + `GemmPathConfig`; reads `STINGRAY_GEMM_PATH` |
| `src/OpenTail.Stingray.Cpu/RepackedGemmPath2.cs` | the port: `block_q8_Kx4`, scalar + AVX2 quantiser, the GEMM |
| `src/OpenTail.Stingray.Cpu/SimdKernels.cs` | one dispatch hook in `TryMatMulBatchedQ4Kx8` (Path 1 untouched) |
| `src/OpenTail.Stingray.Core/KnownEnvironmentVariables.cs` | registered `STINGRAY_GEMM_PATH` |
| `tests/…Tests.ForwardPass/RepackedGemmPath2Tests.cs` | 21 tests, ~2.7 s |
| `scripts/bench-prefill-cli.ps1` | fixed: uses `-f` not `-p` (see the three-harness-bugs section) |
| `docs/repack-gemm/` | README (source study + verdict), this log, `ab-results.md` |

**Nothing is committed.** Path 1 is byte-for-byte unchanged.

## How to turn it on

```
STINGRAY_Q4KX8_CACHE_MB=2048     # REQUIRED — repacked path is off without it (Path 1 too)
STINGRAY_GEMM_PATH=2             # selects the port; 1 or unset = incumbent
```

`RepackedGemmPath2.EngagedCalls` proves it actually ran. Use it — a null A/B is ambiguous between
"no difference" and "never executed", and during this port it was the latter.

## Numbers (SmolLM2-1.7B Q4_K_M, Zen 3 6c/12t, AVX2 only, no OpenBLAS)

| measurement | value |
|---|---|
| Path 2 vs Path 1, isolated matmul | **1.83x** median |
| Path 2 vs Path 1, end-to-end prefill | **1.11x** |
| repack cache off → on (Path 1) | **1.25x** |
| prefill gap to llama.cpp | 2.31x → **~1.70x** |
| decode gap to llama.cpp | **1.26x**, untouched |

## Recommendations, in order of value

1. **Enable the repack cache by default** — 1.25x, no code change. Gated behind the validation at
   `ForwardPass.cs:59`: perplexity gate, greedy-parity checks, and a decision on the
   `PrefillPackedMulti_*` tolerance. That work, not more kernel work, is what puts 1.70x in users'
   hands.
2. **Reconsider the dequant-cache default** — `STINGRAY_PREFILL_DEQUANT_MB=0` is worth ~9% here
   because OpenBLAS is absent. Likely hardware-dependent; a probe beats a flat default.
3. **Path 2 as the repacked kernel** — a further 1.11x once (1) lands. Same test outcomes as Path 1.
4. Everything else is small.

## The GEMV: deliberately not ported, and why

The brief lists it. It is not done, and this is a decision rather than an omission:

- **It would be dead code.** Decode never reaches `TryMatMulBatchedQ4Kx8` — measured directly:
  decode is 28.40 t/s default vs 28.10 t/s with Path 2 + repack, i.e. unchanged. Decode runs the F32
  `MatVec` path. The GEMM is selected only for `M > 3`; the repacked path simply is not on decode's
  route.
- **Correctness no longer needs it.** Increment 8b's zero-padding handles `M <= 3` already.
- **It would reintroduce a hazard.** A second kernel for `M <= 3` means two numerics paths within
  one prompt again — the chunked-vs-unchunked divergence that padding just closed.

Making it useful would mean rewiring decode onto the repacked path, which is a separate change with
its own numerics review. If decode ever becomes the target, port the GEMV *then*, together with that
rewiring.

## Things this port got wrong (kept for calibration)

1. §7's "the int16 chain isn't transferable" — it was already in the tree.
2. "Path 2 has a real defect" — the 2 test failures are pre-existing; Path 1 + repack fails identically.
3. "MAJOR FINDING: repacked path is off by default" — true, but a documented deliberate default, not an oversight.
4. "Quantiser is ~3%, not worth porting" — it was worth 1.19x on the pass.
5. "Quantiser vectorisation will help end-to-end" — it did not, at all.

Items 1 and 3 are the same failure: **not reading the doc comments on code I was already editing.**
Items 4 and 5 are the same failure in opposite directions: **isolated measurements do not predict
end-to-end**, which `ForwardPass.cs:51` already said in plain English.

## Final verification (2026-08-02, end of port)

`dotnet test tests/OpenTail.Stingray.Tests.ForwardPass -c Release`

| configuration | result |
|---|---|
| defaults (Path 1, repack off) | **1191 / 1191 pass** |
| `GEMM_PATH=2` + `Q4KX8_CACHE_MB=2048` | 1189 pass, **2 fail** |

Total rose 1170 → 1191: the 21 new `RepackedGemmPath2Tests`.

The 2 failures are `ContinuousBatchingTests.PrefillPackedMulti_*`, which **Path 1 with the repack
cache fails identically** and which `ForwardPass.cs:55-60` documents in advance as an inherent
consequence of the repacked layout's summation order. They are not a defect in this port.

**A default-configured build is completely clean.** The port is opt-in and cannot affect anyone who
does not set both environment variables.

## Port status: COMPLETE

Every item in the brief is delivered except the GEMV, which is declined on recorded grounds (dead
code — decode measurably never reaches this path). Remaining opportunities are outside the port:

1. The `ForwardPass.cs:59` validation gate — perplexity + greedy parity + a tolerance decision.
   This is what converts ~1.70x from "two env vars away" into what users get. **Highest value.**
2. The dequant-cache default (~9% here; probably hardware-dependent — probe rather than flat default).
3. *Candidate, unmeasured:* both paths allocate their activation scratch per call
   (`NativeMemory.Alloc` in `TryMatMulBatchedQ4Kx8` / `TryMatMulBatched`). With ~5 projections x 24
   layers per chunk that is ~120 alloc/free pairs of 1-10 MB per prefill chunk. Pooling might help
   **both** paths equally, so it would not change the A/B — only absolute throughput. Given this
   session's repeated lesson that isolated gains do not translate, **measure the allocation share
   before writing a pool.**

---

# BREAKTHROUGH: FFN gate+up were bypassing the repacked path entirely

## The 2x2 that pointed at it

Measured both engines with repacking on and off, same prompts, same box:

| | OpenTail | llama.cpp | gap |
|---|---:|---:|---:|
| repack **OFF** | 80.3 t/s | 97.9 t/s | **1.22x** |
| repack **ON** | 97.5 t/s | 162.4 t/s | **1.67x** |
| repack win | **1.21x** | **1.66x** | |

Two things jump out:

1. **With repacking off we are only 1.22x behind.** The baseline engine is nearly at parity — so
   attention, norms, elementwise and dispatch overhead are *not* where the gap lives. That
   invalidates my ranked candidate list from an hour earlier, which put dispatch overhead first.
2. **The entire gap is repack-capture**: llama.cpp gets 1.66x from repacking, we get 1.21x.

Path 2 is ~4.7x over the non-repacked kernel in isolation but delivered 1.21x end-to-end. Too large
a shortfall for Amdahl alone.

## Cause

`ForwardPass.MatMulBatchedDualCached` (line 847) handles the FFN gate+up pair. It calls
`SimdKernels.TryMatMulBatchedDualQ8` **first**, and only falls through to `MatMulBatchedCached` —
the method that contains the repacked-path dispatch — if that declines.

**So FFN gate and up never reached the repacked path, and therefore never reached Path 2.**

For SmolLM2-1.7B (embDim 2048, intermDim 8192) the matmul FLOPs per token are roughly:

```
gate + up : 2 x 8192 x 2048 = 33.6M    <-- bypassed
down      : 1 x 2048 x 8192 = 16.8M
q,k,v,o   : (2048+512+512+2048) x 2048 = 10.5M
```

**Gate+up is ~55% of all matmul work**, and it was running on the older `_8In`-class kernel while
everything else got Path 2.

## Confirmation — one environment variable

`STINGRAY_DISABLE_DUAL_Q8=1` forces gate+up down the `MatMulBatchedCached` route:

```
DISABLE_DUAL_Q8=0 (default)  ->   93.2 t/s
DISABLE_DUAL_Q8=1            ->  141.5 t/s
```

**1.52x from a single flag.** Against llama.cpp's 162.4 t/s that is **~1.15x behind**, down from
1.67x.

## Why this was missed

`MatMulBatchedDualCached` has a long comment about mirroring `MatMulBatched`'s Q8 gate exactly, and
the history of a bug where `TryMatMulBatchedDualQ8` was called *unconditionally*. The fix made the
gate correct — but the dual path still takes precedence over the repacked path, and the repacked
path arrived later (iteration 42). Nobody re-examined the precedence when it did.

This is precisely the case the user flagged: **something checked before can still be the answer.**
The dual-Q8 gate was audited for correctness and left alone; the question of whether it should now
yield to a faster path was never asked.

## Status

Not yet a fix — `STINGRAY_DISABLE_DUAL_Q8=1` disables the dual path wholesale, which also gives
up its one-quantisation-pass-shared-across-two-weights advantage. The right change is to make
`MatMulBatchedDualCached` prefer the repacked path when it is available, and fall back to dual-Q8
only when it is not. Measuring first.

## The fix (not the flag)

`ForwardPass.MatMulBatchedDualCached` now prefers the repacked path when both weights are
repackable, falling through to dual-Q8 only when they are not:

```csharp
if (!useCache1 && !useCache2 && SimdKernels.Q8PrefillEnabled
    && GetRepackedQ4Kx8(in w1, rows, cols) != null
    && GetRepackedQ4Kx8(in w2, rows, cols) != null)
{
    MatMulBatchedCached(output1, in w1, input, N, rows, cols);
    MatMulBatchedCached(output2, in w2, input, N, rows, cols);
    return;
}
```

Both weights are required to be repackable: taking the repacked path for one and dual-Q8 for the
other would mix quantisation schemes inside a single FFN.

**Verified: 140.3 t/s with no env override** (vs 93.2 before, and 141.5 with the blunt
`STINGRAY_DISABLE_DUAL_Q8=1` flag). The code fix captures the win while keeping dual-Q8's
shared-quantisation advantage available wherever the repacked path is not.

Default-configured builds are unaffected: with `STINGRAY_Q4KX8_CACHE_MB` unset,
`GetRepackedQ4Kx8` returns null and the dual-Q8 gate runs exactly as before.

## Gap after the fix

| prompt | OpenTail | llama.cpp | gap |
|---|---:|---:|---:|
| 931 tok | 139.0 t/s | 159.1 t/s | **1.14x** |
| 2431 tok | 119.6 t/s | 144.8 t/s | **1.21x** |

**Prefill gap today: 2.31x -> ~1.14-1.21x.**

Full progression, all measured on the same box and model:

| step | gap |
|---|---:|
| start of day | 2.31x |
| repack cache enabled (Path 1) | ~1.85x |
| Path 2 replaces Path 1 | 1.67x |
| FFN gate+up routed through the repacked path | **1.14x** |

The last step — one `if` — was worth more than the entire kernel port. It was found by asking a
question the codebase had already answered once and never revisited.

## The fix also cleared the `PrefillPackedMulti_*` failures — and corrects a diagnosis in the code

After the precedence fix, **all three configurations pass 1191/1191** (two runs each for the first
two; the one stray failure seen in an earlier default run did not reproduce twice and was a flake):

| configuration | result |
|---|---|
| defaults | 1191 / 1191 |
| `Q4KX8_CACHE_MB=2048` + `GEMM_PATH=1` | 1191 / 1191 |
| `Q4KX8_CACHE_MB=2048` + `GEMM_PATH=2` | 1191 / 1191 |

Those two tests previously failed under **both** repacked paths, and `ForwardPass.cs:55-60`
attributes them to the repacked kernel's summation order:

> It is a NUMERICS change: the repacked kernel splits each row's int32 sum across 2 vector lanes
> where the row-major path uses 8, so the float summation order differs … That is enough to push
> `ContinuousBatchingTests.PrefillPackedMulti_*` past its packed-vs-sequential tolerance.

**That diagnosis appears to be wrong, or at least incomplete.** The likelier cause is the dual-Q8
gate's own threshold:

```csharp
&& SimdKernels.Q8PrefillEnabled && N >= SimdKernels.MinBatchForQ8Prefill
```

`PrefillPackedMulti_MatchesSequentialPrefill` compares prompts of 7 and 4 tokens run separately
against the same two packed into one N=11 pass. Those batch sizes straddle the threshold, so
gate+up took **different kernels depending on batch size** — a batch-size-dependent numerics path,
which is exactly the defect class the surrounding comments warn about. Routing gate+up through the
repacked path removes the threshold from the decision, the path becomes batch-size-independent, and
the tests pass.

Summation order clearly was not sufficient on its own to breach the tolerance, since the repacked
kernel is still in use in the passing runs.

### Why this matters beyond the tests

`ForwardPass.cs:59` names those failures as a prerequisite for flipping the repack default:

> Flipping this default needs the same treatment the Q8-prefill flip got: a perplexity gate plus
> greedy-parity checks, **and a decision on those tests' tolerance.**

**The tolerance decision is no longer needed** — the tests pass as written. What remains for the
default flip is the perplexity gate and greedy-parity checks, plus the two costs the comment lists
that are unaffected by any of this (a second copy of the Q4_K weights, and losing mmap sharing of
the GGUF pages).

This is the third time today that something previously examined and written off turned out to be
the answer. The pattern is consistent: each was a *correct* observation that stopped being true
when something else changed, and nobody re-ran the reasoning.

## CPU prefill phase breakdown — where the remaining gap lives

**Correction first: the profiler already existed.** `PrefillProfileTimers`
(`STINGRAY_PROFILE_PREFILL=1`, categories QkvProj / Attention / OutProj / Ffn / RmsNorm / RoPE /
Other) has been in `OpenTail.Stingray.Engine` all along. I said an hour earlier that CPU prefill had no
phase timing, having grepped only `KnownEnvironmentVariables.cs` and the CUDA files. Fourth instance
today of declaring something absent without searching properly.

2431 tokens, `Q4KX8_CACHE_MB=2048` + `GEMM_PATH=2`, dual-Q8 precedence fixed:

| phase | time | share |
|---|---:|---:|
| FFN (batched GEMM) | 10166 ms | **48.6%** |
| Attention (per-token, **NOT batched**) | 6348 ms | **30.3%** |
| QKV projection | 2892 ms | 13.8% |
| Output projection | 859 ms | 4.1% |
| RoPE | 335 ms | 1.6% |
| RmsNorm | 181 ms | 0.9% |
| Other | 142 ms | 0.7% |

Matmul total: **66.5%**, all now on Path 2.

### Implied location of the remaining ~1.14x

If Path 2's matmuls now match llama.cpp's (same algorithm, same layout, and our scale/min handling
is cheaper), then with matmul at 66.5% of our time:

```
our_total   = M + A          M = 0.665, A = 0.335   (fractions of our total)
their_total = our_total / 1.14 = 0.877
assume M' = M = 0.665   ->   A' = 0.877 - 0.665 = 0.212
A / A' = 0.335 / 0.212 = 1.58x
```

**Our attention is roughly 1.6x slower than llama.cpp's**, and it is 30% of prefill. That is where
the remaining gap is concentrated. The assumption `M' = M` is the weak link — if our matmuls are
still somewhat behind, attention's share of the blame drops correspondingly — but the direction is
clear and the profiler label corroborates it: attention is the one hot phase still running
per-token rather than as a batched GEMM.

### Why this is a credible target rather than another microbenchmark trap

The 30.3% is measured on the real workload at a real context length, not an isolated kernel. The
Amdahl arithmetic therefore uses a share that is already end-to-end. If attention were brought to
parity, the implied result is:

```
1 / (0.665 + 0.212) = 1.14x faster  ->  gap 1.14x -> ~1.00x
```

i.e. **parity**. Even halving the attention deficit would land near 1.07x.

Prior attention work exists and should be read before starting: perf-loop iteration 63 raised
`TokenTile` 8 -> 64 for +5.1% end-to-end / 1.22x on attention, and iterations 13/14/15/17/33 all
touched it. Flash attention was tried and rejected (it loses on CPU — it exists to work around GPU
shared-memory limits). **The untried lever is the one the profiler names: attention is not
batched.** llama.cpp computes prefill attention as batched GEMMs over all query positions; ours
walks tokens.

---

# Attention batching — attempted, measured, REVERTED

Acting on the phase breakdown (attention 30.3%, labelled "per-token, NOT batched"), the score phase
was batched: `SimdKernels.DotF32_4In` computes four query tokens against one key vector, sharing the
key load and collapsing four horizontal reductions into one.

Rationale was sound on paper. At headDim 64 a dot is 8 vector FMAs plus a reduction of ~4-5 ops, so
the reduction is roughly a third of the work, and it is repeated ~10^8 times per layer.

## Result 1: worth far less than predicted

Measured with FFN as an internal control, because the machine drifts between runs and FFN is
untouched by the change:

| | attn/ffn | 
|---|---:|
| before | 0.6244 |
| after (3 runs) | 0.6129 / 0.5741 / 0.5797 |

**~6% off attention, ~2% end-to-end.** Not the 1.6x the gap analysis implied was available.

**This localises attention's real bottleneck.** If batching the QK dots moves only 6%, the score
phase is not where attention's time goes — **phase 3, the weighted-V accumulation, is**. Phase 3
does a read-modify-write of the entire output head (64 floats: 8 loads, 8 FMAs, 8 stores) per
(token, KV position), against phase 1's single float store. Its memory traffic is ~16x phase 1's.

## Result 2: it broke four tests, in the way this codebase keeps warning about

```
failed ContinuousBatchingTests.PrefillPackedMulti_MatchesSequentialPrefill
failed ContinuousBatchingTests.PrefillWithCache_Chunked_MatchesFull
failed ContinuousBatchingTests.PrefillPackedMulti_ChunkedContinuation_MatchesFull
failed ContinuousBatchingTests.ContinuousBatchingEngine_ChunkedPrefill_MatchesUnchunked
```

`DotF32_4In` sums with one accumulator over 8-element strides; `DotF32` uses four accumulators over
32-element strides. Different order, different last bits. A 4-wide kernel can only cover
`t + 4 <= tn`, so the tile remainder falls back to `DotF32` — meaning **a token's arithmetic depends
on how many tokens share its tile, hence on N.** Chunked and unchunked prefill of the same prompt
then disagree.

This is precisely the defect class documented at `ForwardPass.cs:813-824` and re-documented in this
log at increment 8 — and I created it anyway, hours after writing that up. Recorded because the
pattern is evidently easy to reproduce: *any* kernel selected by a size threshold is a numerics
boundary unless every element takes the same path.

## Decision: reverted

Fixing the boundary means either padding the tile remainder, or making `DotF32_4In` bit-identical to
`DotF32` — which needs 4 accumulators per input (16 YMM registers) and four separate reductions,
removing most of the saving. Against a 2% ceiling, neither is worth it.

`DotF32_4In` is **kept in `SimdKernels`, unused**, with all of the above in its doc comment. It is
correct, and a restructured attention that batches *uniformly* could use it. `PrefillCoreAttention`
carries a comment recording why it is not wired in, so the next person does not re-derive this.

Suite back to **1191/1191**.

## Next target for attention, with the analysis

Phase 3, not phase 1. It is a GEMM: `scores[tn][endSeq] @ V[endSeq][headDim] -> out[tn][headDim]`.
The current loop accumulates into memory; a register-tiled version would hold the output tile in
YMM registers across the `i` loop, cutting output traffic from O(tn x endSeq x headDim) to
O(tn x headDim).

Two obstacles, both real:
- Register pressure: headDim 64 is 8 vectors per token, so only ~2 tokens fit in 16 YMM registers.
  Chunking headDim (e.g. 8 dims x 8 tokens) fits, and keeps total V traffic at one pass.
- `scores` is laid out `[t][i]`; a dim-chunked phase 3 wants `[i][t]` to read many tokens' weights
  contiguously at one `i`. Softmax needs `[t][i]`. A transpose between phases 2 and 3 may pay for
  itself, but it is not free.

**Whatever is tried, every token must take the same code path** — see above.

## Attention attempt 2: L1 set-conflict hypothesis — neutral, and a METHODOLOGY FAILURE

### The hypothesis (still believed correct as analysis)

Phase 3 accumulates into `batchAttnOut + (nBase + t) * qDim + h * headDim`. With 32 heads and
headDim 64, `qDim` is 2048 floats = **8192 bytes**, so consecutive tokens' output heads are exactly
8192 B apart. Zen 3's L1d is 32 KB / 8-way / 64 B lines = 64 sets, indexed by `address mod 4096`,
and **8192 mod 4096 == 0** — every token's output head maps to the same L1 sets. Phase 3 cycles all
`tn` tokens per KV position, so 64 lines compete for 8 ways.

Fix attempted: accumulate into a contiguous `TokenTile x headDim` scratch (16 KB), copy out once per
tile. Arithmetic and order untouched, so bit-identical by construction — chosen deliberately after
attempt 1 broke four tests purely by reordering sums.

### The measurement failure

First reading: attn/ffn 0.674 / 0.693 / 0.683 against a "baseline" of 0.6244 — reported to the user
as **~9% slower**, and reverted on that basis.

Then the reverted baseline was re-measured properly: **0.690 / 0.674 / 0.717.**

**The two are statistically identical.** The change was neutral. The "9% slower" was noise, because
the 0.6244 baseline was a *single sample* taken at a different time.

The attn/ffn ratio drifts across **0.62–0.72, a 16% spread**, while both attention experiments today
were chasing 6–9% effects. **Both measurements were below the noise floor of the method used.**
That also retracts attempt 1's "~6% improvement on attention" — equally unresolvable.

This is the same class of error as the `llama-bench` retraction at README section 0 and the
non-interleaved 2400-word comparison: **a difference was read off two numbers that were not
comparable.** The controls that worked all day for the matmul (interleave arms within one pass,
best-of, many samples) were not applied here, because using FFN as an internal control *felt* like
a control. It is not one: it corrects for whole-run drift, not for variance in the attention timer
itself.

### What is actually needed

`tools/attn-bench` already exists — it is what produced the TokenTile sweep recorded in
`PrefillCoreAttention`'s comment (tile 4/8/16/32/64/128/256 -> 0.70/1.00/1.17/1.35/1.48/1.34/1.04x,
"three independent runs"). **Attention work must go through it**, not through end-to-end phase
percentages. Building an isolated harness first is what made the GEMM work tractable; attention was
attacked without one and produced two unresolvable results in a row.

### Current state

Both attention attempts reverted. Suite **1191/1191**. `PrefillCoreAttention` carries a comment
recording the L1 analysis and that the scratch-buffer variant measured neutral, so it is not
re-tried. `SimdKernels.DotF32_4In` remains, unused, with its own findings documented.

### Standing analysis for whoever continues

Phase 3 costs 8 acc loads + 8 V loads + 8 FMAs + 8 stores per (token, KV position) — **4 uops per
useful FMA**. Neither attempt addressed that; attempt 1 targeted phase 1, attempt 2 moved where the
loads went without reducing their number. The uop count is the thing to attack, which requires the
accumulator in REGISTERS across the `i` loop — a token-group x dim-chunk restructure. Estimated
~1.9x on phase 3 if it works, and the ascending-`i` order can be preserved, so it need not change
numerics. **Measure it in `tools/attn-bench`.**

---

# Continuation: phase-3 register tiling — SHIPPED IN TREE, +7.5-8.8% END-TO-END

## Mechanism

The standing analysis was right about the mechanism, but high on its isolated estimate. Phase 3 is
now loop-interchanged into an **8-token x 8-float register microkernel**. Eight YMM accumulators stay
live across the ascending KV-position loop; one V vector feeds eight token outputs, and each output
chunk is stored once. The arithmetic for every output lane is still the same ascending sequence of
FMA operations, so this is loop interchange rather than reassociation.

Because `PagedKvCache.ValueAtHead` includes a page-table lookup and values cross a page every 16
positions, the implementation resolves one V-row pointer table per head before the tiled loop. The
register path is used only with FMA and a head dimension divisible by eight; every other shape keeps
the prior implementation. `STINGRAY_PREFILL_ATTN_REGISTER_VALUES=0` disables it for controlled
A/B measurement; enabled is the default.

## Isolated method — and another stale assumption caught

`tools/attn-bench` had quietly become stale: it still used `ctxLen` as the score-row stride, while
production had already changed to the actual prefill extent (`startPos + N`). That changes both the
scratch size and cache-set mapping. The harness was corrected before accepting a result, then given
a production-faithful `PagedKvCache` mode and alternating A/B order per repetition.

The first 8-register JIT retained seven `active > k` branches inside every hot-loop iteration. JIT
disassembly proved that the accumulators themselves did stay in YMM0-YMM7 with no spills. Splitting
out the overwhelmingly common full-eight-token group removed those branches without changing any
FMA. Final paged-cache results, N=3218, nine round-robin samples per arm:

| independent run | shipped best | register best | ratio | max relative error |
|---:|---:|---:|---:|---:|
| 1 | 450.6 ms | 376.8 ms | **1.20x** | **0** |
| 2 | 415.2 ms | 354.4 ms | **1.17x** | **0** |
| 3 | 394.8 ms | 335.1 ms | **1.18x** | **0** |

This is whole attention, not phase 3 alone. The original ~1.9x estimate was therefore too high as a
whole-attention expectation, but the effect is far above the 16% noise spread that invalidated the
two end-to-end attention-timer attempts.

## End-to-end gate — same binary, interleaved arms, five samples each

`Q4KX8_CACHE_MB=2048`, `GEMM_PATH=2`, `DOTNET_TC_QuickJitForLoops=0`; A/B selected only by the
registered attention switch. Arm order alternated each repetition.

| prompt | register off (t/s) | register on (t/s) | best ratio | median ratio |
|---|---|---|---:|---:|
| 931 tok | 140.9 / 141.8 / 142.7 / **144.9** / 138.1 | 142.0 / **155.8** / 151.5 / 150.6 / 149.3 | **1.075x** | **1.062x** |
| 2431 tok | 116.0 / 120.9 / 119.7 / **121.1** / 119.8 | 127.9 / **131.8** / 127.5 / 129.0 / 125.2 | **1.088x** | **1.068x** |

Unlike attempts 1 and 2, this result does not compare an old single baseline against later samples:
both arms ran in one interleaved pass, with best-of as the primary estimator.

## Fresh llama.cpp reference (indicative, cross-harness)

Fresh best-of-three `llama-completion -st --simple-io -t 6` results were 162.6 t/s for the same
900-word file and 153.66 t/s for the 2400-word file. llama.cpp tokenizes those as 909 / 2409 tokens,
while OpenTail reports 931 / 2431, so this remains an indicative cross-harness comparison:

| prompt file | OpenTail register best | llama.cpp best | remaining gap |
|---|---:|---:|---:|
| 900 words | 155.8 t/s | 162.6 t/s | **1.04x** |
| 2400 words | 131.8 t/s | 153.66 t/s | **1.17x** |

The prior recorded gap was 1.14x / 1.21x at the two sizes. The short-prompt gap is now nearly closed;
the long-prompt gap remains larger and is the next performance target.

## Measurement tooling correction

`scripts/bench-prefill-cli.ps1` used `[int]($samples.Count / 2)` for its median index. PowerShell
rounds an `[int]` conversion, so with the default three runs it converted 1.5 to 2 and reported the
maximum as the "median". It now uses `Math.Floor` explicitly. Best-of results were unaffected.

## Correctness status

- Release builds: CPU, Engine, CLI — clean, zero warnings.
- The 13 `ContinuousBatchingTests` — **13/13**, including the four tests broken by attempt 1.
- Full suite, defaults — **1191/1191**.
- Full suite, `Q4KX8_CACHE_MB=2048` + `GEMM_PATH=1` — **1191/1191**.
- Full suite, `Q4KX8_CACHE_MB=2048` + `GEMM_PATH=2` — **1191/1191**.

Final-build runner note: after the last non-arithmetic cleanup (moving V-pointer initialisation under
the existing `try/finally`), the default runner twice produced 1190/1191 because
`ContinuousBatchingConstraintTests.ConstrainedAndUnconstrained_Coexist_PerSequenceMasking` observed
one completion instead of two. Its complete class passed 4/4 alone, and the full final-build suite
passed **1191/1191 with `--parallel none`**. This is recorded as a runner-order/shared-state flake;
the attention-sensitive `ContinuousBatchingTests` stayed green throughout.

## 2026-08-02 — locating the final CPU prefill gap: llama-shaped Flash attention

This pass restarted from matched raw prompts and treated the earlier attention readings as
unresolved. `tools/attn-bench` was extended with round-robin phase profiling before another
production change was attempted. On the production-shaped paged/register kernel, aggregate
head-worker time was stable enough to locate the work:

| exact prompt tokens | QK | softmax | weighted V |
|---:|---:|---:|---:|
| 900 | 64–68% | 3–4% | 28–34% |
| 2400 | 62–66% | 3–5% | 31–35% |

The remaining attention work was therefore not ordinary softmax and not the already-fixed
register-V loop. llama.cpp's CPU Flash path differs structurally: 64x64 Q/KV tiles, a 6-row by
2-YMM FP32 GEMM microkernel, online softmax, then the same GEMM shape for probabilities times V.

Reproducing the pieces in isolation found three distinct results:

1. Replacing only QK with the 6x2 microkernel reduced the 2400-token QK phase by roughly 18–22%.
2. Online softmax with scalar `MathF.Exp` was a false start: its exp/rescale phase became 42–45%
   of the paired-GEMM kernel and erased almost all of the matrix-product win.
3. Reusing production's AVX2 exp approximation made the complete paired-GEMM online kernel
   **1.22x faster at 900** and **1.51x faster at 2400** than flat register-V attention. Maximum
   absolute output error versus the materialised-softmax reference was about **6.3e-6**.

### Guarded production result

The production A/B is opt-in with `STINGRAY_PREFILL_ATTN_FLASH64=1`, restricted to AVX2+FMA,
`headDim=64`, and `N >= 256`. Short inputs keep the incumbent because 64x64 packing loses below
the measured crossover and because packed multi-prefill currently uses a separate per-token
attention implementation.

An important production-only trap was caught after the first A/B: transposed K packing called
`PagedKvCache.KeyAt` inside both the dimension and key loops — 4096 page lookups per 64-key block.
Resolving 64 K-row pointers once, plus vectorising score scale-and-max, changed the result from a
6–7% end-to-end win to the final numbers below. Arms were interleaved, raw prompts were exact, and
both used `DOTNET_TC_QuickJitForLoops=0`, `Q4KX8_CACHE_MB=2048`, `GEMM_PATH=2`, and
`PREFILL_DEQUANT_MB=0`:

| tokens | incumbent samples (t/s) | Flash64 samples (t/s) | best gain | median gain |
|---:|---|---|---:|---:|
| 900 | 141.4 / 148.3 / 144.9 | 151.7 / 153.5 / 157.3 | **1.061x** | **1.059x** |
| 2400 | 124.8 / 126.1 / 122.8 | 143.9 / 142.0 / 144.2 | **1.144x** | **1.153x** |

Fresh matched llama.cpp references from the same investigation were 163.70 t/s at 900 with FP32
KV (165.02 with its default F16 KV) and 149.8 t/s at 2400 with FP32 KV. The guarded OpenTail path
is therefore still about **4–5% behind**, not faster, but the former ~10% short-prompt / ~19%
long-prompt gaps have both moved to near parity.

Correctness/build gate: Release CLI builds with zero warnings; the attention-sensitive tests pass
13/13 with the flag; the complete ForwardPass suite passes **1191/1191** with the flag and
`--parallel none`.

### What is plausibly left

At this scale the remaining cross-runtime difference is close to the machine's run-to-run spread,
so another claimed 2–4% needs interleaved evidence. The highest-value structural difference is
scheduling: OpenTail assigns one complete head per `Parallel.For` iteration (32 large, equal jobs
on 12 logical processors), whereas llama.cpp dynamically schedules many query-row/head tiles.
That avoids the unavoidable 3-wave versus ideal 2.67-wave tail in a 32/12 split. Secondary costs
are per-head native scratch allocation on every layer and repeated K/V packing. These are targets
to measure, not findings to assume.

## 2026-08-02 — Flash64 scheduling follow-up and default decision

The remaining cross-runtime gap was re-measured before changing code. OpenTail and llama.cpp were
strictly interleaved within one pass, with a discarded warmup and three samples per engine/size.
OpenTail used `DOTNET_TC_QuickJitForLoops=0`, `Q4KX8_CACHE_MB=2048`, `GEMM_PATH=2`,
`PREFILL_DEQUANT_MB=0`, and Flash64. llama.cpp used `llama-completion -st --simple-io -t 6`.

| prompt | OpenTail samples | llama.cpp samples | best-of gap |
|---|---|---|---:|
| 900 words | 154.7 / 160.0 / **163.1** | **177.78** / 169.76 / 171.38 | **1.09x** |
| 2400 words | **153.4** / 142.0 / 148.3 | 151.22 / **156.41** / 135.96 | **1.02x** |

The earlier 4–5% figure therefore did not reproduce as one stable number: the short prompt was
about 9% behind while the long prompt was about 2% behind, and llama.cpp itself varied by 15% at
2400 words. This remains an indicative cross-harness comparison because tokenisation differs
(931/2431 OpenTail tokens versus 909/2409 llama.cpp tokens).

### Scheduling and scratch were separated in the isolated harness

`tools/attn-bench` compared three round-robin arms over identical arithmetic: one complete head per
job with per-head scratch, query-tile jobs with per-job scratch, and query-tile jobs with
per-thread scratch. Nine measured repetitions followed twelve warmup rounds.

| exact N | head jobs | tile jobs / per-job scratch | tile jobs / thread scratch | schedule-only gain |
|---:|---:|---:|---:|---:|
| 900 | 23.4 ms | **15.2 ms** | 15.3 ms | **1.54x** |
| 2400 | 122.0 ms | **91.5 ms** | **91.5 ms** | **1.33x** |

At N=900, the scheduling trace reported 64.6% observed schedule occupancy for head jobs versus
88.8% for query-tile jobs. Scratch reuse contributed no measurable improvement at either size;
the win was scheduling, not allocation. The trace's occupancy figure is an interval-overlap
measure, not literal CPU utilisation: at N=2400 it can exceed 100% because stopwatch intervals
include time while a managed worker is descheduled.

### The isolated scheduling win mostly disappeared in production

The production implementation put both schedules behind a registered same-binary switch and made
both call the same extracted tile worker. An old binary was retained as a third control; the
refactored head schedule was neutral against it. The stronger five-sample two-arm confirmation was:

| prompt | head-job samples | query-tile samples | best ratio | median ratio |
|---|---|---|---:|---:|
| 900 words | 155.2 / 129.9 / 139.4 / 148.5 / **157.2** | 149.4 / 119.7 / 145.2 / 135.9 / **155.8** | **0.991x** | **0.978x** |
| 2400 words | 141.7 / 146.1 / 145.6 / **151.4** / 147.4 | 140.3 / 148.7 / **154.5** / 153.4 / 152.3 | **1.020x** | **1.042x** |

Verdict: query-tile scheduling is neutral/slightly negative at 900 and only +2.0% by the primary
best-of metric at 2400. That is below this machine's end-to-end noise floor. It remains available
as `STINGRAY_PREFILL_ATTN_FLASH64_TILE_JOBS=1`, but is not enabled by default. This is another
case where a large isolated improvement does not transfer through the full workload.

### Flash64 itself passes the default-on bar

A final same-binary, interleaved Flash64 off/on gate used query-tile scheduling on both applicable
Flash64 runs (so it does not exaggerate the 900-token result):

| prompt | Flash64 off | Flash64 on | best gain |
|---|---|---|---:|
| 900 words | 144.7 / 142.4 / **148.4** | **154.6** / 153.8 / 152.4 | **1.042x** |
| 2400 words | 126.5 / **133.6** / 130.0 | 149.5 / **153.8** / 150.2 | **1.151x** |

Two new production tests force the path above its 256-token gate. Head and tile scheduling are
bit-identical at 320 tokens, and a 512-token call matches two 256-token chunks within the existing
numerical contract with the same greedy token. Validation results:

- New Flash64 scheduling/parity tests: **2/2**.
- `ContinuousBatchingTests` with Flash64 + tile jobs + Path 2: **13/13**.
- Full ForwardPass suite with Flash64 + tile jobs + Path 2: **1193/1193**.
- Release CLI build: zero warnings/errors.
- Full ForwardPass suite with the new defaults and no performance environment variables: **1193/1193**.

Decision: Flash64 is now enabled by default on its existing guarded shape (AVX2+FMA, `headDim=64`,
single common head dimension, and `N >= 256`). `STINGRAY_PREFILL_ATTN_FLASH64=0` is the explicit
fallback. Query-tile scheduling remains opt-in pending a result larger than measurement noise.

---

# OPEN LEADS (carried forward — 2026-08-02/03)

## Lead A — CPU KV cache is F32; llama.cpp's is F16 (2x memory)

`PagedKvCache` stores `float*[][]` with `_pageBytes = PageSize * _kvDim * 2 * sizeof(float)`.
The bf16 / q8_0 values of `STINGRAY_KV_DTYPE` are **CUDA-only** (`RunCommand.cs:123`).

```
KV/token = layers x kvDim x 2 x 4 = 24 x 2048 x 2 x 4 = 384 KiB
at 8192 ctx                                            = 3.0 GiB
llama.cpp, same model/context (measured)               = 1536 MiB  <- exactly half, F16
```

**1.5 GiB avoidable, larger than the ~1 GB repack cache that buys 1.8x.** Lazily paged, so only
used tokens are charged, but at long context it is the dominant allocation.

Why it may be free speed rather than a trade:
- **BF16 widening is a zero-extend + 16-bit shift** — no F16C, which .NET does not expose (verified
  during the Path 2 port when mapping `_mm256_cvtph_ps`). F16 would need software conversion; BF16
  does not. BF16 is therefore the natural choice here, not F16.
- **Attention streams the whole K and V caches** (phases 1 and 3) and is 30.3% of prefill. Halving
  element size halves that traffic, so narrowing could make attention *faster*.
- Precedent in-tree: `CudaForwardPass.cs:121-125` — "Arithmetic stays fp32 in the kernels; only the
  *store* is narrowed, so decode is argmax-stable vs fp32 KV."

Requires a perplexity gate (numerics change). **Measure in `tools/attn-bench` before writing the
storage change** — "memory-bound so narrowing helps" is a hypothesis, and two hypotheses of that
shape were refuted on 2026-08-02.

## Lead B — row-major fallback: Q8_KS -> Q8_K (user asked to keep this on the list)

Switch the non-repacked Q4_K kernels from Q8_KS activations (eight sub-scales per super-block) to
Q8_K (one). `SimdKernels.cs:6146` already names the win:

> "The remaining cvt+fma per sub-block is forced by Q8_KS carrying eight activation scales per
> super-block; Q8_K's single scale would let the int32 accumulate across all eight sub-blocks
> instead."

Op count in `AccumQ4KInput`: ~32 vector ops per super-block today, ~25 with Q8_K — roughly 1.28x on
the dot kernel, plausibly ~1.1x end-to-end.

**The accuracy objection that presumably kept Q8_KS is now dead.** Path 2 uses Q8_K and measured
*better* than Path 1's Q8_KS on wikitext-2 (PPL 16.0484 vs 16.0870), on every position bucket.

Ceiling is known and modest: our row-major path is already **1.11-1.16x** off llama.cpp's
`--no-repack` equivalent (88.3 vs 102.4 at 931 tok; 85.0 vs 94.8 at 2431), so 11-16% is all that is
available. Touches several kernels (`_2In`/`_4In`/`_8In`/scalar, plus the Q3_K/Q5_K/Q6_K siblings
share the shape) and needs its own perplexity gate.

Priority: real but narrow — the repack-budget sweep shows the fallback is only reached below
~128 MB of budget on this model.

## Lead C — CLOSED: lazy release of source weights after repacking. Do not build.

The repack budget sweep (2431 tokens, best-of-2):

```
budget     0 MB ->  85.2 t/s        512 MB -> 119.9 t/s  (1.41x)
budget    64 MB ->  85.4 t/s       1024 MB -> 150.9 t/s  (1.77x)
budget   128 MB ->  92.8 t/s       4096 MB -> 154.9 t/s  (1.82x)
budget   256 MB -> 100.1 t/s
```

**Smooth and saturating, no cliff**, complete at ~1 GB (1024 -> 4096 MB is +2.7%). A constrained box
does not need a release mechanism — it needs a smaller budget, and gets proportional speed.
Declining a tensor is free and correct, so degradation is graceful by construction.

Additionally the source weights are clean file-backed mmap pages: the OS reclaims them for free,
with no writeback, precisely when memory is tight. An idle-triggered release would re-implement
that worse, since the kernel knows when memory is scarce and we do not.

## Retracted: "the dequant cache costs ~9%"

Re-measured with repacking forced off and current defaults: 88.3 -> 88.8 t/s (931 tok) and
85.0 -> 86.0 (2431 tok) — **~1%, i.e. noise.** The original +9% was taken before flash64 and before
the dual-Q8 precedence fix, when the phase mix was entirely different. Withdrawn.
