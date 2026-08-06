# llama.cpp repacked Q4_K GEMM — source study and A/B design

**Status:** COMPLETE — source research + measurement done. **Verdict: do not port. See §7.**
**Question:** how much of our remaining ~2.31x CPU-prefill gap to llama.cpp is the repacked 2D
register-tiled GEMM, versus ordinary per-row dot-kernel codegen and orchestration?
**Reference tree:** `examples/cpp/llama.cpp`, binaries `extensions/OpenTail.Stingray/tools/llama.cpp`
(b8585, `cad2d3884`, Clang 19.1.5, `ggml-cpu-haswell.dll`)
**Host:** Zen 3, 6c/12t, AVX2, **no AVX-512, no VNNI**

---

## 0. Retraction — read this before trusting any earlier repack number

An earlier A/B of this question was run with `llama-bench` and `LLAMA_ARG_REPACK`. **It was invalid.**
`llama-bench` builds its model params from `llama_model_default_params()`
(`tools/llama-bench/llama-bench.cpp:1230`) and never calls `common_init_from_params` — which is the
only code path that reads `no_extra_bufts`. `use_extra_bufts` is hardcoded `true` in
`src/llama-model.cpp:2330`. **Repack was ON in both arms.** The reported figures

```
repack "ENABLED"  : pp931 184.81 ± 4.43 | pp4831 138.73 ± 1.84
repack "DISABLED" : pp931 188.69 ± 5.67 | pp4831 130.81 ± 5.31
```

measured identical code paths. The pp931 pair even moves the "wrong" way, and the pp4831 delta sits
inside its own ±5.31 sd. **The conclusion drawn from it — that the repacked GEMM is worth only
0–6% — is withdrawn and must not be cited.** Nothing about the phase-2 premise was refuted.

`llama-cli` *does* honour the flag: `common/common.cpp:1564` sets
`mparams.use_extra_bufts = !params.no_extra_bufts`, and `common/arg.cpp:2283-2286` defines
`--repack, -nr, --no-repack` (env `LLAMA_ARG_REPACK`).

---

## 1. What the flag actually switches

`make_cpu_buft_list` (`src/llama-model.cpp:887`). When `use_extra_bufts` is true, lines 919-935
query the CPU backend for `ggml_backend_dev_get_extra_bufts` and append the repack buffer type to
the list **before** the plain CPU buffer type (added at 938+). Buffer-type order is priority order,
so any weight the repack buft claims is allocated there and physically rewritten at load time by
`tensor_traits::repack` (`ggml/src/ggml-cpu/repack.cpp:4519`).

With the flag off the extra buft is simply absent from the list, so those weights land in the
ordinary CPU buffer and are executed by the ordinary `ggml_compute_forward_mul_mat`.

**Consequence for the experiment:** this is a *load-time weight-layout decision*, not a runtime
kernel toggle. There is no per-op branching, no mixed state, no warm-up dependence. That makes the
A/B unusually clean — the two arms differ in exactly one structural property.

### Proof signal

`ggml_backend_cpu_repack_buffer_type_get_name` returns the literal `"CPU_REPACK"`
(`repack.cpp:4746`). llama.cpp prints per-buffer allocation at load in **normal** output:

```
load_tensors:   CPU_REPACK model buffer size =   xxx.xx MiB
load_tensors:          CPU model buffer size =   yyy.yy MiB
```

Presence of the `CPU_REPACK` line proves the flag took effect. Its size relative to the `CPU` line
directly quantifies the dilution described in §2 — no debug build required. (There is also a
`GGML_LOG_DEBUG("repack tensor %s with %s_%dx%d")` at `repack.cpp:4520`, but that needs debug
logging compiled in; prefer the buffer-size line.)

---

## 2. On AVX2, only Q4_K repacks — this bounds the measurable delta

`ggml_repack_get_optimal_repack_type` (`repack.cpp:4528`) dispatches on tensor type **and ISA**:

| Type | AVX2 path on x86? | Selected traits | Source |
|---|---|---|---|
| Q4_K | **yes** | `q4_K_8x8_q8_K` | 4600-4607 |
| Q5_K | no — NEON only (`matmul_int8` / `dotprod`) | — | 4644-4654 |
| Q6_K | no — NEON only | — | 4655-4665 |
| Q2_K | no — AVX-512 only | — | 4557+ |
| Q4_0, IQ4_NL, Q8_0, MXFP4 | yes | `*_8x8_q8_0` etc. | 4573+, 4667+ |

Gate is `ggml_cpu_has_avx2() && cur->ne[1] % 8 == 0`. SmolLM2-1.7B's output-feature counts
(2048 hidden, 512 GQA KV, 8192 FFN, 49152 vocab) are all `% 8 == 0`, so every Q4_K tensor
qualifies.

**But `Q4_K_M` is a mixed quant.** Its Q6_K tensors take the *identical* ordinary path in both
arms. Any measured delta is therefore diluted by the Q6_K (and F32 norm) share of the work, and the
measured number is a **lower bound** on the repacked GEMM's true per-Q4_K-tensor value. The
`CPU_REPACK` vs `CPU` buffer sizes give the byte-weighted dilution factor needed to un-dilute it.

---

## 3. The GEMM only runs for M > 3 — read the A/B on prefill only

`forward_mul_mat_one_chunk` (`repack.cpp:4204`), dispatch at 4240-4250:

```c
// If there are more than three rows in src1, use gemm; otherwise, use gemv.
if (nrows > 3) {
    gemm<BLOC_TYPE, INTER_SIZE, NB_COLS, PARAM_TYPE>(ne00, ..., nrows - (nrows % 4), ncols);
}
for (int iter = nrows - (nrows % 4); iter < nrows; iter++) {
    gemv<BLOC_TYPE, INTER_SIZE, NB_COLS, PARAM_TYPE>(ne00, ..., 1 /* nrows */, ncols);
}
```

GEMM consumes M in multiples of 4; the remainder falls to GEMV. So:

- **Prefill** (M = token count, hundreds–thousands) → overwhelmingly GEMM.
- **Decode** (M = 1) → purely GEMV.

Only `prompt eval time` is informative for this question. Decode t/s measures GEMV and must not be
mixed in.

Corroborating detail: `forward_mul_mat` quantises src1 in **groups of 4 rows** into `block_q8_Kx4`
(`repack.cpp:4301`, `ggml_quantize_mat_t<INTER_SIZE, PARAM_TYPE>(..., 4, ne10)`), with a scalar
`from_float` tail for the `ne11 % 4` remainder (4306). That matches the GEMM's
`const block_q8_Kx4 * a_ptr_start = (const block_q8_Kx4 *) vy;` at `arch/x86/repack.cpp:2067`.

---

## 4. The `--no-repack` fallback is *not* a fast path — this is why the experiment is worth running

Two candidate fallbacks exist; only one is real here.

**tinyBLAS is declined.** `ggml/src/ggml-cpu/llamafile/sgemm.cpp` handles only F32 (3723), BF16
(3787), F16 (3851), Q8_0 (3935), Q4_0 (3972), Q5_0 (4009), IQ4_NL (4025). **There is no Q4_K case.**
So `llamafile_sgemm` returns false and control reaches `UseGgmlGemm2` (`ggml-cpu.c:1387`).
(Note `ggml-cpu.c:45-47` also `#undef`s `GGML_USE_LLAMAFILE` on ARM SVE/i8mm — irrelevant on x86.)

**The real fallback** is `ggml_compute_forward_mul_mat_one_chunk` (`ggml-cpu.c:1164`): a 16×16
cache-blocked loop wrapped around scalar `vec_dot` calls.

```c
const int64_t blck_0 = 16;        // ggml-cpu.c:1202
const int64_t blck_1 = 16;        // ggml-cpu.c:1203
...
for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ir0 += num_rows_per_vec_dot) {
    vec_dot(ne00, &tmp[ir0 - iir0], ..., src0_row + ir0 * nb01, ..., src1_col, ..., num_rows_per_vec_dot);
}
```

On AVX2 `num_rows_per_vec_dot == vec_dot_num_rows == 1` — the 2-row variant is the ARM mmla path.
So it performs **one `ggml_vec_dot_q4_K_q8_K` per (output row, token) pair**, with no register-level
reuse of the unpacked B tile across tokens. The 16×16 tiling buys cache locality only; it does not
amortise the 4-bit unpack or the scale/min reconstruction.

**That is structurally the same shape as our managed kernel.** `--no-repack` therefore isolates
precisely the open question: what does weight repacking plus 2D register blocking buy *on top of* a
cache-tiled per-row dot loop? The measured delta is a direct estimate of the remaining "boss tower";
our separately-measured ~1.68x per-iteration codegen gap (perf-loop iteration 62) sits *underneath*
it, not inside it.

---

## 5. Why the repacked GEMM should win, mechanically

`ggml_gemm_q4_K_8x8_q8_K` (`arch/x86/repack.cpp:2042`, ~1450 lines to 3492). Fixed shape parameters
at the top — the "used parameters" that explain the whole structure:

```c
const int qk = QK_K;              // 256
const int nb = n / qk;            // super-blocks along K
const int ncols_interleaved = 8;  // 8 B-columns interleaved per repacked block
const int blocklen = 8;
assert (nr % 4 == 0);             // M must arrive in groups of 4  -> block_q8_Kx4
assert (nc % ncols_interleaved == 0);
```

Both ISA arms exist in one function:

- **AVX-512 arm** (`#if defined(__AVX512BW__) && defined(__AVX512DQ__)`, from 2077): processes
  **16 B-columns × 4 token rows** per pass — `__m512 acc_rows[16]` plus `__m512 acc_min_rows[16]`
  (2098-2106), walking two `block_q4_Kx8` pointers (`b_ptr_0`, `b_ptr_1`) at once.
- **AVX2 arm**: the `anc`/`m4bexpanded`/`_mm512_*` machinery above is compiled out; the 256-bit path
  handles 8 B-columns × 4 token rows.

**On this Zen 3 box only the AVX2 arm is live.** Roughly half of that 1450-line function is dead
code here. That materially lowers the cost of any future port — and it means a large measured delta
would be achievable with 256-bit intrinsics we already use.

The structural win is B-tile reuse: unpack, mask (`m4b`), permute (`requiredOrder`,
`_mm256_permutevar8x32_epi32`) and scale each 8-column tile **once**, then accumulate it against 4
token rows. Ceiling is ~4x on unpack/scale/min-correction work; the `madd` FLOPs themselves are paid
by both designs. Note the separate `acc_min_rows` accumulators — llama.cpp keeps the Q4_K min
correction in the same fused pass, which is the same term our iteration-64 vectorisation
(+18.8% prefill) addressed on the per-row side.

---

## 6. A/B design and how to read the result

Runner: `scripts/repack-ab.ps1` → appends to `ab-results.md`.

Controls: same binary, same model, same `-c 8192`, `-t 6` (physical cores), warmup **enabled** so
the timed prefill is not paying first-touch page faults, `-no-cnv`, `-n 1`. Arms interleaved
on/off/on/off so thermal or background drift hits both equally. Best-of, because interference on
this box is one-sided (it can only slow a run down).

| Measured `repack ON / OFF` prefill ratio | Reading |
|---|---|
| ≥ 2x | The repacked 2D GEMM **is** the boss tower; phase-2 premise holds. |
| 1.2–2x | Split target: GEMM matters but so does per-iteration codegen. |
| ≤ 1.2x | Tower is mostly ordinary `ggml_vec_dot_q4_K_q8_K` codegen + threading (the ~1.68x gap) — a much cheaper target than a 1450-line port. |

In all cases the raw ratio is a **lower bound** (§2 dilution). Divide out the Q4_K byte share from
the `CPU_REPACK` / `CPU` buffer sizes before comparing against the 2.31x end-to-end gap.

### Open item not resolvable from source

The Q4_K vs Q6_K **byte** split of `SmolLM2-1.7B-Instruct-Q4_K_M.gguf`. The load-time
`llama_model_loader: - type q4_K: N tensors` census plus the two buffer-size lines settle it; the
script captures both.

---

## 7. Verdict (measured 2026-08-02) — do not port

Full data in `ab-results.md`. Gate passed: `CPU_REPACK` present with repack on, absent with
`--no-repack`, Host model 1005 MiB in both arms. Census confirms §2 exactly — **144 q4_K tensors
repack; 25 q6_K + 49 f32 do not** (729 of 1005 MiB = 72.5% of weight bytes).

| tokens | repack ON | repack OFF | ratio |
|---:|---:|---:|---:|
| 909 | 163.72 t/s | 98.64 t/s | **1.66x** |
| 4709 | 126.37 t/s | 83.90 t/s | **1.51x** |

### The gap decomposes into two roughly equal halves

Our measured gap to llama.cpp is 2.31x (llama-bench, which always runs repacked — consistent with
the ON arm here). At 4709 tokens:

```
total gap            2.31x
  repacked GEMM      1.51x
  everything else    1.53x     (2.31 / 1.51 — codegen, threading, orchestration)
```

**There is no single "boss tower."** That framing was wrong and is retracted along with §0's
llama-bench result. A *perfect* port of `ggml_gemm_q4_K_8x8_q8_K` — ~700 lines of live AVX2
intrinsics (the AVX-512 arm is dead on this host), matching kernels tuned over years — would move
us from 2.31x behind to **1.53x behind**. It does not reach parity. Parity requires closing both
halves, not one.

### The trend runs against the port

The ratio *falls* with context length: **1.66x at 909 tokens, 1.51x at 4709**. Attention is not a
Q4_K matmul, so it takes a growing share as context grows and dilutes the GEMM's contribution
further. Extrapolating, the port is worth even less at 16k — and long context is the real workload.

### Decision

Do not port. Reasons, in order of weight:

1. **It does not achieve the goal.** Best case leaves a 1.53x residual gap.
2. **The goal is an explicit non-goal.** Plan §1.2: *"Replacing llama.cpp on portability, hardware
   coverage or universal serving throughput."* Plan §17: *"This track is not a session
   prerequisite… A benchmark win alone cannot silently make native the default."*
3. **Sessions dominate it.** The product's headline demo is *zero retained-token prefill*. Caching
   eliminates prefill work; a kernel merely accelerates it. On a context revisited N times,
   sessions save ~(1-1/N) of prefill; this port would save 34% of what remains.
4. **Opportunity cost.** 64 perf-loop iterations are done; 0 of ~200 session-plan items are.

### What is still true and worth keeping

Managed C# CPU prefill sits ~2.3x off a mature hand-tuned C++ engine, and ~1.5x off it with
llama.cpp's headline kernel disabled. That is a credible-engine result, not an embarrassing one,
and it is already banked. Iteration 64's Q4_K min-correction vectorisation (+18.8%) was cheap and
real. Stop here and record it.

### Measurement notes

- The ON arm warms across runs (145.0 / 159.6 / 163.7 and 115.5 / 124.4 / 126.4); the OFF arm
  stabilises faster. Best-of absorbs this, but use ≥3 runs if repeating.
- The "un-diluted" column in `ab-results.md` uses byte share as a proxy for time share. Since
  attention consumes time but no weight bytes, the true time fraction is below 72.5%, which makes
  that column a *lower* bound on the kernel's intrinsic speedup. It does not affect the decision —
  the end-to-end ratio is what a port would actually buy.
- Three environment traps cost several runs and are now encoded in `scripts/repack-ab.ps1`:
  `llama-bench` ignores the flag; `llama-cli` is interactive-only in b8585; and without `-st`,
  `llama-completion` hits stdin EOF and raises a console interrupt that kills the parent shell
  when it shares a console (`-NoNewWindow`).

---

## 8. Sub-experiment: activation interleaving (measured 2026-08-02) — not worth shipping

§7 named one idea from the AVX2 arm as separable from the weight repack: llama.cpp pre-interleaves
the **activation** side into `block_q8_Kx4` at quantisation time (`repack.cpp:4301`, 4 rows at a
time), whereas `SimdKernels.DotQ4K_Q8KS_4In` reads `q8_i + chunk*64` from **four separate scratch
allocations**. Tested in isolation (`scratchpad/actinter`), changing only the layout.

**Note we cannot copy llama.cpp's granularity.** They interleave at 8 bytes; our `maddubs` pairing
needs 32 contiguous K-values of one row per load. Chunk (64-byte) granularity is the finest
interleave that preserves our arithmetic — the four reads per chunk become four sequential cache
lines instead of four distant streams.

Correctness gate: `max |split - inter| = 0.000E+000` — byte-identical arithmetic, layout is the
only variable.

| run | speedup (best) | speedup (median) |
|---|---:|---:|
| 1 (cold) | 1.0094x | 0.9960x |
| 2 | 1.0145x | 1.0144x |
| 3 | 1.0236x | 1.0235x |
| 4 | 1.0140x | 1.0136x |

**Result: ~+1.5% isolated.** Real (runs 2-4 agree between best and median to three decimals) but
tiny. Do not ship, for three reasons:

1. **Below the end-to-end noise floor.** Iteration 24 is the precedent: a reproduced 2.4-2.6x
   *isolated* win became a ~12% end-to-end **loss** under production `Parallel.For` contention.
   A 1.5% isolated delta cannot be resolved by `bench-prefill-cli.ps1`, whose own run-to-run
   spread is larger.
2. **The benchmark excludes the cost of building the layout.** Production would have to emit
   interleaved activations during quantisation — strided writes across four row buffers. That
   cost is unmeasured and plausibly exceeds the 1.5%.
3. **The mechanism is nearly exhausted by design.** At `cols=2048` the four activation buffers
   total ~10 KB and stay L1-resident while 2.36 MB of weights stream past. The kernel is
   ALU/load-port bound (~2.4 vector ops/cycle measured), not activation-layout bound. The residual
   1.5% is most likely L1 set-conflict relief (Zen 3: 32 KB, 8-way, 64 sets), which is inherently
   a small effect.

This closes the last separable idea from §7. The remaining gap is the column-parallel accumulator,
which is inseparable from the repacked weight layout — see §7's verdict.
