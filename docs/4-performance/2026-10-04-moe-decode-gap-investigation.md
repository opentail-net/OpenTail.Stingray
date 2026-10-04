# MoE CPU decode gap vs llama.cpp: investigation log (2026-10-04)

Status: **measurement only, no kernel changed.** The one code addition is an opt-in timing diagnostic, `STINGRAY_MOE_PHASE_TIMING=1` (`MoePhaseTiming.cs`, hooks in `ForwardPass.Moe.cs`). Context: the Step 6 speed league in [PerformanceLeague.md](../../PerformanceLeague.md) ("MoE lot") had our CPU decode at 0.21-0.59x of llama.cpp on five MoE models. Machine: Ryzen 7 5700G, **8 cores / 16 threads** (the older "6c/12t" note was wrong), DDR4, 64 GB, AVX2. Default CPU thread count is the physical core count (8); every run below sets `-t` explicitly.

## What was measured

All llama.cpp numbers: `llama-bench -p 0 -n 128 -t N -ngl 0 -r 1` (decode from an empty cache). All ours: the CLI, `docs/reference/benchmark-prompt.txt` (482-493 tokens), 24 generated tokens, temperature 0, `-g 0`. Decode for ours is therefore measured at about 500 tokens of context. Decode t/s.

### 1. The old Qwen3-Coder "1.08x" was wrong
| Qwen3-Coder-30B-A3B Q4_K_M, 6 threads | runs | median |
|---|---|---|
| llama.cpp | 20.4 / 20.5 / 20.0 | 20.4 |
| ours, default (`Parallel.For`) | 8.7 / 8.7 / 8.6 | 8.7 (0.43x) |
| ours, `STINGRAY_CPU_POOL=spin` | 12.0 / 13.2 / 12.8 | 12.8 (0.63x) |

The 2026-09-10 league row had llama.cpp at 6.85 t/s; that reference was not reproducible. The row is marked superseded in PerformanceLeague.md.

### 2. The pool variable
`STINGRAY_CPU_POOL` is only compared with `"spin"`, so `park` and unset are the same code. Qwen1.5-MoE, 6 threads, interleaved: unset 17.8 / 17.7 / 15.6 / 15.4, `park` 17.0 / 15.0 / 16.0 / 15.9. The 12.0 t/s Qwen1.5-MoE decode in this morning's league run was **not** a configuration effect and was not reproduced in any later run (llama.cpp stayed at 24-25 t/s throughout). Cause unknown (machine state at that hour). Consequence: absolute numbers drift 10-40% across sessions; compare engines only back to back, and treat the league's separate-hour ratios as indicative.

### 3. Thread scaling (single run per cell, shuffled order)
| Qwen1.5-MoE-A2.7B Q4_K_M | 1 | 2 | 4 | 6 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|---|
| llama.cpp | 11.2 | 18.0 | 23.5 | 22.7 | 23.0 | 22.2 | 20.5 |
| ours, default | 4.5 | 8.2 | 13.4 | 17.5 | 17.0 | 17.2 | 17.2 |
| ours, spin | 4.6 | 8.4 | 13.1 | 16.1 | 19.0 | 19.7 | **6.0** |

| Qwen3-Coder-30B-A3B Q4_K_M | 1 | 2 | 4 | 6 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|---|
| llama.cpp | 9.2 | 14.9 | 18.5 | 18.9 | 17.7 | 17.2 | 16.9 |
| ours, default | 2.9 | 4.0 | 6.3 | 12.3 | 10.7 | 13.1 | 15.0 |
| ours, spin | 3.1 | 3.9 | 7.7 | 12.3 | 12.9 | 14.6 | 14.5 |

Reading: llama.cpp saturates memory at about 4 threads. Ours scales almost linearly to 6 and never hits that wall. **At 1 thread, where there is no fork/join and no bandwidth ceiling, we are already 2.5-3.2x slower.** The spin pool collapses at 16 threads on Qwen1.5-MoE (6.0 t/s): do not make it a default.

### 4. The gap is not MoE-specific
Dense Qwen3-4B Q4_K_M: llama.cpp 7.95 t/s vs ours 2.8 at 1 thread (0.35x); 16.3 vs 7.9 at 6 threads (0.48x). Per core, our Q4_K decode matvec streams roughly 6-7 GB/s of weights against about 19 GB/s for ggml (arithmetic from the model's bytes per token, approximate).

### 5. MoE phase timing (Qwen1.5-MoE, one decode token = 24 MoE layer calls; microseconds per call)
| Phase | 1 thread | share | 6 threads | share |
|---|---|---|---|---|
| router | 17 | 0.3% | 19 | 0.9% |
| **shared expert** (dense FFN, 5632 wide) | 3036 | **46.6%** | 1036 | **46.6%** |
| sweep A (gate+up, folded) | 1132 | 17.4% | 347 | 15.6% |
| sweep B (down, folded) | 780 | 12.0% | 255 | 11.5% |
| **sequential per-expert loop** (non-folded dtype) | 1531 | **23.5%** | 547 | **24.6%** |
| activation quantisation (A + B) | 10 | 0.1% | 12 | 0.5% |
| empty sweep over A's range (fork/join cost) | 8 | 0.1% | 6 | 0.3% |
| empty sweep over B's range (fork/join cost) | 4 | 0.1% | 5 | 0.2% |
| total | 6514 | | 2224 | |

- **Barrier / fork-join cost is negligible here**: an empty sweep with the same granularity costs about 5 us, against a 2.2 ms layer. (It is measured with warm workers right after a sweep, so cold wake-up latency after serial sections is not included; that is where the spin pool's +47% on Qwen3-Coder may come from, unproven.)
- The shared expert is 46.6% of the MoE time; it goes through the ordinary dense matvec, at about 6.4 GB/s per core.
- **The sequential loop is a coverage gap, not a design choice**: this checkpoint's `ffn_down_exps` is **Q5_0 in 12 layers** and Q8_0 in the other 12, and `IsFoldedDotDType` accepts Q8_0 but not Q5_0 (nor Q2_K), so those 12 layers skip the folded two-sweep path.
- The folded routed sweeps run at about 10 GB/s per core, better than the dense matvec's 6.4.
- The MoE FFN is about two thirds of a decode token (53 ms of about 80 ms at 6 threads); attention, projections and the output head are the rest.

## What this supports and what it does not
Supported by the data: (1) per-core quantised matvec throughput is the dominant gap, for dense and MoE alike; (2) the folded MoE path is missing Q5_0 and Q2_K, which costs about a quarter of MoE time on this checkpoint and presumably more on GLM (Q2_K); (3) fork/join is not the problem at 6 threads on Qwen1.5-MoE.
**Not** shown: whether the kernel is compute-bound or access-bound (no in-cache microbenchmark yet), why spin helps Qwen3-Coder, why the morning's 12 t/s happened, bytes actually read per token, hardware counters. The earlier repository claim that MoE decode is "memory-bound, already at its streaming limit" is contradicted: ggml does about 3x more weight bytes per second per core than we do.

## Suggested next steps (not done)
1. In-cache vs streaming microbenchmark of the Q4_K Q8_KS matvec at 1 thread (decides compute-bound vs access-bound).
2. Add Q5_0 and Q2_K to the folded path (`IsFoldedDotDType`, `DispatchDot`, `ActScratchBytes`/`QuantizeAct`); local, testable, and bit-identity to the sequential path is checkable.
3. Look at the shared-expert and dense decode matvec as the same kernel question.
4. Thread scaling and phase timing for Mixtral and GLM; bytes-per-token counting; AMD uProf counters.

## 6. Microbenchmark: our Q4_K row dot vs ggml's, same rows, one thread (2026-10-04, later)
Tool: `tools/kernel-bench-cs`, mode `q4k-dot` (`Q4KDotBench.cs`). It loads the vendored `ggml-cpu-haswell.dll` (the backend llama.cpp selects on this Zen 3 CPU), takes ggml's own `vec_dot` for Q4_K from `ggml_get_type_traits_cpu`, and times it against our `DotQ4K_Q8KS` and `DotQ4K_Q8KS_2Row` on the SAME random-but-valid Q4_K rows (K = 2048, 1152 bytes per row), each kernel fed its own int8 activation (ours Q8_KS, ggml Q8_K) quantised from the same float vector. No scheduler, no model, no allocation in the timed loop. 600 ms warm-up per cell (so tiered JIT has promoted the kernel), then median of at least 5 passes. Run: `STINGRAY_BENCH_ALLOW_TIERED=1 DOTNET_TieredPGO=0 DOTNET_TC_CallCountingDelayMs=0 OpenTail.Stingray.KernelBench.exe q4k-dot`.

| Working set | ours 1-row ns/row | ours 2-row ns/row | ggml ns/row | ours / ggml (1-row) |
|---|---|---|---|---|
| L1 (16 rows, 18 KB) | 388 | 363 | 106 | 3.6x |
| L2 (256 rows, 295 KB) | 343 | 196 | 128 | 2.7x |
| L3 (4096 rows, 4.7 MB) | 327 | 378 | 117 | 2.8x |
| DRAM (65536 rows, 75 MB) | 349 | 375 | 104 | 3.3x |
| real: shared-expert gate/up 5632 x 2048 (rotated past L3) | 363 | 426 | 111 | 3.3x |
| real: Qwen3-Coder expert gate/up 768 x 2048 | 379 | 384 | 102 | 3.7x |

**Result: the per-row time does not depend on the working set** (L1 to DRAM). Our kernel is therefore compute-bound, about 45 ns per 256-weight super-block (roughly 200 cycles) against about 13 ns for ggml (roughly 55 cycles). That is the 2.5-3.3x per-core gap seen end to end. The 2-row variant that production decode uses is not faster per row than the 1-row one. Results agree numerically (row sums differ about 0.2%, from the two activation formats).

Reading `DotQ4K_Q8KS_Avx2` against what ggml's `vec_dot_q4_K_q8_K` does, the structural differences that plausibly explain the cycle count (not yet proven by experiment): (1) Q8_KS carries one float scale per 32 activations, so each super-block does 8 int-to-float conversions and 8 float FMAs with a freshly computed broadcast scale (`dSub[s] * (d * sc)`), where Q8_K has one scale per 256 and integer accumulation across the super-block; (2) the 6-bit scales and mins are unpacked with scalar code (`GetScaleMinK4` eight times per super-block, plus `LoadQ4KMins` / `MinCorrectionQ4K` and two software half conversions), where ggml shuffles the packed scales as vectors; (3) one dependent float FMA chain through a single accumulator. The Q8_KS format and the exact FP summation order are deliberate (the kernel comment calls it bit-identical to the `_4In`/`_8In` batched kernels, pinned by `MatMulBatchedQ8EquivalenceTests`, and says per-32 scales were chosen for output quality), so a faster kernel is a numerics decision as well as an engineering one.

**JIT sensitivity, an unexpected side result.** The same benchmark with `DOTNET_TieredCompilation=0` gives about 1500 ns/row for our 1-row kernel at every working-set size, 4x slower than the tiered steady state above, while ggml is unaffected. In default tiered mode, cells measured before the kernel was promoted to tier 1 were also 5-10x slow. Consequence worth checking: whether a NativeAOT build (the project's stated deployment target) behaves like the tiered steady state or like `TieredCompilation=0`; the CLI numbers in this note and in the league are all from the default JIT build. Not measured.

## 7. Step 1 done: Q5_0 and Q2_K in the folded MoE decode path (2026-10-04, later)
`IsFoldedDotDType`, `ActScratchBytes`, `QuantizeAct` and `DispatchDot` now cover Q5_0 (Q8_0 activations, `DotQ5_0_Q8_0`) and Q2_K (Q8_K activations, `DotQ2K_Q8K`), exactly what the sequential path's `SimdKernels.MatVec` uses for those dtypes, so the two paths stay bit-identical per row. Checks: `Qwen2MoeGreedyParityTests` (Qwen1.5-MoE: Q5_0 down experts in 12 layers) and `GlmMoeGreedyParityTests` (Q2_K) pass with the same confident-position counts as before (17 and 5; 14 and 16), real runs of 29 s and 213 s; Fast suite 1030 passed. Phase timing on Qwen1.5-MoE at 6 threads: the "sequential per-expert loop" phase (24.6% of the MoE layer before) is gone and the layer total fell from 2224 to 1838 us, measured while the machine was in a SLOWER state than the earlier run (llama.cpp read 18.8 t/s instead of 25 in the same window), so the gain is understated; end-to-end ratios from this session are too noisy to quote.

## 8. Steps 2 and 3 (2026-10-04, evening)

### Step 2: a ggml-style Q4_K kernel prototype, in the bench tool only (`tools/kernel-bench-cs/Q4KProto.cs`, mode `q4k-dot`)
Same structure as `ggml_vec_dot_q4_K_q8_K`: activations in Q8_K (one float scale per 256, per-16 sums), 6-bit scales unpacked once per super-block and broadcast as 16-bit lanes by byte shuffle, `maddubs` + `madd` against the scales accumulating in integer across the super-block, mins folded in through the activation's per-16 sums, one float conversion + FMA per super-block. It is fed ggml's own Q8_K buffer.

| ns/row, one thread, same window | ours 1-row (Q8_KS) | ours 2-row | prototype | ggml |
|---|---|---|---|---|
| L1 (18 KB) | 200 | 181 | 88-100 | 56 |
| L2 (295 KB) | 196 | 182 | 87 | 89 |
| DRAM (75 MB) | 199-207 | 250-280 | 98-107 | 65-79 |
| shared-expert shape | 199 | 255 | 101 | 65 |
| Qwen3-Coder expert shape | 216 | 240 | 96 | 77 |

The prototype is about 2x faster than our current kernel and within 1.25-1.7x of ggml. On the cells where all kernels see the same rows its output equals ggml's to three decimals (checksums -832.681, -3598.620, -3684.363). Accuracy against an exact double-precision dot of the dequantised weights and the original float activation (K = 2048, 8 x 4096 random rows, activations uniform [-1,1] with occasional x8 outliers): **ours Q8_KS 0.66% relative RMS error, ggml Q8_K and the prototype 1.38%**. So the speed-up costs about 2x the activation-quantisation error, which is exactly llama.cpp's behaviour; it is a policy decision, not applied to the engine. Integration would also have to deal with the bit-identity contract between the single-row Q8_KS kernel and the batched `_4In`/`_8In` kernels (`MatMulBatchedQ8EquivalenceTests`) and with an own Q8_K quantiser in ggml's 292-byte layout. A middle option not prototyped: keep Q8_KS but replace the scalar scale unpacking and per-sub-block broadcasts with the shuffle/madd structure (expected gain smaller; it cannot accumulate across sub-blocks in integer because the activation scales differ per 32).

### Step 3: NativeAOT build vs the JIT build
`dotnet publish src/OpenTail.Stingray.Cli -c Release -r win-x64` works once `vswhere.exe` (Visual Studio Installer directory) is on PATH; without it the ILCompiler step fails with "vswhere.exe is not recognized". The AOT `stingray.exe` (22.6 MB) decodes Qwen1.5-MoE at 6.8 / 15.6 / 15.3 t/s against the JIT build's 7.7 / 9.3 / 13.4 in interleaved pairs, with llama.cpp at 13.6 in the same window. So the AOT build is **not** in the slow `TieredCompilation=0` regime (4x slower); it is in the same range as the tiered JIT. The window was too noisy for a finer comparison.

### Measurement contamination (probable, not proven)
At 21:3x a process named `OpenTail.Stingray.Tests.Diffusion` (not started by this work; 21,100 CPU-seconds, 17.7 GB working set) was running on the same machine. The MoE benchmarks in this note therefore ran next to other heavy work at least part of the time. That plausibly explains the session-to-session drift (llama.cpp read 25, then 18.8, then 13.6 t/s on the same model), the unreproduced 12 t/s result this morning, and the low-memory kill of the Hunyuan run. Before quoting any ratio from these notes, re-measure on a machine with nothing else running.
