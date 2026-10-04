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
