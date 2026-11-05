# TODO: completed items split from docs/done/todo-merged-2026-11-01.md (2026-10-30)
> Open items stay in [../todo.md](todo-merged-2026-11-01.md).

Finished items moved verbatim out of [../todo.md](todo-merged-2026-11-01.md). Items still open stay there.

## Scout / Hugging Face: completed items

- [x] **Quant picker** DONE 2026-10-09 (`scout -r repo --quants`; see 061). Original note: (`scout -r repo --quants`): remote-scout every quant, run the estimator + gate, print which fit this machine and the exact `pull` command pinned to the inspected commit. Uses what exists; likely daily-use.
- [x] **Estimator for the hybrid recurrent family (qwen35 / qwen35moe): DONE 2026-10-09** (4 models calibrated, see 061). Original note: **(qwen35 / qwen35moe / qwen4exp): HIGH VALUE.** The Hub's most-downloaded GGUF repos right now are this family (verified top-12 list), and the quant picker returns `unknown` for all of them. Needs the recurrent-state sizing (conv + delta state per recurrent layer, KV only for the full-attention layers)
      and a real calibration run (Qwen3.6-35B-A3B Q6_K is on the archive disk, ~27 GB; loads slowly from there, run it alone). Still open: qwen4exp (not measured), MLA (deepseek2) and RWKV.
- [x] **Scratch arena experiment (measure first)** CLOSED 2026-10-10 at P0, no-go (0.1-0.3% bound; see PerformanceLeague.md). **Plan with phases and times: `docs/done/2026-10-09-scratch-arena-experiment-plan.md` (half a day if it fails at P0; ~1.5-2 working days / ~3 calendar days if it pays off).** (idea from chat: zero-allocation tensor arenas). Facts: 284 `NativeMemory.Alloc*` sites in 51 files, nearly all allocated ONCE at construction; GC share of decode is 0.4-0.9% (PerformanceLeague.md ~1306); an arena can save at most the GC-committed memory (~300 MiB on a 27B, ~2% of peak). Do NOT retrofit all sites (weeks, wrong-number risk, unmeasurable gain).
      Plan: (1) `NativeArena` utility (aligned slab, scoped bump/release, typed views, misuse checks, ~15 tests; hours, touches nothing existing); (2) use it ONLY for batched-prefill chunk scratch (~120 alloc/free pairs of 1-10 MB per chunk, flagged "unmeasured, measure before pooling" in perf-investigation-brief.md);
      (3) A/B on real weights, keep only if measurably faster (rule 7). Also the ~1 MiB/token logits `.ToArray()` copies (ContinuousBatchingEngine, CudaForwardPass) if wanted. Open question: is the hybrid family's ~335 MiB fixed base (dense ~135 MiB) managed heap or something else? Read `managedHeap`/`gcCommitted` from a hybrid run.

- [x] `pull --revision` DONE 2026-10-09 (also pins to the listed commit by default). Still open: route `pull`'s download through `ExternalHttpClient`; make `pull` use `HubClient`'s listing parser (it keeps its own copy).

## RESUME HERE: CpuSgemm replaces OpenBLAS (completed)

- **CpuSgemm replaces OpenBLAS in `CpuBackend.Sgemm`**: DONE, committed `c9b47c69` (Diffusion 162 passed / 262 heavy skipped;
  ForwardPass.Fast 960/0 incl. 176 `CpuSgemmTests`).
  - Files: `src/OpenTail.Stingray.Cpu/CpuSgemm.cs` (TensorSharp port), `CpuBackend.cs`,
    `tests/.../CpuSgemmTests.cs`, `tools/kernel-bench-cs/SgemmBench.cs` + `Program.cs`
    (`sgemm`, `sgemm-sweep`), `docs/4-performance/2026-10-03-cpusgemm-vs-openblas.md`,
    `THIRD_PARTY_NOTICES.md` line, Cli csproj (OpenBLAS packaging item removed).
  - Result: matches/beats OpenBLAS at equal threads on 16 shapes. OpenBLAS's apparent lead was SMT
    (16 threads vs our 8), so CpuSgemm uses logical cores. Real FLUX VAE decode 512px:
    11.20 s vs 12.13 s; PSNR 141.6 dB vs OpenBLAS.
  - HEAD worktree removed.

## Batch-invariant quantized GEMM: completed and closed (investigation stopped 2026-10-09)

- [x] Step 0 DONE 2026-10-03 (committed `a5af0272` + test `fb1f07e6`).
      `STINGRAY_CPU_DECODE_VIA_GEMM=1` routes `FusedMatVec` (Q4_K/Q6_K) + DenseFfn gate/up through
      `MatMulBatchedCached(N=1)` / `MatMulBatchedDualCached`.
      - Kernel invariance (`BatchInvariantGemmTests`, 19): Q4Kx8 Path 2 and Q6K GEMM rows BITWISE
        identical at batch 1..33 vs N=1; outputs checked non-trivial; both kernels eligible on AVX2.
      - Model level (`DecodeViaGemmInvarianceTests`, SmolLM2-360M teacher-forced, 33 positions): GEMM
        decode vs prefill maxAbs 0, 33/33 bit-identical, argmax 33/33. Today's matvec decode: maxAbs
        2.55, 0/33 identical, argmax 29/33.
      - Speed (interleaved x3, idle, 128 tokens, median): 360M 87.8 -> 80.2 t/s (-7%), 1.7B 30.3 ->
        28.8 (-4%), Mistral-7B 8.7 -> 8.3 (-5%). Text differs (numerics changed). Log:
        scratchpad `decode_ab.log`.
      - Suspected cause: at N=1 Path 2 still runs a 4-row Q8Kx4 group (3 zero-padded rows) = 4x the
        arithmetic. Stays opt-in; numerics decision is the user's.
- [x] Step 1a DONE `b31d5f3a`: gate+up DUAL repacked GEMM (one quantisation, one dispatch, bitwise
      = two calls) + per-thread quant scratch (no per-call allocs; depth guard for re-entrancy).
      Prefill SmolLM2-1.7B 707 tok: 214.1 -> 226.4 t/s (+5.7%, 6 pairs vs fb1f07e6, text identical).
      Decode-via-GEMM gap: 1.7B -4.9% -> -3.6%; 7B ~-5% unchanged.
      Kernel facts (KernelBench `q4k-m1`, weights rotated past L3): Q4_K GEMM N=1 vs matvec 1.69x
      (2048^2), 1.49-2.08x (8192x2048), 1.13-1.23x (7B); but MatVecDual (fused gate+up) beats two
      GEMMs 0.89-0.91x at FFN shapes (bandwidth-bound ~36-39 GB/s); Q6_K GEMM N=1 0.81x (ffn_down
      2048x8192), 0.93x (7B down), 0.99x (LM heads).
- [x] Step 1b STOPPED 2026-10-09 (decision: investigation closed, not worth continuing). Q6_K at N=1 - a 1-row-friendly path in `Q6KPrefillGemm`
      with identical per-row arithmetic; and fused q/k/v (one quantisation + one dispatch, like the
      dual). Re-run decode A/B after each.
      - Done: per-thread scratch (no per-call alloc) + `RowBlockKernel1` (register accumulator, same
        op order; 32 BatchInvariantGemm/Q6KPrefillGemm tests pass, bitwise).
      - Kernel bench (`kernel-bench-cs q4k-m1`; run with DOTNET_TieredCompilation=0 - at default
        tiering the first Q6_K shape measures tier-0 code, bogus 0.22x): ffn-down 1.7B 0.81 -> 0.88x,
        7B down 0.96 -> 0.95x, LM heads 0.97-1.00x. Decode A/B 1.7B -3.9% -> -3.2%; 7B noisy (-3..-8%).
      - Left: ffn-down (cache-resident, compute-bound) is the pairing-unpack cost that is no longer
        amortised over a token group at N=1. True parity needs either different Q6_K arithmetic
        (changes prefill numerics - user's call) or a cheaper decode; fused q/k/v not done.
      - Option 1 A/B (2026-10-09, 3 interleaved rounds, median; `STINGRAY_Q6K_GEMM=0` = row-major
        Q6_K family, whose N=1 kernel IS the matvec's): decode 1.7B matvec 30.9 / Q6K-GEMM 29.4 /
        row-major 29.5; 7B 8.5 / 8.0 / 8.0; prefill 1221 tok 223.9 / 219.6 / 224.9 (noise +-3%).
        => Q6_K arithmetic is NOT the cause: the same ~-5% remains with the matvec's own Q6_K kernel.
        The gap is per-call overhead of the batched path at N=1 (quantise + dispatch + Parallel.For
        per matrix, ~200+ calls/token) - next: count/profile calls per token, fuse q/k/v (and o/down),
        and compare against the matvec's fused dispatch.
      - **STOPPING THE INVESTIGATION.** Rationale: the only gain is bit-identical decode/prefill by
        default, at best parity with the legacy matvec (never faster); the remaining -3..-6% is
        per-call overhead across ~200 calls/token, fixable only by invasive dispatch fusion on the
        decode hot path. `STINGRAY_CPU_DECODE_VIA_GEMM=1` stays opt-in (cost ~3-6% decode for
        bit-identical decode); default stays the matvec. Kept: Q6_K N=1 kernel + per-thread scratch
        (small, bitwise identical). Reopen (fuse q/k/v, o, down; profile calls/token) only if a
        feature needs bit-identical decode: speculative verify or batched CPU serving.
- [x] Step 1c DONE `e12c6b62`: rows 2-3 skip in `GemmQ4Kx8Q8Kx4` (`AccumulateRows01`) when the last group
      has <= 2 real rows (exact: lane pairs 0/1 and 2/3 are independent). Bitwise identical.
      Post-1c decode A/B (2026-10-09, 128 tok, 3 alternating pairs, median, GEMM=1 vs matvec):
      360M 93.3 -> 91.2 (-2.2%, noisy), 1.7B 31.1 -> 29.9 (-3.9%), Mistral-7B 8.5 -> 8.3 (-2.4%).
      Remaining gap is mostly Q6_K (Step 1b) + 4-row activation quantisation at N=1. Goal: parity.
- Note: 8 ForwardPass.Fast tests now SKIP because OpenBLAS is gone (`SkipUnless(BlasAvailable)`);
  they test the dead BLAS path. Expected, not a regression.
- [ ] (old) a 1-row variant of Path 2 (and of the Q6K GEMM)
      doing the same per-row operations in the same order, so it stays bit-identical
      (`BatchInvariantGemmTests` enforce it) without padded rows. Re-run the decode A/B:
      `stingray -m models/_models/<gguf> -p "Explain how a printing press works, step by step."
      -n 128 --temp 0 -g 0`, `STINGRAY_CPU_DECODE_VIA_GEMM=0/1` alternating, 3 pairs per model.
      Code: `src/OpenTail.Stingray.Cpu/RepackedGemmPath2.cs` (port of ggml_gemm_q4_K_8x8_q8_K),
      `Q6KPrefillGemm.cs`.
- [ ] Step 1b: if the 1-row variant can't match the matvec, port ManagedQuantGemm (`.cs`, `.Kernels.cs`, `.Decode.cs`; Q4_K/
      Q5_K/Q6_K/Q4_0/Q5_0/Q8_0) as an opt-in kernel; its `TS_CPU_QGEMM_VERIFY` idea as a
      diagnostic (compare against the per-row path on real weights).

## Hybrid-GDN MTP chunked prefill: evidence (completed)

## Evidence (done 2026-10-02, Ornith-1.0-9B Q4_K_M)
- [x] 9B attribution matrix: 41 prompts (30 short / 8 x 500 / 3 x 1200 tokens)
      (`scratchpad/attrib/ornith9b.log`, `.csv`, `an/analyze.cs`).
      - R vs A distance to D: median KL log-ratio 0.00 [-0.04, +0.06]; R closer on 20/41 prompts.
      - Zero flips at certified positions (m > 2δ).
      - Mild tilt toward A on the 3 long prompts only (D sides with A 54 vs R 31; correlated, n=3).
- [x] GDN probe (`scratchpad/probe/probe9b.log`):
      - Local recurrence error at FP32 epsilon: seq 2-4e-7, chunked 1.3-3e-7 (chunked slightly better).
      - Drift is downstream amplification: ~1e-4 at layer 0 -> 3-5e-2 at layer 31.
      - No prefill->decode discontinuity.
      - The `<think>` boundary token flips against double for BOTH FP32 paths.
- [x] Row check (`scratchpad/rowchk/rowchk.log`) at the long40 pos-541 anomaly (KL 13.8):
      batched GEMM vs a double dot of dequantized weights is 9e-7; the per-token int8 decode
      matvec is 1.2e-2 (worst 8e-2). So the "exact" path is the noisy one, and this is not a GEMM bug.
- [x] "Really bad turn" check: none found (no systematic disadvantage, no certified flips, no
      pathological layer, no decode-step jump).
- [x] int8 interaction: the hybrid chunked path is bit-identical with int8 on vs off (Q4_K), with
      no repeated-token collapse (`scratchpad/q8chk`), so the matrix numbers stand.
- [x] Reply to ChatGPT written (full results + 3 questions)
- [x] Investigation recorded (uncommitted; commit with the MTP change):
      - `docs/1-correctness/2026-10-02-prefill-numerics-investigation.md` (step-by-step record)
      - `docs/reference/numerics-investigation-method.md` (reusable method, decision rule);
        linked from CLAUDE.md "Documentation References"
      - `docs/4-performance/patches/2026-10-02-prefill-numerics/` (switches patch, harness sources,
        prompt lists, README)


$line

## int8 prompt processing on by default: completed items

- [x] Flip the default in `SimdKernels.Q8PrefillEnabled` (`!= "0"`); rewrite its doc comment (and
      move it onto the property: it was attached to `BatchedMatVecTierCalls`)
- [x] Build clean; ForwardPass.Fast 748 passed / 0 failed (1 skip: `MatVecThroughput_ByDType`,
      a throughput benchmark); Server.Fast 429/429
- [x] `docs/reference/adr-0003-cpu-int8-prefill-default.md` written (evidence, 2026-10-01 history,
      MoE scope, hybrid/RWKV guard note, rollback criteria)
- [x] `docs/reference/env-var-inventory.md`: `STINGRAY_CPU_PREFILL_Q8` default on, `=0` rollback
- [x] `perf-sweep-plan.md` 10.2 (decision + follow-up list) and Phase 6 (decision, re-measure pending)
- [x] `docs/WHAT-YOU-CAN-DO.md`: stale DeepSeek/int8 sentence fixed
- [x] Checked old int8 defects: the 2026-08-07 repeated-token collapse is fixed by
      `ForwardPass.IsSingleDistinctTokenPrompt`. The hybrid path has no such guard but measured no
      collapse. RWKV is unverified (no checkpoint on disk), noted in ADR-0003
- [x] Heavy suite with int8 on: 166/1/58, the only failure being known item 24. Committed `d7fcf135`

## TensorSharp takeaways: completed items 1-3

- [x] 1. Spin-then-park worker pool: DONE, shipped opt-in `8f45d133` (history below):
      - Ported: `src/OpenTail.Stingray.Cpu/SpinParkWorkerPool.cs` (with `TryFor`, so a busy pool falls
        back to `Parallel.For`). Switches: `STINGRAY_CPU_POOL=spin`, `STINGRAY_CPU_POOL_BLOCKS`
        (default 4; 8 measured best), `STINGRAY_CPU_SPIN`.
      - Routed through `SimdKernels.KernelFor` / `ParallelForCapped` / `ParallelForUncapped` /
        `ParallelForWithScratch`: the matvecs, attention, row kernels, MoE (dense + batched experts),
        hybrid-GDN, and the prefill GEMMs. Default path is unchanged when off (same caps).
      - Tests: `SpinParkWorkerPoolTests` (23). ForwardPass.Fast is 771/771 with the pool off AND on.
        A divide-by-zero on empty ranges was caught and fixed.
      - A/B v1 (one block per index): -25..35% decode. v3 (coarse blocks): 135M +21%, 0.5B +18%,
        360M +20%, Mistral-7B tie; text identical. Long-prompt prefill: tie. MoE (OLMoE, LFM2)
        REGRESSED -12..20% while their loops were still on the ThreadPool, so those were routed too;
        re-measuring now (`scratchpad/pool_ab6.log`).
      - Decide: default on only if MoE/hybrid aren't worse; then docs, plan §1 results, League rows.
- [x] 2. DONE `92a6f37a` (used vendored ggml directly). TensorSharp Q8_K kernels as the second reference in the int8 alignment work (pairs with
      the ggml cross-check above)
- [x] 3. DONE (`Q1_0` verified; Bonsai2 PRISM `0ebec790`). `Q1_0` + Bonsai2 `PQ2_0`/`PTQ1_0` quant types
