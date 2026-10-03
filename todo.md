# RESUME HERE (updated 2026-10-03)

## Now
- **CpuSgemm replaces OpenBLAS in `CpuBackend.Sgemm`**: done, UNCOMMITTED, waiting on the Diffusion
  test suite (ForwardPass.Fast 960/0 green incl. 176 new `CpuSgemmTests`).
  - Files: `src/OpenTail.Stingray.Cpu/CpuSgemm.cs` (TensorSharp port), `CpuBackend.cs`,
    `tests/.../CpuSgemmTests.cs`, `tools/kernel-bench-cs/SgemmBench.cs` + `Program.cs`
    (`sgemm`, `sgemm-sweep`), `docs/4-performance/2026-10-03-cpusgemm-vs-openblas.md`,
    `THIRD_PARTY_NOTICES.md` line, Cli csproj (OpenBLAS packaging item removed).
  - Result: matches/beats OpenBLAS at equal threads on 16 shapes. OpenBLAS's apparent lead was SMT
    (16 threads vs our 8), so CpuSgemm uses logical cores. Real FLUX VAE decode 512px:
    11.20 s vs 12.13 s; PSNR 141.6 dB vs OpenBLAS.
  - After commit: `git worktree remove` the scratchpad `head` worktree.
- **OpenBLAS DLL**: moved to `C:\Git-Public\libopenblas.dll` (user decides when to delete); 6 build-
  output copies removed; never ships again. User: keep the OpenBLAS code (harmless without the DLL).
- **Muse-Glimmer**: CPU wiring committed `8a096e36` (not admitted). Next: level-2 synthetic spec test.

## Next: batch-invariant quantized GEMM (TensorSharp ManagedQuantGemm), user agreed 2026-10-03
Idea: single-token decode goes through the SAME GEMM as prefill (`QGemmMinRows = 1`,
`ManagedQuantGemm.cs:109`), so decode, prefill, speculative verify and batching are bit-identical
by construction. Its M=1 speed claim (1.0-2.1x per-row on K-quants) was measured on their machine.
- **Numerics catch:** ManagedQuantGemm uses ggml Q8_K activations (1 scale/256). Our decode uses
  Q8_KS (1 scale/32), measured ~2.5x more precise on Q4_K. Changing decode numerics is the user's
  call: build opt-in, measure, bring numbers before any default changes (see feedback memory).
- [ ] Step 0 (cheap, existing code): route M=1 decode through our repacked Q4Kx8 prefill GEMM
      (`TryMatMulBatchedQ8`, already ggml Q8_K). Measure decode t/s vs the current matvec,
      uncontended, real weights (SmolLM2-1.7B, Qwen2.5-0.5B, a 7B). If it's close, invariance
      needs no port.
- [ ] Step 1: if step 0 is slow, port ManagedQuantGemm (`.cs`, `.Kernels.cs`, `.Decode.cs`; Q4_K/
      Q5_K/Q6_K/Q4_0/Q5_0/Q8_0) as an opt-in kernel; its `TS_CPU_QGEMM_VERIFY` idea as a
      diagnostic (compare against the per-row path on real weights).
- [ ] Step 2: or a Q8_KS variant of the unified GEMM (keeps our decode precision; no reference
      implementation; the TensorSharp Q8_K kernels are the readable model).
- [ ] Measure: decode t/s, prefill t/s, PPL vs llama.cpp (SmolLM2: F32 0.03%, int8 0.69% today),
      decode-vs-prefill logit identity. Docs: ADR, League rows, perf-sweep-plan 10.2.

## Commits (newest first; none pushed)
- `8a096e36` feat: Muse-Glimmer CPU wiring, ported, not verified
- `85395cdf` docs(coverage): review corrections to the family plans; llama.cpp source at bed0a8566
- `964fb57a` docs(coverage): one port plan per family
- `0ebec790` feat: Bonsai2 PRISM (PQ2_0/PTQ1_0 + Hadamard), ported, not verified (STINGRAY_EXPERIMENTAL_PRISM=1)
- `e63c520d` engine: optimized hybrid-GDN prefill for MTP models (ADR-0002). Qwen3.8-27B
  prefill 2.1 -> 8.8 t/s (~2.0x llama.cpp), text identical
- `92a6f37a` docs: investigation record + method guide + reproduction kit; ggml Q8 cross-check
- `8f45d133` perf(cpu): opt-in spin-then-park worker pool (`STINGRAY_CPU_POOL=spin`)
- `d7fcf135` engine: CPU int8 prefill default on (ADR-0003)

## Then
1. Family ports in plan order (`docs/2-coverage/ported-families-todo.md`): Muse-Glimmer spec test,
   then Qwen 3.8 Flash Next, GLM-5.x, DiffusionGemma, MiniMax-H3, DeepSeek review. NOT admitted,
   NOT advertised.
2. Later: uncontended int8 speed re-measure (Gemma rows), 27B MTP validation + draft acceptance.
- Not mine, leave alone: `stage_diagnostics_report.txt`, `higgs-tts-*.json/txt` stay uncommitted.

---

# TODO: allow chunked prefill for hybrid-GDN models with an MTP head

User decision 2026-10-02: allow the fast (chunked/batched) CPU prefill path for MTP models
(Qwen3.6/3.8-27B). Roll back only if the remaining results take a really bad turn.
Rule: document, document, document.

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

## Before the clean MTP change (agreed with ChatGPT 2026-10-02)
- [ ] ggml vs Stingray Q8 activation cross-check on identical rows (the long40 pos-541 projection
      inputs). P/Invoke the vendored b10306 `ggml-base.dll` (`quantize_row_q8_K_ref`,
      `dequantize_row_q8_K`) and `ggml_get_type_traits_cpu` vec_dot (q4_K/q6_K x q8_K).
      - Per 256-block: amax, scale, % blocks identical, differing q per block, max |dq|, dequant RMS
        vs x (ggml Q8_K vs our Q8_KS).
      - Four-way projection decomposition: FP64(W.x) oracle, FP64(W.deq(q_ggml)),
        FP64(W.deq(q_ours)), actual ggml vec_dot vs our dot. Report absolute error next to relative
        (cancellation rows).

## Code (CPU only; Vulkan/CUDA never had the gate)
- [x] Save the experiment switches as a patch: `docs/4-performance/patches/2026-10-02-prefill-numerics/hybrid-gdn-experiment-switches.patch`
- [ ] Revert the experiment code in `HybridGdnForwardPass.cs`
- [ ] Remove `!_hasMtp` from the `Prefill` gate
- [ ] Rewrite the comments on `GdnChunkedPrefillEnabled`, the `Prefill` gate and `PrefillChunked`
      (decision, evidence, rollback switch; drop the stale "byte-identical" / "only the recurrence
      differs" claims)
- [ ] Build, Fast tests

## Tests (`HybridGdnForwardPassTests.cs`)
- [ ] `MtpDecoder_GreedyParity_LlamaCpp` doc: already diverges from llama.cpp at token 1 on the
      exact path; re-baseline on the fast path when `Qwen3.6-27B-MTP-Q4_K_M.gguf` is available
- [ ] `HybridGdnChunkedPrefill_MatchesSequentialPrefill` doc: 0.9995 cosine floor was calibrated on
      one model; point to the margin certificate (m > 2δ) as the principled criterion

## Docs
- [ ] `docs/reference/adr-0002-hybrid-gdn-mtp-chunked-prefill.md`: context, evidence (matrix +
      probe + row-check tables), decision, rollback switch `STINGRAY_GDN_CHUNKED_PREFILL=0`
      (strict sequential control), the "really bad turn" criteria, open follow-ups.
      Terminology: retire "exact". Use StrictSequential (A, deterministic control path, not ground
      truth), ChunkedRecurrence (R), Optimized (E, production), DoubleGdnReference (D, FP64
      recurrence only, int8 projections). "Numerical operation reference" = FP64 dot over
      dequantized weights x supplied FP32 activations, per operation only. "A reproducible
      computation is not necessarily a more accurate one."
- [ ] `docs/reference/env-var-inventory.md`: real entry for `STINGRAY_GDN_CHUNKED_PREFILL`
- [ ] `docs/4-performance/perf-sweep-plan.md` Phase 2: replace "numerics decision for the user"
      with the decision + ADR link
- [ ] Re-measure Qwen3.8-27B (and Qwen3.6-27B) prefill, uncontended; dated `PerformanceLeague.md` rows
- [ ] `docs/RUNNING.md` 27B rows (measured speed, dated)
- [ ] `docs/STATUS.md` if a hybrid-GDN row is affected
- [ ] Commit only these hunks (other agents may have work in the tree)

# TODO: int8 prompt processing on by default (Option A)

User decision 2026-10-02: `STINGRAY_CPU_PREFILL_Q8` defaults ON for dense models; routed MoE
experts stay F32 (separate gate `STINGRAY_MOE_PREFILL_Q8`, default off). Rollback:
`STINGRAY_CPU_PREFILL_Q8=0`.

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
- [ ] `docs/1-correctness/13-granite4-h-small-moe-ppl-parity-plan.md` (in `docs/done/`): note
      dense on, MoE still off
- [ ] Re-measure prefill with the new default **uncontended, one run at a time**: SmolLM2-1.7B,
      Qwen2.5-0.5B, Gemma-3-4B, Gemma-4-12B; dated League rows; fill ADR-0003's speed table
- [ ] `docs/RUNNING.md`: note rows measured 2026-10-01..02 used the F32 default
- [ ] Commit only these hunks

## Follow-ups: closing the remaining gaps (from the other AI's MoE audit; verify each first)
- [ ] NEW, from the row check: the per-token int8 decode matvec has 1.2e-2 mean / 8e-2 worst
      rel-RMS per projection (vs 9e-7 for the F32 GEMM). Compare our `QuantizeRowToQ8KS` + dot
      against ggml's `quantize_row_q8_K` + `vec_dot_q4_K_q8_K` on the same rows, to tell normal
      int8 noise (activation amax/rms 25-90 here) from a defect
- [ ] Our int8 prefill is not llama.cpp's: SmolLM2 F32 matches llama.cpp PPL to 0.03%, int8 is
      0.69% off, although llama.cpp also uses int8 activations. Check our Q8_KS (8 scales/256) vs
      ggml Q8_K (1 scale/256) per weight type; align where it differs
- [ ] Q3_K batched path uses Q8_KS while decode uses Q8_K (claimed; verify in
      `TryResolveQ8Dispatch`). Unifying on Q8_K would make Q3_K prefill match decode and ggml
- [ ] MoE: quantize the token activations once before expert dispatch (ggml `mul_mat_id` style)
      instead of per expert bucket: k-fold less quantize work, identical activations per token
- [ ] MoE: keep router and SiLU*up in F32, quantize only at GEMM inputs (check current code)
- [ ] MoE: small prompts (N <= 32) through the folded decode-style dispatch (bit-identical to
      decode, fewer tiny GEMMs); measure
- [ ] Then re-run the Granite-4-H-Small PPL/NLL comparison and decide `STINGRAY_MOE_PREFILL_Q8`

# TODO: TensorSharp takeaways (user order 2026-10-02: 1, 2, 3, then 4)

Plan: `docs/2-coverage/2026-10-02-tensorsharp-takeaways-plan.md`; backlog lines added to
`docs/00-current-work.md` (§4 CPU 4a/4b, §2 item 14); CLAUDE.md rule 14 ("ported, not verified"
stays internal). Uncommitted.
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
- [~] 4. IN PROGRESS (Muse-Glimmer wired `8a096e36`). Port families (Qwen 3.8 Flash Next, GLM-5.x, Muse-Glimmer, DiffusionGemma, MiniMax-H3;
      DeepSeek V4 review): NOT admitted, NOT advertised, logged in the plan's table
- Each port adds its file to the TensorSharp section of `THIRD_PARTY_NOTICES.md`

## Follow-ups (not blocking)
- [ ] 27B MTP validation with the same probe (Qwen3.6-27B and Qwen3.8-27B): short/boundary, ~500,
      ~2000 tokens; A vs E vs llama.cpp at the first generated token, `<think>` boundary, first
      post-boundary tokens, one normal position; prefill->decode continuity
- [ ] MTP draft acceptance, StrictSequential vs Optimized prefill, same prompts: accepted per
      round, acceptance by draft position, generated output (identical draft traces not required)
- [ ] llama.cpp b10306 comparison via `llama-server` pre-sampling `n_probs`
- [ ] Optional: name the strict path explicitly (e.g. a `StrictSequential` prefill mode) instead
      of only the env switch
