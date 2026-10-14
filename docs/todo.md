# SCOUT / HUGGING FACE: IDEAS AND TODO (written 2026-10-09)

Built and committed so far (details: `docs/3-product-and-runtime/2026-10-09-checkpoint-scout-and-ai-admission-plan.md`, `docs/reference/061-coverage-tooling.md`):
`stingray scout` (local + `-r owner/repo`), features, structural-parent ranking against 12 shipped signatures, host working-set estimator + budget gate, `scripts/scout-pretest.ps1`,
one external-access policy (`STINGRAY_ALLOW_EXTERNAL`, default allowed, deny wins), `GgufModel.ParseIndex`, Hub client, pinned/bounded Range reader. Playbook: `docs/reference/architecture-admission-agent-playbook.md`; `CLAUDE.md` rule 15.

## First: prove it is useful (do this before building more consumers)
- [ ] Admit ONE real, annoying model with the playbook and write down where scout helped / misled / was irrelevant, with times. Candidates: `ornith-ai/Ornith-1.5-35B-A3B-GGUF`
      (qwen35moe + an extra next-n head: does our loader ignore the unused `nextn` tensors?), a relabel, or the next item in `docs/00-current-work.md`. If it does not save time, fix scout before adding features.

## Ideas, in the order I would do them
- [x] **Quant picker** DONE 2026-10-09 (`scout -r repo --quants`; see 061). Original note: (`scout -r repo --quants`): remote-scout every quant, run the estimator + gate, print which fit this machine and the exact `pull` command pinned to the inspected commit. Uses what exists; likely daily-use.
- [x] **Estimator for the hybrid recurrent family (qwen35 / qwen35moe): DONE 2026-10-09** (4 models calibrated, see 061). Original note: **(qwen35 / qwen35moe / qwen4exp): HIGH VALUE.** The Hub's most-downloaded GGUF repos right now are this family (verified top-12 list), and the quant picker returns `unknown` for all of them. Needs the recurrent-state sizing (conv + delta state per recurrent layer, KV only for the full-attention layers)
      and a real calibration run (Qwen3.6-35B-A3B Q6_K is on the archive disk, ~27 GB; loads slowly from there, run it alone). Still open: qwen4exp (not measured), MLA (deepseek2) and RWKV.
- [x] **Scratch arena experiment (measure first)** CLOSED 2026-10-10 at P0, no-go (0.1-0.3% bound; see PerformanceLeague.md). **Plan with phases and times: `docs/4-performance/2026-10-09-scratch-arena-experiment-plan.md` (half a day if it fails at P0; ~1.5-2 working days / ~3 calendar days if it pays off).** (idea from chat: zero-allocation tensor arenas). Facts: 284 `NativeMemory.Alloc*` sites in 51 files, nearly all allocated ONCE at construction; GC share of decode is 0.4-0.9% (PerformanceLeague.md ~1306); an arena can save at most the GC-committed memory (~300 MiB on a 27B, ~2% of peak). Do NOT retrofit all sites (weeks, wrong-number risk, unmeasurable gain).
      Plan: (1) `NativeArena` utility (aligned slab, scoped bump/release, typed views, misuse checks, ~15 tests; hours, touches nothing existing); (2) use it ONLY for batched-prefill chunk scratch (~120 alloc/free pairs of 1-10 MB per chunk, flagged "unmeasured, measure before pooling" in perf-investigation-brief.md);
      (3) A/B on real weights, keep only if measurably faster (rule 7). Also the ~1 MiB/token logits `.ToArray()` copies (ContinuousBatchingEngine, CudaForwardPass) if wanted. Open question: is the hybrid family's ~335 MiB fixed base (dense ~135 MiB) managed heap or something else? Read `managedHeap`/`gcCommitted` from a hybrid run.
- [ ] Quant picker cost on large-vocab families: ~11 MiB per file (248k-token vocabulary), so a 31-quant repo = 352 MiB / 177 requests. All quants of a repo share their metadata size roughly; learn it from the first file and fetch the exact range in one request. Low priority.
- [ ] **Signature harvest**: remote-scout single-file quants of ADMITTED architectures we have no local file for and `--emit-signature` them (Hub provenance, no download). `qwen3moe` is admitted but ranks against dense `qwen3` only because it has no signature. Prefer single files: split models have no single hash.
- [ ] **Demand-ranked backlog**: `api/models?filter=gguf&sort=downloads&expand[]=gguf&expand[]=downloads` returns architecture, 30-day downloads, gated and total size per repo in ONE call (verified). Aggregate by architecture, mark registered/admitted, rank the unregistered.
      Caveat: downloads are per repository over 30 days, not per file, not inference use. Bounded `--limit`, list-only, never a crawler.
- [ ] **Batch triage**: feed a repo list through remote scout; buckets: identical to an admitted family / identical structure under a new name (relabel candidates, the fast lane, still needs a golden) / near-parent with differences / nothing close / blocked (dtype, tokenizer, size). Machine-readable.
- [ ] **Compare semantics-changing metadata too** (not only tensors): a short curated list (rope scaling type, sliding window, expert gating...) shown as separate informational differences. Not keys that vary with fine-tuning (rope base). qwen3 vs qwen3vl are tensor-identical and differ only in metadata.
- [x] `pull --revision` DONE 2026-10-09 (also pins to the listed commit by default). Still open: route `pull`'s download through `ExternalHttpClient`; make `pull` use `HubClient`'s listing parser (it keeps its own copy).
- [ ] **Scout output as the bug-report format**: a user whose model fails pastes the JSON (no paths, no weights); maintainer reproduces with `scout -r` at the pinned commit.
- [ ] **Shared goldens index** (file sha256 -> golden): goldens are tiny, hash-pinned, path-free. Checking one needs no llama.cpp; only capturing does. Lets scout say "verified for exactly this file".
- [ ] **Estimator**: MLA (DeepSeek2), hybrid/GDN and RWKV are Unknown (blocked); other MoE families unmeasured (only Phi-3.5-MoE: experts not repacked); only one machine calibrated; GPU placement deliberately not estimated (CLAUDE.md rule 13). Re-measure when the engine changes (the Q4_K repack copy is why a Q4_K_M file costs ~1.8x its size).
- [ ] **Leave-one-out** (`scripts/scout-leave-one-out.ps1`) again whenever signatures grow; still n=12. Observed split: <=11 differences = real near-relative, >=14 = nothing close.
- [ ] **Pretest script gaps**: Ctrl+C mid-run and a hard kill of PowerShell untested (child would survive a hard kill); stage 4 has no command; `admit-arch` exit 1 = fail OR tool error (script classifies by verdict line); split exit codes if other callers allow.
- [ ] Unrelated finding: `CLAUDE.md` says `examples/*.cpp` and `tools/llama.cpp` are "checked into this repo". They are git-ignored (0 tracked files), so other contributors do not have them. Decide the wording.
- [ ] Cap/expose index limits and rate behaviour for the Hub (retries, 429), and cache API responses by (repo, commit) if batch features arrive.

## Idea from chat: known-good checkpoints, favourites, local availability (2026-10-09)
**Planned in full: docs/3-product-and-runtime/2026-10-09-known-good-checkpoints-and-first-run-plan.md** (phases P0-P5, gaps G1-G9, acceptance criteria). The notes below are the short form.
Keep THREE things separate and join them with one workflow; build no new download mechanism and no new package format.
1. **Known-good catalogue**: exact repo, pinned revision, file, size, sha256, supported uses, memory needs, test evidence, limitations.
2. **User favourites/defaults**: personal aliases (`chat`, `coding`, `vision`, `speech`) that SELECT from the catalogue and can be overridden.
3. **Local inventory**: which exact files are present, intact, compatible with this machine.
Workflow for `stingray chat` when the checkpoint is absent: resolve favourite or default -> is that exact file installed? -> if not, remote-scout it and confirm the chosen quant is feasible here -> show checkpoint, download size, memory need, licence -> ask -> existing `pull`/installer fetches the pinned file and verifies the hash ->
preflight -> load only if it passes. If the default does not fit, offer another QUALIFIED quant or explain; never substitute an untested look-alike.
- **Most of this already exists.** `src/OpenTail.Stingray.Core/Catalog` already has `CatalogEntry`/`CatalogFile` (repo, pinned revision, path, sha256, size, task, licence + consent, hardware, speed, evidence, run template), a resumable hash-verifying `ModelInstaller`, `stingray setup <task>` and `stingray models`, with `STINGRAY_OFFLINE`.
  Extend that; do not create a parallel catalogue.
- **Missing:** (a) favourites/aliases layer; (b) qualification SCOPE per entry (engine version, backend, quant, context, capabilities such as chat / vision / tools; CPU chat does not imply Vulkan vision); (c) a remote feasibility check before download (scout -r + estimator); (d) the same evaluator used by the loader as by scout (a preflight that blocks unknown/over-budget BEFORE load);
  (e) fall back to another qualified quant; (f) local inventory view (present / intact / compatible).
- **Is it bullet-proof?** No, and it should not claim to be. "Defence in depth" is the right claim: structure understood (scout), exact checkpoint qualified for a stated purpose (catalogue), machine can run it (estimator + preflight), files intact (hash). Acceptance criteria, with where we stand:
  1. exact identity pinned (repo, revision, file, size, sha256), verified after download: EXISTS in the catalog/installer.
  2. fail-closed preflight shared by scout and loader: scout side exists; the LOADER does not call it yet.
  3. resource safety (weights + KV at the requested context + scratch + headroom): estimator exists for plain-attention dense/MoE on CPU only.
  4. qualification tied to a configuration: MISSING.
  5. layered validation, near-tie kept distinct from errors, no promotion on "it loaded": goldens/verify-goldens exist; wiring to the catalogue does not.
  Residual risk to state in docs: a tensor index cannot prove the weights mean what they should; a checksum proves identity, not correctness; a golden covers its inputs and configuration only. Say "structure compatible, file intact, budget feasible, configuration X passed tests Y", never "this model works".
- Open questions for the user: where do favourites live (env/profile JSON/`~/.stingray`)? Does `ask before downloading` stay the default for `chat`? Which uses/backends get a qualification column first (CPU chat only is honest)?

---
# RESUME HERE (updated 2026-10-03)

## Now
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
- [ ] Step 2: or a Q8_KS variant of the unified GEMM (keeps our decode precision; no reference
      implementation; the TensorSharp Q8_K kernels are the readable model).
- [ ] Measure: decode t/s, prefill t/s, PPL vs llama.cpp (SmolLM2: F32 0.03%, int8 0.69% today),
      decode-vs-prefill logit identity. Docs: ADR, League rows, perf-sweep-plan 10.2.

## Commits (newest first; none pushed)
- `b31d5f3a` perf(cpu): gate+up dual repacked GEMM, per-thread quant scratch (prefill +5.7%)
- `fb1f07e6` test: decode-via-GEMM model-level invariance (committed by the user's other AI)
- `a5af0272` feat: STINGRAY_CPU_DECODE_VIA_GEMM experiment (other AI; also committed root dumps + todo.md)
- `c9b47c69` perf(cpu): CpuSgemm replaces OpenBLAS (other AI, my files)
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
