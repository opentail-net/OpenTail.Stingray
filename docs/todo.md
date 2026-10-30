> Finished items moved to [done/todo-done-2026-10.md](done/todo-done-2026-10.md) on 2026-10-30. This file keeps only the open work.

# SCOUT / HUGGING FACE: IDEAS AND TODO (written 2026-10-09)

Built and committed so far (details: `docs/3-product-and-runtime/2026-10-09-checkpoint-scout-and-ai-admission-plan.md`, `docs/reference/061-coverage-tooling.md`):
`stingray scout` (local + `-r owner/repo`), features, structural-parent ranking against 12 shipped signatures, host working-set estimator + budget gate, `scripts/scout-pretest.ps1`,
one external-access policy (`STINGRAY_ALLOW_EXTERNAL`, default allowed, deny wins), `GgufModel.ParseIndex`, Hub client, pinned/bounded Range reader. Playbook: `docs/reference/architecture-admission-agent-playbook.md`; `CLAUDE.md` rule 15.

## First: prove it is useful (do this before building more consumers)
- [ ] Admit ONE real, annoying model with the playbook and write down where scout helped / misled / was irrelevant, with times. Candidates: `ornith-ai/Ornith-1.5-35B-A3B-GGUF`
      (qwen35moe + an extra next-n head: does our loader ignore the unused `nextn` tensors?), a relabel, or the next item in `docs/00-current-work.md`. If it does not save time, fix scout before adding features.

## Ideas, in the order I would do them
- [ ] Quant picker cost on large-vocab families: ~11 MiB per file (248k-token vocabulary), so a 31-quant repo = 352 MiB / 177 requests. All quants of a repo share their metadata size roughly; learn it from the first file and fetch the exact range in one request. Low priority.
- [ ] **Signature harvest**: remote-scout single-file quants of ADMITTED architectures we have no local file for and `--emit-signature` them (Hub provenance, no download). `qwen3moe` is admitted but ranks against dense `qwen3` only because it has no signature. Prefer single files: split models have no single hash.
- [ ] **Demand-ranked backlog**: `api/models?filter=gguf&sort=downloads&expand[]=gguf&expand[]=downloads` returns architecture, 30-day downloads, gated and total size per repo in ONE call (verified). Aggregate by architecture, mark registered/admitted, rank the unregistered.
      Caveat: downloads are per repository over 30 days, not per file, not inference use. Bounded `--limit`, list-only, never a crawler.
- [ ] **Batch triage**: feed a repo list through remote scout; buckets: identical to an admitted family / identical structure under a new name (relabel candidates, the fast lane, still needs a golden) / near-parent with differences / nothing close / blocked (dtype, tokenizer, size). Machine-readable.
- [ ] **Compare semantics-changing metadata too** (not only tensors): a short curated list (rope scaling type, sliding window, expert gating...) shown as separate informational differences. Not keys that vary with fine-tuning (rope base). qwen3 vs qwen3vl are tensor-identical and differ only in metadata.
- [ ] **Scout output as the bug-report format**: a user whose model fails pastes the JSON (no paths, no weights); maintainer reproduces with `scout -r` at the pinned commit.
- [ ] **Shared goldens index** (file sha256 -> golden): goldens are tiny, hash-pinned, path-free. Checking one needs no llama.cpp; only capturing does. Lets scout say "verified for exactly this file".
- [ ] **Estimator**: MLA (DeepSeek2), hybrid/GDN and RWKV are Unknown (blocked); other MoE families unmeasured (only Phi-3.5-MoE: experts not repacked); only one machine calibrated; GPU placement deliberately not estimated (CLAUDE.md rule 13). Re-measure when the engine changes (the Q4_K repack copy is why a Q4_K_M file costs ~1.8x its size).
- [ ] **Leave-one-out** (`scripts/scout-leave-one-out.ps1`) again whenever signatures grow; still n=12. Observed split: <=11 differences = real near-relative, >=14 = nothing close.
- [ ] **Pretest script gaps**: Ctrl+C mid-run and a hard kill of PowerShell untested (child would survive a hard kill); stage 4 has no command; `admit-arch` exit 1 = fail OR tool error (script classifies by verdict line); split exit codes if other callers allow.
- [x] Unrelated finding (resolved: CLAUDE.md rule 8 now says they are git-ignored local checkouts): `CLAUDE.md` said `examples/*.cpp` and `tools/llama.cpp` were "checked into this repo". They are git-ignored (0 tracked files), so other contributors do not have them. Decide the wording.
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
- [ ] `docs/done/13-granite4-h-small-moe-ppl-parity-plan.md`: note
      dense on, MoE still off
- [ ] Re-measure prefill with the new default **uncontended, one run at a time**: SmolLM2-1.7B,
      Qwen2.5-0.5B, Gemma-3-4B, Gemma-4-12B; dated League rows; fill ADR-0003's speed table
- [ ] `docs/RUNNING.md`: note rows measured 2026-10-01..02 used the F32 default
- [ ] Commit only these hunks

## Follow-ups: closing the remaining gaps
- [ ] MoE audit and int8 follow-ups: see [1-correctness/2026-10-02-prefill-numerics-investigation.md](1-correctness/2026-10-02-prefill-numerics-investigation.md) (section "Follow-ups from the MoE audit").

# TODO: TensorSharp takeaways (user order 2026-10-02: 1, 2, 3, then 4)

Plan: `docs/2-coverage/2026-10-02-tensorsharp-takeaways-plan.md`; backlog lines added to
`docs/00-current-work.md` (§4 CPU 4a/4b, §2 item 14); CLAUDE.md rule 14 ("ported, not verified"
stays internal). Uncommitted.
- [~] 4. IN PROGRESS (Muse-Glimmer wired `8a096e36`). Port families (Qwen 3.8 Flash Next, GLM-5.x, Muse-Glimmer, DiffusionGemma, MiniMax-H3;
      DeepSeek V4 review): NOT admitted, NOT advertised, logged in the plan's table
- Each port adds its file to the TensorSharp section of `THIRD_PARTY_NOTICES.md`

## Follow-ups (not blocking)
- [ ] 27B MTP validation: see [reference/adr-0002-hybrid-gdn-mtp-chunked-prefill.md](reference/adr-0002-hybrid-gdn-mtp-chunked-prefill.md) (open follow-ups; owner of the MTP thread).
- [ ] MTP draft acceptance, StrictSequential vs Optimized prefill, same prompts: accepted per
      round, acceptance by draft position, generated output (identical draft traces not required)
- [ ] llama.cpp b10306 comparison via `llama-server` pre-sampling `n_probs`
- [ ] Optional: name the strict path explicitly (e.g. a `StrictSequential` prefill mode) instead
      of only the env switch
