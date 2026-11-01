# Completed items split from todo.md (2026-11-01)

Finished items moved verbatim out of [../todo.md](../todo.md). Open items stay there. Each block is preceded by the heading it sat under. Relative links inside the blocks were rebased for this folder.

> from: ## Ideas, in the order I would do them
- [x] Unrelated finding (resolved: CLAUDE.md rule 8 now says they are git-ignored local checkouts): `CLAUDE.md` said `examples/*.cpp` and `tools/llama.cpp` were "checked into this repo". They are git-ignored (0 tracked files), so other contributors do not have them. Decide the wording.

> from: ## Code (CPU only; Vulkan/CUDA never had the gate)
- [x] *(closed 2026-11-01: the clean change landed as `e63c520d` on 2026-10-02 and the file has no experiment code; only the documented diagnostic and rollback switches remain)* Revert the experiment code in `HybridGdnForwardPass.cs`

> from: ## Code (CPU only; Vulkan/CUDA never had the gate)
- [x] Remove `!_hasMtp` from the `Prefill` gate (verified 2026-10-31: the gate is `GdnChunkedPrefillEnabled` only)

> from: ## Code (CPU only; Vulkan/CUDA never had the gate)
- [x] Rewrite the comments on `GdnChunkedPrefillEnabled`, the `Prefill` gate and `PrefillChunked`
      (decision, evidence, rollback switch; drop the stale "byte-identical" / "only the recurrence
      differs" claims)

> from: ## Code (CPU only; Vulkan/CUDA never had the gate)
- [x] *(closed 2026-11-01: the Fast suites run on every merge in CI, green on `main`)* Build, Fast tests

> from: ## Tests (`HybridGdnForwardPassTests.cs`)
- [x] *(closed 2026-11-01: the caveat and the margin-certificate criterion are in the test's doc comment, `HybridGdnForwardPassTests.cs`)* `HybridGdnChunkedPrefill_MatchesSequentialPrefill` doc: 0.9995 cosine floor was calibrated on
      one model; point to the margin certificate (m > 2δ) as the principled criterion

> from: ## Docs
- [x] `docs/reference/adr-0002-hybrid-gdn-mtp-chunked-prefill.md`: context, evidence (matrix +
      probe + row-check tables), decision, rollback switch `STINGRAY_GDN_CHUNKED_PREFILL=0`
      (strict sequential control), the "really bad turn" criteria, open follow-ups.
      Terminology: retire "exact". Use StrictSequential (A, deterministic control path, not ground
      truth), ChunkedRecurrence (R), Optimized (E, production), DoubleGdnReference (D, FP64
      recurrence only, int8 projections). "Numerical operation reference" = FP64 dot over
      dequantized weights x supplied FP32 activations, per operation only. "A reproducible
      computation is not necessarily a more accurate one."

> from: ## Docs
- [x] `docs/reference/env-var-inventory.md`: real entry for `STINGRAY_GDN_CHUNKED_PREFILL`

> from: ## Docs
- [x] `docs/4-performance/perf-sweep-plan.md` Phase 2: replace "numerics decision for the user"
      with the decision + ADR link

> from: ## Docs
- [x] *(closed 2026-11-01: committed; the changes are on `main`)* Commit only these hunks (other agents may have work in the tree)

> from: ## TODO: int8 prompt processing on by default (Option A)
- [x] Flip the default in `SimdKernels.Q8PrefillEnabled` (`!= "0"`); rewrite its doc comment (and
      move it onto the property: it was attached to `BatchedMatVecTierCalls`)

> from: ## TODO: int8 prompt processing on by default (Option A)
- [x] Build clean; ForwardPass.Fast 748 passed / 0 failed (1 skip: `MatVecThroughput_ByDType`,
      a throughput benchmark); Server.Fast 429/429

> from: ## TODO: int8 prompt processing on by default (Option A)
- [x] `docs/reference/adr-0003-cpu-int8-prefill-default.md` written (evidence, 2026-10-01 history,
      MoE scope, hybrid/RWKV guard note, rollback criteria)

> from: ## TODO: int8 prompt processing on by default (Option A)
- [x] `docs/reference/env-var-inventory.md`: `STINGRAY_CPU_PREFILL_Q8` default on, `=0` rollback

> from: ## TODO: int8 prompt processing on by default (Option A)
- [x] `perf-sweep-plan.md` 10.2 (decision + follow-up list) and Phase 6 (decision, re-measure pending)

> from: ## TODO: int8 prompt processing on by default (Option A)
- [x] `docs/WHAT-YOU-CAN-DO.md`: stale DeepSeek/int8 sentence fixed

> from: ## TODO: int8 prompt processing on by default (Option A)
- [x] Checked old int8 defects: the 2026-08-07 repeated-token collapse is fixed by
      `ForwardPass.IsSingleDistinctTokenPrompt`. The hybrid path has no such guard but measured no
      collapse. RWKV is unverified (no checkpoint on disk), noted in ADR-0003

> from: ## TODO: int8 prompt processing on by default (Option A)
- [x] Heavy suite with int8 on: 166/1/58, the only failure being known item 24. Committed `d7fcf135`

> from: ## TODO: int8 prompt processing on by default (Option A)
- [x] `docs/done/13-granite4-h-small-moe-ppl-parity-plan.md`: note
      dense on, MoE still off

> from: ## TODO: int8 prompt processing on by default (Option A)
- [x] *(closed 2026-11-01: committed; the changes are on `main`)* Commit only these hunks
