# Qwen 3.8 (`qwen4exp`) remaining work

Date: 2026-11-08. Scope: only the open items for Qwen 3.8 Flash Next. General coverage work for all
families stays in [2026-11-07-coverage-completion-programme-plan.md](2026-11-07-coverage-completion-programme-plan.md).
Qwen 3.8 stays **not admitted and not advertised** until Phase 8 (CLAUDE.md rule 14).

Sources for each item are the dated sections of the programme plan (Qwen sections, lines ~109-905) and
[2026-10-03-qwen38-flash-next-port-plan.md](2026-10-03-qwen38-flash-next-port-plan.md). Where those two
disagree, the programme plan's later dates win; the conflicts are listed under "Doc hygiene" below.

Legend: `[ ]` open, `[x]` done. "Runnable now" = no owner decision and no download needed.

---

## Owner decisions needed (blocking the items marked below)

- [ ] **D1. Confirm the working parity criterion** (adopted by Claude 2026-11-08 under standing delegation, reversible).
      Blocks: Phase 0.3 closure, and the final "parity verified" wording.
- [ ] **D2. Allow building the unsloth fork `mtp/qwen4exp-nextn` as a local reference** (third-party code, not in this repo).
      The auto-mode classifier denied this on 2026-11-08 and nothing was retried. Blocks: Phase 4 per-step oracle.
- [ ] **D3. Decide whether the server acts on tool-argument validator failures** (reject, return error, or pass through).
      Blocks: Phase 6 server wiring and the server round trip.
- [ ] **D4. Approve admission** (Phase 8). Needs Phases 0-7 closed or explicitly waived.

---

## Phase 0. Numerical parity (real checkpoint, paged)

- [ ] **0.3 FFN injection gate.** Still open. The residual MoE error from layer 1 is attributed to quantisation noise
      and is not confirmed. Runnable now (checkpoint on disk: `E:\_models\qwen38-flash-next\UD-IQ1_S`).
- [ ] Re-state the Phase 0 section so it no longer says both "closed" and "open" (see Doc hygiene).

## Phase 1. Per-sequence state: broader real-weight verification

Runnable now. Test cases only, no new disk or implementation.

- [ ] More fixed prompts (beyond the one-prompt paged run of 2026-11-07).
- [ ] Incremental decode checked against a full-prefill reference.
- [ ] A prompt that crosses a prefill/decode chunk boundary.
- [ ] A long prompt that exercises QSA K-pool selection (L > 4 for the pool, and L > `indexer_top_k` for selection).
- [ ] A documented tolerance for the real-weight comparison, with the noise envelope it was derived from.

## Phase 2. Batching

- [ ] **Real-checkpoint cancellation memory check.** `CancelledRequest_DoesNotChangeTheOthers` asserts
      `CommittedKvTokens == 0`, but the batching test checks reservations and not cache disposal.
      Runnable now (real checkpoint).
      Design (recorded before code):
      - Test only, no product change. A wrapper `IBatchedForwardPass` over the real `Qwen4ExpBatchedForwardPass` hands
        out its own cache wrapper, which delegates every call to the real cache and counts `Dispose`. The wrapper does
        not implement `IPrefixCacheableBatchedForwardPass`, so the engine runs without the prefix cache and every cache
        is a plain `ISequenceKvCache`.
      - Real checkpoint, skipped when absent. Scenario A: a long request is cancelled after its first chunk. Scenario B:
        a request runs to completion. Both must end with `created == disposed` and `CommittedKvTokens == 0`, and the
        check polls with a timeout so a leak fails rather than hangs.
      - Scope: this shows that the engine disposes the real Qwen cache and releases the reservation on a real request.
        It does not measure the real held bytes (a separate open item).

## Phase 3. Prefix reuse and retained state

- [ ] **Job 3: memory accounting completion.** Design committed in 5c89581f, before code. Parts (a) and (b) are implemented;
        the remaining open item is the real-checkpoint measurement (below). Pieces:
  - [x] `KvBytesPerToken` becomes the exact per-token sum (done; full Core 1417 tests, 0 failed; tests `Qwen4ExpPerTokenBytesTests`, 6; the real-checkpoint value is not measured yet): K and V, raw indexer key, amortised pooled key
        (`ceil(4 x indexerKeyLength / indexerKPool)` bytes), and 4 bytes of PLE history. Counted from the real layer tensors.
        Changes the session ABI fingerprint (acceptable: no persisted qwen4exp sessions exist).
  - [x] `ISnapshotAccountedSequenceKvCache.HeldSnapshotBytes` (new), `RetainedSequenceState` exposes it,
        `HotSession.CurrentResidentBytes` includes it, and eviction returns it. Done with the full Core gate at 1419 tests,
        0 failed. Tests: `HotSessionSnapshotAccountingTests` (2; a mutation check that removes the snapshot from residency
        fails both), `Qwen4ExpSequenceHeldBytesTests` (2).
  - [x] Reservation projection for a turn that will take a snapshot (`IBatchedForwardPass.KeepsTurnSnapshot`, default false;
        true for the Qwen pass). Known gap: no test isolates the reserved figure, so it is covered only indirectly through the
        residency tests.
  - [x] Tests: arithmetic against real list lengths (`Qwen4ExpPerTokenBytesTests`, `Qwen4ExpSequenceHeldBytesTests`); held bytes
        zero before a snapshot and positive after a turn; residency includes them; eviction frees them.
- [x] The turn-start snapshot's fixed state (GDN and PLE conv buffers) is now charged while a snapshot is held (Job 3 part b).
  The copy itself is still made.
- [ ] Real-checkpoint measurement of the fixed bytes per sequence. Not measured yet; no number is quoted anywhere.
- [ ] Cross-session copy-on-write page sharing (Job 8). **Design only**; not started. Deferred.

## Phase 4. MTP speculative decoding

- [ ] **Per-step oracle against the reference.** Blocked: the pinned llama.cpp `bed0a8566` has a stub embedding-state
      setter, so the MTP context cannot receive the trunk state. Next oracle is the unsloth fork (see D2). Also
      recorded: fork builds crash on the head's null recurrent memory (`build_rs_inp_impl`) and fork `llama-server`
      with `-md` segfaults during draft setup.
- [ ] Acceptance rate, multi-token acceptance, rollback on rejection.
- [ ] Separate speed measurement (rule 7 applies later).

## Phase 5. Image and video input

- [ ] Logit-level comparison with the reference (not only text).
- [ ] Exact 40-token equality. The current test compares text only.
- [ ] Image input through the tool and server paths.
- [ ] 5.3 Video input. Untested; deferred by the owner. Leave unchecked until the owner reopens it.

## Phase 6. Tool calls

- [ ] Server wiring: the server does not use the validator yet. Waits on D3.
- [ ] Round trip through the server tool pipeline. Waits on admission (rule 14).

## Phase 7. Serving and product

- [ ] Telemetry for Qwen sessions.
- [ ] Knowledge collections.
- [ ] Developer workflows.
- [ ] Browser UI (7F): **parked by the owner 2026-11-08.** Leave unchecked; do not start.

## Real-weight qualification (from the original port plan's Phase 6, reconciled)

The original plan's Phase 6 checkboxes were never ticked, and the 2026-11-07 paged run was the first real-weight
execution. Keep these open until each is done on the 62 GB checkpoint:

- [ ] 6.1 Sharded GGUF loading and tensor inventory: all shards load via memory-mapped pointers; `per_layer_token_embd`
      stays on disk with no managed RAM allocation.
- [ ] 6.2 PLE sparse touch: instrument `Qwen4ExpPleRowStore` to show a decode touches only the gathered head rows (8-32), not the table.
- [ ] 6.3 Paged correctness run: single-token prefill plus 5 decode steps; a prompt crossing a K-pool boundary; a prompt
      crossing the sparse selection threshold.
- [ ] 6.4 External parity gate: greedy token sequence and logit distributions against the pinned llama.cpp (`bed0a8566`)
      or TensorSharp. Overlaps Phase 0 and Phase 1; close them together.
- [ ] Check the Hugging Face file list for `unsloth/Qwen3.8-Flash-Next-GGUF` (recorded as the "next check" on 2026-10-31).
- [ ] Read the model card licence before any fixture decision (not checked on 2026-10-31).

## Phase 8. Verification and admission

- [ ] Full verification pass (Phases 0-7 closed or explicitly waived by the owner).
- [ ] Performance pass and DRY pass (CLAUDE.md rule 7). Qwen path speed has not been measured (programme plan, Phase 0.2 Q5_K
      fix: "Speed of the Qwen path is not measured yet"). The Q5_K numerics change (`MatVecLlamaNumerics`) is a candidate to measure.
- [ ] Docs and status: `docs/STATUS.md` row and `ModelCompatibility.cs` entry, only after D4 (rule 10 and rule 14).
- [ ] Admission (D4).

---

## Out of scope for this plan

- `deepseek41`, `deepseek32`, `glm-dsa`, `glm5next` work: other families, see the programme plan.
- Vision towers for other families, CUDA and Vulkan graph paths for Qwen 3.8, new ports.

## Doc hygiene (stale or conflicting statements to fix when these items are touched)

- [ ] `2026-10-03-qwen38-flash-next-port-plan.md` status line still says "AWAITING REAL-WEIGHT QUALIFICATION" and the
      "Next check" line predates 2026-11-07. Update, or point to this file.
- [ ] `Architectures/Qwen4ExpArchitecture.cs` `RefusalReason` still describes wrong logit magnitudes. The
      2026-11-08 measurements do not support that (external review check, programme plan).
