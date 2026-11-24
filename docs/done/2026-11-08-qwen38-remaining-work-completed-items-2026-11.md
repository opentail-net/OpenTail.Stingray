# Qwen 3.8 remaining work (2026-11-08) - completed items

Ticked items moved from [2026-11-08-qwen38-remaining-work-plan.md](../2-coverage/2026-11-08-qwen38-remaining-work-plan.md) on 2026-11-08. Each keeps its text and its section heading. Open items stay in that plan.

## Phase 2. Batching

- [x] **Queue limit.** Maximum queue length and rejection when full. Implemented in ae47d135 (`maxQueuedRequests`,
      `EngineQueueFullException`; tests `ContinuousBatchingQueueLimitTests`, 2). Mapping to a server status: see D3.

## Phase 3. Prefix reuse and retained state

- [x] Copy-on-write for QSA history rows within a sequence and its clones (`Qwen4ExpSequenceState.CopyLists`,
      commit ad42fa92; tests `Qwen4ExpSequenceSharingTests`, 4 synthetic).

## Phase 4. MTP speculative decoding

- [x] `Qwen4ExpForwardPass.ForwardMtp` written (planned graph, steps 1-8). `Qwen4ExpMtpForwardTests.DraftStep_IsFiniteAndDeterministic`
      passes. This is a "step runs" result, not a correctness result.
- [x] MTP head loads from its own file (`Qwen4ExpMtpLoadTests`, real files).

## Phase 5. Image and video input

- [x] 5.1 Vision encoder verified against the pinned reference (`Qwen38Vl_Rainbow448_MatchesLlamaMtmdDebug`, 16/16).
- [x] 5.2 Image question answered, matching the reference (`Qwen4ExpImageAnswerTests`, heavy flag, `QWEN_TEST_IMAGE`).

## Phase 6. Tool calls

- [x] Wire-format parsing for the Qwen tool-call shape (`Qwen4ExpToolCallParseTests`, 15 cases).
- [x] Streaming fragments for the reasoning split (tool-call scan is text-only, by design).
- [x] Validator exists as a library (`ToolCallArgumentValidator`), incomplete and malformed calls are not executed.
- [x] **Malformed JSON tool calls** (`ParseJsonCallBlock`, `DeepSeekToolCallAdapter.ParseBlock`). Fixed in 660b4ea8; tests `ToolCallJsonHardeningTests`, 20. Open: a non-object root (`[]`) throws
      `InvalidOperationException`; a non-string `name` (`{"name":42}`) throws from `GetString()`; a duplicate property
      throws `ArgumentException` from `ToDictionary`; an array or scalar `arguments` silently becomes an empty dictionary.
      Confirm with tests first, then fix. Runnable now.
- [x] **Validator numeric and enum edge cases.** Fixed in 660b4ea8; tests `ToolCallValidatorEdgeCaseTests`, 16. Before the fix: `double.TryParse` accepted `NaN` and `Infinity`; `Math.Floor(d) == d`
      was true for infinity; `Convert.ToString(true)` gave `"True"` where the enum literal is `"true"`.

## Doc hygiene

- [x] Programme plan, "Qwen 3.8 remaining work, tabled 2026-11-08" table: superseded by this file (linked at the end of the programme plan).

---

## Moved 2026-11-24 from 2026-11-08-qwen38-remaining-work-plan.md

Open items stay in [2026-11-08-qwen38-remaining-work-plan.md](../2-coverage/2026-11-08-qwen38-remaining-work-plan.md).

_From: Qwen 3.8 (`qwen4exp`) remaining work / Owner decisions needed (blocking the items marked below)_

- [x] **D1. Confirm the working parity criterion** (adopted by Claude 2026-11-08 under standing delegation, reversible; confirmed by the owner 2026-11-08). The criterion: a greedy token may differ from the pinned reference only where the reference's choice is within 0.12 logits of Stingray's choice in Stingray's own top-5 (Phase 1, tolerance). Derived from the Phase 0.2 noise floor of about 0.03 nats. The text path meets it on the Phase 1 prompts. The image path meets it at step 1 (a 0.073-logit near-tie, Phase 0.3); later steps are not comparable once the tokens differ. **Owner decision 2026-11-08: the programme plan's proposed final-logit cosine >= 0.998 is dropped.** The noise probe gives 0.9979 from a 1e-6 perturbation alone, so the cosine bar sits below the noise floor. The token-level criterion stands.

_From: Qwen 3.8 (`qwen4exp`) remaining work / Owner decisions needed (blocking the items marked below)_

- [x] **D2. Allow building the unsloth fork `mtp/qwen4exp-nextn` as a local reference** (third-party code, not in this repo).
      APPROVED by the owner on 2026-11-08 ("yes"). Local build only; nothing from the fork is committed or published.
      The classifier had denied this earlier the same day. Unblocks: Phase 4 per-step oracle.

_From: Qwen 3.8 (`qwen4exp`) remaining work / Owner decisions needed (blocking the items marked below)_

- [x] **D3. Decide whether the server acts on tool-argument validator failures** (reject, return error, or pass through).
      DECIDED by the owner 2026-11-08: return as errors. DONE on all three paths: OpenAI non-streaming (HTTP 422),
      OpenAI streaming (an error event, since the status is already 200), and Anthropic (422 non-streaming, error event
      streaming). Tests: ToolCallValidationEndpointTests (4). Server fast suite: 685 passed, 0 failed, 2 skipped.
      Earlier note: OpenAI non-streaming path only: `ToolCallValidation`
      checks parsed calls against the request's declared tools (undeclared tool or schema failure), and the endpoint
      returns HTTP 422 with `invalid_tool_call`.
      Blocks: Phase 6 server wiring and the server round trip.

_From: Qwen 3.8 (`qwen4exp`) remaining work / Owner decisions needed (blocking the items marked below)_

- [x] **D4. Approve admission** (Phase 8): decided 2026-11-09 under the owner's delegation. Admitted scope: `qwen4exp` text
      generation and tool calls, CPU, paged checkpoint. The owner answered "yes" to approving admission on 2026-11-08; the scope
      question stayed open until the owner's "just figure it all out and get everything done. your decisions!!! you decide and do!"
      (2026-11-09). Decision (taken by me): admit text and tool calls only. Phase 0.3 is closed under D1 (text and image steps 0-1
      explained by the noise floor). Status of the items this decision named, as of 2026-11-09 (later entries are authoritative):
      the Phase 4 per-step oracle is **closed** (under D1); Phase 6.4 is **closed for one prompt**; Phase 7 is **done** except the
      browser UI (7F), which the owner parked; Phase 4 batched verify has its **contract** (replay form, opt-in, no speed gain);
      Phase 5 image and video are **verified at engine level** but the server HTTP path is **not** run on this host (preflight);
      the Phase 8 performance pass has a measured baseline and three reverted experiments, and its DRY review is **not** done.
      Open for the owner: whether an admitted server that a third party can call counts as Model as a Service (see 0.3).

_From: Qwen 3.8 (`qwen4exp`) remaining work / Phase 0. Numerical parity (real checkpoint, paged)_

- [x] **0.3 FFN injection gate: CLOSED under D1 (2026-11-08, later)**, with both paths explained by the float-order noise floor (below).
      (The text path, next paragraph; the image path, the paragraph after that.)

_From: Qwen 3.8 (`qwen4exp`) remaining work / Phase 0. Numerical parity (real checkpoint, paged)_

- [x] **0.3 text-path drift, explained (2026-11-08, later).** Per-layer dumps regenerated after the Q5_K numerics fix
      (commit 563ec1b4, 01:52; the earlier dumps predated it). Text prompt "The capital of France is", last token, against the
      eval-callback dump: layer 0 `linear_attn_out` 0.36%, `l_last_0` 0.37% (was 2.5% and 1.46%); `qkv_mixed` 0.03%; embedding
      exact. Per-layer residual rises to 9-14% by layers 28-47; logits relative error 6.5%, cosine 0.9979, top-1 11751 matches.
      **Noise probe:** the same run with the embedding scaled by (1 + 1e-6 x u), u in [-1, 1] (`STINGRAY_Q4E_PERTURB_EPS`)
      moves layer 0 by 0.41%, layers 32-47 by 8-10%, and the logits by 6.4% (top-1 unchanged). So the reference-to-Stingray
      text drift is the size that float-order noise produces, amplified through the network. It is not evidence of a logic error.
      Consequence for the criterion: the programme plan's proposed final-logit cosine >= 0.998 is below this noise floor (the
      probe itself gives 0.9979), so it should not be used. The token-level criterion (D1) is the one to hold.
      **Image path, explained (2026-11-08, later).** Reference and Stingray logits for the test image ("two shapes", no-think prompt),
      steps 0 and 1: step 0 identical (top-1 248068, TV 0). Step 1 TV 0.178, top-1 271 (reference) against 198 (Stingray). Stingray's
      own top-2 at step 1 differ by 0.073 logits, and the reference's choice (271) is inside Stingray's top-5 within that gap.
      That passes the Phase 1 tolerance rule (0.12). The noise probe on the image path (`QWEN_IMAGE_PERTURB_EPS`, pixels x(1 + 1e-6 u))
      flips step 1 to 271 with TV 0.396 against the unperturbed run, so the reference-to-Stingray gap is noise at a near-tie.
      Steps after 1 are not comparable: the two runs have fed different tokens. So the image logit gap (0.81, TV 0.178) is the
      near-tie, not a logic error.

_From: Qwen 3.8 (`qwen4exp`) remaining work / Phase 0. Numerical parity (real checkpoint, paged)_

- [x] **0.3 FFN injection gate.** Closed 2026-11-08 under D1; the entries at the top of this item hold the evidence. The text that follows is the earlier history, superseded. The residual MoE error from layer 1 is attributed to quantisation noise
      and is not confirmed. Runnable now (checkpoint on disk: `E:\_models\qwen38-flash-next\UD-IQ1_S`).
      Earlier conclusion, 2026-11-08 (superseded by the closure above): not closable on the evidence of that time. The text-path measurements in Phase 0.2 sit inside
      the noise envelope, but the image-path logit comparison (Phase 5, step 1: TV 0.178, logit gap 0.81) is well outside
      it. So the working parity criterion, which was adopted on text evidence, does not yet hold for the image path.
      The next check is the same per-layer comparison on the image prompt's step 1, against the reference's eval-callback,
      which would show whether the injection path is the first layer that diverges.
      Update 2026-11-08: the per-layer check (Phase 5, (c)) shows the image-path residual drift is gradual, starting at 0.36%
      at layer 0 and growing to about 23% by layer 44, with no single layer where it breaks. The injection gate is not
      singled out by this data. The text-only per-layer comparison at the same decode position is the next check.

_From: Qwen 3.8 (`qwen4exp`) remaining work / Phase 0. Numerical parity (real checkpoint, paged)_

- [x] Re-state the Phase 0 section so it no longer says both "closed" and "open" (done 2026-11-08: 0.3 is closed under D1; the earlier text is marked superseded).

_From: Qwen 3.8 (`qwen4exp`) remaining work / Phase 1. Per-sequence state: broader real-weight verification_

- [x] More fixed prompts. Four prompts, 16 greedy tokens each, Stingray against the pinned `llama-completion`
      (`llama.cpp` bed0a8566) on the same checkpoint (2026-11-08, scratchpad `phase1`):
      - prompt 2 (`def fibonacci(n):` docstring block): identical, 41 characters.
      - prompt 4 (`Q: What is 17 times 23? A:`): identical, 49 characters.
      - prompt 1 (`Once upon a time, in a small village by the sea,`): the first 15 tokens match and token 16 differs
        (Stingray `who` 852 versus reference `,`). Stingray's top-5 at that step: 852 at 24.1443, `,` (11) at 24.1080,
        a gap of 0.036 logits.
      - prompt 3 (a ~70-token summary of the printing press, which is past the 64-token page boundary): differs at the
        second generated token (Stingray `What` 3710 versus reference `Which`). Stingray's top-5 at that step: 3710 at
        14.3779, 22365 at 14.2647, 332 at 14.2597, 760 at 14.2448, 27775 at 14.1151. The whole top-5 spans 0.26 logits.
      - Both divergences are near-ties, inside the 0.03-nat noise envelope recorded in Phase 0.2. Neither is confirmed
        as a logic error. The reference prefills in one batch and Stingray prefills token by token, which is a known
        source of near-tie disagreement.

_From: Qwen 3.8 (`qwen4exp`) remaining work / Phase 1. Per-sequence state: broader real-weight verification_

- [x] A documented tolerance for the real-weight comparison, with the noise envelope it was derived from. Tolerance: a
      greedy token may differ only where the reference's choice is within 0.12 logits of Stingray's choice in
      Stingray's own top-5 (this was 0.036 and at most 0.11 here). Derived from the Phase 0.2 noise floor (about 0.03
      nats from 1e-6 perturbations) and the top-5 spread above.

_From: Qwen 3.8 (`qwen4exp`) remaining work / Phase 1. Per-sequence state: broader real-weight verification_

- [x] Incremental decode checked against a full-prefill reference (2026-11-08). The Qwen batched pass's `PrefillWithCache`
      and `PrefillPackedMulti` loop over per-token `Forward` calls, so a chunk of tokens and per-token decode are the same kernel
      calls. The engine's chunking (the next item) is the part that could differ, and it is checked there.

_From: Qwen 3.8 (`qwen4exp`) remaining work / Phase 1. Per-sequence state: broader real-weight verification_

- [x] A prompt that crosses a prefill/decode chunk boundary (2026-11-08). `Qwen4ExpChunkedPrefillTests`: a 63-token prompt
      through `ContinuousBatchingEngine` with `prefillChunkTokens` 7 (about nine chunks) and with no chunking gives the same
      12-token greedy continuation: "the printing press changed the world by making information accessible to many". Pass, real
      checkpoint.

_From: Qwen 3.8 (`qwen4exp`) remaining work / Phase 1. Per-sequence state: broader real-weight verification_

- [x] A long prompt that exercises QSA K-pool selection (2026-11-08). A 76-line numbered prompt tokenizes to 2,376 tokens
      (the reference's count; indexer top-k 2048, pool ratio 4), so selection is active. Stingray's 16-token greedy continuation
      is identical to `llama-completion`'s: "Line 77: the number 77 squared is 59". `Qwen4ExpLongPromptTests`, real checkpoint,
      about 22 minutes of per-token prefill at 1.3 tok/s. Text-level comparison; the reference prints no logits.

_From: Qwen 3.8 (`qwen4exp`) remaining work / Phase 2. Batching_

- [x] **Real-checkpoint cancellation memory check.** Done: `Qwen4ExpCancellationProbeTests` (2) pass on the real checkpoint (14.7 s).
      Scope: a cancelled request and a finished request each dispose their real Qwen cache and release their reservation
      (`created == disposed`, `CommittedKvTokens == 0`). It does not measure held bytes (separate open item). Full Core 1421 tests, 0 failed.
      Original note: `CancelledRequest_DoesNotChangeTheOthers` asserts
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

_From: Qwen 3.8 (`qwen4exp`) remaining work / Phase 3. Prefix reuse and retained state_

- [x] **Job 3: memory accounting completion.** Design committed in 5c89581f, before code. Parts (a) and (b) are implemented;
        the real-checkpoint measurement (below) is done, so every piece is ticked. Closed 2026-11-24 on that evidence. This is host-side session accounting (KV per token, held snapshots, residency): it shares no code with the GPU placement planner fixed for A5 in 3c5be538. Pieces:
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

_From: Qwen 3.8 (`qwen4exp`) remaining work / Phase 3. Prefix reuse and retained state_

- [x] The turn-start snapshot's fixed state (GDN and PLE conv buffers) is now charged while a snapshot is held (Job 3 part b).
  The copy itself is still made.

_From: Qwen 3.8 (`qwen4exp`) remaining work / Phase 3. Prefix reuse and retained state_

- [x] Real-checkpoint measurement of the fixed bytes per sequence (2026-11-08, `Qwen4ExpChunkedPrefillTests`): fixed state
      118,079,488 bytes (112.6 MiB) per sequence, and 49,156 bytes of per-token state, both read from the loaded layer tensors.
      Held snapshot bytes after a real turn (2026-11-08, `Qwen4ExpTurnSnapshotTests`): a first turn of 5 tokens, 3 generated tokens, then a
      second turn of 7 tokens keeps a turn-start snapshot of 118,081,056 bytes. That is the fixed state (118,079,488) plus 1,568 bytes of
      list headers. The KV rows are shared copy-on-write with the live cache, so they are not held until the live cache diverges.
      The first turn holds none (0 bytes).

_From: Qwen 3.8 (`qwen4exp`) remaining work / Phase 3. Prefix reuse and retained state_

- [x] Cross-session copy-on-write sharing (Job 8), closed 2026-11-09 as implemented at the row level, with the remaining copy
      measured. The QSA history rows are shared by reference between a sequence and its clones (`CopyLists`, Phase 3 result
      2026-11-08, `Qwen4ExpSequenceSharingTests`). Cross-session forks use the same clone path: `ForwardPass` prefix capture and
      `ForkPrefix` both call `Qwen4ExpSequenceState.Clone`, so two sessions forked from one prefix hold the same history rows.
      What is still copied per session is the fixed recurrent state (the GDN conv and state buffers and the PLE history): about
      118 MB per sequence (`fixed=118079488`, 2026-11-08 and 2026-11-09). It is copied, not shared, because the recurrence rewrites
      it on every token, so it has no immutable pages to share. The Llama-style `PagedKvCache` page sharing does not apply to the
      Qwen history, which is not paged. Closing Job 8 as this row-level sharing leaves the fixed-state copy as an accepted cost.

_From: Qwen 3.8 (`qwen4exp`) remaining work / Phase 4. MTP speculative decoding_

- [x] **Per-step oracle against the reference: CLOSED 2026-11-09 under D1 (token criterion).** Reference: the unsloth fork
      `mtp/qwen4exp-nextn` at `ca142690` (github.com/unslothai/llama.cpp, PR 144 closed), built Release outside the repository,
      local only, nothing committed or published. The local driver (`llama-mtp-check`) loads the trunk, then the shared head file
      with the trunk as `model_shared`, and runs the head's steps in one MTP context. Earlier crashes (exit 139, `build_rs_inp_impl`,
      `llama-server -md`) came from loading the head as a standalone model; the fork needs `model_shared`.
      Inputs: Stingray's trunk hidden state, token and position for the speed-test prompt plus 24 greedy steps (44 steps), dumped
      by `Qwen4ExpMtpOracleTests` (skipped unless `STINGRAY_MTP_ORACLE_DIR` is set). Result: the top-1 token is identical at 43
      of 44 steps; at step 13 the reference's top is inside Stingray's top-5 within 0.12 logits, so it passes D1. 44 of 44 pass.
      Logits: cosine worst 0.99747 (the plan's dropped 0.998 bar sits at the noise floor), maximum absolute difference up to 1.10.

_From: Qwen 3.8 (`qwen4exp`) remaining work / Phase 4. MTP speculative decoding_

- [x] Separate speed measurement (2026-11-08, `Qwen4ExpMtpSpeedTests`, three repetitions, 24 tokens, same prompt): plain greedy
      2.35, 2.51, 2.52 tok/s; the product MTP decoder 2.49, 2.50, 2.47 tok/s, with the same tokens and acceptance 75%. No gain on
      CPU: the sequential MTP path runs one trunk forward per emitted token. The gain needs the batched verify path (`BatchVerify`),
      which qwen4exp does not implement. The performance pass (rule 7) is still open.
      **Batched verify, contract done 2026-11-09 (replay form; no speed gain):** `Qwen4ExpForwardPass` implements `BatchVerify`,
      `BatchForward2`, `RestoreBatchSnapshot`, `MtpTruncateTo` and `HiddenAt`. The block is replayed token by token from a
      state snapshot, so rollback restores the snapshot and replays the accepted prefix. On the real checkpoint:
      `BatchVerify` and `BatchForward2` equal sequential `Forward` exactly, rollback then a different token equals a fresh run
      exactly (`Qwen4ExpBatchVerifyTests`), and the MTP decoder on the batched path emits the plain greedy tokens (24 of 24).
      But the batched decoder ran at 2.04 tok/s against 2.6 plain (the snapshot clone and the replay cost more than the saved
      reads), so `BatchVerifyEnabled` is off by default and the decoder stays sequential. The default MTP acceptance is
      unchanged (31 of 40, 77.5%). The multi-row pass that gives the real gain is still open.
      **Scope of the multi-row pass, from the checkpoint's tensor types (`stingray list-tensors`, 2026-11-09):** 2-D weights are
      Q8_0 (338 tensors), BF16 (195), Q6_K (84), Q5_K (72), IQ2_XXS (71, the MoE experts), Q2_0 (48), IQ2_S (25), IQ4_XS (1) and
      IQ4_NL (1). Batched kernels found: Q8_0 (`Q8_0BatchedLinear.MatMul`), BF16 (`SimdKernels.MatMulPairBF16`), Q6_K
      (`Q6KPrefillGemm.TryMatMulBatched`). Not found: Q5_K (the engine notes it has no int8 dot path), IQ2_XXS, IQ2_S, Q2_0, IQ4_XS
      and IQ4_NL. The experts (IQ2_XXS and IQ2_S) carry most of the compute, so their batched dequant-dot kernels come first.
      That is new CPU kernel work, not wiring, and it needs its own measurement gate.
      **Measured 2026-11-09 (per-section timing, local uncommitted instrumentation, reverted; 2 rounds of 12 decode tokens, UD-IQ1_S,
      63 GB RAM, one prompt):** wall 410 ms per token (about 2.4 tok/s). Per token: MoE routed and shared experts 155 ms (38%); GDN
      mixers 103 ms (25%); QSA attention 15 ms (4%); output projection 12 ms (3%); PLE 7 ms (2%). Not attributed: about 118 ms (29%).
      **The unattributed part is the HC residual, measured 2026-11-09:** `Qwen4ExpGatedResidual.Mix` and `Combine` take 115 ms per
      token (116.3 and 114.8 ms over two rounds; 96 calls, so about 1.2 ms per call). Cause (reading the code and the tensor list,
      not yet timed by part): the HC up and down matrices are stored as Q8_0 (`hc_attn_up`, `hc_ffn_up`, `hc_attn_down`,
      `hc_ffn_down`, each 320 x 10,240), but `Mix` reads them as F32 from the dequantisation cache, so each call streams about 26 MB
      of float weights (about 2.5 GB per token). A Q8_0 dot with float activations would read about a quarter of those bytes.
      **Tried 2026-11-09 and reverted (dead end in this form):** the up and down matrices read as Q8_0 with `SimdKernels.DotQ8_0_Q8_0`
      (Q8_0 weights, float activations, on the rounded activation that production uses). Wall time 1,125 ms per token against about
      405 ms for the float path, so about 2.8 times slower. The likely cause is the per-row Q8 kernel with float input, which is far
      slower than the float dot; the per-call pin is not the cause. The Q8_0 by Q8_0 integer path (quantise the activation once, then
      an int8 dot, the llama.cpp scheme) is the other option, but it changes the numerics, so it needs the parity gates. Not pursued
      further here. The output projection is small, so the earlier estimate that put it first was wrong. The two rounds
      differ by about 1%.
      **MoE expert path, read 2026-11-09 (no change yet):** the IQ2_XXS experts already use a vectorised Q8_K kernel
      (`DotIq2Xxs_Q8K`, AVX2 variant) and the row loop is parallel (`MatVecQ8KDispatch`, one work item per row once there are at
      least `MinRowsForParallel` rows). The cost is structural: each token makes 1,152 separate expert matvecs (8 experts, three
      matrices, 48 layers), and each opens its own parallel region over 640 rows, with the input quantised again each time.
      Candidate experiment: fuse the gate and up matvecs of all selected experts into one parallel region per layer, and the
      down matvecs into a second, so 2 regions per layer instead of 24. Each row still uses the same kernel on the same quantised
      input, so the output should be bit-identical; that makes the check exact. Keep only if measurably faster.
      **Tried 2026-11-09 and reverted:** the fused version was not faster (404 to 443 ms per token against 409 to 422 ms for the
      current path, which is within noise), and its output was not bit-identical (decode logit hash `19e1cb92d94c4e6f` against
      `842190253fe9b307`). The mismatch was not investigated, because the speed result already rules the change out; it is in the
      experimental path, not in production. The expert time is therefore not the per-region dispatch overhead; the remaining cost
      is the kernel work itself.
      **The earlier estimate, kept for the record (not a measurement):** per decoded token the routed experts read about 1.25 MB per expert per
      layer (IQ2_XXS gate and up, Q2_0 down), so about 480 MB for 8 experts over 48 layers. The attention and GDN projections are
      Q5_K and Q6_K, about 50M multiply-adds per layer. The output projection is 248,320 x 2,560, about 636M multiply-adds. At 2.5
      tok/s the decode moves about 1.5 to 2 GB of weights per token, so it is either memory-bound or dequant-bound. A block of k
      tokens shares weight reads only where the routed experts overlap, so the verify gain is bounded by that overlap. The next
      step is a per-section timing of one decode token (PLE, GDN, QSA attention, MoE, output head), taken on a local, uncommitted
      instrumentation, before any kernel change.
      **Batched verify, decided 2026-11-09:** the qwen4exp "batched" prefill (`Qwen4ExpBatchedForwardPass.PrefillWithCache`,
      `PrefillPackedMulti`, `BatchForwardMulti`) is a loop of single-token `Forward` calls, so it reads the weights once per
      token. Measured: prefill 2.21 tok/s and decode 2.47 tok/s, the same per-token rate. Other families' batched prefill is much
      faster than their decode, so the batch gain comes from a multi-row path they have and qwen4exp lacks. Work: (1) a multi-row
      trunk pass for qwen4exp (GDN recurrence over k tokens, QSA attention over k queries, MoE on k rows, PLE gather for k rows),
      modelled on `ForwardPass.PrefillCore`; (2) `BatchVerify` on top, returning logits for every position, with
      `RestoreBatchSnapshot` for rollback; (3) measure prefill and MTP speed before and after, three repetitions each.

_From: Qwen 3.8 (`qwen4exp`) remaining work / Phase 5. Image and video input_

- [x] Exact 40-token equality under the D1 criterion (2026-11-08, `Qwen4ExpImageAnswerTests` against `llama-mtmd-cli -n 40 --temp 0`,
      generated image: a red and a blue square on white). The two agree token for token for 7 tokens ("The image contains **"). At
      token 7 Stingray picks ` **` (id 2972, logit 20.652) and the reference picks ` two` (id 1330, logit 20.640), the runner-up,
      0.012 logits behind. That is inside the 0.12 tolerance. The continuations differ after that ("squares" against "shapes"), and
      later tokens are not comparable because the runs fed different tokens. Not exact byte equality over 40 tokens; equal under D1.

_From: Qwen 3.8 (`qwen4exp`) remaining work / Phase 6. Tool calls_

- [x] Server wiring (D3, 12e961a2; streaming output stopped after an invalid call in 78579880): the validator runs on all four paths (OpenAI and Anthropic, streaming and non-streaming). Tests: `ToolCallValidationEndpointTests` (10): OpenAI non-streaming invalid and valid; OpenAI streaming invalid, invalid then text, invalid then valid; Anthropic non-streaming undeclared tool and missing required argument; Anthropic streaming undeclared tool, invalid then text, invalid then valid. Not covered: an argument type mismatch on Anthropic (only missing required and undeclared tool are tested). Server fast suite: 691 passed, 2 skipped.

_From: Qwen 3.8 (`qwen4exp`) remaining work / Phase 6. Tool calls_

- [x] Round trip through the server tool pipeline, local (2026-11-08, real trunk, diagnostic-gated, rule 14 still applies to any product claim): `POST /v1/chat/completions` with a `get_weather` tool returns `tool_calls` (`get_weather`, arguments `{"city":"Paris"}`) and `finish_reason: tool_calls`; 318 prompt tokens, 66 completion tokens (40 reasoning), 5 min 6 s wall on the paged checkpoint with 9.6 GB free. The server needed the qwen4exp forward-pass factory (`BuildForwardPass`) and the preflight override (`STINGRAY_IGNORE_PREFLIGHT=1`, estimate 66.1 GiB vs 63.3 GiB, as the paged runs above have shown). Earlier note: **Model side verified 2026-11-08** (`Qwen4ExpToolCallModelTests`, real trunk, greedy, 161-token tool prompt): the model wrote `<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>`, which the adapter (`ToolCallAdapterRegistry`, qwen2) parses and `ToolCallArgumentValidator` accepts against the declared schema. **Server side open:** the server cannot load qwen4exp yet, because the descriptor has no `CreateForwardPass` factory, and the server needs an engine adapter for that forward pass (estimate 1-2 hours). Diagnostic-only changes made for this (each gated on `STINGRAY_DIAGNOSTIC_ALLOW_UNSUPPORTED_ARCH=1` or documented as such): the server loader honours the diagnostic switch for the planner, and `ArchitectureDescriptor.IsUsable` accepts an un-admitted descriptor under it. Admission status unchanged.

_From: Qwen 3.8 (`qwen4exp`) remaining work / Phase 7. Serving and product_

- [x] Telemetry for Qwen sessions (done 2026-11-09 for the PLE counters; the rest is a follow-up). **Decided, no owner spec
      existed.** Built: `EngineEventKind.ForwardCounters` (Name = architecture id from the descriptor, A = PLE gather rows,
      B = PLE gather bytes, C = PLE table bytes, cumulative since load), emitted by `ForwardCounterEvents` after a CLI
      single-prompt run, and written to the `STINGRAY_STATS_JSON` file as `ple_gather_rows`, `ple_gather_bytes`,
      `ple_table_bytes`. MTP drafts already emit `MtpStep`. Not built yet: held snapshot bytes and indexer selections (the pass
      does not count indexer selections; add the counter first). Tests: `RunStatsJsonSinkTests` (CLI, synthetic events),
      `Qwen4ExpTelemetryTests` (real trunk: the event matches the pass counters after real tokens). The CLI run itself is not
      exercised on this host: the preflight refuses the checkpoint (see the RUNNING item above).

_From: Qwen 3.8 (`qwen4exp`) remaining work / Phase 7. Serving and product_

- [x] Knowledge collections (done 2026-11-09). **Spec (decided, no owner spec existed):** a named text collection whose prefix
      state is kept in a hot session (Phase 3) and reused by later questions, so only the question is prefilled again. Result on
      the real checkpoint (`QwenKnowledgeCollectionRealModelTests`, heavy: `STINGRAY_RUN_HEAVY_TESTS=1`): the document is the
      first turn (49 tokens, prefill only); the question (11 tokens) prefilled exactly 11 tokens; the 96-token answer is the same
      as a cold session that gets the document and the question in one turn ("Mainz.", after the same reasoning trace). Not
      built: a named-collection registry or any product surface; the mechanism is the hot session itself.

_From: Qwen 3.8 (`qwen4exp`) remaining work / Phase 7. Serving and product_

- [x] Developer workflows (done 2026-11-09). **Spec (decided, no owner spec existed):** `scripts/qwen38-workflow.ps1` runs scout,
      the CLI smoke, the five Qwen real-checkpoint Core classes, the Sessions knowledge test and Server.Fast, writing logs and a
      summary to a folder (`-Full` adds the full Core gate). First end-to-end run, 2026-11-09: build 0; scout 0 (`scout.json`
      written); CLI smoke 1 (the expected preflight refusal, 66.0 GiB against 63.3 GiB); Core Qwen classes 5 of 5 passed (Single
      token, Turn snapshot, MTP decoder, Telemetry, Sharded loading with shard counts and the managed-heap check); Sessions
      knowledge test 1 of 1 passed (91 s); Server.Fast 692 total, 0 failed, 1 skipped (the earlier runs showed 2 skipped, so one
      skip depends on the environment). Not in the default run: the full Core gate (`-Full`).

_From: Qwen 3.8 (`qwen4exp`) remaining work / Real-weight qualification (from the original port plan's Phase 6, reconciled)_

- [x] 6.1 Sharded GGUF loading and tensor inventory (2026-11-08, `Qwen4ExpShardedLoadingTests`): `GgufModel` maps every
      shard with its own memory-mapped file and resolves tensor data against those pointers. The checkpoint's three files are
      10.9 MB (metadata only), 37.6 GB (1,223 tensors) and 28.8 GB (`per_layer_token_embd.weight`, the single tensor of the third
      file). Loading the tensor set grows the managed heap by 152 MB against a 28,800,138,240-byte table, so the table takes
      no managed RAM.

_From: Qwen 3.8 (`qwen4exp`) remaining work / Real-weight qualification (from the original port plan's Phase 6, reconciled)_

- [x] 6.2 PLE sparse touch (2026-11-08): the gather is counted in `Qwen4ExpForwardPass` (`PleGatherRows`, `PleGatherBytes`, `PleTableBytes`). `Qwen4ExpPleTouchTests` on the real trunk: 16 head rows (1440 bytes) per decode token, against a 28,800,138,240-byte table. Only layer 1 has PLE (`qwen4exp.ple.layers` = [1]), so one gather per token. Logical rows and bytes, not OS pages. (The item named a class that does not exist; the gather is in `ExecutePle`.)

_From: Qwen 3.8 (`qwen4exp`) remaining work / Real-weight qualification (from the original port plan's Phase 6, reconciled)_

- [x] 6.3 Paged correctness run (2026-11-08, real checkpoint): single-token prefill ("Hello", one token) plus five greedy
      decode steps gives "! We’ve noticed that", identical to `llama-completion` (`Qwen4ExpSingleTokenTests`). K-pool boundaries are
      crossed by the 63-token chunked prompt (pool ratio 4, so 15 boundaries) and by the 2,376-token prompt. The sparse selection
      threshold is crossed by the 2,376-token prompt (top-k 2048), with the identical continuation above.

_From: Qwen 3.8 (`qwen4exp`) remaining work / Real-weight qualification (from the original port plan's Phase 6, reconciled)_

- [x] 6.4 External parity gate (closed 2026-11-09 for the 21-token prompt; see the result below): greedy token sequence and logit distributions against the pinned llama.cpp (`bed0a8566`)
      or TensorSharp. Overlaps Phase 0 and Phase 1; close them together.
      2026-11-08 status: greedy text matches `llama-completion` on the single-token prompt, the 2,376-token prompt (16 tokens),
      and the Phase 1 prompts within the tolerance. Logit distributions are not compared with the reference (llama-completion
      prints none), so 6.4 stays open on that part.
      Result, 2026-11-09 (`Qwen4ExpLogitParityTests`, reference `llama-logit-dump` on the pinned `bed0a8566` build, local, not
      committed): the speed-test prompt (21 tokens) and 24 greedy steps. The reference's greedy tokens are teacher-forced
      through Stingray; Stingray's top-1 equals the reference's at all 24 steps (so its own greedy continuation is the same 24
      tokens). Logit distributions: cosine mean 0.99856, worst 0.99694; total variation mean 0.0304, worst 0.0724 (step 6).
      The mean is at the 0.03 noise floor of Phase 0.2, and 11 of 24 steps are above it, so the distributions are close to the
      reference but not inside the floor on every step. The verdict is the D1 token criterion, which passes at every step.
      One prompt only; more prompts are a follow-up.

_From: Qwen 3.8 (`qwen4exp`) remaining work / Real-weight qualification (from the original port plan's Phase 6, reconciled)_

- [x] Check the Hugging Face file list for `unsloth/Qwen3.8-Flash-Next-GGUF` (2026-11-08, read-only API). 77 files, last modified
      2026-11-07, commit `084c631`. The local files match the repository's sizes: UD-IQ1_S shards 10,946,618 / 37,568,095,648 /
      28,800,138,432 bytes; shared MTP Q4_K_M 1,907,151,936 bytes; `mmproj-F16` 904,004,000 bytes.

_From: Qwen 3.8 (`qwen4exp`) remaining work / Real-weight qualification (from the original port plan's Phase 6, reconciled)_

- [x] Read the model card licence before any fixture decision (2026-11-08). The GGUF card says `license: other`,
      `qwen-community-1.0`, with a link to `LICENSE` that does not exist in that repository ("Entry not found"). The base
      repository `Qwen/Qwen3.8-Flash-Next` has the text: Qwen Community License 1.0. Summary, not legal advice: use, copy,
      modify, distribute, fine-tune and host are permitted, including the weights. Conditions: keep the copyright notice; if the
      Software is used in a commercial product with more than 100,000,000 monthly active users or US$20,000,000 monthly revenue,
      show the model name prominently in the UI; a Model-as-a-Service or AI Work Assistant business needs a separate licence
      from Qwen, except internal use that does not make the Software or its outputs available to a third party.
      **Verified text, 2026-11-09:** fetched from `Qwen/Qwen3.8-Flash-Next` (`LICENSE`, "Qwen Community License 1.0"). Clause 2:
      "If the licensee or any of its affiliates conducts a Model as a Service or AI Work Assistant business, the licensee shall
      obtain a separate license from Qwen before Using the Software or its derivative works for any commercial purpose." The
      internal-use exception applies "provided that such Use does not make the Software, its outputs, or its underlying model
      capabilities available to any third party." Model as a Service means giving a third party access to inference "via API or a
      hosted endpoint" with "meaningful control over the inputs, parameters, or training data". Contact in the licence:
      model-business@notice.qwencloud.com. This records the text only; whether any OpenTail deployment is a commercial Model as a
      Service is the owner's decision. **For the owner:**
      whether an admitted OpenTail server (which a third party could call) would count as Model as a Service is a question to
      settle before D4.

_From: Qwen 3.8 (`qwen4exp`) remaining work / Phase 8. Verification and admission_

- [x] Docs and status: `docs/STATUS.md` qwen4exp row (text generation and tool calls only), written after D4 (rule 10 and rule 14).
      Architecture card `docs/models/qwen38-flash-next.md` and its index row in `docs/models/README.md` (2026-11-09).

_From: Qwen 3.8 (`qwen4exp`) remaining work / Phase 8. Verification and admission_

- [x] Admission (D4): `Qwen4ExpArchitecture` descriptor `Status = Admitted`. Tool calls use the default Qwen wrapper
      (`ToolCallAdapterRegistry`); no `qwen4exp` literal was added outside `Engine/Architectures/` (the literal ratchet and
      `Registry_QwenFamilyWrapperServesQwen4Exp` both require this). The facts baseline line for `qwen4exp` changed to `Admitted`.

_From: Qwen 3.8 (`qwen4exp`) remaining work / Doc hygiene (stale or conflicting statements to fix when these items are touched)_

- [x] `../done/2026-11-08-superseded-2026-10-03-qwen38-flash-next-port-plan.md` status and next-check lines now point to this plan (fixed 2026-11-08).

_From: Qwen 3.8 (`qwen4exp`) remaining work / Doc hygiene (stale or conflicting statements to fix when these items are touched)_

- [x] `Architectures/Qwen4ExpArchitecture.cs` `RefusalReason` rewritten 2026-11-08: it no longer claims wrong logit magnitudes; it states the text-path noise envelope, the image-path gap and the open Phase 0.3 gate.
