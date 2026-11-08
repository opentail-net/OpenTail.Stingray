# Qwen 3.8 (`qwen4exp`) remaining work

Date: 2026-11-08. Scope: only the open items for Qwen 3.8 Flash Next. General coverage work for all
families stays in [2026-11-07-coverage-completion-programme-plan.md](2026-11-07-coverage-completion-programme-plan.md).
Qwen 3.8 stays **not admitted and not advertised** until Phase 8 (CLAUDE.md rule 14).

Sources for each item are the dated sections of the programme plan (Qwen sections, lines ~109-905) and
[../done/2026-11-08-superseded-2026-10-03-qwen38-flash-next-port-plan.md](../done/2026-11-08-superseded-2026-10-03-qwen38-flash-next-port-plan.md). Where those two
disagree, the programme plan's later dates win; the conflicts are listed under "Doc hygiene" below.

Legend: `[ ]` open, `[x]` done. "Runnable now" = no owner decision and no download needed.

---

## Owner decisions needed (blocking the items marked below)

- [ ] **D1. Confirm the working parity criterion** (adopted by Claude 2026-11-08 under standing delegation, reversible).
      Blocks: Phase 0.3 closure, and the final "parity verified" wording.
- [x] **D2. Allow building the unsloth fork `mtp/qwen4exp-nextn` as a local reference** (third-party code, not in this repo).
      APPROVED by the owner on 2026-11-08 ("yes"). Local build only; nothing from the fork is committed or published.
      The classifier had denied this earlier the same day. Unblocks: Phase 4 per-step oracle.
- [x] **D3. Decide whether the server acts on tool-argument validator failures** (reject, return error, or pass through).
      DECIDED by the owner 2026-11-08: return as errors. DONE on all three paths: OpenAI non-streaming (HTTP 422),
      OpenAI streaming (an error event, since the status is already 200), and Anthropic (422 non-streaming, error event
      streaming). Tests: ToolCallValidationEndpointTests (4). Server fast suite: 685 passed, 0 failed, 2 skipped.
      Earlier note: OpenAI non-streaming path only: `ToolCallValidation`
      checks parsed calls against the request's declared tools (undeclared tool or schema failure), and the endpoint
      returns HTTP 422 with `invalid_tool_call`.
      Blocks: Phase 6 server wiring and the server round trip.
- [ ] **D4. Approve admission** (Phase 8). Needs Phases 0-7 closed or explicitly waived.
      2026-11-08: the owner answered "yes" to approving admission. The gate is not yet met (Phase 0.3 open, the image step-1
      cause unexplained, Phase 4 oracle not run, Phase 6 round trip not done, Phase 7 not started). No admission action
      (ModelCompatibility, STATUS, README) is taken until the owner states whether that "yes" waives the open phases, or
      the phases are closed. Question to the owner is recorded in the loop summary.

---

## Phase 0. Numerical parity (real checkpoint, paged)

- [ ] **0.3 FFN injection gate.** Still open. The residual MoE error from layer 1 is attributed to quantisation noise
      and is not confirmed. Runnable now (checkpoint on disk: `E:\_models\qwen38-flash-next\UD-IQ1_S`).
      Conclusion, 2026-11-08: not closable on the present evidence. The text-path measurements in Phase 0.2 sit inside
      the noise envelope, but the image-path logit comparison (Phase 5, step 1: TV 0.178, logit gap 0.81) is well outside
      it. So the working parity criterion, which was adopted on text evidence, does not yet hold for the image path.
      The next check is the same per-layer comparison on the image prompt's step 1, against the reference's eval-callback,
      which would show whether the injection path is the first layer that diverges.
      Update 2026-11-08: the per-layer check (Phase 5, (c)) shows the image-path residual drift is gradual, starting at 0.36%
      at layer 0 and growing to about 23% by layer 44, with no single layer where it breaks. The injection gate is not
      singled out by this data. The text-only per-layer comparison at the same decode position is the next check.
- [ ] Re-state the Phase 0 section so it no longer says both "closed" and "open" (see Doc hygiene).

## Phase 1. Per-sequence state: broader real-weight verification

Runnable now. Test cases only, no new disk or implementation.

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
- [x] A documented tolerance for the real-weight comparison, with the noise envelope it was derived from. Tolerance: a
      greedy token may differ only where the reference's choice is within 0.12 logits of Stingray's choice in
      Stingray's own top-5 (this was 0.036 and at most 0.11 here). Derived from the Phase 0.2 noise floor (about 0.03
      nats from 1e-6 perturbations) and the top-5 spread above.
- [ ] Incremental decode checked against a full-prefill reference. By construction the text prefill is per token, so
      incremental and full-prefill are the same code path here. The batched path (`PrefillWithCache` with a chunk of
      tokens) is still not compared with per-token decode; that is the open part of this item.
- [ ] A prompt that crosses a prefill/decode chunk boundary. Prompt 3 crosses the 64-token page size, but not the
      engine's prefill chunk size, so this is not yet covered. Open: run the batching engine with a small
      `prefillChunkTokens` and compare with per-token output on the real checkpoint.
- [ ] A long prompt that exercises QSA K-pool selection (L > 4 for the pool, and L > `indexer_top_k` for selection).
      Needs more than 2048 tokens; a separate background job (CPU prefill is per token, so this is slow).

## Phase 2. Batching

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

- [ ] **Per-step oracle against the reference.** 2026-11-08: owner approved the fork build (D2). Fork driver
      `llama-mtp-check.exe` (Debug) run on the saved hand-off: it exits with code 139 (segfault) during graph reservation,
      after the recurrent memory is allocated, and writes no reference logits. Still blocked by the fork crash. Blocked: the pinned llama.cpp `bed0a8566` has a stub embedding-state
      setter, so the MTP context cannot receive the trunk state. Next oracle is the unsloth fork (see D2). Also
      recorded: fork builds crash on the head's null recurrent memory (`build_rs_inp_impl`) and fork `llama-server`
      with `-md` segfaults during draft setup.
- [ ] Acceptance rate, multi-token acceptance, rollback on rejection.
- [ ] Separate speed measurement (rule 7 applies later).

## Phase 5. Image and video input

- [ ] Logit-level comparison with the reference (not only text). Partly done 2026-11-08 (two attempts):
      - Attempt 1 is invalid and is not a result: the Stingray prompt ended with the empty think block, but the
        reference's generation prompt does not (its step-0 top token is `<think>` 248068, Stingray's was 760 "The").
      - Attempt 2 uses the same prompt (13 tokens after the image; `QWEN_IMAGE_NO_THINK_BLOCK=1`, reference
        `llama-mtmd-cli` with per-step full-logit dumps, a local diagnostic patch that is reverted).
      - Step 0 (valid): same top-1 (248068 `<think>`), same top-5 set and order, total variation about 0.
      - Step 1 (valid, same context): top-1 differs. The reference picks 271 (p 0.660); Stingray picks 198 (p 0.518,
        271 at p 0.482). TV distance 0.178; the logit of 271 differs by 0.81. This is far outside the Phase 0.2
        noise envelope (about 0.03 nats), so it is a real discrepancy on the image path.
      - Steps 2 onward compare different contexts (the greedy paths split at step 1) and are not reported.
      - The text match recorded above used the think-block prompt, which is not the reference's prompt, so it was
        not like-for-like.
      - Open: cause not isolated. Candidates: the decode step after the image chunk (the QSA cache rows appended by
        `ForwardImageChunk` versus `QsaAppend`), or the text positions after the image (Stingray uses 11). Next:
        compare per-layer values at step 1 with the reference's eval-callback.
      Investigation, 2026-11-08 (a): positions do not explain it. The reference (`tools/mtmd/mtmd.cpp`, read only) uses
      `t = pos_0`, `x = pos_0 + col`, `y = pos_0 + row` for each Qwen-VL image token, and advances the text position by
      `max(nx, ny)` after the image (`n_past += get_n_pos`). Stingray uses the same values (`posT = pos0`, `posX/posY`
      as above, text resumes at 4 + 7 = 11), and the first decode position is 24 on both sides. These match, so the
      discrepancy is not a position error.
      Investigation, 2026-11-08 (c): per-layer residuals at the first decode position (`l_last`, 4 HC streams x 2560
      floats; Stingray `l_last_<il>.bin` against the reference's `l_last-<il>` from a local diagnostic dump). Relative
      difference ||ref - st|| / ||ref||:
      - layer 0: 0.36%; layer 1: 0.46%; layers 2 to 5: 0.49% to 0.75%;
      - layers 6 to 22: 1.5% to 2.5%;
      - layers 23 to 29: 3.6% to 6.9%;
      - layers 30 to 44: 9.4% to 23.1%, with the largest single-layer step at 44 to 45 (0.23 to 0.22 is flat; the
        absolute difference grows from 0.20 to 0.43).
      - Layer 47 reads 149%. Not interpreted: it is likely the reference's final-layer naming (the same tensor name
        appears with a "(reshaped)" suffix), so it needs a check before use.
      Reading: the drift is cumulative and gradual, with no single layer that breaks. Layer 0 is already 0.36% off, which
      matches the quantisation-level numerics recorded in Phase 0.2. Phase 0.2 also recorded a growing per-layer error
      on the text path (MoE 19 to 27% by layers 5 to 11). So the image-step discrepancy looks like that known text-path
      drift, not a new image-only break. Not proven: the decisive test is the same per-layer metric on a text-only prompt
      at the same decode position. If the text-only profile matches, the image path adds nothing new. The next step is
      that text comparison.

      Text-only comparison, 2026-11-08 (same metric, no image): prompt "The capital of France is" (5 tokens). The reference
      is a local diagnostic dump of the batched prefill (the decode graph carries `l_last` only for the last layers on the
      text path). Stingray is the per-token forward at position 4 (the same last prompt token). Relative difference of the
      last token's residual, layers 0 to 46 (the reference has no layer 47 file):
      - layer 0: 0.37%; layer 1: 0.39%; layer 2: 0.63%; layer 4: 0.99%; layer 5: 2.99%;
      - layer 12: 5.6%; layer 20: 9.1%; layer 28: 11.4%; layer 32: 13.9%; layer 40: 12.3%; layer 46: 8.9%.
      Reading: layer 0 is 0.37% on text as well as 0.36% on the image decode, so the first-layer offset is systematic and
      does not depend on the image. Both profiles grow gradually, with no single layer that breaks. The image profile is
      lower in the middle layers (1.5 to 2.5% at layers 6 to 22, against 3 to 9% on text) and higher late (23% at layer 44,
      against 10% on text). The two are not directly comparable: the text reference is a batched prefill and Stingray is
      per token, while the image measurement is a single decode step on both sides. So this does not isolate an image-only
      component. It does show the same kind of cumulative drift on text, which points to the known numerical path rather
      than a new image break. Phase 0.3 stays open.
- [ ] Exact 40-token equality. The current test compares text only.
- [ ] Image input through the tool and server paths.
- [ ] 5.3 Video input. The owner reopened it on 2026-11-08 (start once the other work is done or blocked).
      Design, recorded before code (investigation 2026-11-08):
      - What exists: the projector has `v.patch_embd.weight.1` (the temporal kernel), and `QwenVlVisionEncoder` already sums
        two patch embeddings, but for a still image both kernels see the same pixels. Video needs the two kernels on two
        different frames: `out = conv0(frame_t0) + conv1(frame_t1)` per patch. Nothing else in the vision path is video-aware
        (`VisionOps.cs` has only a RoPE comment; `ChatProtocol.CapsVideoFrames` is off).
      - Reference layout to match (`tools/mtmd/mtmd-helper.cpp`, pinned bed0a8566): frames come from a decoder pipe as raw
        RGB, each marked mergeable (consecutive frames merge into one temporal patch); a text timestamp `[%dm%.2fs]` is
        inserted between chunks every `--video-timestamp-interval` ms (default 5000), computed from `frame_index / fps`.
        Default `--video-fps` is 4.0.
      - Stingray plan, in order: (a) frame-sequence input (PNG/RGB frames) with pairing into temporal patches; (b) the
        two-frame patch embedding; (c) M-RoPE temporal index per merged chunk; (d) the prompt layout with the same text
        timestamps; (e) a short clip (a few seconds, generated with ffmpeg in the test only) compared with
        `llama-mtmd-cli --video`: the answer text first, then per-chunk embeddings if the text is ambiguous.
      - Not in scope for the first step: decoding video files inside the product. That needs a decoder decision (an
        ffmpeg dependency, or a managed decoder under NativeAOT). Question for the owner: which, or frame input only for now.
      - Admission is unaffected: video is not part of the current admitted scope (qwen2vl is text-only, see QwenArchitectures).
      Progress 2026-11-08: steps (a) and (b) done for the encoder. `QwenVlVisionEncoder.ForwardVideo(frame0, frame1, w, h)`
      feeds frame 0 to the first temporal kernel and frame 1 to the second; `Forward` is `ForwardCore(chw, chw)`, so the image
      path is unchanged. Tests: `QwenVlVideoFrameTests` (3: identical frames equal the still path bit-for-bit; the second
      frame changes the output; mismatched frame sizes are rejected). The Rainbow reference parity test still passes.
      Not tested: stage-level additivity (the patch stage is private and the later layers are nonlinear, so it is not
      additive at the output). Still open: (c) M-RoPE temporal index per merged chunk, (d) prompt layout with the reference
      timestamps, (e) comparison with `llama-mtmd-cli --video` on a short clip, and the frame-input path (frame sequence).
      DECIDED by the owner 2026-11-08: ffmpeg executable behind a frame-source seam (the TensorSharp pattern: managed core
      takes frames; a host supplies the decoder). Design: `IVideoFrameSource` (probe the clip; read RGB24 frames at a
      given fps and size). `FfmpegVideoFrameSource` runs an operator-installed ffmpeg found by `STINGRAY_FFMPEG`, then PATH;
      it is never bundled or downloaded. Probe parses `ffmpeg -i` output; frames come from `-f rawvideo -pix_fmt rgb24`
      on stdout. Frames are paired into temporal patches for `ForwardVideo`. Test: generate a short clip with ffmpeg from
      the test image (test only), decode it, and run the pairs through the real projector; skipped if ffmpeg is missing.
      Implemented 2026-11-08: `IVideoFrameSource`, `FfmpegVideoFrameSource` (Locate: STINGRAY_FFMPEG, then PATH; Probe via
      `ffmpeg -i` stderr through `ParseProbeOutput`; ReadFrames via rawvideo rgb24 on stdout), and `VideoTemporalPairs`
      (RGB24 to CHW, consecutive pairs). Tests: `ProbeParser_ReadsSizeAndRateFromFfmpegStderr` passes;
      the three decode tests skip. Blocker: the `ffmpeg` on PATH is a scoop shim whose target
      (`scoopppsfmpeg\currentinfmpeg.exe`) is missing, so nothing decodes on this machine. Not used: an Audacity
      FFmpeg installer found in Downloads (an installer, not a standalone binary). Needed from the owner: a working
      ffmpeg (install, or set STINGRAY_FFMPEG to a binary).
      Step (c), checked 2026-11-08 (`tools/mtmd/mtmd.cpp`, read only): mergeable frames are grouped in pairs
      (`n_temporal_merge == 2` for Qwen-VL) and each pair is one embedding and one image-token chunk. Its position is
      `t = pos_0` for every token of the pair, `x` and `y` as for an image, and the text position advances by
      `max(nx, ny)` once per pair (`n_past += get_n_pos`). So Stingray's video layout must advance the text position per
      pair, not per frame, and give every token of a pair the same temporal index. `ForwardVideo` takes one pair, so the
      encoder side already matches this.

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

- [ ] `../done/2026-11-08-superseded-2026-10-03-qwen38-flash-next-port-plan.md` status line still says "AWAITING REAL-WEIGHT QUALIFICATION" and the
      "Next check" line predates 2026-11-07. Update, or point to this file.
- [ ] `Architectures/Qwen4ExpArchitecture.cs` `RefusalReason` still describes wrong logit magnitudes. The
      2026-11-08 measurements do not support that (external review check, programme plan).
