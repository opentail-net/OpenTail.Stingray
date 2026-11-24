# Qwen 3.8 (`qwen4exp`) remaining work

> Completed items moved to [2026-11-08-qwen38-remaining-work-completed-items-2026-11.md](../done/2026-11-08-qwen38-remaining-work-completed-items-2026-11.md) on 2026-11-24. This file keeps only the open items.

Date: 2026-11-08. Scope: only the open items for Qwen 3.8 Flash Next. General coverage work for all
families stays in [2026-11-07-coverage-completion-programme-plan.md](2026-11-07-coverage-completion-programme-plan.md).
Qwen 3.8 is **admitted on 2026-11-09 for CPU text generation and tool calls** (decision D4 below, `docs/STATUS.md`).
Image input, video input and product MTP speculation are **not admitted**. The Qwen Community License question for a hosted
service is open for the owner (section 0.3).

Sources for each item are the dated sections of the programme plan (Qwen sections, lines ~109-905) and
[../done/2026-11-08-superseded-2026-10-03-qwen38-flash-next-port-plan.md](../done/2026-11-08-superseded-2026-10-03-qwen38-flash-next-port-plan.md). Where those two
disagree, the programme plan's later dates win; the conflicts are listed under "Doc hygiene" below.

Legend: `[ ]` open, `[x]` done. "Runnable now" = no owner decision and no download needed.

---

## Owner decisions needed (blocking the items marked below)


---

## Status 2026-11-24 (autonomous loop): what is left, and what blocks it

- **Job 3 (memory accounting): closed** on the real-checkpoint evidence below. Host-side session accounting, not shared with the GPU placement planner fixed for A5 in 3c5be538.
- **Logit-level comparison (open, blocked on a reference build and a heavy run).** The image step-1 top-1 flip (TV 0.178) has no isolated cause. Next step is per-layer values at the same decode position against the reference's eval-callback. Two things block it: (1) the vendored llama.cpp source is at `bed0a8566`, but the binaries are `b10306-cpu`, so a reference build must first be matched to b10306; (2) each dump is a full 62 GB paged run on this 63 GB host. Toolchain is present (cmake, MSVC BuildTools, ninja).
- **Logit item, step 1 done 2026-11-24:** the reference binary reports `version: 10306 (6b5c2efb4)` (`llama-mtmd-cli.exe --version`). The upstream tag `b10306` resolves to `6b5c2ef`, so a clean scratch clone at that tag was made (outside the repo). **Update 2026-11-24:** the owner approved the build. `llama-eval-callback` was built in the scratch clone, but it is text-only and cannot run the image path, so it was not used for this comparison (see the next bullet). Note: the reference was built with clang 20.1.8, so per-layer numbers from an MSVC build will carry a small toolchain difference.
- **Logit item, per-layer comparison on the image path, decode step 1 (2026-11-24).** Reference: the fork build in `examples/llama.cpp/llama.cpp/build-ref` (upstream commit `bed0a8566`; loads qwen4exp, and answers "two squares, one red on the left"). The upstream b10306 binary cannot load qwen4exp. A local dump patch in that fork's `mtmd-cli` (env `Q4E_REF_DUMP_DIR`, first single-token graph per tensor) saved 48 `l_last-N` files of [2560 x 4] F32. The source is restored to its original hash; the binary keeps the env-gated patch. Stingray: `Qwen4ExpImageAnswerTests` with `STINGRAY_Q4E_DUMP_DIR` and `STINGRAY_Q4E_DUMP_POS=24`. Position 24 is the first decode position on the M-RoPE axis (text resumes at 4 + 7 = 11, 13 post tokens); the cache index is 66, so 66 is wrong. Relative difference per layer: layer 0 0.36%, layer 2 0.50%, layer 5 0.79%, layer 10 1.07%, layer 15 1.51%, layer 20 1.98%, layer 24 3.85%, layer 28 5.62%, layer 32 11.9%, layer 36 17.6%, layer 40 23.9%, layer 44 28.1%, layer 46 32.4%. This reproduces the 2026-11-08 profile, so the dump and the reference agree. Layer 47 reads 157%, which is the naming question already noted above; not resolved. Still open: the same-position text-only comparison, which is what decides whether the image path adds anything beyond the known text-path drift. **Same-position text-only comparison, done 2026-11-24.** Reference: `llama-completion` (raw prompt, `-no-cnv`, same fork build, same dump patch, first single-token graph per tensor). Prompt "The capital of France is" (5 tokens, no BOS). Stingray: `Qwen4ExpRefDumpTests.GreedyContinuation_PrintsTokensForComparison` with `STINGRAY_Q4E_DUMP_POS=5`. Both produce " Paris." (IDs 11751, 13). Relative difference per layer at decode position 5: layer 0 1.11%, layer 2 0.84%, layer 10 1.44%, layer 20 2.36%, layer 28 5.98%, layer 32 9.95%, layer 36 9.03%, layer 40 9.47%, layer 44 8.51%, layer 46 7.61%, layer 47 151.9%.
  Comparison with the image path (decode step 1, same metric): image layer 0 0.36%, layer 20 1.98%, layer 28 5.62%, layer 32 11.9%, layer 36 17.6%, layer 40 23.9%, layer 44 28.1%, layer 46 32.4%. So early layers are similar, and the image path adds roughly 2.5 to 3.5 times the late-layer drift (layers 36 to 46). Layer 47 reads about 150% on text as well, so it is a dump-definition mismatch (the reference tensor is not the same quantity as Stingray's dump), not an image effect. The remaining question is whether the extra late-layer drift is image-specific or comes from the longer position (66 tokens against 5). Next step: text prompt of about 66 tokens, same metric. Owner decision: whether the D1 token criterion closes this item with these numbers recorded, or the isolation step above is required.
  Harness notes: `llama-mtmd-cli` in text mode runs its chat template (not a raw prompt), so `llama-completion` was used instead.
  **Length-matched text, done 2026-11-24.** Prompt: four repeats of "The quick brown fox jumps over the lazy dog near the river bank." plus "Then the sun set slowly over the hills and the" = 66 tokens (reference `prompt eval` count 66). Decode at cache position 66 on both sides; both continue " birds flew". (A prompt ending in a full stop gives `<|im_end|>` as the first token, so no decode step; that was avoided.) Relative difference per layer: layer 0 0.46%, layer 2 0.57%, layer 10 1.38%, layer 20 4.98%, layer 28 4.76%, layer 32 4.85%, layer 40 6.34%, layer 46 4.44%, layer 47 219.8%.
  Reading: at matched cache length the image path drifts about 2.5 to 7 times more than text in layers 20 to 46 (image 12 to 32%, text 4 to 6%), while layers 0 to 10 are similar. A confound remains: the image's decode uses M-RoPE position 24 with cache 66, and a text prompt cannot separate RoPE position from cache length. The image-specific share is therefore not isolated, only shown to be larger than text drift. Layer 47 is 150% to 220% on every run, so it is a dump-definition mismatch, not an image effect.
  Layer 47 reading (code, 2026-11-24): `src/models/qwen4exp.cpp` names `res_hc` as `l_last` after each layer's HC combine, and Stingray dumps `_resHc` after the same combine (`Qwen4ExpForwardPass.cs:353`), so the dump point matches for every layer. The mismatch on layer 47 is therefore not the point of the dump. Unconfirmed hypothesis: a later node reuses the `l_last-47` name in the reference graph (the MTP / next-n head reads the residual before the final mixer, `qwen4exp.cpp:501`). Not checked.
- **MTP product path (recommendation, not started):** the sequential MTP path gives no speed gain on CPU (plan item 'Separate speed measurement', 2.35 to 2.52 tok/s plain against 2.47 to 2.50 with MTP), and the batched verify it needs is a replay form with no gain. Wiring the product `MtpDecoder` to the four-stream residual is shared-code work with no CPU benefit. Proposed owner decision: waive it for Phase 8 on CPU, and keep it open for GPU.
- **Image and video server paths (open): the only missing check is the HTTP smoke run with the real checkpoint.** Both code paths are in and the engine tests pass on the real checkpoint. The server cannot load the checkpoint here: host preflight refuses (see the RUNNING.md item in Phase 8). Owner decision needed.
- **Speculative acceptance, product path (open, code).** Chained acceptance and rollback through the product decoder need the MTP decoder wired to the qwen4exp pass (`HasMtpHead`, `MtpForward`, `MtpLastHidden`, `PrefillMtp`, `MtpTruncateTo`, `MtpResetCache`, a per-position hidden history). This is a substantial code task, not started.
- **Phase 8 (verification and admission):** needs the items above closed or explicitly waived by the owner, and the owner decides the RUNNING.md and WHAT-YOU-CAN-DO row.
- **Owner decisions needed:** (a) allow the real-checkpoint server smoke run on this host (preflight override or a different budget), or accept it as a recorded gap; (b) waive or fund the logit-level investigation and the MTP product wiring for Phase 8; (c) the RUNNING.md / WHAT-YOU-CAN-DO row.

## Phase 0. Numerical parity (real checkpoint, paged)


## Phase 1. Per-sequence state: broader real-weight verification

Runnable now. Test cases only, no new disk or implementation.


## Phase 2. Batching


## Phase 3. Prefix reuse and retained state


## Phase 4. MTP speculative decoding

- [ ] Acceptance rate, multi-token acceptance, rollback on rejection. **Lossless with rollback, 2026-11-08** (`Qwen4ExpSpeculativeLosslessTests`, real trunk and head, test-side loop, no product change): greedy speculation with a chain of K=3 head drafts, verified by the trunk, and restored from `Qwen4ExpSequenceState.Snapshot`/`Restore` on rejection, gives exactly the plain greedy tokens for 24 generated tokens (4 rollbacks; 17 of 30 chained drafts accepted). The product MTP decoder is not wired to the qwen4exp pass (it assumes one hidden stream; qwen4exp drafts from the four-stream residual), so the product path is still open. **Acceptance measured 2026-11-08** (`Qwen4ExpMtpAcceptanceTests`, real trunk and shared Q4_K_M head): 31 of 40 greedy positions (**77.5%**), one prompt (21 prompt tokens, 40 decode steps). The head's draft for the next token equals the trunk's own greedy token. Chance is about zero for a 248k vocabulary, so the head is doing real work. Still open: multi-token (chained) acceptance, rollback on rejection (needs the MTP decoder wired to the qwen4exp pass: `HasMtpHead`, `MtpForward`, `MtpLastHidden`, `PrefillMtp`, `MtpTruncateTo`, `MtpResetCache`, and a per-position hidden history), and more prompts.

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
      than a new image break. Superseded: Phase 0.3 is closed under D1 (see the Phase 0 item above).
- [ ] Image input through the tool and server paths. **Server path not wired (checked 2026-11-09):** with `STINGRAY_MMPROJ` the
      server refuses qwen4exp ("only supported for architectures that declare image input (currently Gemma 4, gemma4uv)";
      `InferenceEngineLoader`). The server's image pipeline (`InferenceEngine.GenerateImageChunksAsync`, `ImageContent`) is built for
      Gemma 4. Closing this means a qwen4exp branch in that pipeline (projector embeddings through `ForwardImageChunk` with M-RoPE
      positions), the vision markers in the prompt, and `SupportsImageInput` on the descriptor. The local image path is covered by
      `Qwen4ExpImageAnswerTests`; the tool path with an image is not tested.
      **Design, decided 2026-11-09 (code not started):** (1) `InferenceEngine` holds the vision embedder as an interface, not
      `GemmaUvVisionEmbedder`; the Gemma embedder implements it unchanged. (2) A Qwen embedder wraps `QwenVlVisionEncoder` and
      returns the soft tokens plus the grid per image. (3) `SplicePrefillImages` registers each image with `AddMRopeImage` (as
      `RunCommand.Image` does for the CPU pass) and feeds the tokens with `ForwardImageChunk`. (4) `InferenceEngineLoader` accepts
      `MmprojPath` when the descriptor declares `SupportsImageInput`, and the descriptor for qwen4exp sets it. (5) The prompt uses
      the Qwen vision markers. Tests: Server.Fast with a fake engine for the loader and markers; Vision for the engine path on the
      real mmproj, compared with `Qwen4ExpImageAnswerTests`. The server cannot load the real checkpoint on this host (preflight),
      so the real-model proof is through the test harness, and the server smoke run is a recorded gap.
      **Code landed 2026-11-09 (not yet verified on the real checkpoint):** `SpliceQwenImages` in `InferenceEngine` is a separate
      branch, chosen when `EnableQwenImageInput` ran. It encodes each image with `QwenVlVisionEncoder`, preprocesses with
      `QwenImagePreprocessor` (32-aligned, at most 256 soft tokens, [-1, 1]), and writes each image in one
      `ForwardImageChunk` on `Qwen4ExpForwardPass.DefaultSequence` with M-RoPE positions. Text resumes at start + max(nx, ny).
      The loader accepts an mmproj for `ImageProjectorKind.QwenVl` (set on qwen4exp); the public `SupportsImageInput` stays false.
      Tests: `QwenImagePreprocessorTests` (5, Vision, synthetic); Vision full 218 total, 0 failed, 54 skipped (models absent);
      Server.Fast 692 total, 0 failed. The server prompt still renders one `<|image_pad|>` per image, which the splice expands.
      Prompt markers, 2026-11-09: the server writes the generic `<|image|>` marker; `InferenceEngine.GenerateCore` rewrites it to
      `<|vision_start|><|image_pad|><|vision_end|>` when a Qwen projector is attached, so the server is unchanged.
      Real checkpoint, engine level (`Qwen4ExpEngineImageTests`, Vision, synthetic 224 px image with a red and a blue square on
      white, greedy, 96 tokens, 46 s): the answer is "The colours in this picture are red and blue." This is the engine's own
      splice path, not the direct pass test.
      **Still open:** the server smoke run through HTTP (blocked by preflight on this host), and a comparison of the engine
      answer's token ids with `Qwen4ExpImageAnswerTests` (the text answers match in content; ids are not compared yet).
      Earlier note (the design the code followed): the Gemma splice loop gives each image token one sequential
      position, which is wrong for M-RoPE. The engine calls `_fwd.ResetCache()` and
      then splices, so the default sequence is the right target.
- [ ] 5.3 Video input. The owner reopened it on 2026-11-08 (start once the other work is done or blocked).
      **Engine path done 2026-11-09 (server path open):** `InferenceEngine.GenerateVideoChunksAsync` takes `QwenVideoInput`
      (pre-decoded CHW frames, side, fps; the host decodes, the engine never does). The prompt carries one `<|video_pad|>`; prefill
      expands `VideoPromptLayout.Segments`: text segments as tokens, each frame pair as `ForwardVideo` plus one
      `ForwardImageChunk` with M-RoPE, between the vision markers. `Qwen4ExpEngineVideoTests` (real checkpoint, ffmpeg-made
      red-then-blue clip, 4 fps, 224 px, greedy 96 tokens, 148 s): "The colors red and blue appear in the video, with red being
      the first color shown and blue being the second." This is the same text as the direct pass test and the reference
      (`Qwen4ExpVideoAnswerTests`). **Still open:** the server path. Decode helper written 2026-11-09 (`VideoContent`: base64
      MP4 data URL, the operator's ffmpeg via STINGRAY_FFMPEG or PATH, 224 px, 4 fps, at most 32 frames, one video per request).
      Wired 2026-11-09: the OpenAI endpoint's `video_url` content part (an extension; OpenAI has no standard video part) writes
      `<|video_pad|>` and decodes at request time; `videos` is threading through the builders and both handlers; the media
      dispatch (`ImageContent.GenerateMedia`) sends a video request to `GenerateVideoChunksAsync` (added to `IInferenceEngine`
      with a default that throws). `VideoContentEndpointTests` (3, Server.Fast, fake engine): a non-MP4 or bad-base64 payload,
      and a garbage MP4 (ffmpeg absent or not), each give HTTP 400 before any engine call. The Anthropic endpoints do not take
      video. **Still open:** a real MP4 through HTTP to the qwen4exp engine (needs the preflight override on this host), and the
      Anthropic side if wanted. Still to add: a Server.Fast test with a fake engine that records a successful video call (needs
      an MP4 and ffmpeg on the test host, so it will skip without them).
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
      additive at the output). At that point (c), (d), (e) and the frame-input path were still open; see below.
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
      (`scoop\apps\ffmpeg\current\bin\ffmpeg.exe`) is missing, so nothing decodes on this machine. Not used: an Audacity
      FFmpeg installer found in Downloads (an installer, not a standalone binary). Needed from the owner: a working
      ffmpeg (install, or set STINGRAY_FFMPEG to a binary).
      Step (c), done 2026-11-08 (no code change; pinned by `MRopeVideoPairTests`, 3 passing, ForwardPass.Fast, no weights).
      Reference (`tools/mtmd/mtmd.cpp`, `mtmd_image_tokens_get_decoder_pos` and `mtmd_image_tokens_get_n_pos`): a pair is one
      image-token chunk of nx x ny tokens. Every token has t = pos_0, x and y follow the grid, and the text position advances
      by max(nx, ny) once per chunk. `MRopeImageLayout` already does this for a chunk registered with `AddMRopeImage`, and the
      caller must register one chunk per pair, not one per frame. Tests: a pair shares t; text after a 3x2 chunk advances by 3;
      two pairs each advance once.

      Step (d), done 2026-11-08 (layout only; pinned by `VideoPromptLayoutTests`, 5 passing, Vision, no weights).
      Reference order (`tools/mtmd/mtmd-helper.cpp` `read_next`, ~797-845; `tools/mtmd/mtmd.cpp` `expand_lazy_bitmaps`,
      ~1208-1240): "Video:", then each frame. After a frame that reaches the next threshold (elapsed = frame index / fps;
      thresholds 0, 5000, 10000 ... ms) comes a text "[%dm%.2fs]", before the next frame. Chunking
      (`mtmd_group_mergeable_bitmaps`, ~1108-1125) pairs two frames only when they are adjacent parts, so a text part between
      two frames leaves each of them alone. Each chunk is wrapped in `<|vision_start|>` ... `<|vision_end|>` (`add_media`; Qwen
      `img_beg` / `img_end`, `mtmd.cpp` ~701), and each text part is passed to `add_text` on its own.
      From that rule: with the default 5000 ms, frame 0 is a single chunk (the first timestamp follows it) and the rest pair as
      (1,2), (3,4), ... The 8-frame test clip gives 5 chunks [0] [1,2] [3,4] [5,6] [7], not 4 pairs. So
      `VideoTemporalPairs.Pair` (consecutive pairs, no text) is not the reference layout once timestamps are in the prompt.
      Stingray: `VideoPromptLayout.Parts`, `Chunks` and `Timestamp` (src/OpenTail.Stingray.Vision/VideoPromptLayout.cs).
      A single-frame chunk runs both temporal kernels on that frame (`tools/mtmd/models/qwen2vl.cpp`, `n_batch == 1`), which is
      what `ForwardVideo(f, f)` does, so it matches the still path.

      Step (e), done 2026-11-08: the answer matches the reference. Clip: 2 s, 224x224, 4 fps, red for 1 s then blue
      (`ffmpeg` lavfi, 8 frames). Question as in `Qwen4ExpVideoAnswerTests`.
      Reference: `llama-mtmd-cli --video` from `build-ref` (pinned build), `-fit off -c 4096 -ngl 0 --temp 0 --repeat-penalty 1.0 -n 40`,
      default fps and timestamp interval; 56 s wall. Stingray: `Qwen4ExpVideoAnswerTests`, greedy, 40 new tokens; 206 s wall.
      Both answers, character for character: "The colors red and blue appear in the video, with red being the first color
      shown and blue being the second." (each after the same empty think block). The reference layout log shows 11 entries
      and 5 image chunks, which matches `VideoPromptLayout.Segments` for this clip. Limit: the CLI prints text, not token ids,
      so the comparison is on text; one clip and one question only. The reference's `-fit` default aborts on this checkpoint
      (72.5 GB vs 63 GB RAM), so the reference run uses `-fit off`.
      Frame-input path, done 2026-11-08 (`FrameSequenceVideoSource`: a directory of PNG frames in name order, the rate stated by
      the caller, frames of another size resized bicubic, another rate refused; no decoder). Tests: `FrameSequenceVideoSourceTests`,
      6 tests. The projector-path test runs the frames through `ForwardVideo` with the real mmproj and passes, not skipped.
      Correction found on the way, checked against the metadata (`stingray list-metadata` on the mmproj in `/e/_models`):
      `clip.vision.patch_size` is 16 and `spatial_merge_size` is 2, so 224 px gives 14x14 patches and 7x7 = 49 tokens per
      pair, not 64. The decode test and the encoder's `// 14` comment assumed 14 (the comment is stale; code not changed). Both tests now take the size from
      `QwenVlVisionModel.PatchSize`. The decode test has never run here (it skips without ffmpeg).

## Phase 6. Tool calls


## Phase 7. Serving and product

- [ ] Browser UI (7F): **parked by the owner 2026-11-08.** Leave unchecked; do not start.

## Real-weight qualification (from the original port plan's Phase 6, reconciled)

The original plan's Phase 6 checkboxes were never ticked, and the 2026-11-07 paged run was the first real-weight
execution. Keep these open until each is done on the 62 GB checkpoint:


## Phase 8. Verification and admission

- [ ] Full verification pass (Phases 0-7 closed or explicitly waived by the owner). Partly done 2026-11-09: Core full gate and
      Server.Fast full suite ran on the admission change. The waived items under D4 remain open.
      **Real-checkpoint pass, 2026-11-09:** every Qwen real-checkpoint class ran in the heavy suites (Core 35 classes, Vision 5,
      Sessions 1), with the image and video inputs set. All classes that ran passed with 0 failures. The long-prompt test ran at
      4,182 tokens, past the indexer top-k of 2,048 (2,207 s): passed. The logit-parity test (24 steps: worst cosine 0.996936,
      worst total variation 0.072374) and the MTP-oracle test (44 steps: worst cosine 0.997465) passed on rerun with their inputs.
      The reference-dump tests are developer-only and skipped. Not closed by this pass: the items waived under D4 (the multi-row
      pass, the HTTP runs, the performance pass, docs and the licence).
- [ ] Performance pass and DRY pass (CLAUDE.md rule 7). **DRY pass done 2026-11-09:** (1) Engine: the image and video splices
      share `ForwardQwenChunk` (M-RoPE positions, one `ForwardImageChunk`, the text advance) and `CountTokens`; the Gemma splice
      uses `CountTokens`. Regression: Vision full 225 total, 0 failed; the image and video real-checkpoint answers unchanged.
      (2) Core tests: `Qwen4ExpTestSupport` holds the checkpoint file name, `Argmax` and `Top5Contains`; 8 and 2 private copies
      removed; 23 files use the shared filename. Core gate 1439 total, 0 failed; the changed classes pass on the real checkpoint
      (batched verify, MTP decoder, acceptance, lossless, turn snapshot, logit parity, MTP oracle, MTP speed, sharded loading,
      single token). (3) Left in place on purpose: the model-open boilerplate in about 33 files (each test attaches a different
      setup, such as the MTP head or the batched pass), and the separate Vision and Sessions test projects' own helpers.
      The performance pass is recorded above (measured; no kept optimisation).
      Single-sample speed on this machine (2026-11-08,
      `MeasureTextThroughput_OnTheRealCheckpoint`, paged UD-IQ1_S, 63 GB RAM): prefill 1.30 tok/s over 63 prompt tokens, decode
      1.69 tok/s over 12 steps. One sample, not a performance pass; the performance pass itself has not been done (programme plan, Phase 0.2 Q5_K
      fix: "Speed of the Qwen path is not measured yet"). The Q5_K numerics change (`MatVecLlamaNumerics`) is a candidate to measure.
      Baseline, 2026-11-09 (`Qwen4ExpMtpSpeedTests`, real trunk and shared head, 24 generated tokens, three repetitions, no code
      changed): plain greedy 2.415, 2.479, 2.550 tok/s (mean 2.48); product MTP decoder 2.471, 2.486, 2.489 tok/s (mean 2.48),
      draft acceptance 0.75 over 12 drafts; the MTP tokens equal the plain tokens. On this CPU MTP gives no speed gain, which
      matches its exclusion from the admitted scope. No optimisation has been attempted yet, so these numbers are a baseline,
      not a result of the performance pass.
- [ ] RUNNING.md row and WHAT-YOU-CAN-DO line: not added (open, owner decision). On this 63.3 GiB host the CLI refuses
      the checkpoint: the preflight estimate is 66.0 GiB (`stingray -m ... -p ... -n 128 --temp 0` stops with "Not loading"
      on 2026-11-09). The override `--ignore-preflight` would load it, but CLAUDE.md rule 15 says not to run a checkpoint
      known to exceed the budget, so it was not used for a documented run. Earlier server runs used
      `STINGRAY_IGNORE_PREFLIGHT=1`. Owner to choose: document the override, require a larger host, or a smaller quant.
      Checked 2026-11-09: the same refusal from PowerShell (the shell is not the cause). Lowering the context does not get
      under the limit: estimate 66.0 GiB at the default, 64.2 GiB at `-c 4096`, 63.3 GiB at `-c 2048`, against 63.3 GiB. The
      estimate counts the paged weights as resident, while the plan records a 9.6 GB free run and sparse PLE touch of 1440
      bytes per token (6.2). So the estimate looks pessimistic for mmap-paged models, but it is a shared guard across families,
      and changing it is an owner decision.
      **Resident working set, measured 2026-11-09** (`Process.WorkingSet64`, real trunk, the long-prompt text, greedy prefill, no
      preflight involved): after load 0.09 GiB; after 1,024 tokens 26.71 GiB; 2,048 tokens 29.08 GiB; 3,072 tokens 30.32 GiB;
      4,096 tokens 31.19 GiB; after 32 decode tokens 31.21 GiB. Private memory 3.07 GiB. The increments fall from about 2.4 GiB
      per 1,024 tokens to under 1 GiB, so the working set is levelling off well below the 66.0 GiB estimate and the 63.3 GiB host.
      Limits of this evidence: one prompt, contexts up to 4,096 tokens, one 4,096-token run; a longer and more varied workload is
      not yet measured. An earlier 512-token run (24.6 GiB peak) is consistent. A defensible qwen4exp estimate would need this
      calibration plus a longer diverse run, and the guard change is still the owner's decision.

---

## Out of scope for this plan

- `deepseek41`, `deepseek32`, `glm-dsa`, `glm5next` work: other families, see the programme plan.
- Vision towers for other families, new ports.
- CUDA graph path for Qwen 3.8: **owner decision 2026-11-09: do not attempt CUDA.** This machine has no NVIDIA GPU, so it
  cannot be verified here. Vulkan graph path stays in scope (verifiable on the iGPU for correctness only, CLAUDE.md rule 13).

## Doc hygiene (stale or conflicting statements to fix when these items are touched)

