# Current work

The active engineering backlog: work that is not yet proven or product-complete. Completed work,
measurements, negative results, and superseded plans live in [done](done).

## The goal that orders this list

**Run any GGUF from Hugging Face.** Breadth of models is the objective; throughput is subordinate
to it. A model that runs slowly is a worse outcome than a fast one, but a model that cannot be
loaded at all is not an outcome. Where a performance item and a coverage item compete, coverage
wins — that is the change in priority as of 2026-08-08, and it is why the ordering below differs
from the previous release-hardening ordering.

Two consequences worth stating plainly, because they cut against the previous roadmap:

- CPU/CUDA kernel performance work is now the **last** local priority, not the third.
- DSpark speculative decoding and SafeTensors Phases 4-6 are **parked**, not scheduled. Both are
  implemented far enough to be useful and neither moves the goal.

## Cross-project priority order (popularity-adjusted, 2026-09-01)

The sections below ("Ordered runway", Priorities 1-4) are scoped to the **LLM/GGUF coverage goal**
only. Audio (TTS) and Diffusion/video work is tracked separately (`docs/audio-review-progress.md`,
`docs/diffusion-samples/README.md`) and was never sequenced against the GGUF list. This section is
the actual cross-project execution order — which active thread to work next, across LLM, diffusion,
and TTS — adopted 2026-09-01 after an explicit priority review that layered real Hugging Face
download-volume data onto the previous purely-engineering-leverage ordering. Popularity moved LTX
and Z-Image-Vulkan up sharply; it did not change the Unigram-tokenizer item's technical leverage,
only its relative rank against demand-driven items. Full reasoning: the review argued the project's
strongest near-term external-interest lever is "video + local CPU/GPU independence" (Wan/LTX/Z-Image
popularity dwarfs the individual GGUF architecture gaps), so those move ahead of infrastructure work
that is more elegant but lower-visibility.

**P0 — in this order:**
1. **Wan 2.1 / 2.2 video** — still open. Five real bugs fixed, a ~6x CPU perf improvement landed,
   and a real 256x256/12-step run (2026-09-01) disproved the "just needs more steps" theory — the
   grid artifact persists past the point where VAE-resolution effects could explain it. Every
   specific hypothesis checked so far is individually ruled out (see full handoff in
   `docs/diffusion-samples/README.md`); the real bug is still unlocated. Next real step is a
   numeric (not structural) diff against a real PyTorch/diffusers reference run, not further
   structural re-verification of already-checked pieces.
2. **Z-Image Vulkan backend fix** — **DONE, 2026-09-01.** `QwenTextEncoder`'s GPU path
   unconditionally used BF16 regardless of backend support; `VulkanBackend.Sgemm` had no BF16
   fallback, silently misreading the 2-byte BF16 buffers as 4-byte FP32 and corrupting Qwen3's text
   embeddings to NaN before the DiT ever saw them (found via a `MatQ`-level NaN/Inf trace — the
   corruption appeared at the very first op that consumed real text features). Fixed by gating the
   BF16 path on `BestSgemmPrecision==Bf16`, matching the pattern `ZImageDiT` already used correctly.
   Verified: real, recognizable apple image on the Vulkan GPU path (`zimage-vulkan-fixed.png`).
3. **CosyVoice 2/3 speaker-identity transfer** — **re-scoped, 2026-09-01**: the handoff doc
   describing "cond is all-zero" and "CFG is missing" was stale — both were already fixed by an
   earlier commit (`4cb189f`). Re-verified the real mel extractor (`CosyVoiceMelExtractor.cs`)
   line-for-line against `cosyvoice-frontend.cpp`'s `extract_speech_feat`/`build_mel_basis`, and
   ran a real end-to-end generation with `--ref-audio` — every stage produces real, non-degenerate
   data (real prompt tokens, real mel frames, real speaker embedding, real CFG). **What's left is
   purely a listening judgment call, not implementation**: `docs/audio-samples/
   cosyvoice3-identity-check.wav` needs a human ear to confirm whether the cloned voice actually
   sounds like the reference speaker. See `docs/qwentts-cosyvoice3-handoff.md` §2 for the full
   re-verification and exactly what to check next if it still doesn't sound right.

**P1:**
4. **LTX-Video — make it genuinely real** (a focused follow-on campaign, only after the P0 items
   close). This is the single biggest re-rank from the popularity review: previously scoped low
   because it's real implementation work, not debugging (transformer weights are never applied, text
   conditioning is literal random noise), but the LTX model family's download volume is large enough
   to outweigh that. **Full implementation plan written 2026-09-02**, see
   [055-ltx-video-implementation-plan.md](055-ltx-video-implementation-plan.md) — real tensor
   inventory of the local checkpoint confirms the architecture is a 28-layer/2048-hidden PixArt-
   style DiT (NOT Wan-shaped: patch_size=1 so no real spatial patching happens in the transformer,
   cross-attention operates at 2048-dim on a pre-projected caption sequence rather than raw T5's
   4096-dim, FFN is ordinary 2-layer GELU not gated), plus a dependency-ordered build/verification
   plan and the real gotchas (VAE decoder is itself timestep-conditioned; RoPE uses continuous
   pixel-space coordinates, not integer latent indices). Do not start this while Wan/Z-Image/CosyVoice are still active — it would break
   the consolidation discipline the rest of this document argues for.
5. **SentencePiece Unigram-LM tokenizer.** Best pure infrastructure ROI on the list: one
   implementation unlocks six blocked GGUF architectures at once (`minicpm`, `internlm2`, `ernie4_5`,
   `baichuan`, `orion`, `nanbeige` — see the done-archive's per-architecture "CHECKED and BLOCKED"
   entries), and any future Unigram-tokenized checkpoint besides. Lower individual download numbers
   than the P0 items, but it's recurring leverage, not a one-off unlock.
6. **CPU greedy-decode non-determinism investigation** — low visibility, high correctness stakes;
   run in parallel with whichever other item is active whenever idle capacity allows, per the
   existing note in the "Standing state" archive (2 non-reproducing sightings under CPU contention;
   not yet understood). Do not let this block a demand-driven item above.

**P2:**
7. **HunyuanVideo completion** — DiT already runs clean through every layer against real weights;
   blocked on a real VAE decoder (own class, same shape of work as `WanVaeDecoder3D`) and real dual
   CLIP+LLM text conditioning. Real demand (a popular community Diffusers repack sees tens of
   thousands of downloads/month) but well below Wan/LTX/Z-Image.
8. **`xverse` tokenizer fix** — **root-caused and fixed, 2026-09-02.** The old framing ("absent
   scores") was imprecise: the real defect was architectural, not a missing-data edge case — this
   engine's SPM path (`tokenizer.ggml.model=llama`) used a merges-RANK-TABLE algorithm (built from
   `tokenizer.ggml.merges`) for genuine SentencePiece tokenization, but real SentencePiece BPE
   (confirmed by reading llama.cpp's actual `llm_tokenizer_spm_session::tokenize`) has no merges
   list in the algorithm at all — a candidate merge is valid purely because its concatenated text
   is a vocabulary entry, prioritized by that entry's own `tokenizer.ggml.scores` value (highest
   first, leftmost on a tie). `tokenizer.ggml.merges` is a GGUF export convenience some converters
   also emit, which is why every model shipping both arrays worked by coincidence. Any checkpoint
   without a merges array (`xverse`, but also a real local checkpoint,
   `models/paddleocr-vl-1.6.gguf`, which has scores but no merges — confirmed independently
   broken and now fixed) fragmented to near-character-level.
   Implemented the real algorithm (`GgufTokenizer.SpmMergePiecesByScore`), added `tokenizer.ggml.
   scores` reading, and a 5000-case fuzz-parity suite (`SpmMergeByScoreTests.cs`) against a naive
   O(n²) oracle. Verified against `paddleocr-vl-1.6.gguf`: a 44-char sentence now tokenizes to 10
   sensible tokens with a perfect round-trip decode (previously would have fragmented). Full Core
   suite (642 tests) still green — the old merges-rank algorithm is untouched and still serves the
   byte-level-BPE path, which is a genuinely different, correct use of that mechanism.
   **ARCHITECTURE ADMITTED, same day.** Downloaded `xverse/XVERSE-7B-Chat-GGUF` (Q4_K_M, genuinely
   Apache-2.0), confirmed genuinely `general.architecture=xverse` with neither
   `tokenizer.ggml.merges` nor `tokenizer.ggml.scores` (the exact worst case). Found and fixed a
   SECOND, independent tokenizer bug in the process: this engine never implemented real
   llama.cpp's `add_space_prefix` (default `true` for `tokenizer.ggml.model=llama`, prepends a
   leading space before tokenizing) — the first attempt diverged at token 0 only, with every
   subsequent token already exact, which is what isolated it as a second bug rather than a
   leftover from the first fix. Also had to abandon "retokenize the printed continuation text" as
   an evidence-gathering method for this checkpoint specifically — this vocab has genuine
   tokenizer non-injectivity (multiple valid tokenizations of the same string), so re-encoding the
   model's own printed output reproduced a DIFFERENT sequence (29 tokens) than what the model
   actually sampled (24 tokens); used `llama-server`'s `/completion` endpoint with
   `return_tokens: true` instead, which returns the real, authoritative per-step sampled ids.
   **Result: FULL 24-of-24-token exact greedy match**, zero new architecture code (confirmed
   against `xverse.cpp`: literal plain-Llama trunk). Correction after first writing this up: the
   checkpoint is actually bucket-2, not bucket-1 — the archived note's original "custom Model
   License Agreement" finding was right; the GGUF repo's own HF `cardData.license: apache-2.0`
   only covers the conversion code, not the weights (confirmed via the base model's own README,
   which describes a separate weight license requiring a commercial-use application). Per this
   project's bucket-2 policy, no persisted test — evidence recorded as a comment on the `"xverse"`
   allowlist entry in `ModelCompatibility.cs` instead. `xverse` added to the allowlist; checkpoint
   deleted after the receipt. The tokenizer fixes themselves (`SpmMergePiecesByScore`,
   `AddSpacePrefix`) are permanent and covered by `SpmMergeByScoreTests.cs`'s synthetic fuzz suite
   regardless of xverse's own license.

**P3 and later campaigns (not scheduled, revisit only after the above closes):**
9. **DeepSeek2/MiniCPM3 MLA** — the single biggest architectural lift in the coverage plan
   (5 genuinely new mechanisms); potentially high popularity if ever tackled, but deliberately
   deprioritized behind smaller, faster wins.
10. **Newer LTX families (LTX-2.3/2.5, etc.)** — a later campaign once the base LTX-Video port is
    real; these newer variants individually out-download even the original LTX-Video model.
11. **GPT-OSS** — needs multiple substantial new mechanisms (attention sinks, alternating
    sliding-window attention, biased MoE experts, an OpenAI-specific SwiGLU/gating variant).
    Potentially high popularity, but explicitly a new campaign, not a known/started gap — do not
    chase this opportunistically ahead of the ordered list above just because it's individually
    popular; that violates the consolidation strategy this list is built around. The same
    discipline applies to FLUX.1, SD3/3.5 — strategically attractive, not yet started, not to be
    chased ahead of turn.

---

## Ordered runway

1. [01 — GGUF model coverage](01-gguf-model-coverage-plan.md) — architectures, IQ quant formats,
   tokenizer pre-types, chat templates. **The goal, restated as work.**
2. [02 — Qwen3.5 MoE / Gated DeltaNet](02-qwen35moe-plan.md) — a large, popular GGUF family whose
   hybrid path exists but is not fully evidenced.
3. [03 — Gemma 4 E4B vision](03-gemma4-e4b-vision-plan.md) — multimodal coverage; blocked on a
   usable reference implementation, see the doc.
4. [04 — configuration and operator quality](04-quality-of-life-improvements-plan.md).
5. [05 — CPU architecture kernel coverage](05-cpu-architecture-kernel-opportunities.md) —
   performance only, except its scalar-fallback-format item, which §2 of plan 01 supersedes.

Work needing hardware this machine does not have is in
[90 — external hardware work](90-external-hardware-work.md). It is not part of the local runway.

---

## Also active, separate from the GGUF-coverage goal — session architecture migration

Not ordered by "run any GGUF" — a parallel, session/runtime-layer thread.
[028 — InferenceSession → HotSession migration plan](028-inference-session-to-hotsession-migration-plan.md)
is fully done: Phases 1 (KV memory governance), 2 (cross-session prefix sharing), and 3
(fork/branching) are all implemented and verified against real models, each with its own
`HotSession`-native test.

**Done (2026-08-27)**: [030 — delete InferenceSession/InferenceRuntime](030-delete-inferencesession-todo.md)
has been executed. `InferenceSession`/`InferenceRuntime` and the superseded files/tests around them
(~45 files total, including `KvMemoryGovernor` — Phase 1's predecessor, missed by the original file
list — and `SessionTree`/`BranchVoteResult`/`InferenceSessionConsensusExtensions`) are deleted. All
ten genuinely novel capabilities the two audit passes found are ported onto `HotSession`/the engine
rather than dropped: `ISessionMetadata`/`ISessionMetrics` (`HotSession.Metadata`/`.Metrics`,
`HotSessionMetricsMetadataTests.cs`); `FinishReason`/`ToolCalls` bundling on
`HotSessionTurnResult`; LoRA, tool/skill validation, `OnTokenGenerated`, checkpoint/rollback,
session tree, suspend/resume (`HotSessionCapabilityPortTests.cs`); and `SamplingParams.AllowedChoices`
constrained-choice sampling, which the deletion re-check found was implemented *only* inside
`InferenceSession` — ported into `ContinuousBatchingEngine`'s batched decode loop
(`HotSessionChoiceConstraintTests.cs`) rather than silently lost.

**Done (2026-08-27), follow-up**: [051 — HotSession capability wiring plan](done/051-hotsession-capability-wiring-plan.md)
(now in `docs/done/`; the live `docs/051-hotsession-capability-wiring-plan.md` holds only the
remaining TODOs — LoRA, real new engine work; `OnTokenGenerated`/`ToolCallParser`, deliberately not
wired since the Server layer already does this independently; and `Fork()` skill/instruction
propagation, an open design question). `/v1/sessions/*` gained skills/tool-call validation
(`POST .../skills`, `.../tool-calls/validate`), skill instructions that actually shape the next
turn's prompt (also added to `/v1/chat/completions` and `/v1/messages` via a new `skills` request
field — `SkillWireModels.cs`), previously-invisible `metrics`/`metadata`/`finish_reason`/
`tool_calls`/`allowed_choices` fields, checkpoint/rollback, fork-tree observability, and
suspend/resume. Full solution builds clean; `Tests.Sessions.Fast` (127), `Tests.Server.Fast` (352),
and `Tests.Core` (576) all pass. (`Tests.ForwardPass.Fast` has 7 pre-existing, unrelated
SIMD-tiering bit-equivalence failures on this machine, confirmed present on the pre-deletion
baseline too — environment-specific: `OpenBLAS: not found`.)

**Also found (and fixed) while stress-testing this path**: a severe, unrelated prefill-packing
defect affecting real concurrent `HotSession` traffic at 5–15 simultaneous requests, since
resolved — see the entry below and [031](031-concurrent-decode-batch-tier-divergence-bug.md). Was
never a migration-plan defect and never blocked Phases 1-3 (all three are done/verified
independent of this).

---

## New finding — 4 vision architectures produce degenerate embeddings (2026-09-02)

A full run of `tests/OpenTail.Stingray.Tests.Vision`'s real-weight suite
(`MultimodalRealWeightsTests.cs`, 135 tests, 33 min real-weight run) surfaced 4 genuine failures,
confirmed pre-existing (the test file hasn't changed since before this session, and its call path
— `UnifiedVisionPipeline.Open`/`EmbedImage` — never touches `IComputeBackend`/CUDA/Vulkan at all,
so this is unrelated to any GPU-path work done this session):

- `YoutuVl` and `HunyuanVl`: `EmbedImage` returns a completely all-zero embedding buffer.
  **Re-admitted 2026-09-02 after briefly being dropped — kept, but explicitly lowest priority to
  fix among these 4** (real demand for these two specific architectures is lower than
  `KimiVl`/`MiniCpmV`).
- `KimiVl`: two genuinely different input images produce embeddings whose cosine similarity is
  `NaN` (a degenerate/zero-norm embedding on at least one side).
- `MiniCpmV`: two genuinely different input images produce embeddings with cosine similarity
  `1.0000001` — i.e. identical, meaning the image content isn't actually reaching the encoder
  output for at least one of the two inputs.

131/135 real-weight tests in this suite pass. Not yet root-caused for any of the 4 — no reference
source has been read for these architectures yet this session. Given the failure SHAPE (all-zero or
identical-regardless-of-input) rather than "wrong but non-degenerate" output, the likely culprit
category is a wiring/plumbing bug (an input never reaching the encoder, a buffer never written, an
early-return path) rather than a subtle numerical error — matching the pattern several other real
bugs in this project turned out to be (e.g. Wan's earlier missing-`--vae` mistake, or the
Z-Image/QwenTextEncoder BF16-corruption bug). Real weight test: rerun via
`STINGRAY_RUN_HEAVY_TESTS=1 tests/OpenTail.Stingray.Tests.Vision/bin/Release/net10.0/
OpenTail.Stingray.Tests.Vision.exe -class OpenTail.Stingray.Tests.Vision.MultimodalRealWeightsTests`.

---

## Second new finding — Gemma 3's own text generation is broken on BOTH CPU and Vulkan, unrelated to vision (2026-09-02)

While trying to verify another session's real, unrelated GPU-decode-path change (widening
`ForwardEmbedding`/`SupportsEmbeddingInput` beyond Gemma-4-only in `GpuForwardPass.cs`/
`CudaForwardPass.cs` — that code itself looks like real, careful engineering and compiles clean,
but ships with placeholder-stub tests, `Assert.True(true)` and pure-arithmetic checks that never
call the real code, so it was unverified when found), downloaded a real `gemma-3-4b-it-Q4_K_M.gguf`
+ its real `mmproj-gemma-3-4b-it-f16.gguf` to test end-to-end.

**First, a real, fixed, unrelated bug**: `Gemma3Adapter` (`UnifiedVisionPipeline.cs`) was a literal
copy-paste of `Gemma4VAdapter` with the class renamed but its marker strings never updated —
`PlaceholderMarker`/`ImageOpenMarker` were both `"<|image|>"` (Gemma 4's real marker), which isn't
a real Gemma 3 special token at all, so `--image` on a real Gemma 3 checkpoint always failed with
"expected 1 image placeholder token(s) ... but found 0". Fixed to Gemma 3's real markers
(`<start_of_image>`/`<end_of_image>`, confirmed against `examples/llama.cpp/llama.cpp/tools/mtmd/
mtmd.cpp`'s `PROJECTOR_TYPE_GEMMA3` case).

**With that fixed, a genuinely bigger, pre-existing problem surfaced**: real end-to-end generation
with a real image produces incoherent multi-script gibberish on BOTH backends (different garbage on
each — not even backend-consistent). To isolate whether this was the vision path or something more
fundamental, ran the SAME checkpoint on a PLAIN TEXT prompt ("The capital of France is") with NO
image/mmproj/ForwardEmbedding involved at all:
- **CPU**: stops after 2 tokens, effectively blank output — a real, separate degenerate-generation
  bug.
- **Vulkan**: produces the SAME KIND of incoherent multi-script gibberish
  ("productos teinte expertos...ಬೇಕ...") as the vision run did.

**This means Gemma 3 text generation itself is broken on this engine, on both backends,
independent of vision entirely** — the garbage seen in the vision run was NOT introduced by the
embedding-splice widening being verified; it reproduces with zero vision/embedding code in the
path at all. This is a real, deeper, previously-undocumented correctness gap for `gemma3` as a text
architecture (not just its vision encoder), and it means the embedding-splice widening's own
correctness genuinely cannot be verified until this more fundamental bug is found and fixed first —
there's no working Gemma-3 baseline on either backend to verify against yet. Not root-caused this
pass (no reference source read yet for Gemma 3's specific text-decode mechanics — e.g. its
alternating local/global sliding-window attention pattern, its own norm/scale conventions — the
next real step, following this project's own established methodology, before any further attempt
at either backend).

---

## Priority 1 — model coverage

See [01-gguf-model-coverage-plan.md](01-gguf-model-coverage-plan.md) for the audit and the ordered
work. The three findings that justify its position at the top:

1. **The architecture gate does not run on the CLI inference path.** `ValidateForTextGeneration` is
   called by `doctor`, `static-plan`, and the server loader — not by `RunCommand`. The CLI will
   attempt architectures the server refuses. `docs/cpu-performance-baseline.md` contains a measured
   CPU baseline for OLMoE, an architecture the gate rejects.
2. **The whole IQ quant family is unimplemented.** `DType` declares eight IQ formats;
   `Dequantize.ToFloat32` implements one (`IQ4_NL`). Large models on Hugging Face are distributed
   predominantly as IQ quants, so this excludes more repos than the architecture list does.
3. **DEFECT FOUND AND FIXED (2026-08-08) — Qwen3 tokenized differently from llama.cpp.** Only
   `tekken` had an explicit pre-tokenizer regex; every other byte-BPE model silently got GPT-2's,
   whatever `tokenizer.ggml.pre` declared. Measured on Qwen3-0.6B against `llama-tokenize` b8585,
   three of five probes diverged: `IT'S`, `(hello)`, `a  b`. Nothing errored —
   `Decode(Encode(s)) == s` still holds when the split is wrong, which is why the existing suites
   never caught it. Fixed by `PreTokenizerPatterns`, a ported pre-type → regex-cascade table; all
   parity rows now pass. Digits look like the obvious discriminator but are not — see the plan.

`CLAUDE.md` overstates architecture support — it lists `deepseek2` and OLMoE, neither of which the
gate admits. Correct it in the same change that resolves the gate.

## Priority 2 — model families already part-built

1. **Qwen3.5 MoE / GDN.** Ornith-1.0 9B exercises the hybrid Gated-DeltaNet path end to end, which
   covers items 1 and 2 of that plan in practice. What remains is GDN state-lifecycle conformance
   coverage and a benchmark once correctness is settled. Use
   [reference/qwen35moe-tensor-layout.md](reference/qwen35moe-tensor-layout.md) as the authoritative layout, never the
   superseded SSM plan.
2. **Gemma 4 E4B vision.** **Updated 2026-08-15** — the `gemma4v` mmproj load boundary, fixed-grid
   preprocessing, ViT encoder forward pass, token-reduction pool, and projector are now all
   implemented (`Gemma4VVisionEncoder.cs`), architecture fully reverse-engineered against the real
   mmproj + local llama.cpp source (not guessed from tensor names — see
   [03-gemma4-e4b-vision-plan.md](03-gemma4-e4b-vision-plan.md)'s current implementation contract).
   Passes a real-mmproj structural sanity test but is **not numerically parity-verified**: the
   local llama.cpp build still rejects the paired `gemma4` text GGUF, so no oracle exists yet for
   end-to-end comparison — that is a parity blocker, not an implementation blocker, and does not
   block the remaining work. Still open: embedding splice into the text decoder, decoder-side
   image-token mask semantics (explicitly NOT assumed to match Gemma 3 — needs its own
   investigation before touching `PagedKvCache`), and CLI/API surface.
   Note the 12B is a different, working path (encoder-free `gemma4uv`) and was never blocked by
   any of this.
3. **Gemma 3 vision (SigLIP).** **New, 2026-08-15** — a separate, simpler ViT family
   (`clip.projector_type=gemma3`, `Gemma3VisionModel.cs`/`Gemma3VisionEncoder.cs`) paired with the
   Gemma 3 4B text model rather than E4B. Both `gemma3` and `gemma4` text architectures are already
   admitted, so once the splice/mask work above lands (shared infrastructure, not
   architecture-specific) this path could reach genuine end-to-end multimodal inference sooner
   than E4B, which needs the same work regardless. Loader and encoder implemented and verified
   against the real `models/mmproj-gemma-3-4b-it-f16.gguf`; structural sanity test passes
   (1/1, 604.9s — attention parallelized across heads since it's ~21x gemma4v's compute at 4096
   patches). Two real, non-obvious findings from this checkpoint's export, documented in
   [03-gemma4-e4b-vision-plan.md](03-gemma4-e4b-vision-plan.md)'s addendum: a different metadata
   key convention (`clip.projector_type`, not `clip.vision.projector_type`) and a NAME-vs-FUNCTION
   swap on the `ffn_up`/`ffn_down` tensors (proven via bias-length evidence, not a storage
   transpose). Same caveat as `gemma4v`: not numerically parity-verified, no oracle available.
4. **Llama 4 vision (E4B Scout/Maverick).** **New, 2026-08-15** — a third ViT family
   (`clip.projector_type=llama4`, `Llama4VisionModel.cs`/`Llama4VisionEncoder.cs`), the only one of
   four researched candidates (`llama4`/`qwen2vl`/`qwen3vl`/`glm4v`) whose paired text decoder
   (`llama4`) is already admitted AND needs no new engine-wide RoPE machinery — the other three all
   require genuine multi-axis M-RoPE, which doesn't exist anywhere in this engine yet (see
   [06-llama4-vision-plan.md](06-llama4-vision-plan.md)'s Context section for the full comparison).
   Structural sanity test passes against the real
   `models/mmproj-llama-4-scout-17b-16e-instruct-f16.gguf` (1/1, 132.1s — one 336x336 tile, 34
   blocks, 577 tokens including a real [CLS] token this checkpoint has that neither `gemma4v` nor
   `gemma3` do). Real findings this time: a flat F16 (not 4D F32) patch-embed tensor, no FFN gate
   tensor at all (plain, not gated, FFN), real pre- AND post-layernorm both present, and — the one
   that would have been a silent wrong-answer bug if missed — a NORM/interleaved 2D-RoPE pairing
   convention, genuinely different from `gemma4v`'s NEOX split-half convention, confirmed only by
   reading `clip.cpp`'s shared `build_rope_2d` helper directly rather than assuming the existing
   `ApplyRope2DHalf` helper would transfer. Multi-tile ("llava-uhd") preprocessing and decoder
   splice are both explicitly out of scope, same precedent as the other two encoders — this
   processes one fixed-square tile per call. llama.cpp's own code separately flags this exact
   projector as known to have degraded quality (ggml-org/llama.cpp#13282), independent of whether
   the port itself is correct. Same caveat as the other two: not numerically parity-verified.

## Priority 3 — operator quality

[04-quality-of-life-improvements-plan.md](04-quality-of-life-improvements-plan.md). Configuration
ownership Phase 0 deliverable 1 (both inventories) is now **DONE, 2026-08-15**:

- **Done — inventories regenerated from source.** `docs/cli-option-inventory.md` went from 96
  hand-maintained rows to all **149** (then, as of the 2026-08-15 pass below, **153**) declared
  options, produced by the checked-in `scripts/gen-cli-option-inventory.ps1` (`-Check` fails when
  stale). The count guard now also asserts the ROW count: the declared count had tracked source
  correctly the whole time while the table silently fell 53 rows behind — it was measuring the one
  thing not drifting.
- **Done — stale registry entries retired.** Three `KnownEnvironmentVariables` entries were never
  environment variables: `STINGRAY_ARGMAX_NEG_INF` (a CUDA `#define` in an NVRTC kernel string) and
  the glob patterns `STINGRAY_MOE_` / `STINGRAY_SNAPKV`. Registry 159 → 156. A new test requires
  every entry to appear as a **quoted string literal** in `src/`.
- **Done, 2026-08-15 — Class classification complete for every row of both inventories.** The
  registry had drifted again since 2026-08-08 (grew back to **162** names across later sessions'
  architecture/kernel work; the doc's own reconciliation prose said 156, which was stale) and the
  table had separately drifted from the registry in both directions (5 rows missing, the same 3
  ghost names from the paragraph above still sitting in the table despite being removed from
  source) — re-diffed to zero drift, then every one of the 162 env-var rows and 153 CLI-option rows
  given an explicit Class, not just the ~half that had one via a summary "ownership register" that
  was never propagated into the actual per-row table. Also found and fixed a real
  `gen-cli-option-inventory.ps1` bug: its description parser only matched single-line
  `[Description("...")]`, so multi-line concatenated descriptions (`"..." + "..."`) silently
  produced blank cells — 3 `RunCommand` rows affected, now populated. Full detail in
  [04-quality-of-life-improvements-plan.md](04-quality-of-life-improvements-plan.md) item 1.
- **Still open:** extending source-tracked effective configuration beyond static planning knobs
  (item 2 of the plan), and removing the obsolete/dead-looking switches classification surfaced —
  that needs a per-variable owner call, not a drive-by deletion, so it wasn't done as part of the
  classification pass itself.

## Priority 4 — performance

[05-cpu-architecture-kernel-opportunities.md](05-cpu-architecture-kernel-opportunities.md). All of
it is performance-only; none of it unlocks a model. Its item 3 (native IQ4_NL/MXFP4 kernels) is now
downstream of plan 01 §2 — correctness first, kernels after.

Do not reopen the closed Q4_K repacked-GEMM investigation. Every performance item requires dispatch
proof, interleaved control/candidate samples, named-model end-to-end measurement, and numerical
validation. No single-run result is sufficient.

**Q6_K baseline (2026-08-07).** The checksum-guarded `kernel-bench-cs` harness at `k=8192`,
`rows=512`, `reps=12`, with `DOTNET_TC_QuickJitForLoops=0`, produced independent best times of
0.1676, 0.1760, and 0.1845 ms (checksum `2363.599609`). This replaces the stale 0.2063 ms figure but
is not a new performance claim: a candidate must be interleaved against this implementation in the
same process and beat the observed run-to-run range.

---

## Standing state, not plans

Closed/historical entries (architecture-admission receipts, fixed defects, license findings, test
suite stats, release status, the parked-work list) have moved to
[done/00-current-work-standing-state-2026-08.md](done/00-current-work-standing-state-2026-08.md)
(archived 2026-08-31, per the Archive rule below — nothing there is open work).

## Archive rule

Move a document or section to [done](done) when its outcome is implemented and verified, or when a
measured negative result closes that line of investigation. Add a banner saying what closed and what
carried forward. Keep active documents short: decision, remaining work, acceptance evidence, links.
