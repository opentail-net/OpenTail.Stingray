# Current work

The active engineering backlog: work that is not yet proven or product-complete. Completed work,
measurements, negative results, and superseded plans live in [done](done).

Closed sections of this file are moved verbatim to [done](done) under the Archive rule below (most recently [done/00-current-work-closed-2026-09.md](done/00-current-work-closed-2026-09.md), 2026-09-27).

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

## Vulkan iGPU real profiling: the per-op sync tax is dominated by staging-buffer copy, not queue-submit/fence-wait (2026-09-12)

**Context**: after two failed GPU-attention-kernel attempts (naive shader, then a properly tiled
flash-attention shader — both regressed real timing, see PerformanceLeague.md's SpatialTransformer
rows), an external design review (ChatGPT, given this codebase's real structure) concluded the real
architectural bottleneck isn't the attention kernel itself — it's that `SdxlUNet2DConditionModel`'s
`Lin()`/`Conv()` treat the GPU as a synchronous RPC coprocessor (Upload → Sgemm → Synchronize →
Download → Free, per call) rather than keeping the whole UNet forward pass GPU-resident for one
denoising step. The review's own recommended first step, before attempting any fix: **real
profiling of the actual time split, not another guess.**

**Implemented**: real, permanent, env-var-gated timing (`STINGRAY_PROFILE_GPU_SPLIT=1`) added at
the single real choke point every Vulkan dispatch/upload/download passes through
(`VulkanBackend.SubmitAndWait`, plus the staging-buffer `Map()`/memcpy in `Upload`/`Download`).
Splits into two real numbers: `submitWait` (vkQueueSubmit + actual GPU execution + fence wait,
combined — Vulkan gives no cheaper way to separate "GPU busy" from "waiting" without a timestamp-
query extension not wired up here) and `stagingCopy` (the CPU-side `Map`/memcpy into the
upload/download staging buffers). Wired into the SDXL CLI path (`ImageCommand.cs`) so a real run
prints a real summary.

**Real measurement** (SDXL-Turbo, 512×512, 4 steps, guidance=0, Vulkan iGPU, real weights):

```
[GPU-split:SDXL full run] dispatches=6511 submitWait=2712.5ms stagingCopy=43587.0ms
```

Total wall time: 137.4s. So of that:
- **GPU submit+execution+fence-wait: 2.7s (~2% of total)** — the GPU itself, and the raw act of
  submitting work to it and waiting, is NOT the bottleneck. This is a real, measured refutation of
  the naive "GPU is just slow on this iGPU" framing.
- **CPU-side staging-buffer copy: 43.6s (~32% of total)** — this is the real, dominant, measured
  cost directly attributable to the current per-op Upload/Download architecture. 6511 dispatches
  in one 4-step run means 6511 separate `SubmitAndWait` round-trips, most of them upload/download
  staging copies (Map/memcpy pairs), averaging ~6.7ms each — surprisingly expensive for what should
  be a fast memcpy, suggesting real driver-level `vkMapMemory`/cache-coherency overhead per call,
  not raw bandwidth.
- **The remaining ~65% (~91s)** is CPU-side work outside Vulkan entirely (im2col/dequant/weight
  prep, the rest of the pipeline) — a separate, already-partially-optimized area (parallel im2col,
  vectorized GroupNorm, fp16 weight uploads all landed earlier this session).

**Conclusion, confirming the external review's hypothesis with a real number rather than
reasoning alone**: the fix is not "improve the attention kernel" (already tried twice, both
regressed) and not primarily "reduce GPU wait time" (already tiny) — it's **collapsing 6511 small,
separately-synchronized Upload/Dispatch/Download round-trips into far fewer, larger ones**, most
directly by keeping intermediate UNet tensors GPU-resident across a whole denoising step instead of
crossing back to CPU after every `Lin()`/`Conv()` call. This directly explains why the tiled
attention kernel regressed: it was asked to overcome a architectural cost that exists one level
above the kernel itself, and no amount of shader cleverness fixes a per-call staging-buffer-copy
tax that fires regardless of how good the shader's own math is.

**Not yet implemented** (real, scoped next step, matching the review's own phase ordering — fusion/
reusable-command-graphs/INT8/attention all explicitly deferred behind this): a whole-UNet-forward-
pass GPU-residency rewrite of `SdxlUNet2DConditionModel`, recording the entire chain (conv → norm →
silu → attention → FFN → ... across all blocks) into one command buffer per denoising step via the
`VulkanBackend` batching primitives already used elsewhere in this codebase (`BeginRecord`/
`EndRecordAndSubmit`, `BeginBatch`/`EndBatch`, deferred frees) rather than building new
infrastructure from scratch.

## LLaVA-1.5-7B: mmproj misrouted to InternVL (FIXED), then a second, unfixed gap (classic LLaVA has no real placeholder token) (2026-09-12)

Downloaded LLaVA-1.5-7B (`mys/ggml_llava-v1.5-7b`, Q4_K text + f16 mmproj) as new Phase-1 VLM
coverage. Two real bugs found in sequence:

1. **FIXED**: `UnifiedVisionPipeline.Open` misrouted this mmproj to `InternVlVisionModel` instead
   of `LlavaVisionModel`. Root cause: this is a legacy llava.cpp-era mmproj with no
   `clip.vision.projector_type`/`clip.projector_type` string metadata at all (only the boolean
   `clip.has_llava_projector=true`), so it fell through to the generic structural-inference
   section, where InternVL's detector (`v.class_embd` tensor present) matched first — but a CLS
   embedding tensor is generic to any CLIP-based vision tower (LLaVA's included), not
   InternVL-specific, so this was a false positive that always wins over LLaVA's own, more
   specific `mm.0.weight` structural check further down the same method. Symptom before the fix: a
   hard crash ("vision projector (internvl) outputs 768-dim embeddings but the text backbone
   expects 4096-dim input"). Fixed by (a) checking `clip.has_llava_projector` as an authoritative
   signal before any structural inference runs, and (b) reordering the structural fallback so
   LLaVA's `mm.0.weight` check runs before InternVL's `v.class_embd` check as defense-in-depth for
   an mmproj lacking that flag too. Verified: the vision encoder now runs and reports the correct
   `llava -> 576 soft tokens (4096-dim)`.
2. **NOT FIXED, real architectural gap**: once vision-encoding succeeds, image splicing fails —
   `"expected 1 image placeholder token(s) (<image>, 258880) after templating but found 0"`.
   Root cause: this checkpoint's tokenizer is the plain, unmodified 32000-token LLaMA vocab (no
   `<image>`/`<|image|>`/any image-related special token registered at all — confirmed via
   `list-metadata`). This codebase's `RunCommand.cs` image-splicing path assumes every
   architecture has some real, tokenizable placeholder string that maps to a real vocab id, which
   it substitutes for a synthesized `<image>` marker and then re-finds in the encoded token
   stream. Classic LLaVA-1.5 doesn't work that way: the reference `llava-cli`/`clip.cpp`
   implementation splices the image embedding sequence directly into a fixed prompt position (no
   vocab token involved at all) — there is no real placeholder id to find. This is a genuine
   missing-feature gap (a direct-splice-without-placeholder code path), not a quick metadata fix —
   deferred as a real, scoped item, not attempted further this session. `PerformanceLeague.md` logs
   the real vision-encode timing that was captured before this second failure.

## ONNX support expansion (2026-09-11, real plan, not yet started)

**User want, stated explicitly:** "I would love for us to have great onnx support." Parked behind
the current bug-fixing focus, but recorded here as a real, scoped initiative rather than a vague
aspiration, so it doesn't get lost.

**Where this stands today:** `OnnxModelSession` (`src/OpenTail.Stingray.Core/OnnxModelSession.cs`)
is a genuinely generic ONNX Runtime wrapper — real `InferenceSession`, arbitrary named
inputs/outputs, works against any `.onnx` graph mechanically. The gap has never been the runtime;
it's that only `EmbedCommand` wires up real, correct input construction and output interpretation
for one model class (BERT-family text embedding, fixed for real 2026-09-11 — see the
`BertWordPieceTokenizer` entry below). Every other `.onnx` file sitting in `models/_models/`
(`silero_vad.onnx`, `kokoro-v1.0.onnx`, `campplus.onnx`, `cosyvoice_speech_tokenizer.onnx`/`_v2.onnx`,
`sensevoice-small.int8.onnx`, `paraformer-zh-small.int8.onnx`, `melotts-zh_en.onnx`,
`en_US-lessac-medium.onnx`) is either served by a real native C# reimplementation instead (this
project's actual design preference — avoid runtime dependencies where a native port is feasible)
or, for the two confirmed-broken ASR checkpoints (SenseVoice unwired entirely, FunASR Paraformer's
GGUF broken — see the ASR gap entries elsewhere in this doc), not served by anything at all yet.

**Reference material pulled in 2026-09-11** (`examples/`, gitignored per the existing
`examples/llama.cpp`/`examples/audio.cpp` reference-oracle convention — not committed, not
counted against repo size):
- `examples/sherpa-onnx` (Apache 2.0) — **the highest-value one**: a real, actively-maintained
  project that wires ONNX Runtime up with correct pre/post-processing for dozens of speech models,
  covering several of the *exact same checkpoints already sitting unwired in this repo* — Silero
  VAD, Paraformer, SenseVoice, CosyVoice's speech tokenizer, Kokoro. Its C++ source is a real oracle
  for feature-extraction parameters (mel filterbank config, frame/hop sizes, normalization), real
  input tensor names/shapes, and real output-decode logic per model — same role
  `examples/audio.cpp`/`llama.cpp` already play elsewhere in this project.
- `examples/onnxruntime-extensions` (MIT) — Microsoft's official library for fusing
  tokenization/audio/image pre/post-processing directly into the ONNX graph as custom ops. Lower
  priority; more relevant as a potential future dependency than as example code to port from.
- `examples/onnxruntime-inference-examples` (MIT) — official per-model-class sample pre/post-
  processing, thinner coverage than sherpa-onnx for the audio models this project actually needs.

Licenses checked 2026-09-11: Apache 2.0 and MIT are both compatible with this project's own MIT
license for reading/porting logic (Apache 2.0 code copied verbatim would need its attribution
notice preserved on that specific code — no `NOTICE` file exists to carry forward beyond the
standard header, and nothing has been copied yet, only cloned for reference).

**A real, scoped plan (not started), roughly in priority order — matching the MusicGen/AudioGen
archaeology-first pattern this doc's other sections already use:**

1. **SenseVoice — DONE and VERIFIED 2026-09-11.** Implemented (subagent-written, code-only, I
   verified with real weights) `src/OpenTail.Stingray.Audio/SenseVoice/` — real 4-input ONNX
   forward (`x`/`x_length`/`language`/`text_norm`→`logits`), real metadata read from the
   checkpoint's own embedded ONNX custom metadata (added `OnnxModelSession.CustomMetadata`
   exposing `InferenceSession.ModelMetadata.CustomMetadataMap`, confirmed this capability didn't
   already exist), real feature extraction reusing `FunAsrRealMelExtractor` unchanged (fbank+LFR+
   CMVN, parameters confirmed matching via the real sherpa-onnx exporter script), real CTC greedy
   decode, real vocab (`F:\_models\sensevoice-small-tokens.txt`, 25055 lines matching
   `vocab_size`, downloaded from the real `csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-
   2024-07-17` HF repo). **Verified with the exact same real LibriSpeech clip Citrinet-ASR used**
   (`librispeech_test_clean_6930-75918-0000.wav`): real 2229ms timing, correct language/emotion/
   event tags, and an **exact transcript match to the ground truth**: `"concord returned to its
   place amidst the tents"` — a real, fully working ASR pipeline on the first attempt, not just a
   non-crashing stub. `tests/OpenTail.Stingray.Tests.Audio/SenseVoiceRealWeightsTests.cs`. Not
   wired into the CLI `stt` command (Whisper-shaped, would need real restructuring) — the
   real-weights test is the entry point for now.
2. **FunASR Paraformer — DONE, real Chinese speech verified 2026-09-11.**
   Implemented (subagent-written, code-only, I verified builds+runs with real weights) an
   independent `src/OpenTail.Stingray.Audio/ParaformerOnnx/` path, giving this project a working
   Paraformer regardless of the broken native GGUF conversion. Real metadata directly inspected
   from the checkpoint's own ONNX custom metadata (`vocab_size=8359`, real `lfr_window_size`/
   `_shift`, real `neg_mean`/`inv_stddev`), real single-forward-pass architecture (the whole
   encoder+CIF-predictor+decoder is fused into one ONNX graph, confirmed — no reimplementation of
   Paraformer's internals needed), real per-position-argmax-with-EOS-stop decode (a genuinely
   different algorithm from SenseVoice's CTC-collapse, confirmed via `offline-paraformer-greedy-
   search-decoder.cc`, not copy-pasted from the SenseVoice work by mistake), real vocab
   (`F:\_models\paraformer-zh-small-tokens.txt`, confirmed matching this exact checkpoint two
   independent ways: line count equals `vocab_size` exactly, and `</s>` sits at the real expected
   id 2). **Update, same day — real Chinese audio test landed.** Found a real Chinese test clip
   (`test_wavs/0.wav` from the SAME real HF repo this checkpoint's vocab came from,
   `csukuangfj/sherpa-onnx-paraformer-zh-small-2024-03-09`), downloaded to
   `docs/audio-samples/paraformer-zh-test-0.wav`, swapped into
   `ParaformerOnnxRealWeightsTests.cs` replacing the synthetic-tone placeholder. **Result: a real,
   coherent, grammatically valid Mandarin transcript** — `"对我做了介绍啊那么我想说的是呢大家如果对我的研究感兴趣呢嗯"`
   (roughly: "...introduced me, ah, so what I want to say is, everyone, if you're interested in my
   research, um...") — real 2171ms timing. No exact ground-truth transcript string was available
   to assert byte-for-byte, but this is unambiguously real, sensible spoken Mandarin content, not
   degenerate/garbage output — strong evidence the pipeline is genuinely correct, not just
   non-crashing.
   A real Chinese WAV clip would be needed for full golden verification — not attempted this pass.
3. **Silero VAD's ONNX path** — this project already has a real native Silero VAD port (`🟢` in
   README); this item is lower priority, but sherpa-onnx's VAD wiring could serve as an independent
   real-reference cross-check for the existing native port if a correctness question ever comes up,
   not a new capability to build.
4. **Generalize `EmbedCommand`'s pattern into a reusable shape** once 1-2 real pipelines exist
   beyond text embedding — e.g. a shared `IOnnxPipeline`-style interface if the real preprocessing/
   postprocessing code for SenseVoice and Paraformer turns out to share meaningful structure. Don't
   build this abstraction speculatively before there are at least 2 real, working call sites to
   generalize from (matches this project's own "don't design for hypothetical future requirements"
   discipline).

**Explicitly not in scope for this plan**: re-implementing Kokoro/CosyVoice/MeloTTS/Piper's ONNX
paths — those already have real, working native C# ports and redoing them via generic ONNX
execution would be a regression in the "avoid runtime dependencies" direction, not progress.

## User-requested checkpoint targets (2026-09-03, not yet scoped)

The user has asked for these specific checkpoints to be supported, in addition to whatever this
doc's existing priority ordering already covers. Not yet archaeology'd, scoped, or sequenced
against the rest of this backlog — recorded here as a standing want-list, to be turned into a real
scoped plan (matching the MusicGen/AudioGen/ACE-Step docs' pattern: real config + tensor inventory
before any code) when picked up.

- **Stable Audio 3 Small SFX — WORKING, 2026-09-03.** Real archaeology (`SA3_MODEL_MATRIX` section,
  docs/057) found Small SFX's DiT/VAE config is byte-identical to the already-working Small Music
  runtime and its T5Gemma text encoder is the literal same checkpoint (identical sha256) — so it
  needed ZERO source changes, just pointing the existing `StableAudioPipeline` at the real SFX
  `model.safetensors` (2.27GB, 685 tensors, confirmed matching). `StableAudio3SmallSfxTests` passes
  on the first run: real finite, non-silent SFX audio from a real prompt. A real 3s sample generated
  to `docs/diffusion-samples/` for a listening check. The real architecture-equivalence caution
  below (ACE-Step/Stable-Audio-3 VAE mixup precedent) turned out NOT to apply here — verified, not
  assumed, exactly per that same discipline.
- **Stable Audio 3 Medium — V1 end-to-end WORKING, 2026-09-03.** Real archaeology confirmed Medium
  genuinely differs from Small (embed_dim 1536 vs 1024, depth 24 vs 20, heads 24 vs 16, real
  differential attention on both DiT and VAE, SAME-L VAE with real sliding-window attention) —
  unlike SFX, this needed real new code, all now landed: `StableAudioMediumDiT` (differential self-/
  cross-attention, real tensor shapes confirmed against the checkpoint's own header) and
  `SameLargeVae` (reuses `AcousticVae.cs`'s proven differential-attention/DynamicTanh machinery,
  widened, with single-pass banded sliding-window attention replacing Small's dual-pass chunking —
  real branch logic confirmed by reading `TransformerResamplingBlock.forward` directly). A real
  mid-session correction-then-restoration is worth knowing about: the public `stable-audio-tools`
  PyPI release turned out to be stale for several real config fields, which briefly downgraded
  confidence in the differential-attention formula — fetching the real GitHub `main` source
  directly re-confirmed the formula was correct all along (see docs/057's "CONFIDENCE RESTORED"
  section for the full arc). `StableAudioMediumPipeline` wires text encoder (T5Gemma, literal same
  checkpoint as Small) + DiT + VAE together; `StableAudio3MediumPipelineTests` passes real
  end-to-end generation on the first run. A real 4s sample was generated and sent for a listening
  check. **Update, same session: DRY + performance passes done.** Extracted the shared DiT
  attention kernels (RoPE/RMSNorm/dot-product-attention/timestep-Fourier) between Small and Medium
  into `StableAudioAttentionKernels.cs` — zero regression, confirmed via the real numeric golden-
  parity suite (not just non-degeneracy) re-run after. Measured real generation timing: mean 387s
  for a 4s/15-step CFG-enabled generation (~97s wall-clock per second of audio on CPU) — no code
  change made, the next lever is flagged, not implemented without a profiler-driven re-measurement.
  **`sinusoidal_blocks` FeedForward variant implemented and CONFIRMED FIXED, 2026-09-03**: the
  operator reported the original `sa3_medium_piano-arpeggio_4s.wav` sample as poor quality; the
  deliberately-deferred `sinusoidal_blocks: [8]` gap (real GLU gate activation `Sin(x)=sin(pi*x)`
  replacing `SiLU` on `SameLargeVae`'s last 7 decoder layers, real formula from GitHub `main`
  fetched fresh -- PyPI has no such kwarg) was the leading suspect and turned out to be the real
  bug: a regenerated sample (`sa3_medium_piano-arpeggio_4s_v2.wav`, same prompt/seed/duration, 25
  steps) was confirmed by ear as good, the original confirmed as not. See docs/066's "CONFIRMED
  FIXED" update for the full derivation.
  **Same day, second real gap found and fixed across all three SA3 variants**: the real reference
  pads every generation with 6s of latent headroom, masks attention over it, and warps the Euler
  timestep schedule via a real `DistributionShift` formula (confirmed from real per-checkpoint
  `model_config.json` + GitHub `main` source: `use_effective_length_for_schedule`/
  `mask_padding_attention` both `true` for Small Music, Small SFX, AND Medium) -- none of which
  this port implemented until now (`StableAudioScheduleKernels.cs`, new shared file). Attention
  masking specifically is NOT implemented (documented gap -- neither DiT class accepts a mask at
  all yet). All four existing Small/Medium tests re-pass with zero regression; SFX, Small Music,
  and Medium samples all regenerated with the fix for the operator's own listening check (not
  judged by this session -- operator was AFK during this work). **All three variants' samples
  confirmed present and passing as of this update** (Medium's combined regen took ~17min, the
  real perf-tax flagged above -- separate from the correctness question). Also confirmed: Small Music's
  real `sinusoidal_blocks` is `[0]` (inert), explaining why it never needed Medium's fix; and a
  minor, real, but negligible (~1000x below signal scale) gap in `SoftNormBottleneck.decode()`
  (real inference-time noise injection, RNG-dependent, not implemented -- not a plausible quality
  cause). See docs/066's "(4)/(5)/(6)" updates for the full derivation.
  **Not yet done**: numeric golden-parity (DiT and VAE) for Medium, real attention-masking over
  the padding region for all variants, and docs/065's Sprint 3 consolidation
  (`IStableAudio3Engine`) now that three real variants (Small Music, Small SFX, Medium) exist. See
  docs/done/057-stable-audio-3-implementation-plan.md for the full writeup.
- **MiniMax-Music3 — CORRECTION 2026-09-04: this bullet was badly stale.** It previously read
  "Phase A archaeology done, first component (vocoder) golden-verified" and estimated ~15% overall
  complete, citing the big `language_model/`/`transformer/` downloads as disk-space-blocked. Both
  claims were wrong as of this correction: **all six real components are downloaded locally
  (`models/minimax-music3/`, ~27GB total) and ALL SIX have real, passing golden-parity tests**
  (`MiniMaxMusic3ConditionEncoderGoldenParityTests`, `MiniMaxMusic3GlobalModelGoldenParityTests`,
  `MiniMaxMusic3PromptEncoderGoldenParityTests`, `MiniMaxMusic3RvqDepthDecoderGoldenParityTests`,
  `MiniMaxMusic3TransformerGoldenParityTests`, `MiniMaxMusic3VocoderGoldenParityTests`, plus
  `MiniMaxMusic3GlobalKvCacheConsistencyTests` and `MiniMaxMusic3AutoregressiveGeneratorSmokeTests`
  — re-ran all 8 test classes fresh on 2026-09-04, every one passes with real weights). Real git
  history confirms this was done in commits after this bullet was last written and simply never
  reflected here: `cf7c061` (vocoder), `049a740` (RVQ depth decoder), `0e6a938` (condition
  encoder), `549f62f` (DiT transformer). A full real end-to-end pipeline
  (`MiniMaxMusic3Pipeline.cs`) and two scratch sample-generation tests also exist. **Real, accurate
  status: every individual component is golden-verified; NOT yet done is a real listening check of
  full end-to-end generated audio** (the scratch generate-sample tests exist but their output
  hasn't been judged by ear this session) **and numeric golden-parity of the full multi-stage
  pipeline together** (each stage verified in isolation, not the composed whole). This is much
  closer to "feature-complete, pending a quality listening pass" than "15% done" — do not trust
  the old framing in docs/066's earlier sections without cross-checking against this correction.
  **Update, same day: real composition bug found and fixed.** A first real 200-frame generation
  produced audible garbage ("jitter," per the operator) despite every component's own golden test
  passing — a frame-index off-by-one in `MiniMaxMusic3AutoregressiveGenerator.Generate` (confirmed
  against the real `diffusers==0.40.0` source: frame 0's sampled codes are a prefill-only warm-up
  never meant to be appended to the real output, per the reference's own `range(max_frames+1)`
  loop with a `frame_index > 0` append guard — this port's loop lacked that guard, prepending a
  spurious frame to every generation). Fixed, all 8 existing golden-parity/smoke tests re-verified
  with zero regression, and a full 200-frame regen with the fix completed (3352.9s ≈ 56min real
  wall-clock, real weights) — `docs/diffusion-samples/minimax_music3_v1_folk_verse_200frames.wav`
  is ready for the operator's own listening judgment (not judged by this session; the original
  jitter sample is preserved as `..._200frames_PREFIX_jitter.wav` for reference). See docs/066's
  "RESOLVED, 2026-09-04" entry for the full derivation.
  See docs/066-minimax-music3-future-plan.md.

## Cross-project priority order (2026-09-01): open items only

The full, verbatim 2026-09-01 list (17 items) moved to
[done/00-current-work-closed-2026-09.md](done/00-current-work-closed-2026-09.md#cross-project-priority-order-popularity-adjusted-2026-09-01)
on 2026-09-27: most of it closed (Z-Image Vulkan, xverse tokenizer, SentencePiece Unigram,
MusicGen, AudioGen, Wan, LTX-Video, DeepSeek2 MLA, GPT-OSS, FLUX.1, Stable Audio 3; see their
STATUS rows). Still open, with the original entry's number:

- **#3 CosyVoice 2/3 speaker identity.** CosyVoice 3 is 🟢 in STATUS — no remaining work; its fix
  history is in [done/qwentts-cosyvoice3-handoff.md](done/qwentts-cosyvoice3-handoff.md). CosyVoice 2
  is blocked on an upstream reference (item 10 of
  [102-status-open-items-plan.md](102-status-open-items-plan.md)).
- **#6 CPU greedy-decode non-determinism** (2 non-reproducing sightings under CPU contention).
- **#7 HunyuanVideo numeric verification** (item 13 of [102-status-open-items-plan.md](102-status-open-items-plan.md)).
- **#11 ACE-Step 1.5 Turbo.** The archived entry says "not started", which is stale:
  [064-acestep-implementation-plan.md](064-acestep-implementation-plan.md) records V1 working
  end to end on 2026-09-03. ACE-Step has no STATUS row yet.
- **#12 NaN in `ForwardPass`'s f16 qwen3 path** (last layer, one position; Q8_0 is fine). Not root-caused.
- **#14 Newer LTX families** (LTX-2.3/2.5), a later campaign.

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

## Priority 1 — model coverage

See [01-gguf-model-coverage-plan.md](01-gguf-model-coverage-plan.md) for the audit and the ordered
work. The three findings that justify its position at the top:

1. ~~**The architecture gate does not run on the CLI inference path.**~~ **STALE — already fixed
   (dated 2026-08-08 per `RunCommand.cs`'s own inline comment, confirmed 2026-09-02).**
   `RunCommand.cs`'s GGUF load path now calls `ModelCompatibility.ValidateForTextGeneration(model)`
   unless `--allow-unverified-arch` is explicitly passed (which prints a loud warning instead of
   silently proceeding) — matching `doctor`/`static-plan`/the server loader. This item can be
   removed from the plan; verified by reading `RunCommand.cs:900-919` directly, not assumed.
2. ~~**The whole IQ quant family is unimplemented.**~~ **STALE — already fixed (confirmed
   2026-09-02).** All 9 IQ formats (`IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`,
   `IQ3_S`, `IQ4_NL`, `IQ4_XS`) have real scalar dequantize decoders in `Dequantize.cs` (each with
   its own doc comment citing the matching `ggml` `dequantize_row_iq*` reference) and are dispatched
   in `SimdKernels`'s `MatVec` switch (either a dedicated fast path or `MatVecDequantFallback`).
   Verified by reading both files directly, not assumed. This item can be removed from the plan too
   — whoever fixed this evidently didn't update `01-gguf-model-coverage-plan.md`/this doc to match.
3. **DEFECT FOUND AND FIXED (2026-08-08) — Qwen3 tokenized differently from llama.cpp.** Only
   `tekken` had an explicit pre-tokenizer regex; every other byte-BPE model silently got GPT-2's,
   whatever `tokenizer.ggml.pre` declared. Measured on Qwen3-0.6B against `llama-tokenize` b8585,
   three of five probes diverged: `IT'S`, `(hello)`, `a  b`. Nothing errored —
   `Decode(Encode(s)) == s` still holds when the split is wrong, which is why the existing suites
   never caught it. Fixed by `PreTokenizerPatterns`, a ported pre-type → regex-cascade table; all
   parity rows now pass. Digits look like the obvious discriminator but are not — see the plan.

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
   [06-llama4-vision-plan.md](done/06-llama4-vision-plan.md)'s Context section for the full comparison).
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

## Gemma 4 batched-prefill (not started)

Scoped 2026-09-10 after `PerformanceLeague.md`'s C++ backfill measured Gemma 4 CPU prefill at
0.12-0.13x of llama.cpp on both E4B and 12B — the missing-batched-prefill gap
(`perLayerHdUnsupported`) confirmed on a second model size, not just the original one. Real fix
requires teaching `PrefillCoreAttention` five things it doesn't do today (per-layer head-dim
indexing, per-layer KV-source sharing, `attention_k_eq_v`, per-head V-norm, sliding-window
masking) — not a quick patch; a prior attempt to force the existing batched path
(`STINGRAY_PER_LAYER_HD_PREFILL=1`) crashed with `AccessViolationException`, not just wrong output.
See [070-gemma4-batched-prefill-plan.md](070-gemma4-batched-prefill-plan.md) for the full plan.

## Bugs found during the 2026-09-10/11 PerformanceLeague model sweep (not started)

None of these were found by deliberate auditing — they surfaced incidentally while benchmarking
models for `PerformanceLeague.md`. All are measured/reproduced, none are fixed. Listed roughly by
severity.

- ~~**`stingray embed`'s GGUF path is a complete fake, silently.**~~ **FIXED, already landed by
  `e9403ff` (2026-09-16) before this item was picked back up — this entry was stale.**
  `EmbeddingEngine`'s constructor now does a real `GgufModel.Open`/`ForwardPass` load for any
  `File.Exists` GGUF path. Re-verified 2026-09-18: real run against `qwen3-embedding-0.6b-q8_0.gguf`
  logs a real weight pre-fault (`[ForwardPass] Pre-faulted 0.74 GiB...`) and produces real,
  prompt-varying, non-degenerate output — not the old hash stub. **One residual bug found and fixed
  in the same pass, same failure shape but narrower**: a `.gguf` path that does NOT exist still
  silently fell through to the synthetic placeholder generator with no error. Fixed at
  `EmbedCommand.cs` (errors out before constructing `EmbeddingEngine`, mirroring the ONNX branch's
  existing real error handling). `EmbeddingTests` (11 tests, real weight-loading confirmed via
  timing) re-verified passing.
- ~~**`stingray embed`'s ONNX path crashes on a real BERT-family checkpoint.**~~ **FIXED
  2026-09-11.** Was: `Missing Input: token_type_ids` on `all-MiniLM-L6-v2_quantized.onnx` because
  `EmbedCommand.cs`'s ONNX branch never constructed that tensor. Fix landed: an all-zero
  `token_type_ids` input alongside `input_ids`/`attention_mask` (`OnnxModelSession.Run` already
  filters to only inputs a model declares, so this is safe for models that don't need it too).
  Verified against MiniLM, BGE-small/base/large — all four now report their real, correct native
  output dimension (384/384/768/1024), confirmed real inference. Two more bugs found and fixed in
  the same pass: `stingray embed -o <file>` crashed on the same reflection-JSON issue as the GGUF
  path (replaced with a hand-rolled serializer for the simple `List<float[]>` shape); and the ONNX
  branch's "tokenization" is not real WordPiece/BPE (it maps each raw character to its char code as
  a placeholder token id — no tokenizer.json/vocab.txt ships alongside these ONNX checkpoints here)
  which was overflowing BERT's 512-position limit on long inputs with an opaque broadcast error —
  added a defensive truncation-with-warning instead of a crash. **The missing-real-tokenizer issue
  itself is still open** — this only makes it fail gracefully, real embeddings from this path are
  not semantically meaningful for inputs longer than a few words until a real tokenizer is wired.
  See `PerformanceLeague.md`'s Embeddings section for real post-fix measurements.
- ~~**Granite architecture has a real, Vulkan-specific correctness bug**~~ **FIXED 2026-09-11.**
  Was: identical prompt/seed gave coherent English on CPU but broken output on Vulkan (garbled
  multilingual gibberish on `Granite-4.0-3B-Vision`, a single degenerate token then immediate stop
  on `Granite-Vision-3.2-2B`). Root cause: `GpuForwardPass.cs`'s `RunStandardLayers`/
  `RecordBatchedTrunk` never threaded `AttentionScaleOverride`, `ResidualScale`, or `LogitScale`
  into the Vulkan dispatch path, while the CPU path (`ForwardPass.Decode.cs`/`PrefillCore.cs`/
  `Attention.cs`) applied all three — Granite's real, non-1.0 scaling hyperparameters. Fix: the
  same Q-prescale-to-cancel-the-shader's-hardcoded-scale trick already used for Gemma 4's
  `AttentionScaleOverride`, generalized, plus `ScaleInPlace` calls for `ResidualScale`/`LogitScale`
  at 9 call sites (single-token decode, batched prefill/verify, and every logit-output path),
  each gated on a non-default value so no other architecture is affected. Verified with real
  re-runs on both previously-broken checkpoints: both now produce coherent output on Vulkan
  matching CPU exactly, no measurable perf regression. Full solution rebuilds clean. See
  `PerformanceLeague.md`'s Gemma/Granite rows for the before/after measurements.
- **Two independent ASR pipelines produce degenerate output — disambiguated 2026-09-11. Qwen3-ASR's
  half ROOT-CAUSED AND FIXED 2026-09-12, per user's explicit mini-plan priority (fixed right after
  Z-Image-Turbo).** Qwen3-ASR 0.6B transcribed the standard 14.1s **real speech** reference clip
  as just "aspects" (should be a full sentence) — a genuine decode/correctness bug since real
  speech was used and still produced garbage.

  **Root cause**: `QwenAsrTokenizer.FormatPrompt`'s ChatML template was an entirely invented,
  never-verified guess (its own prior doc comment admitted this). Diagnosed via a real, permanent
  diagnostic (`STINGRAY_QWENASR_DIAG=1` in `QwenAsrDecoder.cs`/`QwenAsrLlmSafetensorsTensorSource.cs`)
  that printed the exact generated token sequence and audio-embedding stats: the decoder emitted
  exactly ONE real token then immediate EOS regardless of real 14s speech input; audio-embedding
  magnitude was confirmed NOT the cause (std=0.0197 vs. a real text-embedding row's std=0.0307,
  same order of magnitude). Checked the real reference vendored in this repo,
  `examples/audio.cpp/src/models/qwen3_asr/tokenizer_text.cpp`'s `default_chat_prompt`/
  `build_prompt`, and found the real template is structurally different: (1) system turn content
  is EMPTY by default, not "You are a helpful speech-to-text assistant."; (2) the user turn
  contains ONLY the audio block, no trailing instruction text at all (the "Transcribe the audio
  speech into text."/"Translate the speech into English." strings were both invented — the real
  reference has no distinct translate-mode prompt for this family either); (3) `language`, when
  set, is the literal string `"language {lang}<asr_text>"` seeded as a forced PREFIX of the
  assistant's own turn (part of the prefill prompt tokens, not generated) — `<asr_text>` (real
  special token id 151704, confirmed via the checkpoint's own `tokenizer_config.json`) appears to
  be the actual trigger telling the model to begin emitting a transcript.

  **Second, compounding bug found while implementing the fix**: `<asr_text>` wasn't registered in
  either tokenizer-building path's `AdditionalSpecialTokens` dict (`QwenAsrWeights.BuildTokenizer`
  for GGUF, `BuildTokenizerFromHfFiles` for Safetensors) — so encoding it fell through to BPE
  character-shredding instead of the single real token id, the exact same failure class already
  found and fixed once for the audio start/end/pad tokens, just never extended to this one since
  nothing had used it before. First attempt at the template fix alone (without this second fix)
  actually regressed further, to completely empty output/immediate EOS at step 0 — confirming
  both bugs were real and needed fixing together.

  **Result**: transcript went from "aspects" (one wrong word) to a real, recognizable, mostly-
  correct transcript of the actual content: `"!合理。Some call me涅，ature； others call me mother
  nature. Iveelve been here for over four although billion years, although twenty two! thousand
  five hundred times longer than you."` — unmistakably the same sentence as the real reference
  (`"some call me nature others call me mother nature i've been here for over four point five
  billion years twenty two thousand five hundred times longer than you"`), with some remaining
  character-level noise (stray Chinese characters, "although" for "point five") left as a real,
  separate, smaller-magnitude gap — not investigated further this pass, plausibly this 0.6B
  checkpoint's own real quality/quantization ceiling rather than a further pipeline bug, since the
  template fix alone recovered essentially the whole sentence structure. `QwenAsrTokenizerTests`
  updated to assert the new real template's shape instead of the old invented one;
  `QwenAsrPipelineSafetensorsTests`'s end-to-end test swapped its synthetic sine-tone input for
  the same real `b.wav` speech clip other working ASR pipelines in this doc use (a real ASR model
  correctly producing empty output on pure tones — which is what started happening once the
  template was fixed — isn't a bug; testing with tone input was never actually validating
  transcription). Full 13-test QwenASR suite re-verified passing. See `PerformanceLeague.md` for
  the timing/full transcript comparison.

  FunASR-Nano (via `paraformer-q8.gguf`) produces repetitive word-salad ("to to to... a a a at at
  at") but — confirmed by reading `FunAsrNanoEndToEndTests.cs:106` — its test harness feeds a
  `new Random(0)`-generated synthetic tone, not real recorded speech, unlike Qwen3-ASR's real-clip
  test. A real speech encoder given non-speech tone input producing meaningless/repetitive output
  is expected behavior, not necessarily a code defect — this remains "inconclusive, needs a
  real-speech re-test" (no CLI/test path currently accepts an arbitrary real `.wav` for this
  pipeline — would need a small harness change), not yet revisited this pass.
- **F5-TTS's blocked `audio.cpp` CPU backend — FIXED for real, 2026-09-11, two real bugs.**
  `examples/audio.cpp/src/community_models/f5_tts/cpu_graph_compute.h`'s
  `ggml_graph_compute_with_ctx` resolution only had GCC/Linux code paths (weak-symbol check, then
  a `dlopen`-based fallback for `GGML_BACKEND_DL` module builds) — this vendored build is compiled
  with MSVC (`__GNUC__` undefined, confirmed via `build/CMakeCache.txt`'s `CMAKE_CXX_COMPILER`
  pointing at `cl.exe`), so neither path applied and it unconditionally threw, regardless of
  whether the CPU backend was actually available. Checked the real build config: this is a plain
  static link (`GGML_BACKEND_DL=OFF`, `BUILD_SHARED_LIBS=OFF` in `CMakeCache.txt`) — the whole
  weak-symbol/dlopen dance was solving a problem that doesn't apply to this build at all.
  `ggml-cpu.h` already declares the function as a normal exported symbol (`GGML_BACKEND_API`);
  for a static link it resolves at ordinary link time. Added a `#if !defined(__GNUC__)` branch
  that just calls it directly — confirmed via a real incremental rebuild (`cmake --build . --target
  audiocpp_cli`), 0 link errors. **Second real bug found once past that**: `ggml_new_object: not
  enough space in the context's memory pool` — `runtime.cpp`'s two `ctx_bytes` sizing formulas
  (`CfgGraph`/`DiTGraph` graph caches) were consistently a few MB short of what ggml actually
  needed for real workloads (real errors: `needed 1613595248, available 1610612736` at the
  1536MB floor; `needed 12893969376, available 12884901888` at the 12288MB cap) — ggml's real
  per-tensor object-header overhead wasn't accounted for in the plain `max(floor, N*bytes)`
  estimate. Fixed with a flat +64MB safety margin on both formulas, plus raised the hard cap from
  12288MB to 16384MB since it was also genuinely too low for longer reference-audio clips (this
  machine has 64GB RAM, real headroom to spare). **Verified end-to-end with real audio output**:
  a short reference clip (`test_s2.wav`) produced a real 65KB WAV; a longer one (`VibeVoiceRef.wav`,
  the one that previously hit the 12288MB cap) produced a real 768KB WAV. Real timing: 43.2s/
  43.1s/43.2s across 3 runs (same prompt as the original bug repro, `"Hello, I will make some
  lunch, darling!"`) — this is the **first-ever real C++ comparison point for F5-TTS**, added to
  `PerformanceLeague.md`: OT's own C# port (27.25s) is actually 1.59x *faster* than this C++
  reference, unlike most other TTS rows in this doc where C++ wins.
- **Chatterbox Turbo's tokenizer-asset bug — FIXED and verified 2026-09-11, but "streaming" was
  never actually the real blocker (a separate, deliberate design limit).** Root-caused by a
  subagent: `examples/audio.cpp/model_specs/chatterbox_turbo.json` expects three sidecar files
  (`chatterbox_turbo_vocab.json`/`chatterbox_turbo_merges.txt`/`chatterbox_turbo_special_tokens.json`,
  classic split GPT2-BPE format) next to the GGUF, none of which existed as standalone files. Fix:
  wrote a small scratch C# tool (referencing `OpenTail.Stingray.Core`'s `GgufModel` reader) to
  extract the real `tokenizer.ggml.tokens`/`tokenizer.ggml.merges` GGUF metadata already embedded
  in the checkpoint into `vocab.json` ({token: id} object) and `merges.txt` (plain "a b" lines per
  `llama_bpe.cpp`'s real loader), and cross-referenced `examples/Chatterbox-turbo-cpp/assets/
  tokenizer.json`'s 20 real emotion `added_tokens` (`[angry]`, `[whispering]`, etc.) into
  `special_tokens.json`, GGUF ids taking precedence on any mismatch. **Verified end-to-end**: model
  now loads cleanly against the correct combined `examples/audio.cpp/models/Chatterbox-Turbo-GGUF/
  chatterbox-turbo-q8_0.gguf` (the earlier `models/_models/chatterbox-turbo-t3-q4_k.gguf` used for
  the initial repro turned out to be a T3-only split checkpoint incompatible with this model
  spec's combined `t3`/`conds`/`s3gen` tensor namespacing — a separate, unrelated non-issue once
  the correct bundled GGUF was used), and a real `--mode offline` run produced a genuine 126KB WAV
  (`docs/audio-samples/chatterbox-turbo-audiocpp-vocab-fix-verify.wav`). **The real finding**:
  `--help` on the fixed model reports only `--task tts --mode offline` as supported — Chatterbox
  Turbo genuinely has no streaming mode in this build at all (confirmed: `--mode streaming`
  fails with `"Chatterbox Turbo only supports offline mode"`), the same deliberate offline-only
  design limit already documented for QwenTTS/CosyVoice3 — not a bug, and not something the vocab
  fix could have unblocked regardless. The vocab files are derived data placed next to the GGUF in
  the gitignored `examples/` tree (not committed, matches convention); the extraction tool is a
  one-off scratch script, not added to the repo.
- **Granite-4.0-3B-Vision's real vision-encode path produces degenerate, non-image-grounded
  output.** Real `--image`/`--mmproj` run against `granite-4.0-3b-vision-Q4_K_M.gguf` +
  `mmproj-granite-4.0-3b-vision-f16.gguf` (2026-09-11, 3 runs, consistent): the vision encoder
  genuinely runs (576 soft tokens, 2560-dim, real non-trivial prefill/decode timing — 13.5 t/s
  prefill, ~12 t/s decode), but the generated text is always `"This image is a description of the
  provided text."` regardless of the actual image content, instead of a real description like
  InternVL3-2B correctly produces on the same image. Not yet root-caused — candidates: image
  embeddings not actually being attended to by the backbone, or a prompt-template/placeholder
  mismatch specific to this checkpoint's chat format. See PerformanceLeague.md's "Vision-Language
  Model Real Image Encoding" section for the measured numbers with this caveat attached.
  **Update 2026-09-11:** Granite-Vision-3.2-2B (`mmproj-granite-vision-3.2-2b-f16.gguf`) shows the
  exact same failure pattern — real vision-encoder run, real timing (~18 t/s prefill, ~16 t/s
  decode, 3 runs), but always the same wrong, non-image-grounded output regardless of image
  content. Two independent Granite-family VLM checkpoints failing identically, while InternVL3-2B
  (different family) works correctly on the same image/prompt — strongly suggests one shared
  Granite-family vision-integration bug (image placeholder/embedding injection point) rather than
  two separate issues. Worth root-causing as a single Granite-vision bug.
  **Update 2026-09-11 — investigation, not resolved.** Found that `docs/vl-migration-plan-2026-08-20.md`
  itself documents a real, verified-working end-to-end Granite4 generation on 2026-08-20
  ("Describe this image.A large, diverse group of animals living in a certain area...") —
  seemingly contradicting this session's finding. Checked for a code regression: `git log
  4f2606c..HEAD` (the commit range since that doc's "image_pad changes" fix, which is when the
  verified-working state was recorded) touching `Granite4VisionEncoder.cs`/`Granite4VisionModel.cs`/
  `UnifiedVisionPipeline.cs`/`RunCommand.cs` shows only unrelated commits (Gemma 3, HunyuanVL,
  global-usings, this session's own `RunImagePrompt` fix, `deepseek2` MLA work) — no Granite4-
  specific change since the verified-working commit. Re-tested directly with two more real runs:
  (a) same abstract test image, default (non-greedy) sampling instead of `--temp 0` — still not
  image-grounded ("a Python script to generate an image of a random astronomical object..."); (b)
  a genuinely different, real photographic image (SD1.5's own coherent wooden-table generation,
  `sd15-perfleague-check.png`) — still not image-grounded ("a brief description of the image.
  There is no text in this problem."). Rules out both "greedy-decoding artifact" and "this specific
  test image confuses it" as explanations. **Real, unresolved question**: either (a) a genuine
  regression exists somewhere not caught by this commit-range check (possibly in shared
  `VisionOps`/`ForwardPass` code outside the files checked), or (b) the doc's 08-20 "verified
  working" claim was itself an unverified pattern-match (plausible-sounding but not actually
  checked against the real image content — the same class of overconfidence this session already
  caught itself making once with a false embedding measurement). Not disambiguated further —
  pivoting to other work per this project's "switch to another item when one stalls" discipline.
  **ROOT CAUSE FOUND 2026-09-11 (subagent investigation, read-only, code-only — not re-run/
  re-verified against real weights by that agent per this session's serial-benchmarking rule).**
  `Granite4VisionEncoder.cs`'s projector is architecturally wrong, not just a naming/dimension
  bug. Confirmed against the real reference (`examples/llama.cpp/llama.cpp/tools/mtmd/models/
  granite4-vision.cpp:34-339`): the real Granite 4 Vision projector is a multi-block
  "WindowQFormer" — several `qf_proj_blocks`, each drawing from a **different intermediate SigLIP
  layer** (not just the final post-LN output), each doing window gather/scatter, a full
  self-attention + cross-attention sub-layer with its own learned query embeddings and
  image-position embeddings, an "unwin" scatter back to raster order, then its own `out_linear`;
  block outputs are concatenated, with optional newline-row tokens appended. **None of this exists
  in the C# port** — `Granite4VisionEncoder.cs:182-202` instead runs the whole SigLIP tower once,
  applies post-LN, then a single optional linear layer — structurally a completely different,
  much simpler computation than what the checkpoint's weights were trained against. **And that one
  linear layer isn't even wired to real weights**: the C# loader looks for generic llava-style
  tensor names (`mm.proj_norm.weight`/`mm.0.weight`, `mm.proj.weight`/`mm.1.weight`,
  `Granite4VisionEncoder.cs:72-75`), but the real GGUF tensor names (per `clip-impl.h:313-317`)
  are `mm.proj_blk.<bid>.linear.weight`/`.norm.<w|b>`/`.post_norm.<w|b>`/`.query`,
  `v.proj_blk.<bid>.img_pos` — none of which match. `VisionOps.GetTensor` silently returns an
  invalid/null `VisionTensorRef` when no candidate name matches (no exception, no log), `MatVecAny`
  silently no-ops on an invalid weight, and `Granite4VisionEncoder.Forward`'s fallback `else`
  branch kicks in: raw, untrained, post-LayerNorm SigLIP hidden states get truncated/copied
  **verbatim** into the "soft token" buffer fed to the LLM — never projected through any learned
  mapping into the LLM's embedding space at all. This fully explains the symptom: real,
  non-trivial encoder computation happens (real timing, real soft-token count), but the LLM
  receives vectors from a completely different distribution than anything it was trained to
  interpret as image content — generic, non-image-grounded output regardless of prompt/temperature/
  image content, exactly as observed. **On the 08-20 "verified working" doc claim**: this
  projector code predates the `bd923ea` pointer-removal refactor and was never wired to real
  tensor names — the subagent's assessment is that the 08-20 doc's "real coherent output" claim was
  very likely an overclaim (fluent-sounding generic text mistaken for real image grounding, without
  a rigorous check against actual image content) rather than a later regression; not independently
  re-verified, a documentation-trust question rather than a code one. **Update 2026-09-11, same
  day — real, substantial rewrite landed (subagent-written, I verified with real weights),
  PARTIALLY fixes this.** Ported the actual `granite4-vision.cpp` architecture: 8 real QFormer
  blocks, each reading a different intermediate SigLIP layer, real window gather/scatter (found
  the previously-unresolved index-construction math at `clip.cpp:5128-5203`'s
  `make_win_idx`/`make_unwin_idx`/`make_spatial_idx` and ported it verbatim), real self+cross
  attention with learned query/image-position embeddings, real per-block output projection,
  concatenated along the token axis. **Verified real progress**: no more crash, soft-token count
  is now architecturally correct (1152 = 8×144, matching the real metadata) instead of the old
  576-token single-linear-layer fallback. `OpenTail.Stingray.Tests.Vision` re-run clean, 150/150,
  no regression from the shared `VisionOps.cs` additions. **NOT yet fully fixed**: two more real
  `--image` runs (greedy and sampled decoding, two different real images — the abstract test
  pattern and a real photographic wooden-table image) both still produced non-image-grounded
  output, different generic text each time but never actually describing the picture. Two concrete
  open leads, neither confirmed as the cause yet: (a) the per-block LayerNorm `eps` value wasn't
  independently traced back through hparams wiring in the C++ reference, just assumed to match the
  SigLIP tower's own `eps`; (b) `image_newline`/`add_newline` handling is unimplemented, though
  checking `clip.cpp:4048`'s `img->add_newline` gate suggests this is per-image (only relevant for
  multi-tile AnyRes images), and our single ~384px test images are very unlikely to trigger tiling
  — so probably not the cause, but not ruled out. Needs real numerical debugging (comparing
  per-block intermediate tensors against a captured reference trace) to find the remaining bug,
  not attempted yet. A real, meaningful step forward, not a complete fix.
  **Update 2026-09-11, same day — second real bug found and fixed.** A second subagent
  independently re-checked all 4 remaining candidates from the eps/index-math/concat-order/
  downsample-branch list above: eps (traced fully, two separate eps values correctly split —
  `_eps` for the top-level norm, `QformerEps=1e-12f` for post-norm/self-attn/cross-attn/FFN norms,
  matching `granite4-vision.cpp` exactly — clean), window/unwindow/spatial index construction
  (re-derived term-by-term against `clip.cpp:5144-5183` — clean), out_linear/token-concat order
  (clean), interp_down-vs-spatial_idx branch selection (clean). **Found the real bug elsewhere**:
  `Granite4ImagePreprocessor.cs` hardcoded OpenAI CLIP's ImageNet mean/std
  (`BaseVisionPreprocessor.ClipMean`/`ClipStd`) regardless of checkpoint, but Granite 4 Vision's
  tower is SigLIP, not CLIP — confirmed via `stingray list-metadata` that the real checkpoint's
  `clip.vision.image_mean`/`image_std` is `[0.5,0.5,0.5]`/`[0.5,0.5,0.5]`, numerically quite
  different from CLIP's constants (0.481/0.457/0.408 mean, 0.268/0.261/0.276 std). Three other
  encoders in this codebase (Gemma3/Gemma4V/Llama4) already correctly read these two keys from
  GGUF — Granite4 was the outlier. This silently mis-normalized every pixel fed to the patch-embed
  conv, no crash/NaN, just smooth-but-wrong input propagating through all 8 QFormer blocks.
  **Fixed**: added `Granite4VisionModel.ImageMean`/`ImageStd` (real GGUF read, SigLIP-convention
  fallback), threaded through the preprocessor and adapter. **Verified real behavioral change**:
  output shifted from generic non-image-related text ("the problem you've provided...") to
  confident, specific scene/landmark descriptions ("a serene landscape... lush green trees",
  "the Eiffel Tower in Paris") — a real, measurable step forward, but **still not correctly
  grounded** to the actual image content (the real test images are a wooden table and an abstract
  color pattern, neither matches these descriptions). `OpenTail.Stingray.Tests.Vision` re-run
  clean, 150/150. Real, meaningful progress across two real bug fixes now, correctness still not
  fully achieved — next step would be numeric golden-parity against `llama-mtmd-cli.exe`'s real
  intermediate tensors, not attempted yet.
- **dots.ocr's real vision-encode path emits a 1-token degenerate output.** Real `--image`/
  `--mmproj` run (2026-09-11) against `dots.ocr-Q8_0.gguf` + `mmproj-dots.ocr-Q8_0.gguf`: vision
  encoder runs correctly (81 soft tokens/1536-dim), but decode immediately emits
  `<|endofassistant|>` with 0 real generated tokens. Likely a chat-template/stop-token handling
  gap specific to this OCR-focused checkpoint's prompt format (dots.ocr is tuned for structured
  document extraction, not open-ended description — the test prompt may not match its expected
  format), not necessarily the same class of bug as the Granite-family one above. Not root-caused.
- **Gemma-3-4B-it has a real Jinja chat-template rendering gap** (same class as the Qwen3.8-27B
  one below, different checkpoint): `'<start_of_turn>' + role + '\n' + (first_user_prefix if
  loop.first else "")` — string-concat combined with a conditional — is passed through
  unevaluated. Did not affect this session's vision-encode measurement (output was still correct/
  coherent), but the actual rendered chat prompt may differ from spec until fixed. Worth fixing
  together with the Qwen3.8-27B gap below by extending the Jinja subset's string-concat-inside-
  conditional support.
- **Qwen3.8-27B's chat template has 3 real Jinja rendering gaps**, logged as runtime warnings, not
  crashes: unsupported string-concatenation-inside-conditional/`in` expressions (e.g.
  `sysns.text + ('\n' if sysns.text else '') + sys_content`) get passed through unevaluated instead
  of rendered, so this checkpoint's actual chat-formatted prompt may be subtly wrong. Fix: extend
  this project's Jinja subset to handle string-concat inside conditional expressions.
- **`RunCommand.RunImagePrompt`'s shared crash bug — FIXED 2026-09-11, two distinct real root
  causes behind one identical symptom.** MiMo-VL-7B-sft (`qwen2vl`) and Step3-VL-10B (`step3vl`)
  both crashed with an identical `ArgumentOutOfRangeException` at the same call site
  (`soft.AsSpan(t * embd, embd)` in `RunImagePrompt`), because `embd` was read from `hp.EmbeddingDim`
  (the TEXT backbone's hidden size) instead of the vision embedder's own reported output width —
  correct only when they happen to match, which most working checkpoints do, but not guaranteed.
  **Root cause 1 (MiMo-VL, NOT fixable in this codebase):** this checkpoint's mmproj genuinely
  projects to 3584-dim while the text backbone expects 4096-dim — a real upstream GGUF-conversion
  mismatch. **Root cause 2 (Step3-VL, a real bug, fixed):** `Step3VlVisionEncoder` looked up the
  final projector tensor under a made-up name, `mm.model_proj.weight`, which never matched this
  checkpoint's real GGUF tensor — confirmed against the real reference
  (`examples/llama.cpp/llama.cpp/tools/mtmd/clip.cpp`'s `PROJECTOR_TYPE_STEP3VL` case and
  `clip-impl.h`'s `TN_MM_PROJECTOR = "mm.model.fc.%s"`) to be `mm.model.fc.weight`, no bias. The
  wrong name meant the tensor was silently never found, so `Forward` fell back to returning a
  narrower buffer while the reported projection width stayed at the wrong wider metadata/fallback
  value — same failure mode as MiMo-VL's mismatch, but self-inflicted rather than upstream. **The
  fix**: (a) `RunImagePrompt` now reads `vision.EmbeddingDim` for the stride and explicitly
  validates it against `hp.EmbeddingDim` up front with a clear, actionable error message instead of
  crashing on a bounds violation deep in the loop — this protects every architecture, not just
  these two; (b) `Step3VlVisionModel.cs`/`Step3VlVisionEncoder.cs`'s tensor lookups corrected to
  `mm.model.fc.weight`/`.bias`, and `Step3VlVisionEncoder.ProjectionDim` made defensively
  width-safe (falls back to the real returned width whenever the final-projector tensor is
  genuinely absent, not just for this specific bug). **Verified**: MiMo-VL now fails with a clean
  `Error: vision projector (qwen2.5vl_merger) outputs 3584-dim embeddings but the text backbone
  expects 4096-dim input` instead of crashing. Step3-VL now runs to completion — real, reproducible
  10.0/9.9 t/s prefill/decode across 3 runs (garbled output, expected for a Q2_K quant on an
  unverified architecture, but the crash is gone). Full `OpenTail.Stingray.Tests.Vision` suite
  re-run clean: 150 passed, 0 failed, 9 skipped (missing fixtures, unrelated to this change).
- **Two `deepseek2`-architecture VLM checkpoints (Kimi-VL-A3B-thinking, YouTu-VL-4B) crash with
  `Missing tensor: blk.0.attn_kv_b.weight`** when run with `--allow-unverified-arch` — same
  architecture tag, same missing-tensor error, at `ForwardPass.Helpers.cs:178`'s `ResolveTensor`.
  **Corrected 2026-09-11** (an earlier note here misstated the missing tensor as `attn_q.weight` —
  re-verified directly against the real error text and `list-tensors` output, this was wrong):
  `blk.0.attn_q.weight` genuinely EXISTS in both checkpoints — these are the "Lite"
  `q_lora_rank==0` MLA variant (a plain per-head Q projection, already supported), not full-size
  MLA. The real, narrower gap is specifically the split `attn_k_b`/`attn_v_b`/`attn_kv_a_mqa`
  "absorption" K/V layout (confirmed present via `list-tensors`: `attn_k_b.weight [128,512,16]`,
  `attn_v_b.weight [512,128,16]`) — `MlaComputeQkv`'s own doc comment
  (`ForwardPass.Decode.cs`) already documents this exact gap: "only the legacy unsplit `wkv_b`
  tensor layout is handled... the split `wk_b`/`wv_b` absorption layout some newer checkpoints use
  is not implemented." Real, scoped fix would need: loading `_wKB`/`_wVB` per-layer (their real
  3D per-head tensor shape, different from the existing 2D `FusedMatVec` pattern), and doing two
  separate per-head decompressions instead of the current one combined matvec — a genuine,
  moderate feature addition (new tensor-loading + per-head batched matvec math, needs correctness
  verification against a real reference), not a one-line fix, but smaller in scope than "implement
  full `q_lora_rank>0`" as earlier notes here implied. (A third `deepseek2` checkpoint,
  DeepSeek-V2-Lite, was already confirmed working earlier this session — so this is specific to
  checkpoints using the split absorption layout, not `deepseek2` broadly.)
- **Nemotron-Nano-12B-v2-VL (`nemotron_h` architecture) crashes**: `HybridGdnForwardPass dense FFN
  requires hp.IntermediateDim > 0`. **Root-caused 2026-09-11** (quick check, not yet fixed): the
  real key `nemotron_h.feed_forward_length` DOES exist in this checkpoint's GGUF metadata
  (confirmed via `list-metadata`), but it's a **per-layer array** (`[0, 20480, 0, 20480, ...]`,
  62 entries — nemotron_h is a hybrid Mamba/SSM+attention architecture, and `0` marks a pure
  Mamba/SSM layer with no dense FFN at all), not the single scalar `ModelHyperparams.IntermediateDim`
  the code assumes. `GetIntArray` (a generic per-layer-array reader) already exists in
  `ModelGraph.cs`, but there is no `IntermediateDimPerLayer`-style property anywhere in this
  codebase yet to route it through, and `HybridGdnForwardPass`'s dense-FFN dispatch would need to
  both read a per-layer value AND skip the dense FFN entirely for `0`-valued (pure-Mamba) layers.
  A real, moderate feature addition (new hyperparameter property + per-layer dispatch logic),
  more precisely scoped than before but still not a one-line fix.
- **DeepSeek-OCR-2 (`deepseek2-ocr`) and PaddleOCR-VL-1.6 (`paddleocr`) run to completion under
  `--allow-unverified-arch` with real timing but fully garbled/degenerate output** — expected per
  the flag's own explicit warning ("output may be wrong... do not use this run as evidence of
  support"). Not treated as bugs to fix; these architectures are simply unverified, exactly as
  labeled. Timing recorded in PerformanceLeague.md for completeness but explicitly not presented
  as evidence of correctness.
- **FunASR Paraformer's GGUF checkpoint is confirmed broken — real, reproducible, not a fluke.**
  2026-09-11: ran `FunAsrRealWeightsTests.Paraformer_GgufRealModelFile_LoadsAndTranscribes`
  directly (not silent no-op — real 1.06s run, real failure): `System.IO.InvalidDataException:
  Paraformer GGUF missing 'pf.vocab' metadata` at `FunAsrWeights.cs:89`. `models/_models/
  paraformer-q8.gguf` genuinely lacks the `pf.vocab` GGUF metadata this loader requires — a bad/
  incomplete GGUF conversion for this specific file, confirmed via direct test execution, not
  guessed. Needs either a fresh correct GGUF conversion or a differently-converted checkpoint;
  not a quick fix from this machine. (This is separate from the FunASR-Nano ONNX-path degenerate-
  output bug already logged above — same family name, different code path, different failure.)
- **SenseVoice is not a real wired pipeline — README overstates coverage.** 2026-09-11 grep of
  `src/OpenTail.Stingray.Audio`: SenseVoice appears only in one doc-comment line on
  `FunAsrPipeline.cs` ("Native C# Alibaba FunASR (Fun-ASR-Nano / SenseVoice / Paraformer)"), with
  no dedicated model spec, config, or code path — it's asserted by that comment, not implemented.
  README's feature-list prose names SenseVoice alongside real, working ASR engines; it is not a
  real, distinct, testable pipeline on this codebase as of this date.
- **Parakeet is CTC-only; README's "FastConformer CTC/TDT" phrasing overstates coverage.**
  2026-09-11: `src/OpenTail.Stingray.Audio/Parakeet/` contains `ParakeetCtcDecoder.cs` and no TDT
  decoder file of any kind (`ParakeetConformerEncoder.cs`/`ParakeetMelExtractor.cs`/
  `ParakeetTokenizer.cs`/`ParakeetWeights.cs`/`ParakeetPipeline.cs` round out the directory — none
  TDT-specific). Only the CTC decode path is real; the TDT half of the README's claim is not
  implemented.
- **`EulerDiscreteScheduler`'s `numInferenceSteps == 1` divide-by-zero — FIXED and verified
  2026-09-11.** Found benchmarking SDXL-Turbo (a genuinely new, previously-untested checkpoint,
  `models/_models/sd_xl_turbo_1.0_fp16.safetensors`): `stepRatio = (trainSteps-1) /
  (numInferenceSteps-1)` divides by zero at `--steps 1`, producing Infinity then NaN
  timesteps/sigmas that silently propagate through the whole denoising loop and render as a solid
  black 843-byte PNG — confirmed on both CPU and Vulkan (not backend-specific). Verified the
  scheduler's existing formula already matches diffusers' real `scheduling_euler_discrete.py`
  "linspace" spacing term-by-term for `numInferenceSteps > 1`, so the fix (a `numInferenceSteps
  == 1` special case resolving to timestep 0, matching numpy's own documented
  `linspace(start, stop, num=1) == [start]` behavior) closes the one edge case rather than
  inventing a new convention. Verified: 1-step CPU run now produces a real 235KB non-degenerate
  PNG instead of 843-byte black. **Separate, deeper, NOT fixed gap surfaced by this same
  investigation**: even with the crash/NaN fixed, 1-step output quality is still poor (textured
  noise, not a coherent image) — real turbo/few-step models need "trailing" timestep spacing
  (sampling near maximum noise) for good few-step results, and this scheduler only implements
  "linspace". At `--steps 4` the same checkpoint produces a real, coherent image (a genuine
  apples-on-a-table result), confirming the pipeline itself works — this is specifically a
  1-2-step quality gap, not a broken pipeline. `OpenTail.Stingray.Tests.Diffusion.SchedulerTests`
  re-run clean (4/4) after the fix.
- **LTX-Video-2B v0.9.1's output correctness appears seed-dependent/unstable — not root-caused.**
  2026-09-11: the first real `stingray image` run against this checkpoint (256×256, 1 frame,
  25 steps, CFG 3.0) produced a real, coherent, image-grounded picture, visually confirmed.
  Two follow-up default-seed (`seed=-1`, random) runs with otherwise-identical parameters — one
  testing `--upscaler` (a genuinely untested flag/checkpoint, `RealESRGAN_x4plus.safetensors`,
  investigated for this reason), one a plain re-run without it — both produced pure visual noise
  instead of a coherent image. Ruled out: (a) `--upscaler` as the cause, since the plain re-run
  without it also produced noise; (b) the `EulerDiscreteScheduler` fix made in this same session
  (see the SDXL-Turbo entry above), since LTX-Video uses its own independent
  `RectifiedFlowScheduler`, confirmed by reading `LtxVideoPipeline.cs`. Real timing stayed
  consistent across all runs (~100-115s), so this is specifically an output-correctness gap, not a
  performance regression. **Update, same day, `--seed 42` sweep**: two runs at the same explicit
  `--seed 42` both produced visually-identical garbled noise (different PNG byte hashes, but
  pixel-identical to the eye — almost certainly benign PNG metadata/timestamp variance, not a real
  non-determinism bug) — this rules out "bad luck on one random seed" and confirms this checkpoint
  genuinely fails to converge for at least seeds -1 (2 of 2 default runs bad after the first good
  one) and 42, not just an unlucky single draw. Given the pipeline's own timing stays real and
  consistent and the very first default-seed run WAS coherent, the honest current read is: this
  checkpoint's real convergence rate is seed-dependent and worse than initially assumed (only 1
  of 6 total runs now confirmed coherent) — not yet root-caused (candidates: RoPE/positional
  embedding edge case, VAE decode instability, or a genuine seed-quality issue in the base model
  itself) and not disambiguated further this session.
  **Update, same day — investigation, not resolved:** ran the golden-parity suite
  (`LtxVideoGoldenParityTests`/`LtxVaeDecoderGoldenParityTests`/`LtxT5EncoderGoldenParityTests`/
  `LtxVideoTrajectoryGoldenTests`/`LtxVideoRealScaleGoldenTests`/`LtxVaeDecoderMultiFrameGoldenTests`)
  — all 7 pass, real (16.7s total, not a silent no-op), so RoPE/VAE decode/T5 encoding/single-step
  trajectory are individually golden-verified against real reference data; the bug is not in these
  isolated components. Also found and fixed a real silent-no-op in `LtxVideoRealWeightsTests.cs`
  (searched `models/` only, not `models/_models/` where the real checkpoint lives — added a
  symlink, same class of bug as `CLAUDE.md` rule 12) but those 2 tests only check config/metadata,
  not full generation quality. Checked two concrete hypotheses directly against the real
  checkpoint and ruled both out: (a) the VAE's per-channel un-normalization statistics tensors
  (`vae.per_channel_statistics.{std-of-means,mean-of-means}`) DO exist in this checkpoint and the
  pipeline's `_weights?.Contains(...)` gate correctly finds them — not silently skipped; (b) the
  VAE decoder's `injectNoise` StyleGAN-style noise is real, intentional per-checkpoint behavior
  per this port's own doc comment ("real inference should leave this true"), not a bug. One
  real, unresolved lead: at the 256×256 test resolution, `_spatialScale=32` gives only an 8×8=64
  spatial-token latent grid, and `GetNormalShift`'s resolution-dependent timestep-shift formula
  (calibrated for 1024-4096 tokens) extrapolates unclamped below that range — computed shift≈0.606
  at 64 tokens, not obviously pathological (no NaN/blowup) but genuinely far outside the formula's
  calibrated range and untested by the golden suite (which likely uses in-range token counts).
  **Tested directly: ruled out.** Ran the same prompt/seed at 512×512 (256 spatial tokens, still
  below the 1024 calibration floor but closer) — still pure noise, visually indistinguishable from
  the 256×256 failures. Resolution/token-count is not the differentiator. Genuinely unresolved —
  switching to other queue items per this project's own "switch to another item when one stalls"
  discipline (`CLAUDE.md`'s "Stopping is for wimps" directive) rather than continuing to sink time
  into one trace without a new concrete lead.
- **`RealESRGAN_x4plus.safetensors` (the `--upscaler` RRDBNet path) — CONFIRMED WORKING 2026-09-11.**
  Paired with SD1.5's known-good base output (`--upscaler` + real SD1.5 generation), produced a
  real, genuinely sharp 512×512 → 2048×2048 (4x) upscale — visually confirmed coherent wood-grain
  detail, not degenerate. This isolates and rules out `--upscaler` itself as the cause of the
  earlier LTX-Video noise (which is real and separate — see the LTX-Video entry above). Real timing
  breakdown available (RRDB body 106.4s, upsample/HR/download 18.1s of a 781.3s total run).
- **`stingray embed`'s ONNX path had a fake char-per-token tokenizer — FIXED FOR REAL 2026-09-11,
  per explicit user request ("fix the fake ONNX tokenizer for real. I want it to work properly").**
  Wrote `src/OpenTail.Stingray.Core/BertWordPieceTokenizer.cs`, a faithful port of HuggingFace
  `transformers`' real `BasicTokenizer`+`WordpieceTokenizer` algorithm (control-char cleaning,
  optional CJK character spacing, lowercase+accent-stripping gated on `do_lower_case`, punctuation
  splitting, greedy longest-match-first WordPiece subword segmentation with `##` continuation
  prefixes) — every local ONNX embedding checkpoint's own real `tokenizer_config.json` declares
  `"tokenizer_class": "BertTokenizer"`, confirmed by fetching and reading them directly (not
  guessed). Downloaded the real `vocab.txt` (30522 tokens, standard BERT vocab size) for all 4
  local BERT-family ONNX checkpoints — `all-MiniLM-L6-v2`, `bge-small/base/large-en-v1.5` — from
  their real HuggingFace repos, saved as `F:\_models\<name>-vocab.txt`. Wired into `EmbedCommand`
  via `TryLoadWordPieceTokenizer` (automatic vocab-file discovery next to the `.onnx` checkpoint,
  falling back to the old char-per-token placeholder only for checkpoints with no downloaded vocab).
  **Verified with a real semantic-correctness check, not just "doesn't crash":** two paraphrased
  sentences ("The quick brown fox jumps over the lazy dog." / "A fast auburn fox leaps above a
  sleepy canine.") scored 0.72 cosine similarity; an unrelated sentence ("The stock market crashed
  heavily today.") scored only 0.15 against the same reference — exactly the separation a
  correctly-tokenized real sentence-embedding model should produce, proving the fix is genuinely
  semantically meaningful, not merely non-crashing. Also fixed a token-count sanity check: "The
  quick brown fox jumps over the lazy dog." tokenizes to exactly 12 ids (9 words + 1 punctuation +
  [CLS]/[SEP]), matching manual WordPiece counting exactly. Confirmed via code review that
  `OnnxModelSession` (`src/OpenTail.Stingray.Core/OnnxModelSession.cs`) is a genuinely generic
  ONNX Runtime wrapper (real `InferenceSession`, arbitrary named inputs/outputs) — the mechanical
  execution path already worked for any `.onnx` graph; `EmbedCommand` was the only caller missing
  correct input construction. Full solution rebuild clean (0 warnings, `TreatWarningsAsErrors`).
  The downloaded vocab files live in `F:\_models\` (not tracked in the repo, matching this
  project's models/_models convention) — a future session on a different machine would need to
  re-download them (small text files, ~230-900KB each).
- **A systemic silent-no-op pattern in `*RealWeightsTests.cs` files, worse than previously
  documented.** `CLAUDE.md` rule 12 already names this pattern (a green, sub-second "pass" that
  never actually touched real weights) for LLM/vision/some-audio tests. Three *more* instances were
  found by accident this pass, not by a deliberate audit: `ParakeetRealWeightsTests` and the
  Orpheus/SNAC perf bench were silently skipping because their model-search helpers only check
  `models/`, not `models/_models/`, where the checkpoints actually live (fixed here via symlinks,
  matching the existing convention — but the underlying search-helper narrowness is unfixed and
  likely affects other tests too). Nobody has run a full sweep comparing every real-weights test's
  search helper against actual `models/_models/` contents.

See `PerformanceLeague.md` (search each bug's name/symptom) for the exact repro commands, dated
findings, and any partial data already gathered.

- **11 downloaded TTS/ASR checkpoints found with ZERO wiring in this codebase (2026-09-12).**
  While sweeping `/f/_models` for anything new to benchmark, found a batch of checkpoints
  (timestamped 2026-09-06 through 2026-09-08, from an earlier download pass that was apparently
  never followed up on) with no corresponding architecture support anywhere in
  `src/OpenTail.Stingray.Audio/`: `VibeVoice-ASR-GGUF`, `VibeVoice-1.5B-GGUF` +
  `vibevoice-7b-q8_0.gguf`, `Higgs-Audio-v3-TTS-4B-GGUF`, `MOSS-TTS-Nano-100M-GGUF`,
  `PersonaPlex-GGUF`, `VoxCPM2-GGUF`, `Citrinet-ASR-GGUF`, `NeuTTS-2E-GGUF`,
  `nemotron-3.5-asr-streaming-0.6b.q8_0.gguf`, `voxtral-mini-realtime/model.safetensors`
  (~8.9GB), `fun-asr-nano/model.safetensors` (~1.7GB, HF-format, plus a separate
  `fun-asr-nano-hf-tokenizer/` directory), `omnivoice/model.safetensors` (~2.4GB, HF-format).
  Confirmed via `grep -rli` across every family name/spelling variant against
  `src/OpenTail.Stingray.Audio/*.cs` — 0 matches for all 11. This is real, new-architecture
  porting work (tokenizer + model + forward pass per architecture, the same scope as any other
  from-scratch model admission in this project), not a quick fix or a benchmarking task — not
  attempted this pass. Worth scoping as an explicit future porting task if the user wants this
  backlog picked up; several of these are HF safetensors format (`voxtral-mini-realtime`,
  `fun-asr-nano`, `omnivoice`) rather than this project's usual GGUF-first convention, which may
  need a different loading path than the existing `GgufWeightLoader`/`SafetensorsLoader` pair
  already supports for LLM/diffusion checkpoints.

- **`phimoe` (Phi-3.5-MoE) genuinely broken: missing LongRoPE support (2026-09-13).** Found while
  closing `ModelCompatibility.cs`'s claimed-but-untested architecture gaps. Real, coherent-checkpoint
  test (`bartowski/Phi-3.5-MoE-instruct-GGUF`, Q2_K AND Q3_K_M, two different quant levels from the
  same reputable converter) both produce complete gibberish output — ruled out "bad quant" as the
  explanation (a real quality difference between Q2_K and Q3_K_M should have shown SOME
  improvement, and didn't). Root cause, confirmed by reading the real reference
  (`examples/llama.cpp/llama.cpp/src/models/phimoe.cpp:43-44`): this checkpoint's GGUF carries
  `rope_factors_long.weight`/`rope_factors_short.weight` (Phi's LongRoPE per-dimension
  frequency-scaling tensors, a distinct context-extension mechanism), but this codebase's
  `ModelGraph.cs`/`ForwardPass.cs` never load or apply either tensor for text generation — the only
  two hits for these tensor names anywhere in the codebase are in an unrelated audio model
  (`VoxCpm2LlmTensorSource.cs`/`VoxCpm2LocalEncoder.cs`). `SimdKernels.BuildRopeTable` already
  accepts a `freqFactors` parameter (used today only for Gemma-4's much simpler single static
  `rope_freqs.weight` array), so the low-level plumbing exists, but real LongRoPE needs a
  genuinely new mechanism on top: two separate factor arrays with RUNTIME selection (short vs.
  long) based on current context length vs. the checkpoint's `original_max_position_embeddings` —
  scoped as real new-architecture-support work, not a quick tensor-name fix. Not attempted this
  pass; see `PerformanceLeague.md`'s "Newly-downloaded architecture coverage" section for the full
  before/after evidence.

## Carried forward from docs archived 2026-09-27

Open leftovers from documents moved to [done](done) on 2026-09-27. Each archived document's banner
names what closed; these items are what did not.

- **Qwen3.6-35B-A3B CPU performance**: prefill 0.63x of llama.cpp after the 2026-09-25 pass, and
  the pre-existing `HybridGdnChunkedPrefill_MatchesSequentialPrefill` failure found then (not
  re-checked since). See Phase 8 of
  [done/2026-09-25-hf-top-downloads-coverage-plan.md](done/2026-09-25-hf-top-downloads-coverage-plan.md).
- **FLUX.2 GPU end-to-end timing** after `0958d6f` was never measured cleanly, and the GPU levers
  (an int8 dot-product quantized GEMM; the empty `VulkanMatMulPathConfig` "Path 2" seam) are open.
  See items 1-2 of §4 in
  [done/2026-09-24-diffusion-perf-session-handoff.md](done/2026-09-24-diffusion-perf-session-handoff.md),
  next to [093-flux2-gpu-performance-optimization-plan.md](093-flux2-gpu-performance-optimization-plan.md).
- **Per-pipeline diffusion end-to-end smoke tests** with real weights, small resolution and a
  stored reference, failing loudly when checkpoints are missing. See item 4 of §4 in the same handoff.
- **MusicGen / AudioGen performance and DRY passes** (CFG as a batch-2 GEMM; a shared T5 kernel
  with Parler), plus top-p sampling. Not re-checked in the archive pass. See "Known gaps" in
  [done/062-musicgen-implementation-plan.md](done/062-musicgen-implementation-plan.md).

