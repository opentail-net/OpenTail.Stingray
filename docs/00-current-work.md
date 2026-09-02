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
3. **CosyVoice 2/3 speaker-identity transfer** — **re-scoped 2026-09-01, judged by ear 2026-09-01:
   quality is sub-par.** The handoff doc describing "cond is all-zero" and "CFG is missing" was
   stale — both were already fixed by an earlier commit (`4cb189f`). Re-verified the real mel
   extractor (`CosyVoiceMelExtractor.cs`) line-for-line against `cosyvoice-frontend.cpp`'s
   `extract_speech_feat`/`build_mel_basis`, and ran a real end-to-end generation with `--ref-audio`
   — every stage produces real, non-degenerate data (real prompt tokens, real mel frames, real
   speaker embedding, real CFG). The operator listened to `docs/audio-samples/
   cosyvoice3-identity-check.wav` and confirmed speaker-identity transfer is not convincing. Since
   every numeric stage checked real, this is likely a genuine subtle bug, not a missing feature —
   next real step is a stage-by-stage numeric diff against the real CosyVoice3 Python/C++
   reference (speaker embedding, prompt/speech tokens, DiT mel output), not further structural
   re-verification. See `docs/qwentts-cosyvoice3-handoff.md` §2 for the full history.

4. **LTX-Video — core port and performance pass complete; trajectory convergence pending** (2026-09-01).
   All individual architecture blocks are implemented, performant, and pass golden numeric parity
   tests (>0.999 cosine-sim against HuggingFace diffusers & official `ltx-video` references):
   - 28-layer PixArt-style DiT (`LtxVideoModel.cs`) with continuous 3D RoPE.
   - Timestep-conditioned 3D causal VAE decoder (`LtxVaeDecoder.cs`).
   - T5-v1.1-XXL text encoder (`T5Encoder.cs`) & exact Unigram SentencePiece tokenizer.
   - Measured performance pass landed: 12.5% faster DiT forward, 1.93x faster VAE decode.
   - **Remaining Multi-Step Convergence Gaps** (separate milestone, see
     [055-ltx-video-implementation-plan.md](055-ltx-video-implementation-plan.md)): end-to-end visual
     smoke test (`ltx_test_apple_512.png`) shows clear prompt semantic adherence but high-contrast/
     dither artifacts. Requires step-by-step multi-step Euler trajectory oracle diffing against
     reference `pipeline_ltx_video.py`, VAE spatial noise gating, and CFG guidance rescaling.
5. **SentencePiece Unigram-LM tokenizer — implemented 2026-09-01.** `GgufTokenizer` now detects
   `tokenizer.ggml.model=t5` (real llama.cpp's `LLAMA_VOCAB_TYPE_UGM` trigger) and routes through
   the existing real Viterbi `UnigramTokenizer` (built for Parler-TTS's T5 encoder) via a new
   `UnigramTokenizer.FromGgufVocab(tokens, scores, unkId, tokenTypes)` factory — uses GGUF's real
   `tokenizer.ggml.token_type` array (NORMAL=1) for the UNK-fallback-score computation instead of
   `FromTokenizerJson`'s bracket heuristic. Also fixed the UGM-specific default `unknown_token_id`
   (2, not SPM's 0). Full solution builds clean; `UnigramTokenizerTests` covers `FromGgufVocab`
   matching `FromTokenizerJson` byte-for-byte on Parler's real vocab, plus a synthetic
   token-type-array case. `precompiled_charsmap` binary normalization remains a documented known
   gap on `UnigramTokenizer`'s class doc (plain-ASCII input unaffected).
   **Correction, same day**: re-downloading `minicpm` to admit it (below) found the "six
   Unigram-LM-blocked architectures" framing was speculative and wrong for at least `minicpm` — its
   `model=llama` + scores-only vocab is plain SPM-BPE with the merges array simply absent (a GGUF
   export convenience, not part of the real algorithm), already fixed by
   `GgufTokenizer.SpmMergePiecesByScore` (the `xverse`-motivated fix), not by today's Unigram-LM
   work at all. `internlm2`/`ernie4_5`/`baichuan`/`orion`/`nanbeige` were all diagnosed the exact
   same way `minicpm` originally was. `ernie4_5` is now ALSO ADMITTED (2026-09-01, zero new
   architecture code — same shape as `exaone`) — and re-checking it surfaced a real THIRD SPM
   tokenizer bug, general-purpose, not `ernie4_5`-specific: `GgufTokenizer.EncodeSpm` mapped any
   post-merge piece with no direct vocab entry straight to `UnknownTokenId`, instead of real
   llama.cpp's per-UTF8-BYTE `<0xXX>` byte-fallback lookup. This plausibly affected every
   already-admitted SPM architecture whenever a byte-fallback-triggering character (control chars,
   rare Unicode) showed up — none of their receipts happened to test one. Fixed via
   `GgufTokenizer.AppendSpmByteFallback`, covered by a new synthetic-vocab test in
   `SpmMergeByScoreTests.cs`. `baichuan` — at least the common
   `shaowenchen/baichuan2-7b-chat-gguf` conversion — needed **zero changes**: that GGUF declares
   `general.architecture=llama` (a converter quirk that splits `W_pack` into ordinary Q/K/V
   tensors), so it was never actually blocked once the SPM fix landed; full 8/8-token greedy match,
   nothing to admit. `internlm2` is now ALSO ADMITTED (2026-09-01): same already-fixed tokenizer
   case, and zero new architecture code needed — no `internlm2`-specific kernel gate existed
   anywhere in `ModelGraph.cs`/`ForwardPass.cs`, so this engine's existing generic pre-norm/GQA/RoPE
   dispatch already covered it; full 8/8-token exact greedy match. `minicpm` is now ADMITTED too —
   see item 8 below. **`orion`/`nanbeige` checked 2026-09-01, closing out the six-architecture
   sweep**: `orion`'s tokenizer axis is also the same already-fixed case, but its architecture side
   still needs real new code (a small LayerNorm-with-bias + gated-SiLU-FFN kernel combination, not
   yet built) — a real, scoped, low-effort task for whenever architecture work is prioritized.
   `nanbeige` turned out to be a bigger, different problem than originally scoped: the current
   Nanbeige4.x family (`nanbeige.num_loops`/`nanbeige.skip_loop_final_norm` in its GGUF metadata)
   is a genuine Looped-Transformer — a fixed set of physical layers reused N times for extra
   effective depth — a real new execution mechanism this engine has never implemented, not a small
   kernel gap like `orion`'s. Do not treat `nanbeige` as a peer-priority item to `orion` going
   forward; it needs its own scoping pass, **and per 2026-09-01 operator review it isn't worth that
   pass regardless — real Nanbeige download volume is too low to justify a genuinely new execution
   mechanism for it. Deprioritized off this list entirely, not just reordered.**
   **`orion` ALSO ADMITTED (2026-09-01)**: the "small new LayerNorm-with-bias + gated-SiLU-FFN"
   architecture task this note originally scoped turned out to already be free too — `UsesLayerNorm`
   is driven purely by tensor presence (`blk.0.attn_norm.bias`, architecture-agnostic detection
   already used by `gptneox`/`falcon`/`codeshell`), `orion`'s gated-SiLU FFN is the same shape the
   plain llama FFN path already builds, and `orion` was already in the NEOX-RoPE dispatch table.
   Allowlist string alone was sufficient. Full 4/4-token exact greedy match.
   **Six-architecture sweep result: 5 ADMITTED with zero new architecture code (`minicpm`,
   `baichuan`, `internlm2`, `ernie4_5` — the last one also surfacing a real general SPM
   byte-fallback bug, fixed — and `orion`), 1 deprioritized (`nanbeige`, looped-transformer
   mechanism, too low demand to pursue). Every architecture originally diagnosed as needing real
   new code turned out not to, once actually checked against real reference source and real
   checkpoints instead of assumed from an earlier read.**
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
    popular; that violates the consolidation strategy this list is built around.
12. **FLUX.1 — run for the first time 2026-09-01: real bugs found and fixed, but output is still
    wrong (structural stub → real-but-broken port, same stage as Wan/LTX).** Downloaded a full
    real checkpoint set for the first time ever (`city96/FLUX.1-schnell-gguf` Q2_K DiT + real
    `ae.safetensors`/`clip_l.safetensors`/T5-XXL fp8 text encoders + real CLIP/T5 tokenizer.jsons,
    all transient, deleted after this pass) and ran `opentail-llm image` end-to-end. Found and
    fixed THREE real bugs along the way, each caught only because this was the first real
    execution:
    1. **Tensor-name prefix mismatch**: `FluxDiT.cs` hardcodes `model.diffusion_model.*` tensor
       names (the real diffusers/safetensors convention), but city96's GGUF converter strips that
       redundant prefix since the file only ever holds DiT tensors — every single weight lookup
       failed immediately. Fixed via a `FindTensor` fallback that tries both conventions.
    2. **Two missing `.weight` suffixes**: `SingleBlock`'s `linear2` and `FinalLayer`'s `linear`
       `MatQ` calls passed the bare tensor-name prefix instead of appending `.weight` (every OTHER
       call site does this via the `LinearNoBias`/`LinearBias` helpers, but these two called `MatQ`
       directly and were never exercised until now).
    3. **Real architectural bug in `SingleBlock`'s MLP**: the existing code applied GEGLU (split
       the MLP hidden state in half, gate one half with the other) — but confirmed against the real
       diffusers `FluxSingleTransformerBlock` reference (`transformer_flux.py`) AND independently
       against this checkpoint's own `linear2` weight shape (`[3072, 15360]` = `[d, 5d]`, only
       consistent with concatenating the FULL un-split, un-halved `4d`-wide MLP output with the
       `d`-wide attention output — GEGLU's halved `2d` would need a `[d, 3d]` shape, which the real
       weight isn't), FLUX's single-block MLP is a plain GELU(tanh-approx) over the full
       `mlp_hidden_dim = 4·d`, never gated/split at all. The code's own inline comments
       (`"Wait, let me reconsider..."`) show this was a half-resolved guess from whenever it was
       first written, never actually checked against reference or run. Fixed: full-width
       `GeluInPlace`, `combined` dim corrected to `d + 4d`, dead `Geglu` helper removed.
    **After all three fixes, the pipeline runs to completion (860.8s, CPU-only, 4 steps) without
    crashing** — a real, substantive improvement — **but the output image
    (`docs/diffusion-samples/flux-schnell-first-run.png`) is a periodic tiling/checkerboard
    pattern, not anything resembling the prompt ("a red apple on a wooden table").** The
    regularity of the pattern (a small repeating tile across the whole 512×512 frame, not random
    noise) points at a patchify/unpatchify or latent-packing ordering bug — img/txt sequence
    concatenation order, patch-to-pixel unpacking, or 2D RoPE position-id assignment are the next
    places to check — but this was NOT investigated further this pass (stopped to report back
    rather than open a much deeper blind investigation). Not yet at LTX-Video's "golden numeric
    parity on every individual block" stage — FLUX has no golden/reference numeric tests at all,
    so the next real step (if picked up) is the same discipline LTX-Video used: per-block
    numeric diffing against the real diffusers reference, not further guessing from output shape
    alone. `SD3/Sd3Pipeline.cs` remains completely unexecuted still — not attempted this pass.

    **Round 2, same day: a FOURTH real bug found and fixed (2D RoPE), but the tiling artifact
    survived it unchanged.** `Flux2DRoPE.cs` had two independent, compounding real bugs, confirmed
    against black-forest-labs/flux's actual `flux/math.py` (`rope`/`apply_rope`) and
    `flux/modules/layers.py` (`EmbedND`), not guessed:
    - **Wrong axis split.** Real FLUX uses THREE position axes (`axes_dim=[16, 56, 56]`): a
      leading always-zero axis (a no-op/identity rotation for plain text-to-image — no time
      dimension), then row (56), then col (56), each independently theta-scaled using ITS OWN
      axis dim (56), not `head_dim` (128). The previous code split `head_dim` evenly in half
      (64 row / 64 col) with no identity portion and the wrong per-axis theta scale.
    - **Wrong rotation convention, and column position silently never used at all.** Real FLUX
      rotates ADJACENT element pairs `(x[2i], x[2i+1])` (the GPT-NeoX/interleaved convention).
      The previous code implemented "rotate-half" instead (`x[i]` paired with `x[i+head_dim/2]`)
      — a different, incompatible convention — and compounding that, its rotation loop only ever
      read frequency slots `[0, head_dim/2)`, meaning the column-axis frequencies (stored at
      `[head_dim/2, head_dim)` under the old half/half split) were computed but **never actually
      read**. Every patch's horizontal position was silently ignored entirely, regardless of the
      convention bug.
    Rewrote `Flux2DRoPE.cs`'s `BuildFreqs`/`ApplyInPlace` to the real 16/56/56 split and real
    interleaved-pair rotation. Full `Tests.Diffusion` suite (91 tests) still green after the
    rewrite. Re-ran the same repro (below): output is BYTE-DIFFERENT from the pre-fix run
    (confirmed via hash, so the fix has a real effect) but the periodic tiling artifact's
    character is visually unchanged. This is itself informative: if RoPE (a purely positional
    signal) were the dominant cause of a periodic per-patch tile, fixing it should have changed
    the artifact's STRUCTURE, not just its exact values — surviving a real, verified fix points
    away from "subtle attention-wiring/positional bug" and toward something more mechanical,
    either upstream of the DiT's semantic content (VAE latent shift/scale conditioning) or
    downstream of it in a way that's structurally periodic regardless of input (e.g. a
    Q2_K-specific dequant/matvec kernel artifact). Patchify/unpatchify (`EulerFlowScheduler.
    PackLatent`/`UnpackLatent`) was also re-checked against the real
    `"b c (h ph) (w pw) -> b (h w) (c ph pw)"` einops rearrange this pass and found already
    correct — channel-outer-then-row-then-col patch layout, row-major patch-grid sequence order,
    both matching real FLUX exactly.

    **Update 2026-09-01, rounds 2-4 (see `docs/056-flux-tiling-artifact-handoff.md` for full
    detail, kept current there — this entry is not being kept in sync line-by-line going
    forward)**: two more real bugs found and fixed against the real BFL reference (a
    flow-matching Euler integration sign inversion, and a `[txt,img]`-vs-`[img,txt]` stream
    token-ordering bug) — six real bugs fixed total now. Tiling artifact still not resolved,
    now with a visible seam. A performance rewrite (`Workspace`/`Parallel.For`) landed in the
    same round; an initial race-condition suspicion was tested and DISPROVEN (two identical runs
    produced byte-identical output), and an initial "slower" timing measurement turned out to be
    shared-machine contention, not a real regression — the performance change is not implicated
    in the artifact. Paused again pending the next round (VAE conditioning is the next suspect).
    Checkpoints used (all deleted after each pass, re-download to resume):
    DiT `city96/FLUX.1-schnell-gguf` → `flux1-schnell-Q2_K.gguf` (4.01 GB); VAE
    `ffxvs/vae-flux` → `ae.safetensors` (335 MB, ungated mirror — `black-forest-labs/
    FLUX.1-schnell`'s own copy is access-gated); CLIP-L `comfyanonymous/flux_text_encoders` →
    `clip_l.safetensors` (246 MB); T5-XXL `comfyanonymous/flux_text_encoders` →
    `t5xxl_fp8_e4m3fn.safetensors` (4.9 GB); CLIP tokenizer `openai/clip-vit-large-patch14` →
    `tokenizer.json`; T5 tokenizer `YuCollection/FLUX.1-schnell-Diffusers` →
    `tokenizer_2/tokenizer.json` (plain `google/t5-v1_1-xxl` has no fast-tokenizer `tokenizer.json`
    of its own). Repro command: `opentail-llm image -m flux1-schnell-Q2_K.gguf --vae
    ae.safetensors --clip-l clip_l.safetensors --clip-tokenizer tokenizer.json --t5xxl
    t5xxl_fp8_e4m3fn.safetensors --t5-tokenizer tokenizer_2/tokenizer.json -p "a red apple on a
    wooden table" --steps 4 --seed 42`.
13. **Stable Audio 3** — **checked 2026-09-02, genuine unwired stub**, same shape LTX-Video was
    before its plan: `StableAudioDiT`/`AcousticVaeDecoder`/`StableAudioPipeline` exist in
    `src/OpenTail.Stingray.Diffusion/StableAudio/` but take no `IWeightLoader` and read no real
    tensors anywhere. No local checkpoint downloaded. Not yet planned/scoped the way LTX-Video was
    — would need its own architecture-inventory-first planning pass before any implementation.

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

## Vision golden-verification pass (2026-09-01): 5 real numeric bugs found and fixed in 2 of the
## 6 confirmed-working architectures; 3 more need a real missing feature, not a bug fix

Elevated the 6 architectures the real-weight differentiation suite confirms as "not obviously
broken" (`Llava`, `Pixtral`, `Exaone4`, `MiMoVl`, `Qwen2.5-VL`, `GLM-4.6V`) toward real golden
numeric parity, following the existing `gemma4uv_ref.py` pattern (hand-written numpy port of the
real llama.cpp mtmd C++ reference, reading the same local mmproj GGUF, checked against the C#
encoder's actual output). This is a stronger claim than the differentiation test, which only
checks "not degenerate" (no NaN, has variance, distinguishes two different images) — it doesn't
check numeric correctness at all.

**`Llava` and `Pixtral`: golden-verified, 5 real bugs found and fixed** (see
`tests/OpenTail.Stingray.Tests.Vision/LlavaVisionEmbedderParityTests.cs`/
`PixtralVisionEmbedderParityTests.cs`, `scripts/llava_ref.py`/`pixtral_ref.py`):
- Llava: `LlavaVisionModel.ProjectionDim` read the wrong metadata key (`clip.vision.
  projection_dim`, CLIP's own unrelated native projection head) instead of deriving the real
  llava-projector output width from `mm.2`'s actual tensor shape — every projector matmul was
  silently truncated. Also: this checkpoint's `ffn_up`/`ffn_down` tensor NAMES are backwards
  relative to their real direction; fixed by deriving role from each tensor's own real input
  width instead of trusting the name.
- Pixtral: the MLP projector looked for the non-existent `mm.0.weight` (real pixtral tensors are
  `mm.1`/`mm.2`) — silently fell back to a raw truncating copy, never running the real projector
  at all. Also: 2D RoPE had the row/col halves swapped and was missing a real `freq_scale_odd`
  factor on the second half.

Both golden-verified now; full `Tests.Diffusion` suite and the differentiation suite re-run clean
after these changes (same 4 pre-existing unrelated failures below, no new regressions).

**`Exaone4`, `MiMoVl`, `Qwen2.5-VL`: NOT a bug-fix task — a real, missing feature (windowed
attention) blocks meaningful golden verification.** Checked before writing any Python reference
(cheaper than discovering it via a failed numeric comparison): none of
`Exaone4VisionEncoder.cs`/`MimoVlVisionEncoder.cs`/`QwenVlVisionEncoder.cs` reference "window" at
all — they always run full (unmasked) attention on every layer. But all three real local
checkpoints declare `clip.vision.n_wa_pattern` (7 for Exaone4, 8 for MiMoVl and Qwen2.5-VL),
meaning the real llama.cpp reference applies LOCAL/masked attention on most layers (only every
Nth layer gets full attention), plus a real window-index reordering step
(`window_idx`/`inv_window_idx` in the real C++) around it. This is fundamentally different in
kind from the Llava/Pixtral bugs above — those were wrong values in an otherwise-complete
computation; this is a missing computation affecting the majority of layers. Building a "golden
test matching current C# scope" the way Pixtral's IMG_BREAK gap was handled would be much less
meaningful here, since windowed vs. full attention changes the actual math for most of the
network, not just a final step. **Real estimated scope: implement windowed local attention +
index reordering for three architectures — a genuine new-feature task, not a golden-test
elevation.** Deliberately not attempted this pass (operator call, 2026-09-01) — do this properly
as its own scoped task later, not opportunistically mid-golden-test-pass.

`GLM-4.6V` has no `n_wa_pattern`/window metadata at all in its real local checkpoint, so it does
not have this gap — tractable the same way Llava/Pixtral were, see below for its own result.

**`GLM-4.6V`: 4 real bugs fixed (attention and patch-merger were both complete no-op stubs), but
golden numeric verification is blocked by a second, different real missing feature (2×2
merge-block patch reordering) — deferred alongside the windowed-attention gap above.**
Cross-checked `Glm4VisionEncoder.cs`/`Glm4VisionModel.cs` against the real
`tools/mtmd/models/glm4v.cpp` (122 lines) before writing any Python reference and found the
existing C# encoder was far more broken than the differentiation test (`Glm4V_RealWeights_
LoadsAndEmbedsImage`, which only checks non-degeneracy) could ever catch:
- `ComputeAttention` never read Q or K, never scored, never softmaxed — it just copied scaled V
  straight through. Fixed: routed through the shared, real `VisionOps.Attention` (the same
  vectorized scaled dot-product + softmax every other working encoder uses).
- The patch merger declared `mm.patch_merger.weight`/`.bias` as fields but never once used them —
  it just concatenated raw 2×2 patch groups and fed that straight into the FC projector. Real
  glm4v.cpp does a genuine strided Conv2D through `mm.patch_merger.weight` (kernel=stride=2,
  1536→4096) first. Fixed: implemented the real conv2d merge.
- The FC projector looked for `mm.fc.weight`/`mm.0.weight`, neither of which exists in this
  checkpoint (confirmed via `list-tensors`) — real name is `mm.model.fc.weight`. Silently fell
  back to a truncating raw copy every time. Fixed: added the real name as the primary candidate.
- The entire projector tail past FC was missing: real glm4v.cpp does
  `mm.post_norm` (plain LayerNorm, eps=1e-5 — distinct from the ViT's own RMS eps) → `ggml_gelu_
  erf` (erf-based GELU, not the tanh-approximation used elsewhere in this codebase) → a gated
  SiLU FFN (`mm.gate`/`mm.up`/`mm.down`, 4096→10944→4096). Added all three stages.

All four fixes verified compiling clean and re-checked against the differentiation suite
(`MultimodalRealWeightsTests`, 11 architectures) — `Glm4V_RealWeights_LoadsAndEmbedsImage` still
passes, same 4 pre-existing unrelated failures as before (`YoutuVl`, `KimiVl`, `MiniCpmV`,
`HunyuanVl`), no new regressions.

**UPDATE (2026-09-02, same day): the reordering turned out to be a non-issue, and GLM-4.6V is now
fully golden-verified with two more real bugs found and fixed.** While tracing the real
position-ID construction for `PROJECTOR_TYPE_GLM4V` in `clip.cpp` (needed to build the golden
Python reference's RoPE), found that GLM4V's real patch/token order is NOT plain row-major:
patches are grouped into 2×2 merge-blocks *before* the transformer runs (real loop: `for y step
merge_ratio { for x step merge_ratio { for dy in 0..1 { for dx in 0..1 } } }`), and position IDs
are assigned per that grouped sequence order, not raster order. Worked through the exact
permute/reshape math in `glm4v.cpp` by hand (ggml_permute + ggml_reshape_4d index algebra) and
confirmed this reorder is purely a MEMORY-LAYOUT convenience for the real code's own
`ggml_conv_2d`-based patch merger (which needs spatially contiguous blocks) — since GLM4V's
attention has no windowing/masking (confirmed earlier), self-attention is fully permutation-
equivariant: reordering all tokens the same way permutes the output identically and changes no
values, as long as (a) RoPE positions are assigned per spatial location rather than per sequence
index, and (b) the merger step gathers the correct spatial neighbors regardless of storage order.
The already-written `ApplyMrope` (indexed by real `px,py`) and the new `ApplyPatchMerger` (which
explicitly computes `srcY,srcX` rather than assuming memory contiguity) already satisfy both
conditions — so the C# encoder needed NO reordering step at all. This reasoning was verified, not
just assumed: built the real golden Python reference (`scripts/glm4v_ref.py`) and ran the parity
test rather than trusting the argument alone.

While building that reference, cross-checking every real tensor name via `list-tensors` surfaced
**two more real, more severe bugs**, on top of the four already fixed above:
- **Q/K/V were being read from tensor names that don't exist in this checkpoint at all.**
  `v.blk.N.attn_q/k/v.weight` are absent — this checkpoint stores a single FUSED
  `v.blk.N.attn_qkv.weight` (out=3×embd). Every one of the 24 layers' `MatVecAny(..., attn_q/k/v,
  ...)` calls was silently a no-op (missing-tensor contract), leaving Q/K/V permanently at their
  zero-initialized buffer contents for the entire forward pass — the real (now-fixed) softmax
  attention from earlier in this pass was mathematically correct but numerically inert the whole
  time, since it was attending over all-zero Q/K/V. Fixed: read the fused tensor once per layer,
  split into Q/K/V via the same three offset slices `ggml_view_4d` uses in the real C++.
- **The second dual patch-embedding conv was fetched but never summed.** Real glm4v.cpp adds TWO
  conv2d outputs (`patch_embeddings_0 + patch_embeddings_1`); `_patchEmbd1W` was being loaded into
  a field and then never referenced anywhere in `ExtractPatches`. Fixed: sum both.
- **Position embeddings were added in the wrong place relative to `norm_embd`.** Real order:
  `+patch_bias → RMSNorm(norm_embd) → +learned_pos_embd` (the position add happens in
  `build_vit`, strictly AFTER the norm). The previous code added `pos_embd` INSIDE the same loop
  as `patch_bias`, before the RMSNorm ran — meaning the position signal was being rescaled/warped
  by the norm on every forward pass instead of added raw afterward. Fixed: moved the pos_embd add
  to after `ApplyRmsNorm`.
- **The 4-section M-RoPE only rotated the first half of `head_dim`, using column position for
  everything.** Derived the real math by hand from `ggml_mrope_cache_init` +
  `GGML_ROPE_TYPE_VISION`'s `rotate_pairs(ne0, n_dims)` call in `ggml-cpu/ops.cpp`: with
  `n_dims=head_dim/2` and 4 equal-size sections, the per-section sector index provably never
  reaches sections 2/3, so only 2 of the 4 declared position channels are ever actually used —
  first quarter of `[0,head_dim/2)` rotates by row/`py`, second quarter by column/`px`, each
  paired with its `+head_dim/2` partner (covering the FULL `head_dim`, not just the first half).
  The previous implementation rotated only `[0,head_dim/2)` (leaving `[head_dim/2,head_dim)`
  completely untouched) and used `px` for every pair, never reading `py` at all. Fixed with the
  derived formula; see `Glm4VisionEncoder.ApplyMrope`'s doc comment for the full derivation.

Built `scripts/glm4v_ref.py` (real numpy port of `glm4v.cpp`'s full forward pass, generated at the
checkpoint's native 336×336/24×24-patch image size so its learned `position_embd`, sized for
exactly 576 positions, applies with no resize) and `Glm4VisionEmbedderParityTests.cs`
(`Forward_MatchesNumpyReference`, same threshold as Pixtral: cosine > 0.97, meanAbs < 5e-2).
**Passed** (21.8s). Re-ran the full differentiation suite afterward: same 4 pre-existing unrelated
failures, no new regressions, full solution builds clean.

GLM-4.6V is now the fourth architecture (after Gemma4UV, Llava, Pixtral) with real golden numeric
verification — six real bugs found and fixed across the two passes on this one architecture,
several of them severe enough (dead attention, dead patch-merger, wrong tensor names entirely)
that the differentiation test's "not degenerate" bar had been passing despite the actual math
being substantially broken. `Exaone4`/`MiMoVl`/`Qwen2.5-VL` remain the only ones still blocked, on
the real windowed-attention gap described above — unlike GLM-4.6V's apparent gap, that one is real
(those checkpoints declare `n_wa_pattern > 0` and their encoders unconditionally run full
attention), not a memory-layout artifact.

---

## Windowed-attention gap closed: Exaone4, MiMoVl, Qwen2.5-VL all golden-verified (2026-09-02)

Implemented real windowed/local attention for the three architectures whose gap was described
above, closing it entirely rather than leaving it deferred. Real mechanism (`tools/mtmd/clip.cpp`'s
shared `PROJECTOR_TYPE_QWEN25VL`/`EXAONE4_5`/`YOUTUVL` case, `tools/mtmd/models/qwen2vl.cpp` +
`exaone4_5.cpp`): layer `il` gets FULL attention only when `(il+1) % n_wa_pattern == 0`; every
other layer only attends within its own `gridWindow x gridWindow` merge-tile spatial window
(`gridWindow = attn_window_size / patch_size / merge_ratio`, real default `attn_window_size=112`
if unset). The real reference computes this via token reordering into contiguous blocks + a dense
mask; implemented it instead by deriving each token's window id directly from its real spatial
(row,col) merge-tile position and masking cross-window scores to `-infinity` before softmax
(`VisionOps.AttentionGqaWindowed`, new) — mathematically identical, no reordering needed, same
technique validated by GLM-4.6V's patch-merger finding above.

Added windowing to `Exaone4VisionEncoder.cs`, `QwenVlVisionEncoder.cs`, and
`MimoVlVisionEncoder.cs`, each gated on real `clip.vision.n_wa_pattern`/`clip.vision.window_size`
metadata (now read into `WindowAttnPattern`/`WindowSize` on all three model classes).

**Along the way, found and fixed the same "M-RoPE only rotates half of head_dim, using the wrong
axis" bug a third and fourth time** (first found and fixed for GLM-4.6V, see above) — it turned out
to also be present in `QwenVlVisionEncoder.ApplyMrope` (its own private copy) AND in the shared
`VisionOps.ApplyMRoPE` helper that `Exaone4VisionEncoder`/`MimoVlVisionEncoder` both call. Same
real derivation applies in all four places (traced from `ggml_mrope_cache_init` +
`GGML_ROPE_TYPE_VISION`'s `rotate_pairs` in `ggml-cpu/ops.cpp`): only 2 of the 4 declared position
channels are ever actually selected, covering the FULL `head_dim` via each index's
`+head_dim/2` partner (first quarter of `[0,head_dim/2)` by row, second quarter by column) — not
two disjoint local quarter-pairs using only column position, as all four previously implemented.

**Additional real bugs found while building golden tests for each architecture** (same
"golden-test-elevation surfaces real bugs the differentiation test can't catch" pattern as
Llava/Pixtral/GLM-4.6V above):
- `QwenVlVisionEncoder`: the second dual-conv patch embedding (`v.patch_embd.weight.1`, confirmed
  present via `list-tensors`) was fetched into an unused field and never summed into the patch
  embedding output — same class of bug as GLM-4.6V's own dual-patch-embed miss.
- `MimoVlVisionEncoder`: real local checkpoints under this class's name only ever store SEPARATE
  `attn_q/k/v` tensors, never a fused `attn_qkv` — but the encoder only ever looked for the fused
  name, so `MatVec`'s "no-op on missing weight" contract silently left Q/K/V at zero for every
  layer the whole time. Added a separate-Q/K/V fallback path (mirrors the one
  `QwenVlVisionEncoder` already had for the reverse case).
- `MimoVlVisionModel`: `head_count_kv` defaulted to a hardcoded `8` (a GQA assumption) when the
  metadata key was absent — but both real local checkpoints under this class's name have NO such
  key at all (confirmed via `list-metadata`) and are plain MHA (`head_count_kv` should equal
  `head_count`, i.e. 16). The wrong default caused K/V projections to read only the first 8
  heads' worth of a weight matrix sized for the full 16, producing garbage K/V. Fixed the default
  to `headCount`.
- `MimoVlVisionEncoder`: post-attention-stack normalization used plain `LayerNorm` (mean-centered)
  instead of `RMSNorm` — a real mismatch against the actual graph these checkpoints run (real
  `qwen2vl.cpp`/`exaone4_5.cpp` use `NORM_TYPE_RMS` uniformly throughout, no mixing).
- `Exaone4VisionEncoder`/`MimoVlVisionEncoder`: both called the shared `VisionOps.ApplyMRoPE` with
  `patchesX`/`patchesY` swapped at the call site (the helper's real parameter order is
  `(q,k,patchesX,patchesY,...)`). Harmless on the square test images used here (row/col symmetric),
  but wrong for real non-square images. Fixed both call sites while already in this code.

Also worth noting: **no local checkpoint for Qwen2.5-VL existed on disk before this pass** —
the earlier README/docs claim that it "passes differentiation" had nothing real behind it locally
(the differentiation test silently returns early when its checkpoint is missing, which looks
identical to "passing" in a bare test-runner summary). Downloaded a real one
(`unsloth/Qwen2.5-VL-7B-Instruct-GGUF`, `mmproj-F16.gguf`, ~1.3GB, confirmed via `list-metadata`:
`n_wa_pattern=8`, `projector_type=qwen2.5vl_merger`) and kept it in `models/` alongside the other
real-weight-suite checkpoints (not a scratch/debug download).

**`MimoVl` real-scope note**: `MimoVlVisionEncoder.cs`'s own doc comment describes the more
elaborate real MIMOVL projector (row/col-banded sliding-window attention sinks, transposed
merge-unit reordering — see `tools/mtmd/models/mimovl.cpp`), but BOTH real local checkpoints under
this class's name are actually `clip.projector_type=qwen2.5vl_merger` (confirmed via
`list-metadata`) — functionally identical to Qwen2.5-VL's own graph, just with plain MHA. This
pass's golden test (`scripts/mimovl_ref.py`) and the windowing/bug fixes above target the graph
these real local checkpoints actually need; the genuine row/col-sink MIMOVL graph remains
unimplemented and untested, since no local checkpoint exercises it.

Added `scripts/exaone4_ref.py`, `scripts/qwen25vl_ref.py`, `scripts/mimovl_ref.py` (real numpy
ports, same methodology as `glm4v_ref.py`) and matching parity tests
(`Exaone4VisionEmbedderParityTests.cs`, `Qwen25VlVisionEmbedderParityTests.cs`,
`MimoVlVisionEmbedderParityTests.cs`). All three pass (cosine > 0.97, meanAbs < 5e-2). Full
differentiation suite re-run clean after every round of fixes, no regressions.

**All 6 of the "confirmed-working" architectures from the original differentiation-suite pass are
now golden-verified**: Gemma4UV, Llava, Pixtral, GLM-4.6V, Exaone4, Qwen2.5-VL, MimoVl (the scoped
graph). Every one of them had at least one real, severe bug (not just cosmetic) that the
differentiation test's "not obviously broken" bar had been passing regardless.

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
