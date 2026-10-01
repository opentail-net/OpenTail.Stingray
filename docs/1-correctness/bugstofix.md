Resolved entries split out to
[done/bugstofix-resolved-2026-08.md](../done/bugstofix-resolved-2026-08.md), 2026-08-15 (most recently
updated 2026-08-27 with the `DeepSeekMoeGraph.cs`/`MlaAttention.cs` column-major layout fixes and
the `IqCodebooks.cs` entry, which had already been marked FIXED but was left here by mistake). This
file keeps only what's still open.

**Closed implementation scope — GLM-4.5 Q5_K×Q8_K:** [archived here](../done/09-glm45-q5k-activation-implementation.md).
The GLM PPL gap is not resolved and the model is not admitted; gate-on real-weight validation is tracked separately as [docs/103 item 19](../103-quickest-first-plan.md).

**Closed investigation moved out** (2026-08-27): the full DeepSeek-V2-Lite MLA/YaRN/MoE-routing
investigation (`ModelCompatibility.cs:460`/`461`, originally logged 2026-08-21) is now at
[done/032-deepseek2-mla-yarn-moe-routing-investigation.md](../done/032-deepseek2-mla-yarn-moe-routing-investigation.md).
tl;dr: multiple real bugs found and fixed along the way (YaRN/kq_scale, expert_weights_norm/scale,
RMSNorm/softmax/SiLU double-precision & ggml-exp fidelity, and — the final entry — a genuine
Q8_0 activation-quantization bug in `MatVecQ8_0` that had silently invalidated the earlier
"native Q8_0" measurement). None of them individually or together produce "Paris" from this
checkpoint+prompt; the router's top-6-of-64 routing decisions are chronically near-tied
(median margin ~0.002) regardless of quantization level (Q2_K through native Q8_0), which is a
property of the trained weights, not of numerical precision. Investigation closed; do not
restart a fourth round of kernel-level chasing on this checkpoint without new evidence.

**Closed item 16 — IQ1_S / IQ1_M / IQ2_XS / IQ2_S coverage verification.** All four formats are
implemented, reference-table verified, admitted, and have valid CPU matvec routes. IQ2_XS/IQ2_S are
also covered by the existing Qwen3.8-27B 24-of-24 exact greedy receipt. IQ1_S/IQ1_M have independent
formula cross-checks but no tractable real-weight receipt; this is an evidence limitation, not an
implementation gap. See the [verification receipt](../done/16-iq-formats-coverage-verification.md),
[GGUF model coverage history](../done/01-gguf-model-coverage-plan.md), and
[CPU implementation record](../done/05-cpu-architecture-kernel-opportunities.md).

## Tracked items

- [x] **01. Real-weight landscape sweep rerun (`docs/103-quickest-first-plan.md` item 2): 649 classes resumed/run; all 22 historical failed class entries now have current dispositions** (2026-09-28 sweep, triage completed 2026-09-30). **Collective plan:** [01 real-weight landscape sweep rerun and failure triage](01-real-weight-landscape-sweep-rerun-plan.md). Closure means every historical red is accounted for; it does not mean every independent correctness investigation is resolved. The detailed matrix records fixed/stale/verified results, item-02/item-07 duplicates, two checkpoint blockers, and deferred reproducible defects.
  - **Caveat on the sweep's own memory data:** the harness resumes from `%TEMP%\stingray-sweep\state.jsonl` and skips any class already marked "done" — the PersonaPlex entries in that file were timestamped `17:57` the same day, from *before* the zero-copy-Q8 fix (see PersonaPlex 7B entry below) landed later in the session, so this run did not re-execute or re-verify them. Its "Top 10 by peak memory" list is stale pre-fix data, not a regression. The PersonaPlex 7B entry's own numbers (11.17 GiB, ~301ms/decode-step) are the current, real ones, independently re-verified by direct class runs, not by this sweep.
  - [x] **Wan layout test failures:** four tests assumed encoder and decoder share one patch-channel order. Updated them to check the Conv3D channel-outer packing and Linear spatial-outer unpacking independently. All four pass; model code is unchanged.
  - [x] **FunASR model identity/lookup:** added architecture-validated Nano (`audiocpp`, `general.name` prefix `Fun-ASR-Nano-2512`) and Paraformer (`paraformer`, `pf.vocab`) checkpoint discovery and applied it to real-weight tests. Both local GGUFs were selected by metadata; Paraformer inspection and Nano encoder golden tests pass. See the collective plan.
  - [x] **FunASR synthetic-tone smoke assertion:** the Paraformer test now checks result structure, language, and duration instead of requiring a segment from a pure tone, which produced no segments with real weights.
  - [x] **FunASR Nano decoder golden — FIXED 2026-10-01:** the synthesized-metadata Nano LLM tensor sources never set `_opentailllm.has_qk_norm`, so `ModelHyperparams.FromGgufMetadata` reported `HasQkNorm=False` and Qwen3's per-head q_norm/k_norm were silently skipped (step-0 logits cos 0.87 vs the audio.cpp reference, repeated token `33108`). Fixed in `FunAsrNanoLlmTensorSource`/`FunAsrNanoLlmGgufTensorSource`; a naive float Qwen3 forward over the same weights had matched the reference (cos 1.0000), isolating the engine-side cause. Second test-side bug: the tied lm_head also scores the appended audio-embedding rows, so argmax/sampling is now restricted to the 151936 text tokens. `FunAsrNanoDecoderGoldenTests` emits `[56568,1773,151645]` exactly; `FunAsrNanoEndToEndTests` now transcribes the real clip to `This little work was finished in the year 1803, and intended for immediate publication.`
  - [ ] **FunASR real-speech transcript (item 07):** the verified Paraformer GGUF and WAV are present. The GGUF returns empty text while the ONNX real-weight path recognizes speech. Stage-by-stage comparison remains in [item 07](07-funasr-gguf-paraformer-real-speech-plan.md); the synthetic-tone test is not ASR evidence.
  - [x] **Qwen ASR smoke test:** the fixture supplies a synthetic tone, not speech, so an empty transcript and segment list are valid. Updated it to assert result structure and duration. Real-speech behavior remains unverified.
  - [x] **MeloTTS fixture integrity:** the complete `_models/melotts-zh_en.onnx` checkpoint is 170,429,550 bytes and loads/generates successfully. The test now skips the empty root placeholder and finds the real checkpoint; hash and direct-run result are in the collective plan.
  - [x] **Parler decoder parity — FIXED 2026-10-01:** the first non-finite (layer 6 self-attention probabilities) was `TensorPrimitives.SoftMax` overflowing to inf/inf = NaN, the same root cause as the 2026-09-05 Chatterbox NaN. `ParlerDecoder` (and the Audiocraft LM kernels, same pattern) now use the max-subtracted `DenseKernels.SoftmaxInPlace`. `ParlerDecoderTests`: F32 control cosine `0.9999999865`, Q8_0 cosine `0.99988` (gate `>0.99` unchanged); Parler Gguf/KvCache/DelayPattern classes pass.
  - [x] **Qwen TTS Code Predictor historical red:** non-finite output does not reproduce with the identity-checked `_models` Talker checkpoint; current class passes 1/1. See the collective plan for path/hash.
  - [ ] **Exaone4 vision embedder parity (deferred):** the main checkpoint exists at `K:\_other_models\EXAONE-4.5-33B-Q4_K_M.gguf`. Fresh MTMD reference and C# Rainbow448 aggregate parity pass, but the NumPy fixture still fails (minimum token cosine `0.825366`); real-image preprocessing/reference reconciliation remains open. See the updated result and SHA-256 in the collective plan.
  - [ ] **Llava vision embedder parity (asset-blocked):** NumPy fixture minimum per-token cosine `-0.21`; matching LLaVA 1.5 7B main checkpoint is absent. The available LLaVA 3 model is not a substitute.
  - [ ] **MimoVL vision embedder parity (deferred):** NumPy fixture minimum per-token cosine `0.249`; current C# aggregate agrees with a fresh MTMD reference within the existing gate. Per-token and real-input preprocessing reconciliation remains open.
  - [ ] **Qwen2.5-VL vision embedder parity (deferred):** NumPy fixture minimum per-token cosine `0.283`; current C# aggregate agrees with a fresh MTMD reference within the existing gate. Per-token and real-input preprocessing reconciliation remains open.
  - [x] **Continuous batching historical mismatch:** the exact method was `PrefillWithCache_DequantCacheOnOff_BitIdentical`, and belongs to item 02. Current class passes 12/12 after fixing async test mutex ownership; historical/current evidence is in the collective plan.
  - [x] **Exaone45 long-prompt parity fixture:** corrected stale expected token count from `196` to `157`; verified with llama.cpp tokenizer and reran long-prompt generation parity successfully (2026-09-30).
  - [x] **Dequant-cache prefill parity:** fixed by disabling the F32 diversion that changed inference math. The exact cache-on/off regression passes with OpenBLAS loaded; see item 02 for the related prefill/decode failure.
  - **Already tracked separately:** `FishSpeechCodecTests`/`FishSpeechFastArTests` (see the Fish Speech entry below). Full sweep logs/XML remain under `%TEMP%\stingray-sweep\logs\<Suite>\<Class>.log`.

- [x] **02. `PrefillDecodeSelfConsistencyTests.F32Prefill_MatchesTokenByTokenDecode` — fixed 2026-09-30** (found 2026-09-28 while re-verifying the PersonaPlex zero-copy-Q8 memory fix).
  - **Failure:** `dotnet test tests/OpenTail.Stingray.Tests.ForwardPass.exe -class OpenTail.Stingray.Tests.ForwardPass.PrefillDecodeSelfConsistencyTests` fails even run alone, single-process, with `STINGRAY_RUN_HEAVY_TESTS=1` — not a concurrency artifact from the mutex/sweep work done the same day.
  - **Context:** pins the invariant that whole-prompt `Prefill(t0..tN)` agrees with `Prefill(t0)` followed by `Forward(t1..tN)` token-by-token, on `SmolLM2-1.7B-Instruct-Q4_K_M.gguf`.
  - **Root cause:** the default dequant cache diverted prefill to F32 matmul whenever OpenBLAS was loaded, even with the Q8 gate disabled; decode continued through quantized MatVec. Disabling the numerically different cache route restored same-kernel parity. All three F32 cases (2, 8, and 33 tokens) pass with OpenBLAS loaded.
- [ ] **03. `docs/sweep-tests-memory-report.md`'s per-test "why" narratives were mostly noise** (found 2026-09-28).
  - **What happened:** the memory sweep report attributed the top-10 memory ranking's 7 PersonaPlex entries to per-test specifics (Mimi codec held in both directions, KV-cache growth, frame count, voice-prompt extraction). The real, shared driver was `PersonaPlexLmTensorSource` eagerly dequantizing the whole 7B Q8_0 checkpoint to fp32 (~28 GB) on every test — fixed same day (zero-copy Q8_0 views, ~11.17 GiB, ~2.2x faster decode too; see `docs/done/audio-review-new-progress.md`'s PersonaPlex 7B entry, `PerformanceLeague.md`).
  - **Why it matters:** the report's rank-by-rank analysis is still useful for the non-PersonaPlex entries (`HunyuanVideoGpuParityTests`, `ZImageGpuRealScaleBisectTests`, `Sd3PerStepTrajectoryParityTests` — genuine double-load-pattern costs), but its 7 PersonaPlex explanations should not be trusted as the real cause without re-deriving them against the now-much-lower baseline.
  - **Next step:** none required — informational, so a future reader of that report doesn't take its PersonaPlex reasoning at face value.
- [x] **04. Granite 4.0 3B Vision (`granite4-vision`) outputs empty decode / early EOS — CLOSED 2026-10-01** (found 2026-09-28 during RUNNING.md verification).
  - **Failure:** Running `stingray -m models/_models/granite-4.0-3b-vision-Q4_K_M.gguf --mmproj models/_models/mmproj-granite-4.0-3b-vision-f16.gguf --image photo.png -p "Describe this picture."` terminated after 0 to 3 tokens (e.g. single period) instead of generating text.
  - **Root cause:** Batched deepstack prefill is 100% numerically sound (tested against per-token: top-1 greedy token identical, top-10 logits within 0.09). The early EOS was caused by defective GGUF metadata Jinja template in the IBM checkpoint, which (1) unconditionally injected a default system message (`"You are a helpful assistant..."`) before user turn even when none was requested, and (2) indented user content by 8 spaces in `render_content(x)`. This forced system prompt causes the Granite 4.0 Vision decoder to shift top-1 logit directly to `<|end_of_text|>` at token 0. Confirmed against `llama-mtmd-cli`: when run with `--jinja`, llama.cpp reproduces the exact same bug (`: \n\n this picture.`), whereas its default ignores Jinja and uses hardcoded `LLM_CHAT_TEMPLATE_GRANITE_4_0` which emits no default system prompt and no 8 spaces.
  - **Resolution:** Added canonical Granite template routing (`<|start_of_role|>user<|end_of_role|>{userMessage}<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>`) to `RunCommand.FormatPrompt` and `ChatTemplateRenderer`, matching `LLM_CHAT_TEMPLATE_GRANITE_4_0` and bypassing the broken metadata Jinja.
  - **Verification:** Unit tests in `ServerLibraryTests` (`Fallback_Granite_EmitsCanonicalGraniteFraming`, `Granite_BypassesBrokenJinjaTemplate_EmitsCanonicalFormat`) and `Granite4VisionTests`. End-to-end regression test in `Granite4VisionE2ETests` passes against real weights and sample image with zero early EOS. Real CLI verification on `flux_apple_vulkan_512_4steps.png` prefilled 737 tokens at 104.9 t/s and decoded 50+ grounded tokens at 11.8 t/s (`"The picture depicts a red apple, set on a warm wooden surface..."`). RUNNING.md updated.
  - **Investigation plan:** [Granite 4.0 3B Vision early-EOS plan](04-granite4-vision-early-eos-plan.md).
- [x] **05. Voxtral Realtime audio.cpp GGUF is not supported by the dedicated STT entry point — CLOSED 2026-09-30 (Q8_0 GGUF support verified and hardened; Q4_K explicitly left as future coverage rather than unfinished correctness work)** (found 2026-09-28 during RUNNING.md verification).
  - **Expected behavior, not a defect:** generic `stingray -m <Voxtral GGUF> -p "..."` rejects `general.architecture=audiocpp` before inference. Maintained unchanged; regression-verified via `VoxtralGgufInventoryTests.GenericTextGeneration_RejectsAudiocppArchitecture`.
  - **Resolution & Hardening:** Added `VoxtralPipeline.LoadGguf(string ggufPath)` and extended `VoxtralPipeline.Load(string pathOrDir)` to detect `.gguf` files. Extracted generic `AudioCppPackedTensorSource` (with `RvcPackedTensorSource` inheriting for backwards compatibility) so Voxtral carries no RVC-specific dependencies. Hardened package validation: `VoxtralAudioEncoderWeights` and `VoxtralTextDecoderWeights` validate `DType.Q8_0` across all 460 raw weight matrices via `GetRawBytes(name, DType.Q8_0)`. Hardened embedded files validation in `VoxtralPipeline.LoadGguf` to verify monotonic non-decreasing offsets, valid bounds in `audiocpp.embedded_files.data`, and lengths (`offsets.Length == names.Length || names.Length + 1`). Embedded `tekken.json` is validated and loaded directly from `audiocpp.embedded_files.*` via `TekkenVocab.Load(ReadOnlySpan<byte>)`. Updated `SttCommand.cs` (`ResolveVoxtralPath`) to route `.gguf` checkpoints to the native Voxtral ASR pipeline.
  - **Verification:** Both programmatic (`VoxtralPipelineEndToEndTests.Transcribe_RealAudio_Gguf_MatchesReferenceTranscript`) and CLI routing integration (`SttCommandRoutingTests.SttCommand_ExecutesVoxtralGguf_Successfully`) reproduce the exact golden reference transcript: `"This little work was finished in the year 1803, and intended for immediate publication."`. Full package validation checks in `VoxtralGgufInventoryTests` verify all 460 raw matrices, byte counts, and embedded bounds. SafeTensors baseline test continues to pass identically. Q4_K is left as a future format coverage extension.
- [ ] **06. GGUF `jais` architecture rejected; admission landed but output is degenerate — real blocker is ALiBi, not an allowlist gap** (found 2026-09-28 during RUNNING.md verification; investigated further same day, in-progress).
  - **Failure (original):** `stingray -m "models/_models/jais-family-590m-chat.Q4_K_M.gguf"` threw `NotSupportedException`: GGUF architecture 'jais' is not supported for text generation (supported list includes `jais2`, but `jais` v1 was unmapped).
  - **Checkpoint:** `mradermacher/jais-family-590m-chat-GGUF`, Q4_K_M (`K:\_other_models\jais-family-590m-chat.Q4_K_M.gguf`, 480 MiB — was missing from every model dir on this machine, which is what stalled the first attempt at this; downloaded via `stingray pull` and kept on K: per this project's disk-space rule, not `models/_models`).
  - **Structural diff confirmed against the real reference** (`examples/llama.cpp/llama.cpp/src/models/jais.cpp` vs `jais2.cpp`): fused `attn_qkv` (vs jais2's separate wq/wk/wv), gated-SiLU-with-bias FFN (vs jais2's non-gated ReLU²), separate untied `output.weight` (vs jais2's tied embeddings), and — the one real gap — **ALiBi position encoding, not RoPE** (`jais.attention.max_alibi_bias` metadata key; `jais.cpp` never calls `inp_pos`/`ggml_rope_ext`). Everything except ALiBi was already generic in this engine (fused-QKV split, `HasFfnBias`, `HasAttnOutputBias`, `HasNormBias`/`UsesLayerNorm` — all tensor-presence-detected, same as gptneox/falcon/codeshell).
  - **Correction to an earlier claim in this investigation:** MPT was cited as "also ALiBi, already admitted" to justify shipping without ALiBi. Checked directly — **MPT is not in `ModelCompatibility.cs`'s allowlist at all** (only appears in `PreTokenizerPatterns.cs`'s tokenizer-pretype table, unrelated to architecture admission). There is no existing ALiBi precedent anywhere in this engine to lean on; it would be genuinely new.
  - **Admission landed (`ModelGraph.cs` sets `noRopeStep=1` for `arch=="jais"`; `ModelCompatibility.cs` allowlists `"jais"`), but real-weight verification shows it doesn't work:** running the actual checkpoint from K: produces degenerate, incoherent output — repeated-token loops (`ex ex ex`, `temporarily temporarily temporarily`, `endendendenden`) mixing English/Arabic word salad, not the "coherent but positionally degraded" text the admission's doc comment predicted. This is stronger evidence of a real missing feature, not a cosmetic gap — **do not check this item off or trust the doc comment's optimistic framing until ALiBi is actually implemented and re-verified.**
  - **What ALiBi actually needs:** a per-attention-head linear bias added to each raw QK score before softmax (`score[i][j] += slope[head] * (j - i)` for causal, roughly; slopes are a fixed geometric sequence derived from `numHeads` and `max_alibi_bias`, see ggml's `ggml_soft_max_ext`/`ggml_alibi` or `jais.cpp`'s own call for the exact formula — don't guess it from memory, read the real reference). **Concrete call sites in this engine that would each need it** (found via `ForwardPass.Attention.cs`): the score computation immediately before every `SimdKernels.SoftmaxInPlace(...)` call — there are at least 3 separate paths (a plain per-token path around line 37/40, a batched-prefill path around line 271/290, and a windowed/KV-shared path around line 417/428) that all independently compute `headScores[i] = ... * scale` then softmax; all three need the bias added at the same point, or a shared helper needs to grow an optional bias parameter used by all of them. `_hp.AttentionScaleOverride`/`_layerHeadDim` near line 112 show where per-model attention-scale plumbing already exists, so ALiBi's per-head slope likely belongs alongside it as a new `ModelHyperparams` field.
  - **Decision needed before implementing:** ALiBi is real, new engine work (attention math, not just a tensor-mapping/allowlist change), and jais v1 is a small, low-priority 590M checkpoint. Worth confirming with the user whether this is worth doing before investing in it, versus leaving the entry open/parked.
- [x] **07. FunASR GGUF Paraformer pipeline returns an empty transcript on real speech — FIXED 2026-10-01** (found 2026-09-27, `docs/103` item 10).
  - **Root causes (three, stacked):** (1) `fsmn_block.weight` is stored `ne=[C,11]` (channel contiguous, index `k*C+ch`) but `FunAsrKernels.FsmnDepthwiseConv` and the python golden both read `ch*11+k` — scrambled FSMN made the residual stream blow up to ~3e5 and the encoder output nearly constant, so CIF alphas collapsed to `[1,0,0,...]`; (2) the encoder input lacked funasr's `x*sqrt(512)` + `SinusoidalPositionEncoder` (`FunAsrEncoder.PrepareInput`, from `funasr/models/sanm/encoder.py:409` / `transformer/embedding.py`); (3) `TensorPrimitives.SoftMax` overflowed to NaN on large scores (now max-subtracted). The stage goldens passed because the python oracle shared bug (1) and fed the encoder directly.
  - **Evidence:** `FunAsrRealSpeechTests.Paraformer_GgufRealSpeech_ProducesNonEmptyMandarinTranscript` — 30 tokens, text identical to the ONNX control; encoder/predictor/decoder/weights goldens regenerated (`scratch-llamacpp-ref/funasr_golden_*.py`, gitignored, fsmn reshape fixed, model path `models/_models/`) and pass at multi-second runtimes.
  - **Original report:**
  - **Failure:** `FunAsrPipeline.Load("models/_models/paraformer-q8.gguf")` on
    `docs/audio-samples/paraformer-zh-test-0.wav` (real Mandarin) produces `''`.
  - **Control:** the ONNX Paraformer on the same clip gives
    "对我做了介绍啊那么我想说的是呢大家如果对我的研究感兴趣呢嗯", so the audio is fine.
  - **Why it looked healthy:** its only end-to-end test feeds a 440 Hz tone, where empty output is
    expected. The stage goldens (encoder/adaptor/decoder) pass on their own.
  - **Investigation plan:** [FunASR GGUF Paraformer real-speech plan](07-funasr-gguf-paraformer-real-speech-plan.md) — first eliminate the wrong-checkpoint lookup trap, then compare real-WAV stages against the production reference path and stop at the first divergence.
  - **Related trap:** `models/paraformer-q8.gguf` (1.0 GB) is not a Paraformer. Its metadata says
    `general.name = Fun-ASR-Nano-2512`, `general.architecture = audiocpp`. The real Paraformer GGUF is
    `models/_models/paraformer-q8.gguf` (237 MB). Tests that look up "paraformer-q8.gguf" in
    `models/` first load the wrong model; rename or move the Nano file.
  - **Lookup/identity fix (2026-09-30):** the Paraformer encoder, predictor, decoder, mel-extractor, weights, and perf tests select `models/_models/paraformer-q8.gguf` by validated GGUF architecture and `pf.vocab`; Nano-specific tests select the root Nano GGUF. Both Paraformer GGUF and WAV are present. `FunAsrPipeline.Architecture` reports the loaded GGUF architecture and the real-weight test asserts `paraformer`. The empty-transcript inference issue remains open for stage-by-stage diagnosis; it is not asset-blocked.
- [ ] **08. Fish Speech S2 Pro correctness-test reconciliation: stale codec oracle and Q4 Fast-AR precision** (found 2026-09-27 by the `docs/103` item 2 landscape sweep).
  - **Observed test failures:**
    - `FishSpeechCodecTests.Decode_RealWeights_MatchesGoldenPcmOutput`: PCM cosine 0.052 vs the old golden.
    - `FishSpeechFastArTests.Forward_RealWeights_MatchesGoldenOracle`: fast-AR logits cosine 0.44 vs the full-precision golden using `models/s2-pro-q4_k_m.gguf`.
  - **The history does not support `d377049` as the origin of these failures:** `7a68185` is an ancestor of `d377049`, and its commit message already records the Fast-AR cosine 0.44 failure as pre-existing. `bd2a612` added the real `quantizer.post_module` transformer; its message says the codec golden predates and omits that transformer. With identical reference-generated codes, the corrected C# codec reached 0.9999999 cosine against the reference. `a7e720` subsequently describes the remaining suite failures as the Q4_K_M Fast-AR precision limitation and stale codec oracle, and reports successful real end-to-end listening verification.
  - **Fast-AR test scope:** `FishSpeechWeights` normalizes Fast-AR weights to Q8_0 at load time. The Q4 checkpoint path therefore includes Q4_K_M dequantization followed by Q8_0 quantization; a 0.44 cosine against original full-precision logits is not alone proof of incorrect math. Existing Q8_0 external-golden and `ForwardStep_MatchesForward_ForSamePrefix` tests are useful controls, but test different claims.
  - **Next step:** follow [`08-fish-speech-s2-pro-golden-reconciliation-plan.md`](08-fish-speech-s2-pro-golden-reconciliation-plan.md): classify current vs `7a68185`, regenerate the codec golden with the full post-module reference path, establish precision-appropriate Q8_0/Q4 Fast-AR tests, and inspect individual `d377049` changes only if a valid comparison demonstrates a regression.
  - **New evidence 2026-10-01 (Fast-AR):** the same golden input through `FishSpeechFastAr.Forward` gives cos **0.9971** vs the full-precision golden with `s2-pro-q8_0.gguf` (a real run: the Q8_0 file is in `models/_models/`) but **0.4406** with `s2-pro-q4_k_m.gguf`; Q4 vs Q8 outputs agree at only 0.506 (argmax 497 vs 324, golden 324), even though every compared Fast-AR tensor dequantizes to cos >= 0.9973 (Q4_K) / 0.9998 (Q6_K) against its Q8_0 twin. So the Q4_K_M checkpoint, not the Fast-AR math, is what diverges; the Q8_0 golden is the correctness gate and the Q4 golden is a quantization-sensitivity measurement (not a defect signal). The codec golden still needs the real 8-layer `post_module` transformer in its oracle, which needs the original PyTorch checkpoint (not on this machine).
  - **Status:** retain the conservative 🟡 rating until permanent, correctly scoped regression coverage is in place. The numeric golden failures are test/oracle issues to reconcile, not established evidence of a current end-to-end audio defect.
- [ ] **10. GLM-4.7-Flash (`deepseek2`) perplexity 0.9-1.4% worse than llama.cpp; not at parity** (logged 2026-09-27; `docs/103-quickest-first-plan.md` item 3).
  - **Checkpoint:** `GLM-4.7-Flash-Q2_K.gguf` (10.6 GB).
  - **Result:** wikitext second-half PPL at -c 2048: ours 8.1757 batched prefill, 8.2100 sequential;
    `llama-perplexity --chunks 1` 8.0997.
  - **Pattern:** same diffuse PPL-gap pattern as the GLM-4.5 investigation (implementation record
    [archived here](../done/09-glm45-q5k-activation-implementation.md)). The generation is coherent
    and the top-k order matches; sequential is worse than batched, so the batched prefill path is
    not the cause.
  - **Next step:** same layer bisection as GLM-4.5, but only trust differences well above the
    4-decimal print resolution of `llama-eval-callback`.
- [ ] **11. LFM2 (`lfm2`) perplexity 0.24% worse than llama.cpp** (logged 2026-09-27; `docs/103-quickest-first-plan.md` item 11a; timeboxed out).
  - **Checkpoint:** `LFM2-1.2B-Q8_0.gguf`. PPL 10.9543 vs `llama-perplexity` 10.9277. Admitted anyway
    (greedy and teacher-forced parity tests pass).
  - **Layer bisection:** 338-token wikitext prompt (with BOS), last token, `llama-eval-callback` vs
    `StageCapture` (the short-conv and Mamba-2 mixers now record `o_proj`).
    - Layer 0 (short conv): `operator_norm` exact, `conv.out_proj` and `l_out` differ by 1e-4,
      which is the callback's print resolution, so layer 0 matches as far as it can be seen.
    - From layer 2 the differences are real (about 5e-4 to 1.6e-3 absolute) and grow gradually;
      no single layer jumps.
  - **Ruled out:** the Q8_0 activation scheme. Our `MatVecQ8_0` already quantises activations to
    32-element Q8_0 blocks as ggml does; storing the block scale as fp16 like ggml (tried) changed
    nothing.
  - **Remaining suspects:** float summation order in attention softmax/accumulation over long
    context; the attention layers' K/V storage precision (llama.cpp's default F16 KV cache vs
    ours); the dump's resolution hides where the error starts. A full-precision dump (tensor
    binary output, not the printed summary) is needed to go further.
  - **Investigation plan:** [LFM2 PPL gap resolution plan](11-lfm2-ppl-gap-plan.md) — freezes the canonical benchmark first, then continues the measured parity investigation.
- [x] **Parakeet (CTC and TDT) held its weights as F32** (logged and resolved 2026-09-27, docs/103 item 14).
  - Was 2.85 GB for the 378 MB q4_k TDT file. Now Q4_K weights stay quantized (repacked Q4Kx8 batched GEMM, as the
    text engine's prefill), only the Q8_0 conv layers are packed F32: 1.1 GB peak, and faster than CrispASR on the
    same files (TDT 1.16x, CTC 1.10x). CrispASR itself peaks at 0.6-0.65 GB; the ~290 MB of F32 conv weights are the
    difference, kept on purpose because the int8 Q8_0 path was ~0.2 s slower per clip. Stopped here.
- [ ] **12. LFM2-MoE (`lfm2moe`) not admitted; reference parity still open** (logged 2026-09-28, docs/103 item 13): the
  earlier Q8-on measurements were per-token 8.1860 vs batched 8.9595 at `-c 512`, and batched 15.8076 vs llama.cpp
  14.8639 at `-c 2048`. After global CPU Q8 became opt-in, the fresh Q4_K_M baseline showed a smaller but real batched
  drift: with Q8 off and OpenBLAS excluded, `[256,1024)` PPL at `-c 1024` was 7.5275 batched vs 7.6135 per-token, and
  `[1024,+)` at `-c 2048` was 14.3531 vs 14.2242. Chunk widths 16/64/256 changed all NLLs in `[256,1024)`.
  - **Isolation/fix, 2026-10-01:** On the actual 256-token WikiText prefix, conv layers 0/1 matched exactly; the first
    batched-vs-token difference was attention layer 2 at position 1 (max delta 0.0002494), growing to max 1.3575 by
    layer 23. LFM2-MoE recurrent batching is now opt-in via `STINGRAY_LFM2_MOE_BATCHED_PREFILL=1`; unset defaults to
    the sequential trunk. Batched CLI runs at `-c 1024` and `-c 2048` then had NLL dumps exactly equal to token-by-token
    (1,023/2,047 targets, zero changed values; PPL `[256,1024)` 7.6135 and `[1024,+)` 14.2242).
  - **Coverage:** `Lfm2MoeBatchedPrefillParityTests` compares every full-vocabulary logit on a real WikiText prefix and
    asserts bit equality against token-by-token execution. Release build and real-weight test passed 2026-10-01.
  - **Still open:** reconcile PPL against llama.cpp using identical tokens/evaluation semantics, explain the experimental
    attention difference, and admit `lfm2moe` only after that reference check. See [12-LFM2-MoE batched/per-token
    parity plan](12-lfm2moe-batched-per-token-parity-plan.md).
- [ ] **13. Granite 4.0-H small (MoE) PPL parity** (logged 2026-09-28, docs/103 item 13): `granite-4.0-h-small-Q2_K`,
  wikitext -c 2048 `[1024,+)` 26.4155 (old Q8 batched default) / 26.5483 (per token) vs
  `llama-perplexity --chunks 1` 26.1080; at -c 512 9.3505 vs 9.4103 (ours lower). The large error (157) was missing
  top-k renormalisation, fixed. Investigation found Q8 activation quantization in the batched MoE expert-down path caused
  all batched/per-token NLL differences; CPU Q8 prefill now defaults off so the batched path matches per-token NLL. The
  remaining reference gap and Q2_K-only local coverage remain open.
  - **Part 1 — parity:** [13-Granite 4.0-H small MoE PPL parity](13-granite4-h-small-moe-ppl-parity-plan.md) — default-off
    verification completed 2026-10-01: unset Q8 overrides produced NLLs identical to per-token at all
    2,047 targets (max delta 0), `[1024,+)` PPL 26.5483. Still compare against llama.cpp and add the
    same-model Q4_K_M receipt when available.
  - **Part 2 — MoE Q8 evaluation:** Measure batched MoE Q8 quality and throughput on Granite 4 H Small and representative
    MoE models, compare identical per-token NLL/logit outputs and corpus PPL, then decide whether any model/weight-dtype
    cases justify opting in. `STINGRAY_CPU_PREFILL_Q8=1` opts into general CPU Q8 prefill; MoE additionally requires
    `STINGRAY_MOE_PREFILL_Q8=1`. Exact numerical parity remains the default.
- [ ] **14. Qwen3-VL / Qwen2.5-VL / PaddleOCR image input: CUDA and Vulkan hybrid still lack it** (logged 2026-09-27, docs/103 item 14).
  - **Plan:** [14-Qwen VL GPU image-input parity](14-qwenvl-gpu-image-input-parity-plan.md) — verifies full Vulkan per model, adds CUDA M-RoPE/deepstack and Vulkan layer-split support, and gates CLI routing on forward-pass capability.
  - 2026-09-28: full Vulkan offload (`GpuForwardPass`) now applies per-pair M-RoPE positions and deepstack, which also covers
    the IMROPE pairs 61-62 for text (M-RoPE models take the per-token trunk). Verified by `Qwen3VlVulkanMRopeParityTests`
    (cosine 0.9995 vs CPU). The CLI now refuses image input for M-RoPE models on any other pass instead of answering wrongly.
    Remaining: CUDA (`CudaForwardPass`, no CUDA GPU here), Vulkan hybrid / layer split.
  - M-RoPE image positions (`ForwardPass.AddMRopeImage`, IMROPE for qwen3vl) and deepstack slices are applied in the
    CPU `ForwardPass` only. The CUDA and Vulkan forward passes would rotate image tokens with 1D positions and skip
    deepstack, so their image answers would be wrong. Text-only use on GPU is fine (text positions are 1D).
  - Also: the GPU rope tables for `qwen3vl` do not zero pairs 61-62 (the IMROPE 4th component), which the CPU table
    now does; that needs checking before GPU text use of `qwen3vl` is called verified.
- [x] **`AceStepPrecomputeSilenceTests` rewrites checked-in files** (found and gated 2026-09-28: now skips unless `STINGRAY_ACESTEP_PRECOMPUTE=1`): running it overwrites
  `src/OpenTail.Stingray.Diffusion/AceStep/silence_latent.bin` / `silence_timbre.bin` and the runtime copies in
  `models/acestep-v15/` (which `AceStepPipeline` reads first). It is a generator; make it opt-in (env gate) or write to a
  scratch path so ordinary test runs cannot change shipped data.
- [ ] **15. LLaVA-NeXT / LLaVA-1.6 AnyRes tiling (llava_uhd) remains unverified with real weights** (logged 2026-09-27):
  - Classic LLaVA-1.5 (single-tile 336x336 ViT + MLP projector) is verified against `llama-mtmd-debug` and `llama-mtmd-cli`
    end to end (`LlamaMtmdVisionParityTests.Llava15_Rainbow336_MatchesLlamaMtmdDebug`).
  - However, dynamic AnyRes multi-tile slicing (`llava_uhd`) is only verified for Granite Vision, not on a LLaVA-NeXT
    or LLaVA-OneVision checkpoint. Needs an actual LLaVA-NeXT checkpoint to verify tile ordering and separators.
  - **Plan:** [15-LLaVA-NeXT / OneVision AnyRes parity](15-llava-next-anyres-parity-plan.md) — validate checkpoint metadata, preprocessing and view composition against llama.cpp, then prove prompt expansion and end-to-end real-weight parity before changing code.
- [ ] **19. RWKV6 CPU coverage**: No WKV6 kernel or RWKV6 model graph exists. Add this only against a concrete model target; require a scalar WKV6 oracle, recurrent-state/reset tests, and real-checkpoint parity before admission. Current source audit: [GGML op coverage verification](17-ggml-op-coverage-verification-plan.md).
- [ ] **20. RWKV7 CPU coverage**: No WKV7 kernel or RWKV7 model graph exists. Track independently from RWKV6 because the recurrence differs; require its own scalar oracle, recurrent-state/reset tests, and real-checkpoint parity before admission. Current source audit: [GGML op coverage verification](17-ggml-op-coverage-verification-plan.md).
- [ ] **21. Generic `SOLVE_TRI` primitive**: A generic lower-triangular solve is absent. The Gated DeltaNet chunk path has only a specialized solve for its derived system. Implement a reusable primitive when there is a concrete consumer; validate lower-triangular multi-RHS behavior against an independent forward-substitution oracle. Current source audit: [GGML op coverage verification](17-ggml-op-coverage-verification-plan.md).
- [x] **18. SpeculativeDecoder.cs StepSampled/PLD bugs**: Resolved 2026-09-30. Sampled PLD now samples the bonus from the last verified row when proposals are shorter than lookahead, tracks accepted-token metrics, appends accepted proposals to lookup history, and synchronizes the optional draft cache with accepted tokens. Regressions cover short proposals, accepted-token metrics, and target/draft cache-position equality in `SpeculativeCascadeTests`.
- [x] **DeepSeekMoeGraph.cs:172 ExpertOffsets off-by-index**: Resolved 2026-08-27 (moved to `docs/done/bugstofix-resolved-2026-08.md`).
- [x] **KvMemoryGovernor TOCTOU race & InferenceSession unguarded Fork() on CUDA**: Moot / resolved 2026-08-27 when superseded session types were removed.

```json
[
  {
    "file": "src/OpenTail.Stingray.Cpu/Dequantize.cs",
    "line": 0,
    "summary": "Drift check requested 2026-08-21: diffed examples/ggml/src/ggml-common.h's quant tables against ours to look for newer formats/moved tables since our IqCodebooks.cs/Dequantize.cs were derived. RESULT: MXFP4/NVFP4 (kvalues_fp4/kvalues_mxfp4) and IQ4_NL/IQ2_XXS/IQ3_XXS/IQ3_S (kvalues_iq4nl/iq2xxs_grid/iq3xxs_grid/iq3s_grid) match ggml's current literals exactly -- no drift, no fix needed there. HOWEVER: ggml-common.h also defines iq2xs_grid (512 entries, for IQ2_XS), iq2s_grid (1024 entries, for IQ2_S), and iq1s_grid (NGRID_IQ1S=2048 entries, for IQ1_S/IQ1_M) -- none of which exist anywhere in IqCodebooks.cs or Dequantize.cs. This isn't drift, it's a format that was never ported: DType.IQ1_S/IQ1_M/IQ2_XS/IQ2_S are declared in the enum (Tensor.cs) but Dequantize.ToFloat32 has no case for any of them (falls to the `default: throw NotSupportedException`), and ModelCompatibility.cs's IsSupportedWeightDType allowlist correctly does not admit them either -- so this is a real coverage gap, not a live correctness bug (nothing can load a GGUF using these dtypes today).",
    "failure_scenario": "Currently none -- these four dtypes are unreachable (no allowlist entry, and the decoder would throw NotSupportedException rather than silently produce wrong output, unlike the previous IQ2_XXS/IQ3_XXS/IQ3_S/IQ4_XS bug this session fixed). Becomes relevant only if a future GGUF is admitted with IQ1_S/IQ1_M/IQ2_XS/IQ2_S weights: would need iq1s_grid/iq2xs_grid/iq2s_grid copied verbatim from ggml-common.h into IqCodebooks.cs plus new decoders in Dequantize.cs (IQ1_S/IQ1_M also need IQ1S_DELTA=0.125f and a sign/shift scheme distinct from the IQ2/3 family), following the same real-table-not-fabricated-formula approach as this session's IQ2_XXS/IQ3_XXS/IQ3_S/IQ4_XS fix."
  },
  {
    "file": "src/OpenTail.Stingray.Engine/ModelCompatibility.cs",
    "line": 0,
    "summary": "Op-list drift check requested 2026-08-21: diffed ggml.h's `enum ggml_op` (examples/ggml/include/ggml.h) against what this engine actually implements. This engine already covers the ops it needs for its admitted architectures: GGML_OP_GATED_DELTAN_ET/SSM_CONV (GdnKernels.cs), GGML_OP_TOP_K (used by MoE routing), GGML_OP_FLASH_ATTN_EXT-equivalent attention, standard RMS_NORM/ROPE/MUL_MAT/GLU/SOFT_MAX paths, etc. Found ONE real, load-bearing gap: GGML_OP_SSM_SCAN (the actual Mamba/Mamba2 state-space recurrence -- distinct from SSM_CONV, which this engine does implement) has NO implementation anywhere (`grep -ri mamba|ssm_scan src/` is empty), and no Mamba/Mamba2/Jamba/Zamba/FalconMamba/Codestral-Mamba architecture string appears in ModelCompatibility.cs's allowlist at all -- consistent (not admitted), not a silent bug, but a real missing capability if any pure-SSM or hybrid-SSM architecture is ever wanted. Also unimplemented and unadmitted: GGML_OP_RWKV_WKV6/RWKV_WKV7 (RWKV6/7 architectures), GGML_OP_LIGHTNING_INDEXER/DSV4_HC_COMB/DSV4_HC_PRE/DSV4_HC_POST (DeepSeek-V4-generation ops -- newer than this session's deepseek2/V2-Lite investigation, ggml added these very recently), GGML_OP_SOLVE_TRI, and GGML_OP_WIN_PART/WIN_UNPART (Swin-style windowed vision attention -- OpenTail.Stingray.Vision covers Gemma3/Gemma4/Llama4 today, not window-attention vision transformers).",
    "failure_scenario": "None today -- every one of these is correctly un-admitted, so no architecture silently produces wrong output via a missing op; requests for these architectures simply fail model-compatibility checks, which is the intended fail-closed behavior. This is purely a capability/future-proofing note: Mamba-family (hybrid or pure SSM) support specifically would need a real GGML_OP_SSM_SCAN kernel (the actual selective-scan recurrence, not just SSM_CONV's causal conv1d prelude that already exists) added to SimdKernels/GdnKernels plus a new ForwardPass graph and ModelCompatibility allowlist entries -- a substantial new architecture family, not a small fix. RWKV6/7 and DeepSeek-V4's DSV4_HC_*/LIGHTNING_INDEXER ops would each similarly require their own dedicated kernel + graph work. None of this blocks anything currently supported."
  }
]
```

**Cut for the cap but still real and worth a look**: `SpeculativeDecoder.cs` StepSampled/PLD bugs — confirmed as a real defect but currently unreachable/latent, no wired call path exercises it yet.

**Historical, now moot** (both files were deleted 2026-08-27 when `InferenceSession`/`InferenceRuntime` and their superseded predecessors were removed — see `docs/030-delete-inferencesession-todo.md`): the `KvMemoryGovernor` TOCTOU race and the `InferenceSession` double-release/unguarded-`Fork()`-on-CUDA findings no longer apply to anything in the codebase.

**Resolved 2026-08-27** (moved to `docs/done/bugstofix-resolved-2026-08.md`): the `DeepSeekMoeGraph.cs:172` `ExpertOffsets[MaxExperts]` off-by-index this note used to flag.
