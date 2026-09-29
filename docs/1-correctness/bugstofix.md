Resolved entries split out to
[done/bugstofix-resolved-2026-08.md](../done/bugstofix-resolved-2026-08.md), 2026-08-15 (most recently
updated 2026-08-27 with the `DeepSeekMoeGraph.cs`/`MlaAttention.cs` column-major layout fixes and
the `IqCodebooks.cs` entry, which had already been marked FIXED but was left here by mistake). This
file keeps only what's still open.

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

## Tracked items

- [ ] **01. Real-weight landscape sweep rerun (`docs/103-quickest-first-plan.md` item 2): 649 classes resumed/run, several new real failures found** (2026-09-28, `scripts/sweep-tests.ps1`, default `-Suites Diffusion,Audio,Vision,ForwardPass`).
  - **Caveat on the sweep's own memory data:** the harness resumes from `%TEMP%\stingray-sweep\state.jsonl` and skips any class already marked "done" — the PersonaPlex entries in that file were timestamped `17:57` the same day, from *before* the zero-copy-Q8 fix (see PersonaPlex 7B entry below) landed later in the session, so this run did not re-execute or re-verify them. Its "Top 10 by peak memory" list is stale pre-fix data, not a regression. The PersonaPlex 7B entry's own numbers (11.17 GiB, ~301ms/decode-step) are the current, real ones, independently re-verified by direct class runs, not by this sweep.
  - **`WanTests` (Diffusion), exit=1, failed=4:** `Assert.Equal() Failure: Values are not within tolerance 9.99999997E-07` (several facts) plus one `Values differ`. Not investigated further.
  - **FunASR cluster, 8 classes, all the SAME root cause:** `FunAsrEncoderTests`, `FunAsrWeightsTests` (3 facts), `FunAsrPredictorTests`, `FunAsrRealDecoderTests`, `FunAsrRealMelExtractorTests`, `FunAsrNanoDecoderGoldenTests`, `Fast.FunAsrPerfBenchTests`, `Fast.FunAsrRealWeightsTests` — all throw `System.IO.InvalidDataException: Paraformer GGUF missing 'pf.vocab' metadata`. This is almost certainly the SAME wrong-file trap already logged in the "FunASR GGUF Paraformer pipeline returns an empty transcript" entry below: `models/paraformer-q8.gguf` (1.0 GB, `general.name=Fun-ASR-Nano-2512`, `general.architecture=audiocpp`) is not a Paraformer; the real one is `models/_models/paraformer-q8.gguf` (237 MB). These tests are presumably finding the wrong file by the same bare-filename lookup. Fixing the rename/move already proposed in that entry should very likely clear all 8 at once — not independently confirmed.
  - **`Fast.QwenAsrRealWeightsTests`:** `Assert.NotEmpty() Failure: Collection was empty` — same empty-transcript shape as the FunASR entry, different engine (Qwen3-ASR). Not investigated whether it shares a cause.
  - **`Fast.MeloTtsRealWeightsTests`:** `MeloTTS ONNX model file must be > 50MB` — the on-disk `melotts-*.onnx` this machine has is presumably a stub/partial file, not a real-weight issue.
  - **`ParlerDecoderTests`:** `cosine similarity 0.9896934984794502 too low vs golden Parler decoder output` — close (0.99) but outside tolerance; possibly a tolerance-too-tight issue rather than a real bug, not determined.
  - **`QwenTtsCodePredictorForwardPassTests`:** `Code Predictor logits contained a non-finite value` (NaN/Inf) — a real numerical bug, not investigated.
  - **Vision embedder parity, 4 classes, all `Forward_MatchesNumpyReference`, all "min per-token cosine too low":** `Exaone4VisionEmbedderParityTests` (0.825), `LlavaVisionEmbedderParityTests` (**-0.21**, negative — much worse than the others), `MimoVlVisionEmbedderParityTests` (0.249), `Qwen25VlVisionEmbedderParityTests` (0.283). Llava's negative cosine and the others' well-below-threshold scores suggest a real per-architecture vision embedder bug each, not a shared tolerance issue — not investigated further.
  - **`ContinuousBatchingTests`, `Exaone45GreedyParityTests` (ForwardPass):** both `Assert.Equal() Failure: Values differ`, no detail captured yet — logs at `%TEMP%\stingray-sweep\logs\ForwardPass\`.
  - **Already tracked, not new:** `FishSpeechCodecTests`/`FishSpeechFastArTests` (see the Fish Speech entry below) and `PrefillDecodeSelfConsistencyTests` (see its own entry above).
  - **Next step:** none of the new failures above have been root-caused; this entry is the raw triage from the sweep, following the standing rule to document precisely and move to the next item rather than block the whole session on one. Full logs/xUnit XML per class are in `%TEMP%\stingray-sweep\logs\<Suite>\<Class>.log`.

- [ ] **02. `PrefillDecodeSelfConsistencyTests.F32Prefill_MatchesTokenByTokenDecode` fails standalone** (found 2026-09-28 while re-verifying the PersonaPlex zero-copy-Q8 memory fix; unrelated to that change).
  - **Failure:** `dotnet test tests/OpenTail.Stingray.Tests.ForwardPass.exe -class OpenTail.Stingray.Tests.ForwardPass.PrefillDecodeSelfConsistencyTests` fails even run alone, single-process, with `STINGRAY_RUN_HEAVY_TESTS=1` — not a concurrency artifact from the mutex/sweep work done the same day.
  - **Context:** pins the invariant that whole-prompt `Prefill(t0..tN)` agrees with `Prefill(t0)` followed by `Forward(t1..tN)` token-by-token, on `SmolLM2-1.7B-Instruct-Q4_K_M.gguf`.
  - **Next step:** not investigated yet — needs a look at whether prefill's batched Q8 activation path (`SimdKernels.MatMulBatched` with `allowQ8`) has drifted from the per-token `MatVec` decode path.
- [ ] **03. `docs/sweep-tests-memory-report.md`'s per-test "why" narratives were mostly noise** (found 2026-09-28).
  - **What happened:** the memory sweep report attributed the top-10 memory ranking's 7 PersonaPlex entries to per-test specifics (Mimi codec held in both directions, KV-cache growth, frame count, voice-prompt extraction). The real, shared driver was `PersonaPlexLmTensorSource` eagerly dequantizing the whole 7B Q8_0 checkpoint to fp32 (~28 GB) on every test — fixed same day (zero-copy Q8_0 views, ~11.17 GiB, ~2.2x faster decode too; see `docs/done/audio-review-new-progress.md`'s PersonaPlex 7B entry, `PerformanceLeague.md`).
  - **Why it matters:** the report's rank-by-rank analysis is still useful for the non-PersonaPlex entries (`HunyuanVideoGpuParityTests`, `ZImageGpuRealScaleBisectTests`, `Sd3PerStepTrajectoryParityTests` — genuine double-load-pattern costs), but its 7 PersonaPlex explanations should not be trusted as the real cause without re-deriving them against the now-much-lower baseline.
  - **Next step:** none required — informational, so a future reader of that report doesn't take its PersonaPlex reasoning at face value.
- [ ] **04. Granite 4.0 3B Vision (`granite4-vision`) outputs empty decode / early EOS** (found 2026-09-28 during RUNNING.md verification).
  - **Failure:** Running `stingray -m models/_models/granite-4.0-3b-vision-Q4_K_M.gguf --mmproj models/_models/mmproj-granite-4.0-3b-vision-f16.gguf --image photo.png -p "Describe this picture."` projects 145 soft tokens (20480-dim across 8 deepstack streams) successfully at 44.3 t/s, but generation terminates after 0 to 3 tokens (e.g. single period) instead of generating text.
  - **Suspect:** Multi-stream deepstack injection mapping into the text model's layers or chat prompt template delimiter handling.
  - **Control:** `granite-vision-3.2-2b` (which uses standard MLP projector) generates full, rich descriptions without issue.
  - **Investigation plan:** [Granite 4.0 3B Vision early-EOS plan](04-granite4-vision-early-eos-plan.md) — reproduces the failure and traces the image-embedding path through the text model to the first divergent output.
- [ ] **05. Voxtral Realtime audio.cpp GGUF is not supported by the dedicated STT entry point** (found 2026-09-28 during RUNNING.md verification).
  - **Expected behavior, not a defect:** generic `stingray -m <Voxtral GGUF> -p "..."` rejects `general.architecture=audiocpp` before inference. Do not add `audiocpp` to the generic text-generation architecture allowlist.
  - **Actual gap:** `SttCommand -m voxtral --model-file <GGUF>` currently resolves a SafeTensors directory, while `VoxtralPipeline.Load` requires `model.safetensors` and filesystem `tekken.json`. The dedicated Voxtral ASR path does not yet load the self-contained audio.cpp-packed GGUF.
  - **Checkpoint:** `mistral-experimental/AudioCPP-Voxtral-Mini-4B-Realtime-2602-GGUF`; use the exact local GGUF as source of truth for metadata, packed tensors, dtypes, and embedded files.
  - **Plan:** [Voxtral Realtime audio.cpp GGUF support](05-voxtral-realtime-audiocpp-gguf-support-plan.md) — inventory the real package, bridge it to the existing Voxtral weight/inference path, route `stt` by checkpoint format, and retain the generic text-generation rejection.
- [ ] **06. GGUF `jais` architecture rejected; admission landed but output is degenerate — real blocker is ALiBi, not an allowlist gap** (found 2026-09-28 during RUNNING.md verification; investigated further same day, in-progress).
  - **Failure (original):** `stingray -m "models/_models/jais-family-590m-chat.Q4_K_M.gguf"` threw `NotSupportedException`: GGUF architecture 'jais' is not supported for text generation (supported list includes `jais2`, but `jais` v1 was unmapped).
  - **Checkpoint:** `mradermacher/jais-family-590m-chat-GGUF`, Q4_K_M (`K:\_other_models\jais-family-590m-chat.Q4_K_M.gguf`, 480 MiB — was missing from every model dir on this machine, which is what stalled the first attempt at this; downloaded via `stingray pull` and kept on K: per this project's disk-space rule, not `models/_models`).
  - **Structural diff confirmed against the real reference** (`examples/llama.cpp/llama.cpp/src/models/jais.cpp` vs `jais2.cpp`): fused `attn_qkv` (vs jais2's separate wq/wk/wv), gated-SiLU-with-bias FFN (vs jais2's non-gated ReLU²), separate untied `output.weight` (vs jais2's tied embeddings), and — the one real gap — **ALiBi position encoding, not RoPE** (`jais.attention.max_alibi_bias` metadata key; `jais.cpp` never calls `inp_pos`/`ggml_rope_ext`). Everything except ALiBi was already generic in this engine (fused-QKV split, `HasFfnBias`, `HasAttnOutputBias`, `HasNormBias`/`UsesLayerNorm` — all tensor-presence-detected, same as gptneox/falcon/codeshell).
  - **Correction to an earlier claim in this investigation:** MPT was cited as "also ALiBi, already admitted" to justify shipping without ALiBi. Checked directly — **MPT is not in `ModelCompatibility.cs`'s allowlist at all** (only appears in `PreTokenizerPatterns.cs`'s tokenizer-pretype table, unrelated to architecture admission). There is no existing ALiBi precedent anywhere in this engine to lean on; it would be genuinely new.
  - **Admission landed (`ModelGraph.cs` sets `noRopeStep=1` for `arch=="jais"`; `ModelCompatibility.cs` allowlists `"jais"`), but real-weight verification shows it doesn't work:** running the actual checkpoint from K: produces degenerate, incoherent output — repeated-token loops (`ex ex ex`, `temporarily temporarily temporarily`, `endendendenden`) mixing English/Arabic word salad, not the "coherent but positionally degraded" text the admission's doc comment predicted. This is stronger evidence of a real missing feature, not a cosmetic gap — **do not check this item off or trust the doc comment's optimistic framing until ALiBi is actually implemented and re-verified.**
  - **What ALiBi actually needs:** a per-attention-head linear bias added to each raw QK score before softmax (`score[i][j] += slope[head] * (j - i)` for causal, roughly; slopes are a fixed geometric sequence derived from `numHeads` and `max_alibi_bias`, see ggml's `ggml_soft_max_ext`/`ggml_alibi` or `jais.cpp`'s own call for the exact formula — don't guess it from memory, read the real reference). **Concrete call sites in this engine that would each need it** (found via `ForwardPass.Attention.cs`): the score computation immediately before every `SimdKernels.SoftmaxInPlace(...)` call — there are at least 3 separate paths (a plain per-token path around line 37/40, a batched-prefill path around line 271/290, and a windowed/KV-shared path around line 417/428) that all independently compute `headScores[i] = ... * scale` then softmax; all three need the bias added at the same point, or a shared helper needs to grow an optional bias parameter used by all of them. `_hp.AttentionScaleOverride`/`_layerHeadDim` near line 112 show where per-model attention-scale plumbing already exists, so ALiBi's per-head slope likely belongs alongside it as a new `ModelHyperparams` field.
  - **Decision needed before implementing:** ALiBi is real, new engine work (attention math, not just a tensor-mapping/allowlist change), and jais v1 is a small, low-priority 590M checkpoint. Worth confirming with the user whether this is worth doing before investing in it, versus leaving the entry open/parked.
- [ ] **07. FunASR GGUF Paraformer pipeline returns an empty transcript on real speech** (found 2026-09-27, `docs/103` item 10).
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
- [ ] **08. Fish Speech S2 Pro correctness-test reconciliation: stale codec oracle and Q4 Fast-AR precision** (found 2026-09-27 by the `docs/103` item 2 landscape sweep).
  - **Observed test failures:**
    - `FishSpeechCodecTests.Decode_RealWeights_MatchesGoldenPcmOutput`: PCM cosine 0.052 vs the old golden.
    - `FishSpeechFastArTests.Forward_RealWeights_MatchesGoldenOracle`: fast-AR logits cosine 0.44 vs the full-precision golden using `models/s2-pro-q4_k_m.gguf`.
  - **The history does not support `d377049` as the origin of these failures:** `7a68185` is an ancestor of `d377049`, and its commit message already records the Fast-AR cosine 0.44 failure as pre-existing. `bd2a612` added the real `quantizer.post_module` transformer; its message says the codec golden predates and omits that transformer. With identical reference-generated codes, the corrected C# codec reached 0.9999999 cosine against the reference. `a7e720` subsequently describes the remaining suite failures as the Q4_K_M Fast-AR precision limitation and stale codec oracle, and reports successful real end-to-end listening verification.
  - **Fast-AR test scope:** `FishSpeechWeights` normalizes Fast-AR weights to Q8_0 at load time. The Q4 checkpoint path therefore includes Q4_K_M dequantization followed by Q8_0 quantization; a 0.44 cosine against original full-precision logits is not alone proof of incorrect math. Existing Q8_0 external-golden and `ForwardStep_MatchesForward_ForSamePrefix` tests are useful controls, but test different claims.
  - **Next step:** follow [`08-fish-speech-s2-pro-golden-reconciliation-plan.md`](08-fish-speech-s2-pro-golden-reconciliation-plan.md): classify current vs `7a68185`, regenerate the codec golden with the full post-module reference path, establish precision-appropriate Q8_0/Q4 Fast-AR tests, and inspect individual `d377049` changes only if a valid comparison demonstrates a regression.
  - **Status:** retain the conservative 🟡 rating until permanent, correctly scoped regression coverage is in place. The numeric golden failures are test/oracle issues to reconcile, not established evidence of a current end-to-end audio defect.
- [ ] **09. GLM-4.5 (`glm4moe`) perplexity 1.9% worse than llama.cpp; not admitted** (logged 2026-09-27; `docs/done/102-status-open-items-plan.md` #16).
  - **Checkpoint:** `cerebras_GLM-4.5-Air-REAP-82B-A12B-Q2_K.gguf`. It mixes quant types: attn_q
    Q2_K, attn_output Q5_K, expert gate/up Q2_K, expert down IQ4_NL.
  - **Result:** wikitext second-half PPL at -c 2048 is 8.7753 vs `llama-perplexity` 8.6125.
  - **Already fixed:** a tokenizer bug (`glm4` pre-type unmapped, commit 4349db3) that made it 37%.
  - **Ruled out:**
    - tokenisation: a 20 KB sample matches `llama-tokenize` exactly;
    - RoPE: the partial NeoX table is built over `rope.dimension_count`=64;
    - batched vs sequential MoE prefill: identical PPL;
    - IQ4_NL dequant and codebook: identical to ggml.
  - **Symptom:** the error grows with position. Next-token log-probs vs `llama-server` given the same
    ids differ by about 0.01 nats at 5 tokens, about 0.1 at 23, and up to 0.85 at 326, with the same
    top-5 order. So look at what accumulates over context: attention over many keys, or
    sigmoid+bias routing drifting as the hidden state drifts.
  - **Next step:** compare per-layer hidden states for the last token of the 326-token wikitext
    prompt (first 1400 bytes of `scripts/kvarn-gate/wiki.test.raw`).
    - llama.cpp side: `examples/llama.cpp/llama.cpp/build-eval/bin/llama-eval-callback.exe`, tensors
      `ffn_inp-N` / `l_out-N`.
    - Our side: `StageCapture` stages `post_attn_resid` / `post_ffn_resid`.
    - Find the first layer whose last-token values drift.
  - **Memory:** llama.cpp needs about 46 GB for this file, so run it alone.
  - **Layer bisection done 2026-09-27:** last token of the 326-token prompt.
    - llama.cpp `llama-eval-callback` `attn_norm/ffn_inp/l_out-N` vs our `StageCapture`, via the
      scratch harness `ZzLayerDumpTmp` (untracked).
    - **No single broken layer.** `attn_norm-0` matches exactly (0.0000). The first difference, about
      0.3%, appears after layer 0's attention. It then grows steadily: about 1% by layer 5, 2-5% by
      layer 15, 5-10% in layers 20-45.
    - **Leading hypothesis:** activation-quantisation scheme differences in every layer's
      attention/FFN matmuls, not a logic bug:
      - our Q5_K matvec keeps activations F32 where ggml quantises them to Q8_K (GLM's `attn_output`
        is Q5_K in every layer);
      - our batched prefill quantises activations per row, not ggml's per-block Q8_K, for the 325
        cached K/V positions.
    - **Next experiment:** dump the `attn_out` stage too (llama.cpp `kqv_out-0`) to split attention
      from `wo`; then make the decode path use ggml's per-dtype activation scheme (Q5_K -> Q8_K) and
      re-measure the layer-0 difference and the PPL.
    - **Caveat found 2026-09-27 (LFM2 bisection):** `llama-eval-callback` prints values to 4
      decimals, so a "0.0001" difference on values near 0.02 is print resolution, not drift. The
      GLM "0.3% after layer 0's attention" may be the same artefact; only differences well above
      1e-4 absolute count.
    - **Layer-0 `attn_out`/`wo` split done, 2026-09-28** (the next experiment named above): compared
      `kqv_out-0` (pre-`wo`, raw attention output) and `node_26`/`ffn_inp-0` (post-`wo`) between
      `llama-eval-callback` and our own `StageCapture` `attn_out`/`o_proj` stages, same real
      checkpoint (`K:\_other_models\cerebras_GLM-4.5-Air-REAP-82B-A12B-Q2_K.gguf`), same 326-token
      wikitext prompt, last-token row (position 325). Method: our own `ZzLayerDumpTmp.cs` (now also
      captures `attn_out`) + a new `ZzMakeGlmIdsTmp.cs` scratch harness to tokenize the exact same
      first-1400-bytes prompt with our own tokenizer (already confirmed matching `llama-tokenize`
      exactly).
      - `kqv_out-0`/`attn_out` (raw attention, pre-`wo`): **exact match** — `...0.0009, 0.0004,
        0.0026` (ref) vs `...0.0009, 0.0004, 0.0027` (ours), a last-digit difference consistent with
        print rounding, not drift.
      - `node_26`/`o_proj` (post-`wo`, the FIRST Q5_K-quantized matmul in the whole model):
        `-0.0045, 0.0024, -0.0140, ..., -0.0029, 0.0092, 0.0037` (ref) vs `-0.0047, 0.0024, -0.0139,
        ..., -0.0030, 0.0093, 0.0038` (ours) — a ~2e-4 difference, small but above the LFM2 caveat's
        1e-4 noise floor.
      - **Conclusion: the divergence's exact origin is now pinned to layer 0's `wo` projection
        specifically, not attention math in general.** RoPE, the QK score computation, and the
        softmax are all now ruled out (their output, `kqv_out`, matches). This is the first
        Q5_K-quantized matmul in the entire model, directly supporting the standing hypothesis
        (ggml quantizes activations to Q8_K for this matmul; we keep them F32) rather than
        something specific to attention.
      - **Not yet done:** the actual fix (making the decode path's Q5_K matvec use ggml's per-block
        Q8_K activation quantization instead of F32) has NOT been implemented or attempted. This is
        real new engineering work on a widely-shared kernel path (`SimdKernels`'s Q5_K matvec is used
        by every Q5_K-quantized model in this codebase, not just GLM), so it needs care and a broad
        regression check before landing, not a quick patch. Scoped but not started.
      - **Full implementation plan (2026-09-28), ready to hand off:**
        [docs/1-correctness/09-glm45-q5k-activation-quant-plan.md](09-glm45-q5k-activation-quant-plan.md) —
        exact function to add (`DotQ5K_Q8K`, mirroring the existing `DotQ8_0_Q8K` shape), the real
        ggml reference to port from, the feature-gate wiring, the verification steps (isolated
        kernel test → re-run the layer-0 bisection harness already built for this → full PPL →
        broad Q5_K regression pass), and the constraints to follow. Start there instead of
        re-deriving the approach from this entry.
- [ ] **10. GLM-4.7-Flash (`deepseek2`) perplexity 0.9-1.4% worse than llama.cpp; not at parity** (logged 2026-09-27; `docs/103-quickest-first-plan.md` item 3).
  - **Checkpoint:** `GLM-4.7-Flash-Q2_K.gguf` (10.6 GB).
  - **Result:** wikitext second-half PPL at -c 2048: ours 8.1757 batched prefill, 8.2100 sequential;
    `llama-perplexity --chunks 1` 8.0997.
  - **Pattern:** same as the GLM-4.5 entry above. The generation is coherent and the top-k order
    matches; the gap is diffuse, not one broken op. Sequential is worse than batched, so the batched
    prefill path is not the cause.
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
- [ ] **12. LFM2-MoE (`lfm2moe`) not admitted: PPL off and per-token vs batched disagree** (logged 2026-09-28, docs/103 item 13):
  `LFM2-8B-A1B-Q4_K_M`, wikitext `[256,1024)` at -c 512: per token 8.1860, batched 8.9595, `llama-perplexity --chunks 1`
  8.7030; at -c 2048 batched 15.8076 vs 14.8639 (+6.3%). Arch wiring is in `ModelGraph` (NeoX, short-conv layers,
  sigmoid gating with `exp_probs_b` as DeepSeek, top-k renormalised as llama.cpp lfm2.cpp norm_w = true) but the arch is
  NOT in the allowlist. Ruled out: parallel expert execution and parallel routing (serial gives identical numbers), and
  the expert matmul kernels (batched vs per-row MatVec on the real blk.2 Q4_K/Q6_K expert weights: relative error 0 at
  n = 1..64). Dense LFM2 1.2B and Granite hybrids agree per-token vs batched within 0.1-0.5%, so the 9% spread is specific
  to this model. Next step: per-layer hidden-state comparison, batched vs per-token vs llama-eval-callback.
- [ ] **13. Granite 4.0-H small (MoE) +1.2% PPL vs llama.cpp** (logged 2026-09-28, docs/103 item 13): `granite-4.0-h-small-Q2_K`,
  wikitext -c 2048 `[1024,+)` 26.4155 (batched) / 26.5483 (per token) vs `llama-perplexity --chunks 1` 26.1080; at -c 512
  9.3505 vs 9.4103 (ours lower). The large error (157) was missing top-k renormalisation, fixed. Not yet bisected; Q2_K
  only locally (a Q4 file would separate quantisation noise from a real difference).
- [ ] **14. Qwen3-VL / Qwen2.5-VL / PaddleOCR image input: CUDA and Vulkan hybrid still lack it** (logged 2026-09-27, docs/103 item 14).
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
- [ ] **16. Dequantize.cs / IqCodebooks.cs coverage gap**: Port `iq1s_grid` (NGRID_IQ1S=2048) and decoders for `IQ1_S`/`IQ1_M` (`IQ1S_DELTA=0.125f`, distinct sign/shift scheme) and `IQ2_XS`/`IQ2_S` when needed by future GGUF models.
- [ ] **17. ModelCompatibility.cs / Kernels missing op coverage** (2026-09-27: the Mamba-2 `SSM_SCAN`/`SSM_CONV` path is now implemented on CPU in `ForwardPass.Mamba2.cs` for Granite 4.0-H / Nemotron-H; Mamba-1 and GPU remain): Implement `GGML_OP_SSM_SCAN` (the selective-scan recurrence, distinct from `SSM_CONV`), `RWKV_WKV6`/`RWKV_WKV7`, and DeepSeek-V4 ops (`LIGHTNING_INDEXER`, `DSV4_HC_*`, `SOLVE_TRI`, `WIN_PART`/`WIN_UNPART`).
- [ ] **18. SpeculativeDecoder.cs StepSampled/PLD bugs**: Confirmed real defect in speculative decode step sampling; currently unreachable/latent as no wired call path exercises it yet.
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
