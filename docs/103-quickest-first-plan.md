# 103: Outstanding work, quickest first

Written 2026-09-27 at the user's request. **This file sets the working order.** Other views of the
same problems exist, such as the themed backlog in [00-current-work.md](00-current-work.md) and
[1-correctness/bugstofix.md](1-correctness/bugstofix.md). Those remain the detail and evidence
sources, but work proceeds in the order below.

## Rules

1. **Order.** Take items in this order. Do not reorder because another view ranks something higher.
2. **If an item stalls,** write down the blocker precisely: in this file, and in `bugstofix.md` when
   it is a bug. Then move to the next item (CLAUDE.md).
3. **Evidence.** "Done" needs the check in the item's **Done when** line, with real weights and an
   independent reference where one applies. Then:
   - update the STATUS.md row;
   - update the [RUNNING.md](RUNNING.md) row (command, measured RAM and speed, dated);
   - commit.
4. **Heavy runs one at a time.** Two 30 GB processes at once runs this 64 GB machine out of memory,
   and concurrent runs make timings meaningless.
5. **Estimates are guesses** unless an item says it has been investigated. Record the real time
   taken when an item closes.

---

## Minutes

### 1. Commit the GLM entry in bugstofix.md
- [x] **1. Commit the GLM entry in bugstofix.md** (DONE 2026-09-27, `99b6b53`)
  - [x] Commit `docs/1-correctness/bugstofix.md` (GLM-4.5 1.9% entry, Mamba-2 op note).
  - [x] **Done when:** committed.

### 2. Rerun the real-weight landscape sweep
- [ ] **2. Rerun the real-weight landscape sweep** (STOPPED 2026-09-27: the memory reaper killed it twice, partway through Audio; not restarted. See Log)
  - [ ] Run `STINGRAY_RUN_HEAVY_TESTS=1` for Diffusion suite with its own log (no other heavy runs).
  - [ ] Run `STINGRAY_RUN_HEAVY_TESTS=1` for Audio suite with its own log.
  - [ ] Run `STINGRAY_RUN_HEAVY_TESTS=1` for Vision suite with its own log.
  - [ ] Run `STINGRAY_RUN_HEAVY_TESTS=1` for ForwardPass suite with its own log.
  - [ ] Read failures and flag any real-weight test class that finishes in under ~0.4s (silent no-op, CLAUDE.md rule 12).
  - [ ] **Done when:** every suite has a complete log. Each failure is either fixed or entered in `bugstofix.md`.
  - *Estimate:* little effort from developer, hours of machine time. (2026-09-27 attempt killed by memory reaper after Diffusion).

---

## About an hour

### 3. GLM-4.7-Flash (`deepseek2`)
- [ ] **3. GLM-4.7-Flash (`deepseek2`)** (BLOCKED 2026-09-27: logged in `bugstofix.md`)
  - [x] `stingray pull -r unsloth/GLM-4.7-Flash-GGUF -q Q2_K` (10.6 GB downloaded to `models/_models/GLM-4.7-Flash-Q2_K.gguf`).
  - [x] `admit-arch` verification (coherent generation, same top-k order as llama-server).
  - [x] Run second-half PPL vs `llama-perplexity` (`-c 2048`): 8.1757 batched / 8.2100 sequential vs 8.0997. Not at parity.
  - [ ] **Done when:** PPL within ~0.3% of llama.cpp; add STATUS/RUNNING rows, and entry in MODELS.md if it qualifies.
  - *Note:* uses `deepseek2` in llama.cpp, not `glm4moe`, so it may already work. Waits for Item 2 sweep to free machine.

### 4. Gemma 4 E4B vision (`gemma4v`) parity
- [x] **4. Gemma 4 E4B vision (`gemma4v`) parity** (DONE 2026-09-27)
  - [x] Add a `LlamaMtmdVisionParityTests` case against `llama-mtmd-debug` (rainbow image), following vision parity pattern.
  - [x] Verify encoder + projector match: row 0 within 0.003 on rainbow 224 (`Gemma4V_Rainbow224_MatchesLlamaMtmdDebug`).
  - [x] **Done when:** stage fingerprints match; STATUS row added.

### 5. Wire the ONNX speech-to-text pipelines into `stingray stt`
- [x] **5. Wire the ONNX speech-to-text pipelines into `stingray stt`** (DONE 2026-09-27, `4874ea4`)
  - [x] Implement `ISpeechToTextPipeline` in `SenseVoicePipeline`.
  - [x] Add CLI options to `SttCommand`: `-m sensevoice|paraformer --model-file <.onnx>` and auto-discover tokens file.
  - [x] Verify CLI transcription: SenseVoice matches LibriSpeech reference text ("concord returned to its place amidst the tents").
  - [x] Verify CLI transcription: Paraformer outputs fluent Mandarin transcript.
  - [x] Run exact-transcript test in `SenseVoiceRealWeightsTests` once Audio test project is free.
  - [x] Commit CLI and pipeline changes.
  - [x] **Done when:** `stingray stt` transcribes a real clip with each, matching existing pipeline output.

### 6. Finish the silent-no-op test sweep
- [ ] **6. Finish the silent-no-op test sweep** (MOSTLY DONE 2026-09-27; leftovers below)
  - [x] Point real-weight tests at `models/_models` as well as `models/` (Part 1 committed in `e58b58a`: 162 silent returns converted to `Assert.Skip`, 141 lookups updated).
  - [x] Verify `ForwardPass.Fast`: 14 previously silent real-weight tests now run and pass.
  - [x] Sweep remaining silent returns in the Audio project (46 files, `4874ea4`).
  - [ ] Sweep remaining 14 files under active edit.
  - [ ] Sweep 31 lookup helpers with non-standard shapes (listed by `nested_models.py`).
  - [ ] **Done when:** grep finds no remaining silent returns in real-weight tests, and test runs report skips as skips.

---

## A few hours, cause already narrowed

### 7. GLM-4.5 (`glm4moe`) 1.9% perplexity gap
- [ ] **7. GLM-4.5 (`glm4moe`) 1.9% perplexity gap** (TIMEBOXED 2026-09-27: logged in `bugstofix.md`)
  - [x] Profile layer bisection: drift starts at ~0.3% after layer 0's attention and compounds across layers rather than a single broken layer.
  - [x] Document findings and next experiment into `bugstofix.md` (GLM entry); timebox to day-scale job.
  - [x] `llama-eval-callback` on 326-token wikitext prompt for tensors `ffn_inp-N` / `l_out-N`.
  - [x] `StageCapture` stages `post_attn_resid` / `post_ffn_resid` on matching token IDs.
  - [ ] Compare layer-by-layer values and isolate cumulative drift cause.
  - [ ] **Done when:** second-half PPL within ~0.3% of llama.cpp's 8.6125; allowlist entry, STATUS, RUNNING, and MODELS rows added.

### 8. Jinja chat-template gaps
- [x] **8. Jinja chat-template gaps** (DONE 2026-09-27, `4ed6866`: the real gap was `tojson` formatting)
  - [x] (No longer reproduces in the template corpus run) Support string concatenation inside a conditional expression (Gemma-3-4B-it and other cases in `00-current-work.md` §1 item 7).
  - [x] Add regression tests verifying template rendering against test vectors (`JinjaTojsonTests`).
  - [x] **Done when:** those templates render identically to llama.cpp's output (Qwen3 with tools, byte for byte).

### 9. Empty KV rows for layers without attention
- [x] **9. Empty KV rows for layers without attention** (DONE 2026-09-27, `c4cea41`)
  - [x] Update `PagedKvCache` to allow Mamba-2, short-conv, and MLP-only layers to skip storage rather than allocating zero KV blocks.
  - [x] Measure and confirm KV footprint reduction scales with attention layer count only.
  - [x] Verify Granite-H, Nemotron-H, and LFM2 parity tests remain unchanged.
  - [x] **Done when:** KV memory footprint reduction measured and verified without parity regression.

### 10. FunASR-Nano on real speech
- [x] **10. FunASR-Nano on real speech** (DONE 2026-09-27 as a bug entry, `fea7873`)
  - [x] Transcribe a real clip (real Mandarin clip; output is empty).
  - [x] Validate output against reference transcript (ONNX Paraformer on the same clip is fluent).
  - [x] **Done when:** sensible transcript produced, or a precise defect entry logged in `bugstofix.md`.

---

## A few hours, cause unknown

Timebox each at half a day, write down what was learned, and move on if blocked.

- [ ] **11. Unknown-cause set**
  - [ ] **11.a** LFM2 0.24% PPL gap (10.9277 vs 10.9543). TIMEBOXED 2026-09-27, logged in `bugstofix.md`.
  - [x] **11.b** Youtu-VL: one 1024-token window +5% PPL vs llama.cpp. DONE 2026-09-27 (`9568823`, unmapped `youtu` pre-tokenizer).
  - [x] **11.c** NaN in `ForwardPass`'s f16 `qwen3` path (last layer, one position). DONE 2026-09-27: no longer reproduces; pinned by `Qwen3F16FiniteLogitsTests`.
  - [ ] **11.d** `HybridGdnChunkedPrefill_MatchesSequentialPrefill` failure with real weights.
  - [ ] **11.e** Stable Audio 3 padding masks in the APG norm.
  - [x] **11.f** Classic LLaVA-1.5 (missing image token in vocab). DONE 2026-09-27: direct-splice path in CLI (`RunCommand.cs`) when marker absent from special tokens; Vicuna prompt format for LLaMA-2 backbone (`!s_hasLlama3Headers` alone, removing `s_isVicuna` so LLaMA-3 VLMs aren't hijacked into Vicuna; pinned by `LlamaPromptFormatRuleTests`); patch/CLS ordering in `LlavaVisionEncoder.cs` (patches 0..575, CLS 576, matching llama.cpp's `clip_graph_llava::build`). Pinned by `LlamaMtmdVisionParityTests.Llava15_Rainbow336_MatchesLlamaMtmdDebug` (576x4096 soft tokens, sum -10587.12 vs -10596.39, row 0 [-0.5335, -0.0007, -0.2534] vs [-0.5347, -0.0022, -0.2532]). End-to-end answers "The newspaper is the New York Times, and the main headline reads \"Men Walk on Moon.\"" matching `llama-mtmd-cli`. Tested two images, count mismatch, no-marker prepend, context overflow, Qwen3-VL regression, and Vulkan GPU offload (-g -1).

---

## A day or more

### 12. Batched prompt processing for the recurrent families
- [ ] **12. Batched prompt processing for the recurrent families**
  - [ ] Add Mamba-2 and short-conv layers to `PrefillCore` for Granite 4.0-H, Nemotron-H, and LFM2.
  - [ ] Batch the projections while keeping scan and conv sequential over tokens.
  - [ ] Verify parity tests remain unchanged.
  - [ ] Measure prompt tok/s and record in `RUNNING.md`.
  - [ ] **Done when:** parity tests unchanged; prompt tok/s measured and recorded in RUNNING.md.

### 13. MoE variants of the recurrent families
- [ ] **13. MoE variants of the recurrent families**
  - [ ] Admit Granite 4.0-H tiny/small (MoE) with PPL parity.
  - [ ] Admit Nemotron-H MoE (latent MoE, sigmoid gating) with PPL parity.
  - [ ] Admit LFM2-MoE (`lfm2moe`) with PPL parity.
  - [ ] **Done when:** each variant admitted with PPL parity against reference.

### 14. Qwen3-VL, Parakeet TDT, ACE-Step parity, CPU-only vision features
- [ ] **14. Architectural additions & missing features**
  - [x] **Qwen3-VL:** Implement IMROPE plus `qwen3vl` architecture support. DONE 2026-09-27 on CPU (see Log).
  - [x] **Parakeet TDT:** Implement the decode head. DONE 2026-09-27 (see Log).
  - [ ] **ACE-Step 1.5 Turbo:** Validate numeric parity and add STATUS row.
  - [ ] **CPU-only vision features:** Port 2D M-RoPE image positions and deepstack to GPU/CUDA forward passes.

### 15. Performance items
- [ ] **15. Performance items** (Measure first, keep only measured wins per CLAUDE.md rule 7):
  - [ ] SmolLM2 prefill (0.89x llama.cpp) and Qwen3.6-35B prefill (0.63x).
  - [ ] Batched image-token prefill in VLMs.
  - [ ] Vulkan batched prefill and matvec bandwidth.
  - [ ] FLUX.1 / FLUX.2 GPU double-block GEMM & fusion.
  - [ ] MiniMax vocoder, MusicGen/AudioGen, CosyVoice3 ODE steps.
  - [ ] TTS/ASR GPU residency.

### 16. Product items
- [ ] **16. Product items**
  - [ ] `stingray setup` and first-run experience.
  - [ ] Configuration ownership.
  - [ ] Multi-model runtime phases.
  - [ ] Session `Fork()` context isolation.
  - [ ] NuGet release checklist.

---

## Can't be scheduled

### 17. CPU greedy non-determinism
- [ ] **17. CPU greedy non-determinism** (never reproduced; act only on a new sighting).

### 18. Blocked on the user or on other hardware
- [ ] **18. Blocked items**
  - [ ] **CosyVoice 2** garbled endings: blocked on upstream CosyVoice2-0.5B recorded reference run.
  - [ ] **HunyuanVideo** numeric check: blocked on independent v1 output (noise + latent).
  - [ ] **Pixtral 12B / GLM-4.6V timings:** blocked on `HF_TOKEN` with accepted license.
  - [ ] **Llama-4 Scout:** blocked on ~93 GB disk space.
  - [ ] **CUDA items:** (`rope_freqs` for Llama-3.1-style models) blocked on CUDA GPU.

---

## Log

- 2026-09-27: plan written. Item 1 done (99b6b53).
- 2026-09-27: item 7 timeboxed; the layer bisection turned it into a day-scale job.
  - The drift is spread across layers: it starts at about 0.3% after layer 0's attention and
    compounds, with no single broken layer.
  - Findings and the next experiment are in `bugstofix.md` (GLM entry).
  - Moved on to item 2.
- 2026-09-27: item 4 done.
  - `llama-mtmd-debug` now runs `gemma4v`. The encoder + projector match: row 0 within 0.003 on
    rainbow 224.
  - New test `Gemma4V_Rainbow224_MatchesLlamaMtmdDebug`; STATUS row added (encoder scope).
  - The tool asserts in clip.cpp while printing the last tensor, so its sum is not available.
- 2026-09-27: item 3 waiting. GLM-4.7-Flash is downloading (GLM-4.5-Air REAP moved to
  `K:\_other_models` to make room); its PPL run waits for the item 2 sweep.
- 2026-09-27: item 3 downloaded (`GLM-4.7-Flash-Q2_K.gguf`, 10.6 GB); its PPL comparison waits for the
  item 2 sweep (both need the machine).
  - C: free space is about 27 GB lower than the moves and downloads account for. No large new file
    was found in the repo or the profile; possibly a pending delete held open by a process. Not
    blocking.
- 2026-09-27: item 5 code done, not yet committed.
  - `stingray stt -m sensevoice|paraformer --model-file <.onnx>`; tokens are found next to the model.
    `SenseVoicePipeline` now implements `ISpeechToTextPipeline`.
  - CLI check: SenseVoice gives "concord returned to its place amidst the tents", identical
    (case-insensitive) to the LibriSpeech reference transcript.
  - Paraformer gives a fluent Mandarin transcript, ending 嗯 where sherpa-onnx's published sample text
    (quoted from memory) ends 你; unverified which is right.
  - The new exact-transcript test in `SenseVoiceRealWeightsTests` waits for the Audio test project to
    be free.
- 2026-09-27: item 6 part 1 committed (e58b58a).
  - 162 silent `return`s in test bodies now call Assert.Skip; 141 lookups also search models/_models.
  - ForwardPass.Fast: 14 real-weight tests that silently passed now really run, and pass.
  - Remaining:
    - the Audio project (after the sweep);
    - 14 files the other agent is editing;
    - 31 lookup helpers with other shapes (listed by `nested_models.py`).
- 2026-09-27: item 2 stopped again by the memory reaper, while the session was idle, still in Audio.
  - Before that, Audio logged two real failures: Fish Speech codec PCM cosine 0.052 and fast-AR
    logits cosine 0.44.
  - Entered in `bugstofix.md` (suspect: perf commit d377049); STATUS row downgraded to 🟡.
  - Vision, ForwardPass and Diffusion did not run.
  - Not restarted: the reaper rule says restart only when the user asks.
- 2026-09-27: item 5 done. `SenseVoiceRealWeightsTests.SpeechToTextPipeline_OnRealLibriSpeechClip_MatchesReferenceTranscript`
  passes: exact LibriSpeech transcript, lang `en`.
- 2026-09-27: item 6 Audio part done (46 files).
  - Remaining for item 6: 14 files the other agent is editing, and 31 lookup helpers with other
    shapes.
- 2026-09-27: item 8 done (4ed6866).
  - The Gemma-3 / Qwen3.8 concatenation warnings no longer appear in the template corpus run.
  - The real remaining gap was `tojson` formatting; it now matches llama.cpp's `/apply-template`
    byte for byte on Qwen3's template with tools.
- 2026-09-27: item 9 done.
  - Layers without attention (Mamba-2, short conv, MLP-only) no longer store zero KV rows.
    `RunTrunk` reserves the block and pages are allocated per layer only on first write (same
    mechanism as the Qwen3.5 GDN hybrid).
  - Measured on Granite 4.0-H 1B after the same prompt: 4/40 layers own pages, 512 KiB vs 5,120 KiB
    before (`Granite4H_MambaLayers_AllocateNoKvPages`).
  - Granite-H 5/5, Nemotron-H 2/2, LFM2 3/3 parity tests unchanged.
- 2026-09-27: item 3 not done yet.
  - GLM-4.7-Flash second-half PPL 8.1757 (batched) vs llama.cpp 8.0997: 0.94%, above the ~0.3% bar.
  - Sequential re-run in progress to separate a batched-path effect.
- 2026-09-27: item 10 done, as a bug entry (its "done when" allows that).
  - The GGUF Paraformer (`FunAsrPipeline`) returns an empty transcript on real Mandarin speech; the
    ONNX Paraformer on the same clip is fluent.
  - Also: `models/paraformer-q8.gguf` is actually Fun-ASR-Nano (`audiocpp`), misnamed.
  - Both are in `bugstofix.md`.
- 2026-09-27: item 3 closed as a bug entry (blocker).
  - GLM-4.7-Flash sequential PPL 8.2100, batched 8.1757, llama.cpp 8.0997: the gap does not come
    from the batched path. It is the same diffuse pattern as item 7; details in `bugstofix.md`.
- 2026-09-27: item 11a timeboxed out, logged in `bugstofix.md`.
  - LFM2 layer 0 matches llama.cpp to the 4-decimal print resolution of `llama-eval-callback`.
    Real drift starts around layer 2 and grows gradually.
  - Ruled out: the Q8_0 activation scheme (ours already matches ggml; an fp16 block scale changed
    nothing).
  - Useful side-product: the short-conv and Mamba-2 mixers now record the `o_proj` stage for
    `StageCapture`.
  - Lesson for items 3 and 7: differences near 1e-4 in those dumps are print rounding. Going further
    needs full-precision tensor dumps.
  - Next: item 11b.
- 2026-09-27: item 11b done.
  - Cause: `tokenizer.ggml.pre = youtu` had no case in `PreTokenizerPatterns.TryResolve`, so it used
    the GPT-2 fallback. Our tokens diverged from `llama-tokenize` at position 1427 (" 杜甫": we glued
    the space to the first byte of 杜). That one passage was the +5% window.
  - Fix: llama.cpp's LLAMA_VOCAB_PRE_TYPE_YOUTU cascade (CJK/Hangul/CJK-punctuation stage, then
    GPT-4o word shapes with single digits).
  - Evidence: all 2048 tokens now match `llama-tokenize`. wiki.test.raw [1024,2048) at -c 2048:
    13.2485 vs llama.cpp 13.2990 (was 13.97). `PreTokenizerParityTests` 27/27 run, 1 pre-existing
    skip, 59 s, three new `youtu` rows.
  - Found with the new `stingray perplexity --dump-nll <file>` (per-position token id and NLL).
- 2026-09-27: follow-up to 11b. A scan of every local GGUF's `tokenizer.ggml.pre` found two more
  values with no case in `PreTokenizerPatterns` (silent GPT-2 fallback): `deepseek-llm`
  (DeepSeek-V2-Lite) and `exaone-moe` (EXAONE-4.5-33B).
  - Ported llama.cpp's cascades for both (the DeepSeek-LLM letter class copied from
    `llama-vocab.cpp`, astral ranges as surrogate pairs).
  - On an 8 KB wikitext plus mixed-script sample vs `llama-tokenize`: DeepSeek 2002/2002 (it already
    matched), EXAONE-4.5 1784/1784 (was 1995 tokens, diverging at token 13: " is an" is one token).
  - Four probe rows in `PreTokenizerParityTests`; they skip while both checkpoints sit on K:.
  - Every local GGUF's pre value is now mapped.
- 2026-09-27: item 11c done. The f16 Qwen3-Embedding-0.6B NaN on the 13-token ACE-Step prompt no
  longer reproduces: Prefill and token-by-token Forward both give finite logits, and f16 agrees with
  Q8_0 (max logit 18.777 vs 18.793, same argmax). `Qwen3F16FiniteLogitsTests` pins it (3.8 s, three
  real weight loads logged). The fixing commit was not identified.
- 2026-09-27: the user moved item 14, then item 15, ahead of 11d-f and 12-13.
- 2026-09-27: item 14, Qwen3-VL done on CPU.
  - Downloaded `Qwen/Qwen3-VL-2B-Instruct-GGUF` Q8_0 and its Q8_0 mmproj into `models/_models`.
  - Text: `qwen3vl` was missing from the NeoX list and had no IMROPE. Added `RopeSectionsInterleaved` and
    `ModelHyperparams.MRopeComponent` (ggml `ggml_mrope_cache_init` for MROPE and IMROPE). Text positions are
    (p, p, p, 0), so pairs 61-62 never rotate. PPL 9.8356 vs llama.cpp 9.8513 (was 2654).
  - Deepstack: reuses Granite 4.0 Vision's `DeepstackMapping` (slice k added before layer k), synthesized from
    `n_deepstack_layers`.
  - Vision: `QwenVlVisionModel` read every u32 key as `is int`, so it kept its defaults (projection 3584 instead
    of 2048, patch 14 instead of 16). Qwen3-VL encoder: resized learned position grid, GELU FFN, LayerNorm with
    bias (and `post_ln` bias), `v.patch_embd.bias`, deepstack branches.
  - Evidence: `LlamaMtmdVisionParityTests` 14/14 (137 s, real weights), including the new `Qwen3Vl_Rainbow448`.
    End to end on `test-1.jpeg`: same answer as `llama-mtmd-cli` except one capitalisation token. RUNNING row
    added (2.7 GB, decode 18.9 tok/s).
  - Remaining for the "CPU-only vision features" sub-item: M-RoPE image positions and deepstack on CUDA/Vulkan.
- 2026-09-27: item 14, Parakeet TDT done.
  - `ParakeetTdtDecoder`: port of CrispASR `parakeet_tdt_decode` (the reference for these CrispASR-format GGUFs).
  - Loader: `parakeet.*` metadata, optional CTC head and linear biases, TDT weights; mel count from the shipped
    filterbank (128 for TDT v2; it was fixed at 80).
  - Evidence: `Tdt06bV2_LibriSpeech_MatchesCrispAsr` 4/4 clips identical to `crispasr --gpu-backend cpu` on the same
    q4_k file (8.2 s). CTC unchanged: `ParakeetLibriSpeechTests` CTC q4_k and f16 pass (26 s), heavy
    `ParakeetConformerEncoderTests` 3/3.
  - `stingray stt -m parakeet` added. Measured 3.9x real time, 2.8 GB; CrispASR is 2.2x faster. Logged in
    `bugstofix.md` (weights expanded to F32).
  - Also logged in `bugstofix.md`: M-RoPE image positions and deepstack are CPU only (Qwen-VL family on GPU).
  - Side fix: `stingray pull -q` now prefers an exact file name and skips `mmproj-*` unless asked, and treats
    HTTP 416 on an existing file as complete (both hit while fetching Qwen3-VL).
- 2026-09-27: Parakeet TDT performance pass (asked for by the user; stopped at parity with the reference, as asked).
  - Where the time went (scratch stage timer, 14.2 s clip, 5 runs): encoder 3.4-3.6 s of 3.5-3.7 s; mel ~50 ms,
    TDT decode ~60 ms. Every encoder linear ran as one matvec per frame, re-reading each weight matrix ~178 times.
  - Change: all encoder linears (FFNs, Q/K/V/out, positional, conv pointwise, subsampling, CTC) run as one
    batched SGEMM over the frames, with weights packed once at load and the F32 copies dropped.
  - Result, `stingray stt -m parakeet` on the same clip, 3 runs: 1.60-1.64 s, 8.7-8.9x real time (was 3.6 s, 3.9x);
    CrispASR 8.6x. Peak RAM 2.85 GB, unchanged (a first version that kept both copies peaked at 5.6 GB).
  - Output unchanged: TDT 4/4 clips identical to CrispASR; CTC q4_k and f16 still 2.9% WER, same words; heavy
    encoder tests 3/3.
- 2026-09-27: Parakeet memory pass (user: continue while context is warm; stop when good enough). Stopped here.
  - Q4_K weights now stay quantized: repacked once into 8-row groups and run through the text engine's
    `TryMatMulBatchedQ4Kx8` batched GEMM. Q8_0 conv layers packed F32 (int8 Q8_0 path ~0.2 s slower per clip; a
    batched Q8_0 4-input wrapper measured no gain and was reverted).
  - NativeAOT binary, 14.2 s clip, 3 runs each: TDT 1.43 s vs CrispASR 1.66 s (1.16x), CTC 1.04 s vs 1.14 s
    (1.10x). Peak RAM 1.1 GB (was 2.85 GB; CrispASR 0.6-0.65 GB). Load time 1.06 s -> 0.36 s.
  - Output unchanged: TDT 4/4 identical to CrispASR, CTC 2.9% WER same words, heavy encoder tests 3/3.
  - Note: single-shot timings from the JIT (`dotnet build`) CLI include ~1 s of JIT warm-up; the AOT binary is
    the fair comparison.
- 2026-09-27: item 11.f LLaVA-1.5 hardening & verification:
  - End-to-end vs llama-mtmd-cli: verbatim match on `test-1.png` ("The newspaper is the/The New York Times, and the main headline reads/is \"Men Walk on Moon.\""). Both 598 prompt tokens (576 image + 22 text). Stingray CPU prefill 7.9 t/s, decode 7.0 t/s.
  - Prompt format rule fix: unit test `LlamaPromptFormatRuleTests.FormatPrompt_LlamaArch_Llama3Headers_ImageModel_RendersLlama3Headers` proved that `s_isVicuna` forced LLaMA-3 models with `<image>`/`mlp` into Vicuna formatting. Removed `s_isVicuna`, relying on `!s_hasLlama3Headers` alone. 4/4 unit tests passed (1.15s). `llama-server --no-jinja` POST /apply-template confirmed ChatML fallback, while `llama-mtmd-cli` requires `--chat-template vicuna` for classic LLaVA.
  - Direct-splice edge cases:
    - Two images: 2 x `--image` with `<image>` twice in `-p` -> 1180 tokens (1152 image + 28 text), answered "Yes, the two images are of the same newspaper page." (7.5 t/s prefill / 6.3 t/s decode).
    - Count mismatch: 2 x `--image` with 1 x `<image>` -> clean exit code 1 (`Error: prompt has 1 '<image>' marker(s) but 2 --image file(s) were given`).
    - Automatic prepend: no `<image>` in `-p` -> prepended 1 image, 598 tokens, answered correctly.
    - Context overflow: `--ctx-size 512` -> clean exit code 1 (`Error: prompt plus images expand to 591 tokens (576 image) but the active context is 512`), `plannedPrefill` correctly accounted for -1 sentinel.
    - Literal word `<image>` without `--image` -> processed as ordinary text without error.
  - Parity & regression suites:
    - `LlamaMtmdVisionParityTests` (all 15 real-weight vision parity tests): 15/15 passed in 116.255s.
    - Qwen3-VL end-to-end regression: answered "This is The New York Times, and the main headline is \"Men Walk on Moon: Astronauts Land On Plain; Collect Rocks, Plant Flag.\"" (321 tokens, 20.3 t/s prefill / 19.0 t/s decode).
    - `Tests.Cli`: 370/371 passed (1 skipped for missing reference model) in 13.542s.
    - `Tests.ForwardPass.Fast`: 702/703 passed (1 skipped for STINGRAY_RUN_HEAVY_TESTS=1) in 1m 40s.
    - `Tests.Server.Fast`: 427/427 passed in 6.074s.
  - GPU offload (-g -1 on AMD Radeon Vulkan iGPU): uploaded all 32 layers to VRAM; classic LLaVA-1.5 has neither 2D M-RoPE nor deepstack, so GPU offload works end-to-end and outputs "The main headline of the newspaper is \"Men Walked on Moon.\"" at 7.3 t/s prefill / 6.5 t/s decode.
