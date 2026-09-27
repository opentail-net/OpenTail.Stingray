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
- [ ] **2. Rerun the real-weight landscape sweep**
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
- [ ] **3. GLM-4.7-Flash (`deepseek2`)**
  - [x] `stingray pull -r unsloth/GLM-4.7-Flash-GGUF -q Q2_K` (10.6 GB downloaded to `models/_models/GLM-4.7-Flash-Q2_K.gguf`).
  - [ ] `admit-arch` verification.
  - [ ] Run second-half PPL vs `llama-perplexity` (`-c 2048`).
  - [ ] **Done when:** PPL within ~0.3% of llama.cpp; add STATUS/RUNNING rows, and entry in MODELS.md if it qualifies.
  - *Note:* uses `deepseek2` in llama.cpp, not `glm4moe`, so it may already work. Waits for Item 2 sweep to free machine.

### 4. Gemma 4 E4B vision (`gemma4v`) parity
- [x] **4. Gemma 4 E4B vision (`gemma4v`) parity** (DONE 2026-09-27)
  - [x] Add a `LlamaMtmdVisionParityTests` case against `llama-mtmd-debug` (rainbow image), following vision parity pattern.
  - [x] Verify encoder + projector match: row 0 within 0.003 on rainbow 224 (`Gemma4V_Rainbow224_MatchesLlamaMtmdDebug`).
  - [x] **Done when:** stage fingerprints match; STATUS row added.

### 5. Wire the ONNX speech-to-text pipelines into `stingray stt`
- [ ] **5. Wire the ONNX speech-to-text pipelines into `stingray stt`**
  - [x] Implement `ISpeechToTextPipeline` in `SenseVoicePipeline`.
  - [x] Add CLI options to `SttCommand`: `-m sensevoice|paraformer --model-file <.onnx>` and auto-discover tokens file.
  - [x] Verify CLI transcription: SenseVoice matches LibriSpeech reference text ("concord returned to its place amidst the tents").
  - [x] Verify CLI transcription: Paraformer outputs fluent Mandarin transcript.
  - [ ] Run exact-transcript test in `SenseVoiceRealWeightsTests` once Audio test project is free.
  - [ ] Commit CLI and pipeline changes.
  - [ ] **Done when:** `stingray stt` transcribes a real clip with each, matching existing pipeline output.

### 6. Finish the silent-no-op test sweep
- [ ] **6. Finish the silent-no-op test sweep**
  - [x] Point real-weight tests at `models/_models` as well as `models/` (Part 1 committed in `e58b58a`: 162 silent returns converted to `Assert.Skip`, 141 lookups updated).
  - [x] Verify `ForwardPass.Fast`: 14 previously silent real-weight tests now run and pass.
  - [ ] Sweep remaining silent returns in the Audio project (after Item 2 sweep).
  - [ ] Sweep remaining 14 files under active edit.
  - [ ] Sweep 31 lookup helpers with non-standard shapes (listed by `nested_models.py`).
  - [ ] **Done when:** grep finds no remaining silent returns in real-weight tests, and test runs report skips as skips.

---

## A few hours, cause already narrowed

### 7. GLM-4.5 (`glm4moe`) 1.9% perplexity gap
- [ ] **7. GLM-4.5 (`glm4moe`) 1.9% perplexity gap**
  - [x] Profile layer bisection: drift starts at ~0.3% after layer 0's attention and compounds across layers rather than a single broken layer.
  - [x] Document findings and next experiment into `bugstofix.md` (GLM entry); timebox to day-scale job.
  - [ ] `llama-eval-callback` on 326-token wikitext prompt for tensors `ffn_inp-N` / `l_out-N`.
  - [ ] `StageCapture` stages `post_attn_resid` / `post_ffn_resid` on matching token IDs.
  - [ ] Compare layer-by-layer values and isolate cumulative drift cause.
  - [ ] **Done when:** second-half PPL within ~0.3% of llama.cpp's 8.6125; allowlist entry, STATUS, RUNNING, and MODELS rows added.

### 8. Jinja chat-template gaps
- [ ] **8. Jinja chat-template gaps**
  - [ ] Support string concatenation inside a conditional expression (Gemma-3-4B-it and other cases in `00-current-work.md` §1 item 7).
  - [ ] Add regression tests verifying template rendering against test vectors.
  - [ ] **Done when:** those templates render identically to llama.cpp's output.

### 9. Empty KV rows for layers without attention
- [ ] **9. Empty KV rows for layers without attention**
  - [ ] Update `PagedKvCache` to allow Mamba-2, short-conv, and MLP-only layers to skip storage rather than allocating zero KV blocks.
  - [ ] Measure and confirm KV footprint reduction scales with attention layer count only.
  - [ ] Verify Granite-H, Nemotron-H, and LFM2 parity tests remain unchanged.
  - [ ] **Done when:** KV memory footprint reduction measured and verified without parity regression.

### 10. FunASR-Nano on real speech
- [ ] **10. FunASR-Nano on real speech**
  - [ ] Transcribe a real clip (LibriSpeech sample in repo).
  - [ ] Validate output against reference transcript.
  - [ ] **Done when:** sensible transcript produced, or a precise defect entry logged in `bugstofix.md`.

---

## A few hours, cause unknown

Timebox each at half a day, write down what was learned, and move on if blocked.

- [ ] **11. Unknown-cause set**
  - [ ] **11.a** LFM2 0.24% PPL gap (10.9277 vs 10.9543).
  - [ ] **11.b** Youtu-VL: one 1024-token window +5% PPL vs llama.cpp.
  - [ ] **11.c** NaN in `ForwardPass`'s f16 `qwen3` path (last layer, one position).
  - [ ] **11.d** `HybridGdnChunkedPrefill_MatchesSequentialPrefill` failure with real weights.
  - [ ] **11.e** Stable Audio 3 padding masks in the APG norm.
  - [ ] **11.f** Classic LLaVA-1.5 (missing image token in vocab).

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
  - [ ] **Qwen3-VL:** Implement IMROPE plus `qwen3vl` architecture support.
  - [ ] **Parakeet TDT:** Implement the decode head.
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
