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
- [x] **2. Rerun the real-weight landscape sweep** (DONE 2026-09-28, `scripts/sweep-tests.ps1`; the 2026-09-27 memory-reaper stall's root cause found and fixed same day — see Log)
  - [x] Built a resumable, one-class-per-process harness (`scripts/sweep-tests.ps1`, commit `4202bff`) specifically because the whole-suite run kept getting reaper-killed with no record of where it stopped.
  - [x] Found and fixed the actual driver of the reaper kills: `PersonaPlexLmTensorSource` eagerly dequantized the whole 7B Q8_0 checkpoint to fp32 on every PersonaPlex real-weight test (peaked 37-45 GB each, 7 of the memory sweep's top-10). Zero-copy Q8_0 fix: 28 GB -> 11.17 GiB, ~2.2x faster decode too. See `docs/done/audio-review-new-progress.md`'s PersonaPlex 7B entry.
  - [x] Added a cross-process mutex to `HeavyTestBase` (now present in Diffusion/Vision too, which had none before) so concurrent heavy-test processes can't stack memory regardless of how they're launched.
  - [x] Ran `STINGRAY_RUN_HEAVY_TESTS=1` for Diffusion, Audio, Vision, ForwardPass via the harness: 649 classes resumed/run to completion, no reaper kill.
  - [x] Read failures and logged them in `bugstofix.md` (new entry, "Real-weight landscape sweep rerun"): a WanTests tolerance failure, an 8-class FunASR cluster sharing one root cause (the already-tracked wrong-checkpoint trap), a QwenASR empty-transcript failure, a MeloTTS stub-file failure, a Parler near-miss, a QwenTTS NaN, 4 vision-embedder parity misses (Llava's cosine is negative), and 2 ForwardPass value-mismatches — triaged, not root-caused (see that entry for detail and log paths).
  - [x] Flagged any quick real-weight pass under ~0.4s as a possible silent no-op (CLAUDE.md rule 12) — see the sweep's own `-Summary` output; none of the flagged ones looked like a NEW no-op beyond the already-known pattern.
  - **Caveat:** the harness resumes from cached `state.jsonl`, so the PersonaPlex classes it reports as "done" are stale (pre-fix) entries from earlier the same day, not a fresh re-verification under the fix — see the new bugstofix entry for detail. The PersonaPlex numbers actually trusted are from direct, independent re-runs, not this sweep.
  - *Estimate:* little effort from developer, hours of machine time. (2026-09-27 attempt killed by memory reaper after Diffusion; 2026-09-28 rerun completed once the actual memory driver was fixed rather than just retried).

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
- [x] **6. Finish the silent-no-op test sweep** (DONE 2026-09-28)
  - [x] Point real-weight tests at `models/_models` as well as `models/` (Part 1 committed in `e58b58a`: 162 silent returns converted to `Assert.Skip`, 141 lookups updated).
  - [x] Verify `ForwardPass.Fast`: 14 previously silent real-weight tests now run and pass.
  - [x] Sweep remaining silent returns in the Audio project (46 files, `4874ea4`).
  - [x] Sweep the remaining 52 files/62 occurrences (`nested_models.py` no longer exists -- rebuilt the file list directly via grep for `if (X is null || ...) return;` at the top of a test body, converted each to `Assert.SkipUnless(negated-condition, "reason")`). Excluded 7 `Zz*Tmp.cs` scratch/debug files (not real regression tests, same judgment as the Diffusion `ZZ_Scratch*` files elsewhere) and 2 occurrences in `VisionOpsBenchmarkTests.cs` that are unsafe pointer null-guards inside private helpers, not test-skip gates -- converting those would have been a real behavior change, not a no-op fix.
  - [x] Verified: all 6 touched projects (Diffusion/Audio/Vision/ForwardPass/Vulkan/Cuda) build clean, 0 warnings; `ChatterboxCfmDecoderTests` (model present) now genuinely runs for real (11.1s, not a silent pass); `Flash64SchedulingTests` (AVX2/FMA + model-path gate) now reports 3 visible `[SKIP]`s with real reasons instead of silently passing.
  - [x] **Done when:** grep finds no remaining silent returns in real-weight tests, and test runs report skips as skips. -- confirmed: only the 7 excluded scratch files and the 1 excluded false-positive file still match the pattern.

---

## A few hours, cause already narrowed

### 7. GLM-4.5 (`glm4moe`) Q5_K×Q8_K kernel and gated decode path
- [x] **7. GLM-4.5 (`glm4moe`) Q5_K×Q8_K kernel and gated decode path** (implementation and synthetic-validation scope complete; real-weight evaluation split to item 19)
  - [x] Profile layer bisection: drift starts at ~0.3% after layer 0's attention and compounds across layers rather than a single broken layer.
  - [x] Document findings and next experiment into `bugstofix.md` (GLM entry); timebox to day-scale job.
  - [x] `llama-eval-callback` on 326-token wikitext prompt for tensors `ffn_inp-N` / `l_out-N`.
  - [x] `StageCapture` stages `post_attn_resid` / `post_ffn_resid` on matching token IDs.
  - [x] **2026-09-28:** split layer 0's attention into pre-`wo` (`kqv_out`/`attn_out`) and post-`wo` (`node_26`/`o_proj`) and compared each independently against `llama-eval-callback` on the real checkpoint. Pre-`wo` matches exactly (RoPE, QK score, softmax all ruled out); the divergence (~2e-4, above the print-resolution noise floor) first appears exactly at `wo`, the model's first Q5_K-quantized matmul. Directly supports the standing activation-quantization hypothesis (F32 vs ggml's Q8_K) rather than an attention-math bug. See the [archived implementation record](done/09-glm45-q5k-activation-implementation.md) for the full numbers and method.
  - [x] Implement a separate ggml-formula Q5_K×Q8_K dot path and a default-off gate only for the CPU decode attention output projection. Synthetic seeded tests compare dispatched and scalar paths against an independent scalar translation for cols 256–4096 (8 seeds each); all pass the predeclared `2e-4 + 2e-5 × |reference|` bound. The existing Q5_K paired-dot and Q8_K-related suites remain green (18 Q8KS, 4 Q3K×Q8K tests).
  - [x] **Done when:** the kernel passes the independent scalar oracle and focused Q5_K/Q8_K regression suites, with the experimental decode gate left off by default.

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
  - [x] **11.a** LFM2 0.24% PPL gap (10.9277 vs 10.9543). CLOSED 2026-10-01: not a gap, the labels were swapped. llama.cpp = 10.9543 (F16 and F32 KV identical), Stingray = 10.9277 (0.24% lower, inside +/-0.96 SE). See `bugstofix.md` item 11.
  - [x] **11.b** Youtu-VL: one 1024-token window +5% PPL vs llama.cpp. DONE 2026-09-27 (`9568823`, unmapped `youtu` pre-tokenizer).
  - [x] **11.c** NaN in `ForwardPass`'s f16 `qwen3` path (last layer, one position). DONE 2026-09-27: no longer reproduces; pinned by `Qwen3F16FiniteLogitsTests`.
  - [x] **11.d** `HybridGdnChunkedPrefill_MatchesSequentialPrefill` failure with real weights. DONE 2026-09-28: not an engine bug; the test's per-logit bound assumed only the recurrence reordered sums (see Log).
  - [x] **11.e** Stable Audio 3 padding masks in the APG norm. DONE 2026-09-28 (see Log).
  - [x] **11.f** Classic LLaVA-1.5 (missing image token in vocab). DONE 2026-09-27: direct-splice path in CLI (`RunCommand.cs`) when marker absent from special tokens; Vicuna prompt format for LLaMA-2 backbone (`!s_hasLlama3Headers` alone, removing `s_isVicuna` so LLaMA-3 VLMs aren't hijacked into Vicuna; pinned by `LlamaPromptFormatRuleTests`); patch/CLS ordering in `LlavaVisionEncoder.cs` (patches 0..575, CLS 576, matching llama.cpp's `clip_graph_llava::build`). Pinned by `LlamaMtmdVisionParityTests.Llava15_Rainbow336_MatchesLlamaMtmdDebug` (576x4096 soft tokens, sum -10587.12 vs -10596.39, row 0 [-0.5335, -0.0007, -0.2534] vs [-0.5347, -0.0022, -0.2532]). End-to-end answers "The newspaper is the New York Times, and the main headline reads \"Men Walk on Moon.\"" matching `llama-mtmd-cli`. Tested two images, count mismatch, no-marker prepend, context overflow, Qwen3-VL regression, and Vulkan GPU offload (-g -1).

---

## A day or more

### 12. Batched prompt processing for the recurrent families
- [x] **12. Batched prompt processing for the recurrent families** DONE 2026-09-28 (see Log)
  - [x] Add Mamba-2 and short-conv layers to `PrefillCore` for Granite 4.0-H, Nemotron-H, and LFM2.
  - [x] Batch the projections while keeping scan and conv sequential over tokens.
  - [x] Verify parity tests remain unchanged.
  - [x] Measure prompt tok/s and record in `RUNNING.md`.
  - [x] **Done when:** parity tests unchanged; prompt tok/s measured and recorded in RUNNING.md (DONE 2026-09-28).

### 13. MoE variants of the recurrent families
- [ ] **13. MoE variants of the recurrent families**
  - [x] Admit Granite 4.0-H tiny/small (MoE) with PPL parity. Small DONE 2026-09-28 (top-k renormalisation fix; +1.2% residual at -c 2048 logged).
  - [ ] Admit Nemotron-H MoE (latent MoE, sigmoid gating) with PPL parity. No local checkpoint (2026-09-28); not downloaded (C: has no free space, K: ~10 GB).
  - [ ] Admit LFM2-MoE (`lfm2moe`) with PPL parity. BLOCKED 2026-09-28 (timeboxed): PPL +6.3% and per-token vs batched disagree by 9%; see bugstofix.
  - [ ] **Done when:** each variant admitted with PPL parity against reference.

### 14. Qwen3-VL, Parakeet TDT, ACE-Step parity, CPU-only vision features
- [x] **14. Architectural additions & missing features** (DONE 2026-09-28, see Log)
  - [x] **Qwen3-VL:** Implement IMROPE plus `qwen3vl` architecture support. DONE 2026-09-27 on CPU (see Log).
  - [x] **Parakeet TDT:** Implement the decode head. DONE 2026-09-27 (see Log).
  - [x] **ACE-Step 1.5 Turbo:** Validate numeric parity and add STATUS row. DONE 2026-09-28 (see Log): missing `<|endoftext|>` fixed; latent cosine 0.994 vs audio.cpp q8_0.
  - [x] **CPU-only vision features:** Port 2D M-RoPE image positions and deepstack to GPU/CUDA forward passes. Vulkan DONE 2026-09-28; CUDA moved to item 18 (no CUDA GPU).

### 15. Performance items
- [x] **15. Performance items** (CLOSED for this machine 2026-09-28, see Log; discrete-GPU items moved to 18):
  - [x] SmolLM2 prefill (0.89x llama.cpp) and Qwen3.6-35B prefill (0.63x). DONE 2026-09-28: Qwen3.6-35B ~0.96x (and Qwen3-Coder-30B 0.57x -> 0.75x); SmolLM2 unchanged at 0.89x, remaining gap is inside the already-tuned Q4_K/Q6_K GEMMs (see Log).
  - [x] Batched image-token prefill in VLMs. DONE 2026-09-28 on CPU (Qwen3-VL 20 -> 78-86 tok/s); Vulkan still per token.
  - [x] Vulkan batched prefill and matvec bandwidth. Matvec re-measured 2026-09-28: Q4_K 42-44 GB/s at all shapes (~85% of DDR4 peak; closed), Q6_K 36-38 GB/s (~72%, open). Batched prefill: moved to 18 (discrete GPU needed).
  - [x] FLUX.1 / FLUX.2 GPU double-block GEMM & fusion. Deferred to backlog 2026-09-28 (item 18). Remaining kernel work (`MultiHeadAttentionTiled` at 88 GFLOP/s, 3.7 s of the 21.6 s double-block loop; the `DoubleBlockGpu` row-offset audit; utility dispatches) is measurable on the iGPU as kernel A vs kernel B, but the GEMMs already run at 600-643 GFLOP/s (093) and take about 16.5 s of the loop, so even a much faster attention leaves the GPU near 18-19 s against the CPU's 17.1 s. It only pays off on a discrete GPU, where attention would become the bottleneck.
  - [x] MiniMax vocoder, MusicGen/AudioGen, CosyVoice3 ODE steps. DONE 2026-09-28: MiniMax vocoder 1.9x, AudioGen 3.3x, MusicGen 2.4x (plus ACE-Step VAE 13.5x). CosyVoice3 left as is: CFG branches already run concurrently on GEMM-sized batches and it is 1.58x faster than audio.cpp.
  - [x] TTS/ASR GPU residency. Moved to 18 (discrete GPU needed).

### 16. Product items
- [ ] **16. Product items**
  - [x] `stingray setup` and first-run experience, front-door step 2. DONE 2026-09-28: catalog (3 entries), model home, SHA-checked resumable installs, `setup`, `models`. Steps 3-4 (path-free task commands, C# facade, docs-as-tests) remain in the front-door design doc.
  - [x] Qwen2.5-0.5B (the catalog's default chat model) CPU speed: decode 23 vs llama.cpp 98 tok/s, prompt 80 vs 379 (0.24x / 0.21x, 2026-09-28). Found while measuring the catalog; much worse than SmolLM2-1.7B's 0.89x. DONE 2026-09-28 (dbdbe2a): int8 Q5_0 kernels, decode 28 -> 66 tok/s (0.29x -> 0.67x), prefill 35 -> ~318 tok/s (~0.84x), perplexity unchanged (11.98 vs llama.cpp 12.01). Decode is still 0.67x of llama.cpp. Numbers in `PerformanceLeague.md`.
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
  - [ ] **CUDA items:** (`rope_freqs` for Llama-3.1-style models; M-RoPE image positions + deepstack in `CudaForwardPass`, done for Vulkan 2026-09-28) blocked on CUDA GPU.
  - [ ] **Discrete-GPU performance items** (from 15, 2026-09-28): Vulkan batched prefill, FLUX.1/FLUX.2 double-block GEMM and fusion (attention shader and audits: measurable on the iGPU but only pays off on a discrete GPU, see item 15), TTS/ASR GPU residency. This machine's iGPU shares DRAM with the CPU and trails it on prefill-shaped work, so a result here would not say whether the GPU code is good (CLAUDE.md rule 13). Blocked on a machine with a discrete GPU.

## Follow-up validation

### 19. GLM-4.5 real-weight Q5_K×Q8_K validation
- [ ] **19. GLM-4.5 real-weight Q5_K×Q8_K validation** (implementation is complete under item 7; this remains the correctness/admission gate)
  - [~] On the real `cerebras_GLM-4.5-Air-REAP-82B-A12B-Q2_K.gguf` checkpoint and matching 326-token prompt, compare gate-off and `ZZ_Q5K_Q8K=1` layer-0 `o_proj` against the existing llama.cpp reference using raw values and record max-abs, mean-abs, RMS, relative-L2, and cosine metrics.
  - [x] If the layer-0 relative-L2 error improves, measure both gate modes' second-half wikitext PPL at `-c 2048` against the unchanged llama.cpp reference 8.6125; record exact command, memory, and measured values.
  - **Measured 2026-10-01 (paired per-token NLL instead of the layer-0 dump, whose harness prints only 6 values at 4 decimals):** `llama-perplexity -c 2048 --chunks 1 --save-all-logits` (CPU, `PPL = 8.6125 +/- 0.70301`) decoded per-token against `stingray perplexity -c 2048 --dump-nll` (per-token decode, 1,023 paired tokens, 0 token mismatches). Gate off: `[1024,+)` PPL 8.7753, paired dNLL +0.01639 (SE 0.01024, +1.6 SE), per-token rms 0.328. `STINGRAY_Q5K_DECODE_Q8K=1`: `[1024,+)` PPL **8.5956** (-0.20% vs llama.cpp, inside the ~0.3% target), paired dNLL -0.00397 (SE 0.00646, -0.6 SE), per-token rms **0.207** (-37%). Gate affects only the non-MLA decode `wo` Q5_K matvec. 82B Q2_K read from K: (33.3 GB; C: has ~15 GB free), ~2.1 tok/s per-token, 15-17 min per run, run one at a time. Reader: scratchpad `pair.cs` (decodes the llama.cpp logits file: header, tokens, per-token `scale`,`min_log_prob`, uint16 log-probs). **Hypothesis supported on real weights; layer-0 raw comparison, the small-Q5_K regression, and the broader inventory remain open, gate stays off by default.** Residual rms 0.207 suggests other K-quant matvecs (non-`wo`) also differ from ggml's Q8_K-activation behaviour.
  - [x] **19a. Small real-weight Q5_K regression (own check, gates the default flip):** pull a small Q5_K_M text model (e.g. SmolLM2-1.7B-Instruct, ~1.2 GB; no local text LLM has Q5_K tensors), run paired per-token NLL vs llama.cpp (`--save-all-logits` / `--dump-nll`) with `STINGRAY_Q5K_DECODE_Q8K` off and on, record PPL, paired mean dNLL and rms for both. Gate-on must not be worse than off beyond noise.
- **Measured 2026-10-01:** gate extended to every decode Q5_K matvec (single/dual/2-in/4-in, folded MoE dot) plus Q5_1 (Q8_1 activations). GLM-4.5 (same paired method, per-token, -c 2048): wo-only `[1024,+)` PPL 8.5956, rms 0.2067; extended 8.6200 (+0.09% vs llama.cpp 8.6125), paired dNLL -0.00131 (-0.2 SE), rms 0.2038. **SmolLM2-1.7B-Instruct Q5_K_M** (144 Q5_K + 25 Q6_K tensors; llama.cpp `PPL = 6.8582 +/- 0.54279`, identical with `-ctk/-ctv f32`): gate off `[1024,+)` 6.8304, paired dNLL -0.00822 (-1.9 SE), rms 0.1416; gate on 6.8410, dNLL -0.00659 (-1.4 SE), rms 0.1484; on-off per-token sd 0.137. **Reading:** the gate is not worse beyond noise on a normal Q5_K model, but it does not improve it either; the clear win is GLM-4.5 (rms 0.328 -> 0.204, on-off sd 0.229 moving toward llama.cpp). Evidence for a default flip is therefore mixed: keep the gate off by default until 19b, and do not claim a general Q5_K fix. Residual rms ~0.14-0.20 on both models has another source (not KV precision, not Q5_K activation format); Q4_K Q8KS (per-32 scale vs ggml per-256) is one untested candidate.
  - [ ] **19b. Broader real-weight Q5_K inventory** with the gate on, against existing golden/parity expectations, before any default change; plus a gate-on/off speed measurement (one process at a time, several runs) for `PerformanceLeague.md`.
  - [ ] **Done when:** real-weight evidence is recorded; only if PPL is within ~0.3% of llama.cpp should GLM-4.5 admission and STATUS/RUNNING/MODELS updates be considered. Keep the gate off by default until then. See the [closed implementation record](done/09-glm45-q5k-activation-implementation.md) and [the detailed plan](1-correctness/09-glm45-q5k-activation-quant-plan.md).

---

## Log

- 2026-09-27: plan written. Item 1 done (99b6b53).
- 2026-09-27: item 7 timeboxed; the layer bisection turned it into a day-scale job.
  - The drift is spread across layers: it starts at about 0.3% after layer 0's attention and
    compounds, with no single broken layer.
  - Findings and the implementation are now in the [closed GLM-4.5 implementation record](done/09-glm45-q5k-activation-implementation.md); remaining validation is item 19 above.
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
- 2026-09-28: item 11.e Stable Audio 3 APG norm padding mask done:
  - Extracted shared `ApplyApg` helper in `StableAudioScheduleKernels.cs` matching `rf_dit.cpp:845-885` and Python `stable-audio-tools`.
  - Global `normSq` and `dot` sum over valid tokens only (`0 .. validTokens * channels - 1`). Epsilon placement fixed to `1.0 / Math.Sqrt(normSq + 1e-16)` inside the sqrt.
  - Padded tokens zero the orthogonal projection component, falling back to pure conditioned guidance at `apg = 1.0` (`condX0 + (cfgScale - 1) * (1 - apg) * diff`).
  - Added missing `ValidLatentTokens` and self-attention V-zeroing (`v.AsSpan((MemoryTokens + valid) * Dim).Clear()`) to `StableAudioMediumDiT.cs`, wired through `StableAudioMediumPipeline.cs`.
  - Replaced duplicate APG math at all 4 call sites in `StableAudioPipeline.cs` and `StableAudioMediumPipeline.cs` (GPU step loop and CPU `PredictVelocity`).
  - Measured numeric parity against `audiocpp_cli.exe` with Euler sampler dumps on Small SFX 6s:
    - Step 0 velocity cosine: 0.999949.
    - Padded token final latent cosine: 0.9999998 (max diff 0.00114 vs reference), confirming APG padding mask math is exact.
  - Tests:
    - Synthetic unit tests `StableAudioApgTests` (3/3 passed).
    - `StableAudioPipelineGoldenParityTests`: 2/2 passed (CPU + GPU real weights, 39.8s).
    - `StableAudioConformanceTests`: 3/3 passed (13.5s).
    - `StableAudio3SmallSfxTests`: 1/1 passed (17.1s).
    - `StableAudio3MediumPipelineTests`: 1/1 passed (35.0s).
- 2026-09-28: item 14, ACE-Step 1.5 Turbo parity against audio.cpp.
  - Reference: `audio-cpp/audio.cpp-gguf` `ACE-Step1.5-GGUF/turbo/ace-step-1.5-turbo-q8_0.gguf` (6,185,460,032 bytes,
    in `K:\_other_models\ACE-Step1.5-GGUF\turbo\` because C: would not free space, see handover 104). Run with the planner off
    (`--request-option thinking=false use_cot_metas=false use_cot_caption=false use_cot_language=false`), `--lyrics "[Instrumental]"`,
    10 s, and `noise_file=` a shared frame-major [250, 64] f32 noise. audio.cpp patched locally (examples/ is gitignored) to dump
    `encoder_hidden`, `context_latents`, `final_latent` under `ACESTEP_DUMP_DIR`. Ours: new `STINGRAY_ACESTEP_NOISE` / `STINGRAY_ACESTEP_DUMP`.
  - Before: condition sequence 77 vs 79 tokens, final latent cosine 0.937, waveform 0.61. Row alignment showed one missing token
    at the end of both the lyric and the caption sequence: the Qwen3-Embedding tokenizer post-processor appends `<|endoftext|>`
    (audio.cpp `tokenize_text` too); ours did not. Fixed in `AceStepQwen3TextEncoder.Tokenize`/`Encode`, pinned by
    `Tokenize_RealWeights_AppendsEndOfTextLikeReferenceTokenizer`.
  - After, f16 text encoder: condition 0.997, final latent 0.982. With the q8_0 text encoder (same quantisation as the reference):
    condition 0.99896, final latent 0.994, waveform 0.945. Remaining gap is consistent with the reference's q8_0 DiT against our
    bf16; closing it needs the 10.1 GB bf16 bundle (no disk for it now).
  - VAE: ours decoding the reference latent matches the reference waveform to 0.99999. The 35% RMS difference seen first is
    `WavWriter` peak-normalising a 1.39 peak to 0.95 where audio.cpp hard-clips (215 samples).
  - Tests: `AceStepQwen3TextEncoderTests` 2/2 (2.4 s), `AceStepConditionEncoderTests` 1/1, `AceStepPipelineEndToEndTests` 1/1 (26.7 s),
    `AceStepDiTTests` 2/2, `AceStepFlowSchedulerTests` 5/5, `AceStepOobleckDecoderTests` 1/1.
  - Speed (RUNNING.md): 10 s of audio in 99-102 s, of which VAE decode 88-90 s; audio.cpp does the whole thing in 37 s on CPU.
    Candidate for item 15.
  - Time taken: about 1.5 hours.
- 2026-09-28: item 14, M-RoPE image positions and deepstack on Vulkan (the last item-14 sub-item that this machine can test).
  - New `RoPENeoxPairPos` shader / `VulkanBackend.RoPEPairPositions`: NEOX RoPE with a per-pair position buffer.
    `MRopeImageLayout` (new, shared by `ForwardPass` and `GpuForwardPass`) holds the image registry and fills the
    per-pair (t, h, w, 0) positions. `GpuForwardPass` uploads them per token, takes the per-token trunk for M-RoPE
    models (so text also gets the unrotated IMROPE pairs 61-62), and accepts 8192-wide deepstack rows in
    `ForwardEmbedding`, adding each slice before its mapped layer. CLI registers images on the GPU pass and now refuses
    image input for M-RoPE models on passes without it (CUDA, Vulkan hybrid) instead of answering wrongly.
  - Evidence: `Qwen3VlVulkanMRopeParityTests` (heavy, 29 s, real weights): final logits Vulkan vs CPU cosine 0.999504,
    same top token; controls: without image registration 0.993721, without deepstack 0.994323. CLI real image
    (AMD block diagram PNG, 784 image tokens) gives a coherent, image-grounded answer on Vulkan (it previously threw on
    the 8192-wide rows). Regression: `VulkanArchLogitParityTests` 18/18 (207 s), `VulkanCpuLogitParityTests` 1/1,
    `VulkanPrecompiledShaderTests` 3/3, `ForwardPass.Fast` 703 (1 skipped), `Tests.Cli` 371 (1 skipped).
  - Qwen3-VL files had disappeared from `models/_models`; re-fetched with `stingray pull` to `K:\_other_models\qwen3vl`
    (tests find them via `STINGRAY_QWEN3VL_DIR`). `test-1.jpeg` is gone too.
  - CUDA left for item 18 (no CUDA GPU). Time taken: about 1.5 hours.
- 2026-09-28: item 15, performance.
  - SmolLM2-1.7B prefill re-measured: 222-232 t/s (4 runs, 1252-token prompt) vs `llama-bench -p 1252 -t 16` 255.7 = 0.89x,
    unchanged. Profile: FFN 62 %, QKV 19 %, attention 7 % (Flash-64; ~154 GFLOP in 396 ms, ~390 GFLOP/s, not the gap),
    out-proj 6 %, RoPE+KV append 3 %. The remaining ~11 % is inside the Q4_K/Q6_K GEMM kernels that already had three
    tuning rounds (docs/done/101); not continued this session.
  - Qwen3.6-35B-A3B prefill: the chunked GDN recurrence ran its 32 heads serially. Parallel over heads (bit-identical):
    recurrence 4453 -> 649 ms, prefill 32.1-32.7 -> 42.4-42.9 t/s (515 tokens, 3 alternating runs each, greedy output
    identical), llama.cpp 65.42 -> 0.50x to 0.65x. Now MoE is 55 % of prefill. PerformanceLeague row added.
  - Qwen3.6-35B-A3B, second step: the GDN pre-recurrence per-token loop (1065 ms of 515 tokens) now runs tokens in
    parallel (conv1d from a contiguous [state; chunk] window). Prefill 41.4-42.7 -> 45.9-47.0 t/s = 0.71x llama.cpp,
    greedy output identical. Also measured: GDN input projections (Q8_0) 1683 ms at ~460 GFLOP/s, ssm-out 543 ms, MoE 55 %.
  - 11.d reproduces on the local Q6_K via the new `STINGRAY_HYBRID_GDN_MODEL` test override: argmax matches, but vocab
    142707 is 0.49947 (sequential) vs 0.092426 (chunked), |diff| 0.407 > tol 0.1025. Identical numbers with the pre-session
    GDN code, so today's parallel changes did not cause or change it.
  - Qwen3.6-35B-A3B, third step: chunked-prefill attention batched over (head, token) instead of one head-parallel
    attention per token. 45.9-47.0 -> 47.9-48.9 t/s = 0.74x (from 0.50x at the start of the session). Bit-identical.
    Remaining: MoE 64 % of prefill (6.7 s of 10.5 s).
  - Qwen3.6-35B-A3B, fourth step: MoE prefill ran its 256 experts serially (six fork/joins each on ~16 tokens) and the
    router per token serially. Experts in parallel (5645 -> ~2860 ms) and router over tokens (580 -> 156 ms):
    46.4-48.1 -> 62.3-63.2 t/s in alternating runs = ~0.96x llama.cpp (65.42). Output identical. This sub-item of 15 is
    at parity; the dense `ForwardPass` MoE path should be checked for the same serial expert loop.
  - Dense `ForwardPass` MoE had the same serial expert loop: both passes now share `MoeBatchedExperts` (DRY) and route
    tokens in parallel. Qwen3-Coder-30B-A3B Q4_K_M, 507 tokens: 48.9-53.1 -> 66.7-69.3 t/s = 0.57x -> 0.75x llama.cpp
    (89.27), output identical. Tests: `MoeBatchedPrefillParityTests` 3/3, `OlmoeGreedyParityTests` 2 (+1 skip),
    `PhiMoeGreedyParityTests` 2/2 (real weights), ForwardPass.Fast 703.
  - Batched image-token prefill (item 15, second sub-item), CPU: `ForwardPass.PrefillEmbeddings` feeds precomputed rows
    through `PrefillCore` (M-RoPE via the registered images, deepstack slices added per layer), with a per-row fallback for
    configurations the batched trunk excludes; the CLI image path uses it. Qwen3-VL 2B, 784-token image: prefill
    19.8-20.1 -> 78.2-85.8 t/s, wall 63 -> 32-35 s. Batched vs per-token logits cosine 0.9994, same top token (new test in
    `Qwen3VlVulkanMRopeParityTests`). Vulkan still feeds image tokens one by one (M-RoPE models take its per-token trunk).
  - ACE-Step VAE decode (found under item 14; taken before the GPU-only sub-items of 15, whose iGPU measurements settle
    little here, CLAUDE.md rule 13): Oobleck `FullConv1d` and `ConvTranspose1d` rewritten as tiled GEMM (im2col / col2im
    around `PackedSgemmF32.Gemm`). Decode 88-90 -> 6.5 s, full 10 s generation 99-102 -> 17.3-17.8 s (audio.cpp CPU 37 s).
    Output within 1 int16 LSB of the old kernels on audio.cpp's latent. `AceStepOobleckDecoderTests`, E2E and
    `AceStepPrecomputeSilenceTests` pass; the two diffusers golden-parity tests skip (fixtures not on disk).
  - MiniMax vocoder (item 15, audio sub-item): the GEMM conv kernels moved to a shared `Primitives/Conv1dGemm` used by the
    ACE-Step Oobleck decoder and the MiniMax-Music3 vocoder. Vocoder decode 8.66-9.14 -> 4.57-4.64 s (200 frames), max abs
    diff 7e-7 vs the old kernels; the ACE-Step decode is byte-identical before/after the extraction.
  - MusicGen/AudioGen (item 15, audio sub-item): `CfmLinearWeight.MatMul` split F16 matmuls over input rows only, so
    t = 1 decode steps ran single-threaded. Small-t calls now split output rows (bit-identical: AudioGen greedy PCM SHA-256
    equal). AudioGen-medium 3 s: 62.0 -> 30.0 s; MusicGen-small 3 s: 16.7 -> 9.5 s. Also benefits every other
    `CfmLinearWeight` caller with small batches (MiniMax RVQ depth decoder, CosyVoice/Chatterbox CFM, F5, QwenASR, T5).
    Tests.Audio 512 (428 skipped without heavy) pass.
  - AudioGen CFG: both guidance branches in one batched decoder step (`AudioGenTransformer.StepBatch`; `Step` is its B = 1
    case). 3 s: 30.0 -> 19.0 s (62.0 at the start of the day), greedy PCM SHA-256 unchanged.
  - DRY: MusicGen and AudioGen transformers were near-copies; both are now thin wrappers over
    `Primitives/AudiocraftLmKernels` (public API unchanged), so MusicGen gets the batched CFG step too. MusicGen-small 3 s:
    9.55 -> 6.98 s (16.7 at the start of the day). PCM SHA-256 unchanged for both; `MusicGenDecoderGoldenParityTests`,
    `AudioGenDecoderGoldenParityTests`, `AudioGenEndToEndGoldenParityTests`, `AudioGenDiagnosticTests` 7/7,
    `AudioGenGenerationSmokeTests` 3/3 pass.
  - Item 15 closed for this machine (2026-09-28). Vulkan matvec re-measured (PerformanceLeague): Q4_K 42-44 GB/s at every
    shape, Q6_K 36-38 GB/s. The GPU-only sub-items moved to 18. Next: item 16.
- 2026-09-28: item 11.d closed.
  - Reproduced on the local Qwen3.6-35B-A3B UD-Q6_K via `STINGRAY_HYBRID_GDN_MODEL`: same argmax, cosine 0.99980, one
    tail logit off by 0.41 (tolerance 0.1025).
  - Isolation (scratch test, 97 tokens): int8 prefill on/off and batched MoE on/off leave it at 0.39-0.41; replacing the
    chunked recurrence with the exact sequential scan inside the chunked path leaves it at 0.42-0.46 (cosine 0.99980).
    So the chunk algorithm is not the source: it is the batched path's different summation order (projections, MoE)
    amplified by 40 layers of top-k routing. Kernel-level chunked vs sequential is pinned by `GdnKernelsTests`.
  - The test now asserts argmax equality and logit cosine >= 0.9995, with the evidence in its doc comment. Passes (40 s).
  - Found on the way: `AceStepPrecomputeSilenceTests` overwrites the checked-in `src/.../AceStep/silence_*.bin` and the
    runtime copies in `models/acestep-v15/` (it regenerated them with today's GEMM conv, a few ULP different). Restored
    both from git; the test is a generator, not a check, and should not run in ordinary test passes (logged in bugstofix).
- 2026-09-28: item 12 done, batched prefill for the recurrent families.
  - `Mamba2Step` / `ShortConvStep` split into in_proj + `Mamba2Mix` / `ShortConvMix` + out_proj; the `Mix` functions
    take n tokens (conv per channel and scan per head, each walking the tokens in order, so every token sees the
    one-token arithmetic). Decode is bit-identical (greedy output hash equal to the previous build, Granite 1B and LFM2).
  - `PrefillCore` runs them with batched projections (`Mamba2Prefill` / `ShortConvPrefill`), reserves the chunk's
    cache blocks up front, sends Nemotron-H MLP-only layers straight to the FFN and skips the FFN on its no-FFN
    layers (the first batched Nemotron run crashed there). SnapKV stays off for these models; TurboQuant keeps the
    per-token path. `STINGRAY_RECURRENT_BATCHED_PREFILL=0` switches back.
  - Prompt (CLI, ~510-580 tokens, per-token -> batched): Granite 4.0-H 350M 72 -> 245, 1B 21 -> 63, LFM2 1.2B 30 -> 101,
    Nemotron Nano 12B v2 Q2_K 6.7 -> 11.7 tok/s (Q2_K has no int8 prefill tier).
  - Quality: wikitext -c 2048 `[1024,+)` bucket, per-token vs batched: LFM2 10.9277 vs 10.9219, Granite 1B 8.7833 vs
    8.7741. Parity tests unchanged: `GraniteHybridGreedyParityTests` 5/5, `Lfm2ParityTests` 3/3, `NemotronHParityTests`
    2/2 (real weights), ForwardPass.Fast 703.
  - Time taken: about 2 hours.
- 2026-09-28: item 13, Granite 4.0-H small (MoE).
  - `granite-4.0-h-small-Q2_K` (already present) loaded but answered in fragments; wikitext -c 512 PPL 157 vs llama.cpp
    9.41, identical on the per-token and batched paths (so not item 12). llama.cpp's granite-hybrid.cpp (and granite.cpp
    for granitemoe) call build_moe_ffn with norm_w = true; ours only renormalised top-k for qwen3moe/phimoe. Fixed in
    `ModelGraph` for granitehybrid and granitemoe.
  - After: -c 512 9.3505 vs 9.4103; -c 2048 `[1024,+)` 26.4155 (per token 26.5483) vs 26.1080 (+1.2%, logged in
    bugstofix). Chat answer correct. Prompt 9.8 tok/s, decode 7.4 tok/s (42-token prompt).
  - Tiny not local; Nemotron-H MoE not local; LFM2-MoE next.
- 2026-09-28: item 13, LFM2-MoE (`LFM2-8B-A1B-Q4_K_M`), timeboxed and not admitted.
  - Wired `lfm2moe` into `ModelGraph` like `lfm2` (short-conv layers, NeoX RoPE) plus top-k renormalisation (llama.cpp
    lfm2.cpp passes norm_w = true); sigmoid gating with `exp_probs_b` already follows the DeepSeek path. Runs cleanly
    (`admit-arch`), but wikitext PPL is +6.3% at -c 2048 (15.8076 vs 14.8639), and at -c 512 per token 8.1860 vs batched
    8.9595 vs llama.cpp 8.7030. Ruled out: parallel experts/routing (serial identical), expert kernels (exact match on
    real weights). Logged in bugstofix with the next step. Not in the allowlist.
  - Nemotron-H MoE: no local checkpoint; Granite 4.0-H tiny: not local (small done above).
- 2026-09-28 (third session): item 16, front-door step 2.
  - Catalog entries were sourced by hash: the README quick start's local files match the HF tree API's lfs sha256
    (`qwen2.5-0.5b-instruct-q4_k_m.gguf`, `en_US-lessac-medium.onnx`, `ggml-base.bin`); the non-LFS `.onnx.json` was hashed
    after download. The local `bge-small-en-v1.5-q8_0.gguf` does NOT match CompendiumLabs' file of that name, so embed is
    not in the catalog yet.
  - `Core/Catalog`: `ModelCatalog` (static table, AOT-safe), `ModelHome` (`STINGRAY_MODEL_HOME`, flat layout so a Piper
    `.onnx.json` sits beside its `.onnx`), `ModelInstaller` (`.part` download, resume, SHA-256, then rename; mismatch
    deletes the partial), `ModelDownloader` (now also used by `pull`). CLI: `setup <task|id> [--yes] [--accept-licence]`,
    `models [task]`; the option binder gained `Positional = true` so `setup chat` works.
  - Real run into a scratch home: speak resumed from a 50 MB partial and verified; transcribe downloaded 141 MB and
    verified; chat re-hashed a copied file. Each printed run command worked; Whisper transcribed the Piper clip word-exact.
    `ModelCatalogTests` (local HTTP stub) cover resume and hash mismatch. Core 699 passed / 48 skipped, CLI 374 / 1 skipped.
  - Found: Qwen2.5-0.5B CPU is 0.24x llama.cpp on decode (logged under 16 above).

