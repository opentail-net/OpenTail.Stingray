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

## Minutes

### 1. Commit the GLM entry in bugstofix.md
- **Steps:** commit `docs/1-correctness/bugstofix.md` (GLM-4.5 1.9% entry, Mamba-2 op note).
- **Done when:** committed.

### 2. Rerun the real-weight landscape sweep
- **Steps:**
  - Run `STINGRAY_RUN_HEAVY_TESTS=1` for Diffusion, Audio, Vision, ForwardPass, one suite at a
    time, each with its own log. Nothing else heavy running.
  - Read failures and any real-weight class that finishes in under about 0.4 s (silent no-op,
    CLAUDE.md rule 12).
- **Done when:** every suite has a complete log. Each failure is either fixed or entered in
  `bugstofix.md`.
- **Estimate:** little effort from me, hours of machine time. The 2026-09-27 attempt was killed by
  the memory reaper after Diffusion (no failures to that point).

## About an hour

### 3. GLM-4.7-Flash (`deepseek2`)
- **Steps:**
  - `stingray pull -r unsloth/GLM-4.7-Flash-GGUF -q Q2_K` (11.3 GB).
  - `admit-arch`, then second-half PPL vs `llama-perplexity` (-c 2048).
- **Done when:** PPL within about 0.3% of llama.cpp. Then a STATUS/RUNNING row, and MODELS.md if
  it qualifies.
- **Note:** it is `deepseek2` in llama.cpp, not `glm4moe`, so it may already work.

### 4. Gemma 4 E4B vision (`gemma4v`) parity
- **Steps:** add a `LlamaMtmdVisionParityTests` case against `llama-mtmd-debug` (rainbow image),
  following today's vision parity pattern.
- **Done when:** stage fingerprints match; STATUS row added.

### 5. Wire the ONNX speech-to-text pipelines into `stingray stt`
- **Steps:** SenseVoice and ONNX Paraformer already work; add the CLI plumbing.
- **Done when:** `stingray stt` transcribes a real clip with each, matching the existing pipeline
  output.

### 6. Finish the silent-no-op test sweep
- **Steps:**
  - Point real-weight tests at `models/_models` as well as `models/`.
  - Turn `if (path is null) return;` into `Assert.Skip*`.
- **Done when:** a grep finds no remaining silent returns in real-weight tests, and a run shows
  skips as skips.
- **Note:** one instance, `PreTokenizerParityTests`, was fixed in 4349db3; 19 rows started
  running.

## A few hours, cause already narrowed

### 7. GLM-4.5 (`glm4moe`) 1.9% perplexity gap
- **Status:** in progress; entry in `bugstofix.md`.
- **Steps:**
  - llama.cpp side: `llama-eval-callback` on the 326-token wikitext prompt (running), tensors
    `ffn_inp-N` / `l_out-N`.
  - Our side: `StageCapture` stages `post_attn_resid` / `post_ffn_resid` on the same ids.
  - Compare the last-token values layer by layer; fix the first layer that drifts.
- **Done when:** second-half PPL within about 0.3% of llama.cpp's 8.6125. Then allowlist entry,
  STATUS, RUNNING and MODELS rows (#16 in `docs/done/102-status-open-items-plan.md`).
- **Risk:** if the drift is spread across layers rather than starting in one, this becomes a day.

### 8. Jinja chat-template gaps
- **Steps:** support string concatenation inside a conditional (Gemma-3-4B-it and the other cases
  in [00-current-work.md](00-current-work.md) §1 item 7).
- **Done when:** those templates render identically to llama.cpp's.

### 9. Empty KV rows for layers without attention
- **Steps:** Mamba-2, short-conv and MLP-only layers call `AppendZeroKv` because `PagedKvCache`
  allocates each position's block on layer 0's append. Let those layers skip storage.
- **Done when:** the KV footprint scales with attention layers only (measured). Granite-H,
  Nemotron-H and LFM2 parity tests are unchanged.

### 10. FunASR-Nano on real speech
- **Steps:** transcribe a real clip (LibriSpeech sample already in the repo).
- **Done when:** a sensible transcript, or a bug entry if not.

## A few hours, cause unknown

Each could be a one-line fix or a day. Timebox each at half a day, then write down what was
learned and move on.

### 11. The unknown-cause set, in this order
- **a.** LFM2 0.24% PPL gap (10.9277 vs 10.9543).
- **b.** Youtu-VL: one 1024-token window +5% PPL vs llama.cpp.
- **c.** NaN in `ForwardPass`'s f16 `qwen3` path (last layer, one position).
- **d.** `HybridGdnChunkedPrefill_MatchesSequentialPrefill` fails with real weights.
- **e.** Stable Audio 3 padding masks in the APG norm.
- **f.** Classic LLaVA-1.5 (no image token in the vocab).

## A day or more

### 12. Batched prompt processing for the recurrent families
- **Scope:** Granite 4.0-H, Nemotron-H, LFM2. Today prompts go token by token: Granite 1B 17 tok/s
  vs about 110 in llama.cpp.
- **Steps:** add Mamba-2 and short-conv layers to `PrefillCore`. Batch the projections; keep the
  scan and conv sequential over tokens.
- **Done when:** parity tests unchanged; prompt tok/s measured and recorded in RUNNING.md.

### 13. MoE variants of the recurrent families
- **Scope:** Granite 4.0-H tiny/small (MoE), Nemotron-H MoE (latent MoE, sigmoid gating), LFM2-MoE
  (`lfm2moe`).
- **Done when:** each is admitted with PPL parity.

### 14. Qwen3-VL, Parakeet TDT, ACE-Step parity, CPU-only vision features
- **Qwen3-VL:** IMROPE plus the `qwen3vl` architecture.
- **Parakeet TDT:** the decode head.
- **ACE-Step 1.5 Turbo:** numeric parity and a STATUS row.
- **CPU-only vision features:** 2D M-RoPE image positions and deepstack, which exist only on the
  CPU `ForwardPass`.

### 15. Performance items
Measure first, keep only measured wins (CLAUDE.md rule 7):
- SmolLM2 prefill (0.89x llama.cpp) and Qwen3.6-35B prefill (0.63x);
- batched image-token prefill in VLMs;
- Vulkan batched prefill and matvec bandwidth;
- FLUX.1 / FLUX.2 GPU;
- MiniMax vocoder, MusicGen/AudioGen, CosyVoice3 steps;
- TTS/ASR GPU residency.

### 16. Product items
- `stingray setup` and the first-run experience;
- configuration ownership;
- multi-model runtime phases;
- session `Fork()`;
- the NuGet release checklist.

## Can't be scheduled

### 17. CPU greedy non-determinism
Never reproduced; act only on a new sighting.

### 18. Blocked on the user or on other hardware
- **CosyVoice 2** garbled endings: needs one upstream CosyVoice2-0.5B run recorded as data.
- **HunyuanVideo** numeric check: needs an independent v1 output (noise + latent).
- **Pixtral 12B / GLM-4.6V timings:** need an `HF_TOKEN` with the licence accepted.
- **Llama-4 Scout:** about 93 GB download; needs disk.
- **CUDA items**, such as `rope_freqs` for Llama-3.1-style models: need a CUDA GPU.

## Log

- 2026-09-27: plan written. Item 7 in progress (`llama-eval-callback` running).
