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
- [x] **1. Commit the GLM entry in bugstofix.md** (DONE 2026-09-27, `99b6b53`) — archived to [done/103-quickest-first-plan-closed-items.md](done/103-quickest-first-plan-closed-items.md)

### 2. Rerun the real-weight landscape sweep
- [x] **2. Rerun the real-weight landscape sweep** (DONE 2026-09-28, `scripts/sweep-tests.ps1`) — details archived to [done/103-quickest-first-plan-closed-items.md](done/103-quickest-first-plan-closed-items.md)

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
- [x] **4. Gemma 4 E4B vision (`gemma4v`) parity** (DONE 2026-09-27) — details archived to [done/103-quickest-first-plan-closed-items.md](done/103-quickest-first-plan-closed-items.md)

### 5. Wire the ONNX speech-to-text pipelines into `stingray stt`
- [x] **5. Wire the ONNX speech-to-text pipelines into `stingray stt`** (DONE 2026-09-27, `4874ea4`) — details archived to [done/103-quickest-first-plan-closed-items.md](done/103-quickest-first-plan-closed-items.md)

### 6. Finish the silent-no-op test sweep
- [x] **6. Finish the silent-no-op test sweep** (DONE 2026-09-28) — details archived to [done/103-quickest-first-plan-closed-items.md](done/103-quickest-first-plan-closed-items.md)

---

## A few hours, cause already narrowed

### 7. GLM-4.5 (`glm4moe`) Q5_K×Q8_K kernel and gated decode path
- [x] **7. GLM-4.5 (`glm4moe`) Q5_K×Q8_K kernel and gated decode path** (implementation and synthetic-validation scope complete; real-weight evaluation split to item 19) — details archived to [done/103-quickest-first-plan-closed-items.md](done/103-quickest-first-plan-closed-items.md)

### 8. Jinja chat-template gaps
- [x] **8. Jinja chat-template gaps** (DONE 2026-09-27, `4ed6866`) — details archived to [done/103-quickest-first-plan-closed-items.md](done/103-quickest-first-plan-closed-items.md)

### 9. Empty KV rows for layers without attention
- [x] **9. Empty KV rows for layers without attention** (DONE 2026-09-27, `c4cea41`) — details archived to [done/103-quickest-first-plan-closed-items.md](done/103-quickest-first-plan-closed-items.md)

### 10. FunASR-Nano on real speech
- [x] **10. FunASR-Nano on real speech** (DONE 2026-09-27 as a bug entry, `fea7873`) — details archived to [done/103-quickest-first-plan-closed-items.md](done/103-quickest-first-plan-closed-items.md)

---

## A few hours, cause unknown

Timebox each at half a day, write down what was learned, and move on if blocked.

- [x] **11. Unknown-cause set** (all initial sub-items 11.a–11.f resolved) — details archived to [done/103-quickest-first-plan-closed-items.md](done/103-quickest-first-plan-closed-items.md)
  - [x] **11.a** LFM2 0.24% PPL gap (CLOSED 2026-10-01)
  - [x] **11.b** Youtu-VL: one 1024-token window +5% PPL vs llama.cpp (DONE 2026-09-27, `9568823`)
  - [x] **11.c** NaN in `ForwardPass`'s f16 `qwen3` path (DONE 2026-09-27)
  - [x] **11.d** `HybridGdnChunkedPrefill_MatchesSequentialPrefill` failure with real weights (DONE 2026-09-28)
  - [x] **11.e** Stable Audio 3 padding masks in the APG norm (DONE 2026-09-28)
  - [x] **11.f** Classic LLaVA-1.5 missing image token / ordering (DONE 2026-09-27)

---

## A day or more

### 12. Batched prompt processing for the recurrent families
- [x] **12. Batched prompt processing for the recurrent families** (DONE 2026-09-28) — details archived to [done/103-quickest-first-plan-closed-items.md](done/103-quickest-first-plan-closed-items.md)

### 13. MoE variants of the recurrent families
- [ ] **13. MoE variants of the recurrent families**
  - [x] Admit Granite 4.0-H tiny/small (MoE) with PPL parity. Small DONE 2026-09-28 (top-k renormalisation fix; +1.2% residual at -c 2048 logged).
  - [ ] Admit Nemotron-H MoE (latent MoE, sigmoid gating) with PPL parity. No local checkpoint (2026-09-28); not downloaded (C: has no free space, K: ~10 GB).
  - [ ] Admit LFM2-MoE (`lfm2moe`) with PPL parity. BLOCKED 2026-09-28 (timeboxed): PPL +6.3% and per-token vs batched disagree by 9%; see bugstofix.
  - [ ] **Done when:** each variant admitted with PPL parity against reference.

### 14. Architectural additions & missing features
- [x] **14. Architectural additions & missing features** (DONE 2026-09-28) — details archived to [done/103-quickest-first-plan-closed-items.md](done/103-quickest-first-plan-closed-items.md)

### 15. Performance items
- [x] **15. Performance items** (CLOSED for this machine 2026-09-28; discrete-GPU items moved to 18) — details archived to [done/103-quickest-first-plan-closed-items.md](done/103-quickest-first-plan-closed-items.md)

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
- **Q4_K Q8KS candidate ruled out 2026-10-01** (Maincoder-1B Q4_K_M: 192 Q4_K + 33 Q6_K; llama.cpp `PPL = 12.0051 +/- 1.04898`; same paired method): default Q8KS (per-32 scales) `ppl` 11.9204, paired dNLL -0.00708 (-2.9 SE), rms **0.0771**; `STINGRAY_LEGACY_DOTQ4K=1` (plain F32 activations, no quantization) 11.9418, dNLL -0.00529, rms **0.0765**. Identical floor with and without activation quantization, so the Q4_K activation format is not a source of the gap and a ggml-exact `DotQ4K_Q8K` is not needed for correctness. Floor scales with the model's quantization noise (Maincoder Q4_K_M 0.077, SmolLM2 Q5_K_M 0.142, GLM-4.5 Q2_K-heavy 0.204). Remaining small systematic offset (ours ~0.005-0.008 nats lower than llama.cpp on the dense models, 2-3 SE) is unexplained.
  - [x] **19b. Broader real-weight Q5_K inventory** with the gate on, against existing golden/parity expectations, before any default change; plus a gate-on/off speed measurement (one process at a time, several runs) for `PerformanceLeague.md`.
- **Measured 2026-10-01 (CLOSED: gate stays off by default).** Paired per-token NLL vs llama.cpp (`--save-all-logits` / `--dump-nll`), gate off -> on, per-token rms: GLM-4.5 0.328 -> 0.204; phi-2 0.121 -> 0.091; Phi-3-mini 0.127 -> 0.112; jais-590M 0.205 -> 0.169; Ornith-9B 0.066 -> 0.063; SmolLM2 0.142 -> 0.148 (the one regression). Paired mean dNLL stays within about 2 SE of zero everywhere (phi3 +0.0043 -> +0.0076, +2.2 SE; jais -0.0179 -> -0.0053), no token mismatches. So fidelity improves on 5 of 6 models at no systematic cost. **Speed (alone, alternating off/on, 5 runs each, CPU, 96-token decode, Release): SmolLM2-1.7B Q5_K_M 21.0 -> 13.3 t/s median (-37%); phi-2 11.7 -> 9.8 t/s (-16%).** The gate is not worth that decode cost as a default; it stays an opt-in for Q5_K-heavy MoE models like GLM-4.5 where the fidelity gain is large. Not covered: the two 27B Qwen UD files (too slow per-token), gemma-4-E4B (1 Q5_K tensor), and GPU/hybrid paths. A faster `DotQ5K_Q8K` would change the trade-off.
  - [ ] **Done when:** real-weight evidence is recorded; only if PPL is within ~0.3% of llama.cpp should GLM-4.5 admission and STATUS/RUNNING/MODELS updates be considered. Keep the gate off by default until then. See the [closed implementation record](done/09-glm45-q5k-activation-implementation.md) and [the detailed plan](1-correctness/09-glm45-q5k-activation-quant-plan.md).

---

## Log

> Historical log entries from 2026-09-27 through 2026-09-28 are archived in [done/103-quickest-first-plan-closed-items.md](done/103-quickest-first-plan-closed-items.md).
