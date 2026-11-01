# Unverified claims in the 2026-10-02/03 coverage wave

> Completed items moved to [../done/2026-10-03-unverified-port-claims-completed-items-2026-10.md](../done/2026-10-03-unverified-port-claims-completed-items-2026-10.md) on 2026-10-31. This file keeps only the open items; a finished item that still has an open child stays here with it.

## Checklist

- [ ] Qwen 3.8 Flash Next: Level 4 paged real-weight run
- [ ] GLM-5.x: real-weight run and K-pool selection check
- [ ] DiffusionGemma: real Q4_K_M parity ladder against llama.cpp (backbone) and TensorSharp (canvas/sampler)
- [ ] MiniMax-H3: beyond foundation (real checkpoint)
- [ ] DeepSeek-V4: real-weight check (the synthetic test does not exercise CSA)
- [ ] DeepSeek-V4.1: real 40-layer schedule (the synthetic fixture has 2 layers)
- [ ] Bonsai2 PRISM: real-weight verification (gated behind `STINGRAY_EXPERIMENTAL_PRISM=1`)

Written 2026-10-03 after two external reviews and a code check. Everything here is something that
**could not be checked yet**, either because the reference was not available, the checkpoint does not fit,
or the check has not been run. None of it blocks the "not admitted, not advertised" policy
(CLAUDE.md rule 14); it is the list of what stands between each family and the next rung of the
status ladder in [../2-coverage/ported-families-todo.md](../2-coverage/ported-families-todo.md).

Labels: **structural** < **synthetic execution** < **synthetic parity** < **reference parity**.

## How this list was built

- **Verified against the code (2026-10-03):** Qwen routed experts were absent from the forward pass; GLM5Next
  carried the previous token's mHC streams forward; DiffusionGemma compared the re-noised canvas for
  stability and committed it; DeepSeek-V4.1 defaulted to 2048 FFN / 64 indexer heads and ignored compression
  ratios. All fixed or refused loudly (see the hub).
- **Verified against primary sources:** DeepSeek-V4.1 `config.json` (HF, `moe_intermediate_size` 2304,
  `index_n_heads` 32, YaRN factor 16, `compress_ratios`, `engram_layer_ids` [1, 14]); DiffusionGemma model card
  (stopping rule, temperature 0.8 to 0.4, entropy bound 0.1) and `generation_config.json`.
- **Not verified, only asserted by a reviewer or by a plan:** everything below.

## Per family

### Qwen 3.8 Flash Next (`qwen4exp`)
| Item | Current Status (2026-10-04) | What would settle it |
|---|---|---|
| PLE n-gram hash, head grouping & table lookup | **Resolved**: 64-bit stateful `Qwen4ExpPleHasher`, n-gram head grouping (8 bigram, 8 trigram), EOS context truncation, 160-wide row concatenation into 2560d, dilated 1D causal conv, and memory-mapped `Qwen4ExpPleRowStore` | Verified in synthetic suite; real checkpoint row gather |
| QSA RoPE & IMRoPE | **Resolved**: Indexer Q and pooled-K RoPE ($\theta = 10\text{M}$, partial rotary 0.25), main attention four-section IMRoPE (`[11, 11, 10]`) using `SimdKernels.ApplyRoPECachedNeoxPartial` | Verified in synthetic suite |
| QSA indexer & K-pool block selection | **Resolved**: Dedicated raw indexer-K cache, K-pool (kpool=4) with active tail retention, multi-head ReLU scoring, top-k block selection, sparse attention mask walk | Verified in synthetic suite |
| Routed experts & fused tensors | **Resolved**: 512-expert MoE (top-10 + 1 shared expert, intermediate 640). Both fused `ffn_gate_up_exps` and separate tensor layouts supported and verified bit-identical (`Qwen4ExpRoutedMoeTests`) | Synthetic parity tests passing |
| Forward pass refusal guard | **Resolved**: Constructor refusal guard removed; real configs with `indexer_top_k > 0` accepted (`Qwen4ExpGuardTests`) | 24 unit/synthetic tests passing |
| Real-weight run | Implementation-complete / synthetic-covered (24 tests passing); real checkpoints remain guarded per CLAUDE.md Rule 14 until verified | Level 4 paged run on `unsloth/Qwen3.8-Flash-Next-GGUF` UD-IQ1_S (~72.5 GB) against llama.cpp / TensorSharp |
| MTP, vision | Deferred | n/a |

### GLM-5.x
| Unchecked | Why unchecked | What would settle it |
|---|---|---|
| `glm5next` K-pool sparse selection | Not implemented; the forward pass throws past `indexer.top_k` keys | Port selection from `glm5-next.cpp`; sparse-vs-full oracle |
| Regression for the mHC token-state fix | Fix landed without an oracle | Two-call versus fresh-call test with sublayer outputs neutralised |
| `glm5next` KDA and mHC numerics | Smoke test only (finite logits) | Independent numeric reference or llama.cpp at `bed0a8566` on a synthetic GGUF |
| `glm-dsa` against the vendored b10306 binaries | The planned synthetic comparison was never run | Synthetic GGUF through `llama-completion`, compare logits |
| Grouped LoRA / expert-slice pointer arithmetic on quantised dtypes | Offsets assume a dense stride | Quantised-dtype fixture for every expert/group slice |
| 236 GiB checkpoint | Real-weight parity pending | GLM-5.3 Q2_K_XL is being downloaded to `E:\_models\glm-5.3` |

### DiffusionGemma
| Item | Current Status (2026-10-04) | What would settle it |
|---|---|---|
| Real GGUF tensor inventory & strict loader | **Resolved**: `DiffusionGemmaTensorSet` strictly loads and validates all 692 tensors with shape assertions (`[2816, 1408, 128]` fused gate/up, `[704, 2816, 128]` down, scales) and SWA vs Full separation | Verified in synthetic tests; real checkpoint load test |
| Gemma-4 MoE backbone & prefill | **Resolved**: Full heterogeneous SWA/Full pass, NeoX RoPE, Q/K norms, full-layer V-from-K derivation, parallel dense (2112) + routed MoE (128 experts, top-8), router & expert down scales, post-norms, dual layer scales. Multi-block prefill now attends to persistent prefix KV across blocks | Parity test against llama.cpp `gemma4.cpp` (Phase 13) |
| Learned self-conditioning MLP | **Resolved**: `self_cond_pre_norm/gate/up/down` loaded and wired into pipeline. Disabled at step 0; at step > 0 computes exact full-vocab soft embeddings -> pre-norm -> GEGLU (2112) -> down -> canvas addition -> weightless norm | Verified in synthetic suite; numerical parity with reference |
| Temperature contract | **Resolved**: Contract aligned to `TemperatureMin = 0.4f`, `TemperatureMax = 0.8f` using reference schedule $T = T_{\min} + (T_{\max} - T_{\min}) \times (\text{curStep}/\text{MaxSteps})$ ($0.8 \to 0.408333$ at step 47) | Step trace verification against TensorSharp |
| EntropyBound token acceptance | **Resolved**: `DiffusionGemmaSampler` now tracks `cumAcceptedEntropy`, accumulating entropy from accepted positions only against the 0.1 nat budget | Step trace verification against TensorSharp |
| Gumbel-max candidate sampling versus plain argmax | TensorSharp reference implements deterministic inverse-CDF multinomial sampling during denoising and argmax for final block commit; vLLM variant unverified | Reference step trace |
| Exact soft embedding performance | About 262k vocab x 256 positions per step; correct implementation present | Perf pass after real parity (rule 7) |
| Real-weight run | Implementation-complete / synthetic-covered (32 tests passing); real checkpoints remain guarded per CLAUDE.md Rule 14 until verified | Download `unsloth/diffusiongemma-26B-A4B-it-GGUF` Q4_K_M (~16.8 GB) and execute Phase 9/13 ladder |

### MiniMax-H3 (foundation only)
Unchecked: Qwen3-VL-32B layer-50 hidden-state conditioning, the 50-block DiT loader and tensor inventory,
VAE encode, reference-image and long-clip conditioning modes, and parity of the dual schedulers, AdaLN
curve and 3D RoPE against upstream. The 13 tests prove layout and pipeline plumbing only.

### DeepSeek-V4 (`deepseek4`)
| Unchecked | What would settle it |
|---|---|
| CSA overlap boundaries (the code itself calls the construction a "working hypothesis") | Ratio-4 test: tokens 0-3 give one block, 4-7 a second; compare with llama.cpp b10306 on a synthetic GGUF |
| Indexer top-k selection | Controlled-score fixture |
| 8-group output LoRA (the synthetic test uses 2 groups and ratio 0, so CSA is not exercised) | Real group count fixture; quantised-dtype group offsets |
| `rope_ext_back` | Tiny vector with known rotation |
| A real checkpoint (98 GB) | Does not fit this host |

### DeepSeek-V4.1 (`deepseek41`) - refuses real configs
| Unchecked | Why unchecked |
|---|---|
| Compressed attention for ratios 1 and 2, the V4.1 indexer (candidate 2048 x block 8, final top-k 512), index-source schedule | Not implemented; loaded tensors are unused |
| YaRN (factor 16, beta 32/1, original 65536) | Not implemented |
| Engram hash function (multiplier 1000003, mod compressed vocab) | A self-consistent test cannot prove it; needs the TensorSharp / PyTorch oracle |
| Engram on quantised tables (the lookup reads raw F32; non-F32 now throws) | Needs a dtype-aware path |
| `output_hc_*` tensors are loaded but not consumed; final norm is applied to the residual instead | Needs a reference decision |
| Layers 37-39 DSpark, three NextN layers, the real 40-layer schedule | The synthetic fixture has 2 layers |
| 335 GB checkpoint | Does not fit even the scratch disk; stays written but not exposed |

### Muse-Glimmer
The synthetic test is **synthetic parity**, but its reference is test-side code written from the same understanding
as the port (level 2, not independent). Unchecked: real-weight parity against llama.cpp `muse-glimmer.cpp`
(reference built at `examples/llama.cpp/llama.cpp/build-ref`, checkpoint `UD-Q4_K_XL` downloading), the
1e-8 post-norm and 3.87 / 1.0 Q/K scaling on real tensors, and the vision tower and DFlash drafter (deferred).

### Bonsai2 PRISM / Q1_0
A real checkpoint produces coherent greedy answers at about 0.4 tokens/s. Still unchecked: a fast ternary kernel, and a teacher-forced golden.

**Reference comparison, 2026-11-01 (plan 5 B2).** The publisher's own reference needs no fork build: the Bonsai model card says stock llama.cpp runs `Ternary-Bonsai-2-27B-Q2_0.gguf` from release b11556 onward (the dev repo's `...-prism-fork-required` file and the `PTQ1_0` file we had on `F:` use a fork-private type id, 143, which stock rejects: "invalid ggml type 143"). So the stock-compatible `Q2_0` file (7,626,008,928 bytes, from `prism-ml/Ternary-Bonsai-2-27B-gguf`) was pulled to `E:\_models\bonsai2`, and the stock `llama-b11556-bin-win-cpu-x64` `llama-completion` was run against it as the independent reference (`-t 8`, greedy, `--top-k 1`). Our CLI ran the same file with `STINGRAY_EXPERIMENTAL_PRISM=1`, `--temp 0 --repeat-penalty 1.0` and an identity chat template (`{{ messages[0].content }}`, so the prompt is raw, as in the reference), CPU.

| Prompt | New tokens | Result |
|---|---|---|
| `The capital of France is` | 16 | identical text: `Paris.` / `The capital of Germany is Berlin.` / `The capital of Italy is` |
| `Water boils at 100 degrees Celsius at sea level because` | 24 | identical text: `the vapor pressure of water equals the atmospheric pressure. At higher altitudes, the atmospheric pressure is lower, so water boils` |

Timing (rule 12, wall clock including load): reference 26.9 s and 41.4 s; ours 48.0 s and 82.9 s (0.5 t/s decode, prefill included). Both runs loaded real weights (7.08 GiB pre-faulted).

This is evidence, not admission: two short greedy prompts, text compared rather than token ids or logits. Rule 14 still applies, so the family stays internal and gets no STATUS row. What would admit it: the `admit-arch` teacher-forced golden against this reference build, a perplexity comparison, and a decision on the 0.4 t/s speed.

## Cross-cutting
- **Provenance:** there is no DeepSeek-V4.1 attribution in the TensorSharp section of `THIRD_PARTY_NOTICES.md`,
  although the port follows TensorSharp. Not audited per family (copied, translated, or independent).
- **MTP / NextN, vision, GPU paths:** deferred everywhere; none is tested.
- **Per-class timing (rule 12):** the new synthetic tests run in milliseconds by design. A green real-weight test
  that also takes milliseconds has not run against weights.
- **Host memory:** families whose smallest quant exceeds 64 GB (Qwen 72.5 GB, DeepSeek 98/335 GB) can only be
  verified with paged, hours-long, correctness-only runs; V4.1 is excluded outright.

## Update 2026-10-03 (later): Muse-Glimmer checked against llama.cpp

`UD-Q4_K_XL` (14.8 GB) on CPU, greedy, raw prompt (`--chat-template "{{ messages[0]['content'] }}"`,
`--repeat-penalty 1.0`) against llama.cpp `bed0a8566` (`build-ref`, CPU, `-no-cnv`):
- "The capital of France is": identical for 20 tokens; the next token is a 0.006-nat near-tie (" in" NLL 0.7108
  vs " proper" 0.7169 in our engine), so the divergence is quantised-accumulation noise, not a model bug.
- "Q: What is 17 times 23? ...": identical through "340+51=391. So".
- "def fibonacci(n):": identical through `else:`.
Still unchecked for Muse-Glimmer: vision tower, DFlash, GPU path, long context beyond the SWA window (prompts here are
under 40 tokens, so the sliding-window mask on real weights is not exercised), admit-arch verdict, timed runs.

Two harness findings from this run (neither is a model bug):
- The CLI's default `--repeat-penalty 1.1` changes greedy output versus llama.cpp's default; always pass `1.0` for parity.
- Muse-Glimmer's real chat template failed in our Jinja evaluator (list `+` list unsupported). Fixed with a regression
  test; the real template renders now, but it has not been compared against a reference rendering.

## Update 2026-10-03 (later still): DiffusionGemma checked against the real checkpoint's tensor inventory

`unsloth/diffusiongemma-26B-A4B-it-GGUF` Q4_K_M (692 tensors) was inventoried and compared with the port and with
llama.cpp's `gemma4.cpp` (same backbone). The port does **not** match the real architecture, and the mismatch was silent:
- **Tensor names:** the port looks for `self_cond.gate/up/down.weight`, `attn_post_norm`, `ffn_post_norm`, `ffn_gate_exps`,
  `ffn_up_exps` (all `Optional`). The file has `self_cond_pre_norm/gate/up/down`, `post_attention_norm`, `post_ffw_norm`,
  `post_ffw_norm_1`, `post_ffw_norm_2`, `pre_ffw_norm_2`, fused `ffn_gate_up_exps`, `ffn_down_exps.scale`,
  `ffn_gate_inp.scale`, `layer_output_scale`, `enc_layer_output_scale`, `rope_freqs`. None would load.
- **Architecture:** no RoPE and no V-norm anywhere; `PrefillPrompt` computes K/V and an FFN per token but never applies
  attention (prompt tokens do not attend to each other); the FFN is not the Gemma-4 layer (dense MLP and expert FFN in
  parallel, each with its own norm, router logits from `rms_norm(attn_out)/sqrt(n_embd) * ffn_gate_inp.scale`, top-8 softmax
  with renormalisation, per-expert down scale, summed, `post_ffw_norm`, residual, `layer_output_scale`).
- **Containment (done):** the forward pass now refuses a checkpoint carrying those markers
  (`DiffusionGemmaRealCheckpointGuardTests`).
- **Unknown, needs the HF/vLLM source:** what `enc_layer_output_scale` vs `layer_output_scale` mean (prefill vs canvas pass?),
  the self-conditioning pre-norm placement, and Gumbel-max sampling. llama.cpp has no diffusion-gemma model file.
- **Likely route to a real check:** rewrite the backbone against `gemma4.cpp`, then verify its causal-prefill path by
  re-labelling a copy of the GGUF as `gemma4` for llama.cpp; the canvas/self-conditioning path still needs the HF reference.

## Update 2026-10-04: DiffusionGemma architecture implementation jump & synthetic verification

The architectural rewrite and tensor contract alignment were completed across `src/OpenTail.Stingray.Diffusion/DiffusionGemma/`:
- **Real GGUF tensor inventory & geometry loader:** `DiffusionGemmaTensorSet.cs` strictly loads all 692 tensors, validates SWA vs Full rules (`attn_v` required on SWA, forbidden on Full), and enforces shape validation on fused expert weights `[2816, 1408, 128]`, down weights `[704, 2816, 128]`, scales, and projections.
- **Gemma-4 MoE backbone:** heterogeneous 25 SWA / 5 Full layers, NeoX RoPE at absolute token positions, per-head Q/K RMSNorm, full-layer V-from-K derivation, parallel dense FFN (2112) + routed MoE (128 experts, top-8), router scaling, expert down scaling, post-norms, dual layer scales (`enc_layer_output_scale` in prefill vs `layer_output_scale` in canvas), and final logit softcapping.
- **Causal multi-block prefill:** `PrefillPrompt` cross-attends across cached prefix (`_promptKCache`/`_promptVCache` with SWA window clipping or full context) and newly committed blocks with correct RoPE position offsets, enabling continuous multi-block generation.
- **Self-conditioning MLP:** fully wired into pipeline; disabled at step 0; at step > 0 computes exact full-vocabulary soft embeddings, pre-norm, GEGLU (2112 intermediate), down projection, canvas addition, and weightless RMSNorm.
- **EntropyBound sampler:** corrected to track `cumAcceptedEntropy` against the 0.1 nat budget; temperature schedule pinned to reference formula $T = 0.4 + 0.4 \times (\text{curStep}/48)$ ($0.8 \to 0.408333$ at step 47).
- **Synthetic test suite:** expanded to 32 tests (`DiffusionGemmaTests`) verifying tensor geometry, SWA/Full separation, V-from-K, router scaling, post-norm/scaling order, global vs window attention, self-conditioning step-0 behavior, multi-block prefix continuity, and deterministic sampling.
- **Remaining for admission:** download real 16.8 GB Q4_K_M GGUF to `F:\_models` (or `E:\_models`), execute Phase 9/13 parity ladder against llama.cpp (backbone) and TensorSharp (canvas/sampler). Checkpoints remain guarded per CLAUDE.md Rule 14 until verified.

## Update 2026-10-03 (evening): Muse-Glimmer admitted (text, CPU only)

The open items listed above for Muse-Glimmer are now closed except those that were never in scope: `admit-arch` returned ADMIT (8/8
exact); a 3,748-token prompt (past the real 2048 sliding window) was identical over 24 generated tokens; and with the window patched
to 64 in the GGUF header (restored afterwards) three ~300-token prompts agreed for 6-20 tokens before near-ties (one measured at
0.05 nat). Speed against llama.cpp on the same CPU: prefill 1.4-1.7 vs 15.2 t/s, decode 1.1-1.7 vs 2.4-2.5 t/s. Still unchecked:
the near-tie margins of the other two patched-window prompts, the vision tower, DFlash, any GPU path, and the checkpoint's license
bucket (so no real-weight parity test is committed).

## Scout dossiers, 2026-11-01 (plan 5 B1): which families fit this machine

`stingray scout -r <repo> --quants --budget 60GB -c 4096 --format json` (scout build 1.0.9+8eb7c90b), one dossier per family, no checkpoint downloaded. Host: 63.3 GiB RAM; scout reserves 8 GiB, so a file fits when its estimated peak is about 52 GiB or less. Scout answers "Unsupported" for every row of a NotAdmitted family whatever its size, so the verdict column below is read from the size and the peak, not from scout's word. MiniMax-H3 is SafeTensors, outside scout's GGUF scope (plan 5 B3).

| Family | Repo @ revision | Smallest quant | Size | Estimated peak | Fits 63.3 GiB? |
|---|---|---|---|---|---|
| Qwen 3.8 Flash Next | `unsloth/Qwen3.8-Flash-Next-GGUF` @ 766911a6 | UD-IQ1_S (3 shards) | 67.6 GiB | 70.5 GiB | **No** |
| GLM-5.x (`glm-dsa`) | `unsloth/GLM-5.3-GGUF` @ 346b3591 | UD-IQ1_S (6 shards) | 201.8 GiB | 245.1 GiB | **No** (the Q2_K_XL on `E:\_models\glm-5.3` is 236.4 GiB, peak 279.2 GiB, complete but cannot be run) |
| DiffusionGemma | `unsloth/diffusiongemma-26B-A4B-it-GGUF` @ f4183a2c | Q4_K_M | 15.7 GiB | 21.5 GiB | **Yes** (needs `DiffusionGemmaPipeline`, not the autoregressive path) |
| DeepSeek V4 Flash | `unsloth/DeepSeek-V4-Flash-0731-GGUF` @ fbbb5b93; `antirez/deepseek-v4-gguf` @ f71f23d5 | UD-IQ1_S (3 shards); IQ2XXS-w2Q2K | 76.9 GiB; 80.8 GiB | 80.4 GiB; 83.7 GiB | **No** |
| DeepSeek V4.1 Flash | `unsloth/DeepSeek-V4.1-Flash-1017-GGUF` @ 8a6ec9cf; `antirez/deepseek-v4.1-flash-gguf` @ 9dcee516 | UD-IQ1_S (5 shards); Q2 | 217.0 GiB; 340.6 GiB | 220.3 GiB; not estimated | **No** |
| Bonsai2 PRISM | `prism-ml/Ternary-Bonsai-2-27B-gguf-dev` @ 1dc2d993 | Q2_0 (`...-prism-fork-required.gguf`) | 7.1 GiB | 9.8 GiB | **Yes**; a copy (`Ternary-Bonsai-2-27B-PTQ1_0.gguf`, 5.9 GB) is already on `F:\_models\bonsai2` |
| MiniMax-H3 | not GGUF | n/a | n/a | n/a | not scouted (B3) |

Corrections this scouting made to earlier statements:
- "V4 (98.6 GB) can be paged from disk and checked against b10306" (table in the tensorsharp plan): the smallest V4 Flash files are 76.9 and 80.8 GiB, over the 63.3 GiB host, so by CLAUDE.md rule 15 the run is not made. No 98.6 GB file exists in the two repos scouted.
- Qwen 3.8 Flash Next "smallest quant ~72.5 GB" is the decimal size of UD-IQ1_S (67.6 GiB); it does not fit either way.
- V4.1: three of the unsloth quants (Q8_K_XL, Q4_K_XL, Q3_K_XL) carry `per_layer_token_embd.weight` with GGML type id 43, which this build does not support; the smaller IQ quants do not and are 217 GiB or more.
- The `Q2_0` in the Bonsai2 dev repo is the only file listed; scout reads `general.architecture` (`qwen35`, admitted) and cannot see the `STINGRAY_EXPERIMENTAL_PRISM` gate, so its "Fits" is about size only.
- The dossiers (JSON) are in the 2026-11-01 session scratchpad, not in the repository.