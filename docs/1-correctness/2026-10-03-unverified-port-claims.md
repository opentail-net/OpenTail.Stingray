# Unverified claims in the 2026-10-02/03 coverage wave

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

### Qwen 3.8 Flash Next (`qwen4exp`) - refuses real configs
| Unchecked | Why unchecked | What would settle it |
|---|---|---|
| PLE n-gram hash, shard and table lookup (no n-gram table is loaded) | Mechanism taken from the plan and the reviewer; llama.cpp `qwen4exp.cpp` not yet read for it | Read the PLE gather in `qwen4exp.cpp`, load the table, test 1/2/3-gram lookups produce distinct, expected rows |
| QSA RoPE (the mixer applies none) | Not yet compared with `build_layer_attn` | Read `build_layer_attn`; synthetic test with a known rotation |
| QSA indexer, K-pool block selection (`n_sel`, select-tail, by-order) | Reference logic is in the graph input builder; not ported | Controlled-score fixture: a distractor block outside the selection must not affect the output |
| Routed experts read the stacked tensor correctly (`ExpertMatVec` slicing, quantised row stride) | Only the router maths has a test | Integrated 2-4 expert fixture with known matrices and exact expected output |
| Fused `ffn_gate_up_exps` tensors | Not loaded; unknown whether real GGUFs use them | Inventory a real GGUF (72.5 GB, does not fit this host) |
| GDN, HC and PLE gate/conv numerics | Only finite-output and component tests exist | llama.cpp `qwen4exp` run on a synthetic GGUF |
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
| Unchecked | Why unchecked | What would settle it |
|---|---|---|
| Gumbel-max candidate sampling versus plain argmax | Claimed by a reviewer from the vLLM reference; the HF/vLLM Python source is not in the repo, the HF repo carries no code, and the model card does not mention it | Read `transformers` `DiffusionGemma*` / the vLLM sampler |
| Learned self-conditioning MLP | Tensors are not loaded; the reference is unavailable; `ApplySelfCondMlp` exists but is unused by the pipeline | Inventory the GGUF tensor names, read the HF modeling code |
| Self-conditioning probabilities use the temperature of the *current* step | Assumed | Reference step trace |
| `TemperatureMin` 0.408 versus the official 0.4 | Config says 0.4; the engine bakes in 0.408 (decay over 47 steps) | Reference step trace; then pin one contract |
| Token-selection rule: "lowest-entropy tokens such that the mutual-information bound stays under 0.1" | The engine uses a cumulative-entropy budget; equivalence is assumed | Reference step trace |
| Exact soft embedding performance | About 262k vocab x 256 positions per step | Perf pass after parity (rule 7) |
| Real-weight run | `unsloth/diffusiongemma-26B-A4B-it-GGUF` Q4_K_M downloading to `F:\_models` | Phase 9 of the plan |

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
A real checkpoint produces coherent greedy answers at about 0.4 tokens/s. Still unchecked: the comparison against
the publisher's reference and a fast ternary kernel.

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

## Update 2026-10-03 (evening): Muse-Glimmer admitted (text, CPU only)

The open items listed above for Muse-Glimmer are now closed except those that were never in scope: `admit-arch` returned ADMIT (8/8
exact); a 3,748-token prompt (past the real 2048 sliding window) was identical over 24 generated tokens; and with the window patched
to 64 in the GGUF header (restored afterwards) three ~300-token prompts agreed for 6-20 tokens before near-ties (one measured at
0.05 nat). Speed against llama.cpp on the same CPU: prefill 1.4-1.7 vs 15.2 t/s, decode 1.1-1.7 vs 2.4-2.5 t/s. Still unchecked:
the near-tie margins of the other two patched-window prompts, the vision tower, DFlash, any GPU path, and the checkpoint's license
bucket (so no real-weight parity test is committed).
