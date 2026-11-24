# DiffusionGemma port plan (`diffusion-gemma` / `diffusion_gemma`) - completed items

Ticked items moved from [docs/2-coverage/2026-10-03-diffusiongemma-port-plan.md](../2-coverage/2026-10-03-diffusiongemma-port-plan.md) on 2026-11-03. Each keeps its section heading and nested lines.

## Open (checked 2026-10-30 against the status line)

- [x] Confirm the self-conditioning MLP width (closed 2026-11-01). The real Q4_K_M GGUF (`E:\_models\diffusiongemma\diffusiongemma-26B-A4B-it-Q4_K_M.gguf`) has `self_cond_gate.weight` and `self_cond_up.weight` as `[2816, 2112]` and `self_cond_down.weight` as `[2112, 2816]` (GGUF dims, first = input), so the MLP width is 2112. `DiffusionGemmaTensorSet` checks these shapes against `SelfCondIntermediateDim = 2112` with `ValidateShape`, which throws on mismatch, so the loader rejects any other width. Read from the header with a scratchpad reader (no weights loaded). This settles the width only; the self-conditioning semantics are not checked by this box.

---

## Moved 2026-11-24 from 2026-10-03-diffusiongemma-port-plan.md

Open items stay in [2026-10-03-diffusiongemma-port-plan.md](../2-coverage/2026-10-03-diffusiongemma-port-plan.md).

_From: Strategic Outcome / Open (checked 2026-10-30 against the status line)_

- [x] Resolve what enc_layer_output_scale and layer_output_scale mean (prefill or canvas pass). Answered 2026-11-08 (coverage plan, item A2): both are the same per-layer scalar. Sources, all read without a weight download. (1) Upstream `google/diffusiongemma-26B-A4B-it`, revision `f7f5b7f5fa82ffc52addd066915886d497f5517b`: `model.safetensors.index.json` has `layer_scalar` for 30 layers under both `model.encoder.language_model.layers.N` and `model.decoder.layers.N`; the BF16 values match in all 30 layers. (2) vLLM `vllm/model_executor/models/diffusion_gemma.py`, commit `de05fd1db93f712288e654bc28a8480a67a7a8e2`: "Single Gemma4 backbone run in two modes ... Same weights, same layers"; `gemma4.py` applies one `layer_scalar` to every text layer in both modes. (3) GGUF `unsloth/diffusiongemma-26B-A4B-it-GGUF`, revision `f4183a2c7a354128d02545752303c4354d165bf0`, Q4_K_M: has both `blk.N.enc_layer_output_scale.weight` and `blk.N.layer_output_scale.weight` (F32, shape [1], N = 0..29); the values match in all 30 layers and match the upstream values. Consequence: prefill and canvas decode multiply by the same value, so TensorSharp's split (`_encScale` for prefill, `_decScale` for canvas) gives the same numbers. Recommendation for the port: read the value from the GGUF and fail when either tensor is missing or the two disagree. Do not default a missing scale to 1.0, which is what TensorSharp's loader does.
