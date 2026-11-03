# DiffusionGemma port plan (`diffusion-gemma` / `diffusion_gemma`) - completed items

Ticked items moved from [docs/2-coverage/2026-10-03-diffusiongemma-port-plan.md](../2-coverage/2026-10-03-diffusiongemma-port-plan.md) on 2026-11-03. Each keeps its section heading and nested lines.

## Open (checked 2026-10-30 against the status line)

- [x] Confirm the self-conditioning MLP width (closed 2026-11-01). The real Q4_K_M GGUF (`E:\_models\diffusiongemma\diffusiongemma-26B-A4B-it-Q4_K_M.gguf`) has `self_cond_gate.weight` and `self_cond_up.weight` as `[2816, 2112]` and `self_cond_down.weight` as `[2112, 2816]` (GGUF dims, first = input), so the MLP width is 2112. `DiffusionGemmaTensorSet` checks these shapes against `SelfCondIntermediateDim = 2112` with `ValidateShape`, which throws on mismatch, so the loader rejects any other width. Read from the header with a scratchpad reader (no weights loaded). This settles the width only; the self-conditioning semantics are not checked by this box.
