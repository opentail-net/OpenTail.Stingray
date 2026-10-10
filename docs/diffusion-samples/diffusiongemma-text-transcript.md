# DiffusionGemma text-diffusion transcript (early wiring run)

DiffusionGemma is a **text** diffusion model (Gemma-4 26B-A4B MoE): it denoises a canvas of 256 tokens into text, so its sample is a transcript, not an image.

This is an early run, kept as proof that the real checkpoint loads and generates through the C# pipeline. It is **not a converged result**.

| | |
|---|---|
| Checkpoint | `unsloth/diffusiongemma-26B-A4B-it-GGUF`, `Q4_K_M` (15.65 GiB), downloaded 2026-10-10 |
| Backend | CPU (no GPU path used) |
| Settings | 1 block (256-token canvas), **2 denoising steps** of the default 48, seed 42, real-checkpoint override `STINGRAY_DIFFUSIONGEMMA_ENABLE_REAL=1` |
| Time | 191.6s for the block (about 95s per step), after a 0.9s load |
| Prompt | `Write one sentence about the ocean.` |

**Output (decoded, cut at the first end-of-sequence token):**

> `<|channel>thought` / `<channel|>`The vast, rhythmic expanse of the ocean mysteries mysteries mysteries mysteries mysteries beneath..

Caveats:
- Two steps is far too few, which explains the repeated "mysteries". A full 48-step run would take over an hour on this CPU.
- This run's prompt used Gemma-3 style turn markers by mistake. The model's own template (`<|turn>user ... <turn|>`) rendered correctly in a later check (16 prompt tokens, no doubled BOS), but a full-length run with it was cancelled before it finished.
- Not verified against an independent reference, so per project policy the family is not admitted or advertised yet. See the [port plan](../2-coverage/2026-10-03-diffusiongemma-port-plan.md).
