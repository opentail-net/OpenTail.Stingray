# DiffusionGemma port plan (`diffusion-gemma` / `diffusion_gemma`)

**Status:** not started. **Policy:** port now, prove later; not admitted, not advertised
(CLAUDE.md rule 14; [ported-families-todo](ported-families-todo.md)).

## Architecture

Source: TensorSharp `docs/models/diffusiongemma.md`, `Models/DiffusionGemma/` (8 files, about 6k
lines including CPU kernels and the sampler). Not in llama.cpp; TensorSharp checked a shard against a
NumPy transcription of the HF reference, but no llama.cpp output is recorded.

A **block text-diffusion** language model on a Gemma-4-style MoE backbone (26B-A4B). It isn't
autoregressive: `Forward(token)` is invalid, and generation runs through a sampler.
- Each denoising step runs over `[prompt | canvas]`. The prompt side is causal and never attends to
  the canvas; the canvas is bidirectional over prompt + canvas.
- Per step: region-aware embedding scale, then prompt/canvas masks, then N Gemma layers.
  - Each layer: local/global QK-norm attention, a dense gated-GELU MLP **and** top-k MoE experts,
    and a prompt-encoder / canvas-decoder scale.
  - Then the output norm, the tied LM head and the final logit softcap.
- Optimized: prefill the prompt K/V once, then repeat canvas decodes reusing it.
- Sampler (`DiffusionEbParams`): 48 max steps, temperature schedule 0.4-0.8, entropy bound 0.1,
  stability/confidence early stop, a seed, block-autoregressive canvases.
- Image input through Gemma 4's vision tower.

Weights: `google/diffusiongemma-26B-A4B-it`, `unsloth/diffusiongemma-26B-A4B-it-GGUF` (about
13-17 GB). **This fits the machine.**

## Reuse in Stingray

- The Gemma 4 forward pieces (per-layer head dims, SWA/global, QK-norm, sandwich norms, softcap),
  the MoE router and experts, and the Gemma 4 vision tower (`UnifiedVisionPipeline`).
- `PagedKvCache` for prompt K/V.

## New work (phases)

- [ ] **0. Read** `DiffusionGemmaModel.cs`, `.Cpu.cs`, `DiffusionGemmaSampler.cs`,
  `DiffusionComputeTurns.cs`. Write the exact mask rules, region scales and sampler algorithm here.
- [ ] **1. Hyperparams + tensors** (`ModelGraph` branch; the dense MLP + MoE per layer).
- [ ] **2. Canvas forward:** a new `DiffusionGemmaForwardPass` with `ForwardCanvas(tokens,
  promptLen)` (unified path) using the prompt-causal / canvas-bidirectional mask. Then
  `PrefillPrompt` + `DecodeCanvas` with prompt-KV reuse.
- [ ] **3. Sampler:** `DiffusionGemmaSampler` (entropy-bounded acceptance, re-noising,
  temperature schedule, early stop, multi-block). Plus a CLI/engine entry, since it doesn't fit
  `InferenceEngine`'s token loop.
- [ ] **4. Vision** (later).
- [ ] **5. Gate:** a `// diffusion-gemma — NOT admitted` block; no server exposure until verified.

## Verification

1. Synthetic tiny model: the canvas forward vs a spec-written reference (masks are the risk), and
   the sampler on a fixed seed vs a reference implementation of its rules.
2. Real checkpoint: coherence of generated text; token-level comparison with TensorSharp on the
   same GGUF and seed (its pure-C# `cpu` backend runs here).
3. Admission via the normal path.

**Effort:** about 1 day to ported + synthetic-verified; more for vision and server integration.
