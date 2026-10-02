# DiffusionGemma port plan (`diffusion-gemma` / `diffusion_gemma`)

**Status:** not started. **Policy:** port now, prove later; not admitted, not advertised
(CLAUDE.md rule 14; [ported-families-todo](ported-families-todo.md)).

## Architecture

References: the **HF reference implementation** (`google/diffusiongemma-26B-A4B-it`) is the
independent one. No llama.cpp port exists (upstream `bed0a8566` has no `diffusion-gemma.cpp`).
Secondary: TensorSharp `docs/models/diffusiongemma.md`, `Models/DiffusionGemma/` (8 files, about
6k lines including CPU kernels and the sampler; it checked a shard against a NumPy transcription of
the HF reference).

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
- [ ] **4. Gate:** keep it out of the text-LLM path entirely. `ModelCompatibility` must refuse
  `diffusion-gemma` for `InferenceEngine` (it can't decode token by token). The new sampler entry
  point is behind an experimental flag; no server exposure until verified.

**Promotion is pipeline-specific, not `ModelCompatibility` / `admit-arch`.** Those are built
around autoregressive text generation, and DiffusionGemma isn't autoregressive (TensorSharp's
`Forward()` throws; generation is the sampler over a canvas). Promotion means:
- a real GGUF;
- reference fixture checks (canvas forward intermediates, sampler decisions);
- a deterministic end-to-end canvas test;
- exposure through a CLI entry / diffusion-style registry;
- a STATUS row.

## Deferred (not in the initial port)

Vision (Gemma 4 tower), the Jev `/v1/systemone` typed-decision endpoint, structured output,
server integration, and GPU paths.

## Verification (levels as in [ported-families-todo](ported-families-todo.md))

1. **Specification tests (level 2):** a synthetic tiny model. The canvas forward against a
   test-side reimplementation (the masks are the risk), and the sampler on a fixed seed against a
   test-side implementation of its rules. Transcription checks, not independent.
2. **Independent implementation (level 3):** the HF/PyTorch reference
   (`google/diffusiongemma-26B-A4B-it`) is the independent implementation; no llama.cpp port
   exists. A checked-in reference fixture (canvas logits for a fixed prompt, canvas and seed)
   produced from it is the target.
3. **Real weights (level 4):** coherence; canvas-logit comparison against that fixture; TensorSharp's
   pure-C# `cpu` backend on the same GGUF and seed as a second reading.

**Effort:** port + specification tests about 1 day. Real-weight verification, the closeout
performance + DRY pass, vision and pipeline promotion are separate.
