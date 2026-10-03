# DiffusionGemma port plan (`diffusion-gemma` / `diffusion_gemma`)

**Status (2026-10-03, revised after inspecting the real checkpoint): NOT DONE.** What exists is a sampler, a block
loop and component tests built on guessed tensor names and a simplified backbone. The real checkpoint is **refused**
by the forward pass (`DiffusionGemmaRealCheckpointGuardTests`) because running it would silently use a degraded model.
Earlier docs that called this family "Done" or "90%" were wrong; see
[the unverified-claims note](../1-correctness/2026-10-03-unverified-port-claims.md).

**Policy:** port now, prove later; not admitted, not advertised (CLAUDE.md rule 14;
[ported-families-todo](ported-families-todo.md)).

## Architecture & Upstream References

**Primary & secondary references:**
- **Hugging Face:** `google/diffusiongemma-26B-A4B-it`. The repo carries only weights and configs; the modeling and
  sampler code lives in `transformers` / `diffusers` and is **not available locally yet** (needed for everything marked
  OPEN below).
- **llama.cpp `src/models/gemma4.cpp`** (local, `bed0a8566`): the same backbone as a causal LM. There is **no
  diffusion-gemma model file** in llama.cpp; its `examples/diffusion` runner is the generic LLaDA/Dream one.
- **Unsloth GGUF** `unsloth/diffusiongemma-26B-A4B-it-GGUF` (Q4_K_M, 692 tensors, `E:`/`F:\_models`): the tensor-name
  ground truth below.
- **TensorSharp:** `docs/models/diffusiongemma.md`, `Models/DiffusionGemma/` (not re-read in this revision).

**DiffusionGemma** is a **block text-diffusion** model on a Gemma-4 MoE backbone (26B total, ~4B active). It is **not
autoregressive**: generation iteratively denoises a 256-token canvas.

---

## Facts verified on the real GGUF (2026-10-03)

Metadata (`diffusion-gemma.*`): 30 layers, hidden 2816, 16 heads, `head_count_kv` per layer (list), `key_length` 512 /
`key_length_swa` 256, `sliding_window` 1024 with a per-layer `sliding_window_pattern`, dense `feed_forward_length` 2112,
`expert_count` 128, `expert_used_count` 8, `expert_feed_forward_length` 704, `rope.freq_base` 1e6 (full) / 1e4 (SWA),
`rope.dimension_count` 512 / 256, `final_logit_softcapping` 30, `vocab_size` 262144, `diffusion.canvas_length` 256,
`attention.causal` false, `shared_kv_layers` 0, `embedding_length_per_layer_input` 0.

Tensors outside the blocks: `token_embd.weight` (Q6_K, tied output), `output_norm.weight`, `rope_freqs.weight` [256],
`self_cond_pre_norm.weight` [2816], `self_cond_gate.weight` [2816 to 2112], `self_cond_up.weight` [2816 to 2112],
`self_cond_down.weight` [2112 to 2816].

Per block (sliding layer 0 shown): `attn_norm`, `attn_q/k/v/output`, `attn_q_norm`, `attn_k_norm`, `post_attention_norm`,
`ffn_norm`, `ffn_gate/up/down` (dense MLP, 2112), `post_ffw_norm_1`, `pre_ffw_norm_2`, `ffn_gate_inp.weight` [2816, 128],
`ffn_gate_inp.scale` [2816], `ffn_gate_up_exps.weight` (**fused**, [2816, 1408, 128]), `ffn_down_exps.weight`
[704, 2816, 128], `ffn_down_exps.scale` [128], `post_ffw_norm_2`, `post_ffw_norm`, `layer_output_scale`,
`enc_layer_output_scale`.

**Full-attention layers (5, 11, 17, 23, 29):** `attn_q` [2816, 8192] (16 x 512), `attn_k` [2816, 1024] (2 x 512),
`attn_output` [8192, 2816], q/k norms [512], and **no `attn_v` tensor** (25 `attn_v` for 30 layers). V is derived from K on
these layers, as in Gemma 4. Sliding layers: Q 16 x 256, KV 8 x 256.

---

### 1. Model geometry & layer schedule (30 layers)

| Parameter | Value |
|---|---|
| Hidden dimension | D = 2816 |
| Layers | 30: **5 full** (5, 11, 17, 23, 29) + **25 sliding** (rest) |
| Sliding layers | Q 16 heads, KV 8 heads, head_dim 256, window 1024, RoPE base 1e4 |
| Full layers | Q 16 heads, KV 2 heads, head_dim 512, global, RoPE base 1e6, `rope_freqs.weight` mask, **no V projection** |
| Embedding scaling | sqrt(D) = 53.0659966 on token embeddings and soft self-conditioning embeddings |
| Dense FFN | 2112, GELU-gated, parallel with the experts (not instead of them) |
| MoE | 128 experts, top-8, expert width 704, fused gate+up, softmax gating, renormalised weights, per-expert down scale |
| Embeddings | tied input/output; final logit softcapping 30 |

### 1b. Layer block contract (from llama.cpp `gemma4.cpp`, MoE layers)

```
h      = rms_norm(x, attn_norm)
q,k    = proj, per-head rms_norm(q_norm / k_norm); v = proj (or K on full layers), plain rms_norm(v) (no weight)
q,k    = RoPE (per-layer base; rope_freqs on full layers)
a      = attention(q, k, v)                       ; sliding / full mask, see section 3
attn   = x + rms_norm(a @ Wo, post_attention_norm)
mlp    = rms_norm( down( gelu(gate(rms_norm(attn, ffn_norm))) * up(...) ), post_ffw_norm_1 )
logits = ffn_gate_inp @ ( rms_norm(attn) / sqrt(D) * ffn_gate_inp.scale )        ; router sees attn, not the normed input
moe    = rms_norm( sum_topk w_e * ( down_e( gelu(gate_e(h2)) * up_e(h2) ) * down_scale_e ), post_ffw_norm_2 )
                 with h2 = rms_norm(attn, pre_ffw_norm_2), w = softmax over all experts, top-8, renormalised, scale 1.0
out    = (attn + rms_norm(mlp + moe, post_ffw_norm)) * layer_output_scale
```

### 2. Self-conditioning

Soft probabilities `P = softmax(logits_{t-1} / T)`, soft embedding `E = P . W_embed * sqrt(D)`, then the learned MLP, then
`canvas += E_sc`.
- **Known from the GGUF:** the MLP is `self_cond_pre_norm` (RMS) then `self_cond_gate`, `self_cond_up` (2816 to 2112) and
  `self_cond_down` (2112 to 2816).
- **OPEN (needs HF source):** where the pre-norm sits, the activation (GELU assumed), whether the output is scaled, and
  how step 1 (no previous prediction) is seeded.

### 3. Attention mask contract & prefix KV reuse

- **Prompt prefill (persistent):** causal; each prompt token attends to earlier prompt tokens **at every layer**, and the
  per-layer attention output feeds the next layer. Builds the persistent prefix KV cache.
- **Canvas denoising (256 tokens):** bidirectional over the canvas; canvas attends to the prompt KV (full layers: all
  positions; sliding layers: the last `min(prompt_len, W - 1)` positions, as the earlier plan stated, **unverified**).
- **OPEN:** `enc_layer_output_scale` versus `layer_output_scale`: presumably the prefill ("encoder") pass uses the former
  and the canvas ("decoder") pass the latter. Not confirmed.

### 4. Sampler (`DiffusionGemmaSampler`)

| Parameter | Value |
|---|---|
| Canvas length | 256 |
| Max denoising steps | 48 |
| Temperature | linear 0.8 to 0.4 (model card, `generation_config.json`); the code uses 0.408 at step 47 of a 47-step decay, **contract to be pinned** |
| Entropy bound | 0.1 nats |
| Stop | mean entropy < 0.005 and the highest-probability predictions identical across two consecutive steps (model card) |

Implemented and fixed 2026-10-03: stability compares the **argmax** history (not the re-noised canvas); the committed block
is the argmax prediction. **OPEN:** Gumbel-max candidate sampling (claimed from the vLLM reference, not confirmed);
whether the token-selection rule ("lowest-entropy tokens such that the mutual-information bound stays under 0.1") equals
the cumulative-entropy budget used here.

### 5. Block-autoregressive canvas lifecycle

Prompt, causal prefill, then repeat { 256-token canvas denoise loop with self-conditioning (up to 48 steps), commit the
argmax block, causally prefill the committed block into the persistent KV cache, next canvas } until the target length
or EOS.

### 6. Components

`DiffusionGemmaForwardPass` (weights, prefill, canvas pass, logits), `DiffusionGemmaSampler`,
`DiffusionGemmaSelfConditioning`, `DiffusionGemmaPipeline` (block lifecycle). `DiffusionGemmaState` was planned and does
not exist; state is local variables in the pipeline.

---

## Hardware constraint & fitting

Q4_K_M is 16.8 GB (verified), Q5_K_M ~19.1 GB, Q8_0 ~26.9 GB; all fit the 64 GB host. The checkpoint was downloaded to
`F:\_models\diffusiongemma` and is kept until the guard test is green, then deleted.

---

## Implementation phases (honest state)

Legend: [x] done and checked, [~] partly there but built on wrong assumptions or unverified, [ ] not done.

- [~] **Phase 0: Contract freeze.**
  - [x] Tensor inventory and geometry frozen from the real GGUF (section "Facts verified").
  - [x] Layer block contract taken from `gemma4.cpp` (section 1b).
  - [ ] OPEN items above need the HF/vLLM source: self-conditioning details, `enc_layer_output_scale`, Gumbel-max,
        temperature minimum, token-selection rule.
- [ ] **Phase 1: Tensor set and loader.** The existing loader (`DiffusionGemmaLayerTensors`, reused DeepSeek4 refs) uses
  guessed names, treats every post-norm and MoE tensor as `Optional`, requires `attn_v` on all layers (the real full layers
  have none) and cannot read fused `ffn_gate_up_exps`. Rewrite as `DiffusionGemmaTensorSet` with `Required` real names and
  per-layer-kind validation, so a name mismatch can never be silent again.
- [ ] **Phase 2: Persistent prompt prefill.** Existing code builds K/V per token but never applies attention, so prompt
  tokens do not attend to each other and layer inputs lack attention. Needs RoPE, V-norm, causal attention, the full
  section 1b block, and the encoder-pass scale.
- [ ] **Phase 3: Canvas forward pass.** Existing code has bidirectional canvas + prefix attention structure but no RoPE, no
  V-norm, a dense+experts FFN that is not the Gemma-4 layer (no router scaling, per-expert scales, parallel norms, fused
  experts) and no layer output scale. Rewrite on section 1b.
- [~] **Phase 4: Self-conditioning.** Exact full-vocabulary soft embedding exists (`ComputeSoftEmbeddings`, 2026-10-03; very
  expensive, a parity path). The learned MLP and its pre-norm are not loaded or applied (the helper `ApplySelfCondMlp`
  exists, unused).
- [~] **Phase 5: Sampler.** Entropy, ranking, acceptance, re-noising, stop test exist; stability fixed. Gumbel-max,
  token-selection rule and temperature contract are OPEN.
- [~] **Phase 6: Block lifecycle.** Loop exists and commits argmax; correctness depends on phases 1-4.
- [~] **Phase 7: Tests.** Component tests pass (sampler, SC maths, tiny synthetic pipeline) but use the old guessed tensor
  names and a simplified backbone; they will be rewritten with phases 1-3. Missing: any parity test.
- [~] **Phase 8: Gate and CLI registry.**
  - [x] `// diffusion-gemma - NOT admitted` block in `ModelCompatibility.cs`.
  - [x] Forward pass refuses the real checkpoint layout (`DiffusionGemmaRealCheckpointGuardTests`).
  - [ ] Pipeline CLI entry point.
- [ ] **Phase 9: Real Q4_K_M verification.** Blocked on phases 1-4. Proposed route:
  1. **Backbone, causal path:** write a copy of the GGUF re-labelled as `gemma4` (rename the `diffusion-gemma.*` keys, map
     `enc_layer_output_scale` to `layer_output_scale`) and compare our prefill logits with llama.cpp `bed0a8566`. This is an
     independent check of section 1b, RoPE, the router and the fused experts, with the caveat that the re-labelling encodes
     the `enc_layer_output_scale` assumption.
  2. **Canvas and self-conditioning path:** needs the HF reference (step traces from identical noise); no llama.cpp oracle exists.

---

## Deferred (explicitly out of scope)

- Gemma-4 vision tower (27 layers, 1152 dim, 280 image tokens): text only.
- Server endpoints, structured output constraints, GPU paths.

---

**Effort:** the earlier "1-2 days" estimate covered a simplified backbone and is void. Realistically this is a Gemma-4 MoE
backbone port (shared with any future `gemma4` MoE support), plus the diffusion-specific parts, plus a verification route
that needs the HF source. Do the backbone first: it is independently checkable against llama.cpp.
