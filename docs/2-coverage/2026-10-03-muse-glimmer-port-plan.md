# Muse-Glimmer port plan (`muse-glimmer`)

**Status:** in progress (2026-10-03). **Policy:** port now, prove later. Not admitted, not
advertised until checkpoint-verified (CLAUDE.md rule 14; [ported-families-todo](ported-families-todo.md)).

## Architecture (text tower)

Source: TensorSharp `docs/models/muse-glimmer.md` and `Models/MuseGlimmer/MuseGlimmerModel.cs`.
TensorSharp cites llama.cpp's `src/models/muse-glimmer.cpp`, which is not in our local llama.cpp
copy (2026-08-07) or in the vendored b10306 binaries.

The 30B model: 52 dense layers, `n_embd` 6656, `n_ff` 19968, 32 Q heads / 2 KV heads, head_dim 128,
vocab 202048.

```
h  = rmsnorm_unweighted(embed)                       (eps = norm eps)
per layer:
  a  = rmsnorm(h, attn_norm)
  q,k,v,g = W·a;  q,k per-head RMSNorm;  RoPE (NORM, interleaved) on SWA layers only (full = NoPE)
  attention: causal; SWA layers see keys with p1 - p0 < n_swa (2048); SWA iff l % P < P-1 (P = 4)
  attn *= sigmoid(g);  o = Wo·attn
  h1 = h  + rmsnorm(o, post_attention_norm, eps = 1e-8)
  h2 = h1 + rmsnorm(SwiGLU(rmsnorm(h1, ffn_norm)), post_ffw_norm, eps = 1e-8)
logits = tanh(lm_head(rmsnorm(h2)) * logit_scale / cap) * cap     (scale before softcap)
```

Metadata: `muse-glimmer.attention.sliding_window_pattern` (scalar period),
`muse-glimmer.attention.sliding_window`, `muse-glimmer.logit_scale`,
`muse-glimmer.final_logit_softcapping`. Tensors: the standard set plus `attn_gate`,
`attn_q_norm`/`attn_k_norm`, `post_attention_norm`, `post_ffw_norm`.

## Reuse in Stingray

- The cohere2 branch in `ModelGraph` (scalar SWA period, `RopeOnlySwaLayers`).
- QK-norm and sandwich-norm detection from tensor presence (Gemma 4 path).
- `LogitScale` (multiply, cohere2 convention), `FinalLogitSoftcap`, applied in that order.
- NORM RoPE (the non-NeoX default).

## New work

- [ ] `ModelGraph`: the `muse-glimmer`/`muse_glimmer` arch.
  - SWA period and `RopeOnlySwaLayers`.
  - Raw logit scale, softcap with default 20.
  - New fields `PostNormEps` (1e-8), `InputEmbeddingRmsNorm`, `AttentionOutputGate`.
- [ ] `ForwardPass` decode:
  - load `attn_gate` per layer;
  - gate projection from `_normBuf` next to Q/K/V, `attnOut *= sigmoid(g)` before `Wo`;
  - `PostNormEps` at both post-norm sites;
  - unweighted embedding RMSNorm in `Forward`.
- [ ] Prefill: gated models take the per-token path in `PrefillDispatch`.
  - `PrefillWithCache` / `BatchForwardMulti` throw; `SupportsBatchVerify` is false.
  - CUDA/Vulkan/GPU passes refuse.
  - Batched prefill support is a follow-up once verified.
- [ ] `ModelCompatibility`: a `// muse-glimmer — NOT admitted` block (not in the allowlist).
  `STINGRAY_DIAGNOSTIC_ALLOW_UNSUPPORTED_ARCH=1` runs it for experiments.
- [ ] The vision tower (50-layer ViT, erf-GELU, Lanczos-3 preprocessing) is a later phase.

## Verification

1. **Now (no checkpoint):** a synthetic tiny `muse-glimmer` GGUF (2-4 layers, P = 2, window 3, small
   dims, random F32 weights). Compare the engine's logits over a 6-10-token prompt against an
   **independent reference forward written in the test from the spec above**. It must cover SWA
   masking past the window, NoPE on full layers, the gate, the 1e-8 post-norms, the embedding norm,
   and scale-then-softcap. Bound: max |Δlogit| ≲ 1e-3 (F32).
2. **With a checkpoint** (Muse-Glimmer-30B GGUF; card lists IQ2 to Q4 sizes, about 7-10 GB for small
   quants): a coherence check, then a token-level comparison with a llama.cpp build that has
   `muse-glimmer.cpp` (newer than b10306) via `stingray admit-arch`, or with TensorSharp.
3. **Admission:** the normal path (independent reference, timed real-weight runs, STATUS row).

**Effort:** about 2-3 hours to "ported + synthetic-verified"; vision about a day more.
