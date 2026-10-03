# Muse-Glimmer port plan (`muse-glimmer`)

**Status:** in progress (2026-10-03). **Policy:** port now, prove later. Not admitted, not
advertised until checkpoint-verified (CLAUDE.md rule 14; [ported-families-todo](ported-families-todo.md)).

## Architecture (text tower)

**Primary reference: llama.cpp `src/models/muse-glimmer.cpp`** (203 lines), in the local source
checkout since the 2026-10-03 pull to `bed0a8566`. It isn't in the vendored b10306 binaries.
Secondary: TensorSharp `docs/models/muse-glimmer.md`, `Models/MuseGlimmer/MuseGlimmerModel.cs`.
The upstream source confirms the spec below, with three refinements:
- `final_logit_softcapping` is **optional with no default**: absent means no softcap (TensorSharp
  defaults to 20; follow llama.cpp).
- `rope.freq_base_swa` is optional, defaulting to the main base.
- Q/K/V come from `create_tensor_qkv`, so handle a fused `attn_qkv` as well as separate Q/K/V.

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

**2026-10-03 checkpoint:** the four items below are wired (CPU). Release build is clean, and the
existing ForwardPass.Fast (784 passed) and Core (679 passed) suites stay green. No Muse-specific
test exists yet. **Next:** the level-2 synthetic specification test (Verification §1).

- [x] `ModelGraph`: the `muse-glimmer`/`muse_glimmer` arch.
  - SWA period (default 4) and `RopeOnlySwaLayers`; optional `rope.freq_base_swa`.
  - Raw logit scale; softcap only when the key is present.
  - New fields `PostNormEps` (1e-8), `InputEmbeddingRmsNorm`, `AttentionOutputGate`.
- [x] `ForwardPass` decode:
  - load `attn_gate` per layer;
  - gate projection from `_normBuf` next to Q/K/V, `attnOut *= sigmoid(g)` before `Wo`;
  - `PostNormEps` at both post-norm sites;
  - unweighted embedding RMSNorm in `Forward`.
- [x] Prefill: gated models take the per-token path in `PrefillDispatch`.
  - `PrefillWithCache` / `BatchForwardMulti` throw; `SupportsBatchVerify` is false.
  - CUDA/Vulkan/GPU passes refuse.
  - Batched prefill support is a follow-up once verified.
- [x] `ModelCompatibility`: a `// muse-glimmer — NOT admitted` block (not in the allowlist).
  `STINGRAY_DIAGNOSTIC_ALLOW_UNSUPPORTED_ARCH=1` runs it for experiments.
## Deferred (not in the initial port)

The vision tower (50-layer ViT, erf-GELU, Lanczos-3 preprocessing), ATEM/tool-calling chat
specifics, the DFlash speculative drafter, batched/paged prefill for gated models, and GPU paths.

## Verification (levels as in [ported-families-todo](ported-families-todo.md))

- [ ] **Specification test (level 2), next:** create a synthetic tiny `muse-glimmer` GGUF (2-4 layers, P = 2,
   window 3, small dims, random F32 weights). Compare the engine's logits over a 6-10-token prompt
   against a test-side reimplementation of the spec above. Cover SWA masking past the window, NoPE on
   full layers, the gate, the 1e-8 post-norms, the embedding norm and scale-then-softcap; bound
   max |Δlogit| ≲ 1e-3 (F32). The spec is cross-read against llama.cpp's source, but this test is
   still not an independent implementation.
- [ ] **Independent implementation (level 3):** run the same synthetic GGUF through a llama.cpp build
   from `bed0a8566` or later (has `muse-glimmer.cpp`) and compare logits. That needs building llama.cpp
   or newer vendored binaries.
- [ ] **Real weights (level 4):** with a checkpoint (Muse-Glimmer-30B; small quants about 7-10 GB),
   coherence, then `stingray admit-arch` against that newer `llama-server`.
- [ ] **Admission (level 5):** admit through the normal text-LLM path.

**Effort:** port + specification test about 2-3 hours. Real-weight verification, the closeout
performance + DRY pass, and admission are separate.
