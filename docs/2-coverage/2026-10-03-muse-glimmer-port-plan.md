# Muse-Glimmer port plan (`muse-glimmer`)

**Status:** in progress (2026-10-03). **Policy:** port now, prove later. Not admitted, not
advertised until checkpoint-verified (CLAUDE.md rule 14; [ported-families-todo](ported-families-todo.md)).

## Architecture (text tower)

**Primary references:**
- Hugging Face Transformers `MuseGlimmer` official implementation.
- Upstream llama.cpp `src/models/muse-glimmer.cpp` (204 lines, in local source checkout since pull to `bed0a8566`).
Secondary: TensorSharp `docs/models/muse-glimmer.md`, `Models/MuseGlimmer/`.
Local Stingray oracle: none in current vendored llama.cpp b10306 binaries (requires newer build).

The 30B model: 52 dense layers, `n_embd` 6656, `n_ff` 19968, 32 Q heads / 2 KV heads, head_dim 128, vocab 202048.

### Upstream specification & GGUF conversion semantics

1. **Embedding:**
   `h = rmsnorm_unweighted(embed, eps = 1e-5)`
2. **Layer Schedule (39 SWA + 13 Full/NoPE layers):**
   - SWA layers: `l % 4 < 3` -> layers 0, 1, 2, 4, 5, 6, ..., 48, 49, 50 (39 layers).
   - Full/NoPE layers: `l % 4 == 3` -> layers 3, 7, 11, ..., 47, 51 (13 layers).
   - RoPE runs on SWA layers only; Full layers use NoPE (no rotary embedding).
   - RoPE representation: HF uses rotate-half layout; GGUF conversion unpermutes into GGML NORM/interleaved form. Stingray operates on the converted GGUF representation.
3. **Sliding-Window Attention Boundary:**
   - Window size `n_swa = 2048`.
   - Query at position $p$ sees positions $[\max(0, p - 2047), p]$ (exactly 2048 visible keys including self).
4. **Per-Layer Norm Weights (`weight + 1` conversion folding):**
   - The original architecture uses centered RMSNorm with $(1 + W)$.
   - The llama.cpp conversion script performs `data_torch = data_torch + 1` for all four per-layer norm tensors (`attn_norm`, `attn_post_norm`, `ffn_norm`, `ffn_post_norm`).
   - The GGUF file therefore stores the already-shifted $(1 + W)$ weights; Stingray and llama.cpp evaluate standard RMSNorm directly on the stored tensors.
   - Post-attention and post-FFN norms use hardcoded `post_norm_eps = 1e-8f`.
5. **QK Scaling & Attention Output Gate:**
   - `qk_scale_factor = 3.87f`, standard attention scale $1/\sqrt{128} \approx 0.088388$.
   - GGUF conversion synthesizes `attn_q_norm.weight = 3.87f` and `attn_k_norm.weight = 1.0f`, absorbing the scale directly into the Q norm.
   - Attention output gate: `gate = sigmoid(attn_gate @ pre_attn_normed_x)` before $W_o$:
     `attn_out = Wo @ (attn * gate)`.
6. **Final Output Norm & Logits:**
   - Final `output_norm`: standard weighted RMSNorm (`eps = 1e-5f`), **untouched** by the `+1` conversion shift.
   - Output multiplier: `output_multiplier = 0.19611613513818404f` (stored in `muse-glimmer.logit_scale`).
   - Softcap: 20 (stored in `muse-glimmer.final_logit_softcapping`).
   - Logits computation contract:
     `logits = 20 * tanh((lm_head(output_norm(h)) * 0.19611613513818404) / 20)`.

```
h  = rmsnorm_unweighted(embed, eps = norm_eps)
per layer (52 total: 39 SWA, 13 Full):
  cur = rmsnorm(h, attn_norm)                                  (converted weight already includes +1)
  gate = wqkv_gate @ cur
  q, k, v = build_qkv(cur)
  q = rmsnorm(q, attn_q_norm)                                  (attn_q_norm = 3.87)
  k = rmsnorm(k, attn_k_norm)                                  (attn_k_norm = 1.0)
  if (is_swa): RoPE(q, k, NORM/interleaved)
  attn = causal_attention(q, k, v, scale = 1/sqrt(head_dim), window = 2048 on SWA, none on Full)
  attn_out = Wo @ (attn * sigmoid(gate))
  h1 = h  + rmsnorm(attn_out, post_attention_norm, eps = 1e-8) (converted weight already includes +1)
  h2 = h1 + rmsnorm(SwiGLU(rmsnorm(h1, ffn_norm)), post_ffw_norm, eps = 1e-8)
logits = 20 * tanh((lm_head(rmsnorm(h2, output_norm)) * 0.19611613513818404) / 20)
```

Metadata: `muse-glimmer.attention.sliding_window_pattern` (scalar period 4),
`muse-glimmer.attention.sliding_window` (2048), `muse-glimmer.logit_scale` (0.19611613513818404),
`muse-glimmer.final_logit_softcapping` (20.0).

## Reuse in Stingray

- The cohere2 branch in `ModelGraph` (scalar SWA period, `RopeOnlySwaLayers`).
- QK-norm and sandwich-norm detection from tensor presence (Gemma 4 path).
- `LogitScale` (multiply, cohere2 convention), `FinalLogitSoftcap`, applied in that order.
- NORM RoPE (the non-NeoX default).

## New work

- [x] `ModelGraph`: the `muse-glimmer`/`muse_glimmer` arch.
  - SWA period (default 4) and `RopeOnlySwaLayers`; optional `rope.freq_base_swa`.
  - Raw logit scale (0.19611613513818404); softcap (20.0) only when key is present.
  - Fields `PostNormEps` (1e-8), `InputEmbeddingRmsNorm`, `AttentionOutputGate`.
- [x] `ForwardPass` decode:
  - load `attn_gate` per layer;
  - gate projection from `_normBuf` next to Q/K/V, `attnOut *= sigmoid(g)` before `Wo`;
  - `PostNormEps` (1e-8) at both post-norm sites;
  - unweighted embedding RMSNorm in `Forward`.
- [ ] Prefill:
  - Note: attention gate `sigmoid(Wgate @ norm_in) * attn` is inherently batchable (standard GEMM).
  - Current initial restriction: gated models take the per-token path in `PrefillDispatch` as a safe initial boundary.
  - Follow-up: wire `_attnGate` into `PrefillCore` batched GEMM path for full prefill speed.
- [x] `ModelCompatibility`: a `// muse-glimmer — NOT admitted` block (not in allowlist).
  `STINGRAY_DIAGNOSTIC_ALLOW_UNSUPPORTED_ARCH=1` runs it for experiments.

## Deferred (not in the initial port)

The vision tower (50-layer ViT, 1536 hidden width, merge size 2, erf-GELU, Lanczos-3), DFlash speculative drafter (companion drafter model, explicitly out of scope for initial port), batched prefill GEMM for gated models, and GPU paths.

## Verification (levels as in [ported-families-todo](ported-families-todo.md))

- [x] **Specification test (level 2):** created a synthetic tiny `muse-glimmer` GGUF (4 layers, P = 2,
   window 3, small dims, random F32 weights; `MuseGlimmerSyntheticTests`). Models converted GGUF semantics:
   `attn_q_norm.weight = 3.87f`, `attn_k_norm.weight = 1.0f`, converted per-layer norm weights, unshifted
   final output norm, SWA masking past the window, NoPE on full layers, attention output gate, 1e-8 post-norms,
   embedding norm and output multiplier scale-then-softcap; verified max |Δlogit| < 1e-3 (F32). Passed 2026-10-03.
- [ ] **Independent implementation (level 3):** run the same synthetic GGUF through a llama.cpp build
   from `bed0a8566` or later (has `muse-glimmer.cpp`) and compare logits. That needs building llama.cpp
   or newer vendored binaries.
- [ ] **Real weights (level 4):** with a checkpoint (Muse-Glimmer-30B; small quants about 7-10 GB, fits
   this machine), coherence, then `stingray admit-arch` against that newer `llama-server`.
- [ ] **Admission (level 5):** admit through the normal text-LLM path.

**Effort:** port + specification test ~2-4 hours. Real-weight verification (fits 64 GB RAM at 7-10 GB),
closeout performance + DRY pass, batched prefill, and admission are separate.
