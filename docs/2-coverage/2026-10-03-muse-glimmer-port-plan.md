# Muse-Glimmer port plan (`muse-glimmer`)

> Completed items moved to [done/2026-10-03-muse-glimmer-port-plan-completed-items-2026-10.md](../done/2026-10-03-muse-glimmer-port-plan-completed-items-2026-10.md) on 2026-10-30. This file keeps only open work.

**Status (revised 2026-10-03): ADMITTED, text only, CPU only.** Real-weight verified against llama.cpp (`admit-arch` 8/8, a 3,748-token prompt past the real window, patched-window runs); prefill is about 10x slower than llama.cpp. Vision tower, DFlash and batched prefill for gated models are not done. **Policy was:** port now, prove later. Not admitted, not
advertised until checkpoint-verified (CLAUDE.md rule 14; [ported-families-todo](ported-families-todo.md)); the checkpoint check has now been done. Licence: Apache 2.0 per secondary sources (2026-10-31; the Hub card was not fetched).

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

- [ ] Prefill:
  - Note: attention gate `sigmoid(Wgate @ norm_in) * attn` is inherently batchable (standard GEMM).
  - Current initial restriction: gated models take the per-token path in `PrefillDispatch` as a safe initial boundary.
  - Follow-up: wire `_attnGate` into `PrefillCore` batched GEMM path for full prefill speed.

## Deferred (not in the initial port)

The vision tower (50-layer ViT, 1536 hidden width, merge size 2, erf-GELU, Lanczos-3), DFlash speculative drafter (companion drafter model, explicitly out of scope for initial port), batched prefill GEMM for gated models, and GPU paths.

## Verification (levels as in [ported-families-todo](ported-families-todo.md))

- [ ] **Independent implementation, synthetic (level 3):** the synthetic-GGUF comparison against llama.cpp was never run
   as such; a CPU llama.cpp build at `bed0a8566` now exists (`examples/llama.cpp/llama.cpp/build-ref`), so it is cheap to do. The
   level-4 result below makes it lower priority.
- [ ] **Follow-up (rule 7 performance pass):** wire `attn_gate` and the embedding norm into the batched `PrefillCore` GEMM path;
   the gate is a plain GEMM, so this should close most of the 10x prefill gap. Not started.

**Effort:** port + specification test ~2-4 hours; level-4 text check done 2026-10-03. Remaining:
closeout performance + DRY pass, batched prefill, and admission are separate.

---

## Research findings (2026-10-31)

Facts are in [research/2026-10-31-coverage-facts.md](research/2026-10-31-coverage-facts.md).

- **Licence:** secondary sources say Apache 2.0. The Hub card was not fetched, so the licence is recorded here as "per secondary sources". Confirm it on `meta-models/Muse-Glimmer-30B` before any distribution.
- **Size and footprint:** one summary says the BF16 base weights are 59.6 GB, while others say more than 55 GB; quantised footprints are reported as under 20 GB. These conflict, so the disk and RAM numbers in this plan should come from the actual GGUF files.
- **Architecture:** the card describes a ~29.6B dense transformer with a ~1.8B ViT-G/14 perception encoder, which matches the plan's vision-tower deferral.

**Next check:** fetch the model card and the GGUF file list for the licence and the exact sizes.
