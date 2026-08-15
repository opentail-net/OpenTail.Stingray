# GLM4V Vision Encoder — Implementation Plan

## Context

SEE "GLM4 VL research.txt" - for full research agent's report!!!!

Following the Llama 4 (`llama4`) vision encoder (`docs/06-llama4-vision-plan.md`), the user
supplied the full `tools/mtmd/models/glm4v.cpp` source and asked whether GLM4V is a reasonable
next target. Vision-encoder-only ports (no text-decoder splice, no image-token attention-mask
wiring — the same scope as `gemma3`, `gemma4v`, and `llama4`) are the established pattern here.

**Caveat worth weighing before starting implementation:** none of the three existing vision
encoders (`gemma3`, `gemma4v`, `llama4`) are numerically parity-verified against a real oracle,
and none are spliced into the text decoder yet (Phase V4 — actually using image embeddings during
generation — hasn't started for any of them). Adding a fourth isolated encoder grows that backlog
rather than shrinking it. GLM4V is also strictly harder than llama4: it needs a real
`ggml_conv_2d`-based patch merger (not a reshape trick), a nontrivial dual-conv patch-embed
reshape, its own self-contained M-RoPE variant, and the only checkpoint found so far is
**Q8_0-quantized**, not F16 like the other three — the first vision encoder that would need a
quantized-weight matvec path (`SimdKernels.MatVecQ8_0`, already present in the codebase) rather
than plain F16. If/when the decision is made to proceed, this plan is ready to execute as-is.

## Architecture Summary (from the pasted `glm4v.cpp`, cross-checked against known `clip.cpp`
shared-helper conventions used by the three existing encoders)

GLM4V reuses the generic `build_vit` loop (like `llama4`, unlike `qwen2vl`/`qwen3vl` which hand-roll
their own per-layer loop) — so no new per-layer control flow is needed, only new pre/post stages
and a different `add_pos` lambda passed into `build_vit`.

1. **Dual-conv patch embed** — two separate `ggml_conv_2d` weights (`patch_embeddings_0` =
   `v.patch_embd.weight`, `patch_embeddings_1` = `v.patch_embd.weight.1`) applied to the input
   image and **summed elementwise first** (channels are never split — the two convs exist only
   because the original HF weight is a single `Conv3d` with temporal kernel=2, split into two
   `Conv2d`s at conversion time; for a still image `conv3d(img,img) ≡ conv2d_0(img)+conv2d_1(img)`,
   confirmed via `conversion/qwen3vl.py:128-141`). The subsequent
   permute/cont_4d/reshape_4d/permute/cont_3d chain is then **pure spatial patch reordering**
   (raster `(h,w)` → 2×2-block-raster), traced index-by-index: final token index
   `t = 2·(h//2)·W + 4·(w//2) + 2·(h%2) + (w%2)`, channel unchanged. This reorder is exactly the
   same block/sub-block token order the M-RoPE position-fill loop produces (item 5) — the two
   must and do line up. Patch bias (`v.patch_embd.bias`, **REQUIRED** — `GGML_ASSERT(patch_bias !=
   nullptr)` — unlike llama4 where it's optional/absent) is added **after** this reshape.
2. **No CLS token** (`class_embedding == nullptr`, asserted) — unlike llama4's CLS-concat.
3. **Optional learned position table** (`position_embeddings`, absent for the "GLM-OCR" variant) —
   resized via bicubic scaling to match the patch grid, reshaped through the *same*
   permute/reshape pattern as the patch embed, then added.
4. **"pos-conv norm"** — an extra `build_norm` (RMSNorm) applied right after position-embedding
   add, before the block loop. Neither llama4 nor gemma3/gemma4v has this extra stage.
5. **Per-block M-RoPE** (self-contained inside the vision graph's `add_pos` lambda, applied to Q/K
   only — does **not** touch the text decoder). **Fully resolved by direct trace of
   `ggml/src/ggml-cpu/ops.cpp` (`ggml_mrope_cache_init`/`rotate_pairs`) and `clip.cpp`'s GLM4V
   position-fill loop — no longer an open unknown.** Despite `mrope_sections` declaring 4 quarters
   and a `positions` tensor of length `n_patches*4`, only streams 0 (row) and 1 (col) are ever
   read by the CPU kernel for `GGML_ROPE_TYPE_VISION` — streams 2/3 are dead duplicates. The
   rotation is **NEOX-style split-half pairing** (channel `k` paired with `k + d_head/2`, not
   adjacent-pair), and YaRN is fully disabled (`ext_factor=0` ⇒ `theta = theta_extrap` exactly,
   `mscale=1`). Concretely, with `H = d_head`, quarters `Q0=[0,H/4) Q1=[H/4,H/2) Q2=[H/2,3H/4)
   Q3=[3H/4,H)`, `inv_freq(k) = rope_theta^(-4k/H)` for `k ∈ [0,H/4)`, `pos_row`/`pos_col` the
   patch's row/col (post-merge-block reorder, see item 1 below):
   ```
   theta_row(k) = pos_row * inv_freq(k);  theta_col(k) = pos_col * inv_freq(k)
   (Q0[k], Q2[k]) rotated by (cos(theta_row(k)), sin(theta_row(k)))   // NEOX split-half
   (Q1[k], Q3[k]) rotated by (cos(theta_col(k)), sin(theta_col(k)))
   ```
   This is the standard Qwen2VL-style vision-rope pattern (`[h,w,h,w]` duplicated-concat +
   rotate-half) applied to Q and K independently, over the *entire* head dim (no untouched tail).
   `pos_row`/`pos_col` come from `clip.cpp`'s shared `PROJECTOR_TYPE_{QWEN2VL,QWEN3VL,GLM4V}`
   position-fill loop: patches are walked in `n_merge`-strided blocks (`y`/`x` outer, step
   `n_merge=2`) with an inner hardwired 2×2 (`dy,dx ∈ {0,1}`) sub-loop, `pos_row=y+dy,
   pos_col=x+dx` — this token order must exactly match the patch-embed reshape's output order
   (item 1), which it does (verified below). Implementation: a small, self-contained 4-quarter
   NEOX-pairing rope helper local to `Glm4vVisionEncoder`, no engine-wide RoPE dispatch changes.
6. **RMSNorm body** (`NORM_TYPE_RMS`), not LayerNorm like llama4 — block structure is otherwise
   the same shape (ln1 → attn → residual → ln2 → ffn → residual).
7. **Patch merger (downsample)** — genuinely different from llama4's reshape-only pixel-shuffle:
   reshape to `[n_embd, n_merge, n_merge, n_token_out]`, permute, then a **real**
   `ggml_conv_2d(mm_patch_merger_w, cur, n_merge, n_merge, 0, 0, 1, 1)` (kernel=stride=n_merge, no
   padding, `mm_patch_merger_w` = HF `visual.downsample`, a genuine `Conv2d(n_embd, n_embd_out,
   kernel_size=n_merge, stride=n_merge)`). Because kernel==stride==n_merge with no padding, output
   spatial size is exactly 1×1 per merged token, so the conv **reduces to a per-output-token dot
   product** — no sliding-window conv logic is actually needed in the C# port, just a loop:
   ```
   out[oc, tok] = bias[oc] + Σ_{ky,kx∈[0,n_merge)} Σ_{ic∈[0,n_embd)}
                      W[oc][ic][ky][kx] * vit_out[ic, tok*n_merge² + ky*n_merge + kx]
   ```
   where `vit_out` is consumed in contiguous runs of `n_merge²` tokens (already in the right order
   from item 1's reorder) and `W` is read in GGUF/HF `[OC,IC,KH,KW]` order. This confirmed
   simplification means the merger is implementable as a dense matvec-like loop, not a real conv2d
   kernel.
8. **FC projector** — `build_mm(mm_fc_w)` → LayerNorm (`NORM_TYPE_NORMAL`, `mm_post_norm_w/b`,
   eps hardcoded to `1e-5`, distinct from the ViT body's own eps) → `ggml_gelu_erf` (erf-based
   GELU — not yet used by any existing encoder; llama4 uses tanh-approx/quick GELU).
9. **FFN projector** — a full gated `build_ffn` (`mm_ffn_up/gate/down_w/b`, activation from
   `hparams.ffn_op`) as a third activation-bearing stage after the ViT body and FC projector.

## Confirmed Tensor Names, Fused QKV, and Hparams

From `clip-impl.h`/`clip.cpp`/`tensor_mapping.py` (all prefix `v.` for vision-tower tensors):

| Field | GGUF tensor name | Required? |
|---|---|---|
| patch conv 0 / 1 | `v.patch_embd.weight` / `v.patch_embd.weight.1` | yes (both, dual-conv) |
| patch bias | `v.patch_embd.bias` | **yes** — `GGML_ASSERT` |
| pos-conv norm | `v.norm_embd.weight` / `.bias` | yes |
| learned position table | `v.position_embd.weight` | optional (absent for GLM-OCR) |
| class embedding | `v.class_embd` | **must be absent** — `GGML_ASSERT` |
| per-block QKV | `v.blk.N.attn_qkv.weight` / `.bias` | **fused, not separate Q/K/V** — confirmed via conversion-script trace (GLM4V shares Qwen3VL's fused-`qkv` HF Linear naming, not Qwen2VL's split naming); bias presence should be verified against the actual downloaded GGUF (`list-tensors`) rather than assumed, loader must treat it as optional either way |
| Q/K norm | `v.blk.N.attn_q_norm.weight` / `attn_k_norm.weight` | optional, variant-only (GLM-OCR has these + no position table; standard GLM4V has neither) — treat as independently-optional flags, not a variant switch |
| patch merger | `mm.patch_merger.weight` / `.bias` | both required |
| FC projector | `mm.model.fc.weight` | weight only, **no bias tensor for this stage** |
| post-norm | `mm.post_norm.weight` / `.bias` | weight required, bias optional |
| FFN projector | `mm.up/gate/down.weight` / `.bias` | weight required per gate, bias optional |

Hparams: `rope_theta = 10000.0` is a **hardcoded constant** for GLM4V (never read from GGUF
metadata — do not look for a `clip.vision.rope_theta` key). `n_merge` defaults to `2`, optionally
overridden by `clip.vision.spatial_merge_size`.

## Files to Add (mirroring the `Llama4Vision*` naming/structure)

- `src/OpenTail.Stingray.Vision/Glm4vVisionModel.cs` — GGUF loader (`clip.projector_type` or
  `clip.vision.projector_type` — **must confirm which key this checkpoint actually uses**, both
  precedents exist across the three existing encoders). Required/optional tensors per the table
  above, including the Q8_0-typed patch-embed/attention/FFN weights, and the fused-QKV split
  (`ggml_view`-equivalent slicing at offsets `0`, `n_head*d_head`, `2*n_head*d_head` along the
  output axis — concatenated, not interleaved per-head).
- `src/OpenTail.Stingray.Vision/Glm4vVisionEncoder.cs` — forward pass: dual-conv patch embed
  (summed) + spatial reorder (confirmed index formula above), patch-bias add, optional position
  table + pos-conv RMSNorm, per-block RMSNorm/fused-QKV-split/attn (with the confirmed NEOX
  split-half M-RoPE on Q/K, quarters 0↔2 by row, 1↔3 by col)/FFN, patch merger (confirmed
  reducible to a dense per-token dot-product loop, not a real sliding conv), FC projector
  (LayerNorm + gelu_erf), gated FFN projector.
- `src/OpenTail.Stingray.Vision/Glm4vImagePreprocessor.cs` — thin wrapper, same pattern as
  `Llama4ImagePreprocessor.cs` (reuse `Gemma4VImagePreprocessor.ResizeNormalize` if mean/std match
  a simple normalize; confirm GLM4V's actual mean/std from `clip.vision.image_mean/std` metadata).
- `tests/OpenTail.Stingray.Tests.Vision/VisionTestPaths.cs` — add `Glm4vMmprojFile` const +
  `FindGlm4vMmproj()`.
- `tests/OpenTail.Stingray.Tests.Vision/Glm4vVisionEncoderTests.cs` — structural-sanity-only test
  (shape, no-NaN/Inf, non-degenerate magnitude), same pattern as `Llama4VisionEncoderTests.cs`. No
  numerical-parity oracle exists for this or any of the other three encoders.
- `scripts/download-model.ps1` — add a `"glm4v-mmproj"` preset pointing at
  `ggml-org/GLM-4.6V-GGUF`'s `mmproj-GLM-4.6V-Q8_0.gguf`.

This document itself (`docs/07-glm4v-vision-plan.md`) should gain a **RESULT** addendum after
implementation, mirroring `docs/06-llama4-vision-plan.md`'s addendum: documenting every real
finding vs. pre-implementation assumption listed above (metadata key, M-RoPE section mapping,
patch-embed reshape correctness, conv2d merger behavior, performance figures, and the same
"not numerically parity-verified" caveat).

## Checkpoint

`ggml-org/GLM-4.6V-GGUF` on Hugging Face, file `mmproj-GLM-4.6V-Q8_0.gguf` (944MB). This is the
only mmproj found for any GLM4V variant; it is **Q8_0-quantized**, not F16. The paired full text
weights (`GLM-4.6V-Q4_K_M.gguf` 70.4GB / `GLM-4.6V-Q8_0-*` shards ~113GB) are not needed — only the
mmproj. Because this is quantized, the loader and encoder must route patch-embed/attention/FFN
matvecs through `SimdKernels.MatVecQ8_0` rather than the F16 matvec path used by the three existing
encoders — this is new plumbing (routing, not new math) worth calling out explicitly during
implementation review. No F16 alternative mmproj was found in an initial search; worth one more
targeted search (e.g. `unsloth/GLM-4.1V-9B-Thinking-GGUF` and similar community requants) before
committing to the Q8_0-only path, since an F16 checkpoint would remove an entire risk axis.

## Verification

Same structural-sanity approach as `llama4`/`gemma3`/`gemma4v`: load the real checkpoint, run a
synthetic image through the encoder, assert output shape matches `n_token_out × projection_dim`,
no NaN/Inf, magnitude non-degenerate. No numerical oracle exists, so this cannot be a parity test —
document that limitation explicitly in the test and in this plan's RESULT addendum, as done for the
other three. Run the full `Tests.Vision` suite afterward to confirm no regressions (expect it to
take noticeably longer than llama4's 132s given Q8_0 dequant overhead).
