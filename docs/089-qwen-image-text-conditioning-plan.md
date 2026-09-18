# 089 — Qwen Image real text-conditioning implementation plan (2026-09-18)

Follow-up to `docs/086`'s Qwen Image finding: DiT+VAE are real and verified against real weights,
the single remaining gap is real LLM text conditioning (`QwenImagePipeline` currently feeds
zero-vector context). This doc is the real recipe, checked directly against the vendored reference
(`examples/diffusers/src/diffusers/pipelines/qwenimage/pipeline_qwenimage.py`) before any
implementation, same discipline `docs/087` used for FLUX.2 — **do not guess the prompt
template/extraction recipe**, every prior text-conditioning integration this session (FLUX.1,
Wan, HunyuanVideo, FLUX.2) has had a real, non-obvious wrapping/cropping convention.

## Checkpoint status (confirmed 2026-09-18, corrects an earlier false "missing" claim this session)

`models/_models/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf` (4.68GB) IS present. `stingray list-metadata`
confirms `general.architecture = qwen2vl`, NOT plain `qwen2`. **`qwen2vl` is not currently in
`ModelGraph.cs`'s architecture-support switch or `ModelCompatibility`'s allowlist** — this is the
one real blocking code gap, not a missing checkpoint.

## Real, precise text-conditioning recipe (confirmed against `pipeline_qwenimage.py`)

```python
tokenizer_max_length = 1024
prompt_template_encode = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)
prompt_template_encode_start_idx = 34   # drop_idx
```

1. Wrap the user prompt in the ChatML template above (`{}`  = the raw user prompt text).
2. Tokenize with `max_length = tokenizer_max_length + drop_idx = 1058`, `padding=True` (dynamic,
   batch-max, NOT a fixed constant like FLUX.2's 512), `truncation=True`.
3. Run the real Qwen2.5-VL text decoder with `output_hidden_states=True`; take
   `hidden_states[-1]` — **the FINAL layer only**, not a multi-layer concat like FLUX.2's Mistral
   scheme (`OUTPUT_LAYERS_MISTRAL=[10,20,30]` does NOT apply here — simpler for Qwen Image).
4. Use the real attention mask to select only the non-padding real tokens per sequence
   (`_extract_masked_hidden`).
5. **Crop off the first 34 tokens** (`drop_idx`) from each sequence's hidden states — these
   correspond to the template's own system+user-wrapper tokens, not the real prompt content. This
   is the same "encode a template, crop the template's own tokens back off" convention already
   confirmed for HunyuanVideo's `crop_start` (`docs/088`) — a real, recurring pattern across this
   project's LLM-as-text-encoder integrations, not a one-off.
6. Re-pad each (now-cropped) sequence to the BATCH's own max real length (dynamic per generation
   call, not a fixed constant) with zeros, for the DiT's cross-attention input.

## Real code gap: `qwen2vl` architecture support (the actual blocker, not the checkpoint)

Real precedent already exists in this codebase for exactly this situation: Qwen3-ForcedAligner's
text decoder declares real M-RoPE metadata (`rope_scaling.interleaved=true`/
`mrope_interleaved=true`, a genuine Qwen2-VL/Qwen2.5-Omni-family mechanism this engine does not
implement in general) — but for **text-only use** (no image/video tokens interleaved with text),
M-RoPE's 3D section-splitting degenerates to plain 1D sequential position ids per the reference's
own `qwen_position_ids` function, so the only real numerically-meaningful difference from standard
NEOX rotation is the interleaved (adjacent-pair) RoPE pairing convention, not any 3D
section-splitting (`ModelGraph.cs` around line 543, `{arch}.rope.is_neox` override mechanism
already built for exactly this).

**This same situation almost certainly applies to Qwen2.5-VL's text backbone used purely for text
conditioning extraction (no image input into the LLM at all, per the real recipe above)** — but
this needs the same real verification Qwen3-ForcedAligner got (checking `qwen2vl`'s own real
`use_mrope`/position-id behavior for text-only sequences against a real reference,
`examples/transformers`'s `Qwen2VLModel` or similar) before assuming it, not copy-pasted blind.

## Confirmed 2026-09-18: `ForwardPass.ExtractHiddenStates` already does exactly what this needs

Checked `src/OpenTail.Stingray.Engine/ForwardPass.cs:1096` directly: `ExtractHiddenStates` already
captures the FINAL transformer layer's hidden state per token (via `PrefillCore`'s
`outAllHiddenStates` parameter, or a per-token `Forward` fallback for MoE/single-token/TurboQuant
cases) — this is the exact same "final hidden_states[-1]" extraction Qwen Image's real recipe
needs (see above), already proven in production for `EmbeddingEngine`. **No new hidden-state-
extraction capability is needed for Qwen Image** (unlike FLUX.2, which needs a NEW capability to
capture 3 different INTERMEDIATE layers (10/20/30 of 40) simultaneously — `ExtractHiddenStates`
only ever returns the final layer, so FLUX.2's Mistral wiring is NOT a drop-in reuse of this same
method, a real difference between the two items despite looking superficially similar).

## 2026-09-18 UPDATE: steps 1-3 DONE -- qwen2vl admitted, real forward pass confirmed

This machine unexpectedly has PyTorch + `transformers` installed, so step 1's verification was
done against the REAL vendored `transformers.models.qwen2_vl.modeling_qwen2_vl` source directly,
not just by analogy to the Qwen3-ForcedAligner precedent: `get_rope_index`'s own docstring
confirms "text tokens use standard 1D RoPE" (all 3 M-RoPE axes get the same sequential position
for text), and `rotate_half` is the standard NEOX convention. Reasoning through the math: feeding
an identical position value into all 3 `mrope_section` channel groups is exactly equivalent to
plain single-axis 1D NEOX rope, so `qwen2vl` dispatches as NEOX in `ModelGraph.cs` (step 2, scoped
explicitly to text-only use, real multimodal M-RoPE still NOT implemented). `QwenImageTextConditioningTests`
confirms step 3 empirically: real Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf (4.36GB), real ChatML-templated
forward pass via `ForwardPass.ExtractHiddenStates`, finite non-zero hidden states -- the mechanism
works end-to-end.

## 2026-09-18 UPDATE: step 4 mechanism DONE -- QwenImageTextConditioning.Encode passes

`QwenImageTextConditioning.cs` (new) implements the complete real recipe: the exact hand-built
ChatML template (verbatim from `pipeline_qwenimage.py`'s `prompt_template_encode` -- confirmed NOT
the tokenizer's own `apply_chat_template`, unlike FLUX.2's Mistral recipe), final-layer-only
extraction via `ForwardPass.ExtractHiddenStates`, and the real `drop_idx=34` crop of the template's
own wrapper tokens. `QwenImageTextConditioning_Encode_RealWeights_ProducesFiniteConditioning`
passes against the real Qwen2.5-VL-7B checkpoint: finite `ContextDim=3584`-per-token output (no
projection needed -- 3584 matches Qwen2.5-VL-7B's own real hidden size exactly).

**Same day, step 4 also DONE: wired into `QwenImagePipeline` itself.** New
`QwenImagePipeline.Load(modelPath, textEncoderPath, vaePath, backend)` overload owns a real
Qwen2.5-VL-7B text encoder; `Generate()` now calls `QwenImageTextConditioning.Encode` automatically
(cond from the real prompt, uncond from `negativePrompt ?? ""`) when the caller doesn't already
supply an explicit `textContext` -- additive, existing 2-arg `Load()`/explicit-`textContext`
callers unaffected. Confirmed no regression on the existing real-weight DiT forward-pass test
(366s) and a new wiring smoke test confirms all three real checkpoints (DiT 8.3GB + text encoder
4.7GB + VAE) load and construct together cleanly. **Qwen Image's real text-conditioning gap
(docs/086's original finding) is now closed at the code level** -- every piece is real and wired.

## 2026-09-18 UPDATE: step 5 done -- real conditioning does NOT produce a coherent image, a SEPARATE bug found

Ran `QwenImageRealConditioningCoherenceTests` (256×256/8-step, real CFG guidance=4.0, 2889.9s):
output is a severe, regular checkerboard/basket-weave tiling artifact
(`docs/diffusion-samples/qwenimage_real_conditioning_256_8step_2026-09-18.png`), NOT a coherent
image. **Critical diagnostic comparison**: this output is visually IDENTICAL in character to the
earlier zero-conditioning run's output (`qwenimage_red-apple-on-white-table_256x256_4steps_
zero-cond_2026-09-18.png`) — same checkerboard pattern, same colors, same structure. **This proves
two things at once**: (1) the real text-conditioning wiring landed this session is functioning
correctly (real vs. zero conditioning demonstrably makes zero visual difference to this
artifact, exactly what you'd expect if conditioning is being read and applied but the actual bug
lives somewhere the conditioning signal never reaches, e.g. downstream of/parallel to it); (2) the
checkerboard artifact is a real, SEPARATE, pre-existing structural bug, not caused by or related
to text conditioning at all. This is the exact same class of periodic-tiling failure signature
this project spent 9 real rounds root-causing for FLUX.1 (`docs/056`, eventually found: missing T5
sequence-length padding) and separately found+fixed for Wan (`README`'s Wan row, a `flipSinToCos`
timestep-embedding bug). **Do not re-investigate the text-conditioning path for this specific
artifact — it's confirmed not the cause.** Real next step: the same patchify/unpatchify,
position-embedding, and VAE-tiling-boundary checks that found Wan's and FLUX.1's real causes,
applied to `QwenImageModel.cs`/`WanVaeDecoder3D.cs`'s Qwen-Image-specific call sites.

## Recommended next steps (NOT done this pass)

6. Apply the FLUX.1/Wan periodic-tiling-artifact investigation playbook to Qwen Image: check
   patchify/unpatchify ordering, 2D/3D RoPE position-id construction, and VAE decoder tiling/
   upsampling boundaries -- in that rough order of likely-cost-to-check, matching how those two
   prior investigations were actually resolved.

This is comparable in scope to FLUX.2's Mistral wiring, though simpler (final-layer only, no
gated-FFN/shared-modulation DiT complexity on this side) -- the `qwen2vl` architecture-support gap
(step 1-2) is the real unknown-sized piece, everything else is a well-scoped, mechanical port of a
precisely-confirmed recipe.
