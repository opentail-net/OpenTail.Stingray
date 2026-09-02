# 057 — Stable Audio 3 implementation plan

Status: **real checkpoint downloaded and tensor-mapped, 2026-09-02. Text encoder (T5Gemma) AND DiT
both fully implemented and golden-verified against the real reference on the first real run each.
Pipeline-level conditioning assembly wired end-to-end and confirmed running (smoke-tested, not yet
golden-verified at the pipeline level). VAE is the only remaining unported piece — architecture
inventoried but resampling mechanism (`TransformerResamplingBlock`) not yet decoded.** See
"Proposed phased implementation order" at the bottom for exact status per component.

## Real weights are available, fully ungated

`stabilityai/stable-audio-3-small-music` (and `-medium`, `-small-sfx`) are gated and need HF
approval. **`stabilityai/stable-audio-3-small-music-base` is NOT gated** and turns out to be
fully self-contained — it bundles all three components this pipeline needs as one download, no
approval required:

- `model.safetensors` (2.27 GB) — the real DiT **and** the real SAME-shaped VAE encoder/decoder
  (`pretransform.model.encoder.*` / `pretransform.model.decoder.*`, 244 tensors) in one file.
- `t5gemma-b-b-ul2/model.safetensors` + `tokenizer.json`/`tokenizer.model` — the real T5Gemma text
  conditioner, bundled as a subfolder with its own tokenizer, so the external gated
  `google/t5gemma-b-b-ul2` repo is not needed at all.

(`stabilityai/SAME-S`'s standalone `model.safetensors` was also downloaded and confirmed
config-identical to the bundled VAE — a second, independent source for the same weights if ever
needed, e.g. to isolate a VAE-only bug from a DiT-only bug.)

All local, downloaded 2026-09-02, not yet deleted (unlike the FLUX/Wan passes' checkpoints — these
are the ones the implementation phases below should be built and tested against, so keep them
until the port is verified):
`~/.cache/huggingface/hub/models--stabilityai--stable-audio-3-small-music-base/`,
`~/.cache/huggingface/hub/models--stabilityai--SAME-S/`.

Base variants (`*-base`) are the un-finetuned checkpoints — real weights, not a placeholder tier;
fine for both implementation and verification.

## Real resolved config (`small-music-base/model_config.json`)

This replaces the earlier doc's hedged "config-gated, not fixed" language — these are the actual
values for the checkpoint above, confirmed against real tensor shapes below, not just the JSON:

- `diffusion_objective`: `rectified_flow` (same `v = x - x0` convention as FLUX/LTX — current
  `StableAudioPipeline.Generate`'s Euler loop already assumes this, which happens to be right).
- `global_cond_type`: `adaLN` (not `prepend` — global conditioning modulates every layer via a
  6-way scale/shift/gate, not by being prepended as extra sequence tokens).
- `cross_attention_cond_ids: [prompt, seconds_total]`, `global_cond_ids: [seconds_total]` — **the
  text prompt is cross-attention-only; `seconds_total` conditions BOTH via cross-attention AND
  adaLN globally.** There is no `seconds_start` conditioner in this checkpoint's config at all
  (unlike `StableAudioDiT.ComputeTimingEmbedding`'s current fake implementation, which fabricates
  a `secondsStart` signal that has no real learned counterpart here).
- `local_add_cond_ids: [inpaint_mask, inpaint_masked_input]`, `local_add_cond_dim: 257` (256 latent
  channels + 1 mask channel) — real inpainting conditioning tensors exist in the checkpoint
  (`to_local_embed.*`, confirmed below) but are irrelevant for a plain text→audio generation path;
  can be left unwired (zero contribution) for a first working port, same way FLUX/LTX shipped
  without every conditioning branch on day one.
- `embed_dim: 1024`, `depth: 20`, `num_heads: 16` (head_dim 64), `cond_token_dim: 768`,
  `global_cond_dim: 768`, `io_channels: 256`, `num_memory_tokens: 64`.
- `attn_kwargs: {qk_norm: rms, differential: false}` — **real per-head RMSNorm on Q and K before
  attention**, a mechanism `StableAudioDiT`'s current stub has no equivalent of at all.
- `norm_type: rms_norm` with `force_fp32: true`.
- `timestep_features_type: expo` (`ExpoFourierFeatures`, not `FourierFeatures` — the current stub's
  `ComputeTimingEmbedding` uses neither, it's a hand-rolled sin/cos blend).
- `mask_padding_attention: true` — real prompt padding must be masked out of cross-attention
  (`conditioner.conditioners.prompt.padding_embedding`, a real learned padding-token embedding
  tensor exists — see below), not left as raw zeros.

## Real tensor map — DiT + VAE (`small-music-base/model.safetensors`, 685 tensors)

Confirmed via `safetensors.safe_open` (Python), not guessed. Per-layer tensors (×20 for the DiT,
×2 outer stages for the VAE) are listed once; the pattern repeats.

**DiT top-level** (`model.model.*`, `conditioner.*`):
```
conditioner.conditioners.prompt.padding_embedding          [768]          # learned pad-token embed, not zero
conditioner.conditioners.seconds_total.embedder.embedding.1.{weight,bias} [768,256]/[768]  # NumberEmbedder MLP
model.model.preprocess_conv.weight   [256,256,1]            # Conv1d, no bias (matches nn.init.zeros_ init)
model.model.postprocess_conv.weight  [256,256,1]
model.model.to_cond_embed.{0,2}.weight   [1024,768] / [1024,1024]   # no bias (SiLU sandwiched, matches dit.py)
model.model.to_global_embed.{0,2}.weight [1024,768] / [1024,1024]   # no bias
model.model.to_timestep_embed.{0,2}.{weight,bias}  [1024,256]+bias / [1024,1024]+bias  # HAS bias
model.model.transformer.global_cond_embedder.{0,2}.{weight,bias}  [1024,1024]+bias / [6144,1024]+bias
model.model.transformer.memory_tokens   [64,1024]           # real learned register/memory tokens
model.model.transformer.project_in.weight   [1024,256]      # no bias
model.model.transformer.project_out.weight  [256,1024]      # no bias
model.model.transformer.rotary_pos_emb.inv_freq  [16]       # RoPE IS active on this checkpoint (dim_heads/4=16)
```

**DiT per-layer** (`model.model.transformer.layers.{i}.*`, i=0..19):
```
pre_norm.gamma [1024]                          # RMSNorm (no beta/alpha at DiT level — plain rms_norm, not dyt)
self_attn.to_qkv.weight  [3072,1024]            # fused QKV, no bias
self_attn.q_norm.gamma / k_norm.gamma  [64]     # per-head RMSNorm on Q/K (head_dim=64) — the real qk_norm=rms
self_attn.to_out.weight  [1024,1024]            # no bias
cross_attend_norm.gamma [1024]
cross_attn.to_q.weight  [1024,1024]             # separate Q (from x) / KV (from cond) projections, no bias
cross_attn.to_kv.weight [2048,1024]             # fused KV, cond_dim(1024, already projected)->2*1024
cross_attn.q_norm.gamma / k_norm.gamma [64]
cross_attn.to_out.weight [1024,1024]
to_local_embed.{0,2}.{weight,bias}  [1024,257]+bias / [1024,1024]+bias   # inpaint-only, safe to leave unwired
to_scale_shift_gate  [6144]                     # per-layer learned bias added to global_cond_embedder output
ff_norm.gamma [1024]
ff.ff.0.proj.{weight,bias}  [8192,1024]+bias    # GLU: 8192 = 2 * (mult=4.0 * 1024)
ff.ff.2.{weight,bias}       [1024,4096]+bias    # projects the GLU's 4096-wide gated output back to 1024
```
No `conformer.*`/`modular_local_embeds.*` tensors anywhere — confirms conformer and modular local
conditioning are both off for this checkpoint; do not build them for a first port.

**Bottleneck** (`pretransform.model.bottleneck.*`, real `SoftNormBottleneck` params):
```
bias [1,256,1], scaling_factor [1,256,1], running_std [1], noise_scaling_factor [1,0,1] (empty — noise_augment_dim=0)
```

**VAE encoder/decoder** (`pretransform.model.{encoder,decoder}.*`) — genuinely more complex than
`autoencoders.py`'s `SAMEEncoder`/`SAMEDecoder` reading alone suggested. Each side is a small stack
of `TransformerResamplingBlock`s (encoder has `layers.0` = one resampling block with 6 inner
transformer sublayers, then `layers.2` = a plain final `Linear`; decoder mirrors this with
`layers.1` = plain initial `Linear`, then `layers.3` = the resampling block). Each
`TransformerResamplingBlock`:
```
mapping.{bias,weight_g,weight_v}   # weight-normalized Linear (PyTorch weight_norm decomposition — reconstruct as weight = weight_g * weight_v/||weight_v||)
new_tokens                          # learnable tokens injected during resampling (real up/downsampling mechanism, not conv stride)
transformers.{0..5}.pre_norm.{alpha,beta,gamma}    # DynamicTanh norm (dyt=true in config), NOT plain RMSNorm — has 3 params, not 1
transformers.{0..5}.self_attn.{q_norm,k_norm}.{alpha,beta,gamma}   # DynamicTanh on Q/K too
transformers.{0..5}.self_attn.to_qkv.weight / to_out.weight
transformers.{0..5}.rope.inv_freq
transformers.{0..5}.ff_norm.{alpha,beta,gamma}
transformers.{0..5}.ff.ff.0.proj.{weight,bias} / ff.ff.2.{weight,bias}   # same GLU FFN shape as the DiT
```
`DynamicTanh` (`transformer.py:325`) was not read in detail this pass — needs its own real formula
check before implementation (`alpha`/`beta`/`gamma` — likely `tanh(alpha*x)*gamma+beta` per the
"DyT" paper this class is presumably named after, but confirm against `transformer.py` directly,
not assumed).

## Real tensor map — T5Gemma text encoder (`t5gemma-b-b-ul2/model.safetensors`, 340 tensors)

Only `model.encoder.*` is needed (`T5GemmaEncoderModel` — the decoder half of the checkpoint,
`model.decoder.*`, is real but unused by this pipeline, per `conditioners.py`'s
`T5GemmaEncoderModel` import).

```
model.encoder.embed_tokens.weight   [256000,768]
model.encoder.layers.{0..11}.self_attn.{q,k,v,o}_proj.weight   [768,768]   # no bias, no separate q/k norm (unlike the DiT)
model.encoder.layers.{0..11}.pre_self_attn_layernorm.weight    [768]
model.encoder.layers.{0..11}.post_self_attn_layernorm.weight   [768]
model.encoder.layers.{0..11}.pre_feedforward_layernorm.weight  [768]
model.encoder.layers.{0..11}.post_feedforward_layernorm.weight [768]
model.encoder.layers.{0..11}.mlp.{gate,up}_proj.weight  [2048,768]
model.encoder.layers.{0..11}.mlp.down_proj.weight       [768,2048]
model.encoder.norm.weight  [768]
```

**This is architecturally a plain Gemma 2/3-family encoder** — real config confirms it exactly:
RoPE (theta 10000), alternating sliding-window (4096) / full attention per layer (even layers
sliding, odd full — `layer_types` in `config.json`), attention-logit softcapping (50.0),
`query_pre_attn_scalar` (64) instead of the usual `1/sqrt(head_dim)` scale, `gelu_pytorch_tanh`
MLP activation, and the same pre-norm+post-norm "sandwich" RMSNorm pattern around both self-attn
and FFN that real Gemma 2/3 use. This engine already has Gemma 3 kernels in
`OpenTail.Stingray.Engine` (softcapping/query_pre_attn_scalar handling referenced in
`GpuForwardPass.cs`/`ModelCompatibility.cs`) — **but those are wired into the causal-LM
`ModelGraph`/`ForwardPass`/GGUF pipeline, not reusable as-is for a bidirectional, safetensors-fed,
encoder-only forward pass.** The realistic reuse is porting the same per-op formulas (RoPE,
softcapped attention, sandwich RMSNorm, GeGLU-tanh MLP) into a small standalone encoder class the
way `T5Encoder.cs` was written standalone for LTX-Video's T5-XXL, not routing through the existing
GGUF `ModelGraph` machinery. No GGUF conversion of T5Gemma exists or is needed — load directly from
its real safetensors + `tokenizer.json` (a fast HF tokenizer this engine doesn't currently have a
reader for — check whether the existing BPE/SPM tokenizer infra reads a plain `tokenizer.json`
directly or needs a small new loader; not checked this pass).

## `StableAudioParams.cs` — fixed to real values this pass

Corrected: `LatentChannels` 64→256, `HiddenSize` 768→1024, `Depth` 12→20, `NumHeads` 12→16,
`TextContextDim` 4096→768. These were previously invented placeholder numbers with misleading doc
comments ("default: 4096" etc.) — now sourced directly from the real downloaded checkpoint's tensor
shapes above, not guessed.

## Real DiT forward-pass spec (fully decoded from `dit.py`/`transformer.py`/`diffusion.py`, 2026-09-02)

Everything below was read from the real reference source directly (not guessed), cross-checked
against the real tensor shapes in the section above. This is precise enough to implement from
directly.

**Top-level conditioning assembly** (`diffusion.py`'s `get_conditioning_inputs`):
- `cross_attn_cond` = `concat([prompt_embeddings (256,768), seconds_total_embedding (1,768)], dim=seq)`
  → `[257, 768]`. `cross_attn_mask` = `concat([prompt_attention_mask (256), ones(1)])`.
- `global_cond` (raw, pre-projection) = `seconds_total`'s own `NumberConditioner` embedding alone
  (768,) — **not** the prompt.
- Real `T5GemmaConditioner.padding_mode = "learned"`: padded prompt positions are **entirely
  replaced** by the real learned `conditioner.conditioners.prompt.padding_embedding` tensor
  (`torch.where(mask, real_embedding, padding_embedding)`), not zeroed and not left as raw encoder
  output — a real, easy-to-miss step between the T5Gemma encoder and the DiT.
- `NumberConditioner` (`seconds_total`, `min_val=0, max_val=384, fourier_features_type=expo`):
  normalize `clamp(seconds_total,0,384)/384` → `ExpoFourierFeatures` → the real
  `conditioner.conditioners.seconds_total.embedder.embedding.1.{weight,bias}` linear (`[768,256]`)
  → `[1,768]`. (`ExpoFourierFeatures`'s own formula not yet read this pass — check `blocks.py`
  before implementing.)

**`DiffusionTransformer._forward`** (`dit.py`):
1. `cross_attn_cond = to_cond_embed(cross_attn_cond)`: `Linear(768→1024,no bias)→SiLU→Linear(1024→1024,no bias)`.
2. `global_embed = to_global_embed(seconds_total_raw)`: same shape `Linear(768→1024)→SiLU→Linear(1024→1024)`, no bias.
3. `timestep_embed = to_timestep_embed(ExpoFourierFeatures(t))`: `Linear(256→1024,+bias)→SiLU→Linear(1024→1024,+bias)`. Since `timestep_cond_type="global"`: `global_embed = global_embed + timestep_embed`.
4. `x = preprocess_conv(x) + x` (Conv1d 256→256 kernel 1, no bias, zero-initialized at training start but real trained weights now) — a residual 1×1 conv "pre-mix", applied on `[256,seqLen]` (channels-first) before transpose to `[seqLen,256]`.

**`ContinuousTransformer.forward`**:
1. `x = project_in(x)`: `Linear(256→1024, no bias)`.
2. **No `prepend_embeds`** for this config (`global_cond_type=adaLN`, not `prepend` — the DiT-level
   `_forward` only builds `prepend_inputs` when `global_cond_type=="prepend"`, which this checkpoint
   is not).
3. `x = cat([memory_tokens.expand(batch,-1,-1) (64,1024), x], dim=1)` → `[64+seqLen, 1024]`. Real
   learned memory tokens (`model.model.transformer.memory_tokens`), **not** zero-initialized filler.
4. `rotary_pos_emb` computed for the FULL prepended length (`64+seqLen` positions, memory tokens
   included at positions `0..63`) — one shared RoPE table for the whole sequence, not per-segment.
5. `global_cond = global_cond_embedder(global_embed)`: `Linear(1024→1024,+bias)→SiLU→Linear(1024→6144,+bias)` → the real per-layer AdaLN input, shared by every layer (each layer adds this to its own learned `to_scale_shift_gate` bias before chunking).
6. Run all 20 `TransformerBlock`s (below), then `x = x[:, 64:, :]` (drop memory tokens) → `project_out`: `Linear(1024→256, no bias)`.

**`TransformerBlock.forward`, adaLN branch** (real per-layer math, `global_cond_dim>0` path):
```
scale_self, shift_self, gate_self, scale_ff, shift_ff, gate_ff = (to_scale_shift_gate[layer] + global_cond).chunk(6)  # each 1024-wide

residual = x
x = RMSNorm_eps1e-5(x, pre_norm.gamma)         # force_fp32 irrelevant in a fp32 C# port
x = x * (1 + scale_self) + shift_self
x = self_attn(x)                                # real formula below
x = x * sigmoid(1 - gate_self)                  # NOT tanh -- sigmoid gating, easy to mis-port
x = x + residual

# cross-attention: unconditional add, no adaLN gating on this sub-block at all
x = x + cross_attn(RMSNorm_eps1e-5(x, cross_attend_norm.gamma))   # to_local_embed / inpainting: skip, unwired in first port

residual = x
x = RMSNorm_eps1e-5(x, ff_norm.gamma)
x = x * (1 + scale_ff) + shift_ff
x = ff(x)                                       # SwiGLU, below
x = x * sigmoid(1 - gate_ff)
x = x + residual
```

**Self-attention** (`Attention` class, `qk_norm="rms"`, no GQA -- `num_heads==kv_heads==16`):
- `qkv = to_qkv(x)` (`[3072,1024]`, no bias) → split into `q,k,v` each `[seq,16,64]`.
- `q = RMSNorm_eps1e-6(q, self_attn.q_norm.gamma)`, same for `k` — **per-head** RMSNorm over the
  64-wide head dim (real `Attention.qk_norm_eps` default `1e-6`, distinct from the `1e-5` used by
  `pre_norm`/`ff_norm`/`cross_attend_norm` above — do not share one epsilon constant for both).
- Apply RoPE to `q,k` — **partial rotary, GPT-J-style, NOT a full-head_dim rotation**: real
  `ContinuousTransformer.rotary_pos_emb = RotaryEmbedding(max(dim_heads//2, 32))` = `RotaryEmbedding(32)`,
  so `inv_freq` has length 16 (confirmed: real checkpoint's `rotary_pos_emb.inv_freq` shape is
  `[16]`) and the rotated slice width (`rot_dim`) is `32`, **not the full 64-wide head_dim**. Only
  the FIRST 32 of each head's 64 channels get rotated (split into two 16-wide halves, standard
  split-half/NeoX rotation within that 32-wide slice); channels `[32,64)` pass through completely
  untouched (`t_unrotated` in `apply_rotary_pos_emb`, real "Wang et al. GPT-J" partial-rotary
  comment in the source). This is a different width than `SplitHalfRoPE`'s existing callers assume
  (they rotate the FULL head_dim) — needs either a new helper or `SplitHalfRoPE.ApplyRoPE` called
  with `headDim=32` against a manually-sliced sub-array per head, not a direct call against the
  full 64-wide head buffer.
- `scores[i,j] = dot(q_i,k_j) / sqrt(head_dim=64)` (plain scaled-dot-product — `Attention` has no
  extra `query_pre_attn_scalar`-style knob; that's a T5Gemma/Gemma-specific mechanism, not shared
  by this DiT's own attention class). No softcapping here either (that's also Gemma-specific).
- Self-attention has no padding here (fixed-length latent + memory tokens, no batch padding for a
  single real generation).
- `out = to_out(attn_out)` (`[1024,1024]`, no bias).

**Cross-attention** (`dim_context=1024` i.e. `cond_embed_dim` after `to_cond_embed`, so `to_kv`'s
real in-width is 1024 not 768 — confirmed by the real tensor shape `cross_attn.to_kv.weight
[2048,1024]`):
- `q = to_q(x)` (`[1024,1024]`, no bias) → `[seq,16,64]`; `kv = to_kv(context)` (`[2048,1024]`, no
  bias) → `k,v` each `[257,16,64]`.
- Same per-head RMSNorm (`q_norm`/`k_norm`, eps `1e-6`) on `q,k`. **No RoPE on cross-attention**
  (`cross_attn_rotary_pos_emb` is off for this config — only self-attention's shared
  `rotary_pos_emb` exists; cross-attention gets no positional signal at all beyond whatever the
  encoder already baked in).
- `scores[i,j] = dot(q_i,k_j)/sqrt(64)`.
- **Real padding handling is V-ZEROING, not additive masking**: `v = v * padding_mask` (zeroing out
  the V rows for padded cross-attn keys) BEFORE the weighted sum — the softmax itself still
  normalizes over ALL 257 keys including padded ones (their score is not set to `-inf`), so padded
  keys still consume softmax probability mass, just contribute zero to the output instead of being
  excluded from the normalizer. This is a real, deliberate (if unusual) choice in the reference —
  replicate it exactly, do not "fix" it to standard additive -inf masking, or the two will diverge
  numerically on any prompt shorter than the real `max_length=256` (i.e. almost every real prompt).
  Since this checkpoint's own real prompt-padding is `mode="learned"` (not zero), most padded
  positions are NOT zero vectors either — they hold the real learned `padding_embedding` — so this
  V-zeroing interacts with a non-zero K/V for those tokens in a way worth double-checking
  numerically against the real reference once implemented, not just reasoned about.

**FFN** (`FeedForward`, `mult=4.0`, real SwiGLU, not GELU-gated):
```
proj = ff.0.proj(x)                    # [8192,1024]+bias -> [seq,8192]
val, gate = proj.chunk(2, dim=-1)       # each [seq,4096]
h = val * SiLU(gate)                    # GLU.forward: x * act(gate), gate is the SECOND half
out = ff.2(h)                           # [1024,4096]+bias -> [seq,1024]
```

**Postprocess**: `x = x[:,:,prepend_length:]` (no-op here, `prepend_length=0` since only memory
tokens were prepended and already stripped inside `ContinuousTransformer`) →
`postprocess_conv(x) + x` (Conv1d 256→256 kernel 1, no bias) → this is the real predicted velocity
`v`, fed into the same Euler `latent += dt * v` loop `StableAudioPipeline.Generate` already has
(rectified-flow convention already correctly assumed there).

Not yet read this pass, needed before implementing: `ExpoFourierFeatures`'s exact formula
(`blocks.py`, referenced by both the DiT's own timestep embedding and `NumberConditioner`'s
`seconds_total` embedding) — check before implementing either.

## Real VAE spec — not yet decoded

`TransformerResamplingBlock` (`autoencoders.py:34`) and `DynamicTanh`'s exact composition within it
were not read this pass beyond the tensor map above. Known so far: `DynamicTanh.forward(x) =
gamma * tanh(alpha * x) + beta` (confirmed from `transformer.py:325`, real formula, not guessed) —
this is what the VAE's `pre_norm`/`ff_norm`/`q_norm`/`k_norm` all use (`dyt=true` in the real
`model_config.json`'s encoder/decoder config), NOT `RMSNorm`. The `mapping.{weight_g,weight_v,bias}`
weight-normalized linear and `new_tokens`-based resampling mechanism still need their own real
reference read (`TransformerResamplingBlock.__init__`/`forward`) before implementation — do not
guess the resampling mechanism from the name alone.

## Proposed phased implementation order

1. **T5Gemma text encoder — DONE, golden-verified, 2026-09-02.**
   `src/OpenTail.Stingray.Diffusion/TextEncoders/T5GemmaEncoder.cs`: real Gemma-2-family formulas
   (RoPE full-head_dim rotation, alternating sliding/full attention -- a no-op at this pipeline's
   real `max_length=256` since the real window is 4096, so implemented as plain bidirectional
   attention with a comment explaining why that's exactly equivalent here, not a shortcut),
   attention-logit softcapping (50.0), `query_pre_attn_scalar` (64), Gemma-family `(1+weight)`
   RMSNorm (NOT plain `weight`-scaled RMSNorm -- a real, easy-to-miss Gemma-specific detail),
   `gelu_pytorch_tanh` gated MLP. Golden test:
   `tests/OpenTail.Stingray.Tests.Diffusion/StableAudioT5GemmaEncoderGoldenParityTests.cs`, fixture
   in `tests/OpenTail.Stingray.Tests.Diffusion/TestData/StableAudioT5GemmaGolden/` (real
   `T5GemmaEncoderModel` output via HF `transformers`, dumped from the real bundled
   `t5gemma-b-b-ul2/model.safetensors`+tokenizer). **Passed on first real run**, cosine > 0.999.
   Weights at `models/stable-audio-3-t5gemma/model.safetensors` (real, ungated, kept locally like
   the project's other real-weight-suite checkpoints). Tokenization was NOT built this pass --
   the golden test feeds pre-tokenized ids from the fixture (same pattern
   `LtxT5EncoderGoldenParityTests` uses); a real `tokenizer.json`/SentencePiece reader for T5Gemma's
   own vocab is still needed before this encoder can be driven by raw prompt strings end-to-end.
2. **DiT — DONE, golden-verified, 2026-09-02.**
   `src/OpenTail.Stingray.Diffusion/StableAudio/StableAudioDiT.cs`: every real mechanism from the
   spec above (partial-rotary RoPE via a dedicated `ApplyPartialRope`, NOT the shared
   `SplitHalfRoPE` helper -- confirmed a new implementation was needed, not just a differently-sized
   call; V-zeroing cross-attn masking; 6-way sigmoid-gated AdaLN; per-head QK-RMSNorm at two
   different epsilons (1e-6 for q/k-norm, 1e-5 for the block-level norms -- real, and easy to
   collapse into one constant by mistake); SwiGLU FFN; 64 learned memory tokens; real
   `ExpoFourierFeatures`, formula confirmed from `blocks.py`). Golden test:
   `StableAudioDiTGoldenParityTests.cs`, fixture in `TestData/StableAudioDiTGolden/` (real
   `DiffusionTransformer.forward` output via the actual `stable_audio_3` package + real loaded
   checkpoint weights, called directly -- no hand-written numpy port needed here either).
   **Passed on the first real run**, cosine > 0.99, at a small (seqLen=8, nCond=25) synthetic-latent
   scale chosen to keep the reference-generation script fast while still exercising every real
   mechanism above. `StableAudioParams`/`StableAudioPipeline`/`StableAudioConformanceTests` all
   updated to the real, fixed-shape API (no more arbitrary synthetic dims -- the real architecture's
   shapes are fixed, so the old "pass any `HiddenSize`/`Depth`" test pattern no longer applies).
   Inpainting's `local_add_cond` branch stays unwired (irrelevant for plain text-to-audio, same
   simplification FLUX/LTX shipped with initially).
   **Pipeline-level wiring**: `StableAudioPipeline.cs` rewritten to actually call the real
   `T5GemmaEncoder`+`StableAudioDiT` (real conditioning assembly: prompt+seconds_total
   concatenation, real learned padding-embedding substitution, real `NumberConditioner` for
   `seconds_total`) -- confirmed running end-to-end in
   `StableAudioPipeline_GeneratesStereoWavFileWithTpdfDither` (real weights, 2 real Euler steps,
   valid WAV produced). Known gap: `StableAudioRequest.PromptTokenIds` requires pre-tokenized ids
   (no T5Gemma tokenizer wired yet, see item 1's note) and the decoded audio is not yet real (VAE
   below is still the placeholder stub) -- this test confirms the real pipeline PLUMBING is
   correct, not that its audio output is.
3. **VAE** — architecture inventoried (`TransformerResamplingBlock` w/ `DynamicTanh` norm, real
   formula `gamma*tanh(alpha*x)+beta` now confirmed), but the resampling mechanism itself
   (`mapping.{weight_g,weight_v,bias}` weight-norm reconstruction + `new_tokens`-driven up/down
   sampling) still needs its own real-source read before implementing -- do not guess this part
   from the tensor names alone, the same way the DiT's RoPE width and cross-attn masking convention
   turned out to be non-obvious from names/shapes alone.
4. **Golden verification** — same discipline as every other pipeline in this project, and as just
   demonstrated working end-to-end for the text encoder: a real reference run (this environment has
   working `torch`+`transformers`, so the actual HF/`stable_audio_3` package can be the oracle
   directly, no hand-written numpy port needed the way T5-XXL/GLM-4.6V required) per component
   (DiT single forward step, VAE decode), checked against this same downloaded checkpoint, before
   calling any piece done. Do not skip this for a component just because it "compiles and produces
   plausible-shaped output" -- this project's history (GLM-4.6V, FLUX, Pixtral) is full of
   components that looked structurally complete and were numerically dead or wrong underneath, and
   this pass's own DiT spec-reading surfaced several details (RoPE only rotating half the expected
   width, V-zeroing instead of additive masking) that would have been very easy to get plausibly-
   but-wrong without reading the real source line by line.
